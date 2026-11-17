# Architecture

The package turns an LC-ESI-MS run into a table of neutral-mass species with their
chromatographic peaks, adduct composition, impurity percentages and quantitation.

```
mzML ──> regions ──> summed spectrum ──> charge network ──> mass histogram ──> candidates
                                                                                   │
                     per-frame NNLS quantification <── template bank <── NNLS refinement
                                   │
                     species (deconvolved EICs) ──> integration ──> purity / potency / report
```

## 1. The logarithmic m/z grid

Every spectrum is resampled onto a grid whose coordinate is

    u = ln(m/z − carrier),   carrier = +m_proton (positive mode), −m_proton (negative mode)

Under this transform a proton-charged ion of neutral mass `M` at charge `z` sits at

    u = ln(M) − ln(z)

so the distance between adjacent charge states, `ln((z+1)/z)`, depends only on `z` and not on
the mass. A charge envelope therefore has the same shape wherever it appears, which is what
lets a convolutional network recognise charge states without having seen that particular mass.
An adduct adding mass `d` at constant charge is simply a proton-charged ion of mass `M + d`,
so adducts need no special encoding in the network at all.

The default grid spans m/z 50 to 10 000 with a step of 2e-5 (about 266 000 bins, equivalent to
a constant resolving power near 50 000). In practice the grid is narrowed to the measured m/z
range of each spectrum, which cuts inference time without changing results.

Resampling preserves peak shape: the raw axis is usually denser than the grid, so bins are
averaged rather than summed, and gaps between raw samples are interpolated. Centroided input
takes a different path and is placed on the grid with the instrument line shape.

## 2. The charge network

`ChargeUNet` is a 1D U-Net with six stride-2 levels, 7-wide kernels and a dilated residual
bottleneck, giving a receptive field of roughly 50 000 bins (about a factor 2.7 in charge). It
reads four channels over a 32 768-bin window:

| Channel | Content |
| --- | --- |
| 0 | `clip(log10(1 + I / noise_sigma), 0, 5)` — referenced to noise, not to the base peak, so a 0.01 % impurity looks the same regardless of what else is in the spectrum |
| 1 | intensity divided by a local rolling maximum |
| 2 | charge-evidence comb: for each hypothesised `z`, the spectrum shifted onto that charge's neighbours, reduced to its maximum |
| 3 | the charge that maximised the comb, scaled to [0, 1] |

It predicts two things per bin: a softmax over 101 classes (0 = non-ion, 1…100 = charge) and an
apex heatmap. The charge head is trained on *soft* targets — the fraction of the bin's intensity
belonging to each charge — because two charge states can coincide exactly (a dimer at `2z` lies
on the monomer at `z`) and a hard label could not represent that.

Deployment runs on ONNX Runtime. PyTorch is only needed for training.

## 3. From charge maps to candidate masses

Each bin above the noise threshold votes for a neutral mass `M = z · exp(u)`, weighted by its
intensity, the network's probability and the apex map. Votes accumulate in a logarithmic mass
histogram, which is smoothed over a fixed 1.2 Da in the mass domain (three passes of a box, close
to a Gaussian of 0.6 Da): when isotopes are resolved every isotopologue votes for its own mass,
and without that smoothing each tooth of the resulting one-dalton comb became a candidate. The
histogram's peaks are the candidate masses, refined to sub-bin precision by parabolic
interpolation.

Two gates remove the artefacts this produces. For each charge state a candidate claims, a
Gaussian-kernel mean shift as wide as the isotope envelope starts at the predicted m/z and climbs
to the nearest envelope; the mass it lands on, corrected for the envelope's skew, is that charge
state's estimate. The estimates are clustered, and the cluster holding most of the intensity
must contain at least two charge states that agree within 150 ppm. Starting at the prediction
keeps an impurity on its own envelope instead of jumping to the main component's apex nearby,
and clustering stops one charge state that lies on another species from vetoing the rest.
Candidates generated from the wings of a stronger envelope converge onto it and are merged away.

Two physical checks decide whether a charge state can count as support at all. Where the
instrument resolves the isotope spacing 1.00336/z at that m/z, the positions of the species' major
isotopologues must hold observed peaks (the comb may slide by up to half a spacing to absorb a
small mass offset); singly charged background read at z = 30 populates one or two of thirty
positions per dalton, and a z = 3 envelope read as its own dimer at z = 6 populates half of them.
Where the spacing is not resolved, an envelope much wider than one peak must appear as a
continuous hump in profile data rather than as a few narrow peaks of a low-charge species.
A further step drops candidates separated from a stronger one by an adduct delta, because the
stronger candidate's adduct columns will account for them.

On isotope-resolved data (Orbitrap and FT-ICR methods, or `isotope_charges: true`) the comb
estimator also reads charges from isotope spacing: for each bin, the largest charge whose
neighbours one spacing away carry a real share of its signal. A true z envelope also scores at
the divisors of z but not at 2z, whose half-spacing positions fall between its peaks. This is
what finds low-charge species, whose charge ladders are too short for the comb.

Two consequences for the gate in that mode. An intense unrelated peak that sits where an
envelope's isotope ladder would continue inherits the ladder's charge and can pull the mass
histogram's peak several daltons off a small molecule, so the envelope search may travel up to
2.5 envelope widths instead of 400 ppm. And under ion pairing an oligonucleotide may show only
one charge state above the noise; with `single_charge: true` such a candidate is kept when the
peaks of its envelope form a ladder whose measured spacing is 1.00336/z for its charge rather
than z - 1 or z + 1, and its isotopologue comb is filled. It is flagged "one charge state:
charge read from isotope spacing". The option is off by default (see `benchmarks.md` for what
it gains and costs on real runs).

## 4. Refinement and quantitation

Candidates are fitted greedily, strongest first, against the residual. For one candidate the
design matrix has a column per (charge, adduct state); the fit is non-negative least squares
weighted by `1/sqrt(I + sigma^2)`, which is the Poisson-appropriate weighting and is what makes
components at 0.01 % of the total ion current visible next to a saturating main peak.

Each candidate is fitted over every charge state the spectrum shows for it, not only over the
charges that voted for it: starting from that band, charges are added outward while an envelope
is observed where the species would put it. A charge map that misreads part of a species'
envelope otherwise leaves those charge states in the residual, where they are later explained
as separate, false species.

Detection uses only the base state — an adducted envelope still shows its base envelope — and the
full adduct combination set is fitted only for components that pass the gates. Only adduct types
the spectrum evidences are in that set: a type is kept when an envelope at M + delta of one of
the strongest candidates passes the same gate as a species would (`require_evidence`); a library
of five ion-pairing adduct types otherwise gives each charge state about fifty columns, and
neighbouring species and adduct columns trade intensity. A component
centred on an adducted form is moved down the adduct ladder when the lighter form is present on
most of the same charge states with at least half the height, and only if the same adduct model
then explains more of the spectrum. Accepted components are subtracted from the residual before
the next candidate is fitted.

After all passes, a component is dropped when at least 80 % of its fitted intensity lies within
two envelope widths of a stronger column of a stronger component (explaining away by shared
evidence). A charge misassignment reads a strong envelope as another mass whose envelopes sit on
the strong one's; a genuine weak species sits between them, and a real dimer keeps its odd charge
states.

The mass is then measured from the raw spectrum rather than the grid, which quantizes at about
20 ppm. Each fitted charge state's envelope is located by the same envelope-wide mean shift and
corrected for the skew of the class's isotope pattern, which gives the average mass directly and
does not hop between resolved isotopologues; the per-charge masses are clustered and the cluster
carrying most of the fitted intensity gives the mass and its spread.

## 5. Across retention time

Deconvolution runs on the summed spectrum of each chromatographic region, where the signal-to-
noise ratio is highest. Regions are built around peaks of the TIC that stand ten times above the
trace's noise (relative floors of 0.2 % of the tallest peak's height and 0.05 % of the total area
only remove ripple); neighbouring peaks are split at their valley, and only peaks whose apexes
are closer than their widths share a region. The components it finds define a template bank — one column per
component and adduct state, with the charge distribution baked in — and every frame in the
region is then quantified by a small non-negative least-squares solve against that bank. This
gives a deconvolved extracted-ion chromatogram per species at full time resolution, and shows
how each adduct's share changes across the peak, without paying for a full deconvolution per
frame.

Components are then linked across frames by mass, giving species. A species whose trace stays
within a third of its maximum at both ends - judged with one region width of quantified frames on
either side, since a broad band can fill its own region - is chemical background and is dropped,
and so is a species whose profile peaks in the outer tenth of its region (unless it is the
region's strongest): a region is built around an elution band, so such a trace is a template
absorbing a neighbour's tail. A species found in adjacent regions of one broad band is merged
into one. In a polarity-switching run each
polarity is processed separately and species are matched by neutral mass afterwards.

## 6. Integration, purity and saturation

Chromatographic integration follows the Agilent CDS model: initial events (slope sensitivity,
peak width, area and height reject, shoulders) plus a timed event table. Valley-separated peaks
share one baseline and are split by a drop line, which conserves the cluster's total area.
See `integration_events.md`.

Saturation is reported at two levels. A clipped, flat-topped MS frame is detector saturation. A
peak whose MS response falls at the apex relative to its own flanks, measured against the UV
trace after cross-correlation alignment, is ionisation-limited: the column is delivering more
analyte than the source can ionise. Both raise warnings that name the affected peaks.

## Module map

| Path | Responsibility |
| --- | --- |
| `chem/` | elements, formulas, isotope patterns, compound classes, adducts, modifications, instrument response |
| `synth/` | synthetic compounds, impurities, charge and adduct distributions, rendering, labels, whole runs |
| `nn/` | grid, featurization, model, losses, training, ONNX export, inference |
| `deconv/` | templates, mass-histogram decoding, NNLS refinement, per-spectrum pipeline |
| `diagnostics.py` | charge oracles, one-to-one scoring, loss and false-species attribution |
| `features/` | regions, per-frame quantification, retention-time linking, annotation |
| `chrom/` | integration events and the integrator |
| `quant/` | purity, calibration, potency, saturation |
| `report/`, `gui/`, `cli.py` | HTML reports, desktop application, command line |
