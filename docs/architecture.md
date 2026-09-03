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
histogram; its peaks are the candidate masses, refined to sub-bin precision by parabolic
interpolation.

Two gates remove the artefacts this produces. A candidate must have an observed peak at its
predicted m/z for at least two charge states, and the masses those peaks imply must agree with
each other. Candidates generated from the wings of a stronger envelope fail both tests. A third
step drops candidates separated from a stronger one by an adduct delta, because the stronger
candidate's adduct columns will account for them.

## 4. Refinement and quantitation

Candidates are fitted greedily, strongest first, against the residual. For one candidate the
design matrix has a column per (charge, adduct state); the fit is non-negative least squares
weighted by `1/sqrt(I + sigma^2)`, which is the Poisson-appropriate weighting and is what makes
components at 0.01 % of the total ion current visible next to a saturating main peak.

Detection uses only the base state — an adducted envelope still shows its base envelope — and the
full adduct combination set is fitted only for components that pass the gates. Accepted
components are subtracted from the residual before the next candidate is fitted.

The mass is then measured from the raw spectrum rather than the grid, which quantizes at about
20 ppm. Each supporting charge state's apex is located and converted to a mass; the observed apex
is the most abundant isotopologue, so the class pattern's average-minus-apex offset is added
before averaging. On a synthetic 17 kDa protein this recovers the mass to 0.2 ppm.

## 5. Across retention time

Deconvolution runs on the summed spectrum of each chromatographic region, where the signal-to-
noise ratio is highest. The components it finds define a template bank — one column per
component and adduct state, with the charge distribution baked in — and every frame in the
region is then quantified by a small non-negative least-squares solve against that bank. This
gives a deconvolved extracted-ion chromatogram per species at full time resolution, and shows
how each adduct's share changes across the peak, without paying for a full deconvolution per
frame.

Components are then linked across frames by mass, giving species. Species whose chromatographic
profile is flat are chemical background and are dropped. In a polarity-switching run each
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
| `features/` | regions, per-frame quantification, retention-time linking, annotation |
| `chrom/` | integration events and the integrator |
| `quant/` | purity, calibration, potency, saturation |
| `report/`, `gui/`, `cli.py` | HTML reports, desktop application, command line |
