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

Resampling preserves peak shape and area: each raw sample stands for its own slice of the axis,
so where the raw axis is denser than the grid a bin is the mean of its samples. Vendor software
writes profile scans thresholded, as short segments around peaks separated by gaps of up to
thousands of bins; a segment's edge sample stands for its own spacing, never for half of the gap
beside it, and nothing is written into the gaps (`core.profile.sample_widths`). A gap is a
spacing more than three times the scan's own spacing there, which is the 20th percentile of the
spacings in blocks of 65, interpolated between block centres and capped at three times the
typical block's: where noise or weak ions leave many short segments most spacings are gaps, and
a median would take them for the spacing. (It still does where isolated points make up more than
four fifths of a block - point-by-point thresholding, which vendor software does not do - and
on scans whose own spacing changes more than threefold over their range, as an FT-ICR scan over
m/z 150-3000 does.) A region's spectrum is the sum of its scans resampled one by one, because on
the union of many scans' axes a gap can no longer be told from the spacing; its raw-point view,
which the gates below read, is the scans interpolated within their own segments onto that union
and added in one sweep over all scans' segments (`Run.sum_frames`). Centroided input takes a
different path and is placed on the grid with the instrument line shape.

The noise level is estimated from the grid spectrum (median absolute deviation of first
differences over its lower intensities). Thresholded data hold almost no noise: what is written
is signal, most of it real low-level ions. Above that noise level the decoder therefore asks
more than "is there signal here" - whether isotopes have the right sizes and whether charge
states co-elute (section 3). An earlier resampling let every segment's edge point stand for half
the gap beside it and interpolated each scan onto another scan's axis, ramping across its own
gaps; that raised the noise estimate of real Orbitrap regions 40-180-fold, and the decoder's
thresholds had been tuned on it (`benchmarks.md`).

## 2. The charge network

Until a trained network has beaten it on the real runs, charges are assigned by the
deterministic **comb estimator** (`CombPredictor`): for every bin and candidate charge z it
multiplies the bin's log-SNR by the mean log-SNR at the positions of the same species' charge
states z ± 1..3 on the grid, takes the best charge, and on isotope-resolved spectra (Orbitrap,
FT-ICR) overrides it with the charge read from the isotopologue spacing 1.00336/z where that
spacing is resolved. It reads the whole spectrum at once.

The network meant to replace it is **LadderNet** (`build_model(..., arch="ladder")`,
`training.md`). For each candidate charge it reads a learned four-layer dilated-convolution
description of the spectrum (peak shape, isotope spacing) at the bin, at the bin's z ± 1..3
neighbours (fixed offsets ln(z/(z ± k))) and at the two positions only a reading at 2z would
fill; a small network shared by all charges, with a learned bias per charge, scores the
reading, and the comb's own score is added with a learnable weight. Its learned part starts at
zero, so the untrained network is the comb (a test checks it), and training learns
corrections. It has 9 500 (`proof`) to 22 000 (`full`) parameters, works on 32 768-bin windows
(a factor 1.93 in m/z, too narrow to hold a doubly charged ion's z = 1 and z = 3 neighbours),
and is exported to ONNX with static shifts. It is trained in two stages - singly charged ions
first without weight in the loss, then with them and dense singly charged background - on
synthetic spectra rendered along the inference path (`gpu_training/README.md`).

The first network, `ChargeUNet`, is a 1D U-Net with six stride-2 levels, 7-wide kernels and a
dilated residual bottleneck, giving a receptive field of roughly 50 000 bins (about a factor
2.7 in charge). Both networks read the same four channels over a 32 768-bin window:

| Channel | Content |
| --- | --- |
| 0 | `clip(log10(1 + I / noise_sigma), 0, 5)` — referenced to noise, not to the base peak, so a 0.01 % impurity looks the same regardless of what else is in the spectrum |
| 1 | intensity divided by a local rolling maximum |
| 2 | charge-evidence comb: for each hypothesised `z`, the spectrum shifted onto that charge's neighbours, reduced to its maximum |
| 3 | the charge that maximised the comb, scaled to [0, 1] |

Each predicts two things per bin: a softmax over 101 classes (0 = non-ion, 1…100 = charge) and
an apex heatmap. The charge head is trained on *soft* targets — the fraction of the bin's intensity
belonging to each charge — because two charge states can coincide exactly (a dimer at `2z` lies
on the monomer at `z`) and a hard label could not represent that.

Deployment runs on ONNX Runtime; the decoder takes the most probable charge among the ion
classes (whether a bin is signal is decided by its intensity). PyTorch is only needed for
training.

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

Two physical checks decide whether a charge state can count as support at all. Where the instrument
resolves the isotope spacing 1.00336/z at that m/z, the positions of the species' major
isotopologues must hold observed peaks (the comb may slide by up to half a spacing to absorb a
small mass offset); singly charged background read at z = 30 populates one or two of thirty
positions per dalton, and a z = 3 envelope read as its own dimer at z = 6 populates half of them. A
species has three major isotopologues at 1.2-1.9 kDa and five at 2.5-3.4 kDa, and an ion of half
the charge fills two or three of them, every other position - enough to pass: a placement whose
filled positions all lie on one such sub-comb does not count while the pattern has major
isotopologues off it (the singly and doubly charged ions of a 915 Da species were otherwise read
as charges 2 and 4 of an 1831 Da one, 1.2 % of the main species under IP-HILIC; the Orbitrap
oligonucleotide method now reaches down to 800 Da and reports that species from its own z = 1 and
z = 2, below. At 3.2 kDa the rule takes the charge-2 state, the 1580 Da shortmer's ions, from a
candidate built on that shortmer's non-covalent dimer, which it had kept out; the dimer is now
dropped by its fitted amount, section 4). Where the spacing is not resolved, an envelope much
wider than one peak must appear as a continuous hump in profile data rather than as a few narrow
peaks of a low-charge species.

On vendor-thresholded data some ion stands above the noise near almost every position, so a comb
position counts as populated only by a peak of at least a third of the height the isotope pattern
predicts there, given the envelope's height where the mean shift found it. A charge ladder assembled
from one strong low-charge ion and whatever weak signal lies at the other positions - the column
wash of the real runs produced dozens of 9-22 kDa "species" that way - fails it. And when the
spectrum is a region's sum, the charge states of the deciding cluster must co-elute (`coelution_r`,
`deconv.coelution`): the scan-by-scan intensity in each one's envelope core must correlate with the
strongest one's at Pearson r >= 0.5. A state that does not is another ion under this species'
window - unless the species' own isotopologues in it co-elute: narrow windows, half a peak width
either side, on the major isotopologues of the comb the strongest state's envelope sets, kept at
r >= 0.9 (`coelution.COMB_RESCUE_R`), and only where the strongest state's comb positions inside its
own window carry at least a quarter of the share the class pattern predicts there (a shortmer's
z = 1 ion, paired with its dimer's z = 3 at a window between the shortmer's isotopologues, fails
that). On the four runs the states that pass on their windows correlate on their isotopologues at a
median 0.96; of those that fail, a third reach 0.5, most of them adduct forms of the main species
beside other forms' peaks, and 4 % reach 0.9. 1883.3 under IP-HILIC, whose z = 3 window also holds
the 1234.2 Da 4-mer's +Na ion (r 0.36 on the window, 0.97 on its isotopologues), is kept this way;
so are NC + K and NC + DIPEA of the IP-RPLC conjugate run (r 0.98 and 0.903): the first now folds
into NC, the second is removed by the adduct dedupe and, found again in the next pass, by the
shared-evidence test. The adduct-evidence test (section 4) still reads the windows only
(`own_comb=False`): on the isotopologues it would evidence DIPEA in NC's region of that run. A state
that fails both leaves the cluster, which must still hold two charge states. The test abstains when
the strongest state carries signal in fewer than four scans. The charge states of one molecule are
the same molecules at different m/z; a ladder of unrelated ions follows as many elution profiles as
it has members. A further step drops candidates separated from a stronger one by an adduct delta,
because the stronger candidate's adduct columns will account for them.

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

The Orbitrap oligonucleotide method reads masses from 800 Da and charges from 1: below about
1.2 kDa z = 1 and z = 2 are a species' only charge states within m/z 400-2500, and below about
802 Da z = 2 leaves that range too. There z = 1 never makes a species on its own
(`singly_charged_species: false`), and a z = 1 state counts - in the candidate pick, the gate, the
fit's charge support, the charges added for the fit and the adduct-evidence test - only where its
envelope has no peak of a third of its height between its isotopologues
(`decode.singly_charged_contradicted`): any charge c >= 2 puts peaks at k/c of a spacing, and every
envelope fills a z = 1 comb (NC's non-covalent dimer [2M-5H]5-, read at z = 1, was otherwise a
2065-2079 Da species at 0.6-1.6 % of NC). The other methods keep singly charged species, and the
time-of-flight oligonucleotide method still starts at 1 kDa and z = 2. Peaks voted for by z = 1
beside another charge state also enter the decoder's list of 200 candidates, and on one synthetic
Orbitrap run they pushed a weak main species out of it (`benchmarks.md`).

## 4. Refinement and quantitation

Candidates are fitted greedily, strongest first, against the residual. For one candidate the
design matrix has a column per (charge, adduct state); the fit is non-negative least squares
weighted by `1/sqrt(I + sigma^2)`, which is the Poisson-appropriate weighting and is what makes
components at 0.01 % of the total ion current visible next to a saturating main peak. `I` is the
largest observed value within one peak width (at least two bins) of the bin, not the bin's own
value: weighting each bin by its own value (Neyman's chi-square) lets a template that sits a
fraction of a bin off its peaks be scaled far down, because the flank bins it over-predicts are
small and weigh a lot. On the real runs that fitted the main species to a fifth of its envelope
while impurities kept 60-80 % of theirs.

On isotope-resolved data a compound class's average composition places the isotopologues of a
particular molecule up to 0.1-0.2 Da from where they are (0.11 Da for the non-conjugated product of
the real runs under the `ps_rna` class): a grid bin at charge four. Before each candidate is fitted,
its templates are slid together by up to 1.5 bins, or 0.3 Da where that is wider, onto the spectrum
and the shift with the largest normalized overlap is kept (`refine.align_template`; only where the
instrument resolves the isotopes of the candidate's central charge). The offsets to absorb are in
daltons: 1.5 bins is 0.04 Da at 1.2 kDa, and with that window half the alignments of the IP-HILIC NC
run were pinned to its edge, a 1235 Da shortmer quantified at a four-hundredth of its amount. Where
the overlap's best is still the window's edge, or lies inside it between the teeth of the
candidate's own isotope comb, that comb decides (`refine.align_candidate`,
`refine.comb_phase_shift`): the residual's bins in each window of a charge state that passed the
gate, folded onto one isotope spacing, give the shift that puts the template's isotopologues on
them, modulo a spacing. A weak species' average mass can be read several tenths of a dalton off - a
co-eluting envelope 2 Da lighter on the same comb, or a neighbour 6 Da away, pulls the mean shift -
and its templates then need more than the window allows: the 1.9 kDa shortmer 1883.3 of the IP-RPLC
NC run, read 0.34 Da light, needs +0.36 Da and at the edge, 0.07 Da beside its isotopologues and off
their narrow peaks, was fitted 300 times low; n-2 of the NC product needs +0.49 Da under IP-RPLC and
-0.37 Da under IP-HILIC. The comb is consulted where the overlap's optimum is the window's edge, and
where it lies inside the window but off the comb: more than 0.15 Da from a tooth (`off_comb_da`) and
holding under a quarter (`between_ratio`) of the larger of the two teeth's overlaps. A mass read
about half a spacing off has both teeth beyond the window, and the optimum then sits between them on
noise (1908.3 and 2887.4 of the IP-RPLC NC run, read 0.54 and 0.53 Da light, 115 and 2000 times
below the better tooth, were missed that way; on the four runs optima on the comb lie within
0.093 Da of a tooth, those between the teeth 0.19-0.48 Da from it, holding at most 15 % of it). At
the edge the comb is consulted where its nearest tooth lies beyond the window, or inside it within
0.15 Da of the edge: that is the tooth cut short by the window, the overlap peaking past the
midpoint of the last step, where no parabola is fitted (on the four runs four alignments end so,
0.0005-0.024 Da from their tooth; 3207.4 of the IP-RPLC NC run, its tooth 0.005 Da inside the
0.289 Da window, was otherwise left at the edge and reported 0.30 Da light). An edge further from a
tooth inside the window, pulled there by a state the gate did not pass, is kept. The comb is taken
only where at least two gate-passed charge states agree on it, each within 0.1 Da of the combined
phase - a state with less than a tenth of the weight cannot veto, its phase being the less certain,
but counts towards the two only where it agrees (1883.3's comb is its z = 2's, 97 % of the weight,
z = 3 agreeing to 0.02 Da) - and half the windows' intensity lines up on it. Of its two teeth either
side of the measured mass the one that carries more of the spectrum is taken, in a weighted fit of
both teeth's templates together with templates of the same comb two to seven spacings beyond the
pair (`refine._better_tooth`): a mass read more than half a spacing off has its nearest tooth on the
wrong side, the edge between them fits neither, and compared alone by their overlaps the two teeth
differ by 0.03-26 %, right and wrong picks sharing that range (FLP + K of the IP-RPLC conjugate run
went one spacing light under a species 3-4 Da lighter on its comb; fitted with it, its own tooth
carries four times the other's). An optimum inside the window on the comb is kept even on the wrong
tooth, for a mass read 0.7-1.3 Da off (2558.3 of the IP-RPLC NC run, `benchmarks.md`): the two
teeth's overlaps do not tell that apart. A candidate whose gate measured another mass's envelopes
(one the adduct dedupe renamed to a lighter partner measured the stronger form's) is left where the
overlap put it, and so is an adduct form of an accepted species - within 1 Da of its mass, or of a
mass the adduct walk moved a candidate from (itself a form of the species it joined), plus an adduct
state, up to three of a type - whose mass its neighbouring forms pull by up to three quarters of a
dalton and whose two teeth fit alike (NC + 3Na of the IP-HILIC NC run, to 1.0 %). Placed on its comb
such a form is fitted whole: NC's +5Na and +3Na+K forms, beyond the two of a type the fit allows and
the three the exemption covers, were reported as a 3.3 % impurity until the mass NC + 3Na was walked
from anchored them. Sodium and potassium forms deeper still, four to eight in all, are recognised by
their comb instead where the region evidences sodium (`refine._on_cation_ladder`): such a form
shares its parent's isotope comb exactly, moved by the ladder's monoisotopic shift, so a candidate
whose comb falls within 0.03 Da of an accepted species' comb so moved, its mass within 1 Da of that
species plus the ladder, keeps the overlap's place as well (2887.4 + 3Na + K, 2991.98 Da of the
IP-HILIC NC run, at an optimum inside the window). Placed on their combs DA-NC + 4Na and NC + 6Na +
K of the IP-HILIC NC run were fitted whole (1.15 % and 0.44 % of NC, 0.79 % and 0.14 % at the edge),
the first taking part of DA-NC's own sodium ladder with it. A species of another composition near
such a mass lies on that comb by chance about one time in sixteen - a 4949 Da candidate of the
IP-RPLC NC run, where no sodium is evidenced, lies on NC - dG's comb moved by six sodium to 0.003 Da
(its report, 4949.38 Da, now merges with a 4948.78 Da piece into 4948.97 Da, `benchmarks.md`).
Searching the overlap itself past the edge instead lets a candidate slide onto whatever lies there:
one the adduct dedupe had renamed to a weak partner 43.5 Da (two sodium) lighter took the IP-RPLC
main species' peaks as its +2Na form, and the main species was lost. The overlap is computed in
quarter-bin steps and the optimum placed between them by a parabola through the best step and its
neighbours, around no shift too unless a step does clearly better: at 140 000 (m/z 200) an Orbitrap
peak at m/z 1700 is narrower than a 2e-5 bin, and the main species of the IP-RPLC conjugate run is
fitted 2 % low with its templates an eighth of a bin off its peaks, 9 % a quarter bin off and 28 %
half a bin off. The shift moves the templates; the reported mass is measured from the raw envelopes
as below, except that a species whose templates its own comb placed is reported at the templates'
position where that lies within half a spacing of the measured mass (`refine.report_comb_masses`),
once per frame after the adduct steps, so that detection and every adduct decision within the frame
read the measured mass (the adduct folds and region merges across the run read the reported one);
when the mass changes afterwards (that measurement, or two copies of one species merged) the shift
is adjusted so that the templates stay where the alignment put them, and a candidate re-centred from
an adducted form onto its base form is aligned afresh. Before this, re-measuring the mass moved the
templates with it, and a re-centred candidate kept the shift found for its adducted form; on that
main species, re-centred from an adducted form, the two cost 5 % of its amount (with only the first
corrected, 20 %).

The overlap search and the adduct walk read only the candidate's own charge states
(`refine.own_charges`): those the gate passed, with those it passed for a duplicate merged in by
mass, and those added outward for the fit (below), each checked by its isotopes and co-elution at
this mass. A voted state the gate measured and rejected - an envelope that moved off, contradicted
its charge by its isotopes, was discontinuous, belonged to another mass or did not co-elute - holds
another species' signal, and one it could not measure (no envelope above the noise there) may hold
anything. Read by the overlap search, an isotope comb of the wrong charge under one rejected state
pulled a 6752 Da species' templates to the far edge of the window, and the species was dropped;
read by the walk, a peak under one walked n-2 onto a phantom (below) before its templates were
placed by its comb. A candidate without a gate record - renamed by the adduct dedupe, or kept
without one - reads every state it carries but those a merged duplicate's gate rejected. The fit
itself still covers every state the candidate carries: restricted to the gate-passed ones it loses
EIC-confirmed species.

Each candidate is fitted over every charge state the spectrum shows for it, not only over the
charges that voted for it: starting from that band, charges are added outward while an envelope is
observed where the species would put it - above the noise, with its isotope comb filled in
proportion where the isotopes are resolved, and co-eluting with the starting band. A charge map that
misreads part of a species' envelope otherwise leaves those charge states in the residual, where
they are later explained as separate, false species.

Detection uses only the base state — an adducted envelope still shows its base envelope — and the
full adduct combination set is fitted only for components that pass the gates. Only adduct types
the spectrum evidences are in that set: a type is kept when an envelope at M + delta of one of the
strongest candidates passes the same gate as a species would (`require_evidence`), its mass
measured as detection measures it and compared with the candidate's average mass plus the adduct's
average-mass shift (K - H is 0.13 Da heavier than its monoisotopic shift, as potassium is 6.7 %
41K); a library of five ion-pairing adduct types otherwise gives each charge state about fifty
columns, and neighbouring species and adduct columns trade intensity. A component centred on an
adducted form is moved down the adduct ladder (by average-mass shifts, as its mass is an average
mass) when the lighter form is present on most of the candidate's own charge states (above: an
unrelated peak 30 times taller under n-2's z = 9 window, a state that failed the gate, walked n-2
down a sodium onto a phantom) with at least half the height, an envelope is centred within half an
envelope width (at least 0.5 Da) of where the step lands (the median of the per-charge mean shifts
that stay within an envelope width of it - on the flank of a stronger species' envelope they all
run off toward its centre), and the same adduct model then explains more of the spectrum. The
lighter trial is placed by the same rule as the candidate it is compared with, on the candidate's
charge states and with the adduct-form exemption: against a trial pinned at the window's edge a
candidate placed by its comb always won. A trial at an accepted species' own mass is no adduct form
of it; placed on that species' comb it fits what the species' fit left, and the candidate joins
that species (NC + 3Na of the IP-HILIC NC run, kept at its window's edge as a form of NC, is joined
to NC this way and no longer reported apart). Accepted components are subtracted from the residual
before the next candidate is fitted.

After all passes, where `suppress_multimers` is on, a component at n times the mass of a stronger
component (n = 2-6, within the multimer tolerance of 500 ppm, at least 2 Da and n x 0.5 Da, with
less than 0.6 of its amount) is dropped first (`refine.drop_multimers`): with resolved isotopes a
multimer's candidate scores like its monomer's in the mass histogram and passes the harmonic test
(the 1580 Da shortmer's [2M-3H]3- dimer scored 0.74 of the shortmer). It goes as a harmonic
reading of the monomer's ions where its own charge states (not multiples of n) carry under a
quarter of its fit, and otherwise only where its strongest own state co-elutes with the monomer's
strongest over the region's scans. Where that cannot be judged - no frames, fewer than four
scans, or no profile in the monomer's trace - it is kept: the oligonucleotide methods take regions
of two scans, and a species at n x M eluting apart from its monomer would otherwise go without
evidence. Then a component is dropped when at least 80 % of its fitted intensity lies within two
envelope widths of a stronger column of a stronger component (explaining away by shared evidence).
An adduct column of a weaker component whose mass (species plus adduct) lies within 0.5 Da of a
stronger component plus any adduct state of the method's list, evidenced or not, is removed from it:
the PS-to-PO impurity plus potassium is the main species plus sodium to within 0.07 Da, and fitted
to the impurity that column carried the main species' sodium adduct into the impurity's amount. A
charge misassignment reads a strong envelope as another mass whose envelopes sit on the strong
one's; a genuine weak species sits between them, and a real dimer keeps its odd charge states -
which the shared-evidence test respects, while the multimer rule drops a dimer that co-elutes with
its monomer.

The mass is then measured from the raw spectrum rather than the grid, which quantizes at about
20 ppm. Each fitted charge state's envelope is located by the same envelope-wide mean shift and
corrected for the skew of the class's isotope pattern, which gives the average mass directly and
does not hop between resolved isotopologues; the per-charge masses are clustered and the cluster
carrying most of the fitted intensity gives the mass and its spread. A comb-placed species is then
reported at its templates' position (above): its comb puts the templates within the class pattern's
composition error of the truth (-0.12 to +0.17 Da on the four runs, the same for one molecule in
both chromatographies; 0.06 Da for both n-2s), where a neighbour on the same comb pulls the mean
shift several tenths of a dalton off (n-2 0.49 Da light under IP-RPLC, by a 4497 Da species six
spacings below it). The template position is taken only within half a spacing of the measured mass,
as the comb is read modulo one spacing: three species of the IP-RPLC NC run whose templates lie
0.52-0.60 Da from their measured mass keep it (`benchmarks.md`). The summed spectrum's axis also
weights points by count, which pulls a mean shift towards the side more scans cover
(`benchmarks.md`).

## 5. Across retention time

Deconvolution runs on the summed spectrum of each chromatographic region, where the signal-to-noise
ratio is highest. Regions are built around peaks of the TIC that stand ten times above the trace's
noise - three times with the oligonucleotide methods, as a 2 % impurity eluting on its own makes a
TIC peak about four times the noise (relative floors of 0.2 % of the tallest peak's height and 0.05
% of the total area only remove ripple; the oligonucleotide methods lower the height floor to 0.1 %,
`regions.min_rel_height`, as 0.2 % deleted the region of a 0.04 % shortmer whose TIC peak stood 3.5
times above the noise at 0.197 % of the tallest - on the real runs the noise floor then decides on
three of four, at 0.17-0.49 % of the tallest peak, and 0.1 % binds on the IP-RPLC conjugate run,
whose noise floor lies at 0.076 %); neighbouring peaks are split at their valley, and only peaks
whose apexes are closer than their widths share a region. The components it finds define a template
bank — one column per component and adduct state, with the charge distribution baked in — and every
frame in the region is then quantified by a small non-negative least-squares solve against that
bank. This gives a deconvolved extracted-ion chromatogram per species at full time resolution, and
shows how each adduct's share changes across the peak, without paying for a full deconvolution per
frame.

Components are then linked across frames by mass, giving species. A species whose trace stays within
a third of its maximum at both ends - judged with one region width of quantified frames on either
side, since a broad band can fill its own region - is chemical background and is dropped, and so is
a species whose profile peaks in the outer tenth of its region (unless it is the region's
strongest): a region is built around an elution band, so such a trace is a template absorbing a
neighbour's tail. That last test is not made at a boundary the region shares with the next one: a
boundary drawn at the valley between two bands is where a species eluting between them peaks, and
the neighbouring region holds the rest of it. A species found in adjacent regions of one broad band
is merged into one; the scan on their shared boundary is counted once, with the larger of its two
readings, and the merge is repeated until nothing more joins (pieces are taken strongest first, and
the last piece of a tail over three regions was left apart: 2653.32 and 2998.45 Da of the IP-RPLC
conjugate run). Pieces of comparable height (each at least a tenth of the other) whose joined trace
falls below half the lower apex between them are two bands of one mass - isomers either side of a
boundary - and stay apart; a scan where neither piece has a reading (a gap the linking bridged, a
frame whose fit gave the species nothing; three scans at most) is read across from its neighbours,
not as a valley, while a longer stretch without a reading is a valley. A piece that a shared
boundary cut off and its region read up to 1.5 Da off its band, beyond the linking tolerance, is
joined to the band where both read the same tooth of the isotope comb in the scans themselves
(`features.link.merge_boundary_pieces`, `refine.tooth_position`: the class pattern matched against
each scan's own points), and the joined species keeps the mass its own isotopes support (n-1 of the
IP-HILIC NC run: one report at 2.35 % instead of 1.9 % and 0.5 %); a deaminated form lies one tooth
away and stays apart. With `linking.set_aside_later_elutions` (on in the two oligonucleotide
methods, off by default) the run's main species' later elutions - its tail past its band and its
column-wash peak: the same molecule by its isotopes, at most a twentieth of it, starting more than
0.2 min after its band ends - are named in its flags instead of being reported, and a species
peaking within 0.1 min of such a wash peak (a later elution that rises and falls, its highest
reading twice its first and last) is set aside the same way under an earlier species of its own mass
at least twenty times stronger (NC's +201.4 Da form at 25.4 min). An isomer eluting later at under a
twentieth of the main species is set aside as well, since its isotopes cannot tell it apart - an
isoAsp or disulfide-scrambled protein variant, a branched polymer beside a linear one - which is why
the other methods leave the step off. In a polarity-switching run each polarity is processed
separately and species are matched by neutral mass and apex (within 0.2 min) afterwards.

A species' amount is its base form, integrated over its trace (`Species.base_intensity`); its total
over every adduct form resolved (`total_intensity`) is reported beside it. Which adduct forms are
resolved depends on the region: a region attaches an adduct type to its species only where it shows
that adduct next to its strongest species, and doing so everywhere makes near-coincident forms of
co-eluting species trade signal (with sodium and potassium forced into every region, the IP-HILIC
NC run lost its PS-to-PO impurity and NC - dG). Where a region does not attach an adduct, the same
forms come out as species of their own; they are folded into the species they belong to. A species
whose mass is a stronger (base form), co-eluting species' plus an adduct state of cation adducts or
of adduct types some region of the run evidenced (the mobile phase's amines, acids and solvents,
whose clusters can lie within a dalton of an oligonucleotide's n+1; average-mass shifts, within 0.6
Da or 2.5 peak widths M/R, whichever is larger; apexes within 0.1 min and traces correlating at
r >= 0.8, both on all forms) is counted as that species' adduct form, unless a known modification of
its class explains the difference as well (peptide oxidation lies 1.04 Da from an ammonium adduct,
an oligonucleotide's n+1 can lie within a dalton of amine and HFIP clusters): a report too many is
the lesser error. Forms are folded before the pieces of a species split over two regions are
merged, so that a form is judged against its parent's piece from the same region. Base forms are
measured for every species alike, and the retention-time apex is read on the base trace for the
same reason, unless the base form is less than a tenth of the species (an adducted form reported
apart).

Base-form shares are exact only where co-eluting species carry the same adduct load: under ion
pairing with acetate the main species' load falls at its own apex, and base-form shares there
understate impurities by up to the ratio of the loads (1.05-1.2 on the IP-HILIC conjugate run,
up to 1.7-1.9 on the IP-HILIC NC run; `benchmarks.md`).

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
| `core/` | data model (spectra, runs, results), methods, vendor-thresholded profiles (`profile.py`) |
| `deconv/` | templates, mass-histogram decoding, co-elution of charge states, NNLS refinement, per-spectrum pipeline |
| `diagnostics.py` | charge oracles, one-to-one scoring, loss and false-species attribution |
| `features/` | regions, per-frame quantification, retention-time linking, annotation |
| `chrom/` | integration events and the integrator |
| `quant/` | purity, calibration, potency, saturation |
| `report/`, `gui/`, `cli.py` | HTML reports, desktop application, command line |
