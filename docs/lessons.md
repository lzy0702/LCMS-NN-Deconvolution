# Lessons from a parallel neural-network deconvolution effort

This package was built twice, by two teams working toward the same goal: a neural network that
turns the multiply-charged electrospray envelopes of an LC-MS run into neutral masses, merges
adducts back into their species and finds impurities next to a dominant component. The other
effort, `lcms-nn-deconv` (a Rust workstation with an oligonucleotide focus), kept a detailed
record of three months of work (2026-08 to 2026-11): 116 numbered experiment sections, the
decoder studies D109-D113, three late "routes" (forward fit, proposal + physics, mass field) and
about 464 000 lines of Rust. None of its models reached the bar the team had set for
integration. Its record also reviewed this package and named what it would borrow from it.

This document distils that record into rules, says what each one changed here, and gives the
measurements. Section numbers (§N) refer to the other project's `docs/nn-current-state.md`.

## 1. Find the bottleneck with oracles before training anything

**There.** The decisive experiment of the whole record was cheap and late (D109, §102): feed the
decoder the true per-bin charge shares instead of a network's. With perfect shares the
information ceiling was a recall of 0.96-0.98 at almost no false peaks; the decoders actually
built reached 0.66-0.84 recall with 8-31 % false peaks (decoder v0 to v2.1, §102-§105). A
second oracle that corrupts the shares at a controlled rate (2z, z/2, z±1) showed that the
simplest decoder needed 98.8 % charge accuracy to keep false peaks under 5 %, against the
61-71 % their networks reached. Months of network work had been aimed at the wrong stage.

**Here.** The same probe, run on day one, gave the same answer for this package: with a perfect
charge map the pipeline found the main component in only half of the protein runs and reported
two to five false species for every true one, barely better than the deterministic comb
estimator. No network could have fixed that. Everything in sections 4 to 8 below was found with
three tools that now ship in `lcmsdeconv.diagnostics`:

- `oracle_predictor(truth, polarity)`: the per-bin charge map a perfect network would produce;
- `confused_oracle_predictor(truth, polarity, accuracy)`: the same map with a fraction of
  charge envelopes relabelled the way networks err, which turns "how good must the network be"
  into a number;
- `attribute_losses(trace, truth, species)`: for every true species, the stage at which it left
  the chain (no region, votes, histogram peak, consistency gate, de-duplication, fit, merge,
  linking), recorded by `process_run(..., trace=[])`.

`scripts/benchmark_synth.py --predictor oracle|comb|confused:0.9|model.onnx` reports all of it.

## 2. Keep the network's job narrow and its objective free of constant solutions

**There.** Most of the first two months went into dense detection heads and their objectives,
and almost every one had an input-independent optimum that training found:

- per-bin sigmoid cross-entropy with about 1 positive per 4 000 bins has the fixed point
  σ* = pw·r / (1 + pw·r); positive weights, focal terms, double normalisation and curricula moved
  or delayed it, never removed it (§28-§43; it saturated within 750 steps even on noise-free,
  single-species data);
- a softmax over 76 000 mass bins stayed exactly uniform for 20 000 steps (§28);
- DETR-style sigmoid coordinates fell into an absorbing state at bin 0 (§21);
- an ordinal charge head hit an exact-zero loss trap from float underflow (§64-§66);
- denoising objectives settled on "copy the input" or "output zero" (D85-D90).

Learned presence scores saturated to constants outside the training distribution (10.5 million
windows scored exactly 0.0, §49), while deterministic proposers and a logistic regression on 19
geometric features (AUC 0.850 against the network's 0.625, §86) kept beating them.

**Here.** The network only assigns a charge to each grid bin, with a softmax over
{non-ion, z = 1..100} trained on soft shares (the fraction of a bin's intensity belonging to
each charge). The other team's review called this the one per-bin objective their record never
falsified: a softmax cannot fire everywhere, and soft shares make "everything is background"
expensive. Detection, separation and quantitation stay deterministic and inspectable.
`lcmsdeconv.nn.checks.degeneracy_report` now verifies on real batches, before training, that
the true labels beat every constant answer: on the current training mixture the truth scores a
cross-entropy of 0.11 against 1.24 for the best constant (the class prior), 3.27 for "all
background" and 6.6 for the comb's guess.

## 3. Train on exactly what inference sees

**There.** Train/evaluation mismatch was the most frequent root cause of a "model failure":
window statistics taken from the truth in training but measured at evaluation (the same window
read as 1 % or 98.5 % background, all 931 of 931 decisions flipped, §46); a noise level nominally
5-30 % that was 5 007 times the signal by energy (§54); a grid default that stretched the axis by
1.36×; monoisotopic against average masses (about 5 Da at 8 kDa).

**Here.** The same class of mismatch was present three times over, and is fixed:

| | Training (before) | Inference |
| --- | --- | --- |
| Spectrum | sticks rendered straight onto the grid, one scan | vendor-thresholded profile scans, summed over a region, resampled onto the grid |
| Noise level in the features | the generator's own parameter | `estimate_noise_sigma` on the grid spectrum |
| Comb feature at window edges | `np.roll` wrapped the far edge in | the same wrap, on different content |

`SynthConfig.render_path = "profile"` (the default) now renders training frames along the
inference path: profile scans on the instrument's axis, shot and electronic noise, vendor
thresholding, a sum over one to eight scans, resampling onto the grid, and the noise level
estimated from the result by the function inference uses. Labels come from per-charge renders
through the same resampling. The comb shifts with zero padding instead of wrapping. The old
renderer is kept as `render_path = "grid"`.

## 4. Generators fail quietly; audit them like code

**There.** Adducts sampled per (species, charge) left 73 % of families with a single charge state,
which then shaped decoder design; "noisy" variants turned out identical to clean ones; two
configurations with the same hash had different contents; truth entries for scans where a
species had no detectable signal capped every engine's recall near 0.1.

**Here.** Four generator defects were found by looking at the truth of the benchmark runs:

1. The exponentially modified Gaussian underflowed to zeros whenever the tail was short
   compared with the width (σ/τ above about 27), deleting about 2 % of all chromatographic peaks
   from the data while keeping them in the truth. One of the "missed main components" of the
   protein benchmark was never in the spectra. It is now computed with the scaled
   complementary error function.
2. Adducts were drawn with a per-charge propensity up to 0.15, so a protein ion at z = 40
   carried about six adducts and its un-adducted form was 0.4-5.6 % of its signal: the true
   mass was nearly invisible. The load is now drawn per ion (`adduct_max_per_ion`, mean
   adducts of each type at the apex charge, default up to 0.3); the old behaviour remains as
   `adduct_max_lambda` and the `heavy` benchmark generator.
3. Impurities drew their charge envelopes independently of their parent, giving a 1 % oxidised
   form charge states its parent never showed. They now inherit the parent's envelope, scaled
   by mass.
4. Oligonucleotide charge states were about 40 % low against a measured ion-pairing run in the
   other project (an 8.2 kDa oligonucleotide centred at z ≈ 10, a related 8.4 kDa species at
   z ≈ 7.5); the model now gives z ≈ 9 at 8.2 kDa.

## 5. Score one-to-one, against truth that can be detected

**There.** Recall that rewards spraying candidates (identical from threshold 0.05 to 0.9), truth
entries no detector could see, metrics computed only on "readable" cases, and engine comparisons
distorted by the wrong resolving power or mass convention all misled the team at some point.

**Here.** Scoring pairs reported and true species one-to-one, serving abundant species first
(so an unresolvable deamidated form one dalton from a 75 kDa protein cannot take the main
component's report). Every true species carries the expected ion count of its second-strongest
charge state (`support_counts`), and recall is reported both over all species and over those
with at least 50 counts, below which two charge states cannot be confirmed above shot noise.
Unmatched reports are classified as charge harmonics, adduct-shifted copies, mass errors or
unexplained.

## 6. Floors tied to the strongest signal delete real species

**There.** Base-peak-relative thresholds limited the classical engines to about 50:1 dynamic
range; weight floors were behind 73 % of one decoder's misses.

**Here.** The region finder discarded chromatographic peaks below 5 % of the tallest one. A
second main component at 4 % of the TIC height (10 % of its area) was never deconvolved. Peaks
are now kept when they stand ten times above the trace's noise, with relative floors of 0.2 %
of height and 0.05 % of area, and neighbouring peaks become separate regions split at their
valley, so one region's summed spectrum is dominated by one elution band.

## 7. Measure on the raw data, with statistics that tolerate a neighbour

**There.** Apex-based masses, isotope folding that merged species one dalton apart, and an
uncorrected elution-tail bias (about 0.26 Da at 8 kDa) cost the classical engines accuracy.

**Here.**

- The mass histogram was smoothed over 25 ppm, so resolved isotopologues one dalton apart each
  became a candidate (39 candidates for a handful of species). It is now smoothed over 1.2 Da in
  the mass domain, which leaves under 1 % of the isotope ripple.
- The consistency gate took the largest raw point within ±400 ppm of each predicted charge
  state, which put an impurity on the main component's apex and merged it away. It now runs a
  Gaussian-kernel mean shift, as wide as the isotope envelope, from the predicted position (it
  climbs to the nearest envelope), corrects for the envelope's skew, and clusters the per-charge
  masses so that one charge state lying on another species cannot veto the rest.
- The adduct "walk-down" compared a base-only fit with a full adduct-ladder fit, so the larger
  model always won and walked main components onto weaker neighbours (an oligonucleotide was
  reported two sodium ions light). It now compares like with like and requires the lighter form
  to carry at least half the current form's height.

Median mass error with a perfect charge map fell from 18-42 ppm to 1-7 ppm.

## 8. Explain away by shared evidence, and fit the whole envelope

**There.** The guard that fixed the classical engine's best-known real-data artefact (a 4 131 Da
"species" at half the mass of an 8 258 Da oligonucleotide, 64 % of the base peak before the fix)
was to drop a candidate whose charge-state peaks are mostly claimed by a stronger candidate.

**Here.** A charge misassignment reads a strong envelope as some other mass. Two changes remove
most of those readings:

- a species is fitted over every charge state the spectrum shows for it, grown outward from the
  charges that voted for it (a 68 kDa protein spanning z = 23-58 had been fitted over z = 32-60,
  leaving its low charge states for artefacts to explain);
- `drop_shared_evidence` removes a component when 80 % of its fitted intensity lies within two
  envelope widths of a stronger column of a stronger component.

The linker also drops a species whose elution profile peaks on the edge of its region (a template
soaking up a neighbour's tail), unless it is the region's strongest species.

## 9. Learn only what a deterministic method cannot do

The other record repeatedly found that information present in the input was not used by the
network (a probe on hand-made geometry features beat the network by 0.15-0.23 AUC, §86-§100),
that injected feature channels became crutches, and that a heuristic charge-family proposer
lifted end-to-end recall from 0.0002 to 0.15 without any retraining (§70). Here the network's
role is the one place where the deterministic comb is measurably weak: assigning charges to
overlapping and low-abundance envelopes. The benchmark now shows exactly how much there is to
gain there (section 11).

## 10. Process: short loops, cheap probes, real data early

The last three weeks of the other record produced no scored result. Gates required bitwise
determinism, and a one-ULP difference between debug and release builds took more than 200
review and brief files to bisect without finding a mechanism. A 241 KB pre-registration went
through five revisions and never ran. Fixed CPU budgets combined with a rule against shrinking
any experiment ended two routes as "resource no-go" before any science was measured. Earlier,
fleet training, installers and inference-backend matrices were built before any model worked,
and three production runs were spent on collapsed heads. What paid off was the opposite: probes
that needed no training (§49, §54, §57, §68, §73, D109), second seeds that reversed single-seed
results, and a rule that a zero result means checking the tool chain first.

Here, every change above was measured on the same benchmark within minutes of being made, and
training is preceded by the checks in `lcmsdeconv.nn.checks` (degenerate solutions, an overfit
test on eight samples, charge accuracy by band with harmonic and neighbour error shares).

## 11. Test on real data as soon as there is any

**There.** The record registered a public GalNAc-oligonucleotide data set in its final weeks and
never evaluated it; no network or decoder result on real data exists in it.

**Here.** Four real Orbitrap runs of oligonucleotide mixtures (with and without a GalNAc
ligand, ion-pairing RP and HILIC) were processed after the synthetic work above and checked
against a reference built from resolved isotope spacing alone. Six failures appeared that the
synthetic benchmark had no way to show:

1. The main species of a run was dropped as "flat background": a broad elution peak that fills
   its own region has a maximum under three times its median, which was the test. Flatness is
   now judged with chromatographic context beyond the region, and a species split over adjacent
   regions is merged.
2. Singly charged background in the column wash (dozens of ions with 1.003 Th isotope spacing)
   formed false 10-60 kDa charge ladders - most of the 85 species the original code reported.
   Where the instrument resolves 1.00336/z, a charge state now counts only if its isotopologue
   positions are populated; where it does not, the envelope must be a continuous hump.
3. Low-charge species (z = 3-4) have ladders too short for the comb, so a 7 % impurity never
   became a candidate. On high-resolution data the comb now reads the charge from isotope
   spacing. On time-of-flight synthetic data the same evidence cost precision (39 % against
   16 % without it for the polymer preset), so it is on by default only for Orbitrap and FT-ICR
   methods.
4. With resolved isotopes, the envelope's centre can fall in the valley between two
   isotopologues; a signal check at that one point rejected real species.
5. Fitting every combination of five ion-pairing adduct types let neighbouring species trade
   intensity; only adduct types the spectrum evidences are now fitted.
6. Vendor-thresholded Orbitrap profiles keep 1-3 % of the m/z axis, which breaks the noise
   estimator's assumptions. A threshold-based estimate was tried, measured against the reference
   with everything else fixed, and reverted: it added junk and lost a real impurity.

On the four runs, reference species found went from 2 of 22 to 9 of 22 and other reported
species from 85 to 19. The synthetic generator still has no singly charged background in its run
tier, which is why failure 2 never showed there.

**Then a real truth table.** The runs are those of a published study (Tutiš et al., Anal. Chem.
2025, 97, 15717-15726, https://doi.org/10.1021/acs.analchem.5c01407), whose Table 1 gives the
average masses of a GalNAc-conjugated phosphorothioate 16-mer and of the impurities spiked into
it at 2 % each. Scoring against those (`scripts/validate_oligo_runs.py`) instead of the
home-made reference changed three conclusions:

1. **The reference was biased and partly blind.** Masses built from resolved isotope positions
   sat 0.33-0.35 Da below the theoretical average masses (5161.91 against 5162.25 Da for the
   non-conjugated product), and the reference only looked inside the strongest TIC bands, where
   none of the separated impurities elute. Against it the pipeline's masses looked "0.1-0.5 Da
   high"; against theory the species of the IP-RPLC conjugate run are within 0.1 Da (conjugate
   +0.06, non-conjugated +0.01, n-1 +0.03, n+1 +0.10 Da), except the PS-to-PO form under the
   conjugate (-0.21 Da). The noise-estimator change rejected in failure 6 was judged
   against the same reference, so that verdict is open again. A reference built with the same
   kind of reasoning as the system under test inherits its blind spots.
2. **Impurities were lost to a displaced candidate, not to noise.** In the IP-HILIC runs the 2 %
   impurities were picked, but 4 Da off their mass: an intense unrelated peak that sits where an
   envelope's isotope ladder would continue inherits the ladder's charge from the isotope-spacing
   evidence and outvotes the envelope in the mass histogram. The gate let a candidate's envelope
   search travel 400 ppm, 2.6 Da at 6.4 kDa. With isotope-read charges it may now climb to the
   envelope's centre from anywhere within the envelope's half-width: 9 to 13 of 20 known species,
   among them the non-conjugated product and its deaminated form (one dalton apart, separated in
   time by IP-HILIC) and its PS-to-PO form. On time-of-flight synthetic data, where this failure
   cannot occur, the same change cost precision (oligonucleotide preset 27 % to 16 %), so it
   applies only when charges are read from isotopes.
3. **One charge state can be enough, at a price.** Under triethylammonium acetate the 6.8 kDa
   compounds carry mostly z = 4, so a 2 % impurity shows one charge state above the noise and the
   two-charge-state rule rejects it. When isotopes are resolved, their measured spacing fixes the
   charge. Accepting such species (`deconvolution.single_charge`) found 2 more of the 20 known
   species but added about 16 reports that no extracted-ion check could confirm, so it is
   available and off by default. Its first two versions made things worse, and only the truth
   table showed it: a comb test with a tolerance of a fifth of the spacing fits z - 1 and z + 1
   over a few isotopologues as well as z, and single-charge candidates admitted into the same
   capped candidate list crowded real ones out (fewer species found with the option than
   without). The charge is now read from the measured spacing of the envelope's peaks, and
   single-charge candidates have a separate allowance.

## 12. Where this leaves the package

See `docs/benchmarks.md` for the tables.

What remains open:

- On the real runs (details in `benchmarks.md`): in the IP-HILIC conjugate run the n+1
  impurity is still missed, the non-conjugated impurity elutes where the TIC shows no band so no
  region is formed, and the deaminated impurity (one dalton above its parent, eluting just
  before it) is absorbed by the parent's template. Abundances next to a strong species are
  unreliable (the PS-to-PO impurity under the conjugate in IP-RPLC is reported at 0.01 % against
  about 5 % by extracted-ion chromatogram). A +201.4 Da form accompanies every major species and
  is reported separately. Average masses are reported; a monoisotopic pattern fit would also
  separate a deaminated form from its parent inside one envelope.
- On synthetic Orbitrap data (`--presets oligo_orbitrap`) precision is 30 % at the real runs'
  resolving power, with 26 unexplained false species in four runs. (It read 13 % while the
  preset rendered 30 000-63 000 and decoded that with a method for 140 000; see section 13.)
- Close impurities inside an unresolved envelope (deamidation at +1 Da, oxidation at +16 Da on
  a 75 kDa protein) are physically unresolvable at time-of-flight resolving power and are
  reported as part of the main component.
- Polymer oligomers one repeat unit apart (44.026 Da for PEG) differ from two sodium
  exchanges by 0.06 Da, so adduct merging and oligomer series remain ambiguous by mass alone.
- The charge network: LadderNet beats the comb on synthetic data but not yet on the real runs;
  see section 13 and `gpu_training/README.md`.
- Tiny per-frame intensities shrank linked species' masses toward zero (a 667 Da report in an
  oligonucleotide method whose range starts at 1 000 Da); fixed in `link_species`, found only
  because a network produced many more such frames than the comb.

## 13. The charge network: what CPU probes can and cannot show

Probes on the full 32 768-bin window: 200 training frames of the default mixture, 24 held-out
frames, batch 4, 450 steps (about eight minutes on two CPU threads), intensity-weighted charge
accuracy on bins that are ions:

- The comb alone, read as an argmax on the same bins, gets 53 % (z = 1: 0 %, z = 2-5: 13 %,
  z = 6-15: 60 %, z = 16-40: 62 %).
- A network trained from scratch reached 12 %. It predicted z = 1 almost everywhere: per bin,
  singly charged background is the most common ion class and is learned first; 33 of the 39
  charge classes present were never predicted.
- The same network starting from the comb (`build_model(..., comb_prior=True)`: the comb
  evidence channels enter the charge logits as fixed offsets and a zero-initialised head learns
  a correction) starts at the comb's 53 %, falls to 39 % within 150 steps and is back at 53 % at
  step 450 with a different profile: z = 1 from 0 to 97 %, z = 2-5 from 13 to 47 %, z = 6-15 from
  60 to 45 %, z = 16-40 from 62 to 52 %.
- Accuracy with background as one more class read 31 %, 22 % and 27 % for that model and fell
  while the accuracy on ions recovered: an argmax over a background class splits the votes, the
  other record's §46. The metrics (`charge_accuracy_given_ion`) and inference (the top two
  charges are taken among the ion classes) now condition on the bin being an ion.

So a residual learner on top of the comb keeps the comb's accuracy and learns the low charges
the comb cannot read; whether it ends above the comb on high charges needs runs two orders of
magnitude longer than this machine allows.

### Give the network the physics: LadderNet

The U-Net above had to discover, from a few hundred spectra, offsets thousands of bins long and
different for every charge. LadderNet (`build_model(..., arch="ladder")`, `docs/training.md`) is
given them: for every candidate charge it reads a learned description of the spectrum at the
bin, at the bin's z ± 1..3 neighbours and at the two positions only a reading at 2z would fill,
and a small network shared by all charges scores the reading. It starts as the comb (its learned
part is initialised to zero; a test checks the equality) and has 9 500 parameters.

Trained on two CPU threads for 45 minutes (240 synthetic spectra, 12 epochs), with singly
charged ions either ignored by the loss or absent from the spectra, it read the right charge on
fresh spectra at charges two and up for 0.88-0.90 of the ion bins weighted by log-SNR, against
0.65 for the comb as the pipeline runs it on time-of-flight spectra (0.83-0.87 against 0.36 on
Orbitrap ones), and for 0.95-0.96 of each spectrum's ion intensity against 0.84. Pooled over
spectra, so that the strongest few decide, it still led at z >= 2 (0.94 against 0.86) but not
at z = 2-5 on time-of-flight data (0.71-0.73 against 0.87): a 32 768-bin window cannot hold the
nearest neighbours of a doubly charged ion, which the comb, reading the whole spectrum, sees.
Getting that comparison fair took four corrections, three of them found by independent reviews:
the two variants had first been validated on different spectra; training printed as its bar
the comb's answer in feature channel 3, computed inside the crop (0.57 where the pipeline's comb
scores 0.64); the "pipeline comb" of the first fair measurement had itself been run on the crop
(0.46); and the evaluation drew each spectrum's mixture component and resolving power from one
random number, so that each component covered a sliver of its range. `lcmsdeconv.nn.baseline`
now runs every estimator, during training and in `scripts/eval_charge_bins.py`, the way
`process_run` does, on spectra drawn as training draws them. In the synthetic benchmark that
became species: precision 34 → 52 % (protein), 28 → 39 % (oligo), 30 → 48 % (Orbitrap oligo at
the real runs' resolving power; 14 → 47 % at the too low one first rendered, see point 4 below),
impurity recall up, charge-harmonic false species almost gone (`benchmarks.md`).

On the four real runs it was not better than the comb: one known species fewer and 13-21 more
unconfirmed reports, almost all tiny, 1.3-3.9 kDa, in the IP-RPLC column wash - singly charged
background read as z = 2-3. What it shows:

1. **Structure beats data at this scale.** A 9 500-parameter network with the right inductive
   bias learned in 45 CPU minutes what a 1 M-parameter U-Net could not approach.
2. **Per-bin accuracy, synthetic species and real species are three different measurements.** A
   23-25-point per-bin gain (log-SNR weighted; 11-12 points of intensity per spectrum, and a
   loss at low charges on the strongest spectra) became 10-30 points of synthetic precision and a
   small loss on real
   runs, because of what the synthetic spectra do not contain: the dense singly charged
   background of a column wash. Which per-bin number to quote is itself a choice to make
   explicit: log-SNR weighting rewards reading weak ions, intensity weighting reading the
   strong ones the decoder's votes rest on.
3. **Leaving a class out of the loss is not neutral at inference.** A network never penalised
   for calling a singly charged ion multiply charged does so on data full of them. "No reward
   for z = 1" is a good first stage - it removed the collapse onto z = 1 that the U-Net probes
   showed - but it needs a second stage that trains on z = 1 among dense background. A short
   CPU version of that stage (`--singly-charged keep --background-ions 200`) cut the unconfirmed
   reports from 28 to 20 and lifted synthetic precision further (protein 65 %, Orbitrap oligo
   57 %), the right direction; the full version is the GPU run in `gpu_training/`.

4. **Check the generator's instrument against the real one.** The synthetic Orbitrap spectra
   were rendered at 30 000-63 000 resolving power (m/z 200); the real runs were recorded on a
   Q Exactive Plus at 140 000 (155 000 measured from their peak widths), and the synthetic
   benchmark decoded its spectra with the real runs' method, which assumes 140 000. Rendered at
   the real resolving power, the comb's synthetic Orbitrap precision is 30 %, not 14 %. The same
   mismatch made the comb's isotope-spacing charges look broken per bin (on 16 spectra at
   30 000-63 000, right for 0.19 of the intensity at z >= 2, against 0.80 without them); a guard
   that repaired the per-bin score cost three to five of the thirteen paper species found on the
   real runs and was not kept (`benchmarks.md`, "Isotope-spacing charges"). The training mixture
   now renders Orbitrap spectra at 35 000-280 000, a quarter of all spectra instead of 15 %.
   CPU probes trained on it are no better on synthetic Orbitrap spectra; on the real runs the
   first stage drops four column-wash reports (28 to 24) and finds one more SI-listed impurity,
   while the second, in six short epochs, learned singly charged ions far less than before and
   kept 27. The GPU run, thirty times longer, decides.

No model is shipped and the comb remains the default. A trained network replaces it only when
it finds at least what the comb finds on the four real runs without adding reports.


## 14. Read vendor-thresholded data as what it is

Instrument software writes profile scans thresholded: short segments around peaks, the noise
between them removed. Two shortcuts in the resampling read the gaps as signal. Resampling gave
a segment's edge point half the gap beside it as its width (up to thousands of grid bins), and
region sums interpolated every scan onto the axis of the scan with most points, ramping across
each scan's own gaps and dropping what lay off that axis. On the real Orbitrap runs the two
raised the noise estimate of a region's grid view 40-180-fold. The fix is local - a sample stands
for its own spacing next to a gap, and a region's grid view is the sum of its scans resampled one
by one - but everything downstream had been tuned on the inflated noise, and four things it had
hidden came out at once:

1. **Above the noise of thresholded data almost everything is signal.** Thermo's own per-peak
   noise (`opentfraw`'s label data) puts the threshold at about 1.6 times the noise and the
   median written peak at about three: what the decoder sees above the corrected noise is real,
   mostly low-level ions. With the corrected noise the envelope gate passed ten times as many
   candidates (39 to 468 in one run) and the pipeline reported 400 species instead of 39. No
   noise threshold separated them: scaling the noise back up 3-100-fold traded the new species
   found for false ones without a clean operating point. What separates a species from a ladder
   of unrelated ions is physics the noise had stood in for: isotopes of the right size (a comb
   position needs a third of the height the pattern predicts, not just signal) and charge states
   that co-elute over the region's scans (Pearson r >= 0.5). The first removed the dozens of
   9-22 kDa column-wash "species", the second most of the rest.
2. **Weighting a fit by each bin's own value biases strong species low.** With the noise floor
   40-180 times too high the least-squares weights were nearly flat. Corrected, they followed
   each bin's own intensity, and a template a fraction of a bin off its peaks was scaled to a
   fifth of the main species' envelope while impurities kept most of theirs (Neyman's
   chi-square bias). The variance is now taken from the local maximum over one peak width.
3. **A class's average composition misplaces resolved isotopologues.** The `ps_rna` pattern
   puts the real runs' 5 kDa product's isotopologues 0.11 Da high: one grid bin at charge four,
   which no least-squares fit absorbs (plain least squares too scales such a template to a third
   of the peaks). Templates are now slid onto the resolved isotopologues before the fit; the
   reported mass is unaffected.
4. **Adduct ladders of neighbours coincide.** The PS-to-PO impurity plus potassium is the main
   species plus sodium to 0.07 Da; once the corrected noise let the potassium adduct be
   evidenced, the impurity's adduct column took the main species' sodium adduct. Such a column
   now goes to the stronger species.

With all four, the comb finds 15/20 of the paper's species and 15/29 SI-listed impurities on the
real runs (13/20 and 5/29 before), with impurity amounts within about 25 % of their extracted-ion
areas under IP-RPLC and a factor of two to three under IP-HILIC (the main IP-RPLC impurity was at
0.2 % against 3 %, and IP-HILIC's main species at half its 2 % impurity), at the cost of more
reports no extracted-ion check confirms (43 against 7, most below 0.1 % of the main species, some
single-charge-state species the check cannot confirm). Synthetic precision rose for every preset.
The lesson is the one of section 3 from the other side: a threshold tuned on a distorted input
encodes the distortion, and fixing the input means re-deriving every test that leaned on it - which
here meant replacing a noise threshold with the physics it had been standing in for. Networks
trained before the fix saw the distorted grid views and must be retrained.

## 15. Make an amount mean the same thing for every species

On the IP-HILIC runs the pipeline reported the conjugate's n-1 at 6.2 % of FLP where an
extracted-ion ratio said 3.0 %, PO at 2.7 % against 3.6 % and n+1 at 5.6 % against 7.9 %. Three
independent investigations - a better reference, the region boundaries, the missed species -
took that apart:

1. **The yardstick was wrong as often as the pipeline.** A +-1.2/z Th window holds two or three
   isotopologue peaks depending on where a species' average mass falls between them: it caught
   43 % of n+1's envelope and 31 % of FLP's, so the crude n+1 ratio was 1.39 times too high
   (2887.4 Da in the NC run: 1.87 times). Windows over the whole envelope, every charge state that
   carries the species, its own elution bounds and fitted overlaps give n+1 5.8 %, PO 2.9 % and
   n-1 3.0 % of FLP. Check the reference before blaming the instrument it measures.
2. **Adduct forms were counted for some species and not for others.** A region attaches an
   adduct type to its species only when it shows that adduct next to its strongest species. The
   test compared an average mass with a monoisotopic shift (K - H is 0.13 Da heavier on average,
   half the tolerance) and measured the adduct form's mass differently from detection, so in
   FLP's region, where one charge state dominates, it found none: FLP's +K, +Na, +Na+K and further
   forms, together 90 % of FLP, came out as ten species of their own, while n-1's region attached
   n-1's (55 % of its signal). n-1 was reported at twice its share. Now the test uses average-mass
   shifts and detection's own measurement, adduct-form species are folded into the co-eluting
   species they belong to, and shares compare base forms, which every species has measured the
   same way. Forcing sodium and potassium into every region instead lost the NC run's PS-to-PO
   impurity and NC - dG: PO-NC plus potassium is NC plus sodium to 0.07 Da. Base forms carry
   their own bias where adduct loads differ - the main species' load falls at its own apex - so
   base-form shares understate impurities by the ratio of the loads: up to 1.2 on the conjugate
   run, 1.7-1.9 on the NC run.
3. **Templates must stay where they were aligned.** At 140 000 (m/z 200) an Orbitrap peak at
   m/z 1700 is narrower than a grid bin, and a template a quarter bin off its peaks is fitted
   9 % low. Re-measuring a component's mass after the fit moved its templates by the change, and a
   candidate re-centred from an adducted form onto its base kept the adducted form's alignment;
   the main IP-RPLC species, re-centred that way, was fitted 5 % low, and every impurity's share
   moved with it. Fixing the second exposed a third defect: a weak candidate stepped down two
   adducts landed on the flank of the NC product's envelope, its freshly aligned template fitted
   NC's isotopologues, and NC's own component in its tail region was dropped for two components
   3 Da either side of it. A re-centred form must now be an envelope centred where the walk lands.
4. **Region boundaries cut species.** The rule that a species peaking at its region's edge is a
   neighbour's tail dropped the NC run's n-1 and the conjugate's deaminated form, which elute at
   the valleys where two regions meet; it now applies only to a region's unshared sides. The
   scan on a shared boundary belongs to both regions, and adding its two readings put the PS-to-PO
   impurity's apex on the boundary; it is counted once.
5. **A review found defects in the fixes themselves.** Templates were aligned over 1.5 grid
   bins: at the 2·10⁻⁵ grid that is 0.04 Da for a 1.2 kDa shortmer, too little to reach its
   peaks, so its amount came out at 0.01 % instead of 4.2 %; the window is now 0.3 Da, whatever
   the mass, and four more SI-listed impurities are found. The re-centring walk stepped by
   monoisotopic shifts and judged where it landed by one charge state, which a neighbour's
   isotopologue can carry; it now steps by average shifts and takes the median of the charge
   states that agree. Adduct forms were folded after region pieces were merged, so a form split
   over two regions was compared with a merged parent whose trace no longer matched; pieces are
   now folded first, and the merge keeps one reading per scan. A fold within a dalton by any
   library adduct put a +201.4 Da impurity under its parent as two TEA and a peptide's oxidation
   under it as ammonium: the tolerance now follows the instrument (2.5 peak widths, at least
   0.6 Da), mobile-phase adducts fold only where some region of the run shows them, and a
   difference that a known modification of the species' class explains as well (an n+1 among
   them) is not folded. Merging region pieces joined two 4842 Da isomers eluting either side of
   a boundary (12.80 and 13.30 min); pieces of comparable height with a deep valley between them
   stay apart. Smaller fixes: the apex is read on the base form's trace, species ids are unique
   over every linked track, the region sum is a sweep over the union axis instead of one pass
   per scan (8.5 times faster), and training scans are thresholded as vendor software writes
   them, as segments around every point above the threshold. On synthetic runs this costs up to
   four false species per preset, most of them artifacts the permissive fold had hidden under
   adduct types the generator never added (`benchmarks.md`). A second review found the sweep
   turning a repeated m/z (converted files can repeat a point, vendor files do not) into NaN for
   the rest of the axis and leaving rounding residues where no scan measured anything, a region
   of profile and centroided scans gridded as centroids whenever its largest scan was centroided,
   scans calibrated a few ppm apart added index by index, and training segments padded to seven
   points where the real ones are five or six. It also found the valley test reading frames
   without a reading (a gap the linking bridged, a frame whose fit gave the species nothing) as
   valleys, so that one band was reported twice; merged adduct forms carrying labels beyond the
   library's limits (+3K where two of a type are listed), which were neither fitted frame by
   frame nor judged at their own mass; a fold's flag lost when region pieces merge; species of
   one mass merged across polarities whatever their retention times; and adduct types given as
   formulas, which the documentation offers, failing outright. Fixed, these change one real-run
   report (a 1900.08 Da piece joins its species) and no synthetic one but a legacy-generator
   duplicate.
6. **What did not work.** Sliding templates over half an isotope spacing instead of 1.5 bins
   (the IP-RPLC main species fell to 69 % of another report), adducts forced into every region
   (above), one template bank over abutting regions (the deaminated NC, 0.98 Da above NC,
   took a quarter of NC), following the alignment overlap past the 0.3 Da window's edge to
   half an isotope spacing (a 1.9 kDa shortmer read 0.34 Da light was quantified, but fourteen
   candidates walked, some onto the peaks of the IP-RPLC main species' adduct forms, and the
   main species was lost), and folding by the whole adduct library rather than the adducts a
   run shows: it takes back one synthetic false species per oligonucleotide preset, and on the
   IP-HILIC NC run, where no region shows an amine and NC + TEA is not reported, folds a 1.9 %
   species 138.75 Da above NC into it as TEA + potassium.

The lesson extends section 5: an amount is only comparable when every species' amount is made
the same way, and a comparison against a reference is only as good as the reference.
