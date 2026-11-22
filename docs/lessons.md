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
1.36×; monoisotopic against average masses (several daltons for an oligonucleotide).

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
4. Oligonucleotide charge states were too low for ion-pairing LC-MS; the model now charges them
   higher.

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
are now kept when they stand ten times above the trace's noise (three times with the
oligonucleotide methods), with relative floors of 0.2 % of height and 0.05 % of area, and
neighbouring peaks become separate regions split at their valley, so one region's summed spectrum
is dominated by one elution band. The oligonucleotide methods lower the height floor to 0.1 %:
at 0.2 % it deleted the region holding a 0.04 % shortmer of the IP-RPLC NC run (3206.4 Da of the
SI), whose TIC peak - mostly a co-eluting 3198.6 Da shortmer's - stood 3.47 times above the noise
at 0.197 % of the tallest: this section's lesson once more (section 17).

## 7. Measure on the raw data, with statistics that tolerate a neighbour

**There.** Apex-based masses, isotope folding that merged species one dalton apart, and an
uncorrected elution-tail bias (a few tenths of a dalton) cost the classical engines accuracy.

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

**There.** The guard that fixed the classical engine's best-known real-data artefact (a "species"
at half the mass of an oligonucleotide) was to drop a candidate whose charge-state peaks are mostly
claimed by a stronger candidate.

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
   formulas, which the documentation offers, failing outright. A third review found the fixes
   too broad in turn: empty scans read across however many (two isomers seen in two scans each,
   half a minute apart, joined; read across three at most, they stay apart, and a 0.000 %
   1900.08 Da sliver the wider rule had joined is reported again), every flag of a merged
   piece copied (a one-charge-state flag on a species of three charge states), formula adducts
   accepted that add no mass or repeat a listed one, the training padding left out of the resume
   fingerprint, and resample_profile still dropping a repeated m/z. Each review narrowed the
   last one's fixes; each fix now has a test that fails without it. The real-run reports are
   those of the first review's fixes, and the synthetic ones but for a legacy-generator duplicate.
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

## 16. Read a weak species' position off its own isotopes

n-2 of the NC product (about 0.1 % of NC) was missed in both chromatographies and the 1.9 kDa
shortmer 1883.3 of the IP-RPLC NC run was quantified at 0.00 %. Four parallel traces, each
replayed stage by stage on the real runs, found three different failures behind two symptoms:

1. **A charge state that failed the gate still steered the candidate.** The gate measures every
   voted charge state but kept them all in the candidate's set, failed ones included. Under
   IP-HILIC n-2's z = 9 window held an unrelated peak 30 times taller than n-2; the adduct walk
   summed it in, stepped n-2 down a sodium onto a phantom at 4482 Da, and that phantom's trace was
   later dropped as flat. The walk now reads only the candidate's own charge states: those the gate
   passed, and those found beside them for the fit (since the second review below). Raising the
   200-candidate cap, the first suspect, was neither needed (n-2 enters the second pass) nor
   sufficient (with 400 it meets the same walk), and added 13 small reports.
2. **A mass read half a spacing off misplaces the templates.** Under IP-RPLC a 4497 Da species six
   isotope spacings below n-2 pulled its mean-shift mass 0.55 Da light; 1883.3 is read 0.34 Da
   light, from an envelope 2 Da lighter on the same comb (-0.15 Da), the union axis's point density
   (-0.12) and its own isotopologue heights (-0.07). Their templates needed +0.49 and +0.36 Da.
   1883.3's stopped at the 0.3 Da window's edge, 0.07 Da short of its isotopologues, and was fitted
   300 times low; n-2's overlap search, led by its failed z = 11, went to the far edge (-0.29 Da,
   0.22 Da beside a wrong isotopologue), where z = 11 carried two thirds of its fit and put its
   apex 0.9 min off. The phase of the candidate's own resolved isotopologues, folded onto one
   spacing, gives the shift directly (within 0.01 Da on synthetic envelopes, and n-2's z = 3 and 4
   agree to 0.003 Da); it is now taken where the overlap search ends on its edge.
3. **A wider search slides onto neighbours, and the comb needs guards too.** The comb phase reads
   whatever comb dominates the windows. The adduct dedupe had renamed FLP's candidate to a weak
   partner 43.5 Da lighter (two sodium to 0.46 Da); aligned to its own comb that partner took FLP's
   peaks as its +2Na form and FLP was lost - exactly as the overlap walk had lost it. A renamed
   candidate has no gate charges of its own, so the comb leaves it alone. Adduct forms of accepted
   species are left alone as well: their masses are pulled by neighbouring forms by up to three
   quarters of a dalton, and the comb then takes the tooth one spacing off - NC + 3Na of the
   IP-HILIC NC run jumped from 0.56 % to 10.5 %, its two teeth fitting the envelope equally to
   0.6 %. On the final code, which takes the better tooth, the two teeth's overlaps above 4 kDa
   differ by 0.03-26 % on the four runs, and right and wrong picks share that range - n-2's correct
   tooth wins by 2.5 and 2.9 %, FLP + K's wrong one by 0.75 % - so no margin rule separates them
   (section 17 chooses the tooth by a joint fit with same-comb neighbours instead, which takes
   FLP + K's own tooth).

Now n-2 is reported in both runs (0.08 % under IP-RPLC against 0.15 % by its windows - 0.074-0.078 %
by its own isotopologues, the windows also holding a same-comb neighbour's upper isotopologues,
section 17 - and 0.14 % under IP-HILIC against 0.11 %), 1883.3 at 0.08 % (0.09-0.10 % by extracted
ions), all 20 paper species are found,
and nothing else on the four runs moves by more than a few tenths of a percent but species that
were edge-pinned (one, reported as an 1831.08 Da shortmer, from 0.004 % to 1.1 %: it was not one,
below). The cost was one false species more or less per synthetic preset.

A review of that version found it still reading evidence the candidate did not own, each fixed. The
comb was read on the spectrum as it came, so a species fitted earlier in the pass still lay in the
windows; it is read on the residual. One coherence figure over all windows let one strong window
speak for the rest - a foreign peak in one gate-passed window moved the comb - so each state
carrying a tenth of the weight must agree within 0.1 Da; and the states the gate rejected, which
the overlap search still read, are no longer read by it. Where the measured mass lay more than half
a spacing from the truth the nearest tooth was the wrong one, and falling back to the window's edge
fitted neither; the tooth whose template fits better is taken. And the adduct walk compared a
candidate placed by its comb with a lighter trial pinned at the edge, so the candidate always won
and adduct forms no longer walked to their base; the trial is placed by the same rule. The better
tooth quantifies species whose templates the edge had misplaced: a 5129.63 Da species of the
IP-RPLC NC run, whose template the comb puts at 5130.18 Da (two phosphorothioates as phosphates,
32.1 Da below NC; the reported mass is 0.5 Da lighter), went from 0.006 % at 17.90 min to 0.36 % at
its extracted-ion apex, 1.3 times its extracted-ion reference, and a 1900.08 Da shortmer (its
template at 1900.64 Da) reported as two 0.000 % slivers (section 15) to 0.11 %, 0.84 times its
reference.

A second review found that version still reading what it should not, and the better tooth taking
false species along with the real ones; each finding is fixed, and each fix but three small ones
(the walked masses kept across passes, the walk trial's exemption and its added states) has a test
that fails without it:

- The walk read every state the gate had not rejected. A voted state the gate could not measure, or
  one a duplicate merged in by mass brought along (nearly all of those had failed the duplicate's
  own gate), would put item 1's phantom walk back: so a test that models it shows (on the IP-HILIC
  run n-2's z = 9 fails its own gate, and the walk kept n-2). The overlap search and the walk now
  read only the candidate's own states - those the gate passed, for it or for a duplicate, and
  those found beside them for the fit - and merged duplicates bring their gate's verdicts. A
  6752.33 Da species, whose templates an isotope comb of the wrong charge under a rejected state
  had pulled to the far edge of the window, is now reported (0.08 %, EIC-confirmed; its templates
  sit at the near edge, held there by the adduct-form exemption), and a report labelled n-1's +Na+K
  form before (6507.77 Da, 0.004 %; by its isotopes n-1 + 58.0 Da, 2 Da short of that form) is a
  6507.42 Da report of 0.023 % (1.3 times its extracted-ion reference), unconfirmed, placed on a
  comb it no longer reads against a rejected state.
- "At least two states agree" counted a state too weak to be checked: a second state with 4-7 % of
  the weight, 0.12-0.22 Da off the combined phase, let a comb read off one window through (2213.17
  Da of the IP-RPLC NC run, whose z = 3 holds 5.6 %, went from 0.028 % to 0.039 %). A weak state
  now counts only where it agrees (1883.3's z = 3 does, to 0.02 Da).
- The walk's trial was placed without the adduct-form exemption its candidate had; it has it now.
  NC + 3Na, kept at its edge as a form of NC, still joins NC: its trial lies on NC's own mass,
  which is no form of NC.
- That walk removed the mass through which NC's +5Na and +3Na+K forms, beyond the two of a type the
  fit allows and the three the adduct-form exemption covers, had been recognised as adduct forms;
  the better tooth, by 0.16 %, placed them on their comb and they were fitted whole - a 3.3 %
  impurity under IP-HILIC made of NC's own sodium ladder. A mass the walk moved a candidate from
  now anchors the forms above it, and they are back at 0.21 %, still a false report
  (`benchmarks.md`).
- The better tooth also lifted an "1831 Da species" from 0.001 % to 0.27 % of NC under IP-RPLC
  (1.1 % under IP-HILIC since the first version): the singly and doubly charged ions of a 915 Da
  species, then below the method's mass range, read at charges 2 and 4 (section 17 extends the
  range to 800 Da, with z = 1 counted beside z = 2). At 1.2-1.9 kDa the isotope
  pattern has three major isotopologues (five at 2.5-3.4 kDa) and a comb at half the charge fills
  two (three) of them, enough for the gate's isotope test; a placement that fills only every other
  position no longer counts. Rejecting a state that is as coherent at half its charge as at its
  own, the first idea, lost three SI-listed shortmers, whose odd charge states are weak.

With these the four runs gave 229 reports, 116 of them EIC-confirmed and 51 unconfirmed (229, 117
and 50 before the second review), with n-2, 1883.3 and every paper species unchanged; on the
synthetic presets recall is unchanged, the oligonucleotide preset reports one false species fewer
under each charge assignment, and the Orbitrap preset's precision stays at 76 % (68 % before this
section's changes).

A third review checked every report the whole section had moved against the runs themselves, with
extracted-ion references and the positions of the isotopologues. Under IP-RPLC the moves were
towards the references but one - a 2980.4 Da species of 0.008 % lost its 0.0013 % report
(found again, at 0.007 %, by section 17's between-teeth rule) - weak shortmers from 0.00-0.73 to
0.83-1.55 times theirs, 5364.15 Da from 0.25 %
to 0.58 % against 0.53 % (its extracted-ion check now fails it: its apex ties between two scans,
and from the earlier one the check reaches a stronger form's tail). Under IP-HILIC the comb had
also placed sodium and potassium forms deeper than the exemption's three of a type: they share
their parent's comb exactly, and fitted whole DA-NC + 4Na went from 0.79 % to 1.15 % - taking part
of DA-NC's own sodium ladder with it - NC + 6Na + K from 0.14 % to 0.44 %, and a shortmer's +4Na
form, folded before, was reported at 0.065 %. Such a form now keeps the edge where its comb lies
within 0.03 Da of an accepted species' comb moved by the ladder's monoisotopic shift, its mass
within 1 Da of that species plus the ladder, and the region evidences sodium; DA-NC + 4Na and NC +
6Na + K are back at their earlier amounts, still reported apart, and the shortmer's +4Na form is
folded into its parent again, with nothing else on the four runs or the synthetic presets changing.
Without the sodium condition a 4949 Da report of the IP-RPLC NC run, where no sodium is evidenced,
fell back to its edge amount: its tooth lies on NC - dG's comb moved by six sodium, to 0.003 Da, by
chance. The four runs now give 228 reports, 115 EIC-confirmed and 51 unconfirmed.

The lesson: a template must be placed by evidence the species owns. A slide over everything nearby
finds the best fit to whatever is there; the gate's per-charge measurements already say which
envelopes are the candidate's, and each later step that places or moves its template should read
only those (the fit itself still covers the rest: restricted to them it loses species). And a
change that lets a template reach further also lets it reach the wrong thing: the comb and its
better tooth quantified real species the window's edge had misplaced, and false ones with them - a
charge error and adduct forms that the edge had held at a fraction of their signal by luck. Every
report a change moves needs checking, not only the ones it was made for.

## 17. Fix the known issues, leave the adduct decisions alone

After section 16 `benchmarks.md` listed what was still wrong on the four runs: 23 of the 29
SI-listed impurity elutions found among 228 reports, 51 of them unconfirmed; n-2 read 0.49 Da light
and 0.27 Da heavy, and under IP-RPLC at half its reference; 1883.3 dropped under IP-HILIC; 1908.3
and 2887.4 of the IP-RPLC NC run missed (and, unlisted, 3206.4); NC reported again at 20.0 and
25.4 min; n-1 of the IP-HILIC NC run split over two regions and read 0.88 Da heavy; a 915 Da species
below the method's range; a non-covalent dimer reported at 0.008 %. The rule for this round: adduct
forms stay listed as species of their own, and adduct handling waits for a module of its own. Six
investigations, each replaying its issues stage by stage on the runs, proposed fixes; an independent
verifier reproduced nine of the ten fixes adopted (all but the chain merge, an extra finding),
checked every report they moved against the raw scans, and for seven of them supplied the variant
that was adopted. The ten were integrated one commit at a time, the four runs and the tests run on
the whole chain after each; the five synthetic benchmarks ran on each fix alone and on the
integrated chain, and again after the review fixes and after item 10. Three reviews (the code, the
outputs, the scope and these documents) and two follow-ups came after, and a last review of the
result.

1. **A reference must own its signal.** n-2 under IP-RPLC sat at 0.08 % against a 0.15 %
   reference, half of it apparently lost. The reference's windows (2.5 envelope widths around each
   charge state, over n-2's elution) also held the upper isotopologues of a 4497.7 Da species six
   spacings below n-2 on its comb and eluting with it, about half of the windows' signal. Split
   from its neighbours on the comb's positions scan by scan, n-2 is 0.074-0.078 % of NC over its
   whole elution and 0.065-0.068 % within its region; the same split gives n-1 2.08 % against the
   windows' 2.03 %. The pipeline was right (1.03 times its own isotopologues) and the reference
   was not. Under IP-HILIC n-2's 0.14 % against 0.11 % is its own signal: its trace follows its
   isotopologues scan by scan, at 1.17-1.31 times them, where NC - dG and 3847.4 of that run are
   1.10. What n-2 does lose is 10-11 % of its IP-RPLC elution, in the next region, where its
   candidate's envelope search climbs 5.2 Da onto the 4497.7 Da species; recovering it needs that
   region's bank to carry its neighbour's species across the boundary, and would add 0.008 %.
2. **A floor tied to the tallest peak deleted a species again.** 3206.4 of the IP-RPLC NC run had no
   region: the TIC peak that would hold it - mostly a 3198.6 Da shortmer's, at 11.66 min against
   3206.4's 11.48 - stood 3.47 times above the noise, enough for the oligonucleotide methods'
   floor of three, but at 0.197 % of the tallest peak, 1.7 % under the relative floor of 0.2 % that
   section 6 had kept as a guard against ripple. The floor is now a method setting
   (`regions.min_rel_height`), 0.1 % for the two oligonucleotide methods and 0.2 % for the others,
   whose noise floor of ten decides anyway and which nothing here could validate: the synthetic
   generator makes impurities co-elute with their parent, and none of 112 synthetic runs changed a
   region at 0.1 %. On the real runs the noise floor then decides on three of four (0.17-0.49 % of
   the tallest peak); on the IP-RPLC conjugate run, whose noise floor lies at 0.076 %, 0.1 % binds
   and admits two more peaks. 3206.4 is found (3207.18 Da at 0.020 % against 0.042 % by extracted
   ions; 13 % of its elution falls before the new region), with a shortmer, its PO form and, in the
   conjugate run, a 2559 Da shortmer at 1.08 times its reference.
3. **A merge taken strongest first is order-dependent.** Pieces of one species in adjacent regions
   are merged strongest first, so a weak tail piece that touched only a middle piece not yet
   joined was compared with the strong piece's trace before it grew (a gap of 0.21 min against the
   0.2 min limit) and was left apart: the last pieces of 2653.32 and 2998.45 Da of the IP-RPLC
   conjugate run, each over three regions. The merge is now repeated until nothing more joins.
4. **Pieces and later elutions are judged by the isotope tooth read in the scans.** A region's
   summed spectrum mixes a band's front or tail with whatever elutes there and can read it a dalton
   off; the scans cannot: one molecule sits on one tooth of its isotope comb however a blend skews
   its average mass, and a deaminated form sits one tooth away (`refine.tooth_position`, the class
   pattern matched against each scan's own points). Two rules read it. A piece peaking on a shared
   boundary, read up to 1.5 Da off a stronger species that runs on past the boundary without a
   valley, is joined to it where both read one tooth, each by a margin of 0.02
   (`features.link.merge_boundary_pieces`): n-1 of the IP-HILIC NC run, a 1.87 % main piece read
   at 4833.86 Da and a 0.47 % tail at 4833.14 Da, is one report of 2.35 % against 2.45 +- 0.18 %.
   The verifier made the joined species keep the mass its own isotopes support instead of the
   stronger piece's, which would have reported n-1 0.88 Da heavy. The IP-RPLC conjugate run's
   0.02-0.05 % boundary reports, which the known issue had called a neighbour's front, read other
   teeth - species lighter than PO and FLP, one peaking on the boundary - and stay. The other rule
   sets the run's main species' later elutions aside (`features.link.set_aside_later_elutions`):
   NC's reports at 20.0 and 25.4 min (1.24 and 1.32 %) are its own tail, falling from 0.3 to 0.07 %
   of its height from 19 to 25 min, and its column-wash peak at 1.3 % of its height, the same
   molecule by its tooth at every time (5162.17 Da, margins 0.037-0.057), and are named in NC's
   flags; the flags name 2.6 of the 5.6 % of NC that elutes after its band. Its +201.4 Da form's
   wash piece goes with them, but only from a later piece that rises and falls: in the verifier's
   probe a slice of NC's tail had carried a later isomer of n-1 off with it.
5. **A window's trace judges whatever lies in it; a species' own isotopologues judge the species.**
   1883.3 under IP-HILIC was dropped by the co-elution test: its z = 3 window held one of its
   isotopologues and, 0.006 m/z beside it, an ion of the 1234.2 Da 4-mer's +Na form eluting earlier,
   so that window's trace correlated at 0.36 with z = 2 while its own isotopologues did at 0.97. A
   state that fails on its window is now judged again on narrow windows on the species' own major
   isotopologues and kept at r >= 0.9; judging the windows' failures on isotopologues at r >= 0.5
   reported 25 species more than at 0.9, mostly adduct forms of the main species (FLP + DIPEA at
   1.78 %), and at 0.8 a 0.30 % form of n+1 came in. The verifier found the rescue pairing the
   1580 Da shortmer's z = 1 ion with its dimer's z = 3 - 99 % of the reference's comb trace lay
   outside its own window - and added an anchor: the reference's comb positions inside its window
   must carry a quarter of the share the pattern predicts there. 1883.3 is found at 1884.51 Da and
   0.33 %, against 0.33 % by its own isotopologues, with two small species whose isotopologues
   co-elute at 0.92 and 0.99.
6. **Multimers go by their fitted amounts, and only on evidence.** With resolved isotopes a
   multimer's candidate scores like its monomer's: the 1580 Da shortmer's z = 2 isotopologues,
   read at z = 4, vote on teeth 2 Da apart that the smoothing does not merge, and its [2M-3H]3-
   dimer scored 0.74 of the shortmer, above the harmonic test's 0.6; section 16's parity rule had
   removed the accident that kept it out. `refine.drop_multimers` drops a component at n x M with
   less than 0.6 of the monomer's fitted amount where its own charge states (not multiples of n)
   carry under a quarter of its fit - a harmonic reading - or where its strongest own state
   co-elutes with the monomer's: the dimer (75 % of its fit on odd states, r 0.91 over 26 scans)
   and a 2469 Da reading of the 1234 Da 4-mer (9 %) go, nothing else on the runs. The verifier
   lowered the share from a half to a quarter, as a genuine 2M species carries about half its fit
   on odd states; the code review found the rule dropping wherever co-elution could not be judged
   (no frames, fewer than four scans, no profile in the monomer's trace), which regions of two
   scans allow in every bundled method, and it now keeps such a component unless it is a harmonic
   (with the co-elution test turned off, `coelution_r: 0`, only harmonics go).
7. **Below 1.2 kDa z = 1 is one of two charge states.** The 915 Da trimer (0.41 % of NC under
   IP-RPLC, 1.15 % under IP-HILIC) lay below the Orbitrap oligonucleotide method's 1 kDa floor; at
   m/z 400-2500 a species under about 1.2 kDa shows only z = 1 and z = 2. Lowering the floor alone
   found nothing; allowing z = 1 as it stood gave 283 reports instead of 228 and lost three
   EIC-confirmed species, because every envelope fills a z = 1 comb: NC's non-covalent dimer
   [2M-5H]5-, read at z = 1, became 2065-2079 Da species at 0.6-1.6 % of NC. With z = 1 never a
   species on its own, and counted in the gate and among the charges added for the fit only where
   its envelope has no peaks between its isotopologues (the candidate pick counts every z = 1 vote
   beside another charge, and the fit still covers a voted z = 1 the gate rejected), the method now
   reaches 800 Da and finds the trimer, 1563.2 and 1234.2 under IP-RPLC and a ladder of shortmers
   whose monoisotopic masses differ by phosphorothioate residues to 0.01 Da, with a 4-mer's silver
   adduct listed apart, real ions at 0.000 % and two unidentified reports. z = 1 reached further
   than intended. Through the adduct-evidence test, which reads it as the gate does, it made sodium
   and potassium evidenced in three low-mass regions (item 13), and on one of 16 synthetic Orbitrap
   seeds peaks voted for by z = 1 beside another charge state pushed a weak 6.42 kDa main species
   out of the decoder's list of 200 candidates. That preset's 3-9 kDa species show no z = 1, and the
   fix's verifier had judged it unaffected for that reason; the benchmark's four seeds do not show
   the loss.
8. **The comb takes over where the overlap's optimum lies between its teeth, and chooses its tooth
   with its neighbours.** Section 16 consulted the comb only where the overlap search ended on the
   window's edge. A mass read about half a spacing off (an envelope 2 Da lighter on the same comb
   pulls the mean shift) has both teeth beyond the 0.29 Da window, and the optimum sits between them
   on whatever weak signal is there, 115-2000 times below the better tooth: 1908.3 and 2887.4 were
   missed, the 2980.4 Da species was lost and a 2602.07 Da species was reported at 0.0003 % at a
   time where its windows are empty. An interior optimum is now kept where it lies on the comb
   (within 0.15 Da of a tooth; on the four runs such optima lie within 0.093 Da, those between the
   teeth 0.19-0.48 Da off) or holds a quarter of the larger tooth's overlap (those between the teeth
   hold at most 15 %), and otherwise the better tooth is taken: 1908.3 and 2887.4 are found at 0.82
   and 0.98 times their extracted-ion references, the 2980.4 and 2602.07 Da species at their times.
   The better tooth was section 16's weak point: compared alone, the two teeth's overlaps differ by
   0.03-26 %, right and wrong picks sharing that range, because an envelope of the same comb two to
   seven spacings away overlaps the candidate's and favours the tooth on its side. Fitted together
   with same-comb templates two to seven spacings beyond the pair, the teeth are right on 49 of the
   50 placements of section 16's code that composition or the raw scans decide, the overlaps also on
   49 - but the joint fit's miss, NC + 6Na + K, is held at the window's edge by the cation-ladder
   rule either way, while the overlaps' miss, FLP + K, is placed a spacing light. Of the
   between-teeth placements added since, three go to the tooth the apex-scan fits reject (2601.07,
   2665.77 and 2983.00 Da). The verifier widened the joint fit from four spacings to seven when n-2
   under IP-HILIC flipped to the wrong tooth for a comb read 0.005-0.02 Da heavier; strong same-comb
   neighbours on both sides still defeat it.
9. **The reported mass follows the templates where the comb placed them.** A comb-placed species
   kept its mean-shift mass, 0.3-0.5 Da off its own isotopologues: n-2 was read 0.49 Da light and
   0.27 Da heavy while its templates sat 0.06 Da from the truth. Read off the comb, the template
   position carries only the class pattern's composition error, -0.12 to +0.17 Da on the four runs
   and the same for one molecule in both chromatographies, where the mean-shift masses of those
   pairs differ by up to 0.6 Da. Such a species is now reported there where that lies within half a
   spacing of its measured mass (`refine.report_comb_masses`). The investigator moved the mass
   inside the fit passes; the verifier found the next pass's filter of species already found then
   reading the moved mass, so that an envelope placed on its far tooth, 0.52-0.59 Da from its gate
   mass, was found again and fitted on the other tooth (a 4486.64 Da report), and the adduct merges
   and drops reading it too. Done once per frame after the adduct steps, every amount on the four
   runs stayed as it was but one merge's: 4948.78 and 4949.38 Da, two species one tooth apart, now
   merge into one report, away from the truth. n-2 is reported at 4503.65 Da in both runs (-0.06 Da,
   from -0.49 and +0.27 Da), 1883.3 under IP-RPLC at 1884.63 Da (+0.02, from -0.34 Da), 3527.5 under
   IP-HILIC at 3527.87 Da (-0.12, from +0.33 Da) and n-1 under IP-HILIC at its tail piece's comb
   position (-0.06 Da).
10. **A knife edge is shut with a tolerance the code already has.** Integrated, the region floor
    (item 2) moved 3207.4's gate mass by 0.025 Da and put its comb's tooth 0.005 Da inside the
    0.289 Da window. The edge path took the comb only for a tooth beyond the window, so 3207.4 kept
    its mean-shift mass, 0.30 Da light, where item 9 alone had put it at +0.01 Da. An alignment that
    stops at the edge within 0.15 Da of a tooth inside the window - the tolerance that already puts
    an interior optimum on the comb - now takes that tooth: it is the tooth cut short by the window,
    the overlap peaking past the midpoint of the last step. It fires on four alignments of the four
    runs, 0.0005-0.024 Da from their tooth, and moves 3207.4 to 3208.69 Da (+0.01) and two
    unconfirmed reports onto the teeth their summed scans read. A threshold of 0.03 Da gave the same
    four runs but would have been a new knife edge, among the distances observed (0.020-0.024 Da
    inside it, 0.038 Da outside). As first written, the rule took the edge for the tooth and then
    let the joint fit of item 8 choose between it and the tooth a spacing beyond the window's other
    edge, which the overlap search never reached - a rule acting against its own reason. Over the
    four runs and all 52 synthetic runs it fires ten times, and the joint fit would have taken that
    far tooth in four: a synthetic 13.2 kDa impurity's template went from 0.003 Da off its truth to
    a spacing heavy, and the legacy polymer preset's one gain (61 to 64 %) was such a pick, which
    also moved a matched species 0.15 Da further from its truth. The edge's own tooth is now taken,
    its distance from the comb wrapped to the nearest tooth as inside the window (above about 17 kDa
    the window can hold the teeth either side of the measured mass); the four runs are unchanged,
    and the legacy polymer preset is back at 61 %. The jump itself remains, one step in: an interior
    optimum on the comb keeps the mean-shift mass, and 3315.35 Da of the same run, 0.28 Da light of
    its tooth, peaks 0.0007 Da short of the midpoint that decides.
11. **Fixes interact, and every one needs the whole chain run again.** Each fix was validated alone
    on the code of section 16; integrated, results appeared that no single validation could show.
    3207.4 needed items 2 and 9 together to lose its correction (item 10). A 1481.14 Da report of
    the IP-HILIC NC run needs the 800 Da range and the between-teeth rule together: it is a cluster
    ion, the sodium form of a calcium-bridged dimer of a co-eluting 710.09 Da compound (its ions fit
    that composition within 1.3 ppm), which its one z = 1 isotopologue gives a second charge state,
    the between-teeth rule templates on its own isotopologues and item 9 a mass 0.43 Da above its
    measured one; requiring z = 1 to show in two scans removes it, and also the deaminated 4-mer, so
    it is left to the adduct module. The 2602.07 Da species got its time and amount from item 8 and,
    from item 9, the tooth the apex-scan fits reject as its reported mass: 1.0 Da from its reference
    where it had been 0.55 Da. And the pooled reading below turned item 4's join of n-1 against it.
    Each step was therefore committed alone, the four runs and the tests run on the chain after it,
    and every changed report traced to the steps that changed it; the synthetic benchmarks ran on
    each fix alone and on the integrated chain, and the four runs, the benchmarks and the tests ran
    on the final tree again after the review fixes, after item 10 and after the last review's fixes.
12. **The pooled reading was tried and not used.** The union axis weights a region's points by
    count; read on each scan's own points, pooled, n-1's main piece is 4833.16 instead of
    4833.86 Da, and that reading was integrated as a last step. It moved 61 other reports by 0.1 Da
    or more, 32 towards their per-scan reading and 14 away. But the adduct folds and the region
    merges read the reported masses: six folds flipped at their 0.5-0.6 Da edges (5301.01, PO-NC +
    Na, 4885.13, 7012.43, 6555.47 and 3368.69 Da), PO-NC + 2Na became a species at 0.82 % - adduct
    decisions this round left alone - and n-1 under IP-HILIC went from -0.06 to +0.18 Da: read
    pooled, its main piece came within 0.5 Da of the tail piece, so the merge that keeps the
    stronger piece's mass joined them before the one that keeps the mass the isotopes support saw
    them. The paper's species got worse (mean error 0.053 to 0.062 Da), the SI's better (0.184 to
    0.168 Da). It waits for the adduct module, with folds and merges decided on one mass, the
    pooled value given to the reported mass only, and boundary pieces left to the tooth.
13. **Adduct outcomes moved although no adduct rule changed.** The adduct dedupe, the fold, the
    cation-ladder rule, the coincident-adduct and shared-evidence drops and the adduct-form
    exemption are the code of section 16; the adduct-evidence test only gained arguments that keep
    it reading charge windows and judge z = 1 as detection does, and no form listed apart before is
    folded away. The in-scope fixes moved adduct outcomes all the same: NC + K of the IP-RPLC
    conjugate run, rescued at r 0.98 (item 5), folds into NC (all forms 1.65 to 1.69 %), and NC +
    DIPEA, rescued at 0.903, is removed by the dedupe and the shared-evidence test; z = 1 evidences
    sodium and potassium in three low-mass regions, giving the IP-HILIC 4-mer adduct columns (all
    forms 2.59 to 4.14 %) and walking its [2M+Na] dimer to 2M, which item 6 then drops; the
    between-teeth rule ends a walk (3058.12 to 2998.05 Da), moves adduct columns between shortmers
    and fits two synthetic adduct forms whole (Orbitrap precision 76 to 70 %); the joint-fit tooth
    puts FLP + K's template on its own tooth (PO all forms 4.82 to 4.86 %); and the comb-placed
    masses leave every fold as it was but move each form's offset from its parent plus the adduct
    (2558.52 Da's +K form now 0.52 Da off, 0.08 Da inside the fold's 0.6 Da). In-scope rules lean on
    adduct rules in turn: only the cation-ladder rule holds 2887.4 + 3Na + K at its optimum and
    masks the joint fit's one miss, FLP + K is comb-placed only because its region does not evidence
    potassium, and item 10 places adduct-walk trials. The list, with its numbers, is in
    `benchmarks.md` for the adduct module.

On the four runs the round gives 250 reports instead of 228, 127 of them EIC-confirmed (115) and 58
unconfirmed (51; 45 of them below 0.1 % of the main species, 40 before); all 20 of the paper's
species, and all 29 SI-listed elutions (23). The scored mass error on the 43 species found before
goes from a median of 0.080 to 0.060 Da and a mean of 0.142 to 0.105 Da, 11 to 6 of them above
0.2 Da and 6 to 2 above 0.3 Da; 19 of 24 impurity amounts lie within 0.88-1.19 of their
extracted-ion references (16 of 23), two of them references read on the species' own isotopologues
(n-2 under IP-RPLC, 1883.3 under IP-HILIC). Five IP-RPLC shortmers are still read 0.52-0.75 Da light
of their own isotopologues: three whose templates sit on the right tooth more than half a spacing
from the measured mass, two whose templates sit on the wrong one. On the synthetic benchmarks recall
is unchanged in every preset and tier, the oligonucleotide preset's median error goes from 1.4 to
1.1 ppm and the Orbitrap preset's precision from 76 to 70 %, and the legacy generator's tables are
as they were; over 16 Orbitrap seeds 75 of 165 reports matched before and 74 now, the main species
of item 7 lost.

The three reviews found more to fix. The code review found the later-elution set-aside running for
every method, so that under the protein top-down method a 12 kDa protein's isomer at 3 % a minute
later was named in a flag instead of reported; it is now a method setting
(`linking.set_aside_later_elutions`), on for the two oligonucleotide methods, where the main
species' tail and column-wash peak are the concern. It found the multimer rule dropping without
evidence (item 6), and rules no test pinned - the pipeline's call to `report_comb_masses`, the z = 1
rules beyond the gate's, the rescue's 0.9 bar, the tooth mass the boundary join reads, the floors
and weights passed through the pipeline - each now with a test that fails when it is removed (28
mutations, all caught). The outputs review found 3207.4's knife edge (item 10), the unconfirmed
count and the moves to explain (2601.07 Da, the 4949 merge), three SI-listed shortmers read half a
spacing light although 29 of 29 are found, and the unexamined 1481.14 Da (item 11); the scope review
the adduct side effects (item 13), a misreading of the rescue (NC + DIPEA does not fold), and 47
statements in these documents that the tree had made stale. A last review, of the result, found the
edge rule handing its tooth to the joint fit and measuring the edge against the nearest tooth only
(item 10), a z = 1 rule these documents stated more widely than the code applies it (item 7), counts
here that their sources do not support, and changes no test caught: the weights the tooth choice
fits with, the cation-ladder guard on the edge path, the multimer rule reading the multimer's trace
instead of the monomer's, `coelution_r: 0` dropping multimers again and a rescue bar of 0.7. Each
now has a test that fails with it.

What did not work, besides the pooled reading: an amplitude test for the comb's second state (a
state counts only where a window without its comb would agree that closely by chance at most 0.5 %
of the time) cut chance agreement on noise from 17.5-20.5 % to 1.0-2.5 %, but changed no report
anywhere and, below about 3 kDa, where a short envelope leaks into the off-comb frequencies it
compares with, lost weak states of the species' own (1883.3's z = 3 among them); taking the other
tooth for 2558.3's on-comb optimum raised it from 0.29 to 0.33 % against 0.36 % but put other
templates on teeth the apex-scan fits reject and churned six adduct-cluster reports, and a shape fit
on the same residual moved 4 of 12 candidates onto rejected teeth, among them a 2.9 % species;
reporting the template position for every comb placement, near tooth or far, made FLP + K a 1.8 %
false species, and for every isotope-resolved species took 2558.3 1.00 Da off and out of the
scorer's reach; a 500 Da floor added nothing; relaxing `report_comb_masses`' half spacing was
considered and not tried: 19 comb-placed components lie 0.52-0.75 Da from their templates, among
them two sodium forms of PO-NC whose folds would flip and two whose templates sit on rejected teeth.

The lesson extends section 16's: evidence must be the species' own - its own isotopologues for the
co-elution of its charge states, its own tooth read in its own scans for its pieces and its later
elutions, its own signal for its reference. And a fix is validated only in the chain it ships in: of
this round's surprises, three came from fixes combined, one from a list of candidates every species
shares, one from masses that decisions downstream of the fix read, and one from a step every method
ran that only the oligonucleotide methods had asked for. No adduct rule changed and adduct outcomes
moved anyway; the adduct module inherits that list, and every rule that leans on the ones it will
redesign.
