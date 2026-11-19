# A transformer whose weights are written, not trained (design)

This is a design for a transformer that performs 2D LC-MS deconvolution with every weight written
by hand in code, following *interpretable weight-space programming* (iWSP; Mehta et al.,
"Interpretable weight-space programming builds transparent transformers and brain encoding
models", [arXiv:2611.18114](https://arxiv.org/abs/2611.18114), 2026). It reads tiles of retention
time x ln(m/z), produces the per-bin charge map the decoder already consumes and, beyond it, the
time-resolved evidence that decides whether a species is real. Nothing in the pipeline changes
yet. Sections 2 and 3 are measurements made for this design on the four real runs and the
synthetic benchmarks; sections 4-8 are the design, the search harness and the plan. Every number
measured here is reproduced by a script in `scripts/iwsp_probes/` (its README lists which);
estimates are called estimates.

**In short.**

- The pipeline's comb estimator *is* a small transformer: four layers of hand-set attention
  heads, run as a numpy program in the exact limit of those heads, reproduce `CombPredictor` in
  every field of `ChargePrediction` on every bin of 128 synthetic spectra and of all 96 predictor
  calls of the four real runs (section 4.3). That is version 0. Building it from the layer
  library and running it as a network is the first test of stage 0.
- On the real runs the charge map is not what loses species. In FLP's tiles version 0 reads
  80-84 % of the known species' envelope intensity as z = 5 where it is z = 4 (it gets 10-15 %
  right) and the isotope filter C2b gets 98 % right, yet both find the same 13/20 paper and 5/29
  SI-listed species; true charges inside every known envelope find 13/20 and 6/29. The missed
  species are strong in correctly resampled sums (median S/N about 1 700) but sit at a median of
  about 220, and as low as 8-16, in the view the package deconvolves; they are also seen at one
  charge state, judged in a summed spectrum, or outside any TIC region. In the time-resolved data
  23 of the 31 show three or more co-eluting isotopologues and at most one co-eluting
  half-spacing peak; 1 of 240 random positions does (section 3).
- How scans reach the grid matters. The package's resampling lets the edge point of every
  vendor-thresholded segment stand for half the gap beside it, and its region sums interpolate
  every scan across the others' gaps; together they raise the noise estimate of the spectra it
  deconvolves 40-180-fold and move the amounts it reports (patched, IP-HILIC NC's main species
  goes from 46 % of its deaminated impurity to 100 % against its 11 %). The measurements here
  resample each scan correctly. Part of what the pipeline loses, it loses to its own resampling;
  the fix is a separate change that needs the decoder recalibrated and the networks retrained
  (sections 2 and 7).
- The design has two parts on 48-scan tiles that cover the whole run: **charge-map circuits**
  that remove false reports (C2b keeps the comb's species with 5 instead of 7 unconfirmed
  reports - a variant chosen on all four runs), and **species circuits** that use the time axis -
  co-elution, isotope chains, elution shape - to flag species the decoder now rejects. A frozen,
  human-written 2D decoder turns per-token output into species; a flag nominates a single-charge
  candidate, whose isotope chain the decoder then tests itself; NNLS still sets the amounts. The
  network cannot bypass the gates, but relabels and flags can add reports, which the adoption
  rule bounds.
- The search follows the article: an agent edits only the network and its `write_weights`, starting
  from version 0, with the real-run truth and this document's circuits and measurements kept out of
  its sandbox; a fixed evaluator scores a pre-registered species-level objective and returns only
  counts for the real runs; the four runs form two folds, so each is scored once by a model whose
  search never saw it; audits check that no weight comes from data and that the executed network is
  the declared one; a 200-word description of the change must rebuild the model from version 0 (the
  article's standalone rebuild is reported as well).
- Before any search, a charge oracle with chain flags, run through the new decoder, must find at
  least two more real species than the comb through the same decoder, with no more reports
  outside the truth table; otherwise the decoder, not a network, is the work. Every circuit here
  was measured on all four runs, so a sealed fifth run, scored against a truth table fixed
  blind, is the adoption gate.

## 1. The approach in the article

An agent (Claude Code) is given a task description, a rules document, a starting Python file holding
a small transformer and an empty `write_weights(model, task)`, and a fixed evaluator it cannot edit.
It writes the weights directly - attention circuits, MLP neurons, lookup tables, constants - runs
the evaluator on a fixed visible set, revises, and repeats for 30-50 iterations. Training of any
kind is forbidden (gradients, optimizers, fitting loops, pretrained weights), and so is computing
the answer anywhere but inside the network. Three checks keep it honest: an auditing agent reviews
every saved model for rule violations and uninterpretable edits; rerunning `write_weights` on
altered task data must leave the weights unchanged; and the final model is scored on a hidden set it
never saw during the loop. A fresh session then reads its code (in the ECoG rebuild, with comments
and run history removed) and describes it in at most 200 words; fresh sessions rebuild it from that
description and the task alone (with at most six visible-set runs in the language-fMRI rebuilds,
none in the ECoG ones). On algorithmic tasks the agents translated known algorithms into exact
transformers (Euclid's algorithm as one block of ReLU units repeated 1 000 times, a prime sieve as
pulse neurons with lateral inhibition) and then compressed them; on NLP tasks they wrote crude but
readable rule lists. The article names the limits: shallow motifs and known algorithms work best;
deep hierarchical representations without backpropagation remain open.

Why it fits this problem:

- **The physics is known.** On the grid u = ln(m/z ∓ m_p) every relation that decides a charge
  is a fixed offset: charge states at ln(z/z'), isotopologues at ln(1 + k 1.0034/M), adducts at
  ln(1 + d/M). The comb is already a written network; LadderNet is that network plus corrections
  learned by gradient descent; an iWSP model is that network plus corrections written, audited
  and explained.
- **Once scans are resampled correctly, the remaining failures are rule-shaped** (section 3):
  species seen at one charge state, envelopes judged in a summed spectrum, regions the TIC never
  forms; the column wash and the override's misreading of strong envelopes are risks for false
  reports and per-bin charges (sections 2-3). Each has a physical signature in the
  time-resolved data that a rule can state.
- **The result is auditable.** Every weight has a stated reason and a 200-word description
  rebuilds the model; for impurity profiling under regulation that matters as much as accuracy.
- **No training, no GPU.** The weights are code; evaluation runs on a CPU.

## 2. What the data look like in two dimensions

Measured on the real runs (`benchmarks.md`: Q Exactive Plus, negative mode, profile spectra at
140 000, vendor-thresholded), each scan resampled onto the grid for m/z 400-2500 (91 525 bins)
with each profile point's width capped as described below the table:

| Quantity | Value |
| --- | --- |
| MS1 scans | IP-RPLC GalNAc run: 609 in 30 min (3.09 s each); IP-RPLC NC run: 692 (2.61 s); IP-HILIC runs: 589-592 in 23 min (2.33 s) |
| Elution width (FWHM of a charge state's envelope trace) | main peaks: IP-RPLC FLP 9-11 scans (28-34 s), NC 7-10 (18-26 s); IP-HILIC FLP 5 (12 s), NC 6 (14 s). Impurities: IP-RPLC 3-6 scans (9-19 s), IP-HILIC 3-8 (7-19 s) |
| Charge-state shares at the apex (envelope sums) | FLP: z = 4 carries 86 % under IP-RPLC (z = 4: 4.9e8; z = 3: 6.6e7; z = 5: 1.4e7; z = 6: 2.2e6; z = 7: 6.4e5) and 88 % under IP-HILIC. NC: z = 3 and z = 4 carry 42-58 % each in IP-RPLC GalNAc, IP-RPLC NC and IP-HILIC NC; 62/38 in IP-HILIC GalNAc, where NC is minor |
| Grid bins with signal in one scan | GalNAc run: 1 600-3 900 (1.7-4.3 %); NC run: 800-1 800 (0.9-1.9 %) |
| Grid bins with signal in a 24-scan tile (75 s), four GalNAc tiles | 9 600-13 300; 4 400-6 600 of them above three sigma of the summed spectrum ("active") |
| Active bins with signal in at most 2 of the 24 scans | 30 % at FLP's apex, 3-5 % elsewhere; median summed intensity a quarter to a third of the persistent bins' (0.22-0.29; FLP tile: 6.0e3 against 2.1e4); 0.1-0.7 % of the comb's mass-vote weight at z >= 2 |
| Where the comb's winning ladder evidence comes from (same tiles) | 77-94 % from persistent neighbours whose traces correlate below 0.5 with the bin's; 2-10 % from one- or two-scan neighbours |
| Trace correlation of a species' own charge states | FLP against its z = 4 trace (medians): z = 3 0.96, z = 5 0.98, z = 6 0.85, z = 7 0.64; NC against its z = 4 trace (IP-RPLC NC, 8 strongest bins): z = 3 0.97-1.00, z = 5 0.88-0.95, z = 6 0.40-0.66 at log-SNR 1.6-2.4, z = 7 -0.12-0.33 |
| Trace correlation of unrelated signal | other bins of FLP's tile: median 0.26; random pairs of peaks above 10 sigma (IP-RPLC NC, 32-scan tiles): 90th percentile 0.29-0.47, 0.2-2.3 % above 0.9 |
| Chance ladder partners (peaks at a z -> z+1 offset) correlating above 0.9 | 0.6 % in the wash tile and 4.9 % in a tile holding a strong fragment (2 887 Da at 10.42 min); 10.8 % at NC's apex, where its own charge states and forms are among them |
| Column wash (96 scans around 27 min, both IP-RPLC runs) | 786-1 549 strong peaks spanning at least 20 scans (participation ratio); 82-89 % of them with a z = 1 isotope partner (17 % at a non-isotopic offset); FWHM median 18-23 scans (90th percentile 34-40); apexes spread over about a minute (IQR); pairwise trace r median 0.54-0.56, 12-17 % of pairs above 0.9 |
| Later elutions | NC (IP-RPLC NC) returns at 25.4 min at 0.98 % of its apex height (0.63 % of its area) and at 27.7 min at 0.13 %; FLP does not return |
| Co-eluting forms beside the paper species (trace correlation with the parent, height relative to it) | +201.4 Da: beside all nine IP-RPLC species, r = 0.89-1.00, 6-15 %; under IP-HILIC co-eluting only beside FLP (r = 0.95, 2.8 %). -16.06 Da (PS to PO): beside all five IP-RPLC GalNAc species, r = 0.79-0.97, 3-13 %. Na and K, where no stronger species shares the window: 1-5 % under IP-RPLC; Na 16-59 %, K 4-46 % under IP-HILIC |

**Resampling.** The runs are vendor-thresholded: a scan is a set of short segments of profile points
separated by gaps of up to thousands of bins. The package's `resample_profile` gives every point the
width 0.5 (u[i+1] - u[i-1]), so a segment's edge point stands for half the gap beside it and lands
in one bin multiplied by up to that many bins. In single scans 24-31 % of the points are wider than
three median spacings (the widest 960-4 892 bins) and they carry 77-97 % of the resampled signal; in
the pipeline's region sums 82-96 %. Those sums add a second distortion: `Run.sum_frames`
interpolates every scan onto the m/z axis of the scan with most points, which fills each scan's gaps
with ramps between its segment edges (4-21 % of the summed signal in five regions of the four runs)
and drops signal off that axis. Together they make the noise estimate of the region's grid view
40-180 times that of the same scans resampled one by one and summed (the widths account for about
4-5 times), and leave 5-20 % of the summed points above the decoder's 3-sigma tests, against all of
them in the per-scan sum.

Every tile measurement of sections 2-4 instead gives each point the smaller of its two neighbour
spacings, at most three median spacings (`scripts/iwsp_probes/scan_grid.py`), as do section 3's
signal-to-noise and isotope-chain probes; the pipeline results of section 3 and the per-call figures
of 4.1-4.3 (active bins, leakage, timings) come from the package as it runs. Patched in-process, the
pipeline with capped widths, or with the grid view replaced by the per-scan sum, finds 13-14/20
paper and 9-10/29 SI-listed species but reports 154-158 species, 52-55 of them unconfirmed, against
39 and 7. Neither patch is the fix: the first keeps the interpolated sum and judges its raw points
against a noise level 4-5 times lower (69-96 % pass, against 5-20 % today), the second against the
per-scan sum's (all pass), so both counts come from a largely disabled envelope gate. The amounts
move too: in IP-HILIC NC, where the package reports the main species at 46 % of its deaminated
impurity DA-NC, both patches report NC at 100 %, DA-NC at 11 % and PO-NC at 5 % (today 20 %);
PO-NC under NC in IP-RPLC goes from 0.18 % to 7.0 %; and IP-HILIC GalNAc's n-1 and PO, which
`benchmarks.md` puts at one charge state above the noise, are found at default settings.
Today's percentages on these runs should not be relied on until the fix is validated. The fix -
per-scan grids summed, for every region and for the per-frame quantification
(`features/quantify.py`), with the decoder's raw-point checks on the same input and its
thresholds recalibrated - is a separate change (section 8); the design's tokenizer resamples this
way from the start. (Measured
with the distortion, an earlier version of this document took one- and two-scan events for a third
of the comb's votes; they carry under 1 %.)

Three things follow. Charges are decided by isotope spacing more than by the ladder: one or two
charge states carry nearly all of a 5-7 kDa phosphorothioate's signal, its z = 1-2 rungs lie above
m/z 2 500, and its isotopologues are 7-10 bins apart. Most of the evidence the comb reads is not
the analyte's: in every tile examined, 77-94 % of the ladder evidence that decides a charge comes
from persistent signal that does not co-elute with the bin, while one- and two-scan events are
weak and carry under 1 % of the votes. And co-elution is specific for strong signal: a species'
main charge states move together (z = 3-5: r = 0.88-1.00) where unrelated peaks rarely do (0.2-2.3 %
of random pairs above 0.9); its outer states (z = 6: 0.85 for FLP but 0.40-0.66 for NC; FLP's
z = 7: 0.64) and the column wash (12-17 % of its pairs above 0.9) are the cases a gate must
handle.

## 3. What limits the pipeline on the real runs

Measured with `scripts/validate_oligo_runs.py` on the four runs, the package as it runs today
(comb: 13/20 paper species and 5/29 SI-listed found, 39 reports, 7 unconfirmed). The whole
four-run validation takes about 30 s, cheap enough to run at every iteration of a search.

**A charge oracle.** Giving the true charge to every active bin inside the isotope envelope of
every known species eluting in a region (z = 2-12, ±3.5 Da / z; comb elsewhere) finds 13/20 and
6/29. Perfect charges on the known species do not recover the missed ones, so a better charge
map alone cannot. (Abundances do move: PO-NC under NC goes from 0.18 % to 9.3 %, but the
resampling patch alone gives 7.0 %, and the oracle leaves the IP-HILIC NC inversion.) Accepting
species seen at one charge state as well (`single_charge: true`) finds 16/20 and 8/29; the comb
with that option finds 15/20 and 7/29, at the price of 12 more unconfirmed reports (74
reported, 19 unconfirmed).

**Per-bin charges do not decide species either.** On the labelled envelope bins of FLP's tiles,
version 0 reads 80-84 % of the labelled intensity as z = 5 where the truth is z = 4. Its isotope
override takes the largest charge whose evidence reaches 0.6 of the best, and on the
log-compressed, clipped log-SNR s of an envelope this strong the flanks of the neighbouring
isotopologues at the z = 5 spacing read high enough. It gets 10-15 % of the labelled intensity
right in those tiles (83 % in the n+1 and NC tile); C2b (4.4) gets 98 %. Both find the same
species.

**Where the missed species are lost** (`process_run(..., trace=...)`), with the comb as it runs
today, and with the oracle's charges and single charge states accepted:

| Stage | Comb, default | Oracle charges, `single_charge: true` |
| --- | --- | --- |
| No TIC region at their retention time (a TIC gap at 9.4-16.1 min in IP-RPLC NC; NC in IP-HILIC GalNAc) | 6 | 6 |
| Never became a candidate (little mass-histogram weight within 1 Da, or not picked) | 17 | 4 |
| Picked, then rejected by the envelope gate | 4 | 9 |
| Lost in the NNLS fit or after it (linking, flat or edge rules, one-to-one matching: DA one dalton from FLP) | 4 | 6 |
| Missed in all | 31 | 25 |

They are not weak in the data: resampled scan by scan, the strongest charge state of the in-region
misses stands at a median signal-to-noise ratio of about 1 700 in the region's sum (about 900 in a
7-scan window around each). In the grid view the package deconvolves, the same windows give a median
of about 220 (IQR 21-680), and 8-16 for the IP-HILIC GalNAc n-1, DA, PO and n+1, the run whose
impurities the pipeline misses entirely: there the inflated noise estimate puts their minor
isotopologues near or below the decoder's 3-sigma tests. Most are 1.2-4.8 kDa fragments and NC
impurities seen mainly at z = 2-3 (the SI lists each fragment by one m/z and one charge), and the
IP-HILIC GalNAc impurities at z = 4. Once charges and the charge count are no longer the barrier,
the envelope gate is: under oracle charges with single charge states accepted, it rejects nine
picked species (four IP-RPLC NC fragments, five IP-HILIC NC species), six of which pass the
time-resolved test below. It judges an envelope in the summed spectrum, where other ions share the
m/z window. What is missing is not charge accuracy but a correctly resampled input, **time-resolved
evidence that an envelope belongs to one eluting species**, and **regions where the TIC shows no
band**.

**The evidence is in the traces.** A test of exactly that kind - in the 7 scans around the
envelope's largest summed scan within ±0.3 min, with signal in at least 4 of them, three or more
isotopologues at 1.0034/z Th whose traces follow the envelope's strongest bin at r >= 0.9, and no
more than one such trace at the half-spacing positions - passes for 23 of the 31 missed species
at their tabulated times (and 14 of the 18 found ones), and for 1 of 240 random (mass, time)
positions (1 641.4 Da at 4.55 min in IP-RPLC NC, among that run's fragments). Eleven of the
twelve IP-RPLC NC fragments pass, four of them in the TIC gap; in IP-HILIC the conjugate's n-1,
DA, PO and n+1 pass, its NC does not. Of the pipeline's unconfirmed reports 3 of 7 pass (10 of 19
with `single_charge: true`), among them an envelope next to FLP with six co-eluting isotopologues -
possibly species the truth table does not list. This is a probe, not the circuit: it fixes a
window where the circuit must follow each trace (4.5), and 1 of 240 bounds its rate on random
positions only below about 2 % (95 %), which stage 0 measures on many more.

**Written corrections, scored end to end.** Five rules were tried by hand, as a search would try
them (section 4.4 gives their circuits):

| Rule | Per bin | Real runs: paper / SI / reported / unconfirmed |
| --- | --- | --- |
| comb (version 0) | synthetic: TOF 0.65 / 0.84 / 0.86; Orbitrap 0.36 / 0.43 / 0.26. Real tiles, labelled envelopes: 0.10 / 0.15 / 0.83 | 13/20, 5/29, 39, 7 |
| ladder mean over in-range neighbours only (a neighbour beyond the acquired m/z range is unknown, not absent) | with the isotope override kept, as validated: unchanged on the labelled envelopes and in synthetic Orbitrap accuracy. The ladder alone, without the override: real tiles 0.92 / 0.87 / 0.41 -> 0.94 / 0.93 / 0.91; synthetic TOF 0.65 -> 0.58 (log-SNR) | 13/20, 5/29, 40, 7 (one more EIC-confirmed report) |
| C2b: isotope matched filter with half-spacing inhibition relative to the first isotopologue, no margin | synthetic Orbitrap 0.62 / 0.73 / 0.81 (z = 2-5 up, z = 6-40 down against the ladder alone); real tiles 0.98 / 0.98 / 0.98 | 13/20, 5/29, 38, **5** |
| C2: absolute half-spacing inhibition and a decision margin of s/4 | synthetic Orbitrap 0.61 / 0.79 / 0.96; real tiles 0.97 / 0.96 / 0.83 | 11/20, 4/29, 30, 5: PO-NC, DA-NC and the SI-listed 4 512.6 Da fragment lost |
| co-elution gate on ladder rungs (12-slot profiles over the region ± one width; weak bins exempt), untuned, θ = 0.3, 0.5, 0.7 | - | 13/20, 5/29, 41, 7 at all three (top charge changed at 6-8 % of active bins; added: a +K form and one EIC-confirmed report) |
| decoder: isotope check requires two adjacent populated isotopologue positions for z >= 2 | - | 13/20, 5/29, 38, 6 (with `single_charge: true`: 15/20, 7/29, 73, 18) |

(Per-bin numbers: z >= 2, weighted by log-SNR / intensity / intensity pooled, on the 96 spectra of
`scripts/eval_charge_bins.py`; on the real tiles, intensity-weighted over the known species'
envelope bins at 19.21, 18.71 and 20.3 min in IP-RPLC GalNAc.) Per-bin accuracy and species come
apart in both directions: C2 is the best per bin on synthetic Orbitrap spectra by intensity (0.79,
pooled 0.96) and loses two paper species; C2b and version 0 differ by a factor of six to ten per
bin on FLP's envelopes and find the same species. This is
the loop the harness of section 5 runs, and why its objective is species, not bins. C2b's two
fewer unconfirmed reports are in-sample: it was chosen from five variants of the filter, each
scored on all four runs, and its gain falls on one run (IP-RPLC GalNAc, 5 -> 3).

## 4. The architecture

```
48-scan tile -> grid bins = keys (s, intensity, position, per-scan trace, context)
                        |
hypotheses = queries (active bin x charge, closed-form physics of position and z)
  layer 1    pointer heads: ladder rungs, isotopologues, half-rungs
             (+ co-elution gates from the traces)                          [4.3, 4.4]
  MLP        comb evidence, isotope filter, harmonics, background          [4.3, 4.4]
  layers 2-4 argmax / broadcast heads: charge per bin                      [4.3]
  consensus  (tied passes) partners read at the consistent charge          [4.4]
  species    charge-state count, isotope-chain flags, mass consensus,
             elution shape, near-isobaric drift                            [4.5]
                        |
  per token: charges, vote weight, flags, apex scan
                        |
frozen 2D decoder (votes in mass x time, candidates, flags at every
charge-count check, partner templates, run-level forms)                     [4.6]
                        |
weighted NNLS, template bank, linking, integration (unchanged)
```

### 4.1 Tokens and the time axis

- **Key tokens:** every grid bin of the acquired m/z range (91 525 for m/z 400-2500), so every
  position a head can ask for exists. Of these, a 48-scan tile of the GalNAc run has 16 400-21 900
  bins with signal and 7 900-10 900 active ones. Channels: s = clip(log10(1 + I/sigma), 0, 5) (the
  comb's feature), the raw intensity, the position, and the time profile.
- **Query tokens (hypotheses):** one per active bin and charge z = 1..Z_max (Z_max = 40 for the
  oligonucleotide methods, 60 elsewhere): 315 000-434 000 per 48-scan tile at Z_max = 40
  (29 000-127 000 per TIC region in the drop-in probes of section 3). A hypothesis's embedding is
  closed-form physics of its own position and z: its six ladder offsets, its isotopologue offsets
  in bins, whether the spacing is resolved, the instrument's peak width there.
- **Time** lives in the channels, not in the sequence: each bin carries its intensity in every
  scan of the tile (over sigma), centred and scaled to unit length - a LayerNorm with gain 1/√48
  (without the gain its length is √48 and the dot product 48 r) - so the dot product of two bins'
  vectors is the Pearson correlation of their traces. Context channels hold each bin's mean
  intensity over the 96 scans before and after the tile, so that a plateau broader than the tile
  shows. Tokens over (scan, bin) would multiply the sequence by the number of scans and need more
  layers to collapse time again; co-elution as one dot product is the article's style. With one
  scan (a summed spectrum, the existing 1D benchmarks, public single spectra) every time term is
  switched off and the model is a 1D charge estimator.
- **Tiles.** 48 consecutive MS1 scans, one tile every 24 scans, the run padded with 24 empty scans
  at each end: every scan lies in two tiles, and every trace maximum lies in the central 24 scans
  of exactly one tile, from which alone its token votes. 48 scans are 2.5 min at 3.1 s and 1.9 min
  at 2.3 s: an analyte's base width (about 2.5 FWHM, 25 scans for FLP under IP-RPLC) fits, and the
  column wash (FWHM 18-23 scans) shows as broader. Every time constant (co-apex windows,
  persistence, templates) is declared in seconds or in units of a trace's own FWHM, not in scans,
  because scans take 2.3-3.1 s across these runs. Tiles replace TIC regions: the whole run is
  read. (The drop-in probes of section 3 kept the TIC regions and pooled each region and one
  region width either side into 12 slots.)
- **The alternative: elution-event tokens.** One token per local maximum of the time-smoothed
  (scan x bin) map above three sigma - one isotopologue peak at its elution apex, 39 000-116 000
  per run, 15-44 % of them one- or two-scan events - with its 33-scan trace in the channels, and
  partners required to peak in the same scan (a score term -beta_c (c_q - c_k)^2). A minor form
  eluting inside a major one's bins (DA-NC, 0.66 min before NC in IP-HILIC) becomes a token of its
  own, where a bin's trace over a tile is dominated by the major peak; the price is that version 0
  is exact only in a summed-spectrum mode (one token per bin, a one-scan trace). The frozen
  tokenizer provides both token sets; rule 9 (5.2) lets the agent choose one per stage, not change
  how either is made.

### 4.2 Primitives

All in float64, in the forms the article's models use; the layer library (5.1) provides exactly
these, and a new one enters only by proposal and audit (rule 10):

- **Pointer head.** Score -beta (q - p_j)^2 between a query position q and key positions p_j
  (beta = 400 per bin^2), plus a zero-valued sink scoring -beta/4, the score of a key half a bin
  away: a hit beats it by beta/4, a key one bin off loses to it by 3 beta/4. Only keys within ±2
  bins of q are evaluated; the omitted ones weigh below e^-(3.75 beta) against the sink. A floor
  neuron relu(read - 1e-100) follows each pointer: the comb decides with strict comparisons against
  zero, and leakage from a key one bin off turned "no evidence" into evidence at 823 bins of one
  real run before the floor was added. With the target empty, keys one and two bins off leak at
  most 5e-130 at beta = 400 (below the floor for any beta above 310). The distance form of the
  score is exact in any precision; the dot-product form, [2 beta q, -beta] . [p_j, p_j^2] minus
  beta q^2, is exact in float64 at integer positions and fails in float32 (errors up to 2.6e5,
  hits and misses on the wrong side of the sink in 54 % of comparisons).
- **Interpolation head.** The comb reads isotope neighbours by linear interpolation at
  fractional positions. For a target i0 + f with 0 < f < 1, a softmax head without a sink centred
  at i0 + 1/2 + ln(f/(1 - f))/(2 beta) gives the two bracketing keys the weights 1 - f and f, and
  every other key a weight below max(f, 1 - f)/min(f, 1 - f) x e^-2 beta of theirs: exact to
  2e-14 with scores computed from offsets p_j - i0, to 1.5e-9 from absolute grid positions, so the
  library uses offsets. At f = 0 (the comb clamps a read below the grid's first bin to it) the
  head is a pointer at i0. It needs no sink because every grid bin is a key; a sink at -beta/4
  would take up to a third of the weight. The prototype uses the equivalent tent kernel
  relu(1 - |q - p_j|).
- **Window head.** A pointer head with a small beta and no sink over ±W bins: a Gaussian kernel
  average, for the mass-consensus and shape heads of 4.5.
- **Argmax head.** A hard argmax over one bin's hypotheses with the sink first and the earliest
  key winning ties - the comb's running top-2 with strict ">" (ArgMax and Gather in ONNX).
- **Broadcast head.** A pointer from each hypothesis to its own bin.
- **MLP idioms**, exact for the values that occur: max(a, b) = a + relu(b - a); a step
  [a > b] = 1 - relu(1 - relu(K (a - b))) with a large K (exact for a - b >= 1/K and 0 for
  a <= b; the form relu(K (a - b)) - relu(K (a - b) - 1) fails once K (a - b) passes 2^53, giving 0
  or 2 up to 2^54 and 0 beyond) and
  [a >= b] = 1 - [b > a]; a select relu(a - M (1 - f)) + relu(b - M f) for a 0/1 flag f and
  0 <= a, b < M (bit-exact at M = 1e6); a gated (GLU-type) product unit where a product is needed.
- **Frozen readouts.** None of the above divides: the confidences best/(best + second + 1e-12)
  and the integration of the decoder are fixed readouts in the library, as the article's encoding
  models end in a fixed ridge regression.

### 4.3 Version 0: the comb, exactly

- **Layer 1** (hypothesis -> bins): six ladder pointer heads, tied across charges, at
  q = p + round(ln(z/(z ± k))/step), k = 1, 2, 3, with value s, summed in the comb's order into
  ACC; two interpolation heads at p ± ln(1 + 1.0033548/(z (m/z ∓ m_p)))/step.
- **MLP 1:** EV = s · ACC / n_z, where n_z counts every rung at a charge z ± k >= 1 whose offset
  is shorter than the grid, rungs past this bin's edge of the spectrum included; NEAR = 0.75 max
  + 0.25 min of the two isotope reads; ISO = s · NEAR where the spacing in bins is at least
  2.5 x max(FWHM in bins, 2), FWHM = (m/z)/R(m/z). On these runs
  the FWHM is 0.5-1.3 bins, so the 2-bin floor always applies and "resolved" means at least 5
  bins: z <= 10 at m/z 1 000.
- **Layer 2** (bin -> its hypotheses): argmax heads for the best ladder charge and the best
  isotope evidence. **Layer 3:** broadcast them back to the hypotheses.
- **MLP 3 / layer 4:** the runner-up charge (the winner masked), and the largest charge whose
  isotope evidence reaches 0.6 of the best (scores z · [ISO >= 0.6 max], argmax): the comb's
  "largest agreeing charge", meant for the divisors of a true isotope charge.
- **Readout:** where the best isotope evidence holds a quarter of s^2, the isotope charge wins
  with confidence 0.9 and a different ladder charge comes second with float32(1) - float32(0.9);
  elsewhere the confidences are best/(best + second + 1e-12) in float32; an apex neuron from two
  pointers at p ± 1 (a local maximum above 3 sigma).

About 170 hand-set numbers plus a Z_max x 7 table of offsets and counts from one formula. As a numpy
program in the exact limit of these heads - hard pointers, tent reads, comparisons in numpy where
the network has step neurons - it matches `CombPredictor` in top1_z, top2_z, top1_p, top2_p and apex
on every active bin of 128 synthetic spectra (64 time-of-flight, 64 Orbitrap at 35 000-280 000;
Z_max 40 and 60; 10.8 million active-bin evaluations) and of all 96 predictor calls of the four real
runs (150 773 active bins, both detection passes), whose species tables come out identical. It has
not yet run as a network built from the layer library; that is the first test of stage 0. It takes
about three times the comb's time on dense synthetic spectra (0.33 s against 0.11 s) and a sixth to
a third of it on real regions (per-call medians 0.007-0.017 s against 0.042-0.047 s). Its weakness
on Orbitrap data is the override (section 3). This is the starting point of the search, as the
untrained LadderNet is the comb.

### 4.4 Charge-map circuits

Each is a handful of weights on top of version 0. Section 3 gives the measured ones; the others
are proposals, not yet built.

- **C2, isotope matched filter** (where isotopes are resolved: these Orbitrap runs, and
  time-of-flight data at R = 30 000 up to about 10 kDa). Six isotope heads read s at the exact
  isotopologue positions ln(1 + k 1.0033548/M)/step for k = ±1, ±2, ±3 (N1, N2, N3 the larger of
  each pair), four at k = ±1/2, ±3/2 (H, their maximum: where only a 2z reading has
  isotopologues); a read takes the maximum over ±1 bin where the instrument's peak is wider than a
  bin. Variant C2b: evidence E(z) = min(s, N1) + 1/2 min(s, N2) + 1/4 min(s, N3) - 2 relu(H - 0.5
  N1) where the spacing is resolved (3 peak widths and 3 bins); the charge with the largest E wins
  where E reaches s/2 and beats the runner-up, and elsewhere the ladder's answer stands (it
  replaces the comb's override). Variant C2: -2 H in place of the relative term, and a winner
  margin of s/4. For the true z the half positions are valleys; a 2z reading has an empty N1; a
  z/2 reading has filled half positions and is inhibited; min(s, .) keeps the flanks of a strong
  envelope from voting for z + 1, the override's failure. C2b loses synthetic Orbitrap accuracy at
  z = 6-40 (0.53 and 0.44 against the ladder's 0.66 and 0.89, log-SNR; C2 at most 0.01), which
  the consensus below is meant to recover.
- **C1, capped comb:** each rung reads min(s_rung, s + 1), so one strong unrelated ion cannot
  decide a weak ion's charge.
- **C4, harmonics:** two more pointers per hypothesis at round(ln(2z/(2z ± 1))/step), rungs only a
  ladder of twice the charge has (LadderNet reads the same two): EV_z loses s times their mean,
  against reading a 2z ion at z. A neuron firing when only the even rungs (k = ±2) are filled
  marks the opposite error, a reading at twice the true charge.
- **C3, singly charged background:** a z = 1 isotope reading (1.0034 Th apart: 125 grid bins at
  m/z 400, 20 at m/z 2 500) that beats every z >= 2 isotope reading sends the bin to z = 1 and
  removes it from other bins' rungs (relu(s - 5 BG) as value); 82-89 % of the strong broad wash
  peaks have such a partner.
- **Co-elution gate:** each ladder pointer also reads the key's unit-length trace and weak flag w (1
  below S/N 15); a product unit forms r = r_query . r_key; a step neuron g = [r + 2 w_query + 2
  w_key > θ] gates the read through the select idiom. A blocked rung reads 0 and still counts in
  n_z. Version 0 is g = 1. Written as a term in the attention score instead, the gate cannot be
  exact: with the sink moved to gamma θ, keys one bin off stay unread only for gamma < beta/(5 - θ)
  (89 at beta = 400), and a softmax then still reads a rung at r = θ - 0.1 with weight e^-8.9 =
  1.4e-4, far above the floor. One-scan events cannot be handled by θ: a spike at an analyte's apex
  correlates with its trace at r ≈ √(2/PR), PR ≈ 1.5 FWHM in scans (0.38 at FWHM 9 scans, 0.52 at 5,
  0.67 at 3, the narrowest impurities here), so sparing 3-scan impurities from spikes would need θ
  above 0.7, which already cuts a strong species' own outer states. Spikes are rejected by their
  one-scan shape (4.5) or the traces despiked first. The gate cuts a species' own outer states too:
  5 of FLP's 14 z = 7 bins sit above the weak exemption at r <= 0.5, and half of NC's 8 strongest z
  = 6 bins (log-SNR 1.6-2.4) correlate at 0.40-0.46.
- **Flat-background veto:** a bin within a third of its maximum in both outer slots (the
  pipeline's own flat-species test, per bin) supports no rung and casts no ladder vote; its
  isotope reading still stands, so a flat multiply charged ion keeps an isotope-read charge.
- **Consensus (tied, two passes):** ladder and isotope heads read the neighbour's hypothesis at
  the *consistent* charge (z ± k on a rung, z on an isotopologue) rather than its raw s, with
  values min(logit, s). A partner then counts only if it is itself plausible at that charge, and
  the weak flank isotopologues of an envelope inherit its charge instead of falling back to
  ladder answers, one way a decision margin can lose weak species (C2's lost PO-NC and DA-NC).

### 4.5 Species circuits

These target the real-run losses of section 3. They write per-token evidence - a confirmation
flag, a vote weight, a partner flag - that the frozen decoder of 4.6 turns into species. None is
built yet; each is measured in stage 0 before any search.

- **Graded co-elution.** Every partner read of the species circuits is gated as in 4.4, but by
  intensity: no gate where the partner's summed S/N is 5 or less (weak traces are noise), r above
  θ required from S/N 30, linear in between (a species' main states z = 3-5: 0.88-1.00; its outer
  states 0.40-0.85 even at S/N 40-240, which this gate cuts; random peaks: 0.2-2.3 % above
  0.9). A persistence gate is no substitute: impurities elute in 3-6 scans under
  IP-RPLC, and on the real envelope bins a 3-scan persistence gate on the ladder lowers
  log-SNR-weighted accuracy in FLP's tile from 0.51 to 0.48.
- **Isotope-chain confirmation** (the single-charge acceptance). Sum heads at the isotopologue
  positions (exact offsets, as in C2) and at the half-spacing positions count co-eluting
  isotopologues (n_iso) and half-spacing peaks (n_half): r >= 0.9 over ±10 s around the envelope's
  own apex (7 scans under IP-RPLC, 9 under IP-HILIC), at least 4 scans with signal (one of 48 fixed
  masks, selected by a one-hot of the apex scan) - close to, but not, the probe's ±3 scans, so its
  rates are re-measured in stage 0. The article's sinusoidal positions can do the isotopologue part
  in one head: a score term A cos(phi_k - phi_q), with phase phi = 2 pi z (m/z ∓ m_p)/1.0034
  selected by the query's charge, peaks at every isotopologue position of that charge at once. A
  flag neuron relu(sum q - 5) over six 0/1 indicators: z >= 2, the spacing resolved, n_iso >= 3,
  n_half <= 1, the isotopologue ratios fitting the class pattern for M = z (m/z ∓ m_p) within 0.6 in
  log10, no second charge state. The probe of section 3 passes 23 of 31 missed species, 14 of 18
  found ones, 3 of 7 unconfirmed reports and 1 of 240 random positions; the circuit's own rates are
  measured in stage 0. The flag only nominates: the decoder repeats the test itself (4.6).
- **Charge-state count.** Six ladder pointer heads, co-elution gated, read the same isotopologue
  at z ± 1, 2, 3 (offsets ln(z/z'), the same ln M), so that a token knows whether its species is
  seen at one charge state or several.
- **Mass consensus (tied, three passes).** Each token carries ln M = u + ln z. A window head
  scores -(lnM_j - lnM_i)^2 M^2/(2 h^2) + ln a_j with value lnM_j, h = √(sigma_env^2 +
  sigma_inst^2) (2.3-2.6 Da here), the kernel of the decoder's refine step: one mean-shift step per
  pass moves a token halfway to its envelope's centre, 87.5 % in three, so every token of an
  envelope votes the species' mass. (The decoder's 0.6 Da histogram kernel, which only removes the
  1.0034 Da ripple, would move it 5-7 % per pass.)
- **Elution shape.** The normalised trace is correlated with three centred Gaussians set in
  seconds - a spike (one scan), an eluting analyte (FWHM 7-35 s here: impurities 7-19 s, main
  peaks up to 34 s) and broad (the wash: FWHM 18-23 scans, 47-71 s) - and three-way lateral
  inhibition picks one; spikes and broad tokens lose vote weight. The wash is only about twice as
  broad as a main peak, so the z = 1 isotope veto of 4.4 carries more of the weight than breadth.
- **Near-isobaric partners** (DA, 0.984 Da above FLP). For a strong token's main envelope, two
  sum heads collect the light and heavy halves scan by scan; a one-sided drift of ln(heavy/light)
  over two consecutive strong scans sets a partner flag with its apex scan. The z = 4 envelopes
  of FLP and FLP + 0.984 Da have a cosine of 0.95 on the grid (Gram condition number 36), so only
  the time profile separates them, and where they co-elute exactly (IP-RPLC) nothing here can.

### 4.6 Readouts and the frozen decoder

The network ends at per-token outputs; a **frozen 2D decoder**, written by people and not
editable by the search, turns them into species. Stage 0 writes it on the four real runs, since
no 2D synthetic data exists before stage 1; after stage 1 it is recalibrated per fold on
synthetic runs and that fold's visible runs, and each fold's decoder is frozen and hashed before
that fold's searches. Every constant is logged with its reason and the runs it was set on, and a
constant kept from stage 0 is marked as set on that fold's hidden runs.

- **Per token:** the `ChargePrediction` fields for the tile's summed spectrum (top two charges,
  their shares, apex), a vote weight w in [0, 1] (persistence, background), a confirmation flag
  (4.5), a partner flag, and the apex scan of its trace.
- **Votes and candidates:** every token votes intensity x share x apex weight x w at
  (ln M, tile start + apex scan); votes are smoothed over 1.2 Da (the existing function) and
  ±6 s; local maxima above 1e-4 of the strongest within ±2 min become candidates (a local floor:
  one tied to the global maximum deletes weak, late species, `lessons.md` section 6); a candidate's
  spectrum is the sum of its own elution window's per-scan grids (apex ± 1.5 x the width of its
  vote trace), which the decoder's raw-point checks read as well; then the existing envelope
  gate, NNLS fit, template bank, linking and flat-background rule run, with their thresholds
  recalibrated on that input. This replaces TIC regions (6 of the 31 missed species had none) and
  detection in region-summed spectra.
- **Flags at every charge-count check.** Today's decoder enforces two charge states three times.
  At picking, a candidate needs two charge states or a flagged token. At the envelope gate, a
  flagged single-charge candidate is judged by a time-resolved chain test the decoder computes
  itself - frozen code, on the candidate's own per-scan grids within ±10 s of its apex: n_iso >= 3,
  n_half <= 1 - instead of by the isotope check on the summed spectrum; unflagged single-charge
  candidates are rejected as today. A candidate that passes counts as charge-supported after the
  fit (the `single_charge` mark). The network's flag only nominates, so it cannot pass the gate
  on its own word; without all three checks, flags could not reach the stage section 3 blames.
- **Partner templates:** a candidate holding a partner flag gets one extra template at
  M + 0.984 Da in its NNLS fit; the partner is reported only if the fit gives it a share above the
  method's floor and it passes the gates like any candidate.
- **Run-level rules** (the decoder's, not the network's, because they need the whole run):
  partners at the method's adduct deltas (Na 21.982, K 37.956, the ion-pairing amines) are
  annotated as forms of the stronger co-eluting parent; a heavier delta recurring beside three or
  more parents becomes a "co-eluting form +delta" - the +201.4 Da form beside all nine IP-RPLC
  species. Its identity is unproven: it behaves like an adduct of the IP-RPLC mobile phase but
  also co-elutes beside FLP under IP-HILIC, and nothing depends on what it is. Lighter partners
  never fold: the PO form, 16.06 Da lighter, recurs beside all five IP-RPLC GalNAc species and is
  a real impurity. The same mass at another apex (NC returning at 0.1-1 %) is a further peak of the
  same species. Annotations never delete a report.
- **Bounded harm.** The network cannot bypass the gates or create candidates other than through
  votes, flags and partner flags. Relabels move mass votes and flags admit single-charge
  candidates, so both can add reports - LadderNet, changing only charges, added 13-21 unconfirmed
  reports; `single_charge: true` added 35 reports for 4 new species, 25 of them outside the truth
  table - which adoption item 1 bounds. Any charge model - the
  comb, LadderNet - runs through the same decoder with w = 1 and no flags; the comb run this way
  ("comb-2D") is the baseline every search is scored against.
- **Drop-in mode.** Without the 2D decoder the network is a `predict_grid` replacement for
  `CombPredictor` (`make_predictor("iwsp:<file>")`), with the region and run passed to it by a
  one-line hook in `process_run`: the mode the probes of section 3 used, kept for debugging.
- **Quantitation stays weighted NNLS.** The article's encoding models end in a fixed readout, a
  ridge regression the agent does not write; here the agent writes per-token evidence and the
  fixed decoder and NNLS read out species and amounts. An exact, audited active-set NNLS already
  exists; written into the network as one tied block repeated many times, as the article's GCD
  model repeats its block, it would re-derive that solver, not improve on it.

### 4.7 Size, speed, precision

Version 0: about 170 hand-set numbers and one table of offsets and counts from one formula. With
the gates and the species circuits, a few thousand numbers from about 15 formulas and at most 30
named constants (beta, θ, thresholds, kernel widths), each with a one-line physical reason.
Written as dense matrices, every co-elution-gated head repeats one constant across the 48 trace
channels of query and key - tens of thousands of non-zero weights that say one thing each - which
is why the rules count formulas and constants and cap non-zero weights at 50 000. LadderNet has
9 500-22 000 trained parameters. On one thread, version 0 takes 0.05-0.06 s per 48-scan tile, a
little less than the comb (0.06-0.07 s); the generic softmax-pointer ladder of
`pointer_attention.py` takes 1.6-2.9 s on the same tiles because it evaluates every bin with
signal at Z_max = 60. The gated
version is not timed: its co-elution reads add about 1.3e8 multiply-adds per tile (11 000 active
bins x 40 charges x 6 rungs x 48 scans), an estimated well under a second against the limit of
5 s per tile. Memory, estimated: about 35 MB of channels per tile, and about 170 MB per gated
head if all charges are gathered at once (about 40 MB evaluated ten charges at a time).

### 4.8 The 200-word description

The article's test of interpretability is that a description of at most 200 words rebuilds the
model. Version 0 with the co-elution gate, the model `coelution_gate.py` ran in section 3 (TIC
regions, 12-slot traces), in 200 words (whitespace-separated):

> Float64, no training. Keys: every grid bin of a region's summed spectrum: s =
> clip(log10(1+I/σ),0,5), I, position p, a centred unit-length 12-slot profile over region ± width.
> Each bin with I > 3σ queries once per charge z ≤ Zmax; other bins output zeros.
> Layer 1: six hard pointer heads, tied across z, score −400(q−p)² against a zero sink at −100,
> read s at p+round(ln(z/(z±k))/step), k=1..3, summed in that order; a read is 0 unless profiles
> correlate (r>0.5) or either s<1.2. Count every rung at charge ≥1 with offset shorter than the
> grid, blocked or past the edge; EV = s·sum/count. Two tent heads read s at p ±
> ln(1+1.0033548/(z·m))/step, m = |m/z − carrier|; ISO = s·(0.75max+0.25min) where that spacing ≥
> 2.5·max(FWHM in bins, 2).
> Layers 2-4, hard argmax over a bin's hypotheses (zero sink first, ties to smaller z): Z1 = argmax
> EV; Z2 = argmax of the others (sink if none). ZI = largest z with ISO ≥ 0.6·maxISO, if maxISO ≥
> 0.25s². Without ZI, p = EV/(EV1+EV2+1e−12) in float64, rounded to float32. With ZI: ZI first (p
> float32 0.9); Z1 second (p float32(1)−float32(0.9)) unless Z1 = ZI, when Z2 stays second. Apex:
> local maximum above 3σ.

No rebuild from it has been attempted; the flat veto and the harmonic term of 4.4 are not in it
because no probe has built them. A description must settle every case the code settles; an
earlier draft left the second charge undetermined where the isotope charge equals the ladder's and
where no rung reads anything.

## 5. The search harness

The article's setup (section 1), adapted.

### 5.1 Files, prompt and sandbox

To be built in stage 0 (section 8); the probes in `scripts/iwsp_probes/` are their prototypes.

| File | Edited by | Content |
| --- | --- | --- |
| `iwsp/tokenizer.py` | nobody | tiles and token channels (4.1): per-scan resampling with capped point widths, sigma, s, per-scan and context channels, positional physics; bin tokens and event tokens |
| `iwsp/layers.py` | nobody | the primitives of 4.2, evaluated exactly and sparsely, and the frozen readouts; tests: sparse against dense softmax attention to 1e-12 on small tiles (scores from offsets), every step and compare neuron at inputs up to 1e10 and differences down to 1e-12 |
| `iwsp/decoder2d.py` | nobody | the frozen 2D decoder of 4.6, with its provenance log |
| `iwsp/model.py` | the agent | `architecture(spec)` (the wiring) and `write_weights(net, spec)`; starts as version 0 |
| `iwsp/TASK.md` | nobody | the task, the harness's API and the objective, plus version 0's description for every session that starts from version 0 (not for the empty-file control search, the standalone rebuild or the control sessions) - nothing else from this document |
| `iwsp/RULES.md` | nobody | the article's rules and those of 5.2 |
| `iwsp/evaluate.py` | nobody | the objective and reports on the visible data (5.3), running the network as built from `model.py`; it runs outside the sandbox with the decoder and the real-run truth, behind one logged command |
| `iwsp/audit.py` | nobody | the checks of 5.6 |

The package needs plumbing first: a predictor registry (`make_predictor("iwsp:<file>")`) that
`eval_charge_bins.py`, `benchmark_synth.py` and `validate_oligo_runs.py` accept, and the region
hook of the drop-in mode.

The agent works in a sandbox with no network that holds only `model.py`, `TASK.md`, the rules,
the read-only tokenizer and layer library, the visible tiles and raw files, and the synthetic
truth. The real-run truth stays outside with the evaluator and the decoder, which the agent
reaches through one logged command that returns counts, as the article's visual-cortex search
reached its data through a restricted interface. `model.py` and `explore()` may import only the
tokenizer and the library, and the audit flags any `explore()` output that names visible species.
This document, its probes and their outputs, `docs/benchmarks.md`, `docs/lessons.md`,
`scripts/validate_oligo_runs.py` and the decoder's provenance log, which together hold the truth
and the measured circuits for all four runs, stay outside. Hidden scoring is a separate command
the operator runs once per selected model. The scorer sets aside a report that matches an
unscored truth species (DA and DA-NC under IP-RPLC): neither credited nor charged.

The file starts at version 0, for three reasons: the comb is the bar for adoption, so every edit
is a measured change from it; a generic GPT of three layers and width 64 with dense attention
cannot cheaply express offsets of 34 657 bins over tens of thousands of tokens and would spend
its budget rebuilding the comb; and version 0 is exact and public. It is not neutral - its isotope
rule was kept because of these four runs (`benchmarks.md`) - but comb-2D carries the same rule.
One control search starts from an empty file with the same library and a `TASK.md` without
version 0: does the agent write the comb itself, as the article's agents wrote Euclid's
algorithm? It is never a candidate for selection.

### 5.2 Rules beyond the article's

1. **Never trained:** no gradients, optimisers, fitting loops, least squares or statistics of
   data used to set weights, and no search over constants in code. A constant changes only by an
   edit between evaluations, logged with its reason and its effect on the objective.
2. **Everything after tokenisation happens inside the network:** no numpy, scipy or lcmsdeconv
   calls on token data outside the primitives, no new preprocessing (smoothing, centroiding,
   baselines), no post-processing of the outputs. An `explore()` function may print visible
   examples; nothing that builds or runs the model may call it.
3. **Known algorithms may be written as weights** - the comb is version 0, and isotope patterns
   and adduct masses are physics; importing the decoder, `CombPredictor` or LadderNet is not
   allowed.
4. **`write_weights` reads only `spec`** - grid step, polarity, charge range, isotope mode, the
   resolving-power law, the proton mass, 1.0033548, the method's adduct deltas, the scan time -
   opens no file, and is deterministic: its weights are hashed at every evaluation.
5. **Constants:** physical constants, `spec` fields, small structural integers and at most 30
   declared tunables, each with a unit and a reason; tables are generated by formulas; time
   constants in seconds or in units of a trace's FWHM. No m/z, mass, retention time or intensity
   that names something in the data ("suppress m/z 1319.39", "the wash starts at 25 min"); a new
   adduct delta is a proposal to the person running the search, not an edit.
6. **Symmetries:** equivariant to shifts along u apart from the declared position-dependent
   physics (isotope spacing, peak width), invariant to shifts in time, charges invariant to a
   common intensity scale.
7. **Precision:** float64; every pointer head declares beta, its search window and its sink, and
   its leakage - including what a gated head reads from a vetoed key - sits below a declared
   readout floor.
8. **Limits:** 5 s per tile (median, one core; confirmed in stage 0), 4 GB, 50 000 non-zero
   weights; and, from the article, "a small improvement that adds ugly complexity is not worth
   it".
9. **Tokens:** the agent may choose bin tokens or event tokens per stage, from the tokenizer's
   two modes.
10. **New primitives** (a clamp activation, a lateral-inhibition argmax) are proposals to the
    person running the search, added to the library with tests and audited like the decoder, not
    written into `model.py`. Proposals of primitives and adduct deltas are decided against a
    pre-registered list of what may be granted, without the hidden runs or the per-species
    traces, and every decision is logged.

### 5.3 Objective, feedback and reports

One pre-registered objective, J = ½ F_syn + ½ R_real. No search starts before the generator has
passed stage 1's calibration and ranking tests.

- **F_syn:** one-to-one species F1 on the visible synthetic runs, counting detectable truth only;
  adduct forms of a found species are not false.
- **R_real** = (2 x paper + SI - X - ½ (forms + same molecule elsewhere) - ¼ Y) / (2 n_paper +
  n_SI) on the visible real runs, where Y counts EIC-confirmed reports outside the truth table
  that also pass three physics checks (no z = 1 isotope partner on the strongest charge state, a
  filled odd isotopologue position for z >= 2, co-eluting charge states) and X every other report
  outside the truth table. A background ladder can pass the extracted-ion check alone (47 of the
  original code's 87 reports did), and a network whose gates keep co-varying rungs could turn
  unconfirmed reports into EIC-confirmed ones; the physics checks are what earn the reduced cost.
- **Per-bin accuracy is not in J**, for the reasons section 3 measured.
- **Eligibility.** J pays 2 for a paper species and 1 for a junk report, so it can gain from
  trades the adoption rule forbids. A model is therefore eligible for selection only if, on the
  visible runs and against comb-2D with the fixes of section 3, it loses no species the comb
  finds, adds no report outside the truth table, adds no report beyond its new species, and keeps
  per-bin accuracy at least the comb's; every report says whether the iterate is eligible.

**Feedback.** Synthetic results come back in full. Real runs come back as counts only (species
found per category, reports per class), at most every fifth iteration, every request logged;
per-species traces of where a species is lost go to the person running the search, not to the
agent. With about ten paper species per fold, richer feedback would let forty iterations tune
thresholds to them. Edits are justified by the last reported objective or by the synthetic
report.

**Reports** (to the operator at every iteration, as a change against version 0 on the same data):
synthetic precision and recall by abundance tier and kind; false species by cause (harmonic, adduct
copy, singly charged ladder, spike-derived, unexplained); the per-bin tables of
`scripts/eval_charge_bins.py`, including its 96-spectrum benchmark for continuity with
`benchmarks.md`; the labelled envelope bins of the known species in the visible real tiles (the only
real per-bin measure); every result split by failure class and by chromatography, so that a gain in
one cannot hide a loss in another; R_real on its own (one paper species moves J by about 0.03 in a
fold, so thresholds on J alone are decided by F_syn); timings, weight counts, tunables and leakage
bounds. Guards set J = -infinity, with the reason: a failed invariance, determinism or symmetry
check, a forbidden import or file read, exceeded limits, or time-of-flight protein F1 more than 0.01
below comb-1D (the baseline of adoption item 4). An evaluation costs about 35 core-minutes (5 min on
8 cores), mostly the synthetic runs (an estimate).

**Noise floor.** Before any search, comb-2D with and without the fixes of section 3 are rerun
under harmless perturbations - noise sigma ±5 %, tiles shifted by half a stride, region margins
±1 scan, a half-bin grid offset - and the spread of their paired difference (species and reports,
per run) is the floor. Every gain threshold - adoption items 2, 3 and 8 and the rebuild test - is
the larger of its stated value and that floor. Dropping alternate scans halves the signal and
the scans per peak, so it is reported as a robustness test, not counted as chance.

### 5.4 Data

**Synthetic.** The current generators cannot make 2D data: `generate_frame_profile` sums its scans
with one shared elution weight, and `generate_run` gives all members of a peak one elution
profile and adds no background. `generate_run` gains:

| Addition | What the generator does | What the real runs show (section 2) |
| --- | --- | --- |
| Instrument | Orbitrap, R proportional to (m/z)^-0.5 with R0 fitted to the runs' peak widths (140 000 nominal at m/z 200); vendor thresholding that leaves segments and gaps as real scans do; 2.3-3.1 s per scan; m/z 400-2500, negative mode | 1-4 % of grid bins with signal per scan |
| Elution | an exponentially modified Gaussian per species, width by class (main peaks broader under load), tail jittered; impurity offsets by kind (shortmers earlier, n+1 later; PO and DA co-eluting under IP-RPLC, 0.46-0.70 min away under IP-HILIC: DA earlier, PO later) | main peaks 5-11 scans FWHM, impurities 3-8 |
| Charge regimes | FLP-like: one state carrying 86-88 %; NC-like: two adjacent states, z = 3 : z = 4 from 43 : 57 (IP-RPLC) to 62 : 38 (IP-HILIC) | FLP's shares similar under both chromatographies; NC's dominant state flips |
| Background | persistent singly charged ions, flat or rising with the gradient; a wash of several hundred to 1 500 broad, mostly singly charged ions, FWHM about twice a main peak's, apexes spread over about a minute, partly co-varying; late re-elution of the main species at 0.1-1 % | wash: pairwise r median 0.54-0.56, 12-17 % of pairs above 0.9, 82-89 % with a z = 1 partner |
| Spikes | one- and two-scan events at the measured rate and intensity | 3-5 % of active bins (30 % at a strong apex), a quarter to a third of persistent bins' intensity |
| Drift and spray | m/z drift of ±2 ppm over a run, 0.5 ppm jitter per scan, a common intensity factor per scan | to be measured |
| Labels | soft charge shares per (bin, scan); species truth with apex time and detectability | |

Calibration (stage 1), per fold from its visible runs only: synthetic tiles within a factor of
1.5 of them in tokens per tile, spike share and intensity, trace breadth, charge-state trace
correlation, charge distribution, wash breadth, wash co-variation and z = 1 partner rate. Impurity
offsets and re-elution are also taken from that fold's visible runs, or left out where they show
none (re-elution was measured only in IP-RPLC NC, hidden in fold B). And the generator must
reproduce every difference between comb-1D, comb-2D, oracle-2D and oracle-flag-2D that exceeds
the noise floor on the visible runs; on two runs most differences will not, and a tie there asks
nothing of the generator. Visible: 32
fixed-seed runs of 10 min with three analyte peaks each, about 250 detectable species (16
IP-RPLC-like and 8 IP-HILIC-like Orbitrap oligonucleotide runs, 8 time-of-flight protein runs as
a guard). Hidden: 96 runs from seeds drawn after the search ends, and 32 shifted-regime runs
(Orbitrap at 70 000, time-of-flight polymers), reported only, not part of adoption.

**Real runs: two diagonal folds.** Each fold shows the agent one run of each chromatography and
hides the other two, so every run is scored once, hidden, by a model whose search never saw it:

| Fold | Visible | Hidden | Comb on the visible runs | Comb on the hidden runs |
| --- | --- | --- | --- | --- |
| A | IP-RPLC NC, IP-HILIC GalNAc | IP-RPLC GalNAc, IP-HILIC NC | paper 5/10, SI 2/14, 2 unconfirmed | paper 8/10, SI 3/15, 5 unconfirmed |
| B | IP-RPLC GalNAc, IP-HILIC NC | IP-RPLC NC, IP-HILIC GalNAc | paper 8/10, SI 3/15, 5 unconfirmed | paper 5/10, SI 2/14, 2 unconfirmed |

Not "IP-RPLC visible, IP-HILIC hidden": the comb already finds all nine IP-RPLC paper species,
and all seven it misses are IP-HILIC ones, so a split by chromatography would hide the largest
real failures from the search. Hidden here means hidden from the agent's loop, not clean. The
comb's constants, the decoder's gates and every probe and circuit of sections 3 and 4 were set or
measured on all four runs, which biases the comparison in both directions - the baseline was
tuned on the hidden runs, and so were the circuits a person would add to it - and is why those
circuits stay out of the agent's sandbox. Each fold's hidden runs also share their analytes with
its visible runs (fold A's hidden IP-RPLC GalNAc species all appear in the visible IP-HILIC GalNAc
table), so a hidden score tests transfer across chromatography, not to unseen analytes. The only
clean real-data test is a sealed fifth run, ideally of another oligonucleotide, and it is the
adoption gate (section 7).

### 5.5 Runs, selection and budget

Per fold, two Opus runs and one Sonnet run of 40 iterations, and in fold A the empty-file control:
seven runs, about 280 evaluations at about 35 core-minutes each - about 20 hours on an 8-core CPU,
about 41 on the 4-core development container. The RTX 4080 adds little: everything is numpy float64
and limited by memory gathers, and the card runs float64 at 1/64 of its float32 rate. A run without
a visible gain of 0.01 in J, or of one species in R_real, by iteration 15 is stopped. After a
plateau the remaining iterations may compress the model (fewer formulas at J within 0.005), as the
article's algorithmic runs did once they reached 100 %; its NLP runs, which like this task have no
exact solution, plateaued late and stayed near 190 000 parameters.

Selection is pre-registered: from each fold's Opus runs, the smallest model (fewest formulas,
then fewest non-zero weights) within 0.005 of the best visible J, scored on that fold's hidden
data once. The two folds' hidden runs differ by about three paper species in difficulty, so the
shipped candidate is the fold model with the larger gain over comb-2D with the fixes on its own
hidden runs (the R_real difference, paired run by run; ties: the smaller model), not the larger
R_real. If neither fold model passes adoption item 3, the shipped candidate is comb-2D with the
fixes. It alone is scored on the sealed fifth run, and rebuild outcomes do not change the
choice.

Agent cost: the article's 34 Opus runs on its 17 algorithmic and NLP tasks cost $449 for 1 334
iterations (Tables A2 and A7), $0.34 per iteration; its ECoG searches cost at least $1 540, a
lower bound covering a 902-iteration final search, precursor and clean-slate searches and runs
stopped at their time limit. At $1-2 per iteration the seven runs cost $280-560, audits and
rebuilds about $150, and exploratory runs - which the article estimates at about twice its final
runs (eight times for the visual cortex) - bring the total to about $1 000-2 000.

### 5.6 Audits and the rebuild test

- **Data invariance:** `write_weights` runs in a subprocess without file access and reads only
  `spec`, so decoy data (other seeds, perturbed visible data - never the hidden runs during the
  search) cannot change its weight hash; the check guards the harness, and its hash must change
  in the documented way with the fields it depends on (resolving power, polarity, step).
  Statistics copied into literals are the auditor's to catch.
- **The executed network:** the evaluator scores the network as built from `model.py` by the
  layer library, never a parallel numpy program; on small inputs that network is compared with a
  dense float64 softmax reference.
- **Automated checks:** only numpy, math and the harness imported, no I/O, no data touched outside
  `explore()`; the literal whitelist of rule 5; symmetry tests - a tile shifted by 1 000 bins
  with the encodings recomputed keeps at least 99 % of active charges on the ladder path (a rigid
  shift changes every implied mass while the data's isotope spacing stays, so isotope circuits are
  tested on species re-rendered at the shifted masses), scans rotated, intensities multiplied by
  10; with every circuit switched off the output equals `CombPredictor` exactly; leakage bounds
  per head.
- **Ablation table:** circuits are removed one at a time, greedily - the one whose removal costs
  least first, re-measuring after each - while J drops by less than 0.005 and no real species is
  lost. One-at-a-time credits mislead when computations share their contribution (88-96 % of the
  explained variance is shared in the article's cortex model, Table A4, and 87 % in its ECoG
  model, Fig. A10), which is why the table follows the article's form but not a fixed
  per-circuit threshold.
- **An auditing agent** reads every saved model and its log: each block's docstring states its
  physical meaning and matches the code; no species-, m/z- or time-specific constants; no
  statistics copied from data in place of physics; every edit justified by its effect on J; no
  reads outside the sandbox.
- **Rebuild:** a fresh session reads the selected model's code, with comments and logs removed,
  and writes two descriptions of at most 200 words: one that may cite version 0's heads (a
  description of the change) and one that stands alone. Per fold, two fresh sessions rebuild from
  each, with the harness, at most six visible evaluations to make the model run (as in the
  article's language-fMRI rebuilds) and no hidden data; two control sessions per fold get the
  empty file, a `TASK.md` without version 0 and no description. The rebuild from version 0 is the
  adoption condition: it loses no paper species the original finds on its hidden runs, keeps at
  least 75 % of the original's hidden R_real gain over version 0 (and at least the noise floor),
  and agrees with the original on what the original changes - charges on the active intensity
  where the original differs from version 0, flags and vote weights - more closely than the
  control sessions do. The standalone rebuild, the article's own test, is reported but not
  required: version 0 with one gate already takes the whole budget (4.8).

## 6. Relation to the comb and LadderNet

- **The comb** is version 0, exactly. Every change is a diff against it, and the comb stays the
  default until a model meets the adoption rule.
- **LadderNet** is the same structure with corrections learned from synthetic spectra: better per
  bin (0.88-0.90 against 0.65 at z >= 2 on time-of-flight spectra, log-SNR), better on synthetic
  species, worse on the real runs because the synthetic spectra lacked the column wash
  (`lessons.md`, section 13) - and, a second possible cause found here, because its training
  renderer resamples thresholded synthetic scans with the same edge widths but on one shared axis,
  an input that matches neither today's real region sums nor the corrected tokenizer. Hand circuits
  will not match its learned reading of weak time-of-flight ions. A written correction has to be
  stated as a rule, which is how a column wash or the override's misreading of a strong envelope
  gets named rather than averaged into weights.
- **They combine.** The frozen 2D decoder takes any charge map, so LadderNet's charges run through
  it as the comb's do (LadderNet-2D, scored like every search); the flat and background vetoes can
  clean what LadderNet reads; the decoder fix of section 3 applies to both. The GPU training of
  `gpu_training/` continues independently, and its best checkpoint is fixed before any hidden
  scoring.
- **Explaining LadderNet.** The article proposes using iWSP to explain black-box models by
  approximating them with a written network. With LadderNet's charge maps as the target instead
  of the truth, the same harness would show which of its learned corrections are rules a person
  can state, and which are not.

## 7. Risks, limits and adoption

- **Synthetic to real.** Rules tuned on synthetic tiles can fail on real runs, as LadderNet's
  corrections did; if generated background is always flat, "flat means background" becomes the
  rule. The calibration and ranking tests of 5.4 and the real-run term of J are the guards; two
  plus two runs of one molecule family are still a small basis, and forty iterations of selection
  can tune thresholds to ten paper species per fold. Measurements made through the distorted
  resampling would have taught the search a rule against an artifact (one-scan events as a third
  of the votes); the generator must be calibrated on correctly resampled scans. The reverse is a
  benefit: a written rule names what it relies on, so a search can expose the generator's
  artefacts, as the article's fMRI search exposed a signal at the start and end of every scan.
- **Gates delete real species.** Co-elution gates cut true partners whose positions are shared,
  a strong species' own outer charge states (FLP's z = 7, NC's z = 6: 4.4) and the Poisson-noisy
  traces of 0.1 % impurities; persistence removes the votes of
  narrow weak species (impurities elute in 3-6 scans) and of species cut at a tile's edge (the
  overlapping tiles reduce that); a breadth rule can delete a broad main band, since the wash is
  only about twice as broad as a main peak, so the z = 1 isotope veto is preferred to breadth.
  Gates must spare weak bins and be judged by species found.
- **Accepting more adds junk.** Single-charge confirmation, regions outside TIC bands and a looser
  background rule can turn the column wash into reports, as `single_charge: true` did (35 more
  reports for 4 new species, 25 of them outside the truth table); the decoder's own narrow windows
  can lose broad species, which stage 0 measures.
- **Rules that help one regime hurt another** (the in-range mean on time-of-flight; C2b at high
  charge). The reports split by kind, charge band and chromatography so that this shows.
- **The resampling fix moves the baseline.** Today's pipeline judges inputs whose noise estimate
  is 40-180 times too high, and patched halfway it reports 154-158 species and different amounts
  (section 2). Comb-1D (the package as it runs) and comb-2D (built on the corrected tokenizer)
  would see different inputs, and part of any gain would be the resampling's, not the decoder's
  or the network's; the package fix, with the decoder recalibrated, is best made before stage 0
  so that both baselines share one input. "Comb-1D" below means the package as released, or with
  the fix if stage 0 makes it, and the fold table of 5.4 is recomputed on it. The training
  renderer (`synth/frames.py`) resamples thresholded synthetic scans with the same edge widths
  on one shared axis, so LadderNet and the checkpoints of the GPU training match neither input;
  they are retrained after the fix, before LadderNet-2D is compared (item 5).
- **Out of reach.** DA under FLP in IP-RPLC and DA-NC under NC co-elute less than one dalton
  apart, with z = 4 envelopes whose cosines are 0.95 (FLP) and 0.92 (NC): that needs exact-formula
  pattern fits. The
  PO abundance under FLP is an NNLS problem. The identity of the +201.4 Da form is chemistry. NC's
  re-elution reports are a linking problem.
- **Time alignment.** The pipeline's apex can sit early (the 1 908 Da fragment, found under
  oracle charges with single charge states accepted, is reported at 5.25 min and tabulated at
  5.52), so evidence must follow each trace, not a fixed window.
- **Fallback value.** Even a model that is not adopted leaves an ablation table showing which
  time-resolved evidence matters; such rules can be folded into the comb.

**Adoption**, pre-registered (its hash committed before the first search). The rule charge models
have had to meet - at least what the comb finds without adding reports - read literally rejects
any model that finds more, since every species found is a report. It becomes, with "outside the
truth table" counting unconfirmed and other EIC-confirmed reports alike:

1. **No loss**, for each fold's model on its hidden runs, against comb-1D: no paper or SI species
   it finds lost (by identity); reports outside the truth table at most its; total reports at most
   its plus the new species found.
2. **A gain**, pooled over the four hidden evaluations: at least two more species, or at least
   three fewer reports outside the truth table, and above the paired noise floor of 5.3. These
   runs shaped the decoder and the circuits, so this is development evidence; item 8 is the test.
3. **From the network:** each fold's model beats comb-2D - with the non-network fixes of section
   3 applied (C2b and the adjacent-isotopologue rule) - on its hidden runs by at least one species
   or two reports outside the truth table, and by more than the paired noise floor; otherwise
   comb-2D with those fixes, the simpler change, is the shipped candidate and must itself pass
   items 1, 2, 4 and 8, with comb-1D as its baseline throughout.
4. **Synthetic, hidden:** Orbitrap oligonucleotide F1 at least comb-2D with the fixes + 0.03 (for
   the fallback, comb-1D + 0.03) with a 95 % bootstrap interval excluding 0; time-of-flight
   proteins within 0.01 of comb-1D; per-bin intensity-weighted accuracy at z >= 2 at least the
   comb's.
5. **Against LadderNet-2D:** at least equal on the counts of item 1 (a tie goes to the written
   model, because it can be audited).
6. **Integrity:** a clean audit, and the rebuild from version 0 passing (the standalone rebuild
   reported).
7. **Speed:** at most 3 min per 30-min run on 8 cores, in numpy float64.
8. **The sealed fifth run**, the adoption gate: a new injection acquired after the search ends,
   ideally of another oligonucleotide, whose truth table is fixed before any compared pipeline
   runs on it, by methods independent of them and blind to their reports, then scored once by the
   same rules. The shipped candidate (5.5) meets item 1 on it and beats comb-1D and - unless it is
   the fallback itself - comb-2D with the fixes, by at least one species or two reports outside the
   truth table and by more than that run's paired noise floor. Before that the model may ship
   only opt-in.

Each fold's model must pass items 1 and 3-7; item 2 is pooled; item 8 decides.

## 8. Plan

| Stage | Work | Go / no-go |
| --- | --- | --- |
| 0 (3-4 days) | the package's resampling fix (per-scan grids summed, raw-point checks on the same input, thresholds recalibrated) or the decision to keep both inputs; plumbing, including the scorer's handling of unscored species; tokenizer and layer library with their tests; version 0 built from the library and run as a network; the 2D decoder with its provenance log; comb-2D with and without the fixes, oracle-2D, oracle-flag-2D (true charges, plus flags on the known single-charge species) and chain-flag-2D (comb charges, flags wherever the chain test fires, no truth) on all four runs; the chain test's pass rate on random positions; the noise floor | version 0, executed, equals `CombPredictor`; sparse and dense attention agree to 1e-12; **oracle-flag-2D beats comb-2D with the fixes, on the same input, by at least two real species with no more reports outside the truth table; chain-flag-2D adds no more reports outside the truth table than real species, and no more than the paired noise floor (the probe's chain test passes 10 of the 19 unconfirmed reports that `single_charge: true` makes, so a zero tolerance would likely fail); comb-2D's time-of-flight protein F1 is within 0.01 of comb-1D - otherwise stop the transformer and fix the decoder** |
| 1 (2-3 days) | the 2D generator additions of 5.4, calibrated per fold; the decoder recalibrated and frozen per fold | synthetic tiles within a factor of 1.5 of the visible real statistics; the ranking test; no search before both pass |
| 2 (1 day) | a 10-iteration pilot; a break-in test of the sandbox; invariance, determinism and audit dry runs; the counts-only feedback | harness bugs fixed; the hidden runs and this document cannot be read |
| 3 (about a week) | the seven searches of 5.5, from version 0 | each run passes or is stopped at iteration 15 |
| 4 (2 days) | hidden scoring, audits, rebuilds, the LadderNet comparison | adoption items 1-7 |
| 5 | the sealed fifth run | adoption item 8 |

The comb stays the default throughout. Independently of the search, three changes are worth
testing on their own: the adjacent-isotopologue condition of section 3, a one-line decoder change
worth trying on LadderNet's charge maps, whose false reports were singly charged ions read at
z = 2-3; C2b, which reads FLP's main envelope right where the comb's override fails, as a candidate
for the comb itself once it has passed the synthetic benchmarks; and the resampling fix, which
needs its thresholds recalibrated before it can replace today's input.
