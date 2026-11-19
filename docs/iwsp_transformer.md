# A transformer whose weights are written, not trained (design)

This is a design for a transformer that performs 2D LC-MS deconvolution with every weight written
by hand in code, following *interpretable weight-space programming* (iWSP; Mehta et al.,
"Interpretable weight-space programming builds transparent transformers and brain encoding
models", [arXiv:2611.18114](https://arxiv.org/abs/2611.18114), 2026). It reads tiles of retention
time x ln(m/z), produces the per-bin charge map the decoder already consumes and, beyond it, the
time-resolved evidence that decides whether a species is real. Nothing in the pipeline changes
yet. Sections 2 and 3 are measurements made for this design on the four real runs and the
synthetic benchmarks; sections 4-8 are the design, the search harness and the plan. Every number
measured here is reproduced by a script in `scripts/iwsp_probes/` (its README lists which).

**In short.**

- The pipeline's comb estimator *is* a small transformer: four layers of hand-set attention
  heads reproduce `CombPredictor` exactly, every field of `ChargePrediction` on every bin, on 128
  synthetic spectra and on all 96 predictor calls of the four real runs (section 4.3). That is
  version 0, and every later version is a readable diff against it.
- On the real runs the charge map is not what loses species. With the true charges given inside
  every known species' envelope, the pipeline finds 13/20 paper species and 6/29 SI-listed ones
  (the comb: 13/20, 5/29). The missed species are visible (median S/N 89) but seen at one charge
  state, judged in a summed spectrum where other ions share their m/z windows, or outside any
  TIC region. In the time-resolved data, 19 of the 31 show three or more co-eluting
  isotopologues at their tabulated times, and none of 240 random positions do (section 3).
- So the design has two parts on 48-scan tiles that cover the whole run: **charge-map
  circuits** that remove false reports (one, an isotope-spacing matched filter, already keeps the
  comb's species with 5 instead of 7 unconfirmed reports), and **species circuits** that use the
  time axis - co-elution, isotope chains, elution shape - to flag species the decoder now
  rejects. A frozen, human-written 2D decoder turns the network's per-token output into species,
  so the network can relabel, down-weight and flag but not invent; NNLS still sets the amounts.
- The search follows the article: an agent edits only the network and its `write_weights`; a
  fixed evaluator scores one pre-registered objective made of synthetic and real species (not
  bins: a rule that lifted per-bin accuracy left species unchanged); the four real runs form two
  folds, so every run is scored once by a model that never saw it; audits check that no weight
  comes from data; and a 200-word description must rebuild the model. Before any search, the
  comb run through the new decoder, and a charge oracle run through it, must show at least two
  more real species than today's pipeline - otherwise the decoder, not a network, is the work.

## 1. The approach in the article

An agent (Claude Code) is given a task description, a rules document, a starting Python file
holding a small transformer and an empty `write_weights(model, task)`, and a fixed evaluator it
cannot edit. It writes the weights directly - attention circuits, MLP neurons, lookup tables,
constants - runs the evaluator on a fixed visible set, revises, and repeats for 30-50
iterations. Training of any kind is forbidden (gradients, optimizers, fitting loops, pretrained
weights), and so is computing the answer anywhere but inside the network. Three checks keep it
honest: an auditing agent reviews every saved model for rule violations and uninterpretable
edits; rerunning `write_weights` on altered task data must leave the weights unchanged; and the
final model is scored on a hidden set, then described in at most 200 words, from which two fresh
sessions rebuild it. On algorithmic tasks the agents translated known algorithms into exact
transformers (Euclid's algorithm as tied ReLU blocks, a prime sieve as pulse neurons with
lateral inhibition); on NLP tasks they wrote crude but readable rule lists. The article names the
limits: shallow motifs and known algorithms work best; deep hierarchies without backpropagation
remain open.

Why it fits this problem:

- **The physics is known.** On the grid u = ln(m/z ∓ m_p) every relation that decides a charge
  is a fixed offset: charge states at ln(z/z'), isotopologues at ln(1 + 1.0034/M), adducts at
  ln(1 + d/M). The comb is already a written network; LadderNet is that network plus corrections
  learned by gradient descent; an iWSP model is that network plus corrections written, audited
  and explained.
- **The remaining failures are rule-shaped** (section 3): species seen at one charge state,
  one-scan spikes, a column wash of singly charged ions, regions the TIC never forms. Each has a
  physical signature in the time-resolved data that a rule can state.
- **The result is auditable.** Every weight has a stated reason and a 200-word description
  rebuilds the model; for impurity profiling under regulation that matters as much as accuracy.
- **No training, no GPU.** The weights are code; evaluation runs on a CPU.

## 2. What the data look like in two dimensions

Measured on the real runs (`benchmarks.md`: Q Exactive Plus, negative mode, profile spectra at
140 000, vendor-thresholded), mostly the IP-RPLC ones, each scan resampled onto the grid for
m/z 400-2500 (91 525 bins) with the pipeline's own `resample_profile`:

| Quantity | Value |
| --- | --- |
| MS1 scans | GalNAc run: 609 in 30 min (3.1 s each); NC run: 692 (2.6 s) |
| Elution width of FLP (6 793.98 Da) and NC (5 162.25 Da), FWHM of a charge state's envelope trace | IP-RPLC: 8-11 scans at 3.1 s (GalNAc run), 6-12 at 2.6 s (NC run), 16-34 s; IP-HILIC: FLP 5 scans |
| FLP at its apex, envelope sums by charge | z = 4: 4.9e8; z = 3: 6.6e7; z = 5: 1.4e7; z = 6: 2.2e6; z = 7: 6.4e5 |
| Grid bins with signal in one scan | 1 600-3 900 (2-4 %) |
| Grid bins with signal in a 24-scan tile (75 s), four tiles | 9 600-13 300; 4 400-6 900 of them above three sigma of the summed spectrum |
| Those active bins with signal in at most 2 of the 24 scans | 21-38 %, carrying 27-45 % of the comb's mass-vote weight at z >= 2 |
| Median summed intensity of such bins against persistent ones (FLP tile) | 2.3e5 against 2.3e5; single-scan events reach 3-5e7 |
| Profile correlation of a species' own charge states | FLP z = 3-7 against z = 4: median r = 0.96 (131 bins); NC z = 3 against z = 4 in IP-RPLC NC: 0.93-1.00 (its 8 strongest bins) |
| Profile correlation of unrelated bins | median 0.25 (FLP tile); random pairs of peaks above 10 sigma (IP-RPLC NC, 32-scan tiles): 90th percentile 0.22-0.36 |
| Chance ladder partners (peaks at a z -> z+1 offset) correlating above 0.9 | 0.5-1.0 % away from the analytes (5.8 % at NC's apex, where NC's own charge states are among them) |
| Column-wash ions (27 min) | rise and fall over about 2 min, four times an analyte's width |
| Co-eluting forms beside the known species (trace correlation with the parent, height relative to it) | +201.4 Da: beside all nine IP-RPLC species, r = 0.89-1.00, 6-12 %; under IP-HILIC only beside FLP (2.8 %): an adduct of the IP-RPLC ion-pairing agent rather than an impurity. -16.06 Da (PS to PO): beside all four IP-RPLC GalNAc species, r = 0.79-0.97, 2.7-5.5 %: real impurities recur and co-elute too. Na and K: 1-5 % under IP-RPLC, 16-46 % under IP-HILIC |

Three things follow. The ladders are short and steep: under ion pairing one charge state carries
most of a phosphorothioate's signal, and the z = 1-2 rungs of a 5-7 kDa species lie above
m/z 2 500, so the isotope spacing (7-10 grid bins here) decides charges more than the ladder
does. A third of the bins the comb reads are one- or two-scan events, as intense as ions in the
summed spectrum and plain in the time axis. And co-elution is specific: a species' own charge
states move together (r = 0.93-1.00) where unrelated signal does not (90th percentile
0.22-0.36); the column wash, whose ions also move together, differs by its breadth.

## 3. What limits the pipeline on the real runs

Measured with `scripts/validate_oligo_runs.py` on the four runs (comb: 13/20 paper species and
5/29 SI-listed found, 39 reports, 7 unconfirmed). The whole four-run validation takes about 30 s,
cheap enough to run at every iteration of a search.

**A charge oracle.** Giving the true charge to every active bin inside the isotope envelope of
every known species eluting in a region (z = 2-12, ±3.5 Da / z; comb elsewhere) finds 13/20 and
6/29. Perfect charges on the known species do not recover the missed ones, so a better charge
map alone cannot. (Abundances do move: PO-NC under NC goes from 0.18 % to 9.3 %.) Accepting
species seen at one charge state as well (`single_charge: true`) finds 16/20 and 8/29; the comb
with that option finds 15/20 and 7/29, at the price of 12 more unconfirmed reports (74
reported, 19 unconfirmed).

**Where the missed species are lost** (`process_run(..., trace=...)`), with the comb as it runs
today, and with the oracle's charges and single charge states accepted:

| Stage | Comb, default | Oracle charges, `single_charge: true` |
| --- | --- | --- |
| No TIC region at their retention time (a TIC gap at 9.4-16.1 min in IP-RPLC NC; NC in IP-HILIC GalNAc) | 6 | 6 |
| Never became a candidate (little mass-histogram weight within 1 Da, or not picked) | 17 | 4 |
| Picked, then rejected by the envelope gate | 4 | 9 |
| Lost in the NNLS fit or after it (linking, flat or edge rules, one-to-one matching: DA one dalton from FLP) | 4 | 6 |
| Missed in all | 31 | 25 |

They are not weak: the strongest charge state of the in-region misses stands at a median
signal-to-noise ratio of 89 in the region's summed spectrum (89 in a 7-scan window around each).
Most are 1.2-4.8 kDa fragments and NC impurities seen mainly at z = 2-3 (the SI lists each
fragment by one m/z and one charge), and the IP-HILIC GalNAc impurities at z = 4. Once charges
and the charge count are no longer the barrier, the envelope gate is: it judges an envelope in
the summed spectrum, where other ions sit in the same m/z window. The 1 908 Da fragment's z = 3
window holds two background ions at constant intensity in every scan beside the fragment's own
isotopologues, which appear and grow as it elutes - indistinguishable in the sum, plain in the
traces. What is missing is not charge accuracy but **time-resolved evidence that an envelope
belongs to one eluting species**, and **regions where the TIC shows no band**.

**The evidence is in the traces.** A test of exactly that kind - in the 7 scans around the
envelope's own maximum, three or more isotopologues at 1.0034/z Th whose traces follow the
envelope's strongest bin at r >= 0.9, and no more than one such trace at the half-spacing
positions - passes for 19 of the 31 missed species at their tabulated times (and 13 of the 18
found ones), and for **none of 240 random (mass, time) positions**. Eleven of the twelve IP-RPLC
NC fragments pass, four of them in the TIC gap; in IP-HILIC, n-1 and n+1 of the conjugate pass,
its DA, PO and NC and NC's own n-1 and n-2 do not. Some of those failures are the probe's: it
places its window at the envelope's largest summed scan within ±0.3 min, and in the IP-HILIC
conjugate run one-scan events of 1e5-1e6 sit in the envelope windows - one draws NC's window
0.24 min past its elution, and moving PO's window by 0.1 min makes it pass. The written circuits
must follow each trace rather than a window. Of the pipeline's unconfirmed reports 2 of 7 pass
(5 of 19 with `single_charge: true`), among them two envelopes next to FLP with six co-eluting
isotopologues each - probably species the truth table does not list.

**Written corrections, scored end to end.** Four rules were tried by hand, as the search would
try them (section 4.4 gives their circuits):

| Rule | Per bin | Real runs: paper / SI / reported / unconfirmed |
| --- | --- | --- |
| comb (version 0) | TOF 0.65 / 0.84 / 0.86; Orbitrap 0.36 / 0.43 / 0.26 | 13/20, 5/29, 39, 7 |
| ladder mean over in-range neighbours only (a neighbour beyond the acquired m/z range is unknown, not absent) | known envelopes in three real tiles up from 0.76/0.82/0.42 to 0.87/0.88/0.84 (intensity); synthetic TOF down, 0.65 -> 0.58 (log-SNR): gains at z = 2-5, larger losses at z = 6-40 | 13/20, 5/29, 40, 7 |
| isotope-spacing matched filter with half-spacing inhibition (C2b, section 4.4) | Orbitrap 0.62 / 0.73 / 0.81 (z = 2-5 up, z = 6-40 down against the plain ladder comb); TOF unchanged | 13/20, 5/29, 38, **5** |
| the same filter with a decision margin (C2) | Orbitrap 0.61 / 0.79 / 0.96 | 11/20, 4/29, 31, 5: PO-NC and DA-NC lost |
| co-elution gate on ladder rungs (12-slot profiles over the region ± one width; weak bins exempt), untuned, three thresholds | - | 13/20, 5/29, 40-42, 6-7 (the added reports: a +K form and EIC-confirmed species, among them NC's 10-mer at 3 208.66 Da that LadderNet also found) |
| decoder: isotope check requires two adjacent populated isotopologue positions for z >= 2 | - | 13/20, 5/29, 38, 6 (with `single_charge: true`: 15/20, 7/29, 73, 18) |

(Per-bin numbers: z >= 2 on the 96 spectra of `scripts/eval_charge_bins.py`, weighted by log-SNR
/ intensity / intensity pooled.) A physically sound rule can help where it was conceived, hurt
elsewhere and leave species unchanged (the in-range mean); another trades accuracy between charge
bands and still removes two false reports (C2b). This is the loop the harness of section 5 runs,
and why its objective is species, not bins.

## 4. The architecture

```
48-scan tile -> grid bins = keys (s, intensity, position, per-scan trace, context)
                        |
hypotheses = queries (active bin x charge, closed-form physics of position and z)
  layer 1    pointer heads: ladder rungs, isotopologues, half-rungs
             (+ co-elution and persistence terms from the traces)          [4.3, 4.4]
  MLP        comb evidence, isotope filter, harmonics, background, spikes  [4.3, 4.4]
  layers 2-4 argmax / broadcast heads: charge per bin                      [4.3]
  consensus  (tied passes) partners read at the consistent charge          [4.4]
  species    isotope-chain flags, mass consensus, elution shape,
             near-isobaric partners, forms                                  [4.5]
                        |
  per token: charges, vote weight, flags, apex scan, annotations
                        |
frozen 2D decoder (votes in mass x time, candidates, existing gates)       [4.6]
                        |
weighted NNLS, template bank, linking, integration (unchanged)
```

### 4.1 Tokens and the time axis

- **Key tokens:** every grid bin of the acquired m/z range (about 92 000 for m/z 400-2500), so
  every position a head can ask for exists. Channels: s = clip(log10(1 + I/sigma), 0, 5) (the
  comb's feature), the raw intensity, the position, and the time profile.
- **Query tokens (hypotheses):** one per active bin (I > 3 sigma) and charge z = 1..Z_max
  (Z_max = 40 for the oligonucleotide methods, 60 elsewhere): about 340 000-440 000 per 48-scan
  tile at Z_max = 40 (32 000-128 000 per TIC region in the drop-in probes of section 3).
  A hypothesis's embedding is closed-form physics of its own position and z: its six ladder
  offsets, the isotope spacing in bins, whether that spacing is resolved, the instrument's peak
  width there.
- **Time** lives in the channels, not in the sequence: each bin carries its intensity in every
  scan of the tile (over sigma), and a LayerNorm without gain turns that trace into a vector of
  zero mean and unit length, so the dot product of two bins' vectors is the Pearson correlation of
  their traces. Context channels hold each bin's mean intensity over the 96 scans before and after
  the tile, so that a plateau broader than the tile shows. Tokens over (scan, bin) would multiply
  the sequence by the number of scans and need more layers to collapse time again; co-elution as
  one dot product is the article's style. With one scan (a summed spectrum, the existing 1D
  benchmarks, public single spectra) every time term is switched off and the model is a 1D charge
  estimator.
- **Tiles.** 48 consecutive MS1 scans (2.5 min at 3.1 s), one every 24 scans, so every scan lies
  in two tiles and a token votes only from the tile whose central 24 scans hold its trace's
  maximum. 48 scans because an analyte's base width (about 25 scans) must fit inside a tile and
  the column wash (about 2 min) must look broader than an analyte. Such tiles of the GalNAc run
  hold 16 400-21 900 tokens, 8 400-10 900 of them above three sigma, a quarter to a third of these
  with signal in at most 2 of the 48 scans; a 30-min run makes 25 tiles. Tiles replace TIC regions:
  the whole run is read. (The drop-in probes of section 3 kept the TIC regions and pooled each
  region and one region width either side into 12 slots.)
- **The alternative: elution-event tokens.** One token per local maximum of the time-smoothed
  (scan x bin) map above three sigma - one isotopologue peak at its elution apex, 148 000-256 000
  per run, 37-67 % of them one- or two-scan events - with its 33-scan trace in the channels, and
  partners required to peak in the same scan (a score term -beta_c (c_q - c_k)^2). A minor form
  eluting inside a major one's bins (DA-NC, 0.6 min before NC in IP-HILIC) becomes a token of its
  own, where a bin's trace over a tile is dominated by the major peak; the price is that version 0
  is exact only in a summed-spectrum mode (one token per bin, a one-scan trace). Which token the
  species stage uses is an architecture choice the search may make (5.2 allows it).

### 4.2 Primitives

All in float64, in the forms the article's models use:

- **Pointer head.** Score -beta (q - p_j)^2 between a query position q and key positions p_j
  (beta = 400), the dot product of [2 beta q, -beta] with [p_j, p_j^2] up to a per-query constant,
  plus a zero-valued sink scoring -beta/4, exactly between a hit (0) and a miss (<= -beta). Only
  keys within ±2 bins of q are evaluated; the omitted ones weigh below e^-(3.75 beta) against the
  sink. A floor neuron relu(read - 1e-100) follows each pointer: the comb decides with strict
  comparisons against zero, and even e^-400 of leakage from a key one bin off turned "no evidence"
  into evidence at 823 bins of one real run before the floor was added. The distance form is
  required in float64: in the dot-product form scores reach beta p^2 ~ 1e12, and float32 errs by
  ~1e5.
- **Interpolation head.** The comb reads isotope neighbours by linear interpolation at
  fractional positions. A softmax head without a sink does this exactly: for a target i0 + f,
  centring the quadratic score at i0 + 1/2 + ln(f/(1 - f))/(2 beta) gives the two bracketing keys
  the weights 1 - f and f, and every other key a weight below max(f, 1 - f)/min(f, 1 - f) x
  e^-2 beta of theirs (checked numerically to 1e-12 at beta = 400). It needs no sink because every
  grid bin is a key, so both bracketing keys exist; a sink at -beta/4 would take up to a third of
  the weight. The prototype uses the equivalent tent kernel relu(1 - |q - p_j|).
- **Argmax head.** A hard argmax over one bin's hypotheses with the sink first and the earliest
  key winning ties: the comb's running top-2 with strict ">".
- **Broadcast head.** A pointer from each hypothesis to its own bin.
- **MLP idioms**, exact for the values that occur: max(a, b) = a + relu(b - a); a step
  relu(K(a - b)) - relu(K(a - b) - 1) with a very large K; a select relu(a - M(1 - f)) +
  relu(b - M f) for a 0/1 flag f; and a gated (GLU-type) product unit where a product is needed.

### 4.3 Version 0: the comb, exactly

- **Layer 1** (hypothesis -> bins): six ladder pointer heads, tied across charges, at
  q = p + round(ln(z/(z ± k))/step), k = 1, 2, 3, with value s, summed in the comb's order into
  ACC; two interpolation heads at p ± ln(1 + 1.0033548/(z (m/z ∓ m_p)))/step.
- **MLP 1:** EV = s · ACC / n_z (n_z: the rungs the comb counts, including those beyond the
  grid's edge); NEAR = 0.75 max + 0.25 min of the two isotope reads; ISO = s · NEAR where the
  spacing is at least 2.5 peak widths.
- **Layer 2** (bin -> its hypotheses): argmax heads for the best ladder charge and the best
  isotope evidence. **Layer 3:** broadcast them back to the hypotheses.
- **MLP 3 / layer 4:** the runner-up charge (the winner masked), and the largest charge whose
  isotope evidence reaches 0.6 of the best (scores z · [ISO >= 0.6 max], argmax): the comb's
  "largest agreeing charge", which handles the divisors of a true isotope charge.
- **MLP 4:** the comb's override - the isotope charge wins with confidence 0.9 where its
  evidence holds a quarter of s^2 - and the confidences best/(best + second + 1e-12), in the
  comb's own float32 arithmetic; an apex neuron from two pointers at p ± 1.

About 170 hand-set numbers plus a Z_max x 7 table of offsets and counts from one formula. A
numpy prototype matches `CombPredictor` in top1_z, top2_z, top1_p, top2_p and apex on every
active bin of 128 synthetic spectra (64 time-of-flight, 64 Orbitrap at 35 000-280 000;
Z_max 40 and 60; 10.8 million active-bin evaluations) and of all 96 predictor calls of the four
real runs (150 773 active bins, both detection passes), whose species tables come out identical.
It runs at about three times the comb's time (0.3 s per dense synthetic spectrum; 0.05 s per real
region). This is the starting point of the search, as the untrained LadderNet is the comb.

### 4.4 Charge-map circuits

Each is a handful of weights on top of version 0. Section 3 gives the measured ones.

- **C2, isotope matched filter** (for Orbitrap data and short ladders). Six isotope heads read
  s at ±1, 2, 3 spacings (N1, N2, N3 the larger of each pair) and four at the half spacings
  ±1/2, ±3/2 (H, their maximum: where only a 2z reading has isotopologues). Evidence
  E(z) = min(s, N1) + 1/2 min(s, N2) + 1/4 min(s, N3) - 2 relu(H - 0.5 N1), where the spacing is
  resolved (3 peak widths and 3 bins); a read takes the maximum over ±1 bin where the
  instrument's peak is wider than a bin. For the true z the half positions are valleys; a 2z
  reading has an empty N1; a z/2 reading has filled half positions and is inhibited. In variant
  C2b the charge with the largest E wins where E reaches half of s and beats the runner-up;
  elsewhere the plain ladder comb's answer stands (it replaces the comb's isotope override).
  Measured in section 3. It loses Orbitrap accuracy at z = 6-40, which the consensus below is
  meant to recover.
- **C1, capped comb:** each rung reads min(s_rung, s + 1), so one strong unrelated ion cannot
  decide a weak ion's charge.
- **C4, harmonics:** two more pointers per hypothesis at round(ln(2z/(2z ± 1))/step), the rungs
  only a ladder twice as dense has (LadderNet reads the same two); EV_z loses s times their mean.
  A neuron firing when only the even rungs (k = ±2) are filled marks a reading at twice the true
  charge.
- **C3, singly charged background:** a z = 1 isotope reading (1.0034 Th apart: 125 grid bins at
  m/z 400, 20 at m/z 2 500, resolved on Orbitrap and TOF alike) that beats every z >= 2 isotope reading
  sends the bin to z = 1 and removes it from other bins' rungs (relu(s - 5 BG) as value).
- **Co-elution gate:** a rung's score gains gamma (r_query . r_key + 2 w_query + 2 w_key), with the
  sink moved to gamma theta: an exact rung is read only if its trace correlates with the bin's
  (r > theta) or either bin is weak (w = 1 below S/N 15), because weak traces are noise. Shared
  windows are the hard case: in IP-RPLC NC, NC's z = 3 bins follow its z = 4 trace at
  r = 0.93-1.00, but the strongest bins of its z = 5 window correlate at 0.04-0.95, because
  another signal there jumps a hundred-thousand-fold within one scan. Version 0 is gamma = 0.
- **Flat-background veto:** a bin within a third of its maximum in both outer slots (the
  pipeline's own flat-species test, per bin) supports no rung and casts no ladder vote; its
  isotope reading still stands, so a flat multiply charged ion keeps an isotope-read charge.
- **Consensus (tied, two passes):** ladder and isotope heads read the neighbour's hypothesis at
  the *consistent* charge (z ± k on a rung, z on an isotopologue) rather than its raw s, with
  values min(logit, s). A partner then counts only if it is itself plausible at that charge, and
  the weak flank isotopologues of an envelope inherit its charge instead of falling back to
  ladder answers (with a decision margin added to C2, PO-NC's weak isotopologues fell back to
  ladder charges and PO-NC was lost on the real runs).

### 4.5 Species circuits

These target the real-run losses of section 3. They write per-token evidence - a confirmation
flag, a vote weight, annotations - that the frozen decoder of 4.6 turns into species; with
event tokens (4.1) the same circuits can assemble species tokens directly.

- **Graded co-elution.** Every partner read of the species circuits is gated as in 4.4, but by
  intensity: no gate where the partner's summed S/N is 5 or less (weak traces are noise), a
  correlation above 0.5 required from S/N 30 (a species' own charge states: r = 0.93-1.00;
  unrelated peaks: 90th percentile 0.22-0.36), linear in between. A hard persistence gate zeroed weak isotopologues
  of real species in the first probe (section 3).
- **Isotope-chain confirmation** (the single-charge acceptance). Sum heads at the isotopologue
  positions ±j 1.0034/z Th (j = 1, 2, 3) and at the half-spacing positions, co-elution gated,
  count co-eluting isotopologues (n_iso) and half-spacing peaks (n_half). The article's
  sinusoidal positions do this in one head: a score term A cos(phi_k - phi_q), with phase
  phi = 2 pi z (m/z ∓ m_p) / 1.0034 selected by the query's charge, peaks at every isotopologue
  position of that charge at once. A flag neuron relu(sum q - 5) over six 0/1 indicators: z >= 2,
  the spacing resolved, n_iso >= 3, n_half <= 1, the isotopologue ratios fitting the class pattern
  for M = z (m/z ∓ m_p) within 0.6 in log10, no second charge state. A real envelope's
  isotopologues co-vary closely (the test of section 3 asks r >= 0.9); a chance ladder at the
  right spacing does not, and neither do constant background ions in the same window. The
  isotopologue part, measured in section 3, holds for 19 of the 31 missed species and for none of
  240 random positions.
- **Mass consensus (tied, three passes).** Each token carries ln M = u + ln z. A head scores
  -(lnM_j - lnM_i)^2 M^2 / (2 x 0.6^2) + ln a_j + ln pi_j (pi: the token's charge distribution)
  with value lnM_j: one Gaussian mean-shift step per pass, the 0.6 Da kernel of the decoder's
  histogram smoothing. Six charge-state heads at the same mass (150 ppm) and z ± 1, 2, 3,
  co-elution gated, count a species' co-eluting charge states, so that a token knows whether its
  species is seen at one charge state or several.
- **Elution shape.** The normalised trace is correlated with three centred Gaussians (sigma 0.7,
  3.5 and 12 scans), and three-way lateral inhibition calls the token a spike, an eluting
  analyte (FWHM 5-10 scans) or broad (the column wash, about 40); spikes and broad tokens lose
  vote weight, which the z = 1 isotope veto of 4.4 backs up.
- **Near-isobaric partners** (DA, 0.984 Da above FLP). For a strong token's main envelope, two
  sum heads collect the light and heavy halves scan by scan; a one-sided drift of ln(heavy/light)
  over two consecutive strong scans proposes a partner at M + 0.984 Da with its apex, for the
  template bank to separate - FLP and FLP + 0.984 Da templates correlate at 0.94, so only the
  time profile separates them, and where they co-elute exactly (IP-RPLC) nothing can.
- **Forms and other elutions (run level).** Heavier co-eluting partners at the method's adduct
  deltas (Na 21.982, K 37.956, the ion-pairing amines) are annotated as forms of the stronger
  parent; a heavier delta recurring beside three or more parents becomes a "co-eluting form
  +delta" - the +201.4 Da form beside all nine IP-RPLC species (section 2). Lighter partners
  never fold: the PO form, 16.06 Da lighter, recurs beside all four IP-RPLC GalNAc species and
  is a real impurity. The same mass at another apex (NC re-eluting in the wash) is annotated as a
  further peak of the same species. Annotations never delete a report.

### 4.6 Readouts and integration

The network ends at per-token outputs; a **frozen 2D decoder**, written by people in stage 0 and
not editable by the search, turns them into species.

- **Per token:** the `ChargePrediction` fields for the tile's summed spectrum (top two charges,
  their shares, apex), a vote weight w in [0, 1] (persistence, background), a confirmation flag
  (a single charge state with a time-resolved isotope chain, 4.5), the apex scan of its trace,
  and optional annotations (partners, forms).
- **The 2D decoder:** every token votes intensity x share x apex weight x w at (ln M, tile start
  + apex scan); votes are smoothed over 1.2 Da (the existing function) and ±2 scans; local maxima
  above 1e-4 of the strongest within ±2 min become candidates (a local floor: one tied to the
  global maximum deletes weak, late species, `lessons.md` section 6); a candidate needs two
  charge states or a confirmation flag; its spectrum is summed over its own elution window
  (apex ± 1.5 x the width of its vote trace); then the existing envelope gate, NNLS fit, template
  bank, linking and flat-background rule run as today. This replaces TIC regions (6 of the 31
  missed species had none) and detection in region-summed spectra.
- **Bounded harm.** The network can relabel charges, lower vote weights and flag single charge
  states; candidate picking, the gates and NNLS stay fixed. Any charge model - the comb,
  LadderNet - runs through the same decoder with w = 1 and no flags; the comb run this way
  ("comb-2D") is the baseline every search is scored against.
- **Drop-in mode.** Without the 2D decoder (or if stage 0 rejects it), the network is a
  `predict_grid` replacement for `CombPredictor` (`make_predictor("iwsp")`), with the region and
  run passed to it by a one-line hook in `process_run`: the mode the probes of section 3 used.
- **Quantitation stays weighted NNLS.** In the article's encoding models the agent writes
  features and a fixed ridge regression reads them out; here the agent writes per-token evidence,
  and the fixed decoder and NNLS read out species and amounts. Explaining strong envelopes away is
  iterative and exact in NNLS; written as a transformer it would be the deep, data-fitted
  computation iWSP handles worst.

### 4.7 Size, speed, precision

Version 0: about 170 hand-set numbers and one table of offsets and counts from one formula. With
the gates and the species circuits, a few thousand numbers from about 15 formulas and at most 30
named constants (beta, gamma, thresholds, the 0.6 Da kernel), each with a one-line physical
reason. Written as dense matrices, every co-elution-gated head repeats one constant across the
48 trace channels of query and key - tens of thousands of non-zero weights that say one thing
each - which is why the rules count formulas and constants and cap non-zero weights at 50 000.
LadderNet has 9 500-22 000 trained parameters. Version 0's ladder layer takes 1.6-2.1 s per
48-scan tile on one core (40-53 s for a 30-min run; 0.05 s per TIC region in the drop-in mode);
the gated version is expected to cost about three times that, inside the limit of 5 s per tile.
Memory, estimated: about 35 MB of channels per tile and 40 MB in flight per gated head, evaluated
one head at a time.

### 4.8 The 200-word description

The article's test of interpretability is that a description of at most 200 words rebuilds the
model. Version 0 with three gates, in the drop-in form the probes of section 3 ran (TIC regions,
12-slot traces), in 198 words:

> Float64, no training, hard-argmax attention. Keys: every grid bin of a region's summed spectrum:
> s = clip(log10(1+I/σ),0,5), raw I, position p, a centred, unit-norm 12-slot elution profile over
> region ± one width. Each bin with I > 3σ queries once per charge z ≤ Zmax.
> Layer 1: six pointer heads, tied across z, score −400(q−p)², zero sink at −100, read s at
> p + round(ln(z/(z±k))/step), k = 1..3, summed in that order; EV = s·sum/count. Two tent heads read
> s at p ± log1p(1.00336/(z·m))/step, m = |m/z − carrier|; ISO = s·(0.75max + 0.25min) where spacing
> exceeds 2.5 peak widths.
> Layers 2-4, over each bin's hypotheses (zero sink first, ties to smaller z): Z1 = argmax EV;
> Z2 = argmax without Z1; ZI = largest z with ISO ≥ 0.6·maxISO, kept if maxISO ≥ 0.25s². If ZI > 0
> it is top charge (p 0.9), a different Z1 second (p 0.1); otherwise p = EV/(EV1+EV2+1e−12). Apex:
> local maximum above 3σ.
> 2D: a rung counts only if profiles correlate (r > 0.5) or either s < 1.2. Flat bins (both outer
> slots ≥ max/3) neither vote z ≥ 2 nor support rungs. EV_z loses s·mean(s at
> p + round(ln(2z/(2z±1))/step)).

## 5. The search harness

The article's setup (section 1), adapted.

### 5.1 Files and sandbox

To be built in stage 0 (section 8); the probes in `scripts/iwsp_probes/` are their prototypes.

| File | Edited by | Content |
| --- | --- | --- |
| `iwsp/tokenizer.py` | nobody | tiles and token channels (4.1): resampling, sigma, s, per-scan and context channels, positional physics |
| `iwsp/layers.py` | nobody | the primitives of 4.2, evaluated exactly and sparsely, with a test against dense softmax attention (agreement to 1e-12 on small tiles); readouts |
| `iwsp/decoder2d.py` | nobody | the frozen 2D decoder of 4.6 |
| `iwsp/model.py` | the agent | `architecture(spec)` (the wiring) and `write_weights(net, spec)`; starts as version 0 |
| `iwsp/RULES.md` | nobody | the article's rules and those of 5.2 |
| `iwsp/evaluate.py` | nobody | the objective and reports on the visible data (5.3) |
| `iwsp/audit.py` | nobody | the checks of 5.6 |

The agent works in a sandbox with no network that holds only `model.py`, the rules, the read-only
harness, the visible tiles and raw files, and a truth file restricted to the visible runs;
`scripts/validate_oligo_runs.py` and `docs/benchmarks.md`, which hold the truth for all four
runs, stay outside, as the data did in the article's visual-cortex search. The evaluator
reports visible results only; hidden scoring is a separate command the operator runs once per
selected model.

The file starts at version 0, for three reasons: the comb is the bar for adoption, so every edit
is a measured change from it; a generic GPT of three layers and width 64 with dense attention
cannot cheaply express offsets of 34 657 bins over 20 000 tokens and would spend its budget
rebuilding the comb; and the 200-word description can cite "the version-0 heads" and spend its
words on what is new. One control run starts from an empty file with the same library: does the
agent write the comb itself, as the article's agents wrote Euclid's algorithm?

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
   resolving-power law, the proton mass, 1.0033548, the method's adduct deltas, the tile length -
   and opens no file.
5. **Constants:** physical constants, `spec` fields, small structural integers and at most 30
   declared tunables, each with a unit and a reason; tables are generated by formulas. No m/z,
   mass, retention time or intensity that names something in the data ("suppress m/z 1319.39",
   "the wash starts at 25 min"); a new adduct delta is a proposal to the person running the
   search, not an edit.
6. **Symmetries:** equivariant to shifts along u apart from the declared position-dependent
   physics (isotope spacing, peak width), invariant to shifts in time, charges invariant to a
   common intensity scale.
7. **Precision:** float64; every pointer head declares beta, its search window and its sink, and
   its leakage sits below a declared readout floor.
8. **Limits:** 5 s per tile (median, one core), 4 GB, 50 000 non-zero weights; and, from the
   article, "a small improvement that adds ugly complexity is not worth it".

### 5.3 Objective and reports

One pre-registered objective, J = ½ F_syn + ½ R_real:

- **F_syn:** one-to-one species F1 on the visible synthetic runs, counting detectable truth only;
  adduct forms of a found species are not false.
- **R_real** = (2 x paper + SI - unconfirmed - ½ (forms + same molecule elsewhere) - ¼ other
  EIC-confirmed) / (2 n_paper + n_SI) on the visible real runs. EIC-confirmed reports cost a
  quarter: they may be real unlisted impurities (section 3), but a co-eluting background ladder
  passes that check too (47 of the original code's 87 reports did).
- **Per-bin accuracy is not in J**, for the reason section 3 measured.

Every iteration reports, as a change against version 0 on the same data: synthetic precision
and recall by abundance tier and kind; false species by cause (harmonic, adduct copy, singly
charged ladder, spike-derived, unexplained); the per-bin tables of `scripts/eval_charge_bins.py`;
for each visible real run, every species found or missed with the stage at which a missed one is
lost, and every other report with its class; timings, weight counts, tunables and leakage bounds.
Guards set J = -infinity, with the reason: a failed invariance or symmetry check, a forbidden
import or file read, exceeded limits, or time-of-flight protein F1 more than 0.02 below version
0. An evaluation costs about 35 core-minutes (5 min on 8 cores); the synthetic runs dominate, the
real-run validation takes 30 s.

### 5.4 Data

**Synthetic.** The current generators cannot make 2D data: `generate_frame_profile` sums its scans
with one shared elution weight, and `generate_run` gives all members of a peak one elution
profile and adds no background. `generate_run` gains, each calibrated against the real runs
before any search:

| Addition | What the generator does | What the real runs show (section 2) |
| --- | --- | --- |
| Instrument | Orbitrap with R0 130 000-160 000 at m/z 200, R proportional to (m/z)^-0.5; vendor threshold keeping 1-3 % of the axis; 3.1 s per scan; m/z 400-2500, negative mode | 140 000 nominal, 155 000 measured from peak widths |
| Elution | an exponentially modified Gaussian per species, width and tail jittered by ±20 %; impurity offsets by kind (shortmers earlier, n+1 later, PO and DA within ±0.4 min), half of them co-eluting exactly | FWHM 8-11 scans; DA 0.4 min before FLP |
| Charge regimes | RPLC-like: z = 3-7 in FLP's ratios (0.13 : 1 : 0.03 : 0.0045 : 0.0013); HILIC-like: one state carries at least 85 % | one dominant charge state under both chromatographies |
| Background | 200-2 000 persistent singly charged ions, flat or rising with the gradient; a wash of 50-200 singly charged ions sharing one profile four times an analyte's width; late re-elution of the main species at 2-5 % | the wash lasts about 2 min; NC is seen again at 20.9 and 27.2 min |
| Spikes | one- and two-scan events without isotope partners, at the measured rate and intensity | 21-38 % of active bins, as intense as ions, up to 5e7 |
| Drift and spray | m/z drift of ±2 ppm over a run, 0.5 ppm jitter per scan, a common intensity factor per scan | to be measured |
| Labels | soft charge shares per (bin, scan); species truth with apex time and detectability | |

Calibration (stage 1): synthetic tiles within a factor of 1.5 of the real ones in tokens per
tile, spike share and intensity, trace breadth, charge-state trace correlation, charge
distribution and wash breadth. Visible: 32 fixed-seed runs of 10 min with three analyte peaks
each, about 250 detectable species (16 RPLC-like and 8 HILIC-like Orbitrap oligonucleotide runs,
8 time-of-flight protein runs as a guard). Hidden: 96 runs from seeds drawn after the search ends,
and 32 shifted-regime runs (Orbitrap at 70 000, time-of-flight polymers), reported only.

**Real runs: two diagonal folds.** Each fold shows the agent one run of each chromatography and
hides the other two, so every run is scored once, hidden, by a model that never saw it:

| Fold | Visible | Hidden | Comb on the visible runs | Comb on the hidden runs |
| --- | --- | --- | --- | --- |
| A | IP-RPLC NC, IP-HILIC GalNAc | IP-RPLC GalNAc, IP-HILIC NC | paper 5/10, SI 2/14, 2 unconfirmed | paper 8/10, SI 3/15, 5 unconfirmed |
| B | IP-RPLC GalNAc, IP-HILIC NC | IP-RPLC NC, IP-HILIC GalNAc | paper 8/10, SI 3/15, 5 unconfirmed | paper 5/10, SI 2/14, 2 unconfirmed |

Not "IP-RPLC visible, IP-HILIC hidden": the comb already finds all nine IP-RPLC paper species,
and all seven it misses are IP-HILIC ones, so a split by chromatography would hide the largest
real failures from the search. Hidden here means hidden from the agent. Every probe of sections
3 and 4, the comb's constants and the decoder's gates have seen all four runs, so the comparison
leans against the search, and new runs - other oligonucleotides, other chromatography, a protein
run - are the hidden set worth acquiring; one sealed fifth run is part of adoption (section 7).

### 5.5 Runs, selection and budget

Per fold, two Opus runs and one Sonnet run of 40 iterations, and in fold A the empty-file control:
seven runs, about 280 evaluations, about 23 hours of an 8-core CPU (no GPU needed). A run without a
visible gain of 0.01 J by iteration 15 is stopped; after a plateau, the remaining iterations
compress the model (fewer formulas at J within 0.005), as the article's runs did. Selection is
pre-registered: from each fold's Opus runs, the smallest model within 0.005 of the best visible J,
scored on the hidden data once. Agent cost: the article's Opus runs on algorithmic tasks cost
$0.34 per iteration ($449 for 1 334 iterations), its large-program ECoG search $1.7 ($1 540 for
902; Table A7). At $1-2 per iteration the seven runs cost $300-560, audits and rebuilds about
$150, and exploratory runs, which cost the article about as much again, bring the total to about
$1 000-1 500.

### 5.6 Audits and the rebuild test

- **Data invariance:** `write_weights` runs in a subprocess without file access; its weight hash
  must not change under decoy data (other seeds, the hidden runs swapped in by the auditor) or
  perturbed truth, and must change in the documented way with the fields it depends on (resolving
  power, polarity, step).
- **Automated checks:** only numpy, math and the harness imported, no I/O, no data touched outside
  `explore()`; the literal whitelist of rule 5; symmetry tests (a tile shifted by 1 000 bins with
  the encodings recomputed keeps at least 99 % of active charges; scans rotated; intensities
  multiplied by 10); with every circuit switched off the output equals `CombPredictor` exactly;
  leakage bounds per head.
- **Ablation table:** for each circuit, the change in J and in real-run species when it is removed;
  a circuit that does not earn 0.005 of J is pruned (the article's unique-variance table).
- **An auditing agent** reads every saved model and its log: each block's docstring states its
  physical meaning and matches the code; no species-, m/z- or time-specific constants; no
  statistics copied from data in place of physics; every edit justified by its effect on J; no
  reads outside the sandbox.
- **Rebuild:** the selected model is described in at most 200 words (citing the version-0 heads,
  giving every constant); per fold, two fresh sessions rebuild it from the description, the
  harness and the visible data, with at most five evaluator calls to fix bugs, not to search. A
  rebuild passes when it loses no paper species the original finds on its hidden runs, keeps at
  least 75 % of the original's hidden gain in J over version 0, and agrees with the original's
  charges on at least 95 % of hidden synthetic active intensity. A control session per fold works
  without a description; it should land near version 0.

## 6. Relation to the comb and LadderNet

- **The comb** is version 0, exactly. Every change is a diff against it, and the comb stays the
  default until a model meets the adoption rule.
- **LadderNet** is the same structure with corrections learned from synthetic spectra: better per
  bin (0.88-0.90 against 0.65 at z >= 2 on time-of-flight spectra, log-SNR), better on synthetic
  species, worse on the real runs because the synthetic spectra lacked the column wash
  (`lessons.md`, section 13). Hand circuits will not match its learned reading of weak
  time-of-flight ions. A written correction has to be stated as a rule, which is how a column
  wash or a one-scan spike gets named rather than averaged into weights.
- **They combine.** The frozen 2D decoder takes any charge map, so LadderNet's charges run through
  it as the comb's do (LadderNet-2D, scored like every search); the flat and spike vetoes can
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
  rule. The calibration of 5.4 and the real-run term of J are the guards; two plus two runs of
  one molecule family are still a small basis, and forty iterations of selection can tune
  thresholds to ten paper species per fold. The reverse is a benefit: a written rule names what it
  relies on, so a search can expose the generator's artefacts, as the article's fMRI search
  exposed a signal at the start and end of every scan.
- **Gates delete real species.** Co-elution gates cut true partners whose positions are shared
  or whose traces are distorted (NC's z = 5 window: r = 0.04-0.95) and fail on the Poisson-noisy
  traces of 0.1 % impurities; persistence removes the votes of narrow weak species and of species
  cut at a tile's edge (the overlapping tiles reduce that); a breadth rule can delete a broad
  main band (in the region where NC re-elutes through the wash, 26 % of the strong active bins
  pass as flat; in 48-scan tiles analytes spread over 12-17 scans and wash ions up to about 27, a
  margin below 2x), so the z = 1 isotope veto is preferred to breadth. Gates must spare weak bins
  and be judged by species found.
- **Accepting more adds junk.** Single-charge confirmation, regions outside TIC bands and a looser
  background rule can turn the column wash into reports, as `single_charge: true` did (+12
  unconfirmed); the decoder's own narrow windows can lose broad species, which stage 0 measures.
- **Rules that help one regime hurt another** (the in-range mean; C2b at high charge). The reports
  split by kind and charge band so that this shows.
- **Out of reach.** DA under FLP in IP-RPLC and DA-NC under NC co-elute less than one dalton
  apart, with isotope patterns whose cosine is 0.96: that needs exact-formula pattern fits. The
  PO abundance under FLP is an NNLS problem. The identity of the +201.4 Da form is chemistry. NC's
  re-elution reports are a linking problem.
- **Time alignment.** The pipeline's apex can sit early (the 1 908 Da fragment: reported at 5.25
  min, eluting at 5.52), so evidence must follow each trace, not a fixed window.
- **Fallback value.** Even a model that is not adopted leaves an ablation table showing which
  time-resolved evidence matters; such rules can be folded into the comb.

**Adoption**, pre-registered (its hash committed before the first search). The rule charge models
have had to meet - at least what the comb finds without adding reports - read literally rejects
any model that finds more, since every species found is a report; it becomes:

1. **No loss** on each fold's hidden runs against today's pipeline: no paper species the comb
   finds lost; SI-listed species found at least the comb's; unconfirmed reports at most the
   comb's; total reports at most the comb's plus the new species found.
2. **A real gain**, pooled over the four hidden evaluations: at least two more species, or at
   least three fewer unconfirmed reports.
3. **From the network**: the model beats comb-2D on the hidden runs by at least one species or two
   unconfirmed reports; otherwise comb-2D, the simpler change, ships instead.
4. **Synthetic, hidden:** Orbitrap oligonucleotide F1 at least comb-2D + 0.03 with a 95 %
   bootstrap interval excluding 0; time-of-flight proteins and polymers within 0.01 of today's
   pipeline; per-bin intensity-weighted accuracy at z >= 2 at least the comb's.
5. **Against LadderNet:** at least equal on the counts of item 1 (a tie goes to the written
   model, because it can be audited).
6. **Integrity:** a clean audit, and both rebuilds passing.
7. **Speed:** at most 3 min per 30-min run on 8 cores, in numpy float64.
8. **A sealed fifth run** - a new injection from the user - meets item 1; until it does, the model
   ships opt-in in the Orbitrap oligonucleotide method.

Both folds must pass items 1-3: that tests the method, not one lucky model.

## 8. Plan

| Stage | Work | Go / no-go |
| --- | --- | --- |
| 0 (2-3 days) | tokenizer, layer library, version 0 as a module with its isotope part tested (the prototype already matches the comb everywhere), the 2D decoder; comb-2D and oracle-2D on all four runs and on synthetic data | version 0 equals `CombPredictor`; sparse and dense attention agree to 1e-12; **comb-2D or oracle-2D finds at least two more real species than today's pipeline with no more unconfirmed reports - otherwise stop the transformer and fix the decoder** |
| 1 (2-3 days) | the 2D generator additions of 5.4 | synthetic tiles within a factor of 1.5 of the real statistics |
| 2 (1 day) | a 10-iteration pilot; a break-in test of the sandbox; invariance and audit dry runs | harness bugs fixed; the hidden runs cannot be read |
| 3 (about a week) | the seven searches of 5.5, starting from the circuits sections 3 and 4 measured: C2b, the isotope-chain confirmation, persistence, co-elution, harmonics, consensus | each run passes or is stopped at iteration 15 |
| 4 (2 days) | hidden scoring, audits, rebuilds, the LadderNet comparison | adoption items 1-7 |
| 5 | opt-in release; a fifth run from the user | adoption item 8, before the model becomes the default |

The comb stays the default throughout. Independently of the search, the adjacent-isotopologue
condition of section 3 is a one-line decoder change worth testing on LadderNet's charge maps,
whose false reports were singly charged ions read at z = 2-3, and C2b - the same species with
two fewer unconfirmed reports - is a candidate for the comb itself once it has passed the
synthetic benchmarks.
