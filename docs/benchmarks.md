# Benchmarks

Two kinds of evidence: synthetic runs, where the truth is known exactly, and four real
Orbitrap runs, scored against a reference built from resolved isotope spacing alone. Both are
reproducible with the scripts in `scripts/`; see `lessons.md` for how each number was used.

## Synthetic runs

`python scripts/benchmark_synth.py --runs 4 --predictor <p>`: four runs per preset, two
chromatographic peaks per run, each a compound with its impurities. A reported species matches a
true one when its mass agrees within 100 ppm or 0.5 Da, whichever is larger, and its retention
time within half a minute; pairing is one-to-one, abundant species first. Recall counts only
*detectable* species (at least 50 expected ion counts in the second-strongest charge state);
below that, two charge states cannot be confirmed above shot noise by any method.

### Default generator

| Charge assignment | Preset | Precision | Main (>=10 %) | 1-10 % | 0.1-1 % | Median ppm |
| --- | --- | --- | --- | --- | --- | --- |
| comb estimator | protein_tof | 65 % (11/17) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 7.2 |
| comb estimator | oligo_iprp | 48 % (15/31) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.4 |
| comb estimator | polymer | 74 % (14/19) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.6 |
| oracle (perfect) | protein_tof | 92 % (12/13) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 9.2 |
| oracle (perfect) | oligo_iprp | 75 % (15/20) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.4 |
| oracle (perfect) | polymer | 79 % (15/19) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.8 |
| oracle, 20 % of envelopes mislabelled | protein_tof | 92 % (12/13) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 9.0 |
| oracle, 20 % of envelopes mislabelled | oligo_iprp | 65 % (15/23) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.4 |
| oracle, 20 % of envelopes mislabelled | polymer | 83 % (15/18) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.8 |
| comb, before the review fixes | protein_tof | 69 % (11/16) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 7.2 |
| comb, before the review fixes | oligo_iprp | 54 % (15/28) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.4 |
| comb, before the review fixes | polymer | 78 % (14/18) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.7 |
| oracle, before the review fixes | protein_tof | 92 % (12/13) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 9.2 |
| oracle, before the review fixes | oligo_iprp | 88 % (15/17) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.4 |
| oracle, before the review fixes | polymer | 83 % (15/18) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.8 |
| comb, after the resampling fix | protein_tof | 69 % (11/16) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 7.2 |
| comb, after the resampling fix | oligo_iprp | 33 % (14/42) | 88 % (7/8) | 57 % (4/7) | 33 % (2/6) | 1.4 |
| comb, after the resampling fix | polymer | 64 % (14/22) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.6 |
| oracle, after the resampling fix | protein_tof | 86 % (12/14) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 9.2 |
| oracle, after the resampling fix | oligo_iprp | 50 % (14/28) | 88 % (7/8) | 57 % (4/7) | 33 % (2/6) | 1.4 |
| oracle, after the resampling fix | polymer | 65 % (15/23) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.8 |
| comb, before the resampling fix | protein_tof | 34 % (10/29) | 100 % (8/8) | 25 % (1/4) | 0 % (0/2) | 6.7 |
| comb, before the resampling fix | oligo_iprp | 28 % (11/40) | 88 % (7/8) | 43 % (3/7) | 17 % (1/6) | 2.1 |
| comb, before the resampling fix | polymer | 40 % (12/30) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| oracle, before the resampling fix | protein_tof | 75 % (12/16) | 100 % (8/8) | 50 % (2/4) | 0 % (0/2) | 7.2 |
| oracle, before the resampling fix | oligo_iprp | 71 % (12/17) | 88 % (7/8) | 43 % (3/7) | 33 % (2/6) | 1.8 |
| oracle, before the resampling fix | polymer | 60 % (12/20) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| 20 % mislabelled, before the fix | protein_tof | 62 % (10/16) | 88 % (7/8) | 50 % (2/4) | 0 % (0/2) | 9.6 |
| 20 % mislabelled, before the fix | oligo_iprp | 67 % (12/18) | 88 % (7/8) | 43 % (3/7) | 33 % (2/6) | 1.8 |
| 20 % mislabelled, before the fix | polymer | 60 % (12/20) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |

The synthetic runs are thresholded too (each scan keeps three points either side of every point
above a fixed level), so the resampling fix and the recalibration it needed
(`lessons.md`, section 14) change them as well: with the comb, precision rose for every preset
and the 1-10 % tier is found more often (1/4 to 3/4 for proteins). With perfect charges the
oligonucleotide preset then reported 28 species against 17. The additional ones were weak forms
of a main species that the inflated noise estimate used to hide: adduct combinations 220-410 Da
above it (two HFIP and a sodium is +358 Da), which the decoder's adduct merge, knowing multiples
of one adduct type only, did not recognize, and shoulders a few daltons from it. Folding adduct
forms into their species after linking (`architecture.md`, section 5), keeping templates where
they were aligned and re-centring candidates only onto envelopes of their own (section 4) brought
it to 17 again, and raised precision for every preset and estimator; the 1-10 % tier of the
oligonucleotide preset is found more often (4/7 to 5/7).

The review fixes of `lessons.md`, section 15, then cost up to four false species per preset and
charge assignment (six under the legacy generator below, which also finds one more main
species); recall is unchanged. With perfect charges the oligonucleotide preset reports 20 species
against 17, the three more all in one run, where four reports of 0.02-0.17 % of its main species
match nothing: before, the adduct fold, which took any library adduct within a dalton, had put
two of them under species as amine clusters (+TEA+2DIPEA, +2DIPEA+Na) although that run's
generator added no amine at all, only sodium, potassium and HFIP; now mobile-phase adducts fold
only where some region shows them, and the first lies within a dalton of a phosphorothioate G n+1
as well. Folding by the whole library again, with the new tolerance and modification guard, takes
one of the three back - and on the real runs folds a 1.9 % species 138.75 Da above the IP-HILIC
NC product as TEA + potassium, 0.53 Da off, although NC + TEA, which would be the stronger form,
is not reported and no region shows TEA (five fewer reports in all, two of them EIC-confirmed),
so the rule stays. In another run a candidate made of an adduct form, measured 1.5 Da light, is
no longer walked down onto its parent: the average-mass ladder lands it 1.47 Da from the parent's
envelope centre, 0.02 Da beyond the half envelope width that now separates a parent from the
flank of a neighbour.

The same runs with the decoder this package had before the changes in `lessons.md` (with the
generator's elution-profile fix and one-to-one scoring applied to it):

| Charge assignment | Preset | Precision | Main (>=10 %) | 1-10 % | 0.1-1 % | Median ppm |
| --- | --- | --- | --- | --- | --- | --- |
| comb estimator | protein_tof | 10 % (8/84) | 62 % (5/8) | 50 % (2/4) | 0 % (0/2) | 21.8 |
| comb estimator | oligo_iprp | 10 % (7/72) | 50 % (4/8) | 29 % (2/7) | 0 % (0/6) | 28.6 |
| comb estimator | polymer | 13 % (7/55) | 62 % (5/8) | 20 % (1/5) | 33 % (1/3) | 8.9 |
| oracle (perfect) | protein_tof | 34 % (11/32) | 75 % (6/8) | 50 % (2/4) | 0 % (0/2) | 42.2 |
| oracle (perfect) | oligo_iprp | 14 % (8/58) | 62 % (5/8) | 29 % (2/7) | 0 % (0/6) | 18.0 |
| oracle (perfect) | polymer | 16 % (8/50) | 75 % (6/8) | 20 % (1/5) | 33 % (1/3) | 30.8 |

### Legacy generator (heavy per-charge adduct load, independent impurity envelopes)

| Decoder | Preset | Precision | Main (>=10 %) | 1-10 % | Median ppm |
| --- | --- | --- | --- | --- | --- |
| before | protein_tof | 7 % (6/91) | 50 % (4/8) | 0 % (0/6) | 37.1 |
| before | oligo_iprp | 5 % (5/97) | 50 % (4/8) | 20 % (1/5) | 10.7 |
| before | polymer | 9 % (4/44) | 25 % (2/8) | 40 % (2/5) | 1.3 |
| after | protein_tof | 11 % (6/53) | 50 % (4/8) | 33 % (2/6) | 17.6 |
| after | oligo_iprp | 14 % (8/57) | 75 % (6/8) | 20 % (1/5) | 2.4 |
| after | polymer | 33 % (10/30) | 88 % (7/8) | 60 % (3/5) | 1.0 |
| resampling fixed | protein_tof | 11 % (6/55) | 50 % (4/8) | 33 % (2/6) | 9.4 |
| resampling fixed | oligo_iprp | 17 % (9/52) | 75 % (6/8) | 40 % (2/5) | 6.0 |
| resampling fixed | polymer | 45 % (13/29) | 100 % (8/8) | 100 % (5/5) | 0.8 |
| amounts made comparable | protein_tof | 10 % (6/58) | 50 % (4/8) | 33 % (2/6) | 9.4 |
| amounts made comparable | oligo_iprp | 26 % (9/34) | 75 % (6/8) | 40 % (2/5) | 2.7 |
| amounts made comparable | polymer | 61 % (14/23) | 100 % (8/8) | 100 % (5/5) | 1.0 |
| now | protein_tof | 10 % (6/60) | 50 % (4/8) | 33 % (2/6) | 9.4 |
| now | oligo_iprp | 24 % (10/41) | 88 % (7/8) | 40 % (2/5) | 2.3 |
| now | polymer | 61 % (14/23) | 100 % (8/8) | 100 % (5/5) | 1.0 |

### Orbitrap regime

`--presets oligo_orbitrap` (not in the default list): phosphorothioate RNA of 3-9 kDa at
120 000-160 000 resolving power at m/z 200, as on the real runs below (recorded at 140 000; 155 000
measured from their peak widths), decoded with the `orbitrap_iprp_neg_oligo` method, so charges
are also read from resolved isotope spacing. Until that was measured the preset rendered
30 000-63 000 (kept as `oligo_orbitrap_lowres`) and decoded it with the 140 000 method:

| Code | Resolving power | Precision | Main (>=10 %) | 1-10 % | 0.1-1 % | 0.01-0.1 % | Median ppm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before the truth table | 30 000-63 000 | 13 % (12/94) | 88 % (7/8) | 57 % (4/7) | 17 % (1/6) | 0 % (0/2) | 4.1 |
| after it | 30 000-63 000 | 14 % (14/99) | 88 % (7/8) | 71 % (5/7) | 17 % (1/6) | 50 % (1/2) | 8.0 |
| at the real resolving power | 120 000-160 000 | 30 % (14/47) | 100 % (8/8) | 57 % (4/7) | 17 % (1/6) | 50 % (1/2) | 2.8 |
| resampling fixed | 120 000-160 000 | 38 % (17/45) | 100 % (8/8) | 86 % (6/7) | 17 % (1/6) | 100 % (2/2) | 2.4 |
| amounts made comparable | 120 000-160 000 | 68 % (17/25) | 100 % (8/8) | 86 % (6/7) | 17 % (1/6) | 100 % (2/2) | 2.5 |
| now | 120 000-160 000 | 68 % (19/28) | 100 % (8/8) | 100 % (7/7) | 33 % (2/6) | 100 % (2/2) | 2.5 |

At the real resolving power precision is in line with the time-of-flight presets and the mass error
a third of what it was: of 33 false species, 5 were adduct-shifted copies, 2 mass errors and 26
unexplained (before the resampling fix; 5, 0 and 23 of 28 after it; with comparable amounts none,
none and 8; now 2, none and 7, and two more species found, whose templates the 0.3 Da alignment
window reaches). `single_charge: true` changes nothing here: the generator gives every species
several charge states, unlike ion-pairing HILIC on the real runs.

#### Isotope-spacing charges

On isotope-resolved data the comb also reads charges from isotope spacing (`architecture.md`,
section 3). Per bin, on synthetic Orbitrap spectra of 1.5-12 kDa phosphorothioate
oligonucleotides, it is wrong for most ions of charge four and up (see the per-bin table in the
next section): a hypothesis one or two charges too high lands on the flank of the neighbouring
isotopologue, and the rule that keeps the largest charge within 0.6 of the best evidence,
compared in log-SNR, accepts it; where a charge's own spacing is unresolved, a lower charge whose
spacing is resolved is read instead (z = 10 as 6, z = 14 as 11). A guard that keeps a reading
only where twice (or three times) the charge would have been resolved, so that the truth cannot
be a multiple of it, restores the per-bin accuracy - and costs real species:

| Isotope-spacing charges | Per bin, 30 000-63 000 | Per bin, 120 000-160 000 | Real runs: paper, SI species | Reported | Unconfirmed |
| --- | --- | --- | --- | --- | --- |
| none (charge ladders only) | 0.53 / 0.80 | 0.62 / 0.82 | 8 / 20, 1 / 29 | 20 | 3 |
| as they are | 0.17 / 0.19 | 0.52 / 0.65 | 13 / 20, 5 / 29 | 39 | 7 |
| kept where 2z is resolved | 0.52 / 0.79 | 0.64 / 0.83 | 10 / 20, 3 / 29 | 27 | 4 |
| kept where 2z and 3z are resolved | 0.53 / 0.81 | 0.64 / 0.83 | 8 / 20, 2 / 29 | 23 | 4 |
| kept up to z = 4 | 0.49 / 0.75 | 0.69 / 0.87 | 10 / 20, 3 / 29 | 31 | 6 |
| kept up to z = 6 | 0.40 / 0.57 | 0.70 / 0.85 | 11 / 20, 3 / 29 | 33 | 6 |
| kept up to z = 8 | 0.31 / 0.41 | 0.63 / 0.78 | 11 / 20, 3 / 29 | 33 | 5 |

(Per bin: the share of z >= 2 ion bins read right, weighted by log-SNR / by intensity, 16
spectra per resolving power, scored as in the per-bin table of the next section. Measured before
the resampling fix.) The
2z guard loses PO-NC, PO, DA-NC (the comb's strongest report in the IP-HILIC NC run) and a
2.56 kDa shortmer reported at 40 %; adding 3z also loses the full-length product of the IP-HILIC
GalNAc run. On the 2·10⁻⁵ grid no peak is taken to be narrower than two bins, so at m/z 1700
nothing above z = 5 counts as resolved, and correct z = 3 readings are thrown away. Keeping only
low-charge readings loses species too: on the real runs the readings at higher charges are
right. For the real runs'
5 kDa oligonucleotides isotopologues sit about ten grid bins apart at every charge, and the
reader's failure modes, which need heavier species or lower resolving power, hardly arise. The
rule stays as it is: the real runs decided, not the per-bin score.

### Charge network (LadderNet)

LadderNet (`docs/training.md`), `proof` size (9 500 parameters), trained on two CPU threads for
45 minutes: 240 synthetic frames over 12 epochs, with singly charged ions either ignored by the
loss (`--singly-charged ignore`) or absent from the spectra (`drop`); "stage 2" continues the
first for six epochs with singly charged ions trained on and up to 200 background ions per
spectrum.

Charge read correctly on the same 96 fresh spectra of the training mixture (72 time-of-flight;
24 Orbitrap at 35 000-280 000), drawn as training draws its validation spectra, for every
estimator, each run as the pipeline runs it - over the whole spectrum, a network in its
32 768-bin tiles - and scored on the spectrum's labelled crop
(`python scripts/eval_charge_bins.py --frames 96 <models>`). "Comb" is the pipeline's
estimator, with isotope-spacing charges on Orbitrap spectra. Shares of the ion bins whose charge
is read right, each spectrum counting the same in every column: at z >= 2 with bins weighted by
log-SNR (the training metric, which also counts bins at 2-3 sigma, where the comb gives no
answer and the decoder casts no vote), by intensity, and by intensity pooled over all spectra
(so that the strongest few decide); by charge band, intensity-weighted:

| Spectra | Charge assignment | z >= 2, log-SNR | z >= 2, intensity | z >= 2, pooled | z = 1 | z = 2-5 | z = 6-15 | z = 16-40 | z > 40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time-of-flight | comb | 0.65 | 0.84 | 0.86 | 0.06 | 0.55 | 0.79 | 0.81 | 0.70 |
| time-of-flight | LadderNet, z = 1 ignored | 0.88 | 0.95 | 0.94 | 0.15 | 0.74 | 0.98 | 0.96 | 0.88 |
| time-of-flight | LadderNet, z = 1 dropped | 0.90 | 0.96 | 0.94 | 0.00 | 0.79 | 0.98 | 0.97 | 0.89 |
| time-of-flight | LadderNet, stage 2, dense background | 0.80 | 0.91 | 0.93 | 0.79 | 0.67 | 0.96 | 0.93 | 0.85 |
| time-of-flight | LadderNet, corrected mixture, z = 1 ignored | 0.89 | 0.95 | 0.93 | 0.00 | 0.80 | 0.98 | 0.95 | 0.88 |
| time-of-flight | LadderNet, corrected mixture, stage 2 | 0.81 | 0.91 | 0.93 | 0.21 | 0.73 | 0.95 | 0.90 | 0.84 |
| Orbitrap | comb | 0.35 | 0.42 | 0.30 | 0.65 | 0.57 | 0.34 | 0.00 | - |
| Orbitrap | comb without isotope charges | 0.51 | 0.73 | 0.95 | 0.02 | 0.68 | 0.80 | 0.91 | - |
| Orbitrap | LadderNet, z = 1 ignored | 0.83 | 0.92 | 0.95 | 0.24 | 0.87 | 0.94 | 0.96 | - |
| Orbitrap | LadderNet, z = 1 dropped | 0.87 | 0.94 | 0.95 | 0.00 | 0.90 | 0.96 | 0.95 | - |
| Orbitrap | LadderNet, stage 2, dense background | 0.70 | 0.86 | 0.96 | 0.89 | 0.78 | 0.86 | 0.79 | - |
| Orbitrap | LadderNet, corrected mixture, z = 1 ignored | 0.87 | 0.95 | 0.97 | 0.00 | 0.92 | 0.96 | 0.93 | - |
| Orbitrap | LadderNet, corrected mixture, stage 2 | 0.73 | 0.87 | 0.95 | 0.30 | 0.81 | 0.85 | 0.69 | - |

The comb rows are measured with the resampling fixed (before it: time-of-flight 0.65, 0.84,
0.86 at z >= 2, Orbitrap 0.36, 0.43, 0.26 and 0.57, 0.77, 0.95 without isotope charges); the
network rows were measured before it, with networks trained on the old grid views, and are not
re-measured until the networks are retrained.

The networks read weak ions far better than the comb (log-SNR weighting) and most of each
spectrum's intensity better too; pooled over spectra they lead at z >= 2 as a whole, but on
time-of-flight data the comb reads the low charges of the strongest spectra better (z = 2-5,
pooled: 0.87 against 0.70-0.74): a network reads 32 768-bin windows, which cannot hold the
nearest neighbours of a doubly charged ion (34 657 bins away at z = 1, 20 273 at z = 3), while
the comb reads the whole spectrum. On these Orbitrap spectra the isotope-spacing charges make the
comb worse per bin (0.42 against 0.73 without them), yet on the real runs they find five more of
the paper's species ("Isotope-spacing charges" above). The second stage 2 learned singly
charged ions far less than the first (0.21-0.30 against 0.79-0.89 at z = 1) in its six short
epochs. Two earlier versions of this table were biased: the first ran the comb on the crop only,
which hid the far neighbours from it too (0.46 at z >= 2 on time-of-flight spectra), and both
drew each spectrum's mixture component and its resolving power from the same random number, so
that each component covered a narrow slice of its range (the Orbitrap spectra 49 000-54 000);
training had also printed the comb's answer in feature channel 3, computed inside the crop, as
the bar to beat. Independent reviews caught all three. The same benchmark as above, with the
network in place of the comb (`--predictor model.onnx`), all rows measured before the resampling
fix (the comb's current rows are in the first table of this section):

| Charge assignment | Preset | Precision | Main (>=10 %) | 1-10 % | 0.1-1 % | Median ppm |
| --- | --- | --- | --- | --- | --- | --- |
| comb estimator | protein_tof | 34 % (10/29) | 100 % (8/8) | 25 % (1/4) | 0 % (0/2) | 6.7 |
| LadderNet, z = 1 ignored | protein_tof | 52 % (11/21) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 6.7 |
| LadderNet, z = 1 dropped | protein_tof | 52 % (11/21) | 100 % (8/8) | 50 % (2/4) | 0 % (0/2) | 9.0 |
| LadderNet, stage 2, dense background | protein_tof | 65 % (11/17) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 6.7 |
| LadderNet, corrected mixture, z = 1 ignored | protein_tof | 58 % (11/19) | 100 % (8/8) | 50 % (2/4) | 0 % (0/2) | 9.0 |
| LadderNet, corrected mixture, stage 2 | protein_tof | 52 % (11/21) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 6.7 |
| comb estimator | oligo_iprp | 28 % (11/40) | 88 % (7/8) | 43 % (3/7) | 17 % (1/6) | 2.1 |
| LadderNet, z = 1 ignored | oligo_iprp | 39 % (13/33) | 88 % (7/8) | 57 % (4/7) | 33 % (2/6) | 3.7 |
| LadderNet, z = 1 dropped | oligo_iprp | 32 % (12/37) | 88 % (7/8) | 43 % (3/7) | 33 % (2/6) | 2.3 |
| LadderNet, stage 2, dense background | oligo_iprp | 39 % (11/28) | 88 % (7/8) | 43 % (3/7) | 17 % (1/6) | 2.4 |
| LadderNet, corrected mixture, z = 1 ignored | oligo_iprp | 38 % (13/34) | 88 % (7/8) | 57 % (4/7) | 33 % (2/6) | 3.4 |
| LadderNet, corrected mixture, stage 2 | oligo_iprp | 40 % (12/30) | 88 % (7/8) | 57 % (4/7) | 17 % (1/6) | 2.4 |
| comb estimator | polymer | 40 % (12/30) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| LadderNet, z = 1 ignored | polymer | 32 % (13/41) | 100 % (8/8) | 80 % (4/5) | 33 % (1/3) | 1.8 |
| LadderNet, z = 1 dropped | polymer | 55 % (12/22) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| LadderNet, stage 2, dense background | polymer | 20 % (12/59) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 1.1 |
| LadderNet, corrected mixture, z = 1 ignored | polymer | 55 % (12/22) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| LadderNet, corrected mixture, stage 2 | polymer | 39 % (13/33) | 100 % (8/8) | 80 % (4/5) | 33 % (1/3) | 2.2 |
| comb estimator | oligo_orbitrap_lowres | 14 % (14/99) | 88 % (7/8) | 71 % (5/7) | 17 % (1/6) | 8.0 |
| LadderNet, z = 1 ignored | oligo_orbitrap_lowres | 47 % (17/36) | 100 % (8/8) | 86 % (6/7) | 33 % (2/6) | 10.6 |
| LadderNet, z = 1 dropped | oligo_orbitrap_lowres | 43 % (16/37) | 100 % (8/8) | 86 % (6/7) | 33 % (2/6) | 8.2 |
| LadderNet, stage 2, dense background | oligo_orbitrap_lowres | 55 % (16/29) | 100 % (8/8) | 86 % (6/7) | 17 % (1/6) | 8.1 |
| comb estimator | oligo_orbitrap | 30 % (14/47) | 100 % (8/8) | 57 % (4/7) | 17 % (1/6) | 2.8 |
| LadderNet, z = 1 ignored | oligo_orbitrap | 48 % (14/29) | 100 % (8/8) | 57 % (4/7) | 17 % (1/6) | 2.8 |
| LadderNet, z = 1 dropped | oligo_orbitrap | 52 % (14/27) | 100 % (8/8) | 57 % (4/7) | 17 % (1/6) | 2.8 |
| LadderNet, stage 2, dense background | oligo_orbitrap | 57 % (13/23) | 88 % (7/8) | 57 % (4/7) | 17 % (1/6) | 2.8 |
| LadderNet, corrected mixture, z = 1 ignored | oligo_orbitrap | 47 % (14/30) | 100 % (8/8) | 57 % (4/7) | 17 % (1/6) | 2.8 |
| LadderNet, corrected mixture, stage 2 | oligo_orbitrap | 50 % (14/28) | 100 % (8/8) | 57 % (4/7) | 17 % (1/6) | 2.8 |

The first stage 2 is the best model so far for proteins and Orbitrap oligonucleotides and the
worst for polymers, where it reads the small oligomers' low charges as background-like (32
unexplained false species against 6 for the comb). The "corrected mixture" rows are the same two
stages trained on the mixture whose Orbitrap spectra span 35 000-280 000 (a quarter of all
spectra) instead of 30 000-63 000 (15 %): no better on the Orbitrap preset, better for polymers,
mixed elsewhere. These CPU runs are short (240 spectra per stage), and the variants trade off
differently per preset.

Charge-harmonic false species, the comb's characteristic error, fell from 6, 5 and 4 (protein,
oligo, polymer) to 0-3. The synthetic runs contain no singly charged background, so these
tables cannot show what a network trained without z = 1 does with it; the real runs can (next
section).

### Reading these tables

- The **oracle** rows are the ceiling of everything downstream of charge assignment. Before the
  decoder changes, a perfect charge map still left two to five false species for every true one;
  no network could have fixed that. After them, perfect charges give 60-75 % precision, and
  mislabelling a fifth of all charge envelopes as 2z, z/2 or z ± 1 costs almost nothing.
- The gap between the **comb** rows and the oracle rows is what a charge network can win. The
  comb's remaining false species are mostly charge misreadings of strong envelopes that happen to
  agree on two charge states; they are systematic, unlike the independent errors of the confused
  oracle, which is why it fares worse than a 20 %-error oracle.
- Main-component recall is limited by physics in a few runs (a deamidated form one dalton from a
  75 kDa protein cannot be separated from it), and impurities below 1 % are mostly lost to shot
  noise and to the main component's envelope tails.
- Polymer oligomers one repeat unit apart (44.026 Da for PEG) differ from two sodium exchanges by
  0.06 Da, so some are merged as adducts.

## Real Orbitrap runs

Four Thermo Q Exactive runs supplied by the user, published by Tutiš et al., "Ion-Pairing
Hydrophilic Interaction Chromatography for Impurity Profiling of Therapeutic Phosphorothioated
Oligonucleotides", Anal. Chem. 2025, 97, 15717-15726
(https://doi.org/10.1021/acs.analchem.5c01407): negative mode, profile spectra at 140 000 resolving
power (m/z 200), m/z 400-2500. A GalNAc-conjugated, fully phosphorothioated 16-mer (FLP, 6793.98
Da) spiked with 2 % each of its n-1, n+1, PS-to-PO (PO), deaminated (DA) and non-conjugated (NC)
forms, and the NC product (5162.25 Da) spiked with 2 % of its deaminated form (DA-NC), each
separated by ion-pairing reversed phase (IP-RPLC, tributylammonium acetate) and ion-pairing HILIC
(IP-HILIC, triethylammonium acetate at 80 °C). Read with `lcmsdeconv.io.thermo.read_thermo_raw`,
processed with the bundled `orbitrap_iprp_neg_oligo` method and the comb estimator, and scored with
`python scripts/validate_oligo_runs.py <the four .raw files>`.

**Truth.** The average masses of the paper's Table 1, confirmed from the sequences of its
Supporting Information (Table S1; the non-conjugated product NC is C160H205N60O78P15S15); for
NC's own impurities, the PS-to-PO form (16.06 Da below NC), the n-1 and n-2 forms the paper
assigns tentatively (329.3 and 658.6 Da below NC: one or two dA phosphorothioates, by
composition), and the 17 synthesis impurities of the SI's Figure S19 with their retention times
under both chromatographies. Figure S19 gives the neutral mass of an extracted isotope peak, not
an average mass, and does not say which isotopologue; near 2.9 and 4.5 kDa the two most abundant
differ by a few percent, so both readings are accepted. Two of its labels are wrong (the z = 2
masses were computed with three protons; m/z 1281.4753 carries another row's mass) and are
corrected. DA co-elutes with FLP under IP-RPLC, and DA-NC with NC, less than 1 Da apart; those
two are not scored. That leaves 20 species named in the paper and 29 SI-listed ones (the 17
impurities in two chromatographies, minus the ones above, plus a 4842 Da form the SI shows at
12.75 min). A reported species matches within 1 Da and 0.35 min.

**Other reports** are sorted into: forms of a found species (+Na, +K, their combinations, or the
+201.4 Da form described below), charge harmonics of a found species, the same molecule at another
retention time (within 0.3 Da), and the rest, which are "EIC-confirmed" when two of their charge
states have extracted-ion chromatograms that peak within 0.25 min of the reported time (the highest
point there, also the highest within 0.1 min either side, carried by three scans or more and three
times the trace's median within 1.5 min), and the two strongest such charge states co-vary (r >= 0.7
within 0.6 min); "unconfirmed" otherwise. Each chromatogram is judged against its own surroundings;
an earlier version divided it by its maximum over the whole run, so that a weak species whose m/z
window also passes the injection peak or the column wash failed (an independent review found five of
fifteen sequence-identified NC impurities rejected that way). On the four runs the check confirms 47
of the 51 tabulated species at their tabulated times and none of 240 random (mass, time) pairs (4
without the three-scan rule, which the same review asked for: vendor- thresholded traces are zero
most of the time, and one stray scan shared by two windows passed). It is a consistency check, not
truth: a false charge ladder made of co-eluting background can pass it, and a species whose
strongest charge states share their m/z windows with other signal can fail it.

| Code, charge assignment | Paper's species | SI-listed | Reported | Forms of found | Same molecule, other time | EIC-confirmed | Unconfirmed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original, comb | 2 / 20 | 0 / 29 | 87 | 0 | 0 | 47 | 38 |
| before the truth table, comb | 9 / 20 | 4 / 29 | 27 | 2 | 2 | 8 | 2 |
| before the resampling fix, comb | 13 / 20 | 5 / 29 | 39 | 4 | 2 | 8 | 7 |
| before the fix, comb, `single_charge: true` | 15 / 20 | 7 / 29 | 74 | 9 | 3 | 21 | 19 |
| before the fix, LadderNet stage 1, z = 1 ignored | 12 / 20 | 4 / 29 | 64 | 5 | 2 | 13 | 28 |
| before the fix, LadderNet stage 1, z = 1 dropped | 12 / 20 | 5 / 29 | 60 | 5 | 2 | 14 | 22 |
| before the fix, LadderNet stage 2, dense background | 12 / 20 | 4 / 29 | 55 | 5 | 2 | 12 | 20 |
| before the fix, LadderNet stage 1, corrected mixture | 12 / 20 | 5 / 29 | 62 | 5 | 2 | 14 | 24 |
| before the fix, LadderNet stage 2, corrected mixture | 12 / 20 | 4 / 29 | 61 | 4 | 2 | 12 | 27 |
| after the resampling fix, comb | 15 / 20 | 15 / 29 | 198 | 21 | 4 | 100 | 43 |
| after the resampling fix, comb, `single_charge: true` | 15 / 20 | 16 / 29 | 328 | 33 | 7 | 119 | 137 |
| **now, comb** | **18 / 20** | **23 / 29** | **227** | **13** | **5** | **114** | **54** |

(Every row but the first two uses the extracted-ion check described above; those two were scored
with its first version. The resampling fix and the recalibration it needed are `lessons.md`,
section 14; "now" adds consistent amounts, region boundaries that no longer cut species and a
region threshold of three times the TIC's noise (section 15). The LadderNets were trained before
the resampling fix and are not re-measured.)

Before the fix no report of any variant was a charge harmonic (half, a third, double or triple a
found species' mass at its time), which the scoring checks for before the extracted-ion test;
now none of the default's is, and one with `single_charge: true`.

What "now" changed, one step at a time (paper's species / SI-listed / reported / unconfirmed):
after the resampling fix 15/20, 15/29, 200, 44; templates kept where they were aligned when a mass
is re-measured 16/20, 15/29, 205, 46; re-centred candidates aligned afresh, alignment between its
steps, adduct evidence at average-mass shifts and adduct forms folded into their species 16/20,
15/29, 183, 46; the edge rule kept off shared region boundaries, a shared scan counted once and the
apex read on the base form 18/20, 15/29, 200, 52; adduct evidence measured as detection measures
it, and re-centring only onto an envelope centred there 18/20, 16/29, 182, 45; regions down to
three times the TIC's noise 18/20, 19/29, 204, 53; adduct forms folded within twice the linking
tolerance (it was 0.3 Da), unless a known modification explains the mass difference as well, 18/20,
19/29, 192, 48; and the fixes of a review of all this (`lessons.md`, section 15: templates aligned
over 0.3 Da rather than 1.5 bins, re-centring by average-mass shifts onto envelopes of its own,
adduct forms folded before region pieces are merged and only for cation adducts and evidenced
mobile-phase adducts, isomers either side of a region boundary kept apart) 18/20, 23/29, 227, 54.
Two further reviews' fixes leave these numbers as they are: a valley test that read across empty
scans however many joined a 1900.08 Da sliver (0.000 %) to its species, but also two isomers seen
in a scan or two each; read across three empty scans at most, it does neither. The lower region
threshold finds the non-conjugated product in the IP-HILIC conjugate run and three SI-listed
impurities; its eight further unconfirmed reports are all below 0.02 % of the main species. The
alignment window finds four more SI-listed impurities and quantifies small shortmers whose
templates 1.5 bins could not reach (the 1235 Da one from 0.01 % to 4.2 %); 23 of its 35 further
reports are EIC-confirmed.

With the resampling fixed and nothing else changed the comb found 15/20 and 10/29 among 400
reports, 222 of them unconfirmed: the decoder's tests had been tuned on a noise estimate 40-180
times too high. Scaling the corrected noise back up 3-100-fold traded the species found for
false ones without a clean operating point (277 reports at 3-fold, 45 at 100-fold with 12/20 and
8/29). What the current defaults use instead - isotopes of the predicted size and co-eluting
charge states (`architecture.md`, section 3) - and how much each contributes:

(`validate_oligo_runs.py --coelution-r R --comb-fill F`; measured after the resampling fix,
before the changes of `lessons.md`, section 15.)

| Setting | Paper's species | SI-listed | Reported | EIC-confirmed | Unconfirmed |
| --- | --- | --- | --- | --- | --- |
| defaults: comb positions filled to 1/3, charge states co-eluting at r >= 0.5 | 15 / 20 | 15 / 29 | 198 | 100 | 43 |
| co-elution r >= 0.3 | 15 / 20 | 16 / 29 | 227 | 106 | 62 |
| co-elution r >= 0.7 | 15 / 20 | 14 / 29 | 178 | 98 | 27 |
| no co-elution test (`coelution_r: 0`) | 15 / 20 | 16 / 29 | 253 | 104 | 86 |
| comb positions filled to 0.2 | 15 / 20 | 15 / 29 | 201 | 101 | 45 |
| comb positions filled to 0.5 | 14 / 20 | 14 / 29 | 187 | 97 | 37 |
| comb positions only above the noise (as before) | 14 / 20 | 15 / 29 | 290 | 137 | 97 |

The co-elution threshold trades SI-listed impurities against unconfirmed reports; 0.5 keeps all
the paper's species. Its "unconfirmed" column is not independent of it: the extracted-ion check
also asks two charge states to co-vary (r >= 0.7 over its own windows).

The LadderNet rows use the CPU-trained `proof` models of the synthetic section. They find the same
species as the comb but one (a 0.01 % n-2 shortmer), and estimate the PS-to-PO impurity under NC at
10 % of NC where the comb reports 0.2 % (about 5 % by extracted-ion chromatogram), and in the GalNAc
mixture find NC's own 7..16 10-mer (3208.66 Da at 14.2 min, 0.13-0.17 %), which the comb misses; but
they add 13-21 more unconfirmed reports, nearly all below 0.13 % of the strongest species, 1.3-3.9
kDa, in the IP-RPLC column wash: singly charged background read as z = 2-3. Stage 1 never penalises
that reading. Stage 2 (z = 1 trained on, and with up to 200 background ions per spectrum,
`--background-ions 200`) brings the number down from 28 to 20: the remedy is in the right direction,
and six CPU epochs are far from enough of it; `gpu_training/README.md` runs both stages on a GPU.
Trained on the corrected mixture (Orbitrap spectra at the real runs' resolving power), stage 1 drops
four of the column-wash reports (28 to 24) and finds one more SI-listed impurity; the second stage 2
learned singly charged ions far less in its six short epochs (per-bin table) and kept 27. What
remains is mostly one singly charged background ion, read as a 2640.8 Da species at five retention
times. (These tables are after a fix to species linking that had shrunk the masses of tracks with
vanishingly weak frames toward zero; it removed two to three junk reports from every variant.) Most
of the SI-listed impurities missed by every variant are small fragments (1.2-3.8 kDa) eluting at
3-15 min.

66 of the original code's 87 reports were heavier than 8 kDa, which nothing in these samples is
(the largest is n+1 at 7139 Da): singly charged background in the column wash read as charge
ladders. 47 of its reports pass the extracted-ion check, which shows how weak that check is for
such ladders. The corrected resampling brought dozens of them back (9-22 kDa, in the IP-RPLC
wash, up to 15 % of the main species) until the isotope comb asked for peaks of the predicted
size; one report of the current code is heavier than 8 kDa (9.4 kDa at 0.015 %, EIC-confirmed).

Found species and their mass errors against the theoretical average masses, current defaults
(comb):

| Run | Species found (error, Da) | Missed |
| --- | --- | --- |
| IP-RPLC, conjugate | FLP (+0.11), PO (+0.02), n+1 (+0.06), n-1 (+0.00), NC (+0.03) | - |
| IP-RPLC, NC | NC (+0.03), PO-NC (+0.02), n-1 (-0.01) | n-2 |
| IP-HILIC, NC | NC (+0.01), DA-NC (-0.03), PO-NC (+0.30), n-1 (+0.88) | n-2 |
| IP-HILIC, conjugate | FLP (+0.02), n-1 (+0.03), DA (-0.10), PO (+0.08), n+1 (+0.03), NC (-0.01) | - |

The IP-HILIC NC run's n-1 is reported at 4833.86 Da, 13.07 min, 1.9 % of NC; the scorer pairs
the truth with a 0.4 % piece of it at 4833.14 Da (13.14 min, the neighbouring region), which is
closer in mass. Its 0.88 Da error is the union axis's (below). Before the resampling fix the
IP-HILIC conjugate run found FLP alone and IP-RPLC NC found n-2 (0.01 %); n-2 is now cut in the
first pass by the 200-candidate cap of the charge-map peaks (rank 280 of 1 971 in the IP-HILIC
run), and in the second its template is aligned by charge states that failed the gate.

Amounts, as a share of the run's main species, base forms compared (`architecture.md`, section
5), against references built from extracted-ion chromatograms. For IP-RPLC: windows of 2.5
isotope-envelope widths (each side) around every charge state that carries the species, over its
whole elution, less a baseline read 0.6-1.2 min away. For IP-HILIC, where co-eluting forms
overlap, a fuller one (`scratchpad` scripts of the investigation, summarized here): every charge
state that carries the species, windows covering 99 % of its isotope envelope, its own elution
bounds with perpendicular drops at valleys, overlapping forms removed by per-scan fits of their
envelopes, and the instrument threshold's losses modelled; the deaminated forms, 0.98 Da above
their parents, split by the envelope's first moment scan by scan. The same reference is given for
all adduct forms (every form up to 320 Da above the species, fitted jointly where co-eluting
species hold each other's forms). The error bars combine the window, baseline, bounds, overlap
and threshold variants.

| Run | Species | Reference, base form | Reference, all forms | Before (all forms resolved) | Now (base form) |
| --- | --- | --- | --- | --- | --- |
| IP-RPLC, conjugate | PO | 3.8 % | - | 4.1 % | 4.0 % |
| IP-RPLC, conjugate | n+1 | 2.4 % | - | 2.8 % | 2.7 % |
| IP-RPLC, conjugate | n-1 | 1.7 % | - | 1.9 % | 2.0 % |
| IP-RPLC, conjugate | NC | 1.6-1.8 % | - | 1.8 % | 1.7 % |
| IP-RPLC, NC | PO-NC | 3.1 % | - | 3.5 % | 3.8 % |
| IP-RPLC, NC | n-1 | 2.0 % | - | 2.2 % | 2.2 % |
| IP-HILIC, conjugate | n-1 | 3.03 +- 0.12 % | 3.23 +- 0.18 % | 6.2 % | 2.8 % |
| IP-HILIC, conjugate | DA | 2.49 +- 0.25 % | 2.93 +- 0.31 % | not found | 2.4 % |
| IP-HILIC, conjugate | PO | 2.93 +- 0.16 % (1.90 skimmed, 4.19 the whole PO band) | 3.24 +- 0.32 % | 2.7 % | 2.9 % |
| IP-HILIC, conjugate | n+1 | 5.77 +- 0.13 % | 6.07 +- 0.26 % | 5.6 % | 5.9 % |
| IP-HILIC, conjugate | NC | 1.78 +- 0.04 % | 1.73 +- 0.07 % | not found | 1.8 % |
| IP-HILIC, NC | DA-NC | 5.78 +- 0.35 % | 9.84 +- 0.67 % | 7.1 % | 5.5 % |
| IP-HILIC, NC | PO-NC | 2.15 +- 0.12 % (0.93 skimmed, 3.17 the whole PO band) | 3.95 +- 0.33 % | 0.8 % | 3.5 % |
| IP-HILIC, NC | n-1 | 2.45 +- 0.18 % | 4.48 +- 0.40 % | not found | 1.9 % |
| IP-HILIC, NC | NC - dG | 0.66 +- 0.02 % | 1.15 +- 0.10 % | 1.2 % | 0.73 % |
| IP-HILIC, NC | 2887.4 | 1.03 +- 0.02 % | 1.30 +- 0.05 % | 1.0 % | 0.91 % |
| IP-HILIC, NC | 3847.4 | 0.47 +- 0.01 % | 0.65 +- 0.03 % | 0.32 % | 0.49 % |
| IP-HILIC, NC | 4512.6 | 0.26 +- 0.01 % | 0.32 +- 0.02 % | 0.45 % | 0.43 % |
| IP-HILIC, NC | 4842 (12.80 min) and 4841.8 (13.30 min) | 1.24 +- 0.09 and 1.64 +- 0.08 % | 1.88 and 2.05 % | 4841.8 only, 1.8 % | 1.37 and 1.76 % |
| IP-HILIC, NC | n-2 | 0.11 +- 0.04 % | 0.21 +- 0.07 % | not found | not found |

("Before" is the code after the resampling fix, which added each species' resolved adduct forms
to its amount; "now" compares base forms.) Against the base-form reference the amounts of 16 of
the 20 impurities found are within 0.88-1.17 of it. The exceptions: n-1 of the IP-HILIC NC run
(0.76, its mass 0.88 Da off), 4512.6 (1.7; it elutes with n-2, which is not reported, and may
hold it), and PO-NC in both chromatographies (1.23 under IP-RPLC). A band
of PO-mass material (other positions of the phosphorothioate-to-phosphate change) elutes just
before each spiked PO isomer; the pipeline counts it with PO-NC, which shares its region (3.5 %
against 2.15 % for the spiked isomer alone and 3.17 % for the whole band), and not with the
conjugate's PO, whose band lies in the neighbouring region. An
earlier version of this table used a crude reference (charge states 3-8, +-1.2/z Th, within 0.4
min of the apex): such a window holds two or three isotopologues depending on where the average
mass falls between them, and caught 43 % of n+1's envelope against 31 % of FLP's, which made n+1
look 1.4 times stronger than it is (and the 2887.4 Da impurity 1.9 times).

Base forms are what every species has measured the same way, but they are not the whole
amount: under IP-HILIC 48-72 % of every species' signal is in adduct forms (sodium, potassium,
acetate clusters), and the main species carries fewer at its own apex (FLP's all-forms-to-base
ratio is 2.5 there against 3.0-3.2 on its flanks; NC's 2.0 against 4.5). Base-form shares therefore understate an impurity by the ratio
of the two adduct loads: 1.05-1.2 on the conjugate run, up to 1.7-1.9 on the NC run (DA-NC 5.5 %
base form, 9.8 % all forms). Summing every adduct form would remove that, but only if every
species' ladder were resolved to the same depth, which the per-region adduct fits do not do.

The paper's tentative n-1 of the NC product (4832.56 Da, "loss of dG") lies 329.3 Da below
NC: one deoxyadenosine phosphorothioate (4832.98 Da by composition) or one deoxyguanosine
phosphodiester, not a dG phosphorothioate (345.27 Da). The figure's mass is that of the third
isotopologue; the reported average, 4832.95 Da, matches the composition. The other shortmers
found (NC minus dG, T or MeC phosphorothioate, 4816.86, 4842.02 and 4842.97 Da) are within
0.12 Da of theirs. Earlier versions of this section scored the runs against a reference
built from resolved isotope spacing; it sat 0.33-0.35 Da below the theoretical masses and looked
only inside the strongest TIC bands, so it missed every separated impurity and made correct
masses look 0.1-0.5 Da high.

What is still wrong on these runs:

- More reports than the extracted-ion check confirms: 54 unconfirmed (44 after the resampling fix),
  nearly all below 0.1 % of the main species. The largest are forms of FLP 177-277 Da above it in
  the IP-HILIC conjugate run (0.6-2.4 %), clusters of sodium, potassium, acetate and the
  ion-pairing amine beyond the adduct library's three, which carry one dominant charge state. A
  region boundary drawn at a valley keeps species that peak on it (the edge rule is not applied
  there), and so also keeps a neighbour's front or tail that the region's bank reads 0.5-1 Da off
  the neighbour's mass: such reports are small (0.03-0.05 % in the IP-RPLC conjugate run), and
  every rule that would drop them also drops the deaminated form, 0.86 Da from FLP.
- Masses measured on a region's summed spectrum weight its points by count. The summed axis holds
  every point any scan measured, so where more scans cover one side of an envelope than the other
  (n-1 of the IP-HILIC NC run: 25 scans on its heavy side, 9 on its light side) the mean shift is
  pulled to the denser side: 0.88 Da high, against 0.17 Da weighting each point by its share of
  the axis and 0.0-0.1 Da reading each scan. Other species move by less than 0.08 Da; changing the
  weighting moves enough reports that it needs its own validation.
- The deaminated forms sit 0.98 Da above their parents, so their envelopes coincide but for one
  isotopologue. DA in the IP-HILIC conjugate run is found in the region before FLP's; its column
  there also takes FLP's rising front, and its apex is read at 16.13 min against 16.09 by the
  envelope's first moment. Where DA co-elutes exactly with FLP (IP-RPLC) it cannot be told apart.
- n-2 of the NC product (0.1 % as base form) is missed in both chromatographies (see above).
- Templates are aligned within 0.3 Da. A 1.9 kDa shortmer of the IP-RPLC NC run (SI 1883.3), its
  average mass read 0.34 Da light, needs its templates moved 0.36 Da and is quantified at 0.00 %
  (aligned, its single-state fit is 80 times larger). Following the overlap past the window's
  edge, up to half an isotope spacing, quantified it but let candidates beside the IP-RPLC main
  species slide onto the peaks of its adduct forms, and the main species was lost. Reading the
  offset from a candidate's own isotopologue peaks, not from a slide over everything nearby,
  would avoid both.
- A +201.4 Da form co-elutes with every major species, strongest under IP-RPLC (8 % of NC by
  extracted-ion chromatogram); the paper does not mention it. It is reported as separate
  species. Its mass matches tributylamine N-oxide (C12H27NO, 201.35 Da), an oxidation product of
  the IP-RPLC ion-pairing amine, but that is a guess.
- NC re-elutes late in the IP-RPLC gradient and wash and is reported each time (at 20.0 and
  25.4 min, 1.2-1.3 % of NC), as the same mass at another retention time; its extracted-ion peaks
  are at 25.4 and 27.7 min, at 0.1-1 % of its height, on a low hump from about 21 min.
- Average masses are reported. For isotope-resolved data a monoisotopic pattern fit would also
  separate a deaminated form from its parent inside one envelope.
