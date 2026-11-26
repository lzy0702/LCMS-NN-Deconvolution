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
| comb estimator | protein_tof | 61 % (11/18) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 7.2 |
| comb estimator | oligo_iprp | 50 % (15/30) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.1 |
| comb estimator | polymer | 74 % (14/19) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.6 |
| oracle (perfect) | protein_tof | 92 % (12/13) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 9.2 |
| oracle (perfect) | oligo_iprp | 75 % (15/20) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.1 |
| oracle (perfect) | polymer | 79 % (15/19) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.8 |
| oracle, 20 % of envelopes mislabelled | protein_tof | 92 % (12/13) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 9.0 |
| oracle, 20 % of envelopes mislabelled | oligo_iprp | 65 % (15/23) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 0.9 |
| oracle, 20 % of envelopes mislabelled | polymer | 83 % (15/18) | 100 % (8/8) | 100 % (5/5) | 33 % (1/3) | 0.8 |
| section 16, comb | oligo_iprp | 50 % (15/30) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.4 |
| section 16, oracle | oligo_iprp | 75 % (15/20) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.4 |
| section 16, 20 % mislabelled | oligo_iprp | 65 % (15/23) | 88 % (7/8) | 71 % (5/7) | 33 % (2/6) | 1.4 |
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
charge assignment (six under the legacy generator below, which also finds one more main species);
recall is unchanged. With perfect charges the oligonucleotide preset reports 20 species against 17,
the three more all in one run, where four reports of 0.02-0.17 % of its main species match nothing:
before, the adduct fold, which took any library adduct within a dalton, had put two of them under
species as amine clusters (+TEA+2DIPEA, +2DIPEA+Na) although that run's generator added no amine at
all, only sodium, potassium and HFIP; now mobile-phase adducts fold only where some region shows
them, and the first lies within a dalton of a phosphorothioate G n+1 as well. Folding by the whole
library again, with the new tolerance and modification guard, takes one of the three back - and on
the real runs folds a 1.9 % species 138.75 Da above the IP-HILIC NC product as TEA + potassium,
0.53 Da off, although NC + TEA, which would be the stronger form, is not reported and no region
shows TEA (five fewer reports in all, two of them EIC-confirmed), so the rule stays. In another run
a candidate made of an adduct form, measured 1.5 Da light, is no longer walked down onto its
parent: the average-mass ladder lands it 1.47 Da from the parent's envelope centre, 0.02 Da beyond
the half envelope width that now separates a parent from the flank of a neighbour. Section 16's
changes - edge-pinned templates placed by the candidate's own isotope comb, template placement and
the adduct walk read on the candidate's own charge states, a comb of half the charge no longer
counted as support - then move the time-of-flight presets by at most one false species (one more
with the comb on protein_tof and under the legacy generator on oligo_iprp, one fewer with the comb
on oligo_iprp and under the legacy generator on protein_tof; the rest unchanged) and take three
from the Orbitrap preset, the regime of the real runs; recall is unchanged.

The "section 16" rows are the oligonucleotide preset before the known-issue fixes of `lessons.md`,
section 17; its protein and polymer rows are the same then and now. Those fixes leave recall
unchanged in every preset and tier of these tables (over 16 Orbitrap seeds, not four, they lose one
main species, below). Reporting a species whose templates its own isotope comb placed at the
templates' position takes the oligonucleotide preset's median error from 1.4 to 1.1 ppm (0.9 with
mislabelled charges); the Orbitrap preset loses precision to two adduct forms now placed on their
own comb (76 to 70 %, below), and the legacy generator's tables are as they were (below). Setting
aside a main species' later elutions took one false polymer species away under each charge
assignment (3408.16 Da at 2.08 min in seed 1001, the main compound's tail read again in the second
peak's region: comb 74 to 78 %, perfect charges 79 to 83 %, mislabelled 83 to 88 %), but only the
oligonucleotide methods turn it on, as it would set aside a later-eluting isomer too
(`architecture.md`, section 5), and the polymer rows are as they were.

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
| section 16 | polymer | 61 % (14/23) | 100 % (8/8) | 100 % (5/5) | 1.0 |
| now | protein_tof | 10 % (6/59) | 50 % (4/8) | 33 % (2/6) | 9.4 |
| now | oligo_iprp | 24 % (10/42) | 88 % (7/8) | 40 % (2/5) | 2.3 |
| now | polymer | 61 % (14/23) | 100 % (8/8) | 100 % (5/5) | 1.0 |

(All three presets are the same at section 16 and now.) For a while section 17 gave the polymer
preset 64 % (14/22): in seed 1000 the templates of a 9197 Da candidate, which stopped at the
alignment window's edge 0.02 Da from a tooth of its own comb inside the window, were placed by the
comb on the tooth one spacing further, the joint fit's choice, an unexplained false species of
13796 Da was no longer reported and a matched 9.25 kDa species moved 0.15 Da further from its truth
(+0.44 to +0.59 Da). The edge's own tooth is now taken (`architecture.md`, section 4), and both are
as they were.

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
| three reviews | 120 000-160 000 | 68 % (19/28) | 100 % (8/8) | 100 % (7/7) | 33 % (2/6) | 100 % (2/2) | 2.5 |
| section 16 | 120 000-160 000 | 76 % (19/25) | 100 % (8/8) | 100 % (7/7) | 33 % (2/6) | 100 % (2/2) | 2.5 |
| now | 120 000-160 000 | 70 % (19/27) | 100 % (8/8) | 100 % (7/7) | 33 % (2/6) | 100 % (2/2) | 2.5 |

At the real resolving power precision is in line with the time-of-flight presets and the mass error
a third of what it was: of 33 false species, 5 were adduct-shifted copies, 2 mass errors and 26
unexplained (before the resampling fix; 5, 0 and 23 of 28 after it; with comparable amounts none,
none and 8; after the reviews 2, none and 7, and two more species found, whose templates the 0.3 Da
alignment window reaches; with edge-pinned templates placed by their comb (section 16), 2, none
and 4; now 4, none and 4). Section 17 replaces an alignment optimum that lies between the comb's
teeth by the better tooth, and two adduct forms read 0.4-0.7 Da off, between their teeth, are now
placed on their own comb and fitted whole: the +K form of an undetected 3452 Da impurity
(seed 1000) and a TEA form of a 12-mer in a spectrum that evidences no TEA, so that the adduct-form
exemption does not know it (seed 1001). They are adduct forms listed as species of their own,
which the adduct module is to decide; recall is the same in every tier.

Over 16 seeds (`--runs 16`, seeds 1000-1015) the code of section 16 reported 165 species, 75 of them
matched, and the current code reports 165 with 74 matched (45 % both; median error 3.0 to 2.5 ppm).
Besides those two adduct forms, a third adduct-shifted false species comes with reporting
comb-placed species at their templates' position (seed 1013), an unexplained one goes with the
joint-fit tooth (seed 1005) and a mass error with the join of boundary pieces (seed 1010). And one
main species is lost (main tier 31/32 to 30/32): the 6.42 kDa compound of seed 1008's second
chromatographic peak, whose second-strongest charge state carries about 170 times fewer expected
ions than the first peak's main compound's, and whose candidate no longer makes the decoder's list
of 200 since the method counts z = 1 - peaks voted for by z = 1 beside another charge state take
places in that list. With 300 places, or with charges from 2, it is found. The four seeds of the
table do not show it. `single_charge: true` changes nothing here: the generator gives every species
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
  decoder changes, a perfect charge map still left two to five false species for every true one; no
  network could have fixed that. After them, perfect charges gave 60-75 % precision (75-92 % in the
  first table now), and mislabelling a fifth of all charge envelopes as 2z, z/2 or z ± 1 costs
  almost nothing.
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
two are not scored (by their envelopes' first moments they elute about 0.5 and 0.6 min after
their parents; see the deaminated forms below). That leaves 20 species named in the paper and 29
SI-listed ones (the 17 impurities in two chromatographies, minus the ones above, plus a 4842 Da
form the SI shows at 12.75 min). A reported species matches within 1 Da and 0.35 min.

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
| amounts made comparable, three reviews, comb | 18 / 20 | 23 / 29 | 227 | 13 | 5 | 114 | 54 |
| section 16, comb | 20 / 20 | 23 / 29 | 228 | 13 | 5 | 115 | 51 |
| section 17, comb | 20 / 20 | 29 / 29 | 250 | 13 | 3 | 127 | 58 |
| **now, comb** | **20 / 20** | **29 / 29** | **252** | **13** | **3** | **127** | **60** |

(Every row but the first two uses the extracted-ion check described above; those two were scored
with its first version. The resampling fix and the recalibration it needed are `lessons.md`,
section 14; "amounts made comparable, three reviews" adds consistent amounts, region boundaries
that no longer cut species and a region threshold of three times the TIC's noise, with that
section's review fixes (section 15); "section 16" templates placed by the candidate's own isotope
comb, template placement and the adduct walk read on its own charge states, a comb of half the
charge no longer counted as support and deep sodium and potassium forms recognised by their comb
(its one charge harmonic is counted in no column); "section 17" the known-issue fixes of that
section, with no charge harmonic reported; and "now" harmonic readings of any ratio dropped and the
first pass's two strongest candidates held at their mass by the adduct dedupe (below). "Same
molecule, other time" falls from 5 to 3 because NC's tail and column-wash elutions in the IP-RPLC
NC run are now named in NC's flags. The LadderNets were trained before the resampling fix and are
not re-measured.)

The 29 SI-listed species found need a qualifier: 1908.3, 2887.4 and 3206.4 of the IP-RPLC NC run are
reported 0.52-0.60 Da light (1909.10, 2887.83 and 3207.18 Da, against 1909.65, 2888.43 and
3207.70 Da by their own isotopologues; IP-HILIC reports them at 1909.56, 2888.39 and 3207.46 Da),
and the scorer matches each to the SI's other isotopologue reading (1908.64, 2887.43 and 3206.70 Da)
within its 1 Da tolerance. Their templates sit on the right tooth, but more than half a spacing from
the measured mass, where the reported mass does not follow them (below). 3206.4 is also quantified
at under half its reference: 0.020 % against 0.042 % by extracted ions (0.48 times), with 13 % of
its z = 3 elution before its region. 2228.3 of the same run is reported 0.75 Da light (2229.15 Da
against 2229.90 Da; IP-HILIC reads 2229.78 Da) with its templates on the tooth one spacing light, as
2558.3's are.

Before the fix no report of any variant was a charge harmonic (half, a third, double or triple a
found species' mass at its time), which the scoring checks for before the extracted-ion test.
Section 16 let one of the default's through - 3160.68 Da at 0.008 %, the 1580.18 Da shortmer's
non-covalent dimer [2M-3H]3-, reported since a comb of half the charge no longer counts as
support - and `refine.drop_multimers` now drops it by its fitted amount and its co-elution with
the shortmer, so none of the default's is; one was with `single_charge: true` after the
resampling fix.

What the rows "amounts made comparable, three reviews" and "section 16" changed, one step at a time
(paper's species / SI-listed / reported / unconfirmed): after the resampling fix (a later
measurement than the table's row, which gives 198 and 43) 15/20, 15/29, 200, 44; templates kept
where they were aligned when a mass is re-measured 16/20, 15/29, 205, 46; re-centred candidates
aligned afresh, alignment between its steps, adduct evidence at average-mass shifts and adduct forms
folded into their species 16/20, 15/29, 183, 46; the edge rule kept off shared region boundaries, a
shared scan counted once and the apex read on the base form 18/20, 15/29, 200, 52; adduct evidence
measured as detection measures it, and re-centring only onto an envelope centred there 18/20, 16/29,
182, 45; regions down to three times the TIC's noise 18/20, 19/29, 204, 53; adduct forms folded
within twice the linking tolerance (it was 0.3 Da), unless a known modification explains the mass
difference as well, 18/20, 19/29, 192, 48; and the fixes of a review of all this (`lessons.md`,
section 15: templates aligned over 0.3 Da rather than 1.5 bins, re-centring by average-mass shifts
onto envelopes of its own, adduct forms folded before region pieces are merged and only for cation
adducts and evidenced mobile-phase adducts, isomers either side of a region boundary kept apart)
18/20, 23/29, 227, 54. Two further reviews' fixes leave these numbers as they are: a valley test
that read across empty scans however many joined a 1900.08 Da sliver (0.000 %; a 0.13 % species
whose templates sat at the alignment window's edge, quantified at 0.11 % since section 16) to its
species, but also two isomers seen in a scan or two each; read across three empty scans at most, it
does neither. Templates of edge-pinned candidates placed by their own isotope comb, template
placement and the adduct walk read on the candidate's own charge states, a comb of half the charge
no longer counted as support, and deep sodium and potassium forms recognised by their comb
(`lessons.md`, section 16), with three reviews' corrections, 20/20, 23/29, 228, 51: n-2 is found in
both chromatographies and the 1.9 kDa shortmer 1883.3 quantified. The lower region threshold finds
the non-conjugated product in the IP-HILIC conjugate run and three SI-listed impurities; its eight
further unconfirmed reports are all below 0.02 % of the main species. The alignment window finds
four more SI-listed impurities and quantifies small shortmers whose templates 1.5 bins could not
reach (the 1235 Da one from 0.01 % to 4.2 %); 23 of its 35 further reports are EIC-confirmed.

What "now" changed against "section 17": the harmonic-reading rule changes no report of the four
runs, and the IP-RPLC NC run and both IP-HILIC runs report exactly what they did. In the IP-RPLC
conjugate run the adduct dedupe no longer moves FLP's candidate onto the 6750.58 Da envelope with
9 % of its score (two sodium below it, to 0.46 Da), nor 3527.9's onto a 3268.96 Da envelope with
1 % of its score (two DIPEA below, to 0.48 Da). The 6750 Da species the move hid is reported
(6750.55 Da, 2.20 %, unconfirmed), and so are 6474.81 Da (0.040 %, EIC-confirmed) and a
3587.35 Da sliver (0.000 %); PO moves from 3.96 to 3.80 % (all forms 4.86 to 4.56 %). 3527.9's
region (14.97-15.69 min) now evidences sodium, and the 3729.15 Da species' new sodium column, 1.4 Da
from 3749.72 Da (0.004 %, EIC-confirmed), removes that report through the shared-evidence test.

What "section 17" changed against "section 16", one fix at a time (`lessons.md`, section 17; the
paper's species 20/20 throughout; SI-listed / reported / EIC-confirmed / unconfirmed): the merge of
a species' region pieces repeated until nothing more joins, 23/29, 226, 115, 49; NC's tail and
column-wash elutions named in its flags, 223, 114, 49; pieces a region boundary cut off joined to
their band by their isotope tooth, 220, 112, 48; the oligonucleotide methods' region floor at
0.1 % of the tallest TIC peak, 24/29, 224, 114, 49; charge states that fail co-elution on their
windows judged again on their own isotopologues, 25/29, 227, 114, 51; multimers dropped by their
fitted amounts, 226, 114, 51; the Orbitrap oligonucleotide method down to 800 Da with z = 1 beside
z = 2, 27/29, 242, 123, 56; alignment optima between the comb's teeth replaced by the better tooth,
29/29, 251, 131, 55; the tooth chosen by a joint fit with same-comb neighbours, unchanged;
comb-placed species reported at their templates' position, 250, 127, 58; and, unchanged in these
counts, the later-elution set-aside made a method setting, multimers dropped only where their
co-elution can be judged, and alignments that stop at the window's edge within 0.15 Da of a tooth
of the comb placed on that tooth (3207.4 from 3208.38 to 3208.69 Da, two unconfirmed reports onto
their teeth). Scored mass errors on the 43 species found before: median 0.080 to 0.060 Da,
mean 0.142 to 0.105 Da, 11 to 6 above 0.2 Da and 6 to 2 above 0.3 Da; on all 49 found now,
median 0.060 Da, mean 0.125 Da, 9 above 0.2 Da and 5 above 0.3 Da.

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

The LadderNet rows use the CPU-trained `proof` models of the synthetic section, and are compared
here with the comb before the resampling fix. They find the same species as that comb but one (a
0.01 % n-2 shortmer), and estimate the PS-to-PO impurity under NC at 10 % of NC where that comb
reported 0.2 % (about 5 % by extracted-ion chromatogram; the current comb reports 3.5-3.8 %), and in
the GalNAc mixture find NC's own 7..16 10-mer (3208.66 Da at 14.2 min, 0.13-0.17 %), which that comb
missed (the current comb reports it, 3208.54 Da at 14.20 min, 0.074 %, EIC-confirmed); but they add
13-21 more unconfirmed reports, nearly all below 0.13 % of the strongest species, 1.3-3.9 kDa, in
the IP-RPLC column wash: singly charged background read as z = 2-3. Stage 1 never penalises that
reading. Stage 2 (z = 1 trained on, and with up to 200 background ions per spectrum,
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
such ladders. The corrected resampling brought dozens of them back (9-22 kDa, in the IP-RPLC wash,
up to 15 % of the main species) until the isotope comb asked for peaks of the predicted size; one
report of the current code is heavier than 8 kDa (9.4 kDa at 0.017 %, EIC-confirmed).

Found species and their mass errors against the theoretical average masses, current defaults
(comb):

| Run | Species found (error, Da) | Missed |
| --- | --- | --- |
| IP-RPLC, conjugate | FLP (+0.11), PO (+0.02), n+1 (+0.06), n-1 (+0.00), NC (+0.03) | - |
| IP-RPLC, NC | NC (+0.03), PO-NC (+0.02), n-1 (-0.01), n-2 (-0.06) | - |
| IP-HILIC, NC | NC (+0.01), DA-NC (-0.03), PO-NC (+0.30), n-1 (-0.06), n-2 (-0.06) | - |
| IP-HILIC, conjugate | FLP (+0.02), n-1 (+0.03), DA (-0.09), PO (+0.08), n+1 (+0.03), NC (-0.01) | - |

The IP-HILIC NC run's n-1 is one report, 4832.92 Da (-0.06 Da) at 13.07 min, 2.35 % of NC: the 0.47
% piece a region boundary cut off (4833.14 Da at 13.14 min) is joined to its band by its isotope
tooth, and the joined species is reported at that piece's comb position; the band's main piece reads
4833.86 Da on the union axis (below), and until section 17 the scorer paired the truth with the
piece. n-2 is reported at its comb position in both runs, 4503.65 Da (4503.22 and 4503.98 Da before
section 17). Before the resampling fix the IP-HILIC conjugate run found FLP alone and IP-RPLC NC
found n-2 (0.01 %). n-2 was then lost two ways (`lessons.md`, section 16): under IP-HILIC an
unrelated peak in a charge window that failed the gate walked it down a sodium onto a phantom, which
was dropped as flat; under IP-RPLC its mass was read 0.55 Da light, the overlap search, following
charge states that had failed the gate, pinned its templates at the window's far edge, 0.2 Da beside
a wrong isotopologue, they fitted other ions, and it was reported at the wrong time.

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
| IP-RPLC, NC | n-2 | 0.074-0.078 % (its own isotopologues; 0.15 % in the windows, below) | - | not found | 0.08 % |
| IP-RPLC, NC | 1883.3 | 0.09-0.10 % | - | 0.00 % | 0.08 % |
| IP-HILIC, conjugate | n-1 | 3.03 +- 0.12 % | 3.23 +- 0.18 % | 6.2 % | 2.8 % |
| IP-HILIC, conjugate | DA | 2.49 +- 0.25 % | 2.93 +- 0.31 % | not found | 2.4 % |
| IP-HILIC, conjugate | PO | 2.93 +- 0.16 % (1.90 skimmed, 4.19 the whole PO band) | 3.24 +- 0.32 % | 2.7 % | 2.9 % |
| IP-HILIC, conjugate | n+1 | 5.77 +- 0.13 % | 6.07 +- 0.26 % | 5.6 % | 5.8 % |
| IP-HILIC, conjugate | NC | 1.78 +- 0.04 % | 1.73 +- 0.07 % | not found | 1.8 % |
| IP-HILIC, NC | DA-NC | 5.78 +- 0.35 % | 9.84 +- 0.67 % | 7.1 % | 5.5 % |
| IP-HILIC, NC | PO-NC | 2.15 +- 0.12 % (0.93 skimmed, 3.17 the whole PO band) | 3.95 +- 0.33 % | 0.8 % | 3.5 % |
| IP-HILIC, NC | n-1 | 2.45 +- 0.18 % | 4.48 +- 0.40 % | not found | 2.35 % |
| IP-HILIC, NC | NC - dG | 0.66 +- 0.02 % | 1.15 +- 0.10 % | 1.2 % | 0.73 % |
| IP-HILIC, NC | 2887.4 | 1.03 +- 0.02 % | 1.30 +- 0.05 % | 1.0 % | 0.91 % |
| IP-HILIC, NC | 3847.4 | 0.47 +- 0.01 % | 0.65 +- 0.03 % | 0.32 % | 0.49 % |
| IP-HILIC, NC | 4512.6 | 0.26 +- 0.01 % | 0.32 +- 0.02 % | 0.45 % | 0.43 % |
| IP-HILIC, NC | 4842 (12.80 min) and 4841.8 (13.30 min) | 1.24 +- 0.09 and 1.64 +- 0.08 % | 1.88 and 2.05 % | 4841.8 only, 1.8 % | 1.37 and 1.75 % |
| IP-HILIC, NC | n-2 | 0.11 +- 0.04 % (0.108-0.121 % by its own isotopologues) | 0.21 +- 0.07 % | not found | 0.14 % |
| IP-HILIC, NC | 1883.3 | 0.33 % (its own isotopologues) | - | not found | 0.33 % |

("Before" is the code after the resampling fix, which added each species' resolved adduct forms to
its amount; "now" compares base forms.) Against the base-form reference the amounts of 19 of the 24
impurities found are within 0.88-1.19 of it; n-1 of the IP-HILIC NC run is now one report at 0.96
(0.76 while a boundary split it), n-2 under IP-RPLC at 0.98-1.03 of its own isotopologues and 1883.3
under IP-HILIC at 1.00. The exceptions: 4512.6 (1.65), PO-NC in both chromatographies (1.21 under
IP-RPLC, 1.61 under IP-HILIC), n-2 under IP-HILIC (1.27; 1.17-1.31 of its own isotopologues, against
1.10 for NC - dG and 3847.4 of that run by the same method, so within the method's spread) and
1883.3 under IP-RPLC (0.84). n-2's IP-RPLC window reference of 0.15 % also held the upper
isotopologues of a 4497.7 Da species six spacings below it on its comb, about half of the window's
signal; split from it on their own isotopologues, scan by scan, n-2 is 0.074-0.078 % of NC over its
whole elution (0.065-0.068 % within its region), and the same split gives n-1 2.08 % against the
window's 2.03 %. A band of PO-mass material (other positions of the phosphorothioate-to-phosphate
change) elutes just before each spiked PO isomer; the pipeline counts it with PO-NC, which shares
its region (3.5 % against 2.15 % for the spiked isomer alone and 3.17 % for the whole band), and not
with the conjugate's PO, whose band lies in the neighbouring region. An earlier version of this
table used a crude reference (charge states 3-8, +-1.2/z Th, within 0.4 min of the apex): such a
window holds two or three isotopologues depending on where the average mass falls between them, and
caught 43 % of n+1's envelope against 31 % of FLP's, which made n+1 look 1.4 times stronger than it
is (and the 2887.4 Da impurity 1.9 times).

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
isotopologue; the reported averages, 4832.97 Da (IP-RPLC) and 4832.92 Da (IP-HILIC), match the
composition. The other shortmers found under IP-RPLC (NC minus dG, T or MeC phosphorothioate,
4816.93, 4842.00 and 4842.91 Da) are within 0.07 Da of theirs (4816.98, 4841.99 and 4842.98 Da).
Earlier versions of this section scored the runs against a reference built from resolved isotope
spacing; it sat 0.33-0.35 Da below the theoretical masses and looked only inside the strongest
TIC bands, so it missed every separated impurity and made correct masses look 0.1-0.5 Da high.

What is still wrong on these runs:

- More reports than the extracted-ion check confirms: 58 unconfirmed (43 after the resampling fix,
  51 before section 17), 45 of them below 0.1 % of the main species. Section 17 added thirteen and
  removed six. Five of the new ones, all above 1 kDa, come with z = 1 in the charge range
  (1009.61 Da, a trimer; 1235.88 Da, the deaminated 4-mer; 1244.84 Da, the trimer plus a dA
  phosphorothioate, monoisotopic 915.17 and 1244.21 Da; 1141.66 and 1283.18 Da, unidentified); three
  are reports whose label the check's ±1.2 Da windows flipped when their mass moved to their
  templates' position (1875.62, 5324.34 and 2228.91 Da); and five are new small reports: 3182.40 Da
  (the PO form of a 3198.6 Da shortmer, one charge state of its own), 4623.81 and 2543.15 Da (their
  own isotopologues co-elute at r 0.92-0.99; in the check's ±1.2 Da windows a foreign ion or a
  correlation of 0.60 fails them), 2653.14 Da of the IP-HILIC conjugate run (the shortmer the
  IP-RPLC conjugate run reports at 2653.32 Da, EIC-confirmed) and 1481.14 Da (a cluster ion, below).
  The six removed are three pieces joined to their bands, two 0.000-0.004 % slivers replaced by
  confirmed reports of their species at the species' own time (2601.07 and 2983.00 Da) and a 0.000 %
  sliver no longer reported (2533.55 Da). The largest unconfirmed reports are forms of FLP
  177-277 Da above it in the IP-HILIC conjugate run (0.6-2.4 %), clusters of sodium, potassium,
  acetate and the ion-pairing amine beyond the adduct library's three, which carry one dominant
  charge state, and a 5364.55 Da report of the IP-RPLC NC run (0.61 %; 0.58 % before the 800 Da
  range moved the grid's bin phase). A region boundary drawn at a valley keeps species that peak on
  it (the edge rule is not applied there). A piece of a band's own front or tail that its region
  read up to 1.5 Da off is now joined to the band where both read the same isotope tooth in the
  scans (n-1 in both NC runs, a 5492.35 Da tail under IP-RPLC); the 0.02-0.05 % reports peaking on
  the IP-RPLC conjugate run's boundary at 18.89 min (6777.31, 6792.99 and 6807.95 Da) are, by their
  isotope teeth, other molecules rather than such pieces - species lighter than PO and FLP by more
  than a spacing, and one peaking on the boundary - and stay, as does the deaminated form, one tooth
  from FLP.
- Masses measured on a region's summed spectrum weight its points by count. The summed axis holds
  every point any scan measured, so where more scans cover one side of an envelope than the other
  (the main piece of n-1 of the IP-HILIC NC run: 25 scans on its heavy side, 9 on its light side)
  the mean shift is pulled to the denser side: 0.88 Da high, against 0.17 Da weighting each point by
  its share of the axis and 0.0-0.1 Da reading each scan. n-1 is now reported at its boundary
  piece's comb position (-0.06 Da), and a species whose templates its own comb placed is reported
  where they sit when that lies within half a spacing of the measured mass; every other species
  keeps the bias, which is not small: read on the region's scans' own points, pooled, 61 other
  reports move by 0.1 Da or more (32 towards their per-scan reading, 14 away, 2 level, 13 without
  one). That reading was tried and is not used: the adduct folds and the region merges read the
  reported masses, and with it six folds flipped at their 0.5-0.6 Da edges (5301.01, PO-NC + Na,
  4885.13, 7012.43, 6555.47 and 3368.69 Da), PO-NC + 2Na was listed as a species (5189.33 Da,
  0.82 %), and n-1 under IP-HILIC went from -0.06 to +0.18 Da: its main piece, read within 0.5 Da of
  the boundary piece, was joined to it by the merge that keeps the stronger piece's mass before the
  tooth-reading merge saw them. It waits for the adduct module, with the folds and merges kept on
  one mass.
- The deaminated forms sit 0.98 Da above their parents, so their envelopes coincide but for one
  isotopologue. DA in the IP-HILIC conjugate run is found in the region before FLP's; its column
  there also takes FLP's rising front, and its apex is read at 16.13 min against 16.09 by the
  envelope's first moment. Under IP-RPLC the deaminated forms do not co-elute exactly: by the
  envelope's first moment DA elutes about 0.5 min after FLP (1.9 % of FLP at 19.71 min) inside
  FLP's region, where the summed spectrum gives one candidate (FLP read 0.11 Da heavy), and DA-NC
  about 0.6 min after NC (3.2 % of NC at 18.90 min), where it is the 5163.17 Da report at 18.90 min
  (6.0 %, its scans reading 5162.98-5163.02 Da at its apex), a blend with NC's tail; the truth
  table's DA-NC at 18.21 min, unscored, should be revisited. Separating them needs an
  exact-composition pattern fit: the class pattern's error on FLP's envelope (-9 to +6 % per
  isotopologue) is larger than what 2 % DA changes (-2 to +3 %).
- n-2 is reported in both chromatographies at its comb position, 4503.65 Da (-0.06 Da). Under
  IP-RPLC its 0.08 % matches its own isotopologues' 0.074-0.078 % (the amounts table). About 10-11 %
  of its IP-RPLC elution lies in the next region, where its candidate's envelope search climbs 5.2
  Da onto the 4497.7 Da species and merges into it; recovering it needs the neighbouring region's
  bank at the shared boundary and would add about 0.008 %. Under IP-HILIC it is 1.27 times its
  reference (1.17-1.31 times its own isotopologues, where NC - dG and 3847.4 of that run are 1.10).
- An alignment optimum inside the window on the wrong tooth is still kept. In the IP-RPLC NC run
  2558.3 is read 0.78 Da light (an envelope 2 Da lighter on the same comb elutes with it); its
  optimum lies on the tooth one spacing light, which fits only 3-4 % worse than the right one, and
  it is reported at 2558.52 Da (-0.64 Da, the mean shift pulled the same way) and 0.29 % against
  0.36 %. 2228.3 of that run is the same case (2229.15 Da against the 2229.90 Da its isotopologues
  read on both charge states, 0.32 % against 0.44 %). Optima between the teeth are now replaced by
  the better tooth (`architecture.md`, section 4): 1908.3 and 2887.4 are found (1909.10 Da at 0.50 %
  against 0.61 %, 2887.83 Da at 0.18 % against 0.18 %), and the 2980.4 Da species is found again
  (2980.42 Da, 0.007 % against 0.008 %), but 1908.3, 2887.4 and 3206.4 are reported half a spacing
  light (above). The 2602.07 Da species of the IP-RPLC conjugate run is reported at its time
  (13.02 min, 0.018 % against 0.02 %, where it was 0.0003 % at 11.94 min) but at 2601.07 Da, one
  tooth light, 1.0 Da from its reference: the between-teeth rule took the tooth the apex-scan fits
  reject, as it does for 2665.77 and 2983.00 Da (their measured masses stand, their templates lying
  more than half a spacing away), and their amounts are still 0.82-1.00 times their references. The
  same rule lowers 2549.17 Da of the IP-RPLC NC run (EIC-confirmed) from 0.025 to 0.023 % against a
  0.029 % reference. 1883.3 under IP-HILIC is found (1884.51 Da, 0.33 % against 0.33 % by its own
  isotopologues): its z = 3 window, which also holds the 1234.2 Da 4-mer's +Na ion, is judged again
  on its own isotopologues.
- The comb's tooth is chosen by a weighted fit of both teeth's templates with same-comb templates
  two to seven spacings beyond the pair (`refine._better_tooth`). On the 50 placements of section
  16's code that composition or the raw scans decide, replayed on the four runs, it takes the
  supported tooth in all but NC + 6Na + K of the IP-HILIC NC run, which the cation-ladder rule holds
  at the window's edge either way; of the between-teeth placements added since, it takes the tooth
  the apex-scan fits reject for 2601.07, 2665.77 and 2983.00 Da (above). It is weak where strong
  same-comb neighbours lie on both sides of a candidate. 4949.38 Da of the IP-RPLC NC run is half of
  a pair of species one spacing apart in about equal amounts (4948.94 and 4949.94 Da), which no
  single tooth explains: with comb-placed masses its piece and the 4948.78 Da piece now merge into
  one report, 4948.97 Da at 17.78 min (0.094 %; before, 0.067 and 0.026 %), and a 0.026 %
  EIC-confirmed species is lost; it needs a two-component split. FLP + K of the IP-RPLC conjugate
  run now sits on its own tooth but is still placed on its comb and fitted whole (the adduct-form
  exemption knows only the adducts a region evidences, sodium and DIPEA there); folded into FLP it
  carries about 3 % of FLP's signal into FLP's all-forms total. That, not its tooth, moves a
  6660.16 Da report's apex from 19.17 to 19.32 min (19.21 by extracted ions) and lowers the
  conjugate's all-forms shares by 2-4 % of their values: held at the window's edge, FLP + K gives
  6660.16 Da at 19.17 min and PO 5.01, n+1 2.79, n-1 2.13 % all forms; on either tooth 19.32 min and
  4.82-4.86, 2.73 and 2.08 %. It is adduct bookkeeping, for the adduct module.
- A comb-placed species is reported at its templates' position only within half a spacing
  (0.5017 Da) of its measured mass, and several decisions lie within hundredths of that: a 4487 Da
  component moves at 0.498 Da, while 19 comb-placed components lie 0.52-0.75 Da from their templates
  and keep their measured mass - 1909.10, 2887.83 and 3207.18 Da (above) among them, PO-NC's +Na and
  +4Na forms (5167.57 and 5232.61 Da, whose folds flip at their templates' position, as under the
  pooled reading above) and 2983.00 and 2665.77 Da, whose templates sit on the tooth the apex-scan
  fits reject. Relaxing the limit was considered and not tried. Two of the moves to the templates go
  against or past a per-scan reading: 7052.52 to 7052.85 Da (7052.50 Da per scan) and 5423.96 to
  5424.37 Da, whose two per-scan readings disagree.
- The reported mass still jumps near the alignment window's edge. An alignment that stops at the
  edge within 0.15 Da of a tooth of the comb inside the window now takes that tooth: 3207.4 of the
  IP-RPLC NC run, its tooth 0.005 Da inside the 0.289 Da window once the 0.1 % region floor had
  moved its gate mass by 0.025 Da, is reported at 3208.69 Da (+0.01 Da) instead of 3208.38 Da. But
  an optimum inside the window on the comb keeps the mean-shift mass, and which the overlap search
  returns turns on its last step: 3207.4's overlap peaks 0.001 Da past the midpoint of that step,
  while 3315.35 Da of the same run, 0.28 Da light of its own tooth at 3315.63 Da, peaks 0.0007 Da
  short of it and keeps its mass, as does 3288.41 Da of the IP-RPLC conjugate run (0.29 Da light of
  its tooth at 3288.70 Da, 0.0077 Da short). Two unconfirmed reports moved with the rule, onto the
  teeth their summed scans read: 2289.54 to 2289.83 Da under IP-RPLC NC, and 7276.54 to 7276.32 Da
  under IP-HILIC conjugate, a weak reading (three of five single scans read 7277.32 Da). The rule
  also places adduct-walk trials (4841.17 Da, the trial for 4885.13 Da, a walk not taken either
  way). As first written it handed the tooth to the joint fit, which can take the one a spacing
  beyond the window's other edge, where the overlap search never reached; traced over the four runs
  and all 52 synthetic runs, the rule fires ten times (four on the four runs, one under the comb
  estimator, five under the legacy generator), and the joint fit would have taken that far tooth in
  four: the trial for 4885.13 Da, an 11.2 kDa synthetic walk trial, a 13.2 kDa synthetic impurity
  (its template 0.003 Da from its truth on the edge's tooth, a spacing heavy on the other) and the
  9197 Da polymer candidate of the legacy table above, whose far tooth took a false species away and
  moved a matched one 0.15 Da further off. The edge's own tooth is taken instead (the alternative
  the rule's investigation had left untested), with the edge's distance from the comb wrapped to the
  nearest tooth as inside the window: above about 17 kDa the window can hold the teeth either side
  of the measured mass, and measured against the nearest alone the own tooth of a constructed 20 kDa
  species, on the other side, was missed (no firing on the four runs or the benchmarks depends on
  it). The four runs are unchanged by both.
- Adduct forms beyond the library (more than two of a type, three in all) are fitted with no
  species, and are reported apart at what their templates explain: held at the window's edge by the
  adduct-form exemption or the cation-ladder rule (the 5270.85, 5250.45 and 5331.67 Da reports
  below), at an optimum inside the window (4842 + 3Na), or on their own comb where neither applies
  (PO-NC + 4Na). Under IP-HILIC NC's +5Na and +3Na+K forms, which co-elute with it and follow its
  sodium load, are a 5270.85 Da report at 0.21 %, unconfirmed (about 4 % of NC by a fit of NC's
  sodium ladder and by extracted ions): its mass, a blend of the two, lies 1.4 Da below the +5Na
  form and 4.5 Da above the +3Na+K form, beyond the cation-ladder rule's 1 Da, and the adduct-form
  exemption holds it at the edge only through the mass NC + 3Na was walked from, with 0.22 Da to
  spare (placed on its comb it was fitted whole, at 3.3 %). DA-NC + 4Na (5250.45 Da, 0.79 %;
  0.8-1.0 % by two fits of the sodium ladder, with the NC + 4Na under it) and NC + 6Na + K (5331.67
  Da, 0.14 %, unconfirmed) are held there by the cation-ladder rule. 4842's +2Na and +3Na forms are
  two reports (0.21 and 0.23 %, against one of 0.28 % before section 16): without its rejected z =
  5 the +2Na form's template explains less than the +3Na form's, and the fold takes no stronger
  form into a weaker one. Sodium forms of the deaminated form of a 5279.76 Da species seen only
  under IP-HILIC (NC + 117.5 Da), which elutes with DA-NC, are reported apart too (5324.34 Da at
  12.83 min, 0.17 %, unconfirmed since its mass moved 0.13 Da to its templates' position). PO-NC's
  sodium ladder is read 0.6-1.5 Da light, pulled by NC's sodium forms 6 Da below it: its +Na form,
  5167.57 Da at 1.15 % (1.07 % by a fit of the ladder), misses the fold by 0.02 Da, and its +4Na
  form, 5232.61 Da at 0.45 %, unconfirmed, has no folded parent within reach.
- The 915 Da species is reported (915.77 Da at 0.41 % of NC under IP-RPLC, 0.48 % by extracted ions;
  915.80 Da at 1.15 % under IP-HILIC, 1.59 %): the Orbitrap oligonucleotide method reaches down to
  800 Da and counts z = 1 beside z = 2, in the gate only where its envelope has no peaks between its
  isotopologues (`architecture.md`, section 3). The shortmer ladder down to 0.9 kDa comes with it -
  1563.2 and 1234.2 under IP-RPLC among it - and so do a 4-mer's silver adduct reported as a species
  of its own (1342.14 Da, 0.028 %), real ions misquantified at 0.000 % (897.34 Da, 0.078 % by
  extracted ions) and unidentified reports (1141.66 and 1283.18 Da); 1563.2's +Na form read at
  double charge stays a 3171.81 Da report under IP-HILIC (0.19 %). The z = 1 rule is untested with a
  trained charge network, and on synthetic short oligonucleotides the new z = 1 candidates change
  what the adduct dedupe sees (in 16 runs of 1.5-4.5 kDa phosphorothioate RNA a 0.17 % impurity was
  lost in one and a missed 0.13 % one reported 0.9 Da light in another). On the synthetic Orbitrap
  preset z = 1 also costs a main species in one of 16 seeds (above).
- A 1481.14 Da report of the IP-HILIC NC run (2.63 min, 0.007 %, unconfirmed) is a cluster ion,
  not an impurity: the sodium form of a calcium-bridged dimer of a 710.09 Da compound whose singly
  charged ions and sodium and potassium forms (m/z 709.09, 731.07, 747.04) elute at 2.5-2.8 min,
  1.7 min before the first oligonucleotide ions. Its ions [2M + Ca + nNa - (4 + n)H]2- (m/z
  728.06, 739.05, 750.04 and 761.03 for n = 0-3) and [2M + Ca + Na - 4H]- (m/z 1479.11) fit that
  composition within 1.3 ppm. With z = 1 counted beside z = 2, its one singly charged isotopologue
  (one scan) gives it a second charge state, and the between-teeth rule puts its templates on its
  own isotopologues; it is reported at the templates' position, 0.43 Da above its measured
  1480.72 Da, because its isotope pattern holds less sulfur than the phosphorothioate template's,
  and the extracted-ion check sees only z = 2 (its z = 1 window peaks on a later ion at 2.86-2.94
  min, and z = 3 holds only background). Telling such clusters from species is left to the adduct
  module: requiring z = 1 to carry signal in two scans removes it, but on the four runs it also
  removes the deaminated 4-mer at 4.43 min (1235.88 Da, 0.070 %), and nothing else.
- 6752.33 Da of the IP-RPLC conjugate run (0.08 %) is the leading edge of a species of about 0.7 %
  whose apex (19.02 min) lies in the next region, where it is not reported; its templates sit at
  the window's near edge, held by the adduct-form exemption as a form of a candidate the dedupe
  renamed. In that region it shares one candidate with a 2.3 times stronger species two spacings
  lighter on its comb (6750.03 Da, 1.66 %, apex 19.32 min), and the adduct dedupe renames FLP's
  candidate onto that blend (below); even without the rename the two need a two-species split.
- The comb's second state counts when its phase agrees within 0.1 Da, which a window of noise does
  about one time in five (17.5-20.5 % of simulated noise windows, 20.3 % of real residual windows
  moved 5-60 m/z): a comb read off one window can still pass. An amplitude test (a state counts
  only where a window without its comb would agree that closely at that amplitude by chance at most
  0.5 %) brings that to 1.0-2.5 % (8.8 % on the moved real windows) and changes no report on the
  four runs or the synthetic benchmarks, but below about 3 kDa, where a short envelope leaks into
  the off-comb frequencies it is compared with, it loses weak states of the species' own (1883.3's
  z = 3 among them); it is not used.
- The adduct dedupe renames a stronger candidate to a lighter partner one adduct ladder below it
  whatever the partner's strength, except the first pass's two strongest candidates, whose adduct
  forms decide the region's adduct types: they keep their mass against a partner with less than
  half their score (moved onto such a partner, the strongest candidate's adduct trials read the
  partner's forms and the region fitted no adduct columns). FLP's candidate under IP-RPLC is held
  that way (it used to become a 6750.58 Da envelope with 9 % of its score, 43.50 Da below it, two
  sodium to 0.46 Da), and the 6750 Da species the rename hid is reported. Weaker candidates still
  move: requiring half the score of every partner loses a 2.9 % EIC-confirmed species of the
  IP-RPLC NC run on a knife-edge of the shared-evidence test (a PS-to-PO candidate, held at its
  mass, is fitted before it); whether a weak partner is a base form or a species of its own waits
  for the adduct module. A held candidate's weak partner, kept as a candidate, is fitted with the
  region's adduct columns, and those of its adduct forms that coincide with the held species' own
  ions can carry most of its fit, so that the shared-evidence test drops it (a 3 % impurity
  0.4 Da off one C9H21NO below a synthetic oligonucleotide, which the rename reported).
- A +201.4 Da form co-elutes with every major species, strongest under IP-RPLC (8 % of NC by
  extracted-ion chromatogram); the paper does not mention it. It is reported as separate species
  (its column-wash elution in the IP-RPLC NC run, 5363.31 Da at 25.42 min, is now named in the
  flags of its own report, 5363.53 Da at 18.28 min, as NC's are in NC's). Its mass matches
  tributylamine N-oxide (C12H27NO, 201.35 Da), an oxidation product of the IP-RPLC ion-pairing
  amine, but that is a guess.
- NC elutes on after its band in the IP-RPLC NC run: a tail at 0.07-0.3 % of its height from 19 to
  25 min and a column-wash peak at 25.4 min (1.3 % of its height). Its pieces that regions report
  at its mass (5162.23 Da at 20.01 min, 1.24 % of NC, and 5162.13 Da at 25.42 min, 1.32 %) are now
  named in NC's flags instead of being reported (`features.link.set_aside_later_elutions`, which
  the oligonucleotide methods turn on: `linking.set_aside_later_elutions`), and NC's amount stays
  its own band's. The flags name only the pieces that were reported, about 2.6 of the 5.6 % of NC
  that elutes after its band, not the total, and an isomer of the main species eluting later at
  under a twentieth of it would be named there too.
- Average masses are reported. For isotope-resolved data a monoisotopic pattern fit would also
  separate a deaminated form from its parent inside one envelope.
- The multimer rule (`refine.drop_multimers`) drops a weaker component at n times a stronger one's
  mass that co-elutes with it, or that is a harmonic reading of it; a half-mass misreading that won
  the fit would keep the misreading and drop the genuine heavier species. Where co-elution cannot
  be judged (no frames, fewer than four scans, or no profile in the monomer's trace) a component
  that is not a harmonic is kept, so a genuine dimer co-eluting with its monomer in a region of two
  or three scans is reported. Neither case occurs on the four runs or in the benchmarks.
- The harmonic-reading rule (`refine.drop_harmonic_readings`) judges a column without a readable
  isotope ladder, where the instrument would not resolve its isotopes, by its position: within
  three envelope sigmas of any ion a stronger component fitted, of any size, it is not the judged
  component's own. That suits time-of-flight data, where the envelopes of large species are humps
  and the readings' columns lie on the stronger species' ions and adduct forms (judged against the
  ions of stronger components it pairs with by mass alone, the rule keeps a good part of the
  readings there), but a genuine species whose envelopes all fall that close to stronger ones goes
  with them. Where the isotopes would be resolved and no ladder is readable it does not judge: so
  judged, 3749.72 Da of the IP-RPLC conjugate run (0.004 %, EIC-confirmed), 1.25 times a stronger
  2998.30 Da species, went, its main column 3 Da from 3714.98 Da's potassium form and interleaved
  with another envelope 1.4 Da away, and with `singly_charged_species` on, so did a singly charged
  ion beside a stronger species' weak adduct column three charges up. A reading stronger than every
  species it reads is never judged. And the stronger species' adduct forms that readings used to
  explain away through the shared-evidence test stay once the readings are gone, as species of their
  own: a main species' tripropylamine N-oxide (TPAO, C9H21NO) form, fitted again as a species in a
  later pass after an adduct walk. Such a form is folded into the main species where a region of the
  run evidenced TPAO and the two traces correlate (`fold_adduct_forms`); otherwise
  `mark_adduct_forms` marks it as the main species' adduct form where it lies within 0.15 Da (or 15
  ppm, whichever is larger) of one to four TPAO above it and co-elutes with it; it is reported apart
  only where neither applies. The fold declines a form two TPAO above a species (318.54 Da) where an
  n+1 (2'-OMe C, 319.21 Da) lies within its tolerance (0.6 Da or 2.5 peak widths, whichever is
  larger: from 4 kDa on at the RDa's resolving power), and the marking, within 0.15 Da, then takes
  it.
- On Confidential sample 1 (eleven IP-RP runs of a fully phosphorothioated DNA, base composition
  T13 C4 G6, from a Waters RDa time-of-flight instrument; `training.md`, recipe v5), with the
  bundled `rda_iprp_neg_oligo` and the lock mass applied as one ratio, the comb leaves a median of
  about 12 % (2-19 %) of a run's deconvolved signal in species at 1.25 times or more, or 0.8 times
  or less, the mass of a stronger co-eluting species that is not itself one (judged strongest
  first), the kind the harmonic-reading rule above judges; the GPU-trained v5 charge networks leave
  a median of under 1 % (at most 3-4 % in a run; `training.md`, recipe v5). With the lock mass
  applied scan by scan the comb left 16 % (`architecture.md`, section 5).
- Knife edges of the region builder: the merge of two IP-RPLC conjugate peaks into one region hangs
  on a float comparison of their apex gap with their width (0.6693310 against 0.6693330 min; a pair
  at 17.65-18.21 min of that run splits by 2e-6 min), and through that region a weak, separate
  2653 Da elution at 10.49 min (0.0047 % by extracted ions) joins 2653.32 Da across the new boundary
  (0.283 to 0.296 %).
- Section 17 changed no adduct rule but the adduct-evidence test, which only gained arguments that
  keep it reading charge windows and judge z = 1 as detection does; its fixes moved adduct outcomes
  all the same, which the adduct module will decide: the co-elution rescue lets NC + K of the
  IP-RPLC conjugate run pass the gate and fold into NC (all forms 1.65 to 1.69 %); NC + DIPEA passes
  too (r 0.903 on its isotopologues), is removed by the adduct dedupe in the first pass, found again
  in the second and dropped by the shared-evidence test, and no threshold on r separates NC + K
  (0.98) from 1883.3 (0.97). The 800 Da range and z = 1 evidence sodium and potassium in three
  low-mass regions (IP-RPLC NC 2.99-3.67 and 3.67-4.82 min, IP-HILIC NC 5.37-5.96 min): the IP-HILIC
  4-mer gets adduct columns (base 4.17 to 4.47 %, all forms 2.59 to 4.14 %; sodium to base 0.40
  against 0.38 by extracted ions), 1564.25 Da folds a +2Na form, 1580.18 Da gets a +K column (all
  forms 0.92 to 1.00 %), 915.77 Da folds its +K form, the 4-mer's [2M+Na] dimer (2491.80 Da,
  0.063 %) walks to 2M and is dropped as a multimer, and the adduct dedupe removes 15 new low-mass
  adduct candidates. The 800 Da floor also moves the grid's bin phase, and with it seven amounts by
  a few percent of their value, among them two sodium forms of 4842 (4907.02 Da 0.254 to 0.234 %,
  4885.13 Da 0.197 to 0.207 %). The between-teeth rule ends the walk from 3058.12 to 2998.05 Da
  (3058.12 Da is dropped by the shared-evidence test, and the +Na+K form of 2998.45 Da lands in a
  new +2K column of 2983.00 Da), lowers the adduct part of 2869.59 Da from 72 to 22 % of its forms
  (all forms 0.064 to 0.023 %), gives the new 2524.45 and 2195.87 Da reports sodium columns of
  50-54 % of their forms, drops a 2533.6 Da sliver through its +Na column, and fits two synthetic
  adduct forms whole. The joint-fit tooth puts FLP + K's template on its own tooth (6831.03 to
  6832.03 Da; PO all forms 4.82 to 4.86 %). Reporting comb-placed masses leaves every fold as it was
  but moves each form's offset from its parent plus the adduct, which the fold takes within 0.6 Da:
  FLP + K from 0.57 to 0.15 Da, 4842 + K 0.59 to 0.22, 3527.5 + Na 0.53 to 0.08, 4842.91 + Na 0.36
  to 0.02, and 2558.52 + K from 0.05 to 0.52 Da, 0.08 Da inside the tolerance. Merged region pieces
  bring their adduct traces (2653.32 Da all forms 0.258 to 0.292 %).
- In-scope rules lean on adduct rules the module will redesign: only the cation-ladder rule holds
  2991.98 Da (2887.4 + 3Na + K) at its optimum (without it the form is placed on its comb and
  fitted whole), and it masks the joint-fit tooth's one miss (NC + 6Na + K); FLP + K is
  comb-placed only because its region does not evidence potassium; the edge rule above places
  adduct-walk trials; and the adduct evidence reads charge windows only (`own_comb=False`):
  judged on their own isotopologues, DIPEA would be evidenced in NC's region of the IP-RPLC
  conjugate run (NC all forms 1.69 to 1.72 %, eight species with +DIPEA columns, n+1's +DIPEA form
  deduped instead of folded).
