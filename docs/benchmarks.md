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
| comb estimator | protein_tof | 34 % (10/29) | 100 % (8/8) | 25 % (1/4) | 0 % (0/2) | 6.7 |
| comb estimator | oligo_iprp | 28 % (11/40) | 88 % (7/8) | 43 % (3/7) | 17 % (1/6) | 2.1 |
| comb estimator | polymer | 40 % (12/30) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| oracle (perfect) | protein_tof | 75 % (12/16) | 100 % (8/8) | 50 % (2/4) | 0 % (0/2) | 7.2 |
| oracle (perfect) | oligo_iprp | 71 % (12/17) | 88 % (7/8) | 43 % (3/7) | 33 % (2/6) | 1.8 |
| oracle (perfect) | polymer | 60 % (12/20) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| oracle, 20 % of envelopes mislabelled | protein_tof | 62 % (10/16) | 88 % (7/8) | 50 % (2/4) | 0 % (0/2) | 9.6 |
| oracle, 20 % of envelopes mislabelled | oligo_iprp | 67 % (12/18) | 88 % (7/8) | 43 % (3/7) | 33 % (2/6) | 1.8 |
| oracle, 20 % of envelopes mislabelled | polymer | 60 % (12/20) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |

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

### Orbitrap regime

`--presets oligo_orbitrap` (not in the default list): phosphorothioate RNA of 3-9 kDa at 30 000
to 63 000 resolving power at m/z 200 (the real runs below were recorded at 140 000), decoded
with the `orbitrap_iprp_neg_oligo` method, so
charges are also read from resolved isotope spacing.

| Code | Precision | Main (>=10 %) | 1-10 % | 0.1-1 % | 0.01-0.1 % | Median ppm |
| --- | --- | --- | --- | --- | --- | --- |
| before the truth table | 13 % (12/94) | 88 % (7/8) | 57 % (4/7) | 17 % (1/6) | 0 % (0/2) | 4.1 |
| now | 14 % (14/99) | 88 % (7/8) | 71 % (5/7) | 17 % (1/6) | 50 % (1/2) | 8.0 |

Precision in this regime is far below the time-of-flight presets: of 85 false species, 16 are
adduct-shifted copies and 67 unexplained. `single_charge: true` changes nothing here: the
generator gives every species several charge states, unlike ion-pairing HILIC on the real runs.

### Charge network (LadderNet)

LadderNet (`docs/training.md`), `proof` size (9 500 parameters), trained on two CPU threads for
45 minutes: 240 synthetic frames over 12 epochs, with singly charged ions either ignored by the
loss (`--singly-charged ignore`) or absent from the spectra (`drop`); "stage 2" continues the
first for six epochs with singly charged ions trained on and up to 200 background ions per
spectrum.

Charge read correctly on the same 96 fresh spectra of the training mixture (83 time-of-flight,
13 Orbitrap; singly charged ions included) for every estimator, each run as the pipeline runs
it - over the whole spectrum, a network in its 32 768-bin tiles - and scored on the spectrum's
labelled crop (`python scripts/eval_charge_bins.py --frames 96 <models>`). "Comb" is the
pipeline's estimator, with isotope-spacing charges on Orbitrap spectra. Shares of the ion bins
whose charge is read right, every spectrum counting the same: at z >= 2 with bins weighted by
log-SNR (the training metric), by intensity, and by intensity pooled over all spectra (so that
the strongest few decide); by charge band, intensity-weighted:

| Spectra | Charge assignment | z >= 2, log-SNR | z >= 2, intensity | z >= 2, pooled | z = 1 | z = 2-5 | z = 6-15 | z = 16-40 | z > 40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time-of-flight | comb | 0.64 | 0.86 | 0.89 | 0.22 | 0.74 | 0.87 | 0.91 | 0.80 |
| time-of-flight | LadderNet, z = 1 ignored | 0.90 | 0.95 | 0.82 | 0.31 | 0.81 | 0.97 | 0.98 | 0.99 |
| time-of-flight | LadderNet, z = 1 dropped | 0.90 | 0.96 | 0.86 | 0.00 | 0.83 | 0.97 | 0.98 | 0.99 |
| time-of-flight | LadderNet, stage 2, dense background | 0.81 | 0.92 | 0.85 | 0.92 | 0.66 | 0.94 | 0.98 | 0.99 |
| Orbitrap | comb | 0.23 | 0.36 | 0.43 | 0.88 | 0.76 | 0.10 | 0.98 | - |
| Orbitrap | comb without isotope charges | 0.60 | 0.84 | 0.88 | 0.12 | 0.81 | 0.86 | 0.96 | - |
| Orbitrap | LadderNet, z = 1 ignored | 0.84 | 0.92 | 0.88 | 0.29 | 0.86 | 0.96 | 1.00 | - |
| Orbitrap | LadderNet, z = 1 dropped | 0.88 | 0.95 | 0.93 | 0.00 | 0.93 | 0.96 | 1.00 | - |
| Orbitrap | LadderNet, stage 2, dense background | 0.73 | 0.86 | 0.84 | 0.92 | 0.78 | 0.91 | 1.00 | - |

The networks read weak ions far better than the comb (log-SNR weighting) and most of each
spectrum's intensity better too, but pooled over spectra the comb leads on time-of-flight data,
at z = 2-5 above all (0.80 against 0.53-0.65 pooled): a network reads 32 768-bin windows, which
cannot hold the nearest neighbours of a doubly charged ion (34 657 bins away at z = 1, 20 273 at
z = 3), while the comb reads the whole spectrum. On Orbitrap spectra the isotope-spacing charges
make the comb worse, not better (0.23 against 0.60 without them). An
earlier version of this table had run the comb on the crop only, which hid the far neighbours
from it too (0.46 at z >= 2 on time-of-flight spectra), and training had printed the comb's
answer in feature channel 3, computed inside the crop, as the bar to beat; an independent review
caught both. The same benchmark as above, with the network in place of the comb
(`--predictor model.onnx`):

| Charge assignment | Preset | Precision | Main (>=10 %) | 1-10 % | 0.1-1 % | Median ppm |
| --- | --- | --- | --- | --- | --- | --- |
| comb estimator | protein_tof | 34 % (10/29) | 100 % (8/8) | 25 % (1/4) | 0 % (0/2) | 6.7 |
| LadderNet, z = 1 ignored | protein_tof | 52 % (11/21) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 6.7 |
| LadderNet, z = 1 dropped | protein_tof | 52 % (11/21) | 100 % (8/8) | 50 % (2/4) | 0 % (0/2) | 9.0 |
| LadderNet, stage 2, dense background | protein_tof | 65 % (11/17) | 100 % (8/8) | 75 % (3/4) | 0 % (0/2) | 6.7 |
| comb estimator | oligo_iprp | 28 % (11/40) | 88 % (7/8) | 43 % (3/7) | 17 % (1/6) | 2.1 |
| LadderNet, z = 1 ignored | oligo_iprp | 39 % (13/33) | 88 % (7/8) | 57 % (4/7) | 33 % (2/6) | 3.7 |
| LadderNet, z = 1 dropped | oligo_iprp | 32 % (12/37) | 88 % (7/8) | 43 % (3/7) | 33 % (2/6) | 2.3 |
| LadderNet, stage 2, dense background | oligo_iprp | 39 % (11/28) | 88 % (7/8) | 43 % (3/7) | 17 % (1/6) | 2.4 |
| comb estimator | polymer | 40 % (12/30) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| LadderNet, z = 1 ignored | polymer | 32 % (13/41) | 100 % (8/8) | 80 % (4/5) | 33 % (1/3) | 1.8 |
| LadderNet, z = 1 dropped | polymer | 55 % (12/22) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
| LadderNet, stage 2, dense background | polymer | 20 % (12/59) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 1.1 |
| comb estimator | oligo_orbitrap | 14 % (14/99) | 88 % (7/8) | 71 % (5/7) | 17 % (1/6) | 8.0 |
| LadderNet, z = 1 ignored | oligo_orbitrap | 47 % (17/36) | 100 % (8/8) | 86 % (6/7) | 33 % (2/6) | 10.6 |
| LadderNet, z = 1 dropped | oligo_orbitrap | 43 % (16/37) | 100 % (8/8) | 86 % (6/7) | 33 % (2/6) | 8.2 |
| LadderNet, stage 2, dense background | oligo_orbitrap | 55 % (16/29) | 100 % (8/8) | 86 % (6/7) | 17 % (1/6) | 8.1 |

Stage 2 is the best model so far for
proteins and Orbitrap oligonucleotides and the worst for polymers, where it reads the small
oligomers' low charges as background-like (32 unexplained false species against 6 for the comb):
these CPU runs are short, and the variants trade off differently per preset.

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
Oligonucleotides", Anal. Chem. 2025, 97, 15717-15726 (https://doi.org/10.1021/acs.analchem.5c01407):
negative mode, profile spectra at 140 000 resolving power (m/z 200), m/z 400-2500. A
GalNAc-conjugated, fully phosphorothioated 16-mer (FLP, 6793.98 Da) spiked with 2 % each of its
n-1, n+1, PS-to-PO (PO), deaminated (DA) and non-conjugated (NC) forms, and the NC product
(5162.25 Da) spiked with 2 % of its deaminated form (DA-NC), each separated by ion-pairing
reversed phase (IP-RPLC, tributylammonium acetate) and ion-pairing HILIC (IP-HILIC,
triethylammonium acetate at 80 °C). Read with `lcmsdeconv.io.read_thermo_raw`, processed with
the bundled `orbitrap_iprp_neg_oligo` method and the comb estimator, and scored with
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

**Other reports** are sorted into: forms of a found species (+Na, +K, their combinations, or
the +201.4 Da form described below), charge harmonics of a found species, the same molecule at
another retention time (within 0.3 Da), and the rest, which are "EIC-confirmed" when two of
their charge states have extracted-ion chromatograms with a peak within 0.25 min of the reported
time, three times above their median within 1.5 min, that co-vary there (r >= 0.7), and
"unconfirmed" otherwise. Each chromatogram is judged against its own surroundings; an earlier
version divided it by its maximum over the whole run, so that a weak species whose m/z window
also passes the injection peak or the column wash failed (an independent review found five of
fifteen sequence-identified NC impurities rejected that way). On the four runs the check
confirms 47 of the 51 tabulated species at their tabulated times and 4 of 240 random
(mass, time) pairs. It is a consistency check, not truth: a false charge ladder made of
co-eluting background can pass it, and a species whose strongest charge states share their m/z
windows with other signal can fail it.

| Code, charge assignment | Paper's species | SI-listed | Reported | Forms of found | Same molecule, other time | EIC-confirmed | Unconfirmed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original, comb | 2 / 20 | 0 / 29 | 87 | 0 | 0 | 47 | 38 |
| before the truth table, comb | 9 / 20 | 4 / 29 | 27 | 2 | 2 | 8 | 2 |
| now, comb | 13 / 20 | 5 / 29 | 39 | 4 | 2 | 8 | 7 |
| now, comb, `single_charge: true` | 15 / 20 | 7 / 29 | 74 | 9 | 3 | 21 | 19 |
| now, LadderNet stage 1, z = 1 ignored | 12 / 20 | 4 / 29 | 64 | 5 | 2 | 13 | 28 |
| now, LadderNet stage 1, z = 1 dropped | 12 / 20 | 5 / 29 | 60 | 5 | 2 | 14 | 22 |
| now, LadderNet stage 2, dense background | 12 / 20 | 4 / 29 | 55 | 5 | 2 | 12 | 20 |

(The "now" rows use the extracted-ion check described above; the two earlier rows were scored
with its first version.)

No report of any variant is a charge harmonic (half, a third, double or triple a found species'
mass at its time), which the scoring now checks for before the extracted-ion test.

The LadderNet rows use the CPU-trained `proof` models of the synthetic section. They find the
same species as the comb but one (a 0.01 % n-2 shortmer), and estimate the PS-to-PO impurity
under NC at 10 % of NC where the comb reports 0.2 % (about 5 % by extracted-ion chromatogram),
and in the GalNAc mixture find NC's own 7..16 10-mer (3208.66 Da at 14.2 min, 0.13-0.17 %),
which the comb misses; but they add 13-21 more unconfirmed reports, nearly all below 0.13 % of
the strongest species, 1.3-3.9 kDa, in the IP-RPLC column wash: singly charged background read
as z = 2-3. Stage 1
never penalises that reading. Stage 2 (z = 1 trained on, and with up to 200 background ions per
spectrum, `--background-ions 200`) brings the number down from 28 to 20: the remedy is in the
right direction, and six CPU epochs are far from enough of it; `gpu_training/README.md` runs
both stages on a GPU. (These tables are after a fix to species linking that had shrunk the masses
of tracks with vanishingly weak frames toward zero; it removed two to three junk reports from
every variant.) Most of the SI-listed impurities missed by every variant are small
fragments (1.2-3.8 kDa) eluting at 3-15 min.

66 of the original code's 87 reports were heavier than 8 kDa, which nothing in these samples is
(the largest is n+1 at 7139 Da): singly charged background in the column wash read as charge
ladders. 47 of its reports pass the extracted-ion check, which shows how weak that check is for
such ladders. No report of the current code is heavier than 8 kDa.

Found species and their mass errors against the theoretical average masses, current defaults
(comb):

| Run | Species found (error, Da) | Missed |
| --- | --- | --- |
| IP-RPLC, conjugate | FLP (+0.06), n-1 (+0.03), n+1 (+0.10), NC (+0.01), PO (-0.21) | - |
| IP-RPLC, NC | NC (+0.02), PO-NC (-0.03), n-1 (-0.03), n-2 (+0.14) | - |
| IP-HILIC, NC | NC (+0.11), DA-NC (-0.01), PO-NC (+0.02) | n-1, n-2 |
| IP-HILIC, conjugate | FLP (+0.06) | n-1, DA, PO, n+1, NC |

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

- IP-HILIC conjugate run: with triethylammonium acetate the impurities carry essentially one
  charge state (z = 4) above the noise, which the two-charge-state rule rejects;
  `deconvolution.single_charge: true` recovers n-1 and PO. n+1 is still missed, NC elutes where
  the TIC shows no band so no region is formed, and DA (one dalton above FLP, eluting 0.4 min
  before it) is absorbed by the FLP template, whose trace then peaks at the region's edge.
- Abundances next to a strong species are unreliable: PO under FLP in IP-RPLC is reported at
  0.01 % against about 5 % by extracted-ion chromatogram, and in the IP-HILIC NC run the main
  species is reported at 46 % of its own 2 % deaminated impurity.
- A +201.4 Da form co-elutes with every major species, strongest under IP-RPLC (8 % of NC by
  extracted-ion chromatogram); the paper does not mention it. It is reported as separate
  species. Its mass matches tributylamine N-oxide (C12H27NO, 201.35 Da), an oxidation product of
  the IP-RPLC ion-pairing amine, but that is a guess.
- NC re-elutes late in the IP-RPLC gradient and wash (20.9 and 27.2 min) and is reported each
  time, as the same mass at another retention time.
- Average masses are reported. For isotope-resolved data a monoisotopic pattern fit would also
  separate a deaminated form from its parent inside one envelope.
