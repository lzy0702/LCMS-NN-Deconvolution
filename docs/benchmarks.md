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
| comb estimator | polymer | 39 % (12/31) | 88 % (7/8) | 80 % (4/5) | 33 % (1/3) | 0.8 |
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

`--presets oligo_orbitrap` (not in the default list): phosphorothioate RNA of 3-9 kDa at 67 000
to 140 000 resolving power (m/z 200), decoded with the `orbitrap_iprp_neg_oligo` method, so
charges are also read from resolved isotope spacing.

| Code | Precision | Main (>=10 %) | 1-10 % | 0.1-1 % | 0.01-0.1 % | Median ppm |
| --- | --- | --- | --- | --- | --- | --- |
| before the truth table | 13 % (12/94) | 88 % (7/8) | 57 % (4/7) | 17 % (1/6) | 0 % (0/2) | 4.1 |
| now | 13 % (14/104) | 88 % (7/8) | 71 % (5/7) | 17 % (1/6) | 50 % (1/2) | 8.0 |

Precision in this regime is far below the time-of-flight presets: of 90 false species, 17 are
adduct-shifted copies and 70 unexplained. `single_charge: true` changes nothing here: the
generator gives every species several charge states, unlike ion-pairing HILIC on the real runs.

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

**Truth.** The average masses of the paper's Table 1; for the NC product's own impurities, the
PS-to-PO form (16.06 Da below NC) and the masses the paper reports for its tentative n-1
(4832.56 Da) and n-2 (4503.53 Da) assignments. Retention times are where extracted-ion
chromatograms of the species peak at two or more charge states together. DA co-elutes with
FLP under IP-RPLC, and DA-NC with NC, less than 1 Da apart; those two are not scored, which
leaves 20 species. A reported species matches within 1 Da (1.5 Da for the paper's tentative
masses) and 0.35 min.

**Other reports** are sorted into: forms of a found species (+Na, +K, or the +201.4 Da form
described below), the same molecule at another retention time, and the rest, which are
"EIC-confirmed" when the extracted-ion chromatograms of their two strongest charge states peak
together at the reported time and "unconfirmed" otherwise. The confirmation is a consistency
check, not truth: a false charge ladder made of co-eluting background can pass it.

| Code | Known species found | Reported | Matching a listed species | Forms of found species | Same molecule, other time | EIC-confirmed | Unconfirmed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 2 / 20 | 87 | 2 | 0 | 0 | 47 | 38 |
| before the truth table | 9 / 20 | 27 | 15 | 2 | 3 | 6 | 1 |
| now | 13 / 20 | 41 | 19 | 5 | 3 | 7 | 7 |
| now, `single_charge: true` | 15 / 20 | 74 | 21 | 9 | 5 | 16 | 23 |

"Listed species" are the 20 known ones plus ten located by extracted-ion chromatograms in the
NC runs (shortmers the paper shows but does not assign, such as NC minus dG, T or MeC).

66 of the original code's 87 reports were heavier than 8 kDa, which nothing in these samples is
(the largest is n+1 at 7139 Da): singly charged background in the column wash read as charge
ladders. 47 of its reports pass the extracted-ion check, which shows how weak that check is for
such ladders. No report of the current code is heavier than 8 kDa.

Found species and their mass errors against the theoretical average masses, current defaults:

| Run | Species found (error, Da) | Missed |
| --- | --- | --- |
| IP-RPLC, conjugate | FLP (+0.06), n-1 (+0.03), n+1 (+0.10), NC (+0.01), PO (-0.21) | - |
| IP-RPLC, NC | NC (+0.02), PO-NC (-0.01), n-1 (+0.39 against the paper's 4832.56), n-2 (+0.32 against 4503.53) | - |
| IP-HILIC, NC | NC (+0.11), DA-NC (-0.01), PO-NC (+0.02) | n-1, n-2 |
| IP-HILIC, conjugate | FLP (+0.06) | n-1, DA, PO, n+1, NC |

The paper's tentative n-1 of the NC product (4832.56 Da, "loss of dG") lies 329.3 Da below
NC: one deoxyadenosine phosphorothioate (4832.98 Da by composition) or one deoxyguanosine
phosphodiester, not a dG phosphorothioate (345.27 Da). The reported 4832.95 Da matches the
composition; the other shortmers found (NC minus dG, T or MeC phosphorothioate, 4816.86,
4842.02 and 4842.97 Da) are within 0.12 Da of theirs. Earlier versions of this section scored the runs against a reference
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
