# Validation

## Synthetic benchmarks

`scripts/benchmark_synth.py` scores the whole chain against known truth and reports mass error,
recall and precision by abundance tier, and adduct-fraction error. Run it with:

```bash
python scripts/benchmark_synth.py --runs 5 --out docs/benchmarks.md
```

`lcmsdeconv evaluate <dir>` scores a single processed directory against its `truth.json`.

Component-level checks that run in the test suite:

| Check | Result |
| --- | --- |
| Isotope patterns against pyteomics | monoisotopic mass agrees to below 2 mDa from small molecules to 100-atom peptides |
| Deconvolution of a 17 kDa protein with 10 % sodium adduct | mass within 0.2 ppm, adduct fraction 12 % against a true 10 %, residual 3.5 % |
| Whole run: 12.3 kDa protein with seven impurities | main mass exact, impurity table populated, 14 s end to end |
| Integrator on analytic Gaussians | isolated peak area exact to 0.00 %, valley cluster total exact to 0.0 % |
| Non-negative least squares through the normal equations | agrees with the direct solve to 2e-10, twelve times faster |
| ONNX export | outputs match PyTorch to about 1e-6 |
| Ionisation saturation | detected on a compressed response, not raised on a linear one |
| Charge estimation without a model | the comb estimator names at least six of eleven charge states of a 20 kDa envelope |
| Impurity recovery, whole run, no trained model | main species exact; three of seven true components recovered, including a 4 % impurity 438 Da below the main |
| A saturated 144 kDa antibody in a two-peak run | found at the correct retention time, 520 ppm high, charge states agreeing to 11 ppm; 180 frames processed in 7.5 s |

## Defects this validation found

Running the pipeline end to end surfaced several defects that unit tests alone had not:

- The elution profile normalized by the maximum of the times passed to it, so evaluating it one
  frame at a time always returned 1.0 and every synthetic run had a flat chromatogram.
- Grid resampling averaged the samples falling in a bin, which makes the result independent of
  how much signal survived vendor thresholding, so deconvolved chromatograms came out flat.
- Candidate charge ranges were unbounded, so one least-squares solve could span sixty charge
  states across the whole grid and take eight seconds.
- Charge misassignment produced candidates at M/n and n*M that were fitted as separate species.
- A heavily adducted species was reported at its adducted mass rather than its base mass.

All are fixed; each has a regression test.

A second round, driven by charge oracles and per-species loss attribution
(`lcmsdeconv.diagnostics`, see `lessons.md`), found more, in the generator and in the decoder:

- The elution profile underflowed to zero when its tail was short against its width, deleting
  about 2 % of synthetic chromatographic peaks from the data while keeping them in the truth.
- A per-charge adduct propensity left the un-adducted form of large proteins at a few percent of
  their signal, so the true mass was nearly invisible.
- Impurities drew charge envelopes unrelated to their parent's; nucleic-acid charge states were
  about 40 % low against a measured ion-pairing run.
- The network was trained on spectra rendered straight onto the grid with the generator's noise
  level, but run on resampled, thresholded, summed profiles with an estimated noise level; the
  comb feature wrapped window edges into each other.
- The region finder dropped chromatographic peaks under 5 % of the tallest, deleting a second
  main component before deconvolution.
- The mass histogram turned every resolved isotopologue into a candidate.
- The consistency gate snapped impurities onto the main component's apex, and one outlying
  charge state could veto a consistent candidate.
- The adduct walk-down compared a base-only fit with a full adduct fit, so it moved main
  components onto weaker neighbours one adduct lighter.
- A species was fitted only over the charges that voted for it, leaving the rest of its envelope
  to be explained as false species.
- Inference imported PyTorch, which only training needs.

## Real Orbitrap runs

Four Thermo Q Exactive runs of oligonucleotide mixtures (a GalNAc-conjugated phosphorothioate
16-mer spiked with 2 % of each of five impurities, and its non-conjugated form spiked with its
deaminated form; ion-pairing reversed phase and ion-pairing HILIC; negative mode, profile at
140k) are read directly from `.raw` (`lcmsdeconv.io.read_thermo_raw`). They are the runs of
Tutiš et al., Anal. Chem. 2025, 97, 15717-15726 (https://doi.org/10.1021/acs.analchem.5c01407),
whose Table 1 gives the average masses; `scripts/validate_oligo_runs.py` scores a run against
them, and checks every other report with extracted-ion chromatograms. Of 20 known species the
original code found 2 among 87 reports; the current code finds 13 among 39, with 15 of 20 when
species seen at one charge state are accepted (`deconvolution.single_charge`). Details, mass
errors and what is still wrong are in `benchmarks.md`; the defects these runs exposed, and what
the truth table changed, are in `lessons.md`, section 11.

## Public data

No real measurements are used for training. Two public sources are reachable without
authentication and are used only to confirm the software behaves sensibly on real spectra:

| Source | Content | Licence |
| --- | --- | --- |
| [UniDec example data](https://github.com/michaelmarty/UniDec) (`unidec/bin/Example Data`) | native mass spectra of bovine serum albumin, alcohol dehydrogenase, GroEL and lipid nanodiscs, as two-column text | modified BSD |
| [ms_deisotope test data](https://github.com/mobiusklein/ms_deisotope) (`tests/test_data`) | Orbitrap mzML with isotopically resolved MS1 of alpha-1-acid glycoprotein glycopeptides | Apache 2.0 |

`scripts/fetch_public_data.py` downloads them into `data/public/` (git-ignored).

Larger repositories worth using on a connected machine, which the sandbox that produced this
package could not reach:

- MassIVE and PRIDE for intact-protein and top-down LC-MS runs (convert with msconvert).
- The Consortium for Top-Down Proteomics benchmark sample for cross-platform intact-mass data.
- Vendor application notes for oligonucleotide ion-pairing LC-MS, which document the adduct
  patterns the ion-pairing adduct library is built around.

## Known limitations

- No model is loaded by default; the deterministic comb estimator is used until one is trained
  and installed as `charge_unet.onnx`. The bundled CPU-trained proof model did not beat the
  estimator (two of seven components against three) and is therefore opt-in. It was trained on
  the old grid-rendered frames, so its features do not match what inference computes.
  A CPU-trained model is a proof of the pipeline, not a production model. Retrain on a GPU (see
  `training.md`) before using the software for decisions: the deconvolution mathematics is
  independent of model quality, but candidate generation is not.
- Mass accuracy degrades under heavy adduction. With a realistic adduct load the main species is
  recovered exactly; with an extreme load (several alkali adducts on most ions) the reported mass
  can sit tens of daltons high, because the mass histogram peaks on an adducted form and the
  base-mass search cannot always walk the whole ladder back. Restricting the adduct library to
  what the mobile phase can actually produce is the practical remedy.
- A genuine dimer and a doubled charge assignment are indistinguishable within one spectrum. The
  `suppress_multimers` setting decides which way to resolve it, and is off for the native and
  size-exclusion methods.
- An adduct whose mass is within about 1 Da of a modification cannot be separated from it at
  modest resolving power. Such species are flagged rather than silently merged.
- Adduct fractions become uncertain when the adduct spacing is below about 0.7 peak widths, which
  happens for sodium on a monoclonal antibody at a resolving power of 10 000.
- Thermo `.raw` files are read directly (optional `opentfraw` dependency); other vendor formats
  must be converted to mzML first.
- Under ion-pairing HILIC with triethylammonium acetate, oligonucleotide impurities may carry a
  single charge state above the noise and are missed unless `deconvolution.single_charge` is on,
  which accepts them when resolved isotope spacing fixes the charge, at the cost of more
  unconfirmed reports.
- Abundances of a minor species overlapping a much stronger one (the PS-to-PO form under its
  parent in IP-RPLC) can be underestimated by orders of magnitude.
