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
  estimator (two of seven components against three) and is therefore opt-in.
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
- Direct reading of vendor formats is out of scope; convert to mzML first.
