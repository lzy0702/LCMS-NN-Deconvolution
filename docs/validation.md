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
| Isotope patterns against pyteomics | monoisotopic mass agrees to below 2 mDa across small molecules to 100-atom peptides |
| Deconvolution of a 17 kDa protein with 10 % sodium adduct | mass within 0.2 ppm, adduct fraction 12 % against a true 10 %, residual 3.5 % |
| Integrator on analytic Gaussians | isolated peak area exact to 0.00 %, valley cluster total exact to 0.0 % |
| ONNX export | outputs match PyTorch to about 1e-6 |
| Ionisation saturation | detected on a compressed response, not raised on a linear one |

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

- The bundled model is CPU-trained and small. Retrain on a GPU (see `training.md`) before using
  the software for decisions; the deconvolution mathematics is independent of model quality but
  candidate generation is not.
- An adduct whose mass is within about 1 Da of a modification cannot be separated from it at
  modest resolving power. Such species are flagged rather than silently merged.
- Adduct fractions become uncertain when the adduct spacing is below about 0.7 peak widths, which
  happens for sodium on a monoclonal antibody at a resolving power of 10 000.
- Direct reading of vendor formats is out of scope; convert to mzML first.
