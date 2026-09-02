# LCMS-NN-Deconvolution

Neural-network deconvolution, adduct merging, extracted-ion chromatogram building,
chromatographic integration and quantitation for LC-ESI-MS of macromolecules.

The multiply-charged electrospray envelope in every MS frame is deconvolved to neutral-mass
species by a 1D convolutional network working on a logarithmic m/z grid, then refined by
weighted non-negative least squares. Deconvolved species are traced across retention time into
extracted-ion chromatograms, adduct forms are merged into their base species with per-adduct
fractions reported, ionisation and detector saturation are flagged against the UV trace, and
peaks are integrated with Agilent-CDS-style integration events for purity and potency.

The model is trained entirely on synthesized data. Nothing is downloaded at run time and
inference runs on ONNX Runtime, so the software runs offline on ordinary lab PCs with no
discrete GPU.

## What it handles

- **Sample types**: peptides and proteins including monoclonal antibodies, DNA and RNA
  oligonucleotides including phosphorothioates, glycans, synthetic polymers (PEG, PPG, PLGA),
  small molecules, and conjugates.
- **Impurities** from 0.01 % to 10 % of the ion current, annotated against a library of
  class-specific modifications (oxidation, deamidation, lysine clipping, glycoforms, n−1 and
  n+1 deletions, depurination, phosphorothioate conversion, end-group variants and more).
- **Instruments**: LC-ESI-ToF primarily, with resolution models for Orbitrap and FT-ICR;
  positive, negative, or polarity switching within a run.
- **Chromatography**: reversed phase, normal phase, HILIC, size exclusion, and ion-pairing
  reversed phase or HILIC, each with its own adduct library (sodium, potassium and ammonium
  under normal reversed phase; triethylamine, di-isopropylethylamine, hexafluoroisopropanol,
  trifluoroacetate, formate or acetate under ion pairing).
- **Saturation**: flat-topped detector frames, and peaks where the MS response compresses at
  the apex relative to its own flanks while the UV response stays linear.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[gui]"            # add [train] for PyTorch, [dev] for tests
```

For an offline machine, build a wheelhouse on a connected one (`pip download`) and install with
`pip install --no-index --find-links wheelhouse lcmsdeconv`.

## Use

```bash
lcmsdeconv methods                                     # list bundled methods
lcmsdeconv synth --preset protein --out runs/demo      # a synthetic run with ground truth
lcmsdeconv process runs/demo/run.mzML --method rplc_pos_protein --out results/
lcmsdeconv report results/ --open
lcmsdeconv gui                                         # desktop application
```

`process` writes `results.json`, `species.csv`, a peak table per integrated signal, and a
self-contained `report.html`.

```python
from lcmsdeconv.core.method import Method
from lcmsdeconv.io.mzml import read_mzml
from lcmsdeconv.process import process_run

result = process_run(read_mzml("run.mzML"), Method.load("rplc_pos_protein"))
for s in result.species[:5]:
    print(f"{s.mass:10.2f} Da  RT {s.rt_apex:.2f}  {s.adduct_fractions()}")
```

## Training

The bundled model is small and CPU-trained. For production accuracy, retrain on a GPU:

```bash
lcmsdeconv train --model-size full --epochs 100 --train-len 50000 --batch-size 16 --workers 8
lcmsdeconv export-onnx models/charge_unet.pt -o src/lcmsdeconv/models/charge_unet.onnx
```

Training data is synthesized on the fly, so there is no dataset to obtain and no epoch repeats.
See `docs/training.md`, including how to target an integrated GPU or NPU through the OpenVINO
and DirectML execution providers.

Setting `model: comb` in a method selects a deterministic charge estimator that needs no model
at all. It is weaker than a trained network but makes the package usable immediately and is the
baseline any model should beat.

## Documentation

| Document | Contents |
| --- | --- |
| `docs/architecture.md` | the logarithmic m/z transform, the network, decoding, refinement, the across-time pipeline |
| `docs/usage.md` | methods, settings, adducts, reading the output, saturation warnings |
| `docs/integration_events.md` | every integration event, baseline codes, how valley clusters are integrated |
| `docs/synthetic_data.md` | what is simulated and why labels cannot drift from features |
| `docs/training.md` | GPU training, model sizes, ONNX deployment, offline installation |
| `docs/validation.md` | benchmarks, public-data checks, known limitations |

## Validation

Public spectra are used only for validation, never for training. Bovine serum albumin and the
alcohol dehydrogenase tetramer are recovered within 1 % of their literature masses from public
native mass spectra. On synthetic data with known truth, a 17 kDa protein carrying a 10 % sodium
adduct is recovered to 0.2 ppm with the adduct fraction measured at 12 %, and the integrator
reproduces analytic Gaussian areas exactly for isolated peaks and conserves the total area of a
valley-split cluster. Run `python scripts/fetch_public_data.py` to enable the public-data tests
and `python scripts/benchmark_synth.py` for the synthetic benchmark.

Known limitations are listed in `docs/validation.md`; the most important is that an adduct whose
mass is within about 1 Da of a modification cannot be separated from it at modest resolving
power, so such species are flagged rather than silently merged.

## Licence

MIT.
