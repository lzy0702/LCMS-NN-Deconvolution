# LCMS-NN-Deconvolution

Neural-network deconvolution, adduct merging, extracted-ion chromatogram building,
chromatographic integration and quantitation for LC-ESI-MS of macromolecules.

The multiply-charged electrospray envelope in every MS frame is deconvolved to neutral-mass
species by a 1D convolutional network working on a logarithmic m/z grid, then refined by
weighted non-negative least squares. Deconvolved species are traced across retention time into
extracted-ion chromatograms, adduct forms are merged into their base species with per-adduct
fractions reported, ionisation and detector saturation are flagged against the UV trace, and
peaks are integrated with Agilent-CDS-style integration events for purity and potency.

The model is trained entirely on synthesized data. Nothing is downloaded at run time: the model
ships inside the package, as do the vocabularies used to read and write mzML (below), and
inference runs on ONNX Runtime with its telemetry off, so the software runs offline on ordinary
lab PCs with no discrete GPU.

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

Reading mzML and mzMLb needs the PSI-MS controlled vocabulary, and writing mzML the Units of
Measurement Ontology as well: pyteomics (5 and later) otherwise fetches PSI-MS from
purl.obolibrary.org for every file it reads, and psims both for the first file a process writes.
The package carries copies, used for every read and write: PSI-MS 4.2.18 and UO
releases/2026-10-12, both CC BY 4.0, in `src/lcmsdeconv/data/cv/` (its `README.md` gives their
sources, checksums and attribution). A term a file uses that the saved PSI-MS lacks (one added
to PSI-MS later) is read with a warning instead of failing the file, and
`python scripts/refresh_vocabularies.py` replaces the copies with the current releases.

ONNX Runtime's telemetry stays off unless you set `ORT_DISABLE_TELEMETRY=0`. Its Linux build
records a device ID under `~/.cache/Microsoft` and can contact its telemetry server unless that
variable is 1 when it loads: lcmsdeconv sets it when it is imported, unless it is set already.
Its Windows build does not read the variable; its telemetry, an event trace, is switched off by
`onnxruntime.disable_telemetry_events()`, which lcmsdeconv calls when it loads ONNX Runtime. A
program that imports `onnxruntime` before `lcmsdeconv` must set the variable itself, and one that
runs ONNX Runtime itself rather than through lcmsdeconv loads it with
`lcmsdeconv.nn.infer.import_onnxruntime()`, which does both.

Thermo `.raw` files are read directly with the `[thermo]` extra (the open-source `opentfraw`
parser, no vendor libraries); other vendor formats need converting to mzML or mzMLb
(`msconvert --mzMLb`) first. From a Waters run (scan IDs `function=N process=P scan=S`) one
function is read: the one with the most MS1 scans, on a tie the lowest number (the low-energy
function of MSE), which leaves out the lockspray reference scans; `--waters-function N` picks
another.

## Use

```bash
lcmsdeconv methods                                     # list bundled methods
lcmsdeconv synth --preset protein --out runs/demo      # a synthetic run with ground truth
lcmsdeconv process runs/demo/run.mzML --method rplc_pos_protein --out results/
lcmsdeconv report results/ --open
lcmsdeconv gui                                         # desktop application
```

`process` writes `results.json`, `species.csv`, a peak table per integrated signal, and a
self-contained `report.html`, which escapes run, method and trace names, so they show as written.

```python
from lcmsdeconv.core.method import Method
from lcmsdeconv.io.mzml import read_mzml
from lcmsdeconv.process import process_run

result = process_run(read_mzml("run.mzML"), Method.load("rplc_pos_protein"))
for s in result.species[:5]:
    print(f"{s.mass:10.2f} Da  RT {s.rt_apex:.2f}  {s.adduct_fractions()}")
```

## Training

The recommended charge network is **LadderNet**, which reads every candidate charge's ladder at
its exact offsets on the log m/z grid and starts as the comb estimator. To train it on your own
NVIDIA GPU, follow **`gpu_training/README.md`**: installation, a GPU check, the whole-spectrum
recipe in its two views (a fixed window, and a ladder that reads a spectrum of any length whole)
with isotope-charge channels, pausing and resuming at any time, and what to send back.

```bash
bash gpu_training/setup_linux.sh        # Windows: gpu_training\setup_windows.bat
bash gpu_training/train_all.sh          # Windows: gpu_training\train_all.bat
python scripts/eval_charge_bins.py runs/ladder_v4_window.onnx runs/ladder_v4_comb.onnx
```

Training data is synthesized on the fly, so there is no dataset to obtain and no epoch repeats.
See `docs/training.md` for both networks (LadderNet and the original U-Net), the checks run
before long trainings, and how to target an integrated GPU or NPU through the OpenVINO and
DirectML execution providers.

Setting `model: comb` in a method selects a deterministic charge estimator that needs no model
at all. It is what the package uses until a trained network beats it on real data: the CPU-trained
LadderNets read charges better per bin on synthetic spectra but report more junk on the real runs
(`docs/benchmarks.md`). It makes the package usable immediately and is the baseline any model
should beat.

To score trained networks against that baseline on your own computer, follow
**`bench/README.md`**: one setup and one command run the four Thermo runs of the validation
(`docs/validation.md`) and the network bars of `docs/training.md` for the comb and every network
found (DirectML on a Windows GPU, else the processor), and write a summary and a zip to send back.

## Diagnostics

Before improving a model, find the stage that loses species. `lcmsdeconv.diagnostics` provides a
perfect per-bin charge oracle (the ceiling of everything downstream of the network), a confused
oracle that mislabels charge envelopes at a chosen rate (how good the network must be), one-to-one
scoring, classification of false species, and attribution of every missed species to the stage
where it was lost, recorded by `process_run(run, method, trace=[])`:

```bash
python scripts/benchmark_synth.py --runs 4 --predictor oracle     # ceiling with perfect charges
python scripts/benchmark_synth.py --runs 4 --predictor comb       # the deterministic estimator
python scripts/benchmark_synth.py --runs 4 --predictor confused:0.8
python scripts/benchmark_synth.py --runs 4 --predictor models/charge_unet_cpu.onnx
```

`docs/lessons.md` explains why these tools come first: it distils the record of a parallel
three-month effort at the same problem, what went wrong there, and what that changed here.

## Documentation

| Document | Contents |
| --- | --- |
| `docs/lessons.md` | lessons from a parallel neural-network effort and the fixes they led to here |
| `docs/architecture.md` | the logarithmic m/z transform, the network, decoding, refinement, the across-time pipeline |
| `docs/usage.md` | methods, settings, adducts, reading the output, saturation warnings |
| `docs/integration_events.md` | every integration event, baseline codes, how valley clusters are integrated |
| `docs/synthetic_data.md` | what is simulated and why labels cannot drift from features |
| `docs/training.md` | GPU training, model sizes, ONNX deployment, offline installation |
| `docs/validation.md` | benchmarks, public-data checks, known limitations |
| `docs/iwsp_transformer.md` | design of a transformer with hand-written weights for 2D deconvolution (interpretable weight-space programming), with the real-run measurements behind it (`scripts/iwsp_probes/`) |

## Validation

Four real Orbitrap runs of a GalNAc-conjugated phosphorothioate oligonucleotide and its impurities,
published with known compositions (Tutiš et al., Anal. Chem. 2025, 97, 15717,
https://doi.org/10.1021/acs.analchem.5c01407), are scored against the paper's species: of 20 known
species the original code found 2 among 87 reports; the current code, since vendor-thresholded
profiles are resampled correctly, the decoder was recalibrated for it, every species' amount is
made the same way and weak species' templates are placed by their own isotope comb, finds all 20,
and all 29 of the impurity elutions its Supporting Information lists (five IP-RPLC shortmers among
them read 0.52-0.75 Da light of their own isotopologues), among 250 reports, with masses within
about 0.1 Da of theory for the main species and base-form impurity shares within 0.88-1.19 of
extracted-ion references for 19 of the 24 impurities in the amounts table (two of them references
read on the species' own isotopologues; `docs/benchmarks.md`, `scripts/validate_oligo_runs.py`).
Public spectra are used only for validation, never for training.
Bovine serum albumin and the alcohol dehydrogenase tetramer are recovered within 1 % of their
literature masses from public native mass spectra. On synthetic data with known truth, a 17 kDa
protein carrying a 10 % sodium adduct is recovered to 0.2 ppm with the adduct fraction measured at
12 %, and the integrator reproduces analytic Gaussian areas exactly for isolated peaks and
conserves the total area of a valley-split cluster. Run `python scripts/fetch_public_data.py` to
enable the public-data tests and `python scripts/benchmark_synth.py` for the synthetic benchmark.

Known limitations are listed in `docs/validation.md`; the most important is that an adduct whose
mass is within about 1 Da of a modification cannot be separated from it at modest resolving
power, so such species are flagged rather than silently merged.

## Licence

MIT (`LICENSE`). The PSI-MS and UO vocabularies in `src/lcmsdeconv/data/cv/` are CC BY 4.0
(attribution, and the citation the PSI-MS maintainers ask for, in their `README.md`), so the
package declares `MIT AND CC-BY-4.0`.
