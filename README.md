# LCMS-NN-Deconvolution

Neural-network deconvolution, adduct merging, EIC extraction, chromatographic integration and
quantitation for LC-ESI-MS of macromolecules (peptides/proteins, oligonucleotides, glycans,
synthetic polymers, small molecules and their conjugates).

The multiply-charged ESI envelope in every MS frame is deconvolved to neutral-mass species by a
1D convolutional neural network working on a logarithmic m/z grid, then refined by weighted
non-negative least squares to quantify components down to ~0.01 % of the total ion current.
Deconvolved species are traced across retention time into extracted-ion chromatograms, adduct
forms (+Na/+K/+NH4 in RPLC, +amine/+acid in ion-pairing RPLC) are merged into their base species
with per-adduct fractions reported, ESI/detector saturation is flagged against the UV trace, and
peaks are integrated with Agilent-CDS-style integration events for purity and potency.

The model is trained entirely on synthesized data; nothing is downloaded at runtime and
inference runs on ONNX Runtime (CPU by default, iGPU/NPU providers optional), so the software
runs offline on ordinary lab PCs.

See `docs/` for the architecture, synthetic-data design, integration-event reference, training
instructions (NVIDIA GPU) and validation notes. Developer setup is in `CLAUDE.md`.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e ".[train,gui,dev]"

# generate a synthetic run and process it
.venv/bin/lcmsdeconv synth --preset protein --out runs/demo
.venv/bin/lcmsdeconv process runs/demo/run.mzML --method rplc_pos_protein --out results/demo
.venv/bin/lcmsdeconv report results/demo --open

# desktop GUI
.venv/bin/lcmsdeconv gui
```

## Status

Under active development. Licensed under the MIT License.
