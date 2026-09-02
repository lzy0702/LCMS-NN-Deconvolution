# LCMS-NN-Deconvolution — developer notes

Python package `lcmsdeconv` (src layout). Install for development:

    python -m venv .venv && .venv/bin/pip install -e ".[train,gui,dev]"

Run checks:

    .venv/bin/ruff check src tests
    .venv/bin/pytest
    QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_gui.py

Key entry points:
- `lcmsdeconv process run.mzML --method configs/rplc_pos_protein.yaml --out results/`
- `lcmsdeconv synth --preset protein --out runs/demo` (synthetic mzML + truth)
- `lcmsdeconv train --config configs/train_gpu.yaml` then `lcmsdeconv export-onnx`
- `lcmsdeconv gui`

Design summary lives in docs/architecture.md. The NN works on a ln(m/z ∓ m_proton) grid
(`lcmsdeconv.nn.grid`); labels come from the synthetic renderer (`lcmsdeconv.synth.render`),
inference runs on onnxruntime (`lcmsdeconv.nn.infer`), and quantitation is a weighted NNLS
template fit (`lcmsdeconv.deconv.refine`). Chromatographic integration follows Agilent-style
events (`lcmsdeconv.chrom.integrate`).
