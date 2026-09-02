# Bundled models

ONNX charge models live here and are loaded automatically by
`lcmsdeconv.nn.infer.bundled_model_path()`.

When this directory holds no model, `lcmsdeconv.deconv.classical.make_predictor` falls back to
the deterministic comb estimator, so the package works out of the box. Set `model: comb` in a
method to select that estimator explicitly.

To install a trained model:

```bash
lcmsdeconv train --model-size full --epochs 100 --train-len 50000 --batch-size 16 --workers 8
lcmsdeconv export-onnx models/charge_unet.pt -o src/lcmsdeconv/models/charge_unet.onnx
```

`export-onnx` also writes a small JSON sidecar with the model's maximum charge, and verifies
that the exported graph reproduces the PyTorch outputs.
