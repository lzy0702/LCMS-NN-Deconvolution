# Bundled models

`lcmsdeconv.nn.infer.bundled_model_path()` loads **`charge_unet.onnx`** automatically, and
nothing else. When that file is absent, `lcmsdeconv.deconv.classical.make_predictor` falls back
to the deterministic comb estimator, so the package works out of the box.

## charge_unet_proof.onnx

A 1.05 M parameter model trained on CPU for two hours (18 epochs, roughly 11 000 synthetic
crops), reaching a validation loss of 0.744 and 52.9 % intensity-weighted charge accuracy. It
exists to prove the training and ONNX deployment path end to end, and as a starting point for
fine-tuning.

It is **not** loaded by default, because on a like-for-like test it did not beat the estimator
it would replace: on a synthetic 12.3 kDa protein with six impurities it recovered two of the
seven true components against the comb estimator's three, while running faster (4.1 s against
7.1 s). Two hours of CPU is not enough training for this task. Select it explicitly if you want
to compare:

```bash
lcmsdeconv process run.mzML --model src/lcmsdeconv/models/charge_unet_proof.onnx --out results/
```

## Installing a trained model

```bash
lcmsdeconv train --model-size full --epochs 100 --train-len 50000 --batch-size 16 --workers 8
lcmsdeconv export-onnx models/charge_unet.pt -o src/lcmsdeconv/models/charge_unet.onnx
python scripts/benchmark_synth.py --runs 3 --out docs/benchmarks.md    # confirm it is better
```

`export-onnx` writes a JSON sidecar with the model's maximum charge and verifies that the
exported graph reproduces the PyTorch outputs.
