# Training

The bundled model (`src/lcmsdeconv/models/charge_unet_proof.onnx`) is a small model trained on
CPU as a working default. For production accuracy, retrain the full model on a GPU.

## On an NVIDIA GPU (for example an RTX 4080 Super)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[train]"

lcmsdeconv train --model-size full --epochs 100 --train-len 50000 --val-len 2000 \
                 --batch-size 16 --workers 8 --out models/charge_unet.pt
lcmsdeconv export-onnx models/charge_unet.pt -o src/lcmsdeconv/models/charge_unet.onnx
```

The full model has about 4.2 million parameters. Data is synthesized on the fly, so there is no
dataset to download and no epoch repeats the same spectra. With eight loader workers the
generator keeps a 4080-class GPU busy; expect a few hours for the schedule above. Training keeps
whichever of the exponential moving average and the live weights validates better, and saves the
best epoch.

`--max-seconds` stops training at a wall-clock budget and still saves the best checkpoint, which
is how the bundled CPU model is produced:

```bash
python scripts/train_proof.py --epochs 30 --train-len 600 --max-seconds 7200
```

## What is being learned

Per grid bin, a distribution over 101 classes (0 = not an ion, 1…100 = charge state) plus an apex
heatmap. Targets are soft: the share of that bin's intensity belonging to each charge. The loss
is an intensity-weighted cross-entropy — bins are weighted by `clip(log10(1 + I/noise), 0, 4)`
plus a small floor, and bins dominated by low-abundance components are weighted more heavily, so
the network is not rewarded for simply predicting "not an ion" everywhere — plus a CenterNet
focal loss on the heatmap.

## Choosing a model size

| Size | Channels | Parameters | Use |
| --- | --- | --- | --- |
| `small` | 8–64, 5 levels | 0.4 M | tests |
| `proof` | 8–80, 6 levels | 1.0 M | bundled CPU-trained default |
| `full` | 16–160, 6 levels | 4.2 M | GPU-trained production model |

## Deployment

Training needs PyTorch; running the software does not. `lcmsdeconv export-onnx` writes an ONNX
file plus a small JSON sidecar and verifies that the exported graph reproduces the PyTorch
outputs. At run time, inference uses ONNX Runtime with the providers named in the method:

```yaml
deconvolution:
  providers: ["OpenVINOExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
```

Unavailable providers are skipped and the CPU provider is always the fallback, so the same model
file runs on a laptop, on an integrated GPU through OpenVINO, and on an NPU where the vendor
supplies an execution provider. Install the extra providers with `pip install .[accel]`.

## Offline installation

Nothing is downloaded at run time: the model ships inside the package. To install on a machine
without internet, build a wheelhouse on a connected machine and copy it across:

```bash
pip download lcmsdeconv -d wheelhouse            # on a connected machine
pip install --no-index --find-links wheelhouse lcmsdeconv   # on the target
```
