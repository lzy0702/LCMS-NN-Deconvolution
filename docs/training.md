# Training

Two charge networks can be trained: **LadderNet** (`--arch ladder`, the recommended one) and the
1D U-Net used so far (`--arch unet`). `src/lcmsdeconv/models/charge_unet_proof.onnx`, a U-Net
trained on CPU for two hours, is **not** loaded by default: on a like-for-like test it recovered
two of seven true components against the deterministic comb estimator's three. Only
`charge_unet.onnx` is picked up automatically, so a model replaces the estimator only once it
has been shown to beat it.

## LadderNet

On the grid u = ln(m/z ∓ m_p) all charge states of one species lie at u = ln M − ln z, so if a
bin carries charge z the same species' charge z + k sits at u + ln(z/(z + k)), whatever its mass:
offsets of thousands of bins, different for every z, which a convolutional network has to
discover from data (CPU probes of the U-Net never got there, see `lessons.md`, section 13).
LadderNet is given them. For each candidate charge it reads a learned local description of the
spectrum (peak shape, isotope spacing; a four-layer dilated convolution) at the bin itself, at
the neighbours z ± 1..3, and at the two positions that only a reading at 2z would fill, halfway
between the z ladder's rungs; a small network shared by all charges, with a learned bias per
charge, scores that reading. The comb estimator's score is added with a learnable weight and the
learned part starts at zero, so the untrained network assigns charges exactly as the comb does
(a test checks this) and training learns corrections. A depthwise convolution over neighbouring
bins lets the bins of one peak agree.

It has 9 500 (`proof`) to 22 000 (`full`) parameters. Ladders are read at the 6 144 strongest
bins of a window (ions occupy a few percent of the bins), which a sparse gather makes cheap in
training; the ONNX export computes the same function densely with static shifts, for a fixed
32 768-bin window, which is what inference uses.

### Singly charged ions

Singly charged ions (chemical background, small molecules) are the most common ions per bin, and
a network learns them first. `--singly-charged` controls them:

- `ignore`: they stay in the spectra, so the input is realistic, but bins carrying them get no
  weight in the loss: predicting them is neither rewarded nor penalised.
- `drop`: the generator produces none.
- `keep`: everything is trained on (a fine-tuning stage after either of the above).

With `ignore` or `drop` the checkpoint is selected by the charge accuracy on ion bins of charge
two or more (`charge_acc_given_ion_z2`, bins weighted by log-SNR), which every epoch prints next
to the comb's on the same validation spectra, then split by kind of spectrum (time-of-flight,
Orbitrap) with bins weighted by log-SNR and by intensity. The comb there is the one the pipeline
runs (`lcmsdeconv.nn.baseline`): over the whole spectrum each validation crop was cut from, with
charges read from isotope spacing on Orbitrap spectra. The comb's answer in feature channel 3 is
no substitute: it is computed inside the crop, where the far neighbours of low charge states
(34 657 bins from z = 2 to z = 1) are missing, and scored 0.57 where the pipeline's comb scores
0.64 on the same spectra (an independent review caught the earlier use of that proxy as the
bar). `scripts/eval_charge_bins.py` makes the same comparison on fresh spectra, with networks run
as the pipeline runs them too. With `ignore`, a bin partly singly charged keeps its other charges
as targets at proportionally lower weight; singly charged ions leave the target altogether.

Install with the `fast` extra (`pip install -e ".[train,fast]"`): without numba the spectra are
generated about 20 times slower (0.26 against 4.9 seconds per spectrum here).

### Running it

```bash
python scripts/train_charge_net.py --smoke                    # a two-minute installation check
python scripts/train_charge_net.py --device cuda --model-size full --singly-charged ignore \
    --epochs 30 --train-len 4000 --batch-size 8 --workers 8 --out runs/ladder_full_ignore.pt
lcmsdeconv export-onnx runs/ladder_full_ignore.pt -o runs/ladder_full_ignore.onnx
```

`gpu_training/README.md` is a step-by-step guide for training on your own GPU (installation,
a check script, run scripts for Windows and Linux, what to send back).

**Pausing and resuming.** Ctrl+C once, or an empty file named `PAUSE` next to the output, ends
the run after the current step and saves the whole training state (model, moving average,
optimiser, learning-rate schedule, position in the epoch, best checkpoint so far) to
`<out>.resume.pt`. The same command with `--resume` continues exactly there. The state is also
saved every `--save-every` minutes (10) and after every epoch, so an interrupted run loses at
most that much. `--minutes` ends a session at a time budget the same way.

## The U-Net on a GPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[train]"

lcmsdeconv train --model-size full --epochs 100 --train-len 50000 --val-len 2000 \
                 --batch-size 16 --workers 8 --out models/charge_unet.pt
lcmsdeconv export-onnx models/charge_unet.pt -o src/lcmsdeconv/models/charge_unet.onnx
```

The full U-Net has about 4.2 million parameters. Data is synthesized on the fly, so there is no
dataset to download. Training keeps whichever of the exponential moving average and the live
weights validates better, and saves the best epoch.

## Training on the input inference sees

Inference deconvolves the summed spectrum of a chromatographic region: vendor-thresholded
profile scans, summed, resampled onto the logarithmic grid, with the noise level estimated from
the result. Training frames are rendered along exactly that path by default
(`SynthConfig.render_path = "profile"`, see `synthetic_data.md`), so the log-SNR feature, the
comb and the loss weights are computed from the same kind of spectrum, with the same noise
estimate, at training and at run time. Earlier models were trained on spectra rendered straight
onto the grid with the generator's own noise parameter, a different input from the one they were
later run on; that mismatch is the most common way a network that trains well fails in use.

## Checks before a long run

`lcmsdeconv.nn.checks` holds three checks that cost seconds and catch the failures that cost
days:

- `degeneracy_report(batch, loss)`: the loss of the true labels against "all background", the
  class prior (the best input-independent answer) and the comb's guess, on real batches. On the
  default mixture the truth scores about 0.11 against 1.24, 3.3 and 6.6; a margin near one
  means training will settle on a constant. Training prints it before the first step.
- `overfit_check(model, samples, loss)`: a model that cannot fit eight fixed samples in a
  hundred steps has broken labels, features or loss, and more data will not help.
- `charge_metrics(logits, batch)`: intensity-weighted charge accuracy overall and by charge band,
  the share of harmonic (2z, z/2) and neighbour (z ± 1) errors, background bins called ions,
  and the number of charge classes present but never predicted.

Every epoch's validation metrics go to `<out>.log.jsonl`, and the saved checkpoint is the one
with the best validation charge accuracy, not the lowest loss: a loss can keep improving while a
model drifts toward a constant answer.

## Starting from the comb

`TrainConfig(comb_prior=True, evidence_channels=...)` builds a network whose charge logits start
as the comb estimator's evidence (fixed offsets from the evidence input channels) and whose
charge head is initialised to zero, so the untrained model already assigns charges like the comb
and training learns a correction. In CPU probes (see `lessons.md`, section 13) a network trained
from scratch reached 12 % charge accuracy on ion bins in 450 steps and predicted almost only
z = 1, while the comb-prior network started at the comb's 53 %, dipped, and returned to 53 %
with the low charges the comb cannot read (z = 2-5: 13 to 47 %) learned. It is the configuration
to train on a GPU.

Charge accuracy is reported two ways: `charge_acc`, an argmax in which the background class
competes, and `charge_acc_given_ion`, the top charge among the ion classes on bins that are
ions. Inference uses the second (`ChargePredictor` reads its top two charges among the ion
classes), checkpoints are selected by it, and only it tracks charge assignment: in the probes
the first fell while the second recovered.

## CPU training script

```bash
python scripts/train_cpu.py --minutes 120 --out models/charge_unet_cpu.pt
lcmsdeconv export-onnx models/charge_unet_cpu.pt -o models/charge_unet_cpu.onnx
python scripts/benchmark_synth.py --runs 4 --predictor models/charge_unet_cpu.onnx
```

It trains on a mixture of positive-mode proteins and polymers (40 %), negative-mode
ion-pairing oligonucleotides (40 %) and a general positive-mode mix (20 %).

## What is being learned

Per grid bin, a distribution over 101 classes (0 = not an ion, 1…100 = charge state) plus an apex
heatmap. Targets are soft: the share of that bin's intensity belonging to each charge. The loss
is an intensity-weighted cross-entropy — bins are weighted by `clip(log10(1 + I/noise), 0, 4)`
plus a small floor, and bins dominated by low-abundance components are weighted more heavily, so
the network is not rewarded for simply predicting "not an ion" everywhere — plus a CenterNet
focal loss on the heatmap.

## What two hours of CPU buys

| | Value |
| --- | --- |
| Epochs completed | 18 |
| Synthetic crops seen | about 11 000 |
| Validation loss | 0.74 |
| Intensity-weighted charge accuracy | 52.9 % |
| Against the comb estimator | two of seven components recovered, against three |

The loss was still falling when the time budget ended. A GPU run sees roughly five hundred
times as many crops in a few hours, which is the regime the architecture was designed for.

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
