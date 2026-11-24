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
32 768-bin window, which is what inference uses. That window spans a factor of 1.93 in m/z, so
it cannot hold the z = 1 and z = 3 neighbours of a doubly charged ion (34 657 and 20 273 bins
away); the comb, which reads the whole spectrum, has the edge there on strong spectra.

`scripts/train_charge_net.py` trains on a mixture of positive-mode proteins and polymers
(30 %), negative-mode ion-pairing oligonucleotides on a time-of-flight instrument (25 %) and on
an Orbitrap at 35 000-280 000 resolving power at m/z 200 (25 %; the real runs of `benchmarks.md`
were recorded at 140 000), and a general positive-mode mix (20 %). The Orbitrap share used to be
15 % at 30 000-63 000, a regime no instrument the package has been tested on works in: there
isotopologues of a 5 kDa oligonucleotide are barely resolved, on the real runs ten grid bins
apart.

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
generated about 25 times slower (0.11 against 3.1 seconds for the same spectra here). The
compiled steps (`synth/fast.py`, the stick builder and the adduct-state sampler) give exactly
the spectra the plain numpy code gives, bit for bit; `tests/test_synth_fast.py` checks every one
against its reference and whole training frames against the original generator.

### Running it

```bash
python scripts/train_charge_net.py --smoke                    # a two-minute installation check
python scripts/train_charge_net.py --model-size full --singly-charged ignore --epochs 40 \
    --train-len 6000 --batch-size 8 --max-ram-gb 16 --max-gpu-memory-gb 14 \
    --out runs/ladder_full_ignore.pt
lcmsdeconv export-onnx runs/ladder_full_ignore.pt -o runs/ladder_full_ignore.onnx
```

The script trains on the GPU unless told `--device cpu`. Data workers default to as many as the
cores and `--max-ram-gb` allow (`lcmsdeconv.nn.machine`), run at lowered priority, and stay
alive for the whole run: the epoch travels with the item indices (`EpochSampler`), so
Windows, which starts every worker as a fresh process, starts them once. The validation spectra
are generated once and kept in memory. A run stopped at the memory budget pauses like Ctrl+C;
the exit code tells the cases apart (0 finished, 3 paused, 4 memory budget).

`gpu_training/README.md` is the guide for training on your own GPU: `setup_windows.bat` (or
`setup_linux.sh`) installs and measures the machine, `train_all.bat` runs both stages within
the RAM and GPU memory limits, resumes after any interruption and packs what to send back.

**Pausing and resuming.** Ctrl+C once, or an empty file named `PAUSE` next to the output, ends
the run after the current step and saves the whole training state (model, moving average,
optimiser, learning-rate schedule, position in the epoch, best checkpoint so far) to
`<out>.resume.pt`. The same command with `--resume` continues exactly there. The state is also
saved every `--save-every` minutes (10) and after every epoch, so an interrupted run loses at
most that much. `--minutes` ends a session at a time budget the same way.

### The whole-spectrum recipe (experimental)

`--recipe v3` trains LadderNet on the view inference reads: a region's whole grid in one window.
It is being tested at proof size on a CPU; the default recipe above is unchanged.

```bash
python scripts/train_charge_net.py --recipe v3 --out runs/ladder_v3.pt                 # arm A0
python scripts/train_charge_net.py --recipe v3 --iso-channels --out runs/ladder_v3_iso.pt  # arm A1
```

- **Window and ladder reach** (`--window`, `TrainConfig.window`; 32 768 bins by default, 131 072
  with v3). Every frame is read in one window: grids of 60 000-130 000 bins here, a few longer
  ones cropped. The ladder keeps every offset shorter than the window, so it reads z = 1 <-> 2,
  3, 4 (34 657, 54 931 and 69 315 bins), which no 32 768-bin crop holds. Ladders are read at
  the 24 576 strongest bins (6 144 per 32 768). The checkpoint and the ONNX sidecar record the
  window, that cap and the channels; `ChargePredictor` reads a grid that fits in one
  zero-padded window, a longer one in windows overlapping by 25 %.
- **One stage, singly charged ions labelled** (`--singly-charged keep`).
- **Spectra.** Every frame is measured over an instrument's scan range (`scan_range =
  "instrument"`: from 250-600 to 1 500-3 500 m/z, never narrower than its ions), so a species'
  missing charge states lie inside the grid, as in real runs. It carries 0-200 singly charged
  background ions, 10-100 times stronger in a quarter of the frames (`background_strong_prob`).
  The Orbitrap oligonucleotides are dna, rna, ps_dna and ps_rna of 1-12 kDa, and in half the
  frames they charge as in the real ion-pairing runs (`oligo_real_like_prob`: apex
  0.0769 M^0.445, one to three charge states). Off, each `SynthConfig` option draws nothing,
  so default frames are bit for bit those of earlier versions (`tests/test_recipe_v3.py`).
- **Isotope-charge channels** (`--iso-channels`, `TrainConfig.iso_channels`). Channel 4 is
  the charge the comb's isotope reader reads at each bin, channel 5 the strength of that
  reading (best evidence over the bin's log-SNR squared). The reader is
  `lcmsdeconv.nn.features.isotope_reading`, which `CombPredictor.isotope_charges` shares. The
  channels are computed over the whole grid, with absolute m/z and the instrument's resolving
  power, on Orbitrap and FT-ICR spectra; on time-of-flight spectra they are empty, as the
  pipeline's comb reads no isotope charges there. The scorer of each charge also reads whether
  the isotopes say that charge. That charge's logit gains up to `iso_scale` (learnable, 10 at
  the start, above the comb prior's 8) where the reading is strong. So the untrained network
  reads charges as the comb with isotope charges does. Such a model needs the instrument:
  `process_run` passes the method's; other callers pass `instrument=` to `ChargePredictor` or
  set `predictor.instrument`.

On this machine (4 cores), a v3 frame takes about 0.3 s to synthesise in one process. A
proof-size step at 2 threads takes 1.3 s per frame (1.4 with the channels), 1.5-1.6 s with both
arms running at once. One training process peaks at about 4.8 GB and its data worker at about
1.2 GB, so two arms together need about 11 GB.

The isotope reader answers only charges whose spacing it resolves on the 2e-5 grid (2.5 peak
widths, a peak at least two bins wide); an envelope of higher charge gets a lower charge, at full
strength. On the synthetic Orbitrap spectra (35 000-280 000 at m/z 200) that is a quarter of the
z >= 2 ion intensity, and where the charge is resolvable the broader synthetic peaks make it read
z + 1 or higher nearly half the time: the reading is right on about a third of the z >= 2 ion
intensity, against 0.74 for the untrained ladder. On the real runs, at 140 000 with charges 2-6
and sharp peaks, the reader is within its range.

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
The same applies to the resampling: a thresholded segment's edge point once stood for half the
gap beside it, in training as at run time. Since that was fixed (`lessons.md`, section 14) the
grid views differ, and models trained before the fix - the CPU `proof` LadderNets and any GPU
run started before it - must be retrained. So must models trained before training scans were
thresholded as vendor software writes them (segments of at least five points around every point
above the threshold; earlier, noise points were kept one by one, half the segments of a training
scan were single points, and its noise level differed from a real scan's by up to threefold). The
padding is a setting (`SynthConfig.vendor_pad`), so a run paused before the change is refused on
resume rather than continued on other frames.

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
