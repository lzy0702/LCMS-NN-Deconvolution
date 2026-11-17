# Training the charge network on your GPU

This folder trains **LadderNet**, the charge-assignment network of `lcmsdeconv`, on synthetic
spectra generated on the fly. Nothing needs to be downloaded except Python packages, and no
data leaves your computer. When it is done, upload the files listed in step 8.

## What is being trained

For every bin of a spectrum on a logarithmic m/z grid, the network decides which charge state
the ions there carry. A species' charge states lie at fixed offsets on that grid (ln(z/(z±k))),
so LadderNet reads each candidate charge's "ladder" at exactly those offsets, plus the positions
that only a doubled charge would fill, and learns how to weigh what it finds. It starts as an
exact copy of the deterministic comb estimator and learns corrections to it. It has about
22 000 parameters (the `full` size). The work is in generating the training spectra, which
happens on your CPU cores while the GPU trains.

Singly charged ions (chemical background) are the most common ions per bin and a network
otherwise learns them first. Stage 1 therefore gives them **no weight** in the loss
(`--singly-charged ignore`): they stay in the spectra, but predicting them is neither rewarded
nor penalised. The alternative, `--singly-charged drop`, removes them from the spectra
altogether.

## Requirements

- An NVIDIA GPU with 6 GB or more and a recent driver (an RTX 4080 Super is plenty). A
  `full`-size training step with a batch of 8 needs about 4 GB; with less, add
  `--batch-size 4`.
- Python 3.10, 3.11 or 3.12.
- As many CPU cores as possible: spectrum generation, not the GPU, sets the pace. Each
  spectrum-generating worker needs about 1 GB of memory; the scripts start up to 8.
- About 2 GB of disk for the Python environment.

## 1. Unpack and open a terminal in this repository's top folder

The folder that contains `pyproject.toml`, `src/`, `scripts/` and `gpu_training/`.

## 2. Create a Python environment

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install PyTorch with CUDA

Pick the command for your system at <https://pytorch.org/get-started/locally/> (Stable, your
OS, Pip, Python, the newest CUDA version your driver supports). For example, for CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 4. Install this package

```bash
pip install -e ".[train]"
```

## 5. Check the installation (one minute)

```bash
python gpu_training/check_gpu.py
python scripts/train_charge_net.py --smoke --device cuda
```

The first prints your GPU and runs three training steps on it. The second trains one tiny epoch
end to end and should finish with `saved runs/smoke.pt`. If either fails, see
**Troubleshooting** below.

## 6. Stage 1: train without singly charged ions in the loss (2-4 hours)

Stage 1 and stage 2 (step 7) are the two runs to do; together they take about 3-6 hours, and
both can be paused and resumed at any time.

Windows: `gpu_training\run_stage1.bat`. Linux or macOS: `bash gpu_training/run_stage1.sh`.
Any option you add is passed on (for example `--workers 6` or `--batch-size 4`). Both run:

```bash
python scripts/train_charge_net.py --device cuda --arch ladder --model-size full \
    --singly-charged ignore --epochs 30 --train-len 4000 --val-len 128 --batch-size 8 \
    --workers 8 --out runs/ladder_full_ignore.pt
```

`--workers` is the number of processes generating spectra: the scripts use the number of
CPU cores minus two, at most 8. Every epoch prints a line such as

```text
epoch 3: charge_acc_given_ion_z2 ema 0.612 live 0.598 (comb 0.574) -> keeping ema; bands {...}
```

The number in brackets is the comb estimator on the same validation spectra: the bar to beat.
The best checkpoint is saved whenever it improves, so you can stop the run at any time
(Ctrl+C) and still have the best model so far. A log of every epoch is written next to it
(`runs/ladder_full_ignore.log.jsonl`).

### Pausing and resuming

You can stop a run at any time and continue later, even in the middle of an epoch:

- **Pause:** press **Ctrl+C once** in the training window, or create an empty file named
  `PAUSE` (or `PAUSE.txt`, as Windows Explorer names a new text file) in the `runs` folder,
  useful if the window is hidden. The current step finishes,
  the complete training state is saved to `runs/ladder_full_ignore.resume.pt`, and the
  program ends with `paused: state saved ...`. Pressing Ctrl+C a second time aborts at once
  without saving.
- **Resume:** run exactly the same command again with `--resume` added, for example
  `gpu_training\run_stage1.bat --resume` on Windows or `bash gpu_training/run_stage1.sh --resume`
  on Linux. It continues from the saved epoch and step with the same learning-rate schedule,
  and refuses to resume if you changed a setting that would alter the run (epochs, batch size,
  spectra per epoch, learning rate, model). A leftover `PAUSE` file is removed automatically.
- If you start the command **without** `--resume` while a paused run's state exists, it stops
  with a message instead of overwriting it; add `--restart` to deliberately start over.
- The state is also saved every 10 minutes (`--save-every` changes that) and after every
  epoch, so a crash, a reboot or a power cut loses at most a few minutes of training: resume
  the same way.

## 7. Stage 2: add singly charged ions and dense background (1-2 hours)

Stage 1 never penalises calling a singly charged ion multiply charged, and on real runs the
column wash is full of singly charged background: a stage-1 network reads some of it as small
multiply charged species. Stage 2 continues from stage 1 with singly charged ions trained on and
up to 200 background ions per spectrum. Windows: `gpu_training\run_stage2.bat`. Linux or macOS:
`bash gpu_training/run_stage2.sh`. Both run:

```bash
python scripts/train_charge_net.py --device cuda --arch ladder --model-size full \
    --singly-charged keep --background-ions 200 --init runs/ladder_full_ignore.pt \
    --epochs 12 --lr 5e-4 --train-len 4000 --val-len 128 --batch-size 8 \
    --workers 8 --out runs/ladder_full_stage2.pt
```

Pausing and resuming work the same way (`--resume`). Here the printed accuracy covers all ion
bins, singly charged ones included, so it is lower than stage 1's.

### Optional: the "drop" variant, for comparison

Stage 1 with singly charged ions removed from the spectra altogether:

```bash
python scripts/train_charge_net.py --device cuda --arch ladder --model-size full \
    --singly-charged drop --epochs 30 --train-len 4000 --val-len 128 --batch-size 8 \
    --workers 8 --out runs/ladder_full_drop.pt
```

## 8. What to upload

The `.pt` checkpoints and `.log.jsonl` logs from the `runs` folder (each checkpoint is well under
1 MB): `runs/ladder_full_ignore.pt`, `runs/ladder_full_stage2.pt` and their `.log.jsonl` files
(and `runs/ladder_full_drop.*` if you ran it). The `.resume.pt` files are not needed. Zip
the `runs` folder if that is easier. The checkpoint is converted to ONNX and scored on the
synthetic benchmark and on the four real runs after you upload it.

## Troubleshooting

- `CUDA is NOT available`: the installed torch is a CPU build or the NVIDIA driver is missing.
  Uninstall torch (`pip uninstall torch`) and install it again with the CUDA command of step 3.
- The GPU is mostly idle: expected, spectrum generation is the bottleneck. More `--workers`
  helps up to the number of cores.
- Workers crash or the computer runs out of memory: lower `--workers` (each worker uses about
  1 GB) or `--batch-size`.
- On Windows a run seems to hang at the start: the first spectra take a while with many
  workers; give it a minute.
