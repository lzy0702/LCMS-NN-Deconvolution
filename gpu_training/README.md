# Training the charge network on your GPU

This folder trains **LadderNet**, the charge-assignment network of `lcmsdeconv`, on synthetic
spectra generated on the fly. Nothing needs to be downloaded except Python packages, and no
data leaves your computer. When it is done, you upload one zip file.

The network trains on the **GPU only**: the scripts refuse to start rather than fall back to the
processor. The training spectra are synthesised by data-worker processes on the processor
while the GPU trains; that part cannot run on the GPU. By default everything stays within
**16 GB of RAM** and **14 GB of GPU memory**, leaving the rest to Windows.

**Retrain after updating.** The package now resamples vendor-thresholded profile scans
correctly (a segment's edge point no longer stands for the gap beside it), which changes the
spectra the network sees in training and in use. A model trained with an earlier version of
this folder was trained on the old grid views: start a new run rather than resuming one.

## Quick start (Windows 11, RTX 4080 Super)

**Before you start**

- **NVIDIA driver.** In a terminal, `nvidia-smi` shows "CUDA Version" at the top right. 13.2 or
  higher lets PyTorch use its main build; with 13.0 or 13.1 the setup uses the CUDA 13.0 build
  instead. To update, install the current Game Ready or Studio driver. The CUDA toolkit is
  **not** needed: PyTorch brings its own CUDA libraries.
- **Python 3.12.** If you don't have it, install it with `winget install -e --id Python.Python.3.12`
  (or from python.org, ticking "Add python.exe to PATH"). Python 3.11, 3.13 and 3.14 also work.
- **Folder.** Unzip the package to a short path on the SSD, for example `C:\lcms`, so that
  `C:\lcms\lcmsdeconv\gpu_training` exists. Avoid folders that OneDrive synchronises (on many
  Windows 11 PCs, the Desktop and Documents): syncing slows checkpoint writing and can lock files.
- **Disk.** About 10 GB free (PyTorch with CUDA is large).

**1. Set up (once, 10-20 minutes):** double-click `gpu_training\setup_windows.bat`, or run it
from a terminal in the `lcmsdeconv` folder. It:

- creates the Python environment `.venv` and installs PyTorch for CUDA and this package;
- checks that PyTorch computes on the GPU;
- measures your machine: a real training step on the GPU under the 14 GB cap, and how much
  memory each data worker takes and how many spectra per second they deliver;
- chooses the number of workers that keeps the training within 16 GB of RAM (written to
  `gpu_training\machine.json`);
- runs a two-minute training to check everything end to end.

At the end it prints the expected training time. Its output is also in `gpu_training\setup.log`.
Other limits: `gpu_training\setup_windows.bat --ram-gb 12 --gpu-memory-gb 12`.

**2. Train (about 2-3 hours; setup prints the estimate for your computer):** double-click
`gpu_training\train_all.bat`. It runs stage 1 and
stage 2 (below), exports both networks to ONNX and packs the results. You can pause at any
time and continue later (see "Pausing and resuming").

**3. Upload** the file it names at the end: `runs\upload_YYYYMMDD-HHMM.zip`. It holds the two
networks, their ONNX exports, the training logs, `machine.json` and `setup.log`.

## The limits, and how they are kept

- **GPU memory, at most 14 GB:** PyTorch's memory is capped at 14 GB less an allowance for the
  CUDA context. A training step needs about 4 GB, so the cap is never close. It also keeps the
  Windows driver from spilling GPU memory into system RAM.
- **RAM, at most 16 GB:** the training process plus its data workers, counted as Task Manager
  counts memory (each process's private working set). Setup measures what one worker takes on
  your machine and starts as many as fit, with a 1.5 GB margin. If the budget is reached
  anyway, the run pauses safely and resumes at once with two workers fewer (this is remembered
  in `machine.json`). For another budget, `train_all.bat --ram-gb 20` starts as many workers as
  the measured sizes allow, or re-measure with `setup_windows.bat --ram-gb 20`.
- **Processor:** the workers run at below-normal priority, so the computer stays responsive for
  other work; when it is idle, they use the cores fully. With the 16 GB budget, about 8 to 12 of
  the 24 cores are busy. To leave more free, use fewer workers (slower):
  `train_all.bat --workers 6`.
- **Sleep:** while training, Windows is kept from going to sleep (the screen may still turn off).
  A forced restart by Windows Update would interrupt the run (it then resumes from the last
  saved state). Pause updates for the day, or set active hours, before a long run.

## What the output means

Before the first epoch, each stage scores the comb estimator on the validation spectra (a
minute). Then every epoch prints lines such as

```text
epoch 8: charge_acc_given_ion_z2 ema 0.903 live 0.907 (comb 0.668) -> keeping live; bands {...}
  charge of z >= 2 ion bins read right (log-SNR / intensity weighted):
    time-of-flight  network 0.895 / 0.931   comb 0.689 / 0.844
    Orbitrap        network 0.850 / 0.917   comb 0.237 / 0.290
  epoch took 2.1 min; 48 spectra/s; RAM 11.8 GB of 16; GPU 3.9 GB reserved; about 1.1 h to go
```

These example numbers come from a 45-minute CPU run of the small model; the last line is
illustrative.

- **The number in brackets** is the comb estimator, scored the same way on the same validation
  spectra. The comb is run as the pipeline runs it: over the whole spectrum, with charges read
  from isotope spacing on Orbitrap spectra.
- **The two lines below it** split that comparison by kind of spectrum, each spectrum counting
  the same. Bins are weighted by log-SNR (the training metric, which counts weak ions as much
  as strong ones) and by intensity (what the decoder's mass votes weigh). A network worth
  uploading beats the comb in all four numbers. Whether that carries over to species is decided
  after you upload: on the strongest spectra (where the comb still does well), on synthetic
  species and on the real runs.
- **The last line** gives speed, memory and the time left. "spectra/s" is set by the data
  workers; the GPU could go several times faster.

The best checkpoint is saved whenever it improves. Logs of every epoch are written to
`runs\*.log.jsonl` and the full console output to `runs\*.console.log`.

## Pausing and resuming

- **Pause:** press **Ctrl+C once** in the training window, or create a file named `PAUSE` in
  the `runs` folder (an empty text file is fine: PAUSE, PAUSE.txt or PAUSE.txt.txt all count). The
  current step finishes, the complete training state is saved, and the program ends. After
  Ctrl+C, Windows then asks "Terminate batch job (Y/N)?": the state is already saved, so either
  answer is safe (N keeps the window open so you can read the last messages). The PAUSE file
  pauses without that question.
- **Continue:** run `gpu_training\train_all.bat` again. A finished stage is skipped, and an
  interrupted one continues from the saved epoch and step with the same learning-rate schedule.
  The same works after a crash, a reboot or a power cut. The state is saved every 10 minutes
  and after every epoch, so at most a few minutes are lost.
- **Abort:** pressing Ctrl+C twice stops at once. The last saved state is kept.
- **Start over:** `gpu_training\train_all.bat --restart` discards the saved states. A stage is
  also started afresh by itself when the stage before it has been trained again.
- **Earlier package versions:** a run they started is refused, because its training spectra
  differ. Start again with `--restart`.
- **One at a time:** a second `train_all.bat` started while one is running stops at once; two
  runs would share their files and double the memory.

## The two stages

Singly charged ions (chemical background) are the most common ions per bin, and a network learns
them first. Training therefore has two stages:

- **Stage 1** (`runs\ladder_full_ignore.pt`), 40 epochs of 6 000 new spectra: singly charged
  ions stay in the spectra but get **no weight** in the loss. Predicting them is neither
  rewarded nor penalised.
- **Stage 2** (`runs\ladder_full_stage2.pt`), 16 epochs of 6 000 spectra of other molecules,
  continuing from stage 1: singly charged ions are trained on, with up to 200 background ions
  per spectrum. On the real
  runs, the column wash is full of singly charged background that a stage-1 network reads as
  small multiply charged species.

In stage 2 the printed accuracy covers all ion bins, singly charged ones included, so it is
lower than stage 1's. Watch the `'z=1'` entry of `bands` rise over the epochs; that is what this
stage is for. If it stays low after half the epochs, mention it when you upload.

The training spectra mix positive-mode proteins and polymers, negative-mode ion-pairing
oligonucleotides on a time-of-flight instrument, the same on an Orbitrap at 35 000-280 000
resolving power (a quarter of the spectra; your runs are at 140 000), and a general mix. Each
stage runs `scripts/train_charge_net.py`, for example for stage 1:

```bash
python scripts/train_charge_net.py --device cuda --arch ladder --model-size full \
    --singly-charged ignore --epochs 40 --train-len 6000 --lr 2e-3 --val-len 128 \
    --batch-size 8 --workers 10 --max-ram-gb 16 --max-gpu-memory-gb 14 \
    --out runs/ladder_full_ignore.pt
```

`gpu_training\run_stage1.bat` and `run_stage2.bat` run one stage only, with the settings of
`train_all.bat`. The script's exit code is 0 when a run finished, 3 when it paused, and 4 when it
stopped at the memory budget.

## Linux

```bash
bash gpu_training/setup_linux.sh
bash gpu_training/train_all.sh
```

## Troubleshooting

- **"Python 3.12 was not found"** (from `setup_windows.bat`): install it as in the quick start,
  then open a new terminal so that PATH is updated.
- **"nvidia-smi was not found"** or **"No PyTorch CUDA build worked with this driver"**:
  install or update the NVIDIA driver and run the setup again.
- **"pip could not download or install PyTorch"**: the setup downloads PyTorch from
  `download.pytorch.org`, which a company proxy or firewall may block; it also needs about
  10 GB of free disk, and no other program may be using `.venv` (close other Python windows).
- **"CUDA is not available, and this run is set to train on the GPU only"**: PyTorch lost its
  CUDA build. Run `gpu_training\setup_windows.bat --reinstall-torch`.
- **"the training process and 1 worker already need more than the RAM budget"**: the budget is
  too low for this training (`gpu_training\machine.json` lists what the training process and
  one worker need). Raise it with `--ram-gb`.
- **`[WinError 1455] The paging file is too small`** while workers start: let Windows manage
  the paging file size (System > About > Advanced system settings > Performance > Advanced >
  Virtual memory), or use fewer workers (`--workers 6`). The setup's measurement already halves
  the number of workers if they fail to start.
- **Slow starts or saves:** real-time virus scanning of the folder can slow them. If you wish,
  exclude it from an administrator PowerShell: `Add-MpPreference -ExclusionPath C:\lcms`.
- **Re-measure the machine** after changing the limits or hardware: `.venv\Scripts\python
  gpu_training\probe.py --ram-gb 16 --gpu-memory-gb 14`.
