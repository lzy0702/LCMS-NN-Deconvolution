# Training the charge network on your GPU

This folder trains **LadderNet**, the charge-assignment network of `lcmsdeconv`, on synthetic
spectra generated on the fly. Nothing needs to be downloaded except Python packages, and no
data leaves your computer. When it is done, you upload one zip file.

The network trains on the **GPU only**: the scripts refuse to start rather than fall back to the
processor. The training spectra are synthesised by data-worker processes on the processor
while the GPU trains; that part cannot run on the GPU. By default everything stays within
**16 GB of RAM** and **14 GB of GPU memory**, leaving the rest to Windows.

## What is trained

Two networks, one after the other, on the same spectra (the same seed), so that they can be
compared:

- **`window`** (`runs\ladder_v3_window.pt`): the ladder reads a fixed window of 131 072 grid
  bins, a region's whole spectrum; inference cuts a longer spectrum into such windows.
- **`comb`** (`runs\ladder_v3_comb.pt`): the same ladder for a spectrum of any length, read
  whole in one pass. Its layers do not depend on the input's length, it is told which bins were
  measured, and where a rung of a charge ladder falls outside the spectrum it reads a learned
  "not measured" value instead of zeros.

Both use the whole-spectrum recipe: full size, 20 epochs of 6 000 new spectra each (singly
charged ions included), learning rate 0.002, and the **isotope-charge channels**: the charge the
comb's isotope reader reads from resolved isotope spacing, the strength of that reading, and
the **isotope limit**, the highest charge whose isotope spacing can be resolved at that m/z
with the instrument's resolving power. Above the limit the reader answers half the charge
(an envelope of charge 8 it cannot resolve reads as 4); the network uses the limit to tell such
readings apart. Untrained, this alone raises the share of charges of z >= 2 read right on
Orbitrap spectra from 0.39 to 0.66 (0.42 to 0.85 weighted by intensity).

Two more runs exist but are not trained by default: `window-plain` and `comb-plain`, the same
without isotope channels. Choose the runs with `--runs`, for example
`gpu_training\setup_and_train.bat --runs comb` or `--runs window,comb,comb-plain`.

## Quick start (Windows 11, RTX 4080 Super)

**Before you start**

- **NVIDIA driver.** In a terminal, `nvidia-smi` shows "CUDA Version" at the top right. 13.2 or
  higher lets PyTorch use its main build; with 13.0 or 13.1 the setup uses the CUDA 13.0 build
  instead. The CUDA toolkit is **not** needed: PyTorch brings its own CUDA libraries.
- **Python 3.12.** If you don't have it, install it with `winget install -e --id Python.Python.3.12`
  (or from python.org, ticking "Add python.exe to PATH"). Python 3.11, 3.13 and 3.14 also work.
- **Folder.** Unzip the package to a short path on the SSD, for example `C:\lcms`, so that
  `C:\lcms\lcmsdeconv\gpu_training` exists. Avoid folders that OneDrive synchronises (on many
  Windows 11 PCs, the Desktop and Documents): syncing slows checkpoint writing and can lock files.
- **Disk.** About 10 GB free (PyTorch with CUDA is large).

**Updating from an earlier package.** Unzip the new package **over the old folder** (the same
place, for example `C:\lcms`; let Windows replace the files). The environment `.venv` stays, so
PyTorch is not downloaded again; because the package changed, the setup measures the computer
and runs its check again (a few minutes). Delete `gpu_training\run_stage1.bat` and
`run_stage2.bat` if they are still there: they belong to the earlier two-stage training. The
earlier runs' files in `runs` are not touched.

**The easy way: one double-click.** Double-click `gpu_training\setup_and_train.bat`. The first
time, it sets up (step 1) and then trains (step 2) in the same window; at the end it names the
file to upload (step 3). Double-click it again after a pause, a crash or a reboot: the training
continues where it stopped, and the setup is skipped as long as it still holds
(`gpu_training\setup_done.json` records it; it runs again by itself if the package has changed,
`.venv` or `machine.json` is missing, or PyTorch no longer computes on the GPU). Options are
passed on: `gpu_training\setup_and_train.bat --ram-gb 12 --gpu-memory-gb 12`; `--redo-setup`
runs the whole setup again. The two steps can also be run one at a time:

**1. Set up (10-20 minutes the first time):** `gpu_training\setup_windows.bat`. It creates the
Python environment `.venv`, installs PyTorch for CUDA and this package, checks that PyTorch
computes on the GPU, and measures your machine: training steps of both networks on the GPU
under the 14 GB cap, and how much memory each data worker takes and how many spectra per second
they deliver. It chooses the number of workers that keeps the training within 16 GB of RAM
(`gpu_training\machine.json`), then checks everything end to end in a few minutes: a few
training steps of both networks, their export to ONNX, and whole spectra read with them. It
prints the expected training time; its output is also in `gpu_training\setup.log`.

**2. Train:** `gpu_training\train_all.bat` trains the runs, exports each finished one to ONNX
(checking that the ONNX model computes what the checkpoint does) and reads whole spectra with
it once, then packs the results. It prints the expected time at the start and the time left
after every epoch. Expect several hours for the two runs; setup's measurement gives the
estimate for your computer.

**3. Upload** the file it names at the end: `runs\upload_YYYYMMDD-HHMM.zip`. It holds every
finished run (checkpoint, ONNX model and its `.json` sidecar, training logs and status), with
`machine.json` and `setup.log`.

## Memory and speed

- **GPU memory:** each step takes 4 spectra of 131 072 bins. The ladder is read at up to
  24 576 bins per spectrum, which is the bulk of the memory, so it is scored 16 charges at a
  time and recomputed in the backward pass: the same results in about a third of the memory
  (measured on a processor, about 2.5 GB per step against 7 GB). Setup measures the real peak
  of both networks on your GPU (`gpu_views` in `machine.json`).
- **Batch size:** `--batch-size 8` takes 8 spectra per step. A batch whose memory, scaled from
  setup's measurement, exceeds the GPU cap is refused before the training starts, with the
  largest size that fits. The learning rate stays the recipe's; the networks were set up at 4.
  A run already started at another size cannot be resumed with a new one: finish it, or start
  over with `--restart`.
- **RAM, at most 16 GB:** the training process plus its data workers, counted as Task Manager
  counts memory. If the budget is reached anyway, the run pauses safely and resumes at once
  with two workers fewer (remembered in `machine.json`). For another budget, `--ram-gb 20`.
- **Speed:** the data workers and the GPU work at the same time and the slower one sets the
  pace; setup says which. More workers (a larger RAM budget) help only if the workers are the
  slower one. The workers run at below-normal priority, so the computer stays usable.
- **Sleep:** while training, Windows is kept from going to sleep (the screen may still turn off).
  Pause Windows Update for the day before a long run.

## What the output means

Before the first epoch, each run scores the comb estimator on the validation spectra (a
minute). Then every epoch prints lines such as

```text
epoch 8: charge_acc_given_ion ema 0.903 live 0.907 (comb 0.668) -> keeping live; bands {...}
  charge of z >= 2 ion bins read right (log-SNR / intensity weighted):
    time-of-flight  network 0.895 / 0.931   comb 0.689 / 0.844
    Orbitrap        network 0.850 / 0.917   comb 0.237 / 0.290
  epoch took 21 min; 48 spectra/s; RAM 11.8 GB of 16; GPU 3.9 GB reserved; about 4.1 h to go
```

(illustrative numbers). The number in brackets is the comb estimator, scored the same way on
the same spectra, as the pipeline runs it. The two lines below split the comparison by kind of
spectrum, bins weighted by log-SNR and by intensity; a network worth uploading beats the comb in
all four. The last line gives speed, memory and the time left. The best checkpoint is saved
whenever it improves; logs are in `runs\*.log.jsonl` and `runs\*.console.log`.

## Pausing and resuming

- **Pause:** press **Ctrl+C once** in the training window, or create a file named `PAUSE` in
  the `runs` folder (PAUSE, PAUSE.txt or PAUSE.txt.txt all count). The current step finishes,
  the complete training state is saved, and the program ends. After Ctrl+C, Windows asks
  "Terminate batch job (Y/N)?": the state is already saved, so either answer is safe.
- **Continue:** double-click `gpu_training\setup_and_train.bat` (or run
  `gpu_training\train_all.bat`) again, with the same `--runs` if you chose some. A finished run
  is skipped, an interrupted one continues from the saved epoch and step. The state is saved
  every 10 minutes and after every epoch.
- **Abort:** pressing Ctrl+C twice stops at once. The last saved state is kept.
- **Start over:** `--restart` discards the saved states of the chosen runs.
- **One at a time:** a second `train_all.bat` started while one is running stops at once.

## Linux

```bash
bash gpu_training/setup_and_train.sh   # setup where needed, then training (as the .bat)
bash gpu_training/setup_linux.sh       # or the two steps one at a time
bash gpu_training/train_all.sh
```

## Troubleshooting

- **"Python 3.12 was not found"**: install it as in the quick start, then open a new terminal.
- **"nvidia-smi was not found"** or **"No PyTorch CUDA build worked with this driver"**:
  install or update the NVIDIA driver and run the setup again.
- **"pip could not download or install PyTorch"**: the setup downloads PyTorch from
  `download.pytorch.org`, which a company proxy or firewall may block; it also needs about
  10 GB of free disk, and no other program may be using `.venv`.
- **"CUDA is not available, and this run is set to train on the GPU only"**: PyTorch lost its
  CUDA build. Run `gpu_training\setup_windows.bat --reinstall-torch`.
- **"needs about N GB of GPU memory"**: use the batch size it names, or raise the cap.
- **"the training process and 1 worker already need more than the RAM budget"**: raise the
  budget with `--ram-gb`.
- **`[WinError 1455] The paging file is too small`** while workers start: let Windows manage
  the paging file size, or use fewer workers (`--workers 6`).
- **Re-measure the machine** after changing the limits or hardware: `.venv\Scripts\python
  gpu_training\probe.py --ram-gb 16 --gpu-memory-gb 14`; `.venv\Scripts\python
  gpu_training\check_gpu.py` runs a few steps of both networks on random input.
