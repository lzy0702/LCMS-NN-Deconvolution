# Training the charge network on your GPU

This folder trains **LadderNet**, the charge-assignment network of `lcmsdeconv`, on synthetic
spectra generated on the fly. Nothing needs to be downloaded except Python packages, and no
data leaves your computer. When it is done, you upload one zip file.

The network trains on the **GPU only**: the scripts refuse to start rather than fall back to the
processor. The training spectra are synthesised by data-worker processes on the processor
while the GPU trains; that part cannot run on the GPU. By default everything stays within
**half of your computer's RAM, at least 16 GB and at most 48 GB** (12 GB on a computer of 16 GB)
and **14 GB of GPU memory**, leaving the rest to Windows.

## What is trained

Two networks, one after the other, on the same spectra (the same seed), so that they can be
compared:

- **`window`** (`runs\ladder_v4_window.pt`): the ladder reads a fixed window of 131 072 grid
  bins, a region's whole spectrum; inference cuts a longer spectrum into such windows.
- **`comb`** (`runs\ladder_v4_comb.pt`): the same ladder for a spectrum of any length, read
  whole in one pass. Its layers do not depend on the input's length, it is told which bins were
  measured, and where a rung of a charge ladder falls outside the spectrum it reads a learned
  "not measured" value instead of zeros.

Both use **recipe v4**, the whole-spectrum recipe in which 40 % of the oligonucleotides of the
real-like Orbitrap spectra show a single charge state (so that the network learns that a lone
envelope can be multiply charged, as its isotope spacing says): full size, 20 epochs of 6 000
new spectra each (singly charged ions included), learning rate 0.002, and the **isotope-charge
channels**: the charge the comb's isotope reader reads from resolved isotope spacing, the
strength of that reading, and the **isotope limit**, the highest charge whose isotope spacing
can be resolved at that m/z with the instrument's resolving power. Above the limit the reader
answers a lower charge it can resolve, mostly one at or a little below the limit; the network
uses the limit to tell such readings apart. Untrained, the share of charges of z >= 2 read
right on the Orbitrap spectra of recipe v3 is 0.66 with the limit (0.85 weighted by
intensity), against 0.62 (0.82) without isotope channels and 0.39 (0.42) with the earlier
layout's two channels, whose wrong readings the limit lets the network discount.

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
and runs its check again (10-15 minutes). Delete `gpu_training\run_stage1.bat` and
`run_stage2.bat` if they are still there: they belong to the earlier two-stage training. The
earlier runs' files in `runs` are not touched. This package trains recipe v4 into new files
(`runs\ladder_v4_*`): a recipe v3 run that has not finished is not continued by it, so finish
that one with the earlier package first if you want it.

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
under the 14 GB cap, how much memory each data worker takes, and which GPU speedups make
training faster on your GPU (below). It chooses as many data workers as the processor is given
(all cores but four) where the RAM budget holds them, and measures how many spectra per second
that many deliver together (`gpu_training\machine.json`; a minute or two). Then it checks
everything end to end in a few minutes: a few training steps of both networks (with the
speedups chosen), their export to ONNX, and whole spectra read with them. It prints which side
will set the pace, the data workers or the GPU, and the expected training time; its output is
also in `gpu_training\setup.log`.

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
  24 576 bins per spectrum, which is the bulk of the memory. Scoring the charges 16 at a time
  and recomputing them in the backward pass gives the same results in about a third of the
  memory (measured on a processor, about 2.5 GB per step against 7 GB) for about a third more
  time per step. Setup measures both ways for both networks on your GPU (`gpu_views` in
  `machine.json`), and training scores all charges at once where that is faster, its peak with
  30 % to spare stays under the cap, and the GPU rather than the data workers sets the pace
  (otherwise chunks cost no time); setup prints the choice. Should a run still run out of GPU
  memory that way, it continues from its last saved state 16 charges at a time, and so do later
  runs of that network (`unchunked_failed` in `machine.json`, kept when setup measures again
  with the same GPU, cap and batch size).
- **Batch size:** `--batch-size 8` takes 8 spectra per step. A batch whose memory, scaled from
  setup's measurement, exceeds the GPU cap is refused before the training starts, with the
  largest size that fits. The learning rate stays the recipe's; the networks were set up at 4.
  A run already started at another size cannot be resumed with a new one: finish it, or start
  over with `--restart`.
- **RAM budget, by default half of the computer's RAM:** the training process plus its data
  workers, counted as Task Manager counts memory. The default is half the RAM, at least 16 GB
  and at most 48 GB, and at most the RAM less 4 GB (12 GB on a computer of 16 GB): 16 GB on a
  computer of 32 GB, as before, 32 GB on one of 64 GB, 48 GB on one of 96 GB or more. Setup
  gives the data workers all cores but four (two of eight or fewer) where the budget holds
  them, and measures their speed at that number. Why: the spectra come from the workers, and
  on a processor with performance and efficiency cores each further worker adds less, while
  batches are delivered in order, so a worker that is ahead waits for the slower ones. On a
  computer of 96 GB with 24 cores (8 performance, 16 efficient), the earlier fixed budget of
  16 GB held 13 workers, which trained at 19 spectra/s (5.4 minutes an epoch) where setup had
  expected 30 from 10 workers measured alone and scaled up; with `--ram-gb 48` the 20 workers
  the processor is given trained at 26-27 spectra/s (3.8 minutes an epoch). Whether the
  workers or the GPU set the pace is in every epoch's line (the share of the time waiting for
  data, below). If the budget is reached anyway, the run pauses safely and resumes at once
  with two workers fewer (remembered in `machine.json`). Another budget: `--ram-gb 24`
  (setup_and_train, setup or train_all); another number of workers: `--workers 16`
  (setup_and_train or train_all). Exporting the window network to ONNX (after its run, and in
  setup's check) takes about 9 GB of RAM for a minute or two; no training runs at that time.
- **Speed:** the data workers and the GPU work at the same time and the slower one sets the
  pace. Setup compares the workers' speed with a training step of the GPU on the runs' first
  training batch (with the GPU speedups chosen), and gives the step with every spectrum at the
  active-bin cap, the slowest a step can be, as the bound; it prints which side sets the pace
  and the expected time (`pace` in `machine.json`), and every epoch measures it again (see
  "What the output means"). More workers help only if the workers are the slower one. The
  workers run at below-normal priority, so the computer stays usable.
- **GPU speedups:** four switches change how the GPU rounds its arithmetic, not what it
  computes: `cudnn_benchmark` (cuDNN times its convolution methods once and keeps the fastest),
  `tf32` (matrix products in TensorFloat-32, as cuDNN's convolutions already are),
  `index_select` (the ladder head reads its rows in a way whose backward pass needs no sort)
  and `fused_adamw` (the optimiser updates every weight in one step). Setup measures them on
  your GPU, each in a process of its own, on the first training batch of both networks with
  the chunking training will use: a step with every switch off, then with each one alone (a
  few steps to warm up, then the median of eight timed steps). A switch is kept if its step is
  at least 3 % faster and its loss and gradients, from the same weights on the same batch, agree
  with those of the step without it (the loss within 0.1 %, the gradients pointing the same way
  to a cosine of 0.999 and of the same size within 1 %: TensorFloat-32 rounding stays well
  inside these, a real change of the computation does not). The kept ones are then combined,
  fastest first, each further one only if it makes the step at least 2 % faster again and still
  agrees. A switch that fails in any way is left off. `gpu_speedups` in `machine.json` records
  the choice for each network,
  `speedup_timings` the measured steps, `speedup_checks` why each switch was kept or not, and
  `cuda_kernels` the 15 GPU kernels that take the most time in the window network's step; setup
  prints a summary (also in `setup.log`). Training uses the chosen switches and prints them as
  each run starts (`GPU speedups tf32, index_select`, or `no GPU speedups`). A `machine.json`
  from an earlier setup has no such record: training then uses none, and says that running
  setup again measures them. A paused run may resume with other switches.
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
  epoch took 21.0 min; 48 spectra/s; of the training time 2 % waiting for data, 95 % GPU busy (79 ms a step); RAM 11.8 GB of 16; GPU 3.9 GB reserved, 3.4 GB allocated at most, 0 allocator retries; about 4.1 h to go
```

(illustrative numbers). The number in brackets is the comb estimator, scored the same way on
the same spectra, as the pipeline runs it. The two lines below split the comparison by kind of
spectrum, bins weighted by log-SNR and by intensity; a network worth uploading beats the comb in
all four. The last line gives speed, what set the pace, memory and the time left:

- **waiting for data**: the share of the epoch's training time (validation not counted) the
  training spent waiting for the data workers' next batch. High (more than a few percent)
  means the workers set the pace: more workers would help, where the RAM budget holds fewer
  than the processor is given (`--ram-gb`) or the processor has cores to spare (`--workers`).
- **GPU busy** and **ms a step**: the GPU time of every 25th step, timed on the GPU itself
  (its median), and that time times the number of steps as a share of the training time. Near
  100 % the GPU sets the pace, and more workers would not help.
- **allocated at most** and **allocator retries**: the most GPU memory the tensors took (below
  "reserved", which adds what PyTorch keeps for reuse), and how often in this epoch an
  allocation succeeded only after PyTorch released its reserve. Retries above 0 mean the memory
  is tight at the cap and the steps slow down: raise the cap with `--gpu-memory-gb` if the GPU
  has room, or train with a smaller batch.

The same numbers are in each epoch's record in `runs\*.log.jsonl` (`pace`) and in
`runs\*.status.json`, with the GPU speedups the run used. The best checkpoint is saved whenever
it improves; logs are in `runs\*.log.jsonl` and `runs\*.console.log`.

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

## Measuring your computer for the developers (profile)

To tune training for your computer, both the GPU side and the synthesis of the training
spectra on the processor, the developers need measurements made on it. The profile makes them
and packs them into one small zip file to send back. It trains nothing for good and reads none
of your files.

**Run it:** double-click `gpu_training\profile.bat` (Linux: `bash gpu_training/profile.sh`).
Like `setup_and_train.bat`, it first runs the setup where it no longer holds (after a new
package: the measurement and the check, 10-15 minutes; `.venv` is kept and PyTorch is not
downloaded again), then the profile itself: **about 45-50 minutes**, with a line for each
experiment and the time left. Keep the computer otherwise idle meanwhile (no games, videos or
other programs using the GPU), plugged in; it keeps Windows from going to sleep. Ctrl+C stops
it and still packs what is done.

**Send back** the file it names at the end: `runs\profile_upload_YYYYMMDD-HHMMSS.zip` (a few
MB). It holds `results.json` (one record per experiment: its exact settings, timings, outcome),
`summary.txt` (the short summary the window prints), each experiment's log, the nvidia-smi
samples as CSV, setup's `machine.json` and the end of `setup.log`. The package folder's path is
replaced by `<package>` and your home folder's by `~`; no computer or user name is recorded.
The large intermediate files (about 1.5 GB of training batches) are deleted at the end.

**What it measures**, each experiment in a process of its own, with a timeout; one that fails
is recorded with its error and the others go on:

- *the computer*: the processor and the kind of each of its logical processors (performance
  or efficient core, as Windows reports it), the Windows version, power plan and power mode,
  the memory, the GPU, its driver and PyTorch;
- *the GPU* (both networks at full size, 4 spectra a step under the 14 GB cap, on the runs'
  first training batches): the time of a step with each GPU speedup alone, all four, setup's
  choice and other head chunkings; a step split into its parts (forward, backward, optimiser
  ...); the GPU kernels that take the time; memory peaks and allocator retries over 40
  batches with several allocator settings; nvidia-smi (utilisation, clocks, power,
  temperature) during a steady minute; and, for information, bf16 arithmetic and 8 spectra a
  step;
- *the processor*: the speed of each logical processor alone; Windows' power throttling and
  the workers' lowered priority on one core of each kind; 20, 22 and 23 workers at once, as
  training uses them today, each pinned to a core of its own, opted out of power throttling,
  and through a new loader that gives the next spectrum to whichever worker is free; where the
  time of a spectrum goes;
- *a short real training run* (two epochs of 400 steps, written to a throwaway file) with what
  was found best.

It changes no setting of the computer: priorities, core assignments and power throttling are
set for its own processes only, which end with each experiment.

**Options:** `--only gpu`, `--only cpu` or `--only e2e` (the short training run) run a part;
`--minutes 30` sets the time budget (default 60: an experiment that would not fit is skipped
and recorded as such; the short training run always runs); `--smoke` runs every experiment at
a tiny size, to check in a few minutes that it runs. For example
`gpu_training\profile.bat --only cpu --minutes 20`.

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
- **Re-measure the machine** after changing the limits, the hardware or the NVIDIA driver (the
  GPU speedups are measured again too): `.venv\Scripts\python gpu_training\probe.py` with the
  default limits, or with your own, for example `--ram-gb 24 --gpu-memory-gb 12`;
  `.venv\Scripts\python gpu_training\check_gpu.py` runs a few steps of both networks on random
  input.
- **Training says machine.json "comes from an earlier setup, with a RAM budget of 16 GB"**:
  that setup kept 16 GB whatever the computer. `setup_and_train.bat` measures again by itself
  after a new package; `train_all.bat` alone keeps the old budget until setup has run, or
  `--ram-gb` gives another.
- **A run stops with an error from cuDNN, from `index_select` or from the optimiser** after
  setup chose GPU speedups: set each network's list under `gpu_speedups` in
  `gpu_training\machine.json` to `[]` and start the training again; it resumes without them.
