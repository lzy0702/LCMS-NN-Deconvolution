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

- **`window`** (`runs\ladder_v5_window.pt`): the ladder reads a fixed window of 131 072 grid
  bins, a region's whole spectrum; inference cuts a longer spectrum into such windows.
- **`comb`** (`runs\ladder_v5_comb.pt`): the same ladder for a spectrum of any length, read
  whole in one pass. Its layers do not depend on the input's length, it is told which bins were
  measured, and where a rung of a charge ladder falls outside the spectrum it reads a learned
  "not measured" value instead of zeros.

Both use **recipe v5**, the whole-spectrum recipe in which every oligonucleotide of 0.9 to
5 kDa in the real-like Orbitrap spectra (three in four) shows a single charge state, so that the
network learns that a lone envelope can be multiply charged where its isotope spacing says so
(above 5 kDa the spacing no longer tells z from z + 1 reliably): full size, 20 epochs of 6 000
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
earlier runs' files in `runs` are not touched. This package trains recipe v5 into new files
(`runs\ladder_v5_*`): a run of an earlier recipe (v3 or v4) that has not finished is not
continued by it, so finish that one with the earlier package first if you want it.

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
(one for every core but the two the training process keeps: 22 on a processor of 8
performance and 16 efficient cores) where the RAM budget holds them, measures how many spectra
per second that many deliver together (a minute or two), and then which way of running them
delivers the most while a training step runs on the GPU (a few minutes, below; all in
`gpu_training\machine.json`). Then it checks everything end to end in a few minutes: a few
training steps of both networks (with the speedups and the way of running the workers chosen),
their export to ONNX, and whole spectra read with them. It prints which side will set the pace,
the data workers or the GPU, and the expected training time; its output is also in
`gpu_training\setup.log`.

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
  time per step without GPU speedups. Setup measures both ways for both networks on your GPU
  (`gpu_views` in `machine.json`), and a network trains 16 charges at a time whenever it uses
  GPU speedups (below). Without them it scores all charges at once only where that is faster,
  its peak times 1.5 stays 2 GB under the cap (less the CUDA context's 0.75 GB), and the GPU
  rather than the data workers sets the pace (otherwise chunks cost no time); setup prints the
  choice. Why so much room: setup measures the peak on one batch, but the number of bins whose
  ladders are read varies from batch to batch, and with all charges at once the memory PyTorch
  reserves grows with it. On the RTX 4080 SUPER of the profile of 2026-11-25, all charges at
  once with the index_select head took the reserve from the 8.9 GB setup measured to 13.2 GB of
  the 13.25 the cap leaves, and the steps then took seconds instead of a tenth of a second, with
  no error; with cuDNN's benchmark as well (setup's choice), and with cuDNN's benchmark alone,
  a test gave no result within 15 minutes, so the rule covers every speedup. In chunks of 16
  the same speedups took about 105 ms a step with about 4 GB reserved, the fastest way that
  held.
- **When the GPU runs out of memory or crawls:** should a run scoring all charges at once
  still run out of GPU memory, or its steps slow to seconds with the memory PyTorch reserves
  near the cap (which raises no error: the training's crawl guard watches for it, below), it
  stops with its state saved and continues at once from there 16 charges at a time, with the
  same GPU speedups, and so do later runs of that network (`unchunked_failed` in
  `machine.json`, kept when setup measures again with the same GPU, cap and batch size).
- **Batch size:** `--batch-size 8` takes 8 spectra per step. A batch whose memory, scaled from
  setup's measurement, exceeds the GPU cap is refused before the training starts, with the
  largest size that fits. The learning rate stays the recipe's; the networks were set up at 4.
  A run already started at another size cannot be resumed with a new one: finish it, or start
  over with `--restart`.
- **RAM budget, by default half of the computer's RAM:** the training process plus its data
  workers, counted as Task Manager counts memory. The default is half the RAM, at least 16 GB
  and at most 48 GB, and at most the RAM less 4 GB (12 GB on a computer of 16 GB): 16 GB on a
  computer of 32 GB, as before, 32 GB on one of 64 GB, 48 GB on one of 96 GB or more. Setup
  gives the data workers one logical processor each of every core but the one or two the
  training process keeps for itself (its main thread and the thread that receives the
  batches), where the budget holds them, and measures their speed at that number. Why: the
  spectra come from the workers, and on a processor with performance and efficiency cores each
  further worker adds less, while torch's DataLoader delivers batches in order, so a worker
  that is ahead waits for the slower ones. On a computer of 96 GB with 24 cores (8
  performance, 16 efficient), the earlier fixed budget of 16 GB held 13 workers, which trained
  at 19 spectra/s (5.4 minutes an epoch) where setup had expected 30 from 10 workers measured
  alone and scaled up; with `--ram-gb 48` the 20 workers the processor was given then trained
  at 26-27 spectra/s (3.8 minutes an epoch); it is given 22 now. Whether the workers or the
  GPU set the pace is in every epoch's line (the share of the time waiting for data, below).
  If the budget is reached anyway, the run pauses safely and resumes at once with two workers
  fewer (remembered in `machine.json`). Another budget: `--ram-gb 24` (setup_and_train, setup
  or train_all); another number of workers: `--workers 16` (setup_and_train or train_all).
  Exporting the window network to ONNX (after its run, and in setup's check) takes about 9 GB
  of RAM for a minute or two; no training runs at that time.
- **Speed:** the data workers and the GPU work at the same time and the slower one sets the
  pace. Setup compares the workers' speed with a training step of the GPU on the runs' first
  training batch (with the GPU speedups chosen), and gives the step with every spectrum at the
  active-bin cap, the slowest a step can be, as the bound; it prints which side sets the pace
  and the expected time (`pace` in `machine.json`), and every epoch measures it again (see
  "What the output means"). More workers help only if the workers are the slower one. The
  workers run at below-normal priority, so the computer stays usable (unless setup measured
  normal priority clearly faster, below).
- **How the data workers run:** setup measures, in a few minutes, which way of running them
  delivers the most spectra on your computer, and training uses that one:
  - *as before*: torch's DataLoader, which hands whole batches to the workers in turn and
    delivers them in order, so that the slowest worker sets the pace of all; the workers free
    to run on any core, Windows' power throttling left to Windows, below-normal priority;
  - *the ordered loader*: single spectra to whichever worker is free, the same batches in the
    same order (training is bit for bit the same either way);
  - with the workers bound to the cores the training process does not keep (*pool*) or each
    to a core of its own (*core*), or free as before;
  - with the workers and the training process opted out of Windows' power throttling or not
    (a process that does not opt out is classified by Windows itself, from its priority and
    whether its window is visible among others, and a throttled process may be held to the
    efficient cores at a low clock);
  - at below-normal or normal priority (Windows only).

  All of them run on one set of workers that switch from way to way while they run (they start
  once; the DataLoader is started for its own turns), each way twice in mirrored order (A B C
  ... C B A, so that the computer warming up or slowing down weighs on every way alike), each
  time on the same spectra and while the training process does what it does in training (a
  real training step of the window network on the GPU for every batch). Setup keeps the way as
  before unless another delivers at least 3 % more, beyond what chance can explain (how far
  apart each way's two measurements are tells how much they vary by chance); each further
  change (the ordered loader, the opt-out, binding, each worker on its own core, normal
  priority) must again deliver 3 % more than every way without it. If the computer was busy
  with something else during the measurement (the way as before measured more than 10 % apart
  at the start and at the end), it keeps the way as before; a way whose own two measurements
  are that far apart, or that not every worker could take, is left out. `data_loader` in
  `machine.json` records the choice and why, `data_loader_test` every way's spectra per second
  (both measurements), what a busy worker delivered on each kind of core and how long the
  training process waited for data; setup prints a table (also in `setup.log`), and the time
  estimate uses the spectra per second measured for the way chosen. A `machine.json` from an
  earlier setup has no such record: the workers then run as before, and training says that
  running setup again measures it. Each run prints the way it uses (`data loading: ...`) and
  how the workers draw the spectra's noise (`spectrum draws: compiled ...`, or numpy's and
  why; also `spectrum_draws` in each run's `.status.json`). Should a run stop with an error of
  the new way (`DataWorkerError`), it continues at once from its last saved state as before,
  and later runs do too (remembered in `machine.json`).

  To choose yourself, pass any of `--loader dataloader` or `ordered`, `--cpu-binding none`,
  `pool` or `core`, `--worker-qos system` or `high` and `--worker-priority below_normal` or
  `normal` to `gpu_training\train_all.bat`; the others stay setup's choice. Exactly as before:
  `gpu_training\train_all.bat --loader dataloader --cpu-binding none --worker-qos system
  --worker-priority below_normal`. Setting the environment variable `LCMSDECONV_LOADER` to
  `dataloader` (`set LCMSDECONV_LOADER=dataloader` in the window before starting) forces
  torch's DataLoader whatever was chosen or passed. `.venv\Scripts\python gpu_training\probe.py
  --no-data-test` measures the computer without this test, and training then runs the workers
  as before.
- **GPU speedups:** four switches change how the GPU rounds its arithmetic, not what it
  computes: `cudnn_benchmark` (cuDNN times its convolution methods once and keeps the fastest),
  `tf32` (matrix products in TensorFloat-32, as cuDNN's convolutions already are),
  `index_select` (the ladder head reads its rows in a way whose backward pass needs no sort)
  and `fused_adamw` (the optimiser updates every weight in one step). Setup measures them on
  your GPU, each in a process of its own, on the first training batch of both networks with
  the ladder scoring 16 charges at a time, the only way training uses them (see "GPU memory"):
  a step with every switch off, then with each one alone (a few steps to warm up, then the
  median of eight timed steps). A switch is kept if its step is at least 3 % faster and its
  loss and gradients, from the same weights on the same batch, agree with those of the step
  without it (the loss within 0.1 %, the gradients pointing the same way to a cosine of 0.999
  and of the same size within 1 %: TensorFloat-32 rounding stays well inside these, a real
  change of the computation does not). The kept ones are then combined, fastest first, each
  further one only if it makes the step at least 2 % faster again and still agrees. A switch
  that fails in any way is left off. `gpu_speedups` in `machine.json` records
  the choice for each network, `speedup_timings` the measured steps, `speedup_chunks` the
  chunking they were measured with, `speedup_reserved_gb` the most GPU memory each step
  reserved, `speedup_checks` why each switch was kept or not, and `cuda_kernels` the 15 GPU
  kernels that take the most time in the window network's step; `gpu_free_gb_at_setup` is the
  GPU memory that was free when the measurement began (the display and other programs hold the
  rest). Setup prints a summary (also in `setup.log`), and chooses each network's chunking only
  after the switches. Training uses the chosen switches and prints them as each run starts
  (`GPU speedups tf32, index_select`, or `no GPU speedups`). A `machine.json` from an earlier
  setup that chose switches for all charges at once trains them 16 charges at a time all the
  same; one without the record uses none, and training says that running setup again measures
  them. A paused run may resume with other switches.
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
  epoch took 21.0 min; 48 spectra/s; of the training time 2 % waiting for data, 95 % GPU busy (79 ms a step); spectra made: performance cores 2410 (2.31/s a busy worker, on processors 0,3-7), efficient cores 3590 (1.42/s a busy worker, on processors 8-23), workers busy 97 %; RAM 11.8 GB of 16; GPU 3.9 GB reserved, 3.4 GB allocated at most, 0 allocator retries; about 4.1 h to go
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
- **spectra made**: the epoch's spectra by the kind of core they were made on (performance,
  efficient; one count where the cores are alike), how many a worker made per second while it
  was busy, on which processors, and how busy the workers were. A busy worker making far fewer
  spectra a second in training than setup measured, or the efficient cores making nearly all
  of them, says the workers are slowed in training (by Windows' power throttling, for
  example); "made twice" counts spectra the ordered loader handed to a second worker because
  the first held them too long.
- **allocated at most** and **allocator retries**: the most GPU memory the tensors took (below
  "reserved", which adds what PyTorch keeps for reuse), and how often in this epoch an
  allocation succeeded only after PyTorch released its reserve. Retries above 0 mean the memory
  is tight at the cap and the steps slow down: raise the cap with `--gpu-memory-gb` if the GPU
  has room, or train with a smaller batch.

The same numbers are in each epoch's record in `runs\*.log.jsonl` (`pace`) and in
`runs\*.status.json`, with the GPU speedups the run used. The best checkpoint is saved whenever
it improves; logs are in `runs\*.log.jsonl` and `runs\*.console.log`.

**What to expect.** On an RTX 4080 SUPER each run starts with `scoring 16 charges at a time;
GPU speedups cudnn_benchmark, index_select` (or the speedups setup chose on your GPU), the
`epoch N step M` lines come about 2-2.5 seconds apart (20 steps of about 0.1 s), and the
epoch's last line shows at most about 8 GB reserved. Lines 20 or more seconds apart, with the
reserve near the cap, mean the GPU is crawling.

**The crawl guard.** The steps of a run can slow from a tenth of a second to several seconds,
without any error, once the memory PyTorch reserves comes close to the cap (on a GPU that also
drives the display, Windows then appears to move part of it to the computer's RAM). In the
profile this began with the runs' first batches, so training judges its steps against the step
setup measured on this GPU with every spectrum at the active-bin cap, the slowest a healthy
step is (train_all passes it): when 3 of the last 8 steps each took more than 5 times that and
more than 1 second (the run's first two steps not counted), with the reserve at 90 % of the cap
or more, a run scoring all charges at once stops with `GPU memory crawl (...)`, its state saved
(exit code 5), and train_all continues it at once 16 charges at a time with the same GPU
speedups, as after running out of GPU memory; pressing Ctrl+C meanwhile does not lose this, the
next start goes on in chunks. In the profile's crawling run the guard would have stopped it
after 9 steps. A run that already scores the charges in chunks only prints
`warning: the GPU slowed down: ...` once and goes on; close other programs that use the GPU if
you see it.

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

## Measuring your computer for the developers (profile)" in gpu_training/README.md (up to "## Troubleshooting") with:

## Measuring your computer for the developers (profile)

To tune training for your computer, both the GPU side and the synthesis of the training
spectra on the processor, the developers need measurements made on it. The profile makes them
and packs them into one small zip file to send back. It trains nothing for good and reads none
of your files.

**Run it:** double-click `gpu_training\profile.bat` (Linux: `bash gpu_training/profile.sh`).
Like `setup_and_train.bat`, it first runs the setup where it no longer holds (after a new
package: the measurement and the check, 10-15 minutes; `.venv` is kept and PyTorch is not
downloaded again), then the profile itself: **about 35 minutes, at most an hour**, with a line
for each experiment, how long it may take and the time left. Keep the computer otherwise idle
meanwhile (no games, videos or other programs using the GPU), plugged in; it keeps Windows from
going to sleep. Ctrl+C stops it and still packs what is done.

**Send back** the file it names at the end: `runs\profile_upload_YYYYMMDD-HHMMSS.zip` (a few
MB), whatever its lines say: an experiment marked `pathological`, `timeout` or `skipped` is a
result too. It holds `results.json` (one record per experiment: its exact settings, timings and
outcome, and for one that stalled, the steps and GPU memory measured before it), `summary.txt`
(the short summary the window prints), each experiment's log, the nvidia-smi samples as CSV,
setup's `machine.json` and the end of `setup.log`. The package folder's path is replaced by
`<package>` and your home folder's by `~`; no computer or user name is recorded. The large
intermediate files (about 1.5 GB of training batches) are deleted at the end.

**What it measures**, each experiment in a process of its own, with a time limit; one that
fails is recorded with its error and the others go on. In this order:

- *the computer*: the processor and the kind of each of its logical processors (performance
  or efficient core, as Windows reports it), the Windows version, power plan and power mode,
  the memory, the GPU, its driver and PyTorch;
- *the GPU* (both networks at full size, 4 spectra a step under the 14 GB cap, on the runs'
  first training batches, each configuration starting with the batch that needs the most
  memory): first a step without speedups and the configuration setup chose, which training
  uses; then each GPU speedup alone, all four, and setup's speedups 16 and 32 charges at a
  time and all at once. The step without speedups sets every later limit: a GPU experiment gets
  eight times as long as that one took (at least 90 seconds), and a step twenty times slower
  than its steps (at least 5 seconds) ends the experiment as `pathological`, keeping what was
  measured before it. A configuration that includes the speedups of one that failed this way,
  at the same head chunking, is not measured again;
- *the processor*, with about 15 minutes kept for it whatever the GPU does: training's own
  data loading with training's number of data workers (22 on a processor of 8 performance and
  16 efficient cores) and with two fewer, through torch's DataLoader and through the ordered
  loader, the workers not bound to cores, sharing the cores the training process does not
  keep, or each on a core of its own, with Windows' power throttling left on or opted out of;
  one worker alone on two performance and two efficient cores; Windows' power throttling and
  the workers' lowered priority on one core of each kind;
- *a short real training run* (two epochs of 400 steps, written to a throwaway file) exactly
  as `train_all` trains the window run on this computer: setup's GPU speedups and head
  chunking, its number of data workers and its way of loading the data;
- then, as far as the time allows: a step split into its parts (forward, backward, optimiser
  ...) and the GPU kernels that take the time, for the fastest configuration whose steps were
  steady and whose memory stayed clear of the cap; memory peaks and allocator retries over 40
  batches with several allocator settings; nvidia-smi (utilisation, clocks, power,
  temperature) during a steady minute; where the time of a spectrum goes; and, for
  information, bf16 arithmetic and 8 spectra a step.

It changes no setting of the computer: priorities, core assignments and power throttling are
set for its own processes only, which end with each experiment.

**Options:** `--only gpu`, `--only cpu` or `--only e2e` (the short training run) run a part;
`--minutes 45` sets the time budget (default 60: an experiment that would not fit is skipped
and recorded as such, detail and information first; the processor's experiments keep their
time and the short training run always runs); `--smoke` runs every experiment at a tiny size,
to check in a few minutes that it runs. For example
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
  GPU speedups and the way the data workers run are measured again too; keep the computer
  otherwise idle meanwhile): `.venv\Scripts\python gpu_training\probe.py` with the
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
- **The training crawls** (the `epoch N step M` lines 20 seconds or more apart, the epoch's
  line showing the GPU memory reserved near the cap): a run scoring all charges at once stops
  and continues in chunks by itself (the crawl guard, above). Otherwise close the other
  programs that use the GPU (games, videos, a browser's hardware acceleration), or lower the
  cap, for example `gpu_training\setup_and_train.bat --gpu-memory-gb 12`: a run in chunks needs
  far less.
- **A run stops with an error from the data workers** (`DataWorkerError`) after setup chose
  another way of running them: train_all continues by itself as before (torch's DataLoader)
  and remembers it. To run them as before from the start, use the options of "How the data
  workers run" above, or `set LCMSDECONV_LOADER=dataloader` before starting the training.
- **The computer is sluggish while training**, and setup chose normal priority for the
  workers: `gpu_training\train_all.bat --worker-priority below_normal` gives other programs
  precedence again.
