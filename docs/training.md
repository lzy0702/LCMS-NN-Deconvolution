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
are generated once and kept in memory. A run stopped at the memory budget pauses like Ctrl+C,
and so does a GPU run whose steps crawl with the memory near the cap (the crawl guard, below);
the exit code tells the cases apart (0 finished, 3 paused, 4 memory budget, 5 GPU memory crawl).

`gpu_training/README.md` is the guide for training on your own GPU: `setup_windows.bat` (or
`setup_linux.sh`) installs and measures the machine, `train_all.bat` trains recipe v5 in both
views (below) within the RAM and GPU memory limits, with the GPU speedups setup measured
faster, resumes after any interruption and packs what to send back. Its RAM budget is by
default half the computer's memory, at least 16 GB, at most 48 and at most the memory less 4
GB (`nn.machine.default_ram_budget_gb`; `--ram-gb` sets another). Setup gives the data workers
one logical processor each of every core but the one or two the training process keeps
(`nn.machine.cpu_workers`, `nn.cpus.plan`: 22 of 8 performance and 16 efficient cores) where
the budget holds them, and measures their speed at that number, in full flow after two batches
from each (`probe.workers_run`): with performance and efficiency cores each further worker
adds less, and torch's DataLoader delivers batches in order with two in work per worker, so a
worker that is ahead waits for the slower ones. On the user's 24-core processor (8
performance, 16 efficient cores, 96 GB) the earlier fixed 16 GB held 13 workers, which trained
at 19 spectra/s where 10 workers measured alone and scaled to 13 had promised 30; the 20 the
processor was then given trained at 26-27.

Setup then measures how that many workers deliver the most spectra (`probe.data_loading_test`, a
few minutes), choosing by measurement what `train_charge_net.py` can switch: `--loader
dataloader` (torch's, as before) or `ordered` (`synth.pool`: single items to whichever worker is
free, the same batches), `--cpu-binding none`, `pool` or `core`, `--worker-qos system` or `high`
(the opt-out of Windows' power throttling, Windows only) and `--worker-priority below_normal` or
`normal` (measured on Windows only: elsewhere a worker's nice value, once raised, cannot be
lowered again while it runs). The DataLoader runs today's way first and last, each time with
workers of its own; between them one ordered loader's workers start once and switch from way to
way while they run (`probe.SwitchedFrames` reads a code in shared memory before every item; the
training process binds itself as training does), each way twice in mirrored order. Every window
times the same spectra (whole rounds of the DataLoader, about eight seconds at today's rate),
after a warm-up and with more items queued behind them, while the main process runs a real
training step of the window view on the GPU for every batch, with the runs' head chunking and
speedups (`probe.TrainingLoad`; where that cannot run, the batch copied to the GPU and the main
thread kept busy for the measured step without holding the interpreter lock).
`probe.choose_data_loading` keeps today's way unless another delivers at least 3 % more beyond
the gain's standard error, each further change (in the order of `probe.data_cost`) again 3 %
more than every way that changes less. The standard error comes from how far apart each way's
two windows are (`probe.data_noise`), after a steady drift over the test is taken out: in the
mirrored order the drift moves a way's two windows apart in proportion to the time between them,
so a straight line through the ordered loader's differences against that time is the drift, and
what it leaves is chance. In simulation (nine ways, every one as fast as today's) the rule
picked another way in 1.2 % of tests at 2 % noise per window and 6.6 % at 3 % (the plain 3 %
margin: 27 % and 48 %), and still found a way truly 8 % faster in 85 % and 59 % of tests.
Today's two windows more than 10 % apart mean the computer was busy, and keep today's way; a way
whose own windows are that far apart, or that not every worker took, is left out; a test whose
ordered loader stops delivering is stopped at each window's time limit (`probe._watchdog`) and
keeps today's way. `machine.json` records the choice (`data_loader`) and the measurements
(`data_loader_test`: each way's spectra per second, both windows, a busy worker's rate on each
class of core, the training process's wait), and the workers' rate becomes the chosen way's;
with the ordered loader its shared slots (`nn.machine.ORDERED_GB_PER_WORKER`) count against the
RAM budget. `train_all.py` passes the choice to every run (none, and so today's way, for a
`machine.json` without it), takes `--loader`, `--cpu-binding`, `--worker-qos` and
`--worker-priority` over it, estimates the time from the rate measured for the way used, and
continues a run whose new way fails (`DataWorkerError`) from its last saved state with today's
way, remembered in `machine.json`. `LCMSDECONV_LOADER=dataloader` forces torch's DataLoader
whatever is passed.

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
  0.0769 M^0.445, one to three charge states). The Orbitrap spectra are thresholded as Thermo
  profile spectra are (`threshold_sigmas_range` 3-6 electronic sigmas, so most bins are zero),
  and half of them are recorded at the real runs' resolving power (130 000-185 000 at m/z 200),
  the other half over 35 000-280 000. Off, each new `SynthConfig` option draws nothing, so
  default frames are bit for bit those of earlier versions (`tests/test_recipe_v3.py`).
- **Isotope-charge channels** (`--iso-channels`, `TrainConfig.iso_channels`). Channel 4 is
  the charge the comb's isotope reader reads at each bin, channel 5 the strength of that
  reading (best evidence over the bin's log-SNR squared), channel 6 the isotope limit (see
  "Two views of the ladder" below). The reader is
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
strength. Its accuracy on synthetic spectra depends on how they are rendered. Real-like
oligonucleotide envelopes, measured as on the real runs (`envlib`: per envelope, z >= 2,
intensity-weighted):

| Orbitrap spectra | isotope reader | zero bins near the envelope | peak FWHM, bins |
| --- | --- | --- | --- |
| threshold 0-3 sigmas, 35 000-280 000 | 0.59 | 0.15 | 1.36 |
| threshold 3-6 sigmas, 35 000-280 000 | 0.78 | 0.90 | 1.36 |
| threshold 3-6 sigmas, 130 000-185 000 (v3, half the spectra) | 0.89 | 0.93 | 1.21 |
| the real runs (140 000 set) | 0.90 | 0.86 | 1.13 |

Unthresholded, the channels fire mostly on noise bins: in an earlier draft of the recipe 88 % of
the strong readings were on noise, and the untrained isotope-channel network called nearly
every active noise bin an ion. A network trained there would learn to distrust a reader that is
right nine times in ten on real spectra.

### Two views of the ladder, and the isotope limit

`--view` (`TrainConfig.view`) chooses how LadderNet reads a spectrum; `gpu_training` trains both
on recipe v5 (below) with isotope channels, on the same spectra (seed 0).

- **`window`** (the default, every network before): a fixed window, the training crop;
  inference cuts a longer grid into windows of that length.
- **`comb`**: the same ladder for an input of any length, read whole. The encoder normalises
  each bin over its channels (a LayerNorm, `ChannelNorm`) instead of over the whole length, so
  that a bin's features do not depend on how long the input is; the last input channel says
  which bins are measured (1 on the grid, 0 on padding); every ladder offset is kept, and a
  tooth that falls outside the input or on an unmeasured bin reads a learned vector
  (`e_unmeasured`, zero at the start) instead of the encoder's projection, while the comb prior
  reads 0 there. Only measured bins are active, at most 24 576 of them as in the window view.
  Its training frames are the window view's, drawn the same way and cut to the same 131 072-bin
  crop, plus the measured channel (`SynthConfig.measured_channel`). `ChargePredictor` reads the
  whole grid in one pass at its own length: the grid and at least 64 zero bins, up to a
  multiple of 64 (the encoder reaches 60 bins beyond the grid's end, where training frames had
  zeros), with no windows. The ONNX export keeps the length axis open and is checked against
  PyTorch at 8 192, 65 536 and 91 712 bins. Checkpoints, resume states and ONNX sidecars record
  the view; a resume with the other view is refused, and states and checkpoints written before
  record none and are window views.
- **`--head-chunk N`** scores N charges at a time and recomputes them in the backward pass
  (`torch.utils.checkpoint`): the same outputs (to about 1e-8) for about a third of the memory.
  A full-size step at 4 spectra of 131 072 bins, every spectrum at the active-bin cap, peaked
  on the processor at 6.7 GB (window) and 7.1 GB (comb) above the process's baseline, 2.4-2.5
  GB in chunks of 16 charges, for 30 % more step time. The GPU package measures both on the
  GPU (`gpu_training/probe.py`): `train_all.head_chunk_for` trains 16 at a time a run with GPU
  speedups (below), and scores all charges at once in a run without them only where that is
  faster, its peak times 1.5 (`UNCHUNKED_MARGIN`) stays 2 GB (`UNCHUNKED_HEADROOM_GB`) under
  the cap less the CUDA context, and the GPU rather than the data workers sets the pace.
  `train_all.run_stage` continues a run that runs out of GPU memory unchunked, or that the
  crawl guard stops (exit code 5, below), from its last saved state in chunks, with the same
  speedups.

**The isotope limit** (channel 6, `nn.features.isotope_limit`): at every bin, the highest charge
whose isotope spacing the reader resolves at that m/z with the instrument's resolving power, by
the reader's own rule (spacing at least 2.5 peak widths, a peak at least two bins wide), up to
60; 0 where none is, and on time-of-flight spectra. At 140 000 (m/z 200) on the 2e-5 grid it is
20 at m/z 500, 10 at 1 000 and 5 at 2 000. Above the limit the reader answers a resolvable lower
charge at full strength, mostly at or a little below the limit, rather than half the true one. The scorer of each charge z also reads
whether z is resolvable there (`clamp(zlim - z + 1, 0, 1)`), and the isotope prior counts only
where twice the read charge is still resolvable (`gate = clamp(zlim - 2 zc + 1, 0, 1)`).
Untrained, on 20 Orbitrap validation spectra of the recipe (the training's validation seeds,
both Orbitrap halves; charge of z >= 2 ion bins read right, weighted by log-SNR / intensity):

| estimator | z >= 2 |
| --- | --- |
| comb with isotope charges (the pipeline's) | 0.381 / 0.423 |
| LadderNet, full, untrained, no isotope channels | 0.617 / 0.823 |
| the same with the first layout's two isotope channels | 0.394 / 0.423 |
| the same with three channels, gated | **0.656 / 0.845** |
| the same with three channels, without the gate | 0.394 / 0.423 |

The gate closes 85 % of the strong readings (strength >= 0.3) on those ion bins, of which 69 %
(log-SNR weighted) are wrong; of those it leaves open, 28 % are. (Untrained, the three-channel
network without the gate is the two-channel one: the learned part starts at zero.) Networks of
the first layout (two isotope channels, in_ch 6, as 07cdc43 saved them) still load and read as
they did (`tests/test_ladder_views.py`); every isotope-channel network now records its
`iso_count`.

### Recipe v4

`--recipe v4` is v3 with lone multiply charged oligonucleotides: in the real-like Orbitrap
spectra (half of them), 40 % of the oligonucleotide main compounds show a single charge state,
their impurities with them (`SynthConfig.oligo_single_state_prob`, `V4_SINGLE_STATE` in
`scripts/train_charge_net.py`). In v3's spectra 99.8 % of the envelopes with no other charge
state of their species in view are singly charged background ions, and where the isotope reader
reads a charge of 2 or more on such a lone peak the label is z = 1 for 97.6 % of the signal: a
network trained there learns that a charge without ladder rungs is z = 1, whatever the isotopes
say. In 80 v4 Orbitrap frames a real-like frame holds 0.95 strong (S/N >= 100) lone envelopes
of z >= 2 (v3: 0.01), and at lone peaks that the reader reads as 2 or more, the label is z = 1
for 49 % of the signal (v3: 97.6 %). Only Orbitrap spectra get them: a lone envelope's charge is
in its absolute m/z, which only the isotope-charge channels carry. The option draws nothing
while it is 0, so default and v3 frames, and their resume fingerprints, are unchanged
(`tests/test_recipe_v4.py`).

### Recipe v5: what the GPU package trains

`--recipe v5` is v4 with clean lone envelopes: three in four Orbitrap spectra are real-like
(v4: half), and in those every oligonucleotide main compound of 0.9 to 5 kDa shows a single
charge state, its impurities with it (`oligo_single_state_prob` 1.0 and
`SynthConfig.oligo_single_state_max_mass` 5 000 Da; `V5_SINGLE_STATE_MAX_MASS` and
`V5_REAL_LIKE` in `scripts/train_charge_net.py`). The ceiling is where the isotope reader stops
being right: on lone envelopes of z >= 2 it reads the charge right on 0.83-0.96 of the signal
below 5 kDa, 0.40-0.67 at 5-8 kDa and 0.01-0.52 above 8 kDa (z + 1 or a divisor), and most of
v4's single-state compounds were heavier. At lone peaks the reader reads as k >= 2, v4's label
was k for 0.20 of the signal, another charge for 0.33 and z = 1 for 0.47, so a network had no
use for the reading there; in v5 it is k for 0.36, another charge for 0.07 and z = 1 for 0.57
(in the real-like spectra k 0.47, z = 1 0.44). A higher isotope-width floor in the reader would
not help above 5 kDa: there the reading's doubt is between z and z + 1, which a floor does not
resolve.

Proof-size arms (10 epochs of 600 spectra, batch 4, seed 0, on the processor; the first layout's
two isotope channels, as A1) scored on the four real runs, intensity-weighted shares of the
envelopes read right:

| check | bar | A1 (v3) | A1V4 (v4) | A1V5 (v5) |
| --- | --- | --- | --- | --- |
| lone multiply charged envelopes (15 cases) | - | 7/15 | 6/15 | 11/15 |
| all envelopes of z >= 2 (120) | >= 0.88 | 0.90 | 0.93 | 0.94 |
| with a ladder rung at S/N >= 100 (67) | >= 0.90 | 0.92 | 0.94 | 0.96 |
| without rungs in any run (3) | - | 0.57 | 0.74 | 0.83 |
| species of 5 kDa or more, z >= 2 (46) | - | 0.96 | 0.95 | 0.97 |
| envelopes of z = 1 (8) | >= 0.80 | 1.00 | 0.93 | 0.80 |
| species: paper / SI / traced | 20/20, >= 28/29, 8/8 | 20/20, 28/29, 7/8 | the same | the same |
| reports not confirmed | <= 64 | 73 | 82 | 76 |
| of those the comb does not make, singly charged background | <= 3 | 1 | 2 | 3 |

v5 reads lone envelopes and rungless species better and keeps the laddered ones; it pays on the
few singly charged envelopes (at the bar). None of the three finds 3206.4 of the IP-HILIC NC run,
whose envelope stands nearly alone (the comb finds it): in the lone-envelope test A1V5 reads it
z = 1 at every charge, while it follows 4512.8 from z = 2 to 5. The synthetic Orbitrap
validation share falls (A1 0.853, A1V5 0.760) because v5's validation spectra hold the hard lone
envelopes; the time-of-flight share, on the same spectra in every recipe, is unchanged
(0.866-0.876). The gated three-channel layout the package trains reads 14 of the 15 lone
envelopes at proof size on v3 already (13 on v4), but the full-size networks trained on the GPU
on v3 read only 5 and 4 of them, while reading laddered envelopes better (0.96 of z >= 2,
0.99-1.00 of the species of 5 kDa or more): at full size the network learns the rule that a peak
without rungs is z = 1 more firmly, so the recipe, not the size, has to teach the exception.

The gated layout at the same proof size, on v4 and v5 with two seeds each; all four best networks
come from the eighth of their ten epochs (charge accuracy on the validation spectra 0.909 and
0.902 for v4, 0.910 and 0.898 for v5):

| check | A1G4 (v4) | A1G4s1 (v4, seed 1) | A1G5 (v5) | A1G5s1 (v5, seed 1) |
| --- | --- | --- | --- | --- |
| lone multiply charged envelopes (15 cases) | 13/15 | 15/15 | 15/15 | 13/15 |
| all envelopes of z >= 2 (120) | 0.93 | 0.82 | 0.88 | 0.89 |
| with a ladder rung at S/N >= 100 (67) | 0.91 | 0.76 | 0.84 | 0.86 |
| species of 5 kDa or more, z >= 2 (46) | 0.87 | 0.58 | 0.73 | 0.78 |
| envelopes of z = 1 (8) | 0.86 | 0.79 | 0.83 | 0.85 |
| reports not confirmed | 84 | 72 | 81 | 86 |
| of those the comb does not make, singly charged background | 6 | 4 | 3 | 9 |
| real-like synthetic envelopes of z >= 2 (default spectra, the runs' charges) | 0.65 | 0.54 | 0.71 | 0.63 |
| envelopes of z >= 2 read mostly as z = 1 (of 120) | 3 | 11 | 9 | 5 |
| a lone 6011.5 Da envelope at z = 3, 4, 5, 6 read as | 4, 4, -, - | 1, 1, 1, 1 | 1, 1, 1, 1 | -, -, -, - |

(-: no usable charge.) Which recipe reads more of the real envelopes turns with the seed: at
seed 0 v4 reads 0.93 of the envelopes of z >= 2 and v5 0.88, at seed 1 v4 0.82 and v5 0.89; of
the species of 5 kDa or more v4 0.87 and 0.58, v5 0.73 and 0.78. On the table's four
real-envelope shares v5's two runs differ by 0.01-0.05, v4's by 0.07-0.29; the singly charged
background differs more between v5's runs (3 and 9; v4 6 and 4), the lone envelopes by two cases
in both (v4 13 and 15, v5 15 and 13). Of the checks in the table, only the real-like synthetic
envelopes rank the recipes the same way at both seeds (v5 0.71 and 0.63, v4 0.65 and 0.54). The
envelopes lost to z = 1 are states of the 5.1-7.1 kDa species, mostly their z = 3 (m/z
1 710-2 380), which these runs show at z = 3 and 4 (z = 5 at 2.5 % of the stronger state or less,
6 % for n+1). How many a network loses varies between the two runs of each recipe (v4 3 and 11,
v5 9 and 5), so the proof arms cannot rank v4 against v5 on the laddered envelopes. The lone
6011.5 Da envelope does not predict the loss either: the two networks above that read it as
z = 1 at every charge lose the most (11 and 9), but the full-size v3 networks also read it as
z = 1 at every charge and read none of those z = 3 states as z = 1 (0.99-1.00 of the species of
5 kDa or more, above). What they miss are the lone envelopes, which v5 is for.

The full-size v5 networks trained on the GPU read them. The window view (GW5) and the comb view
(GC5), 20 epochs each, the best from the 15th and the last (charge accuracy 0.969 and 0.967 on
v5's validation spectra, which hold the hard lone envelopes and so do not compare with v3's),
beside the v3 pair (GW3, GC3), the proof-size A1G5 and the comb, all scored by the same code:

| check | bar | comb | GW3 (v3) | GC3 (v3) | A1G5 (v5, proof size) | GW5 (v5) | GC5 (v5) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lone multiply charged envelopes (15 cases) | - | 15/15 | 5/15 | 4/15 | 15/15 | 14/15 | 11/15 |
| a lone 4512.8 Da envelope at z = 2, 3, 4, 5, 6 read as | - | 2, 3, 4, 5, 6 | 1, 1, 1, 1, 1 | 1, 1, 1, 1, 1 | 2, 3, 4, 5, 6 | 2, 3, 4, 5, 4 | 2, 3, 4, 5, 1 |
| a lone 6011.5 Da envelope at z = 3, 4, 5, 6 read as | - | 3, 4, 6, 7 | 1, 1, 1, 1 | 1, 1, 1, 1 | 1, 1, 1, 1 | 3, 4, 4, 4 | 3, 1, 3, 1 |
| all envelopes of z >= 2 (120) | >= 0.88 | 0.90 | 0.96 | 0.96 | 0.88 | 0.98 | 0.98 |
| with a ladder rung at S/N >= 100 (67) | >= 0.90 | 0.87 | 0.96 | 0.99 | 0.84 | 0.99 | 0.98 |
| without rungs in any run (3) | - | 0.93 | 0.83 | 0.39 | 0.93 | 0.93 | 0.64 |
| species of 5 kDa or more, z >= 2 (46) | - | 0.84 | 0.99 | 1.00 | 0.73 | 1.00 | 0.99 |
| envelopes of z = 1 (8) | >= 0.80 | 0.71 | 0.89 | 1.00 | 0.83 | 0.86 | 0.82 |
| envelopes of z >= 2 read mostly as z = 1 (of 120) | - | 0 | 3 | 4 | 9 | 2 | 2 |
| species: paper / SI / traced | 20/20, >= 28/29, 8/8 | 20/20, 29/29, 8/8 | 20/20, 28/29, 7/8 | the same | the same | the same | the same |
| reports not confirmed | <= 64 | 58 | 92 | 91 | 81 | 102 | 98 |
| of those the comb does not make, singly charged background | <= 3 | - | 4 | 4 | 3 | 9 | 6 |
| real-like synthetic envelopes of z >= 2 (default spectra, the runs' charges) | >= 0.85 | 0.54 | 0.83 | 0.82 | 0.71 | 0.82 | 0.87 |
| synthetic envelopes of z >= 2 (default spectra and charges) | - | 0.39 | 0.92 | 0.93 | 0.83 | 0.92 | 0.92 |

At full size v5 reads the lone envelopes the v3 networks lost (14 and 11 of 15 against 5 and 4)
and keeps the laddered ones (0.98 of the envelopes of z >= 2, 0.99-1.00 of the species of 5 kDa
or more): taught the exception, the full-size networks learn it, the window view nearly fully,
the comb view in part (it reads 4512.8 Da at z = 6, alone and with z = 7 beside it, and 1884.5 Da
at z = 3 and 4 as z = 1). The comb view reads the real-like synthetic envelopes best (0.87
against 0.82-0.83, the only network at the bar, on 203 envelopes in 29 spectra: a narrow
margin). The cost is in the singly charged envelopes and the reports: 0.86 and 0.82 of the
envelopes of z = 1 (v3 0.89 and 1.00), 102 and 98 reports not confirmed (v3 92 and 91), among
the network-only ones 9 and 6 made of singly charged background (v3 4 and 4), all small: the same
column-wash species of 0.4 % of the run's largest in both, the rest 0.03 % or less. 3206.4 Da of
the IP-HILIC NC run stays missed.

The offline benchmark (`bench/README.md`) scored the comb, GW5 and GC5 again on a Windows laptop,
the networks on its GPU through DirectML, with this version's pipeline (the Confidential sample 1
rows below say what changed in it). Every row of the table came out the same except the reports
not confirmed, which that pipeline changes: 60 for the comb, 98 for GW5 and 94 for GC5 against a
bar of 66 (the comb's 60 plus a tenth). The comb's and GW5's are what this computer measures
with it, and GC5's four runs repeated here on the processor give the same 94. On a test spectrum
DirectML read the processor's top charge on every ion bin, its probabilities within 0.0002, and
the four runs' reports were the same species on both computers, but for one trace of GC5's (under
0.01 % of the run's largest) found at another retention time.

The networks were also run on Confidential sample 1: eleven ion-pairing reversed-phase runs from a
Waters RDa time-of-flight instrument (resolving power about 15 000) of a fully phosphorothioated
DNA (C252H328N72O138P24S24, base composition T13 C4 G6) and forms of it, cut to the gradient
(injection front and column wash left out) and processed with the time-of-flight oligonucleotide
method at that resolving power, the ion-pairing amine's N-oxide adduct included.
There is no truth for them. The yardsticks are a non-negative fit of the raw spectra over a 0.2 Da
grid of mass offsets around the runs' main compound (its isotope pattern at each offset, at the
runs' main charges, which hold about 80 % of the forms' signal), read at each known form's offset
in its retention window; and the share of the signal in re-readings, species at 1.25 times or
more, or 0.8 times or less, the mass of a stronger co-eluting species that is not itself one
(judged strongest first; a lower bound: re-readings between 0.8 and 1.25 times are not counted).
Medians over the runs (the deviations: over the 23 forms):

| Confidential sample 1 (11 runs) | comb | A1G5 (v5, proof size) | GW5 (v5) | GC5 (v5) |
| --- | --- | --- | --- | --- |
| re-read share of the signal (range over the runs) | 28 % (16-40) | 1.8 % (0-9) | 0.9 % (0-8) | 0.9 % (0-7) |
| species per run | 139 | 19 | 15 | 14 |
| the known forms' share of the signal | 55 % | 68 % | 70 % | 73 % |
| known forms against the fit: median deviation (points of the run's largest; 23 forms) | 2.7 | 1.4 | 1.7 | 2.3 |
| of the 23, closer to the fit than the comb | - | 11 | 16 | 14 |
| with this version's pipeline: re-read share | 12 % (2-19) | - | 0.4 % (0-3) | 0.4 % (0-4) |
| with this version's pipeline: species per run | 124 | - | 13 | 11 |
| with this version's pipeline: the known forms' share | 68 % | - | 72 % | 75 % |
| with this version's pipeline: median deviation from the fit | 0.95 | - | 2.0 | 1.7 |
| with this version's pipeline: of the 23, closer to the fit than the comb | - | - | 8 | 9 |

("This version's pipeline": harmonic readings of any ratio dropped, the adduct dedupe's hold and
the bundled RDa method with its lock-mass correction and an analysis window, `docs/usage.md`; the
other rows are the pipeline before them.) On time-of-flight data the networks' charge readings
leave almost no re-readings, and their reports hold a seventh to a tenth of the comb's species,
before this version's rule against re-readings and after it (which takes the comb's share from 28
to 12 %). Before it, their amounts of the known forms also agreed with the fit better than the
comb's by the median (GW5 and GC5 closer on 16 and 14 of the 23 forms, A1G5 on 11); the largest
deviation then was the pipeline's in every predictor, a major form read as the +DIPEA adduct of a
spurious lighter species. The adduct dedupe's hold cures that misreading: with it the comb reads
that form as the fit does and agrees with the fit best overall (median 0.95 points, worst 5). The networks, within a median of 1.7-2.0 points, read the same form 16-18
points high and the run's next form 9-12 points high, because they put about a third of the run's
largest form into TPAO adduct forms (their base fraction 0.60-0.63 against the comb's 0.96), which
lifts every other form of that run.

On the laptop of the offline benchmark the rows of this version's pipeline came out the same but
three: the comb's species per run (125 against 124 by the median), GW5's median deviation from
the fit (2.2 against 2.0 points) and the forms GW5 reads closer to the fit than the comb (6
against 8 of the 23). One run repeated here with the comb at two and at four threads gives this
computer's result to the bit, and the laptop's has one species fewer: the differences are the
two computers' arithmetic (for the networks DirectML's too), not the code or the number of
threads. For the comb they are species near the reporting threshold (1-5 in 8 of the 11 runs,
none above 0.08 % of the run's largest). GC5 reports the same species in every run. GW5 does in
nine; in the tenth the laptop has one of 0.005 % fewer, and in the eleventh GW5 divides a cluster
of co-eluting forms differently (a form of 7 % of the run's largest joined to its neighbour, the
run's largest 6.5 % lower): its reading of that cluster is the least stable result of the three.

`gpu_training` trains v5 (`train_all.RECIPE`): `runs/ladder_v5_window.pt` and
`runs/ladder_v5_comb.pt`, checked by `check_model.py` on a v5 Orbitrap spectrum. Setup measures
the data workers on v5's spectra and records the recipe in `machine.json`; with a `machine.json`
measured for another recipe, train_all gives no time estimate and asks for setup to run again.

### GPU speedups and the pace of an epoch

A training step reads nothing back from the GPU but the active bins' positions (LadderNet's
`nonzero`, which sizes the head). The batches are copied from pinned memory without waiting;
the epoch's running loss is summed on the device in float64 and read only at log steps, at the
epoch's end and when the state is saved, so the printed losses, the history and the resume
state carry the values the host sum gave; the focal loss chooses between its two normalisations
on the device (`torch.where`, the same value either way); the moving average updates all
parameters in two launches (`torch._foreach_mul_` / `_foreach_add_`, the same arithmetic) and
copies only the buffers saved with the model (running statistics, where a module keeps them;
LadderNet's offset tables, comb counts and charge numbers are rebuilt with it and never change).

`--gpu-speedups LIST` (`TrainConfig.gpu_speedups`, `nn.train.GPU_SPEEDUPS`) turns on speedups
that change the rounding of the computation, not what is computed, on CUDA only:

- `cudnn_benchmark`: cuDNN times its convolution algorithms on the first step and keeps the
  fastest (training frames all have one length);
- `tf32`: matrix products in TensorFloat-32 (cuDNN's convolutions use it by default);
- `index_select`: LadderNet's sparse head reads the ladder rows with `torch.index_select` and
  int32 indices (`LadderNet.head_gather`), whose CUDA backward pass adds into the rows
  (`index_add_`) instead of sorting the indices for `index_put_`; the same outputs, gradients
  equal up to the order of summation, in the chunked head as in the whole one. It is a plain
  attribute: checkpoints, state dicts and the dense ONNX export do not change;
- `fused_adamw`: AdamW updates every parameter in one fused kernel.

They are recorded in the checkpoint and the status file, and are no part of the resume check: a
run may resume with others (the optimiser state is moved to the implementation the new session
uses). `gpu_training/probe.py` measures each on the user's GPU, each configuration in a process
of its own (cuDNN keeps the algorithm it first chose for a convolution, with or without its
benchmark, for the rest of a process), on the runs' first training batch in each view with the
head scoring 16 charges at a time (`train_all.HEAD_CHUNK`, recorded as `speedup_chunks`), the
only way runs use them: kept if a step with it alone is at least 3 % faster than one without
any and its loss and gradients from the same weights agree (relative loss difference at most
1e-3, gradient cosine at least 0.999, gradient norms within 1 %), then combined greedily,
fastest first, each further one only if it makes the step 2 % faster again
(`probe.choose_speedups`). Each step's peak reserve is recorded too (`speedup_reserved_gb`, and
the GPU memory free when setup began, `gpu_free_gb_at_setup`). Only then does setup choose each
view's head chunking (`head_chunk_for`, with the speedups chosen), so that its own data-loading
test trains the window view as its runs will. train_all passes each run its view's choice; a
`machine.json` without one gives none.

The user's profile of 2026-11-25 (the RTX 4080 SUPER of 16 GB, driving the display under
Windows' WDDM, a cap of 14 GB) showed why speedups and all charges at once do not mix. Setup
had measured all charges at once faster at the active-bin cap (window view 149 against 182 ms a
step, 8.87 GB reserved) and then cuDNN's benchmark with the `index_select` head faster on the
runs' first batch (81 against 125 ms), so both views would have trained that way. On the runs'
batches, whose active rows vary from 28 000 to 98 000, the unchunked head's largest tensors vary
with them: with the `index_select` head (alone, or with the other speedups) the caching
allocator's reserve climbed to 13.2 of the 13.25 GB the cap left for tensors, 1.48 times what
setup had measured on one batch (without speedups 11.45 GB, 1.29 times), and the steps then
took seconds (a 90th percentile of 21 s with all four speedups; setup's own choice, and cuDNN's
benchmark alone, gave no result within 900 s in the window view) with no out-of-memory error
and no allocator retry, so the out-of-memory fallback could not fire. The rule therefore covers
every speedup, not the `index_select` head alone. In chunks of 16 the same speedups took about 105 ms a step with about
4 GB reserved, the fastest configuration that held; a short real training in chunks of 32 ran
at 109 ms a step, 98 % GPU busy, 5 % of the time waiting for data. Hence the rule above, and
the margin of 1.5 with 2 GB to spare for a run without speedups. The four 900 s timeouts used
the profile's whole hour, so none of its processor experiments ran.

**The crawl guard** (`nn.train.crawling`) is the runtime defence against a crawl that raises no
error. On CUDA every training step is timed from the arrival of its batch to its end, and after
it the allocator's reserve is read (`torch.cuda.memory_reserved`). The pace a step is judged
against is the step setup measured on this GPU for this view and chunking with every spectrum
at the active-bin cap, the slowest a healthy step is (`train_all.cap_step_seconds`, passed as
`--expected-step-seconds`, `TrainConfig.expected_step_seconds`): the profile's crawls began
with the runs' first batches, so a run's own early steps are no measure. A run crawls when 3 of
its last 8 steps (its first two not counted) each took more than 5 times the pace and more than
1 s, while the reserve is at least 90 % of the cap on tensors (the GPU memory cap less the
context). The profile's crawling run took 1.37, 1.80, 16.7, 1.88, 0.67, 0.35, 0.08 and 0.10 s
on the runs' eight first batches against setup's 0.149 s, four of any eight steps over 1 s: the
guard stops it after 9 steps, and a run taking 15 s a step from its start after 5; the same
batches without speedups (0.10-0.15 s) and one or two stalls in eight steps never trip it.
Without setup's step (a run started by hand) the pace is the median of the session's steps 5
to 20, and a run crawling from its start is then taken for its pace. A LadderNet whose head
scores all charges at once waits for the GPU (`torch.cuda.synchronize`) at the end of every
step, so that each time is the step's own, and stops on a crawl through the pause path with the
reason `gpu_memory_crawl` (decided before a pause asked for in the same step), which
`train_charge_net.py` turns into exit code 5; `train_all.run_stage` then resumes it with
`--head-chunk 16` and the same speedups and records `unchunked_failed` for the view, also when
Ctrl+C stops train_all meanwhile. Any other CUDA run is timed without that wait (LadderNet's
`nonzero` keeps it within a step of the GPU) and only warns once: chunks would not help it.

Every epoch records what set its pace, without waiting for the GPU at any step (the crawl
guard's wait aside): the share of the training time spent waiting for the data loader, and on
CUDA the median GPU time of every 25th step (CUDA events, read at the epoch's end), that time
times the steps as a share of the
training time, the peak reserved and allocated GPU memory and the allocator's retries in the
epoch; with data workers also what they made on each class of core (`pace.workers`: items, a
busy worker's items a second, the processors, how busy the workers were, items made twice by
the ordered loader), so that a training whose workers run slower than setup measured shows
where. They are in the pace line, in the epoch's record in `<out>.log.jsonl` (`pace`) and in
`<out>.status.json`; `gpu_training/README.md` says how to read them. Setup predicts the same
from its measurements (`train_all.training_hours`, `pace` in `machine.json`): the data workers'
speed at the number training uses (under the data loading chosen, as setup's test measured it
under a training-like load) against a GPU step on the runs' first training batch with
the speedups chosen (`speedup_timings`), the slower setting the pace, and the step with every
spectrum at the active-bin cap (`gpu_views`) as the bound on the time.

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

Nothing is downloaded at run time: the model ships inside the package, as do the PSI-MS and Units
of Measurement Ontology vocabularies used to read and write mzML (`src/lcmsdeconv/data/cv/README.md`), and
ONNX Runtime's telemetry is off unless `ORT_DISABLE_TELEMETRY=0` is set: lcmsdeconv sets the
variable to 1 when it is imported, unless it is set already, which switches off the Linux build's
telemetry, and calls `onnxruntime.disable_telemetry_events()` when it loads ONNX Runtime, which
switches off the Windows build's (that build does not read the variable). To install on a
machine without internet, build a wheelhouse on a connected machine and copy it across:

```bash
pip download lcmsdeconv -d wheelhouse            # on a connected machine
pip install --no-index --find-links wheelhouse lcmsdeconv   # on the target
```
