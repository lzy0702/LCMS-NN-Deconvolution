"""Training loop for the charge network (CPU proof runs and GPU full runs)."""

from __future__ import annotations

import copy
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrainConfig:
    model_size: str = "proof"
    z_max: int = 100
    epochs: int = 3
    train_len: int = 2000
    val_len: int = 200
    batch_size: int = 4
    num_workers: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    heat_weight: float = 0.2
    ema_decay: float = 0.99
    grad_clip: float = 5.0
    base_seed: int = 0
    device: str = "auto"
    out: str = "models/charge_unet_proof.pt"
    max_seconds: float | None = None
    log_every: int = 20
    threads: int | None = None
    #: select the saved checkpoint by validation charge accuracy (``"accuracy"``) rather than
    #: by validation loss: a loss can improve while the model drifts toward a constant answer
    select_by: str = "accuracy"
    #: per-charge comb evidence channels added to the four base input features
    evidence_channels: int = 0
    #: add that evidence straight to the charge logits (see ChargeUNet)
    comb_prior: bool = False
    #: "unet" (ChargeUNet) or "ladder" (LadderNet)
    arch: str = "unet"
    #: singly charged ions in the training data: "keep", "drop" or "ignore" (see SynthConfig)
    singly_charged: str = "keep"
    #: draw new frames every epoch (seeds advance by train_len) instead of repeating them
    fresh_each_epoch: bool = False
    #: keep generated frames in memory (CPU runs that repeat a small pool)
    cache_frames: bool = False
    #: checkpoint to start from (same architecture and size), e.g. a previous stage
    init: str | None = None
    #: continue from ``<out>.resume.pt`` (written every ``save_every_minutes``, at the end of
    #: every epoch, and when the run is paused with Ctrl+C or a file named PAUSE next to ``out``)
    resume: bool = False
    #: start afresh even though ``<out>.resume.pt`` exists (it is overwritten); without this
    #: or ``resume`` such a run refuses to start rather than overwrite a paused run's state
    restart: bool = False
    save_every_minutes: float = 10.0
    #: cap on this process's GPU memory (GB, CUDA context included); None leaves PyTorch's
    #: default, the whole card
    max_gpu_memory_gb: float | None = None
    #: cap on the memory of this process and its data workers together (GB); when it is
    #: exceeded the run pauses, saving its state, so that it can resume with fewer workers
    max_ram_gb: float | None = None
    #: remove PAUSE files left over at the start; gpu_training/train_all.py clears them itself
    #: and turns this off, so that a pause asked for while a stage starts up is not lost
    clear_pause_files: bool = True
    #: LadderNet: grid bins read at once, in training (every frame's crop) and at inference. A
    #: window of 131 072 bins holds a whole region's grid and the ladder's far rungs (z = 1 <-> 2
    #: is 34 657 bins); the active-bin cap scales with it (6 144 per 32 768 bins)
    window: int = 32768
    #: LadderNet: three more input channels, the charge read from resolved isotope spacing, its
    #: strength and the isotope limit (``lcmsdeconv.nn.features.isotope_channels``), computed
    #: over the whole spectrum of Orbitrap and FT-ICR frames (empty on time-of-flight ones)
    iso_channels: bool = False
    #: LadderNet: "window" (a fixed window, read at inference window by window) or "comb" (any
    #: length, the whole grid in one pass at inference; the frames get the measured channel and
    #: are otherwise the window view's, crops of ``window`` bins included)
    view: str = "window"
    #: LadderNet: score this many charges at a time and recompute them in the backward pass
    #: (the same outputs in a fraction of the memory; None: all at once)
    head_chunk: int | None = None
    #: seconds of the training step setup measured on this GPU for this view and head chunking
    #: with every spectrum at the active-bin cap (gpu_training/train_all.py passes it): the
    #: crawl guard's pace (crawling); None: the session's own steps 5 to 20
    expected_step_seconds: float | None = None
    #: GPU speedups to use (names of GPU_SPEEDUPS): applied on CUDA only, recorded with the run,
    #: and free to differ when a run resumes
    gpu_speedups: tuple = ()
    #: how the data workers' spectra reach the training: "dataloader" (torch's DataLoader, which
    #: hands whole batches to its workers in turn, so that the slowest worker sets the pace) or
    #: "ordered" (lcmsdeconv.synth.pool: single items to whichever worker is free, the same
    #: batches); setup measures which delivers more (machine.json). LCMSDECONV_LOADER overrides
    #: it. The settings from here on change where and how spectra are made, never which: they
    #: are free to differ when a run resumes
    loader: str = "dataloader"
    #: the ordered loader's window: items in work or waiting at most (None: five per worker)
    loader_window: int | None = None
    #: where the data workers run (lcmsdeconv.nn.cpus): "none" (not bound, as before), "pool"
    #: (all of them on the cores the training process does not keep) or "core" (each on its own
    #: core; with the ordered loader only, the DataLoader gets "pool")
    cpu_binding: str = "none"
    #: the data workers' priority: "below_normal" (as before: the computer's other programs come
    #: first) or "normal"
    worker_priority: str = "below_normal"
    #: Windows' power throttling of the data workers and the training process: "system" (as
    #: before: Windows' heuristics decide, and they may slow windowless processes of lowered
    #: priority) or "high" (opted out of EcoQoS: full speed whatever the window state)
    worker_qos: str = "system"


#: speedups of GPU training (``TrainConfig.gpu_speedups``, ``--gpu-speedups``). Each changes how
#: the numbers are computed, not what is computed: the results differ by rounding only, so each
#: is used only where setup measured it faster on that GPU with the same loss and gradients
#: (gpu_training/probe.py):
#:
#: - ``cudnn_benchmark``: cuDNN times its convolution algorithms once and keeps the fastest
#:   (training frames all have the same length);
#: - ``tf32``: matrix products in TensorFloat-32 (cuDNN convolutions use it by default);
#: - ``index_select``: LadderNet's head reads the ladder rows with ``torch.index_select``
#:   (``LadderNet.head_gather``), whose backward pass adds into the rows directly instead of
#:   sorting the indices first;
#: - ``fused_adamw``: the optimiser updates all parameters in one fused kernel.
GPU_SPEEDUPS = ("cudnn_benchmark", "tf32", "index_select", "fused_adamw")


def parse_gpu_speedups(text: str | None) -> tuple[str, ...]:
    """``"tf32,index_select"`` -> ``("tf32", "index_select")`` (in GPU_SPEEDUPS order); "none" or
    "" -> (). A ValueError names the known speedups for anything else."""
    names = {t.strip() for t in (text or "").split(",")} - {"", "none"}
    unknown = sorted(names - set(GPU_SPEEDUPS))
    if unknown:
        raise ValueError(f"unknown GPU speedup {', '.join(unknown)}: choose from "
                         f"{', '.join(GPU_SPEEDUPS)}, or none")
    return tuple(s for s in GPU_SPEEDUPS if s in names)


def set_gpu_flags(speedups) -> tuple:
    """Set the process-wide switches of the speedups ``cudnn_benchmark`` and ``tf32`` (on exactly
    when named in ``speedups``); returns what they were, for :func:`restore_gpu_flags`."""
    import torch

    before = (torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.benchmark = "cudnn_benchmark" in speedups
    torch.backends.cuda.matmul.allow_tf32 = "tf32" in speedups
    return before


def restore_gpu_flags(before: tuple) -> None:
    import torch

    torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32 = before


def _active_speedups(config, device: str, model) -> tuple[str, ...]:
    """The GPU speedups this run uses: those ``config`` asks for, on CUDA only, index_select only
    for a LadderNet. Says which, and which are ignored."""
    asked = parse_gpu_speedups(",".join(config.gpu_speedups or ()))
    if not asked:
        return ()
    if not device.startswith("cuda"):
        print(f"GPU speedups ignored ({', '.join(asked)}): this run trains on {device}")
        return ()
    active = tuple(s for s in asked if s != "index_select" or hasattr(model, "head_gather"))
    if len(active) < len(asked):
        print("GPU speedup index_select ignored: it is LadderNet's")
    print(f"GPU speedups: {', '.join(active)}")
    return active


def _use_fused(opt, fused: bool) -> None:
    """Make AdamW ``opt``, whose state was just loaded, update fused or not as this run asks (the
    loaded state brings the choice of the run that saved it). Each parameter's step count moves
    to where that implementation keeps it: on the parameter's device when fused, on the
    processor otherwise."""
    import torch

    for group in opt.param_groups:
        group["fused"] = True if fused else None
        if fused:
            group["foreach"] = None
        for p in group["params"]:
            st = opt.state.get(p) or {}
            if torch.is_tensor(st.get("step")):
                st["step"] = (st["step"].to(device=p.device, dtype=torch.float32) if fused
                              else st["step"].to("cpu"))


def _device(name: str):
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _batch_dict(sample, device):
    x, topk_z, topk_w, heat = sample
    # on CUDA the copies are queued without waiting: the loaders pin their batches in memory
    # (make_dataloaders), so the GPU reads them while the processor goes on
    nb = str(device).startswith("cuda")
    return {
        "features": x.to(device, non_blocking=nb),
        "topk_z": topk_z.to(device, non_blocking=nb),
        "topk_w": topk_w.to(device, non_blocking=nb),
        "heat": heat.to(device, non_blocking=nb),
    }


class EMA:
    """Exponential moving average of a model's parameters, ``shadow``.

    Buffers saved with the model (running statistics, where a module keeps them) are copied from
    it at every update. Buffers it does not save (LadderNet's ladder offsets, comb counts,
    charge numbers and tooth masks) are rebuilt with the model from its settings and never
    change, so they are left as the copy made them."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._params = list(self.shadow.parameters())
        saved = set(model.state_dict())
        self._buffers = [(name, b) for name, b in self.shadow.named_buffers() if name in saved]

    def update(self, model):
        import torch

        with torch.no_grad():
            live = list(model.parameters())
            if self._params and hasattr(torch, "_foreach_mul_") and hasattr(torch, "_foreach_add_"):
                # the same arithmetic as the loop below, in two launches for all parameters
                # instead of two per parameter
                torch._foreach_mul_(self._params, self.decay)
                torch._foreach_add_(self._params, live, alpha=1 - self.decay)
            else:
                for s, m in zip(self._params, live):
                    s.mul_(self.decay).add_(m, alpha=1 - self.decay)
            for name, s in self._buffers:
                s.copy_(model.get_buffer(name))


class _RunningLoss:
    """The sum of the epoch's training losses so far, kept on the device in float64: adding a
    step's loss does not wait for the GPU, only reading the sum does (at log steps, at the
    epoch's end and when the state is saved). The same float64 sum the host kept before."""

    def __init__(self, device: str, total: float = 0.0, n: int = 0):
        import torch

        # Apple's GPUs have no float64: the sum stays on the processor there
        where = "cpu" if str(device).startswith("mps") else device
        self.sum = torch.tensor(float(total), dtype=torch.float64, device=where)
        self.n = int(n)

    def add(self, loss) -> None:
        self.sum += loss.detach().to(self.sum.device)
        self.n += 1

    def reset(self) -> None:
        self.sum.zero_()
        self.n = 0

    def state(self) -> tuple[float, int]:
        """``(sum, steps)`` as the resume state keeps them."""
        return float(self.sum), self.n

    def mean(self) -> float:
        return float(self.sum) / max(1, self.n)


class _EpochWatch:
    """What sets the pace of an epoch, measured without making the processor wait for the GPU:
    the time spent waiting for the next batch from the data loader, the GPU time of every
    ``every``-th step (CUDA events, read once the epoch is over), and on CUDA the memory peaks
    and the allocator's retries (allocations that failed at first and succeeded only after the
    allocator released its cache: memory pressure at the cap)."""

    def __init__(self, device: str, every: int = 25):
        import torch

        self._torch = torch
        self.cuda = str(device).startswith("cuda")
        self.every = int(every)
        self.wait = 0.0
        self.events: list = []
        self._open = None
        self.retries0 = self._retries() if self.cuda else 0

    def _retries(self) -> int:
        return int(self._torch.cuda.memory_stats().get("num_alloc_retries", 0))

    def batches(self, loader):
        """``loader``'s batches; the time spent waiting for each is added to ``wait``."""
        t = time.perf_counter()
        it = iter(loader)
        while True:
            try:
                item = next(it)
            except StopIteration:
                self.wait += time.perf_counter() - t
                return
            self.wait += time.perf_counter() - t
            yield item
            t = time.perf_counter()

    def step_starts(self, k: int) -> None:
        """Before step ``k`` of this epoch (0 = its first in this session)."""
        if self.cuda and k % self.every == 0:
            self._open = self._torch.cuda.Event(enable_timing=True)
            self._open.record()

    def step_ends(self) -> None:
        if self._open is not None:
            end = self._torch.cuda.Event(enable_timing=True)
            end.record()
            self.events.append((self._open, end))
            self._open = None

    def summary(self, seconds: float, steps: int) -> dict:
        """The epoch's record, for ``steps`` training steps in ``seconds`` (validation not
        included): the share of that time spent waiting for data and, on CUDA, the median GPU
        time of the timed steps, the share of the time the GPU was busy by that measure, the
        memory peaks (GB, since the run started in this process) and the allocator's retries
        in this epoch."""
        rec = {"data_wait_share": round(self.wait / max(seconds, 1e-9), 4)}
        if not self.cuda:
            return rec
        torch = self._torch
        if self.events:
            self.events[-1][1].synchronize()
            ms = statistics.median(a.elapsed_time(b) for a, b in self.events)
            rec.update(gpu_step_ms=round(ms, 1), gpu_steps_timed=len(self.events),
                       gpu_busy_share=round(ms * steps / 1000.0 / max(seconds, 1e-9), 4))
        rec.update(gpu_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                   gpu_allocated_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   alloc_retries=self._retries() - self.retries0)
        return rec


#: the crawl guard (crawling). On the RTX 4080 SUPER of the profile of 2026-11-25 (16 GB,
#: driving the display under Windows' WDDM), LadderNet's head scoring all charges at once with
#: GPU speedups let the caching allocator's reserve climb on the runs' varied batches to 13.2 of
#: the 13.25 GB the cap left for tensors; the steps then took seconds (a 90th percentile of
#: 21 s) with no out-of-memory error and no allocator retry, so nothing stopped the run. It did
#: so from its first batches, so a run's own early steps are no measure of its pace: the pace is
#: the step setup measured on this GPU for this view and head chunking with every spectrum at
#: the active-bin cap (the slowest step a healthy run takes; TrainConfig.expected_step_seconds),
#: and only without it the median of the session's steps CRAWL_BASELINE (counted from 0). A step
#: crawls when it takes more than CRAWL_FACTOR times the pace and more than CRAWL_FLOOR_SECONDS;
#: a run crawls when CRAWL_SLOW of its last CRAWL_RECENT steps do (the session's first
#: CRAWL_WARMUP steps, CUDA's and cuDNN's start, not counted) while the allocator reserves
#: CRAWL_RESERVED of the cap or more. In the profile's crawling run the eight batches took
#: 1.37, 1.80, 16.7, 1.88, 0.67, 0.35, 0.08 and 0.10 s against setup's 0.149 s: four of any
#: eight steps over 1 s
CRAWL_WARMUP = 2
CRAWL_BASELINE = (5, 20)
CRAWL_RECENT, CRAWL_SLOW = 8, 3
CRAWL_FACTOR, CRAWL_FLOOR_SECONDS = 5.0, 1.0
CRAWL_RESERVED = 0.9


def _pace(step_seconds, expected: float | None) -> float | None:
    """A run's pace (seconds a step): ``expected``, setup's step at the active-bin cap, where
    known, else the median of the session's steps CRAWL_BASELINE once it has them, else None."""
    if expected and expected > 0:
        return float(expected)
    first, last = CRAWL_BASELINE
    if len(step_seconds) <= last:
        return None
    return statistics.median(step_seconds[first:last + 1])


def _slow_steps(step_seconds, expected: float | None = None) -> list[float]:
    """Those of the last CRAWL_RECENT of ``step_seconds`` (the first CRAWL_WARMUP not counted)
    that crawl: longer than CRAWL_FACTOR times the pace (_pace) and than CRAWL_FLOOR_SECONDS."""
    pace = _pace(step_seconds, expected)
    if pace is None:
        return []
    limit = max(CRAWL_FLOOR_SECONDS, CRAWL_FACTOR * pace)
    recent = step_seconds[max(CRAWL_WARMUP, len(step_seconds) - CRAWL_RECENT):]
    return [s for s in recent if s > limit]


def crawling(step_seconds: list[float], reserved_gb: float, cap_gb: float,
             expected: float | None = None) -> bool:
    """Whether a CUDA run crawls: at least 3 of the last 8 of ``step_seconds`` (the session's
    steps in order, each timed on its own; its first two not counted) took more than 5 times
    the pace and more than 1 s, while the caching allocator reserves ``reserved_gb``, at least
    90 % of ``cap_gb`` (the cap on tensors the run set: its GPU memory cap less the CUDA
    context). The pace is ``expected``, the step setup measured for this view and chunking with
    every spectrum at the active-bin cap; without it, the median of the session's steps 5 to 20,
    so that a run crawling from its start goes unnoticed (that seems its pace) and nothing is
    decided before step 21. False with the memory further from the cap, where slow steps have
    another cause (another program on the GPU, a throttled processor)."""
    if not cap_gb or reserved_gb < CRAWL_RESERVED * cap_gb:
        return False
    return len(_slow_steps(step_seconds, expected)) >= CRAWL_SLOW


class _CrawlGuard:
    """The crawl guard of a CUDA run (:func:`crawling`): times every training step from the
    arrival of its batch to its end, and reads after it what the caching allocator reserves.
    A run whose LadderNet head scores all charges at once (``stops``) waits for the GPU at the
    end of every step, so that each time is that step's own, and stops on a crawl: in chunks the
    head needs a fraction of the memory. Any other run is timed without that wait (LadderNet
    waits for the GPU at every step's active bins, so the times follow the GPU's within a step)
    and only says so, once: chunks would not help it."""

    def __init__(self, cap_gb: float, stops: bool, expected: float | None = None):
        import torch

        self._torch = torch
        self.cap_gb = float(cap_gb)
        self.stops = bool(stops)
        self.expected = float(expected) if expected and expected > 0 else None
        self.seconds: list[float] = []
        self.warned = False
        self.clock = time.perf_counter
        self._start = 0.0

    def step_starts(self) -> None:
        self._start = self.clock()

    def step_ends(self) -> str | None:
        """After a step: why the run stops (a crawl, where the run stops for one), else None."""
        torch = self._torch
        if self.stops:
            torch.cuda.synchronize()
        self.seconds.append(self.clock() - self._start)
        if len(self.seconds) > 4096:  # crawling reads the baseline and the last steps only
            del self.seconds[CRAWL_BASELINE[1] + 1:-CRAWL_RECENT]
        reserved = torch.cuda.memory_reserved() / 2**30
        if not crawling(self.seconds, reserved, self.cap_gb, self.expected):
            return None
        slow = _slow_steps(self.seconds, self.expected)
        against = ("setup's step with every spectrum at the active-bin cap" if self.expected
                   else "early in this session")
        counted = min(CRAWL_RECENT, len(self.seconds) - CRAWL_WARMUP)
        what = (f"{len(slow)} of the last {counted} training steps took "
                f"{min(slow):.1f}-{max(slow):.1f} s against "
                f"{_pace(self.seconds, self.expected):.2f} s, {against}, with {reserved:.1f} of "
                f"the {self.cap_gb:.1f} GB the cap leaves for tensors reserved by PyTorch's "
                f"allocator")
        if self.stops:
            return (f"GPU memory crawl ({what}: the head, scoring all charges at once, has filled "
                    f"the cap on the varied batches)")
        if not self.warned:
            self.warned = True
            print(f"  warning: the GPU slowed down: {what}. The head already scores the "
                  f"charges in chunks, so the run goes on; other programs using the GPU's "
                  f"memory (the display's too) can cause this", flush=True)
        return None


def _crawl_guard(device: str, model, cap_gb: float | None,
                 expected: float | None = None) -> _CrawlGuard | None:
    """The crawl guard of a run on ``device`` (None off CUDA, or without a cap): one that stops
    a LadderNet whose head scores all charges at once (``head_chunk`` None), and only warns
    any other run; ``expected`` is its pace (TrainConfig.expected_step_seconds)."""
    if not (str(device).startswith("cuda") and cap_gb):
        return None
    return _CrawlGuard(cap_gb, stops=getattr(model, "head_chunk", 0) is None, expected=expected)


def train(config: TrainConfig, synth_config=None):
    import torch

    from ..synth.config import SynthConfig
    from ..synth.dataset import make_dataloaders
    from ..synth.pool import choose_loader
    from .features import N_ISOTOPE_CHANNELS
    from .model import VIEWS, build_model

    synth_config = synth_config or SynthConfig(z_max=config.z_max)
    if config.iso_channels and config.arch != "ladder":
        raise ValueError("isotope-charge channels are read by LadderNet only (arch='ladder')")
    if config.iso_channels and config.evidence_channels:
        raise ValueError("isotope-charge channels and comb evidence channels cannot be combined")
    if config.view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, not {config.view!r}")
    comb_view = config.view == "comb"
    if comb_view and (config.arch != "ladder" or config.evidence_channels):
        raise ValueError("the comb view is a LadderNet's (arch='ladder'), without evidence "
                         "channels")
    for sc in ([c for _w, c in synth_config] if isinstance(synth_config, (list, tuple))
               else [synth_config]):
        sc.evidence_channels = config.evidence_channels
        sc.singly_charged = config.singly_charged
        sc.iso_channels = config.iso_channels
        sc.measured_channel = comb_view
        if config.arch == "ladder":
            # the frames are read at the network's window: its offsets and active-bin cap are
            # fixed for that length (the U-Net reads any length, so its crops stay as set)
            sc.crop_size = config.window
    in_ch = (4 + config.evidence_channels + (N_ISOTOPE_CHANNELS if config.iso_channels else 0)
             + (1 if comb_view else 0))
    if config.threads:
        torch.set_num_threads(int(config.threads))
    device = _device(config.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available, and this run is set to train on the GPU only. Check with "
            "'.venv\\Scripts\\python gpu_training\\check_gpu.py' (Linux: .venv/bin/python "
            "gpu_training/check_gpu.py); to reinstall PyTorch for CUDA, run "
            "'gpu_training\\setup_windows.bat --reinstall-torch'.")
    gpu_cap_gb = None  # the GPU memory tensors may take, which the crawl guard watches
    if device.startswith("cuda"):
        gpu_cap_gb = (_cap_gpu_memory(torch, config.max_gpu_memory_gb)
                      if config.max_gpu_memory_gb
                      else torch.cuda.get_device_properties(0).total_memory / 2**30)
    ladder_kw = ({"iso_count": N_ISOTOPE_CHANNELS if config.iso_channels else None,
                  "view": config.view, "head_chunk": config.head_chunk}
                 if config.arch == "ladder" else {})
    model = build_model(config.model_size, z_max=config.z_max, in_ch=in_ch,
                        comb_prior=config.comb_prior, arch=config.arch, window=config.window,
                        iso=config.iso_channels, **ladder_kw).to(device)
    speedups = _active_speedups(config, device, model)
    if "index_select" in speedups:
        model.head_gather = "index_select"  # before the moving average copies the model
    iso_count = int(getattr(model, "iso_count", 0))
    if config.init:
        state = torch.load(config.init, map_location=device, weights_only=False)["state_dict"]
        model.load_state_dict(state)
        print(f"started from {config.init}")
    print(f"model {config.arch}/{config.model_size}: {model.num_params()/1e6:.3f}M params "
          f"on {device}; singly charged ions: {config.singly_charged}")
    if config.arch == "ladder" and (config.window != 32768 or config.iso_channels or comb_view):
        where = (f"comb view: any length (crops of {model.window} bins in training)" if comb_view
                 else f"window {model.window} bins")
        print(f"  {where}, ladders read at up to {model.active_top} bins"
              + (f"; {iso_count} isotope-charge channels" if config.iso_channels else ""))

    if config.num_workers > 0:
        _warm_up(synth_config)
    loader = choose_loader(config.loader)
    main = _MainPlacement(*_placement(config, device, loader))
    main.start()  # this process's power throttling, before its workers start
    try:
        def loaders(kind: str):
            return make_dataloaders(synth_config, config.train_len, config.val_len,
                                    batch_size=config.batch_size, num_workers=config.num_workers,
                                    base_seed=config.base_seed, cache=config.cache_frames,
                                    pin_memory=device.startswith("cuda"), loader=kind,
                                    placement=main.placement, window=config.loader_window)

        try:
            tl, vl = loaders(loader)
        except Exception as exc:
            if loader != "ordered" or config.num_workers <= 0:
                raise
            print(f"the ordered loader did not start ({type(exc).__name__}: {exc}); torch's "
                  f"DataLoader makes the spectra instead", flush=True)
            main.loader = "dataloader"
            main.placement = _for_dataloader(main.placement)
            tl, vl = loaders("dataloader")
        main.describe_loader(tl)
        print(_draws_line(), flush=True)
        try:
            if getattr(tl, "pool", None) is not None:
                main.bind(tl)  # its workers have started: this process takes its own cores
            return _train_loaded(config, synth_config, model, tl, vl, device, speedups, in_ch,
                                 iso_count, main, gpu_cap_gb)
        finally:
            close = getattr(tl, "close", None)  # the ordered loader's workers
            if close is not None:
                close()
    finally:
        main.restore()  # its processors and power throttling, for what follows in this process


def _train_loaded(config, synth_config, model, tl, vl, device: str, speedups, in_ch: int,
                  iso_count: int, main, gpu_cap_gb: float | None = None):
    """train()'s training proper, on its loaders ``tl`` and ``vl``; ``main`` is the training
    process's side of the workers' placement (:class:`_MainPlacement`), ``gpu_cap_gb`` the GPU
    memory tensors may take on CUDA (the crawl guard's cap)."""
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    from .losses import ChargeLoss

    steps_per_epoch = len(tl)
    fused = "fused_adamw" in speedups
    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                fused=True if fused else None)
    steps = max(1, steps_per_epoch) * config.epochs
    sched = OneCycleLR(opt, max_lr=config.lr, total_steps=steps, pct_start=0.2)
    loss_fn = ChargeLoss(heat_weight=config.heat_weight,
                         ignore_charges=(1,) if config.singly_charged == "ignore" else ())
    ema = EMA(model, config.ema_decay)
    # the quantity checkpoints are selected by: accuracy on ion bins (z >= 2 only when singly
    # charged ions are not trained on), the way inference reads charges
    metric = "charge_acc_given_ion" if config.singly_charged == "keep" else "charge_acc_given_ion_z2"

    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log.jsonl")
    resume_path = out_path.with_suffix(".resume.pt")

    def pause_files() -> list[Path]:
        # PAUSE, PAUSE.txt (Explorer's new text file) or PAUSE.txt.txt (Explorer hides the
        # extension it adds): any file named PAUSE, whatever follows the first dot, any case
        return [f for f in out_path.parent.iterdir()
                if f.is_file() and f.name.lower().split(".")[0] == "pause"]

    history: list = []
    best_state = None
    best_loss = float("inf")
    best_key = None
    epoch0, step0 = 0, 0
    running = _RunningLoss(device)  # this epoch's training losses
    comb = None
    if resume_path.exists() and not (config.resume or config.restart):
        raise SystemExit(
            f"{resume_path} holds the state of a paused or interrupted run. Add --resume to "
            f"continue it, or --restart to discard it and start again (this overwrites "
            f"{out_path} too).")
    synth_key = synth_fingerprint(synth_config)
    active_top = getattr(model, "active_top", None)
    if config.resume and resume_path.exists():
        st = torch.load(resume_path, map_location=device, weights_only=False)
        saved = dict(st.get("config", {}))
        saved["active_top"] = st.get("active_top", _default_active_top(saved))
        # states written before it was recorded: the first layout's two channels
        saved["iso_count"] = st.get("iso_count", 2 if saved.get("iso_channels") else 0)
        _check_same_run(saved, {**asdict(config), "active_top": active_top,
                                "iso_count": iso_count})
        if st.get("synth") != synth_key:
            why = ("was written by an earlier version of this package, which did not record its "
                   "synthetic spectra (and whose training mixture differed)"
                   if st.get("synth") is None else
                   "was written by a run that trained on other synthetic spectra")
            raise SystemExit(
                f"cannot resume: {resume_path} {why}; continuing would mix the two. Finish it "
                f"with the version that started it, or start again with --restart.")
        model.load_state_dict(st["model"])
        ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        _use_fused(opt, fused)  # the speedups may differ from the paused session's
        sched.load_state_dict(st["sched"])
        history, best_state, best_loss = st["history"], st["best_state"], st["best_loss"]
        best_key = tuple(st["best_key"]) if st["best_key"] is not None else None
        epoch0, step0, comb = st["epoch"], st["step"], st["comb"]
        running = _RunningLoss(device, *st.get("running", (0.0, 0)))
        if epoch0 >= config.epochs:
            print(f"{resume_path}: this run already finished all {config.epochs} epochs")
        else:
            print(f"resumed from {resume_path}: epoch {epoch0}, step {step0} of "
                  f"{steps_per_epoch}")
    elif config.resume:
        print(f"nothing to resume at {resume_path}; starting from the beginning")
    if epoch0 == 0 and step0 == 0 and log_path.exists():
        # a new run: the log of an earlier one (a restart) is set aside, not appended to
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(log_path.stat().st_mtime))
        log_path.replace(log_path.with_name(f"{out_path.stem}.log.{stamp}.jsonl"))
    if config.clear_pause_files:
        for f in pause_files():
            f.unlink()
            print(f"removed {f} left from the last pause")

    if comb is None:
        # Before spending hours: the true labels must beat every constant answer by a clear
        # margin on real batches, or the optimiser will settle on the constant.
        from .checks import degeneracy_report

        # from the validation spectra, which are in memory: iterating the training loader here
        # would start its persistent workers while the comb's workers below are running too
        first = next(iter(vl), None)
        if first is not None:
            rep = degeneracy_report(_batch_dict(first, "cpu"), loss_fn)
            print(f"loss of constant predictors relative to the truth: {rep['margins']}")
            _log(log_path, {"degeneracy": rep})
    if comb is None or "by_kind" not in comb:
        # the bar to clear: the comb as the pipeline runs it (whole spectrum, isotope-spacing
        # charges on Orbitrap spectra) on the validation frames, scored as a network is
        print("scoring the comb estimator, as the pipeline runs it, on the validation frames ...",
              flush=True)
        comb = comb_baseline(vl, num_workers=config.num_workers, placement=main.placement,
                             pool=getattr(tl, "pool", None))
        _log(log_path, {"comb_baseline": {k: v for k, v in comb.items() if k != "kinds"}})
    bands = {k: round(v, 3) for k, v in comb["bands_given_ion"].items()}
    print(f"comb estimator ({comb['estimator']}): {metric} {comb[metric]:.3f} bands {bands}")
    for line in by_kind_lines(None, comb.get("by_kind")):
        print(line)

    def save_state(epoch: int, step: int, stop_reason: str | None = None) -> None:
        state = {"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                 "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history,
                 "best_state": best_state, "best_key": best_key, "best_loss": best_loss,
                 "epoch": epoch, "step": step, "comb": comb, "config": asdict(config),
                 "running": running.state(), "synth": synth_key, "stop_reason": stop_reason,
                 "active_top": active_top, "iso_count": iso_count,
                 "gpu_speedups": list(speedups)}
        _atomic_save(state, resume_path)
        extra = {"gpu_speedups": list(speedups), "spectrum_draws": _draws_status()}
        if history and history[-1].get("pace"):
            extra["pace"] = history[-1]["pace"]  # the last epoch's
        _write_status(out_path, epoch, step, config.epochs, stop_reason, extra)

    # Ctrl+C asks for a pause: the current step finishes, the state is saved, the run ends.
    # A second Ctrl+C aborts at once. A file named PAUSE next to the output does the same.
    import signal
    import threading

    pause = {"requested": False}

    def on_interrupt(signum, frame):
        if pause["requested"]:
            raise KeyboardInterrupt
        pause["requested"] = True
        print("\n  pause requested: saving after this step (Ctrl+C again aborts)", flush=True)

    old_handler = None
    if threading.current_thread() is threading.main_thread():
        old_handler = signal.signal(signal.SIGINT, on_interrupt)

    def paused() -> bool:
        return pause["requested"] or bool(pause_files())

    def stop_here(epoch: int, step: int, why: str, reason: str = "paused",
                  then: str = "run the same command with --resume to continue") -> tuple:
        save_state(epoch, step, reason)
        print(f"{why}: state saved to {resume_path}; {then} (epoch {epoch}, step {step})",
              flush=True)
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    session_start = time.time()
    last_save = time.time()
    ram = _RamWatch(config.max_ram_gb, extra_gb=getattr(tl, "shared_gb", 0.0))
    stats = getattr(tl, "item_stats", None)  # what the workers made, where
    pool = getattr(tl, "pool", None)
    pace = {"steps": 0, "seconds": 0.0}  # training steps done in this session, and their time
    guard = _crawl_guard(device, model, gpu_cap_gb, config.expected_step_seconds)
    awake = _KeepAwake()
    awake.start()
    flags = set_gpu_flags(speedups) if speedups else None
    try:
        for epoch in range(epoch0, config.epochs):
            model.train()
            if epoch != epoch0:
                running.reset()
            first_step = step0 if epoch == epoch0 else 0
            # new spectra every epoch: epoch e reads items e * train_len ... (EpochSampler)
            tl.sampler.epoch = epoch if config.fresh_each_epoch else 0
            tl.sampler.start = first_step * config.batch_size
            epoch_start = time.time()
            watch = _EpochWatch(device)
            made0 = stats.snapshot() if stats is not None else None
            twice0 = pool.duplicates if pool is not None else 0
            i = first_step - 1
            try:
                for i, sample in enumerate(watch.batches(tl), start=first_step):
                    if not main.bound:  # the workers have started: this process takes its cores
                        main.bind(tl)
                    # nothing in a step reads a value back from the GPU but the active bins'
                    # positions (LadderNet): the loss is read at log steps only, and the crawl
                    # guard of an unchunked head waits for the GPU at the step's end
                    watch.step_starts(i - first_step)
                    if guard is not None:
                        guard.step_starts()
                    batch = _batch_dict(sample, device)
                    out = model(batch["features"])
                    losses = loss_fn(out, batch)
                    opt.zero_grad()
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    opt.step()
                    if sched.last_epoch < steps - 1:
                        sched.step()
                    ema.update(model)
                    running.add(losses["loss"])
                    watch.step_ends()
                    crawl = guard.step_ends() if guard is not None else None
                    if i % config.log_every == 0:
                        print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss "
                              f"{float(losses['loss'].detach()):.4f} ce "
                              f"{float(losses['ce']):.4f} heat {float(losses['heat']):.4f}",
                              flush=True)
                    # a crawl first: a pause asked for during a crawling step (Ctrl+C, likely
                    # while steps take seconds) still records why the run stopped
                    if crawl:
                        return stop_here(epoch, i + 1, crawl, "gpu_memory_crawl",
                                         "run the same command with --resume --head-chunk 16 to "
                                         "continue in chunks, as gpu_training/train_all.py does "
                                         "by itself")
                    if paused():
                        return stop_here(epoch, i + 1, "paused")
                    if config.max_seconds and (time.time() - session_start) > config.max_seconds:
                        return stop_here(epoch, i + 1, "time budget reached", "time")
                    if ram.over():
                        return stop_here(epoch, i + 1,
                                         f"memory budget of {config.max_ram_gb:g} GB reached "
                                         f"({ram.used_gb():.1f} GB in use)", "memory")
                    if time.time() - last_save > 60.0 * config.save_every_minutes:
                        save_state(epoch, i + 1)
                        last_save = time.time()
            except (RuntimeError, OSError):
                # Ctrl+C also reaches the data-loading workers, which then exit; on Windows the
                # interrupted wait can surface as an OSError (InterruptedError) instead
                if pause["requested"]:
                    return stop_here(epoch, i + 1, "paused")
                raise
            trained = time.time() - epoch_start
            pace["steps"] += i + 1 - first_step
            pace["seconds"] += trained
            measured = watch.summary(trained, i + 1 - first_step)
            if speedups:
                measured["gpu_speedups"] = list(speedups)
            if stats is not None:
                measured["workers"] = {**main.record(), **stats.summary(made0, main.kind_of_cpu,
                                                                        trained)}
                if pool is not None:
                    measured["workers"]["made_twice"] = pool.duplicates - twice0
            try:
                val_ema = evaluate(ema.shadow, vl, loss_fn, device, kinds=comb.get("kinds"))
                val_live = evaluate(model, vl, loss_fn, device, kinds=comb.get("kinds"))
            except (RuntimeError, OSError):
                # interrupted while validating: the epoch's training is done, so resuming
                # starts with this validation
                if pause["requested"]:
                    return stop_here(epoch, steps_per_epoch, "paused")
                raise

            def key(v):
                if config.select_by == "accuracy":
                    # accuracy on ion bins: what inference uses (the top charges among the ion
                    # classes); an argmax that includes the background class splits the votes
                    return (v[metric], -v["val_loss"])
                return (-v["val_loss"],)

            best = "ema" if key(val_ema) >= key(val_live) else "live"
            val = val_ema if best == "ema" else val_live
            history.append({"epoch": epoch, "train_loss": running.mean(),
                            "ema": val_ema, "live": val_live, "best": best,
                            "seconds": time.time() - session_start, "pace": measured})
            _log(log_path, history[-1])
            bands = {k: round(v, 3) for k, v in val["bands_given_ion"].items()}
            print(f"epoch {epoch}: {metric} ema {val_ema[metric]:.3f} live "
                  f"{val_live[metric]:.3f} (comb {comb[metric]:.3f}) -> keeping {best}; "
                  f"bands {bands}", flush=True)
            for line in by_kind_lines(val.get("by_kind"), comb.get("by_kind")):
                print(line, flush=True)
            print("  " + _pace_line(pace, epoch, epoch_start, config, steps_per_epoch, ram,
                                    device, measured), flush=True)
            if best_state is None or key(val) > best_key:
                best_key = key(val)
                best_loss = val["val_loss"]
                best_state = copy.deepcopy((ema.shadow if best == "ema" else model).state_dict())
                _save(out_path, best_state, config, in_ch, val, comb, metric, iso_count,
                      gpu_speedups=speedups)
                print(f"  saved {out_path} ({metric} {val[metric]:.3f})")
            running.reset()
            save_state(epoch + 1, 0, "finished" if epoch + 1 >= config.epochs else None)
            last_save = time.time()
            if paused() and epoch + 1 < config.epochs:
                return stop_here(epoch + 1, 0, "paused")
    finally:
        awake.stop()
        if flags is not None:
            restore_gpu_flags(flags)
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)

    if best_state is None:
        best_state = model.state_dict()
        _save(out_path, best_state, config, in_ch, None, comb, metric, iso_count,
              gpu_speedups=speedups)
    print(f"finished; best checkpoint {out_path} (val_loss {best_loss:.4f})")
    model.load_state_dict(best_state)
    return model, history


def _draws_status() -> dict:
    """How this computer draws the training scans' noise (``synth.rng.status``: the compiled
    draws, checked against numpy's here, or numpy's own and why); the data workers decide the
    same, on the same numpy and with the same environment."""
    from ..synth import rng as draws

    return draws.status()


def _draws_line() -> str:
    st = _draws_status()
    if st["draws"] == "compiled":
        return ("spectrum draws: compiled, the same as numpy's draw for draw (checked here in "
                f"{st['check_seconds']:.2f} s)")
    return f"spectrum draws: numpy's, slower ({st['reason']})"


def _warm_up(synth_config) -> None:
    """Generate one spectrum of every configuration in this process before the data workers
    start, so that the compiled numba functions are cached once instead of being compiled by
    every worker at the same time."""
    import numpy as _np

    from ..synth.fast import warm_up
    from ..synth.frames import generate_frame

    warm_up()  # every kernel, also those one frame does not reach
    configs = ([c for _w, c in synth_config] if isinstance(synth_config, (list, tuple))
               else [synth_config])
    for cfg in configs:
        generate_frame(cfg, _np.random.default_rng(0))


def _placement(config, device: str, loader: str) -> tuple:
    """Where the data workers and this process run (:func:`lcmsdeconv.nn.cpus.plan` for this
    computer's cores, with ``config.cpu_binding``, ``worker_priority`` and ``worker_qos``), and
    the cores themselves; printed. The placement is None where it changes nothing (no workers,
    or not bound at below-normal priority with Windows' power throttling: as before)."""
    from .cpus import MODES, PRIORITIES, QOS_MODES, Placement, allowed_cpus, detect, plan

    for name, value, allowed in (("cpu_binding", config.cpu_binding, MODES),
                                 ("worker_priority", config.worker_priority, PRIORITIES),
                                 ("worker_qos", config.worker_qos, QOS_MODES)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
    topo = detect()
    if config.num_workers <= 0:  # no workers: this process's power throttling alone
        return (Placement("none", qos="high") if config.worker_qos == "high" else None), topo
    mode = config.cpu_binding
    if mode == "core" and loader != "ordered":
        print("the data workers share the cores (--cpu-binding pool) instead of one each: "
              "torch's DataLoader hands whole batches to its workers in turn, so the slowest "
              "core would set the pace of all (each on its own core needs --loader ordered)")
        mode = "pool"
    # training on the processor: the training process keeps a core for each of its threads
    main_cores = int(config.threads) if config.threads and not device.startswith("cuda") \
        else None
    placement = plan(topo, workers=config.num_workers, allowed=allowed_cpus(), mode=mode,
                     main_cores=main_cores, priority=config.worker_priority,
                     qos=config.worker_qos)
    print(f"processor: {topo.describe()}; {placement.describe()}", flush=True)
    if (placement.mode, placement.priority, placement.qos) == ("none", "below_normal",
                                                               "system"):
        return None, topo  # as before: lcmsdeconv.synth.dataset.worker_init
    return placement, topo


def _for_dataloader(placement):
    """``placement`` for torch's DataLoader: shared cores instead of one each (see _placement)."""
    if placement is None or placement.mode != "core":
        return placement
    from dataclasses import replace

    pool = tuple(sorted({c for w in placement.workers for c in w}))
    return replace(placement, mode="pool", workers=tuple(pool for _ in placement.workers))


class _MainPlacement:
    """The training process's side of where its data workers run: its power throttling from the
    start of training, its own cores once its workers have started (children inherit their
    parent's affinity when they are created), the record of each epoch, and all of it given
    back when training ends (an export, a resume or another run in this process then has the
    whole processor again)."""

    def __init__(self, placement, topo):
        from .cpus import allowed_cpus

        self.placement = placement
        self.kind_of_cpu = topo.kind_of_cpu() if topo is not None else {}
        self.allowed = allowed_cpus()
        self.loader = None
        self.bound = placement is None or placement.mode == "none"
        self._pinned = False
        self._qos = None  # power throttling before it was set: (set?, previous state)

    def start(self) -> None:
        from .cpus import get_power_throttling, set_power_throttling

        if self.placement is not None and self.placement.qos == "high":
            before = get_power_throttling()
            self._qos = (set_power_throttling(False), before)

    def describe_loader(self, tl) -> None:
        pool = getattr(tl, "pool", None)
        self.loader = "ordered" if pool is not None else "dataloader"
        if pool is not None:
            ring = f", {tl.shared_gb:.2f} GB of shared slots" if pool.ring is not None else ""
            print(f"data loading: ordered, item by item to the free workers, at most "
                  f"{tl.window} items ahead{ring}", flush=True)
        elif tl.num_workers > 0:
            print("data loading: torch's DataLoader, whole batches to the workers in turn",
                  flush=True)

    def bind(self, tl) -> None:
        """Bind this process to its cores and check where its workers may run."""
        if self.bound:
            return
        self.bound = True
        from .cpus import affinity_of, apply_main, cpu_ranges

        done = apply_main(self.placement, qos=False)
        self._pinned = bool(done.get("pinned"))
        pool = getattr(tl, "pool", None)
        if pool is not None:
            pids = pool.pids
        else:  # torch's DataLoader keeps its workers in its iterator (private)
            workers = getattr(getattr(tl, "_iterator", None), "_workers", None) or []
            pids = [w.pid for w in workers]
        masks = affinity_of(pids)
        ok = sum(1 for i, pid in enumerate(pids)
                 if masks.get(pid) == sorted(self.placement.worker_cpus(i)))
        where = (f"bound to processors {cpu_ranges(self.placement.main)}" if self._pinned
                 else "not bound")
        print(f"  {ok} of {len(pids)} data workers bound as planned; this process {where}",
              flush=True)

    def record(self) -> dict:
        p = self.placement
        return {"loader": self.loader, "binding": p.mode if p is not None else "none",
                "priority": p.priority if p is not None else "below_normal",
                "qos": p.qos if p is not None else "system"}

    def restore(self) -> None:
        from .cpus import pin_process, restore_power_throttling

        if self._pinned and self.allowed:
            pin_process(self.allowed)
            self._pinned = False
        if self._qos is not None and self._qos[0]:
            restore_power_throttling(self._qos[1])
        self._qos = None


def _write_status(out_path: Path, epoch: int, step: int, epochs: int,
                  stop_reason: str | None, extra: dict | None = None) -> None:
    """``<out>.status.json``: where the run stands, for gpu_training/train_all.py (which then
    does not need to load the resume state, and PyTorch with it), and ``extra`` for the record
    (the GPU speedups in use, the last epoch's pace measurements)."""
    import json
    import os

    path = out_path.with_suffix(".status.json")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"epoch": epoch, "step": step, "epochs": epochs,
                               "stop_reason": stop_reason, **(extra or {})}) + "\n",
                   encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def _cap_gpu_memory(torch, max_gb: float, context_gb: float = 0.75) -> float:
    """Keep this process's GPU memory below ``max_gb``: PyTorch's caching allocator gets
    ``max_gb`` less an allowance for the CUDA context, and raises out-of-memory rather than
    spill into shared system memory, which Windows drivers otherwise do. Returns the cap on
    tensors (GB). The allocator's reserve may still come so close to it that the steps crawl
    without an error (crawling)."""
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, max(0.05, (max_gb - context_gb) * 2**30 / total))
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"GPU memory capped at {max_gb:g} GB of {total / 2**30:.1f} GB "
          f"({frac * total / 2**30:.1f} GB for tensors, the rest for the CUDA context)")
    return frac * total / 2**30


class _KeepAwake:
    """Keeps Windows from going to sleep while a run trains (the screen may still switch off).
    Elsewhere it does nothing."""

    def __init__(self):
        self.active = False

    def start(self) -> None:
        import sys

        if sys.platform != "win32":
            return
        try:
            import ctypes

            # a private, typed handle on kernel32: the shared ctypes.windll one has untyped
            # prototypes that other libraries may rely on
            self.fn = ctypes.WinDLL("kernel32").SetThreadExecutionState
            self.fn.argtypes = [ctypes.c_uint32]
            self.fn.restype = ctypes.c_uint32
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            self.active = self.fn(0x80000000 | 0x00000001) != 0
        except Exception:  # pragma: no cover - only a convenience
            self.active = False

    def stop(self) -> None:
        if not self.active:
            return
        try:
            self.fn(0x80000000)  # ES_CONTINUOUS alone: sleep is allowed again
        except Exception:  # pragma: no cover
            pass
        self.active = False


class _RamWatch:
    """RAM of this process and its data workers together, checked at most every 15 s: the sum
    of their unique sets (on Windows, private working sets), as the task manager shows them
    (:func:`lcmsdeconv.nn.machine.process_ram_gb`), and ``extra_gb`` of shared memory that none
    of them counts as its own (the ordered loader's slots)."""

    def __init__(self, limit_gb: float | None, every: float = 15.0, extra_gb: float = 0.0):
        self.limit = limit_gb
        self.every = every
        self.extra_gb = float(extra_gb)  # shared memory no process counts as its own
        self.last = 0.0
        self.proc = None
        try:
            import psutil

            self.proc = psutil.Process()
        except ImportError:
            if limit_gb:
                print("psutil is not installed, so the memory budget cannot be watched "
                      "(pip install psutil)")

    def used_gb(self) -> float:
        if self.proc is None:
            return 0.0
        import psutil

        from .machine import process_ram_gb

        total = self.extra_gb
        for p in [self.proc, *self.proc.children(recursive=True)]:
            try:
                total += process_ram_gb(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def over(self) -> bool:
        if not self.limit or self.proc is None or time.time() - self.last < self.every:
            return False
        self.last = time.time()
        return self.used_gb() > self.limit


def _pace_line(pace: dict, epoch: int, epoch_start: float, config, steps_per_epoch: int,
               ram: _RamWatch, device: str, measured: dict | None = None) -> str:
    """Duration and speed of the epoch, what set its pace (``measured``, the epoch's record of
    :class:`_EpochWatch`), memory in use, and the time left."""
    import torch

    m = measured or {}
    took = time.time() - epoch_start
    per_step = pace["seconds"] / max(1, pace["steps"])
    left = per_step * steps_per_epoch * (config.epochs - epoch - 1)
    parts = [f"epoch took {took / 60:.1f} min",
             f"{config.batch_size / max(per_step, 1e-9):.0f} spectra/s"]
    if "data_wait_share" in m:
        share = f"of the training time {100 * m['data_wait_share']:.0f} % waiting for data"
        if m.get("gpu_step_ms") is not None:
            share += (f", {100 * m['gpu_busy_share']:.0f} % GPU busy ({m['gpu_step_ms']:.0f} ms "
                      f"a step)")
        parts.append(share)
    workers = _workers_text(m.get("workers"))
    if workers:
        parts.append(workers)
    if ram.proc is not None:
        parts.append(f"RAM {ram.used_gb():.1f} GB" + (f" of {ram.limit:g}" if ram.limit else ""))
    if device.startswith("cuda"):
        reserved = m.get("gpu_reserved_gb", torch.cuda.max_memory_reserved() / 2**30)
        gpu = f"GPU {reserved:.1f} GB reserved"
        if "gpu_allocated_gb" in m:
            retries = m.get("alloc_retries", 0)
            gpu += (f", {m['gpu_allocated_gb']:.1f} GB allocated at most, {retries} allocator "
                    f"retr{'y' if retries == 1 else 'ies'}")
        parts.append(gpu)
    if config.epochs - epoch - 1 > 0:
        parts.append(f"about {left / 3600:.1f} h to go" if left >= 3600
                     else f"about {left / 60:.0f} min to go")
    return "; ".join(parts)


#: how the pace line names the classes of cores (lcmsdeconv.nn.cpus; "?": all alike)
_CORE_CLASS_NAMES = {"P": "performance cores ", "E": "efficient cores ",
                     "LP": "low-power cores ", "?": ""}


def _workers_text(w: dict | None) -> str | None:
    """The workers' part of an epoch's pace line: spectra made on each class of core, how fast
    a worker made them while busy, on which processors, and how busy the workers were."""
    by = (w or {}).get("by_kind") or {}
    if not by:
        return None
    parts = []
    for kind, d in by.items():
        rate = d.get("per_worker_per_second")
        where = f", on processors {d['processors']}" if d.get("processors") else ""
        parts.append(f"{_CORE_CLASS_NAMES.get(kind, kind + ' ')}{d['items']}"
                     + (f" ({rate:.2f}/s a busy worker{where})" if rate else ""))
    text = "spectra made: " + ", ".join(parts)
    if w.get("workers_busy") is not None:
        text += f", workers busy {100 * w['workers_busy']:.0f} %"
    if w.get("made_twice"):
        text += f", {w['made_twice']} made twice (held too long)"
    return text


def synth_fingerprint(synth_config) -> str:
    """A short digest of the synthetic-data settings (the whole mixture), kept in the resume
    state so that a run is never continued on other spectra than it started with. Settings added
    later count only once they leave their defaults (:func:`lcmsdeconv.synth.config.
    settings_repr`), so a run paused by an earlier version still resumes on the same spectra."""
    import hashlib

    from ..synth.config import settings_repr

    return hashlib.sha256(settings_repr(synth_config).encode()).hexdigest()[:16]


#: settings a resume state written before they existed was trained with
_LATER_RUN_DEFAULTS = {"window": 32768, "iso_channels": False, "view": "window", "iso_count": 0}


def _default_active_top(saved: dict) -> int | None:
    """The active-bin cap of the LadderNet a saved run's settings build (None: a U-Net)."""
    from .constants import ladder_active_top

    if saved.get("arch", "unet") != "ladder":
        return None
    return ladder_active_top(int(saved.get("window", _LATER_RUN_DEFAULTS["window"])))


def _check_same_run(saved: dict, now: dict) -> None:
    """Refuse to resume with settings that change the model, the schedule or the frames."""
    keys = ("arch", "model_size", "singly_charged", "epochs", "train_len", "batch_size", "lr",
            "base_seed", "z_max", "evidence_channels", "comb_prior", "fresh_each_epoch",
            "weight_decay", "heat_weight", "window", "active_top", "iso_channels", "iso_count",
            "view")

    def get(d: dict, k: str):
        return d.get(k, _LATER_RUN_DEFAULTS.get(k))

    diff = {k: (get(saved, k), get(now, k)) for k in keys if get(saved, k) != get(now, k)}
    if diff:
        raise SystemExit(
            "cannot resume: these settings differ from the paused run's (saved, now): "
            f"{diff}. Resume with the original command plus --resume, or use --restart.")


def _atomic_save(obj, path: Path) -> None:
    """Write ``obj`` to ``path`` so that a crash or power cut leaves the old file or the new
    one, never a truncated one: write a temporary file, flush it to disk, then rename."""
    import os

    import torch

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    # on Windows the rename fails while another program (a virus scanner, a backup or sync
    # tool, Explorer's preview) briefly holds either file open; such holds last moments
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def _save(path: Path, state, config, in_ch: int, val, comb, metric: str,
          iso_count: int | None = None, gpu_speedups=()) -> None:
    """Write a checkpoint. ``iso_count``: the isotope channels the network reads (default: as
    many as ``in_ch`` holds, two for six inputs, the first layout's); ``gpu_speedups``: those
    the training used, for the record."""
    from .constants import ladder_active_top
    from .grid import DEFAULT_STEP

    ladder = config.arch == "ladder"
    view = getattr(config, "view", "window")
    if iso_count is None:
        iso_count = (max(0, min(in_ch - 4 - (1 if view == "comb" else 0), 3))
                     if config.iso_channels else 0)
    _atomic_save({"state_dict": state, "config": asdict(config),
                  "synth_config_z_max": config.z_max, "model_size": config.model_size,
                  "z_max": config.z_max, "in_ch": in_ch,
                  "evidence_channels": config.evidence_channels,
                  "comb_prior": config.comb_prior, "arch": config.arch,
                  "singly_charged": config.singly_charged, "selection_metric": metric,
                  "grid_step": DEFAULT_STEP, "window": int(config.window),
                  "active_top": ladder_active_top(config.window) if ladder else None,
                  "iso_channels": bool(config.iso_channels), "iso_count": int(iso_count),
                  "view": view if ladder else "window",
                  "gpu_speedups": list(gpu_speedups or ()),
                  "val": val, "comb_baseline": comb}, path)


def comb_baseline(loader, num_workers: int | None = None, placement=None, pool=None) -> dict:
    """Charge metrics of the comb estimator, as the pipeline runs it, on ``loader``'s frames.

    Every validation frame is generated again with its whole spectrum, which CombPredictor reads
    as ``process_run`` does: all of it at once (a crop would hide the far neighbours of low
    charge states), with isotope-spacing charges on Orbitrap and FT-ICR spectra
    (:mod:`lcmsdeconv.nn.baseline`). Its charges inside each frame's crop are scored exactly as
    :func:`evaluate` scores a network, batch by batch, so the two numbers compare. A frame
    rendered onto its crop alone falls back to the comb's answer in feature channel 3, which
    ``estimator`` then says. Also returns every frame's kind of spectrum (``kinds``, in loader
    order) and per-kind accuracies with both weightings (``by_kind``, see
    :class:`~lcmsdeconv.nn.checks.ChargeScoreboard`).

    The frames are made by ``num_workers`` processes placed as ``placement`` says
    (:func:`lcmsdeconv.synth.dataset.worker_init_for`), or by the training's own workers
    (``pool``, the ordered loader's :class:`~lcmsdeconv.synth.pool.WorkerPool`): the same
    frames, in the same batches.
    """
    import torch
    from torch.utils.data import DataLoader

    from ..synth.dataset import CombBarFrames, mp_context, worker_init_for
    from .checks import ChargeScoreboard, charge_metrics

    source = getattr(loader.dataset, "frames", loader.dataset)  # CachedFrames keeps it
    workers = loader.num_workers if num_workers is None else num_workers
    if pool is not None:
        from ..synth.pool import ordered_batches

        frames = ordered_batches(pool, "comb", CombBarFrames(source), len(source),
                                 loader.batch_size)
    else:
        frames = DataLoader(CombBarFrames(source), batch_size=loader.batch_size,
                            num_workers=workers, shuffle=False,
                            worker_init_fn=worker_init_for(placement, workers),
                            **mp_context(workers))
    sums: dict[str, float] = {}
    bands: dict[str, list[float]] = {}
    kinds: list[str] = []
    hows: set[str] = set()
    board = ChargeScoreboard()
    nb = 0
    for f0, true, zc, kind, how in frames:
        n, length = zc.shape
        zc = zc.long().clamp(0, 100)
        logits = torch.full((n, 101, length), -10.0)
        logits.scatter_(1, zc[:, None, :], 10.0)
        # where the comb has no answer it reads no charge: the ion-class argmax would otherwise
        # land on class 1 and score singly charged bins as correct
        logits[:, 100, :] = torch.where(zc == 0, torch.zeros_like(logits[:, 100, :]),
                                        logits[:, 100, :])
        m = charge_metrics(logits, {"features": f0, "topk_z": true})
        for k in ("weighted_accuracy", "charge_accuracy_given_ion", "charge_accuracy_given_ion_z2"):
            sums[k] = sums.get(k, 0.0) + m[k]
        for k, v in m["bands_given_ion"].items():
            bands.setdefault(k, []).append(v)
        for i in range(n):
            board.add(kind[i], "comb", zc[i].numpy(), true[i, :, 0].numpy(), f0[i, 0].numpy())
        kinds += list(kind)
        hows |= set(how)
        nb += 1
    nb = max(nb, 1)
    estimator = {"pipeline": "whole spectrum, isotope charges on Orbitrap",
                 "channel 3": "channel-3 answer inside the crop"}
    return {"charge_acc": sums.get("weighted_accuracy", 0.0) / nb,
            "charge_acc_given_ion": sums.get("charge_accuracy_given_ion", 0.0) / nb,
            "charge_acc_given_ion_z2": sums.get("charge_accuracy_given_ion_z2", 0.0) / nb,
            "bands_given_ion": {k: float(np.mean(v)) for k, v in bands.items()},
            "estimator": "; ".join(estimator[h] for h in sorted(hows)) or "no frames",
            "kinds": kinds, "by_kind": board.summary()}


def by_kind_lines(net: dict | None, comb: dict | None) -> list[str]:
    """Lines setting a network's per-kind accuracy at z >= 2 beside the comb's, both weighted
    by log-SNR and by intensity (``by_kind`` of :func:`evaluate` and :func:`comb_baseline`)."""
    from .baseline import KIND_NAMES

    def cell(board: dict | None, kind: str, est: str) -> str | None:
        d = (board or {}).get(kind, {}).get(est)
        if not d:
            return None
        a, b = d.get("log-SNR", {}).get("z>=2"), d.get("intensity", {}).get("z>=2")
        if a is None or b is None:
            return None
        return f"{a:.3f} / {b:.3f}"

    lines = []
    for kind, name in KIND_NAMES.items():
        parts = [f"{label} {c}" for label, c in (("network", cell(net, kind, "network")),
                                                 ("comb", cell(comb, kind, "comb"))) if c]
        if parts:
            lines.append(f"    {name:15s} " + "   ".join(parts))
    if lines:
        lines.insert(0, "  charge of z >= 2 ion bins read right (log-SNR / intensity weighted):")
    return lines


def _log(path: Path, record: dict) -> None:
    import json

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def evaluate(model, loader, loss_fn, device, kinds: list[str] | None = None) -> dict:
    """Validation loss and charge metrics (see :func:`lcmsdeconv.nn.checks.charge_metrics`).

    ``charge_acc`` is the log-SNR-weighted top-1 accuracy over ion-bearing bins with the
    background class competing; ``charge_acc_given_ion`` takes the top charge among the ion
    classes only, as inference does (``..._z2``: on bins of charge two or more), and is the
    quantity checkpoints are selected by. With ``kinds`` (each frame's kind of spectrum, in
    loader order) ``by_kind`` adds per-kind accuracies weighted by log-SNR and by intensity.
    """
    import torch

    from .checks import ChargeScoreboard, charge_metrics

    model.eval()
    tot = 0.0
    nb = 0
    sums: dict[str, float] = {}
    bands: dict[str, list[float]] = {}
    never = 0
    board = ChargeScoreboard() if kinds else None
    pos = 0
    with torch.no_grad():
        for sample in loader:
            batch = _batch_dict(sample, device)
            out = model(batch["features"])
            losses = loss_fn(out, batch)
            tot += float(losses["loss"])
            nb += 1
            if board is not None:
                pred = (out["charge_logits"][:, 1:, :].argmax(1) + 1).cpu().numpy()
                true = batch["topk_z"][:, :, 0].cpu().numpy()
                f0 = batch["features"][:, 0, :].cpu().numpy()
                for i in range(pred.shape[0]):
                    if pos + i < len(kinds):
                        board.add(kinds[pos + i], "network", pred[i], true[i], f0[i])
                pos += pred.shape[0]
            m = charge_metrics(out["charge_logits"], batch)
            for k in ("weighted_accuracy", "charge_accuracy_given_ion",
                      "charge_accuracy_given_ion_z2", "harmonic_error",
                      "neighbour_error", "missed_as_background", "background_called_ion"):
                sums[k] = sums.get(k, 0.0) + m[k]
            for k, v in m["bands_given_ion"].items():
                bands.setdefault(k, []).append(v)
            never = max(never, m["classes_never_predicted"])
    nb = max(1, nb)
    res = {"val_loss": tot / nb, "charge_acc": sums.get("weighted_accuracy", 0.0) / nb,
           "charge_acc_given_ion": sums.get("charge_accuracy_given_ion", 0.0) / nb,
           "charge_acc_given_ion_z2": sums.get("charge_accuracy_given_ion_z2", 0.0) / nb}
    for k in ("harmonic_error", "neighbour_error", "missed_as_background", "background_called_ion"):
        res[k] = sums.get(k, 0.0) / nb
    res["bands_given_ion"] = {k: float(np.mean(v)) for k, v in bands.items()}
    res["classes_never_predicted"] = never
    if board is not None:
        res["by_kind"] = board.summary()
    return res
