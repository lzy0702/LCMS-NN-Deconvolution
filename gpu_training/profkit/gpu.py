"""GPU experiments: training steps of the full LadderNet of both views (train_all's recipe, v4
today, isotope-charge channels, as train_all trains them) on the runs' real batches, under the
GPU memory cap.

:func:`prepare` saves the runs' first training batches (made by training's own DataLoader on
the recipe's spectra, seed 0), the batch with every spectrum at the active-bin cap that probe.py
measures, and probe.py's trained-like weights of both views. Then, each in a process of its
own (cuDNN keeps the convolution algorithms a process first chooses):

- :func:`steps`: the step-time distribution of one configuration (GPU speedups, head chunking)
  over the first 8 batches cycled, the free-running pace (as training runs, without waiting
  for the GPU between steps), the at-cap batch, and its loss and gradients against the step
  without speedups from the same weights on the same batch (probe.disagreement);
- :func:`phases`: one configuration's step split into copy, forward, loss, backward, clipping,
  optimiser and moving average by CUDA events, the host's time for each, and the host's wait
  at the active-bin nonzero (the step's one synchronisation);
- :func:`profile`: torch.profiler's CUDA kernels and aten operators that take the most time,
  and the kernels' share of the step (the GPU's busy share);
- :func:`memory`: memory peaks, allocator retries and out-of-memory errors over 40 batches
  (the runner sets PYTORCH_CUDA_ALLOC_CONF for each variant);
- :func:`autocast` and :func:`batch8`, for information (they change the numbers): bf16 over
  the head or the whole forward pass, against float32 from the same weights; 8 spectra a step;
- :func:`smi_loop`: a steady loop of one configuration with nvidia-smi read twice a second.

Every function takes ``device`` (``"cpu"`` runs the same code on the processor, for tests),
``size`` and ``crop`` (smaller networks and spectra for smoke runs and tests)."""

from __future__ import annotations

import gc
import os
import shutil
import statistics
import time
import warnings
from pathlib import Path

import numpy as np

from .common import Unavailable, recipe, recipe_mixture, stats

#: spectra per step (train_all.BATCH) and the GPU memory cap (setup's default)
BATCH = 4
GPU_MEMORY_GB = 14.0
#: input channels of each view: the four base features, three isotope-charge channels, and in
#: the comb view the measured channel
IN_CH = {"window": 7, "comb": 8}
#: the runs' first training batches, cycled in the step measurements
FIRST = 8
NAMES = ("features", "topk_z", "topk_w", "heat")


def require_cuda(device: str = "cuda") -> None:
    if not str(device).startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise Unavailable(f"no CUDA: PyTorch {torch.__version__} (built for CUDA "
                          f"{torch.version.cuda}) sees no GPU")


def window_of(crop: int | None) -> int:
    from train_charge_net import V3_WINDOW

    return int(crop or V3_WINDOW)


def recipe_lr() -> float:
    from train_all import COMMON

    return float(COMMON[COMMON.index("--lr") + 1])


def mixture(crop: int | None = None) -> list:
    """The runs' training spectra as training configures them (common.recipe_mixture:
    train_all's recipe), cropped to ``crop`` bins where given (smoke runs and tests)."""
    mix = recipe_mixture()
    if crop:
        for _w, c in mix:
            c.crop_size = int(crop)
    return mix


def active_rows(x, window: int) -> list[int]:
    """Each spectrum's active bins, whose ladders the network reads (LadderNet.forward's rule
    in the window view: above the active level and among the ``active_top`` strongest)."""
    import torch

    from lcmsdeconv.nn.constants import ladder_active_top

    level = x[:, 0].float()
    top = min(ladder_active_top(window), level.shape[-1])
    kth = torch.topk(level, top, dim=-1).values[:, -1:]
    return [int(v) for v in ((level > 0.5) & (level >= kth)).sum(dim=1)]


def at_cap_batch(mix) -> tuple:
    """probe.gpu_view's batch: items 0-3 of the recipe at seed 777, every bin on the grid raised
    to at least a level of 0.6-0.9, so that the active-bin cap binds (the most a step needs)."""
    import torch

    from lcmsdeconv.synth.dataset import FrameDataset

    ds = FrameDataset(mix, BATCH, base_seed=777)
    items = [ds[i] for i in range(BATCH)]
    batch = [torch.stack([it[j] for it in items]) for j in range(len(NAMES))]
    x = batch[0]
    on_grid = x[:, -1] > 0.5
    level = 0.6 + 0.3 * torch.rand(x[:, 0].shape, generator=torch.Generator().manual_seed(0))
    x[:, 0] = torch.where(on_grid, torch.maximum(x[:, 0], level), x[:, 0])
    return tuple(batch)


def save_batch(path, sample) -> None:
    import torch

    x, topk_z, topk_w, heat = sample
    if int(topk_z.max()) > 32767 or int(topk_z.min()) < -32768:
        raise ValueError("class labels out of the int16 range")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": x.contiguous(), "topk_z": topk_z.to(torch.int16),
                "topk_w": topk_w.contiguous(), "heat": heat.contiguous()}, path)


def load_batch(ctx, name, view: str, pin: bool = False) -> tuple:
    """A saved batch (``name``: its number, or "cap"), as training's loader delivers it for
    ``view`` (the window view has no measured channel), pinned where asked."""
    import torch

    stem = name if isinstance(name, str) else f"b{int(name):03d}"
    d = torch.load(ctx.work / "batches" / f"{stem}.pt", weights_only=True)
    x = d["features"] if view == "comb" else d["features"][:, :-1]
    out = (x.contiguous(), d["topk_z"].to(torch.int64), d["topk_w"], d["heat"])
    return tuple(t.pin_memory() for t in out) if pin else out


def saved_batches(ctx) -> int:
    return len(list((ctx.work / "batches").glob("b*.pt")))


def make_weights(view: str, window: int, size: str = "full") -> dict:
    """probe.measure_speedups' trained-like weights: the untrained network's (seed 0), every
    parameter moved by normal noise of 0.05 (the learned head away from its zero start)."""
    import torch

    from lcmsdeconv.nn.model import build_model

    torch.manual_seed(0)
    model = build_model(size, z_max=100, in_ch=IN_CH[view], arch="ladder", window=window,
                        iso=True, view=view)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model.state_dict()


def prepare(ctx, batches: int = 40, crop: int | None = None, workers: int | None = None,
            size: str = "full", device: str = "cuda") -> dict:
    """The runs' first ``batches`` training batches, the at-cap batch and the weights of both
    views, saved in the work folder for the other GPU experiments."""
    require_cuda(device)
    import torch
    from torch.utils.data import DataLoader

    from lcmsdeconv.nn.machine import cpu_workers
    from lcmsdeconv.nn.train import _warm_up
    from lcmsdeconv.synth.dataset import EpochSampler, FrameDataset, mp_context, worker_init

    window = window_of(crop)
    mix = mixture(crop)
    ctx.work.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(ctx.work).free / 2**30
    asked = int(batches)
    # a batch of 131 072-bin spectra takes about 27 MB on disk; keep 2 GB free besides
    if free_gb < 2.0 + asked * 0.03 * window / 131072:
        batches = max(min(FIRST, asked), min(asked, int((free_gb - 2.0) / 0.03)))
        ctx.say(f"only {free_gb:.1f} GB free on the disk: {batches} batches instead of {asked}")
    t = time.perf_counter()
    _warm_up(mix)  # numba's compiled functions cached once, as training does
    n = int(batches) * BATCH
    w = min(16, cpu_workers()) if workers is None else int(workers)
    kw = (dict(num_workers=w, worker_init_fn=worker_init, prefetch_factor=2, **mp_context(w))
          if w > 0 else {})
    # training's loader: the same dataset, sampler and batches (items 4k .. 4k + 3)
    loader = DataLoader(FrameDataset(mix, n, base_seed=0), batch_size=BATCH,
                        sampler=EpochSampler(n), drop_last=True, **kw)
    rows = []
    for k, sample in enumerate(loader):
        save_batch(ctx.work / "batches" / f"b{k:03d}.pt", sample)
        rows.append(active_rows(sample[0], window))
    made = time.perf_counter() - t
    cap = at_cap_batch(mix)
    save_batch(ctx.work / "batches" / "cap.pt", cap)
    for view in IN_CH:
        torch.save(make_weights(view, window, size), ctx.work / f"weights_{view}.pt")
    size_mb = sum(f.stat().st_size for f in (ctx.work / "batches").glob("*.pt")) / 2**20
    ctx.say(f"{len(rows)} batches made in {made:.0f} s with {w} workers ({size_mb:.0f} MB)")
    return {"config": {"batches": len(rows), "batches_asked": asked,
                       "items": f"0-{4 * len(rows) - 1} of seed 0", "workers": w,
                       "window": window, "size": size, "recipe": recipe(),
                       "disk_free_gb": round(free_gb, 1)},
            "seconds_making": round(made, 1), "saved_mb": round(size_mb, 1),
            "active_rows": rows, "cap_active_rows": active_rows(cap[0], window)}


# ------------------------------------------------------------------ a training step
class Clock:
    """Marks on the device's own timeline: CUDA events on the GPU, read once the work is done;
    the host clock on the processor (where the two are the same)."""

    def __init__(self, device: str):
        import torch

        self.torch = torch
        self.cuda = str(device).startswith("cuda")

    def mark(self):
        if self.cuda:
            e = self.torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def ms(self, a, b) -> float:
        return float(a.elapsed_time(b)) if self.cuda else 1000.0 * (b - a)

    def sync(self) -> None:
        if self.cuda:
            self.torch.cuda.synchronize()


class Trainer:
    """A training step exactly as lcmsdeconv.nn.train takes one: the batch copied to the device
    (from pinned memory, without waiting), forward pass, loss, backward pass, gradient clipping,
    AdamW, the learning-rate schedule, the moving average and the running loss; with the GPU
    speedups ``speedups`` (index_select and fused_adamw here; set_gpu_flags sets the others),
    the head chunking ``chunk``, and bf16 autocast over the head (``"head"``) or the whole
    forward pass (``"forward"``) where asked."""

    def __init__(self, view: str, weights: dict, speedups=(), chunk: int | None = None,
                 device: str = "cuda", size: str = "full", window: int | None = None,
                 autocast: str | None = None, total_steps: int = 100_000):
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR

        from lcmsdeconv.nn.losses import ChargeLoss
        from lcmsdeconv.nn.model import build_model
        from lcmsdeconv.nn.train import EMA, TrainConfig, _RunningLoss

        cfg = TrainConfig()
        self.torch, self.device, self.view = torch, device, view
        self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        model = build_model(size, z_max=cfg.z_max, in_ch=IN_CH[view], arch="ladder",
                            window=window_of(window), iso=True, view=view, head_chunk=chunk)
        model.load_state_dict(weights)
        model = model.to(device).train()
        if "index_select" in speedups:
            model.head_gather = "index_select"  # before the moving average copies the model
        lr = recipe_lr()
        self.opt = AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay,
                         fused=True if "fused_adamw" in speedups else None)
        self.total = int(total_steps)
        self.sched = OneCycleLR(self.opt, max_lr=lr, total_steps=self.total, pct_start=0.2)
        self.ema = EMA(model, cfg.ema_decay)
        self.loss_fn = ChargeLoss(heat_weight=cfg.heat_weight)
        self.running = _RunningLoss(device)
        self.grad_clip = cfg.grad_clip
        self.model = model
        self.params = list(model.parameters())
        self.autocast = autocast
        if autocast == "head":  # after the moving average's copy: the copy is never run
            inner = model._head_rows

            def head_rows(*args, **kwargs):
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    return inner(*args, **kwargs)

            model._head_rows = head_rows
        elif autocast not in (None, "forward"):
            raise ValueError(f"autocast {autocast!r}: head, forward or None")

    def step(self, sample, keep: bool = False, mark=None):
        """One training step on ``sample`` (x, topk_z, topk_w, heat on the host); ``mark(name)``
        is called after each phase; with ``keep`` returns the loss and the gradients before
        clipping (a flat float64 vector), which waits for the GPU."""
        from lcmsdeconv.nn.train import _batch_dict

        torch = self.torch
        mark = mark or (lambda _name: None)
        b = _batch_dict(sample, self.device)
        mark("h2d")
        if self.autocast == "forward":
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                out = self.model(b["features"])
            out = {k: v.float() for k, v in out.items()}
        else:
            out = self.model(b["features"])
        mark("forward")
        losses = self.loss_fn(out, b)
        mark("loss")
        self.opt.zero_grad()
        losses["loss"].backward()
        mark("backward")
        kept = None
        if keep:
            grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                              for p in self.params])
            kept = (float(losses["loss"].detach()), grad.double().cpu().numpy())
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        mark("clip")
        self.opt.step()
        if self.sched.last_epoch < self.total - 1:
            self.sched.step()
        mark("optimiser")
        self.ema.update(self.model)
        self.running.add(losses["loss"])
        mark("ema")
        return kept


def agreement(base: list, other: list) -> dict:
    """How the losses and gradients of ``other``'s first steps agree with ``base``'s (each a
    list of (loss, flat gradient) from the same weights on the same batch): probe.disagreement's
    verdict, with the relative loss difference, the gradients' cosine and the relative
    difference of their norms at each step."""
    from probe import disagreement

    a = {"loss": [c[0] for c in base], "grad": [c[1] for c in base]}
    b = {"loss": [c[0] for c in other], "grad": [c[1] for c in other]}
    rows = []
    for (l0, g0), (l1, g1) in zip(base, other):
        g0, g1 = np.asarray(g0, dtype=np.float64), np.asarray(g1, dtype=np.float64)
        n0, n1 = float(np.linalg.norm(g0)), float(np.linalg.norm(g1))
        rows.append({"loss": l1, "loss_reference": l0,
                     "loss_rel_diff": abs(l1 - l0) / max(abs(l0), 1e-30),
                     "grad_cosine": float(g0 @ g1) / max(n0 * n1, 1e-300) if g0.shape == g1.shape
                     else None, "grad_norm_rel_diff": abs(n1 - n0) / max(n0, 1e-30)})
    why = disagreement(a, b)
    return {"same": why is None, "why": why, "steps": rows}


def _grads_path(ctx, view: str, label: str):
    return ctx.work / "grads" / f"{view}__{label}.npz"


def _save_checks(ctx, view: str, label: str, checks) -> None:
    path = _grads_path(ctx, view, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, loss=np.array([c[0] for c in checks]), grad0=checks[0][1], grad1=checks[1][1])


def _load_checks(ctx, view: str, label: str):
    path = _grads_path(ctx, view, label)
    if not path.exists():
        return None
    with np.load(path) as z:
        return [(float(z["loss"][0]), z["grad0"]), (float(z["loss"][1]), z["grad1"])]


def memory_record() -> dict:
    """The CUDA caching allocator's peaks and counters since the process started (or since the
    last reset of the peaks)."""
    import torch

    def gb(fn):
        try:
            return round(fn() / 2**30, 3)
        except Exception:  # an allocator backend without this statistic (cudaMallocAsync)
            return None

    try:
        s = torch.cuda.memory_stats()
    except Exception:  # the same
        s = {}
    return {"peak_reserved_gb": gb(torch.cuda.max_memory_reserved),
            "peak_allocated_gb": gb(torch.cuda.max_memory_allocated),
            "alloc_retries": int(s.get("num_alloc_retries", 0)),
            "ooms": int(s.get("num_ooms", 0)),
            "device_allocs": s.get("num_device_alloc"), "device_frees": s.get("num_device_free"),
            "sync_all_streams": s.get("num_sync_all_streams")}


class _Setup:
    """What every step experiment starts with: the speedups' process-wide switches (before the
    first convolution), the GPU memory cap, the first batches (pinned) and the view's weights."""

    def __init__(self, ctx, view: str, speedups, device: str, crop, n_batches: int = FIRST,
                 cap: bool = True, pin: bool = True):
        require_cuda(device)
        import torch

        from lcmsdeconv.nn.train import _cap_gpu_memory, set_gpu_flags

        self.torch = torch
        self.cuda = str(device).startswith("cuda")
        self.flags_before = set_gpu_flags(tuple(speedups))
        if self.cuda:
            _cap_gpu_memory(torch, GPU_MEMORY_GB)
        n = min(n_batches, saved_batches(ctx))
        if n < 1:
            raise RuntimeError("no saved batches: the experiment gpu.prepare did not run")
        self.host = [load_batch(ctx, k, view, pin=pin and self.cuda) for k in range(n)]
        self.cap = load_batch(ctx, "cap", view, pin=pin and self.cuda) if cap else None
        self.weights = torch.load(ctx.work / f"weights_{view}.pt", weights_only=True)
        self.window = window_of(crop)
        self.active = [sum(active_rows(b[0], self.window)) for b in self.host]
        self.flags = {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32}

    def done(self) -> None:
        from lcmsdeconv.nn.train import restore_gpu_flags

        restore_gpu_flags(self.flags_before)


def _config(view, speedups, chunk, label, device, size, crop, **more) -> dict:
    return {"view": view, "speedups": list(speedups), "head_chunk": chunk, "label": label,
            "batch": BATCH, "device": device, "size": size, "window": window_of(crop),
            "gpu_memory_gb": GPU_MEMORY_GB, **more}


def _timed(tr: Trainer, clock: Clock, samples, n: int) -> list[tuple]:
    """``n`` steps on ``samples`` cycled, each waited for: (batch, wall ms, GPU ms)."""
    rows = []
    for i in range(n):
        k = i % len(samples)
        clock.sync()
        t0 = time.perf_counter()
        a = clock.mark()
        tr.step(samples[k])
        b = clock.mark()
        clock.sync()
        rows.append((k, 1000.0 * (time.perf_counter() - t0), clock.ms(a, b)))
    return rows


def _free_running(tr: Trainer, clock: Clock, samples, n: int) -> dict:
    """``n`` steps as training runs them, the host waiting for the GPU only where a step does
    (the active-bin nonzero): the wall time per step, and each step's span on the GPU's
    timeline (``gpu_ms``: from one step's start to the next's, so its kernels and the GPU's
    idle time inside it; together they fill the loop, which is why no busy share is taken
    from them: :func:`profile` measures that, from the kernels' own time)."""
    clock.sync()
    t0 = time.perf_counter()
    marks = []
    for i in range(n):
        a = clock.mark()
        tr.step(samples[i % len(samples)])
        marks.append((a, clock.mark()))
    clock.sync()
    wall = 1000.0 * (time.perf_counter() - t0)
    gpu = [clock.ms(a, b) for a, b in marks]
    return {"steps": n, "ms_per_step": round(wall / max(n, 1), 2), "gpu_ms": stats(gpu, 2),
            "spectra_per_second": round(BATCH * n / max(wall / 1000.0, 1e-9), 2)}


def _rows_record(rows, active) -> dict:
    by_batch: dict = {}
    for k, wall, _g in rows:
        by_batch.setdefault(k, []).append(wall)
    return {"wall_ms": stats([r[1] for r in rows], 2), "gpu_ms": stats([r[2] for r in rows], 2),
            "per_batch": [{"batch": k, "active_rows": active[k] if k < len(active) else None,
                           "median_ms": round(statistics.median(v), 2)}
                          for k, v in sorted(by_batch.items())]}


def _oom_types():
    import torch

    return (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else ()


def steps(ctx, view: str, speedups=(), chunk: int | None = None, label: str = "none",
          reference: str = "none", synced: int = 32, free: int = 16, warm: int = 5,
          cap_warm: int = 2, cap_timed: int = 4, check: bool = True, device: str = "cuda",
          size: str = "full", crop: int | None = None) -> dict:
    """The step-time distribution of one configuration (see the module's description)."""
    s = _Setup(ctx, view, speedups, device, crop)
    torch = s.torch
    try:
        tr = Trainer(view, s.weights, speedups, chunk, device, size, crop)
        clock = Clock(device)
        rec = {"config": _config(view, speedups, chunk, label, device, size, crop,
                                 batches=f"the runs' training batches 0-{len(s.host) - 1} cycled",
                                 synced_steps=synced, free_steps=free, warm_steps=warm,
                                 flags=s.flags)}
        if check:
            checks = [tr.step(s.host[0], keep=True) for _ in range(2)]
            _save_checks(ctx, view, label, checks)
            ref = checks if label == reference else _load_checks(ctx, view, reference)
            rec["agreement"] = (agreement(ref, checks) if ref is not None else
                                {"same": None, "why": f"no reference ({reference}) measured"})
            rec["agreement"]["reference"] = reference
        for i in range(warm):
            tr.step(s.host[i % len(s.host)])
        rec["real"] = _rows_record(_timed(tr, clock, s.host, synced), s.active)
        rec["free_running"] = _free_running(tr, clock, s.host, free)
        try:
            for _ in range(cap_warm):
                tr.step(s.cap)
            cap_rows = _timed(tr, clock, [s.cap], cap_timed)
            rec["cap"] = {"wall_ms": stats([r[1] for r in cap_rows], 2),
                          "gpu_ms": stats([r[2] for r in cap_rows], 2),
                          "active_rows": sum(active_rows(s.cap[0], s.window))}
        except _oom_types() as exc:
            rec["cap"] = {"error": f"out of GPU memory: {str(exc)[:300]}"}
            tr.opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if s.cuda:
            rec["memory"] = memory_record()
        rec["median_ms"] = rec["real"]["wall_ms"].get("median")
        return rec
    finally:
        s.done()


#: the phases of a step, in order (Trainer.step's marks)
PHASES = ("h2d", "forward", "loss", "backward", "clip", "optimiser", "ema")


def phases(ctx, view: str, speedups=(), chunk: int | None = None, label: str = "best",
           steps_n: int = 24, warm: int = 5, device: str = "cuda", size: str = "full",
           crop: int | None = None) -> dict:
    """One configuration's step split into phases on the GPU (CUDA events) and on the host,
    the host's wait at the active-bin nonzero, and whether the host keeps the GPU fed."""
    import torch

    s = _Setup(ctx, view, speedups, device, crop, cap=False)
    original = torch.Tensor.nonzero
    stall = {"seconds": 0.0, "calls": 0}

    def timed_nonzero(self, *args, **kwargs):
        t = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            stall["seconds"] += time.perf_counter() - t
            stall["calls"] += 1

    try:
        tr = Trainer(view, s.weights, speedups, chunk, device, size, crop)
        clock = Clock(device)
        for i in range(warm):
            tr.step(s.host[i % len(s.host)])
        clock.sync()
        torch.Tensor.nonzero = timed_nonzero
        per_step = []
        for i in range(steps_n):
            marks = [("start", clock.mark(), time.perf_counter())]
            stall["seconds"], stall["calls"] = 0.0, 0

            def mark(name, marks=marks):
                marks.append((name, clock.mark(), time.perf_counter()))

            tr.step(s.host[i % len(s.host)], mark=mark)
            per_step.append((marks, stall["seconds"], stall["calls"]))
        clock.sync()
    finally:
        torch.Tensor.nonzero = original
        s.done()
    gpu = {p: [] for p in PHASES}
    host = {p: [] for p in PHASES}
    gaps, gpu_total, host_total, host_busy, stalls, calls = [], [], [], [], [], []
    prev_end = None
    for marks, stalled, n_calls in per_step:
        for (_n0, e0, h0), (name, e1, h1) in zip(marks, marks[1:]):
            gpu[name].append(clock.ms(e0, e1))
            host[name].append(1000.0 * (h1 - h0))
        if prev_end is not None:
            gaps.append(clock.ms(prev_end, marks[0][1]))
        prev_end = marks[-1][1]
        gpu_total.append(clock.ms(marks[0][1], marks[-1][1]))
        host_total.append(1000.0 * (marks[-1][2] - marks[0][2]))
        host_busy.append(1000.0 * (marks[-1][2] - marks[0][2] - stalled))
        stalls.append(1000.0 * stalled)
        calls.append(n_calls)
    g, hb = statistics.median(gpu_total), statistics.median(host_busy)
    return {"config": _config(view, speedups, chunk, label, device, size, crop, steps=steps_n,
                              flags=s.flags),
            "phases": {p: {"gpu_ms": stats(gpu[p], 2), "host_ms": stats(host[p], 2)}
                       for p in PHASES},
            "step_gpu_ms": stats(gpu_total, 2), "step_host_ms": stats(host_total, 2),
            "host_busy_ms": stats(host_busy, 2), "nonzero_wait_ms": stats(stalls, 2),
            "nonzero_calls_per_step": stats(calls, 1),
            "gpu_gap_between_steps_ms": stats(gaps, 3),
            "host_busy_over_gpu": round(hb / max(g, 1e-9), 3),
            "verdict": ("launch-bound: the host takes about as long to queue a step as the GPU "
                        "to run it" if hb >= 0.9 * g else
                        "GPU-bound: the host queues a step faster than the GPU runs it")}


def profile(ctx, view: str, speedups=(), chunk: int | None = None, label: str = "none",
            steps_n: int = 3, warm: int = 5, top: int = 30, trace: bool = True,
            device: str = "cuda", size: str = "full", crop: int | None = None) -> dict:
    """torch.profiler over ``steps_n`` steps run as training runs them: the ``top`` CUDA kernels
    by total time, the ``top`` aten operators by their own CUDA time (with input shapes for the
    first ten), the CUDA runtime calls per step (launches, synchronisations, copies), the
    kernels' time against the steps' wall time (``gpu_busy_share``: the share of a step the GPU
    runs kernels, under the profiler, whose own work slows the host, so a lower bound where the
    host sets the pace), and a trace of one step (traces/<id>.json.gz, for chrome://tracing or
    Perfetto). A trace that cannot be written is recorded (``trace_error``); the tables are the
    result all the same."""
    from probe import kernel_rows
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    s = _Setup(ctx, view, speedups, device, crop, cap=False)
    try:
        tr = Trainer(view, s.weights, speedups, chunk, device, size, crop)
        clock = Clock(device)
        for i in range(warm):
            tr.step(s.host[i % len(s.host)])
        clock.sync()
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if s.cuda else [])
        with torch_profile(activities=acts, record_shapes=True) as prof:
            t0 = time.perf_counter()
            for i in range(steps_n):
                tr.step(s.host[i % len(s.host)])
            clock.sync()
            wall_ms = 1000.0 * (time.perf_counter() - t0)
        averages = prof.key_averages()
        trace_file = trace_error = None
        if trace:
            try:
                with torch_profile(activities=acts, record_shapes=True) as one:
                    tr.step(s.host[0])
                    clock.sync()
                trace_file = export_trace(one, ctx.path("traces", f"{ctx.id}.json.gz"),
                                          ctx.path("work", f"{ctx.id}_trace.json"))
            except Exception as exc:  # the trace is a convenience: the tables are the result
                trace_error = f"{type(exc).__name__}: {exc}"[:500]
                ctx.say(f"the trace was not written ({trace_error[:150]}); the tables are kept")
    finally:
        s.done()

    def device_us(e, own: bool) -> float:
        for name in (("self_device_time_total", "self_cuda_time_total") if own else
                     ("device_time_total", "cuda_time_total")):
            v = getattr(e, name, None)
            if v is not None:
                return float(v)
        return 0.0

    ops = []
    runtime: dict = {}
    for e in averages:
        if getattr(e, "device_type", None) == DeviceType.CUDA:
            continue
        key = str(e.key)
        if key.startswith("aten::"):
            ops.append({"name": key, "self_cuda_ms": round(device_us(e, True) / 1000.0, 3),
                        "cuda_total_ms": round(device_us(e, False) / 1000.0, 3),
                        "self_cpu_ms": round(float(e.self_cpu_time_total) / 1000.0, 3),
                        "calls": int(e.count)})
        elif key.startswith(("cuda", "cu")) and not key.startswith("cudnn"):
            runtime[key] = {"calls_per_step": round(e.count / steps_n, 1),
                            "cpu_ms_per_step": round(float(e.cpu_time_total) / 1000.0 / steps_n,
                                                     3)}
    ops.sort(key=lambda r: (-r["self_cuda_ms"], -r["self_cpu_ms"]))
    top_names = [r["name"] for r in ops[:10]]
    shapes: dict = {}
    for e in prof.key_averages(group_by_input_shape=True):
        if str(e.key) in top_names and getattr(e, "device_type", None) != DeviceType.CUDA:
            shapes.setdefault(str(e.key), []).append({
                "shapes": str(getattr(e, "input_shapes", ""))[:300],
                "self_cuda_ms": round(device_us(e, True) / 1000.0, 3), "calls": int(e.count)})
    for name in shapes:
        shapes[name] = sorted(shapes[name], key=lambda r: -r["self_cuda_ms"])[:3]
    kernels = kernel_rows(averages, top) if s.cuda else []
    kernel_ms = sum(device_us(e, True) for e in averages
                    if getattr(e, "device_type", None) == DeviceType.CUDA) / 1000.0
    launches = sum(int(e.count) for e in averages
                   if getattr(e, "device_type", None) == DeviceType.CUDA)
    ms_per_step = wall_ms / max(steps_n, 1)
    return {"config": _config(view, speedups, chunk, label, device, size, crop, steps=steps_n,
                              flags=s.flags),
            "kernels": kernels, "ops": ops[:top], "op_shapes": shapes,
            "runtime_calls": dict(sorted(runtime.items(), key=lambda kv: -kv[1]["calls_per_step"])
                                  [:20]),
            "ms_per_step": round(ms_per_step, 2),
            "kernel_ms_per_step": round(kernel_ms / steps_n, 2),
            "gpu_busy_share": (round(kernel_ms / steps_n / max(ms_per_step, 1e-9), 4)
                               if s.cuda else None),
            "kernels_per_step": round(launches / steps_n, 1),
            "trace": None if trace_file is None else f"traces/{trace_file.name}",
            "trace_error": trace_error}


def export_trace(prof, path, raw=None):
    """``prof``'s trace written to ``path`` (.json.gz) by PyTorch's Python exporter, which opens
    the file with Python's own functions and so takes any path (the native exporter takes a
    narrow-string path, which a folder named in another script can defeat on Windows); where
    that fails (a PyTorch without it, for one), by the native exporter to ``raw`` (.json), then
    compressed. Returns ``path``; OSError naming both failures where no trace was written."""
    import gzip

    path = Path(path)
    try:
        prof.export_chrome_trace(str(path), use_python_export=True)
        if path.exists() and path.stat().st_size > 0:
            return path
        first = "it wrote nothing"
    except Exception as exc:  # TypeError: a PyTorch without the option
        first = f"{type(exc).__name__}: {exc}"
    raw = Path(raw) if raw is not None else path.with_suffix("")
    try:
        prof.export_chrome_trace(str(raw))
        with open(raw, "rb") as src, gzip.open(path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        raw.unlink()
    except Exception as exc:
        raise OSError(f"the Python exporter: {first[:200]}; the native one: "
                      f"{type(exc).__name__}: {exc}") from exc
    if not path.exists() or path.stat().st_size == 0:
        raise OSError(f"no trace was written to {path.name}")
    return path


def memory(ctx, view: str, speedups=(), chunk: int | None = None, label: str = "best",
           variant: str = "default", batches: int = 40, device: str = "cuda", size: str = "full",
           crop: int | None = None) -> dict:
    """Memory over ``batches`` real batches in order, under the cap, with the allocator settings
    of PYTORCH_CUDA_ALLOC_CONF (set by the runner): peaks, retries, out-of-memory errors (the
    step is skipped and the run goes on), the memory reserved after each step, step times, and
    any warning the settings gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = _Setup(ctx, view, speedups, device, crop, n_batches=batches, cap=False, pin=False)
        torch = s.torch
        try:
            tr = Trainer(view, s.weights, speedups, chunk, device, size, crop)
            clock = Clock(device)
            if s.cuda:
                torch.cuda.reset_peak_memory_stats()
            reserved, allocated, walls, oom_steps = [], [], [], []
            for k, sample in enumerate(s.host):
                try:
                    clock.sync()
                    t0 = time.perf_counter()
                    tr.step(sample)
                    clock.sync()
                    walls.append(1000.0 * (time.perf_counter() - t0))
                except _oom_types():
                    oom_steps.append(k)
                    tr.opt.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                if s.cuda:
                    reserved.append(round(torch.cuda.memory_reserved() / 2**30, 3))
                    allocated.append(round(torch.cuda.memory_allocated() / 2**30, 3))
            rec = {"config": _config(view, speedups, chunk, label, device, size, crop,
                                     variant=variant, batches=len(s.host),
                                     alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                                     flags=s.flags),
                   "step_ms": stats(walls[2:] or walls, 2), "oom_steps": oom_steps,
                   "reserved_gb_after_step": reserved, "allocated_gb_after_step": allocated}
            if s.cuda:
                rec.update(memory_record())
                rec["allocator_backend"] = getattr(torch.cuda, "get_allocator_backend",
                                                   lambda: None)()
        finally:
            s.done()
    rec["warnings"] = sorted({f"{w.category.__name__}: {w.message}"[:300] for w in caught})
    return rec


def _checks_and_times(tr: Trainer, s: _Setup, clock: Clock, warm: int, n: int) -> tuple:
    checks = [tr.step(s.host[0], keep=True) for _ in range(2)]
    for i in range(warm):
        tr.step(s.host[i % len(s.host)])
    return checks, _timed(tr, clock, s.host, n)


def autocast(ctx, view: str, speedups=(), chunk: int | None = None, label: str = "best",
             modes=("head", "forward"), steps_n: int = 16, warm: int = 3,
             device: str = "cuda", size: str = "full", crop: int | None = None) -> dict:
    """For information (bf16 changes the numbers): float32 and bf16 autocast over the head or
    the whole forward pass, each from the same weights on the same batch: losses and gradients
    against float32 (probe.disagreement), step times and memory peaks."""
    s = _Setup(ctx, view, speedups, device, crop, cap=False)
    torch = s.torch
    out = {"config": _config(view, speedups, chunk, label, device, size, crop, steps=steps_n,
                             modes=list(modes), flags=s.flags), "modes": {}}
    try:
        clock = Clock(device)
        ref = None
        for mode in (None, *modes):
            name = mode or "float32"
            try:
                if s.cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                tr = Trainer(view, s.weights, speedups, chunk, device, size, crop,
                             autocast=mode)
                checks, rows = _checks_and_times(tr, s, clock, warm, steps_n)
                rec = {"wall_ms": stats([r[1] for r in rows], 2),
                       "gpu_ms": stats([r[2] for r in rows], 2)}
                if s.cuda:
                    rec["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                if mode is None:
                    ref = checks
                elif ref is not None:
                    rec["agreement"] = agreement(ref, checks)
                out["modes"][name] = rec
                del tr
            except Exception as exc:  # a mode that fails is recorded; the next one runs
                out["modes"][name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                if s.cuda:
                    torch.cuda.empty_cache()
    finally:
        s.done()
    return out


def batch8(ctx, view: str, speedups=(), chunk: int | None = None, label: str = "best",
           steps_n: int = 16, warm: int = 3, device: str = "cuda", size: str = "full",
           crop: int | None = None) -> dict:
    """For information: spectra per second at 8 spectra a step (two consecutive real batches
    each) against 4, in the same process; on running out of GPU memory the larger batch is
    measured again 16 charges at a time."""
    s = _Setup(ctx, view, speedups, device, crop, n_batches=2 * FIRST, cap=False)
    torch = s.torch
    out = {"config": _config(view, speedups, chunk, label, device, size, crop, steps=steps_n,
                             flags=s.flags), "batches": {}}
    pairs = [tuple(torch.cat([a, b]) for a, b in zip(s.host[i], s.host[i + 1]))
             for i in range(0, len(s.host) - 1, 2)]
    if s.cuda:
        pairs = [tuple(t.pin_memory() for t in p) for p in pairs]
    clock = Clock(device)

    def measure(bs: int, samples, use_chunk) -> dict:
        """One batch size with one chunking; the trainer lives in this frame only, so that an
        attempt out of GPU memory leaves nothing behind for the next."""
        if s.cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        tr = Trainer(view, s.weights, speedups, use_chunk, device, size, crop)
        for i in range(warm):
            tr.step(samples[i % len(samples)])
        rows = _timed(tr, clock, samples, steps_n)
        med = statistics.median(r[1] for r in rows)
        rec = {"head_chunk": use_chunk, "wall_ms": stats([r[1] for r in rows], 2),
               "spectra_per_second": round(bs * 1000.0 / med, 2)}
        if s.cuda:
            rec.update(memory_record())
        return rec

    try:
        for bs, samples in ((BATCH, s.host[:FIRST]), (2 * BATCH, pairs)):
            for use_chunk in dict.fromkeys([chunk, 16]):
                try:
                    out["batches"][str(bs)] = measure(bs, samples, use_chunk)
                    break
                except _oom_types():
                    out["batches"][str(bs)] = {"head_chunk": use_chunk,
                                               "error": "out of GPU memory"}
                    gc.collect()
    finally:
        s.done()
    return out


def smi_loop(ctx, view: str, speedups=(), chunk: int | None = None, label: str = "best",
             seconds: float = 60.0, warm: int = 5, interval: float = 0.5, device: str = "cuda",
             size: str = "full", crop: int | None = None) -> dict:
    """A steady loop of one configuration for ``seconds``, as training runs (no waiting between
    steps), with nvidia-smi read every ``interval`` s: the pace, and utilisation, clocks, power,
    temperature, memory, performance state and clock-event reasons summarised (the samples in
    csv/<id>.csv)."""
    from .smi import Sampler, summarise

    s = _Setup(ctx, view, speedups, device, crop, cap=False)
    try:
        tr = Trainer(view, s.weights, speedups, chunk, device, size, crop)
        clock = Clock(device)
        for i in range(warm):
            tr.step(s.host[i % len(s.host)])
        clock.sync()
        sampler = Sampler(interval=interval)
        if sampler.smi is None:
            ctx.say("nvidia-smi was not found: the loop runs without its readings")
        sampler.start()
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < seconds:
            tr.step(s.host[n % len(s.host)])
            n += 1
        clock.sync()
        wall = time.perf_counter() - t0
        rows = sampler.stop()
    finally:
        s.done()
    if rows:
        sampler.write_csv(ctx.path("csv", f"{ctx.id}.csv"))
    return {"config": _config(view, speedups, chunk, label, device, size, crop, seconds=seconds,
                              interval=interval, flags=s.flags),
            "steps": n, "seconds": round(wall, 2), "ms_per_step": round(1000.0 * wall / max(n, 1), 2),
            "spectra_per_second": round(BATCH * n / max(wall, 1e-9), 2),
            "nvidia_smi": summarise(rows, list(sampler.fields)),
            "fields": sampler.fields, "smi_errors": sampler.errors,
            "csv": f"csv/{ctx.id}.csv" if rows else None}
