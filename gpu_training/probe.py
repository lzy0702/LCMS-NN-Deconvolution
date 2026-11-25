"""Measure this computer for training, within the given memory limits.

Runs real training steps of the full-size networks of both views (the runs' recipe,
train_all.RECIPE, with isotope-charge channels, at the recipe's batch size, every spectrum at
the active-bin cap: the most memory a step can need) on the GPU under the GPU memory cap,
scoring the charges in chunks and all at once (train_all.py chooses between the two from these
measurements), then starts data workers the way training does on the recipe's spectra and
measures how much memory each one takes and how many spectra per second they deliver together.
It chooses as many workers as the processor is given (one per logical processor of every core
but the one or two the training process keeps, lcmsdeconv.nn.machine.cpu_workers) where the RAM
budget holds them with the training process, fewer where it does not, and measures their speed
at the number it chooses: on a processor with performance and efficiency cores each further
worker adds less, and torch's DataLoader delivers batches in order, so that a worker that is
ahead waits for the slower ones. The RAM budget is by default half the computer's memory, at
least 16 GB and at most 48 (lcmsdeconv.nn.machine).

Then it measures the GPU speedups (lcmsdeconv.nn.train.GPU_SPEEDUPS: cuDNN's benchmark, TF32
matrix products, LadderNet's index_select head, fused AdamW), which change only the rounding of
the computation, on the runs' first training batch in each view, with the head scoring
train_all.HEAD_CHUNK charges at a time (the only way runs use speedups: all charges at once with
them filled the GPU memory cap on the runs' varied batches, train_all.head_chunk_for), each
configuration in a process of its own (_measure_in_child): a speedup is kept for a view if a
step with it alone is at least 3 % faster than one without any and its loss and gradients agree
with that step's; the kept ones are combined greedily, fastest first, each further one only if
it makes the step at least 2 % faster again (choose_speedups). A profile of the window view's
step with its speedups lists the CUDA kernels that take the most time. Only then does it choose
each view's head chunking, from the speedups chosen. Everything goes to
gpu_training/machine.json, which train_all.py reads. The pace it reports compares the workers'
speed with the GPU's on the runs' first training batch (the speedup measurement), and gives the
GPU's step with every spectrum at the active-bin cap as the bound.

Last it measures how that many workers deliver the most spectra (data_loading_test, a few
minutes): torch's DataLoader as before against the ordered loader (lcmsdeconv.synth.pool),
with the workers bound to cores or not, opted out of Windows' power throttling or not, at
below-normal or normal priority, while this process does with every batch what training does (a
real training step of the window view on the GPU). The ordered loader's workers start once and
switch from way to way while they run; every way is timed twice in mirrored order on the same
spectra. Training keeps the way as before unless another delivers at least DATA_MARGIN more
beyond the standard error of the comparison (from how far apart each way's two windows are),
each further change again DATA_MARGIN more than the ways that change less (choose_data_loading),
and keeps it too where the computer was busy meanwhile; machine.json's "data_loader" records the
choice, "data_loader_test" the measurements, and the workers' rate there is the one the chosen
way delivered.

    python gpu_training/probe.py [--ram-gb N] [--gpu-memory-gb 14] [--workers N]
                                 [--no-data-test]

setup.py runs it; run it again on its own to re-measure (for example after changing the limits).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "gpu_training"))

#: memory the training process holds for its 128 validation spectra (about 7 MB each)
VALIDATION_GB = 0.9
#: the GPU speedups: one is kept for a view if a training step with it alone is at least
#: SINGLE_GAIN faster than the step without any, and the two agree (below); the kept ones are
#: then combined, fastest first, each further one only if the combination gets at least
#: COMBINED_GAIN faster again and still agrees
SINGLE_GAIN, COMBINED_GAIN = 0.03, 0.02
#: agreement with the step without speedups, from the same weights on the same batch: the
#: relative difference of the losses, the cosine of the gradients and the relative difference
#: of their norms. TensorFloat-32 rounding stays well inside these; a change of what is computed
#: does not
LOSS_RTOL, MIN_COSINE, NORM_RTOL = 1e-3, 0.999, 1e-2
#: training steps of each configuration: two whose losses and gradients are compared (the first
#: from the shared weights, the second after one optimiser step), WARM_STEPS more, then
#: TIMED_STEPS timed one by one (their median counts)
WARM_STEPS, TIMED_STEPS = 2, 8
#: seconds a configuration's measurement may take before it counts as failed
SPEEDUP_TIMEOUT = 900
#: the CUDA kernel table: steps profiled, kernels kept
PROFILE_STEPS, PROFILE_TOP = 3, 15
#: the data workers' measurement (workers_run): WORKER_WARM_ROUNDS batches from every worker
#: before the timing starts (the start-up, and the head start of the workers that started
#: first), then the timed part, which ends with the first batch after at least WORKER_ROUNDS
#: rounds and WORKER_SECONDS, or after WORKER_MAX_SECONDS and one round
WORKER_WARM_ROUNDS, WORKER_ROUNDS, WORKER_SECONDS, WORKER_MAX_SECONDS = 2, 3, 60.0, 180.0
#: seconds a batch may take to come from the workers before they count as failed to run
WORKER_TIMEOUT = 600.0
#: seconds between two readings of the workers' memory
MEMORY_EVERY = 5.0


def process_gb(proc) -> float:
    """RAM of one process as the task manager shows it (its private working set)."""
    from lcmsdeconv.nn.machine import process_ram_gb

    return process_ram_gb(proc)


def commit_gb(proc) -> float:
    """Memory Windows has promised the process (private bytes), reserved or used; only for the
    record: it is not RAM, but RAM plus the paging file must hold it."""
    return getattr(proc.memory_info(), "private", 0) / 2**30


def recipe_mixture(measured: bool = True) -> list:
    """The training spectra of train_all's runs (its RECIPE), as training configures them (singly
    charged ions labelled, isotope-charge channels, and the comb view's measured channel)."""
    import train_charge_net
    from train_all import RECIPE

    mix = getattr(train_charge_net, f"mixture_{RECIPE}")()
    for _w, c in mix:
        c.singly_charged = "keep"
        c.iso_channels = True
        c.measured_channel = measured
    return mix


def gpu_view(view: str, batch_size: int, head_chunk: int | None) -> dict:
    """Peak GPU memory and time of a training step of the full network of ``view`` on spectra of
    the recipe, every one at the active-bin cap (distinct levels above the active level on all
    its grid bins)."""
    import torch

    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.synth.dataset import FrameDataset

    ds = FrameDataset(recipe_mixture(), batch_size, base_seed=777)
    items = [ds[i] for i in range(batch_size)]
    names = ("features", "topk_z", "topk_w", "heat")
    batch = {k: torch.stack([it[j] for it in items]) for j, k in enumerate(names)}
    x = batch["features"]
    on_grid = x[:, -1] > 0.5
    level = 0.6 + 0.3 * torch.rand(x[:, 0].shape, generator=torch.Generator().manual_seed(0))
    x[:, 0] = torch.where(on_grid, torch.maximum(x[:, 0], level), x[:, 0])
    if view == "window":
        x = x[:, :-1]  # the window view has no measured channel
    batch["features"] = x.contiguous()
    batch = {k: v.cuda() for k, v in batch.items()}
    model = build_model("full", z_max=100, in_ch=x.shape[1], arch="ladder", window=131072,
                        iso=True, view=view, head_chunk=head_chunk).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = ChargeLoss()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(6):
        torch.cuda.synchronize()
        t = time.time()
        loss = loss_fn(model(batch["features"]), batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    out = {"batch": batch_size, "peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
           "allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "step_seconds": round(statistics.median(times[2:]), 4)}
    del model, opt, batch, x
    torch.cuda.empty_cache()
    return out


def gpu_memory_now() -> tuple[float, float] | None:
    """The GPU's memory free and in all now (GB, torch.cuda.mem_get_info), or None where it
    cannot be read. On a GPU that drives the display, Windows' desktop and other programs hold
    part of it, and so does this process's CUDA context once the call has made it; machine.json
    records the free share at the start of setup's GPU measurement, for information."""
    import torch

    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:  # no CUDA, or a driver that cannot say
        return None
    return round(free / 2**30, 2), round(total / 2**30, 2)


def gpu_steps(gpu_memory_gb: float, batch_size: int, head_chunk: int | None) -> dict:
    """Both views at ``batch_size`` (halved until a step fits under the cap) in chunks of
    ``head_chunk`` charges, then at the batch that fit with all charges at once
    (``"unchunked"``)."""
    import torch

    from lcmsdeconv.nn.train import _cap_gpu_memory, _warm_up

    _cap_gpu_memory(torch, gpu_memory_gb)
    _warm_up(recipe_mixture())
    views = {}
    for view in ("window", "comb"):
        b = batch_size
        while b >= 1:
            try:
                views[view] = gpu_view(view, b, head_chunk)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  the {view} view does not fit at {b} spectra per step under the cap",
                      flush=True)
                b //= 2
        if view not in views:
            views[view] = {"batch": batch_size, "peak_gb": None, "step_seconds": None,
                           "error": "out of GPU memory even at one spectrum per step"}
            continue
        whole = None
        try:
            whole = gpu_view(view, views[view]["batch"], None)
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        views[view]["unchunked"] = whole or {
            "batch": views[view]["batch"], "peak_gb": None, "step_seconds": None,
            "error": "out of GPU memory"}
    props = torch.cuda.get_device_properties(0)
    return {"gpu": props.name, "gpu_total_gb": round(props.total_memory / 2**30, 1),
            "gpu_views": views, "torch": torch.__version__, "torch_cuda": torch.version.cuda}


def workers_run(n: int, batch_size: int = 4, dataset=None) -> dict:
    """Start ``n`` data workers as training does (the recipe's spectra in batches of
    ``batch_size``, delivered in order with two batches in work per worker, at lowered priority,
    into pinned memory where CUDA is available) and measure how many spectra per second they
    deliver together and how much memory each one takes. The timing starts once every worker
    has delivered WORKER_WARM_ROUNDS batches, so that neither the start-up nor the head start of
    the workers that started first counts, and stops while every worker is still busy, at
    least WORKER_ROUNDS rounds and WORKER_SECONDS later (after WORKER_MAX_SECONDS and one round
    at most). ``dataset`` stands in for the recipe's spectra (tests)."""
    import psutil
    import torch
    from torch.utils.data import DataLoader

    from lcmsdeconv.synth.dataset import FrameDataset, mp_context, worker_init

    if dataset is None:  # far more spectra than are read: the measurement stops in full flow
        dataset = FrameDataset(recipe_mixture(), n * batch_size * 100_000, base_seed=4242)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n, worker_init_fn=worker_init,
                        prefetch_factor=2, timeout=WORKER_TIMEOUT,
                        pin_memory=torch.cuda.is_available(), **mp_context(n))
    me = psutil.Process()
    memory = {"worker": 0.0, "commit": 0.0, "total": 0.0}

    def read_memory():
        sizes = []
        for p in me.children(recursive=True):
            try:
                sizes.append((process_gb(p), commit_gb(p)))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if sizes:
            memory["worker"] = max(memory["worker"], max(a for a, _b in sizes))
            memory["commit"] = max(memory["commit"], max(b for _a, b in sizes))
            memory["total"] = max(memory["total"], sum(a for a, _b in sizes))

    started = time.time()
    up = t0 = None  # every worker has delivered a batch; the timing starts
    k0 = spectra0 = spectra_up = 0
    k = -1
    now = last_read = started
    spectra = 0
    batches = iter(loader)
    try:
        for k, batch in enumerate(batches):
            now = time.time()  # before anything else this process does
            spectra += len(batch[0] if isinstance(batch, (list, tuple)) else batch)
            if k == n - 1:
                up, spectra_up = now, spectra
            if k == WORKER_WARM_ROUNDS * n - 1:
                t0, k0, spectra0 = now, k, spectra
            elif t0 is not None:
                rounds, seconds = (k - k0) / n, now - t0
                if ((rounds >= WORKER_ROUNDS and seconds >= WORKER_SECONDS)
                        or (rounds >= 1 and seconds >= WORKER_MAX_SECONDS)):
                    break
            if now - last_read >= MEMORY_EVERY or k == n - 1:
                read_memory()
                last_read = time.time()
        read_memory()  # in full flow
    finally:
        stop = getattr(batches, "_shutdown_workers", None)
        if stop is not None:
            stop()  # each worker finishes the batch it is making, then ends
        del batches
    if t0 is None or k <= k0:  # the spectra ran out before the timed part (a short dataset)
        t0, spectra0 = (up, spectra_up) if up is not None and k > n - 1 else (started, 0)
    seconds = max(now - t0, 1e-9)
    return {"workers": n, "startup_seconds": round((up or now) - started, 1),
            "timed_seconds": round(seconds, 1), "timed_spectra": spectra - spectra0,
            "spectra_per_second": round((spectra - spectra0) / seconds, 2),
            "worker_gb": round(memory["worker"], 2), "worker_commit_gb": round(memory["commit"], 2),
            "workers_total_gb": round(memory["total"], 2)}


# ------------------------------------------------------------------ the GPU speedups
def config_name(config) -> str:
    """``("tf32", "index_select")`` -> "tf32+index_select" (in GPU_SPEEDUPS order); () -> "none":
    the names of machine.json's speedup_timings, which train_all looks up (speedup_name)."""
    from train_all import speedup_name

    return speedup_name(config)


def disagreement(base: dict, other: dict) -> str | None:
    """Why the training steps measured in ``other`` do not agree with those of ``base`` (the
    step without speedups), or None: each has the ``loss`` and ``grad`` (a flat vector) of the
    same steps from the same weights on the same batch."""
    import numpy as np

    for k, (l0, l1, g0, g1) in enumerate(zip(base["loss"], other["loss"], base["grad"],
                                             other["grad"]), start=1):
        if not (math.isfinite(l1) and abs(l1 - l0) <= LOSS_RTOL * abs(l0)):
            return f"loss {l1:.6g} against {l0:.6g} at step {k}"
        g0, g1 = np.asarray(g0, dtype=np.float64), np.asarray(g1, dtype=np.float64)
        if g0.shape != g1.shape:
            return f"gradients of {g1.size} values against {g0.size} at step {k}"
        n0, n1 = float(np.linalg.norm(g0)), float(np.linalg.norm(g1))
        if not math.isfinite(n1):
            return f"gradients not finite at step {k}"
        if n0 == 0.0 and n1 == 0.0:
            continue
        cosine = float(g0 @ g1) / max(n0 * n1, 1e-300)
        if cosine < MIN_COSINE:
            return f"gradient cosine {cosine:.5f} at step {k}"
        if abs(n1 - n0) > NORM_RTOL * n0:
            return f"gradient norm {n1:.5g} against {n0:.5g} at step {k}"
    return None


def choose_speedups(measure, toggles=None) -> dict:
    """The selection rule for one view. ``measure(config)`` runs training steps with the speedups
    ``config`` (a tuple of names) on and returns ``{"ms": median milliseconds of a step,
    "loss": [...], "grad": [...]}`` (see :func:`disagreement`); it may raise, which rules that
    configuration out. The baseline is the step without speedups; each speedup alone is kept if
    it is at least SINGLE_GAIN faster and agrees with the baseline; the kept ones are combined
    greedily, fastest first, each further one only if the combination is at least COMBINED_GAIN
    faster than the one before and still agrees. Returns the speedups chosen (GPU_SPEEDUPS
    order), every configuration's median ms (``timings``, by :func:`config_name`) and why each
    was kept or not (``notes``)."""
    from lcmsdeconv.nn.train import GPU_SPEEDUPS

    toggles = tuple(GPU_SPEEDUPS if toggles is None else toggles)
    timings: dict = {}
    notes: dict = {}

    def run(config):
        name = config_name(config)
        try:
            m = measure(tuple(config))
            ms = float(m["ms"])
            if not (math.isfinite(ms) and ms > 0.0):
                raise ValueError(f"a step of {ms} ms")
        except Exception as exc:  # any failure rules the configuration out
            notes[name] = f"failed: {type(exc).__name__}: {exc}"[:400]
            return None
        timings[name] = round(ms, 2)
        return m

    def verdict(m, ref, gain_needed, than=""):
        gain = 1.0 - float(m["ms"]) / float(ref["ms"])
        why = disagreement(base, m)
        faster = (f"{100 * gain:.1f} % faster" if gain >= 0 else f"{-100 * gain:.1f} % slower"
                  ) + than
        if why:
            return False, f"{faster} but not the same: {why}"
        if gain < gain_needed:
            return False, f"{faster}: under {100 * gain_needed:.0f} %"
        return True, f"{faster}, the same loss and gradients"

    base = run(())
    if base is None:
        return {"chosen": [], "timings": timings, "notes": notes}
    notes["none"] = "the reference"
    singles = []
    for t in toggles:
        m = run((t,))
        if m is None:
            continue
        ok, notes[t] = verdict(m, base, SINGLE_GAIN)
        if ok:
            singles.append((float(m["ms"]), toggles.index(t), t, m))
    singles.sort()
    chosen, best = [], None
    for _ms, _i, t, m in singles:
        if not chosen:
            chosen, best = [t], m
            notes[t] += ": kept (the fastest alone)"
            continue
        config = (*chosen, t)
        mc = run(config)
        if mc is None:
            continue
        ok, text = verdict(mc, best, COMBINED_GAIN, f" than {config_name(chosen)}")
        notes[config_name(config)] = text + (": kept" if ok else "")
        if ok:
            chosen.append(t)
            best = mc
    return {"chosen": [s for s in GPU_SPEEDUPS if s in chosen], "timings": timings,
            "notes": notes}


def _training_step(view: str, batch: dict, chunk: int | None, weights: dict, config, lr: float):
    """A training step of the full network of ``view`` as training takes one (loss, backward
    pass, gradient clipping, AdamW, the moving average), with the speedups ``config`` and the
    head chunking ``chunk``, from ``weights``, on the GPU. ``step(keep=True)`` returns the loss
    and the gradients (before clipping) as a flat float64 vector."""
    import torch
    from train_charge_net import V3_WINDOW

    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import EMA, TrainConfig

    cfg = TrainConfig()
    x = batch["features"] if view == "comb" else batch["features"][:, :-1]
    b = {k: v.cuda() for k, v in {**batch, "features": x.contiguous()}.items()}
    model = build_model("full", z_max=cfg.z_max, in_ch=x.shape[1], arch="ladder",
                        window=V3_WINDOW, iso=True, view=view, head_chunk=chunk)
    model.load_state_dict(weights)
    model = model.cuda().train()
    if "index_select" in config:
        model.head_gather = "index_select"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay,
                            fused=True if "fused_adamw" in config else None)
    ema = EMA(model, cfg.ema_decay)
    loss_fn = ChargeLoss(heat_weight=cfg.heat_weight)
    params = list(model.parameters())

    def step(keep: bool = False):
        losses = loss_fn(model(b["features"]), b)
        opt.zero_grad()
        losses["loss"].backward()
        kept = None
        if keep:
            grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                              for p in params])
            kept = (float(losses["loss"].detach()), grad.double().cpu().numpy())
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        ema.update(model)
        return kept

    return step


def speedup_steps(view: str, batch: dict, chunk: int | None, weights: dict, config,
                  lr: float) -> dict:
    """One configuration measured for ``view``: the loss and gradients of its first two training
    steps from ``weights``, then the median time of TIMED_STEPS steps after WARM_STEPS more
    (cuDNN's benchmark, if on, runs in the first), and the memory it took: the most the tensors
    held (``peak_gb``) and the most the caching allocator reserved (``reserved_gb``), from an
    empty cache, so that a view measured after another in the same process counts its own."""
    import gc

    import torch

    gc.collect()  # the network of a view measured before, should anything still hold it
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step = _training_step(view, batch, chunk, weights, config, lr)
    checks = [step(keep=True) for _ in range(2)]
    for _ in range(WARM_STEPS):
        step()
    times = []
    for _ in range(TIMED_STEPS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return {"ms": 1000.0 * statistics.median(times), "loss": [c[0] for c in checks],
            "grad": [c[1] for c in checks],
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


def kernel_rows(averages, top: int = PROFILE_TOP) -> list[dict]:
    """The ``top`` CUDA kernels (and copies) of a profile's averages (``key_averages()``) by
    their total time: name, total ms, calls and share of the time of all of them."""
    from torch.autograd import DeviceType

    rows = []
    for e in averages:
        if getattr(e, "device_type", None) != DeviceType.CUDA:
            continue  # host-side events carry their kernels' time too
        us = getattr(e, "self_device_time_total", None)
        if us is None:
            us = getattr(e, "self_cuda_time_total", 0.0)
        if us and us > 0:
            rows.append((str(e.key), float(us) / 1000.0, int(e.count)))
    total = sum(r[1] for r in rows) or 1.0
    rows.sort(key=lambda r: -r[1])
    return [{"name": name, "total_ms": round(ms, 3), "calls": calls, "share": round(ms / total, 4)}
            for name, ms, calls in rows[:top]]


def cuda_kernels(view: str, batch: dict, chunk: int | None, weights: dict, config,
                 lr: float) -> dict:
    """The CUDA kernels of PROFILE_STEPS training steps of ``view`` with the speedups ``config``
    (after two steps unprofiled), the PROFILE_TOP that take the most time."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    step = _training_step(view, batch, chunk, weights, config, lr)
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILE_STEPS):
            step()
        torch.cuda.synchronize()
    return {"view": view, "speedups": list(config), "steps": PROFILE_STEPS,
            "kernels": kernel_rows(prof.key_averages())}


def _result_base(tmp: Path, name: str, view: str, profile: bool) -> Path:
    """Where :func:`speedup_child` leaves the result of configuration ``name`` for ``view``
    (``.npz`` or ``.json``, or ``.err`` with the error that stopped it)."""
    return tmp / f"{'profile_' if profile else ''}{name}__{view}"


def speedup_child(tmp: Path, name: str, views: list[str], profile: bool) -> int:
    """``probe.py --speedup-child``: measure one configuration (or profile it) for ``views`` in
    this process, from the batch and weights the probe left in ``tmp``; each result, or the
    error that stopped it, goes to a file there."""
    import numpy as np
    import torch

    from lcmsdeconv.nn.train import (
        _cap_gpu_memory,
        parse_gpu_speedups,
        restore_gpu_flags,
        set_gpu_flags,
    )

    config = parse_gpu_speedups(name.replace("+", ","))
    job = json.loads((tmp / "job.json").read_text(encoding="utf-8"))
    before = set_gpu_flags(config)  # before the first convolution: cuDNN keeps its choices
    try:
        _cap_gpu_memory(torch, job["gpu_memory_gb"])
        batch = torch.load(tmp / "batch.pt", weights_only=True)
        for view in views:
            weights = torch.load(tmp / f"weights_{view}.pt", weights_only=True)
            chunk = job["chunks"].get(view)
            out = _result_base(tmp, name, view, profile)
            try:
                if profile:
                    res = cuda_kernels(view, batch, chunk, weights, config, job["lr"])
                    out.with_suffix(".json").write_text(json.dumps(res), encoding="utf-8")
                else:
                    m = speedup_steps(view, batch, chunk, weights, config, job["lr"])
                    np.savez(out.with_suffix(".npz"), ms=m["ms"], loss=np.array(m["loss"]),
                             grad0=m["grad"][0], grad1=m["grad"][1], peak_gb=m["peak_gb"],
                             reserved_gb=m["reserved_gb"])
            except Exception as exc:  # recorded; the next view is measured all the same
                out.with_suffix(".err").write_text(f"{type(exc).__name__}: {exc}"[:1000],
                                                   encoding="utf-8")
                torch.cuda.empty_cache()
    finally:
        restore_gpu_flags(before)
    return 0


def _measure_in_child(tmp: Path, config, views: list[str], profile: bool = False) -> dict:
    """Measure ``config`` for ``views`` in a process of its own (:func:`speedup_child`): cuDNN
    keeps the convolution algorithm it first chooses for each convolution of a process, with or
    without its benchmark, so a configuration measured after another would inherit its choices;
    and a failure cannot spoil the next measurement. Returns ``{view: the measurement, or the
    exception that stopped it}`` (profiles: ``{view: the table, or {"error": ...}}``)."""
    import numpy as np

    name = config_name(config)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--speedup-child", str(tmp),
           "--config", name, "--views", ",".join(views)] + (["--profile"] if profile else [])
    env = dict(os.environ, PYTHONUTF8="1")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=SPEEDUP_TIMEOUT, env=env)
        stopped = (f"the measurement stopped (exit code {proc.returncode}): "
                   f"{(proc.stdout + proc.stderr).strip()[-300:]}")
    except subprocess.TimeoutExpired:
        stopped = f"no result within {SPEEDUP_TIMEOUT} s"
    except Exception as exc:  # e.g. Windows error 1455 (paging file): this configuration only
        stopped = f"the measurement could not start: {type(exc).__name__}: {exc}"[:400]
    out = {}
    for view in views:
        base = _result_base(tmp, name, view, profile)
        if profile:
            try:
                out[view] = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                err = base.with_suffix(".err")
                out[view] = {"view": view, "speedups": list(config), "error": (
                    err.read_text(encoding="utf-8") if err.exists() else stopped)}
        elif base.with_suffix(".npz").exists():
            with np.load(base.with_suffix(".npz")) as z:
                out[view] = {"ms": float(z["ms"]), "loss": [float(v) for v in z["loss"]],
                             "grad": [z["grad0"], z["grad1"]],
                             "peak_gb": round(float(z["peak_gb"]), 2),
                             "reserved_gb": (round(float(z["reserved_gb"]), 2)
                                             if "reserved_gb" in z.files else None)}
        elif base.with_suffix(".err").exists():
            out[view] = RuntimeError(base.with_suffix(".err").read_text(encoding="utf-8"))
        else:
            out[view] = RuntimeError(stopped)
    return out


def first_batch(batch_size: int) -> dict:
    """The first training batch of train_all's runs: items 0 .. batch_size - 1 of the recipe's
    training spectra at seed 0, with the measured channel (the window view leaves it out)."""
    import torch

    from lcmsdeconv.synth.dataset import FrameDataset

    ds = FrameDataset(recipe_mixture(), batch_size, base_seed=0)
    items = [ds[i] for i in range(batch_size)]
    names = ("features", "topk_z", "topk_w", "heat")
    return {k: torch.stack([it[j] for it in items]) for j, k in enumerate(names)}


def measure_speedups(views: list[str], batch_size: int, chunks: dict,
                     gpu_memory_gb: float) -> dict:
    """The GPU speedups of each of ``views`` on the runs' first training batch, with the head
    chunking of ``chunks`` (``{view: chunk}``), under the GPU memory cap, from the same
    trained-like weights (the untrained network's, every parameter moved by normal noise of
    0.05: the learned head away from its zero start), then a profile of the window view's step
    with its chosen speedups. Returns machine.json's ``gpu_speedups`` (``{view: [names]}``),
    ``speedup_timings`` (``{view: {configuration: median ms}}``), ``speedup_checks`` (why each
    was kept or not), the memory of each step (``speedup_peak_gb``, the most the tensors held,
    and ``speedup_reserved_gb``, the most the caching allocator reserved; ``{view:
    {configuration: GB}}``) and ``cuda_kernels``."""
    import torch
    from train_all import COMMON
    from train_charge_net import V3_WINDOW

    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import GPU_SPEEDUPS

    lr = float(COMMON[COMMON.index("--lr") + 1])
    torch.cuda.empty_cache()  # this process's cache would hold GPU memory the children need
    with tempfile.TemporaryDirectory(prefix="lcmsdeconv_probe_",
                                     ignore_cleanup_errors=True) as name:
        tmp = Path(name)
        batch = first_batch(batch_size)
        torch.save(batch, tmp / "batch.pt")
        for view in views:
            in_ch = batch["features"].shape[1] - (0 if view == "comb" else 1)
            torch.manual_seed(0)
            model = build_model("full", z_max=100, in_ch=in_ch, arch="ladder", window=V3_WINDOW,
                                iso=True, view=view)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.05 * torch.randn_like(p))
            torch.save(model.state_dict(), tmp / f"weights_{view}.pt")
        (tmp / "job.json").write_text(json.dumps({
            "lr": lr, "gpu_memory_gb": gpu_memory_gb,
            "chunks": {v: chunks.get(v) for v in views}}), encoding="utf-8")
        measured: dict = {}
        for config in [(), *((t,) for t in GPU_SPEEDUPS)]:  # every view at once
            print(f"  {config_name(config)} ...", flush=True)
            for view, m in _measure_in_child(tmp, config, views).items():
                measured[(config, view)] = m
        results = {}
        for view in views:
            def measure(config, view=view):
                key = (tuple(s for s in GPU_SPEEDUPS if s in config), view)
                if key not in measured:
                    print(f"  {config_name(config)} ({view} view) ...", flush=True)
                    measured[key] = _measure_in_child(tmp, config, [view])[view]
                m = measured[key]
                if isinstance(m, Exception):
                    raise m
                return m

            results[view] = choose_speedups(measure)
        kernels = None
        if "window" in results:
            chosen = tuple(results["window"]["chosen"])
            print(f"  profiling the window view ({config_name(chosen)}) ...", flush=True)
            kernels = _measure_in_child(tmp, chosen, ["window"], profile=True)["window"]
    peaks: dict = {}
    reserved: dict = {}
    for (config, view), m in measured.items():
        if isinstance(m, dict):
            peaks.setdefault(view, {})[config_name(config)] = m["peak_gb"]
            if m.get("reserved_gb") is not None:
                reserved.setdefault(view, {})[config_name(config)] = m["reserved_gb"]
    return {"gpu_speedups": {v: r["chosen"] for v, r in results.items()},
            "speedup_timings": {v: r["timings"] for v, r in results.items()},
            "speedup_checks": {v: r["notes"] for v, r in results.items()},
            "speedup_peak_gb": peaks, "speedup_reserved_gb": reserved, "cuda_kernels": kernels}


def speedup_lines(info: dict, chunks: dict | None = None) -> list[str]:
    """A short account of the speedup measurement (:func:`measure_speedups`) for the console
    and setup.log."""
    lines = []
    chunks = chunks or {}
    for view, timings in (info.get("speedup_timings") or {}).items():
        notes = (info.get("speedup_checks") or {}).get(view) or {}
        chunk = chunks.get(view)
        how = "" if view not in chunks else (
            ", all charges at once" if chunk is None else f", {chunk} charges at a time")
        lines.append(f"  {view} view{how}:")
        reserved = (info.get("speedup_reserved_gb") or {}).get(view) or {}
        for name, note in notes.items():
            ms, gb = timings.get(name), reserved.get(name)
            took = f"{ms:.0f} ms a step" + (f", {gb:.1f} GB reserved" if gb else "") if ms else ""
            lines.append(f"    {name}: " + (f"{took}; " if took else "") + note)
        chosen = (info.get("gpu_speedups") or {}).get(view) or []
        lines.append(f"    training uses: {', '.join(chosen) or 'none'}")
    k = info.get("cuda_kernels") or {}
    if k.get("kernels"):
        lines.append(f"  CUDA kernels of {k['steps']} training steps of the {k['view']} view "
                     f"({', '.join(k['speedups']) or 'no speedups'}), most time first:")
        for row in k["kernels"][:5]:
            lines.append(f"    {100 * row['share']:4.1f} %  {row['total_ms']:8.1f} ms  "
                         f"{row['calls']:5d} calls  {row['name'][:90]}")
    elif k.get("error"):
        lines.append(f"  the CUDA kernel profile failed: {k['error'][:300]}")
    return lines


def worker_line(run: dict) -> str:
    """One measurement of the data workers (workers_run) for the console and setup.log."""
    n = max(run["workers"], 1)
    return (f"  {run['workers']} worker{'s' if n > 1 else ''}: "
            f"{run['spectra_per_second']:.1f} spectra/s together "
            f"({run['spectra_per_second'] / n:.2f} each), timed over "
            f"{run.get('timed_seconds', 0):.0f} s after {run['startup_seconds']:.0f} s to start; "
            f"{run['worker_gb']:.2f} GB of RAM each at most ({run['workers_total_gb']:.1f} GB "
            f"together; {run['worker_commit_gb']:.1f} GB committed each)")


def measure_workers(n: int, measured: list, halve: bool = True) -> dict | None:
    """workers_run(n), with half as many workers each time they fail to run (Windows error 1455,
    the paging file, for example) where ``halve``; each measurement is added to ``measured``.
    None if no number of workers ran."""
    while n >= 1:
        print(f"Measuring {n} data workers on the recipe's spectra (a minute or two) ...",
              flush=True)
        try:
            run = workers_run(n)
        except (OSError, RuntimeError) as exc:  # e.g. Windows error 1455: paging file
            if not halve:
                print(f"  {n} workers failed to run ({exc})", flush=True)
                return None
            print(f"  {n} workers failed to run ({exc}); trying {n // 2}", flush=True)
            n //= 2
            continue
        print(worker_line(run), flush=True)
        measured.append(run)
        return run
    return None


def pace_lines(estimate: dict) -> list[str]:
    """What sets the pace of each of train_all's default runs (train_all.training_hours), with
    what setup measured of the GPU, for the console and setup.log."""
    verdicts = {"workers": "the data workers set the pace",
                "gpu": "the GPU sets the pace",
                "gpu_cap": "the GPU at the active-bin cap is slower than the data workers; "
                           "real batches are lighter, so either may set the pace",
                "either": "the GPU is slower than the data workers' rate scaled up from fewer "
                          "workers, which may promise too much, so either may set the pace"}
    lines = []
    for r in estimate["runs"]:
        gpu = []
        if r["batch"]:
            gpu.append(f"{r['batch_ms']:.0f} ms a step on the runs' first training batch "
                       f"({r['batch']:.0f} spectra/s)")
        if r["cap"]:
            gpu.append(f"{r['cap_ms']:.0f} ms with every spectrum at the active-bin cap "
                       f"({r['cap']:.0f} spectra/s, the slowest a step can be)")
        if not gpu:
            lines.append(f"  {r['name']} run: the GPU was not measured")
            continue
        lines.append(f"  {r['name']} run: the GPU took {' and '.join(gpu)}; the data workers "
                     f"deliver {r['data']:.0f} spectra/s: {verdicts[r['set_by']]}")
    return lines


# ------------------------------------------------------------------ how the data workers run
#: the ways of running the data workers that setup compares (data_loading_test), each as the
#: settings train_charge_net.py takes (DATA_KEYS: --loader, --cpu-binding, --worker-qos,
#: --worker-priority). "today" is how they ran before, and the fallback: torch's DataLoader
#: (whole batches to the workers in turn, delivered in order, so that the slowest worker sets
#: the pace), the workers not bound to cores, Windows' power throttling left to Windows,
#: below-normal priority. The others use the ordered loader (lcmsdeconv.synth.pool: single
#: spectra to whichever worker is free, the same batches), with the workers bound to the cores
#: the training process does not keep ("pool") or each to a core of its own ("core";
#: lcmsdeconv.nn.cpus.plan), opted out of power throttling ("high": EcoQoS off, Windows only) or
#: not, at below-normal or normal priority (Windows only: elsewhere a priority once lowered
#: cannot be raised again)
DATA_KEYS = ("loader", "cpu_binding", "worker_qos", "worker_priority")
DATA_CONFIGS = {
    "today": ("dataloader", "none", "system", "below_normal"),
    "ordered": ("ordered", "none", "system", "below_normal"),
    "ordered_high": ("ordered", "none", "high", "below_normal"),
    "pool": ("ordered", "pool", "system", "below_normal"),
    "pool_high": ("ordered", "pool", "high", "below_normal"),
    "core": ("ordered", "core", "system", "below_normal"),
    "core_high": ("ordered", "core", "high", "below_normal"),
    "core_normal": ("ordered", "core", "system", "normal"),
    "core_high_normal": ("ordered", "core", "high", "normal"),
}
#: the choice (choose_data_loading): the ways in the order of how much they change
#: (data_cost), each taken over the one chosen so far only if its gain over that one, less
#: DATA_SE standard errors of the gain (data_noise: from how far apart each way's two windows
#: are, a steady drift over the test taken out), is at least DATA_MARGIN, and it delivers at
#: least DATA_MARGIN more spectra per second than every way that changes less. Simulated with
#: every way as fast as today's and each window's rate off by 2 % at random, the margin alone
#: took another way in 27 % of the tests, this rule in 1-2 %; it takes a way 8 % faster in 85 %.
#: Today's two windows, the first and the last of the test, further apart than DATA_BUSY mean
#: that the computer was busy with something else: today's way is kept. A way whose own two
#: windows are further apart is left out
DATA_MARGIN, DATA_SE, DATA_BUSY = 0.03, 1.0, 0.10
#: the training process waiting for less than this share of a window: the GPU set its pace
DATA_GPU_BOUND = 0.05
#: each window's timed part: whole rounds of torch's DataLoader (a batch from every worker: it
#: hands them out in turn, so that only whole rounds take the same time again), as many as
#: today's way delivers in about DATA_WINDOW_SECONDS, at least DATA_MIN_ROUNDS; the same
#: spectra in every window
DATA_WINDOW_SECONDS, DATA_MIN_ROUNDS = 8.0, 3
#: a window taking more than DATA_WINDOW_LIMIT times its expected time (at least
#: DATA_WINDOW_MIN_LIMIT seconds; torch's DataLoader, and the ordered loader's warm-up before
#: its first window, WORKER_TIMEOUT more for their start) stops the test, and today's way is
#: kept. The ordered loader waits for a batch without a timeout of its own: a watchdog stops its
#: workers at the limit (_watchdog), and the batch it waits for fails
DATA_WINDOW_LIMIT, DATA_WINDOW_MIN_LIMIT = 10.0, 120.0
#: seconds the watchdog gives the ordered loader's workers to finish the spectrum they are on
DATA_STOP_SECONDS = 2.0
#: warm-up before each window's timed part: torch's DataLoader, whose workers start afresh for
#: each of its windows (spawned ones import PyTorch first, and every new process makes its first
#: spectra more slowly), DATA_DL_WARM_ROUNDS rounds; the ordered loader, whose workers have run
#: since its first window, two spectra a worker
DATA_DL_WARM_ROUNDS = 3
#: the code the data workers read before every spectrum: generation * DATA_CODES + the index of
#: the way they are to take
DATA_CODES = 64
#: the spectra of the test: base seed, and the first index of the windows' warm-up and lead-out
#: spectra (those timed are 0, 1, ... in every window)
DATA_SEED, DATA_WARM_BASE, DATA_LEAD_BASE = 4242, 10_000_000, 20_000_000


def data_settings(name: str) -> dict:
    """The settings of the way ``name`` of DATA_CONFIGS, by DATA_KEYS."""
    return dict(zip(DATA_KEYS, DATA_CONFIGS[name]))


def data_cost(settings: dict) -> int:
    """How much a way of running the data workers changes from today's: the ordered loader 1,
    the opt-out of power throttling 1, the workers bound to their cores 1 or each to its own
    core 2, normal priority 2 (the computer's other programs no longer come first)."""
    return (int(settings["loader"] == "ordered") + int(settings["worker_qos"] == "high")
            + {"none": 0, "pool": 1, "core": 2}[settings["cpu_binding"]]
            + 2 * int(settings["worker_priority"] == "normal"))


def data_checks(topo, allowed, workers: int) -> dict:
    """What data_loading_test can switch while the workers run, checked on this process first:
    ``binding`` (lcmsdeconv.nn.cpus.plan binds on this processor, the logical processors read
    are all those reported, and this process can be bound to its cores and given back its
    processors), ``qos`` (Windows: the opt-out of power throttling set, read back and set back)
    and ``priority`` (Windows, where a worker's priority can be raised again), each with why
    not; and the processor as read."""
    from lcmsdeconv.nn import cpus

    why: dict = {}
    out = {"topology": topo.describe(), "source": topo.source, "hybrid": topo.hybrid,
           "counts": topo.counts(), "allowed": cpus.cpu_ranges(allowed or []), "why_not": why}
    if topo.notes:
        out["notes"] = list(topo.notes)
    plan = cpus.plan(topo, workers=workers, allowed=allowed, mode="core")
    try:
        import psutil

        logical = psutil.cpu_count(logical=True)
    except Exception:
        logical = None
    if plan.mode != "core":
        why["binding"] = plan.notes[0] if plan.notes else "too few cores"
    elif logical and topo.logical_processors + topo.other_groups != logical:
        why["binding"] = (f"{topo.logical_processors} logical processors read, {logical} "
                          f"reported")
    elif not (cpus.pin_process(plan.main) and cpus.allowed_cpus() == sorted(plan.main)):
        why["binding"] = "this process could not be bound to the training process's cores"
    if allowed and not cpus.pin_process(allowed) and "binding" not in why:
        why["binding"] = "this process could not be given back its processors"
    if sys.platform != "win32":
        why["qos"] = "Windows' power throttling: Windows only"
        why["priority"] = "a worker's priority, once lowered, cannot be raised again here"
    else:
        before = cpus.get_power_throttling()
        done = cpus.set_power_throttling(False)
        back = cpus.get_power_throttling()
        cpus.restore_power_throttling(before)
        if not done:
            why["qos"] = "SetProcessInformation refused the opt-out of power throttling"
        elif back is not None and (not back[0] & 1 or back[1] & 1):
            why["qos"] = f"the opt-out read back as (ControlMask, StateMask) {tuple(back)}"
    for key in ("binding", "qos", "priority"):
        out[key] = key not in why
    return out


def data_configs(checks: dict) -> list[str]:
    """The ways of DATA_CONFIGS this computer lets data_loading_test switch to (data_checks)."""
    names = []
    for name in DATA_CONFIGS:
        s = data_settings(name)
        if ((s["cpu_binding"] != "none" and not checks.get("binding"))
                or (s["worker_qos"] == "high" and not checks.get("qos"))
                or (s["worker_priority"] == "normal" and not checks.get("priority"))):
            continue
        names.append(name)
    return names


#: in a data worker of data_loading_test: the way it has taken (the process's state, shared by
#: every copy of SwitchedFrames it holds)
_SWITCHED: dict = {}


def _worker_id() -> int:
    """This data worker's id: the ordered loader's, torch's DataLoader's, or worker_init's."""
    try:
        from lcmsdeconv.synth.pool import current_worker

        wid = current_worker().get("id")
        if wid is not None:
            return int(wid)
    except Exception:
        pass
    try:
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        if info is not None:
            return int(info.id)
    except Exception:
        pass
    from lcmsdeconv.nn.cpus import WORKER_STATE

    return int(WORKER_STATE.get("id", 0))


class SwitchedFrames:
    """``frames``' spectra, unchanged, each made once its data worker has taken the way of
    running the probe asks for: ``control[0]`` (a tensor in shared memory) holds the generation
    times DATA_CODES plus the index of the way in ``configs`` (DATA_KEYS' settings). A worker
    that finds a new generation before a spectrum binds itself (``placements``: the
    lcmsdeconv.nn.cpus.Placement of each binding; not bound: back to the processors it started
    with), sets its power throttling and its priority, and writes what it reached to its row of
    ``status``: 1 or 0 for the binding, the power throttling and the priority (as asked or
    not), then the generation. On Windows a process may change all three of its own whenever it
    likes, so one start of the workers serves every way."""

    def __init__(self, frames, control, status, configs, placements: dict):
        self.frames = frames
        self.control = control
        self.status = status
        self.configs = tuple(configs)
        self.placements = placements

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx):
        code = int(self.control[0])
        if _SWITCHED.get("code") != code:
            self._switch(code)
        return self.frames[idx]

    def _switch(self, code: int) -> None:
        from lcmsdeconv.nn import cpus

        gen, k = divmod(code, DATA_CODES)
        want = self.configs[k]
        st = _SWITCHED
        if "binding" not in st:  # as the worker started: not bound, as the system decides
            st.update(binding="none", qos="system", priority="below_normal",
                      everything=tuple(cpus.allowed_cpus() or ()))
        binding = want["cpu_binding"]
        if binding != st["binding"]:
            target = st["everything"] if binding == "none" else \
                self.placements[binding].worker_cpus(_worker_id())
            if target and cpus.pin_process(target):
                st["binding"] = binding
        if want["worker_qos"] != st["qos"]:
            if cpus.set_power_throttling(False if want["worker_qos"] == "high" else None):
                st["qos"] = want["worker_qos"]
        if want["worker_priority"] != st["priority"]:
            if cpus.set_priority(want["worker_priority"]):
                st["priority"] = want["worker_priority"]
        st["code"] = code
        row = self.status[_worker_id() % self.status.shape[0]]
        row[1] = int(st["binding"] == binding)
        row[2] = int(st["qos"] == want["worker_qos"])
        row[3] = int(st["priority"] == want["worker_priority"])
        row[0] = gen  # last: the row is whole once its generation shows


def _importable():
    """This module as the data workers import it to unpickle SwitchedFrames: "probe", from
    gpu_training (on the path this module put it on), whatever name it was loaded under here."""
    import importlib

    return importlib.import_module("probe")


#: torch.cuda._sleep cycles a second, once measured
_GPU_CYCLES: dict = {}


def busy_for(seconds: float, cuda: bool = False) -> None:
    """Keep this thread busy for ``seconds`` without holding the interpreter lock, as training's
    main thread is while it waits on the GPU: on CUDA a kernel that spins that long on the GPU
    and the wait for it (the driver spins); otherwise hashing, which lets go of the lock."""
    if seconds <= 0:
        return
    if cuda:
        try:
            import torch

            if "per_second" not in _GPU_CYCLES:
                torch.cuda.synchronize()
                t = time.perf_counter()
                torch.cuda._sleep(10_000_000)
                torch.cuda.synchronize()
                _GPU_CYCLES["per_second"] = 1e7 / max(time.perf_counter() - t, 1e-6)
            torch.cuda._sleep(int(seconds * _GPU_CYCLES["per_second"]))
            torch.cuda.synchronize()
            return
        except Exception:  # a PyTorch without it: the processor instead
            pass
    import hashlib

    block = b"\0" * 65536
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        hashlib.sha256(block)


class TrainingLoad:
    """What the training process does with every batch while its data workers make the next
    ones, so that the ways of running them are measured as training runs them: ``step(batch)``
    (a real training step on the GPU: the batch copied there without waiting from pinned
    memory, the forward and backward pass, the optimiser, the moving average); without one,
    the batch copied to the GPU where ``cuda`` and this thread kept busy for ``busy_seconds``
    (busy_for). ``what`` says which; ``close()`` waits for the GPU and undoes what the step
    set up."""

    def __init__(self, step=None, busy_seconds: float = 0.0, cuda: bool = False, what: str = "",
                 restore=None):
        self.step = step
        self.busy_seconds = float(busy_seconds or 0.0)
        self.cuda = bool(cuda)
        self.what = what or (f"the main thread kept busy for {1000 * self.busy_seconds:.0f} ms a "
                             f"batch" if self.busy_seconds else "nothing done with the batches")
        self._restore = restore

    def __call__(self, batch) -> None:
        if self.step is not None:
            self.step(batch)
            return
        if self.cuda:
            for t in batch:
                t.to("cuda", non_blocking=True)
        busy_for(self.busy_seconds, self.cuda)

    def close(self) -> None:
        try:
            if self.cuda:
                import torch

                torch.cuda.synchronize()
        finally:
            if self._restore is not None:
                self._restore()
                self._restore = None


def gpu_training_load(view: str, chunk: int | None, speedups, lr: float, sample) -> TrainingLoad:
    """A real training step of the full network of ``view`` on the GPU for every batch, as the
    runs take it (head chunking ``chunk``, GPU speedups ``speedups``; the window view leaves the
    measured channel out), after three steps on ``sample`` (a batch as the loaders deliver it):
    cuDNN's benchmark, where on, chooses its algorithms in the first."""
    import torch
    from train_charge_net import V3_WINDOW

    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import (
        EMA,
        TrainConfig,
        _batch_dict,
        restore_gpu_flags,
        set_gpu_flags,
    )

    cfg = TrainConfig()
    before = set_gpu_flags(speedups)
    try:
        in_ch = sample[0].shape[1] - (0 if view == "comb" else 1)
        torch.manual_seed(0)
        model = build_model("full", z_max=cfg.z_max, in_ch=in_ch, arch="ladder",
                            window=V3_WINDOW, iso=True, view=view, head_chunk=chunk)
        model = model.cuda().train()
        if "index_select" in speedups:
            model.head_gather = "index_select"
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay,
                                fused=True if "fused_adamw" in speedups else None)
        ema = EMA(model, cfg.ema_decay)
        loss_fn = ChargeLoss(heat_weight=cfg.heat_weight)
        params = list(model.parameters())

        def step(batch) -> None:
            b = _batch_dict(batch, "cuda")
            if view != "comb":
                b["features"] = b["features"][:, :-1].contiguous()
            losses = loss_fn(model(b["features"]), b)
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ema.update(model)

        for _ in range(3):
            step(sample)
        torch.cuda.synchronize()
    except BaseException:
        restore_gpu_flags(before)
        raise
    how = f"{chunk} charges at a time" if chunk else "all charges at once"
    return TrainingLoad(step=step, cuda=True, restore=lambda: restore_gpu_flags(before),
                        what=f"a training step of the {view} view on the GPU for every batch "
                             f"({how}; GPU speedups {', '.join(speedups) or 'none'})")


def training_load(batch_size: int, chunk: int | None, speedups, lr: float,
                  step_seconds: float | None) -> TrainingLoad:
    """The training-like load of data_loading_test: a real training step of the window view on
    the GPU for every batch (gpu_training_load); where that cannot run, every batch copied to
    the GPU and the main thread kept busy for ``step_seconds`` (the step setup measured);
    without CUDA (tests), only the latter."""
    import torch

    if not torch.cuda.is_available():
        return TrainingLoad(busy_seconds=step_seconds or 0.0)
    try:
        b = first_batch(batch_size)
        sample = [b[k].pin_memory() for k in ("features", "topk_z", "topk_w", "heat")]
        return gpu_training_load("window", chunk, tuple(speedups), lr, sample)
    except Exception as exc:  # e.g. out of GPU memory: the load without the network
        torch.cuda.empty_cache()
        print(f"  a training step could not run in this process ({type(exc).__name__}: "
              f"{str(exc)[:200]}); each batch is copied to the GPU instead and the main thread "
              f"kept busy for {1000 * (step_seconds or 0.0):.0f} ms", flush=True)
        return TrainingLoad(busy_seconds=step_seconds or 0.0, cuda=True,
                            what=f"every batch copied to the GPU and the main thread kept busy "
                                 f"for {1000 * (step_seconds or 0.0):.0f} ms")


def _cpu_split(me) -> tuple[float, float]:
    """Processor seconds this computer has spent busy, and those of this process (``me``) and
    its children (psutil); (0, 0) where they cannot be read."""
    import psutil

    try:
        t = psutil.cpu_times()
        busy = sum(t) - sum(getattr(t, f, 0.0) for f in (
            "idle", "iowait", "guest", "guest_nice", "interrupt", "dpc"))
        ours = 0.0
        for p in [me, *me.children(recursive=True)]:
            try:
                c = p.cpu_times()
                ours += c.user + c.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return busy, ours
    except Exception:
        return 0.0, 0.0


def _timed_window(batches, warm: int, timed: int, batch_size: int, load, stats, kinds: dict,
                  limit: float) -> dict:
    """``timed`` batches of ``batches`` after ``warm`` (at least one), ``load`` doing with each
    what training does: spectra a second (timed * batch_size over the time from the arrival of
    the last warm-up batch to that of the last timed one), the share of that time this process
    waited for them, its processor time a batch, what the workers made on each class of core
    (``stats``, an ItemStats; ``kinds``: the class of each logical processor) and how many
    logical processors other programs kept busy meanwhile. Raises TimeoutError past ``limit``
    seconds."""
    import psutil

    me = psutil.Process()
    start = time.perf_counter()
    waited = 0.0
    t0 = t1 = since = made = cpu0 = cpu1 = split0 = split1 = None
    for k in range(warm + timed):
        t = time.perf_counter()
        try:
            batch = next(batches)
        except StopIteration:
            raise RuntimeError(f"the spectra ended after {k} batches") from None
        now = time.perf_counter()
        if k >= warm:
            waited += now - t
        if k == warm - 1:
            t0, since, split0 = now, stats.snapshot(), _cpu_split(me)
            cpu0 = sum(me.cpu_times()[:2])
        elif k == warm + timed - 1:
            t1, split1 = now, _cpu_split(me)
            cpu1 = sum(me.cpu_times()[:2])
            made = stats.summary(since, kinds, now - t0)
        load(batch)
        batch = None
        if time.perf_counter() - start > limit:
            raise TimeoutError(f"a window took more than {limit:.0f} s")
    seconds = max(t1 - t0, 1e-9)
    others = max(0.0, (split1[0] - split0[0]) - (split1[1] - split0[1])) / seconds
    return {"spectra_per_second": timed * batch_size / seconds, "seconds": seconds,
            "spectra": timed * batch_size, "waiting_share": waited / seconds,
            "main_cpu_ms_per_batch": 1000.0 * (cpu1 - cpu0) / timed, "others": others,
            "by_kind": made["by_kind"], "workers_busy": made.get("workers_busy")}


def _epoch(warm: int, timed: int, lead: int, batch_size: int) -> list[int]:
    """The spectra of a window: ``warm`` batches to warm up, the ``timed`` ones (the same in
    every window), ``lead`` more that keep the workers busy until the last timed one is in."""
    B = batch_size
    return ([DATA_WARM_BASE + i for i in range(warm * B)] + list(range(timed * B))
            + [DATA_LEAD_BASE + i for i in range(lead * B)])


def _dataloader_window(frames, workers: int, batch_size: int, epoch: list, warm: int,
                       timed: int, load, kinds: dict, limit: float) -> dict:
    """One window of today's way: torch's DataLoader as training starts it (``workers``
    workers at below-normal priority, two batches in work each, into pinned memory where CUDA
    is available), stopped after the timed batches."""
    import torch
    from torch.utils.data import DataLoader

    from lcmsdeconv.synth.dataset import CountedItems, mp_context, worker_init_for
    from lcmsdeconv.synth.pool import ItemStats

    stats = ItemStats(workers, shared=True)
    loader = DataLoader(CountedItems(frames, stats), batch_size=batch_size, sampler=epoch,
                        num_workers=workers, worker_init_fn=worker_init_for(None, workers),
                        prefetch_factor=2, timeout=WORKER_TIMEOUT, drop_last=True,
                        pin_memory=torch.cuda.is_available(), **mp_context(workers))
    batches = iter(loader)
    try:
        return _timed_window(batches, warm, timed, batch_size, load, stats, kinds, limit)
    finally:
        stop = getattr(batches, "_shutdown_workers", None)
        if stop is not None:
            stop()  # each worker finishes the batch it is making, then ends
        del batches


@contextlib.contextmanager
def _watchdog(seconds: float, stop, what: str):
    """Run ``stop()`` in a thread of its own once ``seconds`` have passed, unless the block has
    ended first, and wait for it: the ordered loader waits for a batch without a timeout of its
    own, and its pool stopped (WorkerPool.close) makes that wait fail at once. A block the
    watchdog stopped raises TimeoutError ("``what`` took more than ... s") in place of what it
    raised, or of its end."""
    lock = threading.Lock()
    state = {"over": False, "fired": False}

    def fire() -> None:
        with lock:
            if state["over"]:
                return
            state["fired"] = True
        stop()

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()

    def over() -> bool:  # whether the watchdog stopped the block (once its stop has finished)
        with lock:
            state["over"] = True
        timer.cancel()
        if state["fired"]:
            timer.join(60.0)  # stop() takes seconds: the workers terminated after a short wait
        return state["fired"]

    late = f"{what} took more than {seconds:.0f} s"
    try:
        yield
    except Exception as exc:
        if over():
            raise TimeoutError(late) from exc
        raise
    except BaseException:  # Ctrl+C: as it is
        over()
        raise
    if over():
        raise TimeoutError(late)


def _merge_windows(windows: list[dict]) -> dict:
    """A way's windows taken together: its spectra over its time, each window's rate and how far
    apart they are (``drift``), and the rest as means or sums."""
    from lcmsdeconv.nn.cpus import cpu_ranges, parse_cpu_list

    seconds = sum(w["seconds"] for w in windows)
    spectra = sum(w["spectra"] for w in windows)
    rates = [w["spectra_per_second"] for w in windows]

    def mean(key, digits):
        vals = [w[key] for w in windows if w.get(key) is not None]
        return round(statistics.fmean(vals), digits) if vals else None

    kinds: dict = {}
    for w in windows:
        for kind, d in (w.get("by_kind") or {}).items():
            a = kinds.setdefault(kind, {"items": 0, "weighted": 0.0, "rated": 0, "moved": 0,
                                        "processors": set()})
            a["items"] += int(d.get("items") or 0)
            a["moved"] += int(d.get("moved") or 0)
            if d.get("per_worker_per_second"):
                a["weighted"] += float(d["per_worker_per_second"]) * int(d.get("items") or 0)
                a["rated"] += int(d.get("items") or 0)
            if d.get("processors"):
                a["processors"].update(parse_cpu_list(d["processors"]))
    out = {"spectra_per_second": round(spectra / max(seconds, 1e-9), 2),
           "windows": [round(r, 2) for r in rates],
           "drift": (round((max(rates) - min(rates)) / statistics.fmean(rates), 3)
                     if len(rates) > 1 and statistics.fmean(rates) > 0 else None),
           "seconds": round(seconds, 1), "spectra": spectra,
           "waiting_share": mean("waiting_share", 3),
           "main_cpu_ms_per_batch": mean("main_cpu_ms_per_batch", 1),
           "others": mean("others", 2), "workers_busy": mean("workers_busy", 3),
           "by_kind": {kind: {"items": a["items"],
                              "per_second": round(a["items"] / max(seconds, 1e-9), 2),
                              "per_worker_per_second": (round(a["weighted"] / a["rated"], 3)
                                                        if a["rated"] else None),
                              "moved": a["moved"], "processors": cpu_ranges(a["processors"])}
                       for kind, a in sorted(kinds.items())}}
    if any("made_twice" in w for w in windows):
        out["made_twice"] = sum(int(w.get("made_twice") or 0) for w in windows)
    if any("applied" in w for w in windows):
        out["applied"] = min(int(w.get("applied") or 0) for w in windows)
    return out


def data_loading_test(workers: int, rate: float | None, load: TrainingLoad | None = None,
                      frames=None, batch_size: int = 4, names=None,
                      window_seconds: float = DATA_WINDOW_SECONDS) -> dict:
    """Which way of running ``workers`` data workers (DATA_CONFIGS) delivers the most spectra
    to a training process, measured in a few minutes on the recipe's spectra (``frames`` stands
    in for them in tests) with ``load`` doing with every batch what training does
    (TrainingLoad). Today's way, torch's DataLoader, is timed first and last, each time with
    workers of its own; between them ONE ordered loader whose workers start once and switch
    from way to way while they run (SwitchedFrames: the training process takes its part as
    training does), each way timed twice in mirrored order (A B C ... C B A), so that a load
    that drifts over the test weighs on every way alike. Every window times the same spectra
    (``rate``, today's spectra a second, sizes them: about ``window_seconds``), after a warm-up
    and with more to come, so that the workers are in full flow. Returns the record of
    machine.json's "data_loader_test": each way's spectra a second over both its windows,
    the windows' rates and how far apart they are, what the workers made on each class of
    core, how long the training process waited and how many logical processors other programs
    kept busy; with "error" where the test stopped (choose_data_loading then keeps today's)."""
    import torch
    from train_all import RECIPE

    from lcmsdeconv.nn import cpus
    from lcmsdeconv.nn.train import _warm_up
    from lcmsdeconv.synth.dataset import FrameDataset
    from lcmsdeconv.synth.pool import (
        OrderedBatches,
        SlotRing,
        WorkerPool,
        default_window,
        ring_slots,
    )

    load = load or TrainingLoad()
    B = int(batch_size)
    topo = cpus.detect()
    allowed = cpus.allowed_cpus() or sorted(c for core in topo.cores for c in core.cpus)
    checks = data_checks(topo, allowed, workers)
    names = [n for n in (names or DATA_CONFIGS) if n in data_configs(checks)]
    if "today" not in names:
        names.insert(0, "today")
    ordered = [n for n in names if n != "today"]
    kinds = topo.kind_of_cpu()
    if frames is None:
        mix = recipe_mixture()
        _warm_up(mix)  # numba's kernels compiled once, before the workers start
        frames = FrameDataset(mix, DATA_LEAD_BASE + 1_000_000, base_seed=DATA_SEED)
    placements = ({b: cpus.plan(topo, workers=workers, allowed=allowed, mode=b)
                   for b in ("pool", "core")} if checks["binding"] else {})
    window = default_window(workers, B)
    rounds = max(DATA_MIN_ROUNDS, math.ceil((rate or 0.0) * window_seconds / (B * workers)))
    timed = rounds * workers  # batches
    warm = {"dataloader": DATA_DL_WARM_ROUNDS * workers, "ordered": math.ceil(2 * workers / B)}
    lead = {"dataloader": 2 * workers, "ordered": math.ceil(window / B) + 1}
    epochs = {k: _epoch(warm[k], timed, lead[k], B) for k in warm}
    pace = max(rate or 0.0, 0.1)

    def limit(kind: str) -> float:
        expected = (warm[kind] + timed) * B / pace
        extra = WORKER_TIMEOUT if kind == "dataloader" else 0.0
        return extra + max(DATA_WINDOW_MIN_LIMIT, DATA_WINDOW_LIMIT * expected)

    record = {"recipe": RECIPE, "workers": workers, "batch": B, "timed_spectra": timed * B,
              "ordered_window": window, "load": load.what, "checks": checks,
              "placements": {b: p.describe() for b, p in placements.items()},
              "order": [], "configs": {}}
    windows: dict = {n: [] for n in names}
    configs = [data_settings(n) for n in ordered]
    qos_before = cpus.get_power_throttling()
    me_memory = {"worker": 0.0, "total": 0.0}
    started = time.perf_counter()
    pool = None

    def run(name: str, fn) -> None:
        w = fn()
        windows[name].append(w)
        record["order"].append(name)
        print(f"  {name}: {w['spectra_per_second']:.1f} spectra/s over {w['seconds']:.0f} s",
              flush=True)

    try:
        run("today", lambda: _dataloader_window(frames, workers, B, epochs["dataloader"],
                                                warm["dataloader"], timed, load, kinds,
                                                limit("dataloader")))
        if ordered:
            spec = frames.item_spec() if hasattr(frames, "item_spec") else None
            ring = SlotRing(ring_slots(window, B, workers), spec) if spec else None
            base = cpus.Placement("none", (), tuple(() for _ in range(workers)),
                                  tuple("?" for _ in range(workers)))
            pool = WorkerPool(workers, init=functools.partial(cpus.worker_init, placement=base),
                              ring=ring)
            stop = functools.partial(pool.close, timeout=DATA_STOP_SECONDS)  # for _watchdog
            control = torch.zeros(1, dtype=torch.int64).share_memory_()
            status = torch.full((workers, 4), -1, dtype=torch.int64).share_memory_()
            frames_switched = _importable().SwitchedFrames(frames, control, status, configs,
                                                           placements)
            gen = 1
            control[0] = gen * DATA_CODES
            ob = OrderedBatches(frames_switched, B, _epoch(warm["ordered"], 0, 0, B), pool,
                                window, pin_memory=torch.cuda.is_available(), name="probe")
            with _watchdog(WORKER_TIMEOUT + limit("ordered"), stop,
                           "the ordered loader's start"):
                for _batch in ob:  # every worker up and warm (each spawned one imports PyTorch)
                    pass
            ob.sampler = epochs["ordered"]
            sequence = ordered + ordered[::-1]
            for turn, name in enumerate(sequence):
                gen += 1
                s = configs[ordered.index(name)]
                cpus.pin_process(placements[s["cpu_binding"]].main
                                 if s["cpu_binding"] != "none" else allowed)
                cpus.set_power_throttling(False if s["worker_qos"] == "high" else None)
                control[0] = gen * DATA_CODES + ordered.index(name)
                twice = pool.duplicates

                def window_of(gen=gen, twice=twice, turn=turn):
                    batches = iter(ob)
                    try:
                        with _watchdog(limit("ordered"), stop, "a window"):
                            w = _timed_window(batches, warm["ordered"], timed, B, load,
                                              pool.stats, kinds, limit("ordered"))
                    finally:
                        batches.close()
                    rows = status.clone()
                    w["applied"] = int(((rows[:, 0] == gen) & (rows[:, 1:] == 1).all(1)).sum())
                    w["made_twice"] = pool.duplicates - twice
                    if turn in (len(ordered) - 1, len(sequence) - 1):  # their memory, twice
                        sizes = _children_gb()
                        if sizes:
                            me_memory["worker"] = max(me_memory["worker"], max(sizes))
                            me_memory["total"] = max(me_memory["total"], sum(sizes))
                    return w

                run(name, window_of)
            pool.close()
            pool = None
            cpus.pin_process(allowed)
            cpus.restore_power_throttling(qos_before)
            record["ordered_worker_gb"] = round(me_memory["worker"], 2)
            record["ordered_workers_total_gb"] = round(me_memory["total"], 2)
            record["ordered_slots_gb"] = round(ring.nbytes / 2**30, 2) if ring else 0.0
        run("today", lambda: _dataloader_window(frames, workers, B, epochs["dataloader"],
                                                warm["dataloader"], timed, load, kinds,
                                                limit("dataloader")))
    except Exception as exc:  # the test stops; today's way is kept
        record["error"] = _error_text(exc)
        print(f"  the test stopped: {record['error']}", flush=True)
    finally:
        if pool is not None:
            pool.close()
        cpus.pin_process(allowed)
        cpus.restore_power_throttling(qos_before)
    record["seconds"] = round(time.perf_counter() - started, 1)
    record["configs"] = {n: {**data_settings(n), **_merge_windows(ws)}
                         for n, ws in windows.items() if ws}
    return record


def _error_text(exc: BaseException, limit: int = 500) -> str:
    """An exception in a line: its first line and, where its message carries a traceback (a
    data worker's error, as torch's DataLoader and lcmsdeconv.synth.pool raise it), its last."""
    lines = [line.strip() for line in str(exc).strip().splitlines() if line.strip()]
    text = f"{type(exc).__name__}: {lines[0] if lines else ''}"
    if len(lines) > 1:
        text += f" ... {lines[-1]}"
    return text[:limit]


def _children_gb() -> list[float]:
    """RAM of each child process of this one (the data workers), as the task manager shows
    it."""
    import psutil

    sizes = []
    for p in psutil.Process().children(recursive=True):
        try:
            sizes.append(process_gb(p))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sizes


def data_noise(configs: dict, order=None) -> dict:
    """How much data_loading_test's rates vary by chance, read from how far apart each way's
    two windows are (``configs``: {name: {"windows": [...], ...}}; ``order``: the ways in the
    order the test timed their windows, its "order"; by default today's first and last and the
    others as ``configs`` lists them, there and back). The mirrored order keeps a steady drift
    over the test (a processor warming up, a program ending) out of the ways' rates, but moves
    each way's two windows apart in proportion to the time between them: that drift is fitted
    as a straight line through the differences of the ordered loader's ways (second window less
    first, relative to their mean) against the time between their windows, where at least two
    ways have both, and taken out. Returns ``window``, the noise of a window's rate (one
    standard deviation, relative: the root mean square of what the line leaves of those
    differences, a degree of freedom less for the line, over √2), ``drift`` (the line's change
    over the whole test, relative; below 0 the computer got slower) and ``ways``, the standard
    error of each way's rate (relative): ``window``, or its own windows' where they are further
    apart than the drift and ``window`` explain (as today's, at the ends of the test with workers
    of their own, can be), over the square root of its number of windows."""
    names = [n for n, c in configs.items() if c.get("windows")]
    if not (order and all(list(order).count(n) == len(configs[n]["windows"]) for n in names)):
        others = [n for n in names if n != "today"]
        order = ["today", *others, *others[::-1], "today"]
    span = max(len(order) - 1, 1)
    pairs = {}  # name: (its second window less its first over their mean, the time between)
    for n in names:
        w = [float(r) for r in configs[n]["windows"]]
        at = [i / span for i, m in enumerate(order) if m == n]
        mean = statistics.fmean(w)
        if len(w) == 2 and len(at) == 2 and mean > 0:
            pairs[n] = ((w[1] - w[0]) / mean, at[1] - at[0])
    fit = {n: p for n, p in pairs.items() if n != "today" and p[1] > 0}
    if len(fit) >= 2:
        drift = sum(d * t for d, t in fit.values()) / sum(t * t for _d, t in fit.values())
        dof = len(fit) - 1
    else:  # too few ways to tell a drift from the noise: none taken out
        fit, drift, dof = pairs, 0.0, len(pairs)
    left = {n: d - drift * t for n, (d, t) in pairs.items()}
    window = math.sqrt(sum(left[n] ** 2 for n in fit) / (2 * dof)) if dof else 0.0
    ways = {n: max(window, abs(left.get(n, 0.0)) / math.sqrt(2))
            / math.sqrt(len(configs[n]["windows"])) for n in names}
    return {"window": window, "drift": drift, "ways": ways}


def choose_data_loading(configs: dict, workers: int | None = None, margin: float = DATA_MARGIN,
                        busy: float = DATA_BUSY, order=None, errors: float = DATA_SE) -> dict:
    """The way training runs its data workers, from data_loading_test's ``configs`` ({name of
    DATA_CONFIGS: {"spectra_per_second", "windows", "drift", "applied", ...}}; ``order``: the
    test's "order", for data_noise): today's unless another delivers at least ``margin`` more
    beyond what the test cannot tell apart. The ways are taken in the order of how much they
    change (data_cost, then DATA_CONFIGS' order), each over the one chosen so far only if its
    gain over that one, less ``errors`` standard errors of the gain (data_noise: the two rates'
    standard errors taken together), is at least ``margin``, and it delivers at least ``margin``
    more than every way that changes less: each further change pays for itself. Today's is kept
    where it was not measured, or where its two windows (the first and the last of the test)
    are more than ``busy`` apart: the computer was busy with something else. A way whose own two
    windows are more than ``busy`` apart, or that fewer than ``workers`` workers took, is left
    out (``left_out``, why). Returns the settings (DATA_KEYS), the way's name (``config``), its
    spectra a second and today's, the gain, a window's noise and the drift over the test
    (``window_noise``, ``test_drift``: data_noise) and why (``reason``)."""
    today = configs.get("today") or {}
    base = today.get("spectra_per_second")
    out = {**data_settings("today"), "config": "today", "spectra_per_second": base,
           "today_spectra_per_second": base, "gain": 0.0, "left_out": {}}
    if not base:
        out["reason"] = "as before: setup's test did not measure the data workers"
        return out
    noise = data_noise(configs, order)
    out.update(window_noise=round(noise["window"], 4), test_drift=round(noise["drift"], 4))
    if (today.get("waiting_share") is not None
            and today["waiting_share"] < DATA_GPU_BOUND):
        out["gpu_bound"] = True
    drift = today.get("drift")
    if drift is not None and drift > busy:
        rates = ", ".join(f"{r:.1f}" for r in today.get("windows") or [])
        out["busy"] = True
        out["reason"] = (f"as before: the computer was busy with something else during setup's "
                         f"test (as before, the data workers delivered {rates} spectra/s in its "
                         f"first and last minute, {100 * drift:.0f} % apart)")
        return out
    candidates = []
    for name, c in configs.items():
        if name == "today" or name not in DATA_CONFIGS:
            continue
        why = None
        if not c.get("spectra_per_second"):
            why = "not measured"
        elif c.get("drift") is not None and c["drift"] > busy:
            why = f"its two windows {100 * c['drift']:.0f} % apart"
        elif workers and c.get("applied") is not None and c["applied"] < workers:
            why = f"{c['applied']} of {workers} workers took it"
        if why:
            out["left_out"][name] = why
            continue
        candidates.append((data_cost(data_settings(name)), list(DATA_CONFIGS).index(name), name))
    rates = {"today": float(base),
             **{n: float(configs[n]["spectra_per_second"]) for _c, _i, n in candidates}}

    def error(name: str, other: str) -> float:  # of the gain of ``name`` over ``other``
        return math.hypot(noise["ways"].get(name, 0.0), noise["ways"].get(other, 0.0))

    choice = "today"
    for cost, _i, name in sorted(candidates):
        less = ["today", *(n for c, _j, n in candidates if c < cost)]
        if (rates[name] / rates[choice] - 1.0 - errors * error(name, choice) >= margin
                and all(rates[name] >= rates[n] * (1.0 + margin) for n in less)):
            choice = name
    rate = rates[choice]
    out.update(data_settings(choice), config=choice, spectra_per_second=rate,
               gain=round(rate / base - 1.0, 3))
    if choice != "today":
        out["reason"] = (f"{100 * (rate / base - 1.0):.0f} % more spectra per second than as "
                         f"before in setup's test ({rate:.1f} against {base:.1f})")
    else:
        best = max(candidates, key=lambda c: rates[c[2]], default=None)
        fastest = ""
        if best is not None:
            fastest = (f"; the fastest, {best[2]}, {100 * (rates[best[2]] / base - 1.0):+.0f} % "
                       f"(± {100 * error(best[2], 'today'):.1f} %)")
        beyond = ("" if errors <= 0 else ", beyond its standard error" if errors == 1 else
                  f", beyond {errors:g} standard errors")
        out["reason"] = (f"as before: no other way was at least {100 * margin:.0f} % faster in "
                         f"setup's test than the ways that change less{beyond}{fastest}")
    return out


#: how data_lines names the classes of cores
_KIND_WORDS = {"P": "performance", "E": "efficient", "LP": "low-power", "?": ""}


def data_lines(test: dict | None, decision: dict) -> list[str]:
    """A short account of the data-loading test (data_loading_test) and its choice for the
    console and setup.log."""
    from train_all import data_words

    lines = []
    configs = (test or {}).get("configs") or {}
    if configs:
        lines.append(f"  {test['workers']} workers, {test['timed_spectra']} spectra timed in each "
                     f"window, {test.get('load')}; {test.get('seconds', 0) / 60:.1f} min. "
                     f"Spectra/s (first and second window), and a busy worker's on each class "
                     f"of core (today: torch's DataLoader as before; the others the ordered "
                     f"loader; pool: the workers on the cores the training process does not "
                     f"keep; core: each on its own; high: opted out of power throttling; "
                     f"normal: normal priority):")
        base = (configs.get("today") or {}).get("spectra_per_second")
        for name, c in configs.items():
            rate = c["spectra_per_second"]
            gain = f" {100 * (rate / base - 1.0):+4.0f} %" if base and name != "today" else \
                "       "
            ws = ", ".join(f"{r:.1f}" for r in c.get("windows") or [])
            per = ", ".join(
                f"{_KIND_WORDS.get(k, k)} {d['per_worker_per_second']:.2f}/s".strip()
                for k, d in (c.get("by_kind") or {}).items() if d.get("per_worker_per_second"))
            extra = ""
            if c.get("waiting_share") is not None:
                extra += f"; waited {100 * c['waiting_share']:.0f} %"
            left = (decision.get("left_out") or {}).get(name)
            if left:
                extra += f"; left out: {left}"
            lines.append(f"    {name:<16} {rate:6.1f}{gain} ({ws}); {per or '-'}{extra}")
        if decision.get("window_noise") is not None:
            lines.append(f"  a window's rate varied by about {100 * decision['window_noise']:.1f} "
                         f"% by chance (how far apart each way's two windows are, after a steady "
                         f"drift of {100 * (decision.get('test_drift') or 0.0):+.1f} % over the "
                         f"test)")
    if (test or {}).get("error"):
        lines.append(f"  the test stopped: {test['error']}")
    if (test or {}).get("checks", {}).get("why_not"):
        lines.append("  not measured here: " + "; ".join(
            f"{k} ({v})" for k, v in test["checks"]["why_not"].items()))
    if decision.get("gpu_bound"):
        lines.append("  the GPU set the pace of today's way already: the ways could not be told "
                     "apart above it")
    lines.append(f"  training uses: {data_words(decision)}: {decision.get('reason')}")
    return lines


def window_step_seconds(gpu: dict, speed: dict, chunk: int | None, speedups) -> float | None:
    """Seconds of a training step of the window view with the head chunking ``chunk`` as setup
    measured it: on the runs' first training batch with ``speedups`` (speedup_timings, where
    they were timed with that chunking: ``speedup_chunks``, else ``head_chunks``, else taken to
    be), else at the active-bin cap (gpu_views); None where not measured."""
    from train_all import speedup_name

    ms = ((speed.get("speedup_timings") or {}).get("window") or {}).get(speedup_name(speedups))
    timed_with = speed.get("speedup_chunks") or speed.get("head_chunks") or {}
    if ms and timed_with.get("window", chunk) == chunk:
        return float(ms) / 1000.0
    m = (gpu.get("gpu_views") or {}).get("window") or {}
    if chunk is None:
        m = m.get("unchunked") or {}
    return float(m["step_seconds"]) if m.get("step_seconds") else None


def measure_data_loading(workers: int, rate: float | None, batch_size: int, chunk: int | None,
                         speedups, lr: float, step_seconds: float | None) -> dict:
    """data_loading_test of ``workers`` workers under a training-like load (training_load: the
    window view's training step, with the runs' head chunking ``chunk`` and GPU speedups
    ``speedups``); a failure is returned as {"error": ...}."""
    try:
        load = training_load(batch_size, chunk, speedups, lr, step_seconds)
    except Exception as exc:
        return {"error": _error_text(exc)}
    try:
        return data_loading_test(workers, rate, load=load, batch_size=batch_size)
    except Exception as exc:
        return {"error": _error_text(exc)}
    finally:
        try:
            load.close()
        except Exception:
            pass


def main() -> int:
    from train_all import (
        BATCH,
        DEFAULT_RUNS,
        HEAD_CHUNK,
        RATE_KEY,
        RATE_WORKERS_KEY,
        RECIPE,
        RUN_TABLE,
        head_chunk_for,
        make_run,
        speedups_for,
        training_hours,
    )

    from lcmsdeconv.nn.machine import default_ram_budget_gb, physical_ram_gb

    total_ram = physical_ram_gb()
    default_ram = default_ram_budget_gb(total_ram)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ram-gb", type=float, default=None,
                    help=f"RAM budget of the training process and its workers together (default: "
                         f"half this computer's memory, at least 16 GB and at most 48, leaving 4 "
                         f"GB to the system: {default_ram:g} GB here)")
    ap.add_argument("--gpu-memory-gb", type=float, default=14.0,
                    help="GPU memory the training process may use, CUDA context included")
    ap.add_argument("--workers", type=int, default=None,
                    help="measure and train with this many workers (default: as many as the "
                         "processor is given and the RAM budget holds)")
    ap.add_argument("--batch-size", type=int, default=BATCH,
                    help=f"spectra per step to measure the GPU at (default {BATCH}, the recipe's)")
    ap.add_argument("--out", type=Path, default=ROOT / "gpu_training" / "machine.json")
    ap.add_argument("--no-data-test", action="store_true",
                    help="do not measure how the data workers deliver spectra fastest (a few "
                         "minutes): training then runs them as before")
    ap.add_argument("--no-gpu", action="store_true", help=argparse.SUPPRESS)  # tests only
    # one configuration of the GPU speedups, measured in a process of its own (measure_speedups)
    ap.add_argument("--speedup-child", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", default="none", help=argparse.SUPPRESS)
    ap.add_argument("--views", default="window", help=argparse.SUPPRESS)
    ap.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.speedup_child is not None:
        return speedup_child(args.speedup_child, args.config, args.views.split(","), args.profile)

    import psutil
    import torch
    from train_all import COMMON, data_words

    from lcmsdeconv.nn.machine import (
        ORDERED_GB_PER_WORKER,
        cpu_cores,
        cpu_workers,
        ram_workers,
        single_threaded_workers,
    )

    ram = args.ram_gb or default_ram
    if args.no_gpu:
        gpu = {"gpu": None, "gpu_views": {}}
    elif not torch.cuda.is_available():
        print("CUDA is not available to PyTorch: run gpu_training/setup.py first.")
        return 1
    else:
        memory = gpu_memory_now()
        now = f"; {memory[0]:.1f} of {memory[1]:.1f} GB free now" if memory else ""
        print(f"Measuring training steps of both views on the GPU ({args.batch_size} spectra per "
              f"step, memory capped at {args.gpu_memory_gb:g} GB{now}) ...", flush=True)
        gpu = gpu_steps(args.gpu_memory_gb, args.batch_size, HEAD_CHUNK)
        if memory:
            gpu["gpu_free_gb_at_setup"] = memory[0]
    # the training process: what this one holds now (PyTorch, the CUDA context) and the
    # validation spectra it keeps in memory
    main_gb = process_gb(psutil.Process()) + VALIDATION_GB
    for view, m in gpu["gpu_views"].items():
        if m.get("peak_gb"):
            print(f"  {gpu['gpu']}, {view} view: {m['step_seconds'] * 1000:.0f} ms per training "
                  f"step of {m['batch']} spectra, {m['peak_gb']:.1f} GB of GPU memory at the "
                  f"peak, scoring {HEAD_CHUNK} charges at a time", flush=True)
            whole = m.get("unchunked") or {}
            if whole.get("peak_gb"):
                print(f"    all charges at once: {whole['step_seconds'] * 1000:.0f} ms, "
                      f"{whole['peak_gb']:.1f} GB", flush=True)
            else:
                print(f"    all charges at once: {whole.get('error', 'not measured')}", flush=True)
        else:
            print(f"  {view} view: {m.get('error')}", flush=True)

    single_threaded_workers()
    cores = cpu_cores()
    by_cpu = cpu_workers(cores)
    # the number of workers to measure: the processor's share where the budget holds it, with a
    # worker's memory taken as WORKER_GB (more than one took on Linux or on Windows) until it is
    # measured, so that the measurement stays within the budget
    planned = args.workers or max(1, min(by_cpu, ram_workers(ram, main_gb=max(main_gb, 1.0))))
    measured: list = []
    run = measure_workers(planned, measured)
    if run is None:
        print("No data worker could be started.")
        return 1
    worker_gb = max(run["worker_gb"], 0.3)
    chosen = args.workers or max(1, min(by_cpu, ram_workers(ram, worker_gb, main_gb)))
    did_not_run = None  # a number of workers that failed to run, where that limits those chosen
    if run["workers"] < planned:  # more workers did not run
        did_not_run = planned if chosen >= run["workers"] else None
        chosen = min(chosen, run["workers"])
    if chosen != run["workers"]:
        # by the memory measured the budget holds another number: their speed is measured at
        # that number, not scaled from this one (on a processor with efficiency cores each
        # further worker adds less, and batches come in order: one that is ahead waits)
        print(f"  the budget holds {chosen} workers of {worker_gb:.2f} GB beside the training "
              f"process ({main_gb:.1f} GB)", flush=True)
        again = measure_workers(chosen, measured, halve=False)
        if again is None:
            if chosen > run["workers"]:
                did_not_run = chosen
            chosen = min(chosen, run["workers"])
            print(f"  keeping {chosen} workers", flush=True)
        else:
            run = again
            worker_gb = max(worker_gb, again["worker_gb"])
            if not args.workers:
                chosen = max(1, min(chosen, ram_workers(ram, worker_gb, main_gb)))
    # the workers' speed at the number chosen: as measured, or in proportion where fewer were
    # chosen than measured (an underestimate: each worker of fewer is at least as fast)
    rate = run["spectra_per_second"] * min(chosen, run["workers"]) / max(run["workers"], 1)
    # a view whose training ran out of GPU memory scoring all charges at once (train_all keeps
    # the record) stays in chunks while the GPU, its cap and the batch are the ones measured
    try:
        before = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    except (OSError, ValueError):
        before = {}
    same = (before.get("gpu"), before.get("gpu_memory_gb"), before.get("gpu_batch")) == (
        gpu.get("gpu"), args.gpu_memory_gb, args.batch_size)
    failed = (before.get("unchunked_failed") or []) if same else []
    # the GPU speedups, on the runs' first training batch, with the head in chunks: the only way
    # runs use them (train_all.head_chunk_for). All charges at once with them filled the GPU
    # memory cap on the runs' varied batches and slowed to seconds a step (the RTX 4080 SUPER of
    # the profile of 2026-11-25), which one batch measured again and again does not show
    speed: dict = {"gpu_speedups": {}}
    if gpu["gpu_views"]:
        from lcmsdeconv.nn.train import GPU_SPEEDUPS

        views = list(dict.fromkeys(RUN_TABLE[name]["view"] for name in DEFAULT_RUNS))
        view_chunks = {v: HEAD_CHUNK for v in views}
        print(f"Measuring the GPU speedups ({', '.join(GPU_SPEEDUPS)}) on the runs' first "
              f"training batch, each configuration in a process of its own ...", flush=True)
        try:
            speed = measure_speedups(views, args.batch_size, view_chunks, args.gpu_memory_gb)
        except Exception as exc:  # the probe goes on: the runs train without speedups
            speed = {"gpu_speedups": {v: [] for v in views},
                     "speedup_error": f"{type(exc).__name__}: {exc}"[:400]}
            print(f"  the measurement failed ({speed['speedup_error']}); the runs train without "
                  f"GPU speedups", flush=True)
        speed = {**speed, "speedup_chunks": view_chunks}
        for line in speedup_lines(speed, view_chunks):
            print(line, flush=True)
    # then each view's head chunking, from the speedups chosen: all charges at once only for a
    # view that trains without speedups, where that was measured faster with ample room and the
    # GPU sets the pace
    chunks = {view: head_chunk_for({**gpu, "unchunked_failed": failed,
                                    "gpu_speedups": speed.get("gpu_speedups") or {}}, view,
                                   args.gpu_memory_gb, args.batch_size, rate)
              for view in gpu["gpu_views"]}
    # how the data workers deliver spectra fastest at the number chosen: torch's DataLoader, as
    # before, against the ordered loader, bound to cores or not, with and without the opt-out of
    # power throttling, at below-normal and normal priority, in mirrored order under a
    # training-like load (the window view's training step, with the chunking and speedups its
    # runs will use)
    data_test, not_run = None, None
    if args.no_data_test or args.no_gpu:
        not_run = "--no-data-test" if args.no_data_test else "the GPU was not measured"
    else:
        chunk = chunks.get("window", HEAD_CHUNK)
        window_speedups = list((speed.get("gpu_speedups") or {}).get("window") or [])
        print(f"Measuring how {chosen} data workers deliver spectra fastest: as before (torch's "
              f"DataLoader) against the ordered loader, bound to cores or not, with and without "
              f"the opt-out of power throttling, at below-normal and normal priority, each twice "
              f"in mirrored order while this process trains on what they deliver (a few "
              f"minutes) ...", flush=True)
        data_test = measure_data_loading(
            chosen, rate, args.batch_size, chunk, window_speedups,
            float(COMMON[COMMON.index("--lr") + 1]),
            window_step_seconds(gpu, speed, chunk, window_speedups))
    if data_test and not data_test.get("error"):
        decision = choose_data_loading(data_test.get("configs") or {}, chosen,
                                       order=data_test.get("order"))
    else:
        decision = choose_data_loading({})
        if data_test:
            decision["reason"] = (f"as before: setup's test of the data workers stopped "
                                  f"({data_test['error']})")
        elif not_run:
            decision["reason"] = f"as before: setup did not test the data workers ({not_run})"
    for line in data_lines(data_test, decision):
        print(line, flush=True)
    tested = chosen  # the number of workers the data-loading test measured
    slots_gb = ORDERED_GB_PER_WORKER if decision["loader"] == "ordered" else 0.0
    ring_limited = False
    if slots_gb and not args.workers:
        # the ordered loader's shared slots, which no process counts as its own, are memory too
        fit = max(1, ram_workers(ram, worker_gb + slots_gb, main_gb))
        if fit < chosen:
            chosen, ring_limited = fit, True
    if decision.get("spectra_per_second") and data_test and not data_test.get("error"):
        # what that way delivered under a training-like load, in proportion for fewer workers
        # (the ordered loader: each worker adds its own spectra)
        rate = float(decision["spectra_per_second"]) * chosen / tested
    info = {"recipe": RECIPE, "workers": chosen, "ram_budget_gb": ram,
            "ram_budget_default": args.ram_gb is None,
            "physical_ram_gb": round(total_ram, 1) if total_ram else None,
            "gpu_memory_gb": args.gpu_memory_gb, "gpu_batch": args.batch_size,
            "head_chunk": HEAD_CHUNK, "head_chunks": chunks, "main_gb": round(main_gb, 2),
            "worker_gb": round(worker_gb, 2),
            "expected_ram_gb": round(main_gb + chosen * (worker_gb + slots_gb), 1),
            RATE_KEY: round(rate, 1), RATE_WORKERS_KEY: chosen, "cores": cores,
            "processor_workers": by_cpu, "measured": measured, **gpu, **speed,
            "data_loader": {**decision, "workers": chosen}}
    if data_test:
        info["data_loader_test"] = data_test
    if failed:
        info["unchunked_failed"] = failed
    # train_all's default runs as it will train them, each at the slower of the data workers
    # and the GPU (on the runs' first training batch where measured, else at the active-bin cap)
    stages = []
    for name in DEFAULT_RUNS:
        st = make_run(name)
        st["head_chunk"] = chunks.get(st["view"], HEAD_CHUNK)
        st["gpu_speedups"] = speedups_for(info, st["view"])
        stages.append(st)
    estimate = training_hours(stages, info, chosen)
    if estimate:
        info["expected_hours"] = round(estimate["hours"], 1)
        info["expected_hours_at_most"] = round(estimate["slowest"], 1)
        info["pace"] = {r["name"]: {
            "data_spectra_per_second": round(r["data"], 1),
            "gpu_spectra_per_second": r["batch"] and round(r["batch"], 1),
            "gpu_at_cap_spectra_per_second": r["cap"] and round(r["cap"], 1),
            "set_by": r["set_by"]} for r in estimate["runs"]}
    args.out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    for view, chunk in chunks.items():
        how = "all charges at once" if chunk is None else f"{HEAD_CHUNK} charges at a time"
        print(f"  {view} view: training will score {how}", flush=True)
    if args.workers:
        why = "as --workers asks" if chosen == args.workers else f"{args.workers} did not run"
    elif ring_limited:
        why = (f"as many as the RAM budget holds with the ordered loader's shared slots; "
               f"{tested} were tested")
    elif did_not_run:
        why = f"as many as ran: {did_not_run} did not"
    elif chosen >= by_cpu:
        why = f"the processor's share, all {cores} cores but {cores - by_cpu}"
    else:
        why = f"as many as the RAM budget holds; the processor would take {by_cpu}"
    budget = (f"the default for this computer's {total_ram:.0f} GB of memory"
              if args.ram_gb is None and total_ram else
              "the default" if args.ram_gb is None else "--ram-gb")
    if data_test and not data_test.get("error") and decision.get("spectra_per_second"):
        speed_text = (f"{rate:.1f} spectra/s as setup's test measured them, loaded as above"
                      + ("" if chosen == tested else
                         f" (in proportion to the {decision['spectra_per_second']:.1f} of "
                         f"{tested})"))
    elif chosen == run["workers"]:
        speed_text = f"measured at that number: {rate:.1f} spectra/s"
    else:
        speed_text = (f"{rate:.1f} spectra/s in proportion to the "
                      f"{run['spectra_per_second']:.1f} of {run['workers']} measured")
    print(f"Chosen: {chosen} data workers ({why}); about {info['expected_ram_gb']:.1f} GB of RAM "
          f"with the training process, within the budget of {ram:g} GB ({budget}); "
          f"{speed_text}", flush=True)
    print(f"Data loading: {data_words(decision)}", flush=True)
    if estimate:
        for line in pace_lines(estimate):
            print(line, flush=True)
        bound = (f" (up to {estimate['slowest']:.1f} h should every step take as long as with "
                 f"every spectrum at the active-bin cap)"
                 if estimate["slowest"] > 1.05 * estimate["hours"] else "")
        print(f"About {estimate['hours']:.1f} h for the two runs{bound}. Each epoch's pace line "
              f"measures the pace again: a share of the training time waiting for data above a "
              f"few percent means the data workers set it.", flush=True)
    print(f"Written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
