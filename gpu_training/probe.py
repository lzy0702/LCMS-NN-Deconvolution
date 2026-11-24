"""Measure this computer for training, within the given memory limits.

Runs real training steps of the full-size networks of both views (recipe v4 with isotope-charge
channels, at the recipe's batch size, every spectrum at the active-bin cap: the most memory a
step can need) on the GPU under the GPU memory cap, scoring the charges in chunks and all at
once (train_all.py chooses between the two from these measurements), then starts data workers
the way training does on the recipe's spectra and measures how much memory each one takes and
how many spectra per second they deliver. From that it chooses the number of workers that keeps
the training process and its workers within the RAM budget.

Then it measures the GPU speedups (lcmsdeconv.nn.train.GPU_SPEEDUPS: cuDNN's benchmark, TF32
matrix products, LadderNet's index_select head, fused AdamW), which change only the rounding of
the computation, on the runs' first training batch in each view, with the head chunking the runs
will use, each configuration in a process of its own (_measure_in_child): a speedup is kept for
a view if a step with it alone is at least 3 % faster than one without any and its loss and
gradients agree with that step's; the kept ones are combined greedily, fastest first, each
further one only if it makes the step at least 2 % faster again (choose_speedups). A profile of
the window view's step with its speedups lists the CUDA kernels that take the most time.
Everything goes to gpu_training/machine.json, which train_all.py reads.

    python gpu_training/probe.py [--ram-gb 16] [--gpu-memory-gb 14] [--workers N]

setup.py runs it; run it again on its own to re-measure (for example after changing the limits).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
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


def process_gb(proc) -> float:
    """RAM of one process as the task manager shows it (its private working set)."""
    from lcmsdeconv.nn.machine import process_ram_gb

    return process_ram_gb(proc)


def commit_gb(proc) -> float:
    """Memory Windows has promised the process (private bytes), reserved or used; only for the
    record: it is not RAM, but RAM plus the paging file must hold it."""
    return getattr(proc.memory_info(), "private", 0) / 2**30


def recipe_mixture(measured: bool = True) -> list:
    """The training spectra of train_all's runs (recipe v4), as training configures them (singly
    charged ions labelled, isotope-charge channels, and the comb view's measured channel)."""
    from train_charge_net import mixture_v4

    mix = mixture_v4()
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


def workers_run(n: int, rounds: int = 4, batch_size: int = 4) -> dict:
    """Start ``n`` workers as training does on the recipe's spectra and measure their speed and
    memory. Speed is timed from the moment every worker has delivered its first batch, so that
    start-up does not count."""
    import psutil
    from torch.utils.data import DataLoader

    from lcmsdeconv.synth.dataset import FrameDataset, mp_context, worker_init

    ds = FrameDataset(recipe_mixture(), n * rounds * batch_size, base_seed=4242)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=n, worker_init_fn=worker_init,
                        **mp_context(n))
    me = psutil.Process()
    started = time.time()
    warm = None
    spectra = 0
    per_worker = commit = total_workers = 0.0
    for k, (x, *_rest) in enumerate(loader):
        if k == n - 1:
            warm = time.time()  # every worker is up and has delivered a batch
        elif k >= n:
            spectra += x.shape[0]
        if k % max(1, n // 2) == 0:
            sizes = []
            for p in me.children(recursive=True):
                try:
                    sizes.append((process_gb(p), commit_gb(p)))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if sizes:
                per_worker = max(per_worker, max(a for a, _b in sizes))
                commit = max(commit, max(b for _a, b in sizes))
                total_workers = max(total_workers, sum(a for a, _b in sizes))
    elapsed = time.time() - (warm or started)
    return {"workers": n, "startup_seconds": round((warm or started) - started, 1),
            "spectra_per_second": round(spectra / max(elapsed, 1e-9), 1),
            "worker_gb": round(per_worker, 2), "worker_commit_gb": round(commit, 2),
            "workers_total_gb": round(total_workers, 2)}


# ------------------------------------------------------------------ the GPU speedups
def config_name(config) -> str:
    """``("tf32", "index_select")`` -> "tf32+index_select" (in GPU_SPEEDUPS order); () -> "none"."""
    from lcmsdeconv.nn.train import GPU_SPEEDUPS

    return "+".join(s for s in GPU_SPEEDUPS if s in config) or "none"


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
    (cuDNN's benchmark, if on, runs in the first), and the memory it took."""
    import torch

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
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30}


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
                             grad0=m["grad"][0], grad1=m["grad"][1], peak_gb=m["peak_gb"])
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
                             "peak_gb": round(float(z["peak_gb"]), 2)}
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
    was kept or not) and ``cuda_kernels``."""
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
    for (config, view), m in measured.items():
        if isinstance(m, dict):
            peaks.setdefault(view, {})[config_name(config)] = m["peak_gb"]
    return {"gpu_speedups": {v: r["chosen"] for v, r in results.items()},
            "speedup_timings": {v: r["timings"] for v, r in results.items()},
            "speedup_checks": {v: r["notes"] for v, r in results.items()},
            "speedup_peak_gb": peaks, "cuda_kernels": kernels}


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
        for name, note in notes.items():
            ms = timings.get(name)
            lines.append(f"    {name}: " + (f"{ms:.0f} ms a step; " if ms else "") + note)
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


def main() -> int:
    from train_all import (
        BATCH,
        DEFAULT_RUNS,
        HEAD_CHUNK,
        RATE_KEY,
        RATE_WORKERS_KEY,
        RECIPE,
        RUN_TABLE,
        SPECTRA,
        head_chunk_for,
    )

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ram-gb", type=float, default=16.0,
                    help="RAM budget of the training process and its workers together")
    ap.add_argument("--gpu-memory-gb", type=float, default=14.0,
                    help="GPU memory the training process may use, CUDA context included")
    ap.add_argument("--workers", type=int, default=None, help="measure with this many workers")
    ap.add_argument("--batch-size", type=int, default=BATCH,
                    help=f"spectra per step to measure the GPU at (default {BATCH}, the recipe's)")
    ap.add_argument("--out", type=Path, default=ROOT / "gpu_training" / "machine.json")
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

    from lcmsdeconv.nn.machine import MARGIN_GB, auto_workers, cpu_cores, single_threaded_workers

    if args.no_gpu:
        gpu = {"gpu": None, "gpu_views": {}}
    elif not torch.cuda.is_available():
        print("CUDA is not available to PyTorch: run gpu_training/setup.py first.")
        return 1
    else:
        print(f"Measuring training steps of both views on the GPU ({args.batch_size} spectra per "
              f"step, memory capped at {args.gpu_memory_gb:g} GB) ...", flush=True)
        gpu = gpu_steps(args.gpu_memory_gb, args.batch_size, HEAD_CHUNK)
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
    n = args.workers or auto_workers(args.ram_gb, main_gb=max(main_gb, 1.0))
    run = None
    while n >= 1:
        print(f"Measuring {n} data workers on the recipe's spectra ...", flush=True)
        try:
            run = workers_run(n)
            break
        except (OSError, RuntimeError) as exc:  # e.g. Windows error 1455: paging file
            print(f"  {n} workers failed to run ({exc}); trying {n // 2}", flush=True)
            n //= 2
    if run is None:
        print("No data worker could be started.")
        return 1
    print(f"  {run['workers']} workers: {run['spectra_per_second']:.0f} spectra/s after "
          f"{run['startup_seconds']:.0f} s to start; {run['worker_gb']:.2f} GB of RAM each at most "
          f"({run['workers_total_gb']:.1f} GB together; {run['worker_commit_gb']:.1f} GB committed "
          f"each)", flush=True)
    worker_gb = max(run["worker_gb"], 0.3)
    fit = int((args.ram_gb - main_gb - MARGIN_GB) // worker_gb)
    by_cpu = cores - 2 if cores <= 8 else cores - 4
    chosen = args.workers or max(1, min(by_cpu, fit))
    rate = run["spectra_per_second"] * chosen / max(run["workers"], 1)
    # a view whose training ran out of GPU memory scoring all charges at once (train_all keeps
    # the record) stays in chunks while the GPU, its cap and the batch are the ones measured
    try:
        before = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    except (OSError, ValueError):
        before = {}
    same = (before.get("gpu"), before.get("gpu_memory_gb"), before.get("gpu_batch")) == (
        gpu.get("gpu"), args.gpu_memory_gb, args.batch_size)
    failed = (before.get("unchunked_failed") or []) if same else []
    chunks = {view: head_chunk_for({**gpu, "unchunked_failed": failed}, view,
                                   args.gpu_memory_gb, args.batch_size, rate)
              for view in gpu["gpu_views"]}
    # train_all.py's default runs, each 20 x 6000 spectra; 10 % more for validation. The data
    # workers and the GPU work at the same time, so the slower of the two sets the pace
    hours, paces = 0.0, {}
    for name in DEFAULT_RUNS:
        view = RUN_TABLE[name]["view"]
        m = gpu["gpu_views"].get(view) or {}
        if m and chunks.get(view, HEAD_CHUNK) is None:
            m = m["unchunked"]  # the step train_all will take
        pace = rate
        if m.get("step_seconds"):
            pace = min(rate, m["batch"] / m["step_seconds"])
        paces[view] = pace
        hours += 1.1 * SPECTRA / max(pace, 1e-6) / 3600
    # the GPU speedups, on the runs' first training batch, with the chunking each view will use
    speed: dict = {"gpu_speedups": {}}
    if gpu["gpu_views"]:
        from lcmsdeconv.nn.train import GPU_SPEEDUPS

        views = list(dict.fromkeys(RUN_TABLE[name]["view"] for name in DEFAULT_RUNS))
        view_chunks = {v: chunks.get(v, HEAD_CHUNK) for v in views}
        print(f"Measuring the GPU speedups ({', '.join(GPU_SPEEDUPS)}) on the runs' first "
              f"training batch, each configuration in a process of its own ...", flush=True)
        try:
            speed = measure_speedups(views, args.batch_size, view_chunks, args.gpu_memory_gb)
        except Exception as exc:  # the probe goes on: the runs train without speedups
            speed = {"gpu_speedups": {v: [] for v in views},
                     "speedup_error": f"{type(exc).__name__}: {exc}"[:400]}
            print(f"  the measurement failed ({speed['speedup_error']}); the runs train without "
                  f"GPU speedups", flush=True)
        for line in speedup_lines(speed, view_chunks):
            print(line, flush=True)
    info = {"recipe": RECIPE, "workers": chosen, "ram_budget_gb": args.ram_gb,
            "gpu_memory_gb": args.gpu_memory_gb, "gpu_batch": args.batch_size,
            "head_chunk": HEAD_CHUNK, "head_chunks": chunks, "main_gb": round(main_gb, 2),
            "worker_gb": round(worker_gb, 2),
            "expected_ram_gb": round(main_gb + chosen * worker_gb, 1),
            RATE_KEY: round(rate, 1), RATE_WORKERS_KEY: chosen,
            "expected_hours": round(hours, 1), "cores": cores, "measured": [run], **gpu,
            **speed}
    if failed:
        info["unchunked_failed"] = failed
    args.out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    gpu_rates = {}
    for v, m in gpu["gpu_views"].items():
        if m and chunks.get(v, HEAD_CHUNK) is None:
            m = m["unchunked"]
        if m.get("step_seconds"):
            gpu_rates[v] = m["batch"] / m["step_seconds"]
    if gpu_rates:
        pace_text = "; ".join(f"the GPU could take {r:.0f} spectra/s in the {v} view"
                              for v, r in gpu_rates.items())
        slower = ("the workers set the pace" if all(r >= rate for r in gpu_rates.values())
                  else "the GPU sets the pace")
        pace_text = f" ({pace_text}, so {slower})"
    else:
        pace_text = " (the GPU was not measured)"
    for view, chunk in chunks.items():
        how = "all charges at once" if chunk is None else f"{HEAD_CHUNK} charges at a time"
        print(f"  {view} view: training will score {how}", flush=True)
    print(f"Chosen: {chosen} data workers (about {info['expected_ram_gb']:.1f} GB of RAM with "
          f"the training process, budget {args.ram_gb:g} GB; {cores} cores). Expected "
          f"{rate:.0f} spectra/s from the workers{pace_text}: about {hours:.1f} h for the two "
          f"runs. Written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
