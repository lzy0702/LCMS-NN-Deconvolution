"""CPU experiments: synthesis of the runs' training spectra (items of train_all's recipe, v5
today, exactly as FrameDataset makes them, with the GPU package's settings: singly charged ions
kept, isotope-charge channels, the measured channel, 131 072-bin crops), by single-threaded
workers as in training.

- :func:`core_map`: one process pinned to each logical processor in turn, the same items on
  each: items per second per processor, then by kind (performance, efficient);
- :func:`throttle`: on one performance and one efficient core, Windows' power throttling
  (EcoQoS) left to Windows or opted out of, at normal and below-normal priority;
- :func:`cprofile_items`: where the time of an item goes, on one core of each kind;
- :func:`loader`: many workers at once delivering batches to a consumer, as training's
  DataLoader does or as the order-preserving dispatcher (:mod:`.dispatch`) does, unpinned or
  one worker pinned per core, with or without the power-throttling opt-out.
"""

from __future__ import annotations

import os
import statistics
import time
from functools import partial

from . import cpusets
from .common import ROOT, Unavailable, recipe, recipe_mixture, stats

BATCH = 4
#: batches each worker delivers before the timing starts (DataLoader) or items per worker
#: (dispatcher: WARM_ROUNDS * BATCH), as probe.workers_run
WARM_ROUNDS = 2
#: the timed part of a loader measurement, and the width of its timeline's bins
LOADER_SECONDS, TIMELINE_SECONDS = 45.0, 5.0
#: far more items than are read: the measurement stops in full flow
ENDLESS = 10_000_000


def recipe_dataset(length: int, base_seed: int = 0, crop: int | None = None):
    """The runs' training spectra (common.recipe_mixture: train_all's recipe as training
    configures it) as a FrameDataset, cropped to ``crop`` bins where given (smoke runs and
    tests)."""
    from lcmsdeconv.synth.dataset import FrameDataset

    mix = recipe_mixture()
    if crop:
        for _w, c in mix:
            c.crop_size = int(crop)
    return FrameDataset(mix, length, base_seed=base_seed), mix


def warm_up(mix) -> float:
    """numba's compiled functions loaded once in this process (training's _warm_up)."""
    from lcmsdeconv.nn.train import _warm_up

    t = time.perf_counter()
    _warm_up(mix)
    return time.perf_counter() - t


def _time_items(ds, n: int) -> list[float]:
    out = []
    for i in range(n):
        t = time.perf_counter()
        ds[i]
        out.append(time.perf_counter() - t)
    return out


def _processors() -> list[dict]:
    return cpusets.logical_processors()["processors"]


def pick_cores(procs: list[dict]) -> dict:
    """One core of each kind for the per-kind experiments: the second performance core (the
    first often serves the system's interrupts) and the first efficient one; on a processor of
    one kind, the second processor and the last."""
    order = sorted(cpusets.usable(procs), key=lambda p: p["cpu"])
    perf = [p for p in order if p.get("kind") == "performance"]
    eff = [p for p in order if p.get("kind") == "efficient"]
    if perf and eff:
        return {"performance": perf[min(1, len(perf) - 1)]["cpu"], "efficient": eff[0]["cpu"]}
    if not order:
        raise Unavailable("no processor this process may run on")
    return {"first": order[min(1, len(order) - 1)]["cpu"], "last": order[-1]["cpu"]}


def core_map(ctx, items: int = 12, crop: int | None = None, priority: str = "below_normal",
             cpus: list | None = None) -> dict:
    """Items per second of one process pinned to each logical processor in turn, the same
    ``items`` items on each (after two unpinned to warm up), at the workers' priority."""
    procs = _processors()
    by_cpu = {p["cpu"]: p for p in procs if p.get("cpu") is not None}
    allowed = cpusets.current_affinity() or sorted(by_cpu)
    todo = [c for c in (cpus or sorted(by_cpu)) if c in allowed]
    ds, mix = recipe_dataset(items, base_seed=0, crop=crop)
    warm = warm_up(mix)
    _time_items(ds, min(2, items))
    prio = cpusets.set_priority(priority)
    rows = []
    try:
        for cpu in todo:
            cpusets.set_affinity([cpu])
            time.sleep(0.05)
            ts = _time_items(ds, items)
            p = by_cpu.get(cpu, {})
            rows.append({"cpu": cpu, "kind": p.get("kind"),
                         "efficiency_class": p.get("efficiency_class"),
                         "items_per_second": round(items / sum(ts), 3),
                         "item_seconds": stats(ts, 4)})
            ctx.say(f"cpu {cpu} ({p.get('kind')}): {items / sum(ts):.2f} items/s")
    finally:
        cpusets.set_affinity(allowed)
        cpusets.set_priority("normal")
    by_kind: dict = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["items_per_second"])
    per_item = [statistics.median(r["item_seconds"]["median"] for r in rows)] if rows else []
    return {"config": {"items": items, "item_indices": f"0-{items - 1} of seed 0",
                       "recipe": recipe(), "crop": crop or 131072, "priority": prio,
                       "warm_up_seconds": round(warm, 1),
                       "threads": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS")}},
            "cores": rows, "by_kind": {k: stats(v, 3) for k, v in by_kind.items()},
            "median_item_seconds": per_item[0] if per_item else None}


def throttle(ctx, items: int = 6, reps: int = 2, crop: int | None = None) -> dict:
    """On one performance and one efficient core: items per second with power throttling left to
    Windows ("default") or opted out of ("opt_out"), at normal and below-normal priority, each
    cell ``reps`` times in mirrored order (drift cancels). Elsewhere than on Windows only the
    priority varies."""
    procs = _processors()
    cores = pick_cores(procs)
    modes = ["default", "opt_out"] if cpusets.WINDOWS else ["default"]
    cells = [(m, p) for m in modes for p in ("normal", "below_normal")]
    ds, mix = recipe_dataset(items, base_seed=0, crop=crop)
    warm_up(mix)
    _time_items(ds, min(2, items))
    allowed = cpusets.current_affinity()
    out: dict = {}
    applied = []
    try:
        for kind, cpu in cores.items():
            cpusets.set_affinity([cpu])
            got: dict = {}
            for rep in range(reps):
                for mode, prio in (cells if rep % 2 == 0 else cells[::-1]):
                    if cpusets.WINDOWS:
                        applied.append(cpusets.set_power_throttling(mode))
                    applied.append(cpusets.set_priority(prio))
                    ts = _time_items(ds, items)
                    got.setdefault(f"{mode}/{prio}", []).append(round(items / sum(ts), 3))
            cell = {k: {"items_per_second": round(statistics.mean(v), 3), "runs": v}
                    for k, v in got.items()}
            effects = {}
            for prio in ("normal", "below_normal"):
                a, b = cell.get(f"default/{prio}"), cell.get(f"opt_out/{prio}")
                if a and b:
                    effects[f"opt_out_gain_{prio}"] = round(
                        b["items_per_second"] / a["items_per_second"] - 1.0, 4)
            for mode in modes:
                a, b = cell.get(f"{mode}/normal"), cell.get(f"{mode}/below_normal")
                if a and b:
                    effects[f"below_normal_change_{mode}"] = round(
                        b["items_per_second"] / a["items_per_second"] - 1.0, 4)
            out[kind] = {"cpu": cpu, "cells": cell, "effects": effects}
            ctx.say(f"{kind} core {cpu}: " + ", ".join(
                f"{k} {v['items_per_second']:.2f}" for k, v in cell.items()))
    finally:
        if cpusets.WINDOWS:
            try:
                cpusets.set_power_throttling("default")
            except Exception:  # the process ends next: nothing outlives it
                pass
        cpusets.set_priority("normal")
        if allowed:
            cpusets.set_affinity(allowed)
    return {"config": {"items": items, "reps": reps, "recipe": recipe(), "crop": crop or 131072,
                       "cores": cores, "modes": modes,
                       "windows_constants": cpusets.WINDOWS_CONSTANTS},
            "cores": out, "applied": applied[:8]}


def _short_path(filename: str) -> str:
    """A profile's file name without the folders that name the user's computer: the package's
    files from the package folder, installed ones from site-packages, the standard library's
    by name."""
    f = str(filename).replace("\\", "/")
    root = str(ROOT).replace("\\", "/")
    if f.startswith(root):
        return f[len(root):].lstrip("/")
    for marker in ("site-packages/", "dist-packages/"):
        if marker in f:
            return marker + f.split(marker, 1)[1]
    if "/lib/python" in f.lower() or "/lib/" in f.lower():
        return "<stdlib>/" + f.rsplit("/", 1)[-1]
    return f.rsplit("/", 1)[-1] if "/" in f else f


def profile_rows(prof, top: int = 30) -> dict:
    """The ``top`` functions of a cProfile by cumulative and by own time."""
    import pstats

    st = pstats.Stats(prof)
    total = float(getattr(st, "total_tt", 0.0)) or 1e-9
    rows = []
    for (filename, line, func), (cc, nc, tt, ct, _callers) in st.stats.items():
        rows.append({"function": f"{_short_path(filename)}:{line}({func})", "calls": nc,
                     "primitive_calls": cc, "own_s": round(tt, 4), "cumulative_s": round(ct, 4),
                     "own_share": round(tt / total, 4)})
    return {"total_s": round(total, 3),
            "by_cumulative": sorted(rows, key=lambda r: -r["cumulative_s"])[:top],
            "by_own": sorted(rows, key=lambda r: -r["own_s"])[:top]}


def cprofile_items(ctx, items: int = 6, crop: int | None = None, top: int = 30) -> dict:
    """cProfile of ``items`` items on one core of each kind (pinned), after a warm-up: the
    ``top`` functions by cumulative and by own time; the full tables in logs/<id>_<kind>.txt."""
    import cProfile
    import io
    import pstats

    procs = _processors()
    cores = pick_cores(procs)
    ds, mix = recipe_dataset(items, base_seed=0, crop=crop)
    warm_up(mix)
    _time_items(ds, min(2, items))
    allowed = cpusets.current_affinity()
    out = {}
    try:
        for kind, cpu in cores.items():
            cpusets.set_affinity([cpu])
            prof = cProfile.Profile()
            t = time.perf_counter()
            prof.enable()
            for i in range(items):
                ds[i]
            prof.disable()
            wall = time.perf_counter() - t
            out[kind] = {"cpu": cpu, "wall_s": round(wall, 3), **profile_rows(prof, top)}
            text = io.StringIO()
            pstats.Stats(prof, stream=text).sort_stats("tottime").print_stats(60)
            ctx.path("logs", f"{ctx.id}_{kind}.txt").write_text(
                text.getvalue().replace(str(ROOT), "<package>"), encoding="utf-8")
    finally:
        if allowed:
            cpusets.set_affinity(allowed)
    return {"config": {"items": items, "recipe": recipe(), "crop": crop or 131072,
                       "cores": cores}, "profiles": out}


# ------------------------------------------------------------------ many workers at once
class TimedItems:
    """A dataset whose items carry, as a last tensor, ``[worker, start, end]`` of their making
    (the worker from DataLoader's worker info, or the dispatcher's), on the clock every process
    of this computer shares (time.perf_counter)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        import torch
        from torch.utils.data import get_worker_info

        from .dispatch import current_worker

        info = get_worker_info()
        wid = info.id if info is not None else current_worker()
        t0 = time.perf_counter()
        item = self.dataset[idx]
        t1 = time.perf_counter()
        stamp = torch.tensor([-1 if wid is None else wid, t0, t1], dtype=torch.float64)
        return (*item, stamp)


def _workers_memory() -> dict:
    try:
        import psutil

        from lcmsdeconv.nn.machine import process_ram_gb

        sizes = []
        for p in psutil.Process().children(recursive=True):
            try:
                sizes.append(process_ram_gb(p))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {"worker_gb_max": round(max(sizes), 3) if sizes else None,
                "workers_gb_total": round(sum(sizes), 2), "processes": len(sizes)}
    except Exception as exc:  # psutil missing: the record says so
        return {"error": str(exc)[:200]}


def loader(ctx, kind: str = "dataloader", workers: int = 20, pinned: bool = False,
           opt_out: bool = False, seconds: float = LOADER_SECONDS, warm_rounds: int = WARM_ROUNDS,
           crop: int | None = None, reserve: int | None = None, window: int | None = None) -> dict:
    """``workers`` workers delivering batches of 4 to this process for ``seconds`` after the
    warm-up: ``kind`` "dataloader" (training's DataLoader: batch k from worker k mod N, in order,
    two batches in hand per worker) or "dispatcher" (each free worker takes the next item;
    batches assembled in order). ``pinned``: each worker on a core of its own and this process
    on the performance cores cpusets.pin_plan reserves, from its first batch on: a process a
    pinned one starts inherits its affinity, so pinned before the workers exist this process
    would squeeze their start (each one importing PyTorch) onto its own one or two cores.
    ``opt_out``: every worker opts out of power throttling (Windows). Records the items per
    second delivered, the consumer's wait, each worker's items and busy share, a timeline, and
    what each worker got."""
    import torch
    from torch.utils.data import DataLoader

    from lcmsdeconv.synth.dataset import mp_context

    from .dispatch import OrderedDispatcher

    if kind not in ("dataloader", "dispatcher"):
        raise ValueError(f"loader kind {kind!r}: dataloader or dispatcher")
    if opt_out and not cpusets.WINDOWS:
        raise Unavailable("the power-throttling opt-out is Windows'")
    procs = _processors()
    kinds = {p["cpu"]: p.get("kind") for p in procs if p.get("cpu") is not None}
    plan = cpusets.pin_plan(procs, workers, reserve) if pinned else None
    allowed = cpusets.current_affinity()
    log_dir = ctx.work / "workers" / ctx.id
    init = partial(cpusets.worker_setup, cores=plan["workers"] if plan else None,
                   opt_out=opt_out, log_dir=str(log_dir))
    ds, mix = recipe_dataset(ENDLESS, base_seed=4242, crop=crop)  # probe.workers_run's spectra
    warm = warm_up(mix)  # before the workers start, as training does
    main_affinity = None  # this process's, once pinned (after its first batch)
    pin = torch.cuda.is_available()
    timed = TimedItems(ds)
    if kind == "dataloader":
        source = DataLoader(timed, batch_size=BATCH, num_workers=workers, worker_init_fn=init,
                            prefetch_factor=2, timeout=600, pin_memory=pin, drop_last=True,
                            **mp_context(workers))
    else:
        source = OrderedDispatcher(timed, range(ENDLESS), BATCH, workers, window=window,
                                   worker_init=init, pin_memory=pin)
    batches = iter(source)
    started = time.perf_counter()
    stamps, waits, arrivals = [], [], []
    t_warm = first = memory = None
    k = k_warm = -1
    per_worker_warm: dict = {}
    try:
        while True:
            t = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                break
            now = time.perf_counter()
            k += 1
            if first is None:
                first = now
                if plan:  # every worker exists by now, started with this process's affinity
                    main_affinity = cpusets.set_affinity(plan["main"])
            waits.append(now - t)  # the consumer blocked, waiting for this batch
            arrivals.append(now)
            for row in batch[-1].tolist():
                stamps.append((int(row[0]), row[1], row[2]))
                if t_warm is None:
                    per_worker_warm[int(row[0])] = per_worker_warm.get(int(row[0]), 0) + 1
            if t_warm is None:
                enough = (k + 1 >= warm_rounds * workers if kind == "dataloader" else
                          (len(per_worker_warm) >= workers
                           and (k + 1) * BATCH >= warm_rounds * BATCH * workers))
                if enough:
                    t_warm, k_warm = now, k
            elif now - t_warm >= seconds:
                memory = _workers_memory()  # in full flow, before the workers stop
                break
    finally:
        stop = getattr(batches, "_shutdown_workers", None)
        if stop is not None:
            stop()  # each DataLoader worker finishes the batch it is making, then ends
        elif hasattr(batches, "close"):
            batches.close()
        if allowed:
            cpusets.set_affinity(allowed)
    if t_warm is None:
        raise RuntimeError(f"the warm-up did not finish: {k + 1} batches in "
                           f"{time.perf_counter() - started:.0f} s")
    t_end = arrivals[-1]
    span = max(t_end - t_warm, 1e-9)
    n_items = (k - k_warm) * BATCH
    window_waits = waits[k_warm + 1:]
    # each worker's items and busy time inside the timed part
    per: dict = {}
    for wid, t0, t1 in stamps:
        w = per.setdefault(wid, {"items": 0, "busy_s": 0.0, "item_s": []})
        overlap = min(t1, t_end) - max(t0, t_warm)
        if overlap > 0:
            w["busy_s"] += overlap
        if t_warm < t1 <= t_end:
            w["items"] += 1
            w["item_s"].append(t1 - t0)
    worker_cpu = {}
    for r in cpusets.worker_records(log_dir):
        worker_cpu[r.get("worker")] = r
    rows = []
    for wid in sorted(per):
        w = per[wid]
        rec = worker_cpu.get(wid, {})
        cpu = rec.get("cpu")
        rows.append({"worker": wid, "cpu": cpu, "kind": kinds.get(cpu) if cpu is not None else None,
                     "items": w["items"], "busy_share": round(w["busy_s"] / span, 3),
                     "item_seconds_median": round(statistics.median(w["item_s"]), 4)
                     if w["item_s"] else None})
    bins: dict = {}
    for t in arrivals[k_warm + 1:]:
        b = int((t - t_warm) // TIMELINE_SECONDS)
        bins[b] = bins.get(b, 0) + BATCH
    kinds_summary: dict = {}
    for r in rows:
        if r["kind"]:
            kinds_summary.setdefault(r["kind"], []).append(r["items"] / span)
    out = {"config": {"kind": kind, "workers": workers, "pinned": pinned, "opt_out": opt_out,
                      "seconds": seconds, "warm_rounds": warm_rounds, "recipe": recipe(),
                      "crop": crop or 131072, "pin_memory": pin, "plan": plan,
                      "main_affinity": main_affinity,
                      "main_pinned": "after the first batch" if plan else None,
                      "window": getattr(source, "window", None),
                      "prefetch_factor": 2 if kind == "dataloader" else None,
                      "start_method": (os.environ.get("LCMSDECONV_MP_CONTEXT")
                                       or ("spawn" if cpusets.WINDOWS else "platform default"))},
           "startup_seconds": round(first - started, 2) if first is not None else None,
           "warm_up_seconds": round(t_warm - started, 2), "numba_warm_up_seconds": round(warm, 1),
           "timed_seconds": round(span, 2), "items": n_items,
           "items_per_second": round(n_items / span, 3),
           "consumer_wait_share": round(sum(window_waits) / span, 4),
           "consumer_wait_ms": stats([1000.0 * w for w in window_waits], 1),
           "per_worker": rows,
           "per_kind_items_per_second_each": {k2: stats(v, 3) for k2, v in kinds_summary.items()},
           "timeline_items_per_second": [round(bins.get(b, 0) / TIMELINE_SECONDS, 2)
                                         for b in range(max(bins) + 1)] if bins else [],
           "worker_setup": [{k2: v for k2, v in r.items() if k2 != "pid"}
                            for r in cpusets.worker_records(log_dir)][:64]}
    out["memory"] = memory
    if kind == "dispatcher":
        out["dispatcher"] = {"assembly_ms_per_batch": round(
            1000.0 * source.assembly_seconds / max(k + 1, 1), 2)}
    w = out["consumer_wait_ms"]
    ctx.say(f"{kind}{' pinned' if pinned else ''}{' opt-out' if opt_out else ''}, {workers} "
            f"workers: {out['items_per_second']:.1f} items/s; a batch waited for "
            f"{w.get('median', 0):.0f} ms (p90 {w.get('p90', 0):.0f})")
    return out

