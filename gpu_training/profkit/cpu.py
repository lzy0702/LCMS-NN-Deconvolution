"""Processor experiments: synthesis of the runs' training spectra (items of train_all's recipe,
v5 today, exactly as training makes them: FrameDataset with the GPU package's settings, singly
charged ions kept, isotope-charge channels, the measured channel as setup's probe measures the
workers, 131 072-bin crops), by single-threaded workers as in training.

- :func:`loader`: training's own data loading (lcmsdeconv.synth.dataset.make_dataloaders, as
  train() builds it) for this process: torch's DataLoader or the ordered loader of
  lcmsdeconv.synth.pool, the workers placed as train() places them (lcmsdeconv.nn.cpus.plan:
  not bound, sharing the cores the training process does not keep, or each on a core of its
  own), opted out of Windows' power throttling or not, at their priority, and this process in
  the training process's part. The spectra per second delivered, the wait for each batch, what
  each worker and each kind of core made (training's own count, ItemStats), the start-up;
- :func:`core_map`: one process pinned to each of a few logical processors in turn (two of
  each kind in the profile: the rate of one worker on a performance and on an efficient core),
  the same items on each;
- :func:`throttle`: on one performance and one efficient core, Windows' power throttling
  (EcoQoS) left to Windows or opted out of, at normal and below-normal priority;
- :func:`cprofile_items`: where the time of an item goes, on one core of each kind.
"""

from __future__ import annotations

import contextlib
import os
import statistics
import time

from . import cpusets
from .common import ROOT, Unavailable, recipe, recipe_mixture, stats

BATCH = 4
#: rounds before a loader measurement's timing starts (a round: a batch from every worker;
#: torch's DataLoader hands batch k to worker k mod N, so that only whole rounds take the same
#: time again); the ordered loader once every worker has made a spectrum and as many spectra
#: as these rounds hold have arrived
WARM_ROUNDS = 2
#: the timed part of a loader measurement (torch's DataLoader: on to the end of a round), and
#: the width of its timeline's bins
LOADER_SECONDS, TIMELINE_SECONDS = 25.0, 5.0
#: the spectra a loader measurement's training loader holds: far more than are read (about
#: 2000 on the computer the package was made for), so that the measurement stops in full flow
LOADER_ITEMS = 200_000
#: their seed (probe.workers_run's spectra)
LOADER_SEED = 4242


def recipe_mix(crop: int | None = None) -> list:
    """The runs' training spectra (common.recipe_mixture: train_all's recipe as training
    configures it), cropped to ``crop`` bins where given (smoke runs and tests)."""
    mix = recipe_mixture()
    if crop:
        for _w, c in mix:
            c.crop_size = int(crop)
    return mix


def recipe_dataset(length: int, base_seed: int = 0, crop: int | None = None):
    """The runs' training spectra (recipe_mix) as a FrameDataset, and the mixture."""
    from lcmsdeconv.synth.dataset import FrameDataset

    mix = recipe_mix(crop)
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


def spread_cores(procs: list[dict], per_kind: int) -> list[int]:
    """``per_kind`` logical processors of each kind this process may use, spread from the first
    to the last of that kind (the very first left out where the kind has more: it often serves
    the system's interrupts), in the order of the kinds."""
    order = sorted(cpusets.usable(procs), key=lambda p: p["cpu"])
    out: list[int] = []
    for kind in dict.fromkeys(p.get("kind") for p in order):
        mine = [p["cpu"] for p in order if p.get("kind") == kind]
        if len(mine) > per_kind:
            mine = mine[1:]
        n = max(1, min(int(per_kind), len(mine)))
        picks = [mine[round(i * (len(mine) - 1) / max(n - 1, 1))] for i in range(n)]
        out += [c for c in dict.fromkeys(picks) if c not in out]
    return out


def core_map(ctx, items: int = 12, crop: int | None = None, priority: str = "below_normal",
             cpus: list | None = None, per_kind: int | None = None) -> dict:
    """Items per second of one process pinned to each logical processor of ``cpus`` in turn
    (default: ``per_kind`` of each kind, spread_cores; without it every one this process may
    use), the same ``items`` items on each (after two unpinned to warm up), at the workers'
    priority."""
    procs = _processors()
    by_cpu = {p["cpu"]: p for p in procs if p.get("cpu") is not None}
    allowed = cpusets.current_affinity() or sorted(by_cpu)
    chosen = cpus or (spread_cores(procs, per_kind) if per_kind else sorted(by_cpu))
    todo = [c for c in chosen if c in allowed]
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
            ctx.say(f"cpu {cpu} ({p.get('kind')}): {items / sum(ts):.2f} spectra/s")
    finally:
        cpusets.set_affinity(allowed)
        cpusets.set_priority("normal")
    by_kind: dict = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["items_per_second"])
    per_item = [statistics.median(r["item_seconds"]["median"] for r in rows)] if rows else []
    return {"config": {"items": items, "item_indices": f"0-{items - 1} of seed 0",
                       "recipe": recipe(), "crop": crop or 131072, "priority": prio,
                       "per_kind": per_kind, "cpus": todo,
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


# ------------------------------------------------------------------ training's data loading
def training_placement(topo, allowed, loader: str, binding: str, qos: str, priority: str,
                       workers: int):
    """Where train() runs ``workers`` data workers of a GPU training run with these settings
    (train_charge_net.py's --loader, --cpu-binding, --worker-qos, --worker-priority), as
    lcmsdeconv.nn.train._placement decides it through lcmsdeconv.nn.cpus.plan: each worker on
    a core of its own ("core") only with the ordered loader, torch's DataLoader, which hands
    whole batches to its workers in turn, sharing the cores instead ("pool"); None where nothing
    changes (not bound, below-normal priority, power throttling left to Windows), as train()
    has it then."""
    from lcmsdeconv.nn import cpus

    mode = "pool" if binding == "core" and loader != "ordered" else binding
    placement = cpus.plan(topo, workers=workers, allowed=allowed, mode=mode, priority=priority,
                          qos=qos)
    if (placement.mode, placement.priority, placement.qos) == ("none", "below_normal", "system"):
        return None
    return placement


class TrainingProcess:
    """This process in the training process's part of a placement, as train() plays it
    (lcmsdeconv.nn.train._MainPlacement): opted out of power throttling from the start where
    the placement says so, before its workers start; bound to its own cores once they run (a
    process takes its parent's affinity when it is created, so bound earlier the workers would
    start on its cores); both given back at the end."""

    def __init__(self, placement):
        from lcmsdeconv.nn import cpus

        self.placement = placement
        self.allowed = cpus.allowed_cpus()
        self.pinned = False
        self._qos = None  # (opted out?, the power throttling before)

    def start(self) -> dict:
        from lcmsdeconv.nn import cpus

        if self.placement is None or self.placement.qos != "high":
            return {}
        before = cpus.get_power_throttling()
        self._qos = (cpus.set_power_throttling(False), before)
        return {"opted_out": self._qos[0]}

    def bind(self, pids: list) -> dict:
        """Bind this process to its cores where the placement binds, and read where the workers
        ``pids`` may run and at which priority."""
        from lcmsdeconv.nn import cpus

        p = self.placement
        out: dict = {"workers": len(pids)}
        if p is not None and p.mode != "none":
            self.pinned = bool(cpus.apply_main(p, qos=False).get("pinned"))
            masks = cpus.affinity_of(pids)
            out["bound_as_planned"] = sum(1 for i, pid in enumerate(pids)
                                          if masks.get(pid) == sorted(p.worker_cpus(i)))
            out["worker_affinity"] = [masks.get(pid) for pid in pids]
        out["main_bound"] = self.pinned
        out["main_affinity"] = cpus.allowed_cpus()
        out["worker_priority"] = priorities(pids)
        return out

    def restore(self) -> None:
        from lcmsdeconv.nn import cpus

        if self.pinned and self.allowed:
            cpus.pin_process(self.allowed)
            self.pinned = False
        if self._qos is not None and self._qos[0]:
            cpus.restore_power_throttling(self._qos[1])
        self._qos = None


def priorities(pids: list) -> dict:
    """{priority: processes} of ``pids``: Windows' priority class by name, elsewhere the nice
    value."""
    try:
        import psutil
    except ImportError:
        return {}
    out: dict = {}
    for pid in pids:
        try:
            v = int(psutil.Process(pid).nice())
        except (psutil.Error, OSError):
            continue
        key = cpusets.PRIORITY_NAMES.get(v, hex(v)) if cpusets.WINDOWS else f"nice {v}"
        out[key] = out.get(key, 0) + 1
    return out


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


def worker_rows(table, span: float, kind_of_cpu: dict) -> list[dict]:
    """Each worker's spectra in the timed part, from training's count of them (an ItemStats
    table: worker x logical processor x (spectra, seconds, processor seconds, moved)): how many,
    the share of the time it was making them, its seconds a spectrum, the logical processors it
    made them on, the one it made most on and that one's kind."""
    from lcmsdeconv.nn.cpus import cpu_ranges

    rows = []
    n_cpus = table.shape[1] - 1  # the last column: a processor that could not be read
    for w in range(table.shape[0]):
        t = table[w]
        items, secs = int(t[:, 0].sum()), float(t[:, 1].sum())
        used = [c for c in range(n_cpus) if t[c, 0] > 0]
        main = max(used, key=lambda c: t[c, 0]) if used else None
        rows.append({"worker": w, "items": items, "busy_share": round(secs / span, 3),
                     "seconds_per_item": round(secs / items, 4) if items else None,
                     "on_processor": round(float(t[:, 2].sum()) / secs, 3) if secs > 0 else None,
                     "cpus": cpu_ranges(used), "cpu": main,
                     "kind": kind_of_cpu.get(main, "?") if main is not None else None,
                     "moved": int(t[:, 3].sum())})
    return rows


def _stop(tl, batches) -> None:
    """End the workers: the ordered loader's pool, or torch's DataLoader's (each finishes the
    batch it is making)."""
    with contextlib.suppress(Exception):  # the process ends next: nothing outlives it
        if getattr(tl, "pool", None) is not None:
            tl.close()
        elif batches is not None and hasattr(batches, "_shutdown_workers"):
            batches._shutdown_workers()


def loader(ctx, loader: str = "dataloader", binding: str = "none", qos: str = "system",
           priority: str = "below_normal", workers: int = 22, seconds: float = LOADER_SECONDS,
           warm_rounds: int = WARM_ROUNDS, crop: int | None = None) -> dict:
    """``workers`` data workers making training's batches of BATCH spectra for this process
    exactly as train() has them made: lcmsdeconv.synth.dataset.make_dataloaders with ``loader``
    ("dataloader": torch's, whole batches to the workers in turn, delivered in order, so that
    the slowest worker sets the pace; "ordered": lcmsdeconv.synth.pool, single spectra to
    whichever worker is free, the same batches) and the placement train() gives ``binding``,
    ``qos`` and ``priority`` (training_placement), the batches into pinned memory where CUDA is
    available, this process in the training process's part (TrainingProcess). After the
    warm-up (WARM_ROUNDS), for ``seconds`` (torch's DataLoader: on to the end of a round): the
    spectra per second delivered; the wait for each batch; what each worker and each kind of
    core made and how busy the workers were (training's own count, ItemStats); the start-up
    (the loaders made, which makes the validation spectra; the first batch; every worker's first
    spectrum); the workers' memory; where the workers and this process ran."""
    import torch

    from lcmsdeconv.nn import cpus
    from lcmsdeconv.synth.dataset import make_dataloaders
    from lcmsdeconv.synth.pool import LOADERS

    for name, value, allowed_values in (("loader", loader, LOADERS),
                                        ("binding", binding, cpus.MODES),
                                        ("qos", qos, cpus.QOS_MODES),
                                        ("priority", priority, cpus.PRIORITIES)):
        if value not in allowed_values:
            raise ValueError(f"{name} {value!r}: one of {', '.join(allowed_values)}")
    if qos == "high" and not cpusets.WINDOWS:
        raise Unavailable("the opt-out of power throttling (--worker-qos high) is Windows'")
    workers = max(1, int(workers))
    mix = recipe_mix(crop)
    warm = warm_up(mix)  # before the workers start, as train() does
    topo = cpus.detect()
    kinds = topo.kind_of_cpu()
    placement = training_placement(topo, cpus.allowed_cpus(), loader, binding, qos, priority,
                                   workers)
    main = TrainingProcess(placement)
    pin = torch.cuda.is_available()
    started = time.perf_counter()
    main_start = main.start()
    tl = batches = placed = None
    arrivals, waits = [], []
    t_first = t_all = t_warm = None
    k = k_warm = -1
    made0 = made = summary = memory = twice = None
    twice0 = 0
    try:
        tl, _vl = make_dataloaders(mix, LOADER_ITEMS, BATCH, batch_size=BATCH,
                                   num_workers=workers, base_seed=LOADER_SEED, pin_memory=pin,
                                   loader=loader, placement=placement)
        t_made = time.perf_counter()
        pool = getattr(tl, "pool", None)
        stats_ = tl.item_stats
        if pool is not None:  # its workers run already: train() binds this process now
            placed = main.bind(pool.pids)
        batches = iter(tl)
        while True:
            t = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:  # every spectrum read: far more than a measurement takes
                break
            now = time.perf_counter()
            del batch
            k += 1
            waits.append(now - t)
            arrivals.append(now)
            if t_first is None:
                t_first = now
                if pool is None:  # torch's workers run once the first batch is in: train()
                    placed = main.bind([w.pid for w in getattr(batches, "_workers", [])])
            if t_warm is None:
                if t_all is None and all(w["items"] > 0 for w in stats_.per_worker()):
                    t_all = now
                # torch's DataLoader: warm_rounds whole rounds in, every worker's included
                if k + 1 >= warm_rounds * workers and (pool is None or t_all is not None):
                    t_warm, k_warm = now, k
                    made0 = stats_.snapshot()
                    twice0 = pool.duplicates if pool is not None else 0
            elif now - t_warm >= seconds and (pool is not None or (k - k_warm) % workers == 0):
                memory = _workers_memory()  # in full flow, before the workers stop
                made = stats_.snapshot() - made0
                summary = stats_.summary(made0, kinds, now - t_warm)
                twice = pool.duplicates - twice0 if pool is not None else None
                break
    finally:
        _stop(tl, batches)
        main.restore()
    if t_warm is None or k <= k_warm:
        raise RuntimeError(f"the warm-up did not finish: {k + 1} batches in "
                           f"{time.perf_counter() - started:.0f} s")
    t_end = arrivals[-1]
    span = max(t_end - t_warm, 1e-9)
    if made is None:  # every spectrum read before the time was up
        made = stats_.snapshot() - made0
        summary = stats_.summary(made0, kinds, span)
        twice = pool.duplicates - twice0 if pool is not None else None
    n_items = (k - k_warm) * BATCH
    timed_waits = waits[k_warm + 1:]
    bins: dict = {}
    for t in arrivals[k_warm + 1:]:
        b = int((t - t_warm) // TIMELINE_SECONDS)
        bins[b] = bins.get(b, 0) + BATCH
    p = placement
    out = {"config": {"loader": loader, "binding": binding, "qos": qos, "priority": priority,
                      "workers": workers, "seconds": seconds, "warm_rounds": warm_rounds,
                      "recipe": recipe(), "crop": crop or 131072, "pin_memory": pin,
                      "items": f"from item 0 of seed {LOADER_SEED}", "topology": topo.describe(),
                      "placement": p.describe() if p is not None else
                      "as before: workers not bound, below-normal priority, power throttling "
                      "left to Windows",
                      "placement_mode": p.mode if p is not None else "none",
                      "main_cpus": list(p.main) if p is not None else [],
                      "worker_kinds": list(p.kinds) if p is not None else [],
                      "window": getattr(tl, "window", None),
                      "prefetch_factor": 2 if loader == "dataloader" else None,
                      "start_method": (os.environ.get("LCMSDECONV_MP_CONTEXT")
                                       or ("spawn" if cpusets.WINDOWS else "platform default"))},
           "startup_seconds": {"loaders": round(t_made - started, 2),
                               "first_batch": round(t_first - started, 2),
                               "every_worker": (round(t_all - started, 2)
                                                if t_all is not None else None),
                               "warm": round(t_warm - started, 2)},
           "numba_warm_up_seconds": round(warm, 1),
           "timed_seconds": round(span, 2), "items": n_items,
           "items_per_second": round(n_items / span, 3),
           "consumer_wait_share": round(sum(timed_waits) / span, 4),
           "consumer_wait_ms": stats([1000.0 * w for w in timed_waits], 1),
           "made": summary, "per_worker": worker_rows(made, span, kinds),
           "made_twice": twice,
           "timeline_items_per_second": [round(bins.get(b, 0) / TIMELINE_SECONDS, 2)
                                         for b in range(max(bins) + 1)] if bins else [],
           "placed": placed, "main_start": main_start, "memory": memory,
           "shared_gb": round(float(getattr(tl, "shared_gb", 0.0)), 3)}
    w = out["consumer_wait_ms"]
    busy = (summary or {}).get("workers_busy")
    ctx.say(f"{loader}, {binding}, {qos}, {workers} worker{'s' if workers != 1 else ''}: "
            f"{out['items_per_second']:.1f} "
            f"spectra/s" + (f", workers busy {100 * busy:.0f} %" if busy is not None else "")
            + f"; a batch waited for {w.get('median', 0):.0f} ms (p90 {w.get('p90', 0):.0f})")
    return out
