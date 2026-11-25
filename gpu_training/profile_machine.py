"""Measure this computer for the developers: training steps on the GPU and the synthesis of
training spectra on the processor, each experiment in a process of its own, packed into one zip
file to send back.

    gpu_training\\profile.bat                 (Windows: the setup where needed, then this)
    bash gpu_training/profile.sh             (Linux)
    python gpu_training/profile_machine.py [--only gpu|cpu|e2e] [--minutes N] [--smoke]

About 35 minutes on a computer like the one the package was made for (24 logical processors, an
RTX 4080 SUPER), at most --minutes (default DEFAULT_MINUTES). Keep the computer otherwise idle
while it runs. It changes no setting of the computer: priorities, affinities and power
throttling are set for its own processes only, which end with each experiment.

The plan, in order. An experiment that would not fit the time left is skipped and recorded as
such, except the environment, the GPU's batches and the training run, which always run; the
time of the reserved experiments still to come (about 20 minutes in all) is kept from the
others, which are skipped, or given less time, first.

1. The environment: the processor and each logical processor's kind (performance or
   efficient), Windows and its power scheme and power mode, the memory, the GPU and its driver,
   PyTorch, nvidia-smi, and setup's record of this computer (gpu_training/machine.json).
2. The GPU (both views of the full LadderNet as train_all trains them, on train_all's recipe,
   4 spectra a step, under the 14 GB cap, on the runs' first training batches; every step
   experiment measures the at-cap batch first). Reserved: each view's step without speedups
   (the baseline every later GPU experiment's timeout and step limit derive from), then setup's
   choice of each view, what training runs. Then each GPU speedup alone, all four, and setup's
   speedups at other head chunkings (16 and 32 charges at a time, all at once). A step far
   slower than its baseline ends the experiment as "pathological", with the steps and memory
   measured before it, and a configuration with a part that failed so at the same chunking is
   not measured.
3. The processor, reserved (about 15 minutes): training's own data loading (torch's DataLoader
   and the ordered loader; the workers not bound, sharing the cores the training process does
   not keep, or each on a core of its own; power throttling left to Windows or opted out of) at
   training's number of data workers (22 of 24 logical processors) and two fewer; one worker
   alone on two performance and two efficient cores; power throttling and priority on one core
   of each kind.
4. A short real training run exactly as train_all trains its window run here (two epochs of
   400 steps): setup's GPU speedups and head chunk, its data workers and their data loading.
5. Then what is detail or information, the first skipped should time run out: the phases,
   kernels and memory of each view's best configuration, nvidia-smi during a steady minute,
   where a spectrum's time goes; bf16 autocast, 8 spectra a step, nvidia-smi on the comb view,
   more allocator settings.

Everything goes to runs/profile_<date-time>/ (results.json with one record per experiment,
summary.txt, each experiment's log, CSV) and is packed into runs/profile_upload_<date-time>.zip:
JSON, logs and CSV only, the package folder's path replaced by <package> and the home folder's
by ~. Large intermediate files (the batches, about 1.5 GB) are deleted at the end.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu_training"))

from profkit import common  # noqa: E402
from profkit.common import (  # noqa: E402
    THREAD_VARS,
    Pathological,
    Unavailable,
    read_json,
    write_json,
)

RUNS = ROOT / "runs"
WINDOWS = sys.platform == "win32"
#: the time budget: the plan's estimate is about 34 minutes on the computer the package was made
#: for, and this leaves room for a slower one and for experiments ended as pathological or at
#: their timeout (an experiment that would not fit is skipped, the reserved ones last)
DEFAULT_MINUTES = 60.0
GROUPS = ("gpu", "cpu", "e2e")
#: what the upload holds: files of these kinds, each at most MAX_FILE_MB
PACK_SUFFIXES = (".json", ".jsonl", ".log", ".txt", ".csv", ".gz")
MAX_FILE_MB = 25.0
#: the end of setup.log packed with the results
SETUP_LOG_TAIL = 300_000
#: an experiment's estimated seconds on the computer the package was made for, and its timeout
#: until its view's baseline has run (then Runner.derived_timeout; the end-to-end check: the
#: workers' start, two epochs of 1600 spectra at 30-40 a second, their validation; a loader
#: experiment: its workers' start, twice for torch's DataLoader, which starts its own for the
#: validation spectra, the warm-up and profkit.cpu.LOADER_SECONDS). A step experiment took 11 s
#: on that computer, the end-to-end check 112 s
COSTS = {"env": (15, 300), "prepare": (45, 1200), "steps": (15, 300), "phases": (20, 300),
         "profile": (35, 600), "memory": (22, 300), "smi": (75, 600), "autocast": (35, 600),
         "batch8": (30, 600), "cores": (35, 300), "throttle": (50, 600),
         "cprofile": (30, 600), "loader": (60, 240), "e2e": (180, 1200)}
#: once a view's step without speedups (its baseline, gpu.steps.<view>.none; the other view's
#: until it has run) has run, a GPU experiment is given TIMEOUT_FACTOR times the baseline's
#: wall time, in proportion to its kind's estimate against a step experiment's, within
#: [TIMEOUT_FLOOR, TIMEOUT_CEILING] seconds; the floor leaves the process its start (PyTorch,
#: the batches) and its first steps, which may compile and autotune
TIMEOUT_FACTOR, TIMEOUT_FLOOR, TIMEOUT_CEILING = 8.0, 90.0, 600.0
#: each GPU experiment's step limit (its plan entry's "stall", read by
#: profkit.common.StallWatchdog): a step after the first STALL_EXEMPT of a start (which may
#: compile and autotune) that takes more than STALL_FACTOR times the baseline's median (at least
#: STALL_FLOOR_S; STALL_NO_BASELINE_S before any baseline) ends the experiment as
#: "pathological"; a step that never returns is ended, with its process, at the experiment's
#: timeout. Kinds whose steps hold more than a training step of 4 spectra have their limit
#: scaled (STALL_WEIGHTS). On the computer the package was made for a step took 0.1-0.16 s and
#: the faulty ones 15-32 s
STALL_FACTOR, STALL_FLOOR_S, STALL_NO_BASELINE_S, STALL_EXEMPT = 20.0, 5.0, 15.0, 2
STALL_WEIGHTS = {"batch8": 2.0, "profile": 3.0}
#: how a step experiment ended that makes every configuration of its view and head chunking
#: with all of its speedups (and more) not worth measuring: it would only repeat the fault
FAULTS = ("timeout", "pathological")
#: the time kept for the reserved experiments still to run (the GPU's baselines and setup's
#: choices, the processor block, the training run) while the others run: their
#: estimates at this computer's pace times RESERVE_MARGIN, at most RESERVE_SHARE of the budget.
#: The others are skipped, and their timeouts shortened, rather than eat into it
RESERVE_SHARE, RESERVE_MARGIN = 0.6, 1.2
#: best_config leaves out a configuration whose steps were unsteady (the 90th percentile of its
#: steps on the real batches above UNSTEADY_P90 times their median) or whose reserved memory
#: came near the cap (its peak at least NEAR_CAP_SHARE of the cap less CONTEXT_GB, the CUDA
#: context's share: lcmsdeconv.nn.train._cap_gpu_memory). On the computer the package was made
#: for such configurations took seconds a step, with no out-of-memory error, at 13.2 GB reserved
#: under a cap of 13.25 for tensors, where the healthy ones stayed at 11.5 GB or less
UNSTEADY_P90, NEAR_CAP_SHARE, CONTEXT_GB = 2.0, 0.95, 0.75
#: the ways of loading the data the processor block measures (train_charge_net.py's --loader,
#: --cpu-binding and --worker-qos; the workers at below-normal priority, as training runs them):
#: at training's number of data workers all of them, at two fewer the first LESS_WORKER_WAYS.
#: First torch's DataLoader as train_all runs it unless setup chose otherwise, then opted out
#: of power throttling (the first remedy should Windows throttle the workers), the ordered
#: loader with each worker on a core of its own, with and without, then the other bindings.
#: Each worker on a core of its own is the ordered loader's alone: train() makes torch's
#: DataLoader share the cores instead. The opt-out is Windows': elsewhere those ways are left out
LOADER_WAYS = (("dataloader", "none", "system"), ("dataloader", "none", "high"),
               ("ordered", "core", "system"), ("ordered", "core", "high"),
               ("dataloader", "pool", "system"), ("ordered", "none", "system"),
               ("ordered", "none", "high"), ("ordered", "pool", "system"),
               ("ordered", "pool", "high"))
LESS_WORKER_WAYS = 4
#: seconds a stopped experiment's process is given to end before the runner goes on without it
STOP_WAIT = 30.0
#: PYTORCH_CUDA_ALLOC_CONF of each memory variant (None: the default allocator)
ALLOC_VARIANTS = {"default": None, "gc08": "garbage_collection_threshold:0.8",
                  "split512": "max_split_size_mb:512",
                  "gc08_split512": "garbage_collection_threshold:0.8,max_split_size_mb:512",
                  "expandable": "expandable_segments:True",
                  "async": "backend:cudaMallocAsync"}


class Console:
    """What the runner prints, also written to runner.log."""

    def __init__(self, path: Path):
        self.path = path

    def say(self, text: str = "") -> None:
        try:
            print(text, flush=True)
        except UnicodeEncodeError:  # a console that cannot show a character
            print(text.encode("ascii", "replace").decode("ascii"), flush=True)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:  # held by another program: the console has shown it
            pass


# ------------------------------------------------------------------ the plan
def entry(eid: str, group: str, title: str, fn: str, kind: str, params: dict | None = None,
          smoke: bool = False, **more) -> dict:
    """A plan entry. ``more``: ``needs`` (ids that must have run ok first), ``essential`` (runs
    whatever the budget), ``reserved`` (its time is kept from the experiments before it that
    are neither: RESERVE_SHARE), ``resolve``, ``view``, ``env``, ``workers``, ``variant``,
    ``setup_choice``."""
    est, timeout = COSTS[kind]
    return {"id": eid, "group": group, "title": title, "fn": fn, "kind": kind,
            "params": params or {}, "estimate": max(3, est // 4) if smoke else est,
            "timeout": timeout, "needs": [], **more}


def setup_choices(views=("window", "comb")) -> dict:
    """What training uses on this computer, from setup's machine.json as train_all reads it
    (profkit.e2e.train_all_choice): each view's GPU speedups and head chunk, the data workers
    and their data loading."""
    from profkit.e2e import train_all_choice

    machine = read_json(ROOT / "gpu_training" / "machine.json") or {}
    chosen = {v: train_all_choice(machine, v) for v in views}
    first = chosen[views[0]]
    return {"machine": machine, "workers": first["workers"], "data": first["data"],
            "views": {v: {"speedups": c["speedups"], "chunk": c["chunk"]}
                      for v, c in chosen.items()}}


def gpu_plan(smoke: bool, choices: dict) -> dict:
    """The GPU experiments by their part of the plan (build_plan orders them): "prepare"; "core",
    reserved: each view's step without speedups (the baseline the later GPU experiments'
    timeouts and step limits derive from), then setup's choice of each view, what training runs;
    "matrix": each speedup alone, all four, setup's speedups at other head chunkings (16 and 32
    charges at a time, all at once where setup chunks); "detail": the phases, kernels and memory
    of each view's best, nvidia-smi on the window view's; "info", run last, most useful first:
    bf16 autocast and 8 spectra a step (for information), nvidia-smi on the comb view, the
    allocator settings least likely to matter. Every step experiment measures its at-cap batch
    first (its time and memory whatever the real batches then do)."""
    from lcmsdeconv.nn.train import GPU_SPEEDUPS

    sz = ({"size": "small", "crop": 16384} if smoke else {})
    few = ({"synced": 4, "free": 2, "warm": 1, "cap_warm": 1, "cap_timed": 1} if smoke else {})
    prepare = entry("gpu.prepare", "gpu", "the runs' first 40 training batches, the at-cap "
                    "batch and the weights", "profkit.gpu:prepare", "prepare",
                    {"batches": 4 if smoke else 40, **({"workers": 2} if smoke else {}), **sz},
                    smoke, essential=True)
    core: list = []
    matrix: list = []
    needs = ["gpu.prepare"]

    def step_entry(view, name, sp, chunk, title_how, **more):
        return entry(f"gpu.steps.{view}.{name}", "gpu",
                     f"steps, {view} view, {'+'.join(sp) or 'no speedups'}, {title_how}",
                     "profkit.gpu:steps", "steps",
                     {"view": view, "speedups": list(sp), "chunk": chunk, "label": name,
                      "cap_first": True, **few, **sz}, smoke, needs=needs, view=view, **more)

    setup_entries = []
    for view in ("window", "comb"):
        choice = choices["views"][view]
        chunk, greedy = choice["chunk"], list(choice["speedups"])
        how = "all charges at once" if chunk is None else f"{chunk} charges at a time"
        configs = [("none", []), *((s, [s]) for s in GPU_SPEEDUPS), ("all", list(GPU_SPEEDUPS))]
        if greedy and greedy not in [c for _n, c in configs]:
            configs.append(("greedy", greedy))
        greedy_name = next((n for n, c in configs if c == greedy), "greedy")
        for name, sp in configs:
            if name == "none":
                core.append(step_entry(view, name, sp, chunk, how, reserved=True))
            elif name == greedy_name:  # setup's choice: what training runs
                setup_entries.append(step_entry(view, name, sp, chunk, how + " (setup's choice)",
                                                reserved=True, setup_choice=True))
            else:
                matrix.append(step_entry(view, name, sp, chunk, how))
        for c in (None, 16, 32):
            if c == chunk:
                continue  # setup's choice, measured above
            tag = "unchunked" if c is None else f"chunk{c}"
            matrix.append(step_entry(view, f"{greedy_name}.{tag}", greedy, c,
                                     "all charges at once" if c is None else f"{c} at a time"))
    core += setup_entries  # both baselines first: every later limit derives from them
    best = {"label": "best"}
    detail: list = []
    info: list = []
    for view in ("window", "comb"):
        detail.append(entry(f"gpu.phases.{view}", "gpu", f"phases of the best step, {view} view",
                            "profkit.gpu:phases", "phases",
                            {"view": view, **best,
                             **({"steps_n": 4, "warm": 1} if smoke else {}), **sz},
                            smoke, needs=needs, resolve={"best": view}))
        detail.append(entry(f"gpu.profile.{view}.best", "gpu", f"profile, {view} view, the best",
                            "profkit.gpu:profile", "profile",
                            {"view": view, **best,
                             **({"steps_n": 1, "warm": 1} if smoke else {}), **sz},
                            smoke, needs=needs, resolve={"best": view}))
    chunk = choices["views"]["window"]["chunk"]
    detail.append(entry("gpu.profile.window.baseline", "gpu", "profile, window view, no speedups",
                        "profkit.gpu:profile", "profile",
                        {"view": "window", "speedups": [], "chunk": chunk, "label": "none",
                         **({"steps_n": 1, "warm": 1} if smoke else {}), **sz}, smoke,
                        needs=needs))
    last_memory = []  # combined settings, expandable segments (which Windows may lack), async
    for view, variants in (("window", ("default", "gc08", "split512")),
                           ("comb", ("default", "gc08", "split512")),
                           ("window", ("gc08_split512", "expandable", "async"))):
        for variant in variants:
            conf = ALLOC_VARIANTS[variant]
            e = entry(f"gpu.memory.{view}.{variant}", "gpu",
                      f"memory over 40 batches, {view} view, {conf or 'default allocator'}",
                      "profkit.gpu:memory", "memory",
                      {"view": view, **best, "variant": variant,
                       "batches": 4 if smoke else 40, **sz}, smoke, needs=needs,
                      resolve={"best": view},
                      env={"PYTORCH_CUDA_ALLOC_CONF": conf} if conf else {"PYTORCH_CUDA_ALLOC_CONF": None})
            (detail if variant == "default" else last_memory).append(e)
    detail.append(entry("gpu.smi.window", "gpu", "a steady minute with nvidia-smi, window view",
                        "profkit.gpu:smi_loop", "smi",
                        {"view": "window", **best, "seconds": 5.0 if smoke else 60.0, **sz},
                        smoke, needs=needs, resolve={"best": "window"}))
    for view in ("window", "comb"):
        info.append(entry(f"gpu.autocast.{view}", "gpu",
                          f"bf16 autocast (information), {view} view", "profkit.gpu:autocast",
                          "autocast", {"view": view, **best,
                                       **({"steps_n": 2, "warm": 1} if smoke else {}), **sz},
                          smoke, needs=needs, resolve={"best": view}))
        info.append(entry(f"gpu.batch8.{view}", "gpu", f"8 spectra a step (information), {view} "
                          f"view", "profkit.gpu:batch8", "batch8",
                          {"view": view, **best, **({"steps_n": 2, "warm": 1} if smoke else {}),
                           **sz}, smoke, needs=needs, resolve={"best": view}))
    e = entry("gpu.smi.comb", "gpu", "half a steady minute with nvidia-smi, comb view",
              "profkit.gpu:smi_loop", "smi",
              {"view": "comb", **best, "seconds": 5.0 if smoke else 30.0, **sz}, smoke,
              needs=needs, resolve={"best": "comb"})
    e["estimate"] = e["estimate"] // 2
    info.append(e)
    return {"prepare": [prepare], "core": core, "matrix": matrix, "detail": detail,
            "info": info + last_memory}


def worker_counts(logical: int, smoke: bool) -> list[int]:
    """The numbers of data workers the loader experiments run: training's
    (lcmsdeconv.nn.machine.cpu_workers: one per logical processor of every core but the two the
    training process keeps, 22 of 24) and two fewer; in a smoke run two and one."""
    from lcmsdeconv.nn.machine import cpu_workers

    full = cpu_workers(int(logical))
    counts = [2, 1] if smoke else [full, full - 2]
    return list(dict.fromkeys(max(1, n) for n in counts))


def loader_ways(windows: bool) -> list[tuple]:
    """LOADER_WAYS this computer can run (the opt-out of power throttling is Windows')."""
    return [w for w in LOADER_WAYS if windows or w[2] != "high"]


def cpu_plan(smoke: bool, logical: int, windows: bool) -> dict:
    """The processor's experiments: "block", reserved: training's data loading
    (profkit.cpu.loader) in each of LOADER_WAYS at training's number of workers, one worker
    alone on two cores of each kind, power throttling and priority on one core of each kind,
    then the first LESS_WORKER_WAYS ways at two fewer workers; and "detail", after the training
    run: where a spectrum's time goes."""
    from train_all import data_words

    crop = {"crop": 16384} if smoke else {}
    short = {"seconds": 4.0, "warm_rounds": 1} if smoke else {}
    counts = worker_counts(logical, smoke)
    ways = loader_ways(windows)

    def loaders(n: int, chosen: list) -> list:
        out = []
        for loader, binding, qos in chosen:
            words = data_words({"loader": loader, "cpu_binding": binding, "worker_qos": qos})
            out.append(entry(f"cpu.loader.n{n}.{loader}.{binding}.{qos}", "cpu",
                             f"{n} worker{'s' if n != 1 else ''}: {words}",
                             "profkit.cpu:loader", "loader",
                             {"loader": loader, "binding": binding, "qos": qos, "workers": n,
                              **short, **crop}, smoke, workers=n,
                             variant=f"{loader}.{binding}.{qos}", reserved=True))
        return out

    block = loaders(counts[0], ways)
    block.append(entry("cpu.cores", "cpu", "spectra per second of one worker alone, two "
                       "logical processors of each kind", "profkit.cpu:core_map", "cores",
                       {"items": 2 if smoke else 12, "per_kind": 2, **crop}, smoke,
                       reserved=True))
    block.append(entry("cpu.throttle", "cpu", "power throttling and priority, one core of each "
                       "kind", "profkit.cpu:throttle", "throttle",
                       {"items": 2 if smoke else 6, "reps": 1 if smoke else 2, **crop}, smoke,
                       reserved=True))
    for n in counts[1:]:
        block += loaders(n, [w for w in ways if w in LOADER_WAYS[:LESS_WORKER_WAYS]])
    detail = [entry("cpu.cprofile", "cpu", "where an item's time goes (cProfile), one core of "
                    "each kind", "profkit.cpu:cprofile_items", "cprofile",
                    {"items": 2 if smoke else 6, **crop}, smoke)]
    return {"block": block, "detail": detail}


def e2e_plan(smoke: bool) -> list:
    """The short real training run, essential and reserved: it runs whatever the budget,
    exactly as train_all trains its window run on this computer (profkit.e2e.train_all_choice:
    setup's GPU speedups and head chunk, its data workers and their data loading; in a smoke
    run two workers), the check that shows whether the data or the GPU sets training's pace."""
    params = {"view": "window"}
    if smoke:
        params.update(size="small", epochs=1, train_len=16, val_len=2, minutes=5.0, workers=2)
    return [entry("e2e.train", "e2e", "a short real training run as train_all trains here",
                  "profkit.e2e:train_check", "e2e", params, smoke, essential=True,
                  reserved=True)]


def build_plan(groups, env: dict, smoke: bool, only_ids=None) -> list:
    """Every experiment to run after ``env``, in order: the GPU's batches; reserved, each view's
    baseline and setup's choice; the GPU's speedups and chunkings; reserved, the processor's
    block (training's data loading, one worker per core, power throttling); the end-to-end
    check (essential); then what is detail or information, the first skipped should time run
    out: the best steps' phases, kernels and memory, where a spectrum's time goes, the GPU's
    information and the allocator settings least likely to matter. A reserved experiment's time
    is kept from those before it (Runner.reserve_after)."""
    common.add_paths()
    choices = setup_choices()
    logical = int((env.get("cpu") or {}).get("logical") or os.cpu_count() or 2)
    windows = (env.get("os") or {}).get("build") is not None or WINDOWS
    g = gpu_plan(smoke, choices) if "gpu" in groups else {}
    c = cpu_plan(smoke, logical, windows) if "cpu" in groups else {}
    plan = [*g.get("prepare", []), *g.get("core", []), *g.get("matrix", []),
            *c.get("block", []), *(e2e_plan(smoke) if "e2e" in groups else []),
            *g.get("detail", []), *c.get("detail", []), *g.get("info", [])]
    for e in plan:
        e["fallback"] = choices["views"].get(e.get("view") or (e.get("resolve") or {}).get("best"))
    if only_ids:
        keep = {e["id"] for e in plan if any(e["id"].startswith(p) for p in only_ids)}
        for e in plan:
            if e["id"] in keep:
                keep.update(e.get("needs") or [])
        plan = [e for e in plan if e["id"] in keep]
    return plan


# ------------------------------------------------------------------ what was found best
def unsafe(rec: dict) -> str | None:
    """Why a step experiment that ran to its end is no configuration to train with, None where
    it is one: its steps on the real batches were unsteady (their 90th percentile above
    UNSTEADY_P90 times their median), its reserved memory came near the cap (its peak, at the
    at-cap batch or over the whole experiment, at least NEAR_CAP_SHARE of the cap less
    CONTEXT_GB), or both."""
    res = rec.get("result") or {}
    why = []
    wall = (res.get("real") or {}).get("wall_ms") or {}
    med, p90 = wall.get("median") or res.get("median_ms"), wall.get("p90")
    if med and p90 and p90 > UNSTEADY_P90 * med:
        why.append(f"unsteady: p90 {p90:.0f} ms over {UNSTEADY_P90:g} x the median {med:.0f} ms")
    peaks = [m.get("peak_reserved_gb") for m in (res.get("memory") or {},
                                                  (res.get("cap") or {}).get("memory") or {})]
    peak = max((float(x) for x in peaks if x is not None), default=None)
    cap = float((rec.get("config") or {}).get("gpu_memory_gb")
                or (res.get("config") or {}).get("gpu_memory_gb") or 14.0)
    if peak is not None and peak >= NEAR_CAP_SHARE * (cap - CONTEXT_GB):
        why.append(f"near the cap: {peak:.2f} GB reserved, at least {NEAR_CAP_SHARE:g} x "
                   f"({cap:g} - {CONTEXT_GB:g}) GB")
    return "; ".join(why) or None


def best_config(records: list, view: str, fallback: dict | None = None) -> dict:
    """The fastest configuration measured for ``view`` (median wall time of a step on the
    real batches) among those whose losses and gradients agree with the step without speedups,
    whose at-cap step fitted the GPU, and which were steady and kept off the cap (unsafe: the
    others are listed in ``excluded``); setup's choice (``fallback``) if none was measured."""
    best = None
    excluded = []
    for r in records:
        if not r["id"].startswith(f"gpu.steps.{view}.") or r.get("outcome") != "ok":
            continue
        res = r.get("result") or {}
        ms = res.get("median_ms")
        agrees = (res.get("agreement") or {}).get("same") is not False
        fits = "error" not in (res.get("cap") or {})
        why = unsafe(r)
        if why:
            excluded.append({"id": r["id"], "why": why})
            continue
        if ms and agrees and fits and (best is None or ms < best[0]):
            best = (ms, r)
    if best is None:
        fb = fallback or {}
        return {"speedups": list(fb.get("speedups") or []), "chunk": fb.get("chunk"),
                "label": "setup's choice (nothing measured" + (
                    " steady and off the cap)" if excluded else ")"),
                "median_ms": None, "excluded": excluded}
    cfg = best[1]["config"]
    return {"speedups": list(cfg.get("speedups") or []), "chunk": cfg.get("chunk"),
            "label": f"best: {best[1]['id'].split('.', 3)[-1]}", "median_ms": best[0],
            "excluded": excluded}


# ------------------------------------------------------------------ running the experiments
class Runner:
    """Runs plan entries one at a time, each in a process of its own with a timeout, and keeps
    their records (results.json after each)."""

    def __init__(self, run_dir: Path, console: Console, smoke: bool, budget_s: float,
                 header: dict):
        self.run_dir = run_dir
        self.console = console
        self.smoke = smoke
        self.budget_s = budget_s
        self.header = header
        self.records: list = []
        self.plan: list = []
        self.started = time.time()
        self.interrupted = False
        self.save_error: str | None = None  # why results.json could not be written last time
        for sub in ("raw", "logs", "work/plan"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

    def scale(self, group: str | None = None) -> float:
        """The scale of the estimates on this computer: the seconds the experiments that ran to
        their end ("ok") took against their estimates, those of ``group`` where they add up to a
        minute of estimates, else all of them but never under 1; 1 until then; within [0.5, 3].
        An experiment stopped at its timeout or ended as pathological measures its fault, not
        this computer's pace."""
        def ratio(rows):
            est = sum(float(r.get("estimate_s") or 0) for r in rows)
            if est < 60:
                return None
            real = sum(float(r.get("seconds") or 0) for r in rows)
            return min(3.0, max(0.5, real / est))

        done = [r for r in self.records if r.get("outcome") == "ok" and r.get("seconds")]
        if group is None:
            return ratio(done) or 1.0
        mine = ratio([r for r in done if r.get("group") == group])
        # a group not measured yet is never made faster by another's pace (the GPU's
        # experiments ran in half their estimates, the processor's need not)
        return mine or max(1.0, ratio(done) or 1.0)

    # ------------------------------------------------ limits derived from what has run
    def baseline(self, view: str | None) -> dict | None:
        """The step without speedups of ``view`` (the other view's until that one has run ok),
        the reference of every later GPU experiment's timeout and step limits: its id, wall
        seconds and median ms a step; None before either has run ok."""
        views = [view] if view else []
        views += [v for v in ("window", "comb") if v not in views]
        for v in views:
            r = self.outcome_of(f"gpu.steps.{v}.none")
            res = (r or {}).get("result") or {}
            if r and r.get("outcome") == "ok" and res.get("median_ms") and r.get("seconds"):
                return {"id": r["id"], "seconds": float(r["seconds"]),
                        "median_ms": float(res["median_ms"])}
        return None

    @staticmethod
    def _view(e: dict) -> str | None:
        return (e.get("params") or {}).get("view") or e.get("view")

    def derived_timeout(self, e: dict) -> tuple[float, str]:
        """A GPU experiment's timeout and where it comes from: TIMEOUT_FACTOR times its
        baseline's wall time, in proportion to its kind's estimate against a step experiment's,
        within [TIMEOUT_FLOOR, TIMEOUT_CEILING]; the static one (COSTS) before any baseline, for
        the batches' preparation and for the other groups."""
        base = self.baseline(self._view(e))
        if e.get("group") != "gpu" or e.get("kind") == "prepare" or base is None:
            return e["timeout"], "static"
        ratio = COSTS[e["kind"]][0] / COSTS["steps"][0]
        t = TIMEOUT_FACTOR * base["seconds"] * ratio
        return round(min(TIMEOUT_CEILING, max(TIMEOUT_FLOOR, t)), 1), f"baseline {base['id']}"

    def stall_limits(self, e: dict) -> dict | None:
        """The step limit of a GPU experiment's watchdog (its entry's "stall"; see the STALL_
        constants): from its baseline's median step, absolute before any baseline."""
        if e.get("group") != "gpu" or e.get("kind") == "prepare":
            return None
        base = self.baseline(self._view(e))
        out = {"limit_s": STALL_NO_BASELINE_S, "exempt": STALL_EXEMPT, "base_ms": None,
               "baseline": None}
        if base is not None:
            weight = STALL_WEIGHTS.get(e.get("kind"), 1.0)
            out.update(base_ms=base["median_ms"], baseline=base["id"],
                       limit_s=round(max(STALL_FLOOR_S,
                                         STALL_FACTOR * weight * base["median_ms"] / 1000.0), 2))
        return out

    def faulty_part(self, e: dict) -> dict | None:
        """For a step experiment of the GPU's matrix: an earlier step experiment of its view and
        head chunking whose speedups are all among its own and which ended in a fault (FAULTS):
        this configuration would only repeat it. None for the reserved ones (the baselines and
        setup's choices, which run whatever)."""
        if not e["id"].startswith("gpu.steps.") or e.get("reserved"):
            return None
        p = e.get("params") or {}
        mine = set(p.get("speedups") or [])
        for r in self.records:
            c = r.get("config") or {}
            if (r["id"].startswith("gpu.steps.") and r["id"] != e["id"]
                    and r.get("outcome") in FAULTS
                    and c.get("view") == p.get("view") and c.get("chunk") == p.get("chunk")
                    and set(c.get("speedups") or []) <= mine):
                return r
        return None

    def reserve_after(self, k: int) -> float:
        """Seconds kept for the reserved and essential experiments after plan entry ``k``
        (1-based) that have not run: their estimates at this computer's pace times
        RESERVE_MARGIN, at most RESERVE_SHARE of the budget."""
        keep = sum(x["estimate"] * self.scale(x.get("group")) for x in self.plan[k:]
                   if (x.get("reserved") or x.get("essential")) and self.outcome_of(x["id"]) is None)
        return min(RESERVE_MARGIN * keep, RESERVE_SHARE * self.budget_s)

    def save(self, extra: dict | None = None, final: bool = False) -> Path | None:
        """results.json with every record so far, written after each experiment. Never stops
        the profile: where the file is held (open in another program, for example) it is
        written after a later experiment, and if the last save (``final``) fails too, the
        records go to results_final.json instead, which is packed as well. Returns the file
        written, None if none was."""
        data = {**self.header, **(extra or {}), "experiments": self.records}
        try:
            write_json(self.run_dir / "results.json", data, attempts=20 if final else 4)
            self.save_error = None
            return self.run_dir / "results.json"
        except OSError as exc:
            first = self.save_error is None
            self.save_error = f"{type(exc).__name__}: {exc}"
        if not final:
            if first:
                self.console.say(f"    results.json could not be written ({self.save_error[:160]})"
                                 f"; it is written again after the next experiment")
            return None
        alt = self.run_dir / "results_final.json"
        try:
            write_json(alt, {**data, "results_json_error": self.save_error})
        except OSError as exc:
            self.console.say(f"The results could not be written ({exc}): the logs are packed")
            return None
        self.console.say(f"results.json could not be written ({self.save_error[:160]}): the "
                         f"results are in {alt.name}")
        return alt

    def record(self, e: dict, outcome: str, error: str | None = None, **more) -> dict:
        rec = {"id": e["id"], "group": e["group"], "title": e["title"], "fn": e["fn"],
               "config": e["params"], "outcome": outcome, "error": error,
               "estimate_s": e["estimate"], "timeout_s": e["timeout"], **more}
        for key in ("workers", "variant", "view", "timeout_from", "stall", "reserved",
                    "setup_choice"):
            if key in e:
                rec[key] = e[key]
        if e.get("env"):
            rec["env"] = e["env"]
        self.records.append(rec)
        self.save()
        return rec

    def outcome_of(self, eid: str) -> dict | None:
        return next((r for r in self.records if r["id"] == eid), None)

    def resolve(self, e: dict) -> None:
        """Fill in what depends on earlier results: a GPU experiment's timeout and step limit
        (from its baseline), the best configuration of a view."""
        e["timeout"], e["timeout_from"] = self.derived_timeout(e)
        stall = self.stall_limits(e)
        if stall is not None:
            e["stall"] = stall
        how = e.get("resolve") or {}
        if "best" in how:
            b = best_config(self.records, how["best"], e.get("fallback"))
            e["params"].update(speedups=b["speedups"], chunk=b["chunk"], label=b["label"])

    def run(self, e: dict, k: int, n: int) -> dict:
        rec = self.start(e, k, n)
        if rec["outcome"] in ("skipped", "unavailable") and rec.get("seconds") is None:
            self.console.say(f"[{k}/{n}] {e['id']}: {rec['outcome']} ({rec['error'][:110]})")
        return rec

    def start(self, e: dict, k: int, n: int) -> dict:
        for need in e.get("needs") or []:
            got = self.outcome_of(need)
            if got is None or got["outcome"] != "ok":
                why = "it was not run" if got is None else f"it was {got['outcome']}"
                if got and got.get("error"):
                    why += f": {str(got['error']).strip().splitlines()[-1][:200]}"
                outcome = "unavailable" if got and got["outcome"] == "unavailable" else "skipped"
                return self.record(e, outcome, f"needs {need}, and {why}")
        if self.interrupted:
            return self.record(e, "skipped", "the profile was interrupted")
        part = self.faulty_part(e)
        if part is not None:
            err = str(part.get("error") or "").strip().splitlines()
            return self.record(e, "skipped", f"{part['id']}, whose speedups it has, ended "
                                             f"{part['outcome']} at this head chunking"
                                             + (f" ({err[-1][:160]})" if err else ""))
        elapsed = time.time() - self.started
        self.resolve(e)
        need = e["estimate"] * self.scale(e.get("group"))
        if not e.get("essential"):
            # a reserved experiment may take what is left of the budget; any other only what
            # is not kept for the reserved ones after it, and never more (its timeout shortened)
            reserved = bool(e.get("reserved"))
            keep = 0.0 if reserved else self.reserve_after(k)
            room = self.budget_s - elapsed - keep
            if need > room:
                return self.record(e, "skipped", f"the time budget ({self.budget_s / 60:.0f} min) "
                                                 f"would be exceeded" + (
                    f" ({keep / 60:.1f} min of it kept for the reserved experiments still to "
                    f"run)" if keep > 0 else ""))
            if not reserved and room < e["timeout"]:
                e["timeout"], e["timeout_from"] = round(room, 1), "the time budget left"
        left = self.remaining_estimate(k)
        self.console.say(f"[{k}/{n}] {e['title']} ({e['id']}; about {need:.0f} s, at most "
                         f"{e['timeout']:.0f}; {elapsed / 60:.1f} min done, about "
                         f"{left / 60:.0f} min to go)")
        rec = self.launch(e)
        self.console.say(f"    {rec['outcome']} in {rec.get('seconds', 0):.0f} s"
                         + (f": {brief(rec)}" if brief(rec) else ""))
        return rec

    def remaining_estimate(self, k: int) -> float:
        return sum(x["estimate"] * self.scale(x.get("group")) for x in self.plan[k - 1:]
                   if self.outcome_of(x["id"]) is None)

    def launch(self, e: dict) -> dict:
        """Start the experiment's process, echo its [say] lines, stop it at its timeout (with
        every process it started), and record what it left."""
        eid = e["id"]
        write_json(self.run_dir / "work" / "plan" / f"{eid}.json", e)
        log = self.run_dir / "logs" / f"{eid}.log"
        raw = self.run_dir / "raw" / f"{eid}.json"
        cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--child", eid,
               "--run-dir", str(self.run_dir)]
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        added = [var for var in THREAD_VARS if var not in env]
        for var in added:
            env[var] = "1"  # one thread per process, as training's workers
        env[common.THREADS_ADDED_ENV] = ",".join(added)  # the training run drops them again
        if self.smoke:
            env[common.SMOKE_ENV] = "1"
        for k, v in (e.get("env") or {}).items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS
              else {"start_new_session": True})
        t0 = time.time()
        outcome = error = None
        code = None
        with open(log, "wb") as fh:
            fh.write(f"=== {common.now_text()} {eid}\n".encode())
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    **kw)
            pos = fh.tell()
            try:
                while True:
                    code = proc.poll()
                    pos = self.echo(log, pos)
                    if code is not None:
                        break
                    if time.time() - t0 > e["timeout"]:
                        common.kill_tree(proc.pid)
                        outcome, error = "timeout", f"no result within {e['timeout']:g} s"
                        try:
                            proc.wait(timeout=STOP_WAIT)
                        except subprocess.TimeoutExpired:  # stuck in a driver call, for one
                            error += (f"; its process did not end within {STOP_WAIT:g} s of "
                                      f"being stopped and was left behind")
                        break
                    time.sleep(0.25)
            except KeyboardInterrupt:
                common.kill_tree(proc.pid)
                self.interrupted = True
                outcome, error = "interrupted", "Ctrl+C"
        seconds = time.time() - t0
        got = read_json(raw) if raw.exists() else None
        tail = self.tail(log)
        if outcome is None:
            if got is not None:
                outcome, error = got.get("outcome", "error"), got.get("error")
            else:
                outcome = "crashed"
                error = f"exit code {code} without a result; the end of its log:\n{tail}"
        more = {"seconds": round(seconds, 1), "exit_code": code, "log": f"logs/{eid}.log",
                "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))}
        progress = read_json(common.progress_file(self.run_dir, eid))
        if progress and outcome not in ("ok", "pathological", "unavailable"):
            more["progress"] = progress  # how far it got: where a step never returned
        warns = self.warnings(log)
        if warns:
            more["log_warnings"] = warns
        if got is not None:
            more["result"] = got.get("result")
            more["process"] = got.get("process")
        return self.record(e, outcome, error, **more)

    def echo(self, log: Path, pos: int) -> int:
        """Show the experiment's [say] lines written since ``pos``; returns the new position."""
        try:
            with open(log, "rb") as fh:
                fh.seek(pos)
                data = fh.read()
        except OSError:
            return pos
        end = data.rfind(b"\n")
        if end < 0:
            return pos
        for line in data[:end].decode("utf-8", errors="replace").splitlines():
            if line.startswith("[say] "):
                self.console.say("    " + line[6:])
        return pos + end + 1

    @staticmethod
    def warnings(log: Path, most: int = 10) -> list[str]:
        """The distinct warning lines of an experiment's log (an allocator setting this PyTorch
        does not support says so when CUDA starts, early in the log)."""
        try:
            text = log.read_bytes()[:4_000_000].decode("utf-8", errors="replace")
        except OSError:
            return []
        lines = [ln.strip() for ln in text.splitlines() if "warn" in ln.lower()]
        return list(dict.fromkeys(ln[:300] for ln in lines))[:most]

    @staticmethod
    def tail(log: Path, n: int = 4000) -> str:
        try:
            data = log.read_bytes()[-n:]
        except OSError:
            return ""
        return data.decode("utf-8", errors="replace")

    def run_all(self, plan: list) -> None:
        self.plan = plan
        for k, e in enumerate(plan, start=1):
            try:
                self.run(e, k, len(plan))
            except Exception:  # a fault of the runner itself is recorded; the next one runs
                self.runner_failed(e, traceback.format_exc())

    def runner_failed(self, e: dict, text: str) -> None:
        """Record that the runner itself failed on ``e`` (its own record, if it has one yet,
        keeps its outcome and gains the error)."""
        last = text.strip().splitlines()[-1] if text.strip() else "?"
        self.console.say(f"    the runner failed on {e.get('id')}: {last[:200]}")
        try:
            rec = self.outcome_of(e.get("id"))
            if rec is None:
                self.record(e, "error", f"the runner failed:\n{text[-4000:]}")
            else:
                rec["runner_error"] = text[-4000:]
                self.save()
        except Exception:  # nothing more can be kept: the console has said it
            pass


def child_main(eid: str, run_dir: Path) -> int:
    """One experiment in this process (the runner started it): its result, or the error that
    stopped it, goes to raw/<id>.json."""
    common.add_paths()
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    e = read_json(run_dir / "work" / "plan" / f"{eid}.json") or {"id": eid, "fn": "?"}
    ctx = common.Context(run_dir, e)
    t0 = time.time()
    rc = 0
    try:
        module, name = e["fn"].split(":")
        fn = getattr(importlib.import_module(module), name)
        rec = {"outcome": "ok", "result": fn(ctx, **(e.get("params") or {}))}
    except Unavailable as exc:
        rec = {"outcome": "unavailable", "error": str(exc)}
    except Pathological as exc:  # a step far slower than its limit: what came before is kept
        print(str(exc), flush=True)
        rec = {"outcome": "pathological", "error": str(exc), "result": exc.partial}
    except BaseException:  # an error of any kind is the record (KeyboardInterrupt too)
        rec = {"outcome": "error", "error": traceback.format_exc()[-6000:]}
        traceback.print_exc()
        rc = 1
    rec["seconds"] = round(time.time() - t0, 2)
    rec["process"] = process_info()
    write_json(run_dir / "raw" / f"{eid}.json", rec)
    return rc


def process_info() -> dict:
    """The experiment process's settings, for its record."""
    from profkit import cpusets

    out = {"python": sys.version.split()[0], "affinity": cpusets.current_affinity(),
           "priority": cpusets.priority_now(),
           "env": {k: os.environ.get(k) for k in (*THREAD_VARS, "PYTORCH_CUDA_ALLOC_CONF")}}
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        out["torch_threads"] = torch.get_num_threads()
    return out


# ------------------------------------------------------------------ the summary
def _ms(v) -> str:
    return "-" if v is None else f"{v:.1f}"


def progress_words(rec: dict) -> str:
    """Where an experiment that did not end by itself had got to (its progress file): the phase
    it had begun and the steps it had taken; "" without one."""
    p = rec.get("progress") or {}
    if not p.get("phase"):
        return ""
    return f" (it had begun its phase {p['phase']} after {p.get('steps_before_stall', 0)} steps)"


def brief(rec: dict) -> str:
    """One line of an experiment's result for the console."""
    res = rec.get("result") or {}
    fn = rec.get("fn", "")
    if rec.get("outcome") != "ok":
        err = str(rec.get("error") or "").strip().splitlines()
        return (err[-1][:200] if err else "") + progress_words(rec)
    try:
        if fn.endswith(":steps"):
            r = res["real"]["wall_ms"]
            cap = res.get("cap") or {}
            agree = res.get("agreement") or {}
            same = "" if agree.get("same") is None else (
                "; the same loss and gradients" if agree["same"] else f"; NOT the same: "
                                                                      f"{agree.get('why')}")
            return (f"{r['median']:.1f} ms a step (p10 {r['p10']:.1f}, p90 {r['p90']:.1f}), "
                    f"free-running {res['free_running']['ms_per_step']:.1f}, at the cap "
                    + (f"{cap['wall_ms']['median']:.1f}" if "wall_ms" in cap else
                       cap.get("error", "-")[:60]) + same)
        if fn.endswith(":phases"):
            return (f"GPU {res['step_gpu_ms']['median']:.1f} ms a step, host busy "
                    f"{res['host_busy_ms']['median']:.1f}, nonzero wait "
                    f"{res['nonzero_wait_ms']['median']:.1f}: {res['verdict'].split(':')[0]}")
        if fn.endswith(":profile"):
            k = (res.get("kernels") or [{}])[0]
            busy = res.get("gpu_busy_share")
            return (f"{res['kernels_per_step']:.0f} kernels a step, {res['kernel_ms_per_step']:.1f}"
                    f" ms of kernels"
                    + (f" in a step of {res['ms_per_step']:.1f} ms ({100 * busy:.0f} % busy)"
                       if busy is not None and res.get("ms_per_step") else "")
                    + f"; top: {k.get('name', '-')[:60]}"
                    + ("; the trace was not written" if res.get("trace_error") else ""))
        if fn.endswith(":memory"):
            return (f"{res.get('peak_reserved_gb', 0):.2f} GB reserved at most, "
                    f"{res.get('alloc_retries', 0)} retries, {res.get('ooms', 0)} out of memory, "
                    f"step {_ms(res['step_ms'].get('median'))} ms")
        if fn.endswith(":smi_loop"):
            s = res.get("nvidia_smi") or {}
            util = (s.get("utilization.gpu") or {}).get("mean")
            return (f"{res['ms_per_step']:.1f} ms a step, {res['spectra_per_second']:.1f} "
                    f"spectra/s; GPU utilisation {util if util is not None else '-'} %")
        if fn.endswith(":autocast"):
            return "; ".join(f"{m} {_ms((v.get('wall_ms') or {}).get('median'))} ms"
                             for m, v in res["modes"].items())
        if fn.endswith(":batch8"):
            return "; ".join(f"{b}: {v.get('spectra_per_second', v.get('error'))} spectra/s"
                             for b, v in res["batches"].items())
        if fn.endswith(":core_map"):
            return "; ".join(f"{k}: {v['median']:.2f} spectra/s ({v['n']} processors)"
                             for k, v in res["by_kind"].items())
        if fn.endswith(":throttle"):
            words = {"opt_out_gain_normal": "opt-out at normal priority",
                     "opt_out_gain_below_normal": "opt-out at below normal",
                     "below_normal_change_default": "below normal",
                     "below_normal_change_opt_out": "below normal, opted out"}
            return "; ".join(
                f"{k} core {v['cpu']}: " + ", ".join(f"{words.get(e, e)} {100 * x:+.1f} %"
                                                   for e, x in v["effects"].items())
                for k, v in res["cores"].items())
        if fn.endswith(":loader"):
            w = res.get("consumer_wait_ms") or {}
            busy = (res.get("made") or {}).get("workers_busy")
            return (f"{res['items_per_second']:.1f} spectra/s"
                    + (f", workers busy {100 * busy:.0f} %" if busy is not None else "")
                    + f"; a batch waited for {_ms(w.get('median'))} ms (p90 "
                    f"{_ms(w.get('p90'))}); the first after "
                    f"{(res.get('startup_seconds') or {}).get('first_batch')} s")
        if fn.endswith(":cprofile_items"):
            return "; ".join(f"{k} {v['wall_s']:.1f} s" for k, v in res["profiles"].items())
        if fn.endswith(":train_check"):
            last = (res.get("epochs") or [{}])[-1]
            return last.get("line", f"exit code {res.get('exit_code')}")[:200]
        if fn.endswith(":environment"):
            c = res.get("cpu") or {}
            return f"{c.get('name')}; {c.get('counts')}; CUDA {res['torch'].get('cuda')}"
        if fn.endswith(":prepare"):
            return f"{res['config']['batches']} batches in {res['seconds_making']:.0f} s"
    except (KeyError, TypeError, IndexError, AttributeError, ValueError):
        return ""
    return ""


def summary_lines(records: list, env: dict, header: dict) -> list[str]:
    """The human summary: the computer, each view's GPU steps, the processor, the end-to-end
    check, and what failed."""
    lines = []
    by_id = {r["id"]: r for r in records}
    counts: dict = {}
    for r in records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    lines.append(f"Profile of {header['started']} to {header.get('finished', '?')}"
                 f" ({header.get('minutes', 0):.1f} min): {len(records)} experiments, "
                 + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    try:
        cpu, tor, mem = env.get("cpu") or {}, env.get("torch") or {}, env.get("memory") or {}
        osr = env.get("os") or {}
        rel = osr.get("release") or {}
        scheme = (osr.get("power_scheme") or {})
        mode = ((osr.get("power_mode") or {}).get("effective") or {}).get("known_as")
        lines.append(f"Computer: {cpu.get('name')}; {cpu.get('logical')} logical processors "
                     f"{cpu.get('counts')} ({cpu.get('source')}); {mem.get('total_gb')} GB RAM")
        lines.append(f"  {osr.get('platform')}" + (f", {rel.get('DisplayVersion')}" if rel else "")
                     + (f"; power scheme {scheme.get('name') or scheme.get('known_as')}"
                        if scheme else "") + (f"; power mode {mode}" if mode else ""))
        gpus = (env.get("nvidia_smi") or {}).get("gpus") or [{}]
        name = tor.get("gpu") or gpus[0].get("name")
        gpu = (f"GPU {name}, driver {gpus[0].get('driver_version')}" if name else
               "no GPU (" + str((env.get("nvidia_smi") or {}).get("why", "")) + ")")
        lines.append(f"  {gpu}; PyTorch {tor.get('version')} (CUDA {tor.get('cuda_build')}, "
                     f"available: {tor.get('cuda')})")
    except Exception as exc:  # the summary never stops the packing
        lines.append(f"(the computer's summary failed: {exc})")
    for view in ("window", "comb"):
        steps = [r for r in records if r["id"].startswith(f"gpu.steps.{view}.")]
        if not steps:
            continue
        if not any(r["outcome"] == "ok" for r in steps):
            gpu_runs = [r for r in records if r["id"].startswith("gpu.") and r["id"].count(view)]
            lines.append(f"GPU, {view} view: not measured ({len(gpu_runs)} experiments "
                         f"{steps[0]['outcome']}: {brief(steps[0])[:160]})")
            continue
        lines.append(f"GPU, {view} view: ms a step on the runs' first 8 batches (median [p10-p90]"
                     f"; free-running; at the active-bin cap)")
        base = next((r for r in steps if r["id"] == f"gpu.steps.{view}.none"
                     and r["outcome"] == "ok"), None)
        base_ms = base and base["result"].get("median_ms")
        b = best_config(records, view)
        left_out = {x["id"]: x["why"] for x in b["excluded"]}
        for r in steps:
            res = r.get("result") or {}
            name = r["id"].split(".", 3)[-1] + (" (setup's)" if r.get("setup_choice") else "")
            if r["outcome"] != "ok":
                cap = (((res.get("measured") or {}).get("cap") or {}).get("wall_ms") or {})
                lines.append(f"  {name:28s} {r['outcome']}: {brief(r)[:150]}"
                             + (f"; at the cap {cap['median']:.1f} ms before it"
                                if cap.get("median") else ""))
                continue
            w = res["real"]["wall_ms"]
            cap = res.get("cap") or {}
            gain = (f" {100 * (1 - w['median'] / base_ms):+5.1f} %" if base_ms else "")
            agree = res.get("agreement") or {}
            flag = "" if agree.get("same") is not False else "  (NOT the same results)"
            if r["id"] in left_out:
                flag += f"  (left out: {left_out[r['id']]})"
            lines.append(f"  {name:28s} {w['median']:7.1f} [{w['p10']:.1f}-{w['p90']:.1f}]{gain}"
                         f"; {res['free_running']['ms_per_step']:.1f}; "
                         f"{_ms((cap.get('wall_ms') or {}).get('median'))}"
                         f"; {res.get('memory', {}).get('peak_reserved_gb', '-')} GB{flag}")
        lines.append(f"  best: {b['label']} ({'+'.join(b['speedups']) or 'no speedups'}, "
                     f"{'all charges at once' if b['chunk'] is None else b['chunk']})")
        ph = by_id.get(f"gpu.phases.{view}")
        if ph and ph["outcome"] == "ok":
            res = ph["result"]
            parts = [f"{p} {v['gpu_ms'].get('median', 0):.1f}/{v['host_ms'].get('median', 0):.1f}"
                     for p, v in res["phases"].items()]
            lines.append(f"  phases, ms GPU/host: {', '.join(parts)}; nonzero wait "
                         f"{res['nonzero_wait_ms'].get('median', 0):.1f}; {res['verdict']}")
        for tag in ("baseline", "best"):
            pr = by_id.get(f"gpu.profile.{view}.{tag}")
            if pr and pr["outcome"] == "ok":
                ks = pr["result"].get("kernels") or []
                top = ", ".join(f"{100 * k['share']:.0f} % {k['name'][:40]}" for k in ks[:3])
                busy = pr["result"].get("gpu_busy_share")
                busy = "" if busy is None else (f", the GPU busy {100 * busy:.0f} % of a step of "
                                                f"{pr['result'].get('ms_per_step')} ms")
                lines.append(f"  kernels ({tag}): {pr['result']['kernels_per_step']:.0f} a step"
                             f"{busy}; {top}")
        mems = [r for r in records if r["id"].startswith(f"gpu.memory.{view}.")]
        if mems:
            lines.append("  memory: " + "; ".join(
                f"{r['id'].rsplit('.', 1)[-1]} "
                + (f"{r['result'].get('peak_reserved_gb', 0):.1f} GB, "
                   f"{r['result'].get('alloc_retries', 0)} retries, "
                   f"{r['result'].get('ooms', 0)} OOM" if r["outcome"] == "ok" else r["outcome"])
                for r in mems))
        sm = by_id.get(f"gpu.smi.{view}")
        if sm and sm["outcome"] == "ok":
            s = sm["result"].get("nvidia_smi") or {}

            def mean(key, s=s):
                return (s.get(key) or {}).get("mean")

            lines.append(f"  nvidia-smi over {sm['result']['seconds']:.0f} s: "
                         f"{sm['result']['spectra_per_second']:.1f} spectra/s; utilisation "
                         f"{mean('utilization.gpu')} %, SM {mean('clocks.sm')} MHz, "
                         f"{mean('power.draw')} W, {mean('temperature.gpu')} C; reasons "
                         f"{s.get('clock_event_reasons')}")
        for name in ("autocast", "batch8"):
            r = by_id.get(f"gpu.{name}.{view}")
            if r:
                lines.append(f"  {name} (information): {brief(r)}")
    cores = by_id.get("cpu.cores")
    if cores:
        lines.append(f"Processor, one worker alone: {brief(cores)}")
    thr = by_id.get("cpu.throttle")
    if thr:
        lines.append(f"  power throttling and priority: {brief(thr)}")
    loaders = [r for r in records if r["id"].startswith("cpu.loader.")]
    if loaders:
        counts = list(dict.fromkeys(r.get("workers") for r in loaders))
        lines.append("  training's data loading, spectra/s delivered (workers busy; the wait "
                     "for a batch, 90th percentile; the first batch after), by workers: "
                     + ", ".join(str(n) for n in counts))
        for v in dict.fromkeys(r.get("variant") for r in loaders):
            cells = []
            for n in counts:
                r = next((x for x in loaders if x.get("workers") == n and x.get("variant") == v),
                         None)
                if r is None:
                    cells.append(f"{n}: -")
                elif r["outcome"] != "ok":
                    cells.append(f"{n}: {r['outcome']}")
                else:
                    res = r["result"]
                    w = res.get("consumer_wait_ms") or {}
                    busy = (res.get("made") or {}).get("workers_busy")
                    cells.append(f"{n}: {res['items_per_second']:.1f} ("
                                 + (f"{100 * busy:.0f} %; " if busy is not None else "")
                                 + f"{_ms(w.get('p90'))} ms; "
                                 f"{(res.get('startup_seconds') or {}).get('first_batch')} s)")
            lines.append(f"    {str(v).replace('.', ' '):26s} " + " | ".join(cells))
    cp = by_id.get("cpu.cprofile")
    if cp and cp["outcome"] == "ok":
        for kind, prof in cp["result"]["profiles"].items():
            top = ", ".join(f"{r['function'].split('(')[-1].rstrip(')')} "
                            f"{100 * r['own_share']:.0f} %" for r in prof["by_own"][:4])
            lines.append(f"  cProfile, {kind} core: {top}")
    e2e = by_id.get("e2e.train")
    if e2e:
        res = e2e.get("result") or {}
        cfg = res.get("config") or e2e.get("config") or {}
        sp = "+".join(cfg.get("speedups") or []) or "no speedups"
        chunk = cfg.get("head_chunk")
        how = "all charges at once" if chunk is None else f"{chunk} charges at a time"
        data = f"; {cfg['data_words']}" if cfg.get("data_words") else ""
        lines.append(f"End-to-end, as train_all trains ({cfg.get('view')} view, {sp}, {how}, "
                     f"{cfg.get('workers')} workers{data}): {e2e['outcome']}"
                     + (f": {brief(e2e)}" if e2e["outcome"] != "ok" else ""))
        for ep in res.get("epochs") or []:
            p = ep.get("pace") or {}
            st = ep.get("steady") or {}
            steady = (f"; {st['spectra_per_second']} spectra/s from step {st['from_step']} on"
                      if st.get("spectra_per_second") else "")
            busy = (p.get("workers") or {}).get("workers_busy")
            lines.append(f"  epoch {ep.get('epoch')}: {ep.get('spectra_per_second_epoch')} "
                         f"spectra/s, {100 * float(p.get('data_wait_share', 0)):.0f} % waiting "
                         f"for data, {100 * float(p.get('gpu_busy_share', 0)):.0f} % GPU busy "
                         f"({p.get('gpu_step_ms')} ms a step), {p.get('alloc_retries')} retries"
                         + (f", workers busy {100 * float(busy):.0f} %" if busy is not None
                            else "") + steady)
    bad = [r for r in records if r["outcome"] in ("error", "timeout", "crashed", "pathological")]
    for r in bad:
        err = str(r.get("error") or "").strip().splitlines()
        lines.append(f"FAILED {r['id']} ({r['outcome']}): {err[-1][:160] if err else ''}"
                     f"{progress_words(r)}")
    return lines


# ------------------------------------------------------------------ packing
def sanitizer():
    """Replace, in text, the package folder's path by <package> and the home folder's by ~ (as
    written and as JSON writes them, with doubled backslashes)."""
    pairs = []
    for path, name in ((ROOT, "<package>"), (Path.home(), "~")):
        s = str(path)
        if len(s) < 4:
            continue
        for v in {s, s.replace("\\", "/"), s.replace("\\", "\\\\"), json.dumps(s)[1:-1]}:
            pairs.append((v, name))
    pairs.sort(key=lambda p: -len(p[0]))  # the package folder before the home folder above it

    def clean(text: str) -> str:
        for old, new in pairs:
            text = text.replace(old, new)
        return text

    return clean


def pack(run_dir: Path, zip_path: Path) -> dict:
    """The upload: the run folder's JSON, logs, text, CSV and compressed traces (never the work
    folder, nothing over MAX_FILE_MB), setup's record of this computer and the end of setup.log,
    paths made anonymous; returns what was packed and left out."""
    clean = sanitizer()
    packed, left = [], []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(run_dir).as_posix()
            if rel.startswith("work/"):
                left.append({"file": rel, "why": "intermediate"})
                continue
            if f.suffix not in PACK_SUFFIXES:
                left.append({"file": rel, "why": "not a kind the upload holds"})
                continue
            size = f.stat().st_size
            if size > MAX_FILE_MB * 2**20:
                left.append({"file": rel, "why": f"{size / 2**20:.0f} MB"})
                continue
            arc = f"{run_dir.name}/{rel}"
            if f.suffix == ".gz":
                zf.write(f, arc)
            else:
                zf.writestr(arc, clean(f.read_text(encoding="utf-8", errors="replace")))
            packed.append(rel)
        for f in (ROOT / "gpu_training" / "machine.json", ROOT / "gpu_training" / "setup_done.json"):
            if f.exists():
                zf.writestr(f"{run_dir.name}/setup/{f.name}",
                            clean(f.read_text(encoding="utf-8", errors="replace")))
                packed.append(f"setup/{f.name}")
        log = ROOT / "gpu_training" / "setup.log"
        if log.exists():
            data = log.read_bytes()[-SETUP_LOG_TAIL:].decode("utf-8", errors="replace")
            zf.writestr(f"{run_dir.name}/setup/setup_log_tail.txt", clean(data))
            packed.append("setup/setup_log_tail.txt")
        zf.writestr(f"{run_dir.name}/manifest.json",
                    json.dumps({"packed": packed, "left_out": left}, indent=1))
    return {"packed": len(packed), "left_out": left, "zip_mb": round(zip_path.stat().st_size / 2**20, 2)}


# ------------------------------------------------------------------ main
def env_entry(smoke: bool) -> dict:
    return entry("env", "env", "the environment: processor, Windows, memory, GPU, PyTorch",
                 "profkit.environment:environment", "env", {}, smoke, essential=True)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default=None,
                    help="run only these groups, comma-separated: gpu, cpu, e2e (default: all)")
    ap.add_argument("--minutes", type=float, default=DEFAULT_MINUTES,
                    help=f"the time budget in minutes (default {DEFAULT_MINUTES:g}): experiments "
                         f"that would not fit are skipped and recorded as such")
    ap.add_argument("--smoke", action="store_true",
                    help=f"tiny sizes everywhere, to check that it runs (or {common.SMOKE_ENV}=1)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the intermediate files (batches, weights) in the run folder")
    ap.add_argument("--experiments", default=None, help=argparse.SUPPRESS)  # id prefixes
    ap.add_argument("--runs-dir", type=Path, default=None, help=argparse.SUPPRESS)  # tests
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        return child_main(args.child, args.run_dir)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    groups = set(GROUPS) if not args.only else {g.strip() for g in args.only.split(",") if g.strip()}
    unknown = groups - set(GROUPS)
    if unknown:
        ap.error(f"unknown group {', '.join(sorted(unknown))}: choose from {', '.join(GROUPS)}")
    smoke = args.smoke or common.smoke()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = Path(args.runs_dir or RUNS)
    run_dir = runs / f"profile_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console = Console(run_dir / "runner.log")
    header = {"profile_version": 2, "started": common.now_text(), "options": {
        "only": sorted(groups), "minutes": args.minutes, "smoke": smoke,
        "experiments": args.experiments},
        "setup_exit_code": os.environ.get("LCMSDECONV_SETUP_RC")}
    console.say("=" * 78)
    console.say("Profiling this computer for the developers" + (" (smoke size)" if smoke else ""))
    console.say(f"  results: {run_dir}")
    console.say("  Keep the computer otherwise idle. Ctrl+C stops it and still packs what is done.")
    console.say("=" * 78)
    runner = Runner(run_dir, console, smoke, 60.0 * args.minutes, header)
    awake = keep_awake()
    try:
        return profile(args, groups, smoke, stamp, runs, run_dir, console, header, runner)
    finally:
        if awake is not None:
            awake.stop()


def keep_awake():
    """Keep Windows from going to sleep while the profile runs, as training does (the screen
    may still switch off); None where that is not possible."""
    common.add_paths()
    try:
        from lcmsdeconv.nn.train import _KeepAwake

        awake = _KeepAwake()
        awake.start()
        return awake
    except Exception:  # a convenience only
        return None


def profile(args, groups, smoke, stamp, runs, run_dir, console, header, runner) -> int:
    env_rec = runner.launch(env_entry(smoke))
    console.say(f"    environment: {env_rec['outcome']}" + (f": {brief(env_rec)}" if brief(env_rec)
                                                             else ""))
    env = env_rec.get("result") or {}
    header["environment"] = env
    try:
        plan = build_plan(groups, env, smoke,
                          [p.strip() for p in args.experiments.split(",")]
                          if args.experiments else None)
    except Exception:  # without a plan only the environment is packed
        console.say("The plan of experiments could not be made:\n" + traceback.format_exc())
        plan = []
    total = sum(e["estimate"] for e in plan)
    console.say(f"{len(plan)} experiments, each in a process of its own: about {total / 60:.0f} "
                f"min (at most {args.minutes:g}; what would not fit is skipped)")
    try:
        runner.run_all(plan)
    except KeyboardInterrupt:
        runner.interrupted = True
        console.say("Interrupted: packing what is done.")
    finished = time.time()
    header.update(finished=common.now_text(), minutes=round((finished - runner.started) / 60, 2))
    try:
        lines = summary_lines(runner.records, env, header)
    except Exception:  # the summary is a convenience: the records are packed regardless
        lines = ["(the summary failed:)", traceback.format_exc()]
    try:
        (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:  # held by another program: the console shows the summary below
        console.say(f"summary.txt could not be written: {exc}")
    runner.save({"finished": header["finished"], "minutes": header["minutes"]}, final=True)
    zip_path = runs / f"profile_upload_{stamp}.zip"
    try:
        info = pack(run_dir, zip_path)
    except Exception:  # the run folder holds everything, to be zipped by hand
        info = None
        console.say("The zip file could not be made:\n" + traceback.format_exc()[-1500:])
    if not args.keep_work:
        import shutil

        shutil.rmtree(run_dir / "work", ignore_errors=True)
    console.say("")
    for line in lines:
        console.say(line)
    console.say("")
    if info is not None:
        console.say(f"Send this file to the developers: {zip_path} ({info['zip_mb']:.1f} MB)")
    else:
        console.say(f"Zip the folder {run_dir} and send it to the developers instead")
    # an experiment that failed is a result like any other: the profile itself succeeded
    return 130 if runner.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
