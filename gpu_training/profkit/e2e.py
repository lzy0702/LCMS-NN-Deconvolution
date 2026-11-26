"""The end-to-end check: a short real training run exactly as train_all trains its window run on
this computer, from setup's machine.json (train_all_choice: scripts/train_charge_net.py with
train_all's arguments for the run, the recipe, the full LadderNet of the window view with
isotope-charge channels, the learning rate and the seed; the GPU speedups and head chunk
train_all.speedups_for and train_all.head_chunk_for choose; the data workers, their data
loading and its options, train_all.data_loading_for and train_all.data_flags; the RAM budget
and the GPU memory cap), written to a throwaway output in the run's folder.

Two epochs of 1600 spectra, 400 steps of 4 each. At the start of every epoch training's
persistent DataLoader workers all start a batch at once (the loader primes prefetch_factor
batches per worker), so the GPU first waits about one batch time: about 5 % of an epoch of 400
steps, where it was a fifth of one of 80. The first epoch also holds the workers' start; the
second is the one to read. Records each epoch's pace from the run's log (its share of time
waiting for data, the GPU's share busy and its milliseconds a step, the allocator's retries,
spectra per second, what the workers made and how busy they were), and its steady pace without
its first PRIMED x workers steps, from the times the run's step lines reached the console."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from .common import ROOT, THREADS_ADDED_ENV, read_json
from .gpu import BATCH, GPU_MEMORY_GB, require_cuda

#: a pace line's speed: "epoch took 1.2 min; 41 spectra/s; ..."
SPEED = re.compile(r"epoch took ([\d.]+) min; (\d+) spectra/s")
#: a step line, which training prints every log_every (20) steps: "  epoch 1 step 40/400 loss ..."
STEP = re.compile(r"^\s*epoch (\d+) step (\d+)/(\d+) loss")
#: the batches training's DataLoader primes per worker at the start of every epoch
#: (prefetch_factor, lcmsdeconv.synth.dataset.make_dataloaders)
PRIMED = 2


def epoch_rate(pace: dict, batch: int = BATCH) -> float | None:
    """Spectra per second of one epoch's training steps, from its pace record: its steps over
    their time (``train_steps``, ``train_seconds``). A record without them (training before they
    were recorded) gives it through its GPU busy share, steps x gpu_step_ms / gpu_busy_share
    being the steps' time - which holds only while the share is below its cap of 1.0."""
    if pace.get("train_steps") and pace.get("train_seconds"):
        return round(batch * float(pace["train_steps"]) / float(pace["train_seconds"]), 1)
    if pace.get("gpu_step_ms") and pace.get("gpu_busy_share"):
        return round(batch * 1000.0 * float(pace["gpu_busy_share"]) / float(pace["gpu_step_ms"]), 1)
    return None


def run_args(view: str = "window", size: str = "full", epochs: int = 2, train_len: int = 1600,
             val_len: int = 8) -> list:
    """train_all's training arguments for its run of ``view`` with isotope-charge channels
    (train_all.make_run: the recipe, the network, the learning rate, the seed), with the
    check's own network size, epochs and spectra."""
    from train_all import RUN_TABLE, make_run

    name = next(n for n, spec in RUN_TABLE.items() if spec["view"] == view and spec["iso"])
    args = list(make_run(name)["args"])
    for opt, value in (("--model-size", size), ("--epochs", epochs), ("--train-len", train_len),
                       ("--val-len", val_len)):
        if opt in args:
            args[args.index(opt) + 1] = str(value)
        else:
            args += [opt, str(value)]
    return args


def train_all_choice(machine: dict, view: str = "window") -> dict:
    """How train_all, started without options, trains its run of ``view`` on this computer, by
    setup's machine.json (``machine``) and train_all's own functions: the RAM budget and the GPU
    memory cap (machine.json's, else train_all's defaults); the data loading
    (train_all.data_loading_for: setup's choice, None for train_charge_net.py's defaults, which
    are train_all.DATA_DEFAULTS) and its options (train_all.data_flags); the data workers
    (machine.json's, refitted to the RAM budget where train_all.main refits them, else
    lcmsdeconv.nn.machine.auto_workers); the GPU speedups (train_all.speedups_for), the head
    chunk (train_all.head_chunk_for at the data workers' rate, train_all.data_rate) and the
    crawl guard's pace (train_all.cap_step_seconds)."""
    import train_all

    from lcmsdeconv.nn.machine import (
        MAIN_GB,
        ORDERED_GB_PER_WORKER,
        auto_workers,
        cpu_workers,
        default_ram_budget_gb,
        ram_workers,
    )

    ram = float(machine.get("ram_budget_gb") or default_ram_budget_gb())
    gpu = float(machine.get("gpu_memory_gb") or GPU_MEMORY_GB)
    data = train_all.data_loading_for(machine)
    loader = (data or train_all.DATA_DEFAULTS)["loader"]
    slots_gb = ORDERED_GB_PER_WORKER if loader == "ordered" else 0.0
    rec = machine.get("data_loader")
    setup_loader = rec.get("loader") if isinstance(rec, dict) else None
    workers = machine.get("workers")
    if machine.get("worker_gb") and (ram != machine.get("ram_budget_gb")
                                     or (slots_gb and setup_loader != "ordered")):
        fit = ram_workers(ram, float(machine["worker_gb"]) + slots_gb,
                          machine.get("main_gb", MAIN_GB))
        workers = max(1, min(cpu_workers(), fit))
    if workers is None:
        workers = auto_workers(ram, loader=loader)
    workers = int(workers)
    rate = train_all.data_rate(machine, workers, data)
    chunk = train_all.head_chunk_for(machine, view, gpu, train_all.BATCH, rate)
    # the crawl guard's pace, as train_all.run_stage passes it (setup's step at the cap)
    expected = train_all.cap_step_seconds(machine, view, train_all.BATCH)[
        "chunks" if chunk else "all"]
    return {"view": view, "speedups": list(train_all.speedups_for(machine, view)),
            "chunk": chunk, "expected_step_seconds": expected,
            "workers": workers, "data": data, "data_flags": train_all.data_flags(data),
            "data_words": train_all.data_words(data), "data_rate": rate, "ram_gb": ram,
            "gpu_memory_gb": gpu}


def steady_pace(step_times: list, skip: int, batch: int = BATCH) -> dict:
    """Each epoch's pace from the arrival of its step lines (``(epoch, step, seconds)``) without
    its first ``skip`` steps: spectra per second from its first step line at or after ``skip``
    to its last. An epoch with fewer than two such lines has none."""
    by_epoch: dict = {}
    for epoch, step, t in step_times:
        if step >= skip:
            by_epoch.setdefault(epoch, []).append((step, t))
    out: dict = {}
    for epoch, rows in sorted(by_epoch.items()):
        (s0, t0), (s1, t1) = rows[0], rows[-1]
        if s1 > s0 and t1 > t0:
            out[epoch] = {"from_step": s0, "to_step": s1, "seconds": round(t1 - t0, 2),
                          "spectra_per_second": round(batch * (s1 - s0) / (t1 - t0), 1)}
    return out


def train_check(ctx, view: str = "window", workers: int | None = None, epochs: int = 2,
                train_len: int = 1600, val_len: int = 8, minutes: float = 12.0,
                ram_gb: float | None = None, size: str = "full",
                label: str = "train_all's choice") -> dict:
    """The short training run of ``view`` as train_all trains it on this computer
    (train_all_choice), with the check's network size, epochs and spectra; ``workers`` and
    ``ram_gb`` in place of train_all's (smoke runs and tests)."""
    require_cuda()
    machine = read_json(ROOT / "gpu_training" / "machine.json") or {}
    choice = train_all_choice(machine, view)
    workers = int(workers or choice["workers"])
    ram = float(ram_gb or choice["ram_gb"])
    speedups, chunk, flags = choice["speedups"], choice["chunk"], choice["data_flags"]
    out_dir = ctx.run_dir / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder_e2e.pt"
    args = run_args(view, size, epochs, train_len, val_len)
    # train_all.run_stage's command, its options in its order, and the check's own at the end
    expected = choice.get("expected_step_seconds")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_charge_net.py"), "--device", "cuda",
           *args, *(["--head-chunk", str(chunk)] if chunk else []),
           *(["--expected-step-seconds", f"{expected:g}"] if expected else []), "--gpu-speedups",
           ",".join(speedups) or "none", "--workers", str(workers), *flags,
           "--max-ram-gb", f"{ram:g}", "--max-gpu-memory-gb", f"{choice['gpu_memory_gb']:g}",
           "--batch-size", str(BATCH), "--out", str(out), "--keep-pause-files", "--restart",
           "--save-every", "1000", "--minutes", f"{minutes:g}"]
    print("> " + " ".join(cmd[1:]), flush=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")  # as train_all.run_stage's
    for var in [*os.environ.get(THREADS_ADDED_ENV, "").split(","), THREADS_ADDED_ENV]:
        env.pop(var, None)  # the runner's one thread per experiment: training sets its own
    lines, step_times = [], []
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    for line in proc.stdout:
        now = time.perf_counter() - started
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
        m = STEP.match(line)
        if m:
            step_times.append((int(m.group(1)), int(m.group(2)), round(now, 3)))
        if "epoch took" in line:
            ctx.say(line.strip()[:200])
    rc = proc.wait()
    records = []
    log = out.with_suffix(".log.jsonl")
    if log.exists():
        for text in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(text)
            except ValueError:
                continue
            if "pace" in rec:
                records.append({"epoch": rec.get("epoch"), "pace": rec["pace"],
                                "seconds_since_start": rec.get("seconds"),
                                "train_loss": rec.get("train_loss")})
    pace_lines = [ln.strip() for ln in lines if "epoch took" in ln]
    for rec, ln in zip(records, pace_lines):
        m = SPEED.search(ln)
        if m:  # the line's speed is the session's mean so far, over every epoch
            rec["epoch_minutes"] = float(m.group(1))
            rec["spectra_per_second_session"] = int(m.group(2))
        rec["line"] = ln
    skip = PRIMED * int(workers)
    steady = steady_pace(step_times, skip)
    for rec in records:
        rate = epoch_rate(rec["pace"])
        if rate is not None:
            rec["spectra_per_second_epoch"] = rate
        rec["steady"] = steady.get(rec["epoch"])
    for f in out_dir.iterdir():  # the throwaway model and state: never uploaded
        if f.suffix in (".pt", ".onnx") or f.name.endswith(".tmp"):
            f.unlink(missing_ok=True)
    errors = [ln for ln in lines if "Error" in ln or "error" in ln][-5:]
    return {"config": {"view": view, "speedups": list(speedups), "head_chunk": chunk,
                       "workers": workers, "data_loading": choice["data"],
                       "data_words": choice["data_words"], "data_flags": flags,
                       "gpu_memory_gb": choice["gpu_memory_gb"],
                       "train_all": {k: choice[k] for k in ("workers", "ram_gb", "data_rate")},
                       "epochs": epochs, "train_len": train_len,
                       "steps": epochs * train_len // BATCH, "val_len": val_len, "size": size,
                       "recipe": args[args.index("--recipe") + 1] if "--recipe" in args else None,
                       "ram_gb": ram, "label": label, "steady_skips_steps": skip,
                       "command": " ".join(cmd[1:]).replace(str(ROOT), "<package>")},
            "exit_code": rc, "epochs": records, "pace_lines": pace_lines,
            "step_lines": [{"epoch": e, "step": s, "seconds": t} for e, s, t in step_times],
            "gpu_speedups_line": next((ln.strip() for ln in lines
                                       if ln.startswith("GPU speedups")), None),
            "error_lines": errors if rc not in (0, 3) else []}
