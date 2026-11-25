"""The end-to-end check: a short real training run (scripts/train_charge_net.py with
train_all's arguments for its window run: the recipe, the full LadderNet of the window view with
isotope-charge channels, the learning rate and the seed), with the GPU speedups the GPU
experiments found best and the number of data workers the CPU experiments found best, written
to a throwaway output in the run's folder.

Two epochs of 1600 spectra, 400 steps of 4 each. At the start of every epoch training's
persistent DataLoader workers all start a batch at once (the loader primes prefetch_factor
batches per worker), so the GPU first waits about one batch time: about 5 % of an epoch of 400
steps, where it was a fifth of one of 80. The first epoch also holds the workers' start; the
second is the one to read. Records each epoch's pace from the run's log (its share of time
waiting for data, the GPU's share busy and its milliseconds a step, the allocator's retries,
spectra per second), and its steady pace without its first PRIMED x workers steps, from the
times the run's step lines reached the console."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from .common import ROOT, read_json
from .gpu import BATCH, GPU_MEMORY_GB, require_cuda

#: a pace line's speed: "epoch took 1.2 min; 41 spectra/s; ..."
SPEED = re.compile(r"epoch took ([\d.]+) min; (\d+) spectra/s")
#: a step line, which training prints every log_every (20) steps: "  epoch 1 step 40/400 loss ..."
STEP = re.compile(r"^\s*epoch (\d+) step (\d+)/(\d+) loss")
#: the batches training's DataLoader primes per worker at the start of every epoch
#: (prefetch_factor, lcmsdeconv.synth.dataset.make_dataloaders)
PRIMED = 2


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


def train_check(ctx, view: str = "window", speedups=(), chunk: int | None = None,
                workers: int = 20, epochs: int = 2, train_len: int = 1600, val_len: int = 8,
                minutes: float = 12.0, ram_gb: float | None = None, size: str = "full",
                label: str = "best") -> dict:
    require_cuda()
    from lcmsdeconv.nn.machine import default_ram_budget_gb

    machine = read_json(ROOT / "gpu_training" / "machine.json") or {}
    ram = float(ram_gb or machine.get("ram_budget_gb") or default_ram_budget_gb())
    out_dir = ctx.run_dir / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder_e2e.pt"
    args = run_args(view, size, epochs, train_len, val_len)
    cmd = [sys.executable, str(ROOT / "scripts" / "train_charge_net.py"), "--device", "cuda",
           *args, "--batch-size", str(BATCH), "--workers", str(workers), "--gpu-speedups",
           ",".join(speedups) or "none", *(["--head-chunk", str(chunk)] if chunk else []),
           "--max-gpu-memory-gb", f"{GPU_MEMORY_GB:g}", "--max-ram-gb", f"{ram:g}",
           "--out", str(out), "--restart", "--keep-pause-files", "--save-every", "1000",
           "--minutes", f"{minutes:g}"]
    print("> " + " ".join(cmd[1:]), flush=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
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
        # this epoch alone: its steps took gpu_step_ms * steps / gpu_busy_share in all
        p = rec["pace"]
        if p.get("gpu_step_ms") and p.get("gpu_busy_share"):
            rec["spectra_per_second_epoch"] = round(
                BATCH * 1000.0 * float(p["gpu_busy_share"]) / float(p["gpu_step_ms"]), 1)
        rec["steady"] = steady.get(rec["epoch"])
    for f in out_dir.iterdir():  # the throwaway model and state: never uploaded
        if f.suffix in (".pt", ".onnx") or f.name.endswith(".tmp"):
            f.unlink(missing_ok=True)
    errors = [ln for ln in lines if "Error" in ln or "error" in ln][-5:]
    return {"config": {"view": view, "speedups": list(speedups), "head_chunk": chunk,
                       "workers": workers, "epochs": epochs, "train_len": train_len,
                       "steps": epochs * train_len // BATCH, "val_len": val_len, "size": size,
                       "recipe": args[args.index("--recipe") + 1] if "--recipe" in args else None,
                       "ram_gb": ram, "label": label, "steady_skips_steps": skip,
                       "command": " ".join(cmd[1:]).replace(str(ROOT), "<package>")},
            "exit_code": rc, "epochs": records, "pace_lines": pace_lines,
            "step_lines": [{"epoch": e, "step": s, "seconds": t} for e, s, t in step_times],
            "gpu_speedups_line": next((ln.strip() for ln in lines
                                       if ln.startswith("GPU speedups")), None),
            "error_lines": errors if rc not in (0, 3) else []}
