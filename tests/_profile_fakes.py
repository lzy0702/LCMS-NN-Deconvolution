"""Stand-in experiments for the tests of gpu_training/profile_machine.py: each runs in a process
the runner starts, as a real experiment does."""

from __future__ import annotations

import os
import subprocess
import sys
import time


def ok(ctx, value: int = 1) -> dict:
    ctx.say(f"working on {value}")
    return {"pid": os.getpid(), "value": value, "config": {"value": value}}


def fail(ctx) -> dict:
    raise ValueError("this experiment fails on purpose")


def crash(ctx) -> dict:
    print("about to crash", flush=True)
    os._exit(3)


def unavailable(ctx) -> dict:
    from profkit.common import Unavailable

    raise Unavailable("no CUDA: a stand-in for a computer without a GPU")


def hang(ctx, seconds: float = 120.0) -> dict:
    """Starts a process of its own (as data workers are), then never finishes in time."""
    child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])
    ctx.path("work", "grandchild.pid").write_text(str(child.pid))
    time.sleep(seconds)
    return {}


def write_csv(ctx) -> dict:
    ctx.path("csv", "samples.csv").write_text("t,value\n0,1\n")
    ctx.path("work", "big.pt").write_bytes(b"\0" * 1024)
    return {"rows": 1}


def stepper(ctx, steps: int = 8, step_s: float = 0.02, slow_at: int | None = None,
            slow_s: float = 0.0, phases=("warm",), per_phase: int = 2) -> dict:
    """Steps under the experiment's watchdog (the limits its plan entry's "stall" holds), as a
    GPU experiment takes them: step ``slow_at`` (counted from 1) takes ``slow_s``, the others
    ``step_s``; ``per_phase`` steps in each of ``phases`` but the last, which has the rest."""
    wd = ctx.watchdog(snapshot=lambda: {"reserved_gb": 1.5})
    for i in range(1, steps + 1):
        k = min((i - 1) // per_phase, len(phases) - 1)
        if i == 1 or ((i - 1) % per_phase == 0 and (i - 1) // per_phase < len(phases)):
            wd.set_phase(phases[k])
        wd.begin()
        time.sleep(slow_s if i == slow_at else step_s)
        wd.end()
    wd.end_phase()
    return {"steps": steps, "memory": wd.memory}
