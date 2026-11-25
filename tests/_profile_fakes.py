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
