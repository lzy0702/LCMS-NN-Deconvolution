"""Set up the training environment where needed, then train: everything from one double-click.

    gpu_training\\setup_and_train.bat        (Windows: finds Python, then runs this)
    bash gpu_training/setup_and_train.sh    (Linux)

1. gpu_training/setup.py --if-needed: the first time, the whole setup (10-20 minutes: the
   environment .venv with PyTorch for CUDA, a measurement of this computer, a few minutes'
   end-to-end check). Later it is skipped while that setup still holds
   (gpu_training/setup_done.json); after a new package it measures and checks again, keeping
   the environment.
2. gpu_training/train_all.py in .venv: trains the runs (the window and the comb view) on the
   GPU, exports them and packs runs/upload_*.zip. A finished run is skipped and an interrupted
   one continues, so after a pause, a crash or a reboot the same double-click carries on where
   the training stopped.

Options, all optional (a double-click passes none):
  --ram-gb N                       RAM budget for setup and training (default: half the
                                   computer's memory, at least 16 GB and at most 48, leaving
                                   4 GB to the system)
  --gpu-memory-gb N                GPU memory cap for setup and training (default 14)
  --workers N                      data workers for the training (default: what setup chose, as
                                   many as the processor is given and the RAM budget holds)
  --batch-size N                   spectra per training step (default 4, the recipe's)
  --runs A,B                       the runs to train (default window,comb; see train_all.py)
  --restart                        train the runs from the beginning (needed after changing
                                   the batch size of a run already started)
  --redo-setup                     run the whole setup again even if it still holds
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = sys.platform == "win32"
VPY = ROOT / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")
#: train_all.py's exit codes besides 0 (finished)
PAUSED, MEMORY, ABORTED = 3, 4, 130
SCRIPT = "gpu_training\\setup_and_train.bat" if WINDOWS else "bash gpu_training/setup_and_train.sh"
AGAIN = f"double-click {SCRIPT} again" if WINDOWS else f"run {SCRIPT} again"


def rel(*parts: str) -> str:
    """A path inside the package folder, written the way this system writes paths."""
    return str(Path(*parts))


def recorded_budget() -> float | None:
    """The RAM budget setup recorded in gpu_training/machine.json, or None."""
    try:
        budget = json.loads((ROOT / "gpu_training" / "machine.json").read_text(
            encoding="utf-8")).get("ram_budget_gb")
        return float(budget) if budget else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def run(cmd: list) -> int:
    """Run ``cmd`` in the package folder in this window; returns its exit code."""
    print("> " + " ".join(str(c) for c in cmd), flush=True)
    # Ctrl+C reaches every program in the window. The training pauses cleanly on it, so this
    # script ignores it and waits for the training to finish saving its state.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return subprocess.call([str(c) for c in cmd], cwd=ROOT)
    finally:
        signal.signal(signal.SIGINT, previous)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ram-gb", type=float, default=None,
                    help="RAM the training may use, training process and data workers together")
    ap.add_argument("--gpu-memory-gb", type=float, default=None,
                    help="GPU memory the training may use")
    ap.add_argument("--workers", type=int, default=None, help="data workers for the training")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="spectra per training step (default 4, the recipe's)")
    ap.add_argument("--runs", default=None,
                    help="the runs to train, comma-separated (default window,comb)")
    ap.add_argument("--restart", action="store_true",
                    help="discard the saved training states and train the runs afresh")
    ap.add_argument("--redo-setup", action="store_true",
                    help="run the whole setup again even if it still holds")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("LadderNet training on this computer's GPU")
    print("  1. setup, the first time only (10-20 minutes; about 3 GB to download)")
    print("  2. training of the two runs, window and comb view (several hours; setup prints the "
          "estimate)")
    print("Pause at any time: press Ctrl+C once, or put a file named PAUSE in the runs folder.")
    print(f"To continue later, {AGAIN}.")
    print("=" * 78, flush=True)
    if "onedrive" in str(ROOT).lower():
        print(f"Note: this folder ({ROOT}) looks like one that OneDrive synchronises, which slows "
              "saving the training state and can lock files. A folder such as C:\\lcms is "
              "better.", flush=True)

    limits = []
    if args.ram_gb:
        limits += ["--ram-gb", f"{args.ram_gb:g}"]
    if args.gpu_memory_gb:
        limits += ["--gpu-memory-gb", f"{args.gpu_memory_gb:g}"]

    setup = [sys.executable, "gpu_training/setup.py", *limits]
    if not args.redo_setup:
        setup.append("--if-needed")
    rc = run(setup)
    if rc != 0:
        log = rel("gpu_training", "setup.log")
        print(f"\nSetup stopped (exit code {rc}); the training was not started. The reason is "
              f"in the messages above and in {log}. Fix it, then {AGAIN}.", flush=True)
        return rc
    if not VPY.exists():  # setup returned 0 but made no environment: never expected
        print(f"\nThe training environment {VPY} is missing after setup; run the setup again "
              "with --redo-setup.", flush=True)
        return 1

    print("\nStarting the training.", flush=True)
    train = [VPY, "gpu_training/train_all.py", *limits]
    if args.workers:
        train += ["--workers", str(args.workers)]
    if args.batch_size:
        train += ["--batch-size", str(args.batch_size)]
    if args.runs:
        train += ["--runs", args.runs]
    if args.restart:
        train.append("--restart")
    rc = run(train)
    if rc == 0:
        print(f"\nFinished. Upload the zip file named above ({rel('runs', 'upload_*.zip')}).",
              flush=True)
    elif rc == PAUSED:
        print(f"\nPaused; the training state is saved. To continue, {AGAIN}.", flush=True)
    elif rc == ABORTED:
        print(f"\nStopped; the last saved state is kept. To continue from it, {AGAIN}.",
              flush=True)
    elif rc == MEMORY:
        budget = args.ram_gb or recorded_budget()
        now = f" (now {budget:g} GB)" if budget else ""
        example = f"{budget + 8:g}" if budget else "24"
        print(f"\nStopped at the RAM budget{now}; see the messages above and "
              f"{rel('gpu_training', 'machine.json')}. Close other programs or raise the "
              f"budget, for example {SCRIPT} --ram-gb {example}, then start again.", flush=True)
    else:
        print(f"\nThe training stopped (exit code {rc}); the messages above say why, and the "
              f"{rel('runs', '*.console.log')} files hold each run's full output. Saved "
              f"training states are kept: once the cause is fixed, {AGAIN}.", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
