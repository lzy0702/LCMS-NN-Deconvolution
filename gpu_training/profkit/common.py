"""What the profiling experiments share: where things are, the recipe the runs train, the
context an experiment runs in, statistics, JSON-ready records, and stopping a process with
everything it started."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

#: the package folder (the one holding gpu_training, scripts and src)
ROOT = Path(__file__).resolve().parents[2]
GPU_TRAINING = ROOT / "gpu_training"
#: set to 1 for the smoke size of every experiment (as --smoke does)
SMOKE_ENV = "LCMSDECONV_PROFILE_SMOKE"
#: the numerical libraries' thread counts, one per process as in training
#: (lcmsdeconv.nn.machine.single_threaded_workers)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
#: what probe.recipe_mixture sets on every configuration of the recipe, as training does
TRAINING_SETTINGS = ("singly_charged", "iso_channels", "measured_channel")


class Unavailable(Exception):
    """An experiment that cannot run on this computer (no CUDA, an API Windows does not have
    here, ...): recorded as unavailable, with the reason, not as an error."""


def add_paths() -> None:
    """Make the package's code importable: lcmsdeconv (src), train_charge_net (scripts), and
    probe, train_all and profkit (gpu_training)."""
    for p in (ROOT / "src", ROOT / "scripts", GPU_TRAINING):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def smoke() -> bool:
    return os.environ.get(SMOKE_ENV, "") not in ("", "0")


def recipe() -> str:
    """The recipe train_all's runs train (train_all.RECIPE, their train_charge_net.py
    --recipe)."""
    from train_all import RECIPE

    return str(RECIPE)


def recipe_mixture(measured: bool = True) -> list:
    """The runs' training spectra as training configures them: probe.recipe_mixture (singly
    charged ions labelled, the isotope-charge channels, the comb view's measured channel where
    ``measured``), the spectra setup measures, as long as it makes train_all's recipe. Should
    the two part ways (one changed without the other), train_all's recipe
    (train_charge_net.mixture_<recipe>) with probe's settings: the profile measures the spectra
    the runs train on."""
    import probe
    import train_charge_net

    from lcmsdeconv.synth.config import settings_repr

    mix = probe.recipe_mixture(measured)
    make = getattr(train_charge_net, f"mixture_{recipe()}", None)
    if make is None:  # a recipe without a mixture of its own: probe's is the only one
        return mix
    own = make()
    for _w, c in own:
        for key in TRAINING_SETTINGS:
            setattr(c, key, getattr(mix[0][1], key))
    return mix if settings_repr(mix) == settings_repr(own) else own


def percentile(values, q: float) -> float | None:
    """The ``q`` quantile (0..1) of ``values``, interpolated linearly between the two nearest
    ranks (numpy's default); None for no values."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def stats(values, digits: int = 3) -> dict:
    """Median, 10th and 90th percentiles, mean, extremes and count of ``values``."""
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0}

    def r(v):
        return round(v, digits)

    return {"n": len(xs), "median": r(percentile(xs, 0.5)), "p10": r(percentile(xs, 0.1)),
            "p90": r(percentile(xs, 0.9)), "mean": r(sum(xs) / len(xs)), "min": r(min(xs)),
            "max": r(max(xs))}


def to_jsonable(obj):
    """``obj`` with numpy numbers and arrays, paths, tuples and sets made JSON types, and
    non-finite floats made None (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy arrays and numbers, tensors
        return to_jsonable(obj.tolist())
    if hasattr(obj, "item"):
        return to_jsonable(obj.item())
    return str(obj)


def write_json(path: Path, obj, attempts: int = 20) -> None:
    """Write ``obj`` as JSON (UTF-8) through a temporary file, so that a reader never sees half
    a file; ``attempts`` tries a quarter of a second apart where the file is held (on Windows a
    virus scanner or an indexer may hold it for a moment), then PermissionError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt >= attempts - 1:
                raise
            time.sleep(0.25)


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Context:
    """What an experiment gets from the runner: the run's folder (``raw`` holds each finished
    experiment's record, ``work`` large intermediate files that are never uploaded, ``csv`` and
    ``traces`` files that are), its own plan entry, and whether this is a smoke run."""

    def __init__(self, run_dir: Path, entry: dict | None = None, is_smoke: bool | None = None):
        self.run_dir = Path(run_dir)
        self.entry = entry or {}
        self.id = self.entry.get("id", "experiment")
        self.smoke = smoke() if is_smoke is None else bool(is_smoke)
        self.work = self.run_dir / "work"
        self.raw = self.run_dir / "raw"

    def path(self, *parts: str) -> Path:
        """A file in the run folder (its folder is made)."""
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def say(self, text: str) -> None:
        """A progress line the runner shows on the console as well (the rest goes to the log)."""
        print(f"[say] {text}", flush=True)

    def earlier(self, exp_id: str) -> dict | None:
        """The record of an experiment that ran before this one, or None."""
        return read_json(self.raw / f"{exp_id}.json")

    def environment(self) -> dict:
        """The environment record (the experiment ``env``), or {} if it is missing."""
        rec = self.earlier("env") or {}
        return rec.get("result") or {}


def kill_tree(pid: int, timeout: float = 10.0) -> list[int]:
    """Stop process ``pid`` and every process it started (data workers, nvidia-smi, a training
    run), children first; returns the ids stopped. psutil where it is installed (training
    installs it), else taskkill on Windows or the process alone elsewhere."""
    stopped: list[int] = []
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            parent = psutil.Process(pid)
            procs = parent.children(recursive=True) + [parent]
        except psutil.NoSuchProcess:
            return stopped
        for p in procs:
            try:
                p.kill()
                stopped.append(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        psutil.wait_procs(procs, timeout=timeout)
        return stopped
    if sys.platform == "win32":
        import subprocess

        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                           timeout=timeout)
        except (OSError, subprocess.SubprocessError):  # the caller checks whether it ended
            return stopped
        return [pid]
    try:
        os.kill(pid, 9)
        stopped.append(pid)
    except OSError:
        pass
    return stopped


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
