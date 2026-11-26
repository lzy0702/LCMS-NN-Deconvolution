"""The benchmark driver: every suite's jobs within this computer's budgets, then the summary and
the zip to upload.

    bench\\run_bench.bat [options]                        (Windows: .venv-bench's Python)
    .venv-bench/bin/python bench/run_bench.py [options]   (Linux)

The suites are the modules of bench/suites (thermo, bars, rda). Each lists its jobs (a command
with its output file, log, threads, peak memory and the jobs it comes after) and writes its part
of the summary. The driver starts the jobs whose predecessors have finished (successfully or
not: the summaries mark what is missing), keeping the threads of the running jobs within
--threads and their memory within --mem-gb, records the start, end, exit code, seconds and peak
memory of each in results/<id>/timing.json, and goes on when a job fails. Run it again after an
interruption or a failure: a job whose output exists is skipped unless a job it comes after ran
again or its inputs changed (a job may name them, such as the networks and Thermo runs it
reads: the output is redone when it was made with others), the rest run. Ctrl-C stops the
running jobs, and so does an error of the driver itself. On Windows the computer is kept from
going to sleep while the jobs run, and the jobs are opted out of power throttling; a pause of
the driver (the computer slept after all) is recorded with the times of the jobs it spans.
Relative paths in the options are taken from the folder the command runs in. A GPU provider
that setup_bench found not working is refused (onnxruntime would quietly use the processor).

At the end it writes results/<id>/summary.md (every suite's summary) and results/<id>_upload.zip
(the results without .raw and .mzMLb files, with bench/machine.json and the setup log): the file
to upload.

Settings, each overriding the one before: the defaults, bench/config.json, bench/config.local.json
(your own, never shipped) and the options. The settings used are written to
results/<id>/config.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
import _paths  # noqa: E402, F401  (<repo>/src and <repo>/scripts on sys.path)
from tools import common  # noqa: E402

WINDOWS = sys.platform == "win32"
SUITES = BENCH / "suites"
RESULTS = BENCH / "results"
MACHINE = BENCH / "machine.json"
PACKAGE = BENCH / "package.json"
CONFIG_FILES = (BENCH / "config.json", BENCH / "config.local.json")
#: the suites in the order they run when none are named (others found follow by name)
SUITE_ORDER = ("thermo", "bars", "rda")
#: where the networks are found by default: the package's own, and the training package's runs
#: (gpu_training writes them to <repo>/runs; gpu_training/runs is looked at too)
MODEL_DIRS = (BENCH / "models", REPO / "runs", REPO / "gpu_training" / "runs")
#: keys a job of a suite must have (optional: "cwd", and "inputs": any JSON naming what the
#: output is made from, such as the networks' labels and sha256 and the Thermo runs; an output
#: made with other inputs is redone)
JOB_KEYS = ("name", "cmd", "out", "log", "threads", "mem_gb", "after")
#: the keys of bench/config.json and bench/config.local.json (null: the default)
SETTINGS = ("suites", "smoke", "providers", "threads_total", "mem_budget_gb", "threads_per_job",
            "thermo_dir", "rda_dir", "models", "default_models", "results_id")
#: files left out of the upload zip
NO_UPLOAD = (".raw", ".mzmlb", ".tmp")
POLL_S, SAMPLE_S, HEARTBEAT_S = 0.5, 2.0, 600.0
#: a pause of the driver's loop this long means the computer slept (the loop wakes every POLL_S)
STALL_S = 60.0


def say(msg: str = "") -> None:
    """Print ``msg`` as ASCII (the console may not take more)."""
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def stamp(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def rel(path: str | Path) -> str:
    """``path`` relative to the package's top folder where it is inside it."""
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ------------------------------------------------------------------ the computer
def logical_cores() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 4


def ram_gb() -> float:
    """Physical memory, or the memory limit of this process's control group if lower (Linux)."""
    total = None
    try:
        import psutil

        total = psutil.virtual_memory().total
    except ImportError:
        if hasattr(os, "sysconf"):
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = cgroup_limit()
    if limit and (total is None or limit < total):
        total = limit
    return (total or 16 * 2**30) / 2**30


def cgroup_limit() -> int | None:
    """The memory limit of this process's cgroup (v1 or v2) in bytes, if any."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        _id, ctrl, path = line.split(":", 2)
        if "memory" in ctrl.split(","):
            cand = Path("/sys/fs/cgroup/memory") / path.lstrip("/") / "memory.limit_in_bytes"
        elif ctrl == "":
            cand = Path("/sys/fs/cgroup") / path.lstrip("/") / "memory.max"
        else:
            continue
        try:
            value = cand.read_text().strip()
        except OSError:
            continue
        if value.isdigit() and int(value) < 2**60:
            return int(value)
    return None


def job_processes(pid: int) -> list:
    """Process ``pid`` and the processes it started (psutil objects; [] without psutil or once
    it has ended). On Windows a job's interpreter is a child of ``pid``: a virtual
    environment's python.exe is a launcher that runs the base Python as a child process."""
    try:
        import psutil

        proc = psutil.Process(pid)
        return [proc, *proc.children(recursive=True)]
    except Exception:
        return []


def peak_bytes(pid: int, procs: list | None = None) -> int:
    """The peak memory so far of process ``pid`` and of each process it started (``procs``:
    :func:`job_processes`), summed (Windows: their peak working sets, which hold the peak
    between two readings too; elsewhere the current memory of each, and VmHWM of ``pid`` on
    Linux); 0 where it cannot be read."""
    own = kids = 0
    for i, proc in enumerate(job_processes(pid) if procs is None else procs):
        try:
            info = proc.memory_info()
        except Exception:  # it just ended
            continue
        size = getattr(info, "peak_wset", 0) or info.rss
        if i == 0:
            own = size
        else:
            kids += size
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                own = max(own, int(line.split()[1]) * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return own + kids


def pid_alive(pid: int) -> bool:
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        return "python" in psutil.Process(pid).name().lower()
    except ImportError:
        if WINDOWS:
            return True  # cannot tell: the lock is kept
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    except Exception:
        return False


class Power:
    """Windows only (elsewhere it does nothing): keeps the computer from going to sleep while
    the jobs run (the screen may still turn off; Windows does not count processor or GPU load
    as activity), and opts each job's processes out of power throttling (EcoQoS: efficient cores
    at a low clock, which Windows may apply to the processes of a minimized or hidden window)."""

    def __init__(self) -> None:
        self.k = None
        self.awake = False
        self.done: set[tuple] = set()  # (process number, creation time)
        if not WINDOWS:
            return
        try:
            import ctypes
            from ctypes import wintypes

            # a private, typed handle: the shared ctypes.windll one has untyped prototypes
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            for name, args, res in (
                    ("SetThreadExecutionState", [ctypes.c_uint32], ctypes.c_uint32),
                    ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
                     wintypes.HANDLE),
                    ("SetProcessInformation", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               wintypes.DWORD], wintypes.BOOL),
                    ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL)):
                fn = getattr(k, name)
                fn.argtypes, fn.restype = args, res
            self.k, self.ctypes = k, ctypes
        except Exception:  # only a convenience
            self.k = None

    def keep_awake(self, on: bool) -> None:
        if self.k is None or on == self.awake:
            return
        try:
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; ES_CONTINUOUS alone allows sleep again
            ok = self.k.SetThreadExecutionState(0x80000001 if on else 0x80000000) != 0
            self.awake = on and ok
        except Exception:
            self.awake = False

    def full_speed(self, procs: list) -> None:
        """Opt these processes (psutil objects) out of power throttling, each once."""
        if self.k is None:
            return
        ct = self.ctypes
        for proc in procs:
            try:
                key = (proc.pid, proc.create_time())  # Windows reuses process numbers
            except Exception:  # it just ended
                continue
            if key in self.done:
                continue
            self.done.add(key)
            try:
                handle = self.k.OpenProcess(0x0200, False, proc.pid)  # PROCESS_SET_INFORMATION
                if not handle:
                    continue
                # PROCESS_POWER_THROTTLING_STATE: version 1, execution speed controlled, not
                # throttled; information class 4 is ProcessPowerThrottling
                state = (ct.c_uint32 * 3)(1, 1, 0)
                self.k.SetProcessInformation(handle, 4, ct.byref(state), ct.sizeof(state))
                self.k.CloseHandle(handle)
            except Exception:
                pass


# ------------------------------------------------------------------ settings
def label_of(stem: str) -> str:
    """A network's label from its file name: ``ladder_v5_window`` -> ``v5_window``."""
    label = stem[len("ladder_"):] if stem.startswith("ladder_") and len(stem) > 7 else stem
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", label)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_models(explicit: list[tuple[str, Path]], defaults: bool) -> tuple[list[dict], list[str]]:
    """The networks to evaluate: ``explicit`` ((label, path) pairs, --model and the settings
    files), then, with ``defaults``, every network in MODEL_DIRS (not the training package's
    smoke networks). Each needs its sidecar .json; one marked .unchecked (its export check did
    not pass) and a second copy of the same file are left out (the copy's training console log,
    ``<name>.console.log``, is kept when the first has none). Returns (models, notes)."""
    models: list[dict] = []
    notes: list[str] = []
    seen: dict[str, dict] = {}
    cands = list(explicit)
    if defaults:
        for folder in MODEL_DIRS:
            for path in sorted(folder.glob("*.onnx")) if folder.is_dir() else []:
                if path.stem.startswith("smoke_"):
                    continue
                cands.append((label_of(path.stem), path))
    for label, path in cands:
        path = Path(path).resolve()
        if not path.is_file():
            notes.append(f"network {label}: {path} not found; left out")
            continue
        if not path.with_suffix(".json").is_file():
            notes.append(f"network {label}: {rel(path)} has no {path.stem}.json next to it "
                         "(copy it too); left out")
            continue
        if path.with_suffix(".unchecked").exists():
            notes.append(f"network {label}: {rel(path)} is marked .unchecked (its export or "
                         "check did not finish); left out")
            continue
        digest = sha256(path)
        log = path.with_suffix(".console.log")
        log = str(log) if log.is_file() else None
        if digest in seen:
            first = seen[digest]
            notes.append(f"network {label}: {rel(path)} is the same file as {first['label']}; "
                         "left out")
            first["console_log"] = first["console_log"] or log
            continue
        seen[digest] = {"label": label, "onnx": str(path), "console_log": log,
                        "sha256": digest[:16]}
        models.append(seen[digest])
    return models, notes


def parse_model_specs(specs: list[str], base: Path) -> list[tuple[str, Path]]:
    """``LABEL=PATH`` strings as (label, path) pairs, a relative path taken from ``base``."""
    try:
        parsed = common.parse_models(specs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    out = []
    for label, path in parsed.items():
        p = Path(path).expanduser()
        out.append((re.sub(r"[^A-Za-z0-9_.+-]+", "_", label), p if p.is_absolute() else base / p))
    return out


def build_cfg(args: argparse.Namespace) -> tuple[dict, list[str]]:
    """The settings of this run (the cfg the suites read, with a few records) and notes."""
    notes: list[str] = []
    files: dict = {}
    for path in CONFIG_FILES:
        data = read_json(path)
        if not isinstance(data, (dict, type(None))) or (data is None and path.exists()):
            raise SystemExit(f"{rel(path)} is not a JSON object")
        for key, value in (data or {}).items():
            if key not in SETTINGS and not key.startswith("_"):
                notes.append(f"{rel(path)}: unknown setting {key!r} ignored (known: "
                             f"{', '.join(SETTINGS)})")
            elif value is not None:
                files[key] = value
    machine = read_json(MACHINE, {}) or {}
    package = read_json(PACKAGE, {}) or {}

    def pick(cli, key, default):
        return cli if cli is not None else files.get(key, default)

    suites = ([s.strip() for s in args.suites.split(",") if s.strip()] if args.suites
              else files.get("suites"))
    providers = pick(args.providers, "providers", "auto")
    requested = providers
    if providers == "auto" and machine.get("providers_option"):
        providers = machine["providers_option"]
        notes.append(f"providers: {providers} (chosen by setup_bench, bench/machine.json)")
    explicit = parse_model_specs(args.model or [], Path.cwd())
    explicit = parse_model_specs(files.get("models") or [], REPO) + explicit
    models, model_notes = find_models(explicit, defaults=files.get("default_models", True)
                                      and not args.no_default_models)
    notes += model_notes
    smoke = bool(args.smoke or files.get("smoke", False))
    rid = args.results_id or files.get("results_id")
    if not rid:
        rid = str(package.get("results_id") or "dev")
        if requested not in ("auto", machine.get("providers_option")):
            rid += f"_{providers}"
        if smoke:
            rid += "_smoke"
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", rid):
        raise SystemExit(f"--results-id {rid!r}: use letters, digits and _ . + - only")

    def folder(cli, key, default: Path) -> str:
        value = cli if cli is not None else files.get(key)
        if value is None:
            return str(default)
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else (Path.cwd() if cli is not None else REPO) / p)
                   .resolve())

    cfg = {
        "repo": str(REPO), "results_id": rid, "results_dir": str(RESULTS / rid),
        "thermo_dir": folder(args.thermo_dir, "thermo_dir", REPO / "data" / "thermo_raw"),
        "rda_dir": folder(args.rda_dir, "rda_dir",
                          BENCH / "private" / "confidential_sample_1" / "data"),
        "models": models, "providers": providers, "providers_requested": requested,
        "threads_total": int(pick(args.threads, "threads_total", max(1, logical_cores() - 2))),
        "mem_budget_gb": round(float(pick(args.mem_gb, "mem_budget_gb", 0.7 * ram_gb())), 1),
        "threads_per_job": int(pick(args.threads_per_job, "threads_per_job", 4)),
        "suites": suites or [], "smoke": smoke, "package": package.get("name"),
    }
    if cfg["threads_total"] < 1 or cfg["threads_per_job"] < 1 or cfg["mem_budget_gb"] <= 0:
        raise SystemExit("--threads, --threads-per-job and --mem-gb must be positive")
    if cfg["threads_per_job"] > cfg["threads_total"]:
        notes.append(f"{cfg['threads_per_job']} threads per job are more than "
                     f"{cfg['threads_total']} in all: each job gets {cfg['threads_total']}")
        cfg["threads_per_job"] = cfg["threads_total"]
    if cfg["providers"] not in ("auto", "dml", "cuda", "cpu"):
        raise SystemExit(f"--providers {cfg['providers']!r}: auto, dml, cuda or cpu")
    refused = provider_refused(cfg["providers"], machine)
    if refused:
        raise SystemExit(refused)
    return cfg, notes


#: the GPU providers by run_bench's names
GPU_EPS = {"dml": "DmlExecutionProvider", "cuda": "CUDAExecutionProvider"}


def provider_refused(option: str, machine: dict) -> str | None:
    """Why ``--providers option`` cannot be used here, or None: a GPU provider that setup_bench
    found not working (bench/machine.json) or, where setup has not run, that onnxruntime does
    not list. onnxruntime would quietly run every network on the processor instead, and the
    results would carry the GPU provider's name."""
    ep = GPU_EPS.get(option)
    if ep is None:
        return None
    working = machine.get("providers_working")
    source = "setup_bench found working (bench/machine.json)"
    if working is None:
        source = "onnxruntime lists"
        try:
            import onnxruntime as ort

            working = ort.get_available_providers()
        except Exception as exc:
            return f"--providers {option}: onnxruntime does not load ({type(exc).__name__}: {exc})"
    if ep in working:
        return None
    hint = ""
    if option == "cuda":
        hint = (" (it needs onnxruntime's CUDA build, which setup_bench does not install"
                + ("; DirectML, --providers dml, runs the networks on the GPU)" if WINDOWS
                   else ")"))
    return (f"--providers {option}: {ep} is not among the providers {source}: "
            f"{', '.join(working) or 'none'}{hint}. onnxruntime would run the networks on the "
            f"processor while the results say {option}. Use --providers auto or cpu, or run "
            "setup_bench again.")


# ------------------------------------------------------------------ suites and jobs
def load_suites(names: list[str]) -> list:
    found = sorted(p.stem for p in SUITES.glob("*.py") if not p.stem.startswith("_"))
    default = [n for n in SUITE_ORDER if n in found] + [n for n in found if n not in SUITE_ORDER]
    names = names or default
    unknown = [n for n in names if n not in found]
    if unknown:
        raise SystemExit(f"unknown suite {', '.join(unknown)}: choose from {', '.join(found)}")
    mods = []
    for name in dict.fromkeys(names):
        mod = importlib.import_module(f"suites.{name}")
        missing = [a for a in ("NAME", "jobs", "summarize") if not hasattr(mod, a)]
        if missing:
            raise SystemExit(f"bench/suites/{name}.py lacks {', '.join(missing)}")
        mods.append(mod)
    return mods


def fingerprint(inputs) -> str | None:
    """A short digest of a job's "inputs" (None when the job names none)."""
    if inputs is None:
        return None
    text = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def out_key(path: str | Path) -> str:
    """An output file's key in timing.json: its path from the package's top folder as the file
    system compares names (on Windows any case names the same file)."""
    return os.path.normcase(rel(path))


class Job:
    """One job of a suite and what happened to it in this session."""

    def __init__(self, spec: dict, suite: str):
        missing = [k for k in JOB_KEYS if k not in spec]
        if missing:
            raise SystemExit(f"a job of suite {suite} lacks {', '.join(missing)}: {spec}")
        self.spec, self.suite = spec, suite
        self.name = str(spec["name"])
        self.cmd = [str(c) for c in spec["cmd"]]
        self.out, self.log = Path(spec["out"]), Path(spec["log"])
        self.threads, self.mem_gb = int(spec["threads"]), float(spec["mem_gb"])
        self.after = [str(a) for a in spec["after"]]
        self.cwd = spec.get("cwd")
        self.inputs = fingerprint(spec.get("inputs"))
        self.status = "waiting"  # running, ok, failed, existing, interrupted, blocked
        self.ran = False  # started in this session
        self.proc = self.fh = None
        self.t0 = self.t1 = 0.0
        self.rc: int | None = None
        self.peak = 0
        self.suspended = 0.0  # seconds the driver stood still while it ran (the computer slept)

    def has_output(self) -> bool:
        try:
            if self.out.is_dir():
                return any(self.out.iterdir())
            return self.out.stat().st_size > 0
        except OSError:
            return False

    def made_with(self, timing: dict) -> str | None:
        """The inputs (digest) its output file was made with: those of the last job that wrote
        the file successfully (timing.json "outputs", by the file's path, case-insensitive on
        Windows), which need not be this job: on Windows a network relabelled in another case
        writes the same file; else this job's own record (an older driver's timing.json)."""
        made = (timing.get("outputs") or {}).get(out_key(self.out))
        if made is None:
            made = (timing.get("jobs") or {}).get(self.name)
        return (made or {}).get("inputs")

    def outdated(self, timing: dict) -> bool:
        """Whether its output was made with other inputs than it has now (``timing``: the
        contents of timing.json). An output whose inputs are not known is kept."""
        made = self.made_with(timing)
        return self.inputs is not None and made is not None and made != self.inputs

    def argv(self) -> list[str]:
        return [sys.executable if i == 0 and c == "{python}" else c
                for i, c in enumerate(self.cmd)]


def collect_jobs(mods: list, cfg: dict) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    for mod in mods:
        jobs += [Job(spec, mod.NAME) for spec in mod.jobs(cfg)]
    names = [j.name for j in jobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"job names used twice: {', '.join(dup)}")
    known, notes = set(names), []
    for job in jobs:
        unknown = [a for a in job.after if a not in known]
        if unknown:
            notes.append(f"{job.name} comes after jobs that are not part of this run "
                         f"({', '.join(unknown[:3])}{' ...' if len(unknown) > 3 else ''}); "
                         "they are ignored")
            job.after = [a for a in job.after if a in known]
    return jobs, notes


# ------------------------------------------------------------------ running
def child_env(threads: int) -> dict:
    env = dict(os.environ)
    env.update(common.thread_env(threads))
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    return env


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop ``proc`` and the processes it started."""
    if proc.poll() is not None:
        return
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        if proc.poll() is None:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def reap(job: Job) -> int | None:
    """The exit code of the job's process once it has ended, else None. On Linux and macOS the
    processes it left behind in its process group are stopped (while the ended process still
    holds the group's number), then it is reaped here, so that the kernel's record of its peak
    memory is read as well."""
    if WINDOWS or not hasattr(os, "waitid"):
        return job.proc.poll()
    pid = job.proc.pid
    try:
        if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            return None
    except ChildProcessError:  # reaped already
        return job.proc.poll()
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    _pid, status, usage = os.wait4(pid, 0)
    job.peak = max(job.peak, usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    job.proc.returncode = os.waitstatus_to_exitcode(status)
    return job.proc.returncode


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def why_failed(job: Job) -> str:
    if job.status == "blocked":
        return "blocked (its 'after' jobs wait on each other)"
    return f"exit code {job.rc}" if job.rc != 0 else "finished without writing its output"


def log_tail(path: Path, n: int = 4) -> list[str]:
    """The last ``n`` lines the latest attempt of a job wrote to its log."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    starts = [i for i, x in enumerate(lines) if x.startswith("=====") and " start " in x]
    lines = lines[starts[-1] + 1:] if starts else lines
    return [x for x in lines if x.strip() and not x.startswith("=====")][-n:]


class Runner:
    """Runs the jobs within the thread and memory budgets; records timing.json."""

    def __init__(self, cfg: dict, jobs: list[Job]):
        self.cfg, self.jobs = cfg, jobs
        self.by_name = {j.name: j for j in jobs}
        self.results = Path(cfg["results_dir"])
        self.timing_path = self.results / "timing.json"
        self.timing = read_json(self.timing_path, {}) or {}
        self.timing.setdefault("jobs", {})
        self.timing.setdefault("outputs", {})
        self.timing.setdefault("sessions", [])
        self.session = {"start": stamp(), "providers": cfg["providers"],
                        "threads_total": cfg["threads_total"],
                        "mem_budget_gb": cfg["mem_budget_gb"], "smoke": cfg["smoke"]}
        self.timing["sessions"].append(self.session)
        self.running: list[Job] = []
        self.t_start = time.time()
        self.power = Power()
        self.save_failed = False
        self.stalls: list[dict] = []  # times the driver stood still (the computer slept)

    # ---- bookkeeping
    def record(self, job: Job) -> None:
        prev = self.timing["jobs"].get(job.name) or {}
        rec = {"suite": job.suite, "status": job.status, "rc": job.rc, "start": stamp(job.t0),
               "end": stamp(job.t1), "seconds": round(job.t1 - job.t0, 1),
               "threads": job.threads, "mem_gb": job.mem_gb,
               "peak_gb": round(job.peak / 2**30, 2) if job.peak else None,
               "out": rel(job.out), "log": rel(job.log)}
        if job.suspended:
            rec["suspended_s"] = round(job.suspended, 1)
        # the inputs its output was made with: a failed attempt leaves the earlier output's
        made = job.inputs if job.status == "ok" else prev.get("inputs")
        if made is not None:
            rec["inputs"] = made
        self.timing["jobs"][job.name] = rec
        if job.status == "ok":  # what the file holds now, whichever job wrote it
            self.timing["outputs"][out_key(job.out)] = {"job": job.name, "inputs": job.inputs,
                                                        "made": stamp(job.t1)}

    def save(self) -> None:
        """Write timing.json. A failure (another program holds the file open) is a warning: the
        records stay in memory and the next save writes them."""
        counts: dict[str, int] = {}
        for job in self.jobs:
            counts[job.status] = counts.get(job.status, 0) + 1
        self.session.update(end=stamp(), wall_s=round(time.time() - self.t_start, 1),
                            counts=counts)
        if self.stalls:
            self.session["stalls"] = self.stalls
        try:
            common.write_json(self.timing_path, self.timing)
        except OSError as exc:
            if not self.save_failed:
                say(f"Warning: {rel(self.timing_path)} could not be written ({exc}); the next "
                    "save tries again")
            self.save_failed = True
            return
        if self.save_failed:
            say(f"{rel(self.timing_path)} written again")
        self.save_failed = False

    def progress(self) -> str:
        done = sum(j.status in ("ok", "failed", "existing", "blocked") for j in self.jobs)
        return f"[{done}/{len(self.jobs)}]"

    # ---- one job
    def start(self, job: Job, again: str = "") -> None:
        job.log.parent.mkdir(parents=True, exist_ok=True)
        job.out.parent.mkdir(parents=True, exist_ok=True)
        cwd = Path(job.cwd) if job.cwd else self.results
        cwd.mkdir(parents=True, exist_ok=True)
        job.fh = open(job.log, "a", encoding="utf-8")
        argv = job.argv()
        job.fh.write(f"===== {stamp()} start {job.name}: {subprocess.list2cmdline(argv)}\n")
        job.fh.flush()
        flags = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS
                 else {"start_new_session": True})
        job.t0, job.ran, job.status = time.time(), True, "running"
        try:
            job.proc = subprocess.Popen(argv, stdout=job.fh, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, cwd=cwd,
                                        env=child_env(job.threads), **flags)
        except OSError as exc:
            job.fh.write(f"could not start: {exc}\n")
            self.finish(job, 127)
            return
        self.running.append(job)
        used_t = sum(j.threads for j in self.running)
        used_m = sum(j.mem_gb for j in self.running)
        say(f"{time.strftime('%H:%M:%S')} start  {job.name} ({plural(job.threads, 'thread')}, "
            f"{job.mem_gb:g} GB; running {len(self.running)}: {plural(used_t, 'thread')}, "
            f"{used_m:.1f} GB){f'; again: {again}' if again else ''}")

    def finish(self, job: Job, rc: int | None) -> None:
        job.t1, job.rc = time.time(), rc
        secs = job.t1 - job.t0
        if rc == 0 and job.has_output():
            job.status = "ok"
        elif job.status != "interrupted":
            job.status = "failed"
        if job.fh is not None:
            note = "" if rc != 0 or job.status == "ok" else " (no output written)"
            job.fh.write(f"===== {stamp()} end {job.name}: exit code {rc}, {secs:.1f} s"
                         f"{note}\n")
            job.fh.close()
            job.fh = None
        if job in self.running:
            self.running.remove(job)
        self.record(job)
        self.save()
        peak = f", peak {job.peak / 2**30:.2f} GB" if job.peak else ""
        if job.status == "ok":
            say(f"{time.strftime('%H:%M:%S')} done   {job.name} {secs:.0f} s{peak} "
                f"{self.progress()}")
        elif job.status == "failed":
            say(f"{time.strftime('%H:%M:%S')} FAILED {job.name} ({why_failed(job)}, {secs:.0f} s) "
                f"{self.progress()}; log: {rel(job.log)}")
            for line in log_tail(job.log):
                say(f"         | {line[:160]}")

    # ---- the loop
    def ready(self, job: Job) -> bool:
        return all(self.by_name[a].status not in ("waiting", "running") for a in job.after)

    def fits(self, job: Job) -> bool:
        if not self.running:
            return True  # a job larger than the budgets runs alone
        threads = sum(j.threads for j in self.running) + job.threads
        mem = sum(j.mem_gb for j in self.running) + job.mem_gb
        return threads <= self.cfg["threads_total"] and mem <= self.cfg["mem_budget_gb"] + 1e-9

    def stalled(self, since: float, now: float) -> None:
        """The driver stood still from ``since`` to ``now`` (the computer slept, or this
        process was suspended): the jobs running across it are timed with that gap in them."""
        gap = now - since
        for job in self.running:
            job.suspended += gap
        self.stalls.append({"from": stamp(since), "seconds": round(gap, 1),
                            "jobs": [j.name for j in self.running]})
        say(f"{time.strftime('%H:%M:%S')} Note: nothing ran for {gap:.0f} s since "
            f"{time.strftime('%H:%M:%S', time.localtime(since))} (the computer slept?); the times "
            f"of the {plural(len(self.running), 'job')} running then include it (timing.json "
            "suspended_s)")

    def run(self) -> None:
        waiting = [j for j in self.jobs if j.status == "waiting"]
        last_sample = last_beat = last_loop = time.time()
        while waiting or self.running:
            now = time.time()
            if now - last_loop > STALL_S:
                self.stalled(last_loop, now)
            last_loop = now
            changed = False
            for job in list(waiting):
                if not self.ready(job):
                    continue
                again = ""
                if job.has_output():
                    again = ("its inputs changed" if job.outdated(self.timing) else
                             "a job it comes after ran" if any(self.by_name[a].ran
                                                               for a in job.after) else "")
                    if not again:
                        job.status = "existing"
                        waiting.remove(job)
                        changed = True
                        continue
                if self.fits(job):
                    waiting.remove(job)
                    self.start(job, again)
                    changed = True
            for job in list(self.running):
                rc = reap(job)
                if rc is not None:
                    self.finish(job, rc)
                    changed = True
            if not self.running and waiting and not changed:
                for job in waiting:  # waiting on each other: a cycle of "after"
                    job.status = "blocked"
                    say(f"BLOCKED {job.name}: its 'after' jobs wait on each other")
                waiting = []
            now = time.time()
            if now - last_sample >= SAMPLE_S:
                for job in self.running:
                    procs = job_processes(job.proc.pid)
                    self.power.full_speed(procs)
                    job.peak = max(job.peak, peak_bytes(job.proc.pid, procs))
                last_sample = now
            if now - last_beat >= HEARTBEAT_S and self.running:
                say(f"{time.strftime('%H:%M:%S')} running {', '.join(j.name for j in self.running)}"
                    f" {self.progress()}, {(now - self.t_start) / 60:.0f} min so far")
                last_beat = now
            if not changed:
                time.sleep(POLL_S)

    def interrupt(self) -> None:
        """Stop every running job (Ctrl-C, or the driver failed)."""
        for job in list(self.running):
            try:
                kill_tree(job.proc)
            except Exception as exc:  # the next ones must still be stopped
                say(f"Could not stop {job.name} (process {job.proc.pid}): {exc}")
            job.status = "interrupted"
            try:
                rc = job.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                rc = None
            self.finish(job, rc)


# ------------------------------------------------------------------ summary and upload
def machine_line(machine: dict) -> str:
    if not machine:
        return "Machine: unknown (bench/machine.json is missing: run setup_bench first)."
    gpus = "; ".join(machine.get("gpus") or []) or "no GPU listed"
    return (f"Machine: {machine.get('cpu', '?')}, {machine.get('cores_physical', '?')} cores "
            f"({machine.get('cores_logical', '?')} logical), {machine.get('ram_gb', '?')} GB RAM; "
            f"{gpus}; Python {machine.get('python', '?')}, onnxruntime "
            f"{(machine.get('onnxruntime') or {}).get('version', '?')}, chosen provider "
            f"{machine.get('chosen_provider', '?')}.")


def write_summary(cfg: dict, mods: list, runner: Runner) -> Path:
    results = Path(cfg["results_dir"])
    package = read_json(PACKAGE, {}) or {}
    machine = read_json(MACHINE, {}) or {}
    jobs = runner.jobs
    count = {s: sum(j.status == s for j in jobs) for s in ("ok", "failed", "existing", "blocked")}
    lines = [f"# Benchmark {cfg['results_id']}", "",
             f"Package {package.get('name', 'unknown')} (source {package.get('source', '?')}); "
             f"summary written {stamp()}.", "", machine_line(machine), "",
             f"Settings: suites {', '.join(m.NAME for m in mods)}; providers {cfg['providers']}; "
             f"{cfg['threads_total']} threads, {cfg['threads_per_job']} per job; memory budget "
             f"{cfg['mem_budget_gb']} GB; smoke {'yes' if cfg['smoke'] else 'no'}.", "",
             "Networks: " + (", ".join(f"{m['label']} ({Path(m['onnx']).name}, sha256 "
                                       f"{m['sha256']})" for m in cfg["models"]) or "none")
             + "; the comb is the baseline.", "",
             f"This session: {count['ok']} jobs done, {count['failed']} failed, "
             f"{count['existing']} kept from an earlier session"
             + (f", {count['blocked']} blocked" if count["blocked"] else "")
             + f"; {(time.time() - runner.t_start) / 60:.1f} min.", ""]
    if runner.stalls:
        lines += [f"Nothing ran for {sum(s['seconds'] for s in runner.stalls):.0f} s in all "
                  f"({plural(len(runner.stalls), 'pause')}; the computer slept?): the seconds of "
                  "the jobs running then include it (timing.json: suspended_s of each job, "
                  "stalls of the session).", ""]
    failed = [j for j in jobs if j.status in ("failed", "blocked")]
    if failed:
        lines += ["Failed jobs (their logs are in the results):", ""]
        lines += [f"- {j.name}: {why_failed(j)}, log {rel(j.log)}" for j in failed]
        lines.append("")
    for mod in mods:
        try:
            text = mod.summarize(cfg)
        except Exception as exc:  # one suite's summary must not lose the others
            text = f"## {mod.NAME}\n\nThe summary failed: {type(exc).__name__}: {exc}\n"
        lines += [text.rstrip(), ""]
    lines += ["## Job times", "", "| job | status | seconds | threads | memory GB (estimate) | "
              "peak GB |", "| --- | --- | --- | --- | --- | --- |"]
    timing = runner.timing["jobs"]
    for job in jobs:
        t = timing.get(job.name, {})
        status = job.status if job.status != "existing" else f"kept ({t.get('status', 'done')})"
        lines.append(f"| {job.name} | {status} | {t.get('seconds', '')} | {job.threads} | "
                     f"{job.mem_gb:g} | {t.get('peak_gb') or ''} |")
    path = results / "summary.md"
    common.write_atomic(path, "\n".join(lines) + "\n")
    return path


def write_upload(cfg: dict) -> Path:
    """results/<id>_upload.zip: the results folder without raw data files, with the machine
    record and the setup log."""
    results = Path(cfg["results_dir"])
    rid = cfg["results_id"]
    zpath = results.parent / f"{rid}_upload.zip"
    tmp = zpath.with_name(zpath.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(results.rglob("*")):
            if (path.is_file() and path.suffix.lower() not in NO_UPLOAD
                    and path.name != ".lock"):
                zf.write(path, f"{rid}/{path.relative_to(results).as_posix()}")
        for path in (MACHINE, BENCH / "setup_bench.log", PACKAGE):
            if path.is_file():
                zf.write(path, f"{rid}/bench/{path.name}")
    try:
        os.replace(tmp, zpath)
    except PermissionError:  # the old zip is open in another program (Windows)
        zpath = zpath.with_name(f"{rid}_upload_{time.strftime('%H%M%S')}.zip")
        os.replace(tmp, zpath)
    return zpath


# ------------------------------------------------------------------ main
def plan(jobs: list[Job], timing: dict) -> dict[str, str]:
    """What a run would do with each job, by name: "to run" (no output), "again: ..." (its
    output is redone) or "exists" (kept). ``timing``: the contents of timing.json."""
    by_name = {j.name: j for j in jobs}
    state: dict[str, str] = {}

    def visit(job: Job, stack: tuple = ()) -> str:
        if job.name not in state:
            befores = [by_name[a] for a in job.after if a not in stack]
            runs = [visit(b, (*stack, job.name)) != "exists" for b in befores]
            state[job.name] = ("to run" if not job.has_output() else
                               "again: its inputs changed" if job.outdated(timing) else
                               "again: a job it comes after runs" if any(runs) else "exists")
        return state[job.name]

    for job in jobs:
        visit(job)
    return state


def plan_lines(jobs: list[Job], state: dict[str, str]) -> list[str]:
    out = []
    width = max((len(j.name) for j in jobs), default=10)
    for job in jobs:
        after = f"  after {len(job.after)} job{'s' if len(job.after) != 1 else ''}" \
            if job.after else ""
        out.append(f"  {job.name:<{width}}  {job.threads:>2} thr {job.mem_gb:>5.1f} GB  "
                   f"{state[job.name]}{after}")
    return out


def check_summaries(mods: list, cfg: dict) -> list[str]:
    """Each suite's summary of an empty results folder (the dry run's check that the suites
    import and tolerate missing outputs)."""
    import tempfile

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        empty = dict(cfg, results_dir=str(Path(tmp) / "results"))
        for mod in mods:
            try:
                text = mod.summarize(empty)
                out.append(f"  {mod.NAME}: summary ok ({len(text.splitlines())} lines)")
            except Exception as exc:
                out.append(f"  {mod.NAME}: summary FAILED: {type(exc).__name__}: {exc}")
    return out


def take_lock(results: Path) -> Path:
    lock = results / ".lock"
    held = read_json(lock)
    if held and held.get("pid") != os.getpid() and pid_alive(int(held.get("pid", 0))):
        raise SystemExit(f"Another run_bench (process {held['pid']}, started "
                         f"{held.get('started')}) is working on {rel(results)}; wait for it or "
                         f"stop it. If none runs, delete {rel(lock)}.")
    results.mkdir(parents=True, exist_ok=True)
    common.write_json(lock, {"pid": os.getpid(), "started": stamp()})
    return lock


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", help="comma-separated suites to run (default: all: "
                                     + ", ".join(SUITE_ORDER) + ")")
    ap.add_argument("--smoke", action="store_true", default=None,
                    help="a few minutes' end-to-end check: every suite shrunk")
    ap.add_argument("--providers", choices=("auto", "dml", "cuda", "cpu"),
                    help="where the networks run (default: the one setup_bench chose)")
    ap.add_argument("--threads", type=int, help="threads of all jobs together (default: logical "
                                                 "cores - 2)")
    ap.add_argument("--mem-gb", type=float, help="peak memory of all jobs together, GB (default: "
                                                 "0.7 x RAM)")
    ap.add_argument("--threads-per-job", type=int, help="threads of each job (default 4)")
    ap.add_argument("--model", action="append", metavar="LABEL=PATH",
                    help="also evaluate this network (its .json next to it); repeatable")
    ap.add_argument("--no-default-models", action="store_true",
                    help="only the --model networks, not bench/models and runs")
    ap.add_argument("--thermo-dir", help="folder with the four Thermo .raw files (searched with "
                                         "its subfolders; default data/thermo_raw)")
    ap.add_argument("--rda-dir", help="folder with the RDa .mzMLb runs (default: "
                                      "bench/private/confidential_sample_1/data)")
    ap.add_argument("--results-id", help="results folder name under bench/results (default: the "
                                         "package's date; the same id continues a run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the settings and the jobs; run and write nothing")
    args = ap.parse_args(argv)
    cfg, notes = build_cfg(args)
    mods = load_suites(cfg["suites"])
    cfg["suites"] = [m.NAME for m in mods]
    results = Path(cfg["results_dir"])

    dry = " (dry run: nothing runs, nothing is written)" if args.dry_run else ""
    say(f"Benchmark {cfg['results_id']}{dry}")
    say(f"Results:   {rel(results)}")
    say(f"Suites:    {', '.join(cfg['suites'])}{' (smoke)' if cfg['smoke'] else ''}")
    # absolute paths: a relative option is taken from the folder the command runs in
    say("Networks:  " + (", ".join(f"{m['label']} ({m['onnx']})" for m in cfg["models"])
                         or "none (the comb only)"))
    say(f"Providers: {cfg['providers']}; threads {cfg['threads_total']} in all, "
        f"{cfg['threads_per_job']} per job; memory budget {cfg['mem_budget_gb']} GB")
    raws = common.find_thermo_raws(cfg["thermo_dir"]) if Path(cfg["thermo_dir"]).is_dir() else {}
    say(f"Thermo:    {len(raws)} of 4 runs found under {cfg['thermo_dir']}"
        + (f" ({', '.join(raws)})" if raws else "; put the .raw files there or use --thermo-dir"))
    for note in notes:
        say(f"Note:      {note}")
    jobs, job_notes = collect_jobs(mods, cfg)
    for note in job_notes:
        say(f"Note:      {note}")
    if args.dry_run:
        state = plan(jobs, read_json(results / "timing.json", {}) or {})
        kept = sum(s == "exists" for s in state.values())
        say(f"Jobs:      {len(jobs)} ({len(jobs) - kept} to run, {kept} kept: their output "
            "is there)")
        for line in plan_lines(jobs, state):
            say(line)
        lines = check_summaries(mods, cfg)
        for line in lines:
            say(line)
        return 1 if any("FAILED" in x for x in lines) else 0
    if not jobs:
        say("No jobs: nothing to do.")
    if not Path(cfg["thermo_dir"]).exists() and cfg["thermo_dir"] == str(REPO / "data" /
                                                                       "thermo_raw"):
        Path(cfg["thermo_dir"]).mkdir(parents=True, exist_ok=True)
    lock = take_lock(results)
    try:
        return run_jobs(cfg, mods, jobs)
    finally:  # whatever happened: no job of this run is running any more
        lock.unlink(missing_ok=True)


def run_jobs(cfg: dict, mods: list, jobs: list[Job]) -> int:
    """Run the jobs, then write the summary and the upload zip; the exit code."""
    common.write_json(Path(cfg["results_dir"]) / "config.json", cfg)
    runner = Runner(cfg, jobs)
    say(f"{time.strftime('%H:%M:%S')} {len(jobs)} jobs; Ctrl-C stops them (run again to go on)")

    def on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal.default_int_handler)  # also where it was ignored
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_term)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_term)
    runner.power.keep_awake(True)
    try:
        runner.run()
    except BaseException as exc:  # Ctrl-C, or the driver itself failed: stop the jobs first
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        ctrl_c = isinstance(exc, KeyboardInterrupt)
        why = "Ctrl-C" if ctrl_c else f"The driver failed ({type(exc).__name__}: {exc})"
        say(f"{time.strftime('%H:%M:%S')} {why}: stopping "
            f"{plural(len(runner.running), 'running job')} ...")
        runner.interrupt()
        runner.save()
        if not ctrl_c:
            raise
        say("Stopped. Run it again to go on: the finished jobs are kept.")
        return 130
    finally:
        runner.power.keep_awake(False)
    runner.save()
    count = {s: sum(j.status == s for j in jobs) for s in ("ok", "failed", "existing")}
    say(f"{time.strftime('%H:%M:%S')} jobs finished ({count['ok']} done, {count['failed']} "
        f"failed, {count['existing']} kept from an earlier session); writing the summary ...")
    written = {}
    for what, write in (("summary", lambda: write_summary(cfg, mods, runner)),
                        ("upload zip", lambda: write_upload(cfg))):
        try:
            written[what] = write()
        except Exception as exc:  # the job outputs are kept: a rerun writes it
            say(f"Warning: the {what} could not be written ({type(exc).__name__}: {exc}); run "
                "again to write it")
    failed = [j for j in jobs if j.status in ("failed", "blocked")]
    if "summary" in written:
        say(f"Summary:   {rel(written['summary'])}")
    if "upload zip" in written:
        zpath = written["upload zip"]
        say(f"Upload:    {rel(zpath)} ({zpath.stat().st_size / 2**20:.1f} MB)")
    if failed:
        say(f"{len(failed)} jobs failed: {', '.join(j.name for j in failed[:5])}"
            f"{' ...' if len(failed) > 5 else ''}. Their logs are in the upload; run again to "
            "retry them.")
    if runner.save_failed:
        say(f"{rel(runner.timing_path)} could not be written (see the warning above): run again "
            "once no other program holds it open.")
    return 1 if failed or len(written) < 2 or runner.save_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
