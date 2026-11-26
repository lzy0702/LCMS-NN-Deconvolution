"""Set up the benchmark on this computer: run it once; running it again is safe.

    bench\\setup_bench.bat          (Windows: double-click it, or run it in a terminal)
    python3 bench/setup_bench.py   (Linux)

1. Finds Python 3.12 (without internet it has to be the Python the wheels package is for, 3.12;
   with internet 3.11 and 3.13 also work): ``py -3.12`` and the usual install folders on
   Windows, ``python3.12`` and the like elsewhere, preferring a Python that bench/wheels holds
   wheels for.
2. Creates the environment ``.venv-bench`` in the package's top folder, apart from the training
   environment ``.venv``: onnxruntime-directml (the GPU through DirectML) and onnxruntime cannot
   share an environment. An environment whose install did not finish (from PyPI, without
   internet) is made again with a Python that bench/wheels holds wheels for, once one is found.
3. Installs what the benchmark needs, at the versions the package was checked with: without
   internet from ``bench/wheels/<tag>/`` when that folder holds wheels for this Python (a tag
   such as ``cp312-win_amd64``), else from PyPI. onnxruntime-directml on Windows, onnxruntime
   elsewhere. Nothing is installed when everything is there already. On Windows it says when
   the Visual C++ 2015-2022 runtime, which onnxruntime and numba need, is missing.
4. Checks that lcmsdeconv imports from the package's ``src`` folder and that numba and
   onnxruntime load, lists onnxruntime's providers and runs a short inference of every network
   in bench/models on each GPU provider and on the processor (outputs compared, times measured).
5. Writes bench/machine.json: Python, cores, RAM, GPUs, the providers that work for every
   network and the one the benchmark uses (run_bench.py reads it).

Everything printed also goes to bench/setup_bench.log.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
WINDOWS = sys.platform == "win32"
VENV = ROOT / ".venv-bench"
VPY = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")
WHEELS = BENCH / "wheels"
MODELS = BENCH / "models"
MACHINE = BENCH / "machine.json"
LOG = BENCH / "setup_bench.log"
#: what the last install put into .venv-bench (read when setup runs again)
RECORD = VENV / "bench_setup.json"
#: the Pythons the environment may use, the preferred first
PYTHONS = ((3, 12), (3, 13), (3, 11))
#: the benchmark's dependencies: (version checked, lowest accepted). The checked versions are
#: those of the development environment (Python 3.13); each has Windows wheels for Python 3.12
#: and 3.13. The lowest accepted ones apply only where a checked version cannot be installed
#: (Python 3.11), from PyPI.
PINS = {
    "numpy": ("2.5.5", "1.24"), "scipy": ("1.18.1", "1.10"), "pandas": ("3.1.1", "2.0"),
    "pyteomics": ("5.0.2", "4.6"), "lxml": ("6.1.3", "4.9"), "psims": ("1.4.0", "1.3"),
    "h5py": ("3.16.0", "3.8"), "PyYAML": ("6.0.3", "6.0"), "click": ("8.5.1", "8.1"),
    "Jinja2": ("3.1.6", "3.1"), "matplotlib": ("3.11.2", "3.7"), "opentfraw": ("3.1.0", "3.1"),
    "numba": ("0.68.1", "0.58"), "psutil": ("7.2.2", "5.9"),
}
#: onnxruntime: the DirectML build on Windows (its newest release), the processor build elsewhere
ORT_WINDOWS = ("onnxruntime-directml", "1.24.4", "1.17")
ORT_OTHER = ("onnxruntime", "1.32.0", "1.16")
#: their dependencies at the development environment's versions (constraints: they apply where
#: a dependency is installed at all)
CONSTRAINTS = {
    "contourpy": "1.4.0", "cycler": "0.12.1", "fonttools": "4.68.0", "kiwisolver": "1.5.1",
    "packaging": "26.3", "pillow": "13.0.0", "pyparsing": "3.3.3", "python-dateutil": "4.0.0",
    "six": "1.17.0", "tzdata": "2026.6", "SQLAlchemy": "2.1.9", "MarkupSafe": "3.0.4",
    "llvmlite": "0.50.0", "flatbuffers": "25.12.19", "protobuf": "7.36.2", "sympy": "1.14.0",
    "mpmath": "1.3.0", "typing_extensions": "4.16.0",
}
PIP = "26.3"
#: threads of the inference check on the processor (run_bench.py's threads per job)
CHECK_THREADS = 4
#: the GPU providers tried, in the order the benchmark prefers them, with run_bench's names
GPU_PROVIDERS = {"DmlExecutionProvider": "dml", "CUDAExecutionProvider": "cuda"}
#: share of the ion bins whose top charge a GPU provider must read as the processor does
AGREEMENT = 0.98
#: pip output that means the package index could not be reached
NETWORK_TROUBLE = ("Tunnel connection failed", "ProxyError", "Max retries exceeded",
                   "Failed to establish a new connection", "ConnectionError", "Connection reset",
                   "Read timed out", "SSLError", "certificate verify failed",
                   "Temporary failure in name resolution", "getaddrinfo failed",
                   "Retrying (Retry(")
NO_VERSION = ("No matching distribution found", "Could not find a version that satisfies")


def requirements(windows: bool = WINDOWS, pinned: bool = True) -> list[str]:
    """The requirement lines: the checked versions (``pinned``) or the lowest accepted ones."""
    name, pin, low = ORT_WINDOWS if windows else ORT_OTHER
    items = [*PINS.items(), (name, (pin, low))]
    return [f"{n}=={p}" if pinned else f"{n}>={lo}" for n, (p, lo) in items]


def constraint_lines() -> list[str]:
    return [f"{n}=={v}" for n, v in CONSTRAINTS.items()]


def say(msg: str = "") -> None:
    """Print ``msg`` and add it to the log. The messages are English; the paths, pip's output
    and Windows' own messages (localized) in them are shown as they are (main() makes a
    console that cannot show a character print '?' instead of failing)."""
    print(msg, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def child_env() -> dict:
    return dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")


def run(cmd: list, quiet: bool = False, timeout: float | None = None) -> tuple[int, str]:
    """Run ``cmd``, showing (unless ``quiet``) and logging its output: (exit code, its last
    lines). A command still running after ``timeout`` seconds is stopped."""
    say("> " + " ".join(str(c) for c in cmd))
    tail: list[str] = []
    try:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=child_env())
    except OSError as exc:
        say(f"Could not start it: {exc}")
        return 127, str(exc)

    def pump() -> None:
        with open(LOG, "a", encoding="utf-8") as fh:
            for line in proc.stdout:
                fh.write(line)
                tail.append(line)
                del tail[:-60]
                if not quiet:
                    print(line, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
        say(f"Stopped after {timeout:.0f} s.")
    reader.join(timeout=30)
    return rc, "".join(tail)


# ------------------------------------------------------------------ Python
INSPECT = ("import json, struct, sys, sysconfig, venv; "
           "print(json.dumps([sys.version_info[0], sys.version_info[1], sys.version_info[2], "
           "struct.calcsize('P') * 8, sysconfig.get_platform(), sys.executable]))")


def inspect_python(cmd: list[str]) -> dict | None:
    """Version, bits, platform and executable of the Python ``cmd`` starts, or None."""
    try:
        out = subprocess.run([*cmd, "-c", INSPECT], capture_output=True, text=True, timeout=60)
        major, minor, micro, bits, plat, exe = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, TypeError):
        return None
    return {"version": (int(major), int(minor)), "micro": int(micro), "bits": int(bits),
            "platform": str(plat), "exe": str(exe), "cmd": cmd}


def wheel_tag(info: dict) -> str:
    """The folder of bench/wheels for a Python: ``cp312-win_amd64``, ``cp313-linux_x86_64``."""
    major, minor = info["version"]
    return f"cp{major}{minor}-" + info["platform"].replace("-", "_").replace(".", "_")


def has_wheels(tag: str) -> bool:
    folder = WHEELS / tag
    return folder.is_dir() and any(folder.glob("*.whl"))


def wheel_folders() -> list[str]:
    return sorted(p.name for p in WHEELS.iterdir() if has_wheels(p.name)) if WHEELS.is_dir() \
        else []


def tag_version(tag: str) -> str:
    """``cp312-win_amd64`` -> ``3.12``."""
    m = re.match(r"cp(\d)(\d+)-", tag)
    return f"{m.group(1)}.{m.group(2)}" if m else tag


def shipped_tags() -> list[str]:
    """The wheel tags of the wheels packages built with this package (bench/package.json)."""
    try:
        tags = json.loads((BENCH / "package.json").read_text(encoding="utf-8"))["wheel_tags"]
        return [str(t) for t in tags]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def offline_tags() -> list[str]:
    """The tags there are wheels for: in bench/wheels, or in a wheels package of this package."""
    return sorted(set(wheel_folders()) | set(shipped_tags()))


def python_hint(version: str) -> list[str]:
    """How to install Python ``version``."""
    if WINDOWS:
        return [f"    winget install -e --id Python.Python.{version}",
                "or with the installer from https://www.python.org/downloads/ (tick \"Add "
                "python.exe to PATH\"; without internet, an installer brought along)"]
    return [f"    your package manager (python{version} and python{version}-venv)"]


def offline_advice(tag: str) -> list[str]:
    """What to do without internet when bench/wheels holds no wheels for ``tag``."""
    if tag in shipped_tags():  # built, not unzipped
        return [f"Without internet, unzip the wheels package lcmsdeconv_bench_wheels_{tag}_<date>"
                f".zip into the package folder (it fills bench/wheels/{tag}) and run this again."]
    tags = offline_tags()
    if not tags:
        return ["Without internet the dependencies cannot be installed: connect this computer to "
                "the internet and run this again."]
    want = tag_version(tags[0])
    unzip = ("" if has_wheels(tags[0]) else
             f"; unzip the wheels package into the package folder first (it fills "
             f"bench/wheels/{tags[0]})")
    return [f"The dependencies for installing without internet are for Python {want} "
            f"({tags[0]}), not for this environment's Python {tag_version(tag)}. Install Python "
            f"{want} with", *python_hint(want),
            f"then run this again: it makes the environment again with Python {want} and "
            f"installs from bench/wheels{unzip}."]


#: the Visual C++ 2015-2022 runtime files that onnxruntime's and numba's modules load (python.org's
#: Python brings the vcruntime ones only; no wheel brings msvcp140.dll under that name)
VC_DLLS = ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")
VC_REDIST = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def vc_runtime_missing() -> list[str]:
    """Windows: the runtime files that neither System32 nor the folder of the environment's
    Python holds (where Windows looks when a Python module loads them); [] elsewhere."""
    if not WINDOWS:
        return []
    folders = [Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"]
    base = venv_base()
    if base:
        folders.append(Path(base).parent if base.lower().endswith(".exe") else Path(base))
    return [d for d in VC_DLLS if not any((f / d).is_file() for f in folders)]


def vc_runtime_advice(missing: list[str]) -> None:
    what = f"{', '.join(missing)} not found" if missing else "a DLL of it could not be loaded"
    say(f"The Microsoft Visual C++ 2015-2022 runtime (x64) is missing or too old ({what}): "
        "onnxruntime and numba need it. Install it with")
    say("    winget install -e --id Microsoft.VCRedist.2015+.x64")
    say(f"or with vc_redist.x64.exe from {VC_REDIST} (without internet, download it on another "
        "computer), then run this again.")


def python_candidates() -> list[list[str]]:
    """Commands that may start a suitable Python, the preferred versions first."""
    cands: list[list[str]] = []
    if WINDOWS:
        cands += [["py", f"-{a}.{b}"] for a, b in PYTHONS]
        roots = [Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Python"
                 if os.environ.get("LOCALAPPDATA") else None,
                 Path(os.environ.get("ProgramFiles", r"C:\Program Files")), Path("C:\\")]
        for a, b in PYTHONS:
            for root in roots:
                if root is not None and (root / f"Python{a}{b}" / "python.exe").is_file():
                    cands.append([str(root / f"Python{a}{b}" / "python.exe")])
        cands.append(["python"])
    else:
        cands += [[f"python{a}.{b}"] for a, b in PYTHONS] + [["python3"]]
    cands.append([sys.executable])
    return cands


def find_python(explicit: str | None) -> dict | None:
    """The Python to create the environment with: ``explicit``, else the preferred version
    among those found, a Python that bench/wheels holds wheels for first."""
    if explicit:
        found = inspect_python([explicit])
        if found is None or found["version"] not in PYTHONS or found["bits"] != 64:
            say(f"{explicit} is not a 64-bit Python 3.11, 3.12 or 3.13.")
            return None
        return found
    seen: dict[str, dict] = {}
    for cmd in python_candidates():
        info = inspect_python(cmd)
        if info is not None and info["bits"] == 64 and info["version"] in PYTHONS:
            seen.setdefault(os.path.normcase(info["exe"]), info)
    if not seen:
        return None
    return min(seen.values(), key=lambda info: (not has_wheels(wheel_tag(info)),
                                                PYTHONS.index(info["version"])))


# ------------------------------------------------------------------ environment
def venv_python() -> dict | None:
    return inspect_python([str(VPY)]) if VPY.exists() else None


def venv_base() -> str | None:
    """The Python .venv-bench was created from (its pyvenv.cfg)."""
    try:
        text = (VENV / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        return None
    found = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            found[key.strip()] = value.strip()
    return found.get("executable") or found.get("home")


def version_text(info: dict) -> str:
    return f"{info['version'][0]}.{info['version'][1]}"


def wheels_python(current: dict, explicit: str | None, source: str) -> dict | None:
    """A Python to make .venv-bench with again, or None to keep it: one that bench/wheels holds
    wheels for, when it holds none for the environment's Python and the environment's
    dependencies are not installed (its install from PyPI failed without internet, and a Python
    the wheels are for was installed since)."""
    if (source == "pypi" or has_wheels(wheel_tag(current)) or not wheel_folders()
            or install_complete()):
        return None
    found = find_python(explicit)
    return found if found is not None and has_wheels(wheel_tag(found)) else None


def make_venv(explicit: str | None, recreate: bool, source: str) -> dict | None:
    """.venv-bench: the one there when it works with a suitable Python (unless ``recreate``,
    or :func:`wheels_python` finds a Python to make it again with), else created with
    ``explicit`` or the Python :func:`find_python` prefers."""
    current = venv_python()
    if current is not None and current["version"] in PYTHONS and not recreate:
        base = wheels_python(current, explicit, source)
        if base is None:
            say(f"The environment {VENV.name} exists (Python {version_text(current)}."
                f"{current['micro']}): kept.")
            return current
        why = (f"its dependencies are not installed and bench/wheels holds none for its Python "
               f"{version_text(current)}, but does for Python {version_text(base)}, found here")
    else:
        base = find_python(explicit)
        if base is None:
            tags = offline_tags()
            want = tag_version(tags[0]) if tags else "3.12"
            say(f"Python {want} was not found ("
                + (f"without internet it has to be {want}, which the wheels package is for; "
                   "with internet 3.11 and 3.13 also work" if tags else
                   "3.11 and 3.13 also work") + "). Install it with")
            for line in python_hint(want):
                say(line)
            say("then run this again.")
            return None
        why = ("--recreate" if recreate else "it does not work" if current is None
               else f"its Python {version_text(current)} is not 3.11-3.13")
    if source != "pypi" and wheel_folders() and not has_wheels(wheel_tag(base)):
        want = tag_version(wheel_folders()[0])
        say(f"Note: bench/wheels holds the dependencies for Python "
            f"{', '.join(tag_version(t) for t in wheel_folders())} only: with Python "
            f"{version_text(base)} they come from PyPI, which needs internet. Without internet, "
            f"install Python {want} first with")
        for line in python_hint(want):
            say(line)
        say("then run this again: the environment is then made again with it.")
    if VENV.exists():
        say(f"Deleting {VENV} ({why}) ...")
        shutil.rmtree(VENV, ignore_errors=True)
        if VENV.exists():
            say(f"{VENV} could not be deleted completely: close the programs that use it "
                "(other terminals, editors) and run this again.")
            return None
    say(f"Creating the environment {VENV} with Python {base['version'][0]}."
        f"{base['version'][1]}.{base['micro']} ({base['exe']}) ...")
    rc, tail = run([*base["cmd"], "-m", "venv", str(VENV)])
    if rc != 0:
        if "ensurepip" in tail:
            say("This Python cannot create environments (on Debian or Ubuntu install the "
                f"python{base['version'][0]}.{base['version'][1]}-venv package).")
        return None
    ignore = VENV / ".gitignore"  # never committed (Python 3.13's venv writes it too)
    if not ignore.exists():
        try:
            ignore.write_text("# Created by setup_bench\n*\n", encoding="utf-8")
        except OSError:
            pass
    return venv_python()


INSTALLED = r"""
import json, sys
from importlib import metadata
out = {}
for name in json.loads(sys.argv[1]):
    try:
        out[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        out[name] = None
print(json.dumps(out))
"""


def dist_name(line: str) -> str:
    return line.split("==")[0].split(">=")[0].strip()


def installed(names: list[str]) -> dict:
    """{distribution: its version in .venv-bench, or None}."""
    try:
        out = subprocess.run([str(VPY), "-c", INSTALLED, json.dumps(names)], capture_output=True,
                             text=True, timeout=120, env=child_env())
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def satisfied(lines: list[str]) -> bool:
    """Whether .venv-bench holds every requirement (``==``: that version; ``>=``: any)."""
    have = installed([dist_name(x) for x in lines])
    for line in lines:
        version = have.get(dist_name(line))
        if version is None or ("==" in line and version != line.split("==")[1]):
            return False
    return True


def read_record() -> dict:
    try:
        return json.loads(RECORD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def install_complete() -> bool:
    """Whether .venv-bench holds what its last install put there, at this package's versions."""
    record = read_record()
    return bool(record.get("pins") == requirements() and record.get("requirements")
                and satisfied(record["requirements"]))


def resolve_ort_conflict(tag: str) -> bool:
    """On Windows the DirectML build alone must provide ``onnxruntime``: remove the processor
    build if something installed it next to it (the two overwrite each other's files)."""
    if not WINDOWS:
        return True
    have = installed(["onnxruntime", ORT_WINDOWS[0]])
    if not have.get("onnxruntime"):
        return True
    say("The processor build of onnxruntime is installed next to onnxruntime-directml: "
        "removing it and installing the DirectML build again ...")
    run([VPY, "-m", "pip", "uninstall", "-y", "onnxruntime"])
    links = ["--no-index", "--find-links", WHEELS / tag] if has_wheels(tag) else []
    version = have.get(ORT_WINDOWS[0]) or ORT_WINDOWS[1]
    rc, _ = run([VPY, "-m", "pip", "install", "--force-reinstall", "--no-deps", *links,
                 f"{ORT_WINDOWS[0]}=={version}"])
    return rc == 0


def install(env: dict, source: str) -> bool:
    """Install the requirements into .venv-bench (``source``: auto, wheels or pypi)."""
    tag = wheel_tag(env)
    pinned = requirements()
    if source == "auto" and install_complete():
        say(f"The dependencies are installed already (from {read_record().get('source', '?')}).")
        return resolve_ort_conflict(tag)
    use_wheels = has_wheels(tag) and source != "pypi"
    if source == "wheels" and not use_wheels:
        say(f"bench/wheels/{tag} holds no wheels for this Python (wheels found: "
            f"{', '.join(wheel_folders()) or 'none'}). Install from PyPI (leave out --source "
            "wheels), or create the environment with a Python the wheels are for (--recreate "
            "--python PATH).")
        return False
    req, con = VENV / "bench_requirements.txt", VENV / "bench_constraints.txt"
    req.write_text("\n".join(pinned) + "\n", encoding="utf-8")
    con.write_text("\n".join(constraint_lines()) + "\n", encoding="utf-8")
    lines = pinned
    if use_wheels:
        where = f"bench/wheels/{tag}"
        links = ["--no-index", "--find-links", WHEELS / tag]
        say(f"Installing the dependencies from {where} (no internet needed) ...")
        if any((WHEELS / tag).glob(f"pip-{PIP}-*.whl")):
            run([VPY, "-m", "pip", "install", *links, "--upgrade", f"pip=={PIP}"], quiet=True)
        rc, tail = run([VPY, "-m", "pip", "install", *links, "-r", req, "-c", con])
        if rc != 0:
            say(f"Installing from {where} failed (see above; the full output is in {LOG}). "
                "With internet, run this again with --source pypi.")
            return False
    else:
        where = "PyPI"
        if wheel_folders():
            say(f"bench/wheels holds no wheels for this Python ({tag}; found: "
                f"{', '.join(wheel_folders())}): installing from PyPI.")
        say("Installing the dependencies from PyPI (about 200 MB) ...")
        run([VPY, "-m", "pip", "install", "--upgrade", f"pip=={PIP}"], quiet=True)
        rc, tail = run([VPY, "-m", "pip", "install", "-r", req, "-c", con])
        if rc != 0 and any(m in tail for m in NETWORK_TROUBLE):
            say("pip could not reach PyPI (see the messages above).")
            for line in offline_advice(tag):
                say(line)
            return False
        if rc != 0 and any(m in tail for m in NO_VERSION):
            say("Some of the checked versions do not exist for this Python: installing the "
                "newest versions it accepts instead.")
            lines = requirements(pinned=False)
            req.write_text("\n".join(lines) + "\n", encoding="utf-8")
            rc, tail = run([VPY, "-m", "pip", "install", "-r", req])
            where = "PyPI (newest accepted versions)"
        if rc != 0:
            say(f"pip could not install the dependencies (see the error above; the full output "
                f"is in {LOG}).")
            return False
    RECORD.write_text(json.dumps({"pins": pinned, "requirements": lines, "source": where,
                                  "finished": time.strftime("%Y-%m-%d %H:%M")}, indent=1) + "\n",
                      encoding="utf-8")
    return resolve_ort_conflict(tag)


# ------------------------------------------------------------------ the check (in .venv-bench)
def check_models() -> list[Path]:
    """The networks of bench/models (by name) that have their sidecar .json: every one is
    checked (they differ in layers and input length, and each runs on the chosen provider)."""
    return [p for p in sorted(MODELS.glob("*.onnx")) if p.with_suffix(".json").is_file()] \
        if MODELS.is_dir() else []


def test_spectrum(step: float):
    """A negative-mode spectrum on the log grid (m/z 400-4000): the isotope envelopes of three
    oligonucleotide-sized masses over a range of charges, Gaussian peaks at a resolving power of
    30 000, noise of 1 % of the largest peak. Returns (grid, intensity, noise sigma)."""
    import numpy as np

    from lcmsdeconv.nn.grid import LogMzGrid

    rng = np.random.default_rng(0)
    grid = LogMzGrid(400.0, 4000.0, step, -1)
    mz = grid.mz
    y = np.zeros(grid.size)
    k = np.arange(12)
    factorial = np.cumprod(np.r_[1.0, np.arange(1, 12)])
    for mass, zc, amp in ((5600.0, 6, 1.0), (7400.0, 8, 0.6), (9100.0, 10, 0.3)):
        lam = mass * 3.6e-4  # mean extra neutrons of an oligonucleotide-like composition
        iso = np.exp(-lam) * lam ** k / factorial
        for z in range(max(2, zc - 4), zc + 5):
            share = amp * np.exp(-0.5 * ((z - zc) / 2.0) ** 2)
            for n, a in zip(k, iso):
                centre = (mass + n * 1.00336) / z - 1.007276
                if not 400.0 < centre < 4000.0:
                    continue
                sd = centre / 30000.0 / 2.3548
                b = int(grid.mz_to_bin(centre))
                lo, hi = max(0, b - 40), min(grid.size, b + 41)
                y[lo:hi] += share * a * np.exp(-0.5 * ((mz[lo:hi] - centre) / sd) ** 2)
    sigma = 0.01 * float(y.max())
    y = np.clip(y + rng.normal(0.0, sigma, grid.size), 0.0, None)
    return grid, y, sigma


def inference_check(model: Path, threads: int) -> dict:
    """``model`` reads the test spectrum through ChargePredictor, as the pipeline runs it, on each
    GPU provider onnxruntime lists and on the processor: the providers the session uses, the
    times (the first read, with the session's warm-up, and the best of two more) and, for a GPU,
    the share of the ion bins whose top charge it reads as the processor does."""
    import numpy as np
    import onnxruntime as ort

    from lcmsdeconv.core.method import Method
    from lcmsdeconv.nn.infer import ChargePredictor

    instrument = Method.load("orbitrap_iprp_neg_oligo").instrument
    avail = ort.get_available_providers()
    order = [p for p in GPU_PROVIDERS if p in avail] + ["CPUExecutionProvider"]
    out: dict = {"model": model.name, "threads": threads, "providers": {}}
    preds, data = {}, None
    for prov in order:
        rec: dict = {"ok": False}
        out["providers"][prov] = rec
        try:
            plist = [prov] if prov == "CPUExecutionProvider" else [prov, "CPUExecutionProvider"]
            pred = ChargePredictor(model, providers=plist, instrument=instrument)
            rec["session_providers"] = pred.session.get_providers()
            if rec["session_providers"][0] != prov:
                rec["error"] = f"the session runs on {rec['session_providers'][0]} instead"
                continue
            if data is None:
                data = test_spectrum(pred.grid_step or 2e-5)
                out.update(bins=int(data[0].size), view=pred.view)
            times = []
            for _ in range(3):
                t = time.perf_counter()
                result = pred.predict_grid(*data)
                times.append(time.perf_counter() - t)
            rec.update(ok=True, first_s=round(times[0], 3), best_s=round(min(times[1:]), 3))
            preds[prov] = result
        except Exception as exc:  # a provider that fails is reported, not fatal
            rec["error"] = f"{type(exc).__name__}: {exc}"[:800]
    cpu = preds.get("CPUExecutionProvider")
    if cpu is not None:
        ion = data[1] > 3.0 * data[2]
        out["ion_bins"] = int(ion.sum())
        for prov, result in preds.items():
            if prov == "CPUExecutionProvider":
                continue
            agree = float((result.top1_z[ion] == cpu.top1_z[ion]).mean()) if ion.any() else 1.0
            dp = float(np.abs(result.top1_p - cpu.top1_p).max())
            rec = out["providers"][prov]
            rec.update(top1_agreement=round(agree, 5), max_abs_dp=round(dp, 5),
                       ok=agree >= AGREEMENT)
            if agree < AGREEMENT:
                rec["error"] = (f"its top charges agree with the processor's on {agree:.1%} of "
                                f"the ion bins only ({AGREEMENT:.0%} wanted)")
    return out


def dml_adapters(model: Path, in_ch: int, length: int = 131072) -> dict:
    """Sessions of ``model`` on DirectML with onnxruntime's default adapter and with each adapter
    number in turn (device_id 0, 1, ...), timed on the same random input and compared with the
    processor: on a laptop with two GPUs the default one must be the fast one."""
    import numpy as np
    import onnxruntime as ort

    x = np.random.default_rng(0).random((1, in_ch, length), dtype=np.float32)
    ref = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]).run(
        None, {"features": x})[0]

    def timed(providers: list) -> dict:
        try:
            # no fallback: onnxruntime would make a processor session where DirectML fails (an
            # adapter number that does not exist) and time that
            sess = ort.InferenceSession(str(model), providers=providers, enable_fallback=0)
            used = sess.get_providers()
            if not used or used[0] != "DmlExecutionProvider":
                return {"error": f"no DirectML session (it would run on "
                                 f"{used[0] if used else 'nothing'})", "session_providers": used}
            times = []
            for _ in range(3):
                t = time.perf_counter()
                got = sess.run(None, {"features": x})[0]
                times.append(time.perf_counter() - t)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
        return {"first_s": round(times[0], 4), "best_s": round(min(times[1:]), 4),
                "max_abs_diff_vs_cpu": round(float(np.abs(got - ref).max()), 5),
                "session_providers": sess.get_providers()}

    out = {"model": model.name,
           "default": timed(["DmlExecutionProvider", "CPUExecutionProvider"]),
           "by_device_id": []}
    for dev in range(4):  # up to the first number that has no adapter
        rec = timed([("DmlExecutionProvider", {"device_id": str(dev)}), "CPUExecutionProvider"])
        rec["device_id"] = dev
        out["by_device_id"].append(rec)
        if "error" in rec:
            break
    timed_ok = [r for r in out["by_device_id"] if "best_s" in r]
    if timed_ok and "best_s" in out["default"]:
        fastest = min(timed_ok, key=lambda r: r["best_s"])
        if out["default"]["best_s"] > 1.5 * fastest["best_s"]:
            out["note"] = (f"DirectML's default adapter is slower than device_id "
                           f"{fastest['device_id']} ({out['default']['best_s']} s against "
                           f"{fastest['best_s']} s); the benchmark uses the default one")
    return out


def cpu_name() -> str:
    if WINDOWS:
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor()


def gpu_names() -> list[str]:
    """The GPUs as nvidia-smi and (on Windows) the display adapters list them."""
    names: list[str] = []
    opts = dict(capture_output=True, text=True, errors="replace", timeout=60)
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"], **opts).stdout
            names += [f"nvidia-smi: {x.strip()}" for x in out.splitlines() if x.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    if WINDOWS:
        ps = ("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' (driver ' + "
              "$_.DriverVersion + ')' }")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], **opts).stdout
            names += [f"adapter: {x.strip()}" for x in out.splitlines() if x.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    return names


def has_dist(name: str) -> bool:
    from importlib import metadata

    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def probe(out_path: Path, threads: int) -> int:
    """The checks, run by .venv-bench's Python; their record goes to ``out_path``."""
    sys.path.insert(0, str(BENCH))
    from tools import common  # the standard library only: the thread caps come before numpy

    os.environ.update(common.thread_env(threads))
    import platform

    import psutil

    rec: dict = {"python": platform.python_version(), "platform": platform.platform(),
                 "cpu": cpu_name(), "cores_physical": psutil.cpu_count(logical=False),
                 "cores_logical": psutil.cpu_count(logical=True),
                 "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "gpus": gpu_names()}
    import _paths  # noqa: F401  (<repo>/src and <repo>/scripts on sys.path)

    import lcmsdeconv

    src = Path(lcmsdeconv.__file__).resolve()
    rec["lcmsdeconv"] = {"path": str(src.parent), "from_package_src": (ROOT / "src") in src.parents}
    # both load DLLs of the Visual C++ runtime; the pipeline goes on without numba (slowly)
    for name in ("numba", "onnxruntime"):
        try:
            rec[name] = {"version": importlib.import_module(name).__version__}
        except Exception as exc:
            rec[name] = {"error": f"{type(exc).__name__}: {exc}"[:800]}
    if "error" in rec["onnxruntime"]:
        out_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        return 0
    import onnxruntime as ort

    rec["onnxruntime"].update(
        package=next((d for d in (ORT_WINDOWS[0], ORT_OTHER[0], "onnxruntime-gpu")
                      if has_dist(d)), None),
        available_providers=ort.get_available_providers())
    common.cap_threads(threads)
    models = check_models()
    rec["inference_checks"] = [inference_check(m, threads) for m in models]
    if models and "DmlExecutionProvider" in rec["onnxruntime"]["available_providers"]:
        info = json.loads(models[0].with_suffix(".json").read_text(encoding="utf-8"))
        rec["dml_adapters"] = dml_adapters(models[0], int(info.get("in_ch", 7)))
    out_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0


def rel(path: Path) -> str:
    try:
        return Path(os.path.abspath(path)).relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def working_providers(checks: list[dict], avail: list[str]) -> list[str]:
    """The providers that read every network of the checks (as the processor does, for a GPU
    provider); without networks to check, those onnxruntime lists."""
    order = [*GPU_PROVIDERS, "CPUExecutionProvider"]
    if not checks:
        return [p for p in order if p in avail]
    return [p for p in order if all(c["providers"].get(p, {}).get("ok") for c in checks)]


def check(threads: int) -> int:
    """Run the checks in .venv-bench and write bench/machine.json."""
    say("Checking: lcmsdeconv, onnxruntime's providers, a short inference of each network on "
        "each provider ...")
    tmp = BENCH / "machine.json.tmp"
    tmp.unlink(missing_ok=True)
    rc, tail = run([VPY, Path(__file__).resolve(), "--probe", tmp, "--threads", threads],
                   timeout=1800)
    try:
        rec = json.loads(tmp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        say(f"The check failed (exit code {rc}); the full output is in {LOG}.")
        if "DLL load failed" in tail:
            vc_runtime_advice(vc_runtime_missing())
        return 1
    finally:
        tmp.unlink(missing_ok=True)
    lcm = rec.get("lcmsdeconv", {})
    if not lcm.get("from_package_src"):
        say(f"Warning: lcmsdeconv was imported from {lcm.get('path')}, not from this package's "
            "src folder: another installation is in the way.")
    ortr, nb = rec.get("onnxruntime", {}), rec.get("numba", {})
    dll = any("DLL load failed" in str(x.get("error")) for x in (ortr, nb))
    if "error" in nb:
        say(f"Warning: numba does not load ({nb['error']}): the synthetic envelopes would be "
            "computed without it, far more slowly.")
    if "error" in ortr:
        say(f"onnxruntime does not load: {ortr['error']}")
    missing = vc_runtime_missing()
    if ("error" in ortr or "error" in nb) and (dll or missing):
        vc_runtime_advice(missing)
    if "error" in ortr:
        return 1
    avail = ortr.get("available_providers", [])
    say(f"onnxruntime {ortr.get('version')} ({ortr.get('package')}); providers: "
        f"{', '.join(avail)}")
    checks = rec.get("inference_checks") or []
    if not checks:  # nothing to check with: trust what onnxruntime lists
        say("Inference check skipped: no network with its .json in bench/models.")
    for chk in checks:
        say(f"Inference check with {chk['model']} ({chk.get('view')} view, "
            f"{chk.get('bins', '?')} bins, {chk.get('ion_bins', '?')} of them ions):")
        for prov, p in chk["providers"].items():
            if not p.get("ok"):
                say(f"  {prov}: does not work here: {p.get('error')}")
                continue
            extra = (f"; top charge as on the processor on {p['top1_agreement']:.2%} of the ion "
                     f"bins" if "top1_agreement" in p else f" ({chk['threads']} threads)")
            say(f"  {prov}: first read {p['first_s']} s (with warm-up), then {p['best_s']} s"
                f"{extra}")
    working = working_providers(checks, avail)
    note = rec.get("dml_adapters", {}).get("note")
    if note:
        say(f"Note: {note}.")
    chosen = next((p for p in GPU_PROVIDERS if p in working), "CPUExecutionProvider")
    rec.update(written=time.strftime("%Y-%m-%d %H:%M"), python_executable=rel(VPY),
               base_python=venv_base(), install=read_record().get("source"),
               providers_working=working, chosen_provider=chosen,
               providers_option=GPU_PROVIDERS.get(chosen, "cpu"))
    MACHINE.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    say(f"{rec.get('cores_physical')} cores ({rec.get('cores_logical')} logical), "
        f"{rec.get('ram_gb')} GB RAM; the benchmark will use {chosen} "
        f"(--providers {rec['providers_option']}). Written: {rel(MACHINE)}")
    if WINDOWS and chosen == "CPUExecutionProvider":
        say("DirectML does not work here for every network, so the networks run on the "
            "processor (slower). Updating the graphics driver usually makes DirectML work.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", help="create the environment with this Python")
    ap.add_argument("--recreate", action="store_true",
                    help="delete .venv-bench and create it again")
    ap.add_argument("--source", choices=("auto", "wheels", "pypi"), default="auto",
                    help="where to install from (auto: bench/wheels when it holds wheels for "
                         "this Python, else PyPI)")
    ap.add_argument("--skip-check", action="store_true",
                    help="install only: no inference check, no machine.json")
    ap.add_argument("--threads", type=int, default=CHECK_THREADS,
                    help=f"threads of the inference check on the processor ({CHECK_THREADS})")
    ap.add_argument("--probe", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # a character the console cannot show: '?'
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if args.probe:
        return probe(args.probe, max(1, args.threads))
    say(f"=== setup_bench {time.strftime('%Y-%m-%d %H:%M')} in {ROOT}")
    env = make_venv(args.python, args.recreate, args.source)
    if env is None:
        return 1
    missing = vc_runtime_missing()
    if missing:  # said before the install, which works without it; the check needs it
        vc_runtime_advice(missing)
        say("Setup goes on with the install; the check after it needs the runtime.")
    if not install(env, args.source):
        return 1
    if args.skip_check:
        say("Installed; the check was skipped (--skip-check).")
        return 0
    rc = check(max(1, args.threads))
    if rc == 0:
        say("")
        say("Setup finished. Next:")
        if WINDOWS:
            say("    bench\\run_bench.bat --smoke     (a few minutes: everything end to end)")
            say("    bench\\run_bench.bat             (the whole benchmark)")
        else:
            say("    .venv-bench/bin/python bench/run_bench.py --smoke")
            say("    .venv-bench/bin/python bench/run_bench.py")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
