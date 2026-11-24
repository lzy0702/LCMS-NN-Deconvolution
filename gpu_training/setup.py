"""One-time setup of the training environment on this computer.

    gpu_training\\setup_windows.bat          (Windows: finds Python, then runs this)
    python3 gpu_training/setup.py           (Linux)

Every step is skipped when it is already done, so running it again is safe:

1. checks Python (3.11 to 3.14) and the NVIDIA driver (nvidia-smi: the GPU, its memory and the
   newest CUDA version the driver supports; the CUDA toolkit itself is not needed);
2. creates the environment .venv in the package's top folder;
3. installs PyTorch built for CUDA from download.pytorch.org - the newest build the driver
   supports among CUDA 13.2, 13.0, 12.8 and 12.6 - and checks that it computes on the GPU,
   falling back to the next build if it does not (PyPI only has processor builds for Windows);
4. installs this package with what training needs (numba, onnx, psutil);
5. measures the machine within the limits (gpu_training/probe.py): GPU memory and time of a
   training step of both views under the GPU cap, memory per data worker, spectra per second,
   the number of workers that keeps everything within the RAM budget, and the GPU speedups
   that make a training step faster on this GPU with the same loss and gradients
   (gpu_training/machine.json);
6. checks everything end to end in a few minutes (train_all.py --smoke): a few training steps
   of both views on the GPU with the speedups chosen, their export to ONNX, and the predictor
   reading whole spectra.

A setup that finishes records itself in gpu_training/setup_done.json. With --if-needed (what
setup_and_train.bat uses) the whole setup is skipped while that record still holds: the same
package files (SETUP_INPUTS and every Python file of src/lcmsdeconv), the environment and
machine.json present, and PyTorch computing on the GPU. A new package unzipped over the folder
of an earlier one keeps its .venv: PyTorch is not downloaded again, and only the measurement and
the check run again.

Output is also written to gpu_training/setup.log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "gpu_training" / "setup.log"
WINDOWS = sys.platform == "win32"
VENV = ROOT / ".venv"
VPY = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")
MACHINE = ROOT / "gpu_training" / "machine.json"
#: the record of the last setup that finished (read by --if-needed)
DONE = ROOT / "gpu_training" / "setup_done.json"
#: files whose change makes --if-needed run the setup again: the package's dependencies, the
#: setup, the measurement and the end-to-end check themselves, and the training code they
#: measure and check (the environment is kept: only the measurement and the check run again)
SETUP_INPUTS = ("pyproject.toml", "gpu_training/setup.py", "gpu_training/probe.py",
                "gpu_training/train_all.py", "gpu_training/check_model.py",
                "scripts/train_charge_net.py", "src/lcmsdeconv/nn/model.py",
                "src/lcmsdeconv/nn/train.py", "src/lcmsdeconv/nn/features.py",
                "src/lcmsdeconv/nn/infer.py", "src/lcmsdeconv/nn/export.py",
                "src/lcmsdeconv/synth/frames.py")
#: folders whose Python files count as well, all of them: the whole package (the network, its
#: training, the synthetic spectra and the chemistry, profiles and comb they are built from)
SETUP_TREES = ("src/lcmsdeconv",)
#: PyTorch CUDA builds, newest first, with the CUDA version the driver must support
TORCH_BUILDS = [("cu132", (13, 2)), ("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6))]
TORCH_INDEX = "https://download.pytorch.org/whl/{}"
#: pip output that means download.pytorch.org could not be reached (pip then also says it found
#: no matching distribution, as it does when an index has no build for this Python)
NETWORK_TROUBLE = ("Tunnel connection failed", "ProxyError", "Max retries exceeded",
                   "Failed to establish a new connection", "ConnectionError", "Connection reset",
                   "Read timed out", "SSLError", "certificate verify failed",
                   "Temporary failure in name resolution", "getaddrinfo failed",
                   "Retrying (Retry(")


def say(msg: str = "") -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def fingerprint() -> dict:
    """The first 16 hex digits of the sha256 of each file in SETUP_INPUTS ('' if missing), and of
    the names and contents of the Python files in each folder of SETUP_TREES."""
    out = {}
    for rel in SETUP_INPUTS:
        path = ROOT / rel
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""
    for rel in SETUP_TREES:
        digest = hashlib.sha256()
        for path in sorted((ROOT / rel).rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
        out[rel + "/"] = digest.hexdigest()[:16]
    return out


def setup_is_current() -> tuple[bool, str]:
    """Whether the last finished setup still holds, and why (or why not)."""
    if not VPY.exists():
        return False, "the environment .venv does not exist yet"
    try:
        done = json.loads(DONE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, "no setup has finished in this folder yet"
    except (OSError, ValueError):
        return False, f"{DONE.name} cannot be read"
    if not (done.get("probe") and done.get("smoke")):
        return False, "the last setup skipped the measurement or the test run"
    if done.get("fingerprint") != fingerprint():
        return False, "the package has changed since the last setup"
    if not MACHINE.exists():
        return False, "gpu_training/machine.json is missing"
    current, reason = torch_on_gpu()
    if not current:
        return False, "PyTorch no longer computes on the GPU: " + reason.strip()[-300:]
    return True, f"finished {done.get('finished', '?')}, {current}"


def run(cmd: list, check: bool = True, quiet: bool = False, tail: list | None = None) -> int:
    """Run ``cmd``, showing and logging its output; returns its exit code. ``tail``, if given,
    receives the last lines of the output."""
    say("> " + " ".join(str(c) for c in cmd))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env)
    with open(LOG, "a", encoding="utf-8") as fh:
        for line in proc.stdout:
            fh.write(line)
            if tail is not None:
                tail.append(line)
                del tail[:-40]
            if not quiet:
                print(line, end="", flush=True)
    rc = proc.wait()
    if check and rc != 0:
        say(f"That step failed (exit code {rc}); the full output is in {LOG}.")
        raise SystemExit(rc)
    return rc


def cuda_from_driver_api() -> tuple[int, int] | None:
    """The newest CUDA version the installed driver supports, asked of the driver itself
    (cuDriverGetVersion in nvcuda.dll or libcuda.so; what nvidia-smi prints as "CUDA Version")."""
    import ctypes

    for name in (["nvcuda.dll"] if WINDOWS else ["libcuda.so.1", "libcuda.so"]):
        try:
            lib = ctypes.WinDLL(name) if WINDOWS else ctypes.CDLL(name)
            version = ctypes.c_int(0)
            if lib.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value > 0:
                return version.value // 1000, version.value % 1000 // 10
        except (OSError, AttributeError):
            continue
    return None


def cuda_from_text(text: str) -> tuple[int, int] | None:
    """The CUDA version in nvidia-smi's output, in any of the ways it has been written:
    "CUDA Version: 13.0" in the table header, "CUDA Version : 13.0" in nvidia-smi -q, with or
    without a word such as "Driver" between "CUDA" and "Version"."""
    m = re.search(r"CUDA(?:[ \t]+[A-Za-z]+)?[ \t]+Version[ \t]*:[ \t]*(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def builds_for(cuda: tuple[int, int] | None) -> list[str]:
    """The PyTorch CUDA builds to try, newest first: those the driver supports, or all of them
    when its CUDA version could not be read (each is then checked on the GPU)."""
    if cuda is None:
        return [b for b, _need in TORCH_BUILDS]
    return [b for b, need in TORCH_BUILDS if need <= cuda]


def driver() -> dict:
    """The NVIDIA driver as nvidia-smi reports it: GPU, memory, driver version, CUDA version."""
    smi = shutil.which("nvidia-smi")
    if smi is None and WINDOWS:
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(cand) if cand.exists() else None
    if smi is None:
        return {}
    text = dict(capture_output=True, text=True, errors="replace", timeout=60)
    try:
        head = subprocess.run([smi], **text).stdout
        query = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader,nounits"], **text).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return {}
    info = {}
    cuda = cuda_from_driver_api() or cuda_from_text(head)
    if cuda is None:
        try:
            cuda = cuda_from_text(subprocess.run([smi, "-q"], **text).stdout)
        except (OSError, subprocess.SubprocessError):
            pass
    if cuda is not None:
        info["cuda"] = cuda
    if query:
        parts = [p.strip() for p in query[0].split(",")]
        if len(parts) >= 3:
            info.update(gpu=parts[0], memory_mb=float(parts[1]), driver=parts[2])
    return info


#: what a PyTorch build that does not match this GPU or driver says; anything else (a missing
#: DLL, the paging file, a virus scanner) is not fixed by another 3 GB download
CUDA_MISMATCH = ("CUDA is not available", "no kernel image", "driver version is insufficient",
                 "not compiled with CUDA", "CUDA driver", "cudaGetDeviceCount", "sm_",
                 "No module named 'torch'")


def torch_on_gpu() -> tuple[str | None, str]:
    """('version | CUDA version | GPU', '') if the environment's PyTorch computes on the GPU,
    else (None, the reason); the reason is also written to the log."""
    if not VPY.exists():
        return None, "no environment yet"
    code = ("import torch, sys\n"
            "if not torch.cuda.is_available(): sys.exit('CUDA is not available to this PyTorch "
            "' + torch.__version__)\n"
            "x = torch.randn(2048, 2048, device='cuda'); y = (x @ x).sum().item()\n"
            "torch.cuda.synchronize()\n"
            "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), sep=' | ')")
    try:
        out = subprocess.run([str(VPY), "-c", code], capture_output=True, text=True, timeout=900,
                             cwd=ROOT)
    except subprocess.TimeoutExpired:
        reason = "the GPU check did not finish within 15 minutes"
    except OSError as exc:
        reason = f"the GPU check could not run: {exc}"
    else:
        if out.returncode == 0:
            return out.stdout.strip(), ""
        reason = (out.stderr or out.stdout).strip()[-1500:] or f"exit code {out.returncode}"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"GPU check failed: {reason}\n")
    return None, reason


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ram-gb", type=float, default=16.0,
                    help="RAM the training may use, training process and data workers together")
    ap.add_argument("--gpu-memory-gb", type=float, default=14.0,
                    help="GPU memory the training may use (the rest is left to Windows)")
    ap.add_argument("--torch-build", default=None, choices=[b for b, _v in TORCH_BUILDS],
                    help="install this PyTorch CUDA build instead of choosing one")
    ap.add_argument("--reinstall-torch", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--if-needed", action="store_true",
                    help="skip everything while the last finished setup still holds "
                         "(setup_and_train.bat uses this)")
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    say(f"=== setup {time.strftime('%Y-%m-%d %H:%M')} in {ROOT}")
    if args.if_needed:
        holds, why = setup_is_current()
        if holds:
            say(f"Setup is already done ({why}): skipping it.")
            return 0
        say(f"Setting up, because {why}.")
    DONE.unlink(missing_ok=True)  # a setup that stops part-way leaves no record behind

    # 1. Python and the driver
    v = sys.version_info
    say(f"Python {v.major}.{v.minor}.{v.micro} at {sys.executable}")
    if not (3, 11) <= (v.major, v.minor) <= (3, 14):
        say("Python 3.11 to 3.14 is needed (3.12 recommended): PyTorch 2.15 and numpy no longer "
            "support older versions on Windows.")
        return 1
    drv = driver()
    if not drv:
        say("nvidia-smi was not found: install or update the NVIDIA driver "
            "(https://www.nvidia.com/Download/index.aspx), then run this again.")
        return 1
    cuda = drv.get("cuda")
    gpu = (f"GPU {drv.get('gpu', '?')}, {drv.get('memory_mb', 0) / 1024:.1f} GB; driver "
           f"{drv.get('driver', '?')}")
    if cuda is None:
        say(f"{gpu}; the CUDA version it supports could not be read, so the PyTorch builds are "
            "tried newest first and each is checked on the GPU")
    else:
        say(f"{gpu}, supporting CUDA up to {cuda[0]}.{cuda[1]}")
    builds = builds_for(cuda)
    if args.torch_build:
        builds = [args.torch_build]
    if not builds:
        say("This driver is too old for current PyTorch (CUDA 12.6 or newer is needed): update "
            "the NVIDIA driver, then run this again.")
        return 1
    if cuda is not None and builds[0] != "cu132" and not args.torch_build:
        say("Note: a driver supporting CUDA 13.2 lets PyTorch use its main build (cu132); "
            f"with this one, {builds[0]} is used. Updating the regular NVIDIA driver is enough; "
            "the CUDA toolkit is not needed.")

    # 2. the environment
    if not VPY.exists():
        say(f"Creating the environment {VENV} ...")
        run([sys.executable, "-m", "venv", str(VENV)])
    run([VPY, "-m", "pip", "install", "--upgrade", "pip"], quiet=True)

    # 3. PyTorch for CUDA
    current, reason = (None, "") if args.reinstall_torch else torch_on_gpu()
    if current:
        say(f"PyTorch already computes on the GPU: {current}")
    else:
        missing = []  # builds the index has none of for this Python
        for build in builds:
            say(f"Installing PyTorch for CUDA ({build}) from download.pytorch.org (about "
                f"3 GB) ...")
            tail: list = []
            rc = run([VPY, "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch",
                      "--index-url", TORCH_INDEX.format(build)], check=False, tail=tail)
            if rc != 0:
                text = "".join(tail)
                if any(m in text for m in NETWORK_TROUBLE):
                    say("pip could not reach download.pytorch.org (see the messages above): a "
                        "proxy or firewall blocks it, or the connection is down. Check that "
                        "https://download.pytorch.org/whl/ opens in a browser on this computer, "
                        "then run this again.")
                    return 1
                if ("No matching distribution found for torch" in text
                        or "Could not find a version that satisfies the requirement torch" in text):
                    say(f"There is no {build} build for this Python; trying the next one.")
                    missing.append(build)
                    continue
                say("pip could not download or install PyTorch (see the error above). Usual "
                    "causes: no access to download.pytorch.org (a proxy or firewall), less than "
                    "about 10 GB free on the disk, or another program still using .venv (close "
                    f"other Python windows). The full output is in {LOG}.")
                return 1
            current, reason = torch_on_gpu()
            if current:
                say(f"PyTorch computes on the GPU: {current}")
                break
            say(f"The {build} build does not compute on the GPU here:\n    {reason[-600:]}")
            if not any(m in reason for m in CUDA_MISMATCH):
                say("That is not a mismatch between PyTorch and the driver, so another build would "
                    "not help. Fix the cause above (for example install the Microsoft Visual C++ "
                    "Redistributable, or let Windows manage the paging file size), then run this "
                    "again.")
                return 1
            say("Trying the next build.")
        if not current and len(missing) == len(builds):
            say(f"PyTorch has no CUDA build for Python {v.major}.{v.minor} on "
                f"download.pytorch.org ({', '.join(builds)}). Install Python 3.12 "
                "(winget install -e --id Python.Python.3.12), delete the .venv folder and run "
                "this again.")
            return 1
        if not current:
            say("No PyTorch CUDA build worked with this driver: update the NVIDIA driver "
                f"(nvidia-smi), then run this again. Details are in {LOG}.")
            return 1

    # 4. this package
    say("Installing this package for training (numba, onnx, psutil) ...")
    run([VPY, "-m", "pip", "install", "-e", ".[train,fast]"])
    current, reason = torch_on_gpu()  # a dependency must never have replaced the CUDA build
    if current is None:
        say(f"PyTorch no longer computes on the GPU after installing the package:\n    "
            f"{reason[-600:]}\nRun this again with --reinstall-torch.")
        return 1

    # 5. measure the machine within the limits
    if not args.skip_probe:
        run([VPY, "gpu_training/probe.py", "--ram-gb", str(args.ram_gb), "--gpu-memory-gb",
             str(args.gpu_memory_gb)])

    # 6. a short end-to-end check on the GPU: training steps, export, the predictor
    if not args.skip_smoke:
        say("A few minutes' check of both runs on the GPU: training steps, export to ONNX, "
            "whole spectra read by the predictor ...")
        run([VPY, "gpu_training/train_all.py", "--smoke", "--ram-gb", str(args.ram_gb),
             "--gpu-memory-gb", str(args.gpu_memory_gb)])

    DONE.write_text(json.dumps({
        "finished": time.strftime("%Y-%m-%d %H:%M"), "torch": current,
        "probe": not args.skip_probe, "smoke": not args.skip_smoke,
        "ram_gb": args.ram_gb, "gpu_memory_gb": args.gpu_memory_gb,
        "fingerprint": fingerprint()}, indent=1) + "\n", encoding="utf-8")
    say("")
    if args.if_needed:
        say("Setup finished.")
        return 0
    say("Setup finished. Start the training with:")
    say("    gpu_training\\train_all.bat" if WINDOWS else "    bash gpu_training/train_all.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
