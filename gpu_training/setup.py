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
5. measures the machine within the limits (gpu_training/probe.py): GPU memory of a training
   step under the GPU cap, memory per data worker, spectra per second, and the number of
   workers that keeps everything within the RAM budget (gpu_training/machine.json);
6. runs a two-minute training on the GPU to check everything end to end.

Output is also written to gpu_training/setup.log.
"""

from __future__ import annotations

import argparse
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
#: PyTorch CUDA builds, newest first, with the CUDA version the driver must support
TORCH_BUILDS = [("cu132", (13, 2)), ("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6))]
TORCH_INDEX = "https://download.pytorch.org/whl/{}"


def say(msg: str = "") -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


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


def driver() -> dict:
    """The NVIDIA driver as nvidia-smi reports it: GPU, memory, driver version, CUDA version."""
    smi = shutil.which("nvidia-smi")
    if smi is None and WINDOWS:
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(cand) if cand.exists() else None
    if smi is None:
        return {}
    try:
        head = subprocess.run([smi], capture_output=True, text=True, timeout=60).stdout
        query = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True,
                               timeout=60).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return {}
    info = {}
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", head)
    if m:
        info["cuda"] = (int(m.group(1)), int(m.group(2)))
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
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    say(f"=== setup {time.strftime('%Y-%m-%d %H:%M')} in {ROOT}")

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
    cuda = drv.get("cuda", (0, 0))
    say(f"GPU {drv.get('gpu', '?')}, {drv.get('memory_mb', 0) / 1024:.1f} GB; driver "
        f"{drv.get('driver', '?')}, supporting CUDA up to {cuda[0]}.{cuda[1]}")
    builds = [b for b, need in TORCH_BUILDS if need <= cuda]
    if args.torch_build:
        builds = [args.torch_build]
    if not builds:
        say("This driver is too old for current PyTorch (CUDA 12.6 or newer is needed): update "
            "the NVIDIA driver, then run this again.")
        return 1
    if builds[0] != "cu132" and not args.torch_build:
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
        for build in builds:
            say(f"Installing PyTorch for CUDA ({build}) from download.pytorch.org (about "
                f"3 GB) ...")
            tail: list = []
            rc = run([VPY, "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch",
                      "--index-url", TORCH_INDEX.format(build)], check=False, tail=tail)
            if rc != 0:
                text = "".join(tail)
                if ("No matching distribution found for torch" in text
                        or "Could not find a version that satisfies the requirement torch" in text):
                    say(f"There is no {build} build for this Python; trying the next one.")
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

    # 6. a short training run on the GPU
    if not args.skip_smoke:
        say("A two-minute training run on the GPU ...")
        workers = []
        machine = ROOT / "gpu_training" / "machine.json"
        if machine.exists():
            import json

            workers = ["--workers", str(json.loads(machine.read_text(encoding="utf-8"))["workers"])]
        run([VPY, "scripts/train_charge_net.py", "--smoke", "--max-ram-gb", str(args.ram_gb),
             "--max-gpu-memory-gb", str(args.gpu_memory_gb), *workers])

    say("")
    say("Setup finished. Start the training with:")
    say("    gpu_training\\train_all.bat" if WINDOWS else "    bash gpu_training/train_all.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
