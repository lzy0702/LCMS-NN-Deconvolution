"""The environment record: the processor and every logical processor's kind, the operating
system and (Windows) its power scheme and power mode, the memory, the GPU, its driver and
PyTorch, nvidia-smi, the Python packages, and setup's record of this computer
(gpu_training/machine.json). No host name, user name or file of the user is read."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

from . import cpusets, smi
from .common import GPU_TRAINING, ROOT, read_json

WINDOWS = sys.platform == "win32"
GUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
#: the power schemes Windows ships
SCHEMES = {"381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
           "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
           "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
           "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate Performance"}
#: the power mode overlays of Settings (Windows 11: Power mode), as commonly documented
OVERLAYS = {"961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
            "00000000-0000-0000-0000-000000000000": "Balanced",
            "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
            "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance"}
#: environment variables that change how training runs, recorded as set
ENV_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
            "CUDA_MODULE_LOADING", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
            "NUMBA_CACHE_DIR", "LCMSDECONV_MP_CONTEXT", "LCMSDECONV_SETUP_RC",
            "PROCESSOR_IDENTIFIER", "NUMBER_OF_PROCESSORS")


def decode_console(data: bytes) -> str:
    """Console output of a Windows tool: UTF-8 if it is, else the console's (OEM) code page,
    which localised Windows uses (936 on Chinese Windows), else the ANSI one."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    pages = []
    if WINDOWS:
        try:
            import ctypes

            pages.append(f"cp{ctypes.WinDLL('kernel32').GetOEMCP()}")
        except Exception:  # pragma: no cover - the ANSI page below is tried all the same
            pass
        pages.append("mbcs")
    for page in pages:
        try:
            return data.decode(page)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def run_tool(args: list[str], timeout: float = 30.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, decode_console(proc.stdout + proc.stderr)


def parse_power_scheme(text: str) -> dict:
    """``powercfg /getactivescheme`` in any language: the scheme's GUID and the name in brackets
    after it (ASCII or full-width), with Windows' English name for its own schemes."""
    m = re.search(rf"({GUID})\s*[(（]([^)）]*)[)）]", text)
    if m is None:
        g = re.search(GUID, text)
        if g is None:
            return {}
        guid, name = g.group(0).lower(), None
    else:
        guid, name = m.group(1).lower(), m.group(2).strip()
    return {"guid": guid, "name": name, "known_as": SCHEMES.get(guid)}


def _registry(root_name: str, key: str, names) -> dict:
    """Values of a registry key (Windows), those that exist."""
    import winreg

    root = getattr(winreg, root_name)
    out = {}
    try:
        with winreg.OpenKey(root, key) as k:
            for name in names:
                try:
                    out[name] = winreg.QueryValueEx(k, name)[0]
                except OSError:
                    continue
    except OSError:
        pass
    return out


def cpu_name() -> str:
    if WINDOWS:
        try:
            got = _registry("HKEY_LOCAL_MACHINE",
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                            ("ProcessorNameString",))
            if got.get("ProcessorNameString"):
                return str(got["ProcessorNameString"]).strip()
        except Exception:  # the generic name below
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def windows_info() -> dict:
    """The Windows version (11 from build 22000 on), the active power scheme, the power mode
    overlay, hardware-accelerated GPU scheduling, and the processor's power settings (raw)."""
    out: dict = {"platform": platform.platform(), "version": platform.version()}
    try:
        v = sys.getwindowsversion()
        out.update(build=v.build, major=v.major, minor=v.minor, windows_11=v.build >= 22000)
    except AttributeError:
        pass
    out["release"] = _registry("HKEY_LOCAL_MACHINE",
                               r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                               ("ProductName", "DisplayVersion", "EditionID", "CurrentBuild",
                                "UBR"))
    rc, text = run_tool(["powercfg", "/getactivescheme"])
    out["power_scheme"] = {**parse_power_scheme(text), "exit_code": rc, "raw": text.strip()[:300]}
    reg = _registry("HKEY_LOCAL_MACHINE",
                    r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes",
                    ("ActivePowerScheme", "ActiveOverlayAcPowerScheme",
                     "ActiveOverlayDcPowerScheme"))
    overlay = {k: {"guid": str(v).lower(), "known_as": OVERLAYS.get(str(v).lower())}
               for k, v in reg.items() if k.startswith("ActiveOverlay")}
    try:
        overlay["effective"] = _effective_overlay()
    except Exception as exc:  # undocumented function: the registry above is the record
        overlay["effective_error"] = f"{type(exc).__name__}: {exc}"[:200]
    out["power_mode"] = overlay
    hags = _registry("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
                     ("HwSchMode",))
    if "HwSchMode" in hags:
        out["gpu_scheduling"] = {"HwSchMode": hags["HwSchMode"],
                                 "hardware_accelerated": hags["HwSchMode"] == 2}
    rc, text = run_tool(["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR"], 60.0)
    out["processor_power_settings"] = (text[:30000] if rc == 0 else
                                       f"exit code {rc}: {text[:300]}")
    return out


def _effective_overlay() -> dict:
    """PowerGetEffectiveOverlayScheme (powrprof.dll, Windows 10 1709 on): the power mode in
    force (on battery or on mains, whichever applies)."""
    import ctypes

    class Guid(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                    ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    fn = ctypes.WinDLL("powrprof").PowerGetEffectiveOverlayScheme
    fn.argtypes = [ctypes.POINTER(Guid)]
    fn.restype = ctypes.c_uint32
    g = Guid()
    rc = fn(ctypes.byref(g))
    if rc != 0:
        return {"error": int(rc)}
    d4 = bytes(g.Data4)
    guid = (f"{g.Data1:08x}-{g.Data2:04x}-{g.Data3:04x}-{d4[:2].hex()}-{d4[2:].hex()}")
    return {"guid": guid, "known_as": OVERLAYS.get(guid)}


def linux_info() -> dict:
    out: dict = {"platform": platform.platform()}
    for key, path in (("platform_profile", "/sys/firmware/acpi/platform_profile"),
                      ("governor", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
                      ("energy_performance_preference",
                       "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
                      ("no_turbo", "/sys/devices/system/cpu/intel_pstate/no_turbo")):
        try:
            with open(path, encoding="utf-8") as fh:
                out[key] = fh.read().strip()
        except OSError:
            continue
    return out


def memory_info() -> dict:
    try:
        import psutil

        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return {"total_gb": round(vm.total / 2**30, 1), "available_gb": round(vm.available / 2**30, 1),
                "swap_or_pagefile_gb": round(sw.total / 2**30, 1)}
    except Exception as exc:  # psutil missing
        return {"error": f"{type(exc).__name__}: {exc}"}


def torch_info() -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"installed": False, "error": str(exc)}
    out: dict = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                 "cuda": torch.cuda.is_available(), "threads": torch.get_num_threads()}
    try:
        out["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # a build without cuDNN
        out["cudnn"] = None
    if out["cuda"]:
        props = torch.cuda.get_device_properties(0)
        out.update(device_count=torch.cuda.device_count(), gpu=props.name,
                   gpu_total_gb=round(props.total_memory / 2**30, 2),
                   capability=f"{props.major}.{props.minor}",
                   multiprocessors=props.multi_processor_count,
                   allocator_backend=getattr(torch.cuda, "get_allocator_backend", lambda: None)(),
                   bf16=torch.cuda.is_bf16_supported())
        try:
            out["driver_cuda"] = torch._C._cuda_getDriverVersion()
        except Exception:  # private API: only for the record
            pass
    return out


def packages() -> dict:
    from importlib import metadata

    out = {}
    for name in ("numpy", "numba", "psutil", "onnx", "onnxruntime", "scipy", "llvmlite"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def setup_record() -> dict:
    """What setup recorded about this computer: machine.json's choices and measurements, and
    setup_done.json (when, with which PyTorch)."""
    machine = read_json(GPU_TRAINING / "machine.json") or {}
    keep = ("recipe", "workers", "ram_budget_gb", "physical_ram_gb", "gpu_memory_gb", "gpu_batch",
            "head_chunk", "head_chunks", "worker_gb", "main_gb", "cores", "processor_workers",
            "gpu", "gpu_views", "gpu_speedups", "speedup_timings", "speedup_checks",
            "expected_hours", "pace", "measured", "unchunked_failed", "data_loader",
            "data_loader_test")
    # the data workers' measured speed under the recipe's name (v5_spectra_per_second, ...)
    rates = tuple(k for k in machine if k.endswith(("_spectra_per_second", "_workers")))
    return {"machine_json": {k: machine[k] for k in keep + rates if k in machine},
            "setup_done": read_json(GPU_TRAINING / "setup_done.json"),
            "machine_json_found": bool(machine)}


def environment(ctx=None) -> dict:
    """The environment record (experiment ``env``)."""
    procs = cpusets.logical_processors()
    out: dict = {
        "cpu": {"name": cpu_name(), "logical": os.cpu_count(), **procs,
                "affinity": cpusets.current_affinity(), "priority": cpusets.priority_now()},
        "os": windows_info() if WINDOWS else linux_info(),
        "memory": memory_info(),
        "python": {"version": platform.python_version(), "implementation":
                   platform.python_implementation(), "bits": 64 if sys.maxsize > 2**32 else 32},
        "packages": packages(),
        "torch": torch_info(),
        "nvidia_smi": smi.describe(),
        "env_vars": {k: os.environ[k] for k in ENV_VARS if k in os.environ},
        "setup": setup_record(),
        "package_root_has_spaces": " " in str(ROOT),
    }
    try:
        import psutil

        out["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq is not None:
            out["cpu"]["freq_mhz"] = {"current": freq.current, "max": freq.max}
    except Exception:  # psutil missing: os.cpu_count above
        pass
    return out
