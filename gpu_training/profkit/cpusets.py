"""Logical processors and their kind, and what a process may set about where and how it runs:
its affinity, its priority and, on Windows, power throttling (EcoQoS).

Windows names each logical processor's kind in GetSystemCpuSetInformation (kernel32): its
``EfficiencyClass``, higher for faster cores (1 for the performance cores and 0 for the
efficient ones of a hybrid Intel processor; 0 for every core of a processor with one kind).
Linux names them in /sys/devices/cpu_core/cpus and /sys/devices/cpu_atom/cpus on hybrid Intel
processors; elsewhere the cores' capacities or highest frequencies tell them apart.

The Windows constants used here, from the Windows SDK (processthreadsapi.h, winnt.h):

- SYSTEM_CPU_SET_INFORMATION: ``DWORD Size; CPU_SET_INFORMATION_TYPE Type;`` then, for
  ``Type == CpuSetInformation (0)``: ``DWORD Id; WORD Group; BYTE LogicalProcessorIndex,
  CoreIndex, LastLevelCacheIndex, NumaNodeIndex, EfficiencyClass, AllFlags (Parked 0x1,
  Allocated 0x2, AllocatedToTargetProcess 0x4, RealTime 0x8); union { DWORD Reserved; BYTE
  SchedulingClass; }; DWORD64 AllocationTag;`` 32 bytes in all. Entries follow one another,
  each ``Size`` bytes long (later versions of Windows may make them longer).
- SetProcessInformation(process, ProcessPowerThrottling = 4, PROCESS_POWER_THROTTLING_STATE
  {ULONG Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION (1); ULONG ControlMask; ULONG
  StateMask}, 12): ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED (0x1) and StateMask =
  0 opt the process out of EcoQoS (always full speed); StateMask = 0x1 opts it in; ControlMask
  = 0 leaves it to Windows again.
- SetPriorityClass: NORMAL_PRIORITY_CLASS 0x20, BELOW_NORMAL_PRIORITY_CLASS 0x4000 (what
  training's data workers run at: lcmsdeconv.synth.dataset.worker_init).
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

from .common import Unavailable

WINDOWS = sys.platform == "win32"

#: CPU_SET_INFORMATION_TYPE.CpuSetInformation
CPU_SET_INFORMATION = 0
#: an entry's header: Size, Type
ENTRY_HEADER = struct.Struct("<II")
#: a CPU set from offset 8: Id, Group, LogicalProcessorIndex, CoreIndex, LastLevelCacheIndex,
#: NumaNodeIndex, EfficiencyClass, AllFlags, Reserved (SchedulingClass is its low byte),
#: AllocationTag (8-aligned at offset 24)
CPU_SET = struct.Struct("<IHBBBBBBIQ")
CPU_SET_SIZE = ENTRY_HEADER.size + CPU_SET.size  # 32
CPU_SET_FLAGS = ((0x1, "parked"), (0x2, "allocated"), (0x4, "allocated_to_target_process"),
                 (0x8, "realtime"))
#: ERROR_INSUFFICIENT_BUFFER, the first call's answer
ERROR_INSUFFICIENT_BUFFER = 122

#: PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
PROCESS_POWER_THROTTLING = 4
#: PROCESS_POWER_THROTTLING_CURRENT_VERSION
POWER_THROTTLING_VERSION = 1
#: PROCESS_POWER_THROTTLING_EXECUTION_SPEED
EXECUTION_SPEED = 0x1
#: (ControlMask, StateMask) of each power throttling mode
THROTTLING_MODES = {"opt_out": (EXECUTION_SPEED, 0), "eco": (EXECUTION_SPEED, EXECUTION_SPEED),
                    "default": (0, 0)}
WINDOWS_CONSTANTS = {"ProcessPowerThrottling": PROCESS_POWER_THROTTLING,
                     "PROCESS_POWER_THROTTLING_CURRENT_VERSION": POWER_THROTTLING_VERSION,
                     "PROCESS_POWER_THROTTLING_EXECUTION_SPEED": EXECUTION_SPEED,
                     "opt_out": {"ControlMask": EXECUTION_SPEED, "StateMask": 0},
                     "NORMAL_PRIORITY_CLASS": 0x20, "BELOW_NORMAL_PRIORITY_CLASS": 0x4000}
PRIORITY_CLASSES = {"normal": 0x20, "below_normal": 0x4000}
PRIORITY_NAMES = {0x20: "normal", 0x4000: "below_normal", 0x40: "idle", 0x8000: "above_normal",
                  0x80: "high", 0x100: "realtime"}
#: how much "below normal" lowers a process's priority on Linux (worker_init's os.nice(5))
BELOW_NORMAL_NICE = 5
#: the nice value this process started with: "normal" on Linux, never raised above
BASE_NICE = os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else 0


# ------------------------------------------------------------------ Windows CPU sets
def parse_cpu_sets(buf: bytes) -> list[dict]:
    """The CPU sets in a buffer GetSystemCpuSetInformation filled: one dict per logical
    processor, in the buffer's order. Entries of another type are skipped, longer entries read
    as far as this layout goes; a buffer that does not divide into whole entries raises
    ValueError."""
    out = []
    off = 0
    while off < len(buf):
        if off + ENTRY_HEADER.size > len(buf):
            raise ValueError(f"{len(buf) - off} bytes left at offset {off}, too few for an "
                             f"entry header")
        size, kind = ENTRY_HEADER.unpack_from(buf, off)
        if size < ENTRY_HEADER.size or off + size > len(buf):
            raise ValueError(f"an entry of {size} bytes at offset {off} in a buffer of "
                             f"{len(buf)}")
        if kind == CPU_SET_INFORMATION:
            if size < CPU_SET_SIZE:
                raise ValueError(f"a CPU set entry of {size} bytes at offset {off}, fewer than "
                                 f"{CPU_SET_SIZE}")
            (cid, group, lp, core, llc, numa, eff, flags, reserved,
             tag) = CPU_SET.unpack_from(buf, off + ENTRY_HEADER.size)
            out.append({"id": cid, "group": group, "logical": lp, "core": core, "llc": llc,
                        "numa": numa, "efficiency_class": eff,
                        "flags": [name for bit, name in CPU_SET_FLAGS if flags & bit],
                        "scheduling_class": reserved & 0xFF, "allocation_tag": tag})
        off += size
    return out


def pack_cpu_set(cid: int, group: int, lp: int, core: int, eff: int, *, llc: int = 0,
                 numa: int = 0, flags: int = 0, scheduling: int = 0, tag: int = 0,
                 size: int = CPU_SET_SIZE, kind: int = CPU_SET_INFORMATION) -> bytes:
    """One entry as Windows writes it (tests, and the record of the layout read): ``size``
    bytes, the CPU set followed by zeros where ``size`` exceeds this layout's 32."""
    body = CPU_SET.pack(cid, group, lp, core, llc, numa, eff, flags, scheduling, tag)
    data = ENTRY_HEADER.pack(size, kind) + body
    return data[:size] if size < len(data) else data + bytes(size - len(data))


def _kernel32():
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=True)


def _abi():
    """The Windows types by their exact sizes (ULONG and DWORD are 32-bit on Windows; ctypes'
    wintypes maps them to c_ulong, which is 64-bit elsewhere, where the tests run these calls
    through stand-ins): (ULONG, BOOL, HANDLE)."""
    import ctypes

    return ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p


def _last_error() -> int:
    import ctypes

    getter = getattr(ctypes, "get_last_error", None)  # Windows only
    return int(getter()) if getter else 0


def windows_cpu_sets(kernel32=None) -> list[dict]:
    """Every logical processor's CPU set, from GetSystemCpuSetInformation: asked once for the
    size needed, then for the entries. ``kernel32`` stands in for the DLL (tests)."""
    import ctypes

    if kernel32 is None and not WINDOWS:
        raise Unavailable("GetSystemCpuSetInformation is a Windows function")
    ulong, boolean, handle = _abi()
    k32 = kernel32 or _kernel32()
    fn = k32.GetSystemCpuSetInformation
    fn.argtypes = [ctypes.c_void_p, ulong, ctypes.POINTER(ulong), handle, ulong]
    fn.restype = boolean
    needed = ulong(0)
    if fn(None, 0, ctypes.byref(needed), None, 0) or not needed.value:
        raise OSError(f"GetSystemCpuSetInformation asked for {needed.value} bytes (error "
                      f"{_last_error()})")
    for _attempt in range(3):  # a processor can come online between the two calls
        size = needed.value
        buf = ctypes.create_string_buffer(size)
        if fn(buf, size, ctypes.byref(needed), None, 0):
            return parse_cpu_sets(buf.raw[:needed.value])
        if _last_error() not in (ERROR_INSUFFICIENT_BUFFER, 0) or needed.value <= size:
            break
    raise OSError(f"GetSystemCpuSetInformation failed (error {_last_error()})")


# ------------------------------------------------------------------ Linux
def parse_cpulist(text: str) -> list[int]:
    """``"0-3,8,10-11"`` -> [0, 1, 2, 3, 8, 10, 11] (the kernel's CPU list format)."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _ranked(values: dict[int, float]) -> dict[int, int]:
    """Classes from per-processor numbers: 0 for the smallest distinct value, 1 for the next..."""
    levels = sorted(set(values.values()))
    return {cpu: levels.index(v) for cpu, v in values.items()}


def linux_processors(sysfs: str | Path = "/sys") -> dict:
    """Every online logical processor with its class: from cpu_core and cpu_atom (hybrid Intel:
    1 and 0), else from cpu_capacity, else from the highest frequency the core reaches (each
    distinct value a class, the fastest highest); all 0 where they do not differ."""
    sysfs = Path(sysfs)
    cpu_dir = sysfs / "devices" / "system" / "cpu"
    online = _read(cpu_dir / "online")
    cpus = parse_cpulist(online) if online else list(range(os.cpu_count() or 1))
    detail: dict[int, dict] = {c: {} for c in cpus}
    for c in cpus:
        base = cpu_dir / f"cpu{c}"
        for key, rel in (("max_khz", "cpufreq/cpuinfo_max_freq"),
                         ("base_khz", "cpufreq/base_frequency"), ("capacity", "cpu_capacity"),
                         ("core_id", "topology/core_id"),
                         ("package", "topology/physical_package_id")):
            text = _read(base / rel)
            if text is not None and text.lstrip("-").isdigit():
                detail[c][key] = int(text)
    core = _read(sysfs / "devices" / "cpu_core" / "cpus")
    atom = _read(sysfs / "devices" / "cpu_atom" / "cpus")
    classes: dict[int, int] = {}
    source = "none"
    if core or atom:
        source = "sysfs cpu_core/cpu_atom"
        classes.update({c: 1 for c in parse_cpulist(core or "")})
        classes.update({c: 0 for c in parse_cpulist(atom or "")})
    else:
        for key, name in (("capacity", "sysfs cpu_capacity"), ("max_khz", "sysfs cpufreq")):
            values = {c: d[key] for c, d in detail.items() if key in d}
            if len(values) == len(cpus) and len(set(values.values())) > 1:
                classes, source = _ranked(values), name
                break
    procs = [{"cpu": c, "efficiency_class": classes.get(c, 0), **detail[c]} for c in cpus]
    return {"source": source, "processors": procs}


# ------------------------------------------------------------------ either system
def kind_names(procs: list[dict]) -> None:
    """Name each processor's kind in place: "performance" for the highest class and
    "efficient" for the lowest on a processor with two or more, "uniform" otherwise."""
    classes = sorted({p.get("efficiency_class") for p in procs
                      if p.get("efficiency_class") is not None})
    for p in procs:
        e = p.get("efficiency_class")
        if len(classes) < 2 or e is None:
            p["kind"] = "uniform"
        elif e == classes[-1]:
            p["kind"] = "performance"
        elif e == classes[0]:
            p["kind"] = "efficient"
        else:
            p["kind"] = f"class {e}"


def logical_processors(kernel32=None, sysfs: str | Path = "/sys") -> dict:
    """``{"source", "processors": [{"cpu", "efficiency_class", "kind", ...}], "counts"}``:
    every logical processor of this computer, ``cpu`` the number affinity uses (None for one
    outside processor group 0 on Windows, which a process's affinity cannot name)."""
    if WINDOWS or kernel32 is not None:
        try:
            sets = windows_cpu_sets(kernel32)
        except (OSError, ValueError, AttributeError) as exc:
            procs = [{"cpu": c, "efficiency_class": None} for c in range(os.cpu_count() or 1)]
            out = {"source": "none", "error": f"{type(exc).__name__}: {exc}", "processors": procs}
        else:
            procs = [{"cpu": s["logical"] if s["group"] == 0 else None, **s} for s in sets]
            out = {"source": "GetSystemCpuSetInformation", "processors": procs}
    else:
        out = linux_processors(sysfs)
    kind_names(out["processors"])
    counts: dict[str, int] = {}
    for p in out["processors"]:
        counts[p["kind"]] = counts.get(p["kind"], 0) + 1
    out["counts"] = counts
    return out


def usable(procs: list[dict]) -> list[dict]:
    """The processors this process may run on (its affinity), in performance-first order."""
    allowed = current_affinity()
    out = [p for p in procs if p.get("cpu") is not None
           and (allowed is None or p["cpu"] in allowed)]
    return sorted(out, key=lambda p: (-(p.get("efficiency_class") or 0), p["cpu"]))


def current_affinity() -> list[int] | None:
    try:
        import psutil

        return sorted(psutil.Process().cpu_affinity())
    except Exception:  # psutil missing, or a platform without affinity (macOS)
        if hasattr(os, "sched_getaffinity"):
            return sorted(os.sched_getaffinity(0))
        return None


def set_affinity(cpus: list[int]) -> list[int] | None:
    """Pin this process (every thread of it) to ``cpus``; returns the affinity now in force."""
    try:
        import psutil

        psutil.Process().cpu_affinity(list(cpus))
    except ImportError:
        if not hasattr(os, "sched_setaffinity"):
            raise Unavailable("no way to set the affinity here (psutil is not installed)"
                              ) from None
        os.sched_setaffinity(0, set(cpus))
    return current_affinity()


# ------------------------------------------------------------------ priority and throttling
def _typed_process_calls(k32):
    import ctypes

    ulong, boolean, handle = _abi()
    k32.GetCurrentProcess.restype = handle
    k32.GetCurrentProcess.argtypes = []
    for name in ("SetProcessInformation", "GetProcessInformation"):
        fn = getattr(k32, name)
        fn.argtypes = [handle, ctypes.c_int, ctypes.c_void_p, ulong]
        fn.restype = boolean
    return k32


def throttling_struct(mode: str):
    """PROCESS_POWER_THROTTLING_STATE for ``mode`` (a key of THROTTLING_MODES)."""
    import ctypes

    ulong = _abi()[0]

    class State(ctypes.Structure):
        _fields_ = [("Version", ulong), ("ControlMask", ulong), ("StateMask", ulong)]

    control, state = THROTTLING_MODES[mode]
    return State(POWER_THROTTLING_VERSION, control, state)


def set_power_throttling(mode: str = "opt_out", kernel32=None) -> dict:
    """Set this process's power throttling (``mode``: opt_out, eco or default) and read it
    back. Returns what was asked and what Windows answered; raises Unavailable elsewhere."""
    import ctypes

    if kernel32 is None and not WINDOWS:
        raise Unavailable("power throttling (EcoQoS) is Windows'")
    k32 = _typed_process_calls(kernel32 or _kernel32())
    state = throttling_struct(mode)
    out = {"mode": mode, "control_mask": state.ControlMask, "state_mask": state.StateMask}
    ok = k32.SetProcessInformation(k32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                   ctypes.byref(state), ctypes.sizeof(state))
    out["set"] = bool(ok)
    if not ok:
        out["error"] = _last_error()
    back = throttling_struct("default")
    if k32.GetProcessInformation(k32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                 ctypes.byref(back), ctypes.sizeof(back)):
        out["readback"] = {"version": back.Version, "control_mask": back.ControlMask,
                           "state_mask": back.StateMask}
    else:
        out["readback_error"] = _last_error()
    return out


def set_priority(level: str, kernel32=None) -> dict:
    """This process at ``level`` ("normal" or "below_normal"): the Windows priority class, or on
    Linux the nice value the process started with or BELOW_NORMAL_NICE more (never less, so
    that a run under nice stays under it). Returns what is in force afterwards."""
    if level not in PRIORITY_CLASSES:
        raise ValueError(f"priority {level!r}: normal or below_normal")
    if WINDOWS or kernel32 is not None:
        dword, boolean, handle = _abi()
        k32 = kernel32 or _kernel32()
        k32.GetCurrentProcess.restype = handle
        k32.GetCurrentProcess.argtypes = []
        k32.SetPriorityClass.argtypes = (handle, dword)
        k32.SetPriorityClass.restype = boolean
        k32.GetPriorityClass.argtypes = (handle,)
        k32.GetPriorityClass.restype = dword
        ok = k32.SetPriorityClass(k32.GetCurrentProcess(), PRIORITY_CLASSES[level])
        now = int(k32.GetPriorityClass(k32.GetCurrentProcess()))
        return {"level": level, "set": bool(ok), "class": now,
                "now": PRIORITY_NAMES.get(now, hex(now))}
    target = BASE_NICE + (BELOW_NORMAL_NICE if level == "below_normal" else 0)
    out = {"level": level, "nice_asked": target}
    try:
        os.setpriority(os.PRIO_PROCESS, 0, target)
        out["set"] = True
    except OSError as exc:  # raising a priority again needs privileges
        out.update(set=False, error=str(exc))
    out["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
    return out


def priority_now() -> dict:
    if WINDOWS:
        try:
            dword, _boolean, handle = _abi()
            k32 = _kernel32()
            k32.GetCurrentProcess.restype = handle
            k32.GetPriorityClass.argtypes = (handle,)
            k32.GetPriorityClass.restype = dword
            now = int(k32.GetPriorityClass(k32.GetCurrentProcess()))
            return {"class": now, "now": PRIORITY_NAMES.get(now, hex(now))}
        except Exception as exc:  # the record only
            return {"error": str(exc)}
    return {"nice": os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else None}
