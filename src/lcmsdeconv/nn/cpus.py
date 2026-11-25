"""Where the training process and its data workers run on a processor whose cores differ:
performance and efficient cores (the standard library only, psutil where installed; no PyTorch).

Detection (:func:`detect`): the cores of this computer, each with the logical processors it
holds and its class - performance ("P"), efficient ("E") or low-power efficient ("LP"); "?"
where all cores are alike or nothing says otherwise. Read on Windows from the CPU sets
(GetSystemCpuSetInformation: EfficiencyClass, CoreIndex) or, where that fails, from
GetLogicalProcessorInformationEx(RelationProcessorCore), with the level-2 caches
(RelationCache) telling which cores share one (Intel's efficient cores share an L2 by four); on
Linux from sysfs: Intel's hybrid PMUs (/sys/devices/cpu_core/cpus, cpu_atom/cpus,
cpu_lowpower/cpus), else cpu_capacity, else the cpufreq maxima; elsewhere, or when anything
fails, all cores are taken as alike: :func:`detect` never raises. Processors are numbered as
``psutil.Process().cpu_affinity()`` numbers them: on Windows the bits of the affinity mask of
processor group 0 (LogicalProcessorIndex of the CPU sets of group 0; computers of up to 64
logical processors have that group only), on Linux the kernel's CPU numbers.

Placement (:func:`plan`): the training process keeps one or two of the fastest cores (its main
thread waits on the GPU's synchronisations, the pinned-memory thread receives and copies every
batch), ranked by class and processor number, processor 0 last (most interrupts land there);
the data workers get one slot per logical processor of every other core, fastest first.
``mode`` says how a worker is bound: "core" (each to its own core: only with the order-
preserving loader, lcmsdeconv.synth.pool, since torch's DataLoader hands out batches in turn
and the slowest core would set the pace of all), "pool" (each to all the workers' cores: the
system balances them) or "none" (not bound, as before). Windows' SchedulingClass is read and
reported (``python -m lcmsdeconv.nn.cpus``) but not used: it is not documented.

Application: :func:`worker_init` (a DataLoader ``worker_init_fn`` through functools.partial,
which pickles for spawned workers) sets the worker's priority (below normal as before, or
normal), opts it out of Windows' power throttling (EcoQoS) where the placement says so, and
binds it; :func:`apply_main` does the same for the training process: its power throttling at
the start, its binding once its workers have started (children inherit their parent's affinity
when they are created). Nothing here changes what a worker computes: P- and E-cores run the
same instructions, and training frames stay bit for bit the same.

Run ``python -m lcmsdeconv.nn.cpus`` to see what this computer reports and what would be done.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import sys
from dataclasses import asdict, dataclass, field

# ----------------------------------------------------------------------------- Windows constants
#: winerror.h
ERROR_INSUFFICIENT_BUFFER = 122
#: winnt.h: CPU_SET_INFORMATION_TYPE.CpuSetInformation; LOGICAL_PROCESSOR_RELATIONSHIP values
CPU_SET_INFORMATION = 0
RELATION_PROCESSOR_CORE = 0
RELATION_CACHE = 2
RELATION_ALL = 0xFFFF
#: PROCESSOR_CACHE_TYPE: CacheUnified, CacheInstruction, CacheData, CacheTrace
CACHE_UNIFIED, CACHE_INSTRUCTION, CACHE_DATA, CACHE_TRACE = 0, 1, 2, 3
#: PROCESSOR_RELATIONSHIP.Flags: the core has more than one logical processor
LTP_PC_SMT = 0x1
#: SYSTEM_CPU_SET_INFORMATION AllFlags bits
CPU_SET_PARKED, CPU_SET_ALLOCATED, CPU_SET_ALLOCATED_TO_TARGET, CPU_SET_REALTIME = 1, 2, 4, 8
CPU_SET_FLAG_NAMES = ((CPU_SET_PARKED, "parked"), (CPU_SET_ALLOCATED, "allocated"),
                      (CPU_SET_ALLOCATED_TO_TARGET, "allocated to this process"),
                      (CPU_SET_REALTIME, "real-time"))
#: processthreadsapi.h: PROCESS_INFORMATION_CLASS.ProcessPowerThrottling,
#: THREAD_INFORMATION_CLASS.ThreadPowerThrottling
PROCESS_POWER_THROTTLING = 4
THREAD_POWER_THROTTLING = 3
#: PROCESS_POWER_THROTTLING_CURRENT_VERSION (= THREAD_POWER_THROTTLING_CURRENT_VERSION)
POWER_THROTTLING_CURRENT_VERSION = 1
#: PROCESS_POWER_THROTTLING_EXECUTION_SPEED (= THREAD_POWER_THROTTLING_EXECUTION_SPEED): EcoQoS
POWER_THROTTLING_EXECUTION_SPEED = 0x1
#: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION: not used (the workers set no timers)
POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
#: SetPriorityClass classes
IDLE_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS = 0x40, 0x4000, 0x20
PRIORITY_CLASS_NAMES = {0x40: "idle", 0x4000: "below normal", 0x20: "normal",
                        0x8000: "above normal", 0x80: "high", 0x100: "real-time"}

# ---------------------------------------------------------------- Windows structures (winnt.h)
# Fixed-width types throughout: ctypes.wintypes.DWORD and ULONG are c_ulong, 8 bytes on Linux
# (LP64), so structures declared with them would have another layout there than on Windows.
c_u8, c_u16, c_u32, c_u64 = ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64
#: KAFFINITY / ULONG_PTR: pointer-sized (8 bytes for 64-bit Python, the only one PyTorch has)
c_uptr = ctypes.c_size_t


class SYSTEM_CPU_SET_INFORMATION(ctypes.Structure):
    """One entry of GetSystemCpuSetInformation, Type == CpuSetInformation; 32 bytes on 64-bit
    Windows (the AllocationTag aligns the union to 8)::

        DWORD Size; CPU_SET_INFORMATION_TYPE Type;              //  0,  4
        union { struct {
            DWORD Id; WORD Group;                               //  8, 12
            BYTE LogicalProcessorIndex, CoreIndex,              // 14, 15
                 LastLevelCacheIndex, NumaNodeIndex,            // 16, 17
                 EfficiencyClass;                               // 18
            union { BYTE AllFlags; struct {...bits...}; };      // 19
            union { DWORD Reserved; BYTE SchedulingClass; };    // 20 (low byte)
            DWORD64 AllocationTag;                              // 24
        } CpuSet; };

    The flag bits are read from AllFlags (bit fields would be laid out differently by GCC and
    MSVC); SchedulingClass is the first byte of the 4-byte union."""

    _fields_ = [("Size", c_u32), ("Type", c_u32), ("Id", c_u32), ("Group", c_u16),
                ("LogicalProcessorIndex", c_u8), ("CoreIndex", c_u8),
                ("LastLevelCacheIndex", c_u8), ("NumaNodeIndex", c_u8),
                ("EfficiencyClass", c_u8), ("AllFlags", c_u8), ("SchedulingClass", c_u8),
                ("_Reserved", c_u8 * 3), ("AllocationTag", c_u64)]


class GROUP_AFFINITY(ctypes.Structure):
    """``KAFFINITY Mask; WORD Group; WORD Reserved[3];``: 16 bytes on 64-bit Windows."""

    _fields_ = [("Mask", c_uptr), ("Group", c_u16), ("Reserved", c_u16 * 3)]


class PROCESSOR_RELATIONSHIP(ctypes.Structure):
    """``BYTE Flags; BYTE EfficiencyClass; BYTE Reserved[20]; WORD GroupCount;
    GROUP_AFFINITY GroupMask[ANYSIZE_ARRAY];``: GroupCount at 22, GroupMask at 24 (aligned to
    8); 40 bytes with one group mask."""

    _fields_ = [("Flags", c_u8), ("EfficiencyClass", c_u8), ("Reserved", c_u8 * 20),
                ("GroupCount", c_u16), ("GroupMask", GROUP_AFFINITY * 1)]


class CACHE_RELATIONSHIP(ctypes.Structure):
    """``BYTE Level; BYTE Associativity; WORD LineSize; DWORD CacheSize; PROCESSOR_CACHE_TYPE
    Type; BYTE Reserved[18]; WORD GroupCount; GROUP_AFFINITY GroupMask[ANYSIZE_ARRAY];``:
    GroupCount at 30, GroupMask at 32; 48 bytes with one group mask. Windows before 10 20H2 has
    ``BYTE Reserved[20]`` and one GroupMask at the same offset: GroupCount reads 0 there."""

    _fields_ = [("Level", c_u8), ("Associativity", c_u8), ("LineSize", c_u16),
                ("CacheSize", c_u32), ("Type", c_u32), ("Reserved", c_u8 * 18),
                ("GroupCount", c_u16), ("GroupMask", GROUP_AFFINITY * 1)]


class SLPI_EX_HEADER(ctypes.Structure):
    """The head of SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: ``LOGICAL_PROCESSOR_RELATIONSHIP
    Relationship; DWORD Size;``; the union that follows (PROCESSOR_RELATIONSHIP for
    RelationProcessorCore, CACHE_RELATIONSHIP for RelationCache) starts at offset 8. A core's
    entry is 48 bytes with one group, a cache's 56."""

    _fields_ = [("Relationship", c_u32), ("Size", c_u32)]


SLPI_EX_UNION_OFFSET = 8


class POWER_THROTTLING_STATE(ctypes.Structure):
    """PROCESS_POWER_THROTTLING_STATE and THREAD_POWER_THROTTLING_STATE (the same layout):
    ``ULONG Version; ULONG ControlMask; ULONG StateMask;``, 12 bytes."""

    _fields_ = [("Version", c_u32), ("ControlMask", c_u32), ("StateMask", c_u32)]


class PROCESSOR_NUMBER(ctypes.Structure):
    """``WORD Group; BYTE Number; BYTE Reserved;`` (GetCurrentProcessorNumberEx)."""

    _fields_ = [("Group", c_u16), ("Number", c_u8), ("Reserved", c_u8)]


# ------------------------------------------------------------------------------ the topology
@dataclass(frozen=True)
class Core:
    """A physical core: its logical processors (as psutil numbers them), its class and rank (0
    the fastest class), the value the class was read from (``efficiency``: Windows
    EfficiencyClass, Linux cpu_capacity or the highest frequency in kHz; 0 unknown), the
    scheduler's preference among cores of a class where reported (``favored``: Windows
    SchedulingClass, Linux acpi_cppc highest_perf; printed by the report, never used: it is not
    documented) and the group of cores sharing its level-2 cache (``cluster``; -1 unknown)."""

    cpus: tuple
    kind: str = "?"
    rank: int = 0
    favored: int = 0
    cluster: int = -1
    efficiency: int = 0


@dataclass(frozen=True)
class Topology:
    """The cores of a computer as :func:`detect` reads them; ``source`` says from where,
    ``notes`` what was left out or failed, ``raw`` the entries read (the report prints them),
    ``other_groups`` how many logical processors lie outside Windows' processor group 0 (which
    psutil's affinity cannot reach: nothing is bound then)."""

    cores: tuple
    source: str
    notes: tuple = ()
    raw: tuple = field(default=(), compare=False)
    other_groups: int = 0

    @property
    def hybrid(self) -> bool:
        return len({c.rank for c in self.cores}) > 1

    @property
    def logical_processors(self) -> int:
        return sum(len(c.cpus) for c in self.cores)

    def counts(self) -> dict:
        out: dict = {}
        for c in self.cores:
            out[c.kind] = out.get(c.kind, 0) + 1
        return out

    def kind_of_cpu(self) -> dict:
        """{logical processor: class of its core}."""
        return {cpu: c.kind for c in self.cores for cpu in c.cpus}

    def describe(self) -> str:
        kinds = ", ".join(f"{n} {KIND_WORDS.get(k, k)}" for k, n in self.counts().items())
        return (f"{len(self.cores)} cores ({kinds}), {self.logical_processors} logical "
                f"processors [{self.source}]")


KIND_WORDS = {"P": "performance", "E": "efficient", "LP": "low-power efficient", "?": "alike"}
KINDS_BY_RANK = ("P", "E", "LP")


def _kinds_for(ranks) -> dict:
    """{rank: kind}: one rank: "?" (all alike); else P, E, LP by rank."""
    ranks = sorted(set(ranks))
    if len(ranks) <= 1:
        return {r: "?" for r in ranks}
    return {r: KINDS_BY_RANK[min(i, len(KINDS_BY_RANK) - 1)] for i, r in enumerate(ranks)}


def _bits(mask: int) -> list[int]:
    return [i for i in range(mask.bit_length()) if mask >> i & 1]


def _dense_clusters(cores_cpus, groups) -> list[int]:
    """For each core (a tuple of processors), the index of the group (a set of processors, e.g.
    the processors sharing a level-2 cache) that holds all its processors, the groups numbered
    by their lowest processor; -1 where none does."""
    groups = sorted({frozenset(g) for g in groups if g}, key=min)
    out = []
    for cpus in cores_cpus:
        out.append(next((i for i, g in enumerate(groups) if set(cpus) <= g), -1))
    return out


# ------------------------------------------------------------------ Windows: buffers and parsing
def parse_cpu_sets(buf) -> list[dict]:
    """The CpuSetInformation entries of a GetSystemCpuSetInformation buffer, walked by each
    entry's Size (a later Windows may grow the structure; an entry shorter than the fields up to
    SchedulingClass is read as far as it goes)."""
    buf = bytes(buf)
    fixed = ctypes.sizeof(SYSTEM_CPU_SET_INFORMATION)
    out, off = [], 0
    while off + 8 <= len(buf):
        size, kind = struct.unpack_from("<II", buf, off)
        if size < 8 or off + size > len(buf):
            raise ValueError(f"malformed CPU set entry at byte {off} (size {size})")
        if kind == CPU_SET_INFORMATION and size >= 20:
            e = SYSTEM_CPU_SET_INFORMATION.from_buffer_copy(
                buf[off:off + min(size, fixed)].ljust(fixed, b"\0"))
            out.append({"id": e.Id, "group": e.Group, "lp": e.LogicalProcessorIndex,
                        "core": e.CoreIndex, "llc": e.LastLevelCacheIndex,
                        "numa": e.NumaNodeIndex, "efficiency_class": e.EfficiencyClass,
                        "flags": e.AllFlags,
                        "scheduling_class": e.SchedulingClass if size >= 21 else 0})
        off += size
    if off != len(buf):
        raise ValueError(f"{len(buf) - off} bytes left over after the CPU sets")
    return out


def _slpi_entries(buf, relation: int):
    """(offset of the union, size of the entry) of every entry of ``relation`` in a
    GetLogicalProcessorInformationEx buffer, walked by each entry's Size (other relationships,
    present with RelationAll, are skipped)."""
    head = ctypes.sizeof(SLPI_EX_HEADER)
    off = 0
    while off + head <= len(buf):
        h = SLPI_EX_HEADER.from_buffer_copy(buf[off:off + head])
        if h.Size < head or off + h.Size > len(buf):
            raise ValueError(f"malformed processor information entry at byte {off} "
                             f"(size {h.Size})")
        if h.Relationship == relation:
            yield off, h.Size
        off += h.Size
    if off != len(buf):
        raise ValueError(f"{len(buf) - off} bytes left over after the processor information")


def _group_masks(buf, off: int, size: int, first: int, count: int) -> list[tuple]:
    """``count`` GROUP_AFFINITY entries from byte ``first`` of the entry at ``off``."""
    ga = ctypes.sizeof(GROUP_AFFINITY)
    if first + count * ga > off + size:
        raise ValueError(f"entry at byte {off}: {count} group masks overrun it")
    out = []
    for i in range(count):
        g = GROUP_AFFINITY.from_buffer_copy(buf[first + i * ga:first + (i + 1) * ga])
        out.append((g.Group, g.Mask))
    return out


def parse_processor_cores(buf) -> list[dict]:
    """The RelationProcessorCore entries of a GetLogicalProcessorInformationEx buffer: flags,
    efficiency class and the (group, mask) pairs of each core."""
    buf = bytes(buf)
    fixed = PROCESSOR_RELATIONSHIP.GroupMask.offset
    out = []
    for off, size in _slpi_entries(buf, RELATION_PROCESSOR_CORE):
        base = off + SLPI_EX_UNION_OFFSET
        if base + fixed > off + size:
            raise ValueError(f"core entry at byte {off} too short ({size} bytes)")
        flags, eff = buf[base], buf[base + 1]
        (count,) = struct.unpack_from("<H", buf, base + PROCESSOR_RELATIONSHIP.GroupCount.offset)
        out.append({"flags": flags, "efficiency_class": eff,
                    "groups": _group_masks(buf, off, size, base + fixed, count)})
    return out


def parse_caches(buf) -> list[dict]:
    """The RelationCache entries of a GetLogicalProcessorInformationEx buffer: level, type, size
    in bytes and the (group, mask) pairs of the processors sharing each cache."""
    buf = bytes(buf)
    fixed = CACHE_RELATIONSHIP.GroupMask.offset
    out = []
    for off, size in _slpi_entries(buf, RELATION_CACHE):
        base = off + SLPI_EX_UNION_OFFSET
        if base + fixed > off + size:
            raise ValueError(f"cache entry at byte {off} too short ({size} bytes)")
        level, ctype = buf[base], struct.unpack_from("<I", buf, base + 8)[0]
        (nbytes,) = struct.unpack_from("<I", buf, base + 4)
        (count,) = struct.unpack_from("<H", buf, base + CACHE_RELATIONSHIP.GroupCount.offset)
        out.append({"level": level, "type": ctype, "size": nbytes,
                    "groups": _group_masks(buf, off, size, base + fixed, max(count, 1))})
    return out


def l2_groups(caches: list[dict]) -> list[set]:
    """The sets of processor-group-0 processors sharing a level-2 cache (unified or data)."""
    out = []
    for c in caches:
        if c["level"] == 2 and c["type"] in (CACHE_UNIFIED, CACHE_DATA):
            cpus = {x for g, mask in c["groups"] if g == 0 for x in _bits(mask)}
            if cpus:
                out.append(cpus)
    return out


def cores_from_cpu_sets(entries: list[dict], caches: list[dict] | None = None) -> Topology:
    """Cores from CPU sets: grouped by (Group, CoreIndex); processor group 0 only, the group
    psutil's affinity reaches. Classes from EfficiencyClass (higher: faster); level-2 cache
    groups from ``caches`` (:func:`parse_caches`)."""
    by_core: dict = {}
    other_groups = 0
    for e in entries:
        if e["group"] != 0:
            other_groups += 1
            continue
        by_core.setdefault(e["core"], []).append(e)
    classes = sorted({max(x["efficiency_class"] for x in v) for v in by_core.values()},
                     reverse=True)
    rank_of = {c: i for i, c in enumerate(classes)}
    kinds = _kinds_for(rank_of.values())
    ordered = sorted(by_core.values(), key=lambda v: min(x["lp"] for x in v))
    cpus_of = [tuple(sorted(x["lp"] for x in v)) for v in ordered]
    clusters = _dense_clusters(cpus_of, l2_groups(caches or []))
    cores = []
    for v, cpus, cluster in zip(ordered, cpus_of, clusters):
        eff = max(x["efficiency_class"] for x in v)
        cores.append(Core(cpus=cpus, kind=kinds[rank_of[eff]], rank=rank_of[eff],
                          favored=max(x["scheduling_class"] for x in v), cluster=cluster,
                          efficiency=eff))
    notes = (f"{other_groups} logical processors outside processor group 0 left out",) \
        if other_groups else ()
    return Topology(tuple(cores), "windows-cpu-sets", notes, tuple(entries), other_groups)


def cores_from_processor_info(entries: list[dict], caches: list[dict] | None = None) -> Topology:
    """Cores from GetLogicalProcessorInformationEx(RelationProcessorCore); group 0 only."""
    cores_raw = []
    other = 0
    for e in entries:
        cpus = []
        for g, mask in e["groups"]:
            if g == 0:
                cpus += _bits(mask)
            else:
                other += len(_bits(mask))
        if cpus:
            cores_raw.append((tuple(sorted(cpus)), e["efficiency_class"]))
    cores_raw.sort()
    classes = sorted({c for _cpus, c in cores_raw}, reverse=True)
    rank_of = {c: i for i, c in enumerate(classes)}
    kinds = _kinds_for(rank_of.values())
    clusters = _dense_clusters([cpus for cpus, _c in cores_raw], l2_groups(caches or []))
    cores = tuple(Core(cpus=cpus, kind=kinds[rank_of[c]], rank=rank_of[c], cluster=cluster,
                       efficiency=c)
                  for (cpus, c), cluster in zip(cores_raw, clusters))
    notes = (f"{other} logical processors outside processor group 0 left out",) if other else ()
    return Topology(cores, "windows-processor-info", notes, other_groups=other)


_KERNEL32 = None


def win_kernel32():
    """kernel32 with typed prototypes (untyped, a 64-bit handle or pointer is cut to 32 bits),
    on a private handle: the shared ctypes.windll one has prototypes other libraries rely on."""
    global _KERNEL32
    if _KERNEL32 is not None:
        return _KERNEL32
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    proto = {
        "GetSystemCpuSetInformation": ((ctypes.c_void_p, c_u32, ctypes.POINTER(c_u32),
                                        wintypes.HANDLE, c_u32), wintypes.BOOL),
        "GetLogicalProcessorInformationEx": ((ctypes.c_int, ctypes.c_void_p,
                                              ctypes.POINTER(c_u32)), wintypes.BOOL),
        "GetCurrentProcess": ((), wintypes.HANDLE),
        "GetCurrentThread": ((), wintypes.HANDLE),
        "SetProcessInformation": ((wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, c_u32),
                                  wintypes.BOOL),
        "GetProcessInformation": ((wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, c_u32),
                                  wintypes.BOOL),
        "SetThreadInformation": ((wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, c_u32),
                                 wintypes.BOOL),
        "SetPriorityClass": ((wintypes.HANDLE, c_u32), wintypes.BOOL),
        "GetPriorityClass": ((wintypes.HANDLE,), c_u32),
        "SetProcessAffinityMask": ((wintypes.HANDLE, c_uptr), wintypes.BOOL),
        "GetProcessAffinityMask": ((wintypes.HANDLE, ctypes.POINTER(c_uptr),
                                    ctypes.POINTER(c_uptr)), wintypes.BOOL),
        "GetCurrentProcessorNumberEx": ((ctypes.POINTER(PROCESSOR_NUMBER),), None),
        "SetConsoleCtrlHandler": ((ctypes.c_void_p, wintypes.BOOL), wintypes.BOOL),
    }
    for name, (args, res) in proto.items():
        fn = getattr(k, name, None)
        if fn is not None:  # GetSystemCpuSetInformation: Windows 10 and later
            fn.argtypes, fn.restype = args, res
    _KERNEL32 = k
    return k


def _two_calls(call, tries: int = 4) -> bytes:
    """A buffer from an API that reports the size it needs: ``call(buffer, length, needed)``
    returns non-zero on success; asked with no buffer it fails and sets ``needed``. Asked again
    when the size grew meanwhile (processors brought online)."""
    need = c_u32(0)
    if call(None, 0, need):  # nothing to return
        return b""
    for _ in range(tries):
        if need.value == 0:
            break
        buf = ctypes.create_string_buffer(need.value)
        size = need.value
        if call(buf, size, need):
            return buf.raw[:need.value or size]
    raise OSError(f"no processor information (last error {_last_error()})")


def _last_error() -> int:
    try:
        return ctypes.get_last_error()
    except Exception:  # not on Windows
        return 0


def win_cpu_set_buffer(k=None) -> bytes:
    k = k or win_kernel32()
    return _two_calls(lambda buf, n, need: k.GetSystemCpuSetInformation(
        buf, n, ctypes.byref(need), None, 0))


def win_processor_buffer(k=None, relation: int = RELATION_PROCESSOR_CORE) -> bytes:
    k = k or win_kernel32()

    def call(buf, n, need):
        need.value = n  # in: the buffer's length; out: bytes written, or needed
        return k.GetLogicalProcessorInformationEx(relation, buf, ctypes.byref(need))
    return _two_calls(call)


def windows_topology(k=None) -> Topology:
    """The CPU sets (Windows 10 and later), else GetLogicalProcessorInformationEx; their
    EfficiencyClass is zero on all cores of a processor whose cores are alike. The level-2
    caches, where they can be read, say which cores share one."""
    notes = []
    try:
        caches = parse_caches(win_processor_buffer(k, RELATION_CACHE))
    except Exception as exc:  # the cores are still known; only their cache groups are not
        caches = []
        notes.append(f"caches unreadable: {type(exc).__name__}: {exc}")
    try:
        topo = cores_from_cpu_sets(parse_cpu_sets(win_cpu_set_buffer(k)), caches)
        if topo.cores:
            return Topology(topo.cores, topo.source, tuple(notes) + topo.notes, topo.raw,
                            topo.other_groups)
        notes.append("no CPU sets")
    except Exception as exc:  # an older Windows, or a buffer we cannot read
        notes.append(f"CPU sets unreadable: {type(exc).__name__}: {exc}")
    topo = cores_from_processor_info(parse_processor_cores(win_processor_buffer(k)), caches)
    return Topology(topo.cores, topo.source, tuple(notes) + topo.notes,
                    other_groups=topo.other_groups)


# --------------------------------------------------------------------------------- Linux sysfs
def parse_cpu_list(text: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]."""
    out: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as fh:
            return fh.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


#: the hybrid PMUs of Intel processors under Linux, fastest first
LINUX_PMUS = (("cpu_core", "P"), ("cpu_atom", "E"), ("cpu_lowpower", "LP"))
#: a core belongs to the class of the fastest core in it if within this share of its speed
#: (Turbo Boost Max favoured cores run a few percent faster than their class; E-cores of Arrow
#: Lake reach about 0.84 of the P-cores' clock, Zen 5c about 0.65 of Zen 5's)
TIER_GAP = 0.10


def tiers(values: dict) -> dict:
    """{key: rank} for {key: speed}: rank 0 the fastest; a key joins the tier of the fastest
    value above it while within TIER_GAP of that value."""
    order = sorted(values.items(), key=lambda kv: -kv[1])
    out, rank, top = {}, -1, None
    for key, v in order:
        if top is None or v < top * (1.0 - TIER_GAP):
            rank, top = rank + 1, v
        out[key] = rank
    return out


def _linux_l2(sysdir: str, cores) -> list[int]:
    """Level-2 cache groups of ``cores`` from cpuN/cache/index*/ (level 2, unified or data:
    shared_cpu_list), else cpuN/topology/cluster_id; -1 where neither is there."""
    groups = []
    for key in cores:
        base = os.path.join(sysdir, f"cpu{key[0]}", "cache")
        try:
            names = sorted(n for n in os.listdir(base) if n.startswith("index"))
        except OSError:
            continue
        for name in names:
            d = os.path.join(base, name)
            if _read(os.path.join(d, "level")) == "2" and \
                    (_read(os.path.join(d, "type")) or "") in ("Unified", "Data"):
                shared = _read(os.path.join(d, "shared_cpu_list"))
                if shared:
                    groups.append(set(parse_cpu_list(shared)))
                break
    if groups:
        return _dense_clusters(list(cores), groups)
    out = []
    for key in cores:
        cid = _read(os.path.join(sysdir, f"cpu{key[0]}", "topology/cluster_id"))
        out.append(int(cid) if cid and cid.lstrip("-").isdigit() else -1)
    return out


def linux_topology(root: str = "/") -> Topology:
    """Cores from sysfs under ``root`` (a fake tree in tests): logical processors grouped by
    core_cpus_list (thread siblings), classes from the hybrid PMUs, else cpu_capacity, else the
    cpufreq maxima (split where one is more than TIER_GAP below the next faster), else alike."""
    sysdir = os.path.join(root, "sys/devices/system/cpu")
    online = parse_cpu_list(_read(os.path.join(sysdir, "online")) or "")
    if not online:
        online = sorted(int(d[3:]) for d in os.listdir(sysdir)
                        if d.startswith("cpu") and d[3:].isdigit())
    cores: dict = {}
    for cpu in online:
        topo_dir = os.path.join(sysdir, f"cpu{cpu}", "topology")
        sib = _read(os.path.join(topo_dir, "core_cpus_list")) or \
            _read(os.path.join(topo_dir, "thread_siblings_list"))
        key = tuple(x for x in parse_cpu_list(sib) if x in online) if sib else (cpu,)
        cores.setdefault(key or (cpu,), None)
    notes = []
    kind_of: dict = {}
    for pmu, kind in LINUX_PMUS:
        text = _read(os.path.join(root, "sys/devices", pmu, "cpus"))
        if text:
            for cpu in parse_cpu_list(text):
                kind_of[cpu] = kind
    source = "uniform"
    rank: dict = {}
    speed: dict = {}
    if kind_of:
        source = "linux-hybrid-pmu"
        order = {k: i for i, (_p, k) in enumerate(LINUX_PMUS)}
        for key in cores:
            ks = [kind_of.get(c) for c in key if c in kind_of]
            rank[key] = min((order[k] for k in ks), default=len(LINUX_PMUS))
    else:
        for name, rel, src in (("cpu_capacity", "cpu_capacity", "linux-cpu-capacity"),
                               ("cpuinfo_max_freq", "cpufreq/cpuinfo_max_freq",
                                "linux-cpufreq-max")):
            speed = {}
            for key in cores:
                vals = [_read(os.path.join(sysdir, f"cpu{c}", rel)) for c in key]
                vals = [float(v) for v in vals if v and v.replace(".", "", 1).isdigit()]
                if vals:
                    speed[key] = max(vals)
            if len(speed) == len(cores) and len(set(speed.values())) > 1:
                t = tiers(speed)
                if len(set(t.values())) > 1:
                    rank, source = t, src
                    break
                notes.append(f"{name} differs by less than {TIER_GAP:.0%}: cores alike")
    favored = {}
    for key in cores:
        hp = [_read(os.path.join(sysdir, f"cpu{c}", "acpi_cppc/highest_perf")) for c in key]
        hp = [int(v) for v in hp if v and v.isdigit()]
        favored[key] = max(hp) if hp else 0
    ordered = sorted(cores, key=lambda k: k[0])
    clusters = dict(zip(ordered, _linux_l2(sysdir, ordered)))
    ranks = {key: rank.get(key, 0) for key in cores}
    present = sorted(set(ranks.values()))
    dense = {r: i for i, r in enumerate(present)}  # e.g. no cpu_atom: P and LP become 0 and 1
    if source == "linux-hybrid-pmu":
        names = {i: (LINUX_PMUS[r][1] if r < len(LINUX_PMUS) else "?")
                 for r, i in dense.items()}
        if len(present) == 1:
            names = {0: "?"}
    else:
        names = _kinds_for(dense.values())
    out = tuple(Core(cpus=key, kind=names[dense[ranks[key]]], rank=dense[ranks[key]],
                     favored=favored[key], cluster=clusters[key],
                     efficiency=int(speed.get(key, 0)))
                for key in ordered)
    return Topology(out, source, tuple(notes))


def uniform_topology(n: int | None = None) -> Topology:
    """``n`` alike cores of one logical processor each (default: this computer's count)."""
    n = n or os.cpu_count() or 1
    return Topology(tuple(Core(cpus=(i,)) for i in range(n)), "uniform")


def detect(k=None) -> Topology:
    """This computer's cores (see the module's docstring); never raises: whatever fails, the
    cores are taken as alike."""
    try:
        if sys.platform == "win32" or k is not None:
            topo = windows_topology(k)
        elif sys.platform.startswith("linux"):
            topo = linux_topology()
        else:
            return uniform_topology()
        if topo.cores:
            return topo
        return Topology(uniform_topology().cores, "uniform", topo.notes + ("no cores read",))
    except Exception as exc:  # a platform or a version we could not read
        return Topology(uniform_topology().cores, "uniform",
                        (f"detection failed: {type(exc).__name__}: {exc}",))


# ---------------------------------------------------------------------------------- placement
#: how data workers are bound to the processor's cores (Placement.mode, --cpu-binding)
MODES = ("none", "pool", "core")
#: the data workers' priority (Placement.priority, --worker-priority): below normal is today's
PRIORITIES = ("below_normal", "normal")
#: Windows power throttling of the data workers and the training process (Placement.qos,
#: --worker-qos): "system" leaves it to Windows' heuristics (today's), "high" opts them out
#: of EcoQoS (the documented HighQoS)
QOS_MODES = ("system", "high")


@dataclass(frozen=True)
class Placement:
    """Where the training process (``main``) and each data worker (``workers[i]``, for worker id
    i; empty: not bound) run, the class of each worker's core(s) (``kinds``: the cores it was
    given; "?" where not bound), the workers' priority and the power throttling of all of
    them."""

    mode: str
    main: tuple = ()
    workers: tuple = ()
    kinds: tuple = ()
    priority: str = "below_normal"
    qos: str = "system"
    notes: tuple = ()

    def worker_cpus(self, worker_id: int) -> tuple:
        return self.workers[worker_id % len(self.workers)] if self.workers else ()

    def worker_kind(self, worker_id: int) -> str:
        return self.kinds[worker_id % len(self.kinds)] if self.kinds else "?"

    def label(self, worker_id: int) -> str:
        """How messages name worker ``worker_id``'s cores ("" where it is not bound)."""
        cpus = self.worker_cpus(worker_id)
        if not cpus:
            return ""
        if self.mode == "core":
            kind = self.worker_kind(worker_id)
            word = f"{KIND_WORDS[kind]} core" if kind in KINDS_BY_RANK else "core"
            return f"{word}, processor{'s' if len(cpus) > 1 else ''} {cpu_ranges(cpus)}"
        return f"processors {cpu_ranges(cpus)}"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> Placement:
        d = json.loads(text)
        return cls(mode=d["mode"], main=tuple(d["main"]),
                   workers=tuple(tuple(w) for w in d["workers"]), kinds=tuple(d["kinds"]),
                   priority=d.get("priority", "below_normal"), qos=d.get("qos", "system"),
                   notes=tuple(d.get("notes", ())))

    def describe(self) -> str:
        n = len(self.workers)
        how = {"below_normal": "below-normal priority", "normal": "normal priority"}[self.priority]
        qos = {"system": "power throttling left to Windows",
               "high": "opted out of power throttling (EcoQoS)"}[self.qos]
        if self.mode == "none" or not any(self.workers):
            text = f"{n} data workers not bound to cores"
            if self.notes:
                text += f" ({self.notes[0]})"
            return f"{text}; {how}; {qos}"
        by_kind: dict = {}
        for k in self.kinds:
            by_kind[k] = by_kind.get(k, 0) + 1
        kinds = ", ".join(f"{v} {KIND_WORDS.get(k, k)}" for k, v in by_kind.items())
        where = "each on its own core" if self.mode == "core" else \
            f"sharing processors {cpu_ranges(self.workers[0])}"
        return (f"{n} data workers {where} ({kinds}), {how}; training process on processors "
                f"{cpu_ranges(self.main)}; {qos}")


def cpu_ranges(cpus) -> str:
    """[0, 1, 2, 5, 7, 8] -> "0-2,5,7-8"."""
    cpus = sorted(cpus)
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


#: the training process's cores: one on a processor of four cores or fewer, two above (its main
#: thread waits on the GPU's synchronisations, the pinned-memory thread receives and copies every
#: batch); no binding at all below MIN_CORES usable cores
MAIN_CORES_SMALL, MAIN_CORES, SMALL_CPU, MIN_CORES = 1, 2, 4, 3


def plan(topo: Topology, workers: int | None = None, allowed=None, mode: str = "none",
         main_cores: int | None = None, per_core: dict | None = None,
         priority: str = "below_normal", qos: str = "system") -> Placement:
    """Where to run the training process and ``workers`` data workers (None: one per logical
    processor of the workers' cores, ``per_core`` times as many on a class it names, e.g.
    ``{"P": 2}``), on the cores of ``topo`` this process may use (``allowed``: its affinity).

    The training process keeps ``main_cores`` of the fastest cores (MAIN_CORES, MAIN_CORES_SMALL
    on SMALL_CPU cores or fewer), ranked by class, then processor number, processor 0 last. The
    workers get the other cores, fastest class first, so that fewer workers than slots take the
    fastest; more workers than slots wrap round, fastest first. "core" binds each worker to its
    core's logical processors, "pool" every worker to all the workers' cores, "none" binds
    nothing (the workers' number is the same). Below MIN_CORES usable cores nothing is bound and
    the workers default to all logical processors but one."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if priority not in PRIORITIES:
        raise ValueError(f"priority must be one of {PRIORITIES}, not {priority!r}")
    if qos not in QOS_MODES:
        raise ValueError(f"qos must be one of {QOS_MODES}, not {qos!r}")
    allowed = set(allowed) if allowed is not None else None
    cores = []
    for c in topo.cores:
        cpus = tuple(x for x in c.cpus if allowed is None or x in allowed)
        if cpus:
            cores.append(Core(cpus=cpus, kind=c.kind, rank=c.rank, favored=c.favored,
                              cluster=c.cluster, efficiency=c.efficiency))
    if len(cores) < MIN_CORES:
        n = workers or max(1, sum(len(c.cpus) for c in cores) - 1)
        why = () if mode == "none" else (f"only {len(cores)} usable cores",)
        return Placement("none", (), tuple(() for _ in range(n)), tuple("?" for _ in range(n)),
                         priority, qos, why)
    if topo.other_groups and mode != "none":  # binding would leave the other groups idle
        n = workers or max(1, topo.logical_processors + topo.other_groups - 2)
        return Placement("none", (), tuple(() for _ in range(n)), tuple("?" for _ in range(n)),
                         priority, qos, (f"{topo.other_groups} logical processors outside "
                                         f"processor group 0",))
    k = main_cores if main_cores is not None else (
        MAIN_CORES_SMALL if len(cores) <= SMALL_CPU else MAIN_CORES)
    k = max(1, min(k, len(cores) - 1))
    by_main = sorted(cores, key=lambda c: (c.rank, 0 in c.cpus, c.cpus[0]))
    main = by_main[:k]
    rest = sorted((c for c in cores if c not in main), key=lambda c: (c.rank, c.cpus[0]))
    per_core = per_core or {}
    slots = [c for c in rest for _ in range(max(0, per_core.get(c.kind, len(c.cpus))))] or rest
    n = workers if workers else len(slots)
    assigned = [slots[i % len(slots)] for i in range(n)]
    if mode == "none":
        return Placement("none", (), tuple(() for _ in assigned), tuple("?" for _ in assigned),
                         priority, qos)
    main_cpus = tuple(sorted(x for c in main for x in c.cpus))
    if mode == "core":
        bound = tuple(c.cpus for c in assigned)
    else:  # pool: the cores the workers use, all of them for every worker
        used = {c.cpus: c for c in assigned}
        pool = tuple(sorted(x for c in used.values() for x in c.cpus))
        bound = tuple(pool for _ in assigned)
    return Placement(mode, main_cpus, bound, tuple(c.kind for c in assigned), priority, qos)


def worker_slots(topo: Topology | None = None, allowed=None) -> int:
    """Data workers :func:`plan` gives by default: one per logical processor of every core the
    training process does not keep (this computer's cores and affinity by default)."""
    if topo is None:
        topo = detect()
        if allowed is None:
            allowed = allowed_cpus()
    return max(1, len(plan(topo, allowed=allowed, mode="core").workers))


# ---------------------------------------------------------------------------------- applying
def pin_process(cpus) -> bool:
    """Bind this whole process (every thread it has, and those it starts) to ``cpus``. Windows:
    SetProcessAffinityMask (the process mask covers all threads; through psutil where
    installed); Linux: every thread in /proc/self/task (sched_setaffinity binds one thread;
    threads started later inherit their creator's mask). False where it failed or is not
    supported."""
    cpus = sorted(set(cpus))
    if not cpus:
        return False
    try:
        if sys.platform.startswith("linux"):
            for tid in os.listdir("/proc/self/task"):
                try:
                    os.sched_setaffinity(int(tid), cpus)
                except ProcessLookupError:  # the thread ended meanwhile
                    pass
            return set(os.sched_getaffinity(0)) == set(cpus)
        if sys.platform == "win32" and max(cpus) < 64:
            try:
                import psutil

                p = psutil.Process()
                p.cpu_affinity(cpus)
                return sorted(p.cpu_affinity()) == cpus
            except ImportError:
                k = win_kernel32()
                mask = sum(1 << c for c in cpus)
                return bool(k.SetProcessAffinityMask(k.GetCurrentProcess(), mask))
        return False
    except Exception:
        return False


def set_power_throttling(eco: bool | None, k=None, thread: bool = False) -> bool:
    """Windows: this process's (or with ``thread`` this thread's) execution-speed throttling.
    ``eco=False`` opts out (the documented "HighQoS": no EcoQoS whatever the window state, and
    the process is no longer classified by the system's heuristics, which include its priority),
    ``True`` opts in (EcoQoS: efficient cores, lowest frequency), ``None`` hands the decision
    back to Windows' heuristics. Windows 10 1709 and later; False elsewhere or on failure."""
    if k is None:
        if sys.platform != "win32":
            return False
        try:
            k = win_kernel32()
        except Exception:
            return False
    state = POWER_THROTTLING_STATE(
        POWER_THROTTLING_CURRENT_VERSION,
        0 if eco is None else POWER_THROTTLING_EXECUTION_SPEED,
        POWER_THROTTLING_EXECUTION_SPEED if eco else 0)
    try:
        if thread:
            return bool(k.SetThreadInformation(k.GetCurrentThread(), THREAD_POWER_THROTTLING,
                                               ctypes.byref(state), ctypes.sizeof(state)))
        return bool(k.SetProcessInformation(k.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                            ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False


def get_power_throttling(k=None) -> tuple | None:
    """Windows: (ControlMask, StateMask) of this process, where GetProcessInformation reads it
    back (not every Windows does); None otherwise."""
    if k is None:
        if sys.platform != "win32":
            return None
        try:
            k = win_kernel32()
        except Exception:
            return None
    state = POWER_THROTTLING_STATE(POWER_THROTTLING_CURRENT_VERSION, 0, 0)
    try:
        ok = k.GetProcessInformation(k.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                     ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        return None
    return (state.ControlMask, state.StateMask) if ok else None


def restore_power_throttling(before: tuple | None, k=None) -> bool:
    """Set this process's power throttling back to ``before`` (:func:`get_power_throttling`'s
    answer; None: Windows' heuristics)."""
    if before is None or not before[0] & POWER_THROTTLING_EXECUTION_SPEED:
        return set_power_throttling(None, k)
    return set_power_throttling(bool(before[1] & POWER_THROTTLING_EXECUTION_SPEED), k)


_NICED = False  # set_priority lowered this process's priority already (Linux: nice is relative)


def set_priority(priority: str = "below_normal", k=None) -> bool:
    """A data worker's priority: Windows BELOW_NORMAL_PRIORITY_CLASS or NORMAL_PRIORITY_CLASS;
    elsewhere nice +5 for below normal (once per process; normal leaves the nice value as
    inherited, which only a privileged process could lower). The computer's other programs come
    first at below normal; when nothing else needs the processor the workers run at full
    speed either way."""
    global _NICED
    if priority not in PRIORITIES:
        raise ValueError(f"priority must be one of {PRIORITIES}, not {priority!r}")
    try:
        if sys.platform == "win32" or k is not None:
            k = k or win_kernel32()
            cls = BELOW_NORMAL_PRIORITY_CLASS if priority == "below_normal" else \
                NORMAL_PRIORITY_CLASS
            return bool(k.SetPriorityClass(k.GetCurrentProcess(), cls))
        if priority == "below_normal":
            if not _NICED:
                os.nice(5)
                _NICED = True
            return True
        return not _NICED
    except Exception:
        return False


def get_priority(k=None) -> str | None:
    """This process's priority as a word (Windows: its priority class; elsewhere its nice
    value), or None where it cannot be read."""
    try:
        if sys.platform == "win32" or k is not None:
            k = k or win_kernel32()
            cls = int(k.GetPriorityClass(k.GetCurrentProcess()))
            return PRIORITY_CLASS_NAMES.get(cls, hex(cls))
        return f"nice {os.getpriority(os.PRIO_PROCESS, 0)}"
    except Exception:
        return None


_SCHED_GETCPU = None


def current_cpu() -> int | None:
    """The logical processor this thread runs on now (psutil's numbering), or None."""
    global _SCHED_GETCPU
    try:
        if sys.platform == "win32":
            k = win_kernel32()
            pn = PROCESSOR_NUMBER()
            k.GetCurrentProcessorNumberEx(ctypes.byref(pn))
            return int(pn.Number) if pn.Group == 0 else None
        if sys.platform.startswith("linux"):
            if _SCHED_GETCPU is None:
                fn = ctypes.CDLL(None, use_errno=True).sched_getcpu
                fn.argtypes, fn.restype = (), ctypes.c_int
                _SCHED_GETCPU = fn
            cpu = int(_SCHED_GETCPU())
            return cpu if cpu >= 0 else None
    except Exception:
        return None
    return None


#: what worker_init did in this process (read by the probe and by tests)
WORKER_STATE: dict = {}


def worker_init(worker_id: int, placement: Placement | None = None) -> None:
    """A data worker's start (a DataLoader ``worker_init_fn``: pass ``functools.partial(
    worker_init, placement=...)``, which pickles, so spawned workers get it; the ordered loader
    of lcmsdeconv.synth.pool calls it the same way): set this worker's priority (below normal
    without a placement, as lcmsdeconv.synth.dataset.worker_init does), opt it out of power
    throttling where the placement says so (Windows), and bind it to its core(s). Every step is a
    courtesy: one that fails leaves the worker as it was, generating the same spectra."""
    WORKER_STATE.clear()
    WORKER_STATE["id"] = worker_id
    priority = placement.priority if placement is not None else "below_normal"
    WORKER_STATE["priority"] = set_priority(priority)
    if placement is not None and placement.qos == "high":
        WORKER_STATE["qos"] = set_power_throttling(False)
    cpus = placement.worker_cpus(worker_id) if placement is not None else ()
    if cpus:
        WORKER_STATE["cpus"] = cpus
        WORKER_STATE["pinned"] = pin_process(cpus)


def apply_main(placement: Placement | None, bind: bool = True, qos: bool = True) -> dict:
    """The training process's part of ``placement``: its power throttling (``qos``: opted out
    where the placement says so; at the start of training) and its binding (``bind``: to its
    cores, once its data workers have started, since a child inherits its parent's affinity
    when it is created: bound earlier, the workers would start on the training process's
    cores). Returns what was done."""
    done: dict = {}
    if placement is None:
        return done
    if qos and placement.qos == "high":
        done["qos"] = set_power_throttling(False)
    if bind and placement.mode != "none" and placement.main:
        done["pinned"] = pin_process(placement.main)
    return done


def allowed_cpus() -> list | None:
    """The logical processors this process may run on now (psutil's numbering), or None where
    that cannot be read (then every core of the topology is taken as usable)."""
    try:
        if hasattr(os, "sched_getaffinity"):
            return sorted(os.sched_getaffinity(0))
        import psutil

        return sorted(psutil.Process().cpu_affinity())
    except Exception:
        return None


def affinity_of(pids) -> dict:
    """{pid: sorted affinity} of other processes (the workers), as the training process checks
    them once they run; a pid that cannot be read maps to None."""
    out = {}
    for pid in pids:
        try:
            if sys.platform.startswith("linux"):
                out[pid] = sorted(os.sched_getaffinity(pid))
            else:
                import psutil

                out[pid] = sorted(psutil.Process(pid).cpu_affinity())
        except Exception:
            out[pid] = None
    return out


# ------------------------------------------------------------------------------------ report
def report(topo: Topology | None = None, allowed=None, workers: int | None = None) -> dict:
    """What this computer reports and what would be done, as a dict (:func:`main` prints it):
    the cores, the plan of each binding mode, this process's affinity, priority, power
    throttling and processor."""
    topo = topo or detect()
    allowed = allowed_cpus() if allowed is None else allowed
    plans = {m: plan(topo, workers=workers, allowed=allowed, mode=m) for m in MODES}
    return {"platform": sys.platform, "topology": topo.describe(), "source": topo.source,
            "notes": list(topo.notes),
            "cores": [asdict(c) for c in topo.cores],
            "raw": [dict(e) for e in topo.raw],
            "allowed": allowed,
            "plans": {m: {"describe": p.describe(), "main": list(p.main),
                          "workers": [list(w) for w in p.workers], "kinds": list(p.kinds)}
                      for m, p in plans.items()},
            "priority": get_priority(), "power_throttling": get_power_throttling(),
            "current_cpu": current_cpu()}


def _report_lines(rep: dict) -> list[str]:
    lines = [f"processor: {rep['topology']}"]
    lines += [f"  note: {n}" for n in rep["notes"]]
    windows = rep["source"].startswith("windows")
    favored = "SchedulingClass" if windows else "highest_perf"
    eff = "EfficiencyClass" if windows else "capacity/max kHz"
    lines.append(f"  {'core':>4}  {'processors':<12} {'class':<6} {eff:>16} {favored:>15} "
                 f"{'L2 group':>8}")
    for i, c in enumerate(rep["cores"]):
        lines.append(f"  {i:>4}  {cpu_ranges(c['cpus']):<12} {c['kind']:<6} "
                     f"{c['efficiency']:>16} {c['favored']:>15} {c['cluster']:>8}")
    if rep["raw"]:
        lines.append("  CPU sets (logical processor: core, efficiency class, scheduling class, "
                     "last-level cache, flags):")
        for e in rep["raw"]:
            flags = ", ".join(name for bit, name in CPU_SET_FLAG_NAMES if e["flags"] & bit)
            lines.append(f"    {e['lp']:>3}: core {e['core']}, group {e['group']}, efficiency "
                         f"{e['efficiency_class']}, scheduling {e['scheduling_class']}, LLC "
                         f"{e['llc']}" + (f" ({flags})" if flags else ""))
    allowed = rep["allowed"]
    lines.append(f"this process may run on processors "
                 f"{cpu_ranges(allowed) if allowed else 'unknown'}")
    for m, p in rep["plans"].items():
        lines.append(f"--cpu-binding {m}: {p['describe']}")
    lines.append(f"this process: priority {rep['priority'] or 'unknown'}, power throttling "
                 f"(ControlMask, StateMask) {rep['power_throttling'] or 'not readable'}, "
                 f"running on processor {rep['current_cpu']}")
    return lines


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="What this computer's processor reports, and where "
                                             "training would run its data workers.")
    ap.add_argument("--workers", type=int, default=None,
                    help="plan this many data workers (default: one per slot)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)
    rep = report(workers=args.workers)
    if args.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print("\n".join(_report_lines(rep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
