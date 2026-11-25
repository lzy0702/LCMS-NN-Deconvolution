"""Where the training process and its data workers run (lcmsdeconv.nn.cpus), checked on Linux.

The Windows structures are checked against the layouts of the Windows SDK's winnt.h and
processthreadsapi.h, and the parsers against byte buffers built independently with struct.pack
at those offsets (not with the ctypes classes under test); the Windows calls run against a fake
kernel32; the Linux detection against fake sysfs trees; the planner on the topologies of an
8 performance + 16 efficient core processor, Alder Lake with hyper-threading, and small ones.
"""

from __future__ import annotations

import ctypes
import functools
import json
import os
import pickle
import struct
import subprocess
import sys
import textwrap

import pytest

from lcmsdeconv.nn import cpus

# ------------------------------------------------------------------- layouts (64-bit Windows)


def offsets(cls) -> dict:
    return {name: getattr(cls, name).offset for name, *_ in cls._fields_}


def test_cpu_set_information_layout():
    assert ctypes.sizeof(cpus.SYSTEM_CPU_SET_INFORMATION) == 32
    assert ctypes.alignment(cpus.SYSTEM_CPU_SET_INFORMATION) == 8
    assert offsets(cpus.SYSTEM_CPU_SET_INFORMATION) == {
        "Size": 0, "Type": 4, "Id": 8, "Group": 12, "LogicalProcessorIndex": 14,
        "CoreIndex": 15, "LastLevelCacheIndex": 16, "NumaNodeIndex": 17, "EfficiencyClass": 18,
        "AllFlags": 19, "SchedulingClass": 20, "_Reserved": 21, "AllocationTag": 24}


def test_processor_and_cache_relationship_layouts():
    assert ctypes.sizeof(cpus.GROUP_AFFINITY) == 16
    assert offsets(cpus.GROUP_AFFINITY) == {"Mask": 0, "Group": 8, "Reserved": 10}
    assert offsets(cpus.PROCESSOR_RELATIONSHIP) == {"Flags": 0, "EfficiencyClass": 1,
                                                    "Reserved": 2, "GroupCount": 22,
                                                    "GroupMask": 24}
    assert ctypes.sizeof(cpus.PROCESSOR_RELATIONSHIP) == 40
    assert offsets(cpus.CACHE_RELATIONSHIP) == {"Level": 0, "Associativity": 1, "LineSize": 2,
                                                "CacheSize": 4, "Type": 8, "Reserved": 12,
                                                "GroupCount": 30, "GroupMask": 32}
    assert ctypes.sizeof(cpus.CACHE_RELATIONSHIP) == 48
    assert ctypes.sizeof(cpus.SLPI_EX_HEADER) == cpus.SLPI_EX_UNION_OFFSET == 8


def test_small_structures():
    assert ctypes.sizeof(cpus.POWER_THROTTLING_STATE) == 12
    assert ctypes.sizeof(cpus.PROCESSOR_NUMBER) == 4
    assert bytes(cpus.POWER_THROTTLING_STATE(1, 1, 0)) == struct.pack("<III", 1, 1, 0)


def test_wintypes_are_not_fixed_width_off_windows():
    """Why the structures use c_uint32: wintypes.DWORD is c_ulong, 8 bytes on LP64 Linux."""
    from ctypes import wintypes

    assert ctypes.sizeof(wintypes.DWORD) == ctypes.sizeof(ctypes.c_ulong)


# ------------------------------------------------------------- synthetic Windows buffers
CPU_SET_FMT = "<IIIHBBBBBBB3xQ"  # Size Type Id Group LP Core LLC Numa Eff Flags Sched pad Tag
assert struct.calcsize(CPU_SET_FMT) == 32


def cpu_set(lp, core, eff, group=0, llc=0, flags=0, sched=0, size=32):
    raw = struct.pack(CPU_SET_FMT, size, 0, 0x100 + lp + 64 * group, group, lp, core, llc, 0,
                      eff, flags, sched, 0)
    return raw + b"\xaa" * (size - 32)  # a later Windows's longer entry: skipped by its Size


def core_entry(lps, eff, group=0, smt=None):
    """A RelationProcessorCore entry: header, Flags, EfficiencyClass, Reserved[20], GroupCount,
    then GROUP_AFFINITY {Mask(8), Group(2), Reserved(6)}."""
    mask = sum(1 << lp for lp in lps)
    flags = cpus.LTP_PC_SMT if (smt if smt is not None else len(lps) > 1) else 0
    body = struct.pack("<BB20xH", flags, eff, 1) + struct.pack("<QH6x", mask, group)
    assert len(body) == 40
    return struct.pack("<II", cpus.RELATION_PROCESSOR_CORE, 8 + len(body)) + body


def cache_entry(lps, level=2, ctype=0, group=0, count=1):
    """A RelationCache entry: header, CACHE_RELATIONSHIP {Level, Associativity, LineSize,
    CacheSize, Type, Reserved[18], GroupCount, GroupMask}; ``count`` 0 as Windows before 10
    20H2 leaves it (Reserved[20] there, one mask)."""
    mask = sum(1 << lp for lp in lps)
    body = struct.pack("<BBHII18xH", level, 8, 64, 2 << 20, ctype, count)
    body += struct.pack("<QH6x", mask, group)
    assert len(body) == 48
    return struct.pack("<II", cpus.RELATION_CACHE, 8 + len(body)) + body


def arrow_lake_layout(interleaved: bool):
    """8 P-cores (EfficiencyClass 1) and 16 E-cores (0), no SMT: 24 logical processors. The
    numbering is an assumption either way (the parser must not rely on it): P-cores first, or
    pairs of P-cores between clusters of four E-cores."""
    if not interleaved:
        return [(lp, 1) for lp in range(8)] + [(lp, 0) for lp in range(8, 24)]
    kinds = ([1, 1] + [0] * 4) * 4
    return list(enumerate(kinds))


def arrow_lake_caches(layout):
    """Level-2 caches: one per P-core, one per cluster of four E-cores (in numbering order)."""
    out, e = [], []
    for lp, eff in layout:
        if eff == 1:
            out.append(cache_entry([lp]))
        else:
            e.append(lp)
            if len(e) == 4:
                out.append(cache_entry(e))
                e = []
    out.append(cache_entry([lp for lp, _ in layout], level=3))
    out.append(cache_entry([0], level=1, ctype=cpus.CACHE_DATA))
    return b"".join(out)


@pytest.mark.parametrize("interleaved", [False, True])
def test_arrow_lake_cpu_sets_with_their_caches(interleaved):
    layout = arrow_lake_layout(interleaved)
    buf = b"".join(cpu_set(lp, lp, eff, sched=(5 if lp in (6, 7) else 0)) for lp, eff in layout)
    caches = cpus.parse_caches(arrow_lake_caches(layout))
    topo = cpus.cores_from_cpu_sets(cpus.parse_cpu_sets(buf), caches)
    assert topo.source == "windows-cpu-sets" and topo.hybrid
    assert topo.counts() == {"P": 8, "E": 16} or topo.counts() == {"E": 16, "P": 8}
    assert sorted(c.cpus[0] for c in topo.cores if c.kind == "P") == \
        sorted(lp for lp, eff in layout if eff == 1)
    assert all(len(c.cpus) == 1 for c in topo.cores) and topo.logical_processors == 24
    # E-cores share a level-2 cache by four: four clusters; each P-core has one of its own
    e_clusters = {}
    for c in topo.cores:
        if c.kind == "E":
            e_clusters.setdefault(c.cluster, []).append(c.cpus[0])
    assert len(e_clusters) == 4 and all(len(v) == 4 for v in e_clusters.values())
    assert len({c.cluster for c in topo.cores if c.kind == "P"}) == 8
    assert topo.kind_of_cpu()[layout[0][0]] == "P"


def test_alder_lake_smt_cpu_sets_and_processor_info_agree():
    # 8 P-cores with two threads each (LPs 2i, 2i+1, CoreIndex 2i), 8 E-cores (16-23)
    sets = [cpu_set(2 * i + t, 2 * i, 1) for i in range(8) for t in (0, 1)]
    sets += [cpu_set(lp, lp, 0) for lp in range(16, 24)]
    a = cpus.cores_from_cpu_sets(cpus.parse_cpu_sets(b"".join(sets)))
    info = [core_entry((2 * i, 2 * i + 1), 1) for i in range(8)]
    info += [core_entry((lp,), 0) for lp in range(16, 24)]
    b = cpus.cores_from_processor_info(cpus.parse_processor_cores(b"".join(info)))
    assert [(c.cpus, c.kind) for c in a.cores] == [(c.cpus, c.kind) for c in b.cores]
    assert a.cores[0].cpus == (0, 1) and a.cores[0].kind == "P"
    assert a.cores[-1].cpus == (23,) and a.cores[-1].kind == "E"


def test_alike_cores_and_three_classes():
    flat = cpus.cores_from_cpu_sets(cpus.parse_cpu_sets(b"".join(cpu_set(i, i, 0)
                                                                 for i in range(8))))
    assert not flat.hybrid and {c.kind for c in flat.cores} == {"?"}
    # Meteor Lake-like: P (2), E (1), low-power E (0)
    three = [cpu_set(0, 0, 2), cpu_set(1, 1, 2), cpu_set(2, 2, 1), cpu_set(3, 3, 1),
             cpu_set(4, 4, 0), cpu_set(5, 5, 0)]
    topo = cpus.cores_from_cpu_sets(cpus.parse_cpu_sets(b"".join(three)))
    assert [c.kind for c in topo.cores] == ["P", "P", "E", "E", "LP", "LP"]


def test_other_groups_longer_entries_and_old_cache_entries():
    buf = b"".join([cpu_set(0, 0, 1, size=40), cpu_set(1, 1, 0, size=48),
                    cpu_set(0, 0, 1, group=1)])
    entries = cpus.parse_cpu_sets(buf)
    assert [e["lp"] for e in entries] == [0, 1, 0]
    topo = cpus.cores_from_cpu_sets(entries)
    assert [c.cpus for c in topo.cores] == [(0,), (1,)] and topo.notes
    info = core_entry((0,), 1) + cache_entry((0, 1)) + core_entry((1,), 0) + \
        core_entry((5,), 0, group=1)
    t2 = cpus.cores_from_processor_info(cpus.parse_processor_cores(info))
    assert [(c.cpus, c.kind) for c in t2.cores] == [((0,), "P"), ((1,), "E")] and t2.notes
    assert topo.other_groups == t2.other_groups == 1
    # more than 64 logical processors: psutil binds within group 0 only, so nothing is bound
    big = cpus.cores_from_cpu_sets(cpus.parse_cpu_sets(b"".join(
        [cpu_set(lp, lp, 0) for lp in range(64)] + [cpu_set(lp, lp, 0, group=1)
                                                    for lp in range(16)])))
    pl = cpus.plan(big, mode="core")
    assert pl.mode == "none" and "outside processor group 0" in pl.notes[0]
    assert len(pl.workers) == 78
    # a cache entry of a Windows before 10 20H2 (no GroupCount: zero) still has its one mask
    old = cpus.parse_caches(cache_entry((2, 3), count=0))
    assert old == [{"level": 2, "type": 0, "size": 2 << 20, "groups": [(0, 0b1100)]}]
    assert cpus.l2_groups(old + cpus.parse_caches(cache_entry((0, 1), group=1))) == [{2, 3}]


def test_malformed_buffers_raise():
    with pytest.raises(ValueError):
        cpus.parse_cpu_sets(struct.pack("<II", 4, 0) + b"\0" * 24)
    with pytest.raises(ValueError):
        cpus.parse_cpu_sets(cpu_set(0, 0, 0) + struct.pack("<II", 64, 0))  # entry overruns
    with pytest.raises(ValueError):
        cpus.parse_cpu_sets(cpu_set(0, 0, 0) + b"\0" * 4)  # bytes left over
    with pytest.raises(ValueError):
        bad = bytearray(core_entry((0,), 0))
        struct.pack_into("<H", bad, 8 + 22, 3)  # three group masks in a 48-byte entry
        cpus.parse_processor_cores(bytes(bad))
    with pytest.raises(ValueError):
        bad = bytearray(cache_entry((0,)))
        struct.pack_into("<H", bad, 8 + 30, 2)
        cpus.parse_caches(bytes(bad))


# --------------------------------------------------------------------------- fake kernel32
class FakeKernel32:
    """The kernel32 calls lcmsdeconv.nn.cpus makes, answering from prepared buffers and
    recording the calls (ctypes.byref arguments arrive as CArgObjects; ``_obj`` is the
    object)."""

    def __init__(self, cpu_sets: bytes | None, processor_info: bytes, caches: bytes = b""):
        self.cpu_sets, self.processor_info, self.caches = cpu_sets, processor_info, caches
        self.calls = []

    def GetCurrentProcess(self):
        return -1

    def GetCurrentThread(self):
        return -2

    def GetSystemCpuSetInformation(self, buf, n, need, process, flags):
        if self.cpu_sets is None:
            raise AttributeError("no such function (Windows before 10)")
        assert process is None and flags == 0
        if n < len(self.cpu_sets):
            need._obj.value = len(self.cpu_sets)
            return 0
        ctypes.memmove(buf, self.cpu_sets, len(self.cpu_sets))
        need._obj.value = len(self.cpu_sets)
        return 1

    def GetLogicalProcessorInformationEx(self, rel, buf, need):
        data = {cpus.RELATION_PROCESSOR_CORE: self.processor_info,
                cpus.RELATION_CACHE: self.caches}[rel]
        if need._obj.value < len(data):
            need._obj.value = len(data)
            return 0
        ctypes.memmove(buf, data, len(data))
        need._obj.value = len(data)
        return 1

    def _record(self, name, handle, cls, ptr, size):
        data = ctypes.string_at(ctypes.addressof(ptr._obj), size)
        self.calls.append((name, handle, cls, size, struct.unpack("<III", data)))
        return 1

    def SetProcessInformation(self, handle, cls, ptr, size):
        return self._record("SetProcessInformation", handle, cls, ptr, size)

    def SetThreadInformation(self, handle, cls, ptr, size):
        return self._record("SetThreadInformation", handle, cls, ptr, size)

    def GetProcessInformation(self, handle, cls, ptr, size):
        ptr._obj.ControlMask, ptr._obj.StateMask = 1, 0
        return 1

    def SetPriorityClass(self, handle, cls):
        self.calls.append(("SetPriorityClass", handle, cls))
        return 1

    def GetPriorityClass(self, handle):
        return cpus.BELOW_NORMAL_PRIORITY_CLASS


def test_windows_topology_with_fake_kernel32():
    layout = arrow_lake_layout(True)
    sets = b"".join(cpu_set(lp, lp, eff) for lp, eff in layout)
    topo = cpus.windows_topology(FakeKernel32(sets, b"", arrow_lake_caches(layout)))
    assert topo.source == "windows-cpu-sets" and topo.counts() == {"P": 8, "E": 16}
    assert len({c.cluster for c in topo.cores if c.kind == "E"}) == 4 and not topo.notes
    # no CPU sets (an older Windows): GetLogicalProcessorInformationEx instead
    info = b"".join(core_entry((lp,), eff) for lp, eff in arrow_lake_layout(False))
    topo2 = cpus.windows_topology(FakeKernel32(None, info))
    assert topo2.source == "windows-processor-info" and topo2.notes
    assert [c.kind for c in topo2.cores] == ["P"] * 8 + ["E"] * 16
    # nothing readable at all: detect() takes the cores as alike and says why, never raises
    broken = cpus.detect(FakeKernel32(None, b"\x01\x02\x03\x04\xff\xff\xff\xff"))
    assert broken.source == "uniform" and "detection failed" in broken.notes[0]


def test_power_throttling_and_priority_calls():
    k = FakeKernel32(b"", b"")
    assert cpus.set_power_throttling(False, k)  # opt out: HighQoS
    assert cpus.set_power_throttling(True, k)  # EcoQoS
    assert cpus.set_power_throttling(None, k)  # back to the system's heuristics
    assert cpus.set_power_throttling(False, k, thread=True)
    assert k.calls == [
        ("SetProcessInformation", -1, 4, 12, (1, 1, 0)),
        ("SetProcessInformation", -1, 4, 12, (1, 1, 1)),
        ("SetProcessInformation", -1, 4, 12, (1, 0, 0)),
        ("SetThreadInformation", -2, 3, 12, (1, 1, 0)),
    ]
    assert cpus.get_power_throttling(k) == (1, 0)
    k.calls.clear()
    assert cpus.restore_power_throttling((1, 0), k) and cpus.restore_power_throttling(None, k)
    assert [c[-1] for c in k.calls] == [(1, 1, 0), (1, 0, 0)]
    assert cpus.set_priority("below_normal", k) and k.calls[-1] == ("SetPriorityClass", -1,
                                                                   0x4000)
    assert cpus.set_priority("normal", k) and k.calls[-1] == ("SetPriorityClass", -1, 0x20)
    assert cpus.get_priority(k) == "below normal"
    with pytest.raises(ValueError):
        cpus.set_priority("idle", k)
    assert not cpus.set_power_throttling(False)  # off Windows, without a kernel32: nothing
    assert cpus.get_power_throttling() is None


def test_worker_init_follows_the_placement(monkeypatch):
    """Priority, then power throttling where asked, then the worker's own core(s); without a
    placement the priority alone, below normal, as lcmsdeconv.synth.dataset.worker_init."""
    calls = []
    monkeypatch.setattr(cpus, "set_priority", lambda p: calls.append(("priority", p)) or True)
    monkeypatch.setattr(cpus, "set_power_throttling",
                        lambda eco: calls.append(("eco", eco)) or True)
    monkeypatch.setattr(cpus, "pin_process", lambda c: calls.append(("pin", tuple(c))) or True)
    pl = cpus.plan(arrow_lake_topology(), mode="core", priority="normal", qos="high")
    cpus.worker_init(3, pl)
    assert calls == [("priority", "normal"), ("eco", False), ("pin", (5,))]
    assert cpus.WORKER_STATE == {"id": 3, "priority": True, "qos": True, "cpus": (5,),
                                 "pinned": True}
    calls.clear()
    cpus.worker_init(0)
    assert calls == [("priority", "below_normal")] and cpus.WORKER_STATE["id"] == 0
    calls.clear()
    assert cpus.apply_main(pl, bind=False) == {"qos": True} and calls == [("eco", False)]
    assert cpus.apply_main(pl, qos=False) == {"pinned": True} and calls[-1] == ("pin", (1, 2))
    none = cpus.plan(arrow_lake_topology(), mode="none")
    assert cpus.apply_main(none) == {}


# ------------------------------------------------------------------------ fake Linux sysfs
def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text + "\n")


def fake_sysfs(root, cores, pmus=None, capacity=None, maxfreq=None, cluster=None, l2=None):
    """``cores``: list of tuples of CPU numbers (thread siblings); ``l2``: lists of CPUs
    sharing a level-2 cache."""
    all_cpus = sorted(c for core in cores for c in core)
    write(root, "sys/devices/system/cpu/online", f"{all_cpus[0]}-{all_cpus[-1]}")
    for core in cores:
        for c in core:
            base = f"sys/devices/system/cpu/cpu{c}"
            write(root, f"{base}/topology/core_cpus_list", ",".join(map(str, core)))
            if capacity:
                write(root, f"{base}/cpu_capacity", str(capacity[c]))
            if maxfreq:
                write(root, f"{base}/cpufreq/cpuinfo_max_freq", str(maxfreq[c]))
            if cluster:
                write(root, f"{base}/topology/cluster_id", str(cluster[c]))
            for group in l2 or ():
                if c in group:
                    write(root, f"{base}/cache/index0/level", "1")
                    write(root, f"{base}/cache/index0/type", "Data")
                    write(root, f"{base}/cache/index0/shared_cpu_list", ",".join(map(str, core)))
                    write(root, f"{base}/cache/index2/level", "2")
                    write(root, f"{base}/cache/index2/type", "Unified")
                    write(root, f"{base}/cache/index2/shared_cpu_list",
                          ",".join(map(str, group)))
    for name, text in (pmus or {}).items():
        write(root, f"sys/devices/{name}/cpus", text)


def test_linux_alder_lake_pmu_and_level2_groups(tmp_path):
    cores = [(2 * i, 2 * i + 1) for i in range(8)] + [(c,) for c in range(16, 24)]
    l2 = [list(core) for core in cores[:8]] + [list(range(16, 20)), list(range(20, 24))]
    fake_sysfs(str(tmp_path), cores, pmus={"cpu_core": "0-15", "cpu_atom": "16-23"}, l2=l2)
    topo = cpus.linux_topology(str(tmp_path))
    assert topo.source == "linux-hybrid-pmu" and topo.counts() == {"P": 8, "E": 8}
    assert topo.cores[0].cpus == (0, 1)
    e = [c.cluster for c in topo.cores if c.kind == "E"]
    assert e[:4] == [e[0]] * 4 and e[4:] == [e[4]] * 4 and e[0] != e[4]
    # no cache directories: the kernel's cluster_id
    fake_sysfs(str(tmp_path / "b"), cores, pmus={"cpu_core": "0-15", "cpu_atom": "16-23"},
               cluster={c: (c - 16) // 4 + 10 if c >= 16 else c // 2 for c in range(24)})
    assert cpus.linux_topology(str(tmp_path / "b")).cores[-1].cluster == 11


def test_linux_arrow_lake_capacity_and_cpufreq(tmp_path):
    cores = [(c,) for c in range(24)]
    p = set(range(8))
    cap = {c: 1024 if c in p else 700 for c in range(24)}
    fake_sysfs(str(tmp_path / "a"), cores, capacity=cap)
    t = cpus.linux_topology(str(tmp_path / "a"))
    assert t.source == "linux-cpu-capacity" and t.counts() == {"P": 8, "E": 16}
    # cpufreq only, two favoured P-cores 200 MHz above the others: still one class
    freq = {c: (5_700_000 if c in (2, 3) else 5_500_000) if c in p else 4_600_000
            for c in range(24)}
    fake_sysfs(str(tmp_path / "b"), cores, maxfreq=freq)
    t = cpus.linux_topology(str(tmp_path / "b"))
    assert t.source == "linux-cpufreq-max" and t.counts() == {"P": 8, "E": 16}


def test_linux_alike_and_three_tiers(tmp_path):
    cores = [(c,) for c in range(8)]
    freq = {c: 5_100_000 + 50_000 * (c % 3) for c in range(8)}  # preferred-core spread only
    fake_sysfs(str(tmp_path / "a"), cores, maxfreq=freq)
    t = cpus.linux_topology(str(tmp_path / "a"))
    assert not t.hybrid and t.source == "uniform" and t.notes
    cap = {0: 1024, 1: 1024, 2: 800, 3: 800, 4: 800, 5: 400, 6: 400, 7: 400}
    fake_sysfs(str(tmp_path / "b"), cores, capacity=cap)
    t = cpus.linux_topology(str(tmp_path / "b"))
    assert [c.kind for c in t.cores] == ["P", "P", "E", "E", "E", "LP", "LP", "LP"]


def test_this_computer():
    t = cpus.detect()
    with open("/sys/devices/system/cpu/online") as fh:
        online = cpus.parse_cpu_list(fh.read())
    assert t.cores and t.logical_processors == len(online)
    assert cpus.current_cpu() in set(os.sched_getaffinity(0))
    assert cpus.allowed_cpus() == sorted(os.sched_getaffinity(0))
    assert cpus.affinity_of([os.getpid()]) == {os.getpid(): sorted(os.sched_getaffinity(0))}


def test_detect_never_raises(monkeypatch):
    def broken(root="/"):
        raise OSError("no sysfs")

    monkeypatch.setattr(cpus, "linux_topology", broken)
    t = cpus.detect()
    assert t.source == "uniform" and "detection failed" in t.notes[0] and t.cores


def test_parse_cpu_list():
    assert cpus.parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert cpus.parse_cpu_list("") == []
    assert cpus.cpu_ranges([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"


# ------------------------------------------------------------------------------- planner
def arrow_lake_topology(interleaved=False, favored=()):
    buf = b"".join(cpu_set(lp, lp, eff, sched=9 if lp in favored else 0)
                   for lp, eff in arrow_lake_layout(interleaved))
    return cpus.cores_from_cpu_sets(cpus.parse_cpu_sets(buf))


def test_plan_8p_16e_core_mode():
    pl = cpus.plan(arrow_lake_topology(), mode="core")
    assert pl.main == (1, 2)  # two P-cores, not processor 0
    assert len(pl.workers) == 22 and pl.kinds == ("P",) * 6 + ("E",) * 16
    assert list(pl.workers[:6]) == [(0,), (3,), (4,), (5,), (6,), (7,)]
    assert len(set(pl.workers)) == 22  # one worker per core
    assert not set(pl.main) & {c for w in pl.workers for c in w}
    assert pl.label(1) == "performance core, processor 3" and pl.label(6) == \
        "efficient core, processor 8"


def test_the_scheduling_class_is_reported_not_used():
    """SchedulingClass is not documented: favoured cores do not change the plan."""
    topo = arrow_lake_topology(favored=(6, 7))
    assert {c.cpus[0]: c.favored for c in topo.cores}[6] == 9
    assert cpus.plan(topo, mode="core") == cpus.plan(arrow_lake_topology(), mode="core")


def test_plan_interleaved_pool_and_fewer_workers():
    topo = arrow_lake_topology(interleaved=True)
    pl = cpus.plan(topo, workers=10, mode="pool")
    p_cpus = sorted(c.cpus[0] for c in topo.cores if c.kind == "P")
    assert pl.main == (1, 6) and set(pl.main) <= set(p_cpus)
    assert pl.kinds == ("P",) * 6 + ("E",) * 4
    assert len(set(pl.workers)) == 1 and len(pl.workers[0]) == 10  # one shared pool
    assert not set(pl.main) & set(pl.workers[0])
    more = cpus.plan(topo, workers=30, mode="core")  # more workers than cores: wrap round
    assert len(more.workers) == 30 and more.workers[22] == more.workers[0]


def test_plan_none_binds_nothing_and_counts_as_the_others():
    topo = arrow_lake_topology()
    pl = cpus.plan(topo, mode="none")
    assert pl.mode == "none" and not pl.main and len(pl.workers) == 22
    assert not any(pl.workers) and pl.label(0) == "" and pl.worker_cpus(5) == ()
    assert "22 data workers not bound to cores" in pl.describe()
    with pytest.raises(ValueError):
        cpus.plan(topo, mode="cores")
    with pytest.raises(ValueError):
        cpus.plan(topo, qos="eco")


def test_plan_per_core_and_restricted_affinity():
    topo = arrow_lake_topology()
    pl = cpus.plan(topo, mode="core", per_core={"P": 2})
    assert len(pl.workers) == 28 and pl.workers[0] == pl.workers[1] == (0,)
    pl2 = cpus.plan(topo, allowed=range(8, 24), mode="core")  # started on the E-cores only
    assert pl2.main == (8, 9) and len(pl2.workers) == 14 and set(pl2.kinds) == {"E"}


def test_plan_alder_lake_with_hyper_threading():
    cores = [cpus.Core(cpus=(2 * i, 2 * i + 1), kind="P", rank=0) for i in range(8)] + \
        [cpus.Core(cpus=(c,), kind="E", rank=1) for c in range(16, 24)]
    smt = cpus.plan(cpus.Topology(tuple(cores), "test"), mode="core")
    assert smt.main == (2, 3, 4, 5) and len(smt.workers) == 6 * 2 + 8
    assert smt.workers[0] == smt.workers[1] == (0, 1)  # two workers on a core's two threads
    assert smt.kinds.count("P") == 12 and smt.kinds.count("E") == 8


def test_plan_uniform_four_cores_and_two_cores():
    pl = cpus.plan(cpus.uniform_topology(4), mode="core")
    assert pl.main == (1,) and pl.workers == ((0,), (2,), (3,))
    assert set(pl.kinds) == {"?"} and pl.label(0) == "core, processor 0"
    two = cpus.plan(cpus.uniform_topology(2), mode="core")
    assert two.mode == "none" and two.notes == ("only 2 usable cores",)
    assert len(two.workers) == 1 and not two.main


def test_worker_slots_and_the_processors_cap():
    from lcmsdeconv.nn.machine import cpu_workers

    assert [cpu_workers(c) for c in (1, 2, 3, 4, 8, 12, 24)] == [1, 1, 2, 3, 6, 10, 22]
    assert cpus.worker_slots(arrow_lake_topology()) == 22
    assert cpu_workers() == cpus.worker_slots()  # this computer, as it reports itself


def test_placement_pickles_and_round_trips():
    pl = cpus.plan(arrow_lake_topology(), mode="core", priority="normal", qos="high")
    fn = functools.partial(cpus.worker_init, placement=pl)
    again = pickle.loads(pickle.dumps(fn))
    assert again.keywords["placement"] == pl
    assert cpus.Placement.from_json(pl.to_json()) == pl
    text = pl.describe()
    assert "22 data workers each on its own core (6 performance, 16 efficient)" in text
    assert "normal priority" in text and "opted out of power throttling" in text


# ---------------------------------------------------------------------- live, this Linux
def test_pin_process_binds_every_thread():
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("one processor")
    code = textwrap.dedent("""
        import os, threading
        from lcmsdeconv.nn import cpus
        stop = threading.Event()
        ts = [threading.Thread(target=stop.wait) for _ in range(3)]
        [t.start() for t in ts]
        target = sorted(os.sched_getaffinity(0))[-1:]
        ok = cpus.pin_process(target)
        late = threading.Thread(target=stop.wait); late.start()
        masks = set()
        for tid in os.listdir('/proc/self/task'):
            with open(f'/proc/self/task/{tid}/status') as fh:
                line = [x for x in fh if x.startswith('Cpus_allowed_list')][0]
            masks.add(line.split()[1])
        stop.set()
        print(ok, sorted(masks), target, cpus.current_cpu())
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=120)
    ok, masks, target, now = out.stdout.split(" ", 3)
    assert ok == "True", out.stderr
    cpu = sorted(os.sched_getaffinity(0))[-1]
    assert (masks, target, now.strip()) == (f"['{cpu}']", f"[{cpu}]", str(cpu))


def test_the_report_runs():
    out = subprocess.run([sys.executable, "-m", "lcmsdeconv.nn.cpus"], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("processor: ") and "--cpu-binding core:" in out.stdout
    js = subprocess.run([sys.executable, "-m", "lcmsdeconv.nn.cpus", "--json", "--workers", "2"],
                        capture_output=True, text=True, timeout=120)
    rep = json.loads(js.stdout)
    assert set(rep["plans"]) == {"none", "pool", "core"} and rep["cores"]
    assert all(len(p["workers"]) == 2 for p in rep["plans"].values())
