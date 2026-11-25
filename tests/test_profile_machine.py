"""The profiling package (gpu_training/profile_machine.py and gpu_training/profkit): the runner's
isolation, timeouts and packing; the Windows CPU-set parser on synthetic buffers and the Windows
calls through stand-in DLL functions; the processor experiments and the order-preserving
dispatcher on this computer with short spectra; the GPU experiments' code on the processor.
Every check runs without a GPU, nvidia-smi or Windows: those are stood in for."""

from __future__ import annotations

import ctypes
import gzip
import importlib.util
import json
import os
import struct
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "gpu_training"
if str(GPU) not in sys.path:
    sys.path.insert(0, str(GPU))

from profkit import common, cpusets, environment, smi  # noqa: E402

#: short spectra: the recipe's items at this crop take a tenth of a second
CROP = 4096


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """One PyTorch thread, as the runner gives every experiment (and training its processes)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _runner():
    spec = importlib.util.spec_from_file_location("_profile_machine", GPU / "profile_machine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ Windows CPU sets
#: a hybrid processor of 8 performance cores (class 1) and 16 efficient ones (class 0) in an
#: interleaved order, as Arrow Lake numbers them
HYBRID = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]


def _documented_entry(cid, group, lp, core, eff, flags=0, scheduling=0, tag=0, size=32,
                      kind=0, llc=0, numa=0) -> bytes:
    """An entry laid out field by field as the Windows SDK documents SYSTEM_CPU_SET_INFORMATION
    (independently of the parser's own struct)."""
    head = struct.pack("<II", size, kind)
    body = (struct.pack("<I", cid) + struct.pack("<H", group)
            + bytes([lp, core, llc, numa, eff, flags]) + struct.pack("<I", scheduling)
            + struct.pack("<Q", tag))
    assert len(head + body) == 32
    data = head + body
    return data[:size] if size < 32 else data + b"\xee" * (size - 32)


def _hybrid_buffer() -> bytes:
    return b"".join(_documented_entry(256 + i, 0, i, i, eff, flags=0x2 if i == 3 else 0,
                                      scheduling=i % 3, tag=7 * i)
                    for i, eff in enumerate(HYBRID))


def test_the_cpu_set_layout_is_the_documented_one():
    assert cpusets.CPU_SET_SIZE == 32
    a = cpusets.pack_cpu_set(300, 1, 5, 4, 1, llc=2, numa=3, flags=0x9, scheduling=6, tag=99)
    b = _documented_entry(300, 1, 5, 4, 1, flags=0x9, scheduling=6, tag=99, llc=2, numa=3)
    assert a == b
    # the fields at their offsets: logical processor 14, efficiency class 18, flags 19,
    # scheduling class 20, allocation tag 24
    assert (a[14], a[18], a[19], a[20]) == (5, 1, 0x9, 6)
    assert struct.unpack_from("<Q", a, 24)[0] == 99


def test_cpu_sets_of_a_hybrid_processor_are_read_in_the_buffers_order():
    sets = cpusets.parse_cpu_sets(_hybrid_buffer())
    assert [s["logical"] for s in sets] == list(range(24))
    assert [s["efficiency_class"] for s in sets] == HYBRID
    assert [s["id"] for s in sets] == list(range(256, 280))
    assert sets[3]["flags"] == ["allocated"] and sets[4]["flags"] == []
    assert [s["scheduling_class"] for s in sets[:4]] == [0, 1, 2, 0]
    assert sets[5]["allocation_tag"] == 35


def test_longer_entries_and_other_types_are_walked_by_their_size():
    buf = (_documented_entry(256, 0, 0, 0, 1, size=40)  # a later Windows' longer entry
           + struct.pack("<II", 16, 7) + b"\x11" * 8  # an entry of another type
           + _documented_entry(257, 0, 1, 1, 0, flags=0x1, size=48))
    sets = cpusets.parse_cpu_sets(buf)
    assert [(s["logical"], s["efficiency_class"], s["flags"]) for s in sets] == [
        (0, 1, []), (1, 0, ["parked"])]


@pytest.mark.parametrize("buf", [
    _documented_entry(256, 0, 0, 0, 1)[:-1],  # cut inside an entry
    _documented_entry(256, 0, 0, 0, 1) + b"\0\0\0",  # bytes after the last entry
    struct.pack("<II", 4, 0),  # an entry shorter than its own header
    _documented_entry(256, 0, 0, 0, 1, size=24),  # a CPU set shorter than its fields
])
def test_a_malformed_buffer_is_refused(buf):
    with pytest.raises(ValueError):
        cpusets.parse_cpu_sets(buf)


def _fake_cpu_set_dll(data: bytes, calls: list):
    """GetSystemCpuSetInformation as kernel32 answers it: without a buffer (or with one too
    small) it fails and names the size needed; with one large enough it fills it."""
    proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.c_uint32)

    def get(buf, length, returned, process, flags):
        calls.append((bool(buf), int(length), process, int(flags)))
        returned[0] = len(data)
        if not buf or length < len(data):
            return 0
        ctypes.memmove(buf, data, len(data))
        return 1

    fn = proto(get)
    return SimpleNamespace(GetSystemCpuSetInformation=fn, _keep=get)


def test_the_windows_call_asks_for_the_size_then_reads_every_processor():
    calls: list = []
    data = _hybrid_buffer() + _documented_entry(400, 1, 0, 0, 1)  # one in processor group 1
    k32 = _fake_cpu_set_dll(data, calls)
    sets = cpusets.windows_cpu_sets(kernel32=k32)
    assert calls == [(False, 0, None, 0), (True, len(data), None, 0)]
    assert len(sets) == 25 and sets[-1]["group"] == 1
    procs = cpusets.logical_processors(kernel32=k32)
    assert procs["source"] == "GetSystemCpuSetInformation"
    assert procs["counts"] == {"performance": 9, "efficient": 16}
    assert [p["cpu"] for p in procs["processors"][:24]] == list(range(24))
    assert procs["processors"][-1]["cpu"] is None  # group 1: no affinity bit in group 0
    assert [p["kind"] for p in procs["processors"][:3]] == ["performance", "performance",
                                                             "efficient"]


# ------------------------------------------------------------------ Linux
def _sysfs(tmp_path: Path, online: str, files: dict) -> Path:
    root = tmp_path / "sys"
    (root / "devices" / "system" / "cpu").mkdir(parents=True)
    (root / "devices" / "system" / "cpu" / "online").write_text(online + "\n")
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{text}\n")
    return root


def test_cpu_lists_are_read_as_the_kernel_writes_them():
    assert cpusets.parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert cpusets.parse_cpulist("5\n") == [5] and cpusets.parse_cpulist("") == []


def test_linux_hybrid_processors_are_named_by_cpu_core_and_cpu_atom(tmp_path):
    root = _sysfs(tmp_path, "0-23", {"devices/cpu_core/cpus": "0-7",
                                      "devices/cpu_atom/cpus": "8-23"})
    out = cpusets.linux_processors(root)
    cpusets.kind_names(out["processors"])
    assert out["source"] == "sysfs cpu_core/cpu_atom"
    assert [p["efficiency_class"] for p in out["processors"]] == [1] * 8 + [0] * 16
    assert out["processors"][0]["kind"] == "performance"
    assert out["processors"][8]["kind"] == "efficient"


def test_linux_falls_back_to_capacity_then_to_the_highest_frequency(tmp_path):
    caps = {f"devices/system/cpu/cpu{c}/cpu_capacity": 1024 if c < 2 else 512 for c in range(4)}
    out = cpusets.linux_processors(_sysfs(tmp_path / "a", "0-3", caps))
    assert out["source"] == "sysfs cpu_capacity"
    assert [p["efficiency_class"] for p in out["processors"]] == [1, 1, 0, 0]
    freqs = {f"devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq": 5700000 if c % 2 else 4600000
             for c in range(4)}
    out = cpusets.linux_processors(_sysfs(tmp_path / "b", "0-3", freqs))
    assert out["source"] == "sysfs cpufreq"
    assert [p["efficiency_class"] for p in out["processors"]] == [0, 1, 0, 1]
    same = {f"devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq": 4000000 for c in range(4)}
    out = cpusets.linux_processors(_sysfs(tmp_path / "c", "0-3", same))
    cpusets.kind_names(out["processors"])
    assert out["source"] == "none"
    assert {p["kind"] for p in out["processors"]} == {"uniform"}


# ------------------------------------------------------------------ pinning
@pytest.mark.parametrize("workers, main, perf_workers, idle", [
    (20, 2, 6, 2),  # two performance cores for training; 6 + 14 for workers; 2 efficient idle
    (22, 2, 6, 0),
    (23, 1, 7, 0),  # one core for training when the workers need all the others
])
def test_pinning_gives_training_performance_cores_and_workers_one_core_each(
        monkeypatch, workers, main, perf_workers, idle):
    monkeypatch.setattr(cpusets, "current_affinity", lambda: None)
    procs = [{"cpu": i, "efficiency_class": e} for i, e in enumerate(HYBRID)]
    cpusets.kind_names(procs)
    plan = cpusets.pin_plan(procs, workers)
    kinds = {p["cpu"]: p["kind"] for p in procs}
    assert len(plan["main"]) == main and {kinds[c] for c in plan["main"]} == {"performance"}
    assert plan["main"] == [0, 1][:main]
    assert len(plan["workers"]) == workers and not plan["shared"]
    assert len(set(plan["workers"]) | set(plan["main"])) == workers + main  # no core twice
    assert plan["worker_kinds"] == {"efficient": workers - perf_workers,
                                    "performance": perf_workers}
    assert len(plan["idle"]) == idle and all(kinds[c] == "efficient" for c in plan["idle"])


# ------------------------------------------------------------------ priority and throttling
class _FakeProcessDll:
    """GetCurrentProcess, Set/GetProcessInformation and Set/GetPriorityClass as kernel32 answers
    them, recording what they were asked."""

    HANDLE = 0x1234

    def __init__(self):
        self.calls: list = []
        self.state = b"\0" * 12
        self.priority = 0x20
        P = ctypes.CFUNCTYPE
        self._fns = [
            P(ctypes.c_void_p)(lambda: self.HANDLE),
            P(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)(
                self._set_info),
            P(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)(
                self._get_info),
            P(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)(self._set_priority),
            P(ctypes.c_uint32, ctypes.c_void_p)(lambda h: self.priority)]
        (self.GetCurrentProcess, self.SetProcessInformation, self.GetProcessInformation,
         self.SetPriorityClass, self.GetPriorityClass) = self._fns

    def _set_info(self, handle, cls, ptr, size):
        self.state = ctypes.string_at(ptr, size)
        self.calls.append(("set", handle, cls, size, struct.unpack("<III", self.state)))
        return 1

    def _get_info(self, handle, cls, ptr, size):
        self.calls.append(("get", handle, cls, size))
        ctypes.memmove(ptr, self.state, size)
        return 1

    def _set_priority(self, handle, cls):
        self.calls.append(("priority", handle, cls))
        self.priority = cls
        return 1


@pytest.mark.parametrize("mode, masks", [("opt_out", (1, 0)), ("eco", (1, 1)),
                                         ("default", (0, 0))])
def test_power_throttling_is_set_with_the_sdk_constants(mode, masks):
    k32 = _FakeProcessDll()
    got = cpusets.set_power_throttling(mode, kernel32=k32)
    # ProcessPowerThrottling = 4; PROCESS_POWER_THROTTLING_STATE {Version 1, ControlMask,
    # StateMask}, 12 bytes; ControlMask EXECUTION_SPEED (1) with StateMask 0 is the opt-out
    assert k32.calls[0] == ("set", _FakeProcessDll.HANDLE, 4, 12, (1, *masks))
    assert k32.calls[1] == ("get", _FakeProcessDll.HANDLE, 4, 12)
    assert got["set"] and got["readback"] == {"version": 1, "control_mask": masks[0],
                                              "state_mask": masks[1]}


def test_priority_classes_are_those_of_training_workers():
    k32 = _FakeProcessDll()
    assert cpusets.set_priority("below_normal", kernel32=k32)["now"] == "below_normal"
    assert cpusets.set_priority("normal", kernel32=k32)["now"] == "normal"
    assert [c[2] for c in k32.calls] == [0x4000, 0x20]


def test_on_linux_normal_is_the_starting_nice_and_never_less(monkeypatch):
    if not hasattr(os, "setpriority"):
        pytest.skip("POSIX only")
    asked = []
    monkeypatch.setattr(cpusets, "WINDOWS", False)
    monkeypatch.setattr(cpusets, "BASE_NICE", 10)
    monkeypatch.setattr(os, "setpriority", lambda which, who, value: asked.append(value))
    cpusets.set_priority("below_normal")
    cpusets.set_priority("normal")
    assert asked == [15, 10]
    with pytest.raises(cpusets.Unavailable):
        cpusets.set_power_throttling("opt_out")


# ------------------------------------------------------------------ the environment record
@pytest.mark.parametrize("text, guid, name, known", [
    ("Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\r\n",
     "381b4222-f694-41f0-9685-ff5bb260df2e", "Balanced", "Balanced"),
    ("电源方案 GUID: 8C5E7FDA-E8BF-4A96-9A85-A6E23A8C635C  (高性能)",
     "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c", "高性能", "High performance"),
    ("电源方案 GUID: e9a42b02-d5df-448d-aa00-03f14749eb61 （卓越性能）",
     "e9a42b02-d5df-448d-aa00-03f14749eb61", "卓越性能", "Ultimate Performance"),
    ("GUID: 11111111-2222-3333-4444-555555555555",
     "11111111-2222-3333-4444-555555555555", None, None),
])
def test_the_power_scheme_is_read_in_any_language(text, guid, name, known):
    got = environment.parse_power_scheme(text)
    assert (got["guid"], got["name"], got["known_as"]) == (guid, name, known)
    assert environment.parse_power_scheme("powercfg: access denied") == {}


def test_console_output_is_decoded_without_failing():
    assert environment.decode_console("高性能".encode()) == "高性能"
    text = environment.decode_console("高性能".encode("gbk"))  # Chinese Windows' code page 936
    if sys.platform == "win32":
        assert isinstance(text, str)  # as the console's code page reads it
    else:
        assert "�" in text  # no code page to try here: replaced, never an exception


# ------------------------------------------------------------------ nvidia-smi
def _fake_smi(tmp_path: Path) -> str:
    """An nvidia-smi that knows the throttle reasons only by their old name."""
    values = {"name": "Fake GPU 4080", "driver_version": "621.24", "memory.total": "16376",
              "utilization.gpu": "97", "utilization.memory": "41", "clocks.sm": "2745",
              "clocks.mem": "11501", "power.draw": "281.50", "temperature.gpu": "62",
              "memory.used": "13012", "pstate": "P2",
              "clocks_throttle_reasons.active": "0x0000000000000004"}
    script = tmp_path / "nvidia-smi"
    script.write_text(f"""#!{sys.executable}
import sys
values = {values!r}
fields = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--query-gpu=")][0].split(",")
unknown = [f for f in fields if f not in values]
if unknown:
    print(f'Field "{{unknown[0]}}" is not a valid field to query.')
    sys.exit(2)
print(", ".join(values[f] for f in fields))
""")
    script.chmod(0o755)
    return str(script)


def test_the_sampler_reads_the_fields_this_driver_knows(tmp_path):
    if sys.platform == "win32":
        pytest.skip("the stand-in nvidia-smi is a script")
    path = _fake_smi(tmp_path)
    s = smi.Sampler(smi=path, interval=0.05)
    assert s.fields["clocks_event_reasons.active"] == "clocks_throttle_reasons.active"
    s.start()
    time.sleep(0.6)
    rows = s.stop()
    assert len(rows) >= 3 and rows[0]["clocks.sm"] == 2745.0 and rows[0]["pstate"] == "P2"
    out = smi.summarise(rows, list(s.fields))
    assert out["samples"] == len(rows) and out["utilization.gpu"]["mean"] == 97.0
    assert out["clock_event_reasons"] == {"sw_power_cap": 1.0}
    assert out["pstate"] == {"P2": 1.0}
    s.write_csv(tmp_path / "s.csv")
    assert (tmp_path / "s.csv").read_text().splitlines()[0].startswith("t,utilization.gpu")
    desc = smi.describe(path)
    assert desc["available"] and desc["gpus"][0]["name"] == "Fake GPU 4080"
    assert "vbios_version" in desc["unknown_fields"]


def test_clock_event_reasons_are_decoded():
    assert smi.decode_reasons("0x0000000000000000") == []
    assert smi.decode_reasons(0x1 | 0x4 | 0x40) == ["gpu_idle", "sw_power_cap",
                                                    "hw_thermal_slowdown"]
    assert smi.decode_reasons(0x1000) == ["other_0x1000"]
    rows = [{"t": 0.0, "clocks_event_reasons.active": 0}, {"t": 0.5,
                                                           "clocks_event_reasons.active": 4}]
    assert smi.summarise(rows)["clock_event_reasons"] == {"none": 0.5, "sw_power_cap": 0.5}


# ------------------------------------------------------------------ the runner
def _plan(pm, *specs):
    out = []
    for eid, fn, params, timeout in specs:
        e = pm.entry(eid, "cpu", eid, f"tests._profile_fakes:{fn}", "env", params, smoke=True)
        e["timeout"] = timeout
        out.append(e)
    return out


def test_each_experiment_runs_in_a_process_of_its_own_and_failures_do_not_stop_the_others(
        tmp_path):
    import psutil

    pm = _runner()
    run = tmp_path / "run"
    runner = pm.Runner(run, pm.Console(tmp_path / "runner.log"), True, 600.0,
                       {"started": "now"})
    plan = _plan(pm, ("t.ok", "ok", {"value": 7}, 60), ("t.fail", "fail", {}, 60),
                 ("t.crash", "crash", {}, 60), ("t.hang", "hang", {}, 3),
                 ("t.none", "unavailable", {}, 60), ("t.after", "ok", {"value": 8}, 60))
    t = time.time()
    runner.run_all(plan)
    assert time.time() - t < 60
    got = {r["id"]: r for r in runner.records}
    assert [r["outcome"] for r in runner.records] == ["ok", "error", "crashed", "timeout",
                                                      "unavailable", "ok"]
    assert got["t.ok"]["result"]["value"] == 7 and got["t.after"]["result"]["value"] == 8
    assert got["t.ok"]["result"]["pid"] != os.getpid()
    assert "ValueError: this experiment fails on purpose" in got["t.fail"]["error"]
    assert "exit code 3" in got["t.crash"]["error"] and "about to crash" in got["t.crash"]["error"]
    assert "no result within 3 s" in got["t.hang"]["error"]
    assert got["t.none"]["error"].startswith("no CUDA")
    grandchild = int((run / "work" / "grandchild.pid").read_text())
    if psutil.pid_exists(grandchild):  # stopped with its parent (at most a zombie)
        assert psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE
    saved = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in saved["experiments"]] == [e["id"] for e in plan]
    assert got["t.ok"]["config"] == {"value": 7} and got["t.ok"]["process"]["env"][
        "OMP_NUM_THREADS"] == "1"
    assert "working on 7" in (tmp_path / "runner.log").read_text(encoding="utf-8")
    assert (run / "logs" / "t.fail.log").exists()


def test_what_needs_a_failed_experiment_or_more_time_than_is_left_is_skipped(tmp_path):
    pm = _runner()
    runner = pm.Runner(tmp_path / "run", pm.Console(tmp_path / "runner.log"), True, 100.0,
                       {"started": "now"})
    plan = _plan(pm, ("a", "fail", {}, 60), ("b", "ok", {}, 60), ("c", "ok", {}, 60),
                 ("d", "ok", {}, 60), ("e", "unavailable", {}, 60), ("f", "ok", {}, 60))
    plan[1]["needs"] = ["a"]
    plan[2]["estimate"] = 1000  # more than the budget
    plan[3]["estimate"], plan[3]["essential"] = 1000, True  # runs all the same
    plan[5]["needs"] = ["e"]
    runner.run_all(plan)
    out = {r["id"]: (r["outcome"], r["error"]) for r in runner.records}
    assert out["b"][0] == "skipped" and "needs a, and it was error" in out["b"][1]
    assert out["c"][0] == "skipped" and "time budget" in out["c"][1]
    assert out["d"][0] == "ok"
    assert out["f"][0] == "unavailable" and "no CUDA" in out["f"][1]


def test_a_process_that_does_not_end_when_stopped_is_recorded_and_left_behind(
        tmp_path, monkeypatch):
    """A process stuck where it cannot be stopped (in a GPU driver call, say): the runner
    records the timeout and goes on to the next experiment."""
    import psutil

    pm = _runner()
    monkeypatch.setattr(pm.common, "kill_tree", lambda pid, timeout=10.0: [])  # it never ends
    monkeypatch.setattr(pm, "STOP_WAIT", 0.5)
    runner = pm.Runner(tmp_path / "run", pm.Console(tmp_path / "runner.log"), True, 600.0,
                       {"started": "now"})
    before = {p.pid for p in psutil.Process().children(recursive=True)}
    try:
        runner.run_all(_plan(pm, ("t.hang", "hang", {}, 2), ("t.after", "ok", {"value": 3}, 60)))
        out = {r["id"]: r for r in runner.records}
        assert out["t.hang"]["outcome"] == "timeout"
        assert "did not end within 0.5 s of being stopped" in out["t.hang"]["error"]
        assert out["t.after"]["outcome"] == "ok" and out["t.after"]["result"]["value"] == 3
    finally:  # the process left behind, and the one it started
        for p in psutil.Process().children(recursive=True):
            if p.pid not in before:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass


def test_results_held_open_by_another_program_never_stop_the_profile(tmp_path, monkeypatch):
    """results.json open in another program (Excel, say): the experiments go on, it is written
    again after each, and if the last save fails too the records go to results_final.json."""
    pm = _runner()
    real = pm.write_json
    held = {"on": True}

    def write_json(path, obj, attempts=20):
        if Path(path).name == "results.json" and held["on"]:
            raise PermissionError(13, "the file is open in another program")
        return real(path, obj, attempts=1)

    monkeypatch.setattr(pm, "write_json", write_json)
    run = tmp_path / "run"
    runner = pm.Runner(run, pm.Console(tmp_path / "runner.log"), True, 600.0, {"started": "now"})
    runner.run_all(_plan(pm, ("a", "ok", {"value": 1}, 60), ("b", "ok", {"value": 2}, 60)))
    assert [r["outcome"] for r in runner.records] == ["ok", "ok"]
    assert "PermissionError" in runner.save_error and not (run / "results.json").exists()
    log = (tmp_path / "runner.log").read_text(encoding="utf-8")
    assert log.count("results.json could not be written") == 1  # said once, not after each
    final = runner.save({"finished": "then"}, final=True)
    assert final == run / "results_final.json"
    saved = json.loads(final.read_text(encoding="utf-8"))
    assert [r["id"] for r in saved["experiments"]] == ["a", "b"]
    assert "PermissionError" in saved["results_json_error"]
    held["on"] = False  # released: the next save writes results.json again
    assert runner.save() == run / "results.json" and runner.save_error is None


def test_a_fault_of_the_runner_itself_is_recorded_and_the_next_experiment_runs(tmp_path):
    pm = _runner()
    runner = pm.Runner(tmp_path / "run", pm.Console(tmp_path / "runner.log"), True, 600.0,
                       {"started": "now"})
    launch = runner.launch

    def faulty(e):
        if e["id"] == "a":
            raise RuntimeError("the runner trips over something")
        return launch(e)

    runner.launch = faulty
    runner.run_all(_plan(pm, ("a", "ok", {}, 60), ("b", "ok", {"value": 5}, 60)))
    out = {r["id"]: r for r in runner.records}
    assert out["a"]["outcome"] == "error" and "the runner failed" in out["a"]["error"]
    assert "the runner trips over something" in out["a"]["error"]
    assert out["b"]["outcome"] == "ok" and out["b"]["result"]["value"] == 5


def test_the_upload_holds_results_logs_and_csv_and_nothing_large(tmp_path, monkeypatch):
    pm = _runner()
    run = tmp_path / "runs" / "profile_x"
    for rel, data in {"results.json": json.dumps({"where": str(ROOT / "gpu_training"),
                                                  "home": str(Path.home() / "x")}),
                      "summary.txt": f"run from {ROOT}\n", "logs/a.log": f"{ROOT}/src\n",
                      "csv/smi.csv": "t,v\n0,1\n", "work/batches/b000.pt": "x" * 2000,
                      "e2e/ladder_e2e.log.jsonl": "{}\n", "e2e/ladder_e2e.pt": "weights",
                      "raw/a.json": "{}"}.items():
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding="utf-8")
    (run / "traces").mkdir()
    with gzip.open(run / "traces" / "t.json.gz", "wt") as fh:
        fh.write("{}")
    big = run / "logs" / "huge.log"
    big.write_bytes(b"0" * 2048)
    monkeypatch.setattr(pm, "MAX_FILE_MB", 1024 / 2**20)
    info = pm.pack(run, tmp_path / "up.zip")
    with zipfile.ZipFile(tmp_path / "up.zip") as zf:
        names = {n.split("/", 1)[1] for n in zf.namelist()}
        results = zf.read("profile_x/results.json").decode()
        summary = zf.read("profile_x/summary.txt").decode()
    assert {"results.json", "summary.txt", "logs/a.log", "csv/smi.csv", "raw/a.json",
            "traces/t.json.gz", "e2e/ladder_e2e.log.jsonl", "manifest.json"} <= names
    assert not any(n.startswith("work/") or n.endswith(".pt") for n in names)
    assert "logs/huge.log" not in names
    assert {e["file"] for e in info["left_out"]} >= {"work/batches/b000.pt",
                                                     "e2e/ladder_e2e.pt", "logs/huge.log"}
    assert str(ROOT) not in results and "<package>" in results and "<package>" in summary
    assert str(Path.home()) not in results


def _steps_record(name, ms, same=True, cap_error=False, speedups=(), chunk=None):
    return {"id": f"gpu.steps.window.{name}", "outcome": "ok",
            "config": {"speedups": list(speedups), "chunk": chunk},
            "result": {"median_ms": ms, "agreement": {"same": same},
                       "cap": {"error": "out of GPU memory"} if cap_error else {}}}


def test_the_best_configuration_is_the_fastest_that_agrees_and_fits():
    pm = _runner()
    records = [_steps_record("none", 100.0), _steps_record("tf32", 90.0, speedups=["tf32"]),
               _steps_record("index_select", 70.0, same=False, speedups=["index_select"]),
               _steps_record("all", 60.0, cap_error=True, speedups=["tf32", "fused_adamw"]),
               _steps_record("tf32.chunk16", 85.0, speedups=["tf32"], chunk=16)]
    best = pm.best_config(records, "window")
    assert (best["speedups"], best["chunk"], best["median_ms"]) == (["tf32"], 16, 85.0)
    assert best["label"] == "best: tf32.chunk16"
    fallback = pm.best_config([], "comb", {"speedups": ["tf32"], "chunk": 16})
    assert fallback["speedups"] == ["tf32"] and fallback["median_ms"] is None
    loaders = [{"id": f"cpu.loader.n{n}.dataloader", "outcome": "ok", "variant": "dataloader",
                "workers": n, "result": {"items_per_second": v}}
               for n, v in ((20, 60.0), (22, 61.0), (23, 70.0))]
    assert pm.best_workers(loaders, [20, 22], 22) == 20  # within 2 %: the fewer
    loaders[1]["result"]["items_per_second"] = 64.0
    assert pm.best_workers(loaders, [20, 22], 22) == 22
    assert pm.best_workers([], [20, 22], 21) == 21


def test_the_plan_measures_each_speedup_alone_all_four_and_setups_choice(monkeypatch):
    pm = _runner()
    choices = {"machine": {}, "workers": 20,
               "views": {"window": {"speedups": ["tf32", "index_select"], "chunk": None},
                         "comb": {"speedups": ["tf32"], "chunk": 16}}}
    monkeypatch.setattr(pm, "setup_choices", lambda: choices)
    env = {"cpu": {"logical": 24}, "os": {"build": 26100}}
    plan = pm.build_plan({"gpu", "cpu", "e2e"}, env, smoke=False)
    ids = [e["id"] for e in plan]
    assert len(ids) == len(set(ids))
    steps = [i for i in ids if i.startswith("gpu.steps.window.")]
    assert steps == [f"gpu.steps.window.{n}" for n in (
        "none", "cudnn_benchmark", "tf32", "index_select", "fused_adamw", "all", "greedy",
        "greedy.chunk16", "greedy.chunk8", "greedy.chunk32")]
    assert [i for i in ids if i.startswith("gpu.steps.comb.")][-1] == "gpu.steps.comb.tf32.unchunked"
    assert "gpu.steps.comb.greedy" not in ids  # setup's choice is tf32 alone, measured already
    loaders = [e for e in plan if e["id"].startswith("cpu.loader.")]
    assert sorted({e["workers"] for e in loaders}) == [20, 22, 23] and len(loaders) == 15
    assert {e["variant"] for e in loaders} == {"dataloader", "dispatcher", "dataloader_pinned",
                                               "dispatcher_pinned", "dataloader_pinned_optout"}
    memory = {e["id"]: e["env"]["PYTORCH_CUDA_ALLOC_CONF"] for e in plan
              if e["id"].startswith("gpu.memory.")}
    assert memory["gpu.memory.window.default"] is None
    assert memory["gpu.memory.window.gc08"] == "garbage_collection_threshold:0.8"
    assert memory["gpu.memory.window.expandable"] == "expandable_segments:True"
    # after the end-to-end run, the information-only experiments, then the allocator settings
    # least likely to matter: the first skipped should time run out
    assert ids.index("e2e.train") < ids.index("gpu.autocast.window") < ids.index("gpu.smi.comb")
    assert ids[-3:] == ["gpu.memory.window.gc08_split512", "gpu.memory.window.expandable",
                        "gpu.memory.window.async"]
    assert max(ids.index(f"gpu.{x}.{v}") for x in ("autocast", "batch8")
               for v in ("window", "comb")) < ids.index("gpu.memory.window.gc08_split512")
    e2e = next(e for e in plan if e["id"] == "e2e.train")
    assert e2e["essential"] and e2e["estimate"] >= 240  # two epochs of 400 steps
    minutes = sum(e["estimate"] for e in plan) / 60
    assert 25 <= minutes <= 45
    assert pm.DEFAULT_MINUTES >= 1.35 * minutes  # room for a computer slower than estimated
    assert pm.worker_counts(24, False) == [20, 22, 23] and pm.worker_counts(4, True) == [2, 3]
    # on Linux no power-throttling opt-out
    plan = pm.build_plan({"cpu"}, {"cpu": {"logical": 24}, "os": {}}, smoke=False)
    if sys.platform != "win32":
        assert not any(e["id"].endswith("_optout") for e in plan)


def test_gpu_experiments_record_no_cuda_cleanly(tmp_path, monkeypatch):
    from profkit import gpu

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(common.Unavailable, match="no CUDA"):
        gpu.steps(common.Context(tmp_path), "window")
    monkeypatch.undo()
    # through the runner, each in a process of its own: the first says why, the next needs it
    pm = _runner()
    runner = pm.Runner(tmp_path / "run", pm.Console(tmp_path / "runner.log"), True, 600.0,
                       {"started": "now"})
    plan = [pm.entry("gpu.prepare", "gpu", "prepare", "profkit.gpu:prepare", "prepare",
                     {"batches": 1}, smoke=True),
            pm.entry("gpu.steps.window.none", "gpu", "steps", "profkit.gpu:steps", "steps",
                     {"view": "window"}, smoke=True, needs=["gpu.prepare"])]
    if real_cuda():
        pytest.skip("this computer has a GPU")
    runner.run_all(plan)
    first, second = runner.records
    assert first["outcome"] == "unavailable" and first["error"].startswith("no CUDA")
    assert second["outcome"] == "unavailable" and "needs gpu.prepare" in second["error"]


def real_cuda() -> bool:
    import subprocess

    out = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                         capture_output=True, text=True, timeout=300)
    return out.stdout.strip() == "True"


# ------------------------------------------------------------------ the GPU experiments' code
@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Two of the runs' batches, the at-cap batch and the weights, at short spectra and the
    small network, made on the processor."""
    from profkit import gpu

    ctx = common.Context(tmp_path_factory.mktemp("gpu"))
    rec = gpu.prepare(ctx, batches=2, crop=CROP, workers=0, size="small", device="cpu")
    return ctx, rec


GPU_KW = dict(device="cpu", size="small", crop=CROP)


def test_the_batches_are_training_s_first_batches(prepared):
    from profkit import gpu

    from lcmsdeconv.synth.dataset import FrameDataset

    ctx, rec = prepared
    assert rec["config"]["batches"] == 2 and len(rec["active_rows"]) == 2
    ds = FrameDataset(gpu.mixture(CROP), 8, base_seed=0)
    b1 = gpu.load_batch(ctx, 1, "comb")
    for j in range(4):  # items 4-7 of seed 0, stacked; the window view drops the last channel
        want = torch.stack([ds[i][j] for i in range(4, 8)])
        assert torch.equal(b1[j], want.to(b1[j].dtype)), j
    assert gpu.load_batch(ctx, 1, "window")[0].shape[1] == 7
    cap = gpu.load_batch(ctx, "cap", "comb")[0]
    on_grid = cap[:, -1] > 0.5
    assert bool((cap[:, 0][on_grid] >= 0.6).all())


def test_step_experiments_measure_and_compare_against_the_step_without_speedups(prepared):
    from profkit import gpu

    ctx, _rec = prepared
    few = dict(synced=2, free=2, warm=1, cap_warm=1, cap_timed=1, **GPU_KW)
    base = gpu.steps(ctx, "window", [], None, "none", **few)
    assert base["agreement"]["same"] and base["real"]["wall_ms"]["n"] == 2
    assert base["free_running"]["steps"] == 2 and base["cap"]["wall_ms"]["n"] == 1
    # the steps' spans fill the free-running loop: no busy share is taken from them
    assert "gpu_busy_share" not in base["free_running"]
    assert base["real"]["per_batch"][0]["active_rows"] == sum(
        gpu.active_rows(gpu.load_batch(ctx, 0, "window")[0], CROP))
    alt = gpu.steps(ctx, "window", ["index_select"], 5, "index_select.chunk5", **few)
    assert alt["agreement"]["same"], alt["agreement"]  # the same up to rounding
    assert alt["agreement"]["reference"] == "none" and alt["config"]["head_chunk"] == 5
    assert alt["agreement"]["steps"][0]["grad_cosine"] > 0.9999


def test_the_phase_split_names_every_phase_and_the_one_nonzero(prepared):
    from profkit import gpu

    ctx, _rec = prepared
    rec = gpu.phases(ctx, "comb", [], None, "none", steps_n=2, warm=1, **GPU_KW)
    assert list(rec["phases"]) == list(gpu.PHASES)
    assert rec["nonzero_calls_per_step"]["median"] == 1.0
    assert rec["step_gpu_ms"]["n"] == 2 and "bound" in rec["verdict"]
    assert torch.Tensor.nonzero.__name__ == "nonzero"  # the timing wrapper is gone again


def test_profile_memory_autocast_batch8_and_the_smi_loop_run(prepared, monkeypatch):
    from profkit import gpu

    ctx, _rec = prepared
    rec = gpu.profile(ctx, "window", [], None, "none", steps_n=1, warm=1, top=5, **GPU_KW)
    assert rec["ops"] and len(rec["ops"]) <= 5 and rec["trace"] == "traces/experiment.json.gz"
    with gzip.open(ctx.run_dir / rec["trace"], "rt", encoding="utf-8") as fh:
        assert "traceEvents" in json.load(fh)
    assert rec["trace_error"] is None and rec["ms_per_step"] > 0
    assert rec["gpu_busy_share"] is None  # the kernels' share of a step: CUDA only
    rec = gpu.memory(ctx, "window", [], None, "none", batches=2, **GPU_KW)
    assert rec["oom_steps"] == [] and rec["step_ms"]["n"] >= 1 and "warnings" in rec
    rec = gpu.autocast(ctx, "window", [], None, "none", steps_n=1, warm=0, **GPU_KW)
    assert set(rec["modes"]) == {"float32", "head", "forward"}
    for mode in ("head", "forward"):
        assert "error" not in rec["modes"][mode], rec["modes"][mode]
        assert rec["modes"][mode]["agreement"]["steps"][0]["loss_rel_diff"] < 0.05
    rec = gpu.batch8(ctx, "window", [], None, "none", steps_n=1, warm=0, **GPU_KW)
    assert rec["batches"]["8"]["spectra_per_second"] > 0
    monkeypatch.setattr(smi, "find_smi", lambda: None)
    rec = gpu.smi_loop(ctx, "window", [], None, "none", seconds=0.3, warm=0, **GPU_KW)
    assert rec["steps"] >= 1 and rec["nvidia_smi"] == {"samples": 0} and rec["csv"] is None


def test_a_trace_that_cannot_be_written_leaves_the_profiles_tables(prepared, monkeypatch):
    """The trace's export failing (on Windows its native exporter can fail on a folder named in
    another script): recorded, and the kernel and operator tables are the result all the same.
    Where the Python exporter is missing (an older PyTorch) or fails, the native one writes
    it."""
    from profkit import gpu
    from torch.profiler import profile as torch_profile

    ctx, _rec = prepared
    native = torch_profile.export_chrome_trace

    def older(self, path, *args, **kwargs):  # a PyTorch without use_python_export
        if "use_python_export" in kwargs:
            raise TypeError("export_chrome_trace() got an unexpected keyword argument")
        return native(self, path, *args, **kwargs)

    def failing(self, path, *args, **kwargs):  # the Python exporter fails on this trace
        if kwargs.get("use_python_export"):
            raise RuntimeError("an event it cannot write")
        return native(self, path, *args, **kwargs)

    trace = ctx.run_dir / "traces" / "experiment.json.gz"
    for stand_in in (older, failing):
        monkeypatch.setattr(torch_profile, "export_chrome_trace", stand_in)
        trace.unlink(missing_ok=True)
        rec = gpu.profile(ctx, "window", [], None, "none", steps_n=1, warm=0, top=5, **GPU_KW)
        assert rec["trace"] == "traces/experiment.json.gz" and rec["trace_error"] is None
        with gzip.open(trace, "rt", encoding="utf-8") as fh:
            assert "traceEvents" in json.load(fh)
        assert not list((ctx.run_dir / "work").glob("*_trace.json"))  # the uncompressed one

    def refuse(self, path, *args, **kwargs):
        raise OSError(f"cannot open {path}")

    monkeypatch.setattr(torch_profile, "export_chrome_trace", refuse)
    rec = gpu.profile(ctx, "window", [], None, "none", steps_n=1, warm=0, top=5, **GPU_KW)
    assert rec["trace"] is None and rec["trace_error"].startswith("OSError: the Python exporter")
    assert rec["trace_error"].count("cannot open") == 2  # both exporters tried
    assert rec["ops"] and len(rec["ops"]) <= 5 and rec["ms_per_step"] > 0


# ------------------------------------------------------------------ the processor experiments
def test_the_per_core_map_measures_every_processor_this_process_may_use(tmp_path):
    from profkit import cpu

    before = cpusets.current_affinity()
    rec = cpu.core_map(common.Context(tmp_path), items=1, crop=CROP)
    assert [r["cpu"] for r in rec["cores"]] == before
    assert all(r["items_per_second"] > 0 for r in rec["cores"])
    assert set(rec["by_kind"]) == {"uniform"} and rec["by_kind"]["uniform"]["n"] == len(before)
    assert cpusets.current_affinity() == before  # set back


def test_throttling_and_priority_and_the_profile_of_an_item_run_here(tmp_path):
    from profkit import cpu

    ctx = common.Context(tmp_path, {"id": "cpu.cprofile"})
    rec = cpu.throttle(ctx, items=1, reps=1, crop=CROP)
    cells = rec["cores"]["first"]["cells"]
    assert set(cells) == ({"default/normal", "default/below_normal", "opt_out/normal",
                           "opt_out/below_normal"} if cpusets.WINDOWS else
                          {"default/normal", "default/below_normal"})
    assert rec["config"]["windows_constants"]["ProcessPowerThrottling"] == 4
    assert os.getpriority(os.PRIO_PROCESS, 0) == cpusets.BASE_NICE  # set back
    rec = cpu.cprofile_items(ctx, items=1, crop=CROP, top=5)
    prof = rec["profiles"]["first"]
    assert len(prof["by_own"]) == 5 and prof["by_cumulative"][0]["cumulative_s"] > 0
    assert not any(str(ROOT) in r["function"] for r in prof["by_own"])
    assert (tmp_path / "logs" / "cpu.cprofile_first.txt").exists()


def test_the_dispatchers_batches_equal_the_dataloaders(monkeypatch):
    """Spawned workers, as on Windows: the same items, in the same order, the same tensors."""
    from profkit import cpu
    from profkit.dispatch import OrderedDispatcher
    from torch.utils.data import DataLoader

    from lcmsdeconv.synth.dataset import EpochSampler

    monkeypatch.setenv("LCMSDECONV_MP_CONTEXT", "spawn")
    ds, mix = cpu.recipe_dataset(12, base_seed=0, crop=CROP)
    cpu.warm_up(mix)
    want = list(DataLoader(ds, batch_size=4, num_workers=2, sampler=EpochSampler(12),
                           drop_last=True, multiprocessing_context="spawn"))
    disp = OrderedDispatcher(ds, list(range(12)), 4, 2, window=3)
    got = list(disp)
    assert len(got) == len(want) == 3
    for a, b in zip(want, got):
        assert len(a) == len(b) == 4
        for x, y in zip(a, b):
            assert x.dtype == y.dtype and torch.equal(x, y)
    assert sum(w["items"] for w in disp.per_worker().values()) == 12
    assert sorted(p for p, *_rest in disp.log) == list(range(12))


def test_the_dispatcher_reports_a_failing_item_and_stops_its_workers():
    """Item 5 fails at once while the others take a while, so its error arrives first: the
    batches before it are delivered all the same, then it is raised, at its turn, as DataLoader
    raises a failed batch."""
    import psutil
    from profkit.dispatch import OrderedDispatcher

    class Broken:
        def __len__(self):
            return 8

        def __getitem__(self, i):
            if i == 5:
                raise KeyError("item 5 is broken")
            time.sleep(0.4)
            return (torch.full((3,), float(i)),)

    disp = OrderedDispatcher(Broken(), list(range(8)), 2, 6, window=6)
    seen, pids = [], []
    with pytest.raises(RuntimeError, match="item 5 is broken"):
        for batch in disp:
            pids = pids or [p.pid for p in disp._workers]
            seen.append(batch[0][:, 0].tolist())
    assert seen == [[0.0, 1.0], [2.0, 3.0]] and len(pids) == 6
    for pid in pids:  # the workers ended with the error (at most zombies until reaped)
        assert (not psutil.pid_exists(pid)
                or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE), pid


@pytest.mark.parametrize("kind, pinned", [("dispatcher", True), ("dataloader", True),
                                          ("dataloader", False)])
def test_the_loader_experiment_measures_workers_in_full_flow(tmp_path, kind, pinned):
    from profkit import cpu

    before = cpusets.current_affinity()
    ctx = common.Context(tmp_path, {"id": f"cpu.loader.{kind}"})
    rec = cpu.loader(ctx, kind, workers=2, pinned=pinned, seconds=1.5, warm_rounds=1,
                     crop=CROP)
    assert rec["items_per_second"] > 0 and rec["timed_seconds"] >= 1.5
    assert 0.0 <= rec["consumer_wait_share"] <= 1.5
    assert {r["worker"] for r in rec["per_worker"]} == {0, 1}
    setup = rec["worker_setup"]
    assert {r["worker"] for r in setup} == {0, 1}
    assert rec["config"]["recipe"] == common.recipe()
    if pinned:
        plan = rec["config"]["plan"]
        assert sorted(r["cpu"] for r in setup) == sorted(plan["workers"])
        assert all(r["affinity"] == [r["cpu"]] for r in setup)
        # the workers started on every core this process may use, not squeezed onto the cores
        # it keeps for itself (two of four here): it pinned itself only once they ran
        assert all(r["affinity_at_start"] == before for r in setup)
        assert rec["config"]["main_affinity"] == sorted(plan["main"])
    else:
        assert rec["config"]["main_affinity"] is None
    assert cpusets.current_affinity() == before == sorted(os.sched_getaffinity(0))


# ------------------------------------------------------------------ the end-to-end check
#: a stand-in for scripts/train_charge_net.py: training's step lines (every 20 steps), pace
#: lines and log, a step every 10 ms after an epoch's first, which waits for its batch
FAKE_TRAINING = '''
import json
import sys
import time
from pathlib import Path

args = sys.argv[1:]


def arg(name):
    return args[args.index(name) + 1]


out = Path(arg("--out"))
steps = int(arg("--train-len")) // int(arg("--batch-size"))
print("GPU speedups: none", flush=True)
for epoch in range(int(arg("--epochs"))):
    for i in range(steps):
        time.sleep(0.3 if i == 0 else 0.01)
        if i % 20 == 0:
            print(f"  epoch {epoch} step {i}/{steps} loss 1.0000 ce 1.0000 heat 0.1000",
                  flush=True)
    pace = {"data_wait_share": 0.1, "gpu_step_ms": 10.0, "gpu_busy_share": 0.8}
    with open(out.with_suffix(".log.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"epoch": epoch, "seconds": 1.0, "train_loss": 1.0,
                             "pace": pace}) + "\\n")
    print("  epoch took 0.1 min; 40 spectra/s; of the training time 10 % waiting for data",
          flush=True)
out.write_bytes(b"weights")
out.with_suffix(".resume.pt").write_bytes(b"state")
'''


def test_the_end_to_end_check_records_each_epochs_pace_and_its_steady_pace(tmp_path,
                                                                           monkeypatch):
    """Training's own command line as train_all gives it for the window run, with the check's
    length; each epoch's pace from the log, and without its first 2 x workers steps from the
    step lines."""
    from profkit import e2e

    root = tmp_path / "package"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "train_charge_net.py").write_text(FAKE_TRAINING, encoding="utf-8")
    monkeypatch.setattr(e2e, "ROOT", root)
    monkeypatch.setattr(e2e, "require_cuda", lambda device="cuda": None)
    ctx = common.Context(tmp_path / "run", {"id": "e2e.train"})
    rec = e2e.train_check(ctx, workers=2, train_len=480, size="small", ram_gb=4.0)
    assert rec["exit_code"] == 0 and not rec["error_lines"], rec
    cfg = rec["config"]
    assert (cfg["recipe"], cfg["steps"], cfg["steady_skips_steps"]) == (common.recipe(), 240, 4)
    cmd = cfg["command"].split()
    for opt, value in (("--recipe", common.recipe()), ("--model-size", "small"),
                       ("--epochs", "2"), ("--train-len", "480"), ("--val-len", "8"),
                       ("--batch-size", "4"), ("--workers", "2"), ("--view", "window")):
        assert cmd.count(opt) == 1 and cmd[cmd.index(opt) + 1] == value, opt
    assert "--iso-channels" in cmd and cmd[cmd.index("--lr") + 1] == "2e-3"
    assert [ep["epoch"] for ep in rec["epochs"]] == [0, 1]
    for ep in rec["epochs"]:
        assert ep["spectra_per_second_epoch"] == 320.0  # 4 spectra x 0.8 busy / 10 ms
        assert (ep["steady"]["from_step"], ep["steady"]["to_step"]) == (20, 100)
        assert ep["steady"]["spectra_per_second"] > 0 and ep["line"].startswith("epoch took")
    assert len(rec["step_lines"]) == 12
    assert not list((ctx.run_dir / "e2e").glob("*.pt"))  # the throwaway model and state


def test_the_steady_pace_leaves_out_each_epochs_first_steps():
    from profkit import e2e

    times = [(0, 0, 5.0), (0, 20, 6.0), (0, 40, 7.0), (0, 60, 8.0),
             (1, 0, 20.0), (1, 20, 21.0), (1, 40, 21.5)]
    got = e2e.steady_pace(times, skip=20)
    assert got[0] == {"from_step": 20, "to_step": 60, "seconds": 2.0, "spectra_per_second": 80.0}
    assert got[1]["spectra_per_second"] == 160.0  # 20 steps of 4 spectra in half a second
    assert e2e.steady_pace(times, skip=60) == {}  # one line left in epoch 0: no pace


def test_the_profile_measures_train_all_s_recipe_even_where_probe_names_another(monkeypatch):
    import probe
    import train_all
    import train_charge_net
    from profkit import e2e

    from lcmsdeconv.synth.config import settings_repr

    assert common.recipe() == train_all.RECIPE
    for measured in (True, False):
        assert (settings_repr(common.recipe_mixture(measured))
                == settings_repr(probe.recipe_mixture(measured)))
    # train_all moved on to another recipe: the profile measures its spectra (and probe, which
    # reads train_all.RECIPE, makes them too)
    other = "v3" if train_all.RECIPE != "v3" else "v4"
    args = list(train_all.COMMON)
    args[args.index("--recipe") + 1] = other
    monkeypatch.setattr(train_all, "RECIPE", other)
    monkeypatch.setattr(train_all, "COMMON", args)
    want = getattr(train_charge_net, f"mixture_{other}")()
    for _w, c in want:
        c.singly_charged, c.iso_channels, c.measured_channel = "keep", True, True
    assert settings_repr(common.recipe_mixture()) == settings_repr(want)
    assert settings_repr(probe.recipe_mixture()) == settings_repr(want)
    # a probe that still made the old recipe's spectra: the profile keeps train_all's

    def stale(measured=True):
        mix = train_charge_net.mixture_v5()
        for _w, c in mix:
            c.singly_charged, c.iso_channels, c.measured_channel = "keep", True, measured
        return mix

    monkeypatch.setattr(probe, "recipe_mixture", stale)
    assert settings_repr(common.recipe_mixture()) == settings_repr(want)
    run = e2e.run_args("window", "full", 2, 1600, 8)
    assert run[run.index("--recipe") + 1] == other


def test_the_runner_profiles_the_processor_end_to_end(tmp_path):
    """--only cpu at the smoke size, through the command line: the environment and the per-core
    map in processes of their own, the summary, and the zip."""
    pm = _runner()
    rc = pm.main(["--only", "cpu", "--smoke", "--runs-dir", str(tmp_path),
                  "--experiments", "cpu.cores"])
    assert rc == 0
    zips = list(tmp_path.glob("profile_upload_*.zip"))
    assert len(zips) == 1
    run = next(tmp_path.glob("profile_2*"))
    saved = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in saved["experiments"]] == ["env", "cpu.cores"]
    assert [r["outcome"] for r in saved["experiments"]] == ["ok", "ok"]
    assert saved["environment"]["cpu"]["logical"] == os.cpu_count()
    assert not (run / "work").exists()
    summary = (run / "summary.txt").read_text(encoding="utf-8")
    assert "Processor, one worker per core" in summary
    with zipfile.ZipFile(zips[0]) as zf:
        names = zf.namelist()
    assert any(n.endswith("results.json") for n in names)
    assert any(n.endswith("logs/cpu.cores.log") for n in names)


def test_the_windows_launcher_sets_up_if_needed_then_profiles():
    bat = (GPU / "profile.bat").read_bytes()
    assert bat.count(b"\r\n") == bat.count(b"\n")  # every line ends with CRLF
    text = bat.decode("ascii")
    assert 'cd /d "%~dp0\\.."' in text
    setup = text.index("gpu_training\\setup.py --if-needed")
    run = text.index('".venv\\Scripts\\python.exe" gpu_training\\profile_machine.py %*')
    assert setup < text.index('set "LCMSDECONV_SETUP_RC=%ERRORLEVEL%"') < run
    assert text.rstrip().endswith("exit /b %RC%")
    sh = (GPU / "profile.sh").read_text()
    assert "setup.py --if-needed" in sh and 'profile_machine.py "$@"' in sh
    assert os.access(GPU / "profile.sh", os.X_OK)


def test_the_runner_works_from_a_folder_with_spaces_and_other_scripts_in_its_name(tmp_path):
    """The package unzipped to a folder such as "C:\\Users\\某人\\My Files\\lcms": the
    experiment's process is started with that path, and its output reaches the log as UTF-8."""
    import shutil

    root = tmp_path / "My Files 数据" / "lcms"
    shutil.copytree(GPU / "profkit", root / "gpu_training" / "profkit",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(GPU / "profile_machine.py", root / "gpu_training")
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "_profile_fakes.py").write_text(
        "def hello(ctx):\n    ctx.say('数据 ok')\n    return {'where': str(ctx.run_dir)}\n",
        encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_profile_machine_copy",
                                                  root / "gpu_training" / "profile_machine.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    assert pm.ROOT == root
    run = root / "runs" / "profile_x"
    runner = pm.Runner(run, pm.Console(tmp_path / "runner.log"), True, 600.0, {"started": "now"})
    runner.run_all(_plan(pm, ("t.hello", "hello", {}, 60)))
    rec = runner.records[0]
    assert rec["outcome"] == "ok", rec["error"]
    assert rec["result"]["where"] == str(run)
    assert "数据 ok" in (tmp_path / "runner.log").read_text(encoding="utf-8")
    pm.pack(run, tmp_path / "up.zip")
    with zipfile.ZipFile(tmp_path / "up.zip") as zf:
        text = zf.read("profile_x/results.json").decode("utf-8")
    assert "<package>/runs/profile_x" in text and str(root) not in text
