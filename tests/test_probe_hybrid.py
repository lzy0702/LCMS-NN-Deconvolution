"""Setup's test of how the data workers deliver spectra fastest (gpu_training/probe.py:
data_loading_test, choose_data_loading), what setup records of it, and how train_all.py passes
the choice to every run (today's way where machine.json says nothing), estimates the time with
the measured rate and falls back to today's way when the new one fails."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

GPU = Path(__file__).resolve().parents[1] / "gpu_training"
TODAY = {"loader": "dataloader", "cpu_binding": "none", "worker_qos": "system",
         "worker_priority": "below_normal"}
CORE_HIGH = {"loader": "ordered", "cpu_binding": "core", "worker_qos": "high",
             "worker_priority": "below_normal"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_gpu_{name}", GPU / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _way(rate: float, drift: float = 0.01, applied: int | None = None,
         waited: float = 0.3) -> dict:
    """A way's record as data_loading_test makes it: its rate, two windows ``drift`` apart."""
    return {"spectra_per_second": rate, "windows": [rate * (1 - drift / 2), rate * (1 + drift / 2)],
            "drift": drift, "applied": applied, "waiting_share": waited}


def _windows(first: float, second: float, applied: int | None = None) -> dict:
    """A way's record from its two windows' rates."""
    rate = (first + second) / 2
    return {"spectra_per_second": rate, "windows": [first, second],
            "drift": abs(second - first) / rate, "applied": applied, "waiting_share": 0.3}


# ------------------------------------------------------------------------------ the choice
def test_a_way_is_taken_only_for_a_few_percent_more_than_the_one_chosen_so_far():
    probe = _load("probe")
    configs = {"today": _way(28.0), "ordered": _way(35.0), "ordered_high": _way(35.5),
               "pool": _way(36.4), "pool_high": _way(36.5), "core": _way(36.8),
               "core_high": _way(38.0, applied=22), "core_normal": _way(38.5),
               "core_high_normal": _way(38.9)}
    d = probe.choose_data_loading(configs, 22)
    # ordered (+25 %), then pool (4 % more), then core_high (4 % more than pool, and 3.3 % more
    # than core, the way without the opt-out); the fastest, core_high_normal, is not 3 % more
    # than core_high: normal priority is not worth it. The windows, each pair 1 % apart, vary
    # by 0.4 % beyond a steady drift: the gains are not within reach of chance
    assert d["config"] == "core_high"
    assert {k: d[k] for k in TODAY} == CORE_HIGH
    assert d["spectra_per_second"] == 38.0 and d["today_spectra_per_second"] == 28.0
    assert d["gain"] == pytest.approx(10 / 28, abs=1e-3)
    assert d["reason"] == ("36 % more spectra per second than as before in setup's test (38.0 "
                           "against 28.0)")
    assert d["left_out"] == {} and not d.get("busy") and not d.get("gpu_bound")
    assert d["window_noise"] == pytest.approx(0.0038, abs=2e-4)
    # every change from today's has its price (data_cost), and the ladder follows it
    costs = {n: probe.data_cost(probe.data_settings(n)) for n in probe.DATA_CONFIGS}
    assert costs == {"today": 0, "ordered": 1, "ordered_high": 2, "pool": 2, "pool_high": 3,
                     "core": 3, "core_high": 4, "core_normal": 5, "core_high_normal": 6}


def test_each_further_change_pays_for_itself_against_the_ways_that_change_less():
    probe = _load("probe")
    configs = {"today": _way(28.0), "ordered": _way(35.0), "ordered_high": _way(35.5),
               "pool": _way(36.2), "pool_high": _way(36.5), "core": _way(37.0),
               "core_high": _way(38.0), "core_normal": _way(38.5),
               "core_high_normal": _way(38.9)}
    # core_high is 5 % more than pool, but the opt-out of power throttling only 2.7 % more
    # than core without it, and core only 2.2 % more than pool: pool
    d = probe.choose_data_loading(configs, 22)
    assert d["config"] == "pool" and d["spectra_per_second"] == 36.2


def test_today_is_kept_where_nothing_is_a_few_percent_faster():
    probe = _load("probe")
    d = probe.choose_data_loading({"today": _way(30.0), "ordered": _way(30.5),
                                   "core": _way(30.8)}, 4)
    assert d["config"] == "today" and {k: d[k] for k in TODAY} == TODAY
    assert d["spectra_per_second"] == 30.0 and d["gain"] == 0.0
    assert d["reason"] == ("as before: no other way was at least 3 % faster in setup's test than "
                           "the ways that change less, beyond its standard error; the fastest, "
                           "core, +3 % (± 0.6 %)")
    # nothing measured: today's, and why
    d = probe.choose_data_loading({})
    assert d["config"] == "today" and d["spectra_per_second"] is None
    assert "did not measure" in d["reason"]


def test_a_gain_within_reach_of_chance_does_not_count():
    probe = _load("probe")
    # the ordered loader 5 % faster, its windows and those of the other ways within 0.5 %
    steady = {"today": _windows(29.9, 30.1), "ordered": _windows(31.4, 31.6),
              "pool": _windows(30.2, 30.4), "core": _windows(30.1, 30.2)}
    d = probe.choose_data_loading(steady, 4)
    assert d["config"] == "ordered" and d["window_noise"] < 0.005
    # the same rates, each way's windows 5 % apart, now this way, now that: a window's rate
    # varies by about 4 %, and 5 % more is within reach of chance (with the margin alone,
    # ordered)
    noisy = {"today": _windows(29.3, 30.7), "ordered": _windows(32.4, 30.6),
             "pool": _windows(29.5, 31.1), "core": _windows(30.9, 29.4)}
    d = probe.choose_data_loading(noisy, 4)
    assert d["config"] == "today" and d["window_noise"] > 0.03
    assert re.search(r"the fastest, ordered, \+5 % \(± [3-6]\.\d %\)$", d["reason"])
    assert probe.choose_data_loading(noisy, 4, errors=0.0)["config"] == "ordered"
    # today's first window 6 % slower than its last (something else ran then): a way 5 % faster
    # than today's mean, but only 2 % faster than its undisturbed window, is not taken; one
    # 6 % faster than that window is
    slow_start = {"today": _windows(28.2, 30.0), "ordered": _windows(30.6, 30.6),
                  "pool": _windows(30.5, 30.5), "core": _windows(30.6, 30.6)}
    assert probe.choose_data_loading(slow_start, 4)["config"] == "today"
    faster = {**slow_start, "ordered": _windows(31.9, 31.9)}
    assert probe.choose_data_loading(faster, 4)["config"] == "ordered"


def test_a_steady_drift_over_the_test_is_not_taken_for_chance():
    probe = _load("probe")
    order = ["today", "ordered", "pool", "core", "core", "pool", "ordered", "today"]

    def drifting(rate: float, name: str) -> dict:  # the computer 7 % slower by the test's end
        first, second = (rate * (1 - 0.07 * i / 7) for i, n in enumerate(order) if n == name)
        return _windows(first, second)

    configs = {n: drifting(r, n) for n, r in (("today", 30.0), ("ordered", 36.0),
                                              ("pool", 35.8), ("core", 37.5))}
    noise = probe.data_noise(configs, order)
    assert noise["window"] == pytest.approx(0.0, abs=1e-9)
    assert noise["drift"] == pytest.approx(-0.07 / (1 - 0.035), rel=1e-6)
    assert noise["ways"] == pytest.approx({n: 0.0 for n in configs}, abs=1e-9)
    # the mirrored order keeps the drift out of the rates, and data_noise out of the noise:
    # core, 4.2 % more than ordered, is taken, although each way's windows are up to 7 % apart
    d = probe.choose_data_loading(configs, 4, order=order)
    assert d["config"] == "core" and d["test_drift"] == pytest.approx(-0.0725, abs=1e-4)
    # without the order the test's own is assumed: today's first and last, the others there
    # and back as listed
    assert probe.data_noise(configs) == noise
    lines = "\n".join(probe.data_lines({"workers": 4, "timed_spectra": 96, "configs": configs},
                                       d))
    assert "a window's rate varied by about 0.0 % by chance" in lines
    assert re.search(r"a steady drift of -7\.[23] % over the test", lines)


def test_a_busy_computer_keeps_today_and_unsteady_ways_are_left_out():
    probe = _load("probe")
    # today's first and last windows 15 % apart: something else ran meanwhile
    busy = probe.choose_data_loading({"today": _way(28.0, drift=0.15), "core": _way(40.0)}, 4)
    assert busy["config"] == "today" and busy["busy"]
    assert "busy with something else" in busy["reason"] and "15 % apart" in busy["reason"]
    # a way whose own windows are far apart, or that not every worker took, is left out
    d = probe.choose_data_loading({
        "today": _way(28.0), "ordered": _way(33.0, applied=4), "pool": _way(40.0, drift=0.2),
        "core": _way(45.0, applied=3), "core_high": {"spectra_per_second": None}}, 4)
    assert d["config"] == "ordered" and d["spectra_per_second"] == 33.0
    assert d["left_out"] == {"pool": "its two windows 20 % apart",
                             "core": "3 of 4 workers took it", "core_high": "not measured"}
    # the GPU set the pace of today's way already: said so
    capped = probe.choose_data_loading({"today": _way(28.0, waited=0.01),
                                        "ordered": _way(28.3)}, 4)
    assert capped["gpu_bound"] and capped["config"] == "today"


def test_the_ways_measured_are_those_this_computer_lets_switch():
    probe = _load("probe")
    every = {"binding": True, "qos": True, "priority": True}
    assert probe.data_configs(every) == list(probe.DATA_CONFIGS)
    # Linux: binding, but no power throttling, and nice cannot be lowered again
    assert probe.data_configs({"binding": True, "qos": False, "priority": False}) == [
        "today", "ordered", "pool", "core"]
    assert probe.data_configs({"binding": False, "qos": True, "priority": True}) == [
        "today", "ordered", "ordered_high"]
    assert probe.data_configs({}) == ["today", "ordered"]
    for name in probe.DATA_CONFIGS:  # every way is one train_charge_net.py takes
        settings = probe.data_settings(name)
        train_all = _load("train_all")
        assert all(settings[k] in train_all.DATA_CHOICES[k] for k in train_all.DATA_DEFAULTS)
    assert probe.data_settings("today") == TODAY


def test_the_checks_bind_this_process_and_give_its_processors_back(monkeypatch):
    import psutil

    from lcmsdeconv.nn import cpus

    probe = _load("probe")
    before = cpus.allowed_cpus()
    checks = probe.data_checks(cpus.detect(), before, 2)
    assert cpus.allowed_cpus() == before
    if sys.platform != "win32":
        assert not checks["qos"] and not checks["priority"]
        assert set(checks["why_not"]) >= {"qos", "priority"}
    if sys.platform.startswith("linux") and len(before) >= 3:
        assert checks["binding"], checks["why_not"]
    # too few cores to bind
    small = probe.data_checks(cpus.uniform_topology(2), [0, 1], 1)
    assert not small["binding"] and "usable cores" in small["why_not"]["binding"]
    # cores read that are not all the logical processors the system reports
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 64)
    odd = probe.data_checks(cpus.uniform_topology(4), [0, 1, 2, 3], 2)
    assert not odd["binding"] and "4 logical processors read, 64 reported" in odd["why_not"][
        "binding"]
    assert cpus.allowed_cpus() == before


# ------------------------------------------------------------------- the workers' side
def test_a_worker_takes_each_way_before_its_next_spectrum(monkeypatch):
    import torch

    from lcmsdeconv.nn import cpus

    probe = _load("probe")
    mod = probe._importable()  # the module the workers unpickle SwitchedFrames from
    assert mod.__name__ == "probe" and mod.SwitchedFrames.__module__ == "probe"
    calls: list = []
    ok = {"pin": True}
    monkeypatch.setattr(cpus, "pin_process", lambda c: calls.append(("pin", tuple(c))) or ok["pin"])
    monkeypatch.setattr(cpus, "set_power_throttling", lambda eco: calls.append(("qos", eco)) or True)
    monkeypatch.setattr(cpus, "set_priority", lambda p: calls.append(("priority", p)) or True)
    monkeypatch.setattr(cpus, "allowed_cpus", lambda: [0, 1, 2, 3, 4, 5])
    monkeypatch.setattr(mod, "_SWITCHED", {})
    monkeypatch.setattr(mod, "_worker_id", lambda: 1)
    topo = cpus.Topology(tuple(cpus.Core(cpus=(i,), kind="P" if i < 3 else "E",
                                         rank=0 if i < 3 else 1) for i in range(6)), "test")
    placements = {b: cpus.plan(topo, workers=3, mode=b) for b in ("pool", "core")}
    names = ["ordered", "core_high_normal", "pool"]
    control = torch.zeros(1, dtype=torch.int64)
    status = torch.full((3, 4), -1, dtype=torch.int64)
    frames = [(torch.tensor([i]),) for i in range(10)]
    ds = mod.SwitchedFrames(frames, control, status, [mod.data_settings(n) for n in names],
                            placements)
    assert len(ds) == 10

    def take(gen, way, idx):
        control[0] = gen * mod.DATA_CODES + names.index(way)
        assert ds[idx] is frames[idx]  # the spectra themselves, unchanged

    take(1, "ordered", 3)  # as the worker started: nothing to change
    assert calls == [] and status[1].tolist() == [1, 1, 1, 1]
    take(2, "core_high_normal", 4)
    assert calls == [("pin", placements["core"].worker_cpus(1)), ("qos", False),
                     ("priority", "normal")]
    assert status[1].tolist() == [2, 1, 1, 1] and status[0].tolist() == [-1] * 4
    calls.clear()
    take(2, "core_high_normal", 5)  # the same generation: nothing again
    assert calls == []
    take(3, "pool", 6)
    assert calls == [("pin", placements["pool"].worker_cpus(1)), ("qos", None),
                     ("priority", "below_normal")]
    calls.clear()
    take(4, "ordered", 7)  # not bound: back to the processors it started with
    assert calls == [("pin", (0, 1, 2, 3, 4, 5))]
    # a binding that fails is reported, and tried again with the next generation
    ok["pin"] = False
    take(5, "pool", 8)
    assert status[1].tolist() == [5, 0, 1, 1]
    ok["pin"] = True
    take(6, "pool", 9)
    assert status[1].tolist() == [6, 1, 1, 1]


class _Quick:
    """Stand-ins for the recipe's spectra: items of fixed shapes that take ``seconds`` of
    processor time each (identical for an index, as the real ones are)."""

    def __init__(self, seconds: float = 0.004):
        self.seconds = seconds

    def __len__(self) -> int:
        return 10**9

    def item_spec(self):
        import torch

        return [((8,), torch.float32), ((2,), torch.int64)]

    def __getitem__(self, i):
        import torch

        end = time.thread_time() + self.seconds
        while time.thread_time() < end:
            pass
        return torch.full((8,), float(i)), torch.tensor([i, -i])


def _live_children(before: set) -> list:
    import psutil

    helpers = set()
    for module, helper, attr in (("resource_tracker", "_resource_tracker", "_pid"),
                                 ("forkserver", "_forkserver", "_forkserver_pid")):
        try:
            mod = importlib.import_module(f"multiprocessing.{module}")
        except ImportError:
            continue
        pid = getattr(getattr(mod, helper, None), attr, None)
        if pid:
            helpers.add(pid)
    deadline = time.time() + 10
    while True:
        alive = [p for p in psutil.Process().children(recursive=True)
                 if p.pid not in before | helpers and p.status() != psutil.STATUS_ZOMBIE]
        if not alive or time.time() > deadline:
            return alive
        time.sleep(0.2)


@pytest.mark.parametrize("context", [None, "spawn"])
def test_one_start_of_ordered_workers_serves_every_way_in_mirrored_order(monkeypatch, context):
    """With the platform's start method (fork here) and with spawn (as on Windows: each
    worker unpickles SwitchedFrames from the module "probe")."""
    import psutil

    from lcmsdeconv.nn import cpus

    if context:
        monkeypatch.setenv("LCMSDECONV_MP_CONTEXT", context)
    else:
        monkeypatch.delenv("LCMSDECONV_MP_CONTEXT", raising=False)
    probe = _load("probe")
    before = cpus.allowed_cpus()
    children = {p.pid for p in psutil.Process().children(recursive=True)}
    seen = []
    load = probe.TrainingLoad(step=lambda batch: seen.append(len(batch[0])))
    names = None if context is None else ["today", "core"]
    rec = probe.data_loading_test(2, 150.0, load=load, frames=_Quick(), batch_size=4,
                                  window_seconds=0.4, names=names)
    assert "error" not in rec, rec.get("error")
    ways = [n for n in rec["configs"] if n != "today"]
    expected = probe.data_configs(rec["checks"])
    if names:
        expected = [n for n in expected if n in names]
    assert list(rec["configs"]) == expected and "ordered" in ways or "core" in ways
    assert rec["order"] == ["today", *ways, *ways[::-1], "today"]
    assert rec["timed_spectra"] % (2 * 4) == 0  # whole rounds of the DataLoader
    for name, c in rec["configs"].items():
        assert len(c["windows"]) == 2 and c["drift"] is not None and c["spectra_per_second"] > 0
        assert c["spectra"] == 2 * rec["timed_spectra"]
        assert sum(k["items"] for k in c["by_kind"].values()) > 0
        assert c["waiting_share"] is not None and c["workers_busy"] is not None
        assert {k: c[k] for k in TODAY} == probe.data_settings(name)
        if name != "today":
            assert c["applied"] == 2 and c["made_twice"] >= 0
    assert seen and set(seen) == {4}  # every batch went through the training-like load
    assert cpus.allowed_cpus() == before  # this process got its processors back
    assert _live_children(children) == []  # the workers ended with the test
    d = probe.choose_data_loading(rec["configs"], 2)
    lines = "\n".join(probe.data_lines(rec, d))
    assert "training uses: " in lines and "today " in lines


def test_a_test_that_stops_keeps_today(monkeypatch):
    probe = _load("probe")

    class Broken(_Quick):
        def __getitem__(self, i):
            raise ValueError("no spectrum here")

    rec = probe.data_loading_test(2, 150.0, frames=Broken(), batch_size=4, window_seconds=0.2)
    assert "no spectrum here" in rec["error"]
    assert rec["configs"] == {}
    d = probe.choose_data_loading(rec["configs"], 2)
    assert d["config"] == "today"


class _StuckInPool(_Quick):
    """Spectra the ordered loader's workers never deliver (indices ``start`` to ``end``),
    torch's DataLoader's do: an ordered loader that stops delivering."""

    def __init__(self, start: int, end: int):
        super().__init__()
        self.start, self.end = start, end

    def __getitem__(self, i):
        from lcmsdeconv.synth.pool import current_worker

        if current_worker() and self.start <= i < self.end:
            time.sleep(3600)
        return super().__getitem__(i)


@pytest.mark.parametrize("where", ["start", "window"])
def test_an_ordered_loader_that_stops_delivering_stops_the_test(monkeypatch, where):
    """The ordered loader waits for a batch without a timeout of its own: a watchdog stops its
    workers at the window's limit (its warm-up before the first window: WORKER_TIMEOUT more),
    and the test stops with an error; today's way is kept."""
    import psutil

    monkeypatch.delenv("LCMSDECONV_MP_CONTEXT", raising=False)
    probe = _load("probe")
    monkeypatch.setattr(probe, "DATA_WINDOW_MIN_LIMIT", 3.0)
    monkeypatch.setattr(probe, "WORKER_TIMEOUT", 5.0)
    frames = {"start": _StuckInPool(probe.DATA_WARM_BASE, probe.DATA_LEAD_BASE),  # warm-up's
              "window": _StuckInPool(0, 8)}[where]  # the timed spectra
    children = {p.pid for p in psutil.Process().children(recursive=True)}
    t0 = time.perf_counter()
    rec = probe.data_loading_test(2, 150.0, frames=frames, batch_size=4,
                                  names=["today", "ordered"], window_seconds=0.2)
    took = time.perf_counter() - t0
    late = {"start": "the ordered loader's start took more than 8 s",
            "window": "a window took more than 3 s"}[where]
    assert rec["error"] == f"TimeoutError: {late}"
    assert rec["order"] == ["today"] and list(rec["configs"]) == ["today"]
    assert took < 60
    assert _live_children(children) == []  # the stuck workers were stopped too


def test_the_watchdog_stops_only_a_block_that_overruns():
    import threading

    probe = _load("probe")
    stops = []
    with probe._watchdog(0.3, lambda: stops.append("stop"), "a quick block"):
        pass
    time.sleep(0.6)
    assert stops == []  # ended in time: never stopped
    # a block waiting for what the stop makes fail, as the ordered loader's batches do
    stopped = threading.Event()
    with pytest.raises(TimeoutError, match="^a slow block took more than 1 s$") as err:
        with probe._watchdog(1.0, stopped.set, "a slow block"):
            assert stopped.wait(30)
            raise RuntimeError("the data workers were stopped")
    assert isinstance(err.value.__cause__, RuntimeError)
    # one that ends after the stop came all the same: stopped too
    with pytest.raises(TimeoutError, match="^a late block took more than 0 s$"):
        with probe._watchdog(0.2, lambda: stops.append("late"), "a late block"):
            time.sleep(0.6)
    assert stops == ["late"]


def test_the_training_like_load_keeps_the_main_thread_busy_without_the_lock():
    import threading

    probe = _load("probe")
    load = probe.TrainingLoad(busy_seconds=0.2)
    assert "busy for 200 ms" in load.what
    ticks = []

    def other():  # another thread of this process keeps running meanwhile
        end = time.perf_counter() + 0.15
        while time.perf_counter() < end:
            ticks.append(1)

    t = threading.Thread(target=other)
    t0 = time.perf_counter()
    t.start()
    load([])
    t.join()
    assert time.perf_counter() - t0 >= 0.2 and len(ticks) > 1000
    load.close()
    assert probe.TrainingLoad().what == "nothing done with the batches"


def test_the_load_follows_the_step_setup_measured_and_a_failure_is_returned(monkeypatch):
    import torch

    probe = _load("probe")
    step = {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.153,
            "unchunked": {"batch": 4, "peak_gb": 5.0, "step_seconds": 0.12}}
    gpu = {"gpu_views": {"window": step}}
    speed = {"speedup_timings": {"window": {"none": 100.0, "tf32+index_select": 80.0}}}
    # the runs' first training batch with the speedups chosen, else the step at the cap
    assert probe.window_step_seconds(gpu, speed, 16, ["index_select", "tf32"]) == 0.08
    assert probe.window_step_seconds(gpu, {}, 16, ["tf32"]) == 0.153
    assert probe.window_step_seconds(gpu, {}, None, []) == 0.12
    assert probe.window_step_seconds({}, {}, 16, []) is None
    # no CUDA: the main thread kept busy as long as a step
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    load = probe.training_load(4, 16, ["tf32"], 2e-3, 0.08)
    assert load.step is None and not load.cuda and load.busy_seconds == 0.08
    # the test failing in any way: an error, and the load closed
    closed = []
    load = probe.TrainingLoad(restore=lambda: closed.append(True))
    monkeypatch.setattr(probe, "training_load", lambda *_a: load)

    def broken(*_a, **_k):
        raise OSError("no room for the slots")

    monkeypatch.setattr(probe, "data_loading_test", broken)
    out = probe.measure_data_loading(4, 20.0, 4, 16, [], 2e-3, 0.08)
    assert out == {"error": "OSError: no room for the slots"} and closed == [True]


# ------------------------------------------------------------------------------ setup
def _probe_main(monkeypatch, tmp_path, total: float = 96.0, test=None, argv=()):
    """probe.py main on a computer of ``total`` GB and 24 cores with a GPU stood in for, its
    data workers measured at 1.2 spectra/s each (0.88 GB), and the data-loading test returning
    ``test(workers)``. Returns machine.json, what the test was asked and the printed text."""
    import psutil
    import torch

    import lcmsdeconv.nn.machine as nn_machine

    probe = _load("probe")
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(total=int(total * 2**30)))
    monkeypatch.setattr(nn_machine, "cpu_cores", lambda: 24)
    monkeypatch.setattr(nn_machine, "single_threaded_workers", lambda: None)
    monkeypatch.setattr(probe, "process_gb", lambda proc: 1.5)
    monkeypatch.setattr(probe, "workers_run", lambda n, **_k: {
        "workers": n, "startup_seconds": 30.0, "timed_seconds": 60.0, "timed_spectra": 1000,
        "spectra_per_second": round(1.2 * n, 2), "worker_gb": 0.88, "worker_commit_gb": 1.6,
        "workers_total_gb": 0.88 * n})
    step = {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.153}
    monkeypatch.setattr(probe, "gpu_steps", lambda *_a: {
        "gpu": "Fake GPU", "gpu_total_gb": 16.0, "gpu_views": {"window": dict(step),
                                                               "comb": dict(step)}})
    monkeypatch.setattr(probe, "measure_speedups", lambda *_a: {
        "gpu_speedups": {"window": ["tf32"], "comb": []},
        "speedup_timings": {"window": {"none": 100.0, "tf32": 95.0}, "comb": {"none": 105.0}}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    asked = []

    def measure(workers, rate, batch, chunk, speedups, lr, step_seconds):
        asked.append({"workers": workers, "rate": rate, "batch": batch, "chunk": chunk,
                      "speedups": list(speedups), "lr": lr, "step_seconds": step_seconds})
        return test(workers) if test else None

    monkeypatch.setattr(probe, "measure_data_loading", measure)
    out = tmp_path / "machine.json"
    monkeypatch.setattr(sys, "argv", ["probe.py", "--out", str(out), *argv])
    assert probe.main() == 0
    return json.loads(out.read_text()), asked


def _table(workers: int, today: float = 27.0, best: float = 36.0, **extra) -> dict:
    """data_loading_test's record: today's way and the ordered loader, each worker on its own
    core and opted out of power throttling, ``best`` spectra a second."""
    probe = _load("probe")
    configs = {"today": {**TODAY, **_way(today, waited=0.4)},
               "ordered": {**probe.data_settings("ordered"), **_way(today * 1.1, applied=workers)},
               "core_high": {**CORE_HIGH, **_way(best, applied=workers),
                             "by_kind": {"P": {"items": 400, "per_worker_per_second": 2.3},
                                         "E": {"items": 600, "per_worker_per_second": 1.4}}}}
    return {"recipe": "v5", "workers": workers, "batch": 4, "timed_spectra": 264,
            "load": "a training step of the window view on the GPU for every batch",
            "seconds": 250.0, "checks": {"why_not": {}}, "order": [], "configs": configs,
            **extra}


def test_setup_records_the_way_it_measured_fastest_and_its_rate(tmp_path, monkeypatch, capsys):
    info, asked = _probe_main(monkeypatch, tmp_path, test=_table)
    text = capsys.readouterr().out
    # measured at the number of workers chosen, with today's rate, under the window view's
    # training step with the runs' chunking and speedups (95 ms with tf32)
    assert asked == [{"workers": 22, "rate": 26.4, "batch": 4, "chunk": 16,
                      "speedups": ["tf32"], "lr": 2e-3, "step_seconds": 0.095}]
    rec = info["data_loader"]
    assert {k: rec[k] for k in TODAY} == CORE_HIGH and rec["config"] == "core_high"
    assert rec["workers"] == 22 and rec["spectra_per_second"] == 36.0
    assert rec["today_spectra_per_second"] == 27.0
    assert info["data_loader_test"] == _table(22)
    # the time estimate takes what that way delivered under the training-like load
    assert (info["v5_spectra_per_second"], info["v5_workers"], info["workers"]) == (36.0, 22, 22)
    assert info["pace"]["window"]["data_spectra_per_second"] == 36.0
    # 0.06 GB of shared slots a worker for the ordered loader
    assert info["expected_ram_gb"] == round(2.4 + 22 * (0.88 + 0.06), 1)
    assert re.search(r"core_high +36\.0 +\+33 % \(35\.8, 36\.2\); performance 2\.30/s, "
                     r"efficient 1\.40/s; waited 30 %", text)
    assert ("training uses: the ordered loader, each worker on a core of its own, opted out of "
            "power throttling (EcoQoS), below-normal priority: 33 % more spectra per second") in text
    assert "Chosen: 22 data workers (the processor's share" in text
    assert "36.0 spectra/s as setup's test measured them" in text


def test_the_ordered_loaders_slots_count_against_the_ram_budget(tmp_path, monkeypatch, capsys):
    # 32 GB: a budget of 16 holds 13 workers of 0.88 GB, 12 with the slots
    info, asked = _probe_main(monkeypatch, tmp_path, total=32, test=_table)
    assert asked[0]["workers"] == 13
    assert info["workers"] == info["v5_workers"] == info["data_loader"]["workers"] == 12
    assert info["v5_spectra_per_second"] == round(36.0 * 12 / 13, 1)
    text = capsys.readouterr().out
    assert ("Chosen: 12 data workers (as many as the RAM budget holds with the ordered loader's "
            "shared slots; 13 were tested)") in text


@pytest.mark.parametrize("how", ["stopped", "skipped", "busy"])
def test_setup_keeps_today_where_the_test_does_not_decide(tmp_path, monkeypatch, capsys, how):
    test = {"stopped": lambda n: {"error": "RuntimeError: a worker died"},
            "skipped": None,
            "busy": lambda n: {**_table(n), "configs": {
                **_table(n)["configs"], "today": {**TODAY, **_way(27.0, drift=0.2)}}}}[how]
    argv = ["--no-data-test"] if how == "skipped" else []
    info, asked = _probe_main(monkeypatch, tmp_path, test=test, argv=argv)
    rec = info["data_loader"]
    assert {k: rec[k] for k in TODAY} == TODAY and rec["config"] == "today"
    assert asked == ([] if how == "skipped" else asked) and len(asked) == (how != "skipped")
    reason = {"stopped": "setup's test of the data workers stopped (RuntimeError: a worker died)",
              "skipped": "setup did not test the data workers (--no-data-test)",
              "busy": "the computer was busy with something else"}[how]
    assert reason in rec["reason"]
    assert ("data_loader_test" in info) == (how != "skipped")
    # the workers' rate: today's way, measured by the test where it measured it
    assert info["v5_spectra_per_second"] == (27.0 if how == "busy" else 26.4)
    assert info["workers"] == 22
    assert "training uses: torch's DataLoader, workers not bound" in capsys.readouterr().out


# ------------------------------------------------------------------------------ train_all
NEW_MACHINE = {
    "recipe": "v5", "workers": 22, "ram_budget_gb": 48.0, "ram_budget_default": True,
    "gpu_memory_gb": 14.0, "gpu_batch": 4, "head_chunk": 16,
    "head_chunks": {"window": 16, "comb": 16}, "main_gb": 2.4, "worker_gb": 0.88,
    "v5_spectra_per_second": 36.0, "v5_workers": 22, "cores": 24, "gpu": "Fake GPU",
    "gpu_views": {"window": {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.153},
                  "comb": {"batch": 4, "peak_gb": 3.2, "step_seconds": 0.16}},
    "gpu_speedups": {"window": ["tf32"], "comb": []},
    "speedup_timings": {"window": {"none": 100.0, "tf32": 95.0}, "comb": {"none": 105.0}},
    "data_loader": {**CORE_HIGH, "config": "core_high", "spectra_per_second": 36.0,
                    "today_spectra_per_second": 27.0, "workers": 22,
                    "reason": "33 % more spectra per second than as before in setup's test "
                              "(36.0 against 27.0)"},
    "data_loader_test": _table(22)}


class _Training:
    """subprocess.Popen for train_all: records the command, prints ``lines`` and exits ``rc``
    (the last script entry repeats)."""

    def __init__(self, record: list, script: list):
        self.record, self.script = record, script

    def __call__(self, cmd, **_kw):
        self.record.append(cmd)
        lines, rc = self.script[min(len(self.record), len(self.script)) - 1]
        return SimpleNamespace(stdout=iter(lines), wait=lambda: rc)


def _train_all(tmp_path, monkeypatch, machine: dict | None, script=None):
    train_all = _load("train_all")
    monkeypatch.setattr(train_all, "ROOT", tmp_path)
    monkeypatch.setattr(train_all, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(train_all, "MACHINE", tmp_path / "gpu_training" / "machine.json")
    (tmp_path / "runs").mkdir(parents=True)
    (tmp_path / "gpu_training").mkdir()
    if machine is not None:
        train_all.MACHINE.write_text(json.dumps(machine))
    for name in train_all.RUN_TABLE:  # what the trainings leave, for the export
        (tmp_path / train_all.RUN_TABLE[name]["out"]).write_text("checkpoint")
    commands: list = []
    monkeypatch.setattr(train_all.subprocess, "Popen",
                        _Training(commands, script or [([], 0)]))
    monkeypatch.setattr(train_all.subprocess, "run",
                        lambda cmd, **_kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(train_all.signal, "signal", lambda *_a: None)
    return train_all, commands


def _data_of(cmd) -> dict:
    """The data-loading options of a training command, by setting."""
    opts = {"--loader": "loader", "--cpu-binding": "cpu_binding", "--worker-qos": "worker_qos",
            "--worker-priority": "worker_priority"}
    return {key: cmd[cmd.index(opt) + 1] for opt, key in opts.items() if opt in cmd}


def _computer(monkeypatch, total_gb: float = 96, cores: int = 24):
    import psutil

    import lcmsdeconv.nn.machine as nn_machine

    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(total=int(total_gb * 2**30)))
    monkeypatch.setattr(nn_machine, "cpu_cores", lambda: cores)


def test_every_run_takes_setups_way(tmp_path, monkeypatch, capsys):
    _computer(monkeypatch)
    train_all, commands = _train_all(tmp_path, monkeypatch, NEW_MACHINE)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu"])
    assert train_all.main() == 0
    assert [_data_of(c) for c in commands] == [CORE_HIGH, CORE_HIGH]
    assert all(c[c.index("--workers") + 1] == "22" for c in commands)
    assert all(c[-1] == "--restart" for c in commands)  # still the last argument
    text = capsys.readouterr().out
    assert ("[train_all] data loading: the ordered loader, each worker on a core of its own, "
            "opted out of power throttling (EcoQoS), below-normal priority (setup's choice: 33 % "
            "more spectra per second") in text
    # the estimate takes the rate setup measured that way: 36 spectra/s against the GPU's 42
    assert "window 1.0 h at 36 spectra/s, set by the data workers" in text


def test_an_earlier_machine_json_trains_exactly_as_before(tmp_path, monkeypatch, capsys):
    _computer(monkeypatch)
    older = {k: v for k, v in NEW_MACHINE.items() if k not in ("data_loader", "data_loader_test")}
    train_all, commands = _train_all(tmp_path, monkeypatch, older)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu"])
    assert train_all.main() == 0
    assert [_data_of(c) for c in commands] == [{}, {}]  # train_charge_net.py's defaults
    text = capsys.readouterr().out
    assert ("data loading: torch's DataLoader, workers not bound to cores, power throttling left "
            "to Windows, below-normal priority (as before: gpu_training/machine.json does not say "
            "how the data workers run fastest") in text
    # a value this package does not know: as before too
    odd = {**NEW_MACHINE, "data_loader": {**CORE_HIGH, "loader": "turbo"}}
    train_all, commands = _train_all(tmp_path / "odd", monkeypatch, odd)
    assert train_all.main() == 0
    assert [_data_of(c) for c in commands] == [{}, {}]
    assert "names a data loading this package does not know" in capsys.readouterr().out


def test_options_override_setups_way_and_the_estimate_follows(tmp_path, monkeypatch, capsys):
    _computer(monkeypatch)
    train_all, commands = _train_all(tmp_path, monkeypatch, NEW_MACHINE)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu", "--loader",
                                      "dataloader", "--cpu-binding", "none", "--worker-qos",
                                      "system"])
    assert train_all.main() == 0
    assert [_data_of(c) for c in commands] == [TODAY, TODAY]
    text = capsys.readouterr().out
    assert ("(setup's choice, with --loader dataloader, --cpu-binding none, --worker-qos system "
            "as asked)") in text
    # today's way as setup measured it: 27 spectra/s
    assert "window 1.4 h at 27 spectra/s, set by the data workers" in text
    # one setting alone: the rest stays setup's
    train_all, commands = _train_all(tmp_path / "one", monkeypatch, NEW_MACHINE)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu", "--worker-priority",
                                      "normal"])
    assert train_all.main() == 0
    assert [_data_of(c) for c in commands] == [{**CORE_HIGH, "worker_priority": "normal"}] * 2
    # not measured that way: setup's rate for its own choice
    assert "window 1.0 h at 36 spectra/s" in capsys.readouterr().out


def test_the_estimate_uses_the_rate_measured_for_each_way():
    train_all = _load("train_all")
    m = NEW_MACHINE
    assert train_all.data_rate(m, 22) == 36.0
    assert train_all.data_rate(m, 22, CORE_HIGH) == 36.0
    assert train_all.data_rate(m, 11, TODAY) == pytest.approx(13.5)
    assert train_all.rate_workers(m, TODAY) == 22
    # not measured that way, or measured on another recipe's spectra: setup's choice's rate
    assert train_all.data_rate(m, 22, {**CORE_HIGH, "worker_priority": "normal"}) == 36.0
    other = {**m, "data_loader_test": {**m["data_loader_test"], "recipe": "v3"}}
    assert train_all.data_rate(other, 22, TODAY) == 36.0
    assert train_all.measured_data_rate(m, None) is None


def test_the_ordered_loaders_slots_count_against_the_budget_in_training(tmp_path, monkeypatch):
    _computer(monkeypatch, 32)
    old = {**{k: v for k, v in NEW_MACHINE.items() if k not in ("data_loader",
                                                                 "data_loader_test")},
           "workers": 13, "ram_budget_gb": 16.0, "v5_workers": 13}
    train_all, commands = _train_all(tmp_path, monkeypatch, old)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu", "--loader", "ordered"])
    assert train_all.main() == 0
    # setup chose 13 for torch's DataLoader; with the ordered loader's slots 16 GB hold 12
    assert [c[c.index("--workers") + 1] for c in commands] == ["12", "12"]
    # setup's own choice of the ordered loader already counts them: its number is kept
    chosen = {**old, "workers": 12, "data_loader": {**NEW_MACHINE["data_loader"],
                                                    "workers": 12}}
    train_all, commands = _train_all(tmp_path / "chosen", monkeypatch, chosen)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu"])
    assert train_all.main() == 0
    assert [c[c.index("--workers") + 1] for c in commands] == ["12", "12"]


FAILURE = ["epoch 0 step 3/1500 loss 1.2\n", "Traceback (most recent call last):\n",
           "lcmsdeconv.synth.pool.DataWorkerError: data worker 3 (pid 4242, efficient core, "
           "processor 9) exited unexpectedly with exit code 3221225477 while generating item "
           "1234 of 'train'\n"]


@pytest.mark.parametrize("saved", [True, False])
def test_a_run_whose_new_data_loading_fails_continues_as_before(tmp_path, monkeypatch, capsys,
                                                               saved):
    train_all, commands = _train_all(tmp_path, monkeypatch, NEW_MACHINE,
                                     script=[(FAILURE, 1), (["done\n"], 0)])
    stage = train_all.make_run("window")
    out = tmp_path / stage["out"]
    if saved:  # the run saved its state before the failure
        out.with_suffix(".resume.pt").write_text("state")
        out.with_suffix(".status.json").write_text(json.dumps({"epoch": 0, "epochs": 20}))
    pool = {"workers": 22, "data": dict(CORE_HIGH)}
    assert train_all.run_stage(stage, pool, [], "cuda", False) == 0
    assert [_data_of(c) for c in commands] == [CORE_HIGH, TODAY]
    assert commands[1][-1] == ("--resume" if saved else "--restart")
    assert pool["data"] == TODAY  # the later runs of this train_all too
    machine = json.loads(train_all.MACHINE.read_text())
    rec = machine["data_loader"]
    assert {k: rec[k] for k in TODAY} == TODAY and rec["failed"] == CORE_HIGH
    assert "exited unexpectedly" in rec["reason"]
    # the time estimate now takes today's way as setup measured it
    assert (machine["v5_spectra_per_second"], machine["v5_workers"]) == (27.0, 22)
    assert machine["data_loader_test"] == NEW_MACHINE["data_loader_test"]
    text = capsys.readouterr().out
    assert "the data loading (the ordered loader, each worker on a core of its own" in text
    assert "continuing as before, torch's DataLoader" in text
    # a later train_all reads it: today's way, setup's measurement of it
    assert train_all.data_loading_for(machine) == TODAY


@pytest.mark.parametrize("data, lines", [
    (dict(TODAY), FAILURE),          # today's way failed: nothing to fall back to
    (None, FAILURE),                 # train_charge_net.py's defaults: today's way
    (dict(CORE_HIGH), ["ValueError: no\n"]),  # another failure: not the data loading's
])
def test_failures_of_todays_way_or_of_anything_else_are_not_retried(tmp_path, monkeypatch,
                                                                   data, lines):
    train_all, commands = _train_all(tmp_path, monkeypatch, NEW_MACHINE,
                                     script=[(lines, 1), (["done\n"], 0)])
    stage = train_all.make_run("window")
    pool = {"workers": 22} if data is None else {"workers": 22, "data": data}
    assert train_all.run_stage(stage, pool, [], "cuda", False) == 1
    assert len(commands) == 1  # not tried again
    assert json.loads(train_all.MACHINE.read_text()) == NEW_MACHINE  # nothing remembered
