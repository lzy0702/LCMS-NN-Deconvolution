"""The one-click GPU training script: setup is skipped while it holds, then training starts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

GPU = Path(__file__).resolve().parents[1] / "gpu_training"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_gpu_{name}", GPU / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_mod(tmp_path, monkeypatch):
    """gpu_training/setup.py pointed at a scratch package folder, with a working GPU."""
    mod = _load("setup")
    for rel in mod.SETUP_INPUTS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"contents of {rel}\n")
    vpy = tmp_path / ".venv" / "bin" / "python"
    vpy.parent.mkdir(parents=True)
    vpy.write_text("")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "VPY", vpy)
    monkeypatch.setattr(mod, "LOG", tmp_path / "gpu_training" / "setup.log")
    monkeypatch.setattr(mod, "MACHINE", tmp_path / "gpu_training" / "machine.json")
    monkeypatch.setattr(mod, "DONE", tmp_path / "gpu_training" / "setup_done.json")
    monkeypatch.setattr(mod, "torch_on_gpu", lambda: ("torch 2.x+cu130 on the GPU", ""))
    mod.MACHINE.write_text(json.dumps({"workers": 8}))
    return mod


def _record(mod, **changes):
    done = {"finished": "2026-11-23 08:00", "torch": "torch", "probe": True, "smoke": True,
            "fingerprint": mod.fingerprint(), **changes}
    mod.DONE.write_text(json.dumps(done))


def test_a_finished_setup_holds(setup_mod):
    _record(setup_mod)
    holds, why = setup_mod.setup_is_current()
    assert holds, why


@pytest.mark.parametrize("breakage, expected", [
    ("no record", "no setup has finished"),
    ("no venv", ".venv does not exist"),
    ("package changed", "package has changed"),
    ("probe skipped", "skipped the measurement"),
    ("no machine.json", "machine.json is missing"),
    ("no gpu", "no longer computes on the GPU"),
    ("unreadable record", "cannot be read"),
])
def test_setup_runs_again_when_it_no_longer_holds(setup_mod, monkeypatch, breakage, expected):
    _record(setup_mod, probe=breakage != "probe skipped")
    if breakage == "no record":
        setup_mod.DONE.unlink()
    elif breakage == "no venv":
        setup_mod.VPY.unlink()
    elif breakage == "package changed":
        (setup_mod.ROOT / "pyproject.toml").write_text("a new dependency\n")
    elif breakage == "no machine.json":
        setup_mod.MACHINE.unlink()
    elif breakage == "no gpu":
        monkeypatch.setattr(setup_mod, "torch_on_gpu", lambda: (None, "CUDA error: no kernel"))
    elif breakage == "unreadable record":
        setup_mod.DONE.write_text("{not json")
    holds, why = setup_mod.setup_is_current()
    assert not holds and expected in why


def test_if_needed_skips_every_step_while_the_setup_holds(setup_mod, monkeypatch):
    _record(setup_mod)

    def no_step(*args, **kwargs):
        raise AssertionError("a setup step ran although the setup holds")

    monkeypatch.setattr(setup_mod, "run", no_step)
    monkeypatch.setattr(setup_mod, "driver", no_step)
    monkeypatch.setattr(sys, "argv", ["setup.py", "--if-needed"])
    assert setup_mod.main() == 0
    assert setup_mod.DONE.exists()  # the record is kept, not rewritten
    assert "already done" in setup_mod.LOG.read_text()


def test_a_full_setup_drops_the_old_record_before_it_starts(setup_mod, monkeypatch):
    _record(setup_mod)
    monkeypatch.setattr(setup_mod, "driver", lambda: {})  # stops at the driver check
    monkeypatch.setattr(sys, "argv", ["setup.py"])
    assert setup_mod.main() == 1
    assert not setup_mod.DONE.exists()


@pytest.fixture
def oneclick(tmp_path, monkeypatch):
    """gpu_training/setup_and_train.py with its two steps recorded instead of run."""
    mod = _load("setup_and_train")
    vpy = tmp_path / "python"
    vpy.write_text("")
    monkeypatch.setattr(mod, "VPY", vpy)
    calls: list = []
    codes = {"setup": 0, "train": 0}

    def fake_run(cmd):
        step = "setup" if "gpu_training/setup.py" in cmd else "train"
        calls.append((step, [str(c) for c in cmd]))
        return codes[step]

    monkeypatch.setattr(mod, "run", fake_run)
    return mod, calls, codes


def test_double_click_sets_up_if_needed_then_trains(oneclick):
    mod, calls, _codes = oneclick
    assert mod.main([]) == 0
    assert [step for step, _cmd in calls] == ["setup", "train"]
    assert calls[0][1] == [sys.executable, "gpu_training/setup.py", "--if-needed"]
    assert calls[1][1] == [str(mod.VPY), "gpu_training/train_all.py"]


def test_limits_reach_both_steps_and_workers_the_training(oneclick):
    mod, calls, _codes = oneclick
    assert mod.main(["--ram-gb", "12", "--gpu-memory-gb", "10", "--workers", "6"]) == 0
    setup, train = calls[0][1], calls[1][1]
    assert setup[2:] == ["--ram-gb", "12", "--gpu-memory-gb", "10", "--if-needed"]
    assert train[2:] == ["--ram-gb", "12", "--gpu-memory-gb", "10", "--workers", "6"]


def test_redo_setup_runs_the_whole_setup(oneclick):
    mod, calls, _codes = oneclick
    assert mod.main(["--redo-setup"]) == 0
    assert "--if-needed" not in calls[0][1]


def test_no_training_after_a_failed_setup(oneclick, capsys):
    mod, calls, codes = oneclick
    codes["setup"] = 1
    assert mod.main([]) == 1
    assert [step for step, _cmd in calls] == ["setup"]
    assert "the training was not started" in capsys.readouterr().out


@pytest.mark.parametrize("rc, says", [(3, "Paused"), (130, "last saved state is kept"),
                                      (4, "RAM budget"), (1, "stopped (exit code 1)")])
def test_the_training_outcome_is_explained(oneclick, capsys, rc, says):
    mod, _calls, codes = oneclick
    codes["train"] = rc
    assert mod.main([]) == rc
    assert says in capsys.readouterr().out


def test_windows_scripts_keep_crlf_and_find_the_one_click_script():
    bat = (GPU / "setup_and_train.bat").read_bytes()
    assert bat.count(b"\r\n") == bat.count(b"\n")  # every line ends with CRLF
    text = bat.decode("ascii")
    assert 'cd /d "%~dp0\\.."' in text
    assert "gpu_training\\setup_and_train.py %*" in text
    assert text.rstrip().endswith("exit /b %RC%")


def test_batch_size_and_restart_reach_the_training(oneclick):
    mod, calls, _codes = oneclick
    assert mod.main(["--batch-size", "16", "--restart"]) == 0
    assert calls[0][1][-1] == "--if-needed" and "--batch-size" not in calls[0][1]
    assert calls[1][1][2:] == ["--batch-size", "16", "--restart"]


#: what setup measured on the GPU at the recipe's batch of 4: the comb view needs more
MACHINE = {"gpu_views": {"window": {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.2},
                         "comb": {"batch": 4, "peak_gb": 5.0, "step_seconds": 0.25}}}


@pytest.mark.parametrize("view, batch, fits, largest", [
    ("window", 4, True, None), ("window", 16, True, None), ("window", 18, True, None),
    ("window", 20, False, 18), ("comb", 8, True, None), ("comb", 12, False, 11),
    ("comb", 1, True, None)])
def test_a_larger_batch_must_fit_under_the_gpu_cap_in_each_view(view, batch, fits, largest):
    train_all = _load("train_all")
    problem = train_all.batch_problem(batch, MACHINE, 14.0, view)
    assert (problem is None) == fits, problem
    if not fits:
        assert f"--batch-size {largest}" in problem and f"the {view} view" in problem
    assert "at least 1" in train_all.batch_problem(0, MACHINE, 14.0, view)


def test_batch_check_needs_a_measurement_of_this_recipe():
    train_all = _load("train_all")
    # nothing measured, or an earlier package's measurement (one peak at its batch of 8)
    for machine in ({}, {"gpu_peak_gb": 4.0}, {"gpu_views": {"comb": {"batch": 4}}}):
        assert train_all.batch_problem(64, machine, 14.0, "comb") is None
    # a cap lowered below what setup measured
    lowered = train_all.batch_problem(4, MACHINE, 4.0, "comb")
    assert lowered is not None and "--batch-size 3" in lowered


def test_runs_are_chosen_by_name():
    train_all = _load("train_all")
    assert train_all.parse_runs(None) == ["window", "comb"]
    assert train_all.parse_runs("comb, window-plain,comb") == ["comb", "window-plain"]
    with pytest.raises(SystemExit, match="unknown run stage1"):
        train_all.parse_runs("stage1")
    runs = {n: train_all.make_run(n) for n in train_all.RUN_TABLE}
    assert {n: (r["view"], r["out"]) for n, r in runs.items()} == {
        "window": ("window", "runs/ladder_v5_window.pt"),
        "comb": ("comb", "runs/ladder_v5_comb.pt"),
        "window-plain": ("window", "runs/ladder_v5_window_plain.pt"),
        "comb-plain": ("comb", "runs/ladder_v5_comb_plain.pt")}
    for name, run in runs.items():
        a = run["args"]
        assert a[a.index("--recipe") + 1] == "v5" and a[a.index("--view") + 1] == run["view"]
        assert ("--iso-channels" in a) == (not name.endswith("plain"))
        for flag, value in (("--model-size", "full"), ("--singly-charged", "keep"),
                            ("--epochs", "20"), ("--train-len", "6000"), ("--val-len", "128"),
                            ("--lr", "2e-3"), ("--seed", "0")):
            assert a[a.index(flag) + 1] == value, flag
        assert run["spectra"] == 120000
    smoke = train_all.make_run("comb", smoke=True)
    assert smoke["out"] == "runs/smoke_v5_comb.pt"
    a = smoke["args"]
    assert a[a.index("--epochs", a.index("--epochs") + 1) + 1] == "1"  # the smoke's own last
    assert a[-2:] == ["--batch-size", "2"]


def test_time_estimate_takes_the_slower_of_workers_and_gpu():
    train_all = _load("train_all")
    machine = {**MACHINE, "v5_spectra_per_second": 30.0, "v5_workers": 10}
    runs = [train_all.make_run("window"), train_all.make_run("comb")]
    text = train_all.time_estimate(runs, machine, 10)
    # window: the workers' 30 spectra/s (the GPU could take 20 at 0.2 s per 4); comb: 16
    assert "window 1.8 h at 20 spectra/s" in text and "comb 2.3 h at 16 spectra/s" in text
    assert "about 4.1 h" in text
    assert train_all.time_estimate(runs, {}, 10) is None


def test_the_two_stage_scripts_of_an_earlier_package_are_answered(monkeypatch, capsys):
    train_all = _load("train_all")
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--only", "1"])
    assert train_all.main() == 1
    assert "two-stage recipe" in capsys.readouterr().out


def test_setup_reruns_the_measurement_when_the_training_code_changes(setup_mod):
    _record(setup_mod)
    for rel in ("gpu_training/train_all.py", "src/lcmsdeconv/nn/model.py",
                "scripts/train_charge_net.py"):
        assert rel in setup_mod.SETUP_INPUTS
    (setup_mod.ROOT / "src/lcmsdeconv/nn/model.py").write_text("a new network\n")
    holds, why = setup_mod.setup_is_current()
    assert not holds and "package has changed" in why


def test_runs_reach_the_training(oneclick):
    mod, calls, _codes = oneclick
    assert mod.main(["--runs", "comb"]) == 0
    assert calls[1][1][2:] == ["--runs", "comb"]


def test_the_upload_holds_every_finished_run(tmp_path, monkeypatch):
    import zipfile

    train_all = _load("train_all")
    monkeypatch.setattr(train_all, "ROOT", tmp_path)
    monkeypatch.setattr(train_all, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(train_all, "MACHINE", tmp_path / "gpu_training" / "machine.json")
    (tmp_path / "runs").mkdir()
    for name, status in (("window", "finished"), ("comb", None), ("comb-plain", "finished")):
        base = tmp_path / train_all.RUN_TABLE[name]["out"]
        for suffix in (".pt", ".onnx", ".json", ".log.jsonl", ".resume.pt"):
            base.with_suffix(suffix).write_text(name)
        base.with_suffix(".status.json").write_text(json.dumps(
            {"epoch": 20 if status else 3, "step": 0, "epochs": 20, "stop_reason": status}))
    chosen = [train_all.make_run("comb")]  # trained now; window and comb-plain finished before
    runs = train_all.finished_runs(chosen)
    assert [r["name"] for r in runs] == ["comb", "window", "comb-plain"]
    names = zipfile.ZipFile(train_all.pack(runs, "")).namelist()
    assert "runs/ladder_v5_window.onnx" in names and "runs/ladder_v5_comb_plain.json" in names
    assert "runs/ladder_v5_comb.status.json" in names
    assert not any(n.endswith(".resume.pt") for n in names)


def _with_unchunked(whole):
    """MACHINE with the window view's step measured with all charges at once: (peak GB, step s)."""
    views = {v: dict(m) for v, m in MACHINE["gpu_views"].items()}
    if whole:
        views["window"]["unchunked"] = {"batch": 4, "peak_gb": whole[0], "step_seconds": whole[1]}
    return {"gpu_views": views}


@pytest.mark.parametrize("whole, gpu_gb, batch, failed, chunk", [
    ((8.0, 0.15), 14.0, 4, [], None),        # 8 x 1.3 = 10.4 GB under 14 - 0.75, and faster
    ((10.5, 0.15), 14.0, 4, [], 16),         # 13.7 GB: too little room
    ((8.0, 0.15), 11.0, 4, [], 16),          # a lower cap
    ((8.0, 0.195), 14.0, 4, [], 16),         # hardly faster than the chunked 0.2 s
    ((8.0, 0.15), 14.0, 8, [], 16),          # a larger batch than measured
    ((8.0, 0.15), 14.0, 2, [], None),        # a smaller one
    ((8.0, 0.15), 14.0, 4, ["window"], 16),  # a run of this view ran out of GPU memory
    ((None, None), 14.0, 4, [], 16),         # out of GPU memory in setup
    (None, 14.0, 4, [], 16),                 # not measured (an earlier package's setup)
])
def test_all_charges_at_once_only_where_measured_faster_with_room(whole, gpu_gb, batch, failed,
                                                                 chunk):
    train_all = _load("train_all")
    machine = {**_with_unchunked(whole), "unchunked_failed": failed}
    assert train_all.head_chunk_for(machine, "window", gpu_gb, batch) == chunk
    assert train_all.head_chunk_for(machine, "comb", gpu_gb, batch) == 16  # measured chunked only


def test_time_estimate_follows_the_chunk_choice():
    train_all = _load("train_all")
    machine = {**_with_unchunked((8.0, 0.1)), "v5_spectra_per_second": 30.0, "v5_workers": 10}
    run = train_all.make_run("window")
    assert "window 1.8 h at 20 spectra/s" in train_all.time_estimate([run], machine, 10)
    run["head_chunk"] = None  # the GPU could then take 40 spectra/s: the workers set the pace
    assert "window 1.2 h at 30 spectra/s" in train_all.time_estimate([run], machine, 10)


def _train_all_in(tmp_path, monkeypatch):
    train_all = _load("train_all")
    monkeypatch.setattr(train_all, "ROOT", tmp_path)
    monkeypatch.setattr(train_all, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(train_all, "MACHINE", tmp_path / "gpu_training" / "machine.json")
    (tmp_path / "runs").mkdir()
    (tmp_path / "gpu_training").mkdir()
    train_all.MACHINE.write_text(json.dumps({"workers": 6}))
    return train_all


class _FakeTraining:
    """subprocess.Popen for train_all: records the command, prints ``lines`` and exits ``rc``."""

    def __init__(self, record: list, script: list):
        self.record, self.script = record, script

    def __call__(self, cmd, **_kw):
        self.record.append(cmd)
        lines, rc = self.script[min(len(self.record), len(self.script)) - 1]
        proc = type("Proc", (), {})()
        proc.stdout, proc.wait = iter(lines), (lambda: rc)
        return proc


@pytest.mark.parametrize("state, then", [
    ("none", "--restart"),    # nothing saved before the failure
    ("saved", "--resume"),    # a state saved before it: resumed from
    ("stale", "--restart"),   # a restart's failure before it saved: not the state it set aside
])
def test_a_run_out_of_gpu_memory_unchunked_continues_in_chunks(tmp_path, monkeypatch, state,
                                                               then):
    import os
    import time

    train_all = _train_all_in(tmp_path, monkeypatch)
    stage = train_all.make_run("window")
    stage["head_chunk"] = None
    out = tmp_path / stage["out"]
    if state != "none":
        out.with_suffix(".resume.pt").write_text("state")
        out.with_suffix(".status.json").write_text(json.dumps({"epoch": 3, "epochs": 20}))
    if state == "stale":
        old = time.time() - 3600
        os.utime(out.with_suffix(".resume.pt"), (old, old))
    commands = []
    oom = ["epoch 3 step 20/1500\n", "torch.OutOfMemoryError: CUDA out of memory. Tried to\n"]
    monkeypatch.setattr(train_all.subprocess, "Popen",
                        _FakeTraining(commands, [(oom, 1), (["done\n"], 0)]))
    restart = state == "stale"
    assert train_all.run_stage(stage, {"workers": 6}, [], "cuda", restart) == 0
    assert len(commands) == 2 and "--head-chunk" not in commands[0]
    assert commands[0][-1] == ("--resume" if state == "saved" else "--restart")
    assert commands[1][commands[1].index("--head-chunk") + 1] == "16"
    assert commands[1][-1] == then
    machine = json.loads(train_all.MACHINE.read_text())
    assert machine["unchunked_failed"] == ["window"] and machine["workers"] == 6
    assert train_all.head_chunk_for({**_with_unchunked((8.0, 0.1)), **machine}, "window",
                                    14.0, 4) == 16


def test_other_failures_and_chunked_runs_are_not_retried(tmp_path, monkeypatch):
    train_all = _train_all_in(tmp_path, monkeypatch)
    commands = []
    oom = ["torch.OutOfMemoryError: CUDA out of memory.\n"]
    monkeypatch.setattr(train_all.subprocess, "Popen",
                        _FakeTraining(commands, [(oom, 1), (["done\n"], 0)]))
    chunked = train_all.make_run("comb")  # already in chunks: nothing left to try
    assert train_all.run_stage(chunked, {"workers": 6}, [], "cuda", False) == 1
    whole = train_all.make_run("window")
    whole["head_chunk"] = None
    commands.clear()
    monkeypatch.setattr(train_all.subprocess, "Popen",
                        _FakeTraining(commands, [(["ValueError: no\n"], 1), (["done\n"], 0)]))
    assert train_all.run_stage(whole, {"workers": 6}, [], "cuda", False) == 1
    assert len(commands) == 1 and "unchunked_failed" not in train_all.MACHINE.read_text()


def test_the_smoke_always_starts_afresh_and_checks_again(tmp_path, monkeypatch):
    from types import SimpleNamespace

    train_all = _train_all_in(tmp_path, monkeypatch)
    for name in train_all.DEFAULT_RUNS:  # an earlier setup's smoke, finished and exported
        base = tmp_path / train_all.make_run(name, smoke=True)["out"]
        for suffix in (".pt", ".resume.pt", ".onnx"):
            base.with_suffix(suffix).write_text("earlier package")
        base.with_suffix(".status.json").write_text(
            json.dumps({"epoch": 1, "epochs": 1, "stop_reason": "finished"}))
    commands, checks = [], []
    monkeypatch.setattr(train_all.subprocess, "Popen", _FakeTraining(commands, [([], 0)]))
    monkeypatch.setattr(train_all.subprocess, "run",
                        lambda cmd, **_kw: checks.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(train_all.signal, "signal", lambda *_a: None)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--smoke", "--device", "cpu",
                                      "--workers", "1"])
    assert train_all.main() == 0
    assert len(commands) == 2 and all(c[-1] == "--restart" for c in commands)
    assert sum("export-onnx" in c for c in checks) == 2
    assert sum("gpu_training/check_model.py" in c for c in checks) == 2


def test_a_model_failing_the_predictor_check_is_checked_again(tmp_path, monkeypatch):
    from types import SimpleNamespace

    train_all = _train_all_in(tmp_path, monkeypatch)
    stage = train_all.make_run("comb")
    ckpt = tmp_path / stage["out"]
    ckpt.write_text("checkpoint")
    calls, check = [], {"rc": 1}

    def fake_run(cmd, **_kw):
        calls.append(cmd)
        if "export-onnx" in cmd:
            ckpt.with_suffix(".onnx").write_text("model")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=check["rc"])

    monkeypatch.setattr(train_all.subprocess, "run", fake_run)
    assert train_all.export(stage) is None
    assert ckpt.with_suffix(".onnx").exists() and ckpt.with_suffix(".unchecked").exists()
    names = __import__("zipfile").ZipFile(train_all.pack([stage], "")).namelist()
    assert "runs/ladder_v5_comb.onnx" in names and "runs/ladder_v5_comb.unchecked" in names
    check["rc"] = 0
    assert train_all.export(stage) == ckpt.with_suffix(".onnx") and len(calls) == 4
    assert not ckpt.with_suffix(".unchecked").exists()
    assert train_all.export(stage) == ckpt.with_suffix(".onnx") and len(calls) == 4  # done


@pytest.mark.parametrize("rel", ["src/lcmsdeconv/synth/dataset.py",
                                 "src/lcmsdeconv/chem/isotopes.py"])
def test_setup_runs_again_when_any_package_code_changes(setup_mod, rel):
    other = setup_mod.ROOT / rel  # not in SETUP_INPUTS
    assert rel not in setup_mod.SETUP_INPUTS
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("frames\n")
    _record(setup_mod)
    assert setup_mod.setup_is_current()[0]
    other.write_text("other frames\n")
    holds, why = setup_mod.setup_is_current()
    assert not holds and "package has changed" in why


def test_all_charges_at_once_only_where_the_gpu_sets_the_pace():
    train_all = _load("train_all")
    machine = _with_unchunked((8.0, 0.15))  # chunked: 4 spectra in 0.2 s, 20 per second
    assert train_all.head_chunk_for(machine, "window", 14.0, 4, 30.0) is None
    assert train_all.head_chunk_for(machine, "window", 14.0, 4, 20.0) == 16  # workers' pace
    assert train_all.head_chunk_for(machine, "window", 14.0, 4, None) is None  # not measured
    assert train_all.data_rate({"v5_spectra_per_second": 30.0, "v5_workers": 10}, 5) == 15.0
    assert train_all.data_rate({}, 5) is None
    # measured on the spectra of an earlier recipe: no rate for this one
    assert train_all.data_rate({"v3_spectra_per_second": 30.0, "v3_workers": 10}, 5) is None


def test_an_interrupted_export_or_check_is_done_again(tmp_path, monkeypatch):
    from types import SimpleNamespace

    train_all = _train_all_in(tmp_path, monkeypatch)
    stage = train_all.make_run("comb")
    ckpt = tmp_path / stage["out"]
    ckpt.write_text("checkpoint")
    calls = []

    def closed_during_the_check(cmd, **_kw):
        calls.append(cmd)
        if "export-onnx" in cmd:
            ckpt.with_suffix(".onnx").write_text("model")
            return SimpleNamespace(returncode=0)
        raise KeyboardInterrupt

    monkeypatch.setattr(train_all.subprocess, "run", closed_during_the_check)
    with pytest.raises(KeyboardInterrupt):
        train_all.export(stage)
    assert ckpt.with_suffix(".onnx").exists() and ckpt.with_suffix(".unchecked").exists()
    monkeypatch.setattr(train_all.subprocess, "run",
                        lambda cmd, **_kw: calls.append(cmd) or SimpleNamespace(returncode=0))
    assert train_all.export(stage) == ckpt.with_suffix(".onnx")
    assert sum("export-onnx" in c for c in calls) == 2
    assert not ckpt.with_suffix(".unchecked").exists()


def test_a_view_out_of_gpu_memory_stays_in_chunks_for_its_later_runs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    train_all = _train_all_in(tmp_path, monkeypatch)
    train_all.MACHINE.write_text(json.dumps({**_with_unchunked((8.0, 0.1)), "workers": 6,
                                             "v5_spectra_per_second": 60.0, "v5_workers": 6}))
    for name in ("window", "window-plain"):
        (tmp_path / train_all.RUN_TABLE[name]["out"]).write_text("checkpoint")
    commands = []
    oom = ["torch.OutOfMemoryError: CUDA out of memory.\n"]
    monkeypatch.setattr(train_all.subprocess, "Popen",
                        _FakeTraining(commands, [(oom, 1), ([], 0), ([], 0)]))
    monkeypatch.setattr(train_all.subprocess, "run",
                        lambda cmd, **_kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(train_all.signal, "signal", lambda *_a: None)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--runs", "window,window-plain",
                                      "--device", "cpu"])
    assert train_all.main() == 0
    chunks = [c[c.index("--head-chunk") + 1] if "--head-chunk" in c else "all" for c in commands]
    assert chunks == ["all", "16", "16"]  # window, its fallback, then window-plain


@pytest.mark.parametrize("cap, kept", [(14.0, True), (12.0, False)])
def test_measuring_again_keeps_a_view_out_of_gpu_memory_in_chunks(tmp_path, monkeypatch, cap,
                                                                  kept):
    import torch

    import lcmsdeconv.nn.machine as nn_machine

    probe = _load("probe")
    step = {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.8,
            "unchunked": {"batch": 4, "peak_gb": 5.0, "step_seconds": 0.5}}
    views = {"window": dict(step), "comb": dict(step)}
    monkeypatch.setattr(probe, "gpu_steps", lambda *_a: {"gpu": "Fake GPU", "gpu_total_gb": 16.0,
                                                         "gpu_views": views})
    monkeypatch.setattr(probe, "workers_run", lambda n, **_k: {
        "workers": n, "startup_seconds": 1.0, "spectra_per_second": 10.0 * n, "worker_gb": 0.5,
        "worker_commit_gb": 0.6, "workers_total_gb": 0.5 * n})
    asked = []

    # on a GPU, by processes of their own (tests/test_gpu_speedups.py checks how)
    def speedups(view_names, batch, chunks, gpu_gb):
        asked.append((view_names, batch, chunks, gpu_gb))
        return {"gpu_speedups": {"window": ["tf32"], "comb": []},
                "speedup_timings": {"window": {"none": 500.0, "tf32": 450.0}, "comb": {}}}

    monkeypatch.setattr(probe, "measure_speedups", speedups)
    # how the data workers run fastest: not measured here (tests/test_probe_hybrid.py)
    monkeypatch.setattr(probe, "measure_data_loading", lambda *_a, **_k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nn_machine, "single_threaded_workers", lambda: None)
    out = tmp_path / "machine.json"
    out.write_text(json.dumps({"gpu": "Fake GPU", "gpu_memory_gb": 14.0, "gpu_batch": 4,
                               "unchunked_failed": ["window"]}))
    monkeypatch.setattr(sys, "argv", ["probe.py", "--workers", "4", "--gpu-memory-gb", str(cap),
                                      "--out", str(out)])
    assert probe.main() == 0
    info = json.loads(out.read_text())
    assert info.get("unchunked_failed") == (["window"] if kept else None)
    assert info["head_chunks"] == {"window": 16 if kept else None, "comb": None}
    # the speedups are measured with the chunking each view's run will use
    assert asked == [(["window", "comb"], 4, {"window": 16 if kept else None, "comb": None}, cap)]
    assert info["gpu_speedups"] == {"window": ["tf32"], "comb": []}
    assert info["recipe"] == "v5" and info["v5_workers"] == 4


# ------------------------------------------------------------------ data workers and RAM budget
def _computer(monkeypatch, total_gb: float, cores: int):
    """lcmsdeconv.nn.machine on a computer of ``total_gb`` GB of memory, as psutil reads it, and
    ``cores`` cores."""
    import psutil

    import lcmsdeconv.nn.machine as nn_machine

    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(total=int(total_gb * 2**30)))
    monkeypatch.setattr(nn_machine, "cpu_cores", lambda: cores)
    return nn_machine


@pytest.mark.parametrize("total, budget", [
    (8, 4), (16, 12), (32, 16), (64, 32), (96, 48), (128, 48),
    # what Windows reports of the memory fitted, a little less
    (7.7, 4), (15.7, 12), (31.8, 16), (63.7, 32), (95.7, 48), (127.7, 48)])
def test_the_default_ram_budget_follows_the_computers_memory(monkeypatch, total, budget):
    """Half the memory, at least 16 GB and at most 48, and 4 GB left to the system."""
    nn_machine = _computer(monkeypatch, total, 24)
    assert nn_machine.physical_ram_gb() == pytest.approx(total)
    assert nn_machine.default_ram_budget_gb() == budget
    assert nn_machine.default_ram_budget_gb(total) == budget


def test_without_psutil_the_ram_budget_is_16_gb(monkeypatch):
    import lcmsdeconv.nn.machine as nn_machine

    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil fails
    assert nn_machine.physical_ram_gb() is None
    assert nn_machine.default_ram_budget_gb() == 16.0
    assert nn_machine.half_ram_gb() == 8.0  # the training script's, as before


@pytest.mark.parametrize("total, workers", [(128, 22), (96, 22), (64, 22), (32, 13), (16, 9)])
def test_the_processor_sets_the_workers_where_the_memory_holds_them(monkeypatch, total, workers):
    """The user's computer: 24 cores, 0.88 GB a worker and 2.4 GB for the training process, as
    setup measured them, within setup's default budget. From 64 GB on, the processor's 22 (every
    core but the two the training process keeps); on 32 GB the 13 of a 16 GB budget, as before;
    an explicit budget wins."""
    nn_machine = _computer(monkeypatch, total, 24)
    assert nn_machine.cpu_workers() == 22
    budget = nn_machine.default_ram_budget_gb()
    assert nn_machine.auto_workers(budget, worker_gb=0.88, main_gb=2.4) == workers
    assert nn_machine.auto_workers(16.0, worker_gb=0.88, main_gb=2.4) == 13
    assert [nn_machine.cpu_workers(c) for c in (1, 2, 4, 8, 12, 24)] == [1, 1, 3, 6, 10, 22]


@pytest.mark.parametrize("total, cores, workers", [
    (16, 24, 2), (20, 24, 4), (24, 24, 5), (31.8, 24, 9), (64, 24, 22), (128, 24, 22),
    (16, 8, 2), (32, 8, 6)])
def test_the_training_script_takes_half_the_memory_without_a_budget(monkeypatch, total, cores,
                                                                    workers):
    """scripts/train_charge_net.py without --max-ram-gb (auto_workers without a budget): half the
    computer's memory with a worker taken as WORKER_GB, as before setup's default followed the
    memory, so no 16 GB below a computer of 32 GB."""
    nn_machine = _computer(monkeypatch, total, cores)
    assert nn_machine.half_ram_gb() == pytest.approx(total / 2)
    assert nn_machine.auto_workers() == workers


def _hybrid_rate(n: int) -> float:
    """Spectra per second of ``n`` data workers on a processor of 8 performance and 16 efficient
    cores, a model of the user's (setup: 10 workers alone at 23.4; training: 20 at 26-27): each
    worker beyond the eighth adds less. 22 workers: 27.8."""
    return 2.6 * min(n, 8) + 0.5 * max(0, n - 8)


def _probe_on(monkeypatch, tmp_path, total: float, cores: int = 24, worker_gb: float = 0.88,
              fails=lambda n: False, argv=(), gpu=None, speedups=None):
    """probe.py on a computer of ``total`` GB and ``cores`` cores, its data workers stood in for:
    each measurement is recorded, delivers _hybrid_rate(n) spectra per second and takes
    ``worker_gb`` a worker, or fails to run where ``fails(n)``. Without ``gpu`` (gpu_steps'
    result) the GPU is not measured; ``speedups`` is measure_speedups' result. Returns
    machine.json and the numbers of workers measured."""
    import torch

    probe = _load("probe")
    nn_machine = _computer(monkeypatch, total, cores)
    monkeypatch.setattr(nn_machine, "single_threaded_workers", lambda: None)
    monkeypatch.setattr(probe, "process_gb", lambda proc: 1.5)  # 2.4 GB with the validation
    asked = []

    def workers_run(n, **_kw):
        asked.append(n)
        if fails(n):
            raise OSError("[WinError 1455] The paging file is too small for this operation")
        return {"workers": n, "startup_seconds": 30.0, "timed_seconds": 60.0,
                "timed_spectra": int(60 * _hybrid_rate(n)),
                "spectra_per_second": round(_hybrid_rate(n), 2), "worker_gb": worker_gb,
                "worker_commit_gb": 1.6, "workers_total_gb": worker_gb * n}

    monkeypatch.setattr(probe, "workers_run", workers_run)
    # how the data workers run fastest: not measured here (tests/test_probe_hybrid.py)
    monkeypatch.setattr(probe, "measure_data_loading", lambda *_a, **_k: None)
    if gpu is not None:
        monkeypatch.setattr(probe, "gpu_steps", lambda *_a: gpu)
        monkeypatch.setattr(probe, "measure_speedups", lambda *_a: speedups)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    out = tmp_path / "machine.json"
    monkeypatch.setattr(sys, "argv", ["probe.py", *([] if gpu else ["--no-gpu"]), "--out",
                                      str(out), *argv])
    assert probe.main() == 0
    return json.loads(out.read_text()), asked


@pytest.mark.parametrize("total, worker_gb, asked, chosen, budget", [
    (96, 0.88, [22], 22, 48),     # the processor's share at once: a 48 GB budget holds it
    (32, 0.88, [10, 13], 13, 16),  # 10 by the safe estimate of a worker, 13 by its measurement
    (32, 2.0, [10, 6], 6, 16),     # workers larger than the estimate: fewer, measured again
])
def test_the_probe_measures_the_workers_at_the_number_it_chooses(tmp_path, monkeypatch, total,
                                                                 worker_gb, asked, chosen,
                                                                 budget):
    info, measured = _probe_on(monkeypatch, tmp_path, total, worker_gb=worker_gb)
    assert measured == asked and [m["workers"] for m in info["measured"]] == asked
    assert info["workers"] == info["v5_workers"] == chosen
    # the speed of that many workers, not of fewer scaled up
    assert info["v5_spectra_per_second"] == round(_hybrid_rate(chosen), 1)
    assert (info["ram_budget_gb"], info["ram_budget_default"]) == (budget, True)
    assert info["physical_ram_gb"] == total and info["processor_workers"] == 22
    assert info["pace"]["window"]["set_by"] == "workers"  # the GPU was not measured


def test_the_probe_keeps_fewer_workers_where_more_do_not_run(tmp_path, monkeypatch, capsys):
    # 22 do not start (the paging file): half as many are measured and chosen
    info, asked = _probe_on(monkeypatch, tmp_path, 96, fails=lambda n: n > 12)
    assert asked == [22, 11] and info["workers"] == 11
    assert info["v5_spectra_per_second"] == round(_hybrid_rate(11), 1)
    assert "as many as ran: 22 did not" in capsys.readouterr().out
    # the 13 the memory holds do not run: the 10 measured first are kept, not scaled up, and
    # the reason is that more did not run, not the budget
    info, asked = _probe_on(monkeypatch, tmp_path, 32, fails=lambda n: n > 12)
    assert asked == [10, 13] and info["workers"] == 10
    assert info["v5_spectra_per_second"] == round(_hybrid_rate(10), 1)
    text = capsys.readouterr().out
    assert "Chosen: 10 data workers (as many as ran: 13 did not)" in text
    assert "as many as the RAM budget holds" not in text
    # 22 do not start and 11 do, but workers of 5 GB fit the budget of 48 GB only 8 times: the
    # budget sets the number, measured again at 8
    info, asked = _probe_on(monkeypatch, tmp_path, 96, worker_gb=5.0, fails=lambda n: n > 12)
    assert asked == [22, 11, 8] and info["workers"] == 8
    assert "Chosen: 8 data workers (as many as the RAM budget holds" in capsys.readouterr().out


def test_workers_and_budget_asked_for_win(tmp_path, monkeypatch):
    info, asked = _probe_on(monkeypatch, tmp_path, 96, argv=["--workers", "8"])
    assert asked == [8] and info["workers"] == info["v5_workers"] == 8
    info, asked = _probe_on(monkeypatch, tmp_path, 96, argv=["--ram-gb", "16"])
    assert asked == [10, 13] and info["workers"] == 13
    assert (info["ram_budget_gb"], info["ram_budget_default"]) == (16, False)


class _SlowSpectra:
    """Spectra that take ``seconds`` each to make, asleep (the host's load does not matter)."""

    def __init__(self, n: int, seconds: float):
        self.n, self.seconds = n, seconds

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import time

        import torch

        time.sleep(self.seconds)
        return torch.zeros(8), torch.zeros(2)


def _multiprocessing_helpers() -> set:
    """Process ids of multiprocessing's own helpers, which outlive the data workers they served
    (until this process ends): the resource tracker that the spawn and forkserver start methods
    run, and the fork server."""
    pids = set()
    for module, helper, attr in (("resource_tracker", "_resource_tracker", "_pid"),
                                 ("forkserver", "_forkserver", "_forkserver_pid")):
        try:
            mod = importlib.import_module(f"multiprocessing.{module}")
        except ImportError:  # not on this platform
            continue
        pid = getattr(getattr(mod, helper, None), attr, None)
        if pid:
            pids.add(pid)
    return pids


def test_the_workers_are_timed_in_full_flow_then_stopped(monkeypatch):
    """With the platform's start method: fork (Linux up to Python 3.13), spawn (Windows, macOS)
    or forkserver (Linux from Python 3.14)."""
    import psutil

    probe = _load("probe")
    monkeypatch.delenv("LCMSDECONV_MP_CONTEXT", raising=False)
    monkeypatch.setattr(probe, "WORKER_SECONDS", 1.5)
    monkeypatch.setattr(probe, "MEMORY_EVERY", 0.5)
    me = psutil.Process()
    before = {p.pid for p in me.children(recursive=True)}  # left by earlier tests
    run = probe.workers_run(2, dataset=_SlowSpectra(1_000_000, 0.02))
    # two workers, each a batch of 4 in 0.08 s: 100 spectra per second together
    assert run["workers"] == 2 and run["timed_seconds"] >= 1.5
    assert 60.0 <= run["spectra_per_second"] <= 110.0
    assert run["timed_spectra"] >= 3 * 2 * 4  # three rounds at least, after two to warm up
    assert run["workers_total_gb"] >= run["worker_gb"] >= 0.0
    left = before | _multiprocessing_helpers()
    alive = [p for p in me.children(recursive=True)
             if p.pid not in left and p.status() != psutil.STATUS_ZOMBIE]
    assert alive == []  # the workers ended with the measurement


def test_the_probe_compares_the_workers_with_the_gpu_on_a_real_batch(tmp_path, monkeypatch,
                                                                     capsys):
    """The user's GPU: 153 ms a step at the active-bin cap (26 spectra/s); 22 workers deliver
    27.8 spectra/s (the model). On the runs' first training batch the GPU takes 95 ms with its
    speedups (42 spectra/s): the workers set the pace; at 180 ms (22 spectra/s) the GPU would."""
    step = {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.153}
    gpu = {"gpu": "Fake GPU", "gpu_total_gb": 16.0,
           "gpu_views": {"window": dict(step), "comb": dict(step)}}
    for ms, set_by, verdict in ((95.0, "workers", "the data workers set the pace"),
                                (180.0, "gpu", "the GPU sets the pace")):
        speedups = {"gpu_speedups": {"window": ["tf32"], "comb": []},
                    "speedup_timings": {"window": {"none": 1.1 * ms, "tf32": ms},
                                        "comb": {"none": ms}}}
        info, asked = _probe_on(monkeypatch, tmp_path, 96, gpu=gpu, speedups=speedups)
        text = capsys.readouterr().out
        assert asked == [22] and info["head_chunks"] == {"window": 16, "comb": 16}
        pace = info["pace"]["window"]
        assert pace["set_by"] == set_by and pace["data_spectra_per_second"] == 27.8
        assert pace["gpu_spectra_per_second"] == round(4 / (ms / 1000), 1)
        assert pace["gpu_at_cap_spectra_per_second"] == 26.1
        assert (f"window run: the GPU took {ms:.0f} ms a step on the runs' first training batch "
                f"({4000 / ms:.0f} spectra/s) and 153 ms with every spectrum at the active-bin "
                f"cap (26 spectra/s, the slowest a step can be)") in text
        assert verdict in text and "Chosen: 22 data workers (the processor's share" in text
        assert "the default for this computer's 96 GB of memory" in text
        assert info["expected_hours"] <= info["expected_hours_at_most"]


#: machine.json of the setup before the budget followed the computer (the user's: 16 GB, 13
#: workers, their speed scaled from 10)
OLD_MACHINE = {
    "recipe": "v5", "workers": 13, "ram_budget_gb": 16.0, "gpu_memory_gb": 14.0, "gpu_batch": 4,
    "head_chunk": 16, "head_chunks": {"window": 16, "comb": 16}, "main_gb": 2.4,
    "worker_gb": 0.88, "expected_ram_gb": 13.8, "v5_spectra_per_second": 30.4, "v5_workers": 13,
    "expected_hours": 2.8, "cores": 24, "gpu": "Fake GPU", "gpu_total_gb": 16.0,
    "gpu_views": {"window": {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.153},
                  "comb": {"batch": 4, "peak_gb": 3.2, "step_seconds": 0.16}},
    "gpu_speedups": {"window": ["tf32"], "comb": []},
    "speedup_timings": {"window": {"none": 100.0, "tf32": 95.0}, "comb": {"none": 105.0}}}
#: the same computer measured by this setup: the default budget, 20 workers measured as 20
NEW_MACHINE = {**OLD_MACHINE, "workers": 20, "ram_budget_gb": 48.0, "ram_budget_default": True,
               "physical_ram_gb": 95.7, "v5_spectra_per_second": 26.8, "v5_workers": 20,
               "processor_workers": 20}
#: an earlier package's: no record of the speedups or of the chunking they were measured with
OLDER_MACHINE = {k: v for k, v in OLD_MACHINE.items()
                 if k not in ("head_chunks", "gpu_speedups", "speedup_timings")}


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


@pytest.mark.parametrize("which", ["older", "old", "new", "none"])
def test_train_all_reads_the_machine_json_of_any_setup(tmp_path, monkeypatch, capsys, which):
    _computer(monkeypatch, 96, 24)
    train_all = _train_all_in(tmp_path, monkeypatch)
    machine = {"older": OLDER_MACHINE, "old": OLD_MACHINE, "new": NEW_MACHINE}.get(which)
    if machine is None:
        train_all.MACHINE.unlink()
    else:
        train_all.MACHINE.write_text(json.dumps(machine))
    for name in train_all.DEFAULT_RUNS:  # what the trainings leave, for the export
        (tmp_path / train_all.RUN_TABLE[name]["out"]).write_text("checkpoint")
    commands = []
    monkeypatch.setattr(train_all.subprocess, "Popen", _FakeTraining(commands, [([], 0)]))
    monkeypatch.setattr(train_all.subprocess, "run",
                        lambda cmd, **_kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(train_all.signal, "signal", lambda *_a: None)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu"])
    assert train_all.main() == 0
    text = capsys.readouterr().out
    # without machine.json: this computer's default budget and as many workers as it holds
    # by the estimate of a worker (WORKER_GB), up to the processor's 22
    workers, ram = {"older": (13, "16"), "old": (13, "16"), "new": (20, "48"),
                    "none": (22, "48")}[which]
    assert [_flag(c, "--workers") for c in commands] == [str(workers)] * 2
    assert all(_flag(c, "--max-ram-gb") == ram for c in commands)
    assert f"{workers} data workers, 4 spectra per step, RAM budget {ram} GB" in text
    assert ("setup now takes half the computer's memory by default, 48 GB here" in text) == (
        which in ("older", "old"))
    if which == "none":
        assert "h of training from the start" not in text
        return
    estimate = text[text.index("[train_all] about "):].splitlines()[0]
    if which == "older":  # no speedup measurement: the GPU at the active-bin cap, 26 spectra/s
        assert "window 1.4 h at 26 spectra/s, set by the GPU at the active-bin cap" in estimate
        assert "no GPU speedups: gpu_training/machine.json does not say" in text
    else:  # the first training batch: 42 spectra/s with tf32 against the workers' 30 or 27
        hours, pace = ("1.2", "30") if which == "old" else ("1.4", "27")
        assert f"window {hours} h at {pace} spectra/s, set by the data workers" in estimate
        assert f"comb {hours} h at {pace} spectra/s, set by the data workers" in estimate
        # at the active-bin cap (26 and 25 spectra/s) the GPU would be slower than 30, not 27
        assert ("up to 2.9 h should every step take as long as with every spectrum at the "
                "active-bin cap" in estimate) == (which == "old")


def test_more_workers_than_setup_measured_do_not_name_the_gpu_as_the_pace(tmp_path, monkeypatch,
                                                                         capsys):
    """The user's case: setup measured 13 workers (30.4 spectra/s) within 16 GB, training ran
    with --ram-gb 48 and so 20 workers (22 now), which delivered 26-27 spectra/s. Scaled up to 22
    they would be faster than the GPU (42 and 38 spectra/s on the first training batch), so the
    GPU is not named as what sets the pace, and the command that measures 22 is."""
    _computer(monkeypatch, 96, 24)
    train_all = _train_all_in(tmp_path, monkeypatch)
    train_all.MACHINE.write_text(json.dumps(OLD_MACHINE))
    for name in train_all.DEFAULT_RUNS:
        (tmp_path / train_all.RUN_TABLE[name]["out"]).write_text("checkpoint")
    commands = []
    monkeypatch.setattr(train_all.subprocess, "Popen", _FakeTraining(commands, [([], 0)]))
    monkeypatch.setattr(train_all.subprocess, "run",
                        lambda cmd, **_kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(train_all.signal, "signal", lambda *_a: None)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu", "--ram-gb", "48"])
    assert train_all.main() == 0
    assert [_flag(c, "--workers") for c in commands] == ["22"] * 2
    estimate = capsys.readouterr().out.split("[train_all] about ")[1].splitlines()[0]
    assert "window 0.9 h at 42 spectra/s, set by the GPU or by the data workers" in estimate
    assert "comb 1.0 h at 38 spectra/s, set by the GPU or by the data workers" in estimate
    assert "set by the GPU on setup's first training batch" not in estimate
    assert ("setup's for 13, scaled to 22: an upper bound (on a processor with efficiency cores "
            "each further worker adds less), so the time may be longer and the data workers may "
            "set the pace; to measure 22: ") in estimate
    assert "probe.py --ram-gb 48 --gpu-memory-gb 14" in estimate


@pytest.mark.parametrize("machine", [OLDER_MACHINE, OLD_MACHINE, NEW_MACHINE],
                         ids=["older", "old", "new"])
def test_a_ram_pause_resumes_with_two_workers_fewer(tmp_path, monkeypatch, capsys, machine):
    train_all = _train_all_in(tmp_path, monkeypatch)
    train_all.MACHINE.write_text(json.dumps(machine))
    commands = []
    budget = ["memory budget of 16 GB reached (16.2 GB in use): state saved\n"]
    monkeypatch.setattr(train_all.subprocess, "Popen", _FakeTraining(
        commands, [(budget, train_all.MEMORY), (["done\n"], 0)]))
    stage = train_all.make_run("window")
    pool = {"workers": machine["workers"]}
    assert train_all.run_stage(stage, pool, ["--max-ram-gb", "16"], "cuda", False) == 0
    n = machine["workers"]
    assert [_flag(c, "--workers") for c in commands] == [str(n), str(n - 2)]
    assert [c[-1] for c in commands] == ["--restart", "--resume"]
    after = json.loads(train_all.MACHINE.read_text())
    assert after == {**machine, "workers": n - 2}  # remembered, nothing else changed
    assert f"the RAM budget was reached; resuming with {n - 2} workers" in capsys.readouterr().out


def test_the_estimate_names_what_sets_the_pace_and_its_bounds():
    train_all = _load("train_all")
    assert train_all.speedup_name(["index_select", "tf32"]) == "tf32+index_select"
    assert train_all.speedup_name([]) == "none"
    assert _load("probe").config_name(("index_select", "tf32")) == "tf32+index_select"
    machine = {**OLD_MACHINE, "v5_spectra_per_second": 30.0, "v5_workers": 10,
               "gpu_views": {"window": {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.25,
                                        "unchunked": {"batch": 4, "peak_gb": 5.0,
                                                      "step_seconds": 0.125}}},
               "speedup_timings": {"window": {"tf32": 100.0}}}
    run = train_all.make_run("window")
    run["gpu_speedups"] = ["tf32"]
    # the first training batch: 40 spectra/s, the workers' 30 set the pace; at the active-bin
    # cap the GPU would take 16 a second
    text = train_all.time_estimate([run], machine, 10)
    assert "window 1.2 h at 30 spectra/s, set by the data workers" in text
    assert "up to 2.3 h should every step take as long as with every spectrum at the " in text
    # another number of workers than setup measured: scaled, and said so
    text = train_all.time_estimate([run], machine, 8)
    assert "window 1.5 h at 24 spectra/s" in text
    assert "the data workers' rate is setup's for 10, scaled to 8" in text
    assert "upper bound" not in text  # fewer: each is at least as fast as measured
    # more than measured (the user's 13 measured, 20 trained): scaled up, the rate is an upper
    # bound, so a GPU slower than it may or may not set the pace, and the measurement is named
    text = train_all.time_estimate([run], machine, 20, "probe.py --ram-gb 48")
    assert "window 0.9 h at 40 spectra/s, set by the GPU or by the data workers" in text
    assert ("setup's for 10, scaled to 20: an upper bound (on a processor with efficiency cores "
            "each further worker adds less), so the time may be longer and the data workers may "
            "set the pace; to measure 20: probe.py --ram-gb 48") in text
    assert train_all.training_hours([run], machine, 20)["runs"][0]["set_by"] == "either"
    # scaled up and still slower than the GPU: the workers set the pace all the same
    text = train_all.time_estimate([run], machine, 13)
    assert "window 0.9 h at 39 spectra/s, set by the data workers" in text
    assert "scaled to 13: an upper bound" in text and "so the time may be longer" in text
    assert "may set the pace" not in text and "to measure" not in text
    # a chunking other than the one the speedups were measured with: the active-bin cap
    run["head_chunk"] = None
    rates = train_all.gpu_rates(machine, run)
    assert rates["batch"] is None and rates["cap"] == 32.0
    assert "window 1.2 h at 30 spectra/s, set by the data workers" in train_all.time_estimate(
        [run], machine, 10)
    machine["speedup_timings"]["window"]["tf32"] = 160.0  # 25 spectra/s on the first batch
    run["head_chunk"] = 16
    assert "set by the GPU on setup's first training batch" in train_all.time_estimate(
        [run], machine, 10)
