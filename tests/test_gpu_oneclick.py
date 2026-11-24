"""The one-click GPU training script: setup is skipped while it holds, then training starts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
        "window": ("window", "runs/ladder_v3_window.pt"),
        "comb": ("comb", "runs/ladder_v3_comb.pt"),
        "window-plain": ("window", "runs/ladder_v3_window_plain.pt"),
        "comb-plain": ("comb", "runs/ladder_v3_comb_plain.pt")}
    for name, run in runs.items():
        a = run["args"]
        assert a[a.index("--recipe") + 1] == "v3" and a[a.index("--view") + 1] == run["view"]
        assert ("--iso-channels" in a) == (not name.endswith("plain"))
        for flag, value in (("--model-size", "full"), ("--singly-charged", "keep"),
                            ("--epochs", "20"), ("--train-len", "6000"), ("--val-len", "128"),
                            ("--lr", "2e-3"), ("--seed", "0")):
            assert a[a.index(flag) + 1] == value, flag
        assert run["spectra"] == 120000
    smoke = train_all.make_run("comb", smoke=True)
    assert smoke["out"] == "runs/smoke_v3_comb.pt"
    a = smoke["args"]
    assert a[a.index("--epochs", a.index("--epochs") + 1) + 1] == "1"  # the smoke's own last
    assert a[-2:] == ["--batch-size", "2"]


def test_time_estimate_takes_the_slower_of_workers_and_gpu():
    train_all = _load("train_all")
    machine = {**MACHINE, "v3_spectra_per_second": 30.0, "v3_workers": 10}
    runs = [train_all.make_run("window"), train_all.make_run("comb")]
    text = train_all.time_estimate(runs, machine, 10, 4)
    # window: the workers' 30 spectra/s (the GPU could take 20 at 0.2 s per 4); comb: 16
    assert "window 1.8 h at 20 spectra/s" in text and "comb 2.3 h at 16 spectra/s" in text
    assert "about 4.1 h" in text
    assert train_all.time_estimate(runs, {}, 10, 4) is None


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
    assert "runs/ladder_v3_window.onnx" in names and "runs/ladder_v3_comb_plain.json" in names
    assert "runs/ladder_v3_comb.status.json" in names
    assert not any(n.endswith(".resume.pt") for n in names)
