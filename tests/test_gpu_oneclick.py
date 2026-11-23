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
