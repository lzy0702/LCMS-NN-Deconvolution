"""gpu_training/setup.py reads the CUDA version the NVIDIA driver supports, however it is written."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SETUP = Path(__file__).resolve().parents[1] / "gpu_training" / "setup.py"


@pytest.fixture
def setup_mod():
    spec = importlib.util.spec_from_file_location("_gpu_setup_driver", SETUP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text, cuda", [
    ("| NVIDIA-SMI 581.57   Driver Version: 581.57   CUDA Version: 13.0     |", (13, 0)),
    ("    CUDA Version                              : 13.5", (13, 5)),  # nvidia-smi -q
    ("| NVIDIA-SMI 621.24   Driver Version: 621.24   CUDA Driver Version: 13.5 |", (13, 5)),
    ("CUDA UMD Version: 13.5", (13, 5)),
    ("| NVIDIA-SMI 621.24   Driver Version: 621.24 |", None),
    ("", None),
])
def test_cuda_version_read_in_every_known_wording(setup_mod, text, cuda):
    assert setup_mod.cuda_from_text(text) == cuda


def test_builds_tried_newest_first_and_all_of_them_when_unknown(setup_mod):
    every = [b for b, _need in setup_mod.TORCH_BUILDS]
    assert setup_mod.builds_for(None) == every
    assert setup_mod.builds_for((13, 5)) == every
    assert setup_mod.builds_for((13, 0)) == ["cu130", "cu128", "cu126"]
    assert setup_mod.builds_for((12, 4)) == []


def _fake_smi(header: str, detail: str):
    def run(cmd, **kwargs):
        if cmd[1:] == []:
            out = header
        elif cmd[1:] == ["-q"]:
            out = detail
        else:
            out = "NVIDIA GeForce RTX 4080 SUPER, 16376, 621.24\n"
        return SimpleNamespace(stdout=out, returncode=0)
    return run


def test_the_driver_itself_is_asked_first(setup_mod, monkeypatch):
    monkeypatch.setattr(setup_mod.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(setup_mod, "cuda_from_driver_api", lambda: (13, 5))
    monkeypatch.setattr(subprocess, "run", _fake_smi("no version here", "nor here"))
    info = setup_mod.driver()
    assert info["cuda"] == (13, 5)
    assert info["driver"] == "621.24" and info["memory_mb"] == 16376


def test_falls_back_to_the_header_then_to_the_detailed_query(setup_mod, monkeypatch):
    monkeypatch.setattr(setup_mod.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(setup_mod, "cuda_from_driver_api", lambda: None)
    monkeypatch.setattr(subprocess, "run", _fake_smi("| CUDA Version: 13.2 |", ""))
    assert setup_mod.driver()["cuda"] == (13, 2)
    monkeypatch.setattr(subprocess, "run", _fake_smi("| no version |", "CUDA Version   : 13.5"))
    assert setup_mod.driver()["cuda"] == (13, 5)
    monkeypatch.setattr(subprocess, "run", _fake_smi("| no version |", "nothing"))
    info = setup_mod.driver()
    assert "cuda" not in info and info["gpu"] == "NVIDIA GeForce RTX 4080 SUPER"


def test_driver_api_absent_is_not_an_error(setup_mod):
    # this test machine has no NVIDIA driver: the query must answer None, not raise
    result = setup_mod.cuda_from_driver_api()
    assert result is None or isinstance(result, tuple)
