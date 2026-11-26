"""onnxruntime's telemetry stays off when lcmsdeconv runs a network.

onnxruntime's Linux build writes a device ID and an event database under
~/.cache/Microsoft/DeveloperTools when it is imported, unless ORT_DISABLE_TELEMETRY is set
first; lcmsdeconv sets it when it is imported, unless the user has. The Windows build does not
read the variable; lcmsdeconv.nn.infer.import_onnxruntime calls
onnxruntime.disable_telemetry_events() for it. Each case with a real onnxruntime runs in a fresh
interpreter with an empty home directory.
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import lcmsdeconv
from lcmsdeconv.nn.infer import import_onnxruntime

MODEL = Path(lcmsdeconv.__file__).resolve().parent / "models" / "charge_unet_proof.onnx"


def _run(code: str, home: Path, **env) -> str:
    """The output of ``code`` in a fresh interpreter whose home directory is ``home``."""
    base = {k: v for k, v in os.environ.items()
            if k not in ("ORT_DISABLE_TELEMETRY", "XDG_CACHE_HOME")}
    home.mkdir()
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
        env=base | {"HOME": str(home), "OMP_NUM_THREADS": "1",
                    "PYTHONPATH": str(Path(lcmsdeconv.__file__).resolve().parents[1])} | env)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


def _files(home: Path) -> list[str]:
    return sorted(str(p.relative_to(home)) for p in home.rglob("*") if p.is_file())


@pytest.mark.skipif(not MODEL.exists(), reason="the bundled proof model is not installed")
def test_inference_leaves_no_telemetry_files(tmp_path):
    _run("import onnxruntime", tmp_path / "control")
    if not _files(tmp_path / "control"):
        pytest.skip("this onnxruntime build writes no telemetry files, so nothing tells them apart")
    out = _run(
        "import os\n"
        "import numpy as np\n"
        "from lcmsdeconv.nn.grid import LogMzGrid\n"
        "from lcmsdeconv.nn.infer import ChargePredictor\n"
        "grid = LogMzGrid(500.0, 600.0)\n"
        f"p = ChargePredictor({str(MODEL)!r})\n"
        "p.predict_grid(grid, np.random.default_rng(0).random(grid.size), 1.0)\n"
        "print(os.environ['ORT_DISABLE_TELEMETRY'])\n", tmp_path / "home")
    assert out.split()[-1] == "1"
    assert _files(tmp_path / "home") == []


def test_a_value_the_user_set_wins(tmp_path):
    out = _run("import os\n"
               "from lcmsdeconv.nn.infer import import_onnxruntime\n"
               "import_onnxruntime()\n"
               "print(os.environ['ORT_DISABLE_TELEMETRY'])\n", tmp_path / "home",
               ORT_DISABLE_TELEMETRY="0")
    assert out.split()[-1] == "0"


def test_importing_lcmsdeconv_first_is_enough(tmp_path):
    """A program that imports lcmsdeconv, then onnxruntime itself, runs with it off too."""
    _run("import onnxruntime", tmp_path / "control")
    if not _files(tmp_path / "control"):
        pytest.skip("this onnxruntime build writes no telemetry files, so nothing tells them apart")
    out = _run("import os\n"
               "import lcmsdeconv.io.mzml\n"
               "print(os.environ['ORT_DISABLE_TELEMETRY'])\n"
               "import onnxruntime\n", tmp_path / "home")
    assert out.split()[-1] == "1"
    assert _files(tmp_path / "home") == []


def test_the_windows_switch_is_thrown_unless_the_user_keeps_telemetry_on(monkeypatch):
    """import_onnxruntime calls disable_telemetry_events() (what the Windows build reads) when
    the variable says off, as it does unless the user set it otherwise (a stub onnxruntime)."""
    calls = []
    stub = types.ModuleType("onnxruntime")
    stub.disable_telemetry_events = lambda: calls.append(os.environ["ORT_DISABLE_TELEMETRY"])
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    assert import_onnxruntime() is stub and calls == ["1"]
    for value, called in (("0", False), ("false", False), ("true", True), ("YES", True)):
        monkeypatch.setenv("ORT_DISABLE_TELEMETRY", value)
        calls.clear()
        import_onnxruntime()
        assert calls == ([value] if called else []), value
