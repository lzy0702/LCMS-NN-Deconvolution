"""Training with data-worker processes as on Windows: persistent spawned workers, resource limits,
and the training script's exit codes."""

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _synth():
    from lcmsdeconv.synth.compounds import ClassConfig
    from lcmsdeconv.synth.config import SynthConfig

    return SynthConfig(classes=[ClassConfig("peptide", (8000.0, 20000.0))], mode="rplc",
                       polarity=1)


def _cfg(out, **kw):
    base = dict(arch="ladder", model_size="small", epochs=2, train_len=4, val_len=2,
                batch_size=2, num_workers=0, threads=1, out=str(out), log_every=100,
                fresh_each_epoch=True, device="cpu")
    base.update(kw)
    return base


def test_epoch_sampler_reads_new_items_every_epoch_and_resumes_inside_one():
    from lcmsdeconv.synth.dataset import EpochSampler

    s = EpochSampler(6)
    assert list(s) == [0, 1, 2, 3, 4, 5]
    s.epoch, s.start = 2, 4
    assert list(s) == [16, 17] and len(s) == 2


def test_spawned_persistent_workers_train_exactly_as_without_workers(tmp_path, monkeypatch):
    """Workers started the Windows way, kept for the whole run, see the same spectra in the
    same order as a run without workers, epoch after epoch, and through a pause."""
    from lcmsdeconv.nn.train import TrainConfig, train

    torch.manual_seed(0)
    plain, _h = train(TrainConfig(**_cfg(tmp_path / "plain.pt")), synth_config=_synth())
    monkeypatch.setenv("LCMSDECONV_MP_CONTEXT", "spawn")
    out = tmp_path / "workers.pt"
    torch.manual_seed(0)
    train(TrainConfig(max_seconds=1e-6, **_cfg(out, num_workers=2)), synth_config=_synth())
    state = torch.load(out.with_suffix(".resume.pt"), weights_only=False)
    assert (state["epoch"], state["step"], state["stop_reason"]) == (0, 1, "time")
    workers, history = train(TrainConfig(resume=True, **_cfg(out, num_workers=2)),
                             synth_config=_synth())
    assert [h["epoch"] for h in history] == [0, 1]
    for (k, a), b in zip(plain.state_dict().items(), workers.state_dict().values()):
        assert torch.allclose(a, b), k
    assert torch.load(out.with_suffix(".resume.pt"), weights_only=False)["stop_reason"] == "finished"


def test_memory_budget_pauses_the_run_for_a_resume_with_fewer_workers(tmp_path):
    pytest.importorskip("psutil")
    from lcmsdeconv.nn.train import TrainConfig, train

    out = tmp_path / "net.pt"
    train(TrainConfig(max_ram_gb=0.01, **_cfg(out)), synth_config=_synth())
    state = torch.load(out.with_suffix(".resume.pt"), weights_only=False)
    assert (state["epoch"], state["step"], state["stop_reason"]) == (0, 1, "memory")


def test_gpu_only_run_refuses_to_fall_back_to_the_processor(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    from lcmsdeconv.nn.train import TrainConfig, train

    with pytest.raises(SystemExit, match="GPU only"):
        train(TrainConfig(**_cfg(tmp_path / "net.pt", device="cuda")), synth_config=_synth())


def test_training_script_exit_codes(tmp_path):
    """0 when the run finished, 3 when it paused (here at its time budget), and the paused run
    completes when resumed."""
    out = tmp_path / "net.pt"
    args = [sys.executable, str(ROOT / "scripts" / "train_charge_net.py"), "--device", "cpu",
            "--model-size", "small", "--epochs", "1", "--train-len", "2", "--val-len", "1",
            "--batch-size", "1", "--workers", "0", "--threads", "1", "--out", str(out)]
    paused = subprocess.run(args + ["--minutes", "0.000001"], capture_output=True, text=True,
                            timeout=900)
    assert paused.returncode == 3, paused.stdout + paused.stderr
    done = subprocess.run(args + ["--resume"], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr
    assert out.exists()
