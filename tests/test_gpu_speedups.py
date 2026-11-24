"""The GPU training speedups and what training measures about its pace.

Always on (the same values up to float rounding): pinned batches copied without waiting, the
running loss kept on the device, the focal loss without a host read, a moving average in two
launches that copies only buffers that can change. Measured by setup and used only where faster
with the same loss and gradients (``--gpu-speedups``): cuDNN's benchmark, TF32 matrix products,
LadderNet's index_select head and fused AdamW. Every check here runs on the processor; the
CUDA-only parts are exercised with fakes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "gpu_training"
#: a short window keeps the networks small; the comb view keeps every offset whatever it is
W, TOP = 16384, 3072


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_speedups_{name}", GPU / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ladder(view: str = "window", chunk: int | None = None, z_ladder: int = 12):
    """A LadderNet with isotope channels (the current three), few charges and a learned part
    that is not zero, so that gradients reach every parameter through the head."""
    from lcmsdeconv.nn.model import LadderNet

    in_ch = 7 + (1 if view == "comb" else 0)
    torch.manual_seed(0)
    model = LadderNet(in_ch, 100, z_ladder=z_ladder, width=8, proj=4, hidden=12, window=W,
                      active_top=TOP, iso=True, view=view, head_chunk=chunk)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model


def _features(in_ch: int, length: int = W, measured: int | None = None, seed: int = 0,
              batch: int = 2):
    """Random peaks, ladders, isotope charges, strengths and limits; with a measured channel
    (last), the bins from ``measured`` on are padding."""
    rng = np.random.default_rng(seed)
    m = length if measured is None else measured
    x = np.zeros((batch, in_ch, length), dtype=np.float32)
    for b in range(batch):
        idx = rng.integers(0, m, 1500)
        x[b, 0, idx] = rng.uniform(0.0, 5.0, idx.size)
        for c in rng.integers(0, m, 30):
            for d in (0, 693, 1386, 2079):
                if 0 <= c - d < m:
                    x[b, 0, c - d] = 4.0
        x[b, 1, :m] = rng.uniform(0.0, 1.0, m)
        z = rng.integers(0, 14, m).astype(np.float32)
        z[rng.random(m) < 0.5] = 0.0
        x[b, 4, :m] = z
        x[b, 5, :m] = np.where(z > 0, rng.uniform(0.0, 1.0, m), 0.0)
        x[b, 6, :m] = rng.integers(0, 20, m)
        if in_ch == 8:
            x[b, 7, :m] = 1.0
    return torch.from_numpy(x)


def _step(model, x, weight):
    model.zero_grad()
    out = model(x)
    loss = (out["charge_logits"] * weight).sum() + out["apex_logit"].square().sum()
    loss.backward()
    return out, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


# ------------------------------------------------------------------ the index_select head
@pytest.mark.parametrize("view", ["window", "comb"])
@pytest.mark.parametrize("chunk", [None, 5])
def test_index_select_head_gives_the_same_logits_and_gradients(view, chunk, monkeypatch):
    ref = _ladder(view, chunk)
    alt = copy.deepcopy(ref)
    alt.head_gather = "index_select"
    in_ch = ref.in_ch
    x = _features(in_ch, measured=W - 900 if view == "comb" else None, seed=1)
    weight = torch.randn(2, 101, W, generator=torch.Generator().manual_seed(2))
    calls = {"n": 0}
    original = torch.index_select

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "index_select", counting)
    out_a, grad_a = _step(ref.train(), x, weight)
    assert calls["n"] == 0
    out_b, grad_b = _step(alt.train(), x, weight)
    assert calls["n"] >= (1 if chunk is None else 3)  # every chunk, and again when recomputed
    for k in ("charge_logits", "apex_logit"):
        assert torch.equal(out_a[k], out_b[k]), k  # a gather copies: the same values exactly
    # the backward pass adds the same terms in another order: the gradients agree to rounding.
    # (A bias before a one-channel group norm has a true gradient of zero, which is rounding
    # alone, so each parameter is compared on the scale of the whole gradient too.)
    assert grad_a.keys() == grad_b.keys()
    flat_a = torch.cat([g.flatten() for g in grad_a.values()]).double()
    flat_b = torch.cat([grad_b[n].flatten() for n in grad_a]).double()
    assert float((flat_b - flat_a).norm()) <= 1e-6 * float(flat_a.norm())
    scale = float(flat_a.abs().max())
    for name, g in grad_a.items():
        err = float((grad_b[name] - g).abs().max())
        assert err <= 1e-5 * float(g.abs().max()) + 1e-8 * scale, (name, err)
    assert float(grad_a["proj.weight"].abs().sum()) > 0.0  # the gather's backward pass counts


def test_the_gather_is_no_part_of_the_saved_model_or_its_onnx_export(tmp_path):
    onnx = pytest.importorskip("onnx")
    ref = _ladder("window")
    alt = copy.deepcopy(ref)
    alt.head_gather = "index_select"
    a, b = ref.state_dict(), alt.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert not any("gather" in k for k in a)
    graphs = []
    x = _features(7, batch=1, seed=3)
    for name, model in (("index", ref), ("index_select", alt)):
        model.eval()
        model.dense = True
        path = tmp_path / f"{name}.onnx"
        torch.onnx.export(model, x, str(path), input_names=["features"],
                          output_names=["charge_logits", "apex_logit"], opset_version=17,
                          dynamo=False)
        graphs.append(onnx.load(str(path)).graph)
    assert graphs[0] == graphs[1]
    # and the dense function (the export) is the index_select network's in training
    alt.dense = False
    with torch.no_grad():
        sparse = alt(x)
        alt.dense = True
        dense = alt(x)
    for k in ("charge_logits", "apex_logit"):
        assert torch.allclose(sparse[k], dense[k], atol=1e-4), k


# ------------------------------------------------------------------ the focal loss
def _focal_reference(apex_logit, heat, alpha: float = 2.0, beta: float = 4.0):
    """centernet_focal as it was, reading the number of positives on the host."""
    p = torch.sigmoid(apex_logit).squeeze(1).clamp(1e-4, 1 - 1e-4)
    pos = (heat >= 1.0 - 1e-6).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - heat) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    n_pos = pos.sum()
    if float(n_pos) < 1.0:
        return neg_loss.sum() / max(heat.numel(), 1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


@pytest.mark.parametrize("positives", [0, 1, 9])
def test_focal_loss_and_its_gradient_are_the_reference_in_both_branches(positives):
    from lcmsdeconv.nn.losses import centernet_focal

    g = torch.Generator().manual_seed(positives)
    heat = torch.rand(3, 400, generator=g) * 0.95
    heat.view(-1)[torch.randperm(1200, generator=g)[:positives]] = 1.0
    logit = 3.0 * torch.randn(3, 1, 400, generator=g)
    a, b = logit.clone().requires_grad_(), logit.clone().requires_grad_()
    got, want = centernet_focal(a, heat), _focal_reference(b, heat)
    assert torch.equal(got, want)
    got.backward()
    want.backward()
    assert torch.equal(a.grad, b.grad)


# ------------------------------------------------------------------ the moving average
def _ema_reference(shadow, model, decay: float) -> None:
    """EMA.update as it was: one launch per parameter, every buffer copied."""
    with torch.no_grad():
        for s, m in zip(shadow.parameters(), model.parameters()):
            s.mul_(decay).add_(m, alpha=1 - decay)
        for s, m in zip(shadow.buffers(), model.buffers()):
            s.copy_(m)


def _norm_net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv1d(2, 4, 3, padding=1), torch.nn.BatchNorm1d(4),
                               torch.nn.SiLU(), torch.nn.Conv1d(4, 1, 1))


@pytest.mark.parametrize("kind, foreach", [("ladder", True), ("batchnorm", True),
                                           ("batchnorm", False)])
def test_moving_average_equals_the_reference_loop(kind, foreach, monkeypatch):
    from lcmsdeconv.nn.train import EMA

    if not foreach:
        monkeypatch.delattr(torch, "_foreach_mul_")  # the loop it falls back to
    model = _ladder("comb", 5) if kind == "ladder" else _norm_net()
    ema = EMA(model, 0.9)
    ref = copy.deepcopy(model).eval()
    copied = [name for name, _b in ema._buffers]
    if kind == "ladder":
        # ladder offsets, comb counts, charge numbers and tooth masks: constants, never copied
        assert {n for n, _b in model.named_buffers()} == {"off_t", "comb_t", "comb_den",
                                                          "z_ids", "present_t"}
        assert copied == []
    else:
        assert copied == ["1.running_mean", "1.running_var", "1.num_batches_tracked"]
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(4):
        if kind == "ladder":
            x = _features(8, measured=W - 500, seed=step, batch=1)
            out = model.train()(x)
            loss = out["charge_logits"].square().mean() + out["apex_logit"].mean()
        else:
            x = torch.randn(3, 2, 40, generator=torch.Generator().manual_seed(step))
            loss = model.train()(x).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        _ema_reference(ref, model, 0.9)
    for (n, s), (_n, r) in zip(ema.shadow.named_parameters(), ref.named_parameters()):
        assert torch.equal(s, r), n
    for (n, s), (_n, r) in zip(ema.shadow.named_buffers(), ref.named_buffers()):
        assert torch.equal(s, r), n  # running statistics follow the model, constants stay
    if kind == "batchnorm":
        assert int(ema.shadow[1].num_batches_tracked) == 4


# ------------------------------------------------------------------ choosing the speedups
def _fake_measure(ms: dict, bad: dict | None = None, raises: tuple = ()):
    """measure(config) for choose_speedups: the step's median time from ``ms`` (by configuration
    name), the reference loss and gradients, changed for configurations in ``bad``, and an
    exception for those in ``raises``."""
    probe = _load("probe")
    rng = np.random.default_rng(0)
    grads = [rng.normal(size=200), rng.normal(size=200)]
    seen = []

    def measure(config):
        name = probe.config_name(config)
        seen.append(name)
        if name in raises:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        loss, grad = [1.25, 1.2], [g.copy() for g in grads]
        change = (bad or {}).get(name)
        if change == "loss":
            loss[1] *= 1.01
        elif change == "direction":
            grad[1] = grad[1] + 0.3 * rng.normal(size=200)
        elif change == "norm":
            grad[0] = grad[0] * 1.05
        elif change == "tf32-like":  # rounding far below the tolerances
            loss = [v * (1 + 2e-5) for v in loss]
            grad = [g * (1 + 1e-4) + 1e-4 * rng.normal(size=200) for g in grad]
        return {"ms": ms[name], "loss": loss, "grad": grad}

    return probe, measure, seen


def test_a_speedup_needs_three_percent_and_the_same_loss_and_gradients():
    ms = {"none": 100.0, "cudnn_benchmark": 98.0, "tf32": 90.0, "index_select": 80.0,
          "fused_adamw": 85.0}
    probe, measure, seen = _fake_measure(ms, bad={"tf32": "tf32-like", "index_select": "direction",
                                                  "fused_adamw": "loss"})
    res = probe.choose_speedups(measure)
    assert res["chosen"] == ["tf32"]
    notes = res["notes"]
    assert "under 3 %" in notes["cudnn_benchmark"]  # 2 % faster: not enough
    assert "not the same: gradient cosine" in notes["index_select"]
    assert "not the same: loss" in notes["fused_adamw"]
    assert "kept" in notes["tf32"]
    assert seen == ["none", "cudnn_benchmark", "tf32", "index_select", "fused_adamw"]
    assert res["timings"] == ms
    # a norm off by 5 % is no rounding either
    probe, measure, _seen = _fake_measure(ms, bad={"index_select": "norm"})
    assert "gradient norm" in probe.choose_speedups(measure)["notes"]["index_select"]


def test_a_speedup_that_fails_is_left_out_and_recorded():
    ms = {"none": 100.0, "cudnn_benchmark": 90.0, "tf32": 95.0, "index_select": 80.0,
          "fused_adamw": 99.0, "tf32+index_select": 76.0}
    probe, measure, _seen = _fake_measure(ms, raises=("cudnn_benchmark",))
    res = probe.choose_speedups(measure)
    assert "cudnn_benchmark" not in res["chosen"]
    assert res["notes"]["cudnn_benchmark"].startswith("failed: RuntimeError: CUDA error")
    assert "cudnn_benchmark" not in res["timings"]
    assert res["chosen"] == ["tf32", "index_select"]
    # nothing can be compared without the reference
    probe, measure, seen = _fake_measure(ms, raises=("none",))
    res = probe.choose_speedups(measure)
    assert res["chosen"] == [] and seen == ["none"] and "failed" in res["notes"]["none"]


def test_speedups_combine_greedily_fastest_first():
    ms = {"none": 100.0, "cudnn_benchmark": 96.0, "tf32": 90.0, "index_select": 80.0,
          "fused_adamw": 92.0,
          "tf32+index_select": 75.0,                     # 6 % faster than index_select: kept
          "tf32+index_select+fused_adamw": 74.0,         # 1.3 % faster again: not enough
          "cudnn_benchmark+tf32+index_select": 70.0}     # 6.7 % faster, but not the same
    probe, measure, seen = _fake_measure(ms, bad={"cudnn_benchmark+tf32+index_select": "loss"})
    res = probe.choose_speedups(measure)
    assert res["chosen"] == ["tf32", "index_select"]
    # the singles, then each combination in the order of the singles' speed
    assert seen == ["none", "cudnn_benchmark", "tf32", "index_select", "fused_adamw",
                    "tf32+index_select", "tf32+index_select+fused_adamw",
                    "cudnn_benchmark+tf32+index_select"]
    notes = res["notes"]
    assert "kept (the fastest alone)" in notes["index_select"]
    assert notes["tf32+index_select"].endswith(": kept")
    assert "faster than tf32+index_select: under 2 %" in notes["tf32+index_select+fused_adamw"]
    assert "not the same" in notes["cudnn_benchmark+tf32+index_select"]
    lines = "\n".join(probe.speedup_lines({"gpu_speedups": {"window": res["chosen"]},
                                           "speedup_timings": {"window": res["timings"]},
                                           "speedup_checks": {"window": notes}},
                                          {"window": None}))
    assert "window view, all charges at once:" in lines
    assert "training uses: tf32, index_select" in lines and "index_select: 80 ms a step" in lines


def test_the_kernel_table_counts_gpu_kernels_only():
    from torch.autograd import DeviceType

    probe = _load("probe")

    def ev(key, us, count, device=DeviceType.CUDA):
        return SimpleNamespace(key=key, self_device_time_total=us, count=count,
                               device_type=device)

    events = [ev("aten::mm", 9000.0, 3, DeviceType.CPU),  # a host op: its kernels' time again
              ev("indexing_backward_kernel", 6000.0, 3), ev("sgemm_128x64", 3000.0, 30),
              ev("Memcpy HtoD (Pinned -> Device)", 1000.0, 12), ev("idle", 0.0, 1)]
    rows = probe.kernel_rows(events, top=2)
    assert [r["name"] for r in rows] == ["indexing_backward_kernel", "sgemm_128x64"]
    assert rows[0] == {"name": "indexing_backward_kernel", "total_ms": 6.0, "calls": 3,
                       "share": 0.6}


# ------------------------------------------------------------------ the probe's plumbing
def test_each_configuration_is_measured_in_a_process_of_its_own(tmp_path, monkeypatch):
    """The parent reads what the child left: a measurement, an error, or nothing at all."""
    probe = _load("probe")
    commands = []

    def child(cmd, **_kw):
        commands.append(cmd)
        name = cmd[cmd.index("--config") + 1]
        if "--profile" in cmd:
            (tmp_path / f"profile_{name}__window.json").write_text(json.dumps(
                {"view": "window", "speedups": ["tf32"], "steps": 3, "kernels": []}))
        else:
            np.savez(tmp_path / f"{name}__window.npz", ms=12.5, loss=np.array([1.0, 0.9]),
                     grad0=np.ones(5), grad1=np.zeros(5), peak_gb=2.345)
            (tmp_path / f"{name}__comb.err").write_text("OutOfMemoryError: CUDA out of memory")
        return SimpleNamespace(returncode=0 if name != "tf32" else 3, stdout="", stderr="boom")

    monkeypatch.setattr(probe.subprocess, "run", child)
    out = probe._measure_in_child(tmp_path, ("tf32",), ["window", "comb", "other"])
    assert commands[0][1:3] == [str(GPU / "probe.py"), "--speedup-child"]
    assert commands[0][-2:] == ["--views", "window,comb,other"]
    assert out["window"]["ms"] == 12.5 and out["window"]["loss"] == [1.0, 0.9]
    assert out["window"]["peak_gb"] == 2.35 and list(out["window"]["grad"][0]) == [1.0] * 5
    assert isinstance(out["comb"], RuntimeError) and "out of memory" in str(out["comb"])
    assert isinstance(out["other"], RuntimeError) and "exit code 3" in str(out["other"])
    prof = probe._measure_in_child(tmp_path, ("tf32",), ["window", "comb"], profile=True)
    assert prof["window"]["kernels"] == [] and "exit code 3" in prof["comb"]["error"]


def test_measure_speedups_prefetches_the_singles_then_combines_each_view(tmp_path, monkeypatch):
    probe = _load("probe")
    ms = {"window": {"none": 100.0, "cudnn_benchmark": 99.0, "tf32": 90.0, "index_select": 80.0,
                     "fused_adamw": 99.0, "tf32+index_select": 70.0},
          "comb": {"none": 100.0, "cudnn_benchmark": 99.0, "tf32": 99.0, "index_select": 95.0,
                   "fused_adamw": 99.0}}
    calls, jobs = [], []
    rng = np.random.default_rng(0)
    grads = [rng.normal(size=50), rng.normal(size=50)]

    def fake_child(tmp, config, views, profile=False):
        jobs.append(json.loads((tmp / "job.json").read_text()))
        assert (tmp / "batch.pt").exists() and (tmp / "weights_window.pt").exists()
        calls.append((probe.config_name(config), tuple(views), profile))
        if profile:
            return {v: {"view": v, "speedups": list(config), "steps": 3, "kernels": []}
                    for v in views}
        return {v: {"ms": ms[v][probe.config_name(config)], "loss": [1.0, 0.9], "grad": grads,
                    "peak_gb": 1.5} for v in views}

    batch = {"features": torch.zeros(2, 8, 64), "topk_z": torch.zeros(2, 64, 3, dtype=torch.long),
             "topk_w": torch.zeros(2, 64, 3), "heat": torch.zeros(2, 64)}
    monkeypatch.setattr(probe, "first_batch", lambda b: batch)
    monkeypatch.setattr(probe, "_measure_in_child", fake_child)
    res = probe.measure_speedups(["window", "comb"], 2, {"window": None, "comb": 16}, 14.0)
    assert res["gpu_speedups"] == {"window": ["tf32", "index_select"], "comb": ["index_select"]}
    assert calls[:5] == [(n, ("window", "comb"), False) for n in
                         ("none", "cudnn_benchmark", "tf32", "index_select", "fused_adamw")]
    assert calls[5:] == [("tf32+index_select", ("window",), False),
                         ("tf32+index_select", ("window",), True)]
    assert jobs[0]["chunks"] == {"window": None, "comb": 16} and jobs[0]["lr"] == 2e-3
    assert res["speedup_timings"]["window"]["tf32+index_select"] == 70.0
    assert res["speedup_peak_gb"]["comb"]["none"] == 1.5
    assert res["cuda_kernels"]["speedups"] == ["tf32", "index_select"]


def test_a_failed_speedup_measurement_does_not_stop_the_probe(tmp_path, monkeypatch):
    import lcmsdeconv.nn.machine as nn_machine

    probe = _load("probe")
    views = {"window": {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.8},
             "comb": {"batch": 4, "peak_gb": 3.0, "step_seconds": 0.8}}
    monkeypatch.setattr(probe, "gpu_steps", lambda *_a: {"gpu": "Fake GPU", "gpu_total_gb": 16.0,
                                                         "gpu_views": views})
    monkeypatch.setattr(probe, "workers_run", lambda n, **_k: {
        "workers": n, "startup_seconds": 1.0, "spectra_per_second": 10.0 * n, "worker_gb": 0.5,
        "worker_commit_gb": 0.6, "workers_total_gb": 0.5 * n})

    def broken(*_a, **_k):
        raise OSError("no room for the temporary files")

    monkeypatch.setattr(probe, "measure_speedups", broken)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nn_machine, "single_threaded_workers", lambda: None)
    out = tmp_path / "machine.json"
    monkeypatch.setattr(sys, "argv", ["probe.py", "--workers", "4", "--out", str(out)])
    assert probe.main() == 0
    info = json.loads(out.read_text())
    assert info["gpu_speedups"] == {"window": [], "comb": []}
    assert "no room" in info["speedup_error"]
    assert info["recipe"] == "v4" and info["v4_spectra_per_second"] == 40.0
    assert "v3_spectra_per_second" not in info


def test_the_child_side_measures_each_configuration_on_the_processor(tmp_path, monkeypatch):
    """measure_speedups end to end with the children's own code (speedup_child, speedup_steps,
    the training step, the profile), each run in this process instead of a process of its own,
    on two real recipe-v4 frames at a short window with a small LadderNet and the CUDA calls
    stood in for: the result files the children write are the ones the parent reads, a
    configuration that fails, or whose process cannot even start, is ruled out alone, and the
    process-wide flags come back as they were."""
    import lcmsdeconv.nn.model as nn_model
    import lcmsdeconv.nn.train as nn_train
    from lcmsdeconv.synth.dataset import FrameDataset

    probe = _load("probe")
    import train_charge_net  # probe put scripts/ on the path

    for name in ("reset_peak_memory_stats", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(nn_train, "_cap_gpu_memory", lambda *a, **k: None)
    monkeypatch.setattr(train_charge_net, "V3_WINDOW", W)
    monkeypatch.setattr(probe, "WARM_STEPS", 0)
    monkeypatch.setattr(probe, "TIMED_STEPS", 1)
    monkeypatch.setattr(probe, "PROFILE_STEPS", 1)
    real_build = nn_model.build_model
    monkeypatch.setattr(nn_model, "build_model",
                        lambda size, **kw: real_build("small", **{**kw, "window": W}))

    def first_batch(batch_size):
        mix = train_charge_net.mixture_v4(W)
        for _w, c in mix:
            c.singly_charged, c.iso_channels, c.measured_channel = "keep", True, True
        ds = FrameDataset(mix, batch_size, base_seed=0)
        items = [ds[i] for i in range(batch_size)]
        names = ("features", "topk_z", "topk_w", "heat")
        return {k: torch.stack([it[j] for it in items]) for j, k in enumerate(names)}

    monkeypatch.setattr(probe, "first_batch", first_batch)
    real_steps = probe.speedup_steps

    def steps(view, batch, chunk, weights, config, lr):
        if "cudnn_benchmark" in config and view == "comb":
            raise RuntimeError("CUDNN_STATUS_NOT_SUPPORTED")  # this view only
        return real_steps(view, batch, chunk, weights, config, lr)

    monkeypatch.setattr(probe, "speedup_steps", steps)
    flags = (torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)
    started = []

    def child(cmd, **_kw):
        name = cmd[cmd.index("--config") + 1]
        started.append(name)
        if name == "fused_adamw":
            raise OSError("[WinError 1455] The paging file is too small")
        rc = probe.speedup_child(Path(cmd[cmd.index("--speedup-child") + 1]), name,
                                 cmd[cmd.index("--views") + 1].split(","), "--profile" in cmd)
        assert (torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32) == flags
        return SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr(probe.subprocess, "run", child)
    res = probe.measure_speedups(["window", "comb"], 2, {"window": None, "comb": 4}, 14.0)
    assert started[:5] == ["none", "cudnn_benchmark", "tf32", "index_select", "fused_adamw"]
    for view in ("window", "comb"):
        timings, notes = res["speedup_timings"][view], res["speedup_checks"][view]
        assert timings["none"] > 0 and notes["none"] == "the reference"
        # every configuration that ran agrees with the reference on the processor
        for name in ("tf32", "index_select") + (("cudnn_benchmark",) if view == "window" else ()):
            assert timings[name] > 0 and "not the same" not in notes[name], notes
        assert notes["fused_adamw"].startswith("failed") and "1455" in notes["fused_adamw"]
        assert set(res["gpu_speedups"][view]) <= {"cudnn_benchmark", "tf32", "index_select"}
    assert res["speedup_checks"]["comb"]["cudnn_benchmark"].startswith("failed")
    assert "CUDNN_STATUS_NOT_SUPPORTED" in res["speedup_checks"]["comb"]["cudnn_benchmark"]
    k = res["cuda_kernels"]
    assert k["view"] == "window" and ("kernels" in k or "error" in k)
    lines = probe.speedup_lines(res, {"window": None, "comb": 4})
    assert any("training uses:" in ln for ln in lines)


# ------------------------------------------------------------------ training with them
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


def test_speedup_names_are_checked():
    from lcmsdeconv.nn.train import GPU_SPEEDUPS, parse_gpu_speedups

    assert parse_gpu_speedups("none") == parse_gpu_speedups("") == parse_gpu_speedups(None) == ()
    assert parse_gpu_speedups(" index_select,tf32 ") == ("tf32", "index_select")
    assert parse_gpu_speedups(",".join(reversed(GPU_SPEEDUPS))) == GPU_SPEEDUPS
    with pytest.raises(ValueError, match="unknown GPU speedup amp: choose from cudnn_benchmark"):
        parse_gpu_speedups("tf32,amp")


def test_on_the_processor_speedups_are_ignored_and_the_pace_is_recorded(tmp_path, capsys):
    from lcmsdeconv.nn.train import TrainConfig, train

    out = tmp_path / "net.pt"
    model, history = train(TrainConfig(**_cfg(out, gpu_speedups=("tf32", "index_select"))),
                           synth_config=_synth())
    text = capsys.readouterr().out
    assert "GPU speedups ignored (tf32, index_select): this run trains on cpu" in text
    assert model.head_gather == "index"
    assert "of the training time" in text and "% waiting for data" in text
    for h in history:
        assert set(h["pace"]) == {"data_wait_share"} and 0.0 <= h["pace"]["data_wait_share"] <= 1
    records = [json.loads(line) for line in out.with_suffix(".log.jsonl").open()]
    epochs = [r for r in records if "epoch" in r]
    assert [set(r["pace"]) for r in epochs] == [{"data_wait_share"}] * 2
    st = json.loads(out.with_suffix(".status.json").read_text())
    assert st["gpu_speedups"] == [] and set(st["pace"]) == {"data_wait_share"}
    assert (st["epoch"], st["stop_reason"]) == (2, "finished")
    ckpt = torch.load(out, weights_only=False)
    assert ckpt["gpu_speedups"] == [] and ckpt["config"]["gpu_speedups"] == ("tf32", "index_select")


def test_a_run_resumes_with_other_speedups_and_the_epoch_loss_is_unchanged(tmp_path, monkeypatch):
    """The resume check leaves the speedups out; and the epoch's training loss is the mean of
    its steps' losses, summed in float64 as the host summed them, across a pause."""
    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.train import TrainConfig, _check_same_run, train

    steps = []
    forward = ChargeLoss.forward

    def recording(self, outputs, batch):
        res = forward(self, outputs, batch)
        if outputs["charge_logits"].requires_grad:  # a training step, not validation
            steps.append(float(res["loss"].detach()))
        return res

    monkeypatch.setattr(ChargeLoss, "forward", recording)
    out = tmp_path / "net.pt"
    train(TrainConfig(max_seconds=1e-6, **_cfg(out, gpu_speedups=("tf32",))),
          synth_config=_synth())
    state = torch.load(out.with_suffix(".resume.pt"), weights_only=False)
    assert (state["epoch"], state["step"], state["stop_reason"]) == (0, 1, "time")
    running = state["running"]
    assert isinstance(running, tuple) and running == (steps[0], 1)  # as earlier files hold it
    assert type(running[0]) is float
    _check_same_run(state["config"], {**state["config"], "gpu_speedups": ("fused_adamw",)})
    _m, history = train(TrainConfig(resume=True,
                                    **_cfg(out, gpu_speedups=("index_select", "fused_adamw"))),
                        synth_config=_synth())
    assert [h["epoch"] for h in history] == [0, 1] and len(steps) == 4
    first = 0.0
    for v in steps[:2]:
        first += v
    second = 0.0
    for v in steps[2:]:
        second += v
    assert history[0]["train_loss"] == first / 2 and history[1]["train_loss"] == second / 2


def test_switching_fused_adamw_on_a_loaded_state():
    """A state saved by a run with or without fused AdamW, loaded by one that asks the other way:
    the optimiser updates as the new run asks, from the same moments and step count."""
    from lcmsdeconv.nn.train import _use_fused

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(50))
    for fused_before, fused_now in ((False, True), (True, False)):
        a = torch.optim.AdamW([p], lr=1e-2, fused=True if fused_before else None)
        for _ in range(3):
            p.grad = torch.randn(50)
            a.step()
        saved = copy.deepcopy(a.state_dict())
        b = torch.optim.AdamW([p], lr=1e-2, fused=True if fused_now else None)
        b.load_state_dict(saved)
        assert bool(b.param_groups[0]["fused"]) == fused_before  # the saved run's choice
        _use_fused(b, fused_now)
        assert bool(b.param_groups[0]["fused"]) == fused_now
        st = b.state[p]
        assert float(st["step"]) == 3.0 and st["step"].dtype == torch.float32
        before = p.detach().clone()
        p.grad = torch.randn(50)
        b.step()
        assert float(b.state[p]["step"]) == 4.0 and not torch.equal(p.detach(), before)


def test_the_training_script_takes_the_speedups_and_refuses_unknown_ones(tmp_path):
    import subprocess

    script = str(ROOT / "scripts" / "train_charge_net.py")
    res = subprocess.run([sys.executable, script, "--device", "cpu", "--gpu-speedups",
                          "tf32,fast", "--out", str(tmp_path / "x.pt")], capture_output=True,
                         text=True, timeout=600)
    assert res.returncode == 2 and "unknown GPU speedup fast" in res.stderr
    out = tmp_path / "net.pt"
    res = subprocess.run([sys.executable, script, "--device", "cpu", "--model-size", "small",
                          "--epochs", "1", "--train-len", "2", "--val-len", "1", "--batch-size",
                          "1", "--workers", "0", "--threads", "1", "--gpu-speedups",
                          "fused_adamw,tf32", "--out", str(out)], capture_output=True,
                         text=True, timeout=900)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "GPU speedups ignored (tf32, fused_adamw): this run trains on cpu" in res.stdout
    st = json.loads(out.with_suffix(".status.json").read_text())
    assert st["stop_reason"] == "finished" and st["gpu_speedups"] == []
    assert "data_wait_share" in st["pace"]
    assert torch.load(out, weights_only=False)["config"]["gpu_speedups"] == ("tf32", "fused_adamw")


def test_batches_are_copied_to_the_gpu_without_waiting():
    from lcmsdeconv.nn.train import _batch_dict

    class Tensor:
        def to(self, device, **kw):
            self.device, self.kw = device, kw
            return self

    for device, waits in (("cuda", False), ("cuda:0", False), ("cpu", True)):
        sample = tuple(Tensor() for _ in range(4))
        batch = _batch_dict(sample, device)
        assert list(batch) == ["features", "topk_z", "topk_w", "heat"]
        assert all(t.device == device and t.kw == {"non_blocking": not waits} for t in sample)


def test_the_speedups_apply_on_cuda_only_and_the_flags_come_back(capsys):
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _active_speedups, restore_gpu_flags, set_gpu_flags

    ladder = build_model("small", 100, 4, arch="ladder", window=8192)
    unet = build_model("small", 100, 4, arch="unet")
    every = TrainConfig(gpu_speedups=("fused_adamw", "index_select", "tf32", "cudnn_benchmark"))
    assert _active_speedups(every, "cuda", ladder) == ("cudnn_benchmark", "tf32", "index_select",
                                                       "fused_adamw")
    assert _active_speedups(every, "cuda", unet) == ("cudnn_benchmark", "tf32", "fused_adamw")
    assert _active_speedups(every, "cpu", ladder) == ()
    assert _active_speedups(TrainConfig(), "cuda", ladder) == ()
    text = capsys.readouterr().out
    assert "index_select ignored: it is LadderNet's" in text and "ignored (" in text
    with pytest.raises(ValueError, match="unknown GPU speedup"):
        _active_speedups(TrainConfig(gpu_speedups=("bf16",)), "cuda", ladder)
    start = (torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)
    before = set_gpu_flags(("tf32", "cudnn_benchmark"))
    try:
        assert before == start
        assert torch.backends.cudnn.benchmark and torch.backends.cuda.matmul.allow_tf32
        inner = set_gpu_flags(())  # every switch off unless named
        assert not torch.backends.cudnn.benchmark and not torch.backends.cuda.matmul.allow_tf32
        restore_gpu_flags(inner)
        assert torch.backends.cudnn.benchmark and torch.backends.cuda.matmul.allow_tf32
    finally:
        restore_gpu_flags(before)
    assert (torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32) == start


def test_the_epoch_watch_times_every_25th_step_on_the_gpu(monkeypatch):
    """CUDA events stand in for the GPU: each record advances a clock by 10 ms."""
    from lcmsdeconv.nn.train import TrainConfig, _EpochWatch, _pace_line, _RamWatch

    clock = [0.0]
    created = []

    class Event:
        def __init__(self, enable_timing=False):
            assert enable_timing
            created.append(self)

        def record(self):
            clock[0] += 10.0
            self.t = clock[0]

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return end.t - self.t

    retries = [2]
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: {"num_alloc_retries": retries[0]})
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 3.5 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3.0 * 2**30)
    watch = _EpochWatch("cuda")
    for k in range(60):
        watch.step_starts(k)
        watch.step_ends()
    retries[0] = 5
    assert len(created) == 6  # steps 0, 25 and 50: a start and an end each
    rec = watch.summary(2.0, 60)
    assert rec == {"data_wait_share": 0.0, "gpu_step_ms": 10.0, "gpu_steps_timed": 3,
                   "gpu_busy_share": 0.3, "gpu_reserved_gb": 3.5, "gpu_allocated_gb": 3.0,
                   "alloc_retries": 3}
    ram = _RamWatch(None)
    ram.proc = None
    line = _pace_line({"steps": 60, "seconds": 2.0}, 0, 0.0, TrainConfig(epochs=1, batch_size=4),
                      60, ram, "cuda", rec)
    assert "of the training time 0 % waiting for data, 30 % GPU busy (10 ms a step)" in line
    assert "GPU 3.5 GB reserved, 3.0 GB allocated at most, 3 allocator retries" in line


def test_the_epoch_watch_measures_the_wait_for_data():
    import time as _time

    from lcmsdeconv.nn.train import _EpochWatch

    def slow():
        for k in range(3):
            _time.sleep(0.05)
            yield k

    watch = _EpochWatch("cpu")
    got = []
    for item in watch.batches(slow()):
        got.append(item)
        _time.sleep(0.02)  # a step
    assert got == [0, 1, 2] and 0.14 <= watch.wait < 2.0  # not the steps' 0.06 s
    rec = watch.summary(0.25, 3)
    assert rec == {"data_wait_share": round(watch.wait / 0.25, 4)}


# ------------------------------------------------------------------ the GPU package
def _train_all_in(tmp_path, monkeypatch, machine: dict):
    train_all = _load("train_all")
    monkeypatch.setattr(train_all, "ROOT", tmp_path)
    monkeypatch.setattr(train_all, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(train_all, "MACHINE", tmp_path / "gpu_training" / "machine.json")
    (tmp_path / "runs").mkdir()
    (tmp_path / "gpu_training").mkdir()
    train_all.MACHINE.write_text(json.dumps(machine))
    for name in train_all.RUN_TABLE:  # what the (fake) trainings leave, for the export
        for smoke in (False, True):
            (tmp_path / train_all.make_run(name, smoke=smoke)["out"]).write_text("checkpoint")
    commands = []

    def popen(cmd, **_kw):
        commands.append(cmd)
        proc = SimpleNamespace(stdout=iter(()), wait=lambda: 0)
        return proc

    monkeypatch.setattr(train_all.subprocess, "Popen", popen)
    monkeypatch.setattr(train_all.subprocess, "run",
                        lambda cmd, **_kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(train_all.signal, "signal", lambda *_a: None)
    return train_all, commands


def _speedups_of(cmd) -> str:
    return cmd[cmd.index("--gpu-speedups") + 1]


def test_each_run_gets_the_speedups_setup_chose_for_its_view(tmp_path, monkeypatch, capsys):
    machine = {"workers": 6, "gpu_speedups": {"window": ["index_select", "tf32"],
                                              "comb": [], "other": ["fused_adamw"]}}
    train_all, commands = _train_all_in(tmp_path, monkeypatch, machine)
    assert train_all.speedups_for(machine, "window") == ["tf32", "index_select"]
    assert train_all.speedups_for(machine, "comb") == []
    assert train_all.speedups_for({"gpu_speedups": {"window": ["tf32", "turbo"]}},
                                  "window") == ["tf32"]
    for smoke in (True, False):  # the smoke run checks them too
        commands.clear()
        monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu", "--workers", "1",
                                          "--restart", *(["--smoke"] if smoke else [])])
        assert train_all.main() == 0
        assert [_speedups_of(c) for c in commands] == ["tf32,index_select", "none"]
        assert all(c[-1] == "--restart" for c in commands)  # still the last argument
    text = capsys.readouterr().out
    assert "scoring 16 charges at a time; GPU speedups tf32, index_select" in text
    assert "scoring 16 charges at a time; no GPU speedups" in text
    assert "running setup again measures them" not in text


def test_an_earlier_machine_json_gives_no_speedups_and_asks_for_setup(tmp_path, monkeypatch,
                                                                       capsys):
    v3 = {"recipe": "v3", "workers": 6, "v3_spectra_per_second": 30.0, "v3_workers": 6}
    train_all, commands = _train_all_in(tmp_path, monkeypatch, v3)
    assert train_all.data_rate(v3, 6) is None  # measured on the spectra of another recipe
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--device", "cpu", "--runs", "window"])
    assert train_all.main() == 0
    assert [_speedups_of(c) for c in commands] == ["none"]
    text = capsys.readouterr().out
    assert "holds no measurement of this recipe (v4" in text
    assert "no GPU speedups: gpu_training/machine.json does not say" in text
    assert "running setup again measures them" in text


def test_the_package_trains_recipe_v4():
    train_all = _load("train_all")
    probe = _load("probe")
    import train_charge_net

    assert train_all.RECIPE == "v4" and (train_all.RATE_KEY, train_all.RATE_WORKERS_KEY) == (
        "v4_spectra_per_second", "v4_workers")
    for name in train_all.RUN_TABLE:
        run = train_all.make_run(name)
        assert run["args"][run["args"].index("--recipe") + 1] == "v4"
        assert run["out"].startswith("runs/ladder_v4_")
        assert train_all.make_run(name, smoke=True)["out"].startswith("runs/smoke_v4_")
    mix, v4 = probe.recipe_mixture(), train_charge_net.mixture_v4()
    assert [w for w, _c in mix] == [w for w, _c in v4]
    for (_w, c), (_w4, c4) in zip(mix, v4):
        assert (c.singly_charged, c.iso_channels, c.measured_channel) == ("keep", True, True)
        assert c.oligo_single_state_prob == c4.oligo_single_state_prob
        assert c.classes == c4.classes and c.resolution_range == c4.resolution_range
    assert any(c.oligo_single_state_prob > 0 for _w, c in mix)


def test_check_model_reads_a_spectrum_of_recipe_v4(monkeypatch):
    check_model = _load("check_model")
    import train_charge_net

    class Asked(Exception):
        pass

    def v4():
        raise Asked

    monkeypatch.setattr(train_charge_net, "mixture_v4", v4)
    with pytest.raises(Asked):
        check_model.spectra()
