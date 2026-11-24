"""LadderNet's two views and the isotope-limit channel.

The window view reads a fixed window; the comb view reads any length whole, with the measured
channel last and a learned vector where a tooth falls outside the input or on an unmeasured bin.
The isotope channels gain a third, the isotope limit, which gates the isotope prior; networks of
the first layout (two isotope channels, as commit 07cdc43 saved them) read as before
(``tests/data/ref_07cdc43.json``, made by ``tests/data/make_ref_07cdc43.py``).
"""

import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pytest

from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.synth.compounds import ClassConfig
from lcmsdeconv.synth.config import SynthConfig
from lcmsdeconv.synth.frames import generate_frame

from ._digests import arrays_digest, canary

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
REF07 = json.loads((DATA / "ref_07cdc43.json").read_text())
W = 131072


def _script():
    spec = importlib.util.spec_from_file_location("train_charge_net",
                                                  ROOT / "scripts" / "train_charge_net.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ladder(view="comb", iso=True, z_ladder=12, **kw):
    """A LadderNet with few charges (the far rungs z = 1 <-> 2, 3, 4 included) and a learned
    part that is not zero."""
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn.model import LadderNet

    in_ch = 4 + (3 if iso else 0) + (1 if view == "comb" else 0)
    torch.manual_seed(0)
    model = LadderNet(in_ch, 100, z_ladder=z_ladder, width=8, proj=4, hidden=12, window=W,
                      active_top=24576, iso=iso, view=view, **kw)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model.eval()


def _features(length, in_ch, measured=None, seed=0, n_peaks=4000):
    """Random peaks, ladders whose rungs lie as far apart as z = 1 <-> 2, 3, 4, isotope charges,
    strengths and limits; with a measured channel (last), bins from ``measured`` on are padding
    (zero input)."""
    rng = np.random.default_rng(seed)
    m = length if measured is None else measured
    x = np.zeros((1, in_ch, length), dtype=np.float32)
    idx = rng.integers(0, m, n_peaks)
    x[0, 0, idx] = rng.uniform(0.0, 5.0, idx.size)
    for b in rng.integers(0, m, 40):
        for d in (0, 34657, 54931, 69315):
            if 0 <= b - d < m:
                x[0, 0, b - d] = 4.0
    x[0, 1, :m] = rng.uniform(0.0, 1.0, m)
    if in_ch >= 7:
        z = rng.integers(0, 30, m).astype(np.float32)
        z[rng.random(m) < 0.5] = 0.0
        x[0, 4, :m] = z
        x[0, 5, :m] = np.where(z > 0, rng.uniform(0.0, 1.0, m), 0.0)
        x[0, 6, :m] = rng.integers(0, 40, m)
    if measured is not None:
        x[0, -1, :m] = 1.0
    return x


# ------------------------------------------------------------------ the isotope limit
def test_isotope_limit_is_the_highest_charge_the_reader_resolves():
    from lcmsdeconv.nn.features import ISOTOPE_RESOLUTION_FACTOR, isotope_limit
    from lcmsdeconv.nn.grid import LogMzGrid

    grid = LogMzGrid(300.0, 3000.0, 2e-5, -1)
    for inst in (InstrumentModel("orbitrap", 140000.0), InstrumentModel("orbitrap", 35000.0),
                 InstrumentModel("fticr", 400000.0)):
        zlim = isotope_limit(grid, inst)
        assert zlim.shape == (grid.size,) and zlim.max() <= 60
        bins = np.linspace(0, grid.size - 1, 400).astype(int)
        mz = grid.bin_to_mz(bins)
        off = np.abs(mz - grid.carrier)
        width = np.maximum(np.log1p(mz / inst.resolving_power(mz) / off) / grid.step, 2.0)

        def resolved(z, off=off, width=width):
            return np.log1p(1.0033548 / z / off) / grid.step >= ISOTOPE_RESOLUTION_FACTOR * width

        lim = zlim[bins]
        assert (resolved(np.maximum(lim, 1)) | (lim == 0)).all()
        assert not resolved(lim + 1)[lim < 60].any()
        assert np.all(np.diff(zlim.astype(int)) <= 0)  # lower charges resolve at higher m/z
    # the real runs' Orbitrap: z = 10 at m/z 1 000, z = 5 at 2 000 (two-bin peaks on the grid)
    orbi = isotope_limit(grid, InstrumentModel("orbitrap", 140000.0))
    assert orbi[grid.mz_to_bin(1000.0)] == 10 and orbi[grid.mz_to_bin(2000.0)] == 5


def test_isotope_limit_channel_is_cut_from_the_whole_grid_and_bounds_the_reading():
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.features import isotope_channels, isotope_limit

    cfg = SynthConfig(classes=[ClassConfig("ps_rna", (4000.0, 12000.0))], mode="iprp",
                      polarity=-1, instrument_kind="orbitrap",
                      resolution_range=(100000.0, 200000.0), crop_size=8192, iso_channels=True)
    fs = generate_frame(cfg, np.random.default_rng(4))
    assert fs.features.shape[0] == 7
    grid, y, sigma = bl.frame_grid(cfg, fs)
    inst = bl.frame_instrument(fs.meta)
    whole = isotope_channels(grid, y, sigma, inst)
    np.testing.assert_array_equal(whole[2], isotope_limit(grid, inst).astype(np.float32))
    np.testing.assert_array_equal(fs.features[4:], whole[:, fs.b0:fs.b0 + cfg.crop_size])
    read = whole[0] > 0
    assert read.sum() > 100 and (whole[0][read] <= whole[2][read]).all()  # never above the limit
    # the first layout is the first two channels; a time-of-flight spectrum gets none, limit too
    np.testing.assert_array_equal(isotope_channels(grid, y, sigma, inst, count=2), whole[:2])
    tof = isotope_channels(grid, y, sigma, InstrumentModel("tof", 40000.0))
    assert tof.shape == (3, grid.size) and not tof.any()


def test_untrained_network_gates_isotope_readings_that_may_be_half_an_unresolved_charge():
    """The isotope prior counts where twice the read charge is still resolvable (zlim >= 2 zc);
    elsewhere the untrained network reads the ladder alone. Dense equals sparse."""
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn.model import build_model

    torch.manual_seed(0)
    model = build_model("proof", 100, 7, arch="ladder", window=W, iso=True).eval()
    plain = build_model("proof", 100, 4, arch="ladder", window=W).eval()
    assert (model.iso_count, model.mlp1.in_features) == (3, plain.mlp1.in_features + 4)
    x = torch.from_numpy(_features(W, 7, seed=1))
    with torch.no_grad():
        logits = model(x)["charge_logits"][0]
        logits_plain = plain(x[:, :4])["charge_logits"][0]
    level = x[0, 0]
    active = (level > 0.5) & (level >= torch.topk(level, model.active_top).values[-1])
    zc, st, zl = x[0, 4], x[0, 5], x[0, 6]
    top = logits[1:].argmax(0) + 1
    strong = active & (st >= 0.3) & (zc >= 1) & (zc <= model.z_ladder)
    open_, closed = strong & (2 * zc <= zl), strong & (2 * zc > zl + 1)
    assert open_.sum() > 300 and closed.sum() > 300
    assert bool((top[open_] == zc[open_].long()).all())
    assert torch.equal(logits[:, closed], logits_plain[:, closed])
    light = _ladder("window")  # the same function densely (fewer charges, a lighter encoder)
    with torch.no_grad():
        sparse = light(x)
        light.dense = True
        dense = light(x)
    for k in ("charge_logits", "apex_logit"):
        assert torch.allclose(sparse[k], dense[k], atol=1e-4), k


# ------------------------------------------------------------------ the first layout, as saved
def _same_arithmetic_07() -> bool:
    import torch

    env = REF07["env"]
    return (env["machine"] == platform.machine() and env["numpy"] == np.__version__
            and env["canary"] == canary() and env["torch"] == torch.__version__.split("+")[0])


def test_a_first_layout_network_saved_by_07cdc43_reads_as_before(tmp_path):
    """A checkpoint 07cdc43 wrote (two isotope channels, in_ch 6, no view recorded) loads as a
    window view of the first layout, computes what it computed then, exports to ONNX, and the
    predictor reads a spectrum with it, checkpoint or ONNX with an old sidecar, as 07cdc43's did."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    import shutil

    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.export import export_onnx, load_torch_model, verify_parity
    from lcmsdeconv.nn.infer import ChargePredictor

    ckpt = tmp_path / "iso2.pt"
    shutil.copy(DATA / "laddernet_iso2_07cdc43.pt", ckpt)
    raw = torch.load(ckpt, weights_only=False)
    assert "view" not in raw and "iso_count" not in raw and raw["in_ch"] == 6
    model, _ = load_torch_model(ckpt)
    assert (model.view, model.iso, model.iso_count, model.window) == ("window", True, 2, 8192)
    rng = np.random.default_rng(7)  # the reference input (make_ref_07cdc43.iso2_input)
    x = np.zeros((1, 6, 8192), dtype=np.float32)
    idx = rng.integers(0, 8192, 1500)
    x[0, 0, idx] = rng.uniform(0.0, 5.0, idx.size)
    x[0, 1] = rng.uniform(0.0, 1.0, 8192)
    z = rng.integers(0, 20, 8192).astype(np.float32)
    z[rng.random(8192) < 0.5] = 0.0
    x[0, 4] = z
    x[0, 5] = np.where(z > 0, rng.uniform(0.0, 1.0, 8192), 0.0)
    with torch.no_grad():
        logits = model(torch.from_numpy(x))["charge_logits"][0].numpy()
    ref = REF07["outputs"]["samples"]
    np.testing.assert_allclose([logits[c, b] for c, b, _v in ref], [v for _c, _b, v in ref],
                               rtol=0, atol=1e-5)

    onnx_path = export_onnx(ckpt, tmp_path / "iso2.onnx")
    side = json.loads(onnx_path.with_suffix(".json").read_text())
    assert (side["iso_count"], side["view"], side["in_ch"]) == (2, "window", 6)
    assert verify_parity(ckpt, onnx_path)["ok"]
    # the sidecar 07cdc43 wrote, which records neither
    reads = REF07["predict_grid"]
    onnx_path.with_suffix(".json").write_text(json.dumps(reads["sidecar"]))
    cfg = _script().mixture_v3()[3][1]
    fs = generate_frame(cfg, np.random.default_rng(reads["frame_seed"]))
    grid, y, sigma = bl.frame_grid(cfg, fs)
    inst = InstrumentModel(fs.meta["kind"], float(fs.meta["resolution"]))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as recorded
    try:
        for backend, path in (("torch", ckpt), ("onnx", onnx_path)):
            pred = ChargePredictor(path, instrument=inst)
            assert (pred.view, pred.iso_count, pred.window) == ("window", 2, 8192)
            p = pred.predict_grid(grid, y, sigma)
            rec = reads[backend]
            assert p.top1_z.shape == (grid.size,)
            if _same_arithmetic_07():  # bit for bit where the reference was computed alike
                assert grid.size == reads["grid_size"]
                assert int((p.top1_z > 0).sum()) == rec["charged"]
                assert arrays_digest(p.top1_z, p.top2_z) == rec["digest"], backend
                assert abs(float(p.top1_p.astype(np.float64).sum()) - rec["top1_p_sum"]) < 1e-3
                assert abs(float(p.apex.astype(np.float64).sum()) - rec["apex_sum"]) < 1e-3
    finally:
        torch.set_num_threads(threads)


# ------------------------------------------------------------------ the comb view
def test_comb_view_dense_equals_sparse_at_any_length():
    """With isotope channels, an unmeasured tail and lengths below and above the far rungs."""
    torch = pytest.importorskip("torch")
    model = _ladder("comb")
    assert model.reach == 69315 and all(v is not None for o in model.offsets[1:] for v in o[:3])
    for length, measured in ((3200, 3000), (40000, 39000), (70016, 69500)):
        x = torch.from_numpy(_features(length, 8, measured=measured))
        with torch.no_grad():
            model.dense = False
            sparse = model(x)
            model.dense = True
            dense = model(x)
        model.dense = False
        for k in ("charge_logits", "apex_logit"):
            assert torch.allclose(sparse[k], dense[k], atol=1e-4), (length, k)


def test_comb_view_reads_unmeasured_teeth_as_e_unmeasured_and_no_further():
    """Outputs on the measured bins do not depend on how long the unmeasured tail is, and a
    tooth outside the input reads e_unmeasured: changing it moves only the bins whose ladder
    reaches past the measured bins."""
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn.infer import COMB_TAIL

    model = _ladder("comb")
    m = 40000
    x = torch.from_numpy(_features(m + 20032, 8, measured=m, seed=3))
    # the encoder reaches 60 bins: past a tail of COMB_TAIL zero bins, as long as any
    short = x[..., :m + COMB_TAIL]
    with torch.no_grad():
        a, b = model(short), model(x)
    for k in ("charge_logits", "apex_logit"):
        assert torch.allclose(a[k][..., :m], b[k][..., :m], atol=1e-5), k
    with torch.no_grad():
        model.e_unmeasured.add_(1.0)
        c = model(short)["charge_logits"][0, :, :m]
    moved = (c - a["charge_logits"][0, :, :m]).abs().amax(0) > 1e-4
    reach_out = torch.zeros(m, dtype=torch.bool)  # a tooth lands past the measured bins
    for o in model.offsets:
        for s in o:
            if s is not None and s > 0:
                reach_out[max(0, m - s):] = True
            elif s is not None and s < 0:
                reach_out[:min(m, -s)] = True
    near = reach_out.clone()  # and the smoothing over two bins either side
    for d in (1, 2):
        near[d:] |= reach_out[:-d]
        near[:-d] |= reach_out[d:]
    assert moved.sum() > 100 and not (moved & ~near).any()


def test_comb_view_needs_the_measured_channel_and_keeps_every_offset():
    pytest.importorskip("torch")
    from lcmsdeconv.nn.model import build_model

    with pytest.raises(ValueError, match="8 input channels"):
        build_model("small", 100, 7, arch="ladder", window=W, iso=True, iso_count=3, view="comb")
    small = build_model("small", 100, 5, arch="ladder", window=8192, view="comb")
    window = build_model("small", 100, 4, arch="ladder", window=8192)
    assert small.reach == 69315 and window.reach < 8192  # no window limit on the offsets
    with pytest.raises(ValueError, match="LadderNet"):
        build_model("small", 100, 4, arch="unet", view="comb")


def test_comb_view_exports_with_an_open_length_axis(tmp_path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    from lcmsdeconv.nn.export import COMB_PARITY_LENGTHS, export_onnx, verify_parity
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _save

    torch.manual_seed(0)
    model = build_model("small", 100, 8, arch="ladder", window=W, iso=True, view="comb")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ckpt = tmp_path / "comb.pt"
    _save(ckpt, model.state_dict(), TrainConfig(arch="ladder", model_size="small", window=W,
                                                iso_channels=True, view="comb"), 8, None, None,
          "charge_acc_given_ion")
    onnx_path = export_onnx(ckpt, tmp_path / "comb.onnx")
    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    assert (meta["view"], meta["iso_count"], meta["in_ch"], meta["active_top"]) == \
        ("comb", 3, 8, 24576)
    res = verify_parity(ckpt, onnx_path)
    assert res["ok"] and res["lengths"] == list(COMB_PARITY_LENGTHS), res
    # a real grid's length (398-2 500 m/z, padded to a multiple of 64), measured to its end
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model.eval()
    x = _features(91648, 8, measured=91603, seed=5)
    with torch.no_grad():
        ref = model(torch.from_numpy(x))["charge_logits"].numpy()
    assert np.abs(sess.run(None, {"features": x})[0] - ref).max() < 1e-3


def test_comb_predictor_reads_a_long_grid_whole_in_one_pass(tmp_path):
    """A grid longer than the training crop: one pass at its own length (padded to a multiple of
    64), the measured channel 1 on the grid's bins, isotope channels of the whole grid."""
    pytest.importorskip("torch")
    from lcmsdeconv.nn.features import isotope_channels
    from lcmsdeconv.nn.grid import LogMzGrid
    from lcmsdeconv.nn.infer import ChargePredictor, comb_length
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _save

    ckpt = tmp_path / "comb.pt"
    _save(ckpt, build_model("small", 100, 8, arch="ladder", window=W, iso=True,
                            view="comb").state_dict(),
          TrainConfig(arch="ladder", model_size="small", window=W, iso_channels=True,
                      view="comb"), 8, None, None, "charge_acc_given_ion")
    orbi = InstrumentModel("orbitrap", 140000.0)
    pred = ChargePredictor(ckpt, instrument=orbi)
    assert (pred.view, pred.iso_count) == ("comb", 3)
    grid = LogMzGrid(150.0, 3000.0, 2e-5, -1)
    assert grid.size > W
    rng = np.random.default_rng(0)
    y = np.zeros(grid.size)
    for b in rng.integers(0, grid.size - 10, 300):
        y[b:b + 3] = rng.uniform(20.0, 2000.0)
    seen = []
    run = pred._run
    pred._run = lambda feats: (seen.append(feats), run(feats))[1]
    p = pred.predict_grid(grid, y, 1.0)
    length = comb_length(grid.size)
    assert length % 64 == 0 and grid.size + 64 <= length < grid.size + 128
    assert len(seen) == 1 and seen[0].shape == (8, length)
    assert (seen[0][7, :grid.size] == 1).all() and not seen[0][7, grid.size:].any()
    np.testing.assert_array_equal(seen[0][4:7, :grid.size], isotope_channels(grid, y, 1.0, orbi))
    assert p.top1_z.shape == (grid.size,) and (p.top1_z[y > 3] > 0).all()


# ------------------------------------------------------------------ comb-view frames and runs
def test_measured_channel_marks_the_grid_and_leaves_the_frame_as_drawn():
    from lcmsdeconv.nn import baseline as bl

    small = SynthConfig(classes=[ClassConfig("ps_rna", (4000.0, 12000.0))], mode="iprp",
                        polarity=-1, instrument_kind="orbitrap",
                        resolution_range=(100000.0, 200000.0), crop_size=8192, iso_channels=True)
    whole = _script().mixture_v3()[2][1]  # an Orbitrap spectrum of the recipe: grid < W
    whole.iso_channels = True
    for base, seed in ((small, 4), (whole, 2)):
        crop = base.crop_size
        comb = SynthConfig(**{**base.__dict__, "measured_channel": True})
        a = generate_frame(base, np.random.default_rng(seed))
        b = generate_frame(comb, np.random.default_rng(seed))
        assert b.features.shape[0] == a.features.shape[0] + 1 and a.b0 == b.b0
        np.testing.assert_array_equal(b.features[:-1], a.features)
        for k in ("topk_z", "topk_w", "heat"):
            np.testing.assert_array_equal(getattr(b, k), getattr(a, k))
        grid, _y, _s = bl.frame_grid(base, a)
        covered = min(crop, grid.size - a.b0)
        assert (b.features[-1, :covered] == 1).all() and not b.features[-1, covered:].any()
        if crop == 8192:
            assert grid.size > crop and covered == crop  # a crop of a longer grid: all measured
        else:
            assert covered == grid.size < crop  # the zero-padded tail is not


def test_training_records_the_view_and_a_resume_keeps_it(tmp_path):
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.nn.train import TrainConfig, _check_same_run, train

    synth = SynthConfig(classes=[ClassConfig("ps_rna", (3000.0, 6000.0))], mode="iprp",
                        polarity=-1, instrument_kind="orbitrap",
                        resolution_range=(100000.0, 150000.0))
    out = tmp_path / "comb.pt"
    cfg = dict(arch="ladder", model_size="small", epochs=1, train_len=2, val_len=2, batch_size=2,
               num_workers=0, threads=1, out=str(out), log_every=100, window=16384,
               iso_channels=True, view="comb", head_chunk=16)
    model, _h = train(TrainConfig(**cfg), synth_config=synth)
    assert synth.measured_channel and synth.crop_size == 16384  # the frames follow the network
    assert (model.view, model.iso_count, model.in_ch, model.head_chunk) == ("comb", 3, 8, 16)
    ckpt = torch.load(out, weights_only=False)
    assert (ckpt["view"], ckpt["iso_count"], ckpt["in_ch"]) == ("comb", 3, 8)
    pred = ChargePredictor(out)
    assert (pred.view, pred.iso_count) == ("comb", 3)
    state = torch.load(out.with_suffix(".resume.pt"), weights_only=False)
    assert state["iso_count"] == 3 and state["config"]["view"] == "comb"
    with pytest.raises(SystemExit, match="view"):
        train(TrainConfig(resume=True, **{**cfg, "view": "window"}), synth_config=synth)
    # resume states written before the view existed are window views; before the isotope limit,
    # isotope-channel ones read two channels
    old = {"arch": "ladder", "model_size": "small", "epochs": 1, "active_top": 6144}
    _check_same_run(old, {**old, "view": "window", "iso_count": 0})
    with pytest.raises(SystemExit, match="view"):
        _check_same_run(old, {**old, "view": "comb", "iso_count": 0})
    with pytest.raises(SystemExit, match="iso_count"):
        _check_same_run({**old, "iso_channels": True, "iso_count": 2},
                        {**old, "iso_channels": True, "iso_count": 3})
