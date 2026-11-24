"""The whole-spectrum training recipe (``--recipe v3``) and its options.

Every option is off by default and must leave the package as it was: default training frames,
the comb estimator, the resume fingerprints and LadderNet are compared with what commit 1917c37
did (``tests/data/ref_1917c37.json``, made by ``tests/data/make_ref_1917c37.py``). The digests
compare only where the machine computes the transcendental functions and random draws exactly
as the recording machine did (``_digests.canary``); elsewhere those tests skip. The rest checks
what the options do: the 131 072-bin window and its ladder reach, the isotope-charge channels
(training and inference compute the same ones; the untrained network reads them as the comb
does), the instrument scan range, strong background and real-like ion-pairing oligonucleotides.
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

from ._digests import arrays_digest, canary, frame_digest, ladder_input

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
REF = json.loads((DATA / "ref_1917c37.json").read_text())
W = 131072


def _script():
    spec = importlib.util.spec_from_file_location("train_charge_net",
                                                  ROOT / "scripts" / "train_charge_net.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same_arithmetic() -> bool:
    env = REF["env"]
    return (env["machine"] == platform.machine() and env["numpy"] == np.__version__
            and env["canary"] == canary())


needs_same_arithmetic = pytest.mark.skipif(
    not _same_arithmetic(), reason="digests were recorded where numpy computes differently")


# ------------------------------------------------------------------ defaults: as before
@needs_same_arithmetic
def test_default_frames_are_the_frames_before_the_options():
    from lcmsdeconv.synth.dataset import FrameDataset

    mixture = _script().mixture
    for rec in REF["frames"]:
        if "dataset_item" in rec:
            fs = FrameDataset(mixture(), 2, base_seed=0).frame(rec["dataset_item"])[1]
        else:
            k, stage = rec["component"], rec["stage"]
            cfg = (mixture() if stage == "ignore" else mixture(background_ions=200))[k][1]
            cfg.singly_charged = stage
            fs = generate_frame(cfg, np.random.default_rng(rec["seed"]))
        assert frame_digest(fs) == rec["digest"], rec


@needs_same_arithmetic
def test_comb_with_an_instrument_reads_as_before():
    from lcmsdeconv.deconv.classical import CombPredictor
    from lcmsdeconv.nn import baseline as bl

    mixture = _script().mixture
    for rec in REF["comb"]:
        k, stage, seed = rec["frame"].split("/")
        cfg = (mixture() if stage == "ignore" else mixture(background_ions=200))[int(k)][1]
        cfg.singly_charged = stage
        fs = generate_frame(cfg, np.random.default_rng(int(seed)))
        grid, y, sigma = bl.frame_grid(cfg, fs)
        pred = CombPredictor(z_max=rec["z_max"],
                             instrument=InstrumentModel(rec["kind"], rec["resolution"]))
        p = pred.predict_grid(grid, y, sigma)
        ns = max(sigma, 1e-9)
        s = np.clip(np.log10(1.0 + np.clip(y, 0, None) / ns), 0.0, 5.0)
        zi, conf = pred.isotope_charges(grid, s, y > 3.0 * ns)
        assert arrays_digest(p.top1_z, p.top1_p, p.top2_z, p.top2_p, p.apex) == rec["predict_grid"]
        assert arrays_digest(zi, conf) == rec["isotope_charges"]


def test_comb_isotope_charges_are_the_shared_reading_and_its_strength_rule():
    from lcmsdeconv.deconv.classical import CombPredictor
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.features import isotope_channels, isotope_reading

    cfg = _script().mixture(crop=8192)[2][1]  # Orbitrap oligonucleotides
    fs = generate_frame(cfg, np.random.default_rng(5))
    grid, y, sigma = bl.frame_grid(cfg, fs)
    inst = bl.frame_instrument(fs.meta)
    s = np.clip(np.log10(1.0 + np.clip(y, 0, None) / sigma), 0.0, 5.0)
    active = y > 3.0 * sigma
    zi, conf = CombPredictor(z_max=60, instrument=inst).isotope_charges(grid, s, active)
    zraw, best = isotope_reading(grid, s, active, inst, z_max=60)
    strong = best >= 0.25 * s ** 2
    np.testing.assert_array_equal(zi, np.where(active & strong, zraw, 0))
    assert (zi > 0).sum() > 100
    assert (zraw[~active] == 0).all() and (best[~active] == 0).all()
    ch = isotope_channels(grid, y, sigma, inst)
    np.testing.assert_array_equal(ch[0], zraw.astype(np.float32))
    on = zraw > 0
    np.testing.assert_allclose(ch[1, on], np.clip(best[on] / s[on] ** 2, 0, 1), rtol=1e-6)
    assert (ch[1, ~on] == 0).all() and ch[1].max() <= 1.0


def test_resume_fingerprints_of_default_spectra_are_unchanged():
    from lcmsdeconv.nn.train import synth_fingerprint

    script = _script()
    keep = script.mixture(background_ions=200)
    for _w, c in keep:
        c.singly_charged = "keep"
    assert synth_fingerprint(script.mixture()) == REF["fingerprints"]["mixture"]
    assert synth_fingerprint(keep) == REF["fingerprints"]["mixture_keep_200"]
    for attr, value in (("scan_range", "instrument"), ("background_strong_prob", 0.25),
                        ("oligo_real_like_prob", 0.5), ("iso_channels", True),
                        ("measured_channel", True)):
        mix = script.mixture()
        setattr(mix[1][1], attr, value)
        assert synth_fingerprint(mix) != REF["fingerprints"]["mixture"], attr


def test_laddernet_without_isotope_channels_is_the_network_before():
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn.model import build_model

    ref = REF["laddernet"]
    state = torch.load(DATA / "laddernet_proof_1917c37.pt", weights_only=True)
    model = build_model("proof", z_max=100, in_ch=4, arch="ladder")
    assert {k: list(v.shape) for k, v in model.state_dict().items()} == ref["keys"]
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(ladder_input()))
    logits = out["charge_logits"][0].numpy()
    apex = out["apex_logit"][0, 0].numpy()
    got = np.array([logits[c, b] for c, b, _v in ref["samples"]])
    np.testing.assert_allclose(got, [v for _c, _b, v in ref["samples"]], rtol=0, atol=1e-5)
    np.testing.assert_allclose([apex[b] for b, _v in ref["apex_samples"]],
                               [v for _b, v in ref["apex_samples"]], rtol=0, atol=1e-5)
    if torch.__version__.split("+")[0] == REF["env"]["torch"]:  # same initialisation
        torch.manual_seed(0)
        fresh = build_model("proof", z_max=100, in_ch=4, arch="ladder")
        assert arrays_digest(*[t.detach().numpy() for t in fresh.state_dict().values()]) == \
            ref["init_digest"]


@needs_same_arithmetic
def test_predictor_reads_an_old_ladder_checkpoint_as_before(tmp_path):
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.nn.train import TrainConfig, _save

    if torch.__version__.split("+")[0] != REF["env"]["torch"]:
        pytest.skip("the digest was recorded with another PyTorch")
    ref = REF["laddernet"]["predict_grid"]
    ckpt = tmp_path / "old.pt"
    _save(ckpt, torch.load(DATA / "laddernet_proof_1917c37.pt", weights_only=True),
          TrainConfig(arch="ladder", model_size="proof"), 4, None, None, "charge_acc_given_ion")
    raw = torch.load(ckpt, weights_only=False)
    for key in ("active_top", "iso_channels", "iso_count", "view"):  # as written before them
        raw.pop(key)
    torch.save(raw, ckpt)
    pred = ChargePredictor(ckpt)
    assert (pred.window, pred.active_top, pred.iso_channels, pred.view) == \
        (32768, 6144, False, "window")
    cfg = _script().mixture(background_ions=200)[0][1]
    cfg.singly_charged = "keep"
    fs = generate_frame(cfg, np.random.default_rng(2000))
    grid, y, sigma = bl.frame_grid(cfg, fs)
    assert grid.size == ref["grid_size"]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as recorded
    try:
        p = pred.predict_grid(grid, y, sigma)
    finally:
        torch.set_num_threads(threads)
    assert arrays_digest(p.top1_z, p.top2_z) == ref["digest"]
    assert abs(float(p.top1_p.astype(np.float64).sum()) - ref["top1_p_sum"]) < 1e-3


# ------------------------------------------------------------------ window and ladder reach
def _iso_features(length, seed=0, n_peaks=6000):
    """A LadderNet input with isotope channels: random peaks, ladders whose rungs lie as far
    apart as z = 1 <-> 2, 3, 4, and whole isotope charges with strengths."""
    rng = np.random.default_rng(seed)
    x = np.zeros((1, 6, length), dtype=np.float32)
    idx = rng.integers(0, length, n_peaks)
    x[0, 0, idx] = rng.uniform(0.0, 5.0, idx.size)
    for b in rng.integers(70000, length - 10, 40):
        x[0, 0, [b, b - 34657, b - 54931, b - 69315]] = 4.0
    x[0, 1] = rng.uniform(0.0, 1.0, length)
    z = rng.integers(0, 30, length).astype(np.float32)
    z[rng.random(length) < 0.5] = 0.0
    x[0, 4] = z
    x[0, 5] = np.where(z > 0, rng.uniform(0.0, 1.0, length), 0.0)
    return x


def test_whole_window_ladder_reaches_z1_to_z4_and_sparse_equals_dense():
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn.model import build_model

    torch.manual_seed(0)
    model = build_model("small", z_max=100, in_ch=6, arch="ladder", window=W, iso=True)
    assert model.active_top == 24576 and model.window == W
    reach = {abs(v) for o in model.offsets for v in o if v is not None}
    assert {34657, 54931, 69315} <= reach  # z = 1 -> 2, 3, 4
    assert 34657 not in {abs(v) for o in build_model("small", arch="ladder").offsets
                         for v in o if v is not None}  # not within a 32 768-bin window
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()
    x = torch.from_numpy(_iso_features(W))
    with torch.no_grad():
        sparse = model(x)
        model.dense = True
        dense = model(x)
    for k in ("charge_logits", "apex_logit"):
        assert torch.allclose(sparse[k], dense[k], atol=1e-4), k


def test_untrained_iso_laddernet_reads_charges_as_the_comb_with_isotopes():
    """Where the isotope reading is strong (>= 0.3) the untrained network's top charge is the
    isotope charge, whatever the ladder says; elsewhere it is the comb's ladder."""
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn.model import build_model

    torch.manual_seed(0)
    model = build_model("proof", z_max=100, in_ch=6, arch="ladder", window=W, iso=True).eval()
    plain = build_model("proof", z_max=100, in_ch=4, arch="ladder", window=W).eval()
    x = torch.from_numpy(_iso_features(W, seed=1))
    with torch.no_grad():
        logits = model(x)["charge_logits"][0]
        logits_plain = plain(x[:, :4])["charge_logits"][0]
    level = x[0, 0]
    kth = torch.topk(level, model.active_top).values[-1]
    active = (level > model.active_level) & (level >= kth)
    zc, st = x[0, 4], x[0, 5]
    top = logits[1:].argmax(0) + 1
    strong = active & (st >= 0.3) & (zc >= 1) & (zc <= model.z_ladder)
    assert strong.sum() > 1000
    assert bool((top[strong] == zc[strong].long()).all())
    none = active & (zc == 0)
    assert torch.equal(logits[:, none], logits_plain[:, none])  # no reading: the plain ladder


def test_iso_laddernet_exports_to_onnx_at_the_whole_window(tmp_path):
    """The current layout (three isotope channels); tests/test_ladder_views.py exports the first
    layout's networks."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from lcmsdeconv.nn.export import export_onnx, verify_parity
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _save

    torch.manual_seed(0)
    model = build_model("small", z_max=100, in_ch=7, arch="ladder", window=W, iso=True)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ckpt = tmp_path / "iso.pt"
    _save(ckpt, model.state_dict(), TrainConfig(arch="ladder", model_size="small", window=W,
                                                iso_channels=True), 7, None, None,
          "charge_acc_given_ion")
    onnx_path = export_onnx(ckpt, tmp_path / "iso.onnx")
    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    assert (meta["window"], meta["active_top"], meta["iso_channels"], meta["in_ch"],
            meta["iso_count"], meta["view"]) == (W, 24576, True, 7, 3, "window")
    res = verify_parity(ckpt, onnx_path, tol=1e-3)
    assert res["ok"], res


def test_a_network_reading_isotope_channels_needs_the_instrument(tmp_path):
    pytest.importorskip("torch")
    from lcmsdeconv.deconv.classical import make_predictor
    from lcmsdeconv.nn.grid import LogMzGrid
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _save

    model = build_model("small", 100, 6, arch="ladder", window=8192, iso=True)
    ckpt = tmp_path / "iso.pt"
    _save(ckpt, model.state_dict(), TrainConfig(arch="ladder", model_size="small", window=8192,
                                                iso_channels=True), 6, None, None,
          "charge_acc_given_ion")
    grid = LogMzGrid(900.0, 1100.0, 2e-5, -1)
    y = np.zeros(grid.size)
    y[100:110] = 50.0
    pred = ChargePredictor(ckpt)
    assert (pred.iso_channels, pred.window, pred.active_top) == (True, 8192, 1536)
    with pytest.raises(ValueError, match="instrument"):
        pred.predict_grid(grid, y, 1.0)
    orbi = InstrumentModel("orbitrap", 140000.0)
    assert make_predictor(str(ckpt), instrument=orbi).instrument is orbi
    pred.instrument = orbi
    assert pred.predict_grid(grid, y, 1.0).top1_z.shape == (grid.size,)


# ------------------------------------------------------------------ isotope channels in frames
def _v3_orbitrap(crop=W, **kw):
    cfg = _script().mixture_v3(crop)[2][1]
    cfg.singly_charged = "keep"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_training_and_inference_compute_the_same_isotope_channels(tmp_path):
    """The predictor feeds a frame's whole grid to the network exactly as training fed it: the
    same window, base features and isotope channels (charge, strength and isotope limit)."""
    pytest.importorskip("torch")
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _save

    ckpt = tmp_path / "iso.pt"
    _save(ckpt, build_model("small", 100, 7, arch="ladder", window=W, iso=True).state_dict(),
          TrainConfig(arch="ladder", model_size="small", window=W, iso_channels=True), 7, None,
          None, "charge_acc_given_ion")
    pred = ChargePredictor(ckpt)
    seen = []
    run = pred._run
    pred._run = lambda feats: (seen.append(feats), run(feats))[1]
    cfg = _v3_orbitrap(iso_channels=True)
    for seed in (2, 3):
        fs = generate_frame(cfg, np.random.default_rng(seed))
        assert fs.features.shape == (7, W)
        grid, y, sigma = bl.frame_grid(cfg, fs)
        assert grid.size <= W and fs.b0 == 0  # the whole grid in one window
        assert fs.features[4].max() >= 2 and (fs.features[5] > 0.3).sum() > 50
        assert fs.features[6, :grid.size].max() >= 5  # the isotope limit, on every grid bin
        assert not fs.features[:, grid.size:].any()  # the zero-padded tail: no input ...
        assert (fs.topk_z[grid.size:, 0] == 0).all() and not fs.heat[grid.size:].any()  # class 0
        seen.clear()
        pred.instrument = bl.frame_instrument(fs.meta)
        pred.predict_grid(grid, y, sigma)
        assert len(seen) == 1
        np.testing.assert_allclose(seen[0], fs.features, rtol=0, atol=1e-6)


def test_training_and_inference_compute_the_same_comb_view_input(tmp_path):
    """A comb-view network reads at inference what its training frames held: the base features,
    the isotope channels and the measured channel, on the grid's own length (padded to a
    multiple of 64); the frame's tail beyond holds nothing."""
    pytest.importorskip("torch")
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.infer import ChargePredictor, comb_length
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _save

    ckpt = tmp_path / "comb.pt"
    _save(ckpt, build_model("small", 100, 8, arch="ladder", window=W, iso=True,
                            view="comb").state_dict(),
          TrainConfig(arch="ladder", model_size="small", window=W, iso_channels=True,
                      view="comb"), 8, None, None, "charge_acc_given_ion")
    pred = ChargePredictor(ckpt)
    seen = []
    run = pred._run
    pred._run = lambda feats: (seen.append(feats), run(feats))[1]
    cfg = _v3_orbitrap(iso_channels=True, measured_channel=True)
    for seed in (2, 3):
        fs = generate_frame(cfg, np.random.default_rng(seed))
        assert fs.features.shape == (8, W)
        grid, y, sigma = bl.frame_grid(cfg, fs)
        assert grid.size <= W and fs.b0 == 0
        assert (fs.features[7, :grid.size] == 1).all() and not fs.features[7, grid.size:].any()
        seen.clear()
        pred.instrument = bl.frame_instrument(fs.meta)
        pred.predict_grid(grid, y, sigma)
        n = comb_length(grid.size)
        assert len(seen) == 1 and seen[0].shape == (8, n)
        np.testing.assert_allclose(seen[0], fs.features[:, :n], rtol=0, atol=1e-6)


def test_isotope_channels_are_read_on_orbitrap_and_fticr_spectra_only():
    """As the pipeline's comb reads isotope charges: time-of-flight spectra get empty channels
    (there most envelopes' isotope spacing is below what the reader resolves)."""
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.features import isotope_channels

    script = _script()
    tof = script.mixture_v3()[1][1]
    tof.iso_channels = True
    fs = generate_frame(tof, np.random.default_rng(1))
    assert fs.meta["kind"] == "tof" and fs.features.shape[0] == 7 and not fs.features[4:].any()
    grid, y, sigma = bl.frame_grid(tof, fs)
    inst = bl.frame_instrument(fs.meta)
    assert not isotope_channels(grid, y, sigma, inst).any()
    assert isotope_channels(grid, y, sigma, inst, kinds=None)[0].max() > 0  # it could read
    fticr = InstrumentModel("fticr", 400000.0)
    assert isotope_channels(grid, y, sigma, fticr)[0].max() > 0


def test_frames_crop_the_isotope_channels_of_the_whole_grid():
    """A grid longer than the crop: the channels are those of the whole grid, cut to the crop
    (absolute m/z), not computed on the crop alone."""
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.features import isotope_channels

    cfg = SynthConfig(classes=[ClassConfig("ps_rna", (4000.0, 12000.0))], mode="iprp",
                      polarity=-1, instrument_kind="orbitrap",
                      resolution_range=(100000.0, 200000.0), crop_size=8192, iso_channels=True)
    fs = generate_frame(cfg, np.random.default_rng(4))
    grid, y, sigma = bl.frame_grid(cfg, fs)
    assert grid.size > cfg.crop_size
    whole = isotope_channels(grid, y, sigma, bl.frame_instrument(fs.meta))
    seg = whole[:, fs.b0:fs.b0 + cfg.crop_size]
    np.testing.assert_array_equal(fs.features[4:, :seg.shape[1]], seg)
    assert seg[0].max() > 0


# ------------------------------------------------------------------ synthesis options
def test_instrument_scan_range_widens_the_grid_to_one_window():
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.synth.frames import instrument_scan_range

    cfg = SynthConfig(crop_size=W)
    span = 0.98 * W * cfg.grid_step
    rng = np.random.default_rng(0)
    for lo, hi in ((900.0, 1300.0), (700.0, 2600.0), (480.0, 6300.0), (120.0, 9000.0)):
        for _ in range(20):
            a, b = instrument_scan_range(cfg, rng, lo, hi)
            assert a <= lo and b >= hi and b <= cfg.grid_mz_max * 0.995
            if np.log(hi / lo) < span:
                assert np.log(b / a) <= span + 1e-9
                assert a <= 600.0 or a == lo or np.log(b / a) >= span - 1e-9
            else:
                assert (a, b) == (lo, hi)  # the ions alone span more than a window
    # frames: the grid spans the scan range, and fits one window. (With v3's vendor-like
    # threshold the grid spans the points above it instead, which in 90 % of frames start
    # within 1 500 bins of the scan range's low end; the default threshold shows the range.)
    for seed in range(3):
        cfg = _v3_orbitrap(threshold_sigmas_range=(0.0, 3.0))
        fs = generate_frame(cfg, np.random.default_rng(seed))
        lo, hi = fs.meta["scan_range"]
        grid, _y, _s = bl.frame_grid(cfg, fs)
        assert (lo <= 600.0 and hi >= 1500.0) or np.log(hi / lo) >= span - 1e-9
        assert grid.mz_min <= lo * 1.01 and grid.mz_max >= hi * 0.99
        assert grid.size <= W


def test_strong_background_reaches_the_strongest_components():
    def ratios(prob, seed):
        cfg = SynthConfig(classes=[ClassConfig("peptide", (8000.0, 20000.0))], crop_size=8192,
                          background_ions_range=(60, 60), background_strong_prob=prob)
        truths = generate_frame(cfg, np.random.default_rng(seed)).truths
        base = max(t.intensity for t in truths if t.name != "background")
        return np.array([t.intensity / base for t in truths if t.name == "background"])

    weak = np.concatenate([ratios(0.0, s) for s in range(3)])
    strong = np.concatenate([ratios(1.0, s) for s in range(3)])
    assert weak.min() >= 1e-4 * 0.999 and weak.max() <= 3e-2 * 1.001
    assert strong.min() >= 1e-3 * 0.999 and strong.max() <= 3.0 * 1.001
    assert strong.max() > 3e-2 and np.median(strong) > 10 * np.median(weak) / 3


def test_real_like_ion_pairing_oligonucleotides():
    from lcmsdeconv.synth.charge import distribution_from_shape, sample_charge_shape

    rng = np.random.default_rng(0)
    apex, n_states = [], []
    for _ in range(300):
        d = distribution_from_shape(sample_charge_shape(4000.0, "ps_rna", "denatured", rng,
                                                        real_like=True))
        apex.append(max(d, key=d.get))
        n_states.append(sum(v >= 0.05 for v in d.values()))
    assert np.mean(np.array(apex) == 3) > 0.8  # a 4 kDa oligonucleotide: mostly at z = 3
    assert set(n_states) <= {1, 2, 3}
    # the default model puts it higher, with more states
    rng = np.random.default_rng(0)
    d = [distribution_from_shape(sample_charge_shape(4000.0, "ps_rna", "denatured", rng))
         for _ in range(100)]
    assert np.mean([max(x, key=x.get) for x in d]) > 4
    # other classes, and analytes below 900 Da, draw exactly as without the option
    for cls, mass in (("peptide", 4000.0), ("dna", 850.0), ("small_molecule", 400.0)):
        a, b = np.random.default_rng(1), np.random.default_rng(1)
        assert sample_charge_shape(mass, cls, "denatured", a, real_like=True) == \
            sample_charge_shape(mass, cls, "denatured", b)
        assert a.random() == b.random()
    # in frames: every oligonucleotide of a real-like frame, impurities following their parent
    cfg = _v3_orbitrap(8192, oligo_real_like_prob=1.0)
    for seed in range(3):
        fs = generate_frame(cfg, np.random.default_rng(seed))
        assert fs.meta["oligo_real_like"] is True
        for t in fs.truths:
            if t.compound_class in ("dna", "rna", "ps_dna", "ps_rna") and t.mass >= 900:
                assert sum(v >= 0.05 for v in t.charges.values()) <= 3, t.to_dict()
                top = 0.0769 * t.mass ** 0.445 * 1.1 * 1.03  # highest apex, impurities' jitter
                assert max(t.charges, key=t.charges.get) <= top + 1, t.to_dict()


def test_v3_recipe():
    script = _script()
    mix, base = script.mixture_v3(), script.mixture()
    # the Orbitrap component is split in two halves of its weight; the others keep theirs
    assert [w for w, _c in mix] == [0.3, 0.25, 0.125, 0.125, 0.2]
    assert sum(w for w, _c in mix) == pytest.approx(sum(w for w, _c in base))
    for (_w, c), (_w0, c0) in zip(mix, [*base[:3], base[2], base[3]]):
        assert (c.crop_size, c.scan_range, c.background_ions_range, c.background_strong_prob) == \
            (W, "instrument", (0, 200), 0.25)
        assert (c.instrument_kind, c.polarity, c.mode) == (c0.instrument_kind, c0.polarity, c0.mode)
    whole, real = mix[2][1], mix[3][1]
    assert whole is not real and whole.classes is not real.classes
    for orbi in (whole, real):
        assert orbi.instrument_kind == "orbitrap" and orbi.oligo_real_like_prob == 0.5
        assert [(k.name, k.mass_range) for k in orbi.classes] == \
            [(n, (1000.0, 12000.0)) for n in ("dna", "rna", "ps_dna", "ps_rna")]
        assert orbi.threshold_sigmas_range == (3.0, 6.0)  # vendor-like: most bins zero
    assert whole.resolution_range == base[2][1].resolution_range == (35000.0, 280000.0)
    assert real.resolution_range == script.V3_REAL_RESOLUTION == (130000.0, 185000.0)
    for i, (_w, c) in enumerate(mix):
        if i not in (2, 3):  # time-of-flight spectra keep the default threshold and charges
            assert c.oligo_real_like_prob == 0.0 and c.threshold_sigmas_range == (0.0, 3.0)


def test_v3_orbitrap_spectra_are_mostly_zero_like_real_ones():
    """The vendor-like threshold leaves most of an Orbitrap v3 grid at zero (real region grids:
    79-93 %), where the default 0-3 sigmas leave nearly every bin nonzero."""
    from lcmsdeconv.nn import baseline as bl

    zero = {}
    for name, thr in (("v3", None), ("default", (0.0, 3.0))):
        fracs = []
        for seed in range(3):
            cfg = _v3_orbitrap()
            if thr is not None:
                cfg.threshold_sigmas_range = thr
            fs = generate_frame(cfg, np.random.default_rng(200 + seed))
            _grid, y, _sigma = bl.frame_grid(cfg, fs)
            fracs.append(float((y <= 0).mean()))
        zero[name] = np.median(fracs)
    assert zero["v3"] > 0.75 and zero["default"] < 0.4, zero


def test_ladder_window_must_be_a_multiple_of_the_padding():
    """The predictor pads to a multiple of 64 and an exported LadderNet reads exactly its window,
    so any other window would export a model the predictor cannot run."""
    pytest.importorskip("torch")
    from lcmsdeconv.nn.model import build_model

    with pytest.raises(ValueError, match="multiple of 64"):
        build_model("small", 100, 4, arch="ladder", window=20000)
    assert build_model("small", 100, 4, arch="ladder", window=20032).window == 20032


# ------------------------------------------------------------------ training with the options
def test_training_records_the_window_and_isotope_channels(tmp_path):
    torch = pytest.importorskip("torch")
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.nn.train import TrainConfig, _check_same_run, train

    synth = SynthConfig(classes=[ClassConfig("ps_rna", (3000.0, 6000.0))], mode="iprp",
                        polarity=-1, instrument_kind="orbitrap",
                        resolution_range=(100000.0, 150000.0))
    out = tmp_path / "net.pt"
    cfg = dict(arch="ladder", model_size="small", epochs=1, train_len=2, val_len=2, batch_size=2,
               num_workers=0, threads=1, out=str(out), log_every=100, window=16384,
               iso_channels=True)
    model, _h = train(TrainConfig(**cfg), synth_config=synth)
    assert synth.crop_size == 16384 and synth.iso_channels  # the frames follow the network
    assert (model.window, model.active_top, model.iso) == (16384, 3072, True)
    ckpt = torch.load(out, weights_only=False)
    assert (ckpt["window"], ckpt["active_top"], ckpt["iso_channels"], ckpt["in_ch"],
            ckpt["iso_count"], ckpt["view"]) == (16384, 3072, True, 7, 3, "window")
    pred = ChargePredictor(out)
    assert (pred.window, pred.iso_channels) == (16384, True)
    state = torch.load(out.with_suffix(".resume.pt"), weights_only=False)
    assert state["active_top"] == 3072
    with pytest.raises(SystemExit, match="window"):
        train(TrainConfig(resume=True, **{**cfg, "window": 32768}), synth_config=synth)
    with pytest.raises(ValueError, match="LadderNet"):
        train(TrainConfig(**{**cfg, "arch": "unet"}), synth_config=synth)
    # a resume state written before these settings existed is a 32 768-bin network without them
    old = {"arch": "ladder", "model_size": "small", "epochs": 1}
    _check_same_run({**old, "active_top": 6144},
                    {**old, "window": 32768, "iso_channels": False, "active_top": 6144})
    with pytest.raises(SystemExit, match="iso_channels"):
        _check_same_run({**old, "active_top": 6144},
                        {**old, "window": 32768, "iso_channels": True, "active_top": 6144})
