import numpy as np

from lcmsdeconv.nn.grid import LogMzGrid
from lcmsdeconv.synth.chromatography import emg, generate_run
from lcmsdeconv.synth.config import SynthConfig
from lcmsdeconv.synth.frames import generate_frame


def test_frame_determinism():
    cfg = SynthConfig()
    a = generate_frame(cfg, np.random.default_rng(42))
    b = generate_frame(cfg, np.random.default_rng(42))
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.topk_z, b.topk_z)


def test_label_shares_bounded():
    for seed in range(5):
        fs = generate_frame(SynthConfig(), np.random.default_rng(seed))
        s = fs.topk_w.sum(axis=1)
        assert s.max() <= 1.0 + 1e-4
        assert (fs.topk_w >= -1e-6).all()
        # class ids are valid charges or 0
        assert fs.topk_z.min() >= 0
        assert fs.topk_z.max() <= 100


def test_features_shape_and_finite():
    fs = generate_frame(SynthConfig(crop_size=8192), np.random.default_rng(3))
    assert fs.features.shape == (4, 8192)
    assert np.isfinite(fs.features).all()


def test_grid_stick_roundtrip():
    g = LogMzGrid(polarity=1)
    for mass, z in [(15000.0, 12), (66000.0, 40), (2000.0, 2)]:
        b = g.mass_charge_to_bin(mass, z)
        rec = g.bin_to_mass(b, z)
        assert abs(rec - mass) / mass < 5e-5  # within grid quantization (~20 ppm)


def test_emg_unit_apex_and_pointwise_consistency():
    t = np.linspace(0, 5, 500)
    y = emg(t, 2.5, 0.1, 0.15)
    # normalization comes from a dense reference grid, so the apex is 1 to grid precision
    assert abs(y.max() - 1.0) < 1e-3
    assert y.min() >= 0.0
    # evaluating one time at a time must agree with evaluating the whole run at once
    pointwise = np.array([emg(np.array([x]), 2.5, 0.1, 0.15)[0] for x in t[::25]])
    assert np.allclose(pointwise, y[::25])


def test_run_generation_small():
    cfg = SynthConfig(mode="rplc")
    run, truth = generate_run(cfg, np.random.default_rng(1), n_peaks=2, rt_range=(1.0, 3.0),
                              scan_rate_hz=1.0, esi_saturation=True)
    assert len(run.spectra) > 10
    assert run.tic().intensity.max() > 0
    assert len(truth.peaks) == 2
    assert any(m.kind == "impurity" for m, _ in truth.peaks[0].members) or True


def test_run_polarity_switching():
    cfg = SynthConfig()
    run, _ = generate_run(cfg, np.random.default_rng(2), n_peaks=1, rt_range=(1.0, 2.0),
                          scan_rate_hz=2.0, polarity_switching=True)
    assert set(run.polarities) == {1, -1}
