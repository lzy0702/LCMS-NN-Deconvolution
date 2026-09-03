import numpy as np
import pytest

from lcmsdeconv.nn.features import comb_shifts, featurize
from lcmsdeconv.nn.grid import LogMzGrid

torch = pytest.importorskip("torch")


def test_grid_charge_spacing_is_mass_independent():
    g = LogMzGrid(polarity=1)
    for mass in (5000.0, 50000.0, 150000.0):
        u10 = g.mass_charge_to_u(mass, 10)
        u11 = g.mass_charge_to_u(mass, 11)
        assert abs((u10 - u11) - np.log(11 / 10)) < 1e-12


def test_grid_resample_preserves_shape():
    from lcmsdeconv.chem.instrument import InstrumentModel

    inst = InstrumentModel("tof", 30000)
    g = LogMzGrid(polarity=1)
    ax = inst.profile_axis(900, 1100)
    y = 1000 * np.exp(-0.5 * ((ax - 1000.0) / inst.sigma_mz(1000.0)) ** 2)
    grid_y = g.resample_profile(ax, y)
    apex_bin = int(np.argmax(grid_y))
    assert abs(g.bin_to_mz(apex_bin) - 1000.0) < 0.05
    # width on the grid should match the instrument line width (within a bin)
    above = np.count_nonzero(grid_y > 0.5 * grid_y.max())
    expected = (1000.0 / 30000) / (1000.0 * g.step)
    assert 0.5 * expected < above < 2.5 * expected


def test_featurize_channels():
    f = featurize(np.abs(np.random.default_rng(0).normal(0, 5, 4096)), 5.0, 2e-5)
    assert f.shape == (4, 4096)
    assert np.isfinite(f).all()
    assert f[0].min() >= 0


def test_comb_shifts_sign():
    s = comb_shifts(2e-5, 10)
    assert s[5][0] < 0  # shift towards z+1 is negative in u


def test_model_forward_and_export(tmp_path):
    from lcmsdeconv.nn.export import export_onnx, verify_parity
    from lcmsdeconv.nn.model import build_model

    m = build_model("small", z_max=20)
    x = torch.randn(1, 4, 2048)
    out = m(x)
    assert out["charge_logits"].shape == (1, 21, 2048)
    assert out["apex_logit"].shape == (1, 1, 2048)
    ckpt = tmp_path / "m.pt"
    torch.save({"state_dict": m.state_dict(), "model_size": "small", "z_max": 20}, ckpt)
    onnx_path = export_onnx(ckpt, tmp_path / "m.onnx", length=2048)
    res = verify_parity(ckpt, onnx_path, length=2048, tol=1e-3)
    assert res["ok"], res


def test_predictor_tiles_full_grid(tmp_path):
    from lcmsdeconv.nn.export import export_onnx
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.nn.model import build_model

    m = build_model("small", z_max=20)
    ckpt = tmp_path / "m.pt"
    torch.save({"state_dict": m.state_dict(), "model_size": "small", "z_max": 20}, ckpt)
    onnx_path = export_onnx(ckpt, tmp_path / "m.onnx", length=4096)
    g = LogMzGrid(50.0, 1000.0, 2e-4, 1)  # small grid for speed
    inten = np.abs(np.random.default_rng(1).normal(0, 3, g.size))
    pred = ChargePredictor(onnx_path, window=4096).predict_grid(g, inten, 3.0)
    assert pred.top1_z.shape == (g.size,)
    assert pred.apex.min() >= 0 and pred.apex.max() <= 1
