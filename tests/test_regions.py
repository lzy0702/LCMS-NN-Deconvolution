import numpy as np
import pytest

from lcmsdeconv.core.method import IntegrationSettings, Method
from lcmsdeconv.core.model import Run, Spectrum
from lcmsdeconv.features.regions import find_regions


def _run(small_height: float) -> Run:
    """A TIC of one tall peak (3 min) and one small one (7 min) on a noisy baseline."""
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 10.0, 0.02)
    y = (1.0e4 + rng.normal(0.0, 100.0, t.size)
         + 1.0e6 * np.exp(-0.5 * ((t - 3.0) / 0.06) ** 2)
         + small_height * np.exp(-0.5 * ((t - 7.0) / 0.06) ** 2))
    return Run([Spectrum(np.array([500.0]), np.array([v]), float(rt), polarity=-1)
                for rt, v in zip(t, y)])


def _covers(regions, rt):
    return any(r.start <= rt <= r.end for r in regions)


@pytest.mark.parametrize("rel, kept", [(0.002, False), (0.001, True)])
def test_relative_height_floor(rel, kept):
    # 0.15 % of the tallest peak and 15 times the noise: the 3x noise floor keeps it, a 0.2 %
    # height floor does not
    run = _run(1.5e3)
    regions = find_regions(run, "tic", -1, integration=IntegrationSettings(peak_width=0.2),
                           min_snr=3.0, min_rel_height=rel)
    assert _covers(regions, 3.0)
    assert _covers(regions, 7.0) is kept


def test_noise_floor_still_decides_below_relative_floor():
    # 0.15 % of the tallest but under three times the noise: no region at either floor
    run = _run(2.0e2)
    for rel in (0.002, 0.001):
        regions = find_regions(run, "tic", -1, integration=IntegrationSettings(peak_width=0.2),
                               min_snr=3.0, min_rel_height=rel)
        assert not _covers(regions, 7.0)


def test_method_relative_height_floor():
    for name in ("orbitrap_iprp_neg_oligo", "iprp_neg_oligo"):
        assert Method.load(name).regions.min_rel_height == pytest.approx(0.001)
    for name in ("rplc_pos_protein", "polymer_pos", "hilic_glycan"):
        assert Method.load(name).regions.min_rel_height == pytest.approx(0.002)
    assert Method().regions.min_rel_height == pytest.approx(0.002)


def test_process_run_hands_the_methods_floors_to_find_regions(monkeypatch):
    from .test_region_pieces import _process

    method = Method.load("orbitrap_iprp_neg_oligo")
    method.regions.min_rel_height, method.regions.min_snr = 0.0037, 4.5
    _result, calls = _process(monkeypatch, _run(1.5e3), method)
    ((args, kw),) = calls["find_regions"]
    assert kw["min_rel_height"] == 0.0037 and kw["min_snr"] == 4.5
    assert args[1:] == ("tic", -1, method.regions.window_min, method.regions.margin_min,
                        method.regions.min_frames)
