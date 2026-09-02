import numpy as np

from lcmsdeconv.core.model import Chromatogram, Peak, Spectrum
from lcmsdeconv.quant.purity import fit_calibration, ms_purity, potency
from lcmsdeconv.quant.saturation import (
    detect_detector_saturation,
    detect_esi_saturation,
    estimate_uv_delay,
)


def test_calibration_and_potency():
    cal = fit_calibration([[0.1, 1000], [0.5, 5000], [1.0, 10000], [2.0, 20000]])
    assert abs(cal.slope - 10000) / 10000 < 0.02
    assert cal.r_squared > 0.999
    res = potency(10000, cal, sample_amount=1.0)
    assert abs(res["percent_of_nominal"] - 100.0) < 2.0


def test_weighted_calibration_favours_low_levels():
    levels = [[0.01, 105], [0.1, 1000], [1.0, 9900], [10.0, 101000]]
    plain = fit_calibration(levels)
    weighted = fit_calibration(levels, weighting="1/x2")
    assert abs(weighted.amount(105) - 0.01) < abs(plain.amount(105) - 0.01)


def test_detector_saturation_flags_clipped_frames():
    mz = np.linspace(500, 600, 400)
    frames = []
    for i in range(20):
        y = 1000 * np.exp(-0.5 * ((mz - 550) / 0.5) ** 2)
        if 8 <= i <= 12:
            y = np.minimum(y * 5, 1000.0)  # clipped ceiling
        frames.append(Spectrum(mz, y, rt=0.1 * i, polarity=1, is_profile=True))
    flagged, level = detect_detector_saturation(frames, flat_top_points=3)
    assert len(flagged) >= 3


def test_uv_delay_estimation():
    t = np.linspace(0, 10, 1000)
    peak = np.exp(-0.5 * ((t - 5) / 0.1) ** 2)
    tic = Chromatogram(t, peak, "TIC", "tic")
    uv = Chromatogram(t, np.exp(-0.5 * ((t - 5.2) / 0.1) ** 2), "UV", "uv")
    assert abs(estimate_uv_delay(tic, uv) - 0.2) < 0.05


def test_esi_saturation_detected_against_uv():
    t = np.linspace(4, 6, 400)
    true = np.exp(-0.5 * ((t - 5) / 0.08) ** 2)
    uv = Chromatogram(t, true * 100, "UV", "uv")
    compressed = true / (1 + 3 * true)  # ionisation-limited response
    tic = Chromatogram(t, compressed * 1e6, "TIC", "tic")
    peak = Peak(rt=5.0, start=4.6, end=5.4, area=1.0, height=1.0, id=1)
    flags, delay, msgs = detect_esi_saturation(tic, uv, [peak], ratio_drop=0.2)
    assert 1 in flags
    assert msgs


def test_no_false_esi_saturation_on_linear_response():
    t = np.linspace(4, 6, 400)
    true = np.exp(-0.5 * ((t - 5) / 0.08) ** 2)
    uv = Chromatogram(t, true * 100, "UV", "uv")
    tic = Chromatogram(t, true * 1e6, "TIC", "tic")
    peak = Peak(rt=5.0, start=4.6, end=5.4, area=1.0, height=1.0, id=1)
    flags, _, _ = detect_esi_saturation(tic, uv, [peak], ratio_drop=0.25)
    assert 1 not in flags


def test_ms_purity():
    from lcmsdeconv.core.model import Species

    t = np.linspace(0, 1, 10)
    a = Species(0, 10000.0, "average", 1, "peptide", t, np.ones(10) * 100)
    b = Species(1, 10016.0, "average", 1, "peptide", t, np.ones(10) * 2)
    res = ms_purity([a, b])
    assert 97 < res["main_percent"] < 99
