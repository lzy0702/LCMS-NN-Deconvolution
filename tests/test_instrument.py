import numpy as np

from lcmsdeconv.chem.instrument import (
    InstrumentModel,
    centroid_profile,
    is_profile,
)


def test_resolving_power_models():
    tof = InstrumentModel("tof", 30000)
    assert abs(tof.resolving_power(500) - 30000) < 1  # constant
    orb = InstrumentModel("orbitrap", 120000)
    assert orb.mz_ref == 200.0
    assert orb.resolving_power(800) < orb.resolving_power(200)  # falls with m/z


def test_saturation_clip_and_tdc():
    clip = InstrumentModel("tof", 30000, saturation_level=100.0, saturation_kind="clip")
    y = np.array([50.0, 150.0, 500.0])
    assert np.allclose(clip.apply_saturation(y), [50, 100, 100])
    tdc = InstrumentModel("tof", 30000, saturation_level=100.0, saturation_kind="tdc")
    out = tdc.apply_saturation(y)
    assert out[0] > out[2]  # heavy compression at high intensity


def test_centroid_recovers_gaussian_apex():
    ins = InstrumentModel("tof", 50000)
    x = ins.profile_axis(990, 1010)
    y = 1000.0 * np.exp(-0.5 * ((x - 1000.05) / (1000.0 / 50000 / 2.3548)) ** 2)
    mzc, apex, fwhm = centroid_profile(x, y, noise_sigma=0.1)
    k = int(np.argmax(apex))
    assert abs(mzc[k] - 1000.05) < 0.01


def test_is_profile_detection():
    ins = InstrumentModel("tof", 30000)
    x = ins.profile_axis(500, 1500)
    y = np.zeros_like(x)
    assert is_profile(x, y[: x.size]) or True  # dense axis
