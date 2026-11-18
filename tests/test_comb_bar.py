"""The bar a charge network is measured against: the comb as the pipeline runs it."""

import numpy as np
import pytest


@pytest.mark.parametrize("kind", ["tof", "orbitrap"])
def test_bar_is_the_pipeline_comb_over_the_whole_spectrum(kind):
    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.deconv.classical import CombPredictor
    from lcmsdeconv.nn.baseline import comb_charges, frame_grid
    from lcmsdeconv.synth.compounds import ClassConfig
    from lcmsdeconv.synth.config import SynthConfig
    from lcmsdeconv.synth.frames import generate_frame

    cfg = SynthConfig(classes=[ClassConfig("ps_dna", (3000.0, 9000.0))], mode="iprp",
                      polarity=-1, instrument_kind=kind, crop_size=8192)
    fs = generate_frame(cfg, np.random.default_rng(3))
    z, how = comb_charges(cfg, fs)
    assert how == "pipeline"
    grid, y, noise = frame_grid(cfg, fs)
    seg = slice(fs.b0, min(fs.b0 + cfg.crop_size, y.size))
    n = seg.stop - seg.start
    # the whole spectrum is the one the frame's features were cut from
    ch0 = np.clip(np.log10(1.0 + np.clip(y[seg], 0, None) / noise), 0.0, 5.0)
    assert np.allclose(fs.features[0][:n], ch0, atol=1e-5)
    # and the bar is CombPredictor over all of it, isotope charges on for Orbitrap spectra
    inst = InstrumentModel(kind, fs.meta["resolution"]) if kind == "orbitrap" else None
    want = CombPredictor(z_max=60, instrument=inst).predict_grid(grid, y, noise).top1_z[seg]
    assert np.array_equal(z[:n], want)
    assert y.size > cfg.crop_size  # a crop alone would have hidden part of the spectrum


def test_scoreboard_counts_every_spectrum_once_and_no_answer_as_wrong():
    from lcmsdeconv.nn.checks import ChargeScoreboard

    board = ChargeScoreboard()
    f0 = np.array([0.0, 2.0, 2.0, 4.0])  # SNR 0, 99, 99, 9999
    true = np.array([0, 3, 3, 3])
    board.add("tof", "a", np.array([0, 3, 0, 5]), true, f0)  # right on one of 3 ion bins
    board.add("tof", "a", np.array([0, 3, 3, 3]), true, 2 * f0)  # a far stronger spectrum, all right
    log_w = (2.0 / 8.0 + 1.0) / 2.0  # each spectrum's weights sum to one
    assert board.accuracy("tof", "a", "log-SNR") == pytest.approx(log_w)
    assert board.accuracy("tof", "a", "intensity") == pytest.approx((99 / 10197 + 1.0) / 2.0)
    pooled = (99 + 2 * 9999 + 1e8 - 1) / (10197 + 2 * 9999 + 1e8 - 1)  # the strong one decides
    assert board.accuracy("tof", "a", "intensity, pooled") == pytest.approx(pooled)
    assert board.accuracy("tof", "a", "log-SNR", "z=1") is None
    assert board.summary()["tof"]["a"]["log-SNR"]["z=2-5"] == pytest.approx(log_w)
    # every band counts each spectrum once: a spectrum mostly singly charged weighs as much
    # at z >= 2 as one with nothing but multiply charged ions
    board = ChargeScoreboard()
    board.add("tof", "b", np.array([1, 1, 5]), np.array([1, 1, 3]), np.full(3, 2.0))
    board.add("tof", "b", np.array([3]), np.array([3]), np.full(1, 2.0))
    assert board.accuracy("tof", "b", "log-SNR", "z>=2") == pytest.approx(0.5)
    assert board.accuracy("tof", "b", "log-SNR", "z=1") == pytest.approx(1.0)
