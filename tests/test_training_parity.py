"""Training frames must be computed exactly as inference computes its input."""

import numpy as np

from lcmsdeconv.nn.features import featurize
from lcmsdeconv.nn.grid import spectrum_to_grid
from lcmsdeconv.synth.compounds import ClassConfig
from lcmsdeconv.synth.config import SynthConfig
from lcmsdeconv.synth.frames import generate_frame


def test_training_features_equal_inference_features():
    """Same summed spectrum, same grid, same noise estimate, same features."""
    for polarity, classes, mode in ((1, [ClassConfig("peptide", (8000.0, 30000.0))], "rplc"),
                                    (-1, [ClassConfig("ps_rna", (4000.0, 12000.0))], "iprp")):
        cfg = SynthConfig(classes=classes, mode=mode, polarity=polarity, crop_size=8192)
        for seed in (0, 1):
            fs = generate_frame(cfg, np.random.default_rng(seed))
            mz, inten = fs.meta["raw"]
            grid, observed, sigma = spectrum_to_grid(mz, inten, polarity, cfg.grid_step,
                                                     cfg.grid_mz_max)
            assert sigma == fs.meta["noise_sigma"]
            window = np.zeros(cfg.crop_size)
            seg = observed[fs.b0:fs.b0 + cfg.crop_size]
            window[:seg.size] = seg
            feats = featurize(window, sigma, cfg.grid_step, z_max=min(cfg.z_max, 64))
            np.testing.assert_allclose(feats, fs.features, rtol=0, atol=1e-6)


def test_training_labels_are_shares_of_observed_intensity():
    cfg = SynthConfig(classes=[ClassConfig("peptide", (8000.0, 30000.0))], crop_size=8192)
    fs = generate_frame(cfg, np.random.default_rng(3))
    total = fs.topk_w.sum(axis=1)
    assert total.max() <= 1.0 + 1e-5
    ion = fs.topk_z[:, 0] > 0
    assert ion.any()
    # where the log-SNR feature is high the bin is dominated by an ion, not by noise
    strong = fs.features[0] > 2.0
    assert (fs.topk_z[strong, 0] > 0).mean() > 0.9


def test_comb_does_not_wrap_window_edges():
    """Signal at one end of a window must not create comb evidence at the other end."""
    step = 2e-5
    w = np.zeros(4096)
    w[10:20] = 1000.0  # a peak near the left edge only
    feats = featurize(w, 1.0, step, z_max=64)
    assert feats[2, -500:].max() == 0.0
