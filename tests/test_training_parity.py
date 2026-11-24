"""Training frames must be computed exactly as inference computes its input."""

import numpy as np

from lcmsdeconv.nn.features import featurize
from lcmsdeconv.nn.grid import spectrum_to_grid
from lcmsdeconv.synth.compounds import ClassConfig
from lcmsdeconv.synth.config import SynthConfig
from lcmsdeconv.synth.frames import generate_frame


def test_training_features_equal_inference_features():
    """The frame's scans summed as a region's are and gridded as deconvolve_spectrum grids them:
    same grid, same noise estimate, same features; with isotope-charge channels, the same
    channels too (charge, strength and isotope limit: computed over the whole grid with the
    instrument, then cut to the window; on Orbitrap spectra, empty on time-of-flight ones)."""
    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.core.model import Run
    from lcmsdeconv.nn.features import isotope_channels
    from lcmsdeconv.synth.frames import scan_spectra

    for polarity, classes, mode, kind in (
            (1, [ClassConfig("peptide", (8000.0, 30000.0))], "rplc", "tof"),
            (-1, [ClassConfig("ps_rna", (4000.0, 12000.0))], "iprp", "tof"),
            (-1, [ClassConfig("ps_rna", (2000.0, 8000.0))], "iprp", "orbitrap")):
        for iso in (False, True):
            if kind == "orbitrap" and not iso:
                continue
            cfg = SynthConfig(classes=classes, mode=mode, polarity=polarity, crop_size=8192,
                              instrument_kind=kind, iso_channels=iso,
                              resolution_range=(8000.0, 60000.0) if kind == "tof"
                              else (60000.0, 240000.0))
            for seed in (0, 1):
                fs = generate_frame(cfg, np.random.default_rng(seed))
                scans = [s for s in scan_spectra(fs.meta["scans"], polarity) if s.mz.size]
                summed = Run(scans).sum_frames(scans)
                grid, observed, sigma = spectrum_to_grid(summed.mz, summed.intensity, polarity,
                                                         cfg.grid_step, cfg.grid_mz_max, True,
                                                         frames=summed.meta["frames"])
                assert np.isclose(sigma, fs.meta["noise_sigma"], rtol=1e-9)
                window = np.zeros(cfg.crop_size)
                seg = observed[fs.b0:fs.b0 + cfg.crop_size]
                window[:seg.size] = seg
                feats = featurize(window, sigma, cfg.grid_step, z_max=min(cfg.z_max, 64))
                if iso:
                    inst = InstrumentModel(fs.meta["kind"], fs.meta["resolution"])
                    whole = isotope_channels(grid, observed, sigma, inst)
                    part = np.zeros((whole.shape[0], cfg.crop_size), dtype=np.float32)
                    cut = whole[:, fs.b0:fs.b0 + cfg.crop_size]
                    part[:, :cut.shape[1]] = cut
                    feats = np.concatenate([feats, part])
                    if kind == "orbitrap":
                        assert part[0].max() > 0  # charges are read in the window
                    else:
                        assert not part.any()
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


def test_training_scans_are_thresholded_into_segments_of_five_points_or_more():
    """As in the real Thermo runs, no thresholded training scan holds a segment shorter than
    five points (two points either side of each point above the threshold)."""
    from lcmsdeconv.core.profile import gap_mask

    cfg = SynthConfig(classes=[ClassConfig("ps_rna", (4000.0, 12000.0))], mode="iprp",
                      polarity=-1, crop_size=8192)
    fs = generate_frame(cfg, np.random.default_rng(4))
    short = total = 0
    for mz, _y in fs.meta["scans"]:
        if mz.size < 2:
            continue
        cuts = np.flatnonzero(gap_mask(mz)) + 1
        lengths = np.diff(np.concatenate([[0], cuts, [mz.size]]))
        inner = lengths[1:-1]  # the first and last can be cut by the axis ends
        short += int(np.sum(inner < 5))
        total += inner.size
    assert total > 100 and short == 0


def test_the_resume_fingerprint_covers_the_training_threshold():
    """A run paused under one thresholding must not resume on frames thresholded another way."""
    from lcmsdeconv.nn.train import synth_fingerprint

    a = SynthConfig(classes=[ClassConfig("ps_rna", (4000.0, 12000.0))], mode="iprp", polarity=-1)
    b = SynthConfig(classes=[ClassConfig("ps_rna", (4000.0, 12000.0))], mode="iprp", polarity=-1,
                    vendor_pad=3)
    assert synth_fingerprint(a) != synth_fingerprint(b)
