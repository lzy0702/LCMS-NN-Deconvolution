"""The comb estimator as the pipeline runs it, on synthetic frames: the bar a network must clear.

A charge network is worth using only where it reads charges better than the comb does in the
pipeline. That comb reads the whole spectrum - a z = 2 ion's z = 1 neighbour lies 34 657 bins
away on the default grid, beyond any 32 768-bin crop - and on Orbitrap and FT-ICR spectra it
also reads charges from resolved isotope spacing (``process._isotope_mode``). The comb's answer
in feature channel 3 does neither: it is computed inside the crop, so it is a convenient input
feature but no fair bar. These helpers rebuild a frame's whole spectrum from its ``meta["raw"]``
and run any predictor over it exactly as ``process_run`` would, then cut its answer to the
frame's labelled crop.
"""

from __future__ import annotations

import numpy as np

#: instrument kinds whose spectra the pipeline reads with isotope-spacing charges by default
ISOTOPE_KINDS = ("orbitrap", "fticr")
#: display names of the two kinds of spectra the comparisons keep apart
KIND_NAMES = {"tof": "time-of-flight", "orbitrap": "Orbitrap"}


def spectrum_kind(meta: dict) -> str:
    """``"orbitrap"`` for Orbitrap and FT-ICR spectra, ``"tof"`` for the rest."""
    return "orbitrap" if meta.get("kind") in ISOTOPE_KINDS else "tof"


def frame_grid(cfg, fs):
    """``(grid, intensity, noise sigma)`` of the whole spectrum frame ``fs`` was cut from, as
    ``spectrum_to_grid`` builds it at inference; None for a frame rendered onto its crop alone."""
    raw = fs.meta.get("raw")
    if raw is None:
        return None
    from .grid import spectrum_to_grid

    return spectrum_to_grid(raw[0], raw[1], cfg.polarity, cfg.grid_step, cfg.grid_mz_max, True)


def pipeline_comb(meta: dict, z_max: int = 60, isotope_charges: bool | None = None):
    """The CombPredictor ``process_run`` builds for a spectrum of this instrument: isotope-spacing
    charges on Orbitrap and FT-ICR spectra unless ``isotope_charges`` says otherwise. It is given
    the frame's true resolving power, which favours it slightly over a method's nominal one."""
    from ..chem.instrument import InstrumentModel
    from ..deconv.classical import CombPredictor

    iso = spectrum_kind(meta) == "orbitrap" if isotope_charges is None else isotope_charges
    inst = InstrumentModel(meta["kind"], float(meta["resolution"])) if iso else None
    return CombPredictor(z_max=z_max, instrument=inst)


def charges_on_crop(predictor, cfg, fs, grid=None) -> np.ndarray | None:
    """``predictor``'s charge per bin over the frame's whole spectrum, cut to the frame's crop
    (0 where it gives none); None for a frame without a whole spectrum. ``grid`` is
    :func:`frame_grid`'s result when the caller already has it."""
    g = grid if grid is not None else frame_grid(cfg, fs)
    if g is None:
        return None
    full, intensity, noise = g
    top = predictor.predict_grid(full, intensity, noise).top1_z
    length = fs.features.shape[1]
    out = np.zeros(length, dtype=np.int64)
    seg = top[fs.b0:fs.b0 + length]
    out[:seg.size] = seg
    return out


def comb_charges(cfg, fs) -> tuple[np.ndarray, str]:
    """The pipeline comb's charges on the frame's crop and ``"pipeline"``; for a frame rendered
    onto its crop alone, the comb's answer in feature channel 3 and ``"channel 3"``."""
    z = charges_on_crop(pipeline_comb(fs.meta), cfg, fs)
    if z is not None:
        return z, "pipeline"
    zc = np.round(fs.features[3] * min(int(cfg.z_max), 64)).astype(np.int64)
    return zc, "channel 3"
