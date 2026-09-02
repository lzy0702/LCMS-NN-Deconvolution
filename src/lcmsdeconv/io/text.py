"""Two-column text spectra (UniDec-style) and simple spectrum text export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.model import Spectrum


def read_text_spectrum(path: str | Path, rt: float = 0.0, polarity: int = 1) -> Spectrum:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mz = arr[:, 0].astype(np.float64)
    it = arr[:, 1].astype(np.float64)
    order = np.argsort(mz)
    mz, it = mz[order], it[order]
    is_prof = bool(np.mean(it <= 0) > 0.05)
    return Spectrum(mz, it, rt, polarity, 1, Path(path).stem, 0, is_prof)


def write_text_spectrum(spectrum: Spectrum, path: str | Path) -> Path:
    path = Path(path)
    np.savetxt(path, np.column_stack([spectrum.mz, spectrum.intensity]))
    return path
