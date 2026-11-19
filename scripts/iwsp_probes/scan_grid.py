"""One MS1 scan on the u grid, without the gap-edge amplification of ``resample_profile``.

Vendor software writes thresholded profiles: only the points near peaks survive, so a scan is a
set of short segments separated by gaps of up to thousands of bins. ``LogMzGrid.resample_profile``
gives every sample the width 0.5 (u[i+1] - u[i-1]); a segment's edge point then represents half
the gap beside it and lands in one bin multiplied by up to that many bins (2 938 in scan 379 of
the IP-RPLC GalNAc run). ``resample_scan`` gives each sample the smaller of its two neighbour
spacings, at most three times the scan's median spacing, so a segment's edges weigh what its
interior points do and an isolated point weighs at most three ordinary ones. For dense data both
rules agree.

The probes resample every scan with this function. The package's pipeline still uses
``resample_profile`` on its region sums (docs/iwsp_transformer.md, section 2).
"""

from __future__ import annotations

import numpy as np


def sample_widths(u: np.ndarray, cap_spacings: float = 3.0) -> np.ndarray:
    """Width in u of each sample of a sorted axis: min(left gap, right gap), capped."""
    if u.size < 2:
        return np.zeros_like(u)
    gaps = np.diff(u)
    left = np.concatenate([[gaps[0]], gaps])
    right = np.concatenate([gaps, [gaps[-1]]])
    return np.minimum(np.minimum(left, right), cap_spacings * float(np.median(gaps)))


def resample_scan(grid, mz, intensity, cap_spacings: float = 3.0) -> np.ndarray:
    """Integrate a scan's samples onto ``grid`` with widths from :func:`sample_widths`."""
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    out = np.zeros(grid.size, dtype=np.float64)
    m = (mz > grid.mz_min) & (mz < grid.mz_max)
    if not np.any(m):
        return out
    mz, intensity = mz[m], intensity[m]
    if mz.size == 1:
        out[int(np.clip(grid.mz_to_bin(mz[0]), 0, grid.size - 1))] = intensity[0]
        return out
    du = sample_widths(grid.mz_to_u(mz), cap_spacings)
    b = np.clip(grid.mz_to_bin(mz), 0, grid.size - 1)
    np.add.at(out, b, intensity * du / grid.step)
    return out
