"""Profile spectra that vendor software has thresholded.

Instrument software usually writes only the profile points above a noise threshold, so a scan
is a set of short segments separated by gaps of up to thousands of grid bins. Two operations
must not read a gap as signal:

* integrating a sample over the axis width it represents (:func:`sample_widths`): a segment's
  edge point represents its own spacing, not half of the gap beside it;
* interpolating a scan onto another axis (:func:`profile_on_axis`): between two segments the
  scan holds nothing, not a ramp from one segment's edge to the next.

A gap is a spacing more than ``GAP_FACTOR`` times the local median spacing (a running median
over ``GAP_WINDOW`` spacings), measured in ln(m/z) so that it is the same at every m/z.
"""

from __future__ import annotations

import numpy as np

GAP_FACTOR = 3.0
GAP_WINDOW = 33


def local_spacing(gaps: np.ndarray, window: int = GAP_WINDOW) -> np.ndarray:
    """Running median of consecutive spacings: the typical spacing around each one."""
    from scipy.ndimage import median_filter

    if gaps.size == 0:
        return gaps.copy()
    size = max(1, min(window, gaps.size))
    return median_filter(gaps, size=size, mode="nearest")


def gap_mask(x: np.ndarray, factor: float = GAP_FACTOR) -> np.ndarray:
    """For each pair of neighbours of the sorted positive axis ``x``: is the spacing a gap?"""
    lx = np.log(np.asarray(x, dtype=float))
    g = np.diff(lx)
    return g > factor * local_spacing(g)


def sample_widths(u: np.ndarray, factor: float = GAP_FACTOR) -> np.ndarray:
    """Width along the sorted axis ``u`` that each sample represents.

    Where the axis is continuous this is the midpoint rule, half the distance to each
    neighbour, which for dense data makes a resampled bin the mean of its samples. Toward a gap
    a sample takes the half-width it has on its other side, so a segment's edge point stands
    for its own spacing; a sample with gaps on both sides takes the local median spacing.
    """
    u = np.asarray(u, dtype=float)
    n = u.size
    if n < 2:
        return np.zeros(n)
    g = np.diff(u)
    s = local_spacing(g)
    gap = g > factor * s
    half = np.where(gap, np.nan, 0.5 * g)
    left = np.concatenate([[np.nan], half])
    right = np.concatenate([half, [np.nan]])
    s_pt = np.concatenate([[s[0]], np.minimum(s[:-1], s[1:]), [s[-1]]])
    left_f = np.where(np.isnan(left), np.where(np.isnan(right), 0.5 * s_pt, right), left)
    right_f = np.where(np.isnan(right), np.where(np.isnan(left), 0.5 * s_pt, left), right)
    return left_f + right_f


def profile_on_axis(axis: np.ndarray, mz: np.ndarray, intensity: np.ndarray,
                    factor: float = GAP_FACTOR) -> np.ndarray:
    """A thresholded profile scan read at the m/z positions ``axis`` (sorted).

    Linear interpolation between the scan's own points within each of its segments; zero
    outside its range and inside its gaps.
    """
    axis = np.asarray(axis, dtype=float)
    mz = np.asarray(mz, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if mz.size == 0:
        return np.zeros(axis.size)
    if np.any(np.diff(mz) < 0):
        order = np.argsort(mz, kind="stable")
        mz, y = mz[order], y[order]
    v = np.interp(axis, mz, y, left=0.0, right=0.0)
    if mz.size < 2:
        return v
    gaps = gap_mask(mz, factor)
    if not gaps.any():
        return v
    k = np.searchsorted(mz, axis, side="right") - 1
    inside = (k >= 0) & (k < mz.size - 1)
    in_gap = np.zeros(axis.size, dtype=bool)
    ki = k[inside]
    in_gap[inside] = gaps[ki] & (axis[inside] > mz[ki])
    v[in_gap] = 0.0
    return v
