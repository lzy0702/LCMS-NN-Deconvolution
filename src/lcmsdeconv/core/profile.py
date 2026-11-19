"""Profile spectra that vendor software has thresholded.

Instrument software usually writes only the profile points above a noise threshold, so a scan
is a set of short segments separated by gaps of up to thousands of grid bins. Two operations
must not read a gap as signal:

* integrating a sample over the axis width it represents (:func:`sample_widths`): a segment's
  edge point represents its own spacing, not half of the gap beside it;
* interpolating a scan onto another axis (:func:`profile_on_axis`): between two segments the
  scan holds nothing, not a ramp from one segment's edge to the next.

A gap is a spacing more than ``GAP_FACTOR`` times the scan's own spacing there, measured in
ln(m/z) so that it is the same at every m/z. The own spacing is a low percentile
(``SPACING_PERCENTILE``) of the spacings in a window of ``GAP_WINDOW`` around each: where noise
or weak ions leave many one- and two-point segments, gaps are most of the spacings, and a median
would take them for the spacing.
"""

from __future__ import annotations

import numpy as np

GAP_FACTOR = 3.0
GAP_WINDOW = 65
SPACING_PERCENTILE = 20.0
#: a scan's own spacing changes with m/z by at most this factor about its typical value (the
#: spacing of Orbitrap and time-of-flight profiles goes as a power of m/z: a factor of about
#: 2.5 over m/z 400-2500), so a stretch where every spacing is wider holds isolated points
SPACING_SPREAD = 3.0


def local_spacing(gaps: np.ndarray, window: int = GAP_WINDOW) -> np.ndarray:
    """The scan's own spacing around each of its consecutive spacings ``gaps``.

    The low percentile of each block of ``window`` consecutive spacings, interpolated between
    block centres (held at the ends): the first and last spacing are judged against the
    spacings inside the scan, and the whole scan costs one vectorized percentile. A block of
    nothing but isolated points shows only gaps; its spacing is capped at ``SPACING_SPREAD``
    times the median block's.
    """
    g = np.asarray(gaps, dtype=float)
    n = g.size
    if n == 0:
        return g.copy()
    w = max(1, min(window, n))
    nb = n // w
    vals = list(np.percentile(g[: nb * w].reshape(nb, w), SPACING_PERCENTILE, axis=1))
    centres = list(np.arange(nb) * w + 0.5 * (w - 1))
    if n > nb * w:  # the remainder, judged with the spacings just before it
        vals.append(float(np.percentile(g[n - w:], SPACING_PERCENTILE)))
        centres.append(n - 0.5 * (w + 1))
    vals = np.asarray(vals)
    vals = np.minimum(vals, SPACING_SPREAD * float(np.median(vals)))
    return np.interp(np.arange(n, dtype=float), np.asarray(centres), vals)


def gap_mask(x: np.ndarray, factor: float = GAP_FACTOR) -> np.ndarray:
    """For each pair of neighbours of the sorted positive axis ``x``: is the spacing a gap?"""
    lx = np.log(np.asarray(x, dtype=float))
    g = np.diff(lx)
    return g > factor * local_spacing(g)


def scan_spacing(mz: np.ndarray, lo: float, hi: float, min_points: int = 4) -> float | None:
    """A scan's own point spacing (m/z) within [lo, hi]: the low percentile of its spacings
    there, or None when it has fewer than ``min_points`` points in the interval."""
    mz = np.asarray(mz, dtype=float)
    a, b = int(np.searchsorted(mz, lo)), int(np.searchsorted(mz, hi))
    if b - a < min_points:
        return None
    return float(np.percentile(np.diff(mz[a:b]), SPACING_PERCENTILE))


def sample_widths(u: np.ndarray, factor: float = GAP_FACTOR) -> np.ndarray:
    """Width along the sorted axis ``u`` that each sample represents.

    Where the axis is continuous this is the midpoint rule, half the distance to each
    neighbour, which for dense data makes a resampled bin the mean of its samples. Toward a gap
    a sample takes the half-width it has on its other side, so a segment's edge point stands
    for its own spacing; a sample with gaps on both sides takes the scan's own spacing there
    (:func:`local_spacing`).
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


def profiles_on_union(scans, axis: np.ndarray, factor: float = GAP_FACTOR) -> np.ndarray:
    """The sum of thresholded profile scans ``[(mz, intensity), ...]``, each read at the sorted
    m/z positions ``axis`` as :func:`profile_on_axis` reads it.

    One sweep over every scan's pieces instead of one interpolation per scan over the whole
    axis: on a union of N scans' axes the latter costs N times the union (a 95-scan region of
    14 000-point scans took 5 s, this half a second). A scan is linear between consecutive
    points of one segment, so each such piece adds ``y0 + slope * (x - x0)`` on [x0, x1);
    sorted starts and ends with cumulative sums give every axis point its active pieces. A
    point that starts no piece (a segment's last, an isolated one) adds its own value where it
    lies; ``axis`` must hold it.
    """
    axis = np.asarray(axis, dtype=float)
    total = np.zeros(axis.size)
    starts, ends, c0, c1, knots, kvals = [], [], [], [], [], []
    for mz, y in scans:
        x = np.asarray(mz, dtype=float)
        v = np.asarray(y, dtype=float)
        if x.size == 0:
            continue
        if np.any(np.diff(x) < 0):
            order = np.argsort(x, kind="stable")
            x, v = x[order], v[order]
        piece = ~gap_mask(x, factor) if x.size > 1 else np.zeros(0, dtype=bool)
        k = np.flatnonzero(piece)
        slope = (v[k + 1] - v[k]) / (x[k + 1] - x[k])
        starts.append(x[k])
        ends.append(x[k + 1])
        c1.append(slope)
        c0.append(v[k] - slope * x[k])
        alone = np.ones(x.size, dtype=bool)
        alone[k] = False
        knots.append(x[alone])
        kvals.append(v[alone])
    if starts:
        s, e = np.concatenate(starts), np.concatenate(ends)
        a0, a1 = np.concatenate(c0), np.concatenate(c1)
        os_, oe = np.argsort(s, kind="stable"), np.argsort(e, kind="stable")
        cs0 = np.concatenate([[0.0], np.cumsum(a0[os_])])
        cs1 = np.concatenate([[0.0], np.cumsum(a1[os_])])
        ce0 = np.concatenate([[0.0], np.cumsum(a0[oe])])
        ce1 = np.concatenate([[0.0], np.cumsum(a1[oe])])
        i_s = np.searchsorted(s[os_], axis, side="right")
        i_e = np.searchsorted(e[oe], axis, side="right")
        total += (cs0[i_s] - ce0[i_e]) + axis * (cs1[i_s] - ce1[i_e])
    if knots:
        p = np.concatenate(knots)
        np.add.at(total, np.clip(np.searchsorted(axis, p), 0, max(axis.size - 1, 0)),
                  np.concatenate(kvals))
    return total


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
