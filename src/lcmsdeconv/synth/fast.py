"""Compiled equivalents of the steps that dominate training-spectrum synthesis.

Each function here returns exactly - bit for bit - what the function it stands for returns, so
training frames are the same with or without it; the tests compare every one with its
reference on random inputs (``tests/test_synth_fast.py``, ``tests/test_synth_speed.py``). The
kernels repeat the references' own arithmetic, operation for operation and in the same order,
without fast-math (no reassociation, no contraction into fused multiply-adds), and leave every
transcendental function, sort and numpy reduction to the same numpy call as the reference:
those may be computed differently on another processor, where reference and kernel still agree.

The scans' random draws are the exception: :func:`frame_scans` draws them with numpy's own
algorithms compiled (:mod:`lcmsdeconv.synth.rng`), which a self-check compares with numpy's
Generator on this machine once per process before they are used (``tests/test_synth_rng.py``
compares them too).

Without numba every function falls back to its reference. Inference never calls this module;
the shared functions it stands for (``nn.features.featurize``,
``nn.grid.LogMzGrid.resample_profile``) are unchanged.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import rng as draws

try:  # optional acceleration, as in render.py
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(f):
            return f
        return wrap if not args else args[0]


# --------------------------------------------------------------------- resampling of label rows
class ProfileRows:
    """``grid.resample_profile(mz, y)[b0:b0 + length]`` (zero-padded to ``length``) for many
    profiles ``y`` that share one m/z axis ``mz``, as a training frame's per-charge label rows do.

    Everything that depends on the axis alone - which points lie inside the grid, their order
    and repeated values (:func:`lcmsdeconv.core.profile.scan_points`), the width each represents
    (:func:`lcmsdeconv.core.profile.sample_widths`) and its grid bin - is computed once, by the
    reference's own calls; per profile only ``y * width / step`` is added into the crop's bins,
    in the order ``np.add.at`` adds them (:func:`_add_samples`). The points inside the grid of a
    sorted axis (a training frame's) are one run of it, read as a slice; the logarithm is taken
    once, for the widths and the bins; and where widths and step are finite, the samples that
    are zero, most of a label row's, are skipped (:func:`_add_nonzero_samples`).
    """

    def __init__(self, grid, mz, b0: int, length: int):
        from ..core.profile import REPEAT_REL

        mz = np.asarray(mz, dtype=float)
        self.b0, self.length, self.size, self.step = int(b0), int(length), grid.size, grid.step
        self.grid, self.mz = grid, mz
        self.inside = (mz > grid.mz_min) & (mz < grid.mz_max)
        where = np.flatnonzero(self.inside)
        self.empty = where.size == 0
        self.order = self.group = self.count = self.single = self.run = None
        self.skip_zeros = False
        if self.empty:
            return
        if where[-1] - where[0] + 1 == where.size:  # one run: the points inside, as a slice
            self.run = slice(int(where[0]), int(where[-1]) + 1)
        x = mz[self.run] if self.run is not None else mz[self.inside]
        # scan_points: sorted (stably), a repeated m/z read once at the mean of its values
        if x.size > 1 and (_has_repeats(np.ascontiguousarray(x), REPEAT_REL) if HAVE_NUMBA
                           else np.any(np.diff(x) <= REPEAT_REL * np.abs(x[1:]))):
            self.order = np.argsort(x, kind="stable")
            x = x[self.order]
            new = np.concatenate([[True], np.diff(x) > REPEAT_REL * np.abs(x[1:])])
            if not new.all():
                self.group = np.cumsum(new) - 1
                self.count = np.bincount(self.group)
                x = x[new]
        if x.size == 1:  # one point inside: the reference itself (rows are rare then)
            self.single = True
            return
        u = grid.mz_to_u(x)
        self.width = sample_widths(u)
        # grid.mz_to_bin(x), which takes the same logarithm again
        self.bins = np.clip(np.round((u - grid.u_min) / grid.step).astype(np.int64), 0,
                            grid.size - 1)
        # a zero sample adds +-0.0 to its bin while its width and the step are finite
        self.skip_zeros = bool(np.isfinite(self.step) and self.step != 0
                               and np.all(np.isfinite(self.width)))

    def row(self, y) -> np.ndarray:
        return self.row_into(y, np.zeros(self.length))

    def row_into(self, y, out: np.ndarray, start: int = 0) -> np.ndarray:
        """The row of a profile written into ``out`` (``length`` float64 zeros, as :meth:`row`
        starts from), which is returned. ``y`` holds the profile at the axis points ``start``
        to ``start + y.size``, and it is zero at the others (a profile rendered on the part of
        the axis its peaks reach, :func:`lcmsdeconv.synth.render.profile_reach`)."""
        if self.empty:
            return out
        y = np.asarray(y, dtype=float)
        if start < 0 or start + y.size > self.mz.size:
            raise ValueError("the profile's points lie outside the axis")
        if start != 0 or y.size != self.mz.size:
            if (HAVE_NUMBA and self.skip_zeros and self.run is not None
                    and self.order is None):
                # the samples outside the profile's points are zeros, which are skipped anyway
                p0, p1 = max(start, self.run.start), min(start + y.size, self.run.stop)
                if p1 > p0:
                    s0, s1 = p0 - self.run.start, p1 - self.run.start
                    _add_nonzero_samples(self.bins[s0:s1], y[p0 - start:p1 - start],
                                         self.width[s0:s1], float(self.step), self.b0, out)
                return out
            whole = np.zeros(self.mz.size)
            whole[start:start + y.size] = y
            y = whole
        if self.single:
            seg = self.grid.resample_profile(self.mz, y)[self.b0:self.b0 + self.length]
            out[:seg.size] = seg
            return out
        v = y[self.run] if self.run is not None else y[self.inside]
        if self.order is not None:
            v = v[self.order]
            if self.group is not None:
                v = np.bincount(self.group, weights=v) / self.count
        if HAVE_NUMBA:
            add = _add_nonzero_samples if self.skip_zeros else _add_samples
            add(self.bins, np.ascontiguousarray(v), self.width, float(self.step), self.b0, out)
        else:
            full = np.zeros(self.size)
            np.add.at(full, self.bins, v * self.width / self.step)
            seg = full[self.b0:min(self.b0 + self.length, self.size)]
            out[:seg.size] = seg
        return out


@njit(cache=True)
def _add_samples(bins, v, width, step, b0, out):
    """``out[bins[i] - b0] += v[i] * width[i] / step`` for every sample whose bin lies in the
    crop ``[b0, b0 + out.size)``, in sample order: the values and the order of additions of the
    reference's ``np.add.at(full, bins, v * width / step)``, restricted to the crop."""
    n = out.shape[0]
    for i in range(bins.shape[0]):
        j = bins[i] - b0
        if j >= 0 and j < n:
            out[j] += v[i] * width[i] / step


@njit(cache=True)
def _add_nonzero_samples(bins, v, width, step, b0, out):
    """:func:`_add_samples` without the samples that are zero, for finite widths and step: such
    a sample adds +0.0 or -0.0 to its bin, and neither changes a sum that starts from +0.0 (+0.0
    plus -0.0 is +0.0, and such a sum never becomes -0.0: a sum is -0.0 only when both terms
    are). A label row's profile is zero away from its charge's peaks, so most samples are."""
    n = out.shape[0]
    for i in range(bins.shape[0]):
        x = v[i]
        if x == 0.0:
            continue
        j = bins[i] - b0
        if j >= 0 and j < n:
            out[j] += x * width[i] / step


# ----------------------------------------------------------------------- scan resampling
def sample_widths(u: np.ndarray) -> np.ndarray:
    """:func:`lcmsdeconv.core.profile.sample_widths` (default gap factor), equal bit for bit.

    For a strictly increasing finite axis (what the resampling passes: sorted points, repeats
    merged) the block percentiles of :func:`lcmsdeconv.core.profile.local_spacing` come from
    :func:`_block_percentiles`, the interpolation between block centres from
    :func:`_interp_index` and the per-point rule from :func:`_widths`; the median is the
    reference's numpy call. Any other axis goes to the reference."""
    from ..core import profile as ref

    u = np.asarray(u, dtype=float)
    if not HAVE_NUMBA or u.size < 2:
        return ref.sample_widths(u)
    g = np.diff(u)
    if not _positive_finite(g):
        return ref.sample_widths(u)
    n = g.size
    w = max(1, min(ref.GAP_WINDOW, n))
    nb = n // w
    rem = n > nb * w
    vals = _block_percentiles(g, w, nb, rem, ref.SPACING_PERCENTILE)
    centres = np.arange(nb) * w + 0.5 * (w - 1)
    if rem:
        centres = np.append(centres, n - 0.5 * (w + 1))
    vals = np.minimum(vals, ref.SPACING_SPREAD * float(np.median(vals)))
    s = np.empty(n)
    _interp_index(centres, vals, s)  # np.interp(np.arange(n, dtype=float), centres, vals)
    out = np.empty(n + 1)
    _widths(g, s, float(ref.GAP_FACTOR), out)
    return out


def _block_percentiles(g: np.ndarray, w: int, nb: int, rem: bool, percent: float) -> np.ndarray:
    """``np.percentile(block, percent)`` of every block of ``w`` consecutive values of ``g``
    and, with ``rem``, of its last ``w`` values, as ``local_spacing`` takes them.

    numpy's linear method: the order statistics ``a, b`` either side of the virtual index
    ``(w - 1) * q`` and ``b - (b - a) * (1 - t)`` for ``t >= 0.5``, else ``a + (b - a) * t``,
    with ``t`` its fractional part. Order statistics are the same values however they are
    found, so here the blocks are sorted (numpy's sort is several times faster than its
    partition) - for values that are all positive and finite (:func:`_positive_finite`): with
    zeros of either sign or NaNs, which value numpy picks is not certain."""
    q = np.true_divide(percent, 100)
    virtual = (w - 1) * q
    if virtual >= w - 1:  # numpy: both indexes -1, the largest value
        kp = kn = w - 1
        t = float(virtual + 1)
    else:
        kp = int(np.floor(virtual))
        kn = kp + 1
        t = float(virtual - kp)
    srt = np.sort(g[: nb * w].reshape(nb, w), axis=1)
    a, b = srt[:, kp], srt[:, kn]
    if rem:
        last = np.sort(g[g.size - w:])
        a, b = np.append(a, last[kp]), np.append(b, last[kn])
    d = b - a
    return b - d * (1 - t) if t >= 0.5 else a + d * t


@njit(cache=True)
def _positive_finite(g):
    """Whether every value of ``g`` is positive and finite."""
    for i in range(g.shape[0]):
        v = g[i]
        if not (v > 0.0 and v < np.inf):
            return False
    return True


@njit(cache=True)
def _interp_index(xp, fp, out):
    """``np.interp(np.arange(out.size, dtype=float), xp, fp)`` for strictly increasing ``xp``
    (block centres), by numpy's own rule: below the first point ``fp[0]``, above the last
    ``fp[-1]``; at a point (or the last) its value; between ``xp[j]`` and ``xp[j + 1]``
    ``slope * (x - xp[j]) + fp[j]`` with ``slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])``
    (numpy computes the slopes first too), retried from the right end when that is NaN. The
    queries increase, so the interval index only moves forward where numpy searches for it."""
    n = out.shape[0]
    m = xp.shape[0]
    if m == 1:
        for i in range(n):
            out[i] = fp[0]
        return
    slope = np.empty(m - 1)
    for k in range(m - 1):
        slope[k] = (fp[k + 1] - fp[k]) / (xp[k + 1] - xp[k])
    j = 0
    for i in range(n):
        x = float(i)
        if x < xp[0]:
            out[i] = fp[0]
            continue
        if x > xp[m - 1]:
            out[i] = fp[m - 1]
            continue
        while j + 1 < m and xp[j + 1] <= x:
            j += 1
        if j == m - 1 or xp[j] == x:
            out[i] = fp[j]
            continue
        r = slope[j] * (x - xp[j]) + fp[j]
        if np.isnan(r):
            r = slope[j] * (x - xp[j + 1]) + fp[j + 1]
            if np.isnan(r) and fp[j] == fp[j + 1]:
                r = fp[j]
        out[i] = r


@njit(cache=True)
def _widths(g, s, factor, out):
    """The per-point rule of :func:`lcmsdeconv.core.profile.sample_widths` for spacings ``g``
    and local spacings ``s``: half of each neighbouring spacing that is no gap (``g <= factor *
    s``); toward a gap the half-width of the other side; between two gaps half of
    ``np.minimum`` of the neighbouring local spacings (the end points their own). The
    reference's ``np.where`` selections, element by element."""
    n = out.shape[0]
    for i in range(n):
        left = np.nan
        if i > 0 and not (g[i - 1] > factor * s[i - 1]):
            left = 0.5 * g[i - 1]
        right = np.nan
        if i < n - 1 and not (g[i] > factor * s[i]):
            right = 0.5 * g[i]
        if i == 0:
            sp = s[0]
        elif i == n - 1:
            sp = s[n - 2]
        else:
            a = s[i - 1]
            b = s[i]
            sp = a if (a <= b or np.isnan(a)) else b
        if np.isnan(left):
            lf = right if not np.isnan(right) else 0.5 * sp
        else:
            lf = left
        if np.isnan(right):
            rf = left if not np.isnan(left) else 0.5 * sp
        else:
            rf = right
        out[i] = lf + rf


@njit(cache=True)
def _has_repeats(x, rel):
    """``np.any(np.diff(x) <= rel * np.abs(x[1:]))``, the test of ``core.profile.scan_points``."""
    for i in range(x.shape[0] - 1):
        if x[i + 1] - x[i] <= rel * np.abs(x[i + 1]):
            return True
    return False


def positive_points(axis: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(axis[y > 0], y[y > 0])``, by :func:`_compress_positive`."""
    if not HAVE_NUMBA or axis.dtype != np.float64 or y.dtype != np.float64 \
            or axis.shape != y.shape or axis.ndim != 1:
        on = y > 0
        return axis[on], y[on]
    mz, v = np.empty(y.size + 1), np.empty(y.size + 1)
    n = _compress_positive(axis, y, mz, v)
    return mz[:n], v[:n]


@njit(cache=True)
def _compress_positive(axis, y, mz, v):
    """``axis[y > 0]`` and ``y[y > 0]`` into ``mz`` and ``v`` (one element longer than ``y``),
    in order; returns their number. Every point is written and the position advanced only for a
    positive one, which spares the branch a thresholded scan makes unpredictable."""
    n = 0
    for i in range(y.shape[0]):
        mz[n] = axis[i]
        v[n] = y[i]
        n += 1 if y[i] > 0 else 0
    return n


def resample_profile(grid, mz, intensity) -> np.ndarray:
    """``grid.resample_profile(mz, intensity)`` (:meth:`lcmsdeconv.nn.grid.LogMzGrid.
    resample_profile`), equal bit for bit: the reference's steps with the widths from
    :func:`sample_widths`, the log computed once for widths and bins, and the samples added in
    order by :func:`_add_samples`."""
    from ..core.profile import REPEAT_REL, scan_points

    if not HAVE_NUMBA:
        return grid.resample_profile(mz, intensity)
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    out = np.zeros(grid.size, dtype=np.float64)
    m = (mz > grid.mz_min) & (mz < grid.mz_max)
    if not np.any(m):
        return out
    # a training scan lies inside its grid (which spans the scans), so mostly no copy is made
    x, v = (mz, intensity) if m.all() else (mz[m], intensity[m])
    if x.size > 1 and _has_repeats(x, REPEAT_REL):
        x, v = scan_points(x, v)
    if x.size == 1:  # one point inside: the reference itself
        return grid.resample_profile(mz, intensity)
    u = grid.mz_to_u(x)
    du = sample_widths(u)
    # grid.mz_to_bin(x), which takes the same logarithm again
    b = np.clip(np.round((u - grid.u_min) / grid.step).astype(np.int64), 0, grid.size - 1)
    _add_samples(b, v, du, float(grid.step), 0, out)
    return out


def spectrum_to_grid(mz, intensity, polarity: int, step: float, mz_max: float, frames):
    """``nn.grid.spectrum_to_grid(mz, intensity, polarity, step, mz_max, True, frames=frames)``
    for the profile scans of a training frame, equal bit for bit (each scan resampled by
    :func:`resample_profile`). Centroided scans, or none, go to the reference."""
    from ..chem.instrument import estimate_noise_sigma
    from ..nn.grid import LogMzGrid
    from ..nn.grid import spectrum_to_grid as reference

    if not HAVE_NUMBA or not frames or any(f.is_profile is False for f in frames):
        return reference(mz, intensity, polarity, step, mz_max, True, frames=frames)
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    # the rest is the reference's, line for line
    span = [np.asarray(f.mz, dtype=float) for f in frames if np.size(f.mz)]
    if not span and mz.size:
        span = [mz]
    if span:
        lo = max(50.0, min(float(a.min()) for a in span) * 0.995)
        hi = min(mz_max, max(float(a.max()) for a in span) * 1.005)
    else:
        lo, hi = 50.0, mz_max
    if hi <= lo * 1.01:
        lo, hi = 50.0, mz_max
    grid = LogMzGrid(lo, hi, step, polarity)
    by_scan = not all(f.mz.size == mz.size and np.array_equal(f.mz, mz) for f in frames)
    if by_scan:
        observed = np.zeros(grid.size, dtype=np.float64)
        for f in frames:
            observed += resample_profile(grid, f.mz, f.intensity)
    else:
        observed = resample_profile(grid, mz, intensity)
    return grid, observed, estimate_noise_sigma(observed)


# ------------------------------------------------------------------------------- scans
def frame_scans(rng: np.random.Generator, clean: np.ndarray, weights: np.ndarray, gain: float,
                esig: float, sat_level: float | None, thr: float, pad: int,
                axis: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], list]:
    """A training frame's noisy scans and their sum as positive points, ``(raw, scans)``: equal
    bit for bit to the reference's lines (:func:`_frame_scans_numpy`)::

        frames_int = [noisy_scan(rng, clean, w, gain, esig, sat_level, thr, pad)
                      for w in weights]
        total = frames._sum_scans(frames_int)
        raw = positive_points(axis, total)
        scans = [positive_points(axis, s) for s in frames_int]

    with the same draws in the same order, ``rng`` left where they leave it. Each scan is one
    compiled kernel that draws with numpy's own algorithms (:func:`lcmsdeconv.synth.rng.scan`)
    when this process's self-check found them equal to numpy's
    (:func:`lcmsdeconv.synth.rng.available`) and the arguments are what the kernel covers: a
    plain ``Generator`` on ``PCG64``; float64 profile, axis and weights; finite weights and gain,
    the gain not zero; a noise level that is finite and not negative (numpy refuses a negative
    one and -0.0, and a NaN one makes NaN noise); a saturation level that is positive and
    finite (``np.minimum`` keeps a NaN level, and for a zero level which zero it gives depends
    on numpy's code path), or None; an axis long enough for the threshold's window. Anything
    else, and a scan with a rate numpy refuses or one above
    :data:`lcmsdeconv.synth.rng.RATE_MAX`, takes the reference's lines from the generator's
    untouched state, with their result or their exception."""
    if _compiled_scans_apply(rng, clean, weights, gain, esig, sat_level, thr, pad, axis):
        out = draws.scan_frame(rng, np.ascontiguousarray(clean), weights, float(gain),
                               float(esig), math.inf if sat_level is None else float(sat_level),
                               float(thr), int(pad) if thr > 0 else 0,
                               np.ascontiguousarray(axis))
        if out is not None:
            return out
    return _frame_scans_numpy(rng, clean, weights, gain, esig, sat_level, thr, pad, axis)


def _frame_scans_numpy(rng, clean, weights, gain, esig, sat_level, thr, pad, axis):
    """The lines of ``frames.generate_frame_profile`` that :func:`frame_scans` stands for."""
    from .frames import _sum_scans

    frames_int = [noisy_scan(rng, clean, w, gain, esig, sat_level, thr, pad) for w in weights]
    total = _sum_scans(frames_int)
    raw = positive_points(axis, total)  # (axis[total > 0], total[total > 0])
    scans = [positive_points(axis, s) for s in frames_int]
    return raw, scans


def _real(x) -> bool:
    """A Python (or numpy float64) number, not a bool."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _compiled_scans_apply(rng, clean, weights, gain, esig, sat_level, thr, pad, axis) -> bool:
    """Whether :func:`frame_scans` may draw these scans compiled (see there)."""
    if not (HAVE_NUMBA and draws.is_pcg64_generator(rng)
            and isinstance(clean, np.ndarray) and clean.dtype == np.float64 and clean.ndim == 1
            and clean.size > 0 and isinstance(axis, np.ndarray) and axis.dtype == np.float64
            and axis.shape == clean.shape
            and isinstance(weights, np.ndarray) and weights.dtype == np.float64
            and weights.ndim == 1 and weights.size > 0 and bool(np.all(np.isfinite(weights)))
            and _real(gain) and math.isfinite(gain) and gain != 0
            and _real(esig) and math.isfinite(esig) and esig >= 0
            and math.copysign(1.0, esig) > 0
            and (sat_level is None or (_real(sat_level) and 0 < sat_level < math.inf))
            and _real(thr)):
        return False
    if thr > 0 and not (isinstance(pad, (int, np.integer)) and not isinstance(pad, bool)
                        and 0 <= pad and clean.size >= 2 * pad + 1):
        return False
    return draws.available()


def noisy_scan(rng: np.random.Generator, clean: np.ndarray, w: float, gain: float, esig: float,
               sat_level: float | None, thr: float, pad: int) -> np.ndarray:
    """One scan of a training frame, equal bit for bit to the reference's lines::

        lam = np.clip(clean * w / gain, 0.0, None)
        scan = rng.poisson(lam).astype(np.float64) * gain + np.abs(rng.normal(0.0, esig, n))
        scan = np.minimum(scan, sat_level)                      # when saturating
        scan[~(np.convolve(scan >= thr, np.ones(2 * pad + 1), mode="same") > 0)] = 0.0

    with the same draws in the same order. numpy draws nothing for a rate of zero (its count is
    0), so only the nonzero rates are passed to ``rng.poisson`` (checked once per process,
    :func:`_poisson_skips_zeros`); the rates with their positions, the arithmetic and the
    thresholding are one pass each (:func:`_nonzero_rates`, :func:`_scan_values`,
    :func:`_threshold_segments`)."""
    n = clean.size
    if not HAVE_NUMBA:
        lam = np.clip(clean * w / gain, 0.0, None)
        scan = rng.poisson(lam).astype(np.float64) * gain + np.abs(rng.normal(0.0, esig, n))
    else:
        if _poisson_skips_zeros() and clean.dtype == np.float64 and clean.ndim == 1:
            idx = np.empty(n, dtype=np.int64)
            vals = np.empty(n)
            m = _nonzero_rates(np.ascontiguousarray(clean), float(w), float(gain), idx, vals)
            counts = rng.poisson(vals[:m])
            idx = idx[:m]
        else:
            counts = rng.poisson(np.clip(clean * w / gain, 0.0, None))
            idx = np.arange(n, dtype=np.int64)
        noise = rng.normal(0.0, esig, n)
        scan = np.empty(n)
        _scan_values(counts.astype(np.int64, copy=False), idx, float(gain), noise, scan)
    if sat_level is not None:
        scan = np.minimum(scan, sat_level)
    if thr > 0:
        if HAVE_NUMBA and pad >= 0 and n >= 2 * pad + 1:
            _threshold_segments(scan, float(thr), int(pad))
        else:
            scan[~(np.convolve(scan >= thr, np.ones(2 * pad + 1), mode="same") > 0)] = 0.0
    return scan


_SKIPS_ZEROS: list = []


def _poisson_skips_zeros() -> bool:
    """Whether ``rng.poisson`` draws nothing for a rate of zero, so that the counts of the
    nonzero rates alone, from the same generator state, are the counts and leave the generator
    where all the rates would (numpy's ``random_poisson`` returns 0 for it)."""
    if not _SKIPS_ZEROS:
        lam = np.array([0.0, 2.5, 0.0, 0.0, 40.0, 1e-3, 0.0, 7.0, -0.0, 12.0])
        g1, g2 = np.random.default_rng(12345), np.random.default_rng(12345)
        full = g1.poisson(lam)
        part = np.zeros(lam.size, dtype=full.dtype)
        part[lam != 0] = g2.poisson(lam[lam != 0])
        _SKIPS_ZEROS.append(bool(np.array_equal(full, part)
                                 and g1.bit_generator.state == g2.bit_generator.state
                                 and np.array_equal(g1.random(4), g2.random(4))))
    return _SKIPS_ZEROS[0]


@njit(cache=True)
def _nonzero_rates(clean, w, gain, idx, vals):
    """The positions and values of the nonzero rates ``lam = np.clip(clean * w / gain, 0.0,
    None)`` (NaN included, -0.0 not, as ``lam != 0`` selects them), in order; returns their
    number. Each rate is ``clean[i] * w / gain`` as numpy computes it, a negative one 0.0; a
    rate of -0.0 may come out as either zero there, and is not selected either way."""
    m = 0
    for i in range(clean.shape[0]):
        x = clean[i] * w / gain
        if x < 0.0:
            x = 0.0
        idx[m] = i
        vals[m] = x
        m += 1 if x != 0.0 else 0
    return m


@njit(cache=True)
def _scan_values(counts, idx, gain, noise, out):
    """``out = c * gain + abs(noise)`` with ``c`` the Poisson count, ``counts[k]`` at
    ``idx[k]`` and 0 elsewhere: ``rng.poisson(lam).astype(np.float64) * gain +
    np.abs(noise)``, one multiplication and one addition per point as there."""
    k = 0
    m = idx.shape[0]
    for i in range(out.shape[0]):
        c = 0
        if k < m and idx[k] == i:
            c = counts[k]
            k += 1
        out[i] = float(c) * gain + np.abs(noise[i])


@njit(cache=True)
def _threshold_segments(scan, thr, pad):
    """``scan[~keep] = 0.0`` for ``keep = np.convolve(scan >= thr, np.ones(2 * pad + 1),
    mode="same") > 0``: every point farther than ``pad`` points from all points at or above the
    threshold is zeroed (the convolution counts those points exactly; here a running count over
    the window ``i - pad .. i + pad``). Needs ``scan.size >= 2 * pad + 1``."""
    n = scan.shape[0]
    above = np.empty(n, dtype=np.bool_)
    for i in range(n):
        above[i] = scan[i] >= thr
    cnt = 0
    for j in range(min(pad + 1, n)):
        cnt += 1 if above[j] else 0
    for i in range(n):
        if cnt == 0:
            scan[i] = 0.0
        j = i + pad + 1
        if j < n and above[j]:
            cnt += 1
        j = i - pad
        if j >= 0 and above[j]:
            cnt -= 1


# --------------------------------------------------------------------------- featurization
@lru_cache(maxsize=32)
def _comb_table(step: float, z_max: int):
    """The nonzero comb shifts of every charge 0..z_max in the order the reference adds them
    (``nn.features.comb_shifts``), ``[z_max + 1, 6]``, and their number per charge."""
    from ..nn.features import comb_shifts

    shifts = comb_shifts(step, z_max)
    table = np.zeros((z_max + 1, 6), dtype=np.int64)
    count = np.zeros(z_max + 1, dtype=np.int64)
    for z, s in shifts.items():
        nz = [v for v in s if v != 0]
        table[z, :len(nz)] = nz
        count[z] = len(nz)
    table.setflags(write=False)
    count.setflags(write=False)
    return table, count


@njit(cache=True)
def _comb_sum(x, table, count, z, acc):
    """``acc`` = the sum of ``x[i + shift]`` over charge ``z``'s shifts in order, from 0.0, at
    every bin: the reference's ``acc += shifted(logint, sh)`` sequence. Outside the window the
    reference adds 0.0, which changes no sum that starts from 0.0 (it is never -0.0), so those
    additions are left out and the inner loop runs over contiguous slices."""
    n = x.shape[0]
    for i in range(n):
        acc[i] = 0.0
    for t in range(count[z]):
        sh = table[z, t]
        lo = max(0, -sh)
        hi = min(n, n - sh)
        if hi > lo:
            a = acc[lo:hi]  # slices, so that the loop index is known not to be negative
            xs = x[lo + sh:hi + sh]
            for k in range(hi - lo):
                a[k] += xs[k]


@njit(cache=True)
def _comb_best(x, table, count, z_max, val, idx):
    """The maximum over charges 2..z_max of the comb evidence ``x * (acc / count)`` at every
    bin and the index of the charge (``z - 2``) that attains it, by numpy's argmax rule (the
    first maximum; a NaN is maximal and ends the search): ``np.argmax(ev[1:z_max], axis=0)``
    and the values there for ``ev = nn.features.comb_evidence(x, step, z_max)``, without the
    ``[z_max, L]`` array. For an input bounded well inside the float range (the log-intensity
    channel lies in [0, 5]) no evidence value can be NaN and the rule is a strict ``>``."""
    n = x.shape[0]
    acc = np.empty(n)
    bounded = True
    for i in range(n):
        if not (np.abs(x[i]) < 1e300):
            bounded = False
            break
    done = np.zeros(n, dtype=np.bool_)
    for z in range(2, z_max + 1):
        m = count[z]
        if m:
            _comb_sum(x, table, count, z, acc)
        if z == 2:
            for i in range(n):
                v = x[i] * (acc[i] / m) if m else 0.0
                val[i] = v
                idx[i] = 0
                done[i] = np.isnan(v)
        elif bounded and m:
            for i in range(n):
                v = x[i] * (acc[i] / m)
                if v > val[i]:
                    val[i] = v
                    idx[i] = z - 2
        else:
            for i in range(n):
                v = x[i] * (acc[i] / m) if m else 0.0
                if not done[i] and not (v <= val[i]):
                    val[i] = v
                    idx[i] = z - 2
                    done[i] = np.isnan(v)


#: bins of a window the tiled comb (:func:`_comb_best_tiled`) evaluates every charge on at a time
COMB_TILE = 2048
#: below this share of nonzero bins a window's comb is evaluated at those bins only
#: (:func:`_comb_best_sparse`)
COMB_SPARSE_SHARE = 0.3


@njit(cache=True)
def _comb_best_tiled(x, table, count, z_max, val, idx, tile):
    """:func:`_comb_best` for an input bounded well inside the float range, equal at every bin,
    evaluated tile by tile: every charge on ``tile`` bins before the next tile, so that the
    running sums, maxima and charges of a tile stay in the first-level cache instead of the
    whole window being swept once per charge (on a 131 072-bin window, more than the caches
    hold). Each bin's arithmetic is the reference's in its order: the sum over the charge's
    shifts in table order from 0.0 (those outside the window left out), then ``x * (acc / m)``,
    then the strict comparison charge by charge. Returns False, writing nothing, for an input
    that is not bounded or a charge without shifts (the reference takes those)."""
    n = x.shape[0]
    for i in range(n):
        if not (np.abs(x[i]) < 1e300):
            return False
    for z in range(2, z_max + 1):
        if count[z] == 0:
            return False
    acc = np.empty(tile)
    for i0 in range(0, n, tile):
        i1 = min(n, i0 + tile)
        w = i1 - i0
        for z in range(2, z_max + 1):
            m = count[z]
            for k in range(w):
                acc[k] = 0.0
            for t in range(m):
                sh = table[z, t]
                lo = max(i0, -sh)
                hi = min(i1, n - sh)
                if hi > lo:
                    a = acc[lo - i0:hi - i0]
                    xs = x[lo + sh:hi + sh]
                    for k in range(hi - lo):
                        a[k] += xs[k]
            if z == 2:
                for k in range(w):
                    val[i0 + k] = x[i0 + k] * (acc[k] / m)
                    idx[i0 + k] = 0
            else:
                for k in range(w):
                    v = x[i0 + k] * (acc[k] / m)
                    if v > val[i0 + k]:
                        val[i0 + k] = v
                        idx[i0 + k] = z - 2
    return True


@njit(cache=True)
def _comb_best_sparse(x, table, count, z_max, val, idx):
    """:func:`_comb_best` for an input that is finite and non-negative, without negative zeros
    (the log-intensity channel), equal at every bin, evaluated at the nonzero bins only: at a bin
    that is +0.0 every charge's evidence is +0.0 * (acc / m) = +0.0 (acc is not negative), so
    the maximum is +0.0 at the first charge. At the others each charge's sum (shifts in table
    order from 0.0, those outside the window left out), division, product and strict comparison
    are the reference's. A thresholded spectrum is mostly zeros, and its comb costs that much
    less. Returns False, writing nothing, for any other input."""
    n = x.shape[0]
    for i in range(n):
        v = x[i]
        if not (v >= 0.0 and v < 1e300) or (v == 0.0 and math.copysign(1.0, v) < 0.0):
            return False
    for z in range(2, z_max + 1):
        if count[z] == 0:
            return False
    for i in range(n):
        xi = x[i]
        if xi == 0.0:
            val[i] = 0.0
            idx[i] = 0
            continue
        for z in range(2, z_max + 1):
            m = count[z]
            acc = 0.0
            for t in range(m):
                j = i + table[z, t]
                if j >= 0 and j < n:
                    acc += x[j]
            v = xi * (acc / m)
            if z == 2 or v > val[i]:
                val[i] = v
                idx[i] = z - 2
    return True


@njit(cache=True)
def _comb_rows(x, table, count, z_max, ev):
    """``nn.features.comb_evidence(x, step, z_max)`` into the zeroed ``ev`` ``[z_max, L]``:
    row ``z - 1`` holds the evidence of charge z for z = 2..z_max with any shift, as there."""
    n = x.shape[0]
    acc = np.empty(n)
    for z in range(2, z_max + 1):
        m = count[z]
        if m == 0:
            continue
        _comb_sum(x, table, count, z, acc)
        for i in range(n):
            ev[z - 1, i] = x[i] * (acc[i] / m)


def featurize(window: np.ndarray, noise_sigma: float, step: float, z_max: int = 64,
              evidence_channels: int = 0) -> np.ndarray:
    """:func:`lcmsdeconv.nn.features.featurize`, equal bit for bit.

    The log-intensity and local-maximum channels are computed by the reference's numpy and scipy
    calls; the comb evidence by :func:`_comb_best` (or :func:`_comb_rows` when evidence channels
    are asked for), the same additions, division and product per element: at the nonzero bins
    only where fewer than :data:`COMB_SPARSE_SHARE` of them are (:func:`_comb_best_sparse`),
    tile by tile otherwise (:func:`_comb_best_tiled`).
    """
    from ..nn import features as ref

    if not HAVE_NUMBA:
        return ref.featurize(window, noise_sigma, step, z_max=z_max,
                             evidence_channels=evidence_channels)
    from scipy.ndimage import maximum_filter1d

    L = window.size
    ns = max(noise_sigma, 1e-6)
    ch0 = np.clip(np.log10(1.0 + np.clip(window, 0, None) / ns), 0.0, 5.0)
    local_max = maximum_filter1d(window, size=min(4001, (L // 4) * 2 + 1), mode="nearest")
    ch1 = np.where(local_max > 0, window / (local_max + 1e-9), 0.0)
    zc = max(z_max, evidence_channels)
    x = np.ascontiguousarray(ch0, dtype=np.float64)
    table, count = _comb_table(float(step), int(zc))
    ev = None
    if evidence_channels:
        ev = np.zeros((zc, L))
        _comb_rows(x, table, count, int(zc), ev)
        ch2, ch3 = ref.comb_transform(ch0, step, z_max=z_max, evidence=ev[:z_max])
    else:
        # comb_transform's own steps on the best charge of every bin
        best = np.zeros(L)
        bestz = np.zeros(L)
        if z_max > 1:
            val = np.empty(L)
            idx = np.empty(L, dtype=np.int64)
            if np.count_nonzero(x) < COMB_SPARSE_SHARE * L:
                done = _comb_best_sparse(x, table, count, int(z_max), val, idx)
            else:
                done = _comb_best_tiled(x, table, count, int(z_max), val, idx, COMB_TILE)
            if not done:
                _comb_best(x, table, count, int(z_max), val, idx)
            pos = val > 0
            best[pos] = val[pos]
            bestz[pos] = idx[pos] + 2
        if best.max() > 0:
            best = best / best.max()
        ch2, ch3 = best, bestz / z_max
    out = np.empty((ref.N_CHANNELS + evidence_channels, L), dtype=np.float32)
    out[0] = ch0
    out[1] = ch1
    out[2] = ch2
    out[3] = ch3
    if evidence_channels:
        out[ref.N_CHANNELS:] = np.clip(ev[:evidence_channels] / ref.EVIDENCE_SCALE, 0.0, 1.0)
    return out


# ------------------------------------------------------------------------------ labels
#: classes (charges and class 0) up to which :func:`build_labels` keeps its table of order
#: types (3 x classes^2 entries); a crop holds at most a hundred charges
LABEL_MAX_CLASSES = 256


def build_labels(charge_mat: np.ndarray, charges: np.ndarray, noise: np.ndarray, k: int = 3,
                 heat_sigma: float = 1.5, apex_snr: float = 2.0, noise_level: float | None = None,
                 observed: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """:func:`lcmsdeconv.synth.labels.build_labels`, equal bit for bit, for labels that are
    shares of the observed intensity (``observed`` given, ``k`` <= 3, float64 rows: every
    profile training frame's labels); anything else goes to the reference.

    The reference sorts the shares of every bin with one ``np.argsort`` over the whole
    ``[nz + 1, L]`` matrix, a fifth of a whole-spectrum frame's synthesis. Here
    :func:`_shares_pass` computes the shares by the reference's operations in its order (the
    rows added one after another, as numpy sums a C-ordered matrix over its first axis) and
    keeps, as they come, the largest ``k + 1`` nonzero shares of each bin and their number.
    Where the ``k`` largest are distinct and above all others, every correct sort puts the same
    classes first, and they are written directly. Where fewer than ``k`` shares are nonzero, all
    distinct, and the rest exactly zero (most bins: background), the places a sort gives the
    tied zeros depend on the bin's order type alone - which classes hold the nonzero shares, in
    which order - not on the values; numpy's own ``np.argsort`` sorts one column of each type
    found, as the reference sorts every column, so whichever sort numpy runs on this processor
    the result is its own. Bins that fit neither (shares equal where the order matters, NaN) are
    sorted by numpy itself, from the reference's shares. The apex heatmap is the reference's,
    with the median of the noise taken as zero without sorting where more than half of it is
    exactly zero and none negative.
    """
    from .labels import build_labels as reference

    args = (charge_mat, charges, noise, k, heat_sigma, apex_snr, noise_level, observed)
    if not HAVE_NUMBA or observed is None:
        return reference(*args)
    cmat = np.asarray(charge_mat)
    rest = np.asarray(noise)
    nz, L = (cmat.shape[0], rest.size) if cmat.ndim == 2 and rest.ndim == 1 else (-1, -1)
    # (rows in C order: numpy sums those over the first axis row after row, as the kernel does;
    # a matrix in another order it may sum pairwise)
    if not (1 <= k <= 3 and 0 <= nz < LABEL_MAX_CLASSES and L >= 3 and cmat.shape[1] == L
            and cmat.dtype == np.float64 and cmat.flags.c_contiguous
            and rest.dtype == np.float64):
        return reference(*args)
    rest = np.ascontiguousarray(rest)
    obs = np.ascontiguousarray(observed, dtype=np.float64)
    if obs.shape != (L,):
        return reference(*args)
    n = nz + 1
    kk = min(k, n)
    count = np.empty(L, dtype=np.int64)
    top_v = np.empty((L, kk + 1))
    top_r = np.empty((L, kk + 1), dtype=np.int64)
    unsure = np.zeros(L, dtype=np.bool_)
    if not _shares_pass(cmat, cmat.view(np.int64), obs, kk, count, top_v, top_r, unsure):
        return reference(*args)  # a negative, infinite or NaN row value, or -0.0
    topk_pos = np.zeros((L, k), dtype=np.int64)
    topk_w = np.zeros((L, k), dtype=np.float32)
    kind = np.empty(L, dtype=np.int8)
    group = np.empty(L, dtype=np.int64)
    table = np.full(3 * n * n, -1, dtype=np.int64)
    types = np.empty((min(L, 3 * n * n), 3), dtype=np.int64)
    ng = _classify(kk, n, count, top_v, top_r, unsure, topk_pos, topk_w, kind, group, table,
                   types)
    fill = np.zeros((max(ng, 1), kk), dtype=np.int64)
    if ng:
        # one column of each order type: its nonzero shares in their order, +0.0 elsewhere
        cols = np.zeros((n, ng))
        g = np.arange(ng)
        cols[types[:ng, 1], g] = np.where(types[:ng, 0] > 0, 2.0, 0.0)
        cols[types[:ng, 2], g] += np.where(types[:ng, 0] > 1, 1.0, 0.0)
        order = np.argsort(-cols, axis=0)[:kk]  # the reference's call
        for j in range(kk - 1):  # a kind-1 bin has at most kk - 1 nonzero shares
            has = types[:ng, 0] > j
            if not np.array_equal(order[j, has], types[:ng, 1 + j][has]):
                return reference(*args)  # not a sort that puts larger shares first
        fill[:] = order.T
    other = np.flatnonzero(kind == 2)
    if other.size:
        # the reference's shares of those bins, sorted by numpy itself. Their sums over the
        # rows are added row after row, as numpy sums the whole matrix: the columns taken apart
        # (in Fortran order, or a single one) it would sum pairwise
        sub = cmat[:, other]
        yo = np.clip(obs[other], 0.0, None)
        total = np.zeros(other.size)
        for r in range(nz):
            total += sub[r]
        denom = np.maximum(yo, total) + 1e-12
        ion = sub * ((yo > 0) / denom)
        ion_sum = np.zeros(other.size)
        for r in range(nz):
            ion_sum += ion[r]
        shares = np.empty((n, other.size))
        shares[1:] = ion
        shares[0] = np.clip(1.0 - ion_sum, 0.0, 1.0) if nz else 1.0
        order = np.argsort(-shares, axis=0)[:kk]
        cols = np.arange(other.size)
        for j in range(kk):
            topk_pos[other, j] = order[j]
            topk_w[other, j] = shares[order[j], cols]
    class_ids = np.concatenate([[0], np.asarray(charges).astype(np.int64)])
    topk_z = np.zeros((L, k), dtype=np.int16)
    _fill_topk(kk, kind, group, types, fill, topk_pos, class_ids, topk_z)

    heat = np.zeros(L, dtype=np.float64)
    if nz:
        median = 0.0 if _median_is_zero(rest) else float(np.median(rest))
        level = max(median, float(noise_level or 0.0), 1e-6)
        apex = np.zeros(L, dtype=np.bool_)
        _apex_mask(cmat, apex_snr * level, apex)
        bins = np.flatnonzero(apex).astype(np.int64)
        if bins.size:
            hw = int(np.ceil(3 * heat_sigma))
            x = np.arange(-hw, hw + 1)
            kernel = np.exp(-0.5 * (x / heat_sigma) ** 2)
            heat_envelope(bins, hw, kernel, heat)
    return {"topk_z": topk_z, "topk_w": topk_w, "heat": heat.astype(np.float32)}


@njit(cache=True)
def _shares_pass(cmat, bits, observed, kk, count, top_v, top_r, unsure):
    """The shares of :func:`lcmsdeconv.synth.labels.build_labels` (``observed`` given), bin by
    bin, without the share matrix: for every bin the ``kk + 1`` largest nonzero shares in
    descending order with their classes (``top_v``, ``top_r`` ``[L, kk + 1]``; class 0 is the
    non-ion remainder, class ``r + 1`` row ``r``), and their number (``count``); ``unsure``
    marks bins whose shares may be NaN. Every share is the reference's expression, its sums
    taken row after row as numpy sums the C-ordered rows. Returns False, before any share, if a
    row value is negative, -0.0, infinite or NaN (``bits`` is ``cmat`` viewed as int64): a
    share then need not be a non-negative number or +0.0, as the rest relies on."""
    nz = cmat.shape[0]
    L = cmat.shape[1]
    total = np.zeros(L)
    lo = np.int64(0)
    hi = np.int64(0)
    for r in range(nz):
        for i in range(L):
            b = bits[r, i]
            lo = min(lo, b)
            hi = max(hi, b)
            total[i] = cmat[0, i] if r == 0 else total[i] + cmat[r, i]
    if lo < 0 or hi >= np.int64(0x7FF0000000000000):
        return False
    # q = seen / (np.maximum(y, total) + 1e-12) with y = np.clip(observed, 0.0, None)
    q = np.empty(L)
    for i in range(L):
        y = observed[i]
        if y < 0.0:
            y = 0.0
        t = total[i]
        if y != y:
            unsure[i] = True
        seen = 1.0 if y > 0.0 else 0.0
        q[i] = seen / ((y if y >= t else t) + 1e-12)
        count[i] = 0
    ion = total  # reused: the running sum of the ion shares
    for r in range(nz):
        for i in range(L):
            v = cmat[r, i] * q[i]
            ion[i] = v if r == 0 else ion[i] + v
            if v > 0.0:
                _insert(top_v, top_r, count, i, v, r + 1, kk)
    for i in range(L):
        x = 1.0 - ion[i] if nz > 0 else 1.0  # np.clip(1.0 - ion.sum(axis=0), 0.0, 1.0)
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        if x > 0.0:
            _insert(top_v, top_r, count, i, x, 0, kk)
        elif not (x == 0.0):
            unsure[i] = True
    return True


@njit(cache=True, inline="always")
def _insert(top_v, top_r, count, i, v, cls, kk):
    """Share ``v`` of class ``cls`` into bin i's descending list of the ``kk + 1`` largest
    (after any equal one already there)."""
    c = count[i]
    count[i] = c + 1
    if c > kk:
        if not (v > top_v[i, kk]):
            return
        c = kk
    while c > 0 and top_v[i, c - 1] < v:
        top_v[i, c] = top_v[i, c - 1]
        top_r[i, c] = top_r[i, c - 1]
        c -= 1
    top_v[i, c] = v
    top_r[i, c] = cls


@njit(cache=True)
def _classify(kk, n, count, top_v, top_r, unsure, topk_pos, topk_w, kind, group, table, types):
    """Per bin: kind 0 - the ``kk`` largest shares distinct and above all others, written to
    ``topk_pos``/``topk_w``; kind 1 - fewer than ``kk`` nonzero shares, distinct, written, and
    the bin's order type (their number and classes in order) numbered in ``group``, with
    ``table`` ``[3 n^2]`` from type to number and ``types`` ``[., 3]`` from number to type;
    kind 2 - the rest. Returns the number of order types."""
    L = count.shape[0]
    ng = 0
    for i in range(L):
        if unsure[i]:
            kind[i] = 2
            continue
        m = count[i]
        top = min(m, kk)
        ok = True
        for j in range(1, top):
            if not (top_v[i, j - 1] > top_v[i, j]):
                ok = False
        if m > kk and not (top_v[i, kk - 1] > top_v[i, kk]):
            ok = False
        if not ok:
            kind[i] = 2
            continue
        for j in range(top):
            topk_pos[i, j] = top_r[i, j]
            topk_w[i, j] = np.float32(top_v[i, j])
        if m >= kk:
            kind[i] = 0
            continue
        kind[i] = 1
        p1 = top_r[i, 0] if m > 0 else 0
        p2 = top_r[i, 1] if m > 1 else 0
        key = (m * n + p1) * n + p2
        g = table[key]
        if g < 0:
            g = ng
            table[key] = g
            types[g, 0] = m
            types[g, 1] = p1
            types[g, 2] = p2
            ng += 1
        group[i] = g
    return ng


@njit(cache=True)
def _fill_topk(kk, kind, group, types, fill, topk_pos, class_ids, topk_z):
    """The places numpy's sort gave the tied zeros of each kind-1 bin (``fill``, per order
    type), then every bin's classes as charges (``class_ids``), int16 as the reference casts
    them."""
    L = kind.shape[0]
    k = topk_z.shape[1]
    for i in range(L):
        if kind[i] == 1:
            g = group[i]
            for j in range(types[g, 0], kk):
                topk_pos[i, j] = fill[g, j]
        for j in range(k):
            topk_z[i, j] = class_ids[topk_pos[i, j]]


@njit(cache=True)
def _median_is_zero(x):
    """Whether more than half of ``x`` is zero and none of it negative or NaN: then its median
    is zero (whichever zero), as numpy computes it."""
    c = 0
    for i in range(x.shape[0]):
        v = x[i]
        if not (v >= 0.0):
            return False
        if v == 0.0:
            c += 1
    return c >= x.shape[0] // 2 + 1


@njit(cache=True)
def _apex_mask(cmat, thr, mask):
    """``mask[i]`` = some row has ``row[i] >= row[i - 1]``, ``row[i] > row[i + 1]`` and
    ``row[i] >= thr``: the reference's apex bins. It skips the rows whose maximum is below
    ``thr``, which hold no such bin."""
    nz = cmat.shape[0]
    L = cmat.shape[1]
    for r in range(nz):
        for i in range(1, L - 1):
            v = cmat[r, i]
            if v >= thr and v >= cmat[r, i - 1] and v > cmat[r, i + 1]:
                mask[i] = True


@njit(cache=True)
def heat_envelope(apex, hw, kernel, heat):
    """For every apex bin ``b`` in order, ``heat[lo:hi] = np.maximum(heat[lo:hi],
    kernel[lo - b + hw:hi - b + hw])`` with ``lo, hi`` the window ``b - hw .. b + hw`` clipped to
    the array, as ``synth.labels.build_labels`` does with one numpy call per apex; numpy's rule
    (keep the first value when it is not less, or NaN), so equal."""
    n = heat.shape[0]
    for b in apex:
        lo = max(0, b - hw)
        hi = min(n, b + hw + 1)
        for t in range(lo, hi):
            h = heat[t]
            k = kernel[t - b + hw]
            if not (h >= k or np.isnan(h)):
                heat[t] = k


# ------------------------------------------------------------------------------ warm-up
def warm_up() -> None:
    """Compile (or load from numba's cache) every kernel of this module and of
    :mod:`lcmsdeconv.synth.render` with the argument types synthesis passes, so that data
    workers started afterwards find them compiled. One generated frame does not reach all of
    them (evidence channels, repeated axis points)."""
    if not HAVE_NUMBA:
        return
    from .render import _fill_sticks

    f = np.linspace(1.0, 2.0, 8)
    i = np.arange(8, dtype=np.int64)
    _add_samples(i, f, f, 0.5, 0, np.zeros(8))
    _add_nonzero_samples(i, f, f, 0.5, 0, np.zeros(8))
    _positive_finite(f)
    _widths(f[:7].copy(), f[:7].copy(), 3.0, np.empty(8))
    _has_repeats(f, 1e-9)
    _interp_index(f[:3].copy(), f[:3].copy(), np.empty(8))
    _nonzero_rates(f, 0.5, 2.0, np.empty(8, dtype=np.int64), np.empty(8))
    _scan_values(i, i, 2.0, f, np.empty(8))
    _threshold_segments(f.copy(), 1.5, 2)
    _poisson_skips_zeros()
    draws.available()  # the compiled draws' self-check, which compiles the scan kernel
    positive_points(f, f)
    table, count = _comb_table(2e-5, 3)
    _comb_best(f, table, count, 3, np.empty(8), np.empty(8, dtype=np.int64))
    _comb_best_tiled(f, table, count, 3, np.empty(8), np.empty(8, dtype=np.int64), COMB_TILE)
    _comb_best_sparse(f, table, count, 3, np.empty(8), np.empty(8, dtype=np.int64))
    _comb_rows(f, table, count, 3, np.zeros((3, 8)))
    heat_envelope(i, 1, f[:3].copy(), np.zeros(8))
    # every label kernel, with the arrays build_labels makes (an apex, ties of both kinds)
    rows = np.array([[0.0, 2.0, 1.0, 0.0, 3.0, 3.0, 0.0, 0.5],
                     [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    build_labels(rows, np.array([2, 3]), np.zeros(8), noise_level=0.1, observed=f)
    one = np.ones(1, dtype=np.int64)
    _fill_sticks(np.zeros(1, dtype=np.int64), one, f[:1].copy(), f[:1].copy(), f[:1].copy(),
                 f[:1].copy(), np.zeros(1, dtype=np.int64), one, f, f, one, 1e-9, np.empty(8),
                 np.empty(8), np.empty(8, dtype=np.int64), np.empty(8, dtype=np.int64),
                 np.empty(8))
