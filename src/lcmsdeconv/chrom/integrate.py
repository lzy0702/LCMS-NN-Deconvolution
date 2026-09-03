"""Chromatographic peak integration with Agilent-CDS-style integration events.

Detection follows the classical bunched-and-smoothed derivative approach: the signal is
bunched to roughly a dozen points across the expected peak width, the first derivative is
compared with a slope-sensitivity threshold to mark peak starts and ends, apexes are the
downward zero crossings, and the baseline is built from the baseline points that detection
leaves behind. Timed events then override that construction the way an analyst would.

Baseline codes follow the usual two-letter convention: first letter for the peak start,
second for the end, ``B`` baseline, ``V`` valley, ``P`` penetrated, ``T`` tangent-skimmed,
``M`` manual, ``S`` shoulder, ``F`` forced/solvent.
"""

from __future__ import annotations

import numpy as np

from ..core._compat import trapezoid
from ..core.model import Chromatogram, Peak, PeakTable
from .events import build_schedule


def _bunch_and_smooth(y: np.ndarray, points_per_peak: float) -> np.ndarray:
    """Light smoothing scaled to the expected peak width (never wider than the peak)."""
    if y.size < 5:
        return y.astype(float)
    win = int(max(3, min(round(points_per_peak / 4.0), (y.size - 1) // 2)))
    if win % 2 == 0:
        win += 1
    if win <= 3:
        return y.astype(float)
    kernel = np.ones(win) / win
    pad = win // 2
    padded = np.pad(y.astype(float), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _auto_slope_sensitivity(dy: np.ndarray) -> float:
    """Derive a slope threshold from the noise of the derivative when none is given."""
    d = dy[np.isfinite(dy)]
    if d.size < 8:
        return 0.0
    mad = np.median(np.abs(d - np.median(d)))
    return float(5.0 * 1.4826 * mad)


def integrate_chromatogram(chrom: Chromatogram, settings, name: str | None = None) -> PeakTable:
    """Integrate one signal and return its peak table."""
    t = np.asarray(chrom.time, dtype=float)
    y = np.asarray(chrom.intensity, dtype=float)
    if t.size < 5:
        return PeakTable([], name or chrom.name, settings.area_unit)
    order = np.argsort(t)
    t, y = t[order], y[order]

    sched = build_schedule(t, settings)
    dt = float(np.median(np.diff(t))) or 1e-6
    peak_width = max(float(settings.peak_width), 3 * dt)
    ppp = max(4.0, peak_width / dt)

    ys = _bunch_and_smooth(y, ppp)
    dy = np.gradient(ys, t)

    slope_thr = float(settings.slope_sensitivity)
    if slope_thr <= 0:
        slope_thr = _auto_slope_sensitivity(dy)
    if slope_thr <= 0:
        slope_thr = 1e-12

    starts, apexes, ends, valleys = _detect(t, ys, dy, slope_thr, sched, peak_width)
    if not starts:
        return PeakTable([], name or chrom.name, settings.area_unit)

    peaks = _build_peaks(t, y, ys, starts, apexes, ends, valleys, sched, settings)
    peaks = _apply_peak_sum_slices(t, y, peaks, sched, settings)
    peaks = _apply_rejects(peaks, settings)
    table = PeakTable(peaks, name or chrom.name, settings.area_unit)
    table.update_area_pct()
    for i, p in enumerate(table.peaks):
        p.id = i + 1
    return table


def _detect(t, ys, dy, slope_thr, sched, peak_width):
    """State machine over the derivative: returns start/apex/end indices and valley flags."""
    n = t.size
    apexes: list[int] = []
    ends: list[int] = []
    valleys: list[tuple[bool, bool]] = []  # (start_is_valley, end_is_valley)

    i = 1
    in_peak = False
    apex = 0
    start_valley = False
    min_pts = max(2, int(round(peak_width / max(np.median(np.diff(t)), 1e-9) / 8)))
    while i < n - 1:
        on = sched.is_on(t[i])
        if not on:
            if in_peak:
                ends.append(i)
                apexes.append(apex)
                valleys.append((start_valley, False))
                in_peak = False
            i += 1
            continue
        if not in_peak:
            if dy[i] > slope_thr:
                in_peak = True
                apex = i
                start_valley = False
            i += 1
            continue
        # inside a peak
        if ys[i] > ys[apex]:
            apex = i
        going_up_again = dy[i] > slope_thr
        settled = abs(dy[i]) <= slope_thr
        if i > apex + min_pts and going_up_again:
            # valley: a new peak starts before the signal returns to baseline
            v = int(apex + np.argmin(ys[apex:i + 1]))
            ends.append(v)
            apexes.append(apex)
            valleys.append((start_valley, True))
            start_valley = True
            apex = i
            i += 1
            continue
        if i > apex + min_pts and settled and dy[i] >= -slope_thr:
            ends.append(i)
            apexes.append(apex)
            valleys.append((start_valley, False))
            in_peak = False
            i += 1
            continue
        i += 1
    if in_peak:
        ends.append(n - 1)
        apexes.append(apex)
        valleys.append((start_valley, False))
    # rebuild the starts list consistently with valley splits
    out_starts: list[int] = []
    cursor = None
    for k in range(len(ends)):
        if k == 0 or not valleys[k][0]:
            cursor = _find_start(ys, ends[k], apexes[k], out_starts)
        else:
            cursor = ends[k - 1]
        out_starts.append(cursor)
    return out_starts, apexes, ends, valleys


def _find_start(ys, end_idx, apex_idx, previous_starts):
    j = apex_idx
    lower = previous_starts[-1] if previous_starts else 0
    while j > lower and ys[j - 1] <= ys[j]:
        j -= 1
    return j


def _baseline_level(t, y, idx, sched):
    for a, b in sched.baseline_hold:
        if a <= t[idx] <= b:
            k = int(np.searchsorted(t, a))
            return float(y[min(k, y.size - 1)])
    return float(y[idx])


def _build_peaks(t, y, ys, starts, apexes, ends, valleys, sched, settings) -> list[Peak]:
    """Integrate detected peaks, grouping valley-joined peaks under a common baseline.

    Peaks that are not baseline-resolved form a cluster: one baseline is drawn from the start
    of the first peak to the end of the last, and the peaks inside are separated by a vertical
    drop line at the valley (Agilent's default). Turning on ``baseline_all_valleys`` instead
    resets the baseline at every valley, which is the valley-to-valley alternative.
    """
    to_seconds = 60.0 if str(settings.area_unit).endswith("s") else 1.0
    clusters: list[list[int]] = []
    for k in range(len(ends)):
        sv = valleys[k][0] if k < len(valleys) else False
        if sv and clusters:
            clusters[-1].append(k)
        else:
            clusters.append([k])

    peaks: list[Peak] = []
    for cluster in clusters:
        first, last = cluster[0], cluster[-1]
        s0, e1 = starts[first], ends[last]
        if e1 <= s0:
            continue
        all_valleys = bool(getattr(sched, "switches", {}).get("baseline_all_valleys", [False])[
            min(s0, len(t) - 1)]) if hasattr(sched, "switches") else settings.baseline_all_valleys
        for k in cluster:
            s, a, e = starts[k], apexes[k], ends[k]
            if e <= s:
                continue
            sv, ev = valleys[k] if k < len(valleys) else (False, False)
            if all_valleys:
                b_lo, b_hi = _baseline_level(t, y, s, sched), _baseline_level(t, y, e, sched)
                t_lo, t_hi = t[s], t[e]
            else:
                b_lo, b_hi = _baseline_level(t, y, s0, sched), _baseline_level(t, y, e1, sched)
                t_lo, t_hi = t[s0], t[e1]
            for bt in sched.baseline_now:
                if abs(bt - t[s]) <= 2 * (t[1] - t[0]):
                    b_lo, t_lo = float(y[s]), float(t[s])
            seg_t = t[s:e + 1]
            seg_y = y[s:e + 1]
            span = max(t_hi - t_lo, 1e-12)
            base = b_lo + (b_hi - b_lo) * (seg_t - t_lo) / span
            above = seg_y - base
            area = float(trapezoid(np.clip(above, 0, None), seg_t)) * to_seconds
            height = float(np.max(above)) if above.size else 0.0
            apex_t = float(seg_t[int(np.argmax(above))]) if above.size else float(t[a])
            code = ("V" if sv else "B") + ("V" if ev else "B")
            if np.any(above < -0.02 * max(height, 1e-12)):
                code = code[0] + "P"
            for st in sched.solvent_peak:
                if seg_t[0] <= st <= seg_t[-1]:
                    code = "F" + code[1]
            peaks.append(Peak(rt=apex_t, start=float(seg_t[0]), end=float(seg_t[-1]), area=area,
                              height=height, width_half=_width_at_fraction(seg_t, above, 0.5),
                              symmetry=_symmetry(seg_t, above, apex_t),
                              tailing=_tailing(seg_t, above, apex_t),
                              code=code, baseline_start=float(base[0]), baseline_end=float(base[-1])))
    peaks.sort(key=lambda p: p.rt)
    peaks = _apply_splits(t, y, peaks, sched, settings)
    peaks = _apply_tangent_skim(t, y, peaks, sched, settings)
    return peaks


def _width_at_fraction(seg_t, above, frac) -> float:
    if above.size < 3 or above.max() <= 0:
        return 0.0
    half = frac * above.max()
    idx = np.nonzero(above >= half)[0]
    if idx.size < 2:
        return 0.0
    return float(seg_t[idx[-1]] - seg_t[idx[0]])


def _symmetry(seg_t, above, apex_t) -> float:
    if above.size < 3 or above.max() <= 0:
        return 0.0
    half = 0.5 * above.max()
    idx = np.nonzero(above >= half)[0]
    if idx.size < 2:
        return 0.0
    left = apex_t - seg_t[idx[0]]
    right = seg_t[idx[-1]] - apex_t
    return float(right / left) if left > 0 else 0.0


def _tailing(seg_t, above, apex_t) -> float:
    """USP tailing factor: total width at 5 % height over twice the leading half-width."""
    if above.size < 3 or above.max() <= 0:
        return 0.0
    lvl = 0.05 * above.max()
    idx = np.nonzero(above >= lvl)[0]
    if idx.size < 2:
        return 0.0
    a = apex_t - seg_t[idx[0]]
    total = seg_t[idx[-1]] - seg_t[idx[0]]
    return float(total / (2 * a)) if a > 0 else 0.0


def _apply_splits(t, y, peaks, sched, settings) -> list[Peak]:
    if not sched.split_peak:
        return peaks
    to_seconds = 60.0 if str(settings.area_unit).endswith("s") else 1.0
    out: list[Peak] = []
    for p in peaks:
        cuts = [c for c in sched.split_peak if p.start < c < p.end]
        if not cuts:
            out.append(p)
            continue
        bounds = [p.start] + sorted(cuts) + [p.end]
        for a, b in zip(bounds[:-1], bounds[1:]):
            i0 = int(np.searchsorted(t, a))
            i1 = int(np.searchsorted(t, b))
            if i1 <= i0:
                continue
            seg_t, seg_y = t[i0:i1 + 1], y[i0:i1 + 1]
            base = np.linspace(p.baseline_at(a), p.baseline_at(b), seg_t.size)
            above = seg_y - base
            area = float(trapezoid(np.clip(above, 0, None), seg_t)) * to_seconds
            out.append(Peak(rt=float(seg_t[int(np.argmax(above))]), start=float(a), end=float(b),
                            area=area, height=float(np.max(above)),
                            width_half=_width_at_fraction(seg_t, above, 0.5),
                            code="VV", baseline_start=p.baseline_at(a), baseline_end=p.baseline_at(b),
                            flags=["split"]))
    return out


def _apply_tangent_skim(t, y, peaks, sched, settings) -> list[Peak]:
    """Skim a small rider off the tail (or front) of a much larger parent peak."""
    ratio = float(getattr(settings, "tail_skim_height_ratio", 0.0))
    front_ratio = float(getattr(settings, "front_skim_height_ratio", 0.0))
    if ratio <= 0 and front_ratio <= 0:
        return peaks
    to_seconds = 60.0 if str(settings.area_unit).endswith("s") else 1.0
    valley_ratio = float(getattr(settings, "skim_valley_ratio", 0.0)) / 100.0
    out = list(peaks)
    for i in range(1, len(out)):
        parent, child = out[i - 1], out[i]
        if child.code[0] != "V" or child.height <= 0:
            continue
        if ratio > 0 and parent.height / child.height >= ratio and parent.rt < child.rt:
            valley_y = float(np.interp(child.start, t, y))
            if valley_ratio > 0 and valley_y < valley_ratio * (child.height + child.baseline_start):
                continue
            i0 = int(np.searchsorted(t, child.start))
            i1 = int(np.searchsorted(t, child.end))
            if i1 <= i0:
                continue
            seg_t, seg_y = t[i0:i1 + 1], y[i0:i1 + 1]
            base = np.linspace(valley_y, float(np.interp(child.end, t, y)), seg_t.size)
            above = seg_y - base
            child.area = float(trapezoid(np.clip(above, 0, None), seg_t)) * to_seconds
            child.height = float(np.max(above))
            child.baseline_start, child.baseline_end = float(base[0]), float(base[-1])
            child.code = "T" + child.code[1]
            child.flags.append("tangent skim (rear)")
            parent.code = parent.code[0] + "T"
        elif front_ratio > 0 and child.height / max(parent.height, 1e-12) >= front_ratio and parent.rt < child.rt:
            parent.code = "T" + parent.code[1]
            parent.flags.append("tangent skim (front)")
    return out


def _apply_peak_sum_slices(t, y, peaks, sched, settings) -> list[Peak]:
    if not sched.peak_sum_slice:
        return peaks
    to_seconds = 60.0 if str(settings.area_unit).endswith("s") else 1.0
    out = [p for p in peaks if not any(a <= p.rt <= b for a, b in sched.peak_sum_slice)]
    for a, b in sched.peak_sum_slice:
        i0, i1 = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
        if i1 <= i0:
            continue
        seg_t, seg_y = t[i0:i1 + 1], y[i0:i1 + 1]
        base = np.linspace(seg_y[0], seg_y[-1], seg_t.size)
        above = seg_y - base
        out.append(Peak(rt=float(seg_t[int(np.argmax(above))]), start=float(a), end=float(b),
                        area=float(trapezoid(np.clip(above, 0, None), seg_t)) * to_seconds,
                        height=float(np.max(above)), code="MM",
                        baseline_start=float(seg_y[0]), baseline_end=float(seg_y[-1]),
                        flags=["peak sum slice"]))
    out.sort(key=lambda p: p.rt)
    return out


def _apply_rejects(peaks, settings) -> list[Peak]:
    keep = [p for p in peaks
            if p.area >= settings.area_reject and p.height >= settings.height_reject]
    if settings.max_area > 0:
        keep = [p for p in keep if p.area <= settings.max_area]
    if settings.max_height > 0:
        keep = [p for p in keep if p.height <= settings.max_height]
    if settings.area_pct_reject > 0 and keep:
        total = sum(p.area for p in keep)
        if total > 0:
            keep = [p for p in keep if 100.0 * p.area / total >= settings.area_pct_reject]
    return keep


def manual_integrate(chrom: Chromatogram, start: float, end: float,
                     baseline_start: float | None = None, baseline_end: float | None = None,
                     area_unit: str = "signal*s") -> Peak:
    """Integrate a user-drawn peak between two times with an optional manual baseline."""
    t = np.asarray(chrom.time, dtype=float)
    y = np.asarray(chrom.intensity, dtype=float)
    i0, i1 = int(np.searchsorted(t, start)), int(np.searchsorted(t, end))
    i0, i1 = max(0, i0), min(t.size - 1, i1)
    seg_t, seg_y = t[i0:i1 + 1], y[i0:i1 + 1]
    b0 = float(seg_y[0]) if baseline_start is None else float(baseline_start)
    b1 = float(seg_y[-1]) if baseline_end is None else float(baseline_end)
    base = np.linspace(b0, b1, seg_t.size)
    above = seg_y - base
    to_seconds = 60.0 if str(area_unit).endswith("s") else 1.0
    return Peak(rt=float(seg_t[int(np.argmax(above))]), start=float(seg_t[0]), end=float(seg_t[-1]),
                area=float(trapezoid(np.clip(above, 0, None), seg_t)) * to_seconds,
                height=float(np.max(above)),
                width_half=_width_at_fraction(seg_t, above, 0.5), code="MM",
                baseline_start=b0, baseline_end=b1, flags=["manual"])
