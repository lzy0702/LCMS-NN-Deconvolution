"""One-point lock-mass correction from a Waters run's lockspray scans.

A Waters run interleaves its MS scans with a few scans of a reference compound sprayed apart, the
lockspray (Leu-enkephalin: [M-H]- 554.2615, [M+H]+ 556.2766). :func:`lcmsdeconv.io.mzml.read_mzml`
keeps those scans in ``run.meta["lockspray"]``. Here the reference peak is located in each of
them (a reading more than :data:`MAX_PPM` off is another ion and is skipped; readings at one time
are taken at their median), and the ratio of its observed to its true m/z is interpolated over the
run's time (held constant before the first reading and after the last) to one time, the
TIC-weighted mean retention time of the run's MS scans (of those within the analysis window, where
one is set: an injection front or a column wash outside it does not pull the time away from the
species analysed); every MS scan's m/z is divided by that one ratio. One ratio for the whole run, not one per scan: the scans of a region are summed point by
point where they share one m/z axis (:meth:`lcmsdeconv.core.model.Run.sum_frames`), and scans
scaled each by its own ratio no longer share one, so their sum spreads every peak over the scans'
several axes. The instrument's drift is left in the masses: a scan keeps the drift between its own
time and the time the ratio is read at, up to the whole spread of the readings (``drift_ppm``) at
the end of the run farthest from that time, about half of it at each end where the ion current is
centred in the run.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..core.model import Run

#: the reference peak is looked for within this many m/z of the reference
HALF_WINDOW = 0.5
#: and must stand this many times above the window's baseline: the median of its points with
#: signal more than two half-height widths away from the peak
MIN_SNR = 10.0
#: a reading further than this from the reference (ppm) is not the reference but another ion or a
#: stray point within the half window (a background ion 0.30 m/z below 554.2615 reads 541 ppm
#: off), where an instrument in calibration reads it within a few tens of ppm: it is skipped
MAX_PPM = 100.0
#: readings at one time that spread wider than this (ppm) are named in a warning
MAX_SPREAD_PPM = 5.0


def reference_peak(mz, intensity, reference: float, half_window: float = HALF_WINDOW,
                   min_snr: float = MIN_SNR) -> float | None:
    """The observed m/z of the reference peak in one lockspray scan, None when the scan does not
    show it.

    The peak is the most intense point within ``half_window`` m/z of ``reference``; its m/z is
    the centroid of its part above its half-height points: of the profile between them, joined
    point to point by straight lines, less half its height. Each half-height point is
    interpolated linearly between the profile points on either side of half the peak's height.
    At three points a half-height width this reads a Gaussian peak within 0.05 ppm at any
    sampling phase, where the midpoint of the half-height points errs by up to 0.7 ppm and a
    centroid of the sampled points above half height by up to 9 ppm. A peak with a
    single point above half its height (a centroided scan, or a profile sampled more sparsely
    than its peaks are wide) is read at that point. The scan shows no peak when no point with
    signal lies in the window, when the peak is cut by the window's edge (it does not fall to
    half its height on one side within the window), or when it stands less than ``min_snr``
    times above the window's baseline: the median of its points with signal more than two
    half-height widths from the peak (noise: the lockspray was off). A scan holding fewer than
    three such points (thresholded to its peaks) has no baseline to judge.
    """
    mz = np.asarray(mz, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    sel = (mz >= reference - half_window) & (mz <= reference + half_window)
    order = np.argsort(mz[sel], kind="stable")
    x, y = mz[sel][order], y[sel][order]
    n = x.size
    if n == 0:
        return None
    j = int(np.argmax(y))
    top = float(y[j])
    if not top > 0:
        return None
    half = 0.5 * top
    left = j
    while left > 0 and y[left - 1] > half:
        left -= 1
    right = j
    while right < n - 1 and y[right + 1] > half:
        right += 1
    if left == right:
        # one point above half its height, read where it lies; one at the window's edge is the
        # flank of a peak beyond it
        if n > 1 and j in (0, n - 1):
            return None
        centre = xl = xr = float(x[j])
    elif left == 0 or right == n - 1:
        return None  # no half-height point on one side within the window
    else:
        xl = float(np.interp(half, [y[left - 1], y[left]], [x[left - 1], x[left]]))
        xr = float(np.interp(half, [y[right + 1], y[right]], [x[right + 1], x[right]]))
        centre = _top_centroid(np.concatenate([[xl], x[left:right + 1], [xr]]),
                               np.concatenate([[0.0], y[left:right + 1] - half, [0.0]]))
    width = xr - xl
    beyond = y[((x < xl - 2.0 * width) | (x > xr + 2.0 * width)) & (y > 0)]
    if beyond.size >= 3 and top < min_snr * float(np.median(beyond)):
        return None
    return centre


def _top_centroid(x: np.ndarray, y: np.ndarray) -> float:
    """The centroid of the polygon under the points (x, y) joined by straight lines: the integrals
    of x y and of y, exact for each linear segment."""
    x0, x1, y0, y1 = x[:-1], x[1:], y[:-1], y[1:]
    h = x1 - x0
    area = float(np.sum(h * (y0 + y1))) / 2.0
    moment = float(np.sum(h * (x0 * (2.0 * y0 + y1) + x1 * (y0 + 2.0 * y1)))) / 6.0
    return moment / area


def lock_mass_readings(scans, reference: float, half_window: float = HALF_WINDOW
                       ) -> tuple[list[tuple[float, float]], int]:
    """(rt, observed m/z) of the reference in each lockspray scan ``scans`` = [(rt, m/z,
    intensity)] that shows it (:func:`reference_peak`), and the number of scans that do not."""
    readings: list[tuple[float, float]] = []
    skipped = 0
    for rt, mz, intensity in scans:
        obs = reference_peak(mz, intensity, reference, half_window)
        if obs is None:
            skipped += 1
        else:
            readings.append((float(rt), obs))
    return readings, skipped


def _by_time(readings: list[tuple[float, float]]) -> dict[float, list[float]]:
    """The readings' observed m/z by their time, in time order."""
    out: dict[float, list[float]] = {}
    for rt, obs in sorted(readings):
        out.setdefault(rt, []).append(obs)
    return out


def _ratios_by_time(readings: list[tuple[float, float]], reference: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The readings' times, in order, and observed / reference at each: the median of the
    readings at one time, which one stray reading among them does not move."""
    groups = _by_time(readings)
    ratios = [float(np.median(obs)) / reference for obs in groups.values()]
    return np.asarray(list(groups), dtype=float), np.asarray(ratios, dtype=float)


def _listed(readings: list[tuple[float, float]], reference: float) -> str:
    """Readings as text, by time ("-541 ppm at 1.00 min (3 scans)")."""
    parts = []
    for t, obs in _by_time(readings).items():
        lo, hi = (f"{(o / reference - 1.0) * 1e6:+.0f}" for o in (min(obs), max(obs)))
        parts.append((f"{lo} ppm" if lo == hi else f"{lo} to {hi} ppm") + f" at {t:.2f} min"
                     + (f" ({len(obs)} scans)" if len(obs) > 1 else ""))
    return ", ".join(parts)


def tic_weighted_rt(spectra) -> float | None:
    """The TIC-weighted mean retention time of ``spectra`` (their plain mean where none holds
    signal); None for no spectra."""
    if not spectra:
        return None
    rt = np.array([s.rt for s in spectra], dtype=float)
    w = np.array([s.tic for s in spectra], dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = float(np.sum(w))
    return float(np.sum(rt * w) / total) if total > 0 else float(np.mean(rt))


def lock_mass_corrected(run: Run, reference: float, half_window: float = HALF_WINDOW,
                        max_ppm: float = MAX_PPM, rt_range: tuple[float, float] | None = None
                        ) -> tuple[Run, dict | None, list[str]]:
    """``run`` with every MS scan's m/z divided by one ratio, observed / reference: the observed
    m/z of the lock mass ``reference`` read in the run's lockspray scans
    (``run.meta["lockspray"]``), interpolated over time to the TIC-weighted mean retention time of
    the run's MS scans (held constant before the first reading and after the last); with
    ``rt_range`` (start, end minutes: the analysis window) the time is weighted over the MS scans
    within it, the whole run's where none lies within it. One ratio for every scan keeps scans
    that share one m/z axis on one axis, as summing a region's scans needs (the module's docstring
    says why); the instrument's drift between the readings is left in.

    A scan's reading further than ``max_ppm`` from the reference is not the reference (another
    ion in the window, or a stray point of a thresholded scan) and is skipped, with a warning
    naming it; readings at one time are taken at their median, with a warning where they spread
    wider than :data:`MAX_SPREAD_PPM`.

    Returns (run, record, warnings). The run returned is a new one (``run`` itself is left as it
    was) whose ``meta["lock_mass"]`` holds the record: the reference (``reference_mz``), the
    reading of each scan that showed it (``readings``: rt, observed m/z, ppm), the lockspray scans
    (``scans``) and those skipped (``skipped``: without the reference, or read beyond
    ``max_ppm``), the time the ratio was read at (``tic_weighted_rt``), ``applied_ppm``, the
    scaling applied to every scan's m/z in ppm (negative where the instrument read high), and
    ``drift_ppm``, how far the readings moved over the run (the largest less the smallest of
    their ppm at their times). A run corrected already is returned as it is, with its record, so
    the correction is never applied twice (with a warning when ``reference`` is another m/z).
    Where the run has no lockspray scans, or none shows the reference within ``max_ppm``, the run
    is returned uncorrected with no record and a warning.
    """
    reference = float(reference)
    if not reference > 0:
        raise ValueError(f"lock mass {reference!r}: an m/z must be positive")
    done = run.meta.get("lock_mass")
    if done:
        warnings = []
        if abs(float(done.get("reference_mz", reference)) - reference) > 1e-6:
            warnings.append(f"lock mass {reference:.4f}: the run's m/z were corrected with "
                            f"{float(done['reference_mz']):.4f} already and are not corrected "
                            "again")
        return run, done, warnings
    scans = run.meta.get("lockspray") or []
    if not scans:
        return run, None, [f"lock mass {reference:.4f} is set, but the run has no lockspray "
                           "scans: m/z not corrected"]
    found, skipped = lock_mass_readings(scans, reference, half_window)
    readings = [(rt, obs) for rt, obs in found if abs(obs / reference - 1.0) * 1e6 <= max_ppm]
    beyond = [(rt, obs) for rt, obs in found if abs(obs / reference - 1.0) * 1e6 > max_ppm]
    if not readings:
        near = (f" (the peaks read nearest it: {_listed(beyond, reference)})" if beyond else "")
        within = f"{max_ppm:g} ppm" if beyond else f"{half_window:g} m/z"
        return run, None, [f"lock mass {reference:.4f}: none of the {len(scans)} lockspray scans "
                           f"shows it within {within}{near}: m/z not corrected"]
    warnings = []
    if beyond:
        warnings.append(f"lock mass {reference:.4f}: {len(beyond)} lockspray "
                        f"{'reading' if len(beyond) == 1 else 'readings'} more than {max_ppm:g} "
                        f"ppm off skipped as another ion or a stray point "
                        f"({_listed(beyond, reference)})")
    for t, obs in _by_time(readings).items():
        spread = (max(obs) - min(obs)) / reference * 1e6
        if spread > MAX_SPREAD_PPM:
            warnings.append(f"lock mass {reference:.4f}: the {len(obs)} readings at {t:.2f} min "
                            f"spread {spread:.1f} ppm; their median is used")
    skipped += len(beyond)
    times, ratios = _ratios_by_time(readings, reference)
    within = ([s for s in run.spectra if rt_range[0] <= s.rt <= rt_range[1]]
              if rt_range is not None else [])
    when = tic_weighted_rt(within or run.spectra)
    if when is None:
        when = float(np.mean(times))
    factor = 1.0 / float(np.interp(when, times, ratios))
    spectra = [dataclasses.replace(s, mz=(s.mz * factor).astype(s.mz.dtype, copy=False),
                                   meta=dict(s.meta))
               for s in run.spectra]
    record = {
        "reference_mz": reference,
        "readings": [{"rt": round(rt, 4), "mz": round(obs, 5),
                      "ppm": round((obs / reference - 1.0) * 1e6, 2)} for rt, obs in readings],
        "scans": len(scans),
        "skipped": skipped,
        "tic_weighted_rt": round(when, 4),
        "applied_ppm": round((factor - 1.0) * 1e6, 2),
        "drift_ppm": round(float(np.ptp(ratios)) * 1e6, 2),
    }
    corrected = Run(spectra, dict(run.chromatograms), run.name, run.source,
                    {**run.meta, "lock_mass": record})
    return corrected, record, warnings


def lock_mass_text(record: dict) -> str:
    """A lock-mass record (:func:`lock_mass_corrected`) as text: the scaling applied, the
    readings' spread over the run, and the lockspray scans read."""
    return (f"m/z scaled by {record['applied_ppm']:+.1f} ppm (readings spread "
            f"{record['drift_ppm']:.1f} ppm over the run; {record['scans'] - record['skipped']} of "
            f"{record['scans']} lockspray scans read)")
