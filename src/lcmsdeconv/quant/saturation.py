"""Detect detector and ESI saturation, using the UV trace as a linear reference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.model import Chromatogram, Peak, Run, Spectrum


@dataclass
class SaturationReport:
    detector_frames: list[float] = field(default_factory=list)  # RTs with clipped spectra
    detector_level: float | None = None
    uv_delay_min: float = 0.0
    peak_flags: dict[int, list[str]] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def any_saturation(self) -> bool:
        return bool(self.detector_frames or self.peak_flags)

    def to_dict(self) -> dict:
        return {
            "detector_frames": self.detector_frames[:50],
            "n_detector_frames": len(self.detector_frames),
            "detector_level": self.detector_level,
            "uv_delay_min": self.uv_delay_min,
            "peak_flags": {int(k): v for k, v in self.peak_flags.items()},
            "messages": list(self.messages),
        }


def detect_detector_saturation(
    spectra: list[Spectrum], flat_top_points: int = 3, level: float | None = None
) -> tuple[list[float], float | None]:
    """Frames whose most intense peaks are clipped at a common ceiling (flat-topped)."""
    if not spectra:
        return [], None
    maxima = np.array([s.base_peak for s in spectra if s.intensity.size])
    if maxima.size == 0:
        return [], None
    ceiling = level
    if ceiling is None:
        top = np.percentile(maxima, 99)
        # a hard ceiling shows up as many frames sharing nearly the same maximum
        near = np.abs(maxima - top) / max(top, 1e-9) < 0.01
        ceiling = float(top) if near.sum() >= max(3, 0.02 * maxima.size) else None
    flagged: list[float] = []
    for s in spectra:
        if s.intensity.size < 5:
            continue
        m = s.base_peak
        if m <= 0:
            continue
        thr = 0.995 * (ceiling if ceiling else m)
        run = _longest_run(s.intensity >= thr)
        if ceiling is not None and m >= 0.995 * ceiling and run >= flat_top_points:
            flagged.append(s.rt)
        elif ceiling is None and run >= flat_top_points * 2:
            flagged.append(s.rt)
    return flagged, ceiling


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))) != 0)
    return int(np.max(idx[1::2] - idx[0::2])) if idx.size else 0


def estimate_uv_delay(tic: Chromatogram, uv: Chromatogram, max_delay: float = 0.6) -> float:
    """Cross-correlation delay (minutes) between the UV and MS traces."""
    if tic.time.size < 5 or uv.time.size < 5:
        return 0.0
    t0 = max(tic.time.min(), uv.time.min())
    t1 = min(tic.time.max(), uv.time.max())
    if t1 <= t0:
        return 0.0
    n = max(256, min(4096, tic.time.size))
    grid = np.linspace(t0, t1, n)
    a = np.interp(grid, tic.time, tic.intensity)
    b = np.interp(grid, uv.time, uv.intensity)
    a = a - a.mean()
    b = b - b.mean()
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    dt = grid[1] - grid[0]
    max_lag = int(max_delay / dt)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.dot(a, np.roll(b, int(k))) for k in lags])
    best = lags[int(np.argmax(corr))]
    return float(best * dt)


def detect_esi_saturation(
    tic: Chromatogram,
    uv: Chromatogram | None,
    peaks: list[Peak],
    ratio_drop: float = 0.25,
    flat_top_fraction: float = 0.2,
    uv_delay: float | None = None,
) -> tuple[dict[int, list[str]], float, list[str]]:
    """Flag peaks whose MS response compresses at the apex while UV stays linear.

    The MS-to-UV response ratio is computed across each peak; if the ratio at the apex falls
    well below the ratio on the flanks, the ionisation (not the column) is the limiting step,
    which is what ESI saturation looks like. Without a UV trace, only a flat-topped TIC is used.
    """
    flags: dict[int, list[str]] = {}
    messages: list[str] = []
    delay = 0.0
    if uv is not None and uv.time.size > 4:
        delay = estimate_uv_delay(tic, uv, ) if uv_delay is None else float(uv_delay)
    for p in peaks:
        pf: list[str] = []
        m = (tic.time >= p.start) & (tic.time <= p.end)
        if m.sum() >= 5:
            seg = tic.intensity[m]
            top = seg.max()
            if top > 0:
                frac = float(np.mean(seg >= 0.98 * top))
                if frac >= flat_top_fraction:
                    pf.append(f"flat-topped TIC over {frac*100:.0f} % of the peak")
        if uv is not None and uv.time.size > 4:
            um = (uv.time - delay >= p.start) & (uv.time - delay <= p.end)
            if m.sum() >= 5 and um.sum() >= 5:
                tt = tic.time[m]
                ms = tic.intensity[m]
                uvv = np.interp(tt, uv.time - delay, uv.intensity)
                if uvv.max() > 0 and ms.max() > 0:
                    ms_n = ms / ms.max()
                    uv_n = uvv / uvv.max()
                    apex = int(np.argmax(uv_n))
                    ok = uv_n > 0.2
                    if ok.sum() >= 5:
                        ratio = np.where(ok, ms_n / np.clip(uv_n, 1e-9, None), np.nan)
                        apex_r = float(np.nanmean(ratio[max(0, apex - 1):apex + 2]))
                        flank = np.nanpercentile(ratio, 80)
                        if flank > 0 and (flank - apex_r) / flank > ratio_drop:
                            pf.append(
                                f"MS response {100*(flank-apex_r)/flank:.0f} % lower at the apex "
                                "than on the flanks relative to UV (ESI saturation)"
                            )
        if pf:
            flags[p.id] = pf
    if flags:
        messages.append(
            "Saturation detected: quantify these peaks from UV, or dilute and re-inject. "
            "MS areas for flagged peaks under-report the true amount."
        )
    return flags, delay, messages


def analyze_saturation(
    run: Run, peaks: list[Peak], settings, polarity: int | None = None,
    uv: Chromatogram | None = None,
) -> SaturationReport:
    spectra = run.frames(polarity)
    frames, level = detect_detector_saturation(spectra, level=settings.detector_level)
    tic = run.tic(polarity)
    if uv is None:
        uvs = run.uv_traces()
        uv = uvs[0] if uvs else None
    flags, delay, messages = detect_esi_saturation(
        tic, uv, peaks, settings.esi_ratio_drop, settings.flat_top_fraction, settings.uv_delay_min
    )
    if frames:
        messages.append(
            f"{len(frames)} MS frames show a clipped (flat-topped) detector signal; "
            "isotope ratios and intensities in those frames are unreliable."
        )
    return SaturationReport(frames, level, delay, flags, messages)
