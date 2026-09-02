"""Find chromatographic regions worth deconvolving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.model import Chromatogram, Run, Spectrum


@dataclass
class Region:
    id: int
    start: float
    end: float
    polarity: int
    frames: list[Spectrum]

    @property
    def rt(self) -> float:
        return 0.5 * (self.start + self.end)

    def summed(self, run: Run) -> Spectrum:
        return run.sum_frames(self.frames)


def find_regions(
    run: Run,
    source: str = "tic",
    polarity: int | None = None,
    window_min: float = 0.5,
    margin_min: float = 0.05,
    min_frames: int = 2,
    integration=None,
    min_rel_height: float = 0.01,
    min_rel_area: float = 0.001,
) -> list[Region]:
    """Split a run into regions around detected chromatographic peaks.

    Uses the integrator when an :class:`IntegrationSettings` is supplied (peaks on the TIC or
    UV), otherwise falls back to fixed windows covering the whole run.
    """
    frames = run.frames(polarity)
    if not frames:
        return []
    times = np.array([f.rt for f in frames])

    windows: list[tuple[float, float]] = []
    if source in ("tic", "uv", "bpc"):
        chrom = _source_chromatogram(run, source, polarity)
        if chrom is not None and chrom.time.size > 3 and integration is not None:
            from ..chrom.integrate import integrate_chromatogram

            table = integrate_chromatogram(chrom, integration)
            peaks = list(table.peaks)
            if peaks:
                # ignore baseline ripple: a region must carry a real share of the signal
                top = max(p.height for p in peaks)
                total = sum(p.area for p in peaks)
                peaks = [p for p in peaks
                         if p.height >= min_rel_height * top
                         and (total <= 0 or p.area >= min_rel_area * total)]
            windows = _merge_windows([(p.start - margin_min, p.end + margin_min) for p in peaks])
    if not windows:
        t0, t1 = float(times.min()), float(times.max())
        n = max(1, int(np.ceil((t1 - t0) / max(window_min, 1e-6))))
        edges = np.linspace(t0, t1, n + 1)
        windows = list(zip(edges[:-1], edges[1:]))

    regions: list[Region] = []
    for i, (a, b) in enumerate(windows):
        sel = [f for f in frames if a <= f.rt <= b]
        if len(sel) < min_frames:
            continue
        pol = sel[0].polarity
        regions.append(Region(i, float(a), float(b), pol, sel))
    return regions


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping retention-time windows so co-eluting peaks share one deconvolution."""
    if not windows:
        return []
    out: list[list[float]] = []
    for a, b in sorted(windows):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _source_chromatogram(run: Run, source: str, polarity: int | None) -> Chromatogram | None:
    if source == "tic":
        return run.tic(polarity)
    if source == "bpc":
        return run.bpc(polarity)
    if source == "uv":
        uv = run.uv_traces()
        return uv[0] if uv else run.tic(polarity)
    return None
