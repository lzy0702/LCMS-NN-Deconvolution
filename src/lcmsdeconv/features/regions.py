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
    min_rel_height: float = 0.002,
    min_rel_area: float = 0.0005,
    min_rel_tic: float = 0.001,
    min_snr: float = 10.0,
) -> list[Region]:
    """Split a run into regions around detected chromatographic peaks.

    Uses the integrator when an :class:`IntegrationSettings` is supplied (peaks on the TIC or
    UV), otherwise falls back to fixed windows covering the whole run.

    A peak is kept when it stands ``min_snr`` times above the trace's noise. A 2 % impurity
    eluting on its own (the non-conjugated product in the IP-HILIC conjugate run) makes a TIC
    peak four times the noise, which ten times drops; the oligonucleotide methods use three,
    where the decoder's own tests judge what such a region holds (on synthetic protein runs
    three added false species, so the default stays ten). The relative
    floors (``min_rel_height`` of the tallest peak, ``min_rel_area`` of the total) are kept low
    on purpose: a second component at a few percent of the main one is an ordinary sample, and a
    floor tied to the tallest peak silently deletes it before deconvolution has a chance to
    look. Neighbouring peaks become separate regions split at the valley between them, so each
    region's summed spectrum is dominated by one elution band; only peaks whose apexes are
    closer than their widths are deconvolved together.
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
            peaks = sorted(table.peaks, key=lambda p: p.rt)
            if peaks:
                top = max(p.height for p in peaks)
                total = sum(p.area for p in peaks)
                floor = max(min_rel_height * top, min_snr * _trace_noise(chrom.intensity))
                peaks = [p for p in peaks
                         if p.height >= floor and (total <= 0 or p.area >= min_rel_area * total)]
            windows = _peak_windows(peaks, margin_min)
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

    # Deconvolving a stretch of baseline only produces envelopes fitted to noise, so regions
    # that carry a negligible share of the run's ion current are dropped.
    if regions and min_rel_tic > 0:
        totals = np.array([sum(f.tic for f in r.frames) for r in regions], dtype=float)
        if totals.max() > 0:
            keep = totals >= min_rel_tic * totals.max()
            regions = [r for r, k in zip(regions, keep) if k]
    for i, r in enumerate(regions):
        r.id = i
    return regions


def _trace_noise(y: np.ndarray) -> float:
    """Robust noise level of a chromatogram from its first differences (peaks are sparse)."""
    y = np.asarray(y, dtype=float)
    if y.size < 4:
        return 0.0
    d = np.diff(y)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def _peak_windows(peaks, margin_min: float) -> list[tuple[float, float]]:
    """Retention-time windows, one per peak, split at the valley between neighbours.

    Peaks closer than their own widths (apex separation below the larger half-height width)
    are one co-elution and share a window.
    """
    if not peaks:
        return []
    groups: list[list] = [[peaks[0]]]
    for p in peaks[1:]:
        prev = groups[-1][-1]
        width = max(prev.width_half, p.width_half, 1e-6)
        if p.rt - prev.rt < width:
            groups[-1].append(p)
        else:
            groups.append([p])
    spans = [(min(p.start for p in g), max(p.end for p in g)) for g in groups]
    out: list[tuple[float, float]] = []
    for i, (a, b) in enumerate(spans):
        lo, hi = a - margin_min, b + margin_min
        if i > 0:
            prev_end = spans[i - 1][1]
            lo = max(lo, 0.5 * (prev_end + a)) if prev_end + margin_min > a - margin_min else lo
        if i + 1 < len(spans):
            next_start = spans[i + 1][0]
            hi = min(hi, 0.5 * (b + next_start)) if b + margin_min > next_start - margin_min else hi
        out.append((lo, hi))
    return out


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
