"""Core data model shared by all stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._compat import trapezoid


@dataclass
class Spectrum:
    """One MS frame."""

    mz: np.ndarray
    intensity: np.ndarray
    rt: float  # minutes
    polarity: int = 1  # +1 / -1
    ms_level: int = 1
    scan_id: str = ""
    index: int = -1
    is_profile: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tic(self) -> float:
        return float(np.sum(self.intensity))

    @property
    def base_peak(self) -> float:
        return float(np.max(self.intensity)) if self.intensity.size else 0.0

    def sliced(self, mz_min: float | None, mz_max: float | None) -> Spectrum:
        lo = 0 if mz_min is None else int(np.searchsorted(self.mz, mz_min))
        hi = self.mz.size if mz_max is None else int(np.searchsorted(self.mz, mz_max, side="right"))
        return Spectrum(
            self.mz[lo:hi], self.intensity[lo:hi], self.rt, self.polarity, self.ms_level,
            self.scan_id, self.index, self.is_profile, dict(self.meta),
        )


@dataclass
class Chromatogram:
    time: np.ndarray  # minutes
    intensity: np.ndarray
    name: str = "TIC"
    kind: str = "tic"  # tic | bpc | uv | eic | deic | other
    unit: str = "counts"
    meta: dict[str, Any] = field(default_factory=dict)

    def sliced(self, t0: float, t1: float) -> Chromatogram:
        m = (self.time >= t0) & (self.time <= t1)
        return Chromatogram(self.time[m], self.intensity[m], self.name, self.kind, self.unit, dict(self.meta))

    def value_at(self, t: float) -> float:
        return float(np.interp(t, self.time, self.intensity))


@dataclass
class Run:
    """An LC-MS run: MS1 frames plus detector chromatograms (UV etc.)."""

    spectra: list[Spectrum]
    chromatograms: dict[str, Chromatogram] = field(default_factory=dict)
    name: str = ""
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- properties
    @property
    def rts(self) -> np.ndarray:
        return np.array([s.rt for s in self.spectra], dtype=float)

    @property
    def polarities(self) -> list[int]:
        return sorted({s.polarity for s in self.spectra})

    def frames(self, polarity: int | None = None, rt_range: tuple[float, float] | None = None) -> list[Spectrum]:
        out = self.spectra
        if polarity is not None:
            out = [s for s in out if s.polarity == polarity]
        if rt_range is not None:
            out = [s for s in out if rt_range[0] <= s.rt <= rt_range[1]]
        return out

    def tic(self, polarity: int | None = None) -> Chromatogram:
        fr = self.frames(polarity)
        t = np.array([s.rt for s in fr])
        y = np.array([s.tic for s in fr])
        return Chromatogram(t, y, "TIC", "tic")

    def bpc(self, polarity: int | None = None) -> Chromatogram:
        fr = self.frames(polarity)
        t = np.array([s.rt for s in fr])
        y = np.array([s.base_peak for s in fr])
        return Chromatogram(t, y, "BPC", "bpc")

    def uv_traces(self) -> list[Chromatogram]:
        return [c for c in self.chromatograms.values() if c.kind == "uv"]

    def eic(self, mz_lo: float, mz_hi: float, polarity: int | None = None) -> Chromatogram:
        fr = self.frames(polarity)
        t = np.array([s.rt for s in fr])
        y = np.empty(len(fr))
        for i, s in enumerate(fr):
            lo = np.searchsorted(s.mz, mz_lo)
            hi = np.searchsorted(s.mz, mz_hi, side="right")
            y[i] = float(np.sum(s.intensity[lo:hi]))
        return Chromatogram(t, y, f"EIC {mz_lo:.3f}-{mz_hi:.3f}", "eic")

    def sum_frames(self, frames: list[Spectrum]) -> Spectrum:
        """Sum spectra onto a common m/z axis (interpolating when axes differ)."""
        if not frames:
            raise ValueError("No frames to sum")
        ref = frames[0]
        same_axis = all(f.mz.size == ref.mz.size and np.allclose(f.mz, ref.mz) for f in frames[1:])
        if same_axis:
            total = np.sum([f.intensity for f in frames], axis=0)
            mz = ref.mz.copy()
        else:
            # union axis: take the densest axis as reference
            ref = max(frames, key=lambda f: f.mz.size)
            mz = ref.mz.copy()
            total = np.zeros(mz.size)
            for f in frames:
                if f.is_profile is False:
                    idx = np.clip(np.searchsorted(mz, f.mz), 0, mz.size - 1)
                    np.add.at(total, idx, f.intensity)
                else:
                    total += np.interp(mz, f.mz, f.intensity, left=0.0, right=0.0)
        rt = float(np.mean([f.rt for f in frames]))
        return Spectrum(mz, total.astype(np.float32), rt, ref.polarity, 1, "sum", -1, ref.is_profile,
                        {"n_frames": len(frames), "rt_range": (frames[0].rt, frames[-1].rt)})


# ---------------------------------------------------------------- deconvolution results
@dataclass
class Component:
    """One deconvolved species in one spectrum (base + adduct states aggregated)."""

    mass: float
    intensity: float
    mass_type: str = "average"  # average | mono
    charges: dict[int, float] = field(default_factory=dict)
    adducts: dict[str, float] = field(default_factory=dict)  # state label -> intensity
    score: float = 0.0
    mass_spread_ppm: float = 0.0
    compound_class: str = "peptide"
    flags: list[str] = field(default_factory=list)
    id: int = -1

    @property
    def charge_range(self) -> tuple[int, int]:
        if not self.charges:
            return (0, 0)
        return (min(self.charges), max(self.charges))

    def adduct_fractions(self) -> dict[str, float]:
        tot = sum(self.adducts.values())
        if tot <= 0:
            return {}
        return {k: v / tot for k, v in self.adducts.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mass": self.mass,
            "mass_type": self.mass_type,
            "intensity": self.intensity,
            "charges": {int(k): float(v) for k, v in self.charges.items()},
            "adducts": {k: float(v) for k, v in self.adducts.items()},
            "score": self.score,
            "mass_spread_ppm": self.mass_spread_ppm,
            "compound_class": self.compound_class,
            "flags": list(self.flags),
        }


@dataclass
class FrameResult:
    rt: float
    polarity: int
    components: list[Component]
    noise_sigma: float = 0.0
    residual_fraction: float = 0.0
    saturation_flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rt": self.rt,
            "polarity": self.polarity,
            "noise_sigma": self.noise_sigma,
            "residual_fraction": self.residual_fraction,
            "saturation_flags": list(self.saturation_flags),
            "components": [c.to_dict() for c in self.components],
        }


@dataclass
class Species:
    """A deconvolved neutral species traced across retention time."""

    id: int
    mass: float
    mass_type: str
    polarity: int
    compound_class: str
    time: np.ndarray
    intensity: np.ndarray  # deconvolved EIC
    adduct_intensity: dict[str, np.ndarray] = field(default_factory=dict)
    charges: dict[int, float] = field(default_factory=dict)
    score: float = 0.0
    mass_spread_ppm: float = 0.0
    flags: list[str] = field(default_factory=list)
    region_id: int = -1
    annotations: list[str] = field(default_factory=list)
    name: str = ""

    @property
    def total_intensity(self) -> float:
        return float(trapezoid(self.intensity, self.time)) if self.time.size > 1 else float(np.sum(self.intensity))

    @property
    def rt_apex(self) -> float:
        return float(self.time[int(np.argmax(self.intensity))]) if self.time.size else float("nan")

    @property
    def max_intensity(self) -> float:
        return float(np.max(self.intensity)) if self.intensity.size else 0.0

    def adduct_fractions(self) -> dict[str, float]:
        tot = {k: float(np.sum(v)) for k, v in self.adduct_intensity.items()}
        s = sum(tot.values())
        return {k: v / s for k, v in tot.items()} if s > 0 else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mass": self.mass,
            "mass_type": self.mass_type,
            "polarity": self.polarity,
            "compound_class": self.compound_class,
            "rt_apex": self.rt_apex,
            "max_intensity": self.max_intensity,
            "total_intensity": self.total_intensity,
            "adduct_fractions": self.adduct_fractions(),
            "charges": {int(k): float(v) for k, v in self.charges.items()},
            "score": self.score,
            "mass_spread_ppm": self.mass_spread_ppm,
            "flags": list(self.flags),
            "annotations": list(self.annotations),
            "region_id": self.region_id,
        }


# ---------------------------------------------------------------- chromatographic peaks
@dataclass
class Peak:
    rt: float
    start: float
    end: float
    area: float
    height: float
    width_half: float = 0.0
    symmetry: float = 0.0
    tailing: float = 0.0
    area_pct: float = 0.0
    code: str = "BB"
    baseline_start: float = 0.0
    baseline_end: float = 0.0
    id: int = -1
    name: str = ""
    flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def baseline_at(self, t: float) -> float:
        if self.end <= self.start:
            return self.baseline_start
        return self.baseline_start + (self.baseline_end - self.baseline_start) * (t - self.start) / (self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "rt": self.rt,
            "start": self.start,
            "end": self.end,
            "area": self.area,
            "height": self.height,
            "width_half": self.width_half,
            "symmetry": self.symmetry,
            "tailing": self.tailing,
            "area_pct": self.area_pct,
            "code": self.code,
            "flags": list(self.flags),
        }
        return d


@dataclass
class PeakTable:
    peaks: list[Peak]
    signal: str = ""
    area_unit: str = "signal*s"

    def __len__(self) -> int:
        return len(self.peaks)

    def __iter__(self):
        return iter(self.peaks)

    def total_area(self) -> float:
        return float(sum(p.area for p in self.peaks))

    def update_area_pct(self) -> None:
        tot = self.total_area()
        for p in self.peaks:
            p.area_pct = 100.0 * p.area / tot if tot > 0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows = [p.to_dict() for p in self.peaks]
        cols = ["id", "name", "rt", "start", "end", "area", "height", "width_half", "symmetry", "tailing", "area_pct", "code", "flags"]
        return pd.DataFrame(rows, columns=cols)

    def peak_at(self, t: float) -> Peak | None:
        for p in self.peaks:
            if p.start <= t <= p.end:
                return p
        return None
