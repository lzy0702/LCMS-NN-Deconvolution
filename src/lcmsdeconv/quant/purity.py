"""Purity, potency and MS-purity calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.model import Peak, PeakTable, Species


@dataclass
class Calibration:
    """External-standard calibration or a plain response factor."""

    mode: str = "none"  # none | external | response_factor
    slope: float = 1.0
    intercept: float = 0.0
    r_squared: float = 0.0
    weighting: str = "none"
    amount_unit: str = "mg/mL"
    n_levels: int = 0

    def amount(self, area: float) -> float | None:
        if self.mode == "none":
            return None
        if self.mode == "response_factor":
            return area * self.slope
        if abs(self.slope) < 1e-15:
            return None
        return (area - self.intercept) / self.slope

    def to_dict(self) -> dict:
        return {"mode": self.mode, "slope": self.slope, "intercept": self.intercept,
                "r_squared": self.r_squared, "weighting": self.weighting,
                "amount_unit": self.amount_unit, "n_levels": self.n_levels}


def fit_calibration(levels: list[list[float]], weighting: str = "none",
                    force_zero: bool = False, amount_unit: str = "mg/mL") -> Calibration:
    """Least-squares calibration of area against amount, optionally 1/x or 1/x^2 weighted."""
    pts = np.asarray([lv for lv in levels if len(lv) >= 2], dtype=float)
    if pts.shape[0] < 2:
        return Calibration("none")
    amount, area = pts[:, 0], pts[:, 1]
    w = np.ones_like(amount)
    if weighting == "1/x":
        w = 1.0 / np.clip(np.abs(amount), 1e-12, None)
    elif weighting in ("1/x2", "1/x^2"):
        w = 1.0 / np.clip(amount**2, 1e-12, None)
    if force_zero:
        slope = float(np.sum(w * amount * area) / max(np.sum(w * amount**2), 1e-12))
        intercept = 0.0
    else:
        sw = w.sum()
        mx = np.sum(w * amount) / sw
        my = np.sum(w * area) / sw
        sxx = np.sum(w * (amount - mx) ** 2)
        sxy = np.sum(w * (amount - mx) * (area - my))
        slope = float(sxy / max(sxx, 1e-12))
        intercept = float(my - slope * mx)
    pred = slope * amount + intercept
    ss_res = float(np.sum(w * (area - pred) ** 2))
    ss_tot = float(np.sum(w * (area - np.average(area, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return Calibration("external", slope, intercept, r2, weighting, amount_unit, int(pts.shape[0]))


@dataclass
class PurityResult:
    signal: str
    rows: list[dict] = field(default_factory=list)
    main_peak_id: int | None = None
    total_area: float = 0.0

    def to_dict(self) -> dict:
        return {"signal": self.signal, "main_peak_id": self.main_peak_id,
                "total_area": self.total_area, "rows": self.rows}


def area_percent_purity(table: PeakTable, main_peak: Peak | None = None,
                        exclude_flags: tuple[str, ...] = ("solvent",)) -> PurityResult:
    """Classic area-% purity over a peak table."""
    peaks = [p for p in table.peaks if not any(f in p.flags for f in exclude_flags)]
    total = sum(p.area for p in peaks)
    if main_peak is None and peaks:
        main_peak = max(peaks, key=lambda p: p.area)
    rows = []
    for p in sorted(peaks, key=lambda x: x.rt):
        rows.append({
            "id": p.id, "name": p.name, "rt": p.rt, "area": p.area, "height": p.height,
            "area_pct": 100.0 * p.area / total if total > 0 else 0.0,
            "code": p.code, "width_half": p.width_half, "tailing": p.tailing,
            "is_main": main_peak is not None and p.id == main_peak.id,
            "flags": list(p.flags),
        })
    return PurityResult(table.signal, rows, main_peak.id if main_peak else None, total)


def ms_purity(species: list[Species], main: Species | None = None) -> dict:
    """Share of deconvolved ion signal belonging to the main species."""
    if not species:
        return {"main_percent": 0.0, "n_species": 0}
    total = sum(s.total_intensity for s in species)
    if main is None:
        main = max(species, key=lambda s: s.total_intensity)
    return {
        "main_percent": 100.0 * main.total_intensity / total if total > 0 else 0.0,
        "main_mass": main.mass,
        "n_species": len(species),
        "total_intensity": total,
    }


def potency(area: float, calibration: Calibration, sample_amount: float | None = None,
            dilution: float = 1.0) -> dict:
    """Amount (and % of nominal when a sample amount is given) from a calibrated area."""
    amt = calibration.amount(area)
    out = {"area": area, "amount": amt, "unit": calibration.amount_unit,
           "calibration": calibration.to_dict()}
    if amt is not None:
        amt = amt * dilution
        out["amount"] = amt
        if sample_amount:
            out["percent_of_nominal"] = 100.0 * amt / sample_amount
    return out
