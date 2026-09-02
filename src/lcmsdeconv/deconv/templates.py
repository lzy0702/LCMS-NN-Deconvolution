"""Build isotope-envelope templates for (mass, charge, adduct) on the log(m/z) grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chem.adducts import carrier_mass
from ..chem.classes import class_isotope_pattern
from ..chem.instrument import InstrumentModel


@dataclass
class Template:
    bins: np.ndarray  # int grid-bin indices
    values: np.ndarray  # area-normalized weights (sum ~ 1)
    z: int
    adduct_mass: float
    adduct_label: str
    mass: float

    @property
    def apex_bin(self) -> int:
        return int(self.bins[int(np.argmax(self.values))]) if self.bins.size else -1


def build_template(
    mass: float,
    z: int,
    grid,
    instrument: InstrumentModel,
    compound_class: str = "peptide",
    adduct_mass: float = 0.0,
    adduct_label: str = "",
    threshold: float = 1e-3,
) -> Template | None:
    """Area-normalized envelope of a class-average composition of ``mass`` at charge ``z``."""
    carrier = carrier_mass(grid.polarity)
    pat = class_isotope_pattern(mass, compound_class, threshold=threshold)
    offset = mass - pat.average_mass
    iso_mass = pat.masses + offset + adduct_mass
    mz = (iso_mass + z * carrier) / z
    valid = (mz > grid.mz_min) & (mz < grid.mz_max)
    if not np.any(valid):
        return None
    mz = mz[valid]
    abund = pat.abundances[valid]
    u = grid.mz_to_u(mz)
    centers = (u - grid.u_min) / grid.step
    sigma_mz = instrument.sigma_mz(mz)
    sigma_bins = np.maximum(sigma_mz / (mz - carrier) / grid.step, 0.4)
    kf = 4.0 if instrument.shape == "gaussian" else 10.0
    acc: dict[int, float] = {}
    for c, s, a in zip(centers, sigma_bins, abund):
        amp = a / (np.sqrt(2 * np.pi) * s)
        hw = int(kf * s) + 1
        lo, hi = int(c) - hw, int(c) + hw + 1
        for b in range(max(0, lo), min(grid.size, hi)):
            acc[b] = acc.get(b, 0.0) + amp * np.exp(-0.5 * ((b - c) / s) ** 2)
    if not acc:
        return None
    bins = np.fromiter(acc.keys(), dtype=np.int64)
    vals = np.fromiter(acc.values(), dtype=np.float64)
    order = np.argsort(bins)
    bins, vals = bins[order], vals[order]
    total = vals.sum()
    if total <= 0:
        return None
    vals = vals / total
    return Template(bins, vals, z, adduct_mass, adduct_label, mass)


def template_mz(mass: float, z: int, polarity: int, adduct_mass: float = 0.0) -> float:
    carrier = carrier_mass(polarity)
    return (mass + adduct_mass + z * carrier) / z
