"""Build isotope-envelope templates for (mass, charge, adduct) on the log(m/z) grid."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..chem.adducts import carrier_mass
from ..chem.classes import class_isotope_pattern
from ..chem.instrument import InstrumentModel


@dataclass
class Template:
    bins: np.ndarray  # int grid-bin indices (sorted, unique)
    values: np.ndarray  # area-normalized weights (sum == 1)
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
    max_half_width: int = 4000,
) -> Template | None:
    """Area-normalized envelope of a class-average composition of ``mass`` at charge ``z``.

    Results are cached on the mass quantized to 0.1 mDa, so the detection pass and the
    adduct-resolution pass over the same candidate reuse their templates.
    """
    return _cached_template(
        int(round(mass * 1e4)), int(z), grid,
        (instrument.kind, instrument.resolution, instrument.mz_ref, instrument.mz_exponent,
         instrument.shape, instrument.eta),
        compound_class, int(round(adduct_mass * 1e4)), adduct_label,
        float(threshold), int(max_half_width),
    )


@lru_cache(maxsize=20000)
def _cached_template(mass_key, z, grid, instrument_key, compound_class, adduct_key,
                     adduct_label, threshold, max_half_width):
    kind, resolution, mz_ref, mz_exponent, shape, eta = instrument_key
    instrument = InstrumentModel(kind=kind, resolution=resolution, mz_ref=mz_ref,
                                 mz_exponent=mz_exponent, shape=shape, eta=eta)
    return _build_template_impl(mass_key * 1e-4, z, grid, instrument, compound_class,
                                adduct_key * 1e-4, adduct_label, threshold, max_half_width)


def _build_template_impl(mass, z, grid, instrument, compound_class, adduct_mass, adduct_label,
                         threshold, max_half_width) -> Template | None:
    """Vectorized construction: every isotopologue is splatted over a shared window.

    The per-isotopologue windows are flattened and reduced with ``np.bincount``, which avoids a
    Python loop over grid bins; the whole envelope costs a few tens of microseconds.
    """
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
    if abund.sum() <= 0:
        return None

    centers = (grid.mz_to_u(mz) - grid.u_min) / grid.step
    sigma_bins = np.maximum(instrument.sigma_mz(mz) / (mz - carrier) / grid.step, 0.4)
    kf = 4.0 if instrument.shape == "gaussian" else 10.0
    hw = int(min(max_half_width, np.ceil(kf * sigma_bins.max()) + 1))

    offsets = np.arange(-hw, hw + 1)
    ic = np.floor(centers).astype(np.int64)
    bins = ic[:, None] + offsets[None, :]
    frac = centers[:, None] - bins
    amp = (abund / (np.sqrt(2 * np.pi) * sigma_bins))[:, None]
    vals = amp * np.exp(-0.5 * (frac / sigma_bins[:, None]) ** 2)

    bins = bins.ravel()
    vals = vals.ravel()
    keep = (bins >= 0) & (bins < grid.size) & (vals > 0)
    if not np.any(keep):
        return None
    bins, vals = bins[keep], vals[keep]
    lo = int(bins.min())
    acc = np.bincount(bins - lo, weights=vals)
    nz = np.nonzero(acc > 0)[0]
    if nz.size == 0:
        return None
    out_bins = (nz + lo).astype(np.int64)
    out_vals = acc[nz]
    total = out_vals.sum()
    if total <= 0:
        return None
    return Template(out_bins, out_vals / total, z, adduct_mass, adduct_label, mass)


def template_mz(mass: float, z: int, polarity: int, adduct_mass: float = 0.0) -> float:
    carrier = carrier_mass(polarity)
    return (mass + adduct_mass + z * carrier) / z
