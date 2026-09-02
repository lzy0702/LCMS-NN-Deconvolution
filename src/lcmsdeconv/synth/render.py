"""Rendering: scene -> isotopologue sticks -> profile spectra and grid arrays.

The same stick set produces (a) the observed grid intensity and per-charge contributions used
to build neural-network labels and (b) the raw profile spectrum written to mzML, so features and
labels can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..chem.adducts import AdductLibrary, AdductState, carrier_mass
from ..chem.instrument import FWHM_TO_SIGMA, InstrumentModel
from .spec import ComponentTruth, Compound, Sticks

try:  # optional acceleration
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(f):
            return f
        return wrap if not args else args[0]


@dataclass
class ComponentInstance:
    """A compound placed in a scene with its charge and adduct distributions."""

    compound: Compound
    intensity: float
    charges: dict[int, float]
    adducts: dict[int, dict[AdductState, float]]  # z -> {state: fraction}

    def truth(self, comp_id: int) -> ComponentTruth:
        agg: dict[str, float] = {}
        for z, fz in self.charges.items():
            for st, fa in self.adducts.get(z, {AdductState(): 1.0}).items():
                agg[st.label or "base"] = agg.get(st.label or "base", 0.0) + fz * fa
        return ComponentTruth(
            id=comp_id,
            mass=self.compound.mass,
            mono_mass=self.compound.mono_mass,
            average_mass=self.compound.average_mass,
            compound_class=self.compound.compound_class,
            intensity=self.intensity,
            charges=dict(self.charges),
            adducts=agg,
            kind=self.compound.kind,
            parent_id=self.compound.parent_id,
            name=self.compound.name,
        )


@dataclass
class Scene:
    components: list[ComponentInstance]
    polarity: int = 1
    library: AdductLibrary | None = None
    meta: dict = field(default_factory=dict)

    def truths(self) -> list[ComponentTruth]:
        return [ci.truth(i) for i, ci in enumerate(self.components)]


def build_sticks(scene: Scene, min_stick: float = 1e-9) -> Sticks:
    """Flatten a scene into isotopologue sticks (observed m/z, intensity, charge, adduct mass)."""
    carrier = carrier_mass(scene.polarity)
    mz_parts, it_parts, cid_parts, z_parts, ad_parts = [], [], [], [], []
    for cid, ci in enumerate(scene.components):
        pat = ci.compound.pattern
        masses = pat.masses
        abund = pat.abundances
        for z, fz in ci.charges.items():
            if fz <= 0:
                continue
            states = ci.adducts.get(z, {AdductState(): 1.0})
            for st, fa in states.items():
                scale = ci.intensity * fz * fa
                if scale < min_stick:
                    continue
                admass = st.mass
                mz = (masses + admass + z * carrier) / z
                it = scale * abund
                keep = it >= min_stick
                if not np.any(keep):
                    continue
                mz_parts.append(mz[keep])
                it_parts.append(it[keep])
                cid_parts.append(np.full(int(keep.sum()), cid, dtype=np.int64))
                z_parts.append(np.full(int(keep.sum()), z, dtype=np.int64))
                ad_parts.append(np.full(int(keep.sum()), admass, dtype=np.float64))
    if not mz_parts:
        return Sticks.empty()
    return Sticks(
        np.concatenate(mz_parts),
        np.concatenate(it_parts),
        np.concatenate(cid_parts),
        np.concatenate(z_parts),
        np.concatenate(ad_parts),
    )


@njit(cache=True, fastmath=True)
def _splat_uniform(centers, intensities, sigma_bins, out, kfactor):
    n = centers.shape[0]
    L = out.shape[0]
    for i in range(n):
        c = centers[i]
        s = sigma_bins[i]
        if s < 0.3:
            s = 0.3
        amp = intensities[i] / (2.5066282746310002 * s)  # area-normalized (sum ~ intensity)
        hw = int(kfactor * s) + 1
        lo = int(c) - hw
        hi = int(c) + hw + 1
        if lo < 0:
            lo = 0
        if hi > L:
            hi = L
        for b in range(lo, hi):
            d = (b - c) / s
            out[b] += amp * np.exp(-0.5 * d * d)


@njit(cache=True, fastmath=True)
def _splat_rows(centers, intensities, sigma_bins, rows, out, kfactor):
    n = centers.shape[0]
    L = out.shape[1]
    for i in range(n):
        c = centers[i]
        s = sigma_bins[i]
        if s < 0.3:
            s = 0.3
        amp = intensities[i] / (2.5066282746310002 * s)
        r = rows[i]
        hw = int(kfactor * s) + 1
        lo = int(c) - hw
        hi = int(c) + hw + 1
        if lo < 0:
            lo = 0
        if hi > L:
            hi = L
        for b in range(lo, hi):
            d = (b - c) / s
            out[r, b] += amp * np.exp(-0.5 * d * d)


def render_grid_total(sticks: Sticks, grid, instrument: InstrumentModel) -> np.ndarray:
    """Render all sticks onto a full grid (area-normalized). Returns intensity[B]."""
    out = np.zeros(grid.size, dtype=np.float64)
    if len(sticks) == 0:
        return out
    u = grid.mz_to_u(sticks.mz)
    centers = (u - grid.u_min) / grid.step
    sigma_mz = instrument.sigma_mz(sticks.mz)
    sigma_u = sigma_mz / (sticks.mz - grid.carrier)
    sigma_bins = sigma_u / grid.step
    kfactor = 4.0 if instrument.shape == "gaussian" else 10.0
    _splat_uniform(centers.astype(np.float64), sticks.intensity.astype(np.float64),
                   sigma_bins.astype(np.float64), out, float(kfactor))
    return out


def render_grid_by_charge(
    sticks: Sticks, grid, instrument: InstrumentModel, b0: int, b1: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a crop [b0, b1) grouped by charge and by component.

    Returns (charge_mat[nz, L], charge_list[nz], comp_mat[nc, L]) where L = b1 - b0.
    ``comp_mat`` rows follow component id order 0..maxcid.
    """
    L = b1 - b0
    if len(sticks) == 0:
        return np.zeros((0, L)), np.array([], dtype=int), np.zeros((0, L))
    u = grid.mz_to_u(sticks.mz)
    centers = (u - grid.u_min) / grid.step - b0
    sigma_mz = instrument.sigma_mz(sticks.mz)
    sigma_u = sigma_mz / (sticks.mz - grid.carrier)
    sigma_bins = (sigma_u / grid.step).astype(np.float64)
    kfactor = 4.0 if instrument.shape == "gaussian" else 10.0
    pad = (kfactor * sigma_bins + 2).astype(int)
    m = (centers + pad >= 0) & (centers - pad < L)
    if not np.any(m):
        return np.zeros((0, L)), np.array([], dtype=int), np.zeros((0, L))
    centers, inten, zc, cid = centers[m], sticks.intensity[m], sticks.z[m], sticks.comp_id[m]
    sig = sigma_bins[m]

    charges = np.unique(zc)
    zrow = {int(z): r for r, z in enumerate(charges)}
    rows = np.array([zrow[int(z)] for z in zc], dtype=np.int64)
    cmat = np.zeros((len(charges), L), dtype=np.float64)
    _splat_rows(centers.astype(np.float64), inten.astype(np.float64), sig, rows, cmat, float(kfactor))

    ncomp = int(cid.max()) + 1
    comp_rows = cid.astype(np.int64)
    compmat = np.zeros((ncomp, L), dtype=np.float64)
    _splat_rows(centers.astype(np.float64), inten.astype(np.float64), sig, comp_rows, compmat, float(kfactor))
    return cmat, charges.astype(int), compmat


def render_profile(sticks: Sticks, mz_axis: np.ndarray, instrument: InstrumentModel) -> np.ndarray:
    """Render sticks onto a raw (non-uniform) m/z axis with apex-height convention."""
    out = np.zeros(mz_axis.size, dtype=np.float64)
    if len(sticks) == 0:
        return out
    order = np.argsort(sticks.mz)
    mz = sticks.mz[order]
    inten = sticks.intensity[order]
    sigma = instrument.sigma_mz(mz)
    idx = np.searchsorted(mz_axis, mz)
    kfactor = 4.0 if instrument.shape == "gaussian" else 12.0
    _splat_profile(mz_axis, mz, inten, sigma, idx.astype(np.int64), out, float(kfactor))
    return out


@njit(cache=True, fastmath=True)
def _splat_profile(axis, centers, intensities, sigma, idx, out, kfactor):
    n = centers.shape[0]
    L = axis.shape[0]
    for i in range(n):
        c = centers[i]
        s = sigma[i]
        if s <= 0:
            s = 1e-6
        amp = intensities[i]
        j = idx[i]
        # walk left
        b = j
        while b >= 0 and b < L and (c - axis[b]) < kfactor * s:
            d = (axis[b] - c) / s
            out[b] += amp * np.exp(-0.5 * d * d)
            b -= 1
        # walk right
        b = j + 1
        while b < L and (axis[b] - c) < kfactor * s:
            d = (axis[b] - c) / s
            out[b] += amp * np.exp(-0.5 * d * d)
            b += 1


def peak_fwhm_bins(grid, instrument: InstrumentModel, mz: float) -> float:
    sigma_u = FWHM_TO_SIGMA / instrument.resolving_power(mz)
    return float(sigma_u / grid.step / FWHM_TO_SIGMA)
