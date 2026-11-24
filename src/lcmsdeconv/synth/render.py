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


#: the states of a charge that a component lists no adduct states for
_BASE_STATES = {AdductState(): 1.0}


@dataclass
class ComponentInstance:
    """A compound placed in a scene with its charge and adduct distributions."""

    compound: Compound
    intensity: float
    charges: dict[int, float]
    adducts: dict[int, dict[AdductState, float]]  # z -> {state: fraction}

    def truth(self, comp_id: int) -> ComponentTruth:
        agg: dict[str, float] = {}
        labels: dict[int, str] = {}  # the label of each state object (all alive meanwhile)
        for z, fz in self.charges.items():
            for st, fa in self.adducts.get(z, _BASE_STATES).items():
                if id(st) not in labels:
                    labels[id(st)] = st.label or "base"
                label = labels[id(st)]
                agg[label] = agg.get(label, 0.0) + fz * fa
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
    """Flatten a scene into isotopologue sticks (observed m/z, intensity, charge, adduct mass).

    The sticks of every component, charge and adduct state in that order, exactly as
    :func:`_build_sticks_reference` writes them (one compiled loop instead of a dozen numpy
    calls per charge and adduct state)."""
    return sticks_from_parts(stick_parts(scene.components, scene.polarity, min_stick))


def stick_parts(components, polarity: int, min_stick: float = 1e-9, first_id: int = 0) -> list:
    """The sticks of ``components`` (component ids counted from ``first_id``) as a list of
    ``(mz, intensity, comp_id, z, adduct_mass)`` parts, which :func:`sticks_from_parts` joins.

    The parts of a scene's components followed by those of components appended to the scene are
    the parts of the larger scene, so sticks need not be built twice when a scene grows."""
    carrier = carrier_mass(polarity)
    parts: list = []
    batch = _StickRows(carrier, min_stick)
    for i, ci in enumerate(components):
        if _HAVE_NUMBA and batch.add(first_id + i, ci):
            continue
        parts += batch.fill()
        one = _build_sticks_reference(Scene([ci], polarity), min_stick)
        if len(one):
            parts.append((one.mz, one.intensity, np.full(len(one), first_id + i, dtype=np.int64),
                          one.z, one.adduct_mass))
    return parts + batch.fill()


def sticks_from_parts(parts: list) -> Sticks:
    """Join stick parts (:func:`stick_parts`) into one :class:`Sticks`."""
    if not parts:
        return Sticks.empty()
    if len(parts) == 1:
        return Sticks(*parts[0])
    return Sticks(*(np.concatenate([p[f] for p in parts]) for f in range(5)))


#: scalar types whose products numpy computes in float64 exactly as Python computes them
_PLAIN = (float, int, np.float64)


class _StickRows:
    """Components collected as rows (component, charge, adduct state) for :func:`_fill_sticks`.

    ``add`` takes a component whose sticks the compiled loop computes exactly as the reference
    does (a float64 isotope pattern, plain float intensity and fractions) and refuses others,
    which the caller builds with the reference instead."""

    def __init__(self, carrier: float, min_stick: float):
        self.carrier = carrier
        self.min_stick = min_stick
        self.masses: dict[int, float] = {}  # adduct mass of each state object (alive meanwhile)
        self.clear()

    def clear(self) -> None:
        self.rows: list = []
        self.patterns: list = []
        self.cids: list[int] = []

    def add(self, cid: int, ci) -> bool:
        pat = ci.compound.pattern
        m, a = pat.masses, pat.abundances
        if not (isinstance(m, np.ndarray) and isinstance(a, np.ndarray) and m.dtype == np.float64
                and a.dtype == np.float64 and m.ndim == 1 and m.shape == a.shape
                and type(ci.intensity) in _PLAIN):
            return False
        c = len(self.cids)
        rows = []
        for z, fz in ci.charges.items():
            if type(fz) not in _PLAIN:
                return False
            if fz <= 0:
                continue
            states = ci.adducts.get(z, _BASE_STATES)
            fracs = list(states.values())
            if not all(type(f) in _PLAIN for f in fracs):
                return False
            # the reference's ``ci.intensity * fz * fa``, which Python evaluates left to right
            scale = (ci.intensity * fz) * np.array(fracs, dtype=np.float64)
            ok = ~(scale < self.min_stick)
            if not ok.any():
                continue
            am = []
            for st in states:
                k = id(st)
                if k not in self.masses:
                    self.masses[k] = st.mass
                am.append(self.masses[k])
            rows.append((c, z, z * self.carrier, np.array(am, dtype=np.float64)[ok], scale[ok]))
        self.rows += rows
        self.patterns.append((m, a))
        self.cids.append(cid)
        return True

    def fill(self) -> list:
        """The sticks of the collected rows as at most one part; the batch is emptied."""
        rows, patterns, cids = self.rows, self.patterns, self.cids
        self.clear()
        if not rows:
            return []
        n_per = [r[4].size for r in rows]
        row_c = np.repeat(np.array([r[0] for r in rows], dtype=np.int64), n_per)
        row_z = np.repeat(np.array([r[1] for r in rows], dtype=np.int64), n_per)
        row_zf = np.repeat(np.array([float(r[1]) for r in rows], dtype=np.float64), n_per)
        row_zc = np.repeat(np.array([r[2] for r in rows], dtype=np.float64), n_per)
        row_am = np.concatenate([r[3] for r in rows])
        row_scale = np.concatenate([r[4] for r in rows])
        sizes = np.array([m.size for m, _a in patterns], dtype=np.int64)
        stop = np.cumsum(sizes)
        start = stop - sizes
        size = int(np.sum(sizes[row_c]))
        mz, inten, am = np.empty(size), np.empty(size), np.empty(size)
        cid, zz = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64)
        n = _fill_sticks(row_c, row_z, row_zf, row_zc, row_am, row_scale, start, stop,
                         np.concatenate([m for m, _a in patterns]),
                         np.concatenate([a for _m, a in patterns]),
                         np.array(cids, dtype=np.int64), float(self.min_stick),
                         mz, inten, cid, zz, am)
        if n == 0:
            return []
        if n < size:
            mz, inten, cid, zz, am = (x[:n].copy() for x in (mz, inten, cid, zz, am))
        return [(mz, inten, cid, zz, am)]


@njit(cache=True)
def _fill_sticks(row_c, row_z, row_zf, row_zc, row_am, row_scale, start, stop, pat_mass, pat_ab,
                 cids, min_stick, mz, inten, cid, z, am):
    """Sticks of every row (one component, charge and adduct state), in row order: for each
    isotopologue of the row's component whose intensity ``scale * abundance`` reaches
    ``min_stick``, m/z ``(mass + adduct mass + z * carrier) / z``, that intensity, the component
    id, the charge and the adduct mass. The same operations in the same order as the
    reference's per-row numpy expressions (no fast-math), so equal bit for bit; returns the
    number of sticks written."""
    n = 0
    for r in range(row_c.shape[0]):
        c = row_c[r]
        s = row_scale[r]
        for i in range(start[c], stop[c]):
            v = s * pat_ab[i]
            if v >= min_stick:
                mz[n] = (pat_mass[i] + row_am[r] + row_zc[r]) / row_zf[r]
                inten[n] = v
                cid[n] = cids[c]
                z[n] = row_z[r]
                am[n] = row_am[r]
                n += 1
    return n


def _build_sticks_reference(scene: Scene, min_stick: float = 1e-9) -> Sticks:
    """The original per-state construction of :func:`build_sticks` (kept for the tests and for
    components :func:`_fill_sticks` does not take)."""
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


def render_profile(sticks: Sticks, mz_axis: np.ndarray, instrument: InstrumentModel,
                   out: np.ndarray | None = None) -> np.ndarray:
    """Render sticks onto a raw (non-uniform) m/z axis with apex-height convention.

    ``out``, a float64 array as long as the axis, is zeroed and rendered into instead of a new
    array (a training frame renders one profile per charge on the same axis)."""
    if out is None:
        out = np.zeros(mz_axis.size, dtype=np.float64)
    else:
        out[:] = 0.0
    if len(sticks) == 0:
        return out
    order = np.argsort(sticks.mz)
    mz = sticks.mz[order]
    inten = sticks.intensity[order]
    sigma = instrument.sigma_mz(mz)
    idx = np.searchsorted(mz_axis, mz)
    _splat_profile(mz_axis, mz, inten, sigma, idx.astype(np.int64), out,
                   _profile_kfactor(instrument))
    return out


def _profile_kfactor(instrument: InstrumentModel) -> float:
    """How many sigmas from its centre :func:`render_profile` draws a peak."""
    return 4.0 if instrument.shape == "gaussian" else 12.0


def profile_reach(sticks: Sticks, mz_axis: np.ndarray, instrument: InstrumentModel
                  ) -> tuple[int, int]:
    """The part ``lo:hi`` of a sorted axis that :func:`render_profile` writes for ``sticks``:
    no point outside it, and from ``mz_axis[lo:hi]`` alone the same values at every point in it.

    The kernel draws each stick from its insertion point ``np.searchsorted(mz_axis, mz)``,
    which it writes, outwards while points lie closer than the line shape's reach (its sigma
    floored at 1e-6, times :func:`_profile_kfactor`), and never past an end of the axis it is
    given. ``lo:hi`` holds every point within that reach of a stick and every insertion point,
    with two points to spare on either side for rounding, so that rendered on the slice each
    stick starts from the same point and stops where it stops on the whole axis. Sticks or
    widths that are not finite: the whole axis."""
    n = mz_axis.size
    if len(sticks) == 0:
        return 0, 0
    mz = sticks.mz
    sigma = instrument.sigma_mz(mz)
    if not (np.all(np.isfinite(mz)) and np.all(np.isfinite(sigma))):
        return 0, n
    reach = _profile_kfactor(instrument) * np.where(sigma > 0, sigma, 1e-6)
    start = np.searchsorted(mz_axis, mz)
    lo = min(int(start.min()), int(np.searchsorted(mz_axis, float(np.min(mz - reach))))) - 2
    hi = max(int(start.max()) + 1,
             int(np.searchsorted(mz_axis, float(np.max(mz + reach)), side="right"))) + 2
    return max(lo, 0), min(hi, n)


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
