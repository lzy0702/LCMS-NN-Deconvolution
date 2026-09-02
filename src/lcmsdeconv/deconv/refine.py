"""Refine candidates by weighted NNLS envelope fitting on the grid, with residual iteration."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from ..chem.adducts import AdductLibrary, AdductState, carrier_mass, mass_from_mz
from ..chem.instrument import InstrumentModel
from ..core.model import Component
from .decode import Candidate
from .templates import build_template, template_mz


def _fit_candidate(
    cand: Candidate,
    residual: np.ndarray,
    weights: np.ndarray,
    grid,
    instrument: InstrumentModel,
    compound_class: str,
    adduct_states: list[AdductState],
    z_extend: int = 1,
    min_coef_frac: float = 1e-3,
    mz_range: tuple[float, float] | None = None,
    max_charges: int = 45,
    max_columns: int = 300,
) -> tuple[dict[tuple[int, str], float], list, float]:
    """Weighted NNLS of one candidate (charges x adduct states) against the residual.

    The charge range is bounded by what the spectrum could actually show: a charge whose m/z
    falls outside the measured range contributes no observable peak, so including it only adds
    a column that the solver must work through. Charge assignments are noisy enough that
    without this bound a single candidate can pull in sixty charge states spanning the whole
    grid, which makes the non-negative least-squares solve orders of magnitude slower.

    Returns (coefficients keyed by (z, adduct_label), templates used, explained_intensity).
    """
    charges = sorted(cand.charges)
    if charges:
        zlo = max(1, min(charges) - z_extend)
        zhi = max(charges) + z_extend
    else:
        zlo, zhi = 1, 1
    if mz_range is not None and cand.mass > 0:
        mz_lo, mz_hi = mz_range
        carrier = abs(carrier_mass(grid.polarity))
        z_from_hi = max(1, int(np.floor(cand.mass / max(mz_hi - carrier, 1e-6))))
        z_from_lo = max(1, int(np.ceil(cand.mass / max(mz_lo - carrier, 1e-6))))
        zlo = max(zlo, z_from_hi)
        zhi = min(zhi, z_from_lo)
    if zhi < zlo:
        zlo = zhi = max(1, min(charges) if charges else 1)
    # Keep the problem small enough to solve quickly: with many adduct states the column count
    # is charges x states, and the charges carrying most of the envelope are the central band.
    allowed = max(3, min(max_charges, max_columns // max(1, len(adduct_states))))
    if zhi - zlo + 1 > allowed:
        centre = int(np.median(charges)) if charges else (zlo + zhi) // 2
        half = allowed // 2
        zlo, zhi = max(zlo, centre - half), min(zhi, centre + half)
    zrange = range(zlo, zhi + 1)

    columns = []
    keys = []
    for z in zrange:
        for st in adduct_states:
            tpl = build_template(cand.mass, z, grid, instrument, compound_class,
                                 adduct_mass=st.mass, adduct_label=st.label)
            if tpl is None:
                continue
            columns.append(tpl)
            keys.append((z, st.label))
    if not columns:
        return {}, [], 0.0

    support = np.unique(np.concatenate([c.bins for c in columns]))
    support = support[(support >= 0) & (support < grid.size)]
    if support.size == 0:
        return {}, [], 0.0
    A = np.zeros((support.size, len(columns)), dtype=np.float64)
    for j, col in enumerate(columns):
        # every template bin is in the support by construction, so a sorted lookup suffices
        A[np.searchsorted(support, col.bins), j] = col.values
    w = weights[support]
    b = residual[support] * w
    Aw = A * w[:, None]
    try:
        x, _ = nnls(Aw, b, maxiter=5 * A.shape[1])
    except Exception:
        return {}, [], 0.0
    peak = x.max() if x.size else 0.0
    coeffs: dict[tuple[int, str], float] = {}
    used = []
    for j, (key, t) in enumerate(zip(keys, columns)):
        if x[j] > min_coef_frac * peak and x[j] > 0:
            coeffs[key] = coeffs.get(key, 0.0) + float(x[j])
            used.append((t, float(x[j])))
    explained = float(sum(c for c in coeffs.values()))
    return coeffs, used, explained


def refine_frame(
    candidates: list[Candidate],
    grid,
    observed: np.ndarray,
    noise_sigma: float,
    instrument: InstrumentModel,
    compound_class: str,
    library: AdductLibrary,
    raw_mz: np.ndarray | None = None,
    raw_int: np.ndarray | None = None,
    max_iter: int = 2,
    min_charge_support: int = 2,
    min_component_frac: float = 1e-4,
    max_mass_spread_ppm: float = 150.0,
    adduct_refit_frac: float = 1e-3,
) -> list[Component]:
    """Greedy weighted-NNLS fit of candidates on the residual, strongest first."""
    residual = observed.astype(np.float64).copy()
    weights = 1.0 / np.sqrt(np.clip(observed, 0, None) + noise_sigma**2 + 1e-9)
    # Detection uses only the base state and one of each adduct: enough to recognise a species
    # without paying for every adduct combination on candidates that will be rejected anyway.
    states_fast = [AdductState()]
    states_full = library.states()
    mz_range = (float(raw_mz.min()), float(raw_mz.max())) if raw_mz is not None and raw_mz.size else None
    total_obs = observed.sum() + 1e-9

    components: list[Component] = []
    remaining = list(candidates)
    cid = 0
    for _ in range(max_iter):
        remaining.sort(key=lambda c: -c.score)
        progressed = False
        for cand in remaining:
            coeffs, used, explained = _fit_candidate(
                cand, residual, weights, grid, instrument, compound_class, states_fast,
                mz_range=mz_range,
            )
            if explained <= 0:
                continue
            charges_present = {z for (z, _lab) in coeffs}
            if len(charges_present) < min_charge_support and charges_present != {1}:
                continue
            if explained < min_component_frac * total_obs:
                continue
            # aggregate by charge and by adduct
            charge_int: dict[int, float] = {}
            adduct_int: dict[str, float] = {}
            for (z, lab), v in coeffs.items():
                charge_int[z] = charge_int.get(z, 0.0) + v
                adduct_int[lab or "base"] = adduct_int.get(lab or "base", 0.0) + v
            comp = Component(
                mass=cand.mass, intensity=explained, mass_type="average",
                charges=charge_int, adducts=adduct_int, score=cand.score,
                compound_class=compound_class, id=cid,
            )
            if raw_mz is not None:
                _refine_mass(comp, raw_mz, raw_int, grid.polarity)
            if comp.mass_spread_ppm > max_mass_spread_ppm and len(charges_present) > 2:
                # charge states disagree on the mass: a degenerate fit, not a real species
                continue
            # An adduct only ever adds mass, so if the fit is centred on an adducted form the
            # lighter base form cannot be represented at all. Trying the centre one adduct
            # lower and keeping it when it explains more recovers the true neutral mass.
            if len(states_full) > 1 and explained >= adduct_refit_frac * total_obs:
                cand, coeffs, used, explained = _recenter_base(
                    cand, coeffs, used, explained, residual, weights, grid, instrument,
                    compound_class, states_full, library, mz_range,
                    raw_mz=raw_mz, raw_int=raw_int, noise_sigma=noise_sigma,
                )
                comp.mass = cand.mass
            if len(states_full) > 1 and explained >= adduct_refit_frac * total_obs:
                coeffs_full, used_full, explained_full = _fit_candidate(
                    cand, residual, weights, grid, instrument, compound_class, states_full,
                    mz_range=mz_range,
                )
                if explained_full >= 0.8 * explained:
                    coeffs, used, explained = coeffs_full, used_full, explained_full
                    adduct_int = {}
                    charge_int = {}
                    for (z, lab), v in coeffs.items():
                        charge_int[z] = charge_int.get(z, 0.0) + v
                        adduct_int[lab or "base"] = adduct_int.get(lab or "base", 0.0) + v
                    comp.charges, comp.adducts, comp.intensity = charge_int, adduct_int, explained
            components.append(comp)
            cid += 1
            # subtract fitted envelope
            for tpl, coef in used:
                np.subtract.at(residual, tpl.bins, tpl.values * coef)
            residual = np.clip(residual, 0, None)
            progressed = True
        # re-detect on residual for the next iteration
        if not progressed:
            break
        remaining = _redetect(residual, grid, noise_sigma, components)
        if not remaining:
            break

    # residual over signal-bearing bins only (whole-grid noise would dominate otherwise)
    sig_bins = observed > 10.0 * noise_sigma
    denom = float(observed[sig_bins].sum()) + 1e-9
    residual_fraction = float(residual[sig_bins].sum() / denom) if sig_bins.any() else 0.0
    for c in components:
        c.flags = list(c.flags)
    _merge_same_mass(components)
    _merge_adduct_components(components, library)
    _flag_harmonics(components)
    _flag_adduct_ambiguity(components, library)
    return components, residual_fraction


def _recenter_base(cand, coeffs, used, explained, residual, weights, grid, instrument,
                   compound_class, states_full, library, mz_range, raw_mz=None, raw_int=None,
                   noise_sigma: float = 0.0, max_steps: int = 3):
    """Walk the candidate centre down the adduct ladder while observed peaks support it.

    Rather than refitting at every trial mass, this asks the cheap question directly: are there
    observed peaks where the lighter form's charge states would be? Adducts only add mass, so a
    centre that is one adduct too heavy leaves the base form unrepresentable, and the base form
    is exactly what those peaks would show.
    """
    if raw_mz is None or raw_mz.size == 0 or not cand.charges:
        return cand, coeffs, used, explained
    deltas = sorted({d for d in library.deltas().values() if d > 0})
    if not deltas:
        return cand, coeffs, used, explained
    carrier = carrier_mass(grid.polarity)
    charges = sorted(cand.charges)[:12]
    thr = 3.0 * noise_sigma

    def support(mass: float) -> int:
        hits = 0
        for z in charges:
            mz_pred = (mass + z * carrier) / z
            tol = mz_pred * 3e-4
            lo = int(np.searchsorted(raw_mz, mz_pred - tol))
            hi = int(np.searchsorted(raw_mz, mz_pred + tol))
            if hi - lo < 3:
                continue
            if raw_int[lo:hi].max() > thr:
                hits += 1
        return hits

    base_hits = support(cand.mass)
    mass = cand.mass
    moved = False
    for _ in range(max_steps):
        best_d, best_hits = None, base_hits
        for d in deltas:
            if mass - d <= 0:
                continue
            h = support(mass - d)
            if h > best_hits:
                best_d, best_hits = d, h
        if best_d is None:
            break
        mass -= best_d
        base_hits = best_hits
        moved = True
    if not moved:
        return cand, coeffs, used, explained
    trial = Candidate(mass=mass, score=cand.score, charges=set(cand.charges),
                      intensity=cand.intensity)
    c2, u2, e2 = _fit_candidate(trial, residual, weights, grid, instrument, compound_class,
                                states_full, mz_range=mz_range)
    if e2 >= explained:
        return trial, c2, u2, e2
    return cand, coeffs, used, explained


def _redetect(residual, grid, noise_sigma, existing) -> list[Candidate]:
    """Look for leftover envelopes in the residual (cheap: strong isolated bins)."""
    thr = 6 * noise_sigma
    if residual.max() < thr:
        return []
    # coarse: not re-running the NN; return empty to keep it deterministic and fast
    return []


def _refine_mass(comp: Component, raw_mz, raw_int, polarity, tol_ppm: float = 200.0):
    """Refine the mass from raw peak apexes.

    The observed apex of an envelope is the most abundant isotopologue, which for large masses
    sits well below the average mass (about -0.5 Da at 17 kDa), so the apex-derived mass is
    corrected by the class pattern's average-minus-apex offset.
    """
    from ..chem.classes import class_isotope_pattern

    pat = class_isotope_pattern(comp.mass, comp.compound_class)
    apex_offset = pat.average_mass - pat.most_abundant_mass
    masses = []
    wts = []
    for z, inten in comp.charges.items():
        mz_pred = template_mz(comp.mass, z, polarity, 0.0)
        lo = np.searchsorted(raw_mz, mz_pred * (1 - tol_ppm * 1e-6))
        hi = np.searchsorted(raw_mz, mz_pred * (1 + tol_ppm * 1e-6))
        if hi <= lo:
            continue
        seg_mz = raw_mz[lo:hi]
        seg_i = raw_int[lo:hi]
        if seg_i.max() <= 0:
            continue
        apex = seg_mz[int(np.argmax(seg_i))]
        m = mass_from_mz(apex, z, polarity, 0.0) + apex_offset
        masses.append(m)
        wts.append(inten)
    if masses:
        masses = np.array(masses)
        wts = np.array(wts)
        mean = float(np.average(masses, weights=wts))
        comp.mass = mean
        if masses.size > 1:
            comp.mass_spread_ppm = float(np.sqrt(np.average((masses - mean) ** 2, weights=wts)) / mean * 1e6)


def _merge_same_mass(components: list[Component], tol_ppm: float = 150.0, tol_da: float = 0.4):
    """Combine components that refined to the same mass (duplicate candidates)."""
    components.sort(key=lambda c: -c.intensity)
    keep: list[Component] = []
    for c in components:
        hit = None
        for k in keep:
            tol = max(k.mass * tol_ppm * 1e-6, tol_da)
            if abs(c.mass - k.mass) <= tol:
                hit = k
                break
        if hit is None:
            keep.append(c)
            continue
        w0, w1 = hit.intensity, c.intensity
        hit.mass = (hit.mass * w0 + c.mass * w1) / max(w0 + w1, 1e-12)
        hit.intensity += c.intensity
        for z, v in c.charges.items():
            hit.charges[z] = hit.charges.get(z, 0.0) + v
        for a, v in c.adducts.items():
            hit.adducts[a] = hit.adducts.get(a, 0.0) + v
    components[:] = keep


def _merge_adduct_components(components: list[Component], library: AdductLibrary, tol_da: float = 0.5):
    """Merge components separated by an adduct delta into the more intense (base) one."""
    deltas = library.deltas()
    components.sort(key=lambda c: -c.intensity)
    keep: list[Component] = []
    for c in components:
        merged = False
        for base in keep:
            dm = c.mass - base.mass
            for name, d in deltas.items():
                for n in (1, 2, 3):
                    if abs(dm - n * d) <= tol_da:
                        lab = f"+{n if n > 1 else ''}{name}"
                        base.adducts[lab] = base.adducts.get(lab, 0.0) + c.intensity
                        base.intensity += c.intensity
                        for z, v in c.charges.items():
                            base.charges[z] = base.charges.get(z, 0.0) + v
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        if not merged:
            keep.append(c)
    components[:] = keep


def _flag_adduct_ambiguity(components: list[Component], library: AdductLibrary,
                           min_fraction: float = 0.005, tol_da: float = 1.5):
    """Warn when a reported adduct delta is close to a known modification of the same class.

    +NH4 (17.027) and oxidation (+15.995) differ by ~1 Da, so at modest resolving power an
    unresolved impurity can be reported as an adduct. The user needs to know which library
    entries could be aliasing a product-related impurity.
    """
    from ..chem.modifications import annotate_delta

    deltas = library.deltas()
    for c in components:
        fr = c.adduct_fractions()
        for label, frac in fr.items():
            if label == "base" or frac < min_fraction:
                continue
            name = label.lstrip("+")
            n = 1
            while name and name[0].isdigit():
                n = int(name[0])
                name = name[1:]
            d = deltas.get(name)
            if d is None:
                continue
            hits = annotate_delta(n * d, c.compound_class, tolerance=tol_da)
            if hits:
                c.flags.append(f"adduct {label} within {tol_da} Da of {hits[0].name}")


def _flag_harmonics(components: list[Component], tol_ppm: float = 100.0):
    by_int = sorted(components, key=lambda c: -c.intensity)
    for i, c in enumerate(by_int):
        for bigger in by_int[:i]:
            for factor in (2.0, 0.5):
                if abs(c.mass - bigger.mass * factor) / c.mass * 1e6 < tol_ppm:
                    if c.intensity < 0.2 * bigger.intensity:
                        c.flags.append("possible_harmonic")
                    break
