"""Refine candidates by weighted NNLS envelope fitting on the grid, with residual iteration."""

from __future__ import annotations

import numpy as np

from ..chem.adducts import AdductLibrary, AdductState, carrier_mass
from ..chem.instrument import InstrumentModel
from ..core.model import Component
from .decode import Candidate
from .nnls_solve import weighted_nnls
from .templates import build_template


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
    max_charges: int = 70,
    max_columns: int = 700,
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
    x = weighted_nnls(A, residual[support], w)
    if x.size == 0:
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
    residual: np.ndarray | None = None,
    start_id: int = 0,
    strongest_explained: float = 0.0,
    trace: list | None = None,
    trace_pass: int = 0,
) -> tuple[list[Component], float, np.ndarray]:
    """Greedy weighted-NNLS fit of candidates, strongest first.

    Fits against ``residual`` (the observed spectrum by default) while weighting by the original
    measurement, and returns the accepted components, the fraction of signal-bearing intensity
    still unexplained, and the updated residual so the caller can re-detect on what is left.
    """
    residual = observed.astype(np.float64).copy() if residual is None else residual
    weights = 1.0 / np.sqrt(np.clip(observed, 0, None) + noise_sigma**2 + 1e-9)
    # Detection uses only the base state and one of each adduct: enough to recognise a species
    # without paying for every adduct combination on candidates that will be rejected anyway.
    states_fast = [AdductState()]
    states_full = library.states()
    mz_range = (float(raw_mz.min()), float(raw_mz.max())) if raw_mz is not None and raw_mz.size else None
    # An abundance floor relative to the whole grid would depend on how wide the grid is and how
    # noisy the baseline is. Impurity limits are quoted against the main component, so that is
    # what candidates are measured against: the strongest fit seen so far in this frame.
    strongest = float(strongest_explained)

    components: list[Component] = []
    remaining = list(candidates)
    cid = start_id
    remaining.sort(key=lambda c: -c.score)

    def decide(cand_mass: float, outcome: str, mass: float | None = None) -> None:
        if trace is not None:
            trace.append({"pass": trace_pass, "stage": "fit", "candidate": float(cand_mass),
                          "outcome": outcome, "mass": None if mass is None else float(mass)})

    resolution = float(instrument.resolving_power(1000.0)) if instrument is not None else 30000.0
    for cand in remaining:
        cand_mass0 = cand.mass
        if raw_mz is not None and raw_mz.size and cand.charges and cand.charges != {1}:
            # fit the species' whole observed envelope, not only the charges that voted for it:
            # charge states the charge map got wrong are otherwise left in the residual, where
            # they are later explained as separate, false species
            cand.charges = observed_charges(cand.mass, cand.charges, raw_mz, raw_int,
                                            3.0 * noise_sigma, grid.polarity, compound_class,
                                            resolution, mz_range)
        coeffs, used, explained = _fit_candidate(
            cand, residual, weights, grid, instrument, compound_class, states_fast,
            mz_range=mz_range,
        )
        if explained <= 0:
            decide(cand_mass0, "no_fit")
            continue
        charges_present = {z for (z, _lab) in coeffs}
        if (len(charges_present) < min_charge_support and charges_present != {1}
                and not getattr(cand, "single_charge", False)):
            decide(cand_mass0, "charge_support")
            continue
        strongest = max(strongest, explained)
        if explained < min_component_frac * strongest:
            decide(cand_mass0, "abundance_floor")
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
            _refine_mass(comp, raw_mz, raw_int, grid.polarity, resolution=resolution)
        if comp.mass_spread_ppm > max_mass_spread_ppm and len(charges_present) > 2:
            # charge states disagree on the mass: a degenerate fit, not a real species
            decide(cand_mass0, "mass_spread", comp.mass)
            continue
        full = None
        if len(states_full) > 1 and explained >= adduct_refit_frac * strongest:
            full = _fit_candidate(cand, residual, weights, grid, instrument, compound_class,
                                  states_full, mz_range=mz_range)
            # An adduct only ever adds mass, so if the fit is centred on an adducted form the
            # lighter base form cannot be represented at all. Trying the centre one adduct
            # lower, and keeping it when it explains more with the same adduct model, recovers
            # the true neutral mass.
            moved, full2 = _recenter_base(
                cand, full, residual, weights, grid, instrument, compound_class, states_full,
                library, mz_range, raw_mz=raw_mz, raw_int=raw_int, noise_sigma=noise_sigma,
                resolution=resolution,
            )
            if moved is not cand:
                cand, full = moved, full2
                # re-centred: measure the mass again from the raw envelopes at the new centre
                comp.mass = cand.mass
                if raw_mz is not None:
                    _refine_mass(comp, raw_mz, raw_int, grid.polarity, resolution=resolution)
        if full is not None:
            coeffs_full, used_full, explained_full = full
            if explained_full >= 0.8 * explained:
                coeffs, used, explained = coeffs_full, used_full, explained_full
                adduct_int = {}
                charge_int = {}
                for (z, lab), v in coeffs.items():
                    charge_int[z] = charge_int.get(z, 0.0) + v
                    adduct_int[lab or "base"] = adduct_int.get(lab or "base", 0.0) + v
                comp.charges, comp.adducts, comp.intensity = charge_int, adduct_int, explained
        comp.columns = {k: float(v) for k, v in coeffs.items()}
        components.append(comp)
        decide(cand_mass0, "accepted", comp.mass)
        cid += 1
        # subtract fitted envelope
        for tpl, coef in used:
            np.subtract.at(residual, tpl.bins, tpl.values * coef)
        residual = np.clip(residual, 0, None)

    # residual over signal-bearing bins only (whole-grid noise would dominate otherwise)
    sig_bins = observed > 10.0 * noise_sigma
    denom = float(observed[sig_bins].sum()) + 1e-9
    residual_fraction = float(residual[sig_bins].sum() / denom) if sig_bins.any() else 0.0
    for c in components:
        c.flags = list(c.flags)
    if trace is not None:
        trace.append({"pass": trace_pass, "stage": "pre_merge",
                      "masses": [float(c.mass) for c in components]})
    _merge_same_mass(components)
    _merge_adduct_components(components, library)
    if trace is not None:
        trace.append({"pass": trace_pass, "stage": "post_merge",
                      "masses": [float(c.mass) for c in components]})
    _flag_harmonics(components)
    _flag_adduct_ambiguity(components, library)
    return components, residual_fraction, residual


def _recenter_base(cand, full, residual, weights, grid, instrument, compound_class,
                   states_full, library, mz_range, raw_mz=None, raw_int=None,
                   noise_sigma: float = 0.0, resolution: float = 30000.0, max_steps: int = 3,
                   min_ratio: float = 0.5, min_gain: float = 0.01):
    """Walk the candidate centre down the adduct ladder while the spectrum supports it.

    Returns ``(candidate, full_fit)``: the original pair when nothing better is found.

    A step down by an adduct mass ``d`` is taken only when the lighter form's envelopes are
    present on most of the same charge states *and* carry at least ``min_ratio`` of the current
    form's height. Under any realistic adduct load the base form is the strongest member of its
    ladder, so a much weaker envelope one adduct lower is a different species (an impurity
    22 Da light, the next oligomer), not this species' base; walking onto it is what used to
    report a main component two sodium ions too light. The walk is then confirmed by refitting
    with the same adduct model at both centres and keeping the lighter one only if it explains
    more of the spectrum.
    """
    if raw_mz is None or raw_mz.size == 0 or not cand.charges or full is None:
        return cand, full
    deltas = sorted({d for d in library.deltas().values() if d > 0})
    if not deltas:
        return cand, full
    from .decode import envelope_geometry

    carrier = carrier_mass(grid.polarity)
    charges = sorted(cand.charges)[:12]
    thr = 3.0 * noise_sigma

    def heights(mass: float) -> tuple[int, float]:
        env_sigma, offset = envelope_geometry(mass, compound_class)
        hits, total = 0, 0.0
        for z in charges:
            mz_pred = (mass + offset + z * carrier) / z
            half = max(0.5 * env_sigma / z, mz_pred / max(resolution, 1.0))
            lo = int(np.searchsorted(raw_mz, mz_pred - half))
            hi = int(np.searchsorted(raw_mz, mz_pred + half))
            if hi - lo < 1:
                continue
            h = float(raw_int[lo:hi].max())
            if h > thr:
                hits += 1
                total += h
        return hits, total

    start_hits, start_height = heights(cand.mass)
    if start_hits < 2 or start_height <= 0:
        return cand, full
    floor = max(2, int(round(0.6 * start_hits)))
    mass = cand.mass
    current = start_height
    moved = False
    for _ in range(max_steps):
        best_d, best_h = None, -1.0
        for d in deltas:
            if mass - d <= 0:
                continue
            hits, h = heights(mass - d)
            if hits >= floor and h >= min_ratio * current and h > best_h:
                best_d, best_h = d, h
        if best_d is None:
            break
        mass -= best_d
        current = best_h
        moved = True
    if not moved:
        return cand, full
    trial = Candidate(mass=mass, score=cand.score, charges=set(cand.charges),
                      intensity=cand.intensity)
    full2 = _fit_candidate(trial, residual, weights, grid, instrument, compound_class,
                           states_full, mz_range=mz_range)
    if full2[2] >= full[2] * (1.0 + min_gain):
        return trial, full2
    return cand, full


def _refine_mass(comp: Component, raw_mz, raw_int, polarity, tol_ppm: float = 200.0,
                 resolution: float = 30000.0, link_ppm: float = 60.0, link_da: float = 1.5):
    """Measure the mass on the raw spectrum, one charge state at a time.

    Each fitted charge state's envelope is located by a Gaussian-kernel mean shift as wide as
    the envelope (see :func:`lcmsdeconv.deconv.decode.refine_candidate_masses`), which sees
    resolved isotopologues as one envelope instead of hopping between them, and the envelope's
    skew is corrected so the result estimates the average mass. The per-charge masses are
    clustered and the cluster holding most of the fitted intensity gives the mass, so one charge
    state that overlaps another species' envelope does not drag the result.
    ``mass_spread_ppm`` is the weighted spread within that cluster.
    """
    from .decode import _mean_shift, envelope_geometry

    order = np.argsort(raw_mz)
    raw_mz = np.asarray(raw_mz)[order]
    raw_int = np.clip(np.asarray(raw_int, dtype=float)[order], 0, None)
    env_sigma, offset = envelope_geometry(comp.mass, comp.compound_class)
    inst_sigma_da = comp.mass / max(resolution, 1.0) / 2.354820045
    kern_da = float(np.sqrt(env_sigma ** 2 + inst_sigma_da ** 2))
    carrier = carrier_mass(polarity)
    masses, wts = [], []
    for z, inten in comp.charges.items():
        if inten <= 0:
            continue
        mz_pred = (comp.mass + offset + z * carrier) / z
        pos, kw = _mean_shift(raw_mz, raw_int, mz_pred, kern_da / z)
        if kw <= 0:
            continue
        m = z * (pos - carrier) - offset
        if abs(m - comp.mass) > comp.mass * tol_ppm * 1e-6:
            continue
        masses.append(m)
        wts.append(inten)
    if not masses:
        return
    ms = np.array(masses)
    ws = np.array(wts)
    srt = np.argsort(ms)
    ms, ws = ms[srt], ws[srt]
    link = max(comp.mass * link_ppm * 1e-6, link_da)
    groups = np.split(np.arange(ms.size), np.nonzero(np.diff(ms) > link)[0] + 1)
    best = max(groups, key=lambda g: ws[g].sum())
    bm, bw = ms[best], ws[best]
    mean = float(np.average(bm, weights=bw))
    comp.mass = mean
    comp.mass_spread_ppm = (float(np.sqrt(np.average((bm - mean) ** 2, weights=bw)) / mean * 1e6)
                            if bm.size > 1 else 0.0)


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


def drop_shared_evidence(components: list[Component], library: AdductLibrary, polarity: int,
                         resolution: float = 30000.0, shared_fraction: float = 0.8,
                         trace: list | None = None) -> list[Component]:
    """Remove components whose envelopes a stronger component already explains.

    A charge misassignment reads a strong envelope as some other mass: charge 40 read as 39,
    or a whole series read at a fraction of its charge. The resulting "species" has no ions of
    its own; its envelopes sit on the stronger species' envelopes (any of its adduct forms
    included), where a stronger fitted column already accounts for them. A genuine weak species
    sits between them. When at least ``shared_fraction`` of a component's fitted intensity lies
    within two envelope sigmas of a stronger column of a stronger component, the component is
    dropped as an artefact. A real dimer survives, since its odd charge states are its own.
    """
    from .decode import envelope_geometry

    if len(components) < 2:
        return components
    carrier = carrier_mass(polarity)
    masses = {s.label or "base": s.mass for s in library.states()}
    masses.setdefault("base", 0.0)
    order = sorted(components, key=lambda c: -c.intensity)
    kept: list[Component] = []
    kept_cols: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (mz, half-width, coef)
    for c in order:
        cols = [(z, lab, v) for (z, lab), v in c.columns.items() if v > 0]
        if kept and cols:
            total = sum(v for _z, _lab, v in cols)
            shared = 0.0
            for z, lab, v in cols:
                mz = (c.mass + masses.get(lab or "base", 0.0) + z * carrier) / z
                for kmz, khw, kcoef in kept_cols:
                    hit = (np.abs(kmz - mz) <= khw) & (kcoef >= v)
                    if hit.any():
                        shared += v
                        break
            if total > 0 and shared >= shared_fraction * total:
                if trace is not None:
                    trace.append({"pass": -1, "stage": "fit", "candidate": float(c.mass),
                                  "outcome": "shared_evidence", "mass": float(c.mass)})
                continue
        kept.append(c)
        if cols:
            env_sigma, _off = envelope_geometry(c.mass, c.compound_class)
            zs = np.array([z for z, _lab, _v in cols], dtype=float)
            mz = np.array([(c.mass + masses.get(lab or "base", 0.0) + z * carrier) / z
                           for z, lab, _v in cols])
            # two envelope sigmas hold 95 % of an isotope envelope's ions
            hw = np.maximum(2.0 * env_sigma / zs, mz / max(resolution, 1.0))
            kept_cols.append((mz, hw, np.array([v for _z, _lab, v in cols])))
    return kept


def observed_charges(mass: float, charges, raw_mz: np.ndarray, raw_int: np.ndarray, thr: float,
                     polarity: int, compound_class: str, resolution: float,
                     mz_range: tuple[float, float] | None = None, max_gap: int = 1,
                     z_cap: int = 150) -> set[int]:
    """Charges of a species' envelope that the spectrum shows, grown outward from ``charges``.

    Starting from the charge band that voted for the species, each next charge outward is kept
    while an envelope is observed (above ``thr``) where that charge would put the species,
    tolerating ``max_gap`` missing states, and stopping at the edge of the measured m/z range.
    """
    from .decode import envelope_geometry

    charges = {int(z) for z in charges if z >= 1}
    if not charges:
        return charges
    env_sigma, offset = envelope_geometry(mass, compound_class)
    carrier = carrier_mass(polarity)

    def present(z: int) -> bool | None:
        mz = (mass + offset + z * carrier) / z
        if mz_range is not None and not (mz_range[0] <= mz <= mz_range[1]):
            return None
        half = max(0.5 * env_sigma / z, mz / max(resolution, 1.0))
        lo = int(np.searchsorted(raw_mz, mz - half))
        hi = int(np.searchsorted(raw_mz, mz + half))
        return bool(hi > lo and raw_int[lo:hi].max() > thr)

    out = set(charges)
    for direction in (-1, 1):
        z = (min(charges) - 1) if direction < 0 else (max(charges) + 1)
        gap = 0
        while 1 <= z <= z_cap:
            p = present(z)
            if p is None:
                break
            if p:
                out.add(z)
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            z += direction
    return out
