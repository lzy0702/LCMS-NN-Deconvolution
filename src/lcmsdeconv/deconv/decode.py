"""Decode NN charge maps into candidate neutral masses via a mass histogram."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..nn.grid import LogMzGrid


@dataclass
class Candidate:
    mass: float
    score: float
    charges: set[int] = field(default_factory=set)
    intensity: float = 0.0


def accumulate_mass_histogram(
    grid: LogMzGrid,
    intensity: np.ndarray,
    prediction,
    noise_sigma: float,
    snr: float = 3.0,
    prob_min: float = 0.05,
    mass_step: float = 5e-6,
    mass_min: float = 100.0,
    mass_max: float = 500000.0,
    z_min: int = 1,
    z_max: int = 100,
    apex_floor: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Accumulate charge-weighted intensity into a log-mass histogram.

    Returns (log_mass_axis, weighted_intensity, charge_support_sets).
    """
    thr = snr * noise_sigma
    lm_min, lm_max = np.log(mass_min), np.log(mass_max)
    n = int((lm_max - lm_min) / mass_step) + 1
    hist = np.zeros(n, dtype=np.float64)
    charge_sets: list[set[int]] = [set() for _ in range(n)]

    u = grid.u
    for zarr, parr in ((prediction.top1_z, prediction.top1_p), (prediction.top2_z, prediction.top2_p)):
        mask = (intensity > thr) & (zarr >= z_min) & (zarr <= z_max) & (parr >= prob_min)
        if not np.any(mask):
            continue
        idx = np.nonzero(mask)[0]
        z = zarr[idx].astype(np.float64)
        mass = z * np.exp(u[idx])  # proton-charged neutral mass
        good = (mass >= mass_min) & (mass <= mass_max)
        idx, z, mass = idx[good], z[good], mass[good]
        apex_w = apex_floor + (1.0 - apex_floor) * np.asarray(prediction.apex, dtype=float)[idx]
        w = intensity[idx] * parr[idx] * apex_w
        hb = ((np.log(mass) - lm_min) / mass_step).astype(np.int64)
        hb = np.clip(hb, 0, n - 1)
        np.add.at(hist, hb, w)
        for b, zi in zip(hb, z.astype(int)):
            charge_sets[b].add(int(zi))
    lm_axis = lm_min + np.arange(n) * mass_step
    return lm_axis, hist, charge_sets


def pick_candidates(
    lm_axis: np.ndarray,
    hist: np.ndarray,
    charge_sets: list[set[int]],
    min_charge_support: int = 2,
    rel_height: float = 1e-4,
    smooth_bins: int = 5,
    merge_ppm: float = 200.0,
) -> list[Candidate]:
    """Peak-pick the mass histogram into candidate masses."""
    if hist.max() <= 0:
        return []
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    sm = uniform_filter1d(hist, size=max(1, smooth_bins))
    mx = maximum_filter1d(sm, size=max(3, smooth_bins * 2 + 1))
    thr = rel_height * sm.max()
    peaks = np.nonzero((sm == mx) & (sm > thr))[0]
    cands: list[Candidate] = []
    win = smooth_bins * 3
    for p in peaks:
        lo, hi = max(0, p - win), min(len(hist), p + win + 1)
        support: set[int] = set()
        for b in range(lo, hi):
            support |= charge_sets[b]
        if len(support) < min_charge_support and 1 not in support:
            # allow singly-charged species (small molecules) with strong evidence
            if not (support == {1} and sm[p] > 0.01 * sm.max()):
                if len(support) < min_charge_support:
                    continue
        # sub-bin parabolic interpolation for a mass accurate to a few ppm
        lm = float(lm_axis[p])
        if 0 < p < len(sm) - 1:
            y0, y1, y2 = sm[p - 1], sm[p], sm[p + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                shift = 0.5 * (y0 - y2) / den
                lm += float(np.clip(shift, -1.0, 1.0)) * (lm_axis[1] - lm_axis[0])
        mass = float(np.exp(lm))
        cands.append(Candidate(mass=mass, score=float(sm[p]), charges=support,
                               intensity=float(hist[lo:hi].sum())))
    cands.sort(key=lambda c: -c.score)
    # merge near-duplicate masses
    merged: list[Candidate] = []
    for c in cands:
        dup = False
        for m in merged:
            tol = max(m.mass * merge_ppm * 1e-6, 0.3)
            if abs(c.mass - m.mass) <= tol:
                m.charges |= c.charges
                dup = True
                break
        if not dup:
            merged.append(c)
    return merged


def dedupe_adduct_candidates(
    candidates: list[Candidate], deltas: dict[str, float], max_n: int = 3, tol_da: float = 0.5
) -> list[Candidate]:
    """Drop candidates that are an adduct of an already-kept (more intense) candidate.

    Their signal is captured by the kept candidate's adduct columns during fitting, so fitting
    them separately would double-count the same ions and leave a shifted, degenerate mass.
    """
    kept: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: -x.score):
        is_adduct = False
        for base in kept:
            dm = c.mass - base.mass
            if dm <= 0:
                continue
            for d in deltas.values():
                for n in range(1, max_n + 1):
                    if abs(dm - n * d) <= tol_da:
                        base.charges |= c.charges
                        is_adduct = True
                        break
                if is_adduct:
                    break
            if is_adduct:
                break
        if not is_adduct:
            kept.append(c)
    return kept


def refine_candidate_masses(
    candidates: list[Candidate],
    raw_mz: np.ndarray,
    raw_int: np.ndarray,
    polarity: int,
    noise_sigma: float = 0.0,
    tol_ppm: float = 400.0,
    min_charges: int = 2,
    max_spread_ppm: float = 150.0,
    snr: float = 3.0,
    drop_unsupported: bool = True,
    compound_class: str = "peptide",
) -> list[Candidate]:
    """Snap candidate masses onto observed peak apexes and drop unsupported candidates.

    A real neutral mass must produce an observed peak at its predicted m/z for at least
    ``min_charges`` charge states, and the masses those peaks imply must agree. Candidates
    generated from the wings of a stronger envelope fail both tests, which removes the shifted
    duplicates that would otherwise be absorbed by adduct columns and double-count intensity.
    """
    from ..chem.adducts import carrier_mass, mass_from_mz
    from ..chem.classes import class_isotope_pattern

    carrier = carrier_mass(polarity)
    thr = snr * noise_sigma
    out: list[Candidate] = []
    for c in candidates:
        pat = class_isotope_pattern(c.mass, compound_class)
        apex_offset = pat.average_mass - pat.most_abundant_mass
        masses, weights, matched_z = [], [], []
        for z in sorted(c.charges):
            mz_pred = (c.mass + z * carrier) / z
            tol = mz_pred * tol_ppm * 1e-6
            lo = int(np.searchsorted(raw_mz, mz_pred - tol))
            hi = int(np.searchsorted(raw_mz, mz_pred + tol))
            if hi - lo < 3:
                continue
            seg_i = raw_int[lo:hi]
            k = int(np.argmax(seg_i))
            if seg_i[k] <= thr:
                continue
            gk = lo + k
            # require a genuine local maximum, not a shoulder of a neighbouring peak
            if 0 < gk < raw_int.size - 1 and not (raw_int[gk] >= raw_int[gk - 1] and raw_int[gk] >= raw_int[gk + 1]):
                continue
            masses.append(mass_from_mz(float(raw_mz[gk]), z, polarity, 0.0) + apex_offset)
            weights.append(float(seg_i[k]))
            matched_z.append(int(z))
        if len(masses) < min_charges:
            if drop_unsupported and c.charges != {1}:
                continue
            out.append(c)
            continue
        masses = np.array(masses)
        weights = np.array(weights)
        order = np.argsort(masses)
        ms, ws = masses[order], weights[order]
        cw = np.cumsum(ws)
        med = float(ms[int(np.searchsorted(cw, 0.5 * cw[-1]))])
        spread = float(np.sqrt(np.average((masses - med) ** 2, weights=weights)) / med * 1e6)
        if drop_unsupported and spread > max_spread_ppm:
            continue
        c.mass = med
        c.charges = set(matched_z) | c.charges
        out.append(c)
    return out
