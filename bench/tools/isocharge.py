"""Charge of the ions at an m/z read off resolved isotope spacing in a region's summed raw
spectrum, independently of the package's estimators (used by classify_arms.py).

iso_charge(x, y, x0, thr, fwhm): the anchor is the strongest centroided peak (local maximum above
thr) within max(3 FWHM, 0.01 Th) of x0. For every charge z whose spacing s = 1.00336/z is at
least 2.5 FWHM, the four positions anchor -2s, -s, +s, +2s are looked up (a peak within
max(0.6 FWHM, 0.002 Th)); a position counts when its peak is at least 5 % of the anchor. z is
credited when an adjacent position (+-s) counts and, for z >= 2, at least two positions count
(a lone coincidence does not make a ladder); its score is the sum of the counted heights
relative to the anchor, each capped at 1, and a z >= 2 is credited only from a score of 0.8 (the
neighbours of an oligonucleotide isotopologue near its envelope's top are each 40 % of it or
more: the 17 known envelopes checked score 2.0-4.0). A z envelope fills its own positions and
those of its divisors (2s is the spacing of z/2) but not those of 2z, so the answer is the
largest credited z scoring at least 0.6 of the best (the comb's own rule). 'isolated' when no
charge is credited: no isotope partner an ion of an oligonucleotide above 1 kDa would have.
"""

from __future__ import annotations

import numpy as np

SPACING = 1.0033548


def centroid_peaks(x, y, lo, hi, thr):
    """(positions, heights) of the local maxima above ``thr`` in ``lo..hi`` (x sorted)."""
    a, b = np.searchsorted(x, [lo, hi])
    xs, ys = x[a:b], y[a:b]
    if xs.size < 3:
        return np.zeros(0), np.zeros(0)
    ii = np.nonzero((ys[1:-1] >= ys[:-2]) & (ys[1:-1] > ys[2:]) & (ys[1:-1] > thr))[0] + 1
    pos, hgt = [], []
    for i in ii:
        a2, b2 = max(i - 2, 0), min(i + 3, ys.size)
        w = np.clip(ys[a2:b2] - 0.5 * ys[i], 0, None)
        pos.append(float((xs[a2:b2] * w).sum() / w.sum()) if w.sum() > 0 else float(xs[i]))
        hgt.append(float(ys[i]))
    return np.array(pos), np.array(hgt)


def iso_charge(x, y, x0, thr, fwhm, zmax=40, anchor_tol=None):
    """The charge the isotope spacing around ``x0`` reads (module docstring)."""
    P, H = centroid_peaks(x, y, x0 - 2.2, x0 + 2.2, thr)
    if P.size == 0:
        return {"z": None, "note": "no peak"}
    at = anchor_tol if anchor_tol is not None else max(3 * fwhm, 0.01)
    near = np.abs(P - x0) <= at
    if not near.any():
        return {"z": None, "note": "no anchor"}
    j = int(np.argmax(np.where(near, H, -1)))
    xa, ha = P[j], H[j]
    tol = max(0.6 * fwhm, 0.002)
    scores = {}
    for z in range(1, zmax + 1):
        s = SPACING / z
        if s < 2.5 * fwhm:
            break
        hs = []
        for k in (-2, -1, 1, 2):
            d = np.abs(P - (xa + k * s)) <= tol
            r = float(H[d].max() / ha) if d.any() else 0.0
            hs.append(r if r >= 0.05 else 0.0)
        n = sum(h > 0 for h in hs)
        ok = (hs[1] > 0 or hs[2] > 0) and n >= (1 if z == 1 else 2)
        sc = round(sum(min(1.0, h) for h in hs), 3)
        scores[z] = sc if ok and (z == 1 or sc >= 0.8) else 0.0
    if not scores:
        return {"z": None, "note": "unresolved", "anchor": round(xa, 4)}
    best = max(scores.values())
    if best <= 0:
        return {"z": None, "note": "isolated", "anchor": round(xa, 4), "anchor_h": ha}
    zz = max(z for z, s in scores.items() if s >= 0.6 * best)
    top = {z: s for z, s in sorted(scores.items(), key=lambda kv: -kv[1])[:4] if s > 0}
    return {"z": zz, "anchor": round(xa, 4), "anchor_h": ha, "best": best, "top": top}


def supported(x, y, x0, thr, fwhm, z, carrier=-1.00727646688, anchor_tol=None):
    """Whether resolved isotopes support charge z for the ions at x0: the strongest peak within
    max(3 FWHM, 0.01 Th) of x0 has a partner one spacing (1.00336/z) away on either side of at
    least min(0.5, 0.25 r1), r1 = 0.38 per kDa of the mass z implies (an oligonucleotide's
    13C+15N+33S isotopologue against the monoisotopic peak; near an envelope's top both
    neighbours exceed it), and no partner half a spacing away as strong (that is charge 2z).
    Returns (bool, partner height, half-spacing height, threshold)."""
    P, H = centroid_peaks(x, y, x0 - 2.2, x0 + 2.2, thr)
    if P.size == 0:
        return False, 0.0, 0.0, None
    at = anchor_tol if anchor_tol is not None else max(3 * fwhm, 0.01)
    near = np.abs(P - x0) <= at
    if not near.any():
        return False, 0.0, 0.0, None
    j = int(np.argmax(np.where(near, H, -1)))
    xa, ha = P[j], H[j]
    tol = max(0.6 * fwhm, 0.002)
    s = SPACING / z

    def h(pos):
        d = np.abs(P - pos) <= tol
        return float(H[d].max() / ha) if d.any() else 0.0

    part = max(h(xa + s), h(xa - s))
    half = max(h(xa + s / 2), h(xa - s / 2)) if s / 2 >= 2.5 * fwhm else 0.0
    mass = z * (xa - carrier)
    need = min(0.5, 0.25 * 0.38 * mass / 1000.0)
    return bool(part >= need and half < part), round(part, 3), round(half, 3), round(need, 3)


def supported_charge(x, y, x0, thr, fwhm, zmax=15):
    """The largest charge 1..zmax that supported() accepts at x0 (a z envelope also has partners
    at the spacing of z/2, but there the half spacing holds the z partner, which supported()
    rejects), or None: no isotope partner that any oligonucleotide ion would show."""
    best = None
    for z in range(1, zmax + 1):
        if SPACING / z < 2.5 * fwhm:
            break
        if supported(x, y, x0, thr, fwhm, z)[0]:
            best = z
    return best
