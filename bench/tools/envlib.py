"""Envelope measurement shared by the real-run and the synthetic probes (real_env.py, synth_env.py,
blindspot.py).

A known species' charge state is located on the pipeline's grid view of a spectrum from its
mass and its class's isotope pattern: the isotopologues of mass M at charge z sit at
u = ln(m_i / z) on the grid (u = ln(m/z -/+ m_p)), so their spacing in bins is about
1.00335 / (M * step) whatever z is. The pattern is slid over +-0.5 Da to absorb the offset
between a tabulated average mass and the class model, and the envelope is accepted at z when
three or more major isotopologues (>= 20 % of the largest) stand above three sigma, the observed
isotopologue heights follow the model (cosine >= 0.8) and the half-spacing positions are
nearly empty (their mean below half the major isotopologues' mean). Accepted envelopes label
their isotopologue bins (+-1 bin) with z; a bin two envelopes label with different charges is
left out.

Every estimator is scored on the labelled bins the way nn.checks.ChargeScoreboard scores
synthetic ion bins: log-SNR weight (feature channel 0 above 0.5) and intensity weight (SNR
above 3), a predicted charge of 0 counting as wrong.
"""

from __future__ import annotations

import numpy as np

MP = 1.00727646688
ADDUCTS = {"+Na": 21.98194, "+K": 37.95588, "+201.4": 201.4}
WINDOW, OVERLAP = 32768, 0.25
ACTIVE_LEVEL, ACTIVE_TOP = 0.5, 6144
CATS = ("right", "z1", "harmonic", "neighbour", "higher", "lower", "none")


def pattern(mass: float, cls: str):
    """(masses, abundances relative to the largest) of the class model's isotopologues of at
    least 5 % at average mass ``mass``."""
    from lcmsdeconv.chem.classes import class_isotope_pattern

    pat = class_isotope_pattern(mass, cls)
    m = np.asarray(pat.masses, float) - pat.average_mass + mass
    a = np.asarray(pat.abundances, float)
    a = a / a.max()
    keep = a >= 0.05
    return m[keep], a[keep]


def fit_envelope(grid, y3, sigma, mass, z, cls, shift_da=None):
    """Best placement of ``mass``'s pattern at charge ``z`` on ``y3`` (the spectrum's 3-bin
    running maximum); None when it does not fit on the grid."""
    m, a = pattern(mass, cls)
    shifts = np.linspace(-0.5, 0.5, 41) if shift_da is None else np.array([shift_da])
    best = None
    for d in shifts:
        b = np.round((np.log((m + d) / z) - grid.u_min) / grid.step).astype(np.int64)
        if b.min() < 1 or b.max() >= grid.size - 1:
            return None
        v = y3[b]
        sc = float((a * v).sum())
        if best is None or sc > best[0]:
            best = (sc, d, b, v)
    _s, d, b, v = best
    major = a >= 0.2
    cos = float((a * v).sum() / max(np.linalg.norm(a) * np.linalg.norm(v), 1e-300))
    mm = m + d
    pair = major[:-1] & major[1:]
    if pair.any():
        hm = 0.5 * (mm[:-1] + mm[1:])[pair]
        hb = np.round((np.log(hm / z) - grid.u_min) / grid.step).astype(np.int64)
        r_half = float(y3[hb].mean() / max(v[major].mean(), 1e-300))
    else:
        r_half = 0.0
    n_above = int((v[major] > 3.0 * sigma).sum())
    ok = n_above >= 3 and cos >= 0.8 and r_half < 0.5
    return {"z": int(z), "mass": float(mass), "shift_da": float(d), "bins": b, "a": a, "v": v,
            "cos": cos, "r_half": r_half, "n_above": n_above, "ok": bool(ok),
            "apex": float(v.max()), "sum": float(v.sum())}


def window_starts(size: int) -> list[int]:
    """The tiling ChargePredictor.predict_grid uses for 32 768-bin windows."""
    step = max(1, int(WINDOW * (1 - OVERLAP)))
    starts = list(range(0, max(1, size - WINDOW + 1), step))
    if not starts or starts[-1] + WINDOW < size:
        starts.append(max(0, size - WINDOW))
    return starts


def active_map(y, sigma):
    """For every bin: whether LadderNet treats it as active (log-SNR above 0.5 and among the
    window's 6 144 highest) in the window whose answer the predictor keeps, and that window's
    count of bins with log-SNR above 0.5."""
    n = y.size
    level = np.clip(np.log10(1.0 + np.clip(y, 0, None) / max(sigma, 1e-6)), 0.0, 5.0)
    act = np.zeros(n, bool)
    count = np.zeros(n, np.int64)
    best_w = np.zeros(n)
    ramp = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(WINDOW) + 0.5) / WINDOW) + 1e-3
    thr = 3.0 * max(sigma, 1e-9)
    for s0 in window_starts(n):
        s1 = min(n, s0 + WINDOW)
        if y[s0:s1].max(initial=0.0) <= thr:
            continue
        lv = np.zeros(WINDOW)
        lv[:s1 - s0] = level[s0:s1]
        kth = np.partition(lv, WINDOW - ACTIVE_TOP)[WINDOW - ACTIVE_TOP]
        a = (lv > ACTIVE_LEVEL) & (lv >= kth)
        wl = ramp[:s1 - s0]
        idx = np.nonzero(wl > best_w[s0:s1])[0]
        act[s0 + idx] = a[idx]
        count[s0 + idx] = int((lv > ACTIVE_LEVEL).sum())
        best_w[s0 + idx] = wl[idx]
    return act, count, level


def fwhm_bins(y, b):
    """Full width at half maximum, in bins, of the peak whose top is within +-1 bin of b."""
    lo, hi = max(0, b - 1), min(y.size, b + 2)
    c = lo + int(np.argmax(y[lo:hi]))
    h = y[c] / 2.0
    if h <= 0:
        return float("nan")
    i = c
    while i > 0 and y[i - 1] > h:
        i -= 1
    j = c
    while j < y.size - 1 and y[j + 1] > h:
        j += 1
    left = i - (y[i] - h) / max(y[i] - y[i - 1], 1e-300) if i > 0 else float(i)
    right = j + (y[j] - h) / max(y[j] - y[j + 1], 1e-300) if j < y.size - 1 else float(j)
    return float(right - left)


def measure(grid, y, sigma, species, cls="ps_rna", z_lo=1, z_hi=40, adducts=True):
    """Envelopes of ``species`` [(name, masses tuple, info dict)] on one spectrum.

    Returns (envelopes, labels): envelope records with their properties (``_bins`` holds the
    labelled bins), and a per-bin label array (0 = unlabelled, -1 = clash)."""
    from scipy.ndimage import maximum_filter1d

    y = np.asarray(y, float)
    y3 = maximum_filter1d(y, 3, mode="constant")
    labels = np.zeros(y.size, np.int64)
    envs = []
    act, count, level = active_map(y, sigma)
    for name, masses, info in species:
        c = info.get("cls", cls)
        fits = []
        for z in range(z_lo, z_hi + 1):
            cands = [f for f in (fit_envelope(grid, y3, sigma, mm, z, c) for mm in masses)
                     if f is not None]
            if cands:
                fits.append(max(cands, key=lambda f: (f["ok"], f["sum"])))
        ok = [f for f in fits if f["ok"]]
        tot = sum(f["sum"] for f in ok) or 1.0
        for f in ok:
            rec = {"species": name, **{k: v for k, v in info.items() if k != "cls"},
                   "z": f["z"], "mass": f["mass"], "shift_da": f["shift_da"],
                   "cos": f["cos"], "r_half": f["r_half"], "n_above": f["n_above"],
                   "share": f["sum"] / tot, "n_states": len(ok),
                   "snr_apex": f["apex"] / max(sigma, 1e-12)}
            b = f["bins"]
            lab = np.unique(np.concatenate([b - 1, b, b + 1]))
            lab = lab[(lab >= 0) & (lab < y.size)]
            rec["_bins"] = lab
            top = int(b[int(np.argmax(f["v"]))])
            rec["fwhm_bins"] = fwhm_bins(y, top)
            rec["mz_top"] = float(grid.bin_to_mz(top))
            rec["spacing_bins"] = float(np.log((f["mass"] + 1.00335) / f["mass"]) / grid.step)
            lo, hi = max(0, int(b.min()) - 100), min(y.size, int(b.max()) + 101)
            near = np.zeros(hi - lo, bool)
            for bb in b:
                near[max(0, bb - 2 - lo):max(0, bb + 3 - lo)] = True
            rest = y[lo:hi][~near]
            rec["zero_frac"] = float((rest <= 0).mean()) if rest.size else float("nan")
            rec["bg_frac3"] = float((rest > 3 * sigma).mean()) if rest.size else float("nan")
            rec["bg_rel"] = float(rest[rest > 3 * sigma].sum() / max(f["v"].sum(), 1e-300))
            w0, w1 = max(0, top - 2000), min(y.size, top + 2001)
            rec["ch1_apex"] = float(y[top] / max(y[w0:w1].max(), 1e-300))
            sig_bins = lab[level[lab] > ACTIVE_LEVEL]
            rec["active_share"] = float(act[sig_bins].mean()) if sig_bins.size else float("nan")
            rec["window_level_bins"] = int(count[top])
            if adducts:
                for an, dm in ADDUCTS.items():
                    g = fit_envelope(grid, y3, sigma, f["mass"] + dm, f["z"], c,
                                     shift_da=f["shift_da"])
                    rec[f"adduct{an}"] = (float(g["sum"] / max(f["sum"], 1e-300))
                                          if g is not None and g["cos"] >= 0.7 else 0.0)
            envs.append(rec)
            clash = (labels[lab] > 0) & (labels[lab] != f["z"])
            labels[lab[clash]] = -1
            free = labels[lab] == 0
            labels[lab[free]] = f["z"]
    for rec in envs:
        lab = rec["_bins"]
        rec["_bins"] = lab[labels[lab] == rec["z"]]
    return envs, labels


def category(p: np.ndarray, z: int) -> np.ndarray:
    """Index into CATS for each predicted charge ``p`` against the true charge ``z``."""
    cat = np.full(p.shape, CATS.index("lower"), np.int64)
    cat[p > z] = CATS.index("higher")
    cat[np.abs(p - z) == 1] = CATS.index("neighbour")
    harm = ((p == 2 * z) | (p == 3 * z) | ((z % 2 == 0) & (p == z // 2))
            | ((z % 3 == 0) & (p == z // 3)))
    cat[harm & (p != z)] = CATS.index("harmonic")
    cat[(p == 1) & (z != 1)] = CATS.index("z1")
    cat[p <= 0] = CATS.index("none")
    cat[p == z] = CATS.index("right")
    return cat


def score(envs, y, sigma, preds: dict):
    """Adds to every envelope, per estimator, the weight sums and right sums under both
    weightings, and the intensity weight by kind of answer (CATS)."""
    y = np.asarray(y, float)
    s = np.clip(np.log10(1.0 + np.clip(y, 0, None) / max(sigma, 1e-6)), 0.0, 5.0)
    snr = np.clip(y, 0, None) / max(sigma, 1e-12)
    for rec in envs:
        b = rec["_bins"]
        z = rec["z"]
        wl = np.where(s[b] > 0.5, s[b], 0.0)
        wi = np.where(snr[b] > 3.0, snr[b], 0.0)
        rec["w_log"] = float(wl.sum())
        rec["w_int"] = float(wi.sum())
        rec["n_bins"] = int((wl > 0).sum())
        for name, top in preds.items():
            p = np.asarray(top)[b].astype(int)
            right = p == z
            cat = category(p, z)
            rec[f"acc_log:{name}"] = float((wl * right).sum())
            rec[f"acc_int:{name}"] = float((wi * right).sum())
            rec[f"cat:{name}"] = [float((wi * (cat == k)).sum()) for k in range(len(CATS))]
            vals = np.unique(p[wi > 0])
            wsum = {int(v): float(wi[p == v].sum()) for v in vals}
            rec[f"pred:{name}"] = dict(sorted(wsum.items(), key=lambda kv: -kv[1])[:4])
    return envs
