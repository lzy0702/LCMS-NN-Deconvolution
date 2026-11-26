"""Classify each network's network-only unconfirmed reports on the Thermo runs.

A network's report is network-only when vrun.py scored it unconfirmed and the comb's run of the
same code (vrun.py's comb record of the same run) does not report it unconfirmed too (within the
truth matching's 1 Da and 0.35 min). Each is classified with the classification of the reports
analysis (nnloss REPORTS, analyze.py), re-implemented here: the decoder's pass-0 votes for the
report's mass are rebuilt from the network's own per-bin readings of its region's pass-0 input
(Region.summed, then spectrum_to_grid with the method's parameters, as deconvolve_spectrum does;
checked against the traces of the reports analysis when this was written); isotope verdicts
(isocharge.py on the region's summed raw spectrum), relations to other reports and to the truth
(harmonics, adducts and clusters, near-duplicates, comb twins), harmonic parents, z = 1-only
support, the main charge state's trace (spike rule) and the class rules are analyze.py's. The
votes are always pass 0's (the reports analysis took the pass in which a report was first picked).

    python bench/tools/classify_arms.py --runs RUNS_DIR --thermo-dir DIR --model LABEL=NET.onnx ...
        [--threads N] [--providers auto|dml|cuda|cpu] --out classify_arms.json

RUNS_DIR holds vrun.py records (any file names; each record names its run key and label): the
comb's ("comb") and each --model label's. Runs without both records are left out and listed.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/
import _paths  # noqa: E402, F401  (<repo>/src and <repo>/scripts on sys.path)

from tools import common  # noqa: E402

common.preset_threads()  # before numpy: --threads caps its BLAS / OpenMP / numba pools

import numpy as np  # noqa: E402
import validate_oligo_runs as v  # noqa: E402

from tools.isocharge import centroid_peaks, iso_charge, supported, supported_charge  # noqa: E402

MP = v.MP
METHOD = "orbitrap_iprp_neg_oligo"
#: bench run key -> validate_oligo_runs.run_key
RUNS = {"RNC": "IPRPLC_NC", "HNC": "IPHILIC_NC", "RGA": "IPRPLC_GalNAc", "HGA": "IPHILIC_GalNAc"}
WASH = {"IPRPLC_GalNAc": (25.0, 28.0), "IPRPLC_NC": (25.0, 28.0)}
FRONT = 0.6
ADD = {"Na": 21.982, "K": 38.090, "NH3": 17.031, "acetate": 60.052, "formate": 46.025,
       "TEA/HA": 101.190, "DIPEA/DBA": 129.247, "HFIP": 168.038, "H3PO4": 97.995,
       "+201.4": 201.4, "PO (-16.06)": -16.06}
COMBO = dict(ADD)
for _a in ("Na", "K", "TEA/HA", "HFIP", "acetate", "DIPEA/DBA"):
    for _b in ("Na", "K", "TEA/HA", "HFIP", "acetate", "DIPEA/DBA"):
        if _a <= _b:
            COMBO[f"{_a}+{_b}"] = ADD[_a] + ADD[_b]
for _a in ("Na", "K"):
    COMBO[f"3{_a}"] = 3 * ADD[_a]


def fwhm(mz):
    """Peak width of the Thermo runs' Orbitrap (140 000 at m/z 200)."""
    return mz / (140000.0 * (mz / 200.0) ** -0.5)


# ------------------------------------------------------------------ one run's inputs
class RunCtx:
    def __init__(self, key, raw, predictors):
        from lcmsdeconv.core.method import Method
        from lcmsdeconv.features.regions import find_regions
        from lcmsdeconv.io.thermo import read_thermo_raw
        from lcmsdeconv.process import _deconv_params

        self.key = key
        self.method = Method.load(METHOD)
        self.params = _deconv_params(self.method)
        self.run = read_thermo_raw(raw)
        self.predictors = predictors
        m = self.method
        self.regions = {int(r.id): r for r in find_regions(
            self.run, m.regions.source, -1, m.regions.window_min, m.regions.margin_min,
            m.regions.min_frames, integration=m.integration_for("tic"),
            min_snr=getattr(m.regions, "min_snr", 10.0),
            min_rel_height=getattr(m.regions, "min_rel_height", 0.002))}
        self.frames = self.run.frames()
        self.times = np.array([f.rt for f in self.frames])
        self.inp: dict = {}
        self.pred: dict = {}

    def region_of(self, rt):
        inside = [r for r in self.regions.values() if r.start - 0.05 <= rt <= r.end + 0.05]
        return min(inside, key=lambda r: abs(0.5 * (r.start + r.end) - rt)).id if inside else None

    def input(self, rid):
        """(grid, y, sigma, raw m/z, raw intensity) of region ``rid``'s pass-0 input."""
        if rid not in self.inp:
            from lcmsdeconv.nn.grid import spectrum_to_grid

            s = self.regions[rid].summed(self.run)
            grid, y, sigma = spectrum_to_grid(s.mz, s.intensity, getattr(s, "polarity", -1),
                                              self.params.grid_step, self.params.grid_mz_max,
                                              s.is_profile, self.method.instrument,
                                              frames=s.meta.get("frames"))
            o = np.argsort(s.mz)
            self.inp[rid] = (grid, y, float(sigma), np.asarray(s.mz, float)[o],
                             np.asarray(s.intensity, float)[o])
        return self.inp[rid]

    def readings(self, rid, label):
        """Network ``label``'s per-bin readings of region ``rid``'s pass-0 input (active bins)."""
        k = (rid, label)
        if k not in self.pred:
            grid, y, sigma, _x, _y = self.input(rid)
            r = self.predictors[label].predict_grid(grid, y, sigma)
            act = np.flatnonzero(y > 3.0 * max(sigma, 1e-9))
            self.pred[k] = {"idx": act, "u": grid.u[act], "I": y[act].astype(np.float64),
                            "z1": r.top1_z[act].copy(), "p1": r.top1_p[act].copy(),
                            "z2": r.top2_z[act].copy(), "p2": r.top2_p[act].copy(),
                            "apex": r.apex[act].copy(), "noise_sigma": sigma}
        return self.pred[k]

    def main_trace(self, mass, rt, main_z):
        """eic_detail's 'main' entry: scans with signal within +-0.25 min on the main state."""
        idx = np.flatnonzero(np.abs(self.times - rt) <= v.EIC_WINDOW)
        tt = self.times[idx]
        centre = (mass - main_z * MP) / main_z
        if idx.size < 3 or not 400 <= centre <= 2500:
            return {}
        lo, hi = centre - 1.2 / main_z, centre + 1.2 / main_z
        tr = np.array([float(self.frames[i].intensity[np.searchsorted(self.frames[i].mz, lo):
                                                       np.searchsorted(self.frames[i].mz, hi)]
                             .sum()) for i in idx])
        w = np.abs(tt - rt) <= 0.25
        return {"z": main_z, "n_signal": int(np.count_nonzero(tr[w] > 0)), "n_scans": int(w.sum())}


# ------------------------------------------------------------------ analyze.py, adapted
def votes(ctx, rid, label, M):
    p = ctx.readings(rid, label)
    _g, _y, ns, sx, sy = ctx.input(rid)
    u, inten, apex = p["u"], p["I"], p["apex"].astype(float)
    mz = np.exp(u) - MP
    groups = collections.defaultdict(lambda: {"w": 0.0, "w1": 0.0, "bins": set()})
    for which in ("1", "2"):
        zz = p["z" + which].astype(float)
        pp = p["p" + which].astype(float)
        ok = (zz >= 1) & (zz <= 40) & (pp >= 0.05)
        w = inten * pp * (0.2 + 0.8 * apex)
        sel = ok & (np.abs(zz * np.exp(u) - M) <= 1.2)
        for i in np.flatnonzero(sel):
            g = groups[int(zz[i])]
            g["w"] += float(w[i])
            if which == "1":
                g["w1"] += float(w[i])
            g["bins"].add(int(i))
    tot = sum(g["w"] for g in groups.values())
    out = []
    for zr, g in sorted(groups.items(), key=lambda kv: -kv[1]["w"]):
        b = np.array(sorted(g["bins"]))
        a = b[int(np.argmax(inten[b]))]
        x0 = float(mz[a])
        e = {"z": zr, "share": g["w"] / tot if tot > 0 else 0.0,
             "top1_share": g["w1"] / g["w"] if g["w"] else 0,
             "n": int(b.size), "mz": round(x0, 3), "snr": float(inten[a] / ns)}
        r = iso_charge(sx, sy, x0, 3 * ns, fwhm(x0))
        sup = supported(sx, sy, x0, 3 * ns, fwhm(x0), zr)[0]
        zs = zr if (sup or r.get("z") == zr) else supported_charge(sx, sy, x0, 3 * ns, fwhm(x0))
        if zs is None and r.get("z") is not None and r["z"] > 15:
            e["iso"], e["iso_note"] = r["z"], "dense"
        else:
            e["iso"], e["iso_note"] = zs, ("isolated" if zs is None else None)
        out.append(e)
    return out


def iso_verdict(groups):
    agg = collections.Counter()
    for g in groups:
        zi, zr = g.get("iso"), g["z"]
        iso_note = g.get("iso_note")
        if zr == 1:
            k = ("z=1 read as 1" if (zi == 1 or iso_note == "isolated")
                 else ("unread" if zi is None else "z>=2 read as 1"))
        elif zi is None:
            k = "isolated read as z>=2" if iso_note == "isolated" else "unread"
        elif zi > 15 and zi != zr:
            k = "dense (z>15 ladder, uncertain)"
        elif zi == zr:
            k = "agrees"
        elif zi == 1:
            k = "z=1 read as z>=2"
        elif zi > zr:
            k = "too low"
        else:
            k = "too high"
        agg[k] += g["share"]
    return dict(agg)


def relations(key, M, T, own_pct, species, comb_species):
    rel = []
    truth = [(n, (m[0] if isinstance(m, tuple) else m), r) for n, m, r, _k in v.TRUTH[key]]
    cands = [(f"{s['mass']:.2f}@{s['rt']:.2f}", s["mass"], s["rt"], s["pct"]) for s in species
             if not (abs(s["mass"] - M) < 1e-6 and abs(s["rt"] - T) < 1e-6)]
    cands += [(f"truth {n}", m, r, 1e9) for n, m, r in truth]
    for name, m, r, pct in cands:
        if pct < own_pct:
            continue
        dt = abs(r - T)
        if dt <= 0.35:
            for p_, q_ in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)):
                if abs(M - m * p_ / q_) <= 1.0 * max(1.0, p_ / q_):
                    rel.append(("harmonic", f"{p_}/{q_} of {name}"))
            d = M - m
            for lab_, dm in COMBO.items():
                lim = 0.5 if "+" not in lab_[1:] and not lab_.startswith("3") else 0.3
                if abs(d - dm) <= lim:
                    rel.append(("adduct", f"{name} {lab_} ({d:+.2f})"))
            if 1.0 < abs(d) <= 3.5:
                rel.append(("near-duplicate", f"{d:+.2f} Da from {name}"))
        elif dt <= 3.0 and abs(M - m) <= 1.0:
            rel.append(("near-duplicate", f"same mass ({M - m:+.2f}) as {name}, {r - T:+.2f} min"))
    for s in comb_species:
        dt = abs(s["rt"] - T)
        if ((0.35 < dt <= 3.0 and abs(s["mass"] - M) <= 1.0)
                or (dt <= 0.35 and 1.0 < abs(s["mass"] - M) <= 3.5)):
            rel.append(("comb-twin", f"comb reports {s['mass']:.2f}@{s['rt']:.2f}"))
    return rel


def parents(key, M, T, groups, species, comb_species):
    out = []
    pool = [(s["mass"], s["rt"]) for s in species] + [(s["mass"], s["rt"]) for s in comb_species]
    pool += [((m[0] if isinstance(m, tuple) else m), r) for _n, m, r, _k in v.TRUTH[key]]
    for g in groups:
        zi, zr = g.get("iso"), g["z"]
        if zi is None or zi == zr or zi > 15 or g["share"] < 0.1:
            continue
        mp = M * zi / zr
        ent = {"z_iso": zi, "z_read": zr, "share": g["share"], "hit": None}
        if zi != 1:
            hit = [(m, r) for m, r in pool
                   if abs(m - mp) <= 1.5 * max(1.0, zi / zr) and abs(r - T) <= 0.5]
            ent["hit"] = bool(hit) or None
        out.append(ent)
    return out


def classify(rec):
    """(primary class, flags) of a report record (analyze.py's rules)."""
    ver, rel = rec["iso_verdict"], rec["relations"]
    kinds = {r[0] for r in rel}
    flags = []
    one = ver.get("z=1 read as z>=2", 0) + ver.get("isolated read as z>=2", 0)
    z1own = ver.get("z=1 read as 1", 0)
    low, high = ver.get("too low", 0), ver.get("too high", 0)
    agree = ver.get("agrees", 0)
    harm = [p_ for p_ in rec.get("parents", []) if p_["hit"] and p_["share"] >= 0.3
            and any(abs(p_["z_iso"] / p_["z_read"] - f) < 1e-9 for f in (2, 3, 0.5, 1 / 3))]
    if "harmonic" in kinds or harm:
        flags.append("harmonic")
    w_hi = sum(g["share"] for g in rec["votes"] if g["z"] >= 2)
    if w_hi >= 0.1:
        sc = one >= 0.5 * w_hi and one + z1own >= 0.5
    else:
        sc = not any(x["supported"] for x in rec.get("fit_z_support", {}).values())
        if not sc:
            agree = max(agree, 0.5)
    if sc:
        flags.append("singly charged")
    if rec["main_trace"].get("n_signal") is not None and rec["main_trace"]["n_signal"] <= 2:
        flags.append("spike")
    if "adduct" in kinds:
        flags.append("adduct")
    if "near-duplicate" in kinds or "comb-twin" in kinds:
        flags.append("near-duplicate")
    if rec["context"] == "column wash":
        flags.append("column wash")
    if rec["context"] == "injection front":
        flags.append("injection front")
    if low + high >= 0.5:
        flags.append("charge misread")
    if agree >= 0.5:
        flags.append("charges agree")
    if harm or ("harmonic" in flags and (low + high) >= 0.3):
        primary = "harmonic"
    elif "singly charged" in flags:
        primary = "singly charged background"
    elif "harmonic" in flags:
        primary = "harmonic"
    elif "spike" in flags:
        primary = "spike"
    elif "adduct" in flags and agree >= 0.5:
        primary = "adduct or cluster copy"
    elif "near-duplicate" in flags:
        primary = "near-duplicate"
    elif "adduct" in flags:
        primary = "adduct or cluster copy"
    elif "column wash" in flags:
        primary = "column-wash region"
    elif "injection front" in flags:
        primary = "other: injection front"
    elif "charge misread" in flags:
        primary = "other: charges misread"
    elif "charges agree" in flags:
        primary = "other: charges agree with isotopes"
    else:
        primary = "other"
    return primary, flags


# ------------------------------------------------------------------ records
def species_list(d):
    base0 = max((s["base"] for s in d["species"]), default=1.0)
    return [{"mass": s["mass"], "rt": s["rt"], "pct": 100.0 * s["base"] / base0,
             "charges": {int(z): float(x) for z, x in s["charges"].items()}} for s in d["species"]]


def unconf(d):
    return [(o["mass"], o["rt"]) for o in d["others"] if o["why"] == "unconfirmed"]


def classify_report(ctx, label, d, comb_rec, mass, rt):
    key = ctx.key
    sp = species_list(d)
    s = min(sp, key=lambda s: abs(s["mass"] - mass) + abs(s["rt"] - rt))
    rid = ctx.region_of(s["rt"])
    rec = {"run": key, "net": label, "variant": "net", "mass": round(s["mass"], 2),
           "rt": round(s["rt"], 2), "pct": round(s["pct"], 4), "region": rid}
    if rid is None:
        rec["primary"] = "no region"
        return rec
    groups = votes(ctx, rid, label, s["mass"])
    cs, cu = species_list(comb_rec), unconf(comb_rec)
    rec["votes"] = groups
    rec["iso_verdict"] = iso_verdict(groups)
    rec["relations"] = relations(key, s["mass"], s["rt"], s["pct"], sp, cs)
    rec["context"] = ("injection front" if s["rt"] < FRONT else
                      "column wash" if key in WASH and WASH[key][0] <= s["rt"] <= WASH[key][1]
                      else "gradient")
    ch = sorted(s["charges"].items(), key=lambda kv: -kv[1])
    tot = sum(x for _, x in ch) or 1.0
    rec["charges"] = ", ".join(f"{z}:{x / tot:.2f}" for z, x in ch[:4])
    rec["main_trace"] = ctx.main_trace(s["mass"], s["rt"], ch[0][0]) if ch else {}
    rec["parents"] = parents(key, s["mass"], s["rt"], groups, sp, cs)
    w_hi = sum(g["share"] for g in groups if g["z"] >= 2)
    rec["fit_z_support"] = {}
    if w_hi < 0.1:
        _g, _y, ns, sx, sy = ctx.input(rid)
        for zz, xx in ch:
            if zz < 2 or xx < 0.1 * tot:
                continue
            c_ = (s["mass"] - zz * MP) / zz
            P_, H_ = centroid_peaks(sx, sy, c_ - 0.55 / zz, c_ + 0.55 / zz, 3 * ns)
            if P_.size:
                x0_ = float(P_[int(np.argmax(H_))])
                rec["fit_z_support"][zz] = {"supported": supported(sx, sy, x0_, 3 * ns,
                                                                   fwhm(x0_), zz)[0]}
            else:
                rec["fit_z_support"][zz] = {"supported": False}
    rec["primary"], rec["flags"] = classify(rec)
    rec["comb_reports"] = any(abs(c["mass"] - mass) <= v.MASS_TOL and abs(c["rt"] - rt) <= v.RT_TOL
                              for c in cs)
    rec["comb_unconfirmed"] = any(abs(m - mass) <= v.MASS_TOL and abs(t - rt) <= v.RT_TOL
                                  for m, t in cu)
    rec["top_votes"] = ", ".join(
        f"z{g['z']} {g['share']:.2f} iso {g['iso']}"
        + ("" if not g["iso_note"] else " " + g["iso_note"]) for g in groups[:3])
    return rec


def load_records(folder: Path) -> dict:
    """{(bench run key, label): record} of the vrun.py records in ``folder``."""
    short = {long: rk for rk, long in RUNS.items()}
    out = {}
    for p in sorted(folder.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or "label" not in d or d.get("key") not in short:
            continue
        out[(short[d["key"]], d["label"])] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, type=Path, help="folder of vrun.py records")
    ap.add_argument("--thermo-dir", required=True, type=Path,
                    help="folder holding the Thermo .raw files (searched recursively)")
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--providers", default="auto", choices=("auto", "dml", "cuda", "cpu"))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    models = common.parse_models(args.model)
    if not models:
        ap.error("no --model to classify")
    common.cap_threads(args.threads)
    from lcmsdeconv.core.method import Method
    from lcmsdeconv.nn.infer import ChargePredictor

    records = load_records(args.runs)
    raws = common.find_thermo_raws(args.thermo_dir)
    todo = collections.defaultdict(list)  # run -> [(label, mass, rt)]
    covered = {label: [] for label in models}
    missing = {label: [] for label in models}
    for label in models:
        for rk in RUNS:
            comb_rec, net_rec = records.get((rk, "comb")), records.get((rk, label))
            if comb_rec is None or net_rec is None or rk not in raws:
                missing[label].append(rk)
                continue
            covered[label].append(rk)
            cu = unconf(comb_rec)
            for m, t in unconf(net_rec):
                if not any(abs(m - mm) <= v.MASS_TOL and abs(t - tt) <= v.RT_TOL for mm, tt in cu):
                    todo[rk].append((label, m, t))
    inst = Method.load(METHOD).instrument
    prov = common.providers(args.providers)
    predictors = {label: ChargePredictor(path, providers=prov, instrument=inst)
                  for label, path in models.items()}
    res = []
    for rk in RUNS:
        items = todo.get(rk)
        if not items:
            continue
        t0 = time.time()
        ctx = RunCtx(RUNS[rk], raws[rk], predictors)
        for label, m, t in items:
            res.append(classify_report(ctx, label, records[(rk, label)], records[(rk, "comb")],
                                       m, t))
        print(f"{RUNS[rk]}: {len(items)} reports, {len(ctx.inp)} regions, {len(ctx.pred)} "
              f"readings, {time.time() - t0:.0f} s", flush=True)
        del ctx
    counts = {label: dict(collections.Counter(r["primary"] for r in res if r["net"] == label))
              for label in models}
    for label in models:
        print(f"{label}: {sum(counts[label].values())} network-only unconfirmed reports on "
              f"{', '.join(covered[label]) or 'no run'}: {counts[label]}", flush=True)
    common.write_json(args.out, {"reports": res, "runs": covered, "missing": missing,
                                 "counts": counts,
                                 "providers": {k: p.session.get_providers()
                                               for k, p in predictors.items()}})


if __name__ == "__main__":
    main()
