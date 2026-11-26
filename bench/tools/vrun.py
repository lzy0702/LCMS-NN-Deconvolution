"""process_run on one Thermo run with one charge predictor, scored against the truth of
scripts/validate_oligo_runs.py, with taps on the target species' path through the decoder.

    python bench/tools/vrun.py RAW [--model LABEL=NET.onnx] [--mode MODE] [--targets T.json]
        [--sub "NAME|NAME"] [--substates "4,3"] [--tile "NAME:Z:X"] [--confirm]
        [--threads N] [--providers auto|dml|cuda|cpu] --out OUT.json

The method is orbitrap_iprp_neg_oligo with ``single_charge`` off. N = the network (one
``--model``; ChargePredictor with the method's instrument, onnxruntime capped at --threads
intra-op threads); C = make_predictor("comb", ...) as process_run builds it. MODE (default: net
with --model, else comb):
  net          N unchanged
  comb         C unchanged (no --model needed; the record's label is "comb")
  iso          N, with C's isotope reader overriding top-1 wherever it reads a charge, exactly as
               CombPredictor overrides its own ladder (top1 = zi at 0.9, N's top-1 -> top-2 at 0.1)
  netapex      N's (z, p) everywhere, C's apex map everywhere
  combnetapex  C's (z, p) everywhere, N's apex map everywhere
  envfull      N, with C's whole prediction on the --sub targets' envelope bins
  envz         N, with C's (z, p) on those bins (N's apex kept)
  envapex      N's (z, p) with C's apex on those bins
  enviso       N with the isotope override on those bins only
  tile         N, but the tiling origin moved per target region so that --tile's state's apex bin
               sits X bins into its kept window (no other change)
Envelope bins: in regions overlapping the target's apex +-0.35 min, for each state with comb share
>= 0.01 (or --substates), the grid bins from the lightest to the heaviest isotopologue (>= 1 % of
the largest, class ps_rna) +-0.5 Da, divided by z.

The record (JSON): paper / SI species found ([found, listed]), reported species (1-10 kDa),
unconfirmed reports (with --confirm: neither a truth species, a form or harmonic of one, nor
EIC-confirmed), every row and report, the targets found, and the taps' log for the targets.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/
import _paths  # noqa: E402, F401  (<repo>/src and <repo>/scripts on sys.path)

from tools import common  # noqa: E402

common.preset_threads()  # before numpy: --threads caps its BLAS / OpenMP / numba pools

import numpy as np  # noqa: E402

METHOD = "orbitrap_iprp_neg_oligo"
MODES = ("net", "comb", "iso", "netapex", "combnetapex", "envfull", "envz", "envapex", "enviso",
         "tile")
FIELDS = ("top1_z", "top1_p", "top2_z", "top2_p", "apex")

CTX = {"region": None, "call": 0}
LOG: list[dict] = []


def envelope_mass_range(mass: float):
    from lcmsdeconv.chem.classes import class_isotope_pattern

    pat = class_isotope_pattern(mass, "ps_rna")
    m = pat.masses + (mass - pat.average_mass)
    a = pat.abundances
    keep = a >= 0.01 * a.max()
    apex_mass = float(m[int(np.argmax(a))])
    return float(m[keep].min() - 0.5), float(m[keep].max() + 0.5), apex_mass


class Target:
    def __init__(self, d):
        self.name = d["name"]
        self.mass = float(d["mass"])
        self.rt = float(d["rt"])
        self.shares = {int(z): float(w) for z, w in d["shares"].items()}
        self.lo, self.hi, self.apex_mass = envelope_mass_range(self.mass)

    def in_region(self, reg) -> bool:
        return reg is not None and reg[1] <= self.rt + 0.35 and reg[2] >= self.rt - 0.35

    def bins(self, grid, z):
        lo = int(grid.u_to_bin(np.log(self.lo / z)))
        hi = int(grid.u_to_bin(np.log(self.hi / z)))
        return max(lo, 0), min(hi + 1, grid.size)


def kept_window(B: int, b: int, W: int = 32768, overlap: float = 0.25):
    """The window start whose cosine weight ChargePredictor keeps for bin b (array length B)."""
    step = max(1, int(W * (1 - overlap)))
    starts = list(range(0, max(1, B - W + 1), step))
    if not starts or starts[-1] + W < B:
        starts.append(max(0, B - W))
    best, bs = -1.0, None
    for s0 in starts:
        if s0 <= b < s0 + W:
            w = 0.5 - 0.5 * np.cos(2 * np.pi * ((b - s0) + 0.5) / W) + 1e-3
            if w > best:
                best, bs = w, s0
    return bs


def votes(pred, intensity, noise, lo, hi, prob_min=0.05):
    """(weight by charge, total) of the histogram votes cast from bins lo:hi (decoder's rule)."""
    thr = 3.0 * noise
    out: dict[int, float] = {}
    sl = slice(lo, hi)
    inten = intensity[sl]
    apex = 0.2 + 0.8 * np.asarray(pred.apex[sl], dtype=float)
    for zarr, parr in ((pred.top1_z[sl], pred.top1_p[sl]), (pred.top2_z[sl], pred.top2_p[sl])):
        m = (inten > thr) & (zarr >= 1) & (zarr <= 40) & (parr >= prob_min)
        for z in np.unique(zarr[m]):
            mm = m & (zarr == z)
            out[int(z)] = out.get(int(z), 0.0) + float((inten[mm] * parr[mm] * apex[mm]).sum())
    return out, float(sum(out.values()))


def compose(P, Q, sel, fields):
    """P with Q's ``fields`` on the boolean mask ``sel``."""
    from lcmsdeconv.nn.infer import ChargePrediction

    arr = {f: np.array(getattr(P, f), copy=True) for f in FIELDS}
    for f in fields:
        arr[f][sel] = np.asarray(getattr(Q, f))[sel]
    return ChargePrediction(arr["top1_z"], arr["top1_p"], arr["top2_z"], arr["top2_p"],
                            arr["apex"], P.noise_sigma)


def iso_override(P, zi, conf, sel=None):
    from lcmsdeconv.nn.infer import ChargePrediction

    t1z, t1p = P.top1_z.copy(), P.top1_p.copy()
    t2z, t2p = P.top2_z.copy(), P.top2_p.copy()
    use = zi > 0
    if sel is not None:
        use = use & sel
    diff = use & (t1z != zi)
    t2z[diff] = t1z[diff]
    t2p[diff] = (1.0 - conf[diff]).astype(np.float32)
    t1z[use] = zi[use]
    t1p[use] = conf[use]
    return ChargePrediction(t1z, t1p, t2z, t2p, np.array(P.apex, copy=True), P.noise_sigma)


class Composed:
    """The predictor process_run is given: N, C or a composition of the two (MODE)."""

    def __init__(self, net, comb, mode, targets, sub, substates, tile):
        self.net, self.comb, self.mode = net, comb, mode
        self.targets, self.sub, self.substates, self.tile = targets, sub, substates, tile
        self.backend = "composed"
        self.model_path = None

    def _iso(self, grid, intensity, noise):
        ns = max(noise, 1e-9)
        s = np.clip(np.log10(1.0 + np.clip(intensity, 0, None) / ns), 0.0, 5.0)
        active = intensity > self.comb.min_snr * ns
        return self.comb.isotope_charges(grid, s, active)

    def _net(self, grid, intensity, noise, here):
        if self.mode != "tile":
            return self.net.predict_grid(grid, intensity, noise), 0
        spec = [(t, z, x) for (t, z, x) in self.tile if t in here]
        if not spec:
            return self.net.predict_grid(grid, intensity, noise), 0
        from lcmsdeconv.nn.infer import ChargePrediction

        t, z, x = spec[0]
        b = grid.mass_charge_to_bin(t.apex_mass, z)
        k = int((x - b) % 24576)
        padded = np.concatenate([np.zeros(k), intensity])
        P = self.net.predict_grid(grid, padded, noise)
        return ChargePrediction(P.top1_z[k:], P.top1_p[k:], P.top2_z[k:], P.top2_p[k:],
                                P.apex[k:], P.noise_sigma), k

    def _envelope_bins(self, grid, size, here):
        sel = np.zeros(size, dtype=bool)
        for t in here:
            if t.name not in self.sub:
                continue
            for z, w in t.shares.items():
                if (self.substates and z not in self.substates) or (not self.substates and w < 0.01):
                    continue
                lo, hi = t.bins(grid, z)
                sel[lo:hi] = True
        return sel

    def predict_grid(self, grid, intensity, noise_sigma):
        reg = CTX["region"]
        it = CTX["call"]
        CTX["call"] += 1
        here = [t for t in self.targets if t.in_region(reg)]
        B = intensity.size
        N = C = None
        k = 0
        if self.mode != "comb":
            N, k = self._net(grid, intensity, noise_sigma, here)
        if self.mode != "net" or here:
            C = self.comb.predict_grid(grid, intensity, noise_sigma)
        mode = self.mode
        sel = (self._envelope_bins(grid, B, here) if mode.startswith("env")
               else np.zeros(B, dtype=bool))
        if mode in ("net", "tile"):
            P = N
        elif mode == "comb":
            P = C
        elif mode == "iso":
            zi, conf = self._iso(grid, intensity, noise_sigma)
            P = iso_override(N, zi, conf)
        elif mode == "netapex":
            P = compose(N, C, np.ones(B, dtype=bool), ["apex"])
        elif mode == "combnetapex":
            P = compose(C, N, np.ones(B, dtype=bool), ["apex"])
        elif mode == "envfull":
            P = compose(N, C, sel, list(FIELDS))
        elif mode == "envz":
            P = compose(N, C, sel, ["top1_z", "top1_p", "top2_z", "top2_p"])
        elif mode == "envapex":
            P = compose(N, C, sel, ["apex"])
        elif mode == "enviso":
            zi, conf = self._iso(grid, intensity, noise_sigma)
            P = iso_override(N, zi, conf, sel)
        else:
            raise ValueError(mode)
        if here:
            LOG.append(self._record(grid, intensity, noise_sigma, reg, it, B, k, sel, here,
                                    N, C, P))
        return P

    def _record(self, grid, intensity, noise_sigma, reg, it, B, k, sel, here, N, C, P):
        """What each prediction does on the targets' charge states in this call."""
        rec = {"region": reg[0], "start": reg[1], "end": reg[2], "call": it, "B": B,
               "tile_pad": k, "substituted_bins": int(sel.sum()), "states": []}
        zi = None
        if self.mode in ("iso", "enviso"):
            zi, _ = self._iso(grid, intensity, noise_sigma)
        for t in here:
            for z, w in t.shares.items():
                if w < 0.005:
                    continue
                lo, hi = t.bins(grid, z)
                if hi <= lo:
                    continue
                seg = intensity[lo:hi]
                snr = float(seg.max() / max(noise_sigma, 1e-12))
                row = {"target": t.name, "z": z, "share": w, "snr": snr, "lo": lo, "hi": hi}
                # apex term at the envelope's local maxima above 3 sigma
                mid = seg[1:-1]
                mx = np.nonzero((mid >= seg[:-2]) & (mid > seg[2:])
                                & (mid > 3 * noise_sigma))[0] + 1
                for lab, Q in (("N", N), ("C", C), ("P", P)):
                    if Q is None:
                        continue
                    v, tot = votes(Q, intensity, noise_sigma, lo, hi)
                    row[lab] = {"total": tot,
                                "share_z": (v.get(z, 0.0) / tot) if tot > 0 else None,
                                "dest": sorted(((zz, ww / tot) for zz, ww in v.items()),
                                               key=lambda a: -a[1])[:4] if tot > 0 else []}
                    row[lab]["apex_at_maxima"] = (float(np.mean(np.asarray(Q.apex)[lo + mx]))
                                                  if mx.size else None)
                    row[lab]["top1p_at_maxima"] = (float(np.mean(np.asarray(Q.top1_p)[lo + mx]))
                                                   if mx.size else None)
                if zi is not None:
                    zz = zi[lo:hi][seg > 3 * noise_sigma]
                    row["iso_read"] = {int(a): int(b)
                                       for a, b in zip(*np.unique(zz, return_counts=True))}
                # visible rungs in the window N keeps for the state's apex bin
                b = grid.mass_charge_to_bin(t.apex_mass, z)
                s0 = kept_window(B + k, b + k)
                if s0 is not None:
                    vis = {}
                    for z2, w2 in t.shares.items():
                        if z2 == z or w2 < 0.005:
                            continue
                        b2 = grid.mass_charge_to_bin(t.apex_mass, z2) + k
                        if s0 <= b2 < s0 + 32768 and abs(z2 - z) <= 3:
                            lo2, hi2 = t.bins(grid, z2)
                            vis[z2] = (float(intensity[lo2:hi2].max() / max(noise_sigma, 1e-12))
                                       if hi2 > lo2 else 0.0)
                    row["window_x"] = int(b + k - s0)
                    row["visible_rungs_snr"] = vis
                rec["states"].append(row)
        return rec


def install_taps(targets):
    """Region context, uncut candidate ranks and per-state gate verdicts for the targets."""
    from lcmsdeconv.deconv import pipeline as PL
    from lcmsdeconv.features import regions as R

    orig_summed = R.Region.summed

    def summed(self, run):
        CTX["region"] = (self.id, float(self.start), float(self.end), self.polarity)
        CTX["call"] = 0
        return orig_summed(self, run)

    R.Region.summed = summed

    orig_pick = PL.pick_candidates

    def pick(lm_axis, hist, support, **kw):
        out = orig_pick(lm_axis, hist, support, **kw)
        here = [t for t in targets if t.in_region(CTX["region"])]
        if here:
            full = orig_pick(lm_axis, hist, support, **{**kw, "max_candidates": 10 ** 6})
            any_support = orig_pick(lm_axis, hist, support,
                                    **{**kw, "max_candidates": 10 ** 6, "min_charge_support": 1})
            for t in here:
                near = [(i, c) for i, c in enumerate(full) if abs(c.mass - t.mass) <= 1.5]
                peaks = [(c.mass, sorted(c.charges), c.score / max(1e-30, any_support[0].score))
                         for c in any_support if abs(c.mass - t.mass) <= 3.0]
                LOG.append({"tap": "pick", "region": CTX["region"][0], "pass_call": CTX["call"] - 1,
                            "target": t.name, "n_uncut": len(full), "n_kept": len(out),
                            "uncut_rank": [(i, c.mass, sorted(c.charges), c.score)
                                           for i, c in near],
                            "top_score": full[0].score if full else None,
                            "peaks_any_support_3da": peaks})
        return out

    PL.pick_candidates = pick

    orig_gate = PL.refine_candidate_masses

    def gate(cands, *a, **kw):
        caller = sys._getframe(1).f_code.co_name
        here = ([t for t in targets if t.in_region(CTX["region"])] if caller == "detect" else [])
        pre = [(t, c, copy.deepcopy(c)) for t in here for c in cands if abs(c.mass - t.mass) <= 1.5]
        out = orig_gate(cands, *a, **kw)
        for t, c, cc in pre:
            kept = any(o is c for o in out)
            one = orig_gate([copy.deepcopy(cc)], *a, **{**kw, "min_charges": 1})
            LOG.append({"tap": "gate", "region": CTX["region"][0], "pass_call": CTX["call"] - 1,
                        "target": t.name, "cand_mass": cc.mass, "charges": sorted(cc.charges),
                        "kept": kept, "kept_mass": c.mass if kept else None,
                        "states_passing_alone": (sorted(one[0].gate_charges)
                                                 if one and one[0].gate_charges else []),
                        "states_failed": (sorted(one[0].gate_failed)
                                          if one and one[0].gate_failed else [])})
        return out

    PL.refine_candidate_masses = gate


def target_stages(trace, targets):
    """The decoder trace's stage snapshots near the targets, in regions overlapping them."""
    stages = []
    for reg in trace:
        span = (reg["region"], reg["start"], reg["end"], 0)
        near_targets = [t for t in targets if t.in_region(span)]
        for st in reg["stages"]:
            for t in near_targets:
                if "masses" in st:
                    charges = st.get("charges", [None] * len(st["masses"]))
                    near = [(i, m, charges[i]) for i, m in enumerate(st["masses"])
                            if abs(m - t.mass) <= 1.5]
                    stages.append({"region": reg["region"], "pass": st["pass"],
                                   "stage": st["stage"], "target": t.name,
                                   "n": len(st["masses"]), "near": near})
                elif st.get("stage") == "fit" and abs(st["candidate"] - t.mass) <= 1.5:
                    stages.append({"region": reg["region"], "pass": st["pass"], "stage": "fit",
                                   "target": t.name, "candidate": st["candidate"],
                                   "outcome": st["outcome"], "mass": st["mass"]})
    return stages


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path, help="the Thermo .raw file")
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                    help="the network (one; its .json sidecar beside it); without it the comb")
    ap.add_argument("--mode", default=None, choices=MODES,
                    help="default: net with --model, else comb")
    ap.add_argument("--targets", type=Path, default=None,
                    help="target species (JSON list, targets.py); default none")
    ap.add_argument("--sub", default="", help="targets whose envelope bins env* modes change")
    ap.add_argument("--substates", default="", help="their charge states (default: share >= 0.01)")
    ap.add_argument("--tile", default="", help="NAME:Z:X for mode tile")
    ap.add_argument("--confirm", action="store_true",
                    help="EIC-check the reports that match no truth species")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--providers", default="auto", choices=("auto", "dml", "cuda", "cpu"))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    models = common.parse_models(args.model)
    if len(models) > 1:
        ap.error("one --model per run")
    label, net_path = next(iter(models.items()), ("comb", None))
    mode = args.mode or ("net" if net_path else "comb")
    if mode != "comb" and net_path is None:
        ap.error(f"--mode {mode} needs --model")
    if mode == "comb":
        label, net_path = "comb", None
    common.cap_threads(args.threads)

    import validate_oligo_runs as v

    from lcmsdeconv.core.method import Method
    from lcmsdeconv.deconv.classical import make_predictor
    from lcmsdeconv.io.thermo import read_thermo_raw
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.process import _isotope_mode, process_run

    targets = [Target(d) for d in json.loads(args.targets.read_text())] if args.targets else []
    sub = set(args.sub.split("|")) if args.sub else set()
    substates = {int(z) for z in args.substates.split(",")} if args.substates else set()
    tile = []
    if args.tile:
        name, z, x = args.tile.rsplit(":", 2)
        tile = [(next(t for t in targets if t.name == name), int(z), int(x))]
    install_taps(targets)

    t0 = time.perf_counter()
    method = Method.load(METHOD)
    method.deconvolution.single_charge = False
    run = read_thermo_raw(args.raw)
    t_load = time.perf_counter() - t0
    instrument = method.instrument
    comb = make_predictor("comb", z_max=min(int(method.deconvolution.charge_range[1]), 60),
                          instrument=instrument if _isotope_mode(method) else None)
    # the method's instrument, which a network with isotope-charge channels needs
    net = None
    used = None
    if net_path is not None:
        net = ChargePredictor(net_path, providers=common.providers(args.providers),
                              instrument=instrument)
        used = net.session.get_providers()
    pred = Composed(net, comb, mode, targets, sub, substates, tile)
    trace: list = []
    t1 = time.perf_counter()
    result = process_run(run, method, predictor=pred, trace=trace)
    t_run = time.perf_counter() - t1

    key = v.run_key(args.raw)
    species = [s for s in result.species if 1000.0 <= s.mass <= 10000.0]
    if args.confirm:
        def conf(m, t):
            return v.joint_eic_peak(run, m, t)
    else:
        def conf(m, t):
            return False
    sc = v.score(key, species, conf)
    rows = [{"name": n, "mass": m, "rt": r, "kind": k,
             "found": None if s is None else [s.mass, s.rt_apex]} for n, m, r, k, s in sc["rows"]]
    others = [{"mass": s.mass, "rt": s.rt_apex, "why": why} for s, why in sc["others"]]
    out = {"raw": args.raw.name, "key": key, "label": label, "model": net_path, "mode": mode,
           "providers": used, "threads": args.threads, "sub": sorted(sub),
           "substates": sorted(substates), "tile": args.tile, "load_s": t_load, "run_s": t_run,
           "n_reported": len(species),
           "paper": [sum(r["found"] is not None for r in rows if r["kind"] == "paper"),
                     sum(r["kind"] == "paper" for r in rows)],
           "si": [sum(r["found"] is not None for r in rows if r["kind"] == "si"),
                  sum(r["kind"] == "si" for r in rows)],
           "unconfirmed": (sum(o["why"] == "unconfirmed" for o in others)
                           if args.confirm else None),
           "rows": rows, "others": others,
           "species": [{"mass": s.mass, "rt": s.rt_apex, "base": s.base_intensity,
                        "charges": {str(z): float(w) for z, w in s.charges.items()}}
                       for s in species],
           "targets": {t.name: next((r["found"] for r in rows if r["name"] == t.name), None)
                       for t in targets},
           "stages": target_stages(trace, targets), "log": LOG}
    common.write_json(args.out, out)
    tl = ", ".join(f"{n}={'FOUND' if f else 'missed'}" for n, f in out["targets"].items())
    print(f"{key} {label} mode={mode} sub={args.sub} substates={args.substates} "
          f"tile={args.tile}: {tl or 'no targets'}; paper {out['paper']}, SI {out['si']}, "
          f"reported {len(species)}, unconfirmed {out['unconfirmed']}; "
          f"load {t_load:.0f}s run {t_run:.0f}s", flush=True)


if __name__ == "__main__":
    main()
