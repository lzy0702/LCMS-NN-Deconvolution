"""Probes on the four real runs behind docs/iwsp_transformer.md, sections 3 and 4.4.

Each probe changes one thing in the pipeline in this process only (the package is not modified)
and scores the result with scripts/validate_oligo_runs.py's truth and scoring:

    validate  the four-run validation with one change (--rule):
              comb       nothing changed (the reference: 13/20, 5/29, 39 reported, 7 unconfirmed)
              inrange    the comb's ladder mean taken over in-range neighbours only: a neighbour
                         position beyond the acquired m/z range is unknown, not an observed zero
              oracle     the true charge (z = 2-12) on every active bin inside the isotope
                         envelope (+-3.5 Da / z around the average m/z) of every known species
                         eluting in the region being deconvolved (+-0.1 min); comb elsewhere
              adjacent   the decoder's isotope check also requires two adjacent populated
                         isotopologue positions for z >= 2 (a singly charged series read at z
                         fills at most every z-th position)
    trace     for each missed species, the decoder stage at which it is lost (process_run's
              trace), optionally under --rule oracle
    snr       for each missed species, the signal-to-noise ratio of its strongest charge
              state's envelope in its region's summed spectrum and in 7 scans around it

--single-charge sets the method's ``single_charge: true`` for any of them. --resample changes how
spectra reach the grid (docs/iwsp_transformer.md, section 2):
              pipeline   as the package does it: a region's scans summed on one scan's m/z axis,
                         then resampled with each point standing for half the gap beside it
              capped     every resampling uses scan_grid's widths (no point stands for a gap)
              scans      capped, and a region's grid view is the sum of its scans each resampled
                         onto the grid, instead of the resampled sum on one scan's axis (the
                         decoder's raw-point checks still read the summed spectrum)

    python scripts/iwsp_probes/real_run_probes.py validate --rule oracle IPRPLC_*.raw IPHILIC_*.raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validate_oligo_runs as vor  # noqa: E402
from scan_grid import resample_scan  # noqa: E402

from lcmsdeconv.chem.instrument import estimate_noise_sigma  # noqa: E402
from lcmsdeconv.core.method import Method  # noqa: E402
from lcmsdeconv.deconv import classical, decode  # noqa: E402
from lcmsdeconv.features import regions as regions_mod  # noqa: E402
from lcmsdeconv.io.thermo import read_thermo_raw  # noqa: E402
from lcmsdeconv.nn.features import shifted  # noqa: E402
from lcmsdeconv.nn.grid import LogMzGrid  # noqa: E402
from lcmsdeconv.nn.infer import ChargePrediction  # noqa: E402
from lcmsdeconv.process import process_run  # noqa: E402

MP = vor.MP
METHOD = "orbitrap_iprp_neg_oligo"
#: the run and region being processed, for the oracle
STATE: dict = {"key": None, "region": None}


# ------------------------------------------------------------------------------- the changes
def _comb_inrange(self, grid, intensity, noise_sigma):
    """CombPredictor.predict_grid with each charge's ladder mean over in-range neighbours."""
    B = intensity.size
    top1_z = np.zeros(B, dtype=np.int16)
    top1_p = np.zeros(B, dtype=np.float32)
    top2_z = np.zeros(B, dtype=np.int16)
    top2_p = np.zeros(B, dtype=np.float32)
    apex = np.zeros(B, dtype=np.float32)
    ns = max(noise_sigma, 1e-9)
    s = np.clip(np.log10(1.0 + np.clip(intensity, 0, None) / ns), 0.0, 5.0)
    active = intensity > self.min_snr * ns
    if not active.any():
        return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)
    idx = np.arange(B)
    best = np.zeros(B)
    second = np.zeros(B)
    for z in range(1, self.z_max + 1):
        acc = np.zeros(B)
        n = np.zeros(B)
        for k in self.neighbours:
            for zz in (z + k, z - k):
                if zz < 1:
                    continue
                shift = int(round(np.log(z / zz) / grid.step))
                if shift == 0 or abs(shift) >= B:
                    continue
                acc += shifted(s, shift)
                n += (idx + shift >= 0) & (idx + shift < B)  # the change: count in-range rungs
        if not n.any():
            continue
        ev = np.where(n > 0, s * acc / np.maximum(n, 1), 0.0)
        better = ev > best
        second[better] = best[better]
        top2_z[better] = top1_z[better]
        best[better] = ev[better]
        top1_z[better] = z
        between = (~better) & (ev > second)
        second[between] = ev[between]
        top2_z[between] = z
    total = best + second + 1e-12
    top1_p = (best / total).astype(np.float32)
    top2_p = (second / total).astype(np.float32)
    for arr in (top1_z, top1_p, top2_z, top2_p):
        arr[~active] = 0
    if self.instrument is not None:
        zi, conf = self.isotope_charges(grid, s, active)
        use = zi > 0
        top2_z[use] = np.where(top1_z[use] != zi[use], top1_z[use], top2_z[use])
        top2_p[use] = np.where(top1_z[use] != zi[use], 1.0 - conf[use], top2_p[use])
        top1_z[use] = zi[use]
        top1_p[use] = conf[use]
    mid = intensity[1:-1]
    ismax = (mid >= intensity[:-2]) & (mid > intensity[2:]) & (mid > self.min_snr * ns)
    apex[1:-1] = ismax.astype(np.float32)
    return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)


def _oracle_labels(grid, size: int) -> np.ndarray:
    """Charge per bin inside the envelopes of the truth species eluting in the current region."""
    lab = np.zeros(size, dtype=np.int16)
    clash = np.zeros(size, dtype=bool)
    key, reg = STATE["key"], STATE["region"]
    if key is None or reg is None:
        return lab
    for _name, mass, rt, _kind in vor.TRUTH[key]:
        if not (reg[0] - 0.1 <= rt <= reg[1] + 0.1):
            continue
        m = mass[0] if isinstance(mass, tuple) else mass
        for z in range(2, 13):
            mz = (m - z * MP) / z if grid.polarity < 0 else (m + z * MP) / z
            a, b = sorted((int(grid.mz_to_bin(mz - 3.5 / z)), int(grid.mz_to_bin(mz + 3.5 / z))))
            lo, hi = max(0, a), min(size - 1, b)
            if hi < lo:
                continue
            seg = slice(lo, hi + 1)
            clash[seg] |= (lab[seg] > 0) & (lab[seg] != z)
            lab[seg] = np.where(lab[seg] == 0, z, lab[seg])
    lab[clash] = 0
    return lab


def _apply_oracle():
    orig_summed = regions_mod.Region.summed
    orig_predict = classical.CombPredictor.predict_grid

    def summed(self, run, *a, **kw):
        STATE["region"] = (self.start, self.end)
        return orig_summed(self, run, *a, **kw)

    def predict_grid(self, grid, intensity, noise_sigma):
        pred = orig_predict(self, grid, intensity, noise_sigma)
        lab = _oracle_labels(grid, intensity.size)
        use = (lab > 0) & (pred.top1_z > 0)  # only bins the pipeline treats as signal
        pred.top1_z[use] = lab[use]
        pred.top1_p[use] = 1.0
        pred.top2_z[use] = 0
        pred.top2_p[use] = 0.0
        return pred

    regions_mod.Region.summed = summed
    classical.CombPredictor.predict_grid = predict_grid


def _isotopes_contradict_adjacent(mass, z, centre_mz, raw_mz, raw_int, thr, polarity,
                                  compound_class, fwhm, min_populated=0.6, min_rel=0.3):
    """decode.isotopes_contradict_charge, also requiring two adjacent populated positions."""
    from lcmsdeconv.chem.adducts import carrier_mass
    from lcmsdeconv.chem.classes import class_isotope_pattern

    spacing = decode.ISOTOPE_SPACING / z
    if spacing < 2.0 * fwhm:
        return False
    pat = class_isotope_pattern(mass, compound_class)
    a = pat.abundances / max(pat.abundances.max(), 1e-300)
    major = a >= min_rel
    if major.sum() < 2:
        return False
    carrier = carrier_mass(polarity)
    avg = float((pat.abundances * pat.masses).sum() / max(pat.abundances.sum(), 1e-300))
    iso_mz = (pat.masses[major] - avg + mass + z * carrier) / z
    tol = max(0.5 * fwhm, 0.2 * spacing)
    best, best_adjacent = 0, False
    for shift in np.linspace(-0.5, 0.5, 11) * spacing:
        filled = []
        for x in iso_mz + shift:
            lo = int(np.searchsorted(raw_mz, x - tol))
            hi = int(np.searchsorted(raw_mz, x + tol))
            filled.append(bool(hi > lo and raw_int[lo:hi].max() > thr))
        populated = sum(filled)
        adjacent = any(filled[i] and filled[i + 1] for i in range(len(filled) - 1))
        if populated > best or (populated == best and adjacent and not best_adjacent):
            best, best_adjacent = populated, adjacent
    if best < min_populated * major.sum():
        return True
    return z >= 2 and not best_adjacent


def apply_rule(rule: str) -> None:
    if rule == "inrange":
        classical.CombPredictor.predict_grid = _comb_inrange
    elif rule == "oracle":
        _apply_oracle()
    elif rule == "adjacent":
        decode.isotopes_contradict_charge = _isotopes_contradict_adjacent


# ------------------------------------------------------------------------------- resampling
#: summed region spectra by the id of their m/z array: (the array, its frames)
_FRAMES: dict = {}


def apply_resample(mode: str) -> None:
    from lcmsdeconv.core.model import Run
    from lcmsdeconv.deconv import pipeline as pipeline_mod
    from lcmsdeconv.nn.grid import LogMzGrid

    if mode == "pipeline":
        return

    def resample_capped(self, mz, intensity, centroided=None):
        return resample_scan(self, mz, intensity)

    LogMzGrid.resample_profile = resample_capped
    if mode != "scans":
        return
    orig_sum, orig_to_grid = Run.sum_frames, pipeline_mod.spectrum_to_grid

    def sum_frames(self, frames):
        out = orig_sum(self, frames)
        _FRAMES[id(out.mz)] = (out.mz, frames)
        return out

    def to_grid(mz, intensity, polarity, step=2e-5, mz_max=10000.0, is_profile=True,
                instrument=None):
        grid, observed, sigma = orig_to_grid(mz, intensity, polarity, step, mz_max, is_profile,
                                             instrument)
        hit = _FRAMES.get(id(mz))
        if hit is None or hit[0] is not mz or is_profile is False:
            return grid, observed, sigma
        observed = sum(resample_scan(grid, f.mz, f.intensity) for f in hit[1])
        return grid, observed, estimate_noise_sigma(observed)

    Run.sum_frames = sum_frames
    pipeline_mod.spectrum_to_grid = to_grid


# ------------------------------------------------------------------------------- the probes
def _missed(key, species):
    found = [(s.mass, s.rt_apex) for s in species]
    for name, mass, rt, kind in vor.TRUTH[key]:
        if kind not in ("paper", "si"):
            continue
        masses = mass if isinstance(mass, tuple) else (mass,)
        if not any(min(abs(m - mm) for mm in masses) < 1.0 and abs(t - rt) < 0.35
                   for m, t in found):
            yield name, masses, rt


def _method(single_charge: bool) -> Method:
    method = Method.load(METHOD)
    method.deconvolution.single_charge = single_charge
    return method


def trace(paths, single_charge: bool) -> None:
    stages = ["picked", "gated", "harmonics", "adduct_dedupe", "detected"]
    for path in paths:
        key = vor.run_key(path)
        STATE["key"] = key
        run = read_thermo_raw(path)
        tr: list = []
        res = process_run(run, _method(single_charge), trace=tr)
        print(f"== {key}")
        for name, masses, rt in _missed(key, res.species):
            recs = [r for r in tr if r["start"] - 0.05 <= rt <= r["end"] + 0.05]
            if not recs:
                print(f"   {name:22s} {masses[0]:8.2f} @ {rt:5.2f}: no region")
                continue
            rec = recs[0]

            def near(ms, masses=masses):
                return any(min(abs(m - mm) for mm in masses) < 1.0 for m in ms)

            out = []
            for st in rec["stages"]:
                stage = st.get("stage")
                if stage == "votes" and st.get("pass") == 0:
                    mm = np.exp(np.asarray(st["log_mass"]))
                    w = np.asarray(st["weight"])
                    sel = np.zeros(mm.size, bool)
                    for m0 in masses:
                        sel |= np.abs(mm - m0) < 1.0
                    rel = w[sel].sum() / max(w.max(), 1e-12) if sel.any() else 0.0
                    out.append(f"votes {rel:.2g}x the largest bin")
                elif stage in stages and st.get("pass") == 0:
                    out.append(f"{stage} {'Y' if near(st['masses']) else '-'}")
                elif stage == "components":
                    out.append(f"components {'Y' if near(st['masses']) else '-'}")
            later = [st for st in rec["stages"]
                     if st.get("stage") == "detected" and st.get("pass", 0) > 0]
            if later:
                out.append("later passes " + ("Y" if any(near(st["masses"]) for st in later)
                                              else "-"))
            print(f"   {name:22s} {masses[0]:8.2f} @ {rt:5.2f} (region {rec['start']:.2f}-"
                  f"{rec['end']:.2f}): " + "; ".join(out), flush=True)


def snr(paths, single_charge: bool) -> None:
    rows = []
    for path in paths:
        key = vor.run_key(path)
        STATE["key"] = key
        run = read_thermo_raw(path)
        res = process_run(run, _method(single_charge))
        ms1 = [s for s in run.spectra if s.ms_level == 1]
        rt = np.array([s.rt for s in ms1])
        pol = ms1[0].polarity
        lo_mz = min(float(np.min(s.mz)) for s in ms1 if s.mz.size)
        hi_mz = max(float(np.max(s.mz)) for s in ms1 if s.mz.size)
        grid = LogMzGrid(lo_mz, hi_mz, 2e-5, pol)
        cache: dict = {}

        def g(j, ms1=ms1, grid=grid, cache=cache):
            if j not in cache:
                cache[j] = resample_scan(grid, ms1[j].mz, ms1[j].intensity)
            return cache[j]

        for name, masses, srt in _missed(key, res.species):
            m0 = masses[0]
            c = int(np.searchsorted(rt, srt))
            win = range(max(0, c - 3), min(len(ms1), c + 4))
            wsum = sum(g(j) for j in win)
            wsig = estimate_noise_sigma(wsum)
            best = None
            for z in range(2, 13):
                mz = (m0 - z * MP) / z if pol < 0 else (m0 + z * MP) / z
                if not lo_mz < mz < hi_mz:
                    continue
                a, b = sorted((int(grid.mz_to_bin(mz - 3.5 / z)),
                               int(grid.mz_to_bin(mz + 3.5 / z))))
                v = wsum[a:b + 1].max() / wsig
                if best is None or v > best[1]:
                    best = (z, v, a, b)
            if best is None:
                continue
            z, snr_w, a, b = best
            regs = [r for r in res.regions if r.start - 0.05 <= srt <= r.end + 0.05]
            if regs:
                r = regs[0]
                rsum = sum(g(j) for j in range(len(ms1)) if r.start <= rt[j] <= r.end)
                snr_r = rsum[a:b + 1].max() / estimate_noise_sigma(rsum)
                where = f"region {r.start:.2f}-{r.end:.2f}: S/N {snr_r:7.1f}"
            else:
                snr_r = float("nan")
                where = "no region" + " " * 23
            rows.append((snr_r, snr_w))
            print(f"{key:15s} {name:22s} z={z:2d}  {where}   7 scans: S/N {snr_w:7.1f}", flush=True)
    r = np.array([row for row in rows if np.isfinite(row[0])])
    if r.size:
        print(f"\nin-region misses: median S/N {np.median(r[:, 0]):.1f} in the region's sum, "
              f"{np.median(r[:, 1]):.1f} in their own 7 scans")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe", choices=("validate", "trace", "snr"))
    ap.add_argument("raw", nargs="+", type=Path)
    ap.add_argument("--rule", default="comb", choices=("comb", "inrange", "oracle", "adjacent"))
    ap.add_argument("--single-charge", action="store_true")
    ap.add_argument("--resample", default="pipeline", choices=("pipeline", "capped", "scans"))
    args = ap.parse_args()
    apply_resample(args.resample)
    apply_rule(args.rule)
    if args.probe == "validate":
        orig_run_key = vor.run_key

        def run_key(path):
            STATE["key"] = orig_run_key(path)
            return STATE["key"]

        vor.run_key = run_key
        sys.argv = [sys.argv[0], *map(str, args.raw)] + (["--single-charge"]
                                                         if args.single_charge else [])
        vor.main()
    elif args.probe == "trace":
        trace(args.raw, args.single_charge)
    else:
        snr(args.raw, args.single_charge)


if __name__ == "__main__":
    main()
