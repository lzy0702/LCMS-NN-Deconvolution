"""The time-resolved isotope-chain test of docs/iwsp_transformer.md (sections 3 and 4.5).

For a mass M at a time t, for each charge z = 2..12 whose average m/z lies in the acquired range:
  1. the envelope (+-3.5 Da / z) is summed per scan; the scan where it peaks within +-0.3 min of t
     is the apex, and the 7 scans around it are the window;
  2. the anchor is the envelope's strongest grid bin in the window's sum;
  3. isotopologue positions are anchor +- j * 1.0033548 / z (j = 1, 2, 3), half-spacing positions
     anchor +- (j - 1/2) * 1.0033548 / z; a position co-elutes when its strongest bin within
     +-1 bin has signal in at least 4 of the 7 scans and its trace correlates with the anchor's at
     r >= --r;
  4. the charge with the most co-eluting isotopologues minus co-eluting half positions is kept.
M passes at t when it has >= 3 co-eluting isotopologues and <= 1 co-eluting half position.

Populations: the truth species of each run (validate_oligo_runs.py) at their tabulated times,
split by whether the comb pipeline finds them; the pipeline's unconfirmed reports (default and
``single_charge: true``); and random (mass, time) pairs, 1.2-7.2 kDa, 3-25 min.

    python scripts/iwsp_probes/isotope_chains.py IPRPLC_*.raw IPHILIC_*.raw
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

from lcmsdeconv.core.method import Method  # noqa: E402
from lcmsdeconv.io.thermo import read_thermo_raw  # noqa: E402
from lcmsdeconv.nn.grid import LogMzGrid  # noqa: E402
from lcmsdeconv.process import process_run  # noqa: E402

MP = vor.MP
ISO = 1.0033548
POPULATIONS = ("truth, found", "truth, missed", "unconfirmed (default)",
               "unconfirmed (single_charge)", "random")


class Chains:
    def __init__(self, run, r_min: float, mz_lo: float = 400.0, mz_hi: float = 2500.0):
        self.ms1 = [s for s in run.spectra if s.ms_level == 1]
        self.rt = np.array([s.rt for s in self.ms1])
        self.pol = self.ms1[0].polarity
        self.grid = LogMzGrid(mz_lo, mz_hi, 2e-5, self.pol)
        self.mz_lo, self.mz_hi = mz_lo, mz_hi
        self.r_min = r_min
        self._cache: dict = {}

    def scan(self, j: int) -> np.ndarray:
        if j not in self._cache:
            s = self.ms1[j]
            self._cache[j] = resample_scan(self.grid, s.mz, s.intensity)
        return self._cache[j]

    def __call__(self, mass: float, t: float) -> tuple[int, int, int]:
        """(co-eluting isotopologues, co-eluting half positions, charge) of the best charge."""
        grid, rt = self.grid, self.rt
        best = None
        for z in range(2, 13):
            mz = (mass - z * MP) / z if self.pol < 0 else (mass + z * MP) / z
            if not self.mz_lo + 0.5 < mz < self.mz_hi - 0.5:
                continue
            a, b = sorted((int(grid.mz_to_bin(mz - 3.5 / z)), int(grid.mz_to_bin(mz + 3.5 / z))))
            c0, c1 = int(np.searchsorted(rt, t - 0.3)), int(np.searchsorted(rt, t + 0.3))
            if c1 <= c0:
                continue
            env = np.array([self.scan(j)[a:b + 1].sum() for j in range(c0, c1)])
            c = c0 + int(np.argmax(env))
            tile = np.stack([self.scan(j) for j in range(max(0, c - 3), min(len(self.ms1), c + 4))])
            tot = tile.sum(0)
            k = a + int(np.argmax(tot[a:b + 1]))
            anchor = tile[:, k]
            if (anchor > 0).sum() < 4:
                continue
            ra = anchor - anchor.mean()

            def coelutes(mz_pos, tile=tile, tot=tot, ra=ra):
                bb = int(grid.mz_to_bin(mz_pos))
                lo, hi = max(0, bb - 1), min(grid.size - 1, bb + 1)
                p = tile[:, lo + int(np.argmax(tot[lo:hi + 1]))]
                if (p > 0).sum() < 4:
                    return 0
                rp = p - p.mean()
                den = np.linalg.norm(ra) * np.linalg.norm(rp)
                return int(den > 0 and (ra @ rp) / den >= self.r_min)

            mzk = float(grid.bin_to_mz(k))
            iso = sum(coelutes(mzk + sg * j * ISO / z) for j in (1, 2, 3) for sg in (1, -1))
            half = sum(coelutes(mzk + sg * (j - 0.5) * ISO / z) for j in (1, 2, 3) for sg in (1, -1))
            if best is None or iso - half > best[0] - best[1]:
                best = (iso, half, z)
        return best or (0, 0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", nargs="+", type=Path)
    ap.add_argument("--r", type=float, default=0.9, help="trace correlation for co-elution")
    ap.add_argument("--random", type=int, default=60, help="random (mass, time) pairs per run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for path in args.raw:
        key = vor.run_key(path)
        run = read_thermo_raw(path)
        chains = Chains(run, args.r)
        found = [(s.mass, s.rt_apex)
                 for s in process_run(run, Method.load("orbitrap_iprp_neg_oligo")).species]
        for name, mass, t, kind in vor.TRUTH[key]:
            if kind not in ("paper", "si"):
                continue
            masses = mass if isinstance(mass, tuple) else (mass,)
            hit = any(min(abs(m - mm) for mm in masses) < 1.0 and abs(tt - t) < 0.35
                      for m, tt in found)
            res = max((chains(m, t) for m in masses), key=lambda r: r[0] - r[1])
            rows.append((POPULATIONS[0] if hit else POPULATIONS[1], key, f"{name} @ {t}", *res))
        for single in (False, True):
            method = Method.load("orbitrap_iprp_neg_oligo")
            method.deconvolution.single_charge = single
            species = process_run(run, method).species
            sc = vor.score(key, species, lambda m, t, run=run: vor.joint_eic_peak(run, m, t))
            for s, why in sc["others"]:
                if why.startswith("unconfirmed"):
                    rows.append((POPULATIONS[3 if single else 2], key,
                                 f"{s.mass:.1f} @ {s.rt_apex:.2f}", *chains(s.mass, s.rt_apex)))
        t_hi = min(25.0, float(chains.rt.max()) - 1.0)
        for _ in range(args.random):
            m, t = float(rng.uniform(1200, 7200)), float(rng.uniform(3.0, t_hi))
            rows.append((POPULATIONS[4], key, f"{m:.1f} @ {t:.2f}", *chains(m, t)))
        print(f"{key} done", file=sys.stderr, flush=True)
    print(f"co-elution at r >= {args.r}; passing: >= 3 co-eluting isotopologues, <= 1 co-eluting "
          f"half position")
    for pop in POPULATIONS:
        sel = [r for r in rows if r[0] == pop]
        if sel:
            n_pass = sum(1 for r in sel if r[3] >= 3 and r[4] <= 1)
            print(f"  {pop:28s} {n_pass:3d} of {len(sel):3d} pass ({n_pass / len(sel):.0%})")
    for r in rows:
        if r[0] != "random" or (r[3] >= 3 and r[4] <= 1):
            print(f"    {r[0]:28s} {r[1]:15s} {r[2]:30s} isotopologues {r[3]}, half {r[4]}, z {r[5]}")


if __name__ == "__main__":
    main()
