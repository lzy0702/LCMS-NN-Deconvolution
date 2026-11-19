"""What real LC-MS data look like as 2D tiles: docs/iwsp_transformer.md, sections 2 and 3.

Every scan is resampled onto the u grid (m/z 400-2500 unless --mz says otherwise) with the
pipeline's ``resample_profile``. A tile is ``--scans`` consecutive MS1 scans centred on a time.

    scans      scan count and interval, profile points per scan, grid bins with signal per scan
               and per tile
    widths     elution FWHM (in scans) of a species' charge-state envelopes: --species MASS@MIN
    profiles   per-bin trace statistics in tiles: max/mean over the tile, and the Pearson
               correlation with a reference species' envelope trace (--species MASS@MIN, at
               --ref-z) for that species' own charge states and for all other bins
    evidence   where the comb's winning ladder evidence comes from: one- or two-scan "spike"
               neighbours, persistent neighbours whose trace correlates below 0.5 with the bin's,
               and the share of the comb's z >= 2 mass votes on spike bins
    envelopes  bins inside the envelopes of the run's known species (validate_oligo_runs.py
               truth) as labels: the comb against four ladder variants (in-range mean,
               persistence gate, co-elution gate, all three)
    traces     a species' charge-state traces: the correlation of each charge state's strongest
               bins with one charge state's envelope trace (--species MASS@MIN, --ref-z)
    forms      for each known species of the run, the trace correlation and relative height of
               its +201.4 Da, +Na, +K and -16.06 Da forms (raw spectra, best charge, +-0.6 min)
    events     elution-event tokens of the whole run (local maxima in time and m/z) and their
               share of one- or two-scan events
    pairs      trace correlations of random pairs of strong peaks in a tile, and of the pairs
               that sit by chance at a charge-ladder offset

    python scripts/iwsp_probes/tile_statistics.py profiles IPRPLC_GalNAc_conjugated_mix.raw \
        --species 6793.98@19.25 --ref-z 4 --at 19.25 27.0 12.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointer_attention import PointerHead, comb_layer, readout  # noqa: E402

from lcmsdeconv.chem.instrument import estimate_noise_sigma  # noqa: E402
from lcmsdeconv.io.thermo import read_thermo_raw  # noqa: E402
from lcmsdeconv.nn.grid import LogMzGrid  # noqa: E402

MP = 1.00727646688
STEP = 2e-5


class Tiles:
    def __init__(self, path: Path, mz_lo: float, mz_hi: float, scans: int):
        self.run = read_thermo_raw(path)
        self.ms1 = [s for s in self.run.spectra if s.ms_level == 1]
        self.rt = np.array([s.rt for s in self.ms1])
        self.polarity = self.ms1[0].polarity
        self.grid = LogMzGrid(mz_lo, mz_hi, STEP, self.polarity)
        self.T = scans
        self._cache: dict = {}

    def scan(self, j: int) -> np.ndarray:
        if j not in self._cache:
            s = self.ms1[j]
            self._cache[j] = self.grid.resample_profile(np.asarray(s.mz, float),
                                                        np.asarray(s.intensity, float))
        return self._cache[j]

    def tile(self, t0: float) -> np.ndarray:
        c = int(np.searchsorted(self.rt, t0))
        lo = max(0, c - self.T // 2)
        return np.stack([self.scan(j) for j in range(lo, min(len(self.ms1), lo + self.T))])

    def envelope_bins(self, mass: float, z: int, half_da: float = 3.5) -> tuple[int, int]:
        mz = (mass - z * MP) / z if self.polarity < 0 else (mass + z * MP) / z
        a = int(self.grid.mz_to_bin(mz - half_da / z))
        b = int(self.grid.mz_to_bin(mz + half_da / z))
        return min(a, b), max(a, b)


def _unit(p: np.ndarray) -> np.ndarray:
    """Rows centred and scaled to unit length (a LayerNorm without gain): dot = Pearson r."""
    c = p - p.mean(axis=-1, keepdims=True)
    n = np.linalg.norm(c, axis=-1, keepdims=True)
    return np.where(n > 0, c / np.where(n > 0, n, 1.0), 0.0)


def _q(v) -> str:
    return " ".join(f"{np.percentile(v, p):5.2f}" for p in (10, 25, 50, 75, 90))


def cmd_scans(tl: Tiles, args) -> None:
    pts = np.array([s.mz.size for s in tl.ms1])
    print(f"{len(tl.ms1)} MS1 scans, {tl.rt.min():.2f}-{tl.rt.max():.2f} min, median interval "
          f"{np.median(np.diff(tl.rt)) * 60:.2f} s; profile points per scan: median "
          f"{np.median(pts):.0f}; grid bins {tl.grid.size}")
    import time

    for t0 in args.at:
        j = int(np.searchsorted(tl.rt, t0))
        nz = tl.scan(j) > 0
        tot = tl.tile(t0).sum(0)
        noise = estimate_noise_sigma(tot)
        s_all = np.clip(np.log10(1.0 + tot / noise), 0.0, 5.0)
        pos = np.nonzero(s_all > 0)[0]
        start = time.perf_counter()
        comb_layer(pos.astype(float), s_all[pos], tot.size)
        print(f"  {t0:6.2f} min: bins with signal in one scan {nz.sum()} ({nz.mean():.1%}); in a "
              f"{tl.T}-scan tile {np.sum(tot > 0)}, above 3 sigma {np.sum(tot > 3 * noise)}; "
              f"version 0's ladder layer {time.perf_counter() - start:.2f} s")


def cmd_widths(tl: Tiles, args) -> None:
    dt = np.median(np.diff(tl.rt)) * 60
    for spec in args.species:
        mass, t0 = (float(v) for v in spec.split("@"))
        w0, w1 = np.searchsorted(tl.rt, t0 - 1.0), np.searchsorted(tl.rt, t0 + 1.0)
        for z in range(3, 12):
            mz = (mass - z * MP) / z if tl.polarity < 0 else (mass + z * MP) / z
            lo, hi = mz - 4.0 / z, mz + 4.0 / z
            y = np.array([float(np.sum(s.intensity[np.searchsorted(s.mz, lo):
                                                   np.searchsorted(s.mz, hi)]))
                          for s in tl.ms1[w0:w1]])
            if y.max() <= 0:
                continue
            k = int(np.argmax(y))
            a = k
            while a > 0 and y[a - 1] >= y[k] / 2:
                a -= 1
            b = k
            while b < y.size - 1 and y[b + 1] >= y[k] / 2:
                b += 1
            print(f"M {mass} z={z:2d}: apex {tl.rt[w0 + k]:.2f} min, envelope sum {y.max():.3g}, "
                  f"FWHM {b - a + 1} scans ({(b - a + 1) * dt:.0f} s)")


def cmd_profiles(tl: Tiles, args) -> None:
    mass, ts = (float(v) for v in args.species[0].split("@"))
    for t0 in args.at:
        tile = tl.tile(t0)
        tot = tile.sum(0)
        nz = np.nonzero(tot > 0)[0]
        tok = nz[tot[nz] > 3 * np.median(tot[nz])]
        prof = tile[:, tok]
        peak = prof.max(0) / np.maximum(prof.mean(0), 1e-12)
        a, b = tl.envelope_bins(mass, args.ref_z, 4.0)
        ref = _unit(tile[:, a:b + 1].sum(1)[None, :])[0]
        corr = _unit(prof.T) @ ref
        print(f"tile at {t0} min: {tok.size} bins above 3x the median of {nz.size} with signal;"
              f" percentiles 10/25/50/75/90")
        print(f"   max/mean over the tile: {_q(peak)}")
        print(f"   r with the {mass} Da z={args.ref_z} trace: {_q(corr)}")
        if abs(t0 - ts) < 0.3:
            own = np.zeros(tok.size, bool)
            for z in range(3, 8):
                lo, hi = tl.envelope_bins(mass, z, 4.0)
                own |= (tok >= lo) & (tok <= hi)
            print(f"   the species' own z = 3-7 bins: {own.sum()}, median r "
                  f"{np.median(corr[own]):.2f}, median max/mean {np.median(peak[own]):.2f}; "
                  f"other bins median r {np.median(corr[~own]):.2f}")


def cmd_evidence(tl: Tiles, args) -> None:
    head = PointerHead()
    for t0 in args.at:
        tile = tl.tile(t0)
        tot = tile.sum(0)
        noise = estimate_noise_sigma(tot)
        s_all = np.clip(np.log10(1.0 + tot / noise), 0.0, 5.0)
        pos = np.nonzero(s_all > 0)[0]
        posf = pos.astype(float)
        s = s_all[pos]
        prof = tile[:, pos].T
        nscan = (prof > 0).sum(1)
        pz = _unit(prof)
        z1, p1, _ = readout(comb_layer(posf, s, tot.size))
        active = tot[pos] > 3 * noise
        sel = np.nonzero(active & (z1 >= 2))[0]
        vals = np.column_stack([s, (nscan <= 2).astype(float), pz])
        ev_all = ev_spike = ev_far = 0.0
        for z in np.unique(z1[sel]):
            q = sel[z1[sel] == z]
            for k in (1, 2, 3):
                for zz in (z + k, z - k):
                    if zz < 1:
                        continue
                    sh = int(round(np.log(z / zz) / STEP))
                    if sh == 0 or abs(sh) >= tot.size:
                        continue
                    out = head(posf, vals, posf[q] + sh)
                    sv, spike = out[:, 0], out[:, 1] > 0.5
                    r = np.einsum("nt,nt->n", out[:, 2:], pz[q])
                    ev_all += sv.sum()
                    ev_spike += sv[spike].sum()
                    ev_far += sv[(~spike) & (sv > 0) & (r < 0.5)].sum()
        votes = tot[pos] * p1 * (active & (z1 >= 2))
        print(f"tile at {t0} min: {active.sum()} active bins, {np.mean(nscan[active] <= 2):.0%} "
              f"with signal in at most 2 of {tl.T} scans")
        print(f"   winning ladder evidence from spike neighbours {ev_spike / ev_all:.0%}, from "
              f"persistent neighbours correlating below 0.5 {ev_far / ev_all:.0%}")
        print(f"   z >= 2 mass-vote weight on spike bins {votes[nscan <= 2].sum() / votes.sum():.1%};"
              f" median summed intensity, spike / persistent active bins: "
              f"{np.median(tot[pos][(nscan <= 2) & active]):.3g} / "
              f"{np.median(tot[pos][(nscan > 2) & active]):.3g}")


def _ladder(pos, s, prof, size, use_range, use_persist, use_coel, beta=200.0, gamma=20.0,
            rho=0.5, z_max=60):
    """Per charge, the mean over its in-range (or all) rungs of the gated rung log-SNR."""
    persistent = (prof > 0).sum(1) >= 3
    val = s * persistent if use_persist else s
    pz = _unit(prof)
    posf = pos.astype(float)
    m = np.zeros((pos.size, z_max))
    for z in range(1, z_max + 1):
        acc = np.zeros(pos.size)
        cnt = np.zeros(pos.size)
        for k in (1, 2, 3):
            for zz in (z + k, z - k):
                if zz < 1:
                    continue
                sh = int(round(np.log(z / zz) / STEP))
                if sh == 0 or abs(sh) >= size:
                    continue
                tgt = posf + sh
                lo = np.searchsorted(posf, tgt - 2, "left")
                hi = np.searchsorted(posf, tgt + 2, "right")
                width = int((hi - lo).max(initial=0))
                if width:
                    cand_raw = lo[:, None] + np.arange(width)[None, :]
                    ok = cand_raw < hi[:, None]
                    cand = np.minimum(cand_raw, pos.size - 1)
                    d = tgt[:, None] - posf[cand]
                    score = -beta * d * d
                    sink = -beta * 0.25
                    if use_coel:  # co-elution: a soft gate at r = rho, the sink moved to 0
                        score = score + gamma * (np.einsum("nt,nwt->nw", pz, pz[cand]) - rho)
                        sink = 0.0
                    score = np.where(ok, score, -np.inf)
                    mx = np.maximum(score.max(1), sink)
                    w = np.exp(score - mx[:, None])
                    acc += (w * val[cand]).sum(1) / (w.sum(1) + np.exp(sink - mx))
                cnt += ((tgt >= 0) & (tgt <= size - 1)) if use_range else 1.0
        m[:, z - 1] = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    if use_persist:
        m[~persistent] = 0.0
    return m


def cmd_envelopes(tl: Tiles, args) -> None:
    import validate_oligo_runs as vor

    truth = vor.TRUTH[vor.run_key(args.raw)]
    for t0 in args.at:
        c = int(np.searchsorted(tl.rt, t0))
        lo_scan = max(0, c - tl.T // 2)
        t_lo, t_hi = tl.rt[lo_scan], tl.rt[min(len(tl.ms1), lo_scan + tl.T) - 1]
        tile = tl.tile(t0)
        tot = tile.sum(0)
        noise = estimate_noise_sigma(tot)
        s_all = np.clip(np.log10(1.0 + tot / noise), 0.0, 5.0)
        pos = np.nonzero(s_all > 0)[0]
        s, prof = s_all[pos], tile[:, pos].T
        active = tot[pos] > 3 * noise
        label = np.zeros(pos.size, int)
        clash = np.zeros(pos.size, bool)
        names = []
        for name, mass, srt, _kind in truth:
            if not (t_lo - 0.1 <= srt <= t_hi + 0.1):
                continue
            names.append(name)
            m0 = mass[0] if isinstance(mass, tuple) else mass
            for z in range(2, 13):
                a, b = tl.envelope_bins(m0, z)
                inside = (pos >= a) & (pos <= b)
                clash |= inside & (label > 0) & (label != z)
                label[inside & (label == 0)] = z
        lab = (label > 0) & ~clash & active
        print(f"\ntile at {t0} min ({t_lo:.2f}-{t_hi:.2f}): {', '.join(names)}; {lab.sum()} "
              f"labelled active bins")
        for vname, flags in (("comb", (False, False, False)), ("in-range", (True, False, False)),
                             ("persistence", (False, True, False)),
                             ("co-elution", (False, False, True)), ("all three", (True, True, True))):
            z1, p1, _ = readout(_ladder(pos, s, prof, tot.size, *flags))
            right = z1 == label
            votes = tot[pos] * p1 * (active & (z1 >= 2))
            print(f"  {vname:>11}: labelled bins right {np.sum(s[lab] * right[lab]) / np.sum(s[lab]):.3f}"
                  f" (log-SNR) {np.sum(tot[pos][lab] * right[lab]) / np.sum(tot[pos][lab]):.3f}"
                  f" (intensity); share of the tile's z >= 2 vote weight on labelled bins "
                  f"{votes[lab].sum() / max(votes.sum(), 1e-12):.3f}")


def cmd_traces(tl: Tiles, args) -> None:
    mass, t0 = (float(v) for v in args.species[0].split("@"))
    tile = tl.tile(t0)
    a, b = tl.envelope_bins(mass, args.ref_z, 3.0)
    ref = _unit(tile[:, a:b + 1].sum(1)[None, :])[0]
    for z in range(2, 13):
        lo, hi = tl.envelope_bins(mass, z, 3.0)
        if z == args.ref_z or lo <= 0 or hi >= tl.grid.size - 1:
            continue
        env = tile[:, lo:hi + 1]
        tr = env.sum(1)
        strong = np.argsort(-env.sum(0))[:8]
        rs = _unit(env[:, strong].T) @ ref
        jump = np.max(tr[1:] / np.maximum(tr[:-1], 1.0))
        print(f"z={z:2d}: trace max {tr.max():.3g}; largest scan-to-scan rise x{jump:.3g}; r of its "
              f"8 strongest bins with the z={args.ref_z} trace: " + " ".join(f"{r:.2f}" for r in rs))


#: known species of each run (validate_oligo_runs.py truth), checked for co-eluting forms
FORM_SPECIES = {
    "IPRPLC_GalNAc": [("FLP", 6793.98, 19.21), ("n-1", 6448.71, 18.71), ("n+1", 7139.25, 20.34),
                      ("NC", 5162.25, 20.22)],
    "IPRPLC_NC": [("NC", 5162.25, 18.21), ("n-1", 4832.98, 17.54), ("PO-NC", 5146.21, 18.09),
                  ("n-2", 4503.71, 16.55), ("4842", 4842.5, 17.82)],
    "IPHILIC_GalNAc": [("FLP", 6793.98, 16.40), ("n-1", 6448.71, 15.94), ("n+1", 7139.25, 16.98),
                       ("PO", 6777.92, 16.98)],
    "IPHILIC_NC": [("NC", 5162.25, 13.41), ("PO-NC", 5146.21, 14.11), ("n-1", 4832.98, 13.03),
                   ("n-2", 4503.71, 12.60)]}
FORM_DELTAS = {"+201.4": 201.4, "+Na": 21.98194, "+K": 37.95588, "-16.06": -16.06}


def _envelope_trace(tl: Tiles, mass: float, z: int, t0: float, half_min: float = 0.6):
    """Per scan, the raw intensity within +-2 envelope sigmas of the z envelope's centre."""
    from lcmsdeconv.deconv.decode import envelope_geometry

    sig, off = envelope_geometry(mass, "ps_rna")
    c = (mass + off - z * MP) / z if tl.polarity < 0 else (mass + off + z * MP) / z
    hw = 2.0 * sig / z
    a, b = np.searchsorted(tl.rt, t0 - half_min), np.searchsorted(tl.rt, t0 + half_min)
    out = []
    for s in tl.ms1[a:b]:
        mz = np.asarray(s.mz)
        out.append(float(np.sum(np.asarray(s.intensity)[np.searchsorted(mz, c - hw):
                                                         np.searchsorted(mz, c + hw)])))
    return np.array(out)


def cmd_forms(tl: Tiles, args) -> None:
    import validate_oligo_runs as vor

    for name, mass, t0 in FORM_SPECIES[vor.run_key(args.raw)]:
        best = max(((z, _envelope_trace(tl, mass, z, t0)) for z in (3, 4, 5, 6)),
                   key=lambda zt: zt[1].max())
        z, ref = best
        cells = []
        for label, d in FORM_DELTAS.items():
            y = _envelope_trace(tl, mass + d, z, t0)
            r = (float(np.corrcoef(ref, y)[0, 1]) if ref.std() > 0 and y.std() > 0
                 else float("nan"))
            cells.append(f"{label} r={r:+.2f} h={y.max() / max(ref.max(), 1e-9):.3f}")
        print(f"{name:6s} z={z} max {ref.max():.1e}: " + "; ".join(cells))


def cmd_events(tl: Tiles, args) -> None:
    """Elution-event tokens of the whole run: local maxima of the time-smoothed (scan x bin) map
    (Gaussian sigma 1 scan; maximum over +-2 scans and +-1 bin) above 3 x the median per-scan
    noise sigma; a "spike" has raw signal in at most 2 of the 9 scans around its apex."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter

    from lcmsdeconv.chem.instrument import estimate_noise_sigma

    X = np.stack([tl.scan(j).astype(np.float32) for j in range(len(tl.ms1))])
    sig = np.array([estimate_noise_sigma(X[j].astype(float)) for j in range(0, X.shape[0], 10)])
    noise = float(np.median(sig[sig > 0]))
    S = gaussian_filter1d(X, 1.0, axis=0, mode="constant")
    ev = ((S == maximum_filter(S, size=(5, 1), mode="constant"))
          & (S == maximum_filter(S, size=(1, 3), mode="constant")) & (S > 3 * noise))
    c, b = np.nonzero(ev)
    pad = np.pad(X > 0, ((4, 4), (0, 0)))
    spikes = sum(pad[c + k, b] for k in range(9)) <= 2
    print(f"{X.shape[0]} scans x {X.shape[1]} bins: {c.size} event tokens, "
          f"{spikes.mean():.0%} of them with signal in at most 2 of the 9 scans around the apex")


def cmd_pairs(tl: Tiles, args) -> None:
    """How specific is co-elution? Pearson correlation of the traces (summed over +-1 bin) of the
    local maxima above 10 sigma in a tile: 20 000 random pairs, and the pairs that sit by chance at
    a z -> z+1 ladder offset (z = 2..6, +-1 bin)."""
    rng = np.random.default_rng(0)
    for t0 in args.at:
        tile = tl.tile(t0)
        tot = tile.sum(0)
        noise = estimate_noise_sigma(tot)
        mid = tot[1:-1]
        lm = np.nonzero((mid > tot[:-2]) & (mid >= tot[2:]) & (mid > 10 * noise))[0] + 1
        P = _unit(np.stack([tile[:, b - 1:b + 2].sum(1) for b in lm]))
        i, j = rng.integers(0, lm.size, (2, 20000))
        keep = i != j
        r_rand = np.einsum("nt,nt->n", P[i[keep]], P[j[keep]])
        where = {int(b): n for n, b in enumerate(lm)}
        r_lad = [float(P[n] @ P[where[int(b) + sh + e]])
                 for n, b in enumerate(lm) for z in range(2, 7)
                 for sh in (int(round(np.log(z / (z + 1)) / STEP)),) for e in (-1, 0, 1)
                 if int(b) + sh + e in where]
        q = lambda v: " ".join(f"{np.quantile(v, p):+.2f}" for p in (0.1, 0.5, 0.9))  # noqa: E731
        print(f"tile at {t0} min: {lm.size} maxima above 10 sigma; random pairs r q10/50/90 "
              f"{q(r_rand)}, above 0.9: {np.mean(r_rand > 0.9):.1%}; chance ladder pairs "
              f"({len(r_lad)}) above 0.9: {np.mean(np.array(r_lad) > 0.9):.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe", choices=("scans", "widths", "profiles", "evidence", "envelopes",
                                      "traces", "forms", "events", "pairs"))
    ap.add_argument("raw", type=Path)
    ap.add_argument("--at", type=float, nargs="+", default=[19.25, 27.0, 12.0],
                    help="tile centres (min)")
    ap.add_argument("--scans", type=int, default=24, help="scans per tile")
    ap.add_argument("--species", nargs="+", default=["6793.98@19.25"], help="MASS@MIN")
    ap.add_argument("--ref-z", type=int, default=4)
    ap.add_argument("--mz", type=float, nargs=2, default=[400.0, 2500.0])
    args = ap.parse_args()
    tl = Tiles(args.raw, args.mz[0], args.mz[1], args.scans)
    {"scans": cmd_scans, "widths": cmd_widths, "profiles": cmd_profiles,
     "evidence": cmd_evidence, "envelopes": cmd_envelopes, "traces": cmd_traces,
     "forms": cmd_forms, "events": cmd_events, "pairs": cmd_pairs}[args.probe](tl, args)


if __name__ == "__main__":
    main()
