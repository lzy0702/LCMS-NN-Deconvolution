"""What real LC-MS data look like as 2D tiles: docs/iwsp_transformer.md, sections 2 and 3.

Every scan is resampled onto the u grid (m/z 400-2500 unless --mz says otherwise) by
``scan_grid.resample_scan``, which keeps the edge points of vendor-thresholded segments from
standing for the gaps beside them. A tile is ``--scans`` consecutive MS1 scans centred on a time.

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
               truth) as labels: version 0 as the pipeline runs it (Z_max 40, the method's
               Orbitrap model, its isotope override) with and without the in-range mean, and the
               ladder part alone (softmax pointers, Z_max 60, no isotope override) against four
               ladder variants (in-range mean, persistence gate, co-elution gate, all three)
    traces     a species' charge-state traces: the correlation of each charge state's strongest
               bins with one charge state's envelope trace (--species MASS@MIN, --ref-z)
    forms      for each known species of the run, the trace correlation and relative height of
               its +201.4 Da, +Na, +K and -16.06 Da forms (raw spectra, best charge, +-0.6 min)
    events     elution-event tokens of the whole run (local maxima in time and m/z) and their
               share of one- or two-scan events
    pairs      trace correlations of random pairs of strong peaks in a tile, and of the pairs
               that sit by chance at a charge-ladder offset
    wash       the column wash's strong peaks by breadth: z = 1 isotope partners, FWHM, apex
               spread and pairwise trace correlation (use --scans 96 around 27 min)
    gaps       vendor-thresholded scans and the pipeline's region sums under resample_profile: the
               points that stand for a gap and their share of the resampled signal; the region
               sum's noise sigma three ways (the package's view, capped widths, the sum of
               per-scan grids) and the ramps Run.sum_frames' interpolation adds across gaps
    reelution  later elutions of a species (--species MASS@MIN): height and area against the main
               peak, from raw extracted-ion chromatograms

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
from scan_grid import resample_scan  # noqa: E402

from lcmsdeconv.chem.instrument import estimate_noise_sigma  # noqa: E402
from lcmsdeconv.io.thermo import read_thermo_raw  # noqa: E402
from lcmsdeconv.nn.grid import LogMzGrid, spectrum_to_grid  # noqa: E402

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
            self._cache[j] = resample_scan(self.grid, s.mz, s.intensity)
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

    import comb_transformer_v0 as v0

    from lcmsdeconv.core.method import Method
    from lcmsdeconv.deconv.classical import CombPredictor

    inst = Method.load("orbitrap_iprp_neg_oligo").instrument
    for t0 in args.at:
        j = int(np.searchsorted(tl.rt, t0))
        nz = tl.scan(j) > 0
        tot = tl.tile(t0).sum(0)
        noise = estimate_noise_sigma(tot)
        s_all = np.clip(np.log10(1.0 + tot / noise), 0.0, 5.0)
        pos = np.nonzero(s_all > 0)[0]
        start = time.perf_counter()
        comb_layer(pos.astype(float), s_all[pos], tot.size)
        t_soft = time.perf_counter() - start
        start = time.perf_counter()
        v0.forward(v0.write_weights(tl.grid.step, 40, tot.size), tl.grid, tot, noise, 40,
                   instrument=inst)
        t_v0 = time.perf_counter() - start
        start = time.perf_counter()
        CombPredictor(z_max=40, instrument=inst).predict_grid(tl.grid, tot, noise)
        t_comb = time.perf_counter() - start
        print(f"  {t0:6.2f} min: bins with signal in one scan {nz.sum()} ({nz.mean():.1%}); in a "
              f"{tl.T}-scan tile {np.sum(tot > 0)}, above 3 sigma {np.sum(tot > 3 * noise)}; on "
              f"the tile's sum, version 0 (Z_max 40) {t_v0:.3f} s, the comb {t_comb:.3f} s, the "
              f"softmax-pointer ladder of pointer_attention.py (Z_max 60, every bin with signal) "
              f"{t_soft:.2f} s")


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
            cells = []
            for z in range(3, 8):
                lo, hi = tl.envelope_bins(mass, z, 4.0)
                inside = (tok >= lo) & (tok <= hi)
                own |= inside
                if z != args.ref_z and inside.any():
                    s = np.log10(1.0 + tot[tok[inside]] / estimate_noise_sigma(tot))
                    cells.append(f"z={z} {np.median(corr[inside]):.2f} ({inside.sum()} bins, "
                                 f"{np.sum((s >= 1.2) & (corr[inside] <= 0.5))} with s >= 1.2 and "
                                 f"r <= 0.5)")
            print(f"   the species' own bins, median r with the z={args.ref_z} trace by charge: "
                  + "; ".join(cells))
            print(f"   other bins median r {np.median(corr[~own]):.2f}")


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
        from isotope_filter import VARIANTS, IsoCircuit
        from pointer_attention import InRangeComb

        from lcmsdeconv.core.method import Method
        from lcmsdeconv.deconv.classical import CombPredictor

        inst = Method.load("orbitrap_iprp_neg_oligo").instrument
        comb = CombPredictor(z_max=40, instrument=inst)
        full = {name: (pr.top1_z[pos], pr.top1_p[pos].astype(float)) for name, pr in (
            ("version 0", comb.predict_grid(tl.grid, tot, noise)),
            ("version 0 + in-range", InRangeComb(comb).predict_grid(tl.grid, tot, noise)),
            ("C2b", IsoCircuit(inst, z_max=40, **VARIANTS["c2b"]).predict_grid(tl.grid, tot, noise)),
            ("C2", IsoCircuit(inst, z_max=40, **VARIANTS["c2"]).predict_grid(tl.grid, tot, noise)))}
        for vname, flags in (("version 0", None), ("version 0 + in-range", None), ("C2b", None),
                             ("C2", None),
                             ("ladder only", (False, False, False)),
                             ("in-range", (True, False, False)),
                             ("persistence", (False, True, False)),
                             ("co-elution", (False, False, True)), ("all three", (True, True, True))):
            if flags is None:
                z1, p1 = full[vname]
            else:
                z1, p1, _ = readout(_ladder(pos, s, prof, tot.size, *flags))
            right = z1 == label
            votes = tot[pos] * p1 * (active & (z1 >= 2))
            if vname == "version 0":
                w = tot[pos][lab]
                pairs = {}
                for zt, zp, wi in zip(label[lab], z1[lab], w):
                    pairs[(int(zt), int(zp))] = pairs.get((int(zt), int(zp)), 0.0) + wi
                top = sorted(pairs.items(), key=lambda kv: -kv[1])[:4]
                print("  version 0, labelled intensity by (true z, top charge): " + "; ".join(
                    f"({a}, {b}) {v / w.sum():.0%}" for (a, b), v in top))
            print(f"  {vname:>20}: labelled bins right {np.sum(s[lab] * right[lab]) / np.sum(s[lab]):.3f}"
                  f" (log-SNR) {np.sum(tot[pos][lab] * right[lab]) / np.sum(tot[pos][lab]):.3f}"
                  f" (intensity); share of the tile's z >= 2 vote weight on labelled bins "
                  f"{votes[lab].sum() / max(votes.sum(), 1e-12):.3f}")


def cmd_traces(tl: Tiles, args) -> None:
    mass, t0 = (float(v) for v in args.species[0].split("@"))
    tile = tl.tile(t0)
    tot = tile.sum(0)
    noise = estimate_noise_sigma(tot)
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
        both = (tr[1:] > 0) & (tr[:-1] > 0)
        jump = np.max(tr[1:][both] / tr[:-1][both], initial=0.0)
        s = np.log10(1.0 + tot[lo:hi + 1][strong] / noise)
        print(f"z={z:2d}: trace max {tr.max():.3g}, {np.sum(tr == 0)} empty scans; largest rise "
              f"between non-empty scans x{jump:.3g}; r of its 8 strongest bins with the "
              f"z={args.ref_z} trace: " + " ".join(f"{r:.2f}" for r in rs)
              + f" (their log-SNR s in the tile's sum {s.min():.2f}-{s.max():.2f})")


#: the paper's species of each run (validate_oligo_runs.py truth) except the near-isobaric DA forms,
#: checked for co-eluting forms
FORM_SPECIES = {
    "IPRPLC_GalNAc": [("FLP", 6793.98, 19.21), ("PO", 6777.92, 19.21), ("n-1", 6448.71, 18.71),
                      ("n+1", 7139.25, 20.34), ("NC", 5162.25, 20.22)],
    "IPRPLC_NC": [("NC", 5162.25, 18.21), ("n-1", 4832.98, 17.54), ("PO-NC", 5146.21, 18.09),
                  ("n-2", 4503.71, 16.55)],
    "IPHILIC_GalNAc": [("FLP", 6793.98, 16.40), ("n-1", 6448.71, 15.94), ("n+1", 7139.25, 16.98),
                       ("PO", 6777.92, 16.98), ("NC", 5162.25, 13.72)],
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


def _fwhm(y: np.ndarray) -> int:
    k = int(np.argmax(y))
    a = k
    while a > 0 and y[a - 1] >= y[k] / 2:
        a -= 1
    b = k
    while b < y.size - 1 and y[b + 1] >= y[k] / 2:
        b += 1
    return b - a + 1


def cmd_wash(tl: Tiles, args) -> None:
    """The column wash in one tile (use --scans 96): strong local maxima (summed S/N above 30),
    split by the participation ratio PR = (sum x)^2 / sum x^2 of their traces (scans they
    effectively span); per class, how many carry a z = 1 isotope partner (a peak 1.0034 Th higher
    holding 5 % of their height; the same test 1.37 Th higher gives the chance rate), the FWHM of
    the 100 strongest traces (summed over +-1 bin), the spread of their apexes, and the pairwise
    trace correlation of the 300 strongest."""
    for t0 in args.at:
        tile = tl.tile(t0)
        c = int(np.searchsorted(tl.rt, t0))
        lo_scan = max(0, c - tl.T // 2)
        tot = tile.sum(0)
        noise = estimate_noise_sigma(tot)
        act = np.nonzero(tot > 30 * noise)[0]
        left = tot[np.clip(act - 1, 0, None)]
        right = tot[np.clip(act + 1, None, tot.size - 1)]
        lm = act[(tot[act] >= left) & (tot[act] >= right)]
        P = tile[:, lm]
        pr = P.sum(0) ** 2 / np.maximum((P ** 2).sum(0), 1e-300)
        apex = np.argmax(P, axis=0)
        mz = tl.grid.bin_to_mz(lm)
        def partner(th, lm=lm, mz=mz, tot=tot):
            n1 = np.log1p(th / np.abs(mz - tl.grid.carrier)) / STEP
            hit = np.zeros(lm.size, bool)
            for d in (-1, 0, 1):
                tgt = np.clip(np.round(lm + n1).astype(int) + d, 0, tot.size - 1)
                hit |= tot[tgt] >= 0.05 * tot[lm]
            return hit

        z1, chance = partner(1.0033548), partner(1.37)
        tr = np.stack([tile[:, max(b - 1, 0):b + 2].sum(1) for b in lm], axis=1)
        print(f"tile at {t0} min ({tl.T} scans): {lm.size} strong peaks")
        for label, sel in (("PR >= 20", pr >= 20), ("PR 8-20", (pr >= 8) & (pr < 20)),
                           ("PR < 4", pr < 4)):
            idx = np.nonzero(sel)[0]
            if idx.size < 5:
                continue
            top = idx[np.argsort(-tot[lm[idx]])[:300]]
            Q = _unit(P[:, top].T)
            rr = (Q @ Q.T)[np.triu_indices(top.size, 1)]
            a25, a75 = (tl.rt[lo_scan + int(np.percentile(apex[idx], q))] for q in (25, 75))
            fw = np.array([_fwhm(tr[:, k]) for k in top[:100]])
            print(f"   {label:8s}: {idx.size:5d} peaks, {np.mean(z1[idx]):.0%} with a z = 1 isotope "
                  f"partner (chance {np.mean(chance[idx]):.0%}); FWHM median {np.median(fw):.0f} "
                  f"scans (q90 {np.percentile(fw, 90):.0f}); apexes {a25:.2f}-{a75:.2f} min (IQR); "
                  f"pairwise r median {np.median(rr):.2f}, q90 {np.percentile(rr, 90):.2f}, above "
                  f"0.9 {np.mean(rr > 0.9):.1%}")


def cmd_gaps(tl: Tiles, args) -> None:
    """How vendor thresholding meets the package's resample_profile: in the scan nearest each
    --at, and in the sum of the scans within +-0.55 min of it as the pipeline forms region sums
    (Run.sum_frames: every scan interpolated onto the m/z axis of the one with most points), the
    points whose width 0.5 (u[i+1] - u[i-1]) exceeds three median spacings, and their share of
    what resample_profile puts on the grid (intensity x width)."""
    def widths(mz):
        u = tl.grid.mz_to_u(mz)
        du = np.empty_like(u)
        du[1:-1] = 0.5 * (u[2:] - u[:-2])
        du[0], du[-1] = u[1] - u[0], u[-1] - u[-2]
        return du, float(np.median(np.diff(u)))

    lo_mz, hi_mz = tl.grid.mz_min, tl.grid.mz_max
    for t0 in args.at:
        j = int(np.searchsorted(tl.rt, t0))
        s = tl.ms1[j]
        m = (s.mz > lo_mz) & (s.mz < hi_mz)
        mz, y = np.asarray(s.mz, float)[m], np.asarray(s.intensity, float)[m]
        du, med = widths(mz)
        wide = du > 3 * med
        print(f"scan {j} ({tl.rt[j]:.2f} min): {mz.size} points, {np.sum(y == 0)} of them zero; "
              f"{wide.mean():.1%} wider than 3 median spacings ({med / STEP:.2f} bins), the "
              f"widest {du.max() / STEP:.0f} bins; their share of the resampled signal "
              f"{np.sum(y[wide] * du[wide]) / np.sum(y * du):.0%}")
        frames = [f for f in tl.ms1 if abs(f.rt - t0) <= 0.55]
        tot = tl.run.sum_frames(frames)
        m = (tot.mz > lo_mz) & (tot.mz < hi_mz)
        mz, y = np.asarray(tot.mz, float)[m], np.asarray(tot.intensity, float)[m]
        du, med = widths(mz)
        wide = du > 3 * med
        print(f"   the pipeline's sum of {len(frames)} scans: {mz.size} points; {wide.mean():.1%} "
              f"wider than 3 median spacings, carrying {np.sum(y[wide] * du[wide]) / np.sum(y * du):.0%}"
              f" of the resampled signal")
        # the region's grid view three ways, on the grid the pipeline builds for this sum
        grid, view_pkg, sig_pkg = spectrum_to_grid(tot.mz, tot.intensity, tl.polarity)
        view_cap = resample_scan(grid, tot.mz, tot.intensity)
        view_scans = sum(resample_scan(grid, f.mz, f.intensity) for f in frames)
        sigmas = [sig_pkg, estimate_noise_sigma(view_cap), estimate_noise_sigma(view_scans)]
        raw = np.asarray(tot.intensity, float)
        print("   noise sigma of the region's grid view: " + " / ".join(
            f"{name} {s:.3g}" for name, s in zip(("package", "capped widths", "per-scan sum"),
                                                sigmas))
              + "; raw summed points above 3 sigma: " + " / ".join(
                  f"{np.mean(raw > 3 * s):.0%}" for s in sigmas))
        # what Run.sum_frames' interpolation adds: points of the reference axis inside a gap of
        # another scan (nearest point of that scan more than a bin away) get a ramp between its
        # segment edges
        ref = np.asarray(tot.mz, float)
        u_ref = grid.mz_to_u(ref)
        ramp = np.zeros(ref.size)
        for f in frames:
            fm, fy = np.asarray(f.mz, float), np.asarray(f.intensity, float)
            if fm.size < 2:
                continue
            v = np.interp(ref, fm, fy, left=0.0, right=0.0)
            pos = np.searchsorted(fm, ref)
            lo_i, hi_i = np.clip(pos - 1, 0, fm.size - 1), np.clip(pos, 0, fm.size - 1)
            uf = grid.mz_to_u(fm)
            dist = np.minimum(np.abs(u_ref - uf[lo_i]), np.abs(u_ref - uf[hi_i])) / grid.step
            ramp += np.where(dist > 1.0, v, 0.0)
        ramp_grid = resample_scan(grid, ref, ramp)
        print(f"   interpolated ramps across other scans' gaps: {ramp_grid.sum() / view_cap.sum():.1%} "
              f"of the capped region sum, which holds {view_cap.sum() / view_scans.sum():.2f} of the "
              f"per-scan sum's signal")


def cmd_reelution(tl: Tiles, args) -> None:
    """Later elutions of a species (--species MASS@MIN): the summed raw extracted-ion
    chromatogram of its main isotopologues (abundance >= 30 % of the largest) at z = 3-5, +-8
    ppm, smoothed (Gaussian, sigma 1.5 scans); local maxima within +-6 scans above 0.05 % of the
    apex and more than 1.5 min from it, with their height and area relative to the main peak."""
    from lcmsdeconv.chem.classes import class_isotope_pattern

    for spec in args.species:
        mass, _t0 = (float(v) for v in spec.split("@"))
        pat = class_isotope_pattern(mass, "ps_rna")
        a = np.asarray(pat.abundances, float)
        a = a / a.sum()
        iso = np.asarray(pat.masses, float) - float((a * pat.masses).sum()) + mass
        keep = iso[a >= 0.3 * a.max()]
        tr = np.zeros(len(tl.ms1))
        for z in (3, 4, 5):
            for mi in keep:
                mz0 = (mi - z * MP) / z if tl.polarity < 0 else (mi + z * MP) / z
                for k, s in enumerate(tl.ms1):
                    lo, hi = np.searchsorted(s.mz, [mz0 * (1 - 8e-6), mz0 * (1 + 8e-6)])
                    tr[k] += float(np.sum(s.intensity[lo:hi]))
        ap = int(np.argmax(tr))
        g = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
        sm = np.convolve(tr, g / g.sum(), mode="same")
        main = tr[max(0, ap - 15):ap + 16].sum()
        print(f"M {mass}: apex {tl.rt[ap]:.2f} min, height {tr[ap]:.3g}")
        for i in range(3, len(sm) - 3):
            if (sm[i] == sm[max(0, i - 6):i + 7].max() and sm[i] > 5e-4 * sm[ap]
                    and abs(tl.rt[i] - tl.rt[ap]) > 1.5):
                left, right = i, i
                while left > 0 and sm[left] > sm[i] / 2:
                    left -= 1
                while right < sm.size - 1 and sm[right] > sm[i] / 2:
                    right += 1
                w = right - left
                area = tr[max(0, left - w // 2):right + w // 2 + 1].sum()
                print(f"   later maximum at {tl.rt[i]:.2f} min: height {sm[i] / sm[ap]:.2%} of the "
                      f"apex (smoothed), FWHM {w} scans, area {area / main:.2%} of the main peak")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe", choices=("scans", "widths", "profiles", "evidence", "envelopes",
                                      "traces", "forms", "events", "pairs", "wash", "gaps",
                                      "reelution"))
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
     "forms": cmd_forms, "events": cmd_events, "pairs": cmd_pairs,
     "wash": cmd_wash, "gaps": cmd_gaps, "reelution": cmd_reelution}[args.probe](tl, args)


if __name__ == "__main__":
    main()
