"""Circuit C2 of docs/iwsp_transformer.md: an isotope-spacing matched filter with half-spacing
inhibition, as a predictor with CombPredictor's interface (a prototype, not part of the package).

Per active bin and charge z whose isotope spacing d (in bins) is resolved (d >= 3 peak widths
and d >= 3 bins), reads of the log-SNR s at the isotopologue positions. The k-th position lies
ln(1 + k 1.0033548 / M) / step bins away (M = z (m/z -+ m_p)), not k times the first spacing:
    N1, N2, N3 = max(s(+k), s(-k)) for k = 1, 2, 3
    H          = max(s(+-1/2), s(+-3/2))   (the positions only a 2z reading has isotopologues at)
    E(z) = min(s0, N1) + 1/2 min(s0, N2) + 1/4 min(s0, N3) - 2 H                    (C2)
    E(z) = min(s0, N1) + 1/2 min(s0, N2) + 1/4 min(s0, N3) - 2 relu(H - 0.5 N1)     (C2b)
A read is the bin at the position, or the maximum over +-1 bin where the instrument's peak is
wider than one bin. The charge with the largest E wins where E >= s0 / 2 and E beats the runner-up
(by a quarter of s0 for C2, by anything for C2b); elsewhere the plain ladder comb's answer stands.

    real   the four-run validation with C2b (or C2, or the comb) as the pipeline's predictor
    bins   per-bin accuracy on the 96 spectra of scripts/eval_charge_bins.py

    python scripts/iwsp_probes/isotope_filter.py real --variant c2b IPRPLC_*.raw IPHILIC_*.raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from lcmsdeconv.deconv.classical import CombPredictor
from lcmsdeconv.nn.infer import ChargePrediction

SCRIPTS = Path(__file__).resolve().parents[1]
ISO = 1.0033548
VARIANTS = {"c2": {"rel_inhib": None, "margin": 0.25}, "c2b": {"rel_inhib": 0.5, "margin": 0.0}}


class IsoCircuit:
    def __init__(self, instrument, z_max=60, min_snr=3.0, w=(1.0, 0.5, 0.25), inhib=2.0,
                 res_factor=3.0, rel_inhib=None, margin=0.25):
        self.ladder = CombPredictor(z_max=z_max, instrument=None, min_snr=min_snr)
        self.inst = instrument
        self.z_max = z_max
        self.min_snr = min_snr
        self.w = w
        self.inhib = inhib
        self.res_factor = res_factor
        self.rel_inhib = rel_inhib
        self.margin = margin
        self.backend = "iso-circuit"
        self.model_path = None

    def predict_grid(self, grid, intensity, noise_sigma):
        base = self.ladder.predict_grid(grid, intensity, noise_sigma)
        ns = max(noise_sigma, 1e-9)
        s = np.clip(np.log10(1.0 + np.clip(intensity, 0, None) / ns), 0.0, 5.0)
        B = s.size
        idx = np.nonzero(intensity > self.min_snr * ns)[0]
        if idx.size == 0:
            return base
        mz = grid.bin_to_mz(idx)
        off = np.abs(mz - grid.carrier)
        fwhm_b = np.log1p(mz / np.asarray(self.inst.resolving_power(mz)) / off) / grid.step
        wide_peak = fwhm_b > 1.0  # read the maximum over +-1 bin
        s1 = maximum_filter1d(s, 3, mode="constant")

        def side(k, z):
            def read(x):
                x = np.clip(np.round(x).astype(np.int64), 0, B - 1)
                return np.where(wide_peak, s1[x], s[x])
            up = np.log1p(k * ISO / z / off) / grid.step
            down = np.log1p(-k * ISO / z / off) / grid.step
            return np.maximum(read(idx + up), read(idx + down))

        s0 = s[idx]
        best = np.full(idx.size, -np.inf)
        second = np.full(idx.size, -np.inf)
        zbest = np.zeros(idx.size, dtype=np.int16)
        for z in range(1, self.z_max + 1):
            d = np.log1p(ISO / z / off) / grid.step
            res = (d >= self.res_factor * fwhm_b) & (d >= 3.0)
            if not res.any():
                break
            n1, n2, n3 = side(1.0, z), side(2.0, z), side(3.0, z)
            h = np.maximum(side(0.5, z), side(1.5, z))
            inhibit = h if self.rel_inhib is None else np.maximum(h - self.rel_inhib * n1, 0.0)
            e = (self.w[0] * np.minimum(s0, n1) + self.w[1] * np.minimum(s0, n2)
                 + self.w[2] * np.minimum(s0, n3) - self.inhib * inhibit)
            e = np.where(res, e, -np.inf)
            up = e > best
            second = np.where(up, best, np.maximum(second, e))
            zbest = np.where(up, z, zbest)
            best = np.where(up, e, best)
        ok = (best >= 0.5 * s0) & (best > second + self.margin * s0)
        top1, p1 = base.top1_z.copy(), base.top1_p.copy()
        top2, p2 = base.top2_z.copy(), base.top2_p.copy()
        sel = idx[ok]
        top2[sel] = np.where(top1[sel] != zbest[ok], top1[sel], top2[sel])
        top1[sel] = zbest[ok]
        p1[sel] = 0.9
        p2[sel] = 0.1
        return ChargePrediction(top1, p1, top2, p2, base.apex, noise_sigma)


def cmd_real(args) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import validate_oligo_runs as vor

    from lcmsdeconv.core.method import Method
    from lcmsdeconv.io.thermo import read_thermo_raw
    from lcmsdeconv.process import process_run

    tot = {"paper": [0, 0], "si": [0, 0], "reported": 0, "unconfirmed": 0}
    for path in args.raw:
        key = vor.run_key(path)
        method = Method.load("orbitrap_iprp_neg_oligo")
        run = read_thermo_raw(path)
        pred = None if args.variant == "comb" else IsoCircuit(
            method.instrument, z_max=min(int(method.deconvolution.charge_range[1]), 60),
            **VARIANTS[args.variant])
        species = process_run(run, method, predictor=pred).species
        sc = vor.score(key, species, lambda m, t, run=run: vor.joint_eic_peak(run, m, t))
        found = {"paper": [0, 0], "si": [0, 0]}
        missed = []
        for name, _mass, _rt, kind, s in sc["rows"]:
            if kind in found:
                found[kind][1] += 1
                found[kind][0] += s is not None
                if s is None:
                    missed.append(f"{name} ({kind})")
        unconfirmed = sum(1 for _s, why in sc["others"] if why.startswith("unconfirmed"))
        print(f"{key}: paper {found['paper'][0]}/{found['paper'][1]}, SI {found['si'][0]}/"
              f"{found['si'][1]}, reported {len(species)}, unconfirmed {unconfirmed}; missed: "
              + (", ".join(missed) or "none"), flush=True)
        for k in ("paper", "si"):
            tot[k][0] += found[k][0]
            tot[k][1] += found[k][1]
        tot["reported"] += len(species)
        tot["unconfirmed"] += unconfirmed
    print(f"{args.variant}: paper {tot['paper'][0]}/{tot['paper'][1]}, SI {tot['si'][0]}/"
          f"{tot['si'][1]}, reported {tot['reported']}, unconfirmed {tot['unconfirmed']}")


def cmd_bins(args) -> None:
    sys.path.insert(0, str(SCRIPTS))
    from train_charge_net import mixture

    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.checks import BIN_BANDS, POOLED, WEIGHTINGS, ChargeScoreboard
    from lcmsdeconv.synth.dataset import FrameDataset

    frames = FrameDataset(mixture(), args.frames, base_seed=20_000_000)
    board = ChargeScoreboard()
    kinds: dict[str, int] = {}
    names = ["comb", "comb, no isotope charges", "C2", "C2b"]
    for i in range(args.frames):
        cfg, fs = frames.frame(i)
        grid = bl.frame_grid(cfg, fs)
        kind = bl.spectrum_kind(fs.meta)
        kinds[kind] = kinds.get(kind, 0) + 1
        ests = [("comb", bl.pipeline_comb(fs.meta))]
        if kind == "orbitrap":  # C2 reads resolved isotopes; time-of-flight spectra keep the comb
            inst = InstrumentModel(fs.meta["kind"], float(fs.meta["resolution"]))
            ests += [("comb, no isotope charges", bl.pipeline_comb(fs.meta, isotope_charges=False)),
                     ("C2", IsoCircuit(inst, **VARIANTS["c2"])),
                     ("C2b", IsoCircuit(inst, **VARIANTS["c2b"]))]
        for name, est in ests:
            board.add(kind, name, bl.charges_on_crop(est, cfg, fs, grid),
                      fs.topk_z[:, 0].astype(int), fs.features[0])
    bands = [b[0] for b in BIN_BANDS]
    for kind, n in kinds.items():
        for how in (*WEIGHTINGS, POOLED):
            print(f"\n{bl.KIND_NAMES[kind]} ({n} spectra), weighted by {how}: " + " ".join(bands))
            for name in names:
                cells = [board.accuracy(kind, name, how, b) for b in bands]
                if any(c is not None for c in cells):
                    print(f"  {name:>26}: " + " ".join("  -  " if c is None else f"{c:.2f}"
                                                       for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe", choices=("real", "bins"))
    ap.add_argument("raw", nargs="*", type=Path)
    ap.add_argument("--variant", default="c2b", choices=("comb", "c2", "c2b"))
    ap.add_argument("--frames", type=int, default=96)
    args = ap.parse_args()
    (cmd_real if args.probe == "real" else cmd_bins)(args)


if __name__ == "__main__":
    main()
