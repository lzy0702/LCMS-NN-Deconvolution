"""Per-bin charge accuracy of charge models and of the comb estimator on the same spectra.

Every model is scored on exactly the same freshly generated spectra (the training mixture of
scripts/train_charge_net.py), against:

    comb (pipeline)   the CombPredictor the pipeline uses: charge ladder comb, plus charges read
                      from resolved isotope spacing for Orbitrap spectra, as in isotope mode
    comb (feature)    the comb's answer encoded in feature channel 3, the bar training prints

The score is the intensity-weighted share of ion bins whose charge the model reads correctly,
taking the most probable charge among the ion classes (as the decoder does), on bins of charge
two or more and, separately, on singly charged ones; time-of-flight and Orbitrap spectra are
reported apart.

    python scripts/eval_charge_bins.py --frames 64 runs/ladder_full_ignore.onnx runs/ladder_full_stage2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

BANDS = (("z=2-5", 2, 5), ("z=6-15", 6, 15), ("z=16-40", 16, 40), ("z>40", 41, 1000))


def frame_views(cfg, seed: int):
    """One spectrum: (features, true charge per bin, crop grid, window, noise, instrument kind,
    resolution)."""
    from lcmsdeconv.nn.grid import LogMzGrid, spectrum_to_grid
    from lcmsdeconv.synth.frames import generate_frame

    fs = generate_frame(cfg, np.random.default_rng(seed))
    axis, total = fs.meta["raw"]
    grid, observed, noise = spectrum_to_grid(axis, total, cfg.polarity, cfg.grid_step,
                                             cfg.grid_mz_max, True)
    L = fs.features.shape[1]
    b0 = int(fs.b0)
    window = np.zeros(L)
    seg = observed[b0:b0 + L]
    window[:seg.size] = seg
    lo = float(grid.bin_to_mz(b0))
    crop = LogMzGrid(lo, float(grid.bin_to_mz(b0 + L - 1)) * (1 + 1e-12), grid.step, cfg.polarity)
    return fs, crop, window, noise


def score(pred_z: np.ndarray, true_z: np.ndarray, w: np.ndarray, out: dict, key: str) -> None:
    """Accumulate weighted correct / total per band for one model on one spectrum."""
    d = out.setdefault(key, {})
    for name, lo, hi in (("z>=2", 2, 1000), ("z=1", 1, 1)) + BANDS:
        m = (true_z >= lo) & (true_z <= hi)
        c, t = d.get(name, (0.0, 0.0))
        d[name] = (c + float((w * m * (pred_z == true_z)).sum()), t + float((w * m).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="*", help=".onnx or .pt charge models")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20_000_000)
    ap.add_argument("--singly-charged", default="keep", choices=["keep", "drop", "ignore"],
                    help="how the evaluation spectra treat singly charged ions")
    ap.add_argument("--background-ions", type=int, default=None)
    args = ap.parse_args()

    from train_charge_net import mixture

    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.deconv.classical import CombPredictor
    from lcmsdeconv.nn.infer import ChargePredictor

    mix = mixture(background_ions=args.background_ions)
    for _w, c in mix:
        c.singly_charged = args.singly_charged
    weights = np.array([w for w, _c in mix])
    weights = weights / weights.sum()
    predictors = {Path(m).name: ChargePredictor(m) for m in args.models}
    results: dict[str, dict] = {"tof": {}, "orbitrap": {}}
    for i in range(args.frames):
        seed = args.seed + i
        cfg = mix[int(np.random.default_rng(seed).choice(len(mix), p=weights))][1]
        fs, crop, window, noise = frame_views(cfg, seed)
        kind = "orbitrap" if fs.meta["kind"] in ("orbitrap", "fticr") else "tof"
        true_z = fs.topk_z[:, 0].astype(int)
        w = fs.features[0].astype(float)
        w = np.where(w > 0.5, w, 0.0)  # the bins charge_metrics scores
        res = results[kind]
        feat = np.round(fs.features[3] * 64).astype(int)  # the proxy: 0 where it has no answer
        score(np.where(feat > 0, feat, -1), true_z, w, res, "comb (feature)")
        inst = InstrumentModel(fs.meta["kind"], float(fs.meta["resolution"])) if kind == "orbitrap" else None
        comb = CombPredictor(z_max=60, instrument=inst).predict_grid(crop, window, noise)
        score(np.where(comb.top1_z > 0, comb.top1_z, -1), true_z, w, res, "comb (pipeline)")
        for name, p in predictors.items():
            pr = p.predict_grid(crop, window, noise)
            score(np.where(pr.top1_z > 0, pr.top1_z, -1), true_z, w, res, name)
        print(f"frame {i + 1}/{args.frames} ({kind})", flush=True, file=sys.stderr)

    for kind, res in results.items():
        if not res:
            continue
        print(f"\n{kind} spectra (accuracy on ion bins, intensity-weighted)")
        cols = ("z>=2", "z=1") + tuple(b[0] for b in BANDS)
        print(f"| model | {' | '.join(cols)} |")
        print("| --- |" + " --- |" * len(cols))
        for key, d in res.items():
            cells = [f"{c / t:.2f}" if t > 0 else "-" for c, t in (d.get(k, (0, 0)) for k in cols)]
            print(f"| {key} | {' | '.join(cells)} |")


if __name__ == "__main__":
    main()
