"""Per-bin charge accuracy of charge models and of the comb estimator, each run as the pipeline
runs it, on the same spectra.

Every estimator reads the same freshly generated spectra of the training mixture
(scripts/train_charge_net.py) the way ``process_run`` would - over the whole spectrum, a network
in its 32 768-bin tiles, the comb all at once - and is scored on the frame's labelled crop:

    comb                        the CombPredictor the pipeline uses, with charges read from
                                resolved isotope spacing on Orbitrap spectra (isotope mode)
    comb, no isotope charges    the same without them (Orbitrap spectra)
    <model file>                a trained network (.onnx or .pt) through ChargePredictor

The score is the share of the ion bins whose charge an estimator reads correctly (a network's
most probable charge among the ion classes, as the decoder takes it; no answer counts as
wrong). Every spectrum counts the same; within a spectrum bins count by log-SNR (feature channel
0, bins above 0.5: what training selects checkpoints by) or by intensity (bins above three
sigma: what the decoder's mass votes weigh). A third table pools the intensity-weighted bins of
all spectra, so that the few strongest spectra decide it. Charges two and up, singly charged
ions and four charge bands are reported apart, and so are time-of-flight and Orbitrap spectra.

    python scripts/eval_charge_bins.py --frames 96 runs/ladder_full_ignore.onnx runs/ladder_full_stage2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


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

    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.checks import BIN_BANDS, POOLED, WEIGHTINGS, ChargeScoreboard
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.synth.frames import generate_frame

    mix = mixture(background_ions=args.background_ions)
    for _w, c in mix:
        c.singly_charged = args.singly_charged
    weights = np.array([w for w, _c in mix])
    weights = weights / weights.sum()
    predictors = {Path(m).name: ChargePredictor(m) for m in args.models}
    board = ChargeScoreboard()
    order = ["comb", "comb, no isotope charges", *predictors]
    n_kind: dict[str, int] = {}
    for i in range(args.frames):
        seed = args.seed + i
        cfg = mix[int(np.random.default_rng(seed).choice(len(mix), p=weights))][1]
        fs = generate_frame(cfg, np.random.default_rng(seed))
        grid = bl.frame_grid(cfg, fs)
        if grid is None:
            raise SystemExit("frames rendered without a whole spectrum cannot be scored this way")
        kind = bl.spectrum_kind(fs.meta)
        n_kind[kind] = n_kind.get(kind, 0) + 1
        true_z = fs.topk_z[:, 0].astype(int)
        f0 = fs.features[0]
        estimators = [("comb", bl.pipeline_comb(fs.meta))]
        if kind == "orbitrap":
            estimators.append(("comb, no isotope charges",
                               bl.pipeline_comb(fs.meta, isotope_charges=False)))
        estimators += list(predictors.items())
        for name, est in estimators:
            board.add(kind, name, bl.charges_on_crop(est, cfg, fs, grid), true_z, f0)
        print(f"frame {i + 1}/{args.frames} ({bl.KIND_NAMES[kind]})", flush=True, file=sys.stderr)

    cols = [b[0] for b in BIN_BANDS]
    for kind, name in bl.KIND_NAMES.items():
        if not n_kind.get(kind):
            continue
        for how in (*WEIGHTINGS, POOLED):
            print(f"\n{name} spectra ({n_kind[kind]}), share of ion bins read right, weighted by {how}")
            print(f"| charge assignment | {' | '.join(cols)} |")
            print("| --- |" + " --- |" * len(cols))
            for est in order:
                cells = [board.accuracy(kind, est, how, band) for band in cols]
                if all(c is None for c in cells):
                    continue
                print(f"| {est} | " + " | ".join("-" if c is None else f"{c:.2f}" for c in cells) + " |")


if __name__ == "__main__":
    main()
