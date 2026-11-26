"""The real-envelope measurement (real_env.py) on synthetic Orbitrap oligonucleotide frames.

Frames are drawn from the Orbitrap oligonucleotide component of scripts/train_charge_net.py's
mixture() (or of recipe v3's mixture_v3(), --frames-config) as training draws them
(synth.dataset.FrameDataset), singly charged ions kept, and each estimator reads the frame's whole
spectrum as the pipeline would (nn.baseline). Every oligonucleotide component's envelopes are
located and labelled by envlib exactly as on the real runs, so the two sets of numbers compare like
with like; the labels are checked against the renderer's own per-bin truth on the labelled crop.

--charge-model changes the oligonucleotide charge-state model in this process only:
    default     the package's (apex 0.00664 M^0.8 x U(0.85, 1.2), width 12-25 % of the apex)
    real        apex A M^B x U(0.9, 1.1) (A, B fitted to the real runs), width 0.35-0.6 charges
    low         the real apex, the package's width rule
    narrow      the package's apex, width 0.35-0.6 charges
(with v3 frames only default and real, set through the package's own oligo_real_like_prob).

    python bench/tools/synth_env.py --frames 32 --frames-config default --charge-model real
        [--model LABEL=NET.onnx ...] [--threads N] [--providers auto|dml|cuda|cpu] --out synth.json
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/
import _paths  # noqa: E402, F401  (<repo>/src and <repo>/scripts on sys.path)

from tools import common  # noqa: E402

common.preset_threads()  # before numpy: --threads caps its BLAS / OpenMP / numba pools

import numpy as np  # noqa: E402

OLIGO = ("dna", "rna", "ps_dna", "ps_rna")


def patch_charge_model(model: str, a_real: float, b_real: float) -> None:
    """Replace the oligonucleotide charge-state model of lcmsdeconv.synth.charge (this process)."""
    if model == "default":
        return
    from lcmsdeconv.synth import charge as ch

    orig_sample, orig_related = ch.sample_charge_shape, ch.related_shape
    real_apex = model in ("real", "low")
    narrow = model in ("real", "narrow")

    def sample_charge_shape(mass, cls, mode, rng, **kw):
        # (**kw: the package's real_like switch, which this patch's models replace)
        if cls not in OLIGO or mass < 900:
            return orig_sample(mass, cls, mode, rng, **kw)
        if real_apex:
            zc = max(1.0, a_real * mass ** b_real * rng.uniform(0.9, 1.1))
        else:
            zc = ch.apex_charge(mass, cls, mode, rng)
        width = rng.uniform(0.35, 0.6) if narrow else max(zc * rng.uniform(0.12, 0.25), 0.6)
        skew = rng.uniform(-0.4, 0.4)
        second = None
        if not narrow and rng.random() < 0.15:
            second = (zc * rng.uniform(0.5, 0.85), width * 0.8, 0.6)
        return ch.ChargeShape(zc, width, skew, second)

    def related_shape(parent, parent_mass, mass, cls, rng, jitter=0.03, **kw):
        if cls not in OLIGO or parent.single is not None:
            return orig_related(parent, parent_mass, mass, cls, rng, jitter, **kw)
        b = b_real if real_apex else ch.get_class(cls).charge_b
        scale = (max(mass, 1.0) / max(parent_mass, 1.0)) ** b * rng.uniform(1 - jitter, 1 + jitter)
        floor = 0.35 if narrow else 0.6
        second = None
        if parent.second is not None:
            zc2, w2, a2 = parent.second
            second = (zc2 * scale, w2 * scale, a2)
        return replace(parent, centre=max(1.0, parent.centre * scale),
                       width=max(floor, parent.width * scale), second=second)

    ch.sample_charge_shape = sample_charge_shape
    ch.related_shape = related_shape


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--seed", type=int, default=21_000_000)
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--charge-model", default="default",
                    choices=("default", "real", "low", "narrow"))
    ap.add_argument("--a-real", type=float, default=0.0769)
    ap.add_argument("--b-real", type=float, default=0.445)
    ap.add_argument("--background-ions", type=int, default=None)
    ap.add_argument("--frames-config", default="default", choices=("default", "v3-whole", "v3-real"),
                    help="default: mixture()'s Orbitrap oligonucleotides (as ROOTCAUSE measured); "
                         "v3-whole / v3-real: recipe v3's two Orbitrap halves (thresholded, "
                         "35 000-280 000 / 130 000-185 000), with --charge-model default or real "
                         "set through the package's own oligo_real_like_prob (0 or 1)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--providers", default="auto", choices=("auto", "dml", "cuda", "cpu"))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    models = common.parse_models(args.model)
    v3 = args.frames_config != "default"
    if v3 and args.charge_model not in ("default", "real"):
        ap.error("v3 frames take --charge-model default or real")
    common.cap_threads(args.threads)
    if not v3:
        patch_charge_model(args.charge_model, args.a_real, args.b_real)

    from train_charge_net import mixture, mixture_v3

    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.checks import ChargeScoreboard
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.synth.dataset import FrameDataset
    from tools import envlib

    if v3:
        mix = mixture_v3(background_ions=args.background_ions)
        cfg = mix[2 if args.frames_config == "v3-whole" else 3][1]
        cfg.oligo_real_like_prob = 1.0 if args.charge_model == "real" else 0.0
    else:
        mix = mixture(background_ions=args.background_ions)
        cfg = mix[2][1]
    if cfg.instrument_kind != "orbitrap":
        raise SystemExit(f"expected the mixture's Orbitrap component, got {cfg.instrument_kind}")
    cfg.singly_charged = "keep"
    frames = FrameDataset([(1.0, cfg)], args.frames, base_seed=args.seed)
    prov = common.providers(args.providers)
    nets = {label: ChargePredictor(path, providers=prov) for label, path in models.items()}
    board = ChargeScoreboard()
    out = {"envelopes": [], "frames": [], "estimators": ["comb", "comb, ladder only", *nets],
           "charge_model": args.charge_model, "frames_config": args.frames_config,
           "n_frames": args.frames, "seed": args.seed,
           "providers": {k: n.session.get_providers() for k, n in nets.items()}}
    lab_right = lab_total = lab_bg = lab_int_right = lab_int = 0.0
    for i in range(args.frames):
        t0 = time.time()
        fcfg, fs = frames.frame(i)
        grid, y, sigma = bl.frame_grid(fcfg, fs)
        for net in nets.values():  # isotope-channel networks read the frame's resolving power
            net.instrument = bl.frame_instrument(fs.meta)
        ests = {"comb": bl.pipeline_comb(fs.meta, z_max=40),
                "comb, ladder only": bl.pipeline_comb(fs.meta, z_max=40, isotope_charges=False),
                **nets}
        preds = {k: e.predict_grid(grid, y, sigma).top1_z.copy() for k, e in ests.items()}
        main_int = max((t.intensity for t in fs.truths if t.compound_class in OLIGO), default=1.0)
        species = []
        for t in fs.truths:
            if t.compound_class not in OLIGO:
                continue
            species.append((f"{t.kind}{t.id}", (float(t.average_mass),),
                            {"frame": i, "cls": t.compound_class, "kind": t.kind,
                             "rel": float(t.intensity / main_int),
                             "true_charges": {int(k): float(v) for k, v in t.charges.items()},
                             "resolution": float(fs.meta["resolution"])}))
        envs, labels = envlib.measure(grid, y, sigma, species)
        envlib.score(envs, y, sigma, preds)
        # the labels against the renderer's truth on the crop
        L = fs.features.shape[1]
        true_z = fs.topk_z[:, 0].astype(int)
        f0 = fs.features[0]
        crop = np.arange(fs.b0, min(fs.b0 + L, y.size))
        mine = labels[crop]
        sel = (mine > 0) & (f0[:crop.size] > 0.5)
        w = f0[:crop.size][sel]
        tz = true_z[:crop.size][sel]
        lab_right += float((w * (mine[sel] == tz)).sum())
        lab_total += float(w.sum())
        lab_bg += float((w * (tz == 0)).sum())
        snr = 10.0 ** f0[:crop.size][sel] - 1.0
        wi = np.where(snr > 3.0, snr, 0.0)
        lab_int_right += float((wi * (mine[sel] == tz)).sum())
        lab_int += float(wi.sum())
        for name, top in preds.items():
            seg = np.zeros(L, np.int64)
            s = top[fs.b0:fs.b0 + L]
            seg[:s.size] = s
            board.add("orbitrap", name, seg, true_z, f0)
        for e in envs:
            e.pop("_bins")
            out["envelopes"].append(e)
        out["frames"].append({"frame": i, "resolution": float(fs.meta["resolution"]),
                              "n_sum": int(fs.meta["n_sum"]), "sigma": float(sigma),
                              "grid_bins": int(y.size), "zero_frac": float((y <= 0).mean()),
                              "n_species": len(species), "n_envelopes": len(envs)})
        print(f"frame {i + 1}/{args.frames}: R {fs.meta['resolution']:.0f}, {len(species)} "
              f"oligonucleotide components, {len(envs)} envelopes, {time.time() - t0:.0f} s",
              flush=True)
    out["label_check"] = {"log_snr_weight_right": lab_right, "log_snr_weight": lab_total,
                          "share": lab_right / max(lab_total, 1e-300),
                          "share_true_background": lab_bg / max(lab_total, 1e-300),
                          "share_intensity": lab_int_right / max(lab_int, 1e-300)}
    out["crop_scoreboard"] = board.summary()
    common.write_json(args.out, out)
    print("label check on crops:", out["label_check"], flush=True)


if __name__ == "__main__":
    main()
