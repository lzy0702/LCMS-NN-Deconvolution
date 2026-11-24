"""Record what the package did before the whole-spectrum recipe options (commit 1917c37), for
tests/test_recipe_v3.py: digests of default training frames and of the comb estimator's output,
the resume fingerprints of the training mixture, and a proof LadderNet (its state dict and its
outputs at sampled places). Run it on a checkout of that commit, in its own process:

    git worktree add /tmp/old 1917c37
    PYTHONPATH=/tmp/old/src python tests/data/make_ref_1917c37.py /tmp/old tests/data

It writes ref_1917c37.json and laddernet_proof_1917c37.pt into the output directory.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, for _digests
from _digests import arrays_digest, canary, frame_digest, ladder_input  # noqa: E402

#: the frames recorded: (mixture component, singly charged ions, seed); "keep" frames carry up to
#: 200 background ions, as the second training stage draws them
FRAMES = [(k, "ignore", 11 * k + 1) for k in range(4)] + [(k, "keep", 2000 + k) for k in range(4)]
DATASET_ITEMS = 2
#: the comb runs recorded: (frame, instrument kind or None for the frame's own, resolving power
#: or None for the frame's own, highest charge)
COMB = [("2/keep/2002", None, None, 60), ("2/ignore/23", None, None, 40),
        ("1/keep/2001", None, None, 60), ("1/keep/2001", "orbitrap", 140000.0, 40)]


def frame_configs(mixture, k: int, stage: str):
    mix = mixture() if stage == "ignore" else mixture(background_ions=200)
    cfg = mix[k][1]
    cfg.singly_charged = stage
    return cfg


def main(old_root: Path, out: Path) -> None:
    import numba
    import scipy
    import torch

    import lcmsdeconv

    sys.path.insert(0, str(old_root / "scripts"))
    from train_charge_net import mixture

    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.deconv.classical import CombPredictor
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _save, synth_fingerprint
    from lcmsdeconv.synth.dataset import FrameDataset
    from lcmsdeconv.synth.frames import generate_frame

    assert Path(lcmsdeconv.__file__).resolve().is_relative_to(old_root.resolve()), lcmsdeconv.__file__
    ref: dict = {"env": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "torch": torch.__version__.split("+")[0], "numba": numba.__version__,
                         "machine": platform.machine(), "canary": canary()}}

    frames, made = [], {}
    for k, stage, seed in FRAMES:
        cfg = frame_configs(mixture, k, stage)
        fs = generate_frame(cfg, np.random.default_rng(seed))
        made[f"{k}/{stage}/{seed}"] = (cfg, fs)
        frames.append({"component": k, "stage": stage, "seed": seed, "digest": frame_digest(fs)})
    ds = FrameDataset(mixture(), DATASET_ITEMS, base_seed=0)
    for i in range(DATASET_ITEMS):
        frames.append({"dataset_item": i, "digest": frame_digest(ds.frame(i)[1])})
    ref["frames"] = frames

    comb = []
    for name, kind, res, z_max in COMB:
        cfg, fs = made[name]
        grid, y, sigma = bl.frame_grid(cfg, fs)
        kind = kind or fs.meta["kind"]
        res = float(res or fs.meta["resolution"])
        pred = CombPredictor(z_max=z_max, instrument=InstrumentModel(kind, res))
        p = pred.predict_grid(grid, y, sigma)
        ns = max(sigma, 1e-9)
        s = np.clip(np.log10(1.0 + np.clip(y, 0, None) / ns), 0.0, 5.0)
        zi, conf = pred.isotope_charges(grid, s, y > 3.0 * ns)
        comb.append({"frame": name, "kind": kind, "resolution": res, "z_max": z_max,
                     "predict_grid": arrays_digest(p.top1_z, p.top1_p, p.top2_z, p.top2_p, p.apex),
                     "isotope_charges": arrays_digest(zi, conf), "n_isotope": int((zi > 0).sum())})
    ref["comb"] = comb

    keep = mixture(background_ions=200)
    for _w, c in keep:
        c.singly_charged = "keep"
    ref["fingerprints"] = {"mixture": synth_fingerprint(mixture()),
                           "mixture_keep_200": synth_fingerprint(keep)}

    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = build_model("proof", z_max=100, in_ch=4, arch="ladder")
    init = arrays_digest(*[t.detach().numpy() for t in model.state_dict().values()])
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # a learned part that is not zero
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    model.eval()
    state = model.state_dict()
    torch.save(state, out / "laddernet_proof_1917c37.pt")
    x = torch.from_numpy(ladder_input())
    with torch.no_grad():
        o = model(x)
    logits = o["charge_logits"][0].numpy()
    apex = o["apex_logit"][0, 0].numpy()
    level = x[0, 0]
    kth = torch.topk(level, model.active_top).values[-1]
    active = np.flatnonzero(((level > model.active_level) & (level >= kth)).numpy())
    rng = np.random.default_rng(3)
    bins = np.concatenate([rng.choice(active, 120, replace=False),
                           rng.integers(0, logits.shape[1], 40)])
    classes = rng.integers(0, logits.shape[0], bins.size)
    classes[:60] = rng.integers(1, model.z_ladder + 1, 60)
    lad = {"keys": {k: list(v.shape) for k, v in state.items()}, "init_digest": init,
           "samples": [[int(c), int(b), float(logits[c, b])] for c, b in zip(classes, bins)],
           "apex_samples": [[int(b), float(apex[b])] for b in bins[:40]],
           "logit_sum": float(logits.astype(np.float64).sum()),
           "top1_active": arrays_digest((logits[1:, active].argmax(0) + 1).astype(np.int64))}
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ladder.pt"
        _save(ckpt, state, TrainConfig(arch="ladder", model_size="proof"), 4, None, None,
              "charge_acc_given_ion")
        cfg, fs = made["0/keep/2000"]
        grid, y, sigma = bl.frame_grid(cfg, fs)
        p = ChargePredictor(ckpt).predict_grid(grid, y, sigma)
    lad["predict_grid"] = {"frame": "0/keep/2000", "grid_size": int(grid.size),
                           "digest": arrays_digest(p.top1_z, p.top2_z),
                           "top1_p_sum": float(p.top1_p.astype(np.float64).sum()),
                           "apex_sum": float(p.apex.astype(np.float64).sum())}
    ref["laddernet"] = lad
    (out / "ref_1917c37.json").write_text(json.dumps(ref, indent=1) + "\n")
    print("wrote", out / "ref_1917c37.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
