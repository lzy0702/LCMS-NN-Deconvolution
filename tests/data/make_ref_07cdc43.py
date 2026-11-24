"""Record a network of the first isotope-channel layout as commit 07cdc43 wrote and read it, for
tests/test_ladder_views.py: a small window-view LadderNet reading two isotope channels (in_ch 6),
saved by 07cdc43's training code, its outputs on a fixed input, its ONNX sidecar, and what
07cdc43's ChargePredictor reads with it (checkpoint and ONNX export) on the grid of an Orbitrap
spectrum of the whole-spectrum recipe. Run it on a checkout of that commit, in its own process:

    git archive 07cdc43 | tar -x -C /tmp/old
    PYTHONPATH=/tmp/old/src python tests/data/make_ref_07cdc43.py /tmp/old tests/data

It writes ref_07cdc43.json and laddernet_iso2_07cdc43.pt into the output directory.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, for _digests
from _digests import arrays_digest, canary  # noqa: E402

#: the network: LadderNet "small" with two isotope channels, read in 8 192-bin windows
WINDOW = 8192
#: the spectrum read: mixture_v3's Orbitrap component at the real runs' resolving power
FRAME_SEED = 31


def iso2_input(length: int = WINDOW) -> np.ndarray:
    """A fixed input ``[1, 6, length]``: random peaks, isotope charges and strengths."""
    rng = np.random.default_rng(7)
    x = np.zeros((1, 6, length), dtype=np.float32)
    idx = rng.integers(0, length, 1500)
    x[0, 0, idx] = rng.uniform(0.0, 5.0, idx.size)
    x[0, 1] = rng.uniform(0.0, 1.0, length)
    z = rng.integers(0, 20, length).astype(np.float32)
    z[rng.random(length) < 0.5] = 0.0
    x[0, 4] = z
    x[0, 5] = np.where(z > 0, rng.uniform(0.0, 1.0, length), 0.0)
    return x


def main(old_root: Path, out: Path) -> None:
    import torch

    import lcmsdeconv

    sys.path.insert(0, str(old_root / "scripts"))
    from train_charge_net import mixture_v3

    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.export import export_onnx
    from lcmsdeconv.nn.infer import ChargePredictor
    from lcmsdeconv.nn.model import build_model
    from lcmsdeconv.nn.train import TrainConfig, _save
    from lcmsdeconv.synth.frames import generate_frame

    assert Path(lcmsdeconv.__file__).resolve().is_relative_to(old_root.resolve()), lcmsdeconv.__file__
    ref: dict = {"env": {"numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                         "machine": platform.machine(), "canary": canary()}}
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = build_model("small", z_max=100, in_ch=6, arch="ladder", window=WINDOW, iso=True)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # a learned part that is not zero
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    model.eval()
    ckpt = out / "laddernet_iso2_07cdc43.pt"
    _save(ckpt, model.state_dict(), TrainConfig(arch="ladder", model_size="small", window=WINDOW,
                                                iso_channels=True), 6, None, None,
          "charge_acc_given_ion")
    x = torch.from_numpy(iso2_input())
    with torch.no_grad():
        o = model(x)
    logits = o["charge_logits"][0].numpy()
    rng = np.random.default_rng(3)
    bins = rng.integers(0, WINDOW, 200)
    classes = rng.integers(1, model.z_ladder + 1, bins.size)
    ref["outputs"] = {"samples": [[int(c), int(b), float(logits[c, b])]
                                  for c, b in zip(classes, bins)],
                      "logit_sum": float(logits.astype(np.float64).sum())}

    cfg = mixture_v3()[3][1]
    fs = generate_frame(cfg, np.random.default_rng(FRAME_SEED))
    grid, y, sigma = bl.frame_grid(cfg, fs)
    inst = InstrumentModel(fs.meta["kind"], float(fs.meta["resolution"]))
    reads = {"frame_seed": FRAME_SEED, "grid_size": int(grid.size),
             "resolution": float(fs.meta["resolution"])}
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(ckpt, Path(tmp) / "iso2.onnx")
        reads["sidecar"] = json.loads(onnx_path.with_suffix(".json").read_text())
        for backend, path in (("torch", ckpt), ("onnx", onnx_path)):
            p = ChargePredictor(path, instrument=inst).predict_grid(grid, y, sigma)
            reads[backend] = {"digest": arrays_digest(p.top1_z, p.top2_z),
                              "top1_p_sum": float(p.top1_p.astype(np.float64).sum()),
                              "apex_sum": float(p.apex.astype(np.float64).sum()),
                              "charged": int((p.top1_z > 0).sum())}
    ref["predict_grid"] = reads
    (out / "ref_07cdc43.json").write_text(json.dumps(ref, indent=1) + "\n")
    print("wrote", out / "ref_07cdc43.json", "and", ckpt)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
