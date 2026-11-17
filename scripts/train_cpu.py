"""Train the charge network on a CPU with features rendered along the inference path.

The training mixture covers positive-mode reversed-phase proteins and polymers, negative-mode
ion-pairing oligonucleotides and a general positive-mode mix, each rendered as summed,
vendor-thresholded profile scans resampled onto the grid (``SynthConfig.render_path =
"profile"``), so features and noise estimates match what inference computes.

    python scripts/train_cpu.py --minutes 120 --out models/charge_unet_cpu.pt
    lcmsdeconv export-onnx models/charge_unet_cpu.pt -o models/charge_unet_cpu.onnx
"""

from __future__ import annotations

import argparse

from lcmsdeconv.nn.train import TrainConfig, train
from lcmsdeconv.synth.compounds import DEFAULT_CLASSES, ClassConfig
from lcmsdeconv.synth.config import SynthConfig


def mixture(crop: int) -> list[tuple[float, SynthConfig]]:
    proteins = SynthConfig(classes=[ClassConfig("peptide", (2000.0, 90000.0), 3.0),
                                    ClassConfig("peg", (1000.0, 15000.0), 1.0)],
                           mode="rplc", polarity=1, crop_size=crop)
    oligos = SynthConfig(classes=[ClassConfig("dna", (2000.0, 25000.0)),
                                  ClassConfig("rna", (2000.0, 25000.0)),
                                  ClassConfig("ps_dna", (2000.0, 25000.0)),
                                  ClassConfig("ps_rna", (2000.0, 25000.0))],
                         mode="iprp", polarity=-1, crop_size=crop)
    general = SynthConfig(classes=list(DEFAULT_CLASSES), mode="rplc", polarity=1, crop_size=crop)
    return [(0.4, proteins), (0.4, oligos), (0.2, general)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--model-size", default="proof", choices=["small", "proof", "full"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--train-len", type=int, default=1200)
    ap.add_argument("--val-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--crop", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/charge_unet_cpu.pt")
    args = ap.parse_args()
    cfg = TrainConfig(model_size=args.model_size, epochs=args.epochs, train_len=args.train_len,
                      val_len=args.val_len, batch_size=args.batch_size, num_workers=args.workers,
                      lr=args.lr, base_seed=args.seed, out=args.out,
                      max_seconds=args.minutes * 60.0, threads=args.threads)
    train(cfg, synth_config=mixture(args.crop))


if __name__ == "__main__":
    main()
