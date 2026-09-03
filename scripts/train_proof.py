"""Train the bundled CPU proof model and export it to ONNX."""

from __future__ import annotations

import argparse

from lcmsdeconv.nn.export import export_onnx, verify_parity
from lcmsdeconv.nn.train import TrainConfig, train
from lcmsdeconv.synth.config import SynthConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train-len", type=int, default=1500)
    ap.add_argument("--val-len", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=2400)
    ap.add_argument("--out", default="src/lcmsdeconv/models/charge_unet_proof.pt")
    ap.add_argument("--onnx", default="src/lcmsdeconv/models/charge_unet_proof.onnx")
    args = ap.parse_args()

    sc = SynthConfig(crop_size=32768)
    tc = TrainConfig(model_size="proof", epochs=args.epochs, train_len=args.train_len,
                     val_len=args.val_len, batch_size=args.batch_size, num_workers=args.workers,
                     heat_weight=0.1, max_seconds=args.max_seconds, out=args.out, log_every=25)
    model, history = train(tc, sc)
    onnx_path = export_onnx(args.out, args.onnx)
    print("parity:", verify_parity(args.out, onnx_path))
    print("final history:", history[-1] if history else None)


if __name__ == "__main__":
    main()
