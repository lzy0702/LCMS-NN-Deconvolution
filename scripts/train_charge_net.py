"""Train a charge network (LadderNet by default) on synthetic frames, on a CPU or a GPU.

Frames are rendered along the inference path (summed, vendor-thresholded profile scans
resampled onto the logarithmic grid) from a mixture of positive-mode proteins and polymers,
negative-mode ion-pairing oligonucleotides and a general positive-mode mix. Every epoch prints
the charge accuracy on ion bins next to the comb estimator's on the same validation frames, the
bar a network has to clear before it is worth using; the best checkpoint is saved as soon as it
improves, so an interrupted run still leaves a model.

Singly charged ions (chemical background, small molecules) dominate the bins and are learned
first; ``--singly-charged`` controls them:

    drop     generate none: the network never sees a z = 1 ion
    ignore   generate them, but give their bins no weight in the loss
    keep     train on everything (a later fine-tuning stage)

Examples:

    # quick check that everything works (a few minutes)
    python scripts/train_charge_net.py --smoke

    # a stage-1 run on a GPU, singly charged ions ignored, eight data workers
    python scripts/train_charge_net.py --device cuda --singly-charged ignore \\
        --epochs 40 --train-len 4000 --batch-size 8 --workers 8 --out runs/ladder_stage1.pt

    # stage 2: continue from stage 1 with singly charged ions included
    python scripts/train_charge_net.py --device cuda --singly-charged keep \\
        --init runs/ladder_stage1.pt --epochs 10 --lr 3e-4 --out runs/ladder_stage2.pt

    # export for inference (ONNX + JSON sidecar next to it)
    lcmsdeconv export-onnx runs/ladder_stage1.pt -o runs/ladder_stage1.onnx

Pausing and resuming: press Ctrl+C once (or create an empty file named PAUSE in the output's
folder); the current step finishes, the whole training state is saved to ``<out>.resume.pt``
and the run ends. Run the same command again with ``--resume`` to continue exactly where it
stopped, inside an epoch if need be. The state is also saved every ``--save-every`` minutes
and after every epoch, so a crash or a power cut loses at most that much work.
"""

from __future__ import annotations

import argparse
import os

from lcmsdeconv.nn.train import TrainConfig, train
from lcmsdeconv.synth.compounds import DEFAULT_CLASSES, ClassConfig
from lcmsdeconv.synth.config import SynthConfig


def mixture(crop: int = 32768) -> list[tuple[float, SynthConfig]]:
    proteins = SynthConfig(classes=[ClassConfig("peptide", (2000.0, 90000.0), 3.0),
                                    ClassConfig("peg", (1000.0, 15000.0), 1.0)],
                           mode="rplc", polarity=1, crop_size=crop)
    oligos = SynthConfig(classes=[ClassConfig("dna", (2000.0, 25000.0)),
                                  ClassConfig("rna", (2000.0, 25000.0)),
                                  ClassConfig("ps_dna", (2000.0, 25000.0)),
                                  ClassConfig("ps_rna", (2000.0, 25000.0))],
                         mode="iprp", polarity=-1, crop_size=crop)
    orbitrap_oligos = SynthConfig(classes=[ClassConfig("ps_dna", (1500.0, 12000.0)),
                                           ClassConfig("ps_rna", (1500.0, 12000.0))],
                                  mode="iprp", polarity=-1, crop_size=crop,
                                  instrument_kind="orbitrap",
                                  resolution_range=(30000.0, 63000.0))
    general = SynthConfig(classes=list(DEFAULT_CLASSES), mode="rplc", polarity=1, crop_size=crop)
    return [(0.35, proteins), (0.3, oligos), (0.15, orbitrap_oligos), (0.2, general)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", default="ladder", choices=["ladder", "unet"])
    ap.add_argument("--model-size", default="proof", choices=["small", "proof", "full"])
    ap.add_argument("--singly-charged", default="ignore", choices=["drop", "ignore", "keep"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-len", type=int, default=2000, help="frames per epoch")
    ap.add_argument("--val-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="processes generating frames (the bottleneck on a GPU)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for torch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat-frames", action="store_true",
                    help="reuse the same frames every epoch (and cache them in memory, one "
                         "process) instead of drawing new ones")
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--resume", action="store_true",
                    help="continue a paused or interrupted run from <out>.resume.pt")
    ap.add_argument("--save-every", type=float, default=10.0,
                    help="minutes between saves of the resumable training state")
    ap.add_argument("--out", default="runs/ladder.pt")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the installation")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.train_len, args.val_len, args.batch_size = 1, 16, 8, 2
        args.out = "runs/smoke.pt"
    cfg = TrainConfig(model_size=args.model_size, epochs=args.epochs, train_len=args.train_len,
                      val_len=args.val_len, batch_size=args.batch_size,
                      num_workers=args.workers, lr=args.lr, base_seed=args.seed, out=args.out,
                      max_seconds=args.minutes * 60.0 if args.minutes else None,
                      threads=args.threads, device=args.device, arch=args.arch,
                      singly_charged=args.singly_charged,
                      fresh_each_epoch=not args.repeat_frames, cache_frames=args.repeat_frames,
                      init=args.init, resume=args.resume, save_every_minutes=args.save_every)
    train(cfg, synth_config=mixture())


if __name__ == "__main__":
    main()
