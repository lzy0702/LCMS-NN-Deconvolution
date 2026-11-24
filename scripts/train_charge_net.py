"""Train a charge network (LadderNet by default) on synthetic frames, on a CPU or a GPU.

Frames are rendered along the inference path (summed, vendor-thresholded profile scans
resampled onto the logarithmic grid) from a mixture of positive-mode proteins and polymers,
negative-mode ion-pairing oligonucleotides on time-of-flight and on Orbitrap instruments (35 000
to 280 000 resolving power at m/z 200) and a general positive-mode mix. Every epoch prints
the charge accuracy on ion bins next to that of the comb estimator as the pipeline runs it (over
the whole spectrum, isotope-spacing charges on Orbitrap spectra) on the same validation frames,
by kind of spectrum and weighted by log-SNR and by intensity: the bar a network has to clear
before it is worth using. The best checkpoint is saved as soon as it improves, so an
interrupted run still leaves a model.

Singly charged ions (chemical background, small molecules) dominate the bins and are learned
first; ``--singly-charged`` controls them:

    drop     generate none: the network never sees a z = 1 ion
    ignore   generate them, but give their bins no weight in the loss
    keep     train on everything (a later fine-tuning stage)

Examples:

    # quick check that everything works (a few minutes)
    python scripts/train_charge_net.py --smoke

    # a stage-1 run on the GPU, singly charged ions ignored, within 16 GB of RAM
    python scripts/train_charge_net.py --singly-charged ignore --epochs 40 --train-len 6000 \\
        --batch-size 8 --max-ram-gb 16 --max-gpu-memory-gb 14 --out runs/ladder_stage1.pt

    # stage 2: continue from stage 1 with singly charged ions included
    python scripts/train_charge_net.py --device cuda --singly-charged keep \\
        --init runs/ladder_stage1.pt --epochs 10 --lr 3e-4 --out runs/ladder_stage2.pt

    # the whole-spectrum recipe (experimental): one 131 072-bin window, z = 1 labelled in one
    # stage, instrument scan ranges, strong background in a quarter of the spectra, real-like
    # ion-pairing oligonucleotides; --iso-channels adds the isotope-charge input channels, and
    # --view comb trains the ladder that reads a grid of any length whole (the same frames)
    python scripts/train_charge_net.py --recipe v3 --iso-channels --out runs/ladder_v3_iso.pt
    python scripts/train_charge_net.py --recipe v3 --iso-channels --view comb \
        --out runs/ladder_v3_comb.pt
    # v4: v3 with lone multiply charged oligonucleotides in the real-like Orbitrap spectra
    python scripts/train_charge_net.py --recipe v4 --iso-channels --out runs/ladder_v4_iso.pt

    # export for inference (ONNX + JSON sidecar next to it)
    lcmsdeconv export-onnx runs/ladder_stage1.pt -o runs/ladder_stage1.onnx

Pausing and resuming: press Ctrl+C once (or create an empty file named PAUSE, or PAUSE.txt, in
the output's folder); the current step finishes, the whole training state is saved to
``<out>.resume.pt`` and the run ends. Run the same command again with ``--resume`` to continue exactly where it
stopped, inside an epoch if need be; it refuses to resume with settings that would change the
run. The state is also saved every ``--save-every`` minutes and after every epoch, so a crash or
a power cut loses at most that much work. A new run will not start over a paused one's state
unless ``--restart`` says to discard it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lcmsdeconv.nn.machine import auto_workers, single_threaded_workers
from lcmsdeconv.nn.train import TrainConfig, train
from lcmsdeconv.synth.compounds import DEFAULT_CLASSES, ClassConfig
from lcmsdeconv.synth.config import SynthConfig


def mixture(crop: int = 32768, background_ions: int | None = None) -> list[tuple[float, SynthConfig]]:
    proteins = SynthConfig(classes=[ClassConfig("peptide", (2000.0, 90000.0), 3.0),
                                    ClassConfig("peg", (1000.0, 15000.0), 1.0)],
                           mode="rplc", polarity=1, crop_size=crop)
    oligos = SynthConfig(classes=[ClassConfig("dna", (2000.0, 25000.0)),
                                  ClassConfig("rna", (2000.0, 25000.0)),
                                  ClassConfig("ps_dna", (2000.0, 25000.0)),
                                  ClassConfig("ps_rna", (2000.0, 25000.0))],
                         mode="iprp", polarity=-1, crop_size=crop)
    # resolving power at m/z 200 from 35 000 to 280 000, the common Orbitrap settings: the real
    # Q Exactive Plus runs of docs/benchmarks.md were recorded at 140 000 (155 000 measured)
    orbitrap_oligos = SynthConfig(classes=[ClassConfig("ps_dna", (1500.0, 12000.0)),
                                           ClassConfig("ps_rna", (1500.0, 12000.0))],
                                  mode="iprp", polarity=-1, crop_size=crop,
                                  instrument_kind="orbitrap",
                                  resolution_range=(35000.0, 280000.0))
    general = SynthConfig(classes=list(DEFAULT_CLASSES), mode="rplc", polarity=1, crop_size=crop)
    mix = [(0.3, proteins), (0.25, oligos), (0.25, orbitrap_oligos), (0.2, general)]
    if background_ions is not None:
        # real column-wash regions carry far more singly charged background than the default
        # 0-40 ions per spectrum; a network that must learn to reject it has to see it
        for _w, cfg in mix:
            cfg.background_ions_range = (0, int(background_ions))
    return mix


#: the whole-spectrum recipe's window: a region's whole grid (400-2 500 m/z is about 91 600
#: bins) and the ladder's far rungs (z = 1 <-> 4 is 69 315 bins) in one window
V3_WINDOW = 131072


def mixture_v3(window: int = V3_WINDOW, background_ions: int | None = None
               ) -> list[tuple[float, SynthConfig]]:
    """The whole-spectrum recipe (experimental): :func:`mixture`'s four kinds of spectra with the
    same weights, each read whole in one ``window``-bin crop, measured over an instrument's scan
    range (so that missing charge states lie inside the grid), with 0-200 singly charged
    background ions that are 10-100 times stronger in a quarter of the spectra; the Orbitrap
    oligonucleotides span dna, rna, ps_dna and ps_rna from 1 to 12 kDa, half the spectra with
    the charge states of real ion-pairing runs (one to three).

    The Orbitrap spectra are thresholded as Thermo profile spectra are (3-6 electronic noise
    sigmas: most bins zero), and half of them are recorded at the resolving power of the real
    runs (130 000-185 000 at m/z 200), the other half over the whole 35 000-280 000. Measured on
    real-like synthetic envelopes as on the real runs (zero bins near an envelope, peak width,
    isotope-reader accuracy at z >= 2): 0.93, 1.21 bins and 0.89 at that resolving power and
    0.90, 1.36 and 0.78 over the whole range, against 0.86, 1.13 and 0.90 on the real runs; the
    default 0-3 sigmas give 0.15, 1.36 and 0.59. Without the threshold the isotope-charge
    channels fire mostly on noise bins, which real spectra hardly have.

    The result has five components: the Orbitrap one is split into those two halves, each at
    half its weight (``mix[2]`` the whole range, ``mix[3]`` the real runs' resolving power)."""
    mix = mixture(crop=window, background_ions=200 if background_ions is None else background_ions)
    for _w, cfg in mix:
        cfg.scan_range = "instrument"
        cfg.background_strong_prob = 0.25
    weight, orbitrap = mix[2]
    assert orbitrap.instrument_kind == "orbitrap"
    orbitrap.classes = [ClassConfig(c, (1000.0, 12000.0)) for c in ("dna", "rna", "ps_dna", "ps_rna")]
    orbitrap.oligo_real_like_prob = 0.5
    orbitrap.threshold_sigmas_range = (3.0, 6.0)
    at_real = replace(orbitrap, classes=list(orbitrap.classes),
                      resolution_range=V3_REAL_RESOLUTION)
    mix[2] = (weight / 2, orbitrap)
    mix.insert(3, (weight / 2, at_real))
    return mix


#: resolving power (at m/z 200) of half the v3 Orbitrap spectra: the real Q Exactive Plus runs
#: were recorded at 140 000 and measure about 155 000
V3_REAL_RESOLUTION = (130000.0, 185000.0)


def mixture_v4(window: int = V3_WINDOW, background_ions: int | None = None
               ) -> list[tuple[float, SynthConfig]]:
    """Recipe v4 (experimental): :func:`mixture_v3` with lone multiply charged
    oligonucleotides. In the real-like Orbitrap spectra (half of them), a share
    :data:`V4_SINGLE_STATE` of the oligonucleotide main compounds show one charge state only
    (``SynthConfig.oligo_single_state_prob``), their impurities with them.

    In v3's spectra 99.8 % of the envelopes with no other charge state of their species in view
    are singly charged background ions, and where the isotope reader reads a charge of 2 or more
    on such a lone peak the label is z = 1 for 97.6 % of the signal. A network trained on them
    learns that a charge without ladder rungs is z = 1 and overrides a right isotope reading on
    every lone multiply charged envelope (docs/training.md, recipe v4). Only Orbitrap spectra get
    them: a lone envelope's charge is in its absolute m/z, which only the isotope-charge channels
    carry, and those are read on Orbitrap and FT-ICR spectra only."""
    mix = mixture_v3(window, background_ions)
    for _w, cfg in mix:
        if cfg.instrument_kind == "orbitrap":
            cfg.oligo_single_state_prob = V4_SINGLE_STATE
    return mix


#: recipe v4: share of the real-like Orbitrap spectra's oligonucleotide main compounds that show
#: a single charge state
V4_SINGLE_STATE = 0.4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipe", default="default", choices=["default", "v3", "v4"],
                    help="training spectra and settings: 'default' (the mixture above, "
                         "32 768-bin crops), 'v3', the whole-spectrum recipe (experimental: "
                         "one 131 072-bin window, singly charged ions kept, instrument scan "
                         "ranges, strong background, real-like ion-pairing oligonucleotides), "
                         "or 'v4', v3 with lone multiply charged oligonucleotides in the "
                         "real-like Orbitrap spectra")
    ap.add_argument("--window", type=int, default=None,
                    help="LadderNet window in grid bins (default 32 768; 131 072 with --recipe "
                         "v3 or v4)")
    ap.add_argument("--iso-channels", action="store_true",
                    help="add the isotope-charge input channels: charge, strength and isotope "
                         "limit (LadderNet only)")
    ap.add_argument("--view", default="window", choices=["window", "comb"],
                    help="LadderNet: 'window' reads a fixed window (inference cuts the grid into "
                         "windows of the training length); 'comb' reads any length, the whole "
                         "grid in one pass at inference (the frames add the measured channel)")
    ap.add_argument("--head-chunk", type=int, default=None,
                    help="LadderNet: score this many charges at a time and recompute them in "
                         "the backward pass (the same outputs in less GPU memory)")
    ap.add_argument("--arch", default="ladder", choices=["ladder", "unet"])
    ap.add_argument("--model-size", default="proof", choices=["small", "proof", "full"])
    ap.add_argument("--singly-charged", default=None, choices=["drop", "ignore", "keep"],
                    help="default: ignore (keep with --recipe v3 or v4)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-len", type=int, default=2000, help="frames per epoch")
    ap.add_argument("--val-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes generating spectra, the bottleneck on a GPU (default: as "
                         "many as the cores and --max-ram-gb allow; each needs about 1 GB)")
    ap.add_argument("--max-ram-gb", type=float, default=None,
                    help="memory budget of the training process and its workers together: "
                         "sets the default --workers, and the run pauses if it is exceeded")
    ap.add_argument("--max-gpu-memory-gb", type=float, default=None,
                    help="cap on this process's GPU memory, CUDA context included")
    ap.add_argument("--device", default="cuda",
                    help="cuda (default: the network trains on the GPU only), cpu, or auto")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for torch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat-frames", action="store_true",
                    help="reuse the same frames every epoch (and cache them in memory, one "
                         "process) instead of drawing new ones")
    ap.add_argument("--background-ions", type=int, default=None,
                    help="up to this many singly charged background ions per spectrum "
                         "(default 40, 200 with --recipe v3 or v4; more teaches the network to "
                         "reject dense background)")
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--resume", action="store_true",
                    help="continue a paused or interrupted run from <out>.resume.pt")
    ap.add_argument("--restart", action="store_true",
                    help="discard a paused run's saved state and start again")
    ap.add_argument("--keep-pause-files", action="store_true",
                    help="do not remove PAUSE files found at the start (gpu_training/train_all.py "
                         "clears them itself)")
    ap.add_argument("--save-every", type=float, default=10.0,
                    help="minutes between saves of the resumable training state")
    ap.add_argument("--out", default="runs/ladder.pt")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the installation")
    args = ap.parse_args()
    try:
        import numba  # noqa: F401
    except ImportError:
        print("WARNING: numba is not installed, so spectra are generated about 20 times slower. "
              "Install it with: pip install numba", flush=True)
    restart = args.restart
    v3 = args.recipe in ("v3", "v4")  # v4 is v3 with other spectra
    if args.singly_charged is None:
        args.singly_charged = "keep" if v3 else "ignore"
    if args.window is None:
        args.window = V3_WINDOW if v3 else 32768
    if args.workers is None:
        args.workers = auto_workers(args.max_ram_gb)
    if args.workers > 0:
        single_threaded_workers()
    if args.smoke:
        args.epochs, args.train_len, args.val_len, args.batch_size = 1, 16, 8, 2
        args.out = "runs/smoke.pt"
        restart = not args.resume  # a smoke run is a throwaway: it may overwrite the last one
    cfg = TrainConfig(model_size=args.model_size, epochs=args.epochs, train_len=args.train_len,
                      val_len=args.val_len, batch_size=args.batch_size,
                      num_workers=args.workers, lr=args.lr, base_seed=args.seed, out=args.out,
                      max_seconds=args.minutes * 60.0 if args.minutes else None,
                      threads=args.threads, device=args.device, arch=args.arch,
                      singly_charged=args.singly_charged,
                      fresh_each_epoch=not args.repeat_frames, cache_frames=args.repeat_frames,
                      init=args.init, resume=args.resume, restart=restart,
                      save_every_minutes=args.save_every, max_ram_gb=args.max_ram_gb,
                      clear_pause_files=not args.keep_pause_files,
                      max_gpu_memory_gb=args.max_gpu_memory_gb,
                      window=args.window, iso_channels=args.iso_channels, view=args.view,
                      head_chunk=args.head_chunk)
    print(f"{args.workers} data workers; spectra are generated on the processor, the network "
          f"trains on {args.device}", flush=True)
    if v3:
        print(f"recipe {args.recipe} (experimental): {args.window}-bin window, {args.view} "
              f"view, singly charged ions {args.singly_charged}"
              + (", isotope-charge channels" if args.iso_channels else ""), flush=True)
        make = mixture_v4 if args.recipe == "v4" else mixture_v3
        synth = make(args.window, background_ions=args.background_ions)
    else:
        synth = mixture(crop=args.window, background_ions=args.background_ions)
    train(cfg, synth_config=synth)
    raise SystemExit(exit_code(Path(cfg.out)))


#: exit codes: the run finished; it was paused (Ctrl+C, a PAUSE file, --minutes); it stopped at
#: the memory budget (resume it with fewer workers)
FINISHED, PAUSED, MEMORY = 0, 3, 4


def exit_code(out: Path) -> int:
    """How the run whose checkpoint is ``out`` ended, from the state it left."""
    import torch

    resume = out.with_suffix(".resume.pt")
    if not resume.exists():
        return FINISHED
    reason = torch.load(resume, map_location="cpu", weights_only=False).get("stop_reason")
    return {"paused": PAUSED, "time": PAUSED, "memory": MEMORY}.get(reason, FINISHED)


if __name__ == "__main__":
    main()
