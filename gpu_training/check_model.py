"""Read whole spectra with a trained charge model, the way the pipeline reads them.

    python gpu_training/check_model.py runs/ladder_v4_comb.onnx [more models]

Each model (.onnx, or a .pt checkpoint) is loaded through ChargePredictor with the instrument of
the Orbitrap oligonucleotide method and reads two whole grids: an Orbitrap spectrum of recipe v4
(the one the runs train), and the same spectrum on a grid extended 50 000 bins to lower m/z, so
that it is longer than the 131 072-bin training window (a comb-view model reads it in one
pass, a window-view model in overlapping windows). It prints the time taken and the share of
ion bins whose charge it reads right; the exit code is 0 when every model gave a charge for
every bin above three sigma of both grids.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
#: bins added below the spectrum's grid for the long grid
EXTRA_BINS = 50000


def spectra() -> list:
    """``(name, grid, intensity, noise sigma, true charge per bin, log-SNR feature per bin)`` of
    the two grids read."""
    from train_charge_net import mixture_v4

    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.grid import LogMzGrid
    from lcmsdeconv.synth.frames import generate_frame

    cfg = mixture_v4()[3][1]  # Orbitrap oligonucleotides at the real runs' resolving power
    fs = generate_frame(cfg, np.random.default_rng(5))
    grid, y, sigma = bl.frame_grid(cfg, fs)
    true, f0 = np.zeros(grid.size, dtype=np.int64), np.zeros(grid.size)
    n = min(fs.features.shape[1], grid.size - fs.b0)
    true[fs.b0:fs.b0 + n] = fs.topk_z[:n, 0]
    f0[fs.b0:fs.b0 + n] = fs.features[0, :n]
    # the same bins, preceded by EXTRA_BINS empty ones
    k = EXTRA_BINS
    low = float(np.exp(grid.u_min - k * grid.step) + grid.carrier)
    long = LogMzGrid(low, grid.mz_max, grid.step, grid.polarity)
    if long.size != grid.size + k:
        raise RuntimeError(f"the extended grid has {long.size} bins, not {grid.size + k}")

    def extend(a):
        out = np.zeros(long.size, dtype=a.dtype)
        out[k:] = a
        return out

    return [("Orbitrap spectrum", grid, y, sigma, true, f0),
            ("the same, grid longer than the training window", long, extend(y), sigma,
             extend(true), extend(f0))]


def main(argv: list[str]) -> int:
    from lcmsdeconv.core.method import Method
    from lcmsdeconv.nn.checks import ChargeScoreboard
    from lcmsdeconv.nn.infer import ChargePredictor

    if not argv:
        print(__doc__)
        return 2
    instrument = Method.load("orbitrap_iprp_neg_oligo").instrument
    grids = spectra()
    ok = True
    for path in argv:
        pred = ChargePredictor(path, instrument=instrument)
        what = (f"{pred.view} view, {pred.iso_count} isotope channels" if pred.iso_channels
                else f"{pred.view} view")
        print(f"{Path(path).name} ({what}):", flush=True)
        for name, grid, y, sigma, true, f0 in grids:
            t = time.time()
            p = pred.predict_grid(grid, y, sigma)
            took = time.time() - t
            board = ChargeScoreboard()
            board.add("orbitrap", "model", p.top1_z, true, f0)
            acc = board.accuracy("orbitrap", "model", "log-SNR", "z>=2")
            good = (p.top1_z.shape == (grid.size,) and bool((p.top1_z[y > 3 * sigma] > 0).all())
                    and bool(np.isfinite(p.top1_p).all()))
            ok &= good
            acc_text = "-" if acc is None else f"{acc:.2f}"
            print(f"  {name}: {grid.size} bins read in {took:.1f} s; charge of z >= 2 ion bins "
                  f"read right {acc_text} (log-SNR weighted){'' if good else '  FAILED'}",
                  flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
