"""One isotope envelope, placed at different charges: what each estimator reads.

On the grid u = ln(m/z -/+ m_p) a species of mass M at charge z sits at u = ln M - ln z and its
isotopologues are ln((M + 1.00335)/M) / step bins apart whatever z is. A network that sees only
the window's content (no absolute position) must then give a lone envelope the same charge at
every z. This renders one oligonucleotide envelope (class model, Gaussian isotopologue peaks of
the Thermo runs' measured width, no noise; sigma = 1) alone on the runs' grid at z = 2..6, and
with its z + 1 state at a quarter of its height, and reports each estimator's intensity-weighted
majority charge on the envelope's bins (a markdown table, one row per mass and placement:
"top charge (its share)").

    python bench/tools/blindspot.py --mass 4512.8 1884.5 [--model LABEL=NET.onnx ...]
        [--threads N] [--providers auto|dml|cuda|cpu] --out blindspot.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/
import _paths  # noqa: E402, F401  (<repo>/src and <repo>/scripts on sys.path)

from tools import common  # noqa: E402

common.preset_threads()  # before numpy: --threads caps its BLAS / OpenMP / numba pools

import numpy as np  # noqa: E402

from tools import envlib  # noqa: E402


def render(grid, mass, states, height=3000.0, fwhm_bins=1.14):
    """The envelope of ``mass`` at each (z, relative height) of ``states`` on ``grid``."""
    y = np.zeros(grid.size)
    m, a = envlib.pattern(mass, "ps_rna")
    sd = fwhm_bins / 2.3548
    bins = np.arange(grid.size)
    for z, rel in states:
        for mi, ai in zip(m, a):
            c = (np.log(mi / z) - grid.u_min) / grid.step
            lo, hi = int(c) - 5, int(c) + 6
            y[lo:hi] += height * rel * ai * np.exp(-0.5 * ((bins[lo:hi] - c) / sd) ** 2)
    return y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mass", type=float, nargs="+", default=[4512.8])
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--providers", default="auto", choices=("auto", "dml", "cuda", "cpu"))
    ap.add_argument("--out", required=True, type=Path, help="the markdown table")
    args = ap.parse_args(argv)
    models = common.parse_models(args.model)
    common.cap_threads(args.threads)

    from lcmsdeconv.core.method import Method
    from lcmsdeconv.deconv.classical import CombPredictor, make_predictor
    from lcmsdeconv.nn.grid import LogMzGrid
    from lcmsdeconv.nn.infer import ChargePredictor

    inst = Method.load("orbitrap_iprp_neg_oligo").instrument
    ests = {"comb": make_predictor("comb", z_max=40, instrument=inst),
            "comb, ladder only": CombPredictor(z_max=40, instrument=None)}
    prov = common.providers(args.providers)
    for label, path in models.items():
        ests[label] = ChargePredictor(path, providers=prov, instrument=inst)  # isotope channels
    grid = LogMzGrid(399.0, 2500.0, 2e-5, -1)
    lines = ["| mass | states | " + " | ".join(ests) + " |",
             "| --- | --- |" + " --- |" * len(ests)]
    print("\n".join(lines), flush=True)
    for mass in args.mass:
        m, _a = envlib.pattern(mass, "ps_rna")
        for z in range(2, 7):
            for states in ([(z, 1.0)], [(z, 1.0), (z + 1, 0.25)]):
                mzs = [(mass - zz * envlib.MP) / zz for zz, _ in states]
                if min(mzs) < 410 or max(mzs) > 2480:
                    continue
                y = render(grid, mass, states)
                b = np.round((np.log(m / z) - grid.u_min) / grid.step).astype(int)
                lab = np.unique(np.concatenate([b - 1, b, b + 1]))
                w = y[lab]
                cells = []
                for est in ests.values():
                    p = est.predict_grid(grid, y, 1.0).top1_z[lab].astype(int)
                    vals = {int(v): float(w[p == v].sum()) for v in np.unique(p)}
                    top = max(vals, key=vals.get)
                    cells.append(f"{top} ({vals[top] / w.sum():.2f})")
                desc = " + ".join(f"z={zz}" + ("" if r == 1.0 else f" x{r:g}") for zz, r in states)
                lines.append(f"| {mass:.1f} | {desc} | " + " | ".join(cells) + " |")
                print(lines[-1], flush=True)
    common.write_atomic(args.out, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
