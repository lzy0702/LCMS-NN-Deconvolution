"""The comb's ladder evidence as one layer of softmax pointer attention (docs/iwsp_transformer.md).

Tokens are the grid bins with signal. A pointer head reads, for every query token, the value of
the token at a target position: score -beta (t - p)^2 between the target t and each key position
p, plus one sink key with score -beta/4 and value 0. A key exactly at the target beats the sink by
beta/4; a key one bin off loses to it by 3 beta/4; so a missing target reads (almost) 0. Only keys
within +-W bins of the target are scored (searchsorted on the sorted positions); every other key
weighs below exp(-beta (W^2 - 1/4)) relative to the sink, which bounds the difference from dense
softmax attention.

    check     the layer (354 heads for z = 1..60) plus an argmax readout against CombPredictor
              without isotope charges, on spectra of the training mixture and, given a raw file,
              on the summed spectra of two real 24-scan tiles
    inrange   the per-bin benchmark of scripts/eval_charge_bins.py (96 spectra) for the pipeline
              comb and for the comb with each charge's ladder mean over in-range neighbours only

    python scripts/iwsp_probes/pointer_attention.py check --frames 12 IPRPLC_GalNAc_conjugated_mix.raw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCRIPTS = Path(__file__).resolve().parents[1]


class PointerHead:
    def __init__(self, beta: float = 200.0, window: int = 2):
        self.beta = beta
        self.window = window

    def leakage_exponent(self) -> float:
        """Keys more than ``window`` bins from the target weigh below e^-(this) against the sink."""
        return self.beta * (self.window ** 2 - 0.25)

    def __call__(self, key_pos: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
        """``key_pos`` sorted ascending, ``values`` [N] or [N, C], ``target`` [Q]."""
        n = target.size
        lo = np.searchsorted(key_pos, target - self.window, side="left")
        hi = np.searchsorted(key_pos, target + self.window, side="right")
        width = int((hi - lo).max(initial=0))
        vals = values if values.ndim == 2 else values[:, None]
        out = np.zeros((n, vals.shape[1]))
        if width:
            cand_raw = lo[:, None] + np.arange(width)[None, :]
            valid = cand_raw < hi[:, None]
            cand = np.minimum(cand_raw, key_pos.size - 1)
            d = target[:, None] - key_pos[cand]
            score = np.where(valid, -self.beta * d * d, -np.inf)
            sink = -self.beta * 0.25
            m = np.maximum(score.max(axis=1), sink)
            w = np.exp(score - m[:, None])
            z = w.sum(axis=1) + np.exp(sink - m)
            out = np.einsum("nw,nwc->nc", w / z[:, None], vals[cand])
        return out if values.ndim == 2 else out[:, 0]


def comb_layer(pos: np.ndarray, s: np.ndarray, size: int, z_max: int = 60, ks=(1, 2, 3),
               step: float = 2e-5, in_range: bool = False) -> np.ndarray:
    """Version 0's ladder layer: m[N, z_max], each charge's mean neighbour log-SNR.

    One pointer head per (z, z') with z' = z +/- k reads s at round(ln(z/z')/step); the output
    projection averages a charge's heads. Heads whose offset is 0 or not smaller than the grid are
    absent, as in CombPredictor, and the mean runs over every present head (the comb counts a rung
    beyond the grid's edge as an observed zero) or, with ``in_range``, over the rungs inside the
    grid only. The comb's evidence is s * m_z, whose argmax over z is that of m_z where s > 0.
    """
    head = PointerHead()
    m = np.zeros((pos.size, z_max))
    for z in range(1, z_max + 1):
        acc = np.zeros(pos.size)
        n = np.zeros(pos.size)
        for k in ks:
            for zz in (z + k, z - k):
                if zz < 1:
                    continue
                sh = int(round(np.log(z / zz) / step))
                if sh == 0 or abs(sh) >= size:
                    continue
                acc += head(pos, s, pos + sh)
                n += ((pos + sh >= 0) & (pos + sh <= size - 1)) if in_range else 1.0
        m[:, z - 1] = np.where(n > 0, acc / np.maximum(n, 1), 0.0)
    return m


def readout(m: np.ndarray, floor: float = 1e-30):
    """Charge = argmax over z (lowest z on ties), 0 where no charge has evidence above ``floor``;
    the top-2 share best / (best + second) as the comb reports it. ``floor`` is the readout
    neuron's bias: between the heads' leakage (exp(-150) ~ 1e-65 from a key one bin off the
    target) and any real log-SNR."""
    order = np.argsort(-m, axis=1, kind="stable")
    rows = np.arange(m.shape[0])
    best = m[rows, order[:, 0]]
    second = m[rows, order[:, 1]]
    z1 = np.where(best > floor, order[:, 0] + 1, 0)
    z2 = np.where(second > floor, order[:, 1] + 1, 0)
    p1 = np.where(best > floor, best / (best + second + 1e-12), 0.0)
    return z1, p1, z2


def _compare(grid, intensity, noise, label) -> None:
    from lcmsdeconv.deconv.classical import CombPredictor

    comb = CombPredictor(z_max=60).predict_grid(grid, intensity, noise)
    ns = max(noise, 1e-9)
    s_all = np.clip(np.log10(1.0 + np.clip(intensity, 0, None) / ns), 0.0, 5.0)
    pos = np.nonzero(s_all > 0)[0]
    t0 = time.time()
    z1, p1, z2 = readout(comb_layer(pos.astype(float), s_all[pos], intensity.size))
    dt = time.time() - t0
    a = intensity[pos] > 3.0 * ns
    print(f"{label}: {pos.size} tokens, {a.sum()} active; top-1 agree "
          f"{np.mean(z1[a] == comb.top1_z[pos][a]):.6f}, top-2 agree "
          f"{np.mean(z2[a] == comb.top2_z[pos][a]):.6f}, max |p1 diff| "
          f"{np.max(np.abs(p1[a] - comb.top1_p[pos][a])):.2e}; layer {dt:.2f} s", flush=True)


def cmd_check(args) -> None:
    sys.path.insert(0, str(SCRIPTS))
    from train_charge_net import mixture

    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.synth.dataset import FrameDataset

    print(f"a pointer head at beta 200, window +-2 bins: keys outside the window weigh below "
          f"e^-{PointerHead().leakage_exponent():.0f} against the sink")
    frames = FrameDataset(mixture(), args.frames, base_seed=args.seed)
    for i in range(args.frames):
        cfg, fs = frames.frame(i)
        g = bl.frame_grid(cfg, fs)
        _compare(*g, f"synthetic {i:2d} ({bl.spectrum_kind(fs.meta)}, grid {g[0].size})")
    for raw in args.raw:
        from scan_grid import resample_scan

        from lcmsdeconv.chem.instrument import estimate_noise_sigma
        from lcmsdeconv.io.thermo import read_thermo_raw
        from lcmsdeconv.nn.grid import LogMzGrid

        ms1 = [s for s in read_thermo_raw(raw).spectra if s.ms_level == 1]
        rt = np.array([s.rt for s in ms1])
        grid = LogMzGrid(400.0, 2500.0, 2e-5, ms1[0].polarity)
        for t in args.at:
            c = int(np.searchsorted(rt, t))
            tot = sum(resample_scan(grid, ms1[j].mz, ms1[j].intensity) for j in range(c - 12, c + 12))
            _compare(grid, tot, estimate_noise_sigma(tot), f"{Path(raw).name}, 24 scans at {t} min")


class InRangeComb:
    """The pipeline comb with the in-range ladder mean (its isotope override kept on top)."""

    def __init__(self, base):
        self.base = base

    def predict_grid(self, grid, intensity, noise):
        pred = self.base.predict_grid(grid, intensity, noise)
        ns = max(noise, 1e-9)
        s_all = np.clip(np.log10(1.0 + np.clip(intensity, 0, None) / ns), 0.0, 5.0)
        pos = np.nonzero(s_all > 0)[0]
        z1, _p1, _z2 = readout(comb_layer(pos.astype(float), s_all[pos], intensity.size,
                                          z_max=self.base.z_max, step=grid.step, in_range=True))
        top = np.zeros(intensity.size, dtype=np.int16)
        top[pos] = z1
        active = intensity > self.base.min_snr * ns
        top[~active] = 0
        if self.base.instrument is not None:
            zi, _conf = self.base.isotope_charges(grid, s_all, active)
            top[zi > 0] = zi[zi > 0]
        pred.top1_z = top
        return pred


def cmd_inrange(args) -> None:
    sys.path.insert(0, str(SCRIPTS))
    from train_charge_net import mixture

    from lcmsdeconv.nn import baseline as bl
    from lcmsdeconv.nn.checks import BIN_BANDS, POOLED, WEIGHTINGS, ChargeScoreboard
    from lcmsdeconv.synth.dataset import FrameDataset

    frames = FrameDataset(mixture(), args.frames, base_seed=args.seed)
    board = ChargeScoreboard()
    kinds: dict[str, int] = {}
    names = ["comb", "comb, in-range mean", "comb, no isotope charges",
             "comb, no isotope charges, in-range mean"]
    for i in range(args.frames):
        cfg, fs = frames.frame(i)
        grid = bl.frame_grid(cfg, fs)
        kind = bl.spectrum_kind(fs.meta)
        kinds[kind] = kinds.get(kind, 0) + 1
        true_z, f0 = fs.topk_z[:, 0].astype(int), fs.features[0]
        ests = [bl.pipeline_comb(fs.meta)]
        if kind == "orbitrap":
            ests.append(bl.pipeline_comb(fs.meta, isotope_charges=False))
        for j, est in enumerate(ests):
            board.add(kind, names[2 * j], bl.charges_on_crop(est, cfg, fs, grid), true_z, f0)
            board.add(kind, names[2 * j + 1], bl.charges_on_crop(InRangeComb(est), cfg, fs, grid),
                      true_z, f0)
    bands = [b[0] for b in BIN_BANDS]
    for kind, n in kinds.items():
        for how in (*WEIGHTINGS, POOLED):
            print(f"\n{bl.KIND_NAMES[kind]} ({n} spectra), weighted by {how}: " + " ".join(bands))
            for name in names:
                cells = [board.accuracy(kind, name, how, b) for b in bands]
                if any(c is not None for c in cells):
                    print(f"  {name:>40}: " + " ".join("  -  " if c is None else f"{c:.2f}"
                                                       for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe", choices=("check", "inrange"))
    ap.add_argument("raw", nargs="*", type=Path, help="real runs (check only)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20_000_000)
    ap.add_argument("--at", type=float, nargs="+", default=[19.25, 27.0])
    args = ap.parse_args()
    if args.frames is None:
        args.frames = 12 if args.probe == "check" else 96
    (cmd_check if args.probe == "check" else cmd_inrange)(args)


if __name__ == "__main__":
    main()
