"""Version 0 of docs/iwsp_transformer.md: CombPredictor written as a hard-attention transformer.

Key tokens are every grid bin (s, raw intensity, position); every active bin (I > 3 sigma) spawns
one query token per charge z = 1..Zmax (a hypothesis). Each head is evaluated in its exact limit:
a pointer head (score -beta (q - p_j)^2, zero-valued sink at -beta/4) reads the key at its target
or the sink's 0; a tent head interpolates linearly between the two keys around a fractional
target; an argmax head takes the hard argmax over a bin's hypotheses with the sink first (the
earliest key wins ties). ``leakage`` evaluates the pointer heads as softmax instead, with and
without a floor neuron relu(read - 1e-100), to show what leakage does to the comb's strict tests.

    synthetic  ChargePrediction of the network against CombPredictor, field by field, on spectra
               of the default and Orbitrap generators (Zmax 40 and 60)
    real       the same on every predictor call (both detection passes) of process_run on real
               runs, with the comb's species table
    leakage    softmax pointer heads at beta 50 and 400, with and without the floor, on every
               predictor call of a real run

    python scripts/iwsp_probes/comb_transformer_v0.py synthetic --tof 48 --orbitrap 48
    python scripts/iwsp_probes/comb_transformer_v0.py real IPRPLC_NC_mix.raw IPHILIC_NC_mix.raw
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lcmsdeconv.deconv.classical import CombPredictor
from lcmsdeconv.nn.infer import ChargePrediction

ISO = 1.0033548
LADDER_HEADS = [(1, +1), (1, -1), (2, +1), (2, -1), (3, +1), (3, -1)]  # comb order: z+k, z-k
NO_HEAD = np.iinfo(np.int64).max


# ------------------------------------------------------------------ write_weights (closed form)
def write_weights(step: float, zmax: int, B: int) -> dict:
    """Integer ladder offsets per (z, head), the count n_z, constants. No data is read."""
    off = np.full((zmax, 6), NO_HEAD, dtype=np.int64)
    n = np.zeros(zmax)
    for z in range(1, zmax + 1):
        for h, (k, sgn) in enumerate(LADDER_HEADS):
            zz = z + sgn * k
            if zz < 1:
                continue
            sh = int(round(np.log(z / zz) / step))  # the comb's own expression
            if sh == 0 or abs(sh) >= B:
                continue
            off[z - 1, h] = sh
            n[z - 1] += 1
    return {"off": off, "n": n, "M": 100.0, "agree": 0.6, "strong": 0.25,
            "res_factor": 2.5, "conf": np.float32(0.9)}


# ------------------------------------------------------------------ attention primitives
def hard_pointer(values, q):
    """Pointer head in its exact limit: the exact target (score 0) beats the sink (-beta/4), the
    sink beats every other key (<= -beta); an out-of-range target reads the sink's 0."""
    B = values.size
    ok = (q >= 0) & (q < B)
    out = np.zeros(q.shape)
    out[ok] = values[q[ok]]
    return out


pointer_read = hard_pointer


def tent_read(values, q):
    """Tent-kernel head: weights relu(1 - |q - p_j|) on the two keys around q (they sum to 1)."""
    B = values.size
    q = np.clip(q, 0, B - 1.000001)  # the query clamp of the comb's interpolation
    j0 = np.floor(q).astype(np.int64)
    j1 = np.minimum(j0 + 1, B - 1)
    return values[j0] * (1.0 - (q - j0)) + values[j1] * (q - j0)


def argmax_head(scores, values):
    """Hard argmax over each row's keys [sink, z = 1..Zmax]; the earliest key wins ties.
    ``scores`` [N, Z]; the sink (score 0, value 0) is key 0. Returns (index or 0, values)."""
    N = scores.shape[0]
    full = np.concatenate([np.zeros((N, 1)), scores], axis=1)
    win = np.argmax(full, axis=1)  # the first maximum
    vals = [np.concatenate([np.zeros((N, 1)), v], axis=1)[np.arange(N), win] for v in values]
    return win, vals


# ------------------------------------------------------------------ the network
def forward(W, grid, intensity, noise_sigma, zmax, instrument=None, min_snr=3.0):
    B = intensity.size
    ns = max(noise_sigma, 1e-9)
    s = np.clip(np.log10(1.0 + np.clip(intensity, 0, None) / ns), 0.0, 5.0)
    thr = min_snr * ns
    idx = np.nonzero(intensity > thr)[0]
    top1_z = np.zeros(B, np.int16)
    top2_z = np.zeros(B, np.int16)
    top1_p = np.zeros(B, np.float32)
    top2_p = np.zeros(B, np.float32)
    apex = np.zeros(B, np.float32)
    if idx.size:
        Z = np.arange(1, zmax + 1)
        # layer 1: six ladder pointer heads, tied across z; query = p_i + off[z, h]
        acc = np.zeros((idx.size, zmax))
        for h in range(6):
            o = W["off"][:, h]
            has = o != NO_HEAD
            r = pointer_read(s, idx[:, None] + np.where(has, o, 0)[None, :])
            acc = acc + np.where(has[None, :], r, 0.0)  # residual adds in the comb's order
        ev = s[idx][:, None] * (acc / np.maximum(W["n"], 1)[None, :])  # product unit
        ev[:, W["n"] == 0] = 0.0
        # layers 2-4: argmax heads over the bin's hypotheses
        z1, (b1,) = argmax_head(ev, [ev])
        ev2 = ev - W["M"] * (Z[None, :] == z1[:, None])  # the winner masked below the sink
        z2, (b2,) = argmax_head(ev2, [ev])
        D = b1 + b2 + 1e-12  # output registers (Z1, W1, Z2, W2, D); readout p = W / D
        Z1, Z2, W1, W2 = z1.copy(), z2.copy(), b1.copy(), b2.copy()
        if instrument is not None:
            # isotope circuit: positional features of the bin's m/z (embedding)
            mz = grid.bin_to_mz(idx)
            fwhm = mz / np.asarray(instrument.resolving_power(mz), dtype=float)
            offm = np.abs(mz - grid.carrier)
            width = np.maximum(np.log1p(fwhm / offm) / grid.step, 2.0)
            nz = np.log1p((ISO / Z[None, :]) / offm[:, None]) / grid.step  # fractional bins
            res = nz >= W["res_factor"] * width[:, None]
            up = tent_read(s, idx[:, None] + nz)
            dn = tent_read(s, idx[:, None] - nz)
            hi = dn + np.maximum(up - dn, 0.0)  # ReLU max and min
            lo = up - np.maximum(up - dn, 0.0)
            iso = np.where(res, s[idx][:, None] * (0.75 * hi + 0.25 * lo), 0.0)
            _, (biso,) = argmax_head(iso, [iso])
            ok = (iso >= W["agree"] * biso[:, None]) & (biso[:, None] > 0)
            zch, _ = argmax_head(Z[None, :] * ok, [])  # the largest agreeing charge
            zi = np.where(biso >= W["strong"] * s[idx] ** 2, zch, 0)
            F = zi > 0
            G = F & (z1 != zi)
            c = float(W["conf"])                     # float32(0.9) as float64
            c2 = float(np.float32(1.0) - W["conf"])  # the comb's float32 1 - 0.9
            Z2 = np.where(G, z1, Z2)
            W2 = np.where(G, c2 * D, W2)
            Z1 = np.where(F, zi, Z1)
            W1 = np.where(F, c * D, W1)
        top1_z[idx], top2_z[idx] = Z1, Z2
        top1_p[idx], top2_p[idx] = (W1 / D).astype(np.float32), (W2 / D).astype(np.float32)
    # apex: pointer heads at -1 and +1 on the raw intensity
    mid = intensity[1:-1]
    apex[1:-1] = ((mid >= intensity[:-2]) & (mid > intensity[2:]) & (mid > thr)).astype(np.float32)
    return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)


def compare(a: ChargePrediction, b: ChargePrediction) -> dict:
    return {"z1": int((a.top1_z != b.top1_z).sum()), "z2": int((a.top2_z != b.top2_z).sum()),
            "p1": int((a.top1_p != b.top1_p).sum()), "p2": int((a.top2_p != b.top2_p).sum()),
            "apex": int((a.apex != b.apex).sum())}


def soft_pointer(beta: float, floor: bool):
    """Pointer heads as softmax: score -beta d^2 over the target and its two neighbours plus the
    sink at -beta/4 (all other keys negligible), then optionally relu(read - 1e-100)."""
    def read(values, q):
        B = values.size
        num = np.zeros(q.shape)
        den = np.full(q.shape, np.exp(-beta / 4))
        for d in (-1, 0, 1):
            j = q + d
            ok = (j >= 0) & (j < B)
            w = np.exp(-beta * d * d)
            num[ok] += w * values[j[ok]]
            den[ok] += w
        out = num / den
        far = (q < 0) | (q >= B)
        if far.any():
            dist = np.where(q < 0, -q, q - (B - 1))[far].astype(float)
            wk = np.exp(-beta * dist ** 2 + beta / 4)
            out[far] = wk * np.where(q[far] < 0, values[0], values[-1]) / (1 + wk)
        return np.maximum(out - 1e-100, 0.0) if floor else out
    return read


# ------------------------------------------------------------------ checks
def cmd_synthetic(args) -> None:
    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.nn.baseline import frame_grid
    from lcmsdeconv.synth.config import SynthConfig, orbitrap_config
    from lcmsdeconv.synth.frames import generate_frame

    rng = np.random.default_rng(args.seed)
    tot: dict = {}
    t_comb = t_net = 0.0
    for kind, n in (("tof", args.tof), ("orbitrap", args.orbitrap)):
        cfg = SynthConfig(background_ions_range=(0, 40))
        if kind == "orbitrap":
            cfg = orbitrap_config(background_ions_range=(0, 40))
            cfg.resolution_range = (35000.0, 280000.0)
        for _ in range(n):
            fs = generate_frame(cfg, rng)
            grid, inten, noise = frame_grid(cfg, fs)
            for zmax in (40, 60):
                inst = (InstrumentModel(fs.meta["kind"], float(fs.meta["resolution"]))
                        if kind == "orbitrap" else None)
                t0 = time.perf_counter()
                ref = CombPredictor(z_max=zmax, instrument=inst).predict_grid(grid, inten, noise)
                t1 = time.perf_counter()
                out = forward(write_weights(grid.step, zmax, inten.size), grid, inten, noise,
                              zmax, instrument=inst)
                t_comb += t1 - t0
                t_net += time.perf_counter() - t1
                agg = tot.setdefault((kind, zmax), {"spectra": 0, "active": 0, "z1": 0, "z2": 0,
                                                    "p1": 0, "p2": 0, "apex": 0})
                agg["spectra"] += 1
                agg["active"] += int((inten > 3 * max(noise, 1e-9)).sum())
                for k, v in compare(ref, out).items():
                    agg[k] += v
    for k, v in tot.items():
        print(k, "differing:", v)
    print(f"comb {t_comb:.1f} s, network {t_net:.1f} s")


class _Both:
    """A predictor that returns the comb's answer and counts where version 0 differs."""

    def __init__(self, zmax, instrument, variants=None):
        self.comb = CombPredictor(z_max=zmax, instrument=instrument)
        self.zmax, self.instrument = zmax, instrument
        self.backend, self.model_path = "comb", None
        self.variants = variants or {"v0": None}
        self.calls, self.bins, self.diff = 0, 0, {}
        self.per_call, self.t_comb, self.t_net = [], [], []

    def predict_grid(self, grid, intensity, noise_sigma):
        global pointer_read
        t0 = time.perf_counter()
        ref = self.comb.predict_grid(grid, intensity, noise_sigma)
        self.t_comb.append(time.perf_counter() - t0)
        for name, reader in self.variants.items():
            pointer_read = reader or hard_pointer
            t0 = time.perf_counter()
            W = write_weights(grid.step, self.zmax, intensity.size)
            out = forward(W, grid, intensity, noise_sigma, self.zmax, instrument=self.instrument)
            if name == "v0":
                self.t_net.append(time.perf_counter() - t0)
            d = compare(ref, out)
            acc = self.diff.setdefault(name, dict.fromkeys(d, 0))
            for k, v in d.items():
                acc[k] += v
        pointer_read = hard_pointer
        self.calls += 1
        n = int((intensity > 3 * max(noise_sigma, 1e-9)).sum())
        self.bins += n
        self.per_call.append(n)
        return ref


def _run(paths, variants=None) -> None:
    from lcmsdeconv.core.method import Method
    from lcmsdeconv.io.thermo import read_thermo_raw
    from lcmsdeconv.process import process_run

    for path in paths:
        method = Method.load("orbitrap_iprp_neg_oligo")
        p = _Both(min(int(method.deconvolution.charge_range[1]), 60), method.instrument, variants)
        res = process_run(read_thermo_raw(path), method, predictor=p)
        print(f"{Path(path).name}: {p.calls} predictor calls, {p.bins} active bins "
              f"({min(p.per_call)}-{max(p.per_call)} per call, {p.zmax} charges each), "
              f"{len(res.species)} species; differing: {p.diff}; time per call: comb median "
              f"{np.median(p.t_comb):.3f} s, version 0 median {np.median(p.t_net):.3f} s",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe", choices=("synthetic", "real", "leakage"))
    ap.add_argument("raw", nargs="*", type=Path)
    ap.add_argument("--tof", type=int, default=48)
    ap.add_argument("--orbitrap", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.probe == "synthetic":
        cmd_synthetic(args)
    elif args.probe == "real":
        _run(args.raw)
    else:
        _run(args.raw, {f"beta {b:g}, floor {f}": soft_pointer(b, f)
                        for b in (50.0, 400.0) for f in (False, True)})


if __name__ == "__main__":
    main()
