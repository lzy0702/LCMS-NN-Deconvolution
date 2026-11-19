"""The co-elution gate of docs/iwsp_transformer.md on the four real runs (a probe, untuned).

Version 0 (comb_transformer_v0.py) with one change: a ladder rung is read only if its trace
correlates with the bin's (Pearson r > theta) or either bin is weak (s < 1.2, S/N below ~15,
where a trace is mostly noise); otherwise the head reads the sink's 0 and the rung still counts,
like one beyond the spectrum's edge. Traces are pooled into 12 slots: four over one region width
before the region, four over the region, four over one region width after it. The isotope reader
is not gated. Scored with scripts/validate_oligo_runs.py, unchanged; also counts the active bins
whose top charge changed against the comb.

    python scripts/iwsp_probes/coelution_gate.py --theta 0.5 IPRPLC_*.raw IPHILIC_*.raw
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import comb_transformer_v0 as v0  # noqa: E402
from scan_grid import resample_scan  # noqa: E402

import lcmsdeconv.deconv.classical as classical  # noqa: E402
import lcmsdeconv.features.regions as regions  # noqa: E402
from lcmsdeconv.nn.infer import ChargePrediction  # noqa: E402

VALIDATE = Path(__file__).resolve().parents[1] / "validate_oligo_runs.py"
CURRENT: dict = {}


def _unit_rows(p):
    c = p - p.mean(-1, keepdims=True)
    n = np.linalg.norm(c, axis=-1, keepdims=True)
    return np.where(n > 0, c / np.where(n > 0, n, 1), 0.0)


def forward_gated(W, grid, intensity, noise_sigma, zmax, instrument, E, theta, s_weak,
                  min_snr=3.0):
    """Version 0 with the gate on the six ladder heads (E: [bins, 12] unit-norm traces)."""
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
        acc = np.zeros((idx.size, zmax))
        for h in range(6):
            o = W["off"][:, h]
            has = o != v0.NO_HEAD
            q = idx[:, None] + np.where(has, o, 0)[None, :]
            r = v0.hard_pointer(s, q)
            qc = np.clip(q, 0, B - 1)
            rij = np.einsum("nk,nzk->nz", E[idx], E[qc])
            ok = (rij > theta) | (s[idx][:, None] < s_weak) | (s[qc] < s_weak)
            acc = acc + np.where(has[None, :] & ok, r, 0.0)
        ev = s[idx][:, None] * (acc / np.maximum(W["n"], 1)[None, :])
        ev[:, W["n"] == 0] = 0.0
        z1, (b1,) = v0.argmax_head(ev, [ev])
        z2, (b2,) = v0.argmax_head(ev - W["M"] * (Z[None, :] == z1[:, None]), [ev])
        D = b1 + b2 + 1e-12
        Z1, Z2 = z1.copy(), z2.copy()
        P1, P2 = (b1 / D).astype(np.float32), (b2 / D).astype(np.float32)
        if instrument is not None:  # the isotope reader of version 0, ungated
            mz = grid.bin_to_mz(idx)
            fwhm = mz / np.asarray(instrument.resolving_power(mz), dtype=float)
            offm = np.abs(mz - grid.carrier)
            width = np.maximum(np.log1p(fwhm / offm) / grid.step, 2.0)
            nz = np.log1p((v0.ISO / Z[None, :]) / offm[:, None]) / grid.step
            res = nz >= W["res_factor"] * width[:, None]
            up = v0.tent_read(s, idx[:, None] + nz)
            dn = v0.tent_read(s, idx[:, None] - nz)
            hi = dn + np.maximum(up - dn, 0.0)
            lo = up - np.maximum(up - dn, 0.0)
            iso = np.where(res, s[idx][:, None] * (0.75 * hi + 0.25 * lo), 0.0)
            _, (biso,) = v0.argmax_head(iso, [iso])
            agree = (iso >= W["agree"] * biso[:, None]) & (biso[:, None] > 0)
            zch, _ = v0.argmax_head(Z[None, :] * agree, [])
            zi = np.where(biso >= W["strong"] * s[idx] ** 2, zch, 0)
            F = zi > 0
            G = F & (z1 != zi)
            Z2 = np.where(G, z1, Z2)
            P2 = np.where(G, np.float32(1.0) - W["conf"], P2).astype(np.float32)
            Z1 = np.where(F, zi, Z1)
            P1 = np.where(F, W["conf"], P1).astype(np.float32)
        top1_z[idx], top2_z[idx] = Z1, Z2
        top1_p[idx], top2_p[idx] = P1, P2
    mid = intensity[1:-1]
    apex[1:-1] = ((mid >= intensity[:-2]) & (mid > intensity[2:]) & (mid > thr)).astype(np.float32)
    return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)


class Gated:
    """The gated network as the pipeline's predictor; profiles of the region being processed."""

    backend, model_path = "iwsp-gate", None

    def __init__(self, z_max, instrument, theta, s_weak):
        self.z_max, self.instrument = z_max, instrument
        self.theta, self.s_weak = theta, s_weak
        self.cache = None
        self.changed = self.active = 0

    def profiles(self, grid):
        reg, run = CURRENT["region"], CURRENT["run"]
        key = (id(reg), grid.size, grid.mz_min)
        if self.cache and self.cache[0] == key:
            return self.cache[1]
        w = reg.end - reg.start
        edges = np.concatenate([np.linspace(reg.start - w, reg.start, 5)[:-1],
                                np.linspace(reg.start, reg.end, 5)[:-1],
                                np.linspace(reg.end, reg.end + w, 5)])
        P = np.zeros((12, grid.size))
        for f in run.frames(reg.polarity):
            if reg.start - w <= f.rt <= reg.end + w:
                k = int(np.clip(np.searchsorted(edges, f.rt, side="right") - 1, 0, 11))
                P[k] += resample_scan(grid, f.mz, f.intensity)
        E = _unit_rows(P.T)
        self.cache = (key, E)
        return E

    def predict_grid(self, grid, intensity, noise_sigma):
        W = v0.write_weights(grid.step, self.z_max, intensity.size)
        out = forward_gated(W, grid, intensity, noise_sigma, self.z_max, self.instrument,
                            self.profiles(grid), self.theta, self.s_weak)
        ref = classical.CombPredictor(z_max=self.z_max, instrument=self.instrument).predict_grid(
            grid, intensity, noise_sigma)
        a = intensity > 3 * max(noise_sigma, 1e-9)
        self.active += int(a.sum())
        self.changed += int((a & (out.top1_z != ref.top1_z)).sum())
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", nargs="+", type=Path)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--s-weak", type=float, default=1.2)
    args = ap.parse_args()
    made: list[Gated] = []
    orig_summed = regions.Region.summed

    def summed(self, run, *a, **kw):
        CURRENT["region"], CURRENT["run"] = self, run
        return orig_summed(self, run, *a, **kw)

    def make_predictor(model_path, providers=None, z_max=60, instrument=None):
        p = Gated(z_max, instrument, args.theta, args.s_weak)
        made.append(p)
        return p

    regions.Region.summed = summed
    classical.make_predictor = make_predictor
    sys.argv = [str(VALIDATE), *map(str, args.raw)]
    runpy.run_path(str(VALIDATE), run_name="__main__")
    print(f"theta {args.theta}, weak below s = {args.s_weak}: top charge changed at "
          f"{sum(p.changed for p in made)} of {sum(p.active for p in made)} active bins")


if __name__ == "__main__":
    main()
