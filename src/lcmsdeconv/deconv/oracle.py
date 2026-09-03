"""An oracle charge predictor built from ground truth.

Used to validate decoding, NNLS refinement and quantitation independently of the trained
network, and as a reference ("perfect charge assignment") in benchmarks.
"""

from __future__ import annotations

import numpy as np

from ..nn.grid import LogMzGrid
from ..nn.infer import ChargePrediction


class OraclePredictor:
    """Assigns every grid bin the charge of the nearest true ion peak."""

    def __init__(self, sticks, grid: LogMzGrid, window_bins: int = 60):
        self.sticks = sticks
        self.grid = grid
        self.window = window_bins

    def predict_grid(self, grid: LogMzGrid, intensity: np.ndarray, noise_sigma: float) -> ChargePrediction:
        B = intensity.size
        top1_z = np.zeros(B, dtype=np.int16)
        top1_p = np.zeros(B, dtype=np.float32)
        top2_z = np.zeros(B, dtype=np.int16)
        top2_p = np.zeros(B, dtype=np.float32)
        apex = np.zeros(B, dtype=np.float32)
        best = np.zeros(B, dtype=np.float64)
        bins = grid.mz_to_bin(self.sticks.mz)
        for b, z, inten in zip(bins, self.sticks.z, self.sticks.intensity):
            lo, hi = max(0, b - self.window), min(B, b + self.window + 1)
            better = inten > best[lo:hi]
            idx = np.nonzero(better)[0] + lo
            top1_z[idx] = int(z)
            top1_p[idx] = 1.0
            best[idx] = inten
            if 0 <= b < B:
                apex[b] = 1.0
        return ChargePrediction(top1_z, top1_p, top2_z, top2_p, apex, noise_sigma)
