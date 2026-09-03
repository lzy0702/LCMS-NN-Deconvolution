"""Rasterize soft charge-share labels and an apex heatmap on a grid crop."""

from __future__ import annotations

import numpy as np


def build_labels(
    charge_mat: np.ndarray,
    charges: np.ndarray,
    noise: np.ndarray,
    k: int = 3,
    heat_sigma: float = 1.5,
    apex_snr: float = 2.0,
) -> dict[str, np.ndarray]:
    """Build training labels for one crop.

    Parameters
    ----------
    charge_mat : [nz, L] pre-saturation signal per charge row.
    charges : [nz] the charge integer of each row.
    noise : [L] non-ion content (class 0).

    Returns dict with ``topk_z`` [L, k] (int16, charge integers; 0 = non-ion),
    ``topk_w`` [L, k] (float32 shares summing to <=1) and ``heat`` [L] (float32 in [0, 1]).
    """
    L = noise.size
    nz = charge_mat.shape[0]
    total_sig = charge_mat.sum(axis=0) if nz else np.zeros(L)
    denom = total_sig + noise + 1e-9

    # assemble a (nz+1, L) share matrix with class 0 = noise
    shares = np.empty((nz + 1, L), dtype=np.float64)
    shares[0] = noise / denom
    if nz:
        shares[1:] = charge_mat / denom
    class_ids = np.concatenate([[0], charges.astype(np.int64)])

    # top-k classes per bin
    kk = min(k, nz + 1)
    order = np.argsort(-shares, axis=0)[:kk]  # [kk, L]
    topk_z = np.zeros((L, k), dtype=np.int16)
    topk_w = np.zeros((L, k), dtype=np.float32)
    for j in range(kk):
        rows = order[j]
        topk_z[:, j] = class_ids[rows]
        topk_w[:, j] = shares[rows, np.arange(L)]

    # apex heatmap: local maxima of each charge row above apex_snr * local noise
    heat = np.zeros(L, dtype=np.float64)
    if nz:
        noise_level = max(np.median(noise), 1e-6)
        apex_bins: list[int] = []
        for r in range(nz):
            row = charge_mat[r]
            if row.max() < apex_snr * noise_level:
                continue
            mid = row[1:-1]
            ismax = (mid >= row[:-2]) & (mid > row[2:]) & (mid >= apex_snr * noise_level)
            apex_bins.extend((np.nonzero(ismax)[0] + 1).tolist())
        if apex_bins:
            ab = np.unique(np.array(apex_bins))
            hw = int(np.ceil(3 * heat_sigma))
            x = np.arange(-hw, hw + 1)
            kernel = np.exp(-0.5 * (x / heat_sigma) ** 2)
            for b in ab:
                lo = max(0, b - hw)
                hi = min(L, b + hw + 1)
                heat[lo:hi] = np.maximum(heat[lo:hi], kernel[(lo - b + hw):(hi - b + hw)])
    return {"topk_z": topk_z, "topk_w": topk_w, "heat": heat.astype(np.float32)}
