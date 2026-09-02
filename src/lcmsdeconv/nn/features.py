"""Input featurization for the charge network (fixed-length windows)."""

from __future__ import annotations

import numpy as np

N_CHANNELS = 4


def comb_shifts(step: float, z_max: int, ks=(1, 2, 3)) -> dict[int, list[int]]:
    """Bin shifts to a charge z's +/-k neighbours in u-space (ln(z/(z+k))/step)."""
    shifts: dict[int, list[int]] = {}
    for z in range(1, z_max + 1):
        s = []
        for k in ks:
            s.append(int(round(np.log(z / (z + k)) / step)))
            s.append(int(round(np.log(z / max(z - k, 1)) / step)) if z - k >= 1 else 0)
        shifts[z] = s
    return shifts


def comb_transform(logint: np.ndarray, step: float, z_max: int = 64, ks=(1, 2, 3)) -> tuple[np.ndarray, np.ndarray]:
    """Charge-evidence comb over a window. Returns (max_evidence[L], argmax_z[L]/z_max)."""
    L = logint.size
    best = np.zeros(L)
    bestz = np.zeros(L)
    shifts = comb_shifts(step, z_max, ks)
    for z in range(2, z_max + 1):
        acc = np.zeros(L)
        cnt = 0
        for sh in shifts[z]:
            if sh == 0:
                continue
            acc += np.roll(logint, -sh)
            cnt += 1
        if cnt:
            ev = logint * (acc / cnt)
            upd = ev > best
            best[upd] = ev[upd]
            bestz[upd] = z
    if best.max() > 0:
        best = best / best.max()
    return best, bestz / z_max


def featurize(window: np.ndarray, noise_sigma: float, step: float, z_max: int = 64) -> np.ndarray:
    """Return a [N_CHANNELS, L] float32 feature tensor for a window of grid intensities."""
    from scipy.ndimage import maximum_filter1d

    L = window.size
    ns = max(noise_sigma, 1e-6)
    ch0 = np.clip(np.log10(1.0 + np.clip(window, 0, None) / ns), 0.0, 5.0)
    local_max = maximum_filter1d(window, size=min(4001, (L // 4) * 2 + 1), mode="nearest")
    ch1 = np.where(local_max > 0, window / (local_max + 1e-9), 0.0)
    ch2, ch3 = comb_transform(ch0, step, z_max=z_max)
    out = np.empty((N_CHANNELS, L), dtype=np.float32)
    out[0] = ch0
    out[1] = ch1
    out[2] = ch2
    out[3] = ch3
    return out
