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


def shifted(x: np.ndarray, sh: int) -> np.ndarray:
    """``y[i] = x[i + sh]``, zero where ``i + sh`` falls outside the array.

    ``np.roll`` wraps around instead, which made the comb at one edge of a window read the
    spectrum at the other edge: features near window edges then depended on unrelated signal,
    differently in a training crop than in an inference window.
    """
    L = x.size
    out = np.zeros_like(x)
    if sh == 0:
        out[:] = x
    elif 0 < sh < L:
        out[: L - sh] = x[sh:]
    elif -L < sh < 0:
        out[-sh:] = x[: L + sh]
    return out


def comb_evidence(logint: np.ndarray, step: float, z_max: int = 64, ks=(1, 2, 3)) -> np.ndarray:
    """Charge evidence for every charge 1..z_max: ``[z_max, L]``.

    For charge z, a bin's evidence is its log intensity times the mean log intensity at the
    positions where the same species' charge states z +/- k would lie. Charge 1 has no lower
    neighbours and its upper ones are further away than a window, so its row is usually zero.
    """
    L = logint.size
    ev = np.zeros((z_max, L))
    shifts = comb_shifts(step, z_max, ks)
    for z in range(2, z_max + 1):
        acc = np.zeros(L)
        cnt = 0
        for sh in shifts[z]:
            if sh == 0:
                continue
            acc += shifted(logint, sh)
            cnt += 1
        if cnt:
            ev[z - 1] = logint * (acc / cnt)
    return ev


def comb_transform(logint: np.ndarray, step: float, z_max: int = 64, ks=(1, 2, 3),
                   evidence: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Charge-evidence comb over a window. Returns (max_evidence[L], argmax_z[L]/z_max)."""
    ev = comb_evidence(logint, step, z_max, ks) if evidence is None else evidence
    L = logint.size
    best = np.zeros(L)
    bestz = np.zeros(L)
    if ev.shape[0] > 1:
        sub = ev[1:]  # charges 2..z_max, as before
        idx = np.argmax(sub, axis=0)
        val = sub[idx, np.arange(L)]
        pos = val > 0
        best[pos] = val[pos]
        bestz[pos] = idx[pos] + 2
    if best.max() > 0:
        best = best / best.max()
    return best, bestz / z_max


#: Evidence values are products of two log-SNR values (each clipped at 5), so 25 is the ceiling.
EVIDENCE_SCALE = 25.0


def featurize(window: np.ndarray, noise_sigma: float, step: float, z_max: int = 64,
              evidence_channels: int = 0) -> np.ndarray:
    """Return a ``[N_CHANNELS + evidence_channels, L]`` float32 feature tensor for a window.

    With ``evidence_channels`` > 0 the comb's evidence for each charge 1..evidence_channels is
    appended, scaled to [0, 1]. The four base channels summarise it as its maximum and the
    charge that attains it (``z / z_max``); a network has to learn some sixty thresholds to turn
    that one number back into a class, whereas one channel per charge can be read by a
    pointwise layer directly.
    """
    from scipy.ndimage import maximum_filter1d

    L = window.size
    ns = max(noise_sigma, 1e-6)
    ch0 = np.clip(np.log10(1.0 + np.clip(window, 0, None) / ns), 0.0, 5.0)
    local_max = maximum_filter1d(window, size=min(4001, (L // 4) * 2 + 1), mode="nearest")
    ch1 = np.where(local_max > 0, window / (local_max + 1e-9), 0.0)
    zc = max(z_max, evidence_channels)
    ev = comb_evidence(ch0, step, zc)
    ch2, ch3 = comb_transform(ch0, step, z_max=z_max, evidence=ev[:z_max])
    out = np.empty((N_CHANNELS + evidence_channels, L), dtype=np.float32)
    out[0] = ch0
    out[1] = ch1
    out[2] = ch2
    out[3] = ch3
    if evidence_channels:
        out[N_CHANNELS:] = np.clip(ev[:evidence_channels] / EVIDENCE_SCALE, 0.0, 1.0)
    return out
