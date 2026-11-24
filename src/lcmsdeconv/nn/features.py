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


# ------------------------------------------------------------------ isotope-spacing charges
#: the isotope reader's settings, as ``deconv.classical.CombPredictor`` uses them: charges up to
#: 60, bins above three sigma, a spacing resolved when it spans 2.5 peak widths, and the largest
#: charge whose evidence comes within 0.6 of the best
ISOTOPE_Z_MAX = 60
ISOTOPE_MIN_SNR = 3.0
ISOTOPE_RESOLUTION_FACTOR = 2.5
ISOTOPE_AGREEMENT = 0.6
#: number of isotope-charge channels (:func:`isotope_channels`)
N_ISOTOPE_CHANNELS = 2
#: instruments whose spectra the isotope channels are read on: those the pipeline reads isotope
#: charges on by default (``process._isotope_mode``). On time-of-flight spectra the reader is
#: outside the range it works in (most envelopes' spacing is below what it resolves, and it then
#: answers the highest charge it does resolve), so their channels stay empty
ISOTOPE_KINDS = ("orbitrap", "fticr")


def isotope_reading(grid, s: np.ndarray, active: np.ndarray, instrument,
                    z_max: int = ISOTOPE_Z_MAX,
                    resolution_factor: float = ISOTOPE_RESOLUTION_FACTOR,
                    agreement: float = ISOTOPE_AGREEMENT) -> tuple[np.ndarray, np.ndarray]:
    """Charge per bin read from resolved isotope spacing: ``(zraw, best)``, both ``[B]``.

    For each charge z whose isotope spacing 1.00336/z ``instrument`` resolves at a bin's m/z
    (``grid.bin_to_mz``, absolute: the spacing on the grid depends on it), the evidence is the
    bin's log-SNR ``s`` times the log-SNR one spacing either side. A z envelope also scores at
    the divisors of z (two or more isotopes away) but not at 2z, whose half-spacing positions
    fall between its peaks, so ``zraw`` is the largest charge whose evidence is at least
    ``agreement`` times the best (where the best is positive), 0 elsewhere and at bins that are
    not ``active``; ``best`` is the best evidence (0 at inactive bins). This is the reader of
    :meth:`lcmsdeconv.deconv.classical.CombPredictor.isotope_charges`, which adds its own rule
    on the strength of the evidence, and of the network's isotope channels
    (:func:`isotope_channels`).
    """
    B = s.size
    zraw = np.zeros(B, dtype=np.int16)
    best_out = np.zeros(B)
    idx = np.nonzero(active)[0]
    if idx.size == 0:
        return zraw, best_out
    mz = grid.bin_to_mz(idx)
    fwhm = mz / np.asarray(instrument.resolving_power(mz), dtype=float)
    off = np.abs(mz - grid.carrier)
    # a peak is never narrower on the grid than about two bins, whatever the instrument
    width_bins = np.maximum(np.log1p(fwhm / off) / grid.step, 2.0)

    def at(x):  # linear interpolation of s at fractional bin positions
        x = np.clip(x, 0, B - 1.000001)
        i0 = np.floor(x).astype(np.int64)
        f = x - i0
        return s[i0] * (1.0 - f) + s[np.minimum(i0 + 1, B - 1)] * f

    best = np.zeros(idx.size)
    ev_all = []
    for z in range(1, z_max + 1):
        spacing = 1.0033548 / z
        n = np.log1p(spacing / off) / grid.step  # spacing in (fractional) bins
        resolved = n >= resolution_factor * width_bins
        if not resolved.any():
            break
        # neighbours read at their exact fractional position: a hypothesis whose spacing is
        # off by a bin or two lands on the flank of the next isotope, not on its top
        up = at(idx + n)
        dn = at(idx - n)
        # the first and last isotopologue have a neighbour on one side only
        near = 0.75 * np.maximum(up, dn) + 0.25 * np.minimum(up, dn)
        ev = np.where(resolved, s[idx] * near, 0.0)
        ev_all.append((z, ev))
        best = np.maximum(best, ev)
    if not ev_all:
        return zraw, best_out
    chosen = np.zeros(idx.size, dtype=np.int16)
    for z, ev in ev_all:  # ascending z: the last one that agrees wins
        ok = (ev >= agreement * best) & (best > 0)
        chosen[ok] = z
    zraw[idx] = chosen
    best_out[idx] = best
    return zraw, best_out


def isotope_channels(grid, intensity: np.ndarray, noise_sigma: float, instrument,
                     z_max: int = ISOTOPE_Z_MAX, min_snr: float = ISOTOPE_MIN_SNR,
                     kinds: tuple[str, ...] | None = ISOTOPE_KINDS) -> np.ndarray:
    """The two isotope-charge input channels of a whole grid spectrum, ``[2, B]`` float32.

    Channel 0 is the charge :func:`isotope_reading` reads at each bin (0 = none), channel 1 its
    strength ``clip(best / s**2, 0, 1)`` (0 where there is no charge), with ``s`` the bin's
    log-SNR and bins above ``min_snr`` sigma read, exactly as the comb estimator reads them. The
    channels depend on absolute m/z and the instrument's resolving power, so they are computed
    over the whole grid (by training frames before their crop, by the predictor once per
    spectrum) and then cut to each window: one function for both, so that training and
    inference see the same input. They are read on the instrument ``kinds`` only (None: on every
    instrument) and empty for the rest.
    """
    y = np.asarray(intensity, dtype=float)
    out = np.zeros((N_ISOTOPE_CHANNELS, y.size), dtype=np.float32)
    if kinds is not None and getattr(instrument, "kind", None) not in kinds:
        return out
    ns = max(noise_sigma, 1e-9)
    s = np.clip(np.log10(1.0 + np.clip(y, 0, None) / ns), 0.0, 5.0)
    active = y > min_snr * ns
    zraw, best = isotope_reading(grid, s, active, instrument, z_max)
    on = zraw > 0
    out[0] = zraw
    out[1, on] = np.clip(best[on] / s[on] ** 2, 0.0, 1.0)
    return out
