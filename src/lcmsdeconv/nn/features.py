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
#: number of isotope-charge channels (:func:`isotope_channels`): the charge read, its strength
#: and the isotope limit. Networks of the first layout read only the first two
#: (:data:`ISOTOPE_CHANNELS_FIRST_LAYOUT`)
N_ISOTOPE_CHANNELS = 3
ISOTOPE_CHANNELS_FIRST_LAYOUT = 2
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


def isotope_limit(grid, instrument, z_max: int = ISOTOPE_Z_MAX,
                  resolution_factor: float = ISOTOPE_RESOLUTION_FACTOR,
                  bins: np.ndarray | None = None) -> np.ndarray:
    """The isotope limit of every grid bin (or of ``bins``): the highest charge, up to
    ``z_max``, whose isotope spacing :func:`isotope_reading` counts as resolved at the bin's m/z
    with ``instrument``'s resolving power (spacing in bins at least ``resolution_factor`` peak
    widths, a peak never narrower than two bins); 0 where no charge is. ``int16``.

    The reader answers no charge above it, so a reading close to the limit may be a lower
    charge standing for an envelope it cannot resolve (2z read as z).

    Whether a spacing is resolved can only fall with the charge: from one charge to the next
    the spacing in bins shrinks by z / (z + 1), by far more than rounding moves it. So the limit
    is the highest resolved charge, found by the reader's expression itself at that charge and
    the next, from an estimate of it (:func:`_settled_limit`), instead of the expression at
    every charge in turn (:func:`_limit_by_charges`, which answers where the estimate cannot be
    relied on): the same values, in a few passes over the bins instead of one per charge."""
    b = np.arange(grid.size) if bins is None else np.asarray(bins)
    zlim = np.zeros(b.size, dtype=np.int16)
    if b.size == 0:
        return zlim
    # the reader's own expressions, evaluated at every bin instead of the active ones
    mz = grid.bin_to_mz(b)
    fwhm = mz / np.asarray(instrument.resolving_power(mz), dtype=float)
    off = np.abs(mz - grid.carrier)
    width_bins = np.maximum(np.log1p(fwhm / off) / grid.step, 2.0)
    thr = resolution_factor * width_bins
    settled = _settled_limit(off, thr, grid.step, z_max) if _settling_is_exact() else None
    if settled is None:
        return _limit_by_charges(off, thr, grid.step, z_max, zlim)
    zlim[:] = settled
    return zlim


def _resolved(off: np.ndarray, thr: np.ndarray, step: float, z) -> np.ndarray:
    """The reader's test of charge ``z`` (an int or an int array) at bins with offsets ``off``
    and thresholds ``thr``: the spacing ``1.0033548 / z`` in bins reaches the threshold."""
    return np.log1p((1.0033548 / z) / off) / step >= thr


def _limit_by_charges(off, thr, step, z_max, zlim: np.ndarray) -> np.ndarray:
    """:func:`isotope_limit` charge by charge into ``zlim``, as the reader tests them."""
    for z in range(1, z_max + 1):
        resolved = _resolved(off, thr, step, z)
        if not resolved.any():
            break
        zlim[resolved] = z  # ascending z: the highest resolved charge stays
    return zlim


def _settled_limit(off, thr, step, z_max) -> np.ndarray | None:
    """The highest charge up to ``z_max`` that :func:`_resolved` passes at every bin (0 for
    none), int64, or None where this does not apply (positive finite offsets, finite thresholds
    and step, and ``z_max`` an int16 charge are needed).

    The continuous solution of ``log1p((1.0033548 / z) / off) / step = thr`` is ``z = 1.0033548
    / (off * expm1(thr * step))``; rounded down and clipped to ``0 .. z_max`` it is the answer
    except within rounding of a whole charge. Each bin is then settled by the test itself: the
    estimate stands where its charge passes (or is 0) and the next fails (or it is ``z_max``),
    and elsewhere the charge steps down or up until it does. Decided by the test, at the
    charges it is decided at in turn, this is the answer the loop over the charges gives,
    provided numpy gives a value the same bits wherever it sits in an array (checked once per
    process, :func:`_settling_is_exact`), as the bins here are taken apart."""
    if not (isinstance(z_max, (int, np.integer)) and 1 <= z_max <= np.iinfo(np.int16).max
            and np.isfinite(step) and step > 0 and np.all(np.isfinite(thr))
            and np.all(np.isfinite(off)) and np.all(off > 0)):
        return None
    z_max = int(z_max)
    with np.errstate(all="ignore"):
        est = 1.0033548 / (off * np.expm1(thr * step))
    z = np.clip(np.floor(np.nan_to_num(est, nan=0.0, posinf=z_max, neginf=0.0)), 0, z_max)
    z = z.astype(np.int64)
    holds = (z == 0) | _resolved(off, thr, step, np.maximum(z, 1))
    holds &= (z >= z_max) | ~_resolved(off, thr, step, np.minimum(z + 1, z_max))
    todo = np.flatnonzero(~holds)
    for _ in range(z_max + 2):  # a charge moves one way, at most z_max steps
        if todo.size == 0:
            return z
        zt = z[todo]
        down = np.zeros(todo.size, dtype=bool)
        has = zt >= 1
        down[has] = ~_resolved(off[todo[has]], thr[todo[has]], step, zt[has])
        up = np.zeros(todo.size, dtype=bool)
        below = ~down & (zt < z_max)
        up[below] = _resolved(off[todo[below]], thr[todo[below]], step, zt[below] + 1)
        z[todo[down]] -= 1
        z[todo[up]] += 1
        todo = todo[down | up]
    return None


_SETTLING: list = []


def _settling_is_exact() -> bool:
    """Whether :func:`_settled_limit` gives :func:`_limit_by_charges`'s values on this machine,
    checked once per process: numpy's ``log1p`` gives an element the same bits wherever it sits
    (shifted, strided, reordered, alone), and the two agree on offsets and thresholds across
    the range of real grids, with bins placed on and next to the boundaries of charges."""
    if not _SETTLING:
        ok = False
        try:
            rng = np.random.default_rng(7)
            x = np.exp(rng.uniform(np.log(1e-9), np.log(10.0), 1031))
            y = np.log1p(x)
            perm = rng.permutation(x.size)
            ok = (np.array_equal(np.log1p(x[perm]), y[perm])
                  and np.array_equal(np.log1p(x[1:]), y[1:])
                  and np.array_equal(np.log1p(x[::3]), y[::3])
                  and all(np.log1p(x[i:i + 1])[0] == y[i] for i in range(0, x.size, 97)))
            step = 2e-5
            for z_max in (60, 7, 1):
                off = np.exp(rng.uniform(np.log(50.0), np.log(9000.0), 3000))
                thr = 2.5 * np.maximum(np.exp(rng.uniform(np.log(0.5), np.log(300.0), 3000)), 2.0)
                # thresholds a charge just reaches, and just misses
                z = rng.integers(1, z_max + 1, 600)
                edge = np.log1p((1.0033548 / z) / off[:600]) / step
                thr[:600] = np.where(np.arange(600) % 2 == 0, edge, np.nextafter(edge, np.inf))
                want = _limit_by_charges(off, thr, step, z_max, np.zeros(off.size, np.int16))
                got = _settled_limit(off, thr, step, z_max)
                ok = ok and got is not None and np.array_equal(got, want)
        except Exception:
            ok = False
        _SETTLING.append(bool(ok))
    return _SETTLING[0]


def isotope_channels(grid, intensity: np.ndarray, noise_sigma: float, instrument,
                     z_max: int = ISOTOPE_Z_MAX, min_snr: float = ISOTOPE_MIN_SNR,
                     kinds: tuple[str, ...] | None = ISOTOPE_KINDS,
                     count: int = N_ISOTOPE_CHANNELS) -> np.ndarray:
    """The isotope-charge input channels of a whole grid spectrum, ``[count, B]`` float32.

    Channel 0 is the charge :func:`isotope_reading` reads at each bin (0 = none), channel 1 its
    strength ``clip(best / s**2, 0, 1)`` (0 where there is no charge), with ``s`` the bin's
    log-SNR and bins above ``min_snr`` sigma read, exactly as the comb estimator reads them.
    Channel 2 is the bin's isotope limit (:func:`isotope_limit`, every bin, as a charge value):
    the highest charge the reader resolves there, so that a network can tell a reading that may
    be the half of an unresolved charge from one that cannot. ``count`` = 2 gives the first
    layout, without it. The channels depend on absolute m/z and the instrument's resolving
    power, so they are computed over the whole grid (by training frames before their crop, by
    the predictor once per spectrum) and then cut to each window: one function for both, so
    that training and inference see the same input. They are read on the instrument ``kinds``
    only (None: on every instrument) and empty for the rest, the limit included.
    """
    if count not in (ISOTOPE_CHANNELS_FIRST_LAYOUT, N_ISOTOPE_CHANNELS):
        raise ValueError(f"isotope channels come as 2 (charge, strength) or 3 (and the isotope "
                         f"limit), not {count}")
    y = np.asarray(intensity, dtype=float)
    out = np.zeros((count, y.size), dtype=np.float32)
    if kinds is not None and getattr(instrument, "kind", None) not in kinds:
        return out
    ns = max(noise_sigma, 1e-9)
    s = np.clip(np.log10(1.0 + np.clip(y, 0, None) / ns), 0.0, 5.0)
    active = y > min_snr * ns
    zraw, best = isotope_reading(grid, s, active, instrument, z_max)
    on = zraw > 0
    out[0] = zraw
    out[1, on] = np.clip(best[on] / s[on] ** 2, 0.0, 1.0)
    if count > 2:
        out[2] = isotope_limit(grid, instrument, z_max)
    return out
