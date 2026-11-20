"""Decode NN charge maps into candidate neutral masses via a mass histogram."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..nn.grid import LogMzGrid


class ChargeSupport:
    """Which charges contributed to each mass-histogram bin, without a per-bin Python set.

    Contributions are kept as flat (bin, charge) arrays sorted by bin; the charges supporting a
    bin range are then a slice, which keeps histogram accumulation vectorized.
    """

    __slots__ = ("bins", "charges", "n", "_starts")

    def __init__(self, bins: np.ndarray, charges: np.ndarray, n: int):
        order = np.argsort(bins, kind="stable")
        self.bins = bins[order]
        self.charges = charges[order]
        self.n = n
        self._starts = np.searchsorted(self.bins, np.arange(n + 1))

    def charges_in(self, lo: int, hi: int) -> set[int]:
        a = int(self._starts[max(0, min(lo, self.n))])
        b = int(self._starts[max(0, min(hi, self.n))])
        if b <= a:
            return set()
        return set(np.unique(self.charges[a:b]).tolist())


@dataclass
class Candidate:
    mass: float
    score: float
    charges: set[int] = field(default_factory=set)
    intensity: float = 0.0
    # seen at one charge state only, the charge fixed by resolved isotope spacing
    single_charge: bool = False
    # where the class's isotope pattern puts this species' resolved isotopologues against where
    # the data show them, in ln(mass) units: templates are built at mass * exp(template_shift)
    template_shift: float = 0.0
    # the charge states whose own envelopes passed the gate (refine_candidate_masses), as
    # opposed to every state the charge map voted for: None before the gate
    gate_charges: set[int] | None = None


def accumulate_mass_histogram(
    grid: LogMzGrid,
    intensity: np.ndarray,
    prediction,
    noise_sigma: float,
    snr: float = 3.0,
    prob_min: float = 0.05,
    mass_step: float = 5e-6,
    mass_min: float = 100.0,
    mass_max: float = 500000.0,
    z_min: int = 1,
    z_max: int = 100,
    apex_floor: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, ChargeSupport]:
    """Accumulate charge-weighted intensity into a log-mass histogram.

    Returns (log_mass_axis, weighted_intensity, charge_support_sets).
    """
    thr = snr * noise_sigma
    lm_min, lm_max = np.log(mass_min), np.log(mass_max)
    n = int((lm_max - lm_min) / mass_step) + 1
    hist = np.zeros(n, dtype=np.float64)
    pair_bins: list[np.ndarray] = []
    pair_z: list[np.ndarray] = []

    u = grid.u
    for zarr, parr in ((prediction.top1_z, prediction.top1_p), (prediction.top2_z, prediction.top2_p)):
        mask = (intensity > thr) & (zarr >= z_min) & (zarr <= z_max) & (parr >= prob_min)
        if not np.any(mask):
            continue
        idx = np.nonzero(mask)[0]
        z = zarr[idx].astype(np.float64)
        mass = z * np.exp(u[idx])  # proton-charged neutral mass
        good = (mass >= mass_min) & (mass <= mass_max)
        idx, z, mass = idx[good], z[good], mass[good]
        apex_w = apex_floor + (1.0 - apex_floor) * np.asarray(prediction.apex, dtype=float)[idx]
        w = intensity[idx] * parr[idx] * apex_w
        hb = ((np.log(mass) - lm_min) / mass_step).astype(np.int64)
        hb = np.clip(hb, 0, n - 1)
        np.add.at(hist, hb, w)
        pair_bins.append(hb)
        pair_z.append(z.astype(np.int32))
    lm_axis = lm_min + np.arange(n) * mass_step
    if pair_bins:
        support = ChargeSupport(np.concatenate(pair_bins), np.concatenate(pair_z), n)
    else:
        support = ChargeSupport(np.array([], dtype=np.int64), np.array([], dtype=np.int32), n)
    return lm_axis, hist, support


#: Isotope spacing (13C - 12C), the period of a resolved isotope pattern in the mass domain.
ISOTOPE_SPACING = 1.0033548


def smooth_mass_histogram(lm_axis: np.ndarray, hist: np.ndarray, smooth_bins: int = 5,
                          smooth_da: float = 1.2, passes: int = 3) -> np.ndarray:
    """Smooth a log-mass histogram over a fixed width in daltons, not in bins.

    When isotopes are resolved, every isotopologue peak votes for its own mass and the histogram
    of one species is a comb with a 1.0034 Da period, whose every tooth would otherwise become a
    candidate. Three passes of a 1.2 Da box approximate a Gaussian of 0.6 Da, which leaves under
    one percent of that ripple while barely widening envelopes that are several daltons wide
    anyway (large molecules, unresolved isotopes). The width is never narrower than
    ``smooth_bins`` histogram bins.
    """
    n = hist.size
    if n < 3:
        return hist.astype(float)
    step = float(lm_axis[1] - lm_axis[0])
    mass = np.exp(lm_axis)
    hw = np.maximum(smooth_bins // 2, np.round(0.5 * smooth_da / (mass * step))).astype(np.int64)
    hw = np.minimum(hw, 5000)
    idx = np.arange(n)
    lo = np.clip(idx - hw, 0, n)
    hi = np.clip(idx + hw + 1, 0, n)
    out = hist.astype(float)
    for _ in range(max(1, passes)):
        cs = np.concatenate([[0.0], np.cumsum(out)])
        out = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return out


def pick_candidates(
    lm_axis: np.ndarray,
    hist: np.ndarray,
    charge_support: ChargeSupport,
    min_charge_support: int = 2,
    rel_height: float = 1e-4,
    smooth_bins: int = 5,
    merge_ppm: float = 200.0,
    max_candidates: int = 200,
    smooth_da: float = 1.2,
    single_charge_ok: bool = False,
) -> list[Candidate]:
    """Peak-pick the mass histogram into candidate masses.

    ``single_charge_ok`` keeps peaks voted for by a single charge state (z >= 2) as well; the
    gate (:func:`refine_candidate_masses`) then accepts them only when resolved isotope spacing
    fixes that charge.
    """
    if hist.max() <= 0:
        return []
    sm = smooth_mass_histogram(lm_axis, hist, smooth_bins, smooth_da)
    thr = rel_height * sm.max()
    mid = sm[1:-1]
    peaks = np.nonzero((mid > sm[:-2]) & (mid >= sm[2:]) & (mid > thr))[0] + 1
    step = float(lm_axis[1] - lm_axis[0])
    cands: list[Candidate] = []
    singles: list[Candidate] = []
    for p in peaks:
        # the support window follows the smoothing width so that every vote that built the
        # peak is counted, plus the bins of the original fixed window
        hw = max(smooth_bins * 3, int(round(0.75 * smooth_da / (np.exp(lm_axis[p]) * step))))
        lo, hi = max(0, p - hw), min(len(hist), p + hw + 1)
        support = charge_support.charges_in(lo, hi)
        single = len(support) < min_charge_support and support != {1}
        if single and not (single_charge_ok and len(support) == 1):
            continue
        # sub-bin parabolic interpolation for a mass accurate to a few ppm
        lm = float(lm_axis[p])
        if 0 < p < len(sm) - 1:
            y0, y1, y2 = sm[p - 1], sm[p], sm[p + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                shift = 0.5 * (y0 - y2) / den
                lm += float(np.clip(shift, -1.0, 1.0)) * step
        mass = float(np.exp(lm))
        (singles if single else cands).append(
            Candidate(mass=mass, score=float(sm[p]), charges=support,
                      intensity=float(hist[lo:hi].sum())))
    cands.sort(key=lambda c: -c.score)
    cands = cands[:max_candidates]
    # Peaks voted for by one charge state are far more numerous (every noise envelope makes
    # one) and must not push the regular candidates out of the list: they get a separate,
    # smaller allowance after them, so a near-duplicate merges into the regular candidate.
    singles.sort(key=lambda c: -c.score)
    cands += singles[: max_candidates // 4]
    # merge near-duplicate masses
    merged: list[Candidate] = []
    for c in cands:
        dup = False
        for m in merged:
            tol = max(m.mass * merge_ppm * 1e-6, 0.3)
            if abs(c.mass - m.mass) <= tol:
                m.charges |= c.charges
                dup = True
                break
        if not dup:
            merged.append(c)
    return merged


def dedupe_adduct_candidates(
    candidates: list[Candidate], deltas: dict[str, float], max_n: int = 3, tol_da: float = 0.5
) -> list[Candidate]:
    """Drop candidates that are an adduct of an already-kept (more intense) candidate.

    Their signal is captured by the kept candidate's adduct columns during fitting, so fitting
    them separately would double-count the same ions and leave a shifted, degenerate mass.
    """
    kept: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: -x.score):
        merged = False
        for base in kept:
            dm = c.mass - base.mass
            if abs(dm) <= tol_da:
                continue
            for d in deltas.values():
                for n in range(1, max_n + 1):
                    if abs(abs(dm) - n * d) > tol_da:
                        continue
                    base.charges |= c.charges
                    if dm < 0:
                        # the lighter form is the base species: the stronger candidate we kept
                        # is its adduct, so the base identity moves down to this mass. Its
                        # gate measured the heavier form's envelopes, not this mass's: they no
                        # longer say where this mass's isotopologues sit
                        base.mass = c.mass
                        base.gate_charges = None
                    merged = True
                    break
                if merged:
                    break
            if merged:
                break
        if not merged:
            kept.append(c)
    return kept


def _mean_shift(x: np.ndarray, w: np.ndarray, start: float, sigma: float,
                max_iter: int = 25) -> tuple[float, float]:
    """Gaussian-kernel mean shift on weighted points sorted by ``x``.

    Returns (converged position, kernel-weighted intensity there). Started at a species'
    predicted position it climbs to the nearest envelope, and with a kernel as wide as the
    envelope it sees resolved isotopologues as one envelope rather than a comb.
    """
    pos = float(start)
    mass_w = 0.0
    for _ in range(max_iter):
        lo = int(np.searchsorted(x, pos - 3.0 * sigma))
        hi = int(np.searchsorted(x, pos + 3.0 * sigma))
        if hi <= lo:
            return pos, 0.0
        k = w[lo:hi] * np.exp(-0.5 * ((x[lo:hi] - pos) / sigma) ** 2)
        mass_w = float(k.sum())
        if mass_w <= 0:
            return pos, 0.0
        new = float((k * x[lo:hi]).sum() / mass_w)
        if abs(new - pos) < 1e-4 * sigma:
            pos = new
            break
        pos = new
    return pos, mass_w


def envelope_geometry(mass: float, compound_class: str) -> tuple[float, float]:
    """(isotope-envelope sigma in Da, mean-shift offset of the envelope from its average mass).

    The offset is what a mean shift with an envelope-wide kernel converges to on the noise-free
    pattern, minus the average mass: isotope envelopes are skewed, so the smoothed mode sits a
    little below the average, and subtracting the same quantity measured on theory removes that
    bias from observed masses.
    """
    from ..chem.classes import class_isotope_pattern

    pat = class_isotope_pattern(mass, compound_class)
    a = pat.abundances / max(pat.abundances.sum(), 1e-300)
    avg = float((a * pat.masses).sum())
    sig = float(np.sqrt(max((a * (pat.masses - avg) ** 2).sum(), 1e-12)))
    kern = max(sig, 0.35)
    order = np.argsort(pat.masses)
    conv, _w = _mean_shift(pat.masses[order], a[order], avg, kern)
    return sig, conv - avg


def isotopes_contradict_charge(mass: float, z: int, centre_mz: float, raw_mz: np.ndarray,
                               raw_int: np.ndarray, thr: float, polarity: int,
                               compound_class: str, fwhm: float,
                               min_populated: float = 0.6, min_rel: float = 0.3,
                               height: float = 0.0, min_fill: float = 1.0 / 3.0) -> bool:
    """True when resolved isotopes show the envelope at ``centre_mz`` is not charge ``z``.

    At charge z the isotopologues of a species lie 1.00336/z apart. When the instrument resolves
    that spacing, the major isotopologue positions (abundance at least ``min_rel`` of the
    largest) must hold observed peaks; a charge state whose positions are mostly empty is not
    this species at this charge. Singly charged background ions show one peak per 1.003 Th, so
    reading them at z = 30 predicts thirty positions per dalton of which one or two are
    populated - the test removes the false high-charge ladders such background otherwise
    forms on real data. When the spacing is not resolved the test cannot speak and returns
    False.

    A position counts as populated by a peak above ``thr`` and, given the envelope's
    ``height`` (its strongest isotopologue, where the envelope was found), of at least
    ``min_fill`` of the height the isotope pattern predicts there. Vendor software keeps only
    signal, so in a correctly resampled spectrum some ion stands above the noise near almost
    every position; what a comb read at the wrong charge lacks is peaks of the right size.
    """
    from ..chem.adducts import carrier_mass
    from ..chem.classes import class_isotope_pattern

    spacing = ISOTOPE_SPACING / z
    if spacing < 2.0 * fwhm:
        return False
    pat = class_isotope_pattern(mass, compound_class)
    a = pat.abundances / max(pat.abundances.max(), 1e-300)
    major = a >= min_rel
    if major.sum() < 2:
        return False
    carrier = carrier_mass(polarity)
    avg = float((pat.abundances * pat.masses).sum() / max(pat.abundances.sum(), 1e-300))
    iso_mz = (pat.masses[major] - avg + mass + z * carrier) / z
    need = np.maximum(thr, min_fill * a[major] * max(height, 0.0))
    tol = max(0.5 * fwhm, 0.2 * spacing)
    # The comb is slid across one spacing and the best placement counts: the measured mass can
    # be off by a fraction of a dalton (a neighbouring adduct or impurity pulls the envelope's
    # centre), which at low charge is a large fraction of the spacing, while background read at
    # the wrong charge cannot fill a comb at any placement.
    best = 0
    for shift in np.linspace(-0.5, 0.5, 11) * spacing:
        populated = 0
        for x, level in zip(iso_mz + shift, need):
            lo = int(np.searchsorted(raw_mz, x - tol))
            hi = int(np.searchsorted(raw_mz, x + tol))
            if hi > lo and raw_int[lo:hi].max() > level:
                populated += 1
        best = max(best, populated)
    return best < min_populated * major.sum()


def isotope_ladder_spacing(centre_mz: float, half_width: float, raw_mz: np.ndarray,
                           raw_int: np.ndarray, thr: float, fwhm: float,
                           min_rel: float = 0.2, min_peaks: int = 4) -> float | None:
    """Spacing of the isotopologue ladder around ``centre_mz``, measured from its peaks.

    The local maxima of the profile within ``half_width`` that stand above ``thr`` and above
    ``min_rel`` of the strongest are centroided; maxima closer than one ``fwhm`` are one peak.
    When at least ``min_peaks`` of them form a ladder - consecutive gaps that agree with their
    median within 10 % - the median gap is returned, else None. The spacing is 1.00336/z, so
    it reads the charge directly, without assuming one.
    """
    lo = int(np.searchsorted(raw_mz, centre_mz - half_width))
    hi = int(np.searchsorted(raw_mz, centre_mz + half_width))
    if hi - lo < 5:
        return None
    x, y = raw_mz[lo:hi], raw_int[lo:hi]
    level = max(thr, min_rel * float(y.max()))
    idx = np.nonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > level))[0] + 1
    if idx.size < min_peaks:
        return None
    pos = []
    for i in idx:
        a, b = max(i - 2, 0), min(i + 3, y.size)
        w = np.clip(y[a:b] - 0.5 * y[i], 0, None)
        pos.append(float((x[a:b] * w).sum() / w.sum()) if w.sum() > 0 else float(x[i]))
    peaks = [pos[0]]
    for q in pos[1:]:
        if q - peaks[-1] >= fwhm:
            peaks.append(q)
    if len(peaks) < min_peaks:
        return None
    gaps = np.diff(peaks)
    d = float(np.median(gaps))
    if d <= 0 or np.mean(np.abs(gaps - d) <= 0.1 * d) < 0.75:
        return None
    return d


def isotope_charge_unique(mass: float, z: int, centre_mz: float, raw_mz: np.ndarray,
                          raw_int: np.ndarray, thr: float, polarity: int, compound_class: str,
                          fwhm: float, min_populated: float = 0.8, height: float = 0.0) -> bool:
    """True when resolved isotopes fix the envelope at ``centre_mz`` at charge ``z`` alone.

    One charge state is evidence enough for a species when the instrument resolves its
    isotopologue spacing, 1.00336/z: the peaks of the envelope must form a ladder whose
    measured spacing is that of charge z rather than z - 1 or z + 1 (within a third of the
    difference), and the comb of isotopologue positions at z must be filled (``min_populated``
    of the major positions above ``thr``). A species at 2z would show a ladder of half the
    spacing, one at z/2 double; an envelope made of several overlapping species shows no clean
    ladder and is declined.
    """
    if z < 2 or ISOTOPE_SPACING / z < 2.0 * fwhm:
        return False
    env_sigma, _offset = envelope_geometry(mass, compound_class)
    d = isotope_ladder_spacing(centre_mz, 2.0 * env_sigma / z, raw_mz, raw_int, thr, fwhm)
    if d is None:
        return False
    expected = ISOTOPE_SPACING / z
    if abs(d - expected) > expected / (3.0 * (z + 1)):
        return False
    return not isotopes_contradict_charge(mass, z, centre_mz, raw_mz, raw_int, thr, polarity,
                                          compound_class, fwhm, min_populated=min_populated,
                                          height=height)


def envelope_discontinuous(centre_mz: float, half_width: float, raw_mz: np.ndarray,
                           raw_int: np.ndarray, fwhm: float, min_coverage: float = 0.35,
                           rel_level: float = 0.2, spacing: float | None = None) -> bool:
    """True when an envelope that should be a continuous hump is a few separate spikes.

    An envelope whose isotopes the instrument cannot separate, and which is much wider than one
    peak, appears in profile data as a continuous hump. If the profile within ``half_width`` of
    the envelope's centre stays above ``rel_level`` of its maximum over less than
    ``min_coverage`` of that interval, the "envelope" is made of narrow peaks belonging to
    something else - typically the resolved isotopes of a low-charge species, read at a high
    charge as a much heavier one.

    A spacing of more than three of the scans' own point spacings is a gap. ``spacing`` gives
    that spacing; without it, it is read off ``raw_mz``, which is right for one scan but not for
    the union of scans calibrated one by one, whose points fall several to a scan spacing.
    """
    if half_width < 3.0 * fwhm:
        return False  # a single peak is the whole envelope: nothing to test
    lo = int(np.searchsorted(raw_mz, centre_mz - half_width))
    hi = int(np.searchsorted(raw_mz, centre_mz + half_width))
    if hi - lo < 3:
        return True
    x = raw_mz[lo:hi]
    y = raw_int[lo:hi]
    level = rel_level * float(y.max())
    if level <= 0:
        return True
    dx = np.diff(x)
    own = spacing if spacing is not None and spacing > 0 else (
        float(np.median(dx)) if dx.size else 0.0)
    gap = 3.0 * own
    above = (y[:-1] >= level) & (y[1:] >= level) & (dx <= max(gap, 1e-12))
    return float(dx[above].sum()) < min_coverage * 2.0 * half_width


def refine_candidate_masses(
    candidates: list[Candidate],
    raw_mz: np.ndarray,
    raw_int: np.ndarray,
    polarity: int,
    noise_sigma: float = 0.0,
    tol_ppm: float = 400.0,
    min_charges: int = 2,
    max_spread_ppm: float = 150.0,
    snr: float = 3.0,
    drop_unsupported: bool = True,
    compound_class: str = "peptide",
    resolution: float = 30000.0,
    link_ppm: float = 60.0,
    link_da: float = 1.5,
    min_cluster_weight: float = 0.5,
    instrument=None,
    profile: bool = False,
    single_charge_ok: bool = False,
    envelope_move: bool = False,
    frames=None,
    coelution_r: float = 0.5,
) -> list[Candidate]:
    """Measure candidate masses on the raw spectrum and drop unsupported candidates.

    A real neutral mass must produce an envelope at its predicted m/z for at least
    ``min_charges`` charge states, and the masses those envelopes imply must agree. For each
    charge state a Gaussian-kernel mean shift, started at the predicted position with a kernel
    as wide as the envelope, climbs to the nearest envelope; the mass it lands on is corrected
    for the envelope's skew so that it estimates the average mass. The per-charge masses are
    then clustered (single linkage at ``max(link_ppm, link_da)``) and the cluster carrying most
    of the intensity decides: it needs ``min_charges`` members, at least ``min_cluster_weight``
    of the intensity and a spread within ``max_spread_ppm``. Charge states in other clusters are
    envelopes of other species that happened to lie under one charge state's window, and no
    longer veto an otherwise consistent candidate.

    A mean shift that has to travel further than ``tol_ppm`` (with ``envelope_move``, or 2.5
    envelope widths if more) belongs to some other envelope and is discarded for that charge
    state. Candidates carried by the wings of a stronger envelope
    converge onto that envelope and are removed by the later duplicate merge.

    With ``single_charge_ok`` a candidate seen at one charge state only is kept when resolved
    isotope spacing fixes that charge (:func:`isotope_charge_unique`); it is marked
    ``single_charge``. Under ion pairing, oligonucleotides often carry one dominant charge state,
    and an impurity at a few percent then shows that state alone above the noise.

    ``frames``, a :class:`~lcmsdeconv.deconv.coelution.FrameStack` of the scans the spectrum
    sums, adds the test that the noise threshold cannot make on vendor-thresholded data: the
    charge states of the deciding cluster must co-elute with its strongest one (scan-by-scan
    traces of their envelope cores correlating at ``coelution_r`` or better,
    :func:`~lcmsdeconv.deconv.coelution.coeluting`). A state that does not is another
    species' envelope under this one's window and leaves the cluster, which must still have
    ``min_charges`` members. ``coelution_r <= 0`` turns the test off.
    """
    from ..chem.adducts import carrier_mass
    from .coelution import coeluting

    carrier = carrier_mass(polarity)
    thr = snr * noise_sigma
    out: list[Candidate] = []
    order = np.argsort(raw_mz)
    raw_mz = raw_mz[order]
    raw_int = np.clip(raw_int[order].astype(float), 0, None)

    def accept_single(c: Candidate, masses, matched_z, where, i: int) -> bool:
        """Keep ``c`` on its one charge state ``i`` if resolved isotopes fix that charge."""
        if not (single_charge_ok and instrument is not None):
            return False
        pos_i, fwhm_i, _half_i, core_i = where[i]
        if not isotope_charge_unique(masses[i], matched_z[i], pos_i, raw_mz, raw_int, thr,
                                     polarity, compound_class, fwhm_i, height=core_i):
            return False
        c.mass = float(masses[i])
        c.charges = {int(matched_z[i])}
        c.gate_charges = set(c.charges)
        c.single_charge = True
        out.append(c)
        return True

    for c in candidates:
        env_sigma, offset = envelope_geometry(c.mass, compound_class)
        inst_sigma_da = c.mass / max(resolution, 1.0) / 2.354820045
        kern_da = float(np.sqrt(env_sigma ** 2 + inst_sigma_da ** 2))
        max_move_da = c.mass * tol_ppm * 1e-6
        if envelope_move:
            # a candidate anywhere within its envelope's half-width may climb to the envelope's
            # centre: when charges are read from isotope spacing, a strong unrelated peak that
            # sits on an envelope's isotope ladder inherits its charge and can pull the
            # histogram peak several daltons off for small molecules
            max_move_da = max(max_move_da, 2.5 * kern_da)
        masses, weights, matched_z, where = [], [], [], []
        for z in sorted(c.charges):
            mz_pred = (c.mass + offset + z * carrier) / z
            pos, kw = _mean_shift(raw_mz, raw_int, mz_pred, kern_da / z)
            if kw <= 0:
                continue
            m = z * (pos - carrier) - offset
            if abs(m - c.mass) > max_move_da:
                continue
            # the envelope must actually stand above the noise around where the shift converged;
            # with resolved isotopes the converged centre can sit in the valley between two
            # isotopologues, so the core of the envelope is searched, not the single point
            half = max(0.5 * kern_da / z, 2.0 * pos / max(resolution, 1.0))
            lo = int(np.searchsorted(raw_mz, pos - half))
            hi = int(np.searchsorted(raw_mz, pos + half))
            core = float(raw_int[lo:hi].max()) if hi > lo else 0.0
            if core <= thr:
                continue
            r_here = float(instrument.resolving_power(pos)) if instrument is not None else resolution
            fwhm = pos / max(r_here, 1.0)
            if isotopes_contradict_charge(m, z, pos, raw_mz, raw_int, thr, polarity,
                                          compound_class, fwhm, height=core):
                continue
            if profile and ISOTOPE_SPACING / z < 2.0 * fwhm:
                hw = env_sigma / z
                own = frames.spacing(pos - hw, pos + hw) if frames is not None else None
                if envelope_discontinuous(pos, hw, raw_mz, raw_int, fwhm, spacing=own):
                    continue
            masses.append(m)
            weights.append(kw)
            matched_z.append(int(z))
            where.append((pos, fwhm, half, core))

        if len(masses) < min_charges:
            if len(masses) == 1 and accept_single(c, masses, matched_z, where, 0):
                continue
            if drop_unsupported and c.charges != {1}:
                continue
            out.append(c)
            continue
        ms = np.array(masses)
        ws = np.array(weights)
        zs = np.array(matched_z)
        srt = np.argsort(ms)
        ms, ws, zs = ms[srt], ws[srt], zs[srt]
        link = max(c.mass * link_ppm * 1e-6, link_da)
        breaks = np.nonzero(np.diff(ms) > link)[0] + 1
        groups = np.split(np.arange(ms.size), breaks)
        best = max(groups, key=lambda g: ws[g].sum())
        if frames is not None and coelution_r > 0 and best.size >= 2:
            members = srt[best]
            traces = frames.traces([where[i][0] for i in members], [where[i][2] for i in members])
            best = best[coeluting(traces, int(np.argmax(ws[best])), coelution_r)]
        if best.size < min_charges or ws[best].sum() < min_cluster_weight * ws.sum():
            if (best.size == 1 and ws[best].sum() >= min_cluster_weight * ws.sum()
                    and accept_single(c, masses, matched_z, where, int(srt[best[0]]))):
                continue
            if drop_unsupported:
                continue
            out.append(c)
            continue
        bm, bw = ms[best], ws[best]
        cw = np.cumsum(bw)
        med = float(bm[int(np.searchsorted(cw, 0.5 * cw[-1]))])
        spread = float(np.sqrt(np.average((bm - med) ** 2, weights=bw)) / med * 1e6)
        if drop_unsupported and spread > max_spread_ppm:
            continue
        c.mass = med
        c.gate_charges = set(zs[best].tolist())
        c.charges = c.gate_charges | c.charges
        out.append(c)
    return out


def suppress_harmonics(candidates: list[Candidate], tol_ppm: float = 500.0,
                       max_order: int = 6, score_ratio: float = 0.6,
                       suppress_multimers: bool = True) -> list[Candidate]:
    """Drop candidates that are a charge harmonic of a stronger one.

    Reading an envelope's charges as z/n puts a spurious candidate at M/n carrying exactly the
    same ions, and reading them as n*z puts one at n*M. Both are artefacts of the charge
    assignment rather than separate species, and both are dropped when clearly weaker than the
    candidate they mirror.

    Multiples (n*M) are the ambiguous direction: a genuine dimer is indistinguishable from a
    doubled charge assignment within one spectrum. ``suppress_multimers`` should be turned off
    for native and size-exclusion work, where oligomers are the point of the experiment.
    """
    kept: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: -x.score):
        harmonic = False
        for s in kept:
            if c.score >= score_ratio * s.score:
                continue
            tol = max(c.mass * tol_ppm * 1e-6, 2.0)
            for n in range(2, max_order + 1):
                if abs(c.mass - s.mass / n) <= tol:
                    harmonic = True
                    break
                if suppress_multimers and abs(c.mass - s.mass * n) <= max(tol, n * 0.5):
                    harmonic = True
                    break
            if harmonic:
                break
        if not harmonic:
            kept.append(c)
    return kept


def drop_harmonics_of(candidates: list[Candidate], masses: list[float], tol_ppm: float = 500.0,
                      max_order: int = 6, suppress_multimers: bool = True) -> list[Candidate]:
    """Drop candidates that are a charge harmonic of an already-accepted mass.

    Once a species has been fitted and subtracted it no longer appears in the candidate list,
    so a later pass over the residual has nothing to recognise its harmonics against. Passing
    the accepted masses back in keeps M/n and n*M artefacts out of the later passes.
    """
    if not masses:
        return candidates
    kept: list[Candidate] = []
    for c in candidates:
        harmonic = False
        for m in masses:
            tol = max(c.mass * tol_ppm * 1e-6, 2.0)
            for n in range(2, max_order + 1):
                if abs(c.mass - m / n) <= tol:
                    harmonic = True
                    break
                if suppress_multimers and abs(c.mass - m * n) <= max(tol, n * 0.5):
                    harmonic = True
                    break
            if harmonic:
                break
        if not harmonic:
            kept.append(c)
    return kept
