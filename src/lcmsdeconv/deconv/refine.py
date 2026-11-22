"""Refine candidates by weighted NNLS envelope fitting on the grid, with residual iteration."""

from __future__ import annotations

import numpy as np

from ..chem.adducts import AdductLibrary, AdductState, carrier_mass
from ..chem.instrument import InstrumentModel
from ..core.model import Component
from .decode import Candidate
from .nnls_solve import fit_weights, weighted_nnls
from .templates import build_template


def _fit_candidate(
    cand: Candidate,
    residual: np.ndarray,
    weights: np.ndarray,
    grid,
    instrument: InstrumentModel,
    compound_class: str,
    adduct_states: list[AdductState],
    z_extend: int = 1,
    min_coef_frac: float = 1e-3,
    mz_range: tuple[float, float] | None = None,
    max_charges: int = 70,
    max_columns: int = 700,
) -> tuple[dict[tuple[int, str], float], list, float]:
    """Weighted NNLS of one candidate (charges x adduct states) against the residual.

    The charge range is bounded by what the spectrum could actually show: a charge whose m/z
    falls outside the measured range contributes no observable peak, so including it only adds
    a column that the solver must work through. Charge assignments are noisy enough that
    without this bound a single candidate can pull in sixty charge states spanning the whole
    grid, which makes the non-negative least-squares solve orders of magnitude slower.

    Returns (coefficients keyed by (z, adduct_label), templates used, explained_intensity).
    """
    charges = sorted(cand.charges)
    if charges:
        zlo = max(1, min(charges) - z_extend)
        zhi = max(charges) + z_extend
    else:
        zlo, zhi = 1, 1
    if mz_range is not None and cand.mass > 0:
        mz_lo, mz_hi = mz_range
        carrier = abs(carrier_mass(grid.polarity))
        z_from_hi = max(1, int(np.floor(cand.mass / max(mz_hi - carrier, 1e-6))))
        z_from_lo = max(1, int(np.ceil(cand.mass / max(mz_lo - carrier, 1e-6))))
        zlo = max(zlo, z_from_hi)
        zhi = min(zhi, z_from_lo)
    if zhi < zlo:
        zlo = zhi = max(1, min(charges) if charges else 1)
    # Keep the problem small enough to solve quickly: with many adduct states the column count
    # is charges x states, and the charges carrying most of the envelope are the central band.
    allowed = max(3, min(max_charges, max_columns // max(1, len(adduct_states))))
    if zhi - zlo + 1 > allowed:
        centre = int(np.median(charges)) if charges else (zlo + zhi) // 2
        half = allowed // 2
        zlo, zhi = max(zlo, centre - half), min(zhi, centre + half)
    zrange = range(zlo, zhi + 1)

    columns = []
    keys = []
    tmass = cand.mass * float(np.exp(getattr(cand, "template_shift", 0.0)))
    for z in zrange:
        for st in adduct_states:
            tpl = build_template(tmass, z, grid, instrument, compound_class,
                                 adduct_mass=st.mass, adduct_label=st.label)
            if tpl is None:
                continue
            columns.append(tpl)
            keys.append((z, st.label))
    if not columns:
        return {}, [], 0.0

    support = np.unique(np.concatenate([c.bins for c in columns]))
    support = support[(support >= 0) & (support < grid.size)]
    if support.size == 0:
        return {}, [], 0.0
    A = np.zeros((support.size, len(columns)), dtype=np.float64)
    for j, col in enumerate(columns):
        # every template bin is in the support by construction, so a sorted lookup suffices
        A[np.searchsorted(support, col.bins), j] = col.values
    w = weights[support]
    x = weighted_nnls(A, residual[support], w)
    if x.size == 0:
        return {}, [], 0.0
    peak = x.max() if x.size else 0.0
    coeffs: dict[tuple[int, str], float] = {}
    used = []
    for j, (key, t) in enumerate(zip(keys, columns)):
        if x[j] > min_coef_frac * peak and x[j] > 0:
            coeffs[key] = coeffs.get(key, 0.0) + float(x[j])
            used.append((t, float(x[j])))
    explained = float(sum(c for c in coeffs.values()))
    return coeffs, used, explained


def own_charges(cand: Candidate) -> set[int] | None:
    """The charge states measured as the candidate's own envelopes at its mass: those its gate
    passed (with those the gate passed for a duplicate merged into it) and those
    :func:`observed_charges` added, less any the gate rejected. A voted state the gate rejected
    holds another species' signal, and one it could not measure (no envelope above the noise
    there) may hold anything; neither is the candidate's. None for a candidate with no gate
    record (renamed by the adduct dedupe, or never gated)."""
    gate = getattr(cand, "gate_charges", None)
    if gate is None:
        return None
    own = set(gate) | set(getattr(cand, "observed_extra", None) or ())
    return own - set(getattr(cand, "gate_failed", None) or ())


def align_template(cand: Candidate, spectrum: np.ndarray, grid, instrument: InstrumentModel,
                   compound_class: str, max_bins: float = 1.5, step_bins: float = 0.25,
                   max_charges: int = 8, max_da: float = 0.3) -> float:
    """Shift (ln mass) that best lines the class's isotope pattern up with ``spectrum``.

    A compound class stands for an average composition, which puts each resolved isotopologue
    up to 0.1-0.2 Da from where a given molecule has it - one grid bin at Orbitrap resolving
    power and charge four. A least-squares fit cannot absorb that: a template a bin off the
    peaks it should explain is scaled to a third of them. Every charge state's base template
    is slid over +-``max_bins``, or +-``max_da`` daltons where that is wider (the same shift in
    ln(m/z) for all of them, as a mass error would be), and the shift with the largest
    normalized overlap with the spectrum is kept, no shift unless one does clearly better. The
    overlap is smooth in the shift, so a parabola through the best step and its neighbours puts
    the optimum between steps: at Orbitrap resolving power a peak is narrower than a bin, and a
    quarter bin off its peaks a template is scaled 2-9 % low. Only where the instrument resolves
    the isotopes of the candidate's central charge state; elsewhere a bin is a small part of an
    envelope and the shift is 0.
    """
    # the candidate's own envelopes (:func:`own_charges`): a charge state that failed the gate
    # holds another species' comb, which the overlap would follow
    own = own_charges(cand)
    if own is None:
        own = set(cand.charges) - set(getattr(cand, "gate_failed", None) or ())
    zs = _align_charges(cand.mass, own or cand.charges, grid, instrument, max_charges)
    if zs is None:
        return 0.0
    reach = _align_reach(cand.mass, grid, max_bins, step_bins, max_da)
    shifts = np.arange(-reach, reach + 1e-9, step_bins)
    scores = np.array([_overlap(cand.mass * float(np.exp(s * grid.step)), zs, spectrum, grid,
                                instrument, compound_class) for s in shifts])
    k = int(np.argmax(scores))
    if scores[k] <= 0:
        return 0.0
    zero = int(np.argmin(np.abs(shifts)))
    # centred on no shift unless a step explains clearly more; the parabola refines either
    if scores[zero] >= (1.0 - 1e-3) * scores[k]:
        k = zero
    best = float(shifts[k])
    if 0 < k < shifts.size - 1:
        a, b, c = scores[k - 1], scores[k], scores[k + 1]
        den = a - 2.0 * b + c
        if den < 0:
            best += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) * step_bins
    return best * grid.step


#: the spacing of an isotope comb (13C - 12C), Da
ISOTOPE_DA = 1.0033548


def _align_charges(mass: float, charges, grid, instrument: InstrumentModel | None,
                   max_charges: int = 8) -> list[int] | None:
    """The charge states a template alignment reads (at most ``max_charges`` about the middle),
    or None where the instrument does not resolve the isotopes of the central one."""
    zs = sorted(int(z) for z in charges if z >= 1)
    if not zs or instrument is None:
        return None
    zc = zs[len(zs) // 2]
    mz_c = (mass + zc * carrier_mass(grid.polarity)) / zc
    if ISOTOPE_DA / zc < 2.0 * mz_c / max(float(instrument.resolving_power(mz_c)), 1.0):
        return None
    if len(zs) > max_charges:
        mid = len(zs) // 2
        zs = zs[max(0, mid - max_charges // 2): mid + max_charges // 2]
    return zs


def _align_reach(mass: float, grid, max_bins: float = 1.5, step_bins: float = 0.25,
                 max_da: float = 0.3) -> float:
    """Half-width of the alignment window in bins: +-max_bins, or +-max_da where that is wider.
    The offsets to absorb are in daltons, and 1.5 bins is 0.04 Da at 1.2 kDa (where half the
    alignments of the IP-HILIC NC run were pinned to the edge of the window, and a 1235 Da
    shortmer was quantified 400 times low)."""
    reach = max(max_bins, max_da / max(mass, 1.0) / grid.step)
    return float(np.floor(reach / step_bins) * step_bins)


def _overlap(mass: float, zs, spectrum: np.ndarray, grid, instrument: InstrumentModel,
             compound_class: str) -> float:
    """Normalized overlap of the class's base templates at ``mass`` with ``spectrum``."""
    total = 0.0
    for z in zs:
        t = build_template(mass, z, grid, instrument, compound_class)
        if t is None:
            continue
        norm = float(np.sqrt(t.values @ t.values))
        if norm > 0:
            total += float(spectrum[t.bins] @ t.values) / norm
    return total


def comb_phase_shift(mass: float, charges, mz: np.ndarray, intensity: np.ndarray,
                     polarity: int, instrument: InstrumentModel, compound_class: str,
                     width_sigmas: float = 2.5, min_share: float = 0.1,
                     max_disagree_da: float = 0.1) -> tuple[float, float, int]:
    """Where a candidate's own resolved isotopologues sit against the class template:
    ``(shift in Da, coherence, charge states used)``, read off the points ``mz``/``intensity``
    (a spectrum, or a grid's bins and values).

    Each resolved charge state's points within ``width_sigmas`` envelope widths of the
    candidate's position are folded onto one isotope spacing: the phase of their intensity-
    weighted sum against the template's own isotopologues is the shift that puts the template
    on them, modulo one spacing (taken within half a spacing of the candidate's mass). The
    states are added as unit phasors weighted by their coherent intensity. ``coherence`` is the
    length of that sum over the windows' total intensity: the share of it that lines up on one
    comb. It is 0 unless at least two states agree, each within ``max_disagree_da`` of the
    combined phase, and every state carrying ``min_share`` of the weight agrees (a foreign
    peak in one window). A weaker state cannot veto - its phase is the less certain - but it
    counts towards the two only where it agrees.
    """
    from ..chem.classes import class_isotope_pattern

    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if mz.size == 0 or mass <= 0 or instrument is None:
        return 0.0, 0.0, 0
    carrier = carrier_mass(polarity)
    pat = class_isotope_pattern(mass, compound_class)
    a = pat.abundances / max(float(pat.abundances.sum()), 1e-300)
    iso_mass = pat.masses + (mass - pat.average_mass)
    p_t = complex(np.sum(a * np.exp(2j * np.pi * iso_mass / ISOTOPE_DA)))
    if abs(p_t) <= 0:
        return 0.0, 0.0, 0
    env_sigma = float(np.sqrt(np.sum(a * (pat.masses - pat.average_mass) ** 2)))
    units, weights, window_total = [], [], 0.0
    for z in sorted(int(z) for z in charges if int(z) >= 1):
        mz_c = (mass + z * carrier) / z
        fwhm = mz_c / max(float(instrument.resolving_power(mz_c)), 1.0)
        if ISOTOPE_DA / z < 2.0 * fwhm:
            continue  # isotopes not resolved at this charge
        hw = width_sigmas * env_sigma / z + 2.0 * fwhm
        lo, hi = np.searchsorted(mz, [mz_c - hw, mz_c + hw])
        if hi - lo < 3:
            continue
        w = np.clip(intensity[lo:hi], 0.0, None)
        if w.sum() <= 0:
            continue
        p_o = complex(np.sum(w * np.exp(2j * np.pi * z * (mz[lo:hi] - carrier) / ISOTOPE_DA)))
        d = p_o * np.conj(p_t)
        if abs(d) <= 0:
            continue
        units.append(d / abs(d))
        weights.append(abs(p_o))
        window_total += float(w.sum())
    if not units:
        return 0.0, 0.0, 0
    total = complex(np.sum(np.array(weights) * np.array(units)))
    shift = float(np.angle(total)) / (2.0 * np.pi) * ISOTOPE_DA
    coherence = abs(total) / window_total
    wsum = float(np.sum(weights))
    agreeing = 0
    for u, w in zip(units, weights):
        own = float(np.angle(u * np.conj(total))) / (2.0 * np.pi) * ISOTOPE_DA
        if abs(own) <= max_disagree_da:
            agreeing += 1
        elif w >= min_share * wsum:
            coherence = 0.0
    if agreeing < 2:
        coherence = 0.0
    return shift, float(coherence), len(units)


def tooth_position(mass: float, charges, scans, instrument: InstrumentModel,
                   compound_class: str, reach_da: float = 1.6, step_da: float = 0.02,
                   width_sigmas: float = 2.5) -> tuple[float, float] | None:
    """Which tooth of its isotope comb a species' resolved envelopes sit on: ``(average mass,
    margin)``, or None where no resolved charge state of ``charges`` holds signal in ``scans``.

    The class's isotope pattern is placed at ``mass`` plus shifts of up to ``reach_da`` and
    matched against every scan's own points within the envelope's windows (each scan on its own
    axis: on the union of many scans' axes the points are densest where most scans measured);
    the shift with the largest summed overlap gives the average mass. ``margin`` is how much
    better that fits than the best placement one isotope spacing away (1 - other / best).

    Two readings of one molecule land on one tooth wherever a region's summed spectrum put
    their average masses (a neighbour's envelope can pull them a dalton apart); a deaminated
    form, 0.98 Da heavier, lands one tooth away, and an isomer on the same one.
    """
    from ..chem.classes import class_isotope_pattern

    if mass <= 0 or instrument is None or not scans:
        return None
    pat = class_isotope_pattern(mass, compound_class)
    a = pat.abundances / max(float(pat.abundances.sum()), 1e-300)
    keep = a >= 1e-3 * a.max()
    a, rel = a[keep], (pat.masses - pat.average_mass)[keep]
    env_sigma = float(np.sqrt(np.sum(a * rel ** 2) / max(float(a.sum()), 1e-300)))
    carrier = carrier_mass(scans[0].polarity)
    shifts = np.arange(-reach_da, reach_da + 0.5 * step_da, step_da)
    scores = np.zeros(shifts.size)
    used = False
    for z in sorted({int(z) for z in charges if int(z) >= 1}):
        mz_c = (mass + z * carrier) / z
        fwhm = mz_c / max(float(instrument.resolving_power(mz_c)), 1.0)
        if ISOTOPE_DA / z < 2.0 * fwhm:
            continue  # isotopes not resolved at this charge
        sig = fwhm / 2.354820045
        hw = (width_sigmas * env_sigma + reach_da) / z + 2.0 * fwhm
        # isotopologue m/z for every shift: (shift, isotopologue)
        pos = (mass + shifts[:, None] + rel[None, :] + z * carrier) / z
        for s in scans:
            lo, hi = np.searchsorted(s.mz, [mz_c - hw, mz_c + hw])
            if hi - lo < 3:
                continue
            x = np.asarray(s.mz[lo:hi], dtype=float)
            y = np.clip(np.asarray(s.intensity[lo:hi], dtype=float), 0.0, None)
            if y.sum() <= 0:
                continue
            # the template at the scan's points, one isotopologue at a time: (shift, point)
            t = np.zeros((shifts.size, x.size))
            for j in range(a.size):
                t += a[j] * np.exp(-0.5 * ((x[None, :] - pos[:, j:j + 1]) / sig) ** 2)
            scores += t @ y
            used = True
    if not used or scores.max() <= 0:
        return None
    k = int(np.argmax(scores))
    away = np.abs(np.abs(shifts - shifts[k]) - ISOTOPE_DA) <= 0.25
    other = float(scores[away].max()) if away.any() else 0.0
    return float(mass + shifts[k]), float(1.0 - other / scores[k])


def align_candidate(cand: Candidate, spectrum: np.ndarray, grid, instrument: InstrumentModel,
                    compound_class: str, min_coherence: float = 0.5,
                    forms_of: list[float] | None = None, form_shifts=(),
                    form_tol_da: float = 1.0, grid_mz: np.ndarray | None = None,
                    ladder_of=None, ladder_tol_da: float = 0.03) -> float:
    """:func:`align_template`, and where its best shift is the window's edge, the candidate's own
    isotope comb (:func:`comb_phase_shift` over the charge states that passed the gate, read on
    ``spectrum`` - the residual, so that species already fitted do not count).

    A weak species' average mass can be read several tenths of a dalton off - a neighbour on the
    same isotope comb pulls the mean shift - and its templates then need more than the window
    allows: n-2 of the NC product 0.37-0.49 Da, a 1.9 kDa shortmer 0.36 Da. Pinned at the edge,
    between two isotopologues, they fit another species' peaks or nothing (300 times low).
    Searching the overlap past the edge finds those shifts too, but lets a candidate beside a
    strong species slide onto that species' peaks. The comb is read off the candidate's own
    envelopes instead, and taken when (a) the overlap ran into the window's edge, (b) the comb
    lies beyond the window and (c) at least two gate-passed charge states agree on it
    (``min_coherence``). Of the comb's two teeth either side of the measured mass, the one whose
    template fits the envelope better is taken: a mass read more than half a spacing off has its
    nearest tooth on the wrong side, and the edge between them fits neither. A candidate whose
    gate measured another mass's envelopes (renamed by the adduct dedupe) has no gate charges and
    is not touched, and neither is one within ``form_tol_da`` of an adduct form of an accepted
    species (masses ``forms_of``: accepted species, and masses the adduct walk moved candidates
    from; adduct shifts ``form_shifts``): an adduct form's mass is read off by its neighbouring
    forms, by up to three quarters of a dalton, and its two teeth fit alike (NC + 3Na of the
    IP-HILIC NC run, read 0.73 Da light, to 1 %). Nor is a sodium/potassium form deeper than
    that (four to eight of them, :func:`_on_cation_ladder` over ``ladder_of``, the accepted
    species and where their templates sit) whose comb lies on its parent's comb moved by the
    ladder's monoisotopic shift. The comb is read over the gate-passed states, the teeth compared
    over the middle ones of them; the overlap search reads the candidate's own states
    (:func:`own_charges`), which also hold those added for the fit.
    """
    shift = align_template(cand, spectrum, grid, instrument, compound_class)
    gate = getattr(cand, "gate_charges", None)
    if not gate:
        return shift
    if forms_of and any(abs(cand.mass - m - d) <= form_tol_da for m in forms_of
                        for d in form_shifts):
        return shift
    zs = _align_charges(cand.mass, gate, grid, instrument)
    if zs is None:
        return shift
    reach = _align_reach(cand.mass, grid)
    if abs(shift / grid.step) < reach - 1e-6:
        return shift  # an optimum inside the window
    if grid_mz is None:
        grid_mz = grid.bin_to_mz(np.arange(grid.size))
    comb, coherence, n = comb_phase_shift(cand.mass, gate, grid_mz, spectrum, grid.polarity,
                                          instrument, compound_class)
    if n == 0 or coherence < min_coherence or abs(comb) <= reach * cand.mass * grid.step:
        return shift
    other = comb - np.sign(comb) * ISOTOPE_DA
    if (_overlap(cand.mass + other, zs, spectrum, grid, instrument, compound_class)
            > _overlap(cand.mass + comb, zs, spectrum, grid, instrument, compound_class)):
        comb = other
    # a sodium/potassium form deeper than the exemption's three of a type (DA-NC + 4Na and
    # NC + 6Na + K of the IP-HILIC NC run) sits exactly on its parent's comb moved by the
    # ladder's monoisotopic shift; placed there it is fitted whole as a false species, so it
    # keeps the edge like the exempt forms. A species of another composition near such a mass
    # lies on that comb only by chance
    if ladder_of and _on_cation_ladder(cand.mass, comb, ladder_of, compound_class, form_tol_da,
                                       ladder_tol_da):
        return shift
    return float(np.log1p(comb / cand.mass))


def _template_phase(mass: float, compound_class: str) -> float:
    """Where the class template's isotopologues sit for a species of average mass ``mass``,
    modulo one isotope spacing (Da)."""
    from ..chem.classes import class_isotope_pattern

    pat = class_isotope_pattern(mass, compound_class)
    a = pat.abundances / max(float(pat.abundances.sum()), 1e-300)
    iso = pat.masses + (mass - pat.average_mass)
    phasor = np.sum(a * np.exp(2j * np.pi * iso / ISOTOPE_DA))
    return float(np.angle(phasor)) / (2.0 * np.pi) * ISOTOPE_DA


def _wrap_spacing(x: float) -> float:
    """``x`` (Da) modulo one isotope spacing, between minus and plus half a spacing."""
    return (x / ISOTOPE_DA - round(x / ISOTOPE_DA)) * ISOTOPE_DA


def _on_cation_ladder(mass: float, comb_da: float, ladder_of, compound_class: str,
                      tol_da: float, phase_tol_da: float, min_total: int = 4,
                      max_total: int = 8, max_k: int = 2) -> bool:
    """Whether a candidate read at ``mass``, whose comb puts its template at ``mass + comb_da``,
    is an ``a`` Na + ``b`` K form (``min_total`` <= a + b <= ``max_total``, b <= ``max_k``) of a
    species in ``ladder_of`` - pairs (average mass, template position less that mass, Da): within
    ``tol_da`` of its mass plus the ladder's average-mass shift, and its comb within
    ``phase_tol_da`` of the species' comb plus the ladder's monoisotopic shift. The forms share
    their parent's isotope comb exactly (sodium has one isotope), whatever their measured masses;
    a species of another composition near such a mass has its comb anywhere within the spacing,
    and within ``phase_tol_da`` of that one about one time in sixteen."""
    from ..chem.adducts import ADDUCTS

    na, k = ADDUCTS["Na"], ADDUCTS["K"]
    # the comb as comb_phase_shift measured it, against the class template at the read mass: the
    # template at mass + comb_da can come from the next 0.2 % pattern bin, whose comb sits up to
    # 0.09 Da elsewhere, and the verdict would depend on which of the comb's teeth was taken
    phase_c = _template_phase(mass, compound_class) + comb_da
    for mp, sp in ladder_of:
        if mass <= mp:
            continue
        phase_p = None
        for b in range(0, max_k + 1):
            for a in range(max(0, min_total - b), max_total - b + 1):
                if abs(mass - mp - a * na.avg_mass - b * k.avg_mass) > tol_da:
                    continue
                if phase_p is None:
                    phase_p = _template_phase(mp + sp, compound_class)
                if abs(_wrap_spacing(phase_c - phase_p - a * na.mass - b * k.mass)) <= phase_tol_da:
                    return True
    return False


def refine_frame(
    candidates: list[Candidate],
    grid,
    observed: np.ndarray,
    noise_sigma: float,
    instrument: InstrumentModel,
    compound_class: str,
    library: AdductLibrary,
    raw_mz: np.ndarray | None = None,
    raw_int: np.ndarray | None = None,
    max_iter: int = 2,
    min_charge_support: int = 2,
    min_component_frac: float = 1e-4,
    max_mass_spread_ppm: float = 150.0,
    adduct_refit_frac: float = 1e-3,
    residual: np.ndarray | None = None,
    start_id: int = 0,
    strongest_explained: float = 0.0,
    trace: list | None = None,
    trace_pass: int = 0,
    frames=None,
    coelution_r: float = 0.5,
    known_masses: list[float] | None = None,
    walked: list[float] | None = None,
    known_shifts: list[float] | None = None,
    singly_charged_ok: bool = True,
) -> tuple[list[Component], float, np.ndarray]:
    """Greedy weighted-NNLS fit of candidates, strongest first.

    Fits against ``residual`` (the observed spectrum by default) while weighting by the original
    measurement, and returns the accepted components, the fraction of signal-bearing intensity
    still unexplained, and the updated residual so the caller can re-detect on what is left.
    ``frames`` and ``coelution_r`` decide which charge states beyond the voting band are the
    species' own (:func:`observed_charges`). ``known_masses`` are the species accepted by
    earlier passes over the same spectrum, ``known_shifts`` where their templates sit (Da from
    each mass), and ``walked`` the masses the adduct walk moved candidates from in them; the
    masses walked from in this pass are appended to it. Without ``singly_charged_ok`` a fit
    to z = 1 alone is unsupported like any other single charge state, and z = 1 is added for
    the fit only where its envelope is singly charged (:func:`observed_charges`).
    """
    residual = observed.astype(np.float64).copy() if residual is None else residual
    weights = fit_weights(observed, noise_sigma, grid.step,
                          float(instrument.resolving_power(1000.0)) if instrument is not None
                          else 30000.0)
    # Detection uses only the base state and one of each adduct: enough to recognise a species
    # without paying for every adduct combination on candidates that will be rejected anyway.
    states_fast = [AdductState()]
    states_full = library.states()
    mz_range = (float(raw_mz.min()), float(raw_mz.max())) if raw_mz is not None and raw_mz.size else None
    # An abundance floor relative to the whole grid would depend on how wide the grid is and how
    # noisy the baseline is. Impurity limits are quoted against the main component, so that is
    # what candidates are measured against: the strongest fit seen so far in this frame.
    strongest = float(strongest_explained)

    components: list[Component] = []
    remaining = list(candidates)
    cid = start_id
    remaining.sort(key=lambda c: -c.score)

    def decide(cand_mass: float, outcome: str, mass: float | None = None) -> None:
        if trace is not None:
            trace.append({"pass": trace_pass, "stage": "fit", "candidate": float(cand_mass),
                          "outcome": outcome, "mass": None if mass is None else float(mass)})

    resolution = float(instrument.resolving_power(1000.0)) if instrument is not None else 30000.0
    # where adduct forms of accepted species sit: the library's states, and up to three of one
    # type (sodium ladders run past the two a fit allows per type)
    avg = library.avg_deltas()
    form_shifts = sorted({s.avg_mass for s in states_full if s.counts}
                         | {n * d for d in avg.values() for n in (1, 2, 3)})
    grid_mz = grid.bin_to_mz(np.arange(grid.size))
    # a mass the walk moved a candidate from is an adduct form of the species it joined: forms
    # one adduct state above it belong to the same species (NC + 3Na of the IP-HILIC NC run
    # walked onto NC; without it as an anchor NC's +5Na and +3Na+K forms were comb-placed and
    # reported as a 3.3 % impurity)
    walked_from: list[float] = walked if walked is not None else []
    # sodium ladders deeper than the exemption's three (refine._on_cation_ladder) only where the
    # region evidences sodium: elsewhere a species can sit on such a comb by chance (a 4949 Da
    # report of the IP-RPLC NC run lies on NC - dG's comb moved by six sodium, to 0.003 Da)
    sodium = "Na" in library.names()
    for cand in remaining:
        cand_mass0 = cand.mass
        if raw_mz is not None and raw_mz.size and cand.charges and cand.charges != {1}:
            # fit the species' whole observed envelope, not only the charges that voted for it:
            # charge states the charge map got wrong are otherwise left in the residual, where
            # they are later explained as separate, false species
            voted = set(cand.charges)
            cand.charges = observed_charges(cand.mass, cand.charges, raw_mz, raw_int,
                                            3.0 * noise_sigma, grid.polarity, compound_class,
                                            resolution, mz_range, frames=frames,
                                            coelution_r=coelution_r, instrument=instrument,
                                            singly_charged_ok=singly_charged_ok)
            cand.observed_extra = set(cand.charges) - voted
        forms = list(known_masses or ()) + [c.mass for c in components] + walked_from
        ladder = ((list(zip(known_masses or (), known_shifts or ()))
                   + [(c.mass, c.mass * float(np.expm1(c.template_shift))) for c in components])
                  if sodium else None)
        cand.template_shift = align_candidate(
            cand, residual, grid, instrument, compound_class,
            forms_of=forms, form_shifts=form_shifts, grid_mz=grid_mz, ladder_of=ladder)
        coeffs, used, explained = _fit_candidate(
            cand, residual, weights, grid, instrument, compound_class, states_fast,
            mz_range=mz_range,
        )
        if explained <= 0:
            decide(cand_mass0, "no_fit")
            continue
        charges_present = {z for (z, _lab) in coeffs}
        if (len(charges_present) < min_charge_support
                and (charges_present != {1} or not singly_charged_ok)
                and not getattr(cand, "single_charge", False)):
            decide(cand_mass0, "charge_support")
            continue
        strongest = max(strongest, explained)
        if explained < min_component_frac * strongest:
            decide(cand_mass0, "abundance_floor")
            continue
        # aggregate by charge and by adduct
        charge_int: dict[int, float] = {}
        adduct_int: dict[str, float] = {}
        for (z, lab), v in coeffs.items():
            charge_int[z] = charge_int.get(z, 0.0) + v
            adduct_int[lab or "base"] = adduct_int.get(lab or "base", 0.0) + v
        comp = Component(
            mass=cand.mass, intensity=explained, mass_type="average",
            charges=charge_int, adducts=adduct_int, score=cand.score,
            compound_class=compound_class, id=cid, template_shift=cand.template_shift,
        )
        if raw_mz is not None:
            _refine_mass(comp, raw_mz, raw_int, grid.polarity, resolution=resolution)
        if comp.mass_spread_ppm > max_mass_spread_ppm and len(charges_present) > 2:
            # charge states disagree on the mass: a degenerate fit, not a real species
            decide(cand_mass0, "mass_spread", comp.mass)
            continue
        full = None
        if len(states_full) > 1 and explained >= adduct_refit_frac * strongest:
            full = _fit_candidate(cand, residual, weights, grid, instrument, compound_class,
                                  states_full, mz_range=mz_range)
            # An adduct only ever adds mass, so if the fit is centred on an adducted form the
            # lighter base form cannot be represented at all. Trying the centre one adduct
            # lower, and keeping it when it explains more with the same adduct model, recovers
            # the true neutral mass.
            moved, full2 = _recenter_base(
                cand, full, residual, weights, grid, instrument, compound_class, states_full,
                library, mz_range, raw_mz=raw_mz, raw_int=raw_int, noise_sigma=noise_sigma,
                resolution=resolution, grid_mz=grid_mz, forms_of=forms, form_shifts=form_shifts,
                ladder_of=ladder,
            )
            if moved is not cand:
                walked_from.append(float(cand.mass))
                cand, full = moved, full2
                # re-centred: measure the mass again from the raw envelopes at the new centre,
                # with the templates where the lighter form's own alignment put them
                comp.mass = cand.mass
                comp.template_shift = cand.template_shift
                if raw_mz is not None:
                    _refine_mass(comp, raw_mz, raw_int, grid.polarity, resolution=resolution)
        if full is not None:
            coeffs_full, used_full, explained_full = full
            if explained_full >= 0.8 * explained:
                coeffs, used, explained = coeffs_full, used_full, explained_full
                adduct_int = {}
                charge_int = {}
                for (z, lab), v in coeffs.items():
                    charge_int[z] = charge_int.get(z, 0.0) + v
                    adduct_int[lab or "base"] = adduct_int.get(lab or "base", 0.0) + v
                comp.charges, comp.adducts, comp.intensity = charge_int, adduct_int, explained
        comp.columns = {k: float(v) for k, v in coeffs.items()}
        components.append(comp)
        decide(cand_mass0, "accepted", comp.mass)
        cid += 1
        # subtract fitted envelope
        for tpl, coef in used:
            np.subtract.at(residual, tpl.bins, tpl.values * coef)
        residual = np.clip(residual, 0, None)

    # residual over signal-bearing bins only (whole-grid noise would dominate otherwise)
    sig_bins = observed > 10.0 * noise_sigma
    denom = float(observed[sig_bins].sum()) + 1e-9
    residual_fraction = float(residual[sig_bins].sum() / denom) if sig_bins.any() else 0.0
    for c in components:
        c.flags = list(c.flags)
    if trace is not None:
        trace.append({"pass": trace_pass, "stage": "pre_merge",
                      "masses": [float(c.mass) for c in components]})
    _merge_same_mass(components)
    _merge_adduct_components(components, library)
    if trace is not None:
        trace.append({"pass": trace_pass, "stage": "post_merge",
                      "masses": [float(c.mass) for c in components]})
    _flag_harmonics(components)
    _flag_adduct_ambiguity(components, library)
    return components, residual_fraction, residual


def _recenter_base(cand, full, residual, weights, grid, instrument, compound_class,
                   states_full, library, mz_range, raw_mz=None, raw_int=None,
                   noise_sigma: float = 0.0, resolution: float = 30000.0, max_steps: int = 3,
                   min_ratio: float = 0.5, min_gain: float = 0.01, grid_mz=None,
                   forms_of=None, form_shifts=(), ladder_of=None):
    """Walk the candidate centre down the adduct ladder while the spectrum supports it.

    Returns ``(candidate, full_fit)``: the original pair when nothing better is found.

    A step down by an adduct mass ``d`` is taken only when the lighter form's envelopes are
    present on most of the same charge states *and* carry at least ``min_ratio`` of the current
    form's height. Under any realistic adduct load the base form is the strongest member of its
    ladder, so a much weaker envelope one adduct lower is a different species (an impurity
    22 Da light, the next oligomer), not this species' base; walking onto it is what used to
    report a main component two sodium ions too light. The walk is then confirmed by refitting
    with the same adduct model at both centres and keeping the lighter one only if it explains
    more of the spectrum.
    """
    if raw_mz is None or raw_mz.size == 0 or not cand.charges or full is None:
        return cand, full
    # the candidate's mass is an average mass: step by average-mass shifts (K - H is 0.13 Da
    # heavier than its monoisotopic shift)
    deltas = sorted({d for d in library.avg_deltas().values() if d > 0})
    if not deltas:
        return cand, full
    from .decode import envelope_geometry

    carrier = carrier_mass(grid.polarity)
    # the walk reads the candidate's own envelopes (:func:`own_charges`): a charge state whose
    # window failed the gate, or that it could not measure, may hold some other species' signal
    # (an unrelated peak 30 times taller under n-2's z9 window walked it down to a phantom
    # 22 Da lighter)
    own = own_charges(cand)
    if own is None:
        own = set(cand.charges) - set(cand.gate_failed or ())
    charges = sorted(own)[:12]
    thr = 3.0 * noise_sigma

    def heights(mass: float) -> tuple[int, float]:
        env_sigma, offset = envelope_geometry(mass, compound_class)
        hits, total = 0, 0.0
        for z in charges:
            mz_pred = (mass + offset + z * carrier) / z
            half = max(0.5 * env_sigma / z, mz_pred / max(resolution, 1.0))
            lo = int(np.searchsorted(raw_mz, mz_pred - half))
            hi = int(np.searchsorted(raw_mz, mz_pred + half))
            if hi - lo < 1:
                continue
            h = float(raw_int[lo:hi].max())
            if h > thr:
                hits += 1
                total += h
        return hits, total

    start_hits, start_height = heights(cand.mass)
    if start_hits < 2 or start_height <= 0:
        return cand, full
    floor = max(2, int(round(0.6 * start_hits)))
    mass = cand.mass
    current = start_height
    moved = False
    for _ in range(max_steps):
        best_d, best_h = None, -1.0
        for d in deltas:
            if mass - d <= 0:
                continue
            hits, h = heights(mass - d)
            if hits >= floor and h >= min_ratio * current and h > best_h:
                best_d, best_h = d, h
        if best_d is None:
            break
        mass -= best_d
        current = best_h
        moved = True
    if not moved:
        return cand, full
    # the lighter form must be an envelope of its own, centred where the walk lands: a weak
    # candidate stepped down by adducts can land on the flank of a much stronger species'
    # envelope, whose isotopologues an aligned template then fits (the IP-RPLC NC product's
    # tail was split into 5159.3 and 5166.1 Da that way, its own component dropped)
    env_sigma, _off = envelope_geometry(mass, compound_class)
    centre = _envelope_centre(mass, charges, raw_mz, raw_int, grid.polarity, compound_class,
                              resolution)
    if centre is None or abs(centre - mass) > max(0.5 * env_sigma, 0.5):
        return cand, full
    trial = Candidate(mass=mass, score=cand.score, charges=set(cand.charges),
                      intensity=cand.intensity, gate_charges=cand.gate_charges,
                      gate_failed=cand.gate_failed, observed_extra=cand.observed_extra)
    # the lighter form's isotopologues are its own: the heavier centre's alignment is not theirs.
    # It is placed by the rule that placed the candidate it is compared with, the adduct-form
    # guard included: a candidate placed by its comb against a trial pinned at the window's edge
    # would always win. A trial at an accepted species' own mass is not a form of it: placed on
    # that species' comb it fits what the species left, and the walk joins the candidate to it
    # (NC + 3Na of the IP-HILIC NC run, guarded at its edge, joined NC this way)
    trial.template_shift = align_candidate(trial, residual, grid, instrument, compound_class,
                                           forms_of=forms_of, form_shifts=form_shifts,
                                           grid_mz=grid_mz, ladder_of=ladder_of)
    full2 = _fit_candidate(trial, residual, weights, grid, instrument, compound_class,
                           states_full, mz_range=mz_range)
    if full2[2] >= full[2] * (1.0 + min_gain):
        return trial, full2
    return cand, full


def _envelope_centre(mass: float, charges, raw_mz, raw_int, polarity: int, compound_class: str,
                     resolution: float) -> float | None:
    """Where the envelopes near ``mass`` are centred: the median of the per-charge mean shifts of
    :func:`_refine_mass`, started at ``mass``, over the charge states whose shift stays within
    one envelope width of it; None when none does. A charge state whose window holds a
    stronger co-eluting species' envelope converges onto that one and is left out, and on the
    flank of a stronger envelope every charge state runs off toward its centre."""
    from .decode import _mean_shift, envelope_geometry

    env_sigma, offset = envelope_geometry(mass, compound_class)
    kern_da = float(np.sqrt(env_sigma ** 2 + (mass / max(resolution, 1.0) / 2.354820045) ** 2))
    carrier = carrier_mass(polarity)
    ms = []
    for z in charges:
        pos, kw = _mean_shift(raw_mz, raw_int, (mass + offset + z * carrier) / z, kern_da / z)
        m = z * (pos - carrier) - offset
        if kw > 0 and abs(m - mass) <= env_sigma:
            ms.append(m)
    return float(np.median(ms)) if ms else None


def _set_mass(comp: Component, mass: float) -> None:
    """Move a component's mass without moving its templates: the template shift is adjusted so
    that templates stay where the fit aligned them (mass * exp(template_shift) is kept)."""
    if comp.mass > 0 and mass > 0:
        comp.template_shift += float(np.log(comp.mass / mass))
    comp.mass = float(mass)


def _refine_mass(comp: Component, raw_mz, raw_int, polarity, tol_ppm: float = 200.0,
                 resolution: float = 30000.0, link_ppm: float = 60.0, link_da: float = 1.5):
    """Measure the mass on the raw spectrum, one charge state at a time.

    Each fitted charge state's envelope is located by a Gaussian-kernel mean shift as wide as
    the envelope (see :func:`lcmsdeconv.deconv.decode.refine_candidate_masses`), which sees
    resolved isotopologues as one envelope instead of hopping between them, and the envelope's
    skew is corrected so the result estimates the average mass. The per-charge masses are
    clustered and the cluster holding most of the fitted intensity gives the mass, so one charge
    state that overlaps another species' envelope does not drag the result.
    ``mass_spread_ppm`` is the weighted spread within that cluster.
    """
    from .decode import _mean_shift, envelope_geometry

    order = np.argsort(raw_mz)
    raw_mz = np.asarray(raw_mz)[order]
    raw_int = np.clip(np.asarray(raw_int, dtype=float)[order], 0, None)
    env_sigma, offset = envelope_geometry(comp.mass, comp.compound_class)
    inst_sigma_da = comp.mass / max(resolution, 1.0) / 2.354820045
    kern_da = float(np.sqrt(env_sigma ** 2 + inst_sigma_da ** 2))
    carrier = carrier_mass(polarity)
    masses, wts = [], []
    for z, inten in comp.charges.items():
        if inten <= 0:
            continue
        mz_pred = (comp.mass + offset + z * carrier) / z
        pos, kw = _mean_shift(raw_mz, raw_int, mz_pred, kern_da / z)
        if kw <= 0:
            continue
        m = z * (pos - carrier) - offset
        if abs(m - comp.mass) > comp.mass * tol_ppm * 1e-6:
            continue
        masses.append(m)
        wts.append(inten)
    if not masses:
        return
    ms = np.array(masses)
    ws = np.array(wts)
    srt = np.argsort(ms)
    ms, ws = ms[srt], ws[srt]
    link = max(comp.mass * link_ppm * 1e-6, link_da)
    groups = np.split(np.arange(ms.size), np.nonzero(np.diff(ms) > link)[0] + 1)
    best = max(groups, key=lambda g: ws[g].sum())
    bm, bw = ms[best], ws[best]
    mean = float(np.average(bm, weights=bw))
    _set_mass(comp, mean)
    comp.mass_spread_ppm = (float(np.sqrt(np.average((bm - mean) ** 2, weights=bw)) / mean * 1e6)
                            if bm.size > 1 else 0.0)


def _merge_same_mass(components: list[Component], tol_ppm: float = 150.0, tol_da: float = 0.4):
    """Combine components that refined to the same mass (duplicate candidates)."""
    components.sort(key=lambda c: -c.intensity)
    keep: list[Component] = []
    for c in components:
        hit = None
        for k in keep:
            tol = max(k.mass * tol_ppm * 1e-6, tol_da)
            if abs(c.mass - k.mass) <= tol:
                hit = k
                break
        if hit is None:
            keep.append(c)
            continue
        w0, w1 = hit.intensity, c.intensity
        _set_mass(hit, (hit.mass * w0 + c.mass * w1) / max(w0 + w1, 1e-12))
        hit.intensity += c.intensity
        for z, v in c.charges.items():
            hit.charges[z] = hit.charges.get(z, 0.0) + v
        for a, v in c.adducts.items():
            hit.adducts[a] = hit.adducts.get(a, 0.0) + v
        for key, v in c.columns.items():
            hit.columns[key] = hit.columns.get(key, 0.0) + v
    components[:] = keep


def _merge_adduct_components(components: list[Component], library: AdductLibrary, tol_da: float = 0.5):
    """Merge components separated by an adduct delta into the more intense (base) one."""
    deltas = library.avg_deltas()
    names = library.names()
    components.sort(key=lambda c: -c.intensity)
    keep: list[Component] = []
    for c in components:
        merged = False
        for base in keep:
            dm = c.mass - base.mass
            for name, d in deltas.items():
                for n in (1, 2, 3):
                    if abs(dm - n * d) <= tol_da:
                        lab = f"+{n if n > 1 else ''}{name}"
                        base.adducts[lab] = base.adducts.get(lab, 0.0) + c.intensity
                        base.intensity += c.intensity
                        for z, v in c.charges.items():
                            base.charges[z] = base.charges.get(z, 0.0) + v
                        # its columns too, under the combined labels: the coincident-adduct
                        # step rebuilds a component from its columns
                        step = AdductState.from_label(lab, names)
                        for (z, clab), v in c.columns.items():
                            key = (z, step.plus(AdductState.from_label(clab, names), names).label)
                            base.columns[key] = base.columns.get(key, 0.0) + v
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        if not merged:
            keep.append(c)
    components[:] = keep


def _flag_adduct_ambiguity(components: list[Component], library: AdductLibrary,
                           min_fraction: float = 0.005, tol_da: float = 1.5):
    """Warn when a reported adduct delta is close to a known modification of the same class.

    +NH4 (17.027) and oxidation (+15.995) differ by ~1 Da, so at modest resolving power an
    unresolved impurity can be reported as an adduct. The user needs to know which library
    entries could be aliasing a product-related impurity.
    """
    from ..chem.modifications import annotate_delta

    deltas = library.deltas()
    for c in components:
        fr = c.adduct_fractions()
        for label, frac in fr.items():
            if label == "base" or frac < min_fraction:
                continue
            name = label.lstrip("+")
            n = 1
            while name and name[0].isdigit():
                n = int(name[0])
                name = name[1:]
            d = deltas.get(name)
            if d is None:
                continue
            hits = annotate_delta(n * d, c.compound_class, tolerance=tol_da)
            if hits:
                c.flags.append(f"adduct {label} within {tol_da} Da of {hits[0].name}")


def _flag_harmonics(components: list[Component], tol_ppm: float = 100.0):
    by_int = sorted(components, key=lambda c: -c.intensity)
    for i, c in enumerate(by_int):
        for bigger in by_int[:i]:
            for factor in (2.0, 0.5):
                if abs(c.mass - bigger.mass * factor) / c.mass * 1e6 < tol_ppm:
                    if c.intensity < 0.2 * bigger.intensity:
                        c.flags.append("possible_harmonic")
                    break


def drop_coincident_adducts(components: list[Component], library: AdductLibrary,
                            tol_da: float = 0.5, trace: list | None = None) -> list[Component]:
    """Give a weaker species' adduct envelopes that coincide with a stronger species' to it.

    Adduct ladders overlap: the PS-to-PO impurity of an oligonucleotide plus potassium
    (M - 16.03 + 37.96 Da) is the oligonucleotide plus sodium (M + 21.98 Da) to within 0.07 Da,
    and two envelopes at one mass are one envelope. The stronger species' adduct is the likelier
    source - its ladder is what the mobile phase decorates - and fitted to the weaker species
    such a column turns the main species' adducts into an impurity's amount. So an adduct column
    of a component whose mass (species plus adduct) lies within ``tol_da`` of a stronger
    component's mass plus any adduct state of the library is removed from it, whether or not the
    stronger component fitted that state; a component left with no column is dropped.
    """
    if len(components) < 2:
        return components
    # average masses compared: average-mass shifts
    shifts = sorted({s.avg_mass for s in library.states()} | {0.0})
    shift_of = {s.label or "base": s.avg_mass for s in library.states()}
    kept: list[Component] = []
    for c in sorted(components, key=lambda c: -c.intensity):
        cols = dict(c.columns)
        if kept and cols:
            for z, lab in list(cols):
                if not lab or lab == "base" or cols[(z, lab)] <= 0:
                    continue
                shift = shift_of.get(lab)
                if shift is None:  # a combined label beyond the library's limits
                    st = library.state_of(lab)
                    shift = st.avg_mass if st is not None else 0.0
                m = c.mass + shift
                if any(abs(k.mass + d - m) <= tol_da for k in kept for d in shifts):
                    del cols[(z, lab)]
            if len(cols) < len(c.columns):
                total = float(sum(v for v in cols.values() if v > 0))
                if total <= 0:
                    if trace is not None:
                        trace.append({"pass": -1, "stage": "fit", "candidate": float(c.mass),
                                      "outcome": "coincident_adducts", "mass": float(c.mass)})
                    continue
                charges: dict[int, float] = {}
                adducts: dict[str, float] = {}
                for (z, lab), v in cols.items():
                    charges[z] = charges.get(z, 0.0) + v
                    adducts[lab or "base"] = adducts.get(lab or "base", 0.0) + v
                c.columns, c.charges, c.adducts, c.intensity = cols, charges, adducts, total
        kept.append(c)
    return kept


def drop_shared_evidence(components: list[Component], library: AdductLibrary, polarity: int,
                         resolution: float = 30000.0, shared_fraction: float = 0.8,
                         trace: list | None = None) -> list[Component]:
    """Remove components whose envelopes a stronger component already explains.

    A charge misassignment reads a strong envelope as some other mass: charge 40 read as 39,
    or a whole series read at a fraction of its charge. The resulting "species" has no ions of
    its own; its envelopes sit on the stronger species' envelopes (any of its adduct forms
    included), where a stronger fitted column already accounts for them. A genuine weak species
    sits between them. When at least ``shared_fraction`` of a component's fitted intensity lies
    within two envelope sigmas of a stronger column of a stronger component, the component is
    dropped as an artefact. A real dimer survives, since its odd charge states are its own.
    """
    from .decode import envelope_geometry

    if len(components) < 2:
        return components
    carrier = carrier_mass(polarity)
    masses = {s.label or "base": s.mass for s in library.states()}
    masses.setdefault("base", 0.0)
    order = sorted(components, key=lambda c: -c.intensity)
    kept: list[Component] = []
    kept_cols: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (mz, half-width, coef)
    for c in order:
        cols = [(z, lab, v) for (z, lab), v in c.columns.items() if v > 0]
        if kept and cols:
            total = sum(v for _z, _lab, v in cols)
            shared = 0.0
            for z, lab, v in cols:
                mz = (c.mass + masses.get(lab or "base", 0.0) + z * carrier) / z
                for kmz, khw, kcoef in kept_cols:
                    hit = (np.abs(kmz - mz) <= khw) & (kcoef >= v)
                    if hit.any():
                        shared += v
                        break
            if total > 0 and shared >= shared_fraction * total:
                if trace is not None:
                    trace.append({"pass": -1, "stage": "fit", "candidate": float(c.mass),
                                  "outcome": "shared_evidence", "mass": float(c.mass)})
                continue
        kept.append(c)
        if cols:
            env_sigma, _off = envelope_geometry(c.mass, c.compound_class)
            zs = np.array([z for z, _lab, _v in cols], dtype=float)
            mz = np.array([(c.mass + masses.get(lab or "base", 0.0) + z * carrier) / z
                           for z, lab, _v in cols])
            # two envelope sigmas hold 95 % of an isotope envelope's ions
            hw = np.maximum(2.0 * env_sigma / zs, mz / max(resolution, 1.0))
            kept_cols.append((mz, hw, np.array([v for _z, _lab, v in cols])))
    return kept


def drop_multimers(components: list[Component], polarity: int, resolution: float = 30000.0,
                   frames=None, coelution_r: float = 0.5, tol_ppm: float = 500.0,
                   max_order: int = 6, ratio: float = 0.6, own_share: float = 0.25,
                   trace: list | None = None) -> list[Component]:
    """Remove components at n times the mass of a stronger, co-eluting component (n = 2 to
    ``max_order``): multimers the source makes of a species' own molecules.

    A non-covalent dimer's charge states that are multiples of two hold its monomer's ions, but
    the odd ones are its own, so :func:`drop_shared_evidence` keeps it. Before the fit,
    :func:`~lcmsdeconv.deconv.decode.suppress_harmonics` drops a candidate at n*M only when its
    mass-histogram score is clearly below the monomer's, and with resolved isotopes a multimer's
    score is not: the monomer's isotopologues, read at n times their charge, put teeth n daltons
    apart, each scored on its own instead of smoothed into one envelope, and the charge
    ladder at n*M finds the monomer's states and the multimer's own between them. The 1580 Da
    shortmer of the IP-RPLC NC run and its [2M-3H]3- dimer scored 1 : 0.74, and the dimer was
    reported at 0.8 % of the shortmer. The fitted amounts measure each species' own envelopes,
    so they decide here: a component within the multimer tolerance of ``suppress_harmonics``
    (``tol_ppm``, at least 2 Da and n * 0.5 Da) of n times a component with more than 1 /
    ``ratio`` times its intensity is dropped where it co-elutes with it - a multimer is made of
    the monomer's molecules as they elute. The strongest charge state of its own (not a multiple
    of n) is compared with the monomer's strongest over the scans the spectrum sums
    (:func:`~lcmsdeconv.deconv.coelution.coeluting`, which abstains on too few scans). A
    component whose own states carry less than ``own_share`` of its fit is the monomer's ions
    read at n times their charge, a harmonic, and goes without the test: what its own states
    hold is too little to show an elution profile (the 1234 Da shortmer's ions read as a
    2469 Da species of the IP-HILIC NC run, 9 % of its fit on odd states). The bar is half of
    what a genuine species at 2M carries on its odd states over an even charge ladder: such a
    species, eluting on its own, is judged by its elution, not by its charge-state distribution
    (the dimer of the IP-RPLC NC run has 75 % of its fit on odd states). Methods for native
    and size-exclusion work, where oligomers are the point, turn this off with
    ``suppress_multimers: false``.
    """
    from .coelution import coeluting
    from .decode import envelope_geometry

    if len(components) < 2:
        return components
    carrier = carrier_mass(polarity)

    def window(c: Component, own_of: int = 1) -> tuple[float, float] | None:
        """The envelope core of ``c``'s strongest base-form state (z not a multiple of
        ``own_of`` when that is above one)."""
        cols = [(v, z) for (z, lab), v in c.columns.items()
                if v > 0 and (not lab or lab == "base") and (own_of < 2 or z % own_of)]
        if not cols:
            return None
        _v, z = max(cols)
        env_sigma, offset = envelope_geometry(c.mass, c.compound_class)
        mz = (c.mass + offset + z * carrier) / z
        return mz, max(env_sigma / z, mz / max(resolution, 1.0))

    def coelutes(c: Component, k: Component, n: int) -> bool:
        total = sum(v for v in c.columns.values() if v > 0)
        own_fit = sum(v for (z, _lab), v in c.columns.items() if v > 0 and z % n)
        if own_fit < own_share * total:
            return True  # the monomer's ions at n times their charge: a harmonic
        own, mono = window(c, n), window(k)
        if own is None or mono is None or frames is None or coelution_r <= 0:
            return True
        tr = frames.traces([mono[0], own[0]], [mono[1], own[1]])
        return bool(coeluting(tr, 0, coelution_r)[1])

    kept: list[Component] = []
    for c in sorted(components, key=lambda c: -c.intensity):
        tol = max(c.mass * tol_ppm * 1e-6, 2.0)
        monomers = [(k, n) for k in kept if c.intensity < ratio * k.intensity
                    for n in range(2, max_order + 1)
                    if abs(c.mass - n * k.mass) <= max(tol, n * 0.5)]
        monomer = next((m for m in monomers if coelutes(c, *m)), None)
        if monomer is not None:
            if trace is not None:
                trace.append({"pass": -1, "stage": "fit", "candidate": float(c.mass),
                              "outcome": "multimer", "mass": float(monomer[0].mass)})
            continue
        kept.append(c)
    return kept


def observed_charges(mass: float, charges, raw_mz: np.ndarray, raw_int: np.ndarray, thr: float,
                     polarity: int, compound_class: str, resolution: float,
                     mz_range: tuple[float, float] | None = None, max_gap: int = 1,
                     z_cap: int = 150, frames=None, coelution_r: float = 0.5,
                     instrument=None, singly_charged_ok: bool = True) -> set[int]:
    """Charges of a species' envelope that the spectrum shows, grown outward from ``charges``.

    Starting from the charge band that voted for the species, each next charge outward is kept
    while an envelope is observed where that charge would put the species, tolerating
    ``max_gap`` missing states, and stopping at the edge of the measured m/z range. Observed
    means above ``thr`` and, where the instrument resolves the isotopes, with the isotope comb
    of that charge filled in proportion (:func:`~lcmsdeconv.deconv.decode.
    isotopes_contradict_charge` relative to the envelope's own height). With ``frames`` (a
    :class:`~lcmsdeconv.deconv.coelution.FrameStack` of the scans the spectrum sums) the
    envelope must also co-elute with the starting band (correlation ``coelution_r`` or better):
    above the noise of a thresholded spectrum there is signal of some ion near almost every
    position, and only its isotopes and its time course tell whether it is this species.
    Without ``singly_charged_ok`` z = 1 is observed only where its envelope has no peaks between
    its isotopologues (:func:`~lcmsdeconv.deconv.decode.singly_charged_contradicted`).
    """
    from .coelution import correlation, shows_profile
    from .decode import envelope_geometry, isotopes_contradict_charge, singly_charged_contradicted

    charges = {int(z) for z in charges if z >= 1}
    if not charges:
        return charges
    env_sigma, offset = envelope_geometry(mass, compound_class)
    carrier = carrier_mass(polarity)

    def window(z: int) -> tuple[float, float]:
        mz = (mass + offset + z * carrier) / z
        return mz, max(0.5 * env_sigma / z, mz / max(resolution, 1.0))

    ref = None
    if frames is not None and coelution_r > 0:
        band = sorted(charges)
        ref = frames.traces([window(z)[0] for z in band], [window(z)[1] for z in band]).sum(0)
        if not shows_profile(ref):
            ref = None

    def present(z: int) -> bool | None:
        mz, half = window(z)
        if mz_range is not None and not (mz_range[0] <= mz <= mz_range[1]):
            return None
        lo = int(np.searchsorted(raw_mz, mz - half))
        hi = int(np.searchsorted(raw_mz, mz + half))
        core = float(raw_int[lo:hi].max()) if hi > lo else 0.0
        if core <= thr:
            return False
        r_here = float(instrument.resolving_power(mz)) if instrument is not None else resolution
        if isotopes_contradict_charge(mass, z, mz, raw_mz, raw_int, thr, polarity,
                                      compound_class, mz / max(r_here, 1.0), height=core):
            return False
        if (z == 1 and not singly_charged_ok
                and singly_charged_contradicted(mz, max(2.0 * env_sigma, 1.5), raw_mz, raw_int,
                                                thr, core)):
            return False
        if ref is not None:
            return correlation(ref, frames.traces([mz], [half])[0]) >= coelution_r
        return True

    out = set(charges)
    for direction in (-1, 1):
        z = (min(charges) - 1) if direction < 0 else (max(charges) + 1)
        gap = 0
        while 1 <= z <= z_cap:
            p = present(z)
            if p is None:
                break
            if p:
                out.add(z)
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            z += direction
    return out
