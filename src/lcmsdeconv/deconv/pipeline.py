"""Per-spectrum deconvolution pipeline: NN charge maps -> candidates -> NNLS components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chem.adducts import AdductLibrary
from ..chem.instrument import InstrumentModel
from ..core.model import Component, FrameResult, Spectrum
from ..nn.grid import spectrum_to_grid
from ..nn.infer import ChargePredictor
from .coelution import FrameStack
from .decode import (
    accumulate_mass_histogram,
    dedupe_adduct_candidates,
    drop_harmonics_of,
    pick_candidates,
    refine_candidate_masses,
    suppress_harmonics,
)
from .nnls_solve import fit_weights
from .refine import drop_coincident_adducts, drop_shared_evidence, refine_frame


@dataclass
class DeconvParams:
    compound_class: str = "auto"
    class_candidates: tuple[str, ...] = ("peptide", "dna", "rna", "glycan", "peg")
    mass_range: tuple[float, float] = (200.0, 300000.0)
    charge_range: tuple[int, int] = (1, 100)
    snr: float = 3.0
    prob_min: float = 0.05
    min_charge_support: int = 2
    min_relative_abundance: float = 1e-4
    refine_iterations: int = 2
    max_components: int = 30
    max_mass_spread_ppm: float = 150.0
    suppress_multimers: bool = True
    grid_step: float = 2e-5
    grid_mz_max: float = 10000.0
    adduct_mode: str = "rplc"
    adduct_include: tuple[str, ...] = ()
    adduct_exclude: tuple[str, ...] = ()
    adduct_max_total: int = 3
    adduct_max_per_type: int = 2
    adduct_require_evidence: bool = True
    # accept a species seen at one charge state when resolved isotope spacing fixes the charge
    # (high-resolution instruments; see refine_candidate_masses)
    single_charge_isotopes: bool = False
    # charges are read from resolved isotope spacing (Orbitrap, FT-ICR): candidates may then be
    # displaced by peaks that inherit an envelope's ladder charge, and the gate lets them climb
    # to the envelope's centre from anywhere within its half-width
    isotope_charges: bool = False
    # a summed spectrum's charge states must co-elute over its scans (Pearson r of their
    # scan-by-scan traces at least this; 0 turns the test off); see deconv.coelution
    coelution_r: float = 0.5


def choose_class_auto(mass: float, candidates) -> str:
    """Pick a class by mass-defect agreement (mono fractional mass trend)."""
    # cheap proxy: compare observed-independent expected apex-mono offset; default to peptide
    return candidates[0] if candidates else "peptide"


def deconvolve_spectrum(
    spectrum: Spectrum,
    predictor: ChargePredictor,
    params: DeconvParams,
    instrument: InstrumentModel | None = None,
    trace: list | None = None,
) -> FrameResult:
    """Deconvolve one spectrum into neutral-mass components.

    ``trace``, when a list is passed, receives a snapshot of the candidate list after every
    stage (and every fit decision), so a diagnostic can tell at which stage a known species was
    lost. Passing nothing leaves the computation unchanged.
    """
    polarity = spectrum.polarity
    if instrument is None:
        instrument = InstrumentModel("tof", 30000.0)
    # the grid is restricted to the measured m/z range (empty grid costs inference time for
    # nothing); training frames go through this same function
    grid, observed, noise_sigma = spectrum_to_grid(
        spectrum.mz, spectrum.intensity, polarity, params.grid_step, params.grid_mz_max,
        spectrum.is_profile, instrument, frames=spectrum.meta.get("frames"))
    if observed.max() <= 0:
        return FrameResult(spectrum.rt, polarity, [], noise_sigma, 0.0, meta={"grid": grid})

    library = AdductLibrary.from_mode(params.adduct_mode, polarity,
                                      include=params.adduct_include, exclude=params.adduct_exclude,
                                      max_per_type=params.adduct_max_per_type,
                                      max_total=params.adduct_max_total)
    raw_order = np.argsort(spectrum.mz)
    raw_mz, raw_int = spectrum.mz[raw_order], spectrum.intensity[raw_order]
    # the scans a region's spectrum sums: their own point spacing, and the co-elution of charge
    # states (the co-elution tests abstain with fewer than four scans)
    frames = FrameStack.from_spectrum(spectrum, min_frames=1)
    # the class sets the isotope-envelope width used to locate each charge state's envelope;
    # with automatic selection the class is only known after detection, and the first
    # candidate class is the best prior
    gate_class = (params.compound_class if params.compound_class != "auto"
                  else (params.class_candidates[0] if params.class_candidates else "peptide"))

    def snap(stage: str, cands, it: int, **extra) -> None:
        if trace is not None:
            trace.append({"pass": it, "stage": stage,
                          "masses": [float(c.mass) for c in cands],
                          "charges": [sorted(int(z) for z in c.charges) for c in cands],
                          "scores": [float(c.score) for c in cands], **extra})

    def detect(signal: np.ndarray, it: int = 0):
        """Charge assignment, mass histogram and candidate cleanup for one spectrum."""
        prediction = predictor.predict_grid(grid, signal, noise_sigma)
        lm_axis, hist, charge_support = accumulate_mass_histogram(
            grid, signal, prediction, noise_sigma, snr=params.snr, prob_min=params.prob_min,
            mass_min=params.mass_range[0], mass_max=params.mass_range[1],
            z_min=params.charge_range[0], z_max=params.charge_range[1],
        )
        if trace is not None:
            nz = np.flatnonzero(hist)
            trace.append({"pass": it, "stage": "votes", "log_mass": lm_axis[nz].astype(np.float64),
                          "weight": hist[nz].astype(np.float64)})
        cands = pick_candidates(lm_axis, hist, charge_support,
                                min_charge_support=params.min_charge_support,
                                rel_height=params.min_relative_abundance,
                                single_charge_ok=params.single_charge_isotopes)
        snap("picked", cands, it)
        if not cands:
            return []
        cands = refine_candidate_masses(cands, raw_mz, raw_int, polarity,
                                        noise_sigma=noise_sigma, snr=params.snr,
                                        max_spread_ppm=params.max_mass_spread_ppm,
                                        compound_class=gate_class,
                                        resolution=float(instrument.resolving_power(1000.0)),
                                        instrument=instrument,
                                        profile=spectrum.is_profile is not False,
                                        single_charge_ok=params.single_charge_isotopes,
                                        envelope_move=params.isotope_charges,
                                        frames=frames, coelution_r=params.coelution_r)
        snap("gated", cands, it)
        cands = _dedupe_by_mass(cands)
        cands = suppress_harmonics(cands, suppress_multimers=params.suppress_multimers)
        snap("harmonics", cands, it)
        cands = dedupe_adduct_candidates(cands, library.avg_deltas())
        snap("adduct_dedupe", cands, it)
        if cands:
            floor = params.min_relative_abundance * max(c.score for c in cands)
            cands = [c for c in cands if c.score >= floor]
        cands = cands[: params.max_components]
        snap("detected", cands, it)
        return cands

    candidates = detect(observed)
    if not candidates:
        return FrameResult(spectrum.rt, polarity, [], noise_sigma, 1.0, meta={"grid": grid})

    # every adduct the mobile phase could make, evidenced or not: what a stronger species'
    # adduct ladder may put where a weaker one's adduct column would sit
    mode_library = library
    if params.adduct_require_evidence and library.adducts:
        names = evidenced_adducts(candidates, library, raw_mz, raw_int, polarity, noise_sigma,
                                  params, gate_class, instrument, spectrum.is_profile is not False,
                                  frames=frames)
        if trace is not None:
            trace.append({"pass": 0, "stage": "adducts", "kept": sorted(names),
                          "library": library.names()})
        library = AdductLibrary.from_mode(params.adduct_mode, polarity,
                                          include=params.adduct_include,
                                          exclude=tuple(params.adduct_exclude)
                                          + tuple(n for n in library.names() if n not in names),
                                          max_per_type=params.adduct_max_per_type,
                                          max_total=params.adduct_max_total)

    cls = params.compound_class
    auto_selected = cls == "auto"
    if auto_selected:
        cls = _auto_class(candidates, grid, observed, noise_sigma, instrument, library,
                          params.class_candidates)

    # Fit, then look again at what is left. Subtracting the strong envelopes uncovers the
    # low-abundance species that the first pass could not separate from their tails, which is
    # where impurities below one percent of the ion current live.
    components: list[Component] = []
    residual = observed.astype(np.float64).copy()
    residual_fraction = 1.0
    # masses the adduct walk moved candidates from: adduct forms of the species they joined
    walked: list[float] = []
    for iteration in range(max(1, params.refine_iterations)):
        if iteration > 0:
            found_masses = [e.mass for e in components]
            candidates = [c for c in detect(residual, iteration)
                          if all(abs(c.mass - e) > max(e * 1e-4, 0.5) for e in found_masses)]
            candidates = drop_harmonics_of(candidates, found_masses,
                                           suppress_multimers=params.suppress_multimers)
            snap("not_yet_found", candidates, iteration)
            if not candidates:
                break
        new, residual_fraction, residual = refine_frame(
            candidates, grid, observed, noise_sigma, instrument, cls, library,
            raw_mz=raw_mz, raw_int=raw_int,
            min_charge_support=params.min_charge_support,
            min_component_frac=params.min_relative_abundance,
            max_mass_spread_ppm=params.max_mass_spread_ppm,
            residual=residual, start_id=len(components),
            strongest_explained=max((c.intensity for c in components), default=0.0),
            trace=trace, trace_pass=iteration, frames=frames, coelution_r=params.coelution_r,
            known_masses=[c.mass for c in components], walked=walked,
        )
        if not new:
            break
        components.extend(new)
    components = drop_shared_evidence(components, library, polarity,
                                      resolution=float(instrument.resolving_power(1000.0)),
                                      trace=trace)
    components = drop_coincident_adducts(components, mode_library, trace=trace)
    snap("components", components, -1)

    for c in components:
        c.compound_class = cls
        if auto_selected:
            # at time-of-flight resolving power the isotope envelopes of different compound
            # classes are nearly identical, so an automatic choice is a hint, not a finding
            c.flags.append(f"compound class {cls} chosen automatically")
    return FrameResult(spectrum.rt, polarity, components, noise_sigma, residual_fraction,
                       meta={"resolution": instrument.resolution, "grid": grid,
                             "adducts": library.names()})


def _dedupe_by_mass(candidates, tol_ppm: float = 100.0, tol_da: float = 0.3):
    """Collapse candidates that refined onto the same mass, keeping the strongest."""
    kept = []
    for c in sorted(candidates, key=lambda x: -x.score):
        hit = None
        for k in kept:
            if abs(c.mass - k.mass) <= max(k.mass * tol_ppm * 1e-6, tol_da):
                hit = k
                break
        if hit is None:
            kept.append(c)
        else:
            hit.charges |= c.charges
            hit.intensity += c.intensity
            _merge_gate_verdicts(hit, c)
    return kept


def _merge_gate_verdicts(kept, dup) -> None:
    """Give ``kept`` the gate's verdicts on the charge states ``dup`` brings: within the
    by-mass tolerance the duplicate's gate measured the same windows, so a state it passed is
    this mass's own envelope and one it rejected holds another species' signal. Where the two
    disagree, ``kept``'s own verdict stands. Without this a rejected state merged in was read
    again by the overlap search and the adduct walk (an isotope comb of the wrong charge under
    one state pulled a 6752 Da species' templates to the far edge of their window)."""
    if dup.gate_charges is None and dup.gate_failed is None:
        return
    own_pass = set(kept.gate_charges or ())
    own_fail = set(kept.gate_failed or ())
    passed = own_pass | (set(dup.gate_charges or ()) - own_fail)
    failed = own_fail | (set(dup.gate_failed or ()) - own_pass)
    if kept.gate_charges is not None:
        kept.gate_charges = passed
    kept.gate_failed = failed - passed


def _auto_class(candidates, grid, observed, noise_sigma, instrument, library, class_candidates) -> str:
    """Choose the class whose templates best explain the strongest candidate (weighted R^2).

    This discriminates poorly below about 60 000 resolving power, where the isotope envelopes of
    a peptide, an oligonucleotide and a polymer of the same mass differ by less than the peak
    width. Name the class in the method whenever it is known; every bundled method does.
    """
    from .refine import _fit_candidate

    cand = max(candidates, key=lambda c: c.score)
    weights = fit_weights(observed, noise_sigma, grid.step, float(instrument.resolving_power(1000.0)))
    states = library.states()
    best_cls, best_score = class_candidates[0], -np.inf
    for cls in class_candidates:
        coeffs, used, explained = _fit_candidate(cand, observed, weights, grid, instrument, cls, states)
        if not used:
            continue
        model = np.zeros_like(observed)
        for t, coef in used:
            np.add.at(model, t.bins, t.values * coef)
        support = np.unique(np.concatenate([t.bins for t, _ in used]))
        r = observed[support] - model[support]
        ss_res = float(np.sum((r * weights[support]) ** 2))
        ss_tot = float(np.sum((observed[support] * weights[support]) ** 2)) + 1e-9
        score = 1.0 - ss_res / ss_tot
        if score > best_score:
            best_score, best_cls = score, cls
    return best_cls


def evidenced_adducts(candidates, library: AdductLibrary, raw_mz, raw_int, polarity: int,
                      noise_sigma: float, params: DeconvParams, compound_class: str,
                      instrument, profile: bool, n_species: int = 2, frames=None) -> set[str]:
    """Adduct types whose envelopes the spectrum actually shows next to its strongest species.

    An adduct library describes what a mobile phase *could* contribute. Fitting every
    combination of five types (about fifty adduct columns per charge state for ion-pairing
    methods) lets neighbouring species and adduct columns trade intensity, so abundances come
    out wrong even when every mass is right. A type is kept when, for one of the
    ``n_species`` strongest candidates, a candidate at M + delta with the same charges passes
    :func:`refine_candidate_masses` - at least two charge states agreeing on the mass, with the
    isotope comb checked where it is resolved, and the per-charge masses measured as detection
    measures them - at the candidate's average mass plus the adduct's average-mass shift (for
    K - H it is 0.13 Da more than the monoisotopic shift, half the tolerance). A region whose
    strongest species shows an adduct form that this test misses reports that form as a
    species of its own, while other regions attach the same adduct to their species.
    """
    from .decode import Candidate

    kept: set[str] = set()
    top = sorted(candidates, key=lambda c: -c.score)[:n_species]
    for adduct in library.adducts:
        for c in top:
            trial = Candidate(mass=c.mass + adduct.mass, score=c.score, charges=set(c.charges))
            ok = refine_candidate_masses([trial], raw_mz, raw_int, polarity,
                                         noise_sigma=noise_sigma, snr=params.snr,
                                         max_spread_ppm=params.max_mass_spread_ppm,
                                         compound_class=compound_class,
                                         resolution=float(instrument.resolving_power(1000.0)),
                                         instrument=instrument, profile=profile,
                                         tol_ppm=50.0, frames=frames,
                                         coelution_r=params.coelution_r,
                                         envelope_move=params.isotope_charges)
            if ok and abs(ok[0].mass - (c.mass + adduct.avg_mass)) <= max(2e-5 * c.mass, 0.3):
                kept.add(adduct.name)
                break
    return kept
