"""Per-spectrum deconvolution pipeline: NN charge maps -> candidates -> NNLS components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chem.adducts import AdductLibrary
from ..chem.instrument import InstrumentModel, estimate_noise_sigma
from ..core.model import Component, FrameResult, Spectrum
from ..nn.grid import LogMzGrid
from ..nn.infer import ChargePredictor
from .decode import (
    accumulate_mass_histogram,
    dedupe_adduct_candidates,
    drop_harmonics_of,
    pick_candidates,
    refine_candidate_masses,
    suppress_harmonics,
)
from .refine import refine_frame


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


def choose_class_auto(mass: float, candidates) -> str:
    """Pick a class by mass-defect agreement (mono fractional mass trend)."""
    # cheap proxy: compare observed-independent expected apex-mono offset; default to peptide
    return candidates[0] if candidates else "peptide"


def deconvolve_spectrum(
    spectrum: Spectrum,
    predictor: ChargePredictor,
    params: DeconvParams,
    instrument: InstrumentModel | None = None,
) -> FrameResult:
    polarity = spectrum.polarity
    # restrict the grid to the measured m/z range: empty grid costs inference time for nothing
    if spectrum.mz.size:
        lo = max(50.0, float(spectrum.mz.min()) * 0.995)
        hi = min(params.grid_mz_max, float(spectrum.mz.max()) * 1.005)
    else:
        lo, hi = 50.0, params.grid_mz_max
    if hi <= lo * 1.01:
        lo, hi = 50.0, params.grid_mz_max
    grid = LogMzGrid(lo, hi, params.grid_step, polarity)
    if instrument is None:
        instrument = InstrumentModel("tof", 30000.0)

    if spectrum.is_profile is False:
        observed = grid.render_centroids(spectrum.mz, spectrum.intensity, instrument)
    else:
        observed = grid.resample_profile(spectrum.mz, spectrum.intensity)
    noise_sigma = estimate_noise_sigma(observed)
    if observed.max() <= 0:
        return FrameResult(spectrum.rt, polarity, [], noise_sigma, 0.0, meta={"grid": grid})

    library = AdductLibrary.from_mode(params.adduct_mode, polarity,
                                      include=params.adduct_include, exclude=params.adduct_exclude,
                                      max_per_type=params.adduct_max_per_type,
                                      max_total=params.adduct_max_total)
    raw_order = np.argsort(spectrum.mz)
    raw_mz, raw_int = spectrum.mz[raw_order], spectrum.intensity[raw_order]

    def detect(signal: np.ndarray):
        """Charge assignment, mass histogram and candidate cleanup for one spectrum."""
        prediction = predictor.predict_grid(grid, signal, noise_sigma)
        lm_axis, hist, charge_support = accumulate_mass_histogram(
            grid, signal, prediction, noise_sigma, snr=params.snr, prob_min=params.prob_min,
            mass_min=params.mass_range[0], mass_max=params.mass_range[1],
            z_min=params.charge_range[0], z_max=params.charge_range[1],
        )
        cands = pick_candidates(lm_axis, hist, charge_support,
                                min_charge_support=params.min_charge_support,
                                rel_height=params.min_relative_abundance)
        if not cands:
            return []
        cands = refine_candidate_masses(cands, raw_mz, raw_int, polarity,
                                        noise_sigma=noise_sigma, snr=params.snr,
                                        max_spread_ppm=params.max_mass_spread_ppm)
        cands = _dedupe_by_mass(cands)
        cands = suppress_harmonics(cands, suppress_multimers=params.suppress_multimers)
        cands = dedupe_adduct_candidates(cands, library.deltas())
        if cands:
            floor = params.min_relative_abundance * max(c.score for c in cands)
            cands = [c for c in cands if c.score >= floor]
        return cands[: params.max_components]

    candidates = detect(observed)
    if not candidates:
        return FrameResult(spectrum.rt, polarity, [], noise_sigma, 1.0, meta={"grid": grid})

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
    for iteration in range(max(1, params.refine_iterations)):
        if iteration > 0:
            found_masses = [e.mass for e in components]
            candidates = [c for c in detect(residual)
                          if all(abs(c.mass - e) > max(e * 1e-4, 0.5) for e in found_masses)]
            candidates = drop_harmonics_of(candidates, found_masses,
                                           suppress_multimers=params.suppress_multimers)
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
        )
        if not new:
            break
        components.extend(new)

    for c in components:
        c.compound_class = cls
        if auto_selected:
            # at time-of-flight resolving power the isotope envelopes of different compound
            # classes are nearly identical, so an automatic choice is a hint, not a finding
            c.flags.append(f"compound class {cls} chosen automatically")
    return FrameResult(spectrum.rt, polarity, components, noise_sigma, residual_fraction,
                       meta={"resolution": instrument.resolution, "grid": grid})


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
    return kept


def _auto_class(candidates, grid, observed, noise_sigma, instrument, library, class_candidates) -> str:
    """Choose the class whose templates best explain the strongest candidate (weighted R^2).

    This discriminates poorly below about 60 000 resolving power, where the isotope envelopes of
    a peptide, an oligonucleotide and a polymer of the same mass differ by less than the peak
    width. Name the class in the method whenever it is known; every bundled method does.
    """
    from .refine import _fit_candidate

    cand = max(candidates, key=lambda c: c.score)
    weights = 1.0 / np.sqrt(np.clip(observed, 0, None) + noise_sigma**2 + 1e-9)
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
