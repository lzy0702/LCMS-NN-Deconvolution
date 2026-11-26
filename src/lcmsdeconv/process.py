"""End-to-end processing of an LC-MS run: deconvolve, link, integrate, quantify."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chem.adducts import AdductLibrary
from .chrom.integrate import integrate_chromatogram
from .core.method import (
    Method,
    analysis_window,
    rt_window_setting,
    rt_window_text,
    window_bounds,
)
from .core.model import Chromatogram, PeakTable, Run, Species
from .deconv.pipeline import DeconvParams, deconvolve_spectrum
from .features.annotate import annotate_species, impurity_table
from .features.link import (
    _is_flat,
    fold_adduct_forms,
    link_species,
    merge_adjacent_species,
    merge_boundary_pieces,
    merge_species_across_polarity,
    set_aside_later_elutions,
)
from .features.quantify import TemplateBank
from .features.regions import Region, find_regions, source_noise
from .io.lockmass import lock_mass_corrected
from .nn.grid import LogMzGrid
from .nn.infer import ChargePredictor  # noqa: F401  (public re-export)
from .quant.purity import (
    Calibration,
    area_percent_purity,
    fit_calibration,
    ms_purity,
    potency,
)
from .quant.saturation import SaturationReport, analyze_saturation


@dataclass
class ProcessResult:
    run_name: str
    method: Method
    species: list[Species] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    peak_tables: dict[str, PeakTable] = field(default_factory=dict)
    chromatograms: dict[str, Chromatogram] = field(default_factory=dict)
    impurities: list[dict] = field(default_factory=list)
    purity: dict[str, Any] = field(default_factory=dict)
    potency: dict[str, Any] = field(default_factory=dict)
    saturation: SaturationReport | None = None
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    #: the analysis window as the method sets it ([start, end] minutes, None for an open end),
    #: None when the whole run was analysed
    rt_window: list[float | None] | None = None
    #: the lock-mass correction applied (:func:`lcmsdeconv.io.lockmass.lock_mass_corrected`)
    lock_mass: dict | None = None
    #: how many name-alias pairs the run and the method were aliased with
    #: (:mod:`lcmsdeconv.aliases`); recorded in the summary when there were any, never their text
    alias_pairs: int = 0

    def species_eic(self, species: Species) -> Chromatogram:
        return Chromatogram(species.time, species.intensity,
                            f"m {species.mass:.2f}", "deic", "counts")

    def summary(self) -> dict:
        head = {"run": self.run_name, "method": self.method.name}
        if self.alias_pairs:
            head["alias_pairs"] = self.alias_pairs
        return head | {
            "n_species": len(self.species),
            "n_regions": len(self.regions),
            "rt_window": self.rt_window,
            "lock_mass": self.lock_mass,
            "species": [s.to_dict() for s in self.species],
            "impurities": self.impurities,
            "purity": self.purity,
            "potency": self.potency,
            "saturation": self.saturation.to_dict() if self.saturation else None,
            "peak_tables": {k: [p.to_dict() for p in v.peaks] for k, v in self.peak_tables.items()},
            "warnings": self.warnings,
            "timings": self.timings,
        }


def _isotope_mode(method: Method) -> bool:
    """Whether the charge estimator reads charges from resolved isotope spacing.

    ``deconvolution.isotope_charges`` decides; unset, it is on for Orbitrap and FT-ICR
    instruments.
    """
    iso = getattr(method.deconvolution, "isotope_charges", None)
    if iso is None:
        inst = method.instrument
        iso = inst is not None and inst.kind in ("orbitrap", "fticr")
    return bool(iso)


def _deconv_params(method: Method) -> DeconvParams:
    d = method.deconvolution
    return DeconvParams(
        compound_class=d.compound_class,
        class_candidates=tuple(d.class_candidates),
        mass_range=(float(d.mass_range[0]), float(d.mass_range[1])),
        charge_range=(int(d.charge_range[0]), int(d.charge_range[1])),
        snr=d.snr_threshold,
        prob_min=float(d.nn_probability_min),
        min_charge_support=d.min_charge_support,
        suppress_multimers=d.suppress_multimers,
        min_relative_abundance=d.min_relative_abundance,
        refine_iterations=d.refine_iterations,
        max_components=d.max_components,
        grid_step=d.grid_step,
        adduct_mode=d.adducts.mode,
        adduct_include=tuple(d.adducts.include),
        adduct_exclude=tuple(d.adducts.exclude),
        adduct_max_total=d.adducts.max_total,
        adduct_max_per_type=d.adducts.max_per_type,
        adduct_require_evidence=getattr(d.adducts, "require_evidence", True),
        single_charge_isotopes=bool(getattr(d, "single_charge", False)),
        singly_charged_species=bool(getattr(d, "singly_charged_species", True)),
        isotope_charges=_isotope_mode(method),
        coelution_r=float(getattr(d, "coelution_r", 0.5)),
    )


def process_run(
    run: Run,
    method: Method | None = None,
    predictor: ChargePredictor | None = None,
    model_path: str | None = None,
    progress=None,
    trace: list | None = None,
    aliases=None,
) -> ProcessResult:
    """Process one run end to end.

    ``trace``, when a list is passed, receives one record per deconvolved region holding the
    stage-by-stage candidate snapshots of :func:`deconvolve_spectrum` and the species linked
    from it. :mod:`lcmsdeconv.diagnostics` turns that into a per-species loss attribution.

    ``aliases`` (:class:`lcmsdeconv.aliases.Aliases`), where given with pairs, makes the result
    show NEW for OLD in every name: the run and the method are processed as aliased copies
    (:meth:`~lcmsdeconv.aliases.Aliases.alias_run`, and
    :meth:`~lcmsdeconv.aliases.Aliases.alias_method` with ``run``, so that the method's trace
    references name the same traces as before; ``run`` and ``method`` themselves are left as
    they were), and ``result.alias_pairs`` counts the pairs. A run and method aliased already
    are passed without it (aliasing twice can differ from aliasing once).

    With ``acquisition.lock_mass_mz`` set, the scans' m/z are first corrected against the run's
    lockspray scans (:func:`lcmsdeconv.io.lockmass.lock_mass_corrected`; ``run`` itself is left
    as it was, and a run corrected already is not corrected again). With ``regions.rt_window``
    set, everything is then analysed within that window only: regions, TIC and UV integration,
    saturation, purity and content; ``result.chromatograms`` keep the whole traces for display.
    The regions' relative floors are taken within the window, their noise floor on the whole
    run's trace.
    """
    method = method or Method()
    t_start = time.perf_counter()
    if aliases:  # the method with the run as read: its trace references follow the traces
        run, method = aliases.alias_run(run), aliases.alias_method(method, run)
    result = ProcessResult(run_name=run.name, method=method,
                           alias_pairs=len(aliases) if aliases else 0)
    run = _lock_mass(run, method, result)
    window = analysis_window(method)
    result.rt_window = None if window is None else rt_window_setting(method.regions.rt_window)
    # what is analysed: the scans and traces within the analysis window
    view = run if window is None else run.windowed(*window)
    if window is not None and not view.spectra:
        span = (f"{min(s.rt for s in run.spectra):.2f}-{max(s.rt for s in run.spectra):.2f} min"
                if run.spectra else "no time")
        result.warnings.append(f"the analysis window ({rt_window_text(result.rt_window)}) "
                               f"holds no MS scans; the run's scans span {span}")
    instrument = method.instrument
    params = _deconv_params(method)

    if predictor is None:
        from .deconv.classical import make_predictor

        predictor = make_predictor(model_path or method.deconvolution.model,
                                   providers=list(method.deconvolution.providers),
                                   z_max=min(int(method.deconvolution.charge_range[1]), 60),
                                   instrument=instrument if _isotope_mode(method) else None)
    if getattr(predictor, "iso_channels", False) and getattr(predictor, "instrument", None) is None:
        # a network's isotope-charge input channels need the instrument whether or not the comb
        # reads isotope charges: its kind decides whether they are read (as in training), its
        # resolving power what they read
        predictor.instrument = instrument

    polarities = run.polarities
    if method.acquisition.polarity == "positive":
        polarities = [p for p in polarities if p > 0] or [1]
    elif method.acquisition.polarity == "negative":
        polarities = [p for p in polarities if p < 0] or [-1]

    # the whole traces are kept for display; those within the window are integrated
    uv = uv_in = None
    uvs = run.uv_traces()
    if uvs:
        name = method.acquisition.uv_channel
        uv = next((c for c in uvs if c.name == name), uvs[0])
        result.chromatograms[uv.name] = uv
        uv_in = uv if window is None else uv.sliced(*window)
    tics_in: dict[str, Chromatogram] = {}

    all_species: list[Species] = []
    evidenced: dict[int, set[str]] = {}
    # the boundaries two regions share (polarity, time) and the scan spacing, by polarity
    boundaries: list[tuple[int, float]] = []
    steps: dict[int, float] = {}
    sid = 0
    for pol in polarities:
        key = f"TIC{'+' if pol > 0 else '-'}"
        result.chromatograms[key] = run.tic(pol)
        tics_in[key] = result.chromatograms[key] if window is None else view.tic(pol)
        # within a window the relative floors are the window's, but the noise is the whole run's:
        # a window crowded with peaks reads a higher noise on its own trace
        noise = None if window is None else source_noise(run, method.regions.source, pol)
        regions = find_regions(view, method.regions.source, pol, method.regions.window_min,
                               method.regions.margin_min, method.regions.min_frames,
                               integration=method.integration_for("tic"),
                               min_snr=getattr(method.regions, "min_snr", 10.0),
                               min_rel_height=getattr(method.regions, "min_rel_height", 0.002),
                               noise=noise)
        if window is not None:
            # a region's scans lie within the window; its margin may not
            for r in regions:
                r.start, r.end = max(r.start, window[0]), min(r.end, window[1])
        result.regions.extend(regions)
        # a boundary two regions share is drawn at the valley between their bands
        rts = np.array([f.rt for f in view.frames(pol)])
        step = float(np.median(np.diff(rts))) if rts.size > 1 else 0.0
        steps[pol] = step
        for ri, region in enumerate(regions):
            shared = (ri > 0 and region.start - regions[ri - 1].end <= 0.5 * step,
                      ri + 1 < len(regions) and regions[ri + 1].start - region.end <= 0.5 * step)
            if shared[0]:
                boundaries.append((pol, float(region.start)))
            if progress:
                progress(f"deconvolving region {region.id} ({region.start:.2f}-{region.end:.2f} min)")
            summed = region.summed(view)
            region_trace: list | None = [] if trace is not None else None
            fr = deconvolve_spectrum(summed, predictor, params, instrument, trace=region_trace)
            if trace is not None:
                trace.append({"region": region.id, "start": region.start, "end": region.end,
                              "polarity": pol, "stages": region_trace, "species": []})
            evidenced.setdefault(pol, set()).update(fr.meta.get("adducts", ()))
            if not fr.components:
                continue
            grid = fr.meta.get("grid") or LogMzGrid(50.0, params.grid_mz_max, params.grid_step, pol)
            library = AdductLibrary.from_mode(params.adduct_mode, pol,
                                              include=params.adduct_include,
                                              exclude=params.adduct_exclude,
                                              max_per_type=params.adduct_max_per_type,
                                              max_total=params.adduct_max_total)
            bank = TemplateBank(fr.components, grid, instrument, library)
            frame_results = []
            if method.deconvolution.quantify_frames:
                for spec in region.frames:
                    frame_results.append(bank.quantify(spec))
            else:
                frame_results = [fr]
            sp = link_species(frame_results, method.linking.mass_tolerance_ppm,
                              method.linking.mass_tolerance_da, method.linking.min_frames,
                              method.linking.max_gap_frames, method.linking.noise_peak_ratio,
                              region_id=region.id, start_id=sid,
                              flat_test=not method.deconvolution.quantify_frames,
                              shared_edges=shared)
            n_linked = len(sp)
            if method.deconvolution.quantify_frames and sp:
                # the context frames only judge flatness: the whole run's, beyond the window too
                sp = _drop_flat_species(sp, bank, run, region, pol, method)
            # ids are numbered over every linked track: count the dropped ones too
            sid += n_linked + 1
            all_species.extend(sp)
            if trace is not None:
                trace[-1]["species"] = [{"mass": s.mass, "rt_apex": s.rt_apex} for s in sp]

    # a species' adduct forms count with it however each region's decoder reported them: cation
    # adducts (sodium, potassium, ammonium: every electrospray run has them), and the mobile
    # phase's amines, acids and solvents where some region of the run evidenced them (their
    # clusters lie within a dalton of an oligonucleotide's n+1). Region pieces are folded before they
    # are merged: a form reported apart in one region is judged against its parent's piece in
    # that region, and the merge then keeps one reading per scan. A weak form's average mass is
    # read to a few tenths of a dalton (potassium forms 0.4-0.5 Da light on the IP-RPLC runs),
    # and from envelopes of width M / R to a few such widths
    fold_libraries = {}
    for pol in polarities:
        lib = AdductLibrary.from_mode(params.adduct_mode, pol, include=params.adduct_include,
                                      exclude=params.adduct_exclude,
                                      max_per_type=params.adduct_max_per_type,
                                      max_total=params.adduct_max_total)
        if params.adduct_require_evidence:
            lib = AdductLibrary([a for a in lib.adducts if a.category == "cation"
                                 or a.name in evidenced.get(pol, set())],
                                lib.max_per_type, lib.max_total)
        fold_libraries[pol] = lib
    all_species = fold_adduct_forms(
        all_species, fold_libraries, tolerance_da=0.6,
        tolerance_ppm=2.5e6 / max(float(instrument.resolving_power(1000.0)), 1.0))
    all_species = merge_adjacent_species(all_species, method.linking.mass_tolerance_ppm,
                                         method.linking.mass_tolerance_da)
    # a piece of a band that its region read up to a dalton off, and, where the method sets them
    # aside, the main species found again later in the run (its tail beyond its band, the column
    # wash): judged by their isotopes, read in the scans themselves
    same = _same_molecule(view, instrument, params.compound_class)
    all_species = merge_boundary_pieces(all_species, boundaries, same, steps,
                                        method.linking.mass_tolerance_ppm,
                                        method.linking.mass_tolerance_da)
    if getattr(method.linking, "set_aside_later_elutions", False):
        all_species = set_aside_later_elutions(all_species, same, steps,
                                               method.linking.mass_tolerance_ppm,
                                               method.linking.mass_tolerance_da)
    if len(polarities) > 1:
        all_species = merge_species_across_polarity(all_species)
    all_species.sort(key=lambda s: -s.base_intensity)
    result.species = all_species

    # co-eluting adduct ladders: a species 1-4 adducts of the method's library (its mode's adducts
    # and 'include') above a stronger species it co-elutes with is that species' adduct form, not
    # an impurity; the impurity table and MS purity count it with its parent
    from .features.annotate import mark_adduct_forms

    mark_adduct_forms(all_species, AdductLibrary.from_mode(params.adduct_mode, 0,
                                                           include=params.adduct_include,
                                                           exclude=params.adduct_exclude))

    # deconvolved EICs and their integration
    deic_settings = method.integration_for("deic")
    for s in all_species[: method.report.max_species_plots * 4]:
        ch = result.species_eic(s)
        result.chromatograms[ch.name] = ch
        table = integrate_chromatogram(ch, deic_settings, name=ch.name)
        if len(table):
            result.peak_tables[ch.name] = table

    # UV / TIC integration (within the analysis window) and purity
    if uv is not None:
        result.peak_tables[uv.name] = integrate_chromatogram(uv_in, method.integration_for("uv"),
                                                             uv.name)
    for key, ch in tics_in.items():
        result.peak_tables[key] = integrate_chromatogram(ch, method.integration_for("tic"), key)

    # area-% purity and potency from the purity signal's peak table
    _purity_and_potency(result, method)

    if all_species:
        annotate_species(all_species)
        result.impurities = impurity_table(all_species, floor_pct=method.quant.impurity_floor_pct)

    # saturation
    peaks_for_sat = []
    for key, tab in result.peak_tables.items():
        if key.startswith("TIC"):
            peaks_for_sat = tab.peaks
            break
    result.saturation = analyze_saturation(view, peaks_for_sat, method.saturation,
                                           polarities[0] if len(polarities) == 1 else None, uv_in)
    if result.saturation and result.saturation.messages:
        result.warnings.extend(result.saturation.messages)

    result.timings["total_s"] = time.perf_counter() - t_start
    return result


def _lock_mass(run: Run, method: Method, result: ProcessResult) -> Run:
    """The run whose scans are analysed: corrected against its lockspray scans where the method
    sets ``acquisition.lock_mass_mz`` (once: a run corrected already keeps its correction), as
    read otherwise. The correction applied is recorded in ``result.lock_mass`` and anything that
    stopped it in ``result.warnings``."""
    lock = getattr(method.acquisition, "lock_mass_mz", None)
    if lock is None:
        result.lock_mass = run.meta.get("lock_mass")
        return run
    # the ratio's time is weighted over the scans analysed (the window), not a front or wash
    run, result.lock_mass, warnings = lock_mass_corrected(run, float(lock),
                                                          rt_range=analysis_window(method))
    result.warnings.extend(warnings)
    return run


def _purity_table_name(result: ProcessResult, method: Method) -> str | None:
    """The peak table the area-% purity and the potency are read from: the UV trace's where
    ``quant.purity_signal`` is uv and the run has one, the table so named where there is one,
    else the first TIC's; None where there is none."""
    signal = method.quant.purity_signal
    uv = next((k for k, ch in result.chromatograms.items() if ch.kind == "uv"), None)
    if signal == "uv" and uv is not None and uv in result.peak_tables:
        return uv
    if signal in result.peak_tables:
        return signal
    return next((k for k in result.peak_tables if k.startswith("TIC")), None)


def _calibration(method: Method) -> Calibration:
    cal_cfg = method.quant.calibration
    if cal_cfg.mode == "external" and cal_cfg.levels:
        return fit_calibration(cal_cfg.levels, cal_cfg.weighting, cal_cfg.force_zero,
                               cal_cfg.amount_unit)
    if cal_cfg.mode == "response_factor":
        return Calibration("response_factor", cal_cfg.response_factor, 0.0, 0.0,
                           amount_unit=cal_cfg.amount_unit)
    return Calibration("none")


def _purity_and_potency(result: ProcessResult, method: Method) -> None:
    """``result.purity`` (area-% purity of the purity signal's peak table, with the MS purity of
    ``result.species``) and ``result.potency`` (from the table's largest peak, where the method
    calibrates), as the result's peak tables now stand."""
    name = _purity_table_name(result, method)
    table = result.peak_tables.get(name) if name is not None else None
    if table is not None:
        result.purity = area_percent_purity(table).to_dict()
        result.purity["ms"] = ms_purity(result.species) if result.species else {}
    cal = _calibration(method)
    result.potency = {}
    if cal.mode != "none" and table is not None and len(table):
        main = max(table.peaks, key=lambda p: p.area)
        result.potency = potency(main.area, cal, method.quant.sample_amount)


def integration_kind(chrom: Chromatogram) -> str:
    """The integration settings a trace is integrated with: "uv", "deic" or "tic"."""
    return chrom.kind if chrom.kind in ("uv", "deic") else "tic"


def reintegrate(result: ProcessResult, name: str, method: Method | None = None
                ) -> PeakTable | None:
    """Integrate the trace ``name`` of ``result`` again (the GUI's re-integrate button), with
    ``method``'s integration settings for its kind (default: the result's own method).

    As :func:`process_run` does, a TIC or UV trace is integrated within the analysis window the
    result was processed with (``result.rt_window``; ``result.chromatograms`` hold the whole
    traces, for display), a deconvolved trace whole. The table replaces the result's, and where
    it is the purity signal's, the area-% purity and the potency are read from it again (with
    the quantitation settings the result was processed with), so the report agrees with it.
    Returns the table; None where the result holds no trace ``name``.
    """
    chrom = result.chromatograms.get(name)
    if chrom is None:
        return None
    kind = integration_kind(chrom)
    bounds = window_bounds(result.rt_window)
    if bounds is not None and kind in ("tic", "uv"):
        chrom = chrom.sliced(*bounds)
    table = integrate_chromatogram(chrom, (method or result.method).integration_for(kind), name)
    result.peak_tables[name] = table
    if name == _purity_table_name(result, result.method):
        _purity_and_potency(result, result.method)
    return table



def _same_molecule(run: Run, instrument, compound_class: str, min_margin: float = 0.02,
                   max_charges: int = 3):
    """``same(a, band_a, b, band_b)``: whether species ``a``, read in the scans of the time band
    ``band_a``, and ``b``, read in those of ``band_b``, are one molecule by their resolved
    isotopes: the class's pattern fits both best on one tooth of the isotope comb
    (:func:`lcmsdeconv.deconv.refine.tooth_position`, over ``b``'s strongest charge states),
    each better than one tooth away by ``min_margin``: then the average mass ``b``'s band puts the
    molecule at, else None. Where the isotopes are not resolved, a band holds no signal or a fit
    is no better than its neighbouring tooth, the answer is no."""
    from .deconv.refine import tooth_position

    def same(a: Species, band_a, b: Species, band_b) -> float | None:
        if instrument is None:
            return None
        zs = [z for z, v in sorted(b.charges.items(), key=lambda kv: -kv[1]) if v > 0][:max_charges]
        cls = b.compound_class or a.compound_class or compound_class
        if not zs or not cls or cls == "auto":
            return None
        m = 0.5 * (a.mass + b.mass)
        ta = tooth_position(m, zs, run.frames(a.polarity, band_a), instrument, cls)
        tb = tooth_position(m, zs, run.frames(b.polarity, band_b), instrument, cls)
        if ta is None or tb is None or min(ta[1], tb[1]) < min_margin:
            return None
        return tb[0] if abs(ta[0] - tb[0]) <= 0.5 else None

    return same


def _drop_flat_species(species, bank, run, region, polarity, method, context_min: float = 0.3):
    """Drop species whose trace, seen with some chromatographic context, does not fall away.

    Chemical background stays up across the whole run; a real species elutes. Judging that
    inside one region fails when a broad band is split into several regions, because the trace
    then stays up at both edges of each. The trace is therefore extended by one region width
    (at least ``context_min``) of quantified frames on either side before testing.
    """
    width = max(region.end - region.start, context_min)
    before = run.frames(polarity, (region.start - width, region.start - 1e-9))
    after = run.frames(polarity, (region.end + 1e-9, region.end + width))
    if not before and not after:
        return species
    ctx_before = [bank.quantify(f) for f in before]
    ctx_after = [bank.quantify(f) for f in after]
    tol_ppm, tol_da = method.linking.mass_tolerance_ppm, method.linking.mass_tolerance_da

    def amount(fr, mass):
        return sum(c.intensity for c in fr.components
                   if abs(c.mass - mass) <= max(mass * tol_ppm * 1e-6, tol_da))

    kept = []
    for s in species:
        trace = np.concatenate([[amount(fr, s.mass) for fr in ctx_before], s.intensity,
                                [amount(fr, s.mass) for fr in ctx_after]])
        if trace.size > 6 * method.linking.min_frames and _is_flat(trace, method.linking.noise_peak_ratio):
            continue
        kept.append(s)
    return kept
