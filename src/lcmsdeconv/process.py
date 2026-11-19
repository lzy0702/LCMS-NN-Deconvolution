"""End-to-end processing of an LC-MS run: deconvolve, link, integrate, quantify."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chem.adducts import AdductLibrary
from .chrom.integrate import integrate_chromatogram
from .core.method import Method
from .core.model import Chromatogram, PeakTable, Run, Species
from .deconv.pipeline import DeconvParams, deconvolve_spectrum
from .features.annotate import annotate_species, impurity_table
from .features.link import (
    _is_flat,
    fold_adduct_forms,
    link_species,
    merge_adjacent_species,
    merge_species_across_polarity,
)
from .features.quantify import TemplateBank
from .features.regions import Region, find_regions
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

    def species_eic(self, species: Species) -> Chromatogram:
        return Chromatogram(species.time, species.intensity,
                            f"m {species.mass:.2f}", "deic", "counts")

    def summary(self) -> dict:
        return {
            "run": self.run_name,
            "method": self.method.name,
            "n_species": len(self.species),
            "n_regions": len(self.regions),
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
) -> ProcessResult:
    """Process one run end to end.

    ``trace``, when a list is passed, receives one record per deconvolved region holding the
    stage-by-stage candidate snapshots of :func:`deconvolve_spectrum` and the species linked
    from it. :mod:`lcmsdeconv.diagnostics` turns that into a per-species loss attribution.
    """
    method = method or Method()
    t_start = time.perf_counter()
    result = ProcessResult(run_name=run.name, method=method)
    instrument = method.instrument
    params = _deconv_params(method)

    if predictor is None:
        from .deconv.classical import make_predictor

        predictor = make_predictor(model_path or method.deconvolution.model,
                                   providers=list(method.deconvolution.providers),
                                   z_max=min(int(method.deconvolution.charge_range[1]), 60),
                                   instrument=instrument if _isotope_mode(method) else None)

    polarities = run.polarities
    if method.acquisition.polarity == "positive":
        polarities = [p for p in polarities if p > 0] or [1]
    elif method.acquisition.polarity == "negative":
        polarities = [p for p in polarities if p < 0] or [-1]

    uv = None
    uvs = run.uv_traces()
    if uvs:
        name = method.acquisition.uv_channel
        uv = next((c for c in uvs if c.name == name), uvs[0])
        result.chromatograms[uv.name] = uv

    all_species: list[Species] = []
    evidenced: dict[int, set[str]] = {}
    sid = 0
    for pol in polarities:
        tic = run.tic(pol)
        result.chromatograms[f"TIC{'+' if pol > 0 else '-'}"] = tic
        regions = find_regions(run, method.regions.source, pol, method.regions.window_min,
                               method.regions.margin_min, method.regions.min_frames,
                               integration=method.integration_for("tic"),
                               min_snr=getattr(method.regions, "min_snr", 10.0))
        result.regions.extend(regions)
        # a boundary two regions share is drawn at the valley between their bands
        rts = np.array([f.rt for f in run.frames(pol)])
        step = float(np.median(np.diff(rts))) if rts.size > 1 else 0.0
        for ri, region in enumerate(regions):
            shared = (ri > 0 and region.start - regions[ri - 1].end <= 0.5 * step,
                      ri + 1 < len(regions) and regions[ri + 1].start - region.end <= 0.5 * step)
            if progress:
                progress(f"deconvolving region {region.id} ({region.start:.2f}-{region.end:.2f} min)")
            summed = region.summed(run)
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
    if len(polarities) > 1:
        all_species = merge_species_across_polarity(all_species)
    all_species.sort(key=lambda s: -s.base_intensity)
    result.species = all_species

    # deconvolved EICs and their integration
    deic_settings = method.integration_for("deic")
    for s in all_species[: method.report.max_species_plots * 4]:
        ch = result.species_eic(s)
        result.chromatograms[ch.name] = ch
        table = integrate_chromatogram(ch, deic_settings, name=ch.name)
        if len(table):
            result.peak_tables[ch.name] = table

    # UV / TIC integration and purity
    if uv is not None:
        result.peak_tables[uv.name] = integrate_chromatogram(uv, method.integration_for("uv"), uv.name)
    for key, ch in list(result.chromatograms.items()):
        if ch.kind == "tic":
            result.peak_tables[key] = integrate_chromatogram(ch, method.integration_for("tic"), key)

    purity_signal = method.quant.purity_signal
    purity_table = None
    if purity_signal == "uv" and uv is not None and uv.name in result.peak_tables:
        purity_table = result.peak_tables[uv.name]
    elif purity_signal in result.peak_tables:
        purity_table = result.peak_tables[purity_signal]
    else:
        tics = [v for k, v in result.peak_tables.items() if k.startswith("TIC")]
        purity_table = tics[0] if tics else None
    if purity_table is not None:
        result.purity = area_percent_purity(purity_table).to_dict()
        result.purity["ms"] = ms_purity(all_species) if all_species else {}

    if all_species:
        annotate_species(all_species)
        result.impurities = impurity_table(all_species, floor_pct=method.quant.impurity_floor_pct)

    # potency
    cal_cfg = method.quant.calibration
    if cal_cfg.mode == "external" and cal_cfg.levels:
        cal = fit_calibration(cal_cfg.levels, cal_cfg.weighting, cal_cfg.force_zero, cal_cfg.amount_unit)
    elif cal_cfg.mode == "response_factor":
        cal = Calibration("response_factor", cal_cfg.response_factor, 0.0, 0.0,
                          amount_unit=cal_cfg.amount_unit)
    else:
        cal = Calibration("none")
    if cal.mode != "none" and purity_table is not None and len(purity_table):
        main = max(purity_table.peaks, key=lambda p: p.area)
        result.potency = potency(main.area, cal, method.quant.sample_amount)

    # saturation
    peaks_for_sat = []
    for key, tab in result.peak_tables.items():
        if key.startswith("TIC"):
            peaks_for_sat = tab.peaks
            break
    result.saturation = analyze_saturation(run, peaks_for_sat, method.saturation,
                                           polarities[0] if len(polarities) == 1 else None, uv)
    if result.saturation and result.saturation.messages:
        result.warnings.extend(result.saturation.messages)

    result.timings["total_s"] = time.perf_counter() - t_start
    return result


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
