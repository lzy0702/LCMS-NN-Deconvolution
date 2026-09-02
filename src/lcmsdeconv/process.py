"""End-to-end processing of an LC-MS run: deconvolve, link, integrate, quantify."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .chem.adducts import AdductLibrary
from .chrom.integrate import integrate_chromatogram
from .core.method import Method
from .core.model import Chromatogram, PeakTable, Run, Species
from .deconv.pipeline import DeconvParams, deconvolve_spectrum
from .features.annotate import annotate_species, impurity_table
from .features.link import link_species, merge_species_across_polarity
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
        min_relative_abundance=d.min_relative_abundance,
        refine_iterations=d.refine_iterations,
        max_components=d.max_components,
        grid_step=d.grid_step,
        adduct_mode=d.adducts.mode,
        adduct_include=tuple(d.adducts.include),
        adduct_exclude=tuple(d.adducts.exclude),
        adduct_max_total=d.adducts.max_total,
        adduct_max_per_type=d.adducts.max_per_type,
    )


def process_run(
    run: Run,
    method: Method | None = None,
    predictor: ChargePredictor | None = None,
    model_path: str | None = None,
    progress=None,
) -> ProcessResult:
    """Process one run end to end."""
    method = method or Method()
    t_start = time.perf_counter()
    result = ProcessResult(run_name=run.name, method=method)
    instrument = method.instrument
    params = _deconv_params(method)

    if predictor is None:
        from .deconv.classical import make_predictor

        predictor = make_predictor(model_path or method.deconvolution.model,
                                   providers=list(method.deconvolution.providers),
                                   z_max=min(int(method.deconvolution.charge_range[1]), 60))

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
    sid = 0
    for pol in polarities:
        tic = run.tic(pol)
        result.chromatograms[f"TIC{'+' if pol > 0 else '-'}"] = tic
        regions = find_regions(run, method.regions.source, pol, method.regions.window_min,
                               method.regions.margin_min, method.regions.min_frames,
                               integration=method.integration_for("tic"))
        result.regions.extend(regions)
        for region in regions:
            if progress:
                progress(f"deconvolving region {region.id} ({region.start:.2f}-{region.end:.2f} min)")
            summed = region.summed(run)
            fr = deconvolve_spectrum(summed, predictor, params, instrument)
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
                              region_id=region.id, start_id=sid)
            sid += len(sp) + 1
            all_species.extend(sp)

    if len(polarities) > 1:
        all_species = merge_species_across_polarity(all_species)
    all_species.sort(key=lambda s: -s.total_intensity)
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
