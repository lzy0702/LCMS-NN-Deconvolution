"""Processing method (parameters) with YAML persistence.

A method mirrors what a chromatography data system stores: instrument description,
deconvolution settings, adduct library, region discovery, per-signal integration events,
quantitation and reporting options.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from ..chem.instrument import InstrumentModel


@dataclass
class IntegrationEvent:
    """A timed integration event (Agilent-style)."""

    time: float
    event: str
    value: float | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.time, "event": self.event, "value": self.value}


@dataclass
class IntegrationSettings:
    slope_sensitivity: float = 1.0  # signal units per minute
    peak_width: float = 0.1  # minutes (expected width at half height)
    area_reject: float = 0.0
    height_reject: float = 0.0
    area_pct_reject: float = 0.0
    shoulders: str = "off"  # off | drop | tangent
    tangent_skim_mode: str = "standard"  # standard | exponential | old_exponential | new_exponential
    tail_skim_height_ratio: float = 0.0  # parent/child height ratio above which the child is skimmed (0 = off)
    front_skim_height_ratio: float = 0.0
    skim_valley_ratio: float = 20.0  # % of child height the valley must exceed to skim
    baseline_all_valleys: bool = False
    detect_negative_peaks: bool = False
    fixed_peak_width: bool = False
    max_area: float = 0.0
    max_height: float = 0.0
    area_unit: str = "signal*s"
    timed_events: list[IntegrationEvent] = field(default_factory=list)


@dataclass
class AdductSettings:
    mode: str = "rplc"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    max_per_type: int = 3
    max_total: int = 4
    max_types_mixed: int = 2
    min_fraction_report: float = 0.005


@dataclass
class DeconvSettings:
    mass_range: list[float] = field(default_factory=lambda: [200.0, 300000.0])
    charge_range: list[int] = field(default_factory=lambda: [1, 100])
    mz_range: list[float | None] = field(default_factory=lambda: [None, None])
    compound_class: str = "auto"
    class_candidates: list[str] = field(default_factory=lambda: ["peptide", "dna", "rna", "glycan", "peg"])
    mass_type: str = "auto"  # average | mono | auto (mono when isotopes are resolved)
    adducts: AdductSettings = field(default_factory=AdductSettings)
    min_relative_abundance: float = 1e-4
    snr_threshold: float = 3.0
    model: str | None = None  # path to ONNX model; None = bundled default
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    per_frame_nn: bool = False
    refine_iterations: int = 2
    nn_probability_min: int | float = 0.05
    min_charge_support: int = 2
    grid_step: float = 2e-5
    max_components: int = 200
    quantify_frames: bool = True


@dataclass
class RegionSettings:
    source: str = "tic"  # tic | uv | fixed
    window_min: float = 0.5  # used for fixed windows or when no peaks are found
    margin_min: float = 0.05
    min_frames: int = 2


@dataclass
class LinkSettings:
    mass_tolerance_ppm: float = 30.0
    mass_tolerance_da: float = 0.5
    min_frames: int = 2
    max_gap_frames: int = 2
    noise_peak_ratio: float = 3.0  # EIC max/median below which a species is chemical noise


@dataclass
class CalibrationSettings:
    mode: str = "none"  # none | external | response_factor
    levels: list[list[float]] = field(default_factory=list)  # [[amount, area], ...]
    weighting: str = "none"  # none | 1/x | 1/x2
    force_zero: bool = False
    response_factor: float = 1.0
    amount_unit: str = "mg/mL"


@dataclass
class QuantSettings:
    purity_signal: str = "uv"  # uv | tic | deic
    impurity_floor_pct: float = 0.01
    main_species: str = "largest"  # largest | mass:<value>
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    sample_amount: float | None = None


@dataclass
class SaturationSettings:
    esi_ratio_drop: float = 0.25
    flat_top_fraction: float = 0.2
    detector_level: float | None = None
    uv_delay_min: float | None = None  # None = estimate by cross-correlation


@dataclass
class ReportSettings:
    title: str | None = None
    plots: bool = True
    max_species_plots: int = 12


@dataclass
class AcquisitionSettings:
    polarity: str = "auto"  # auto | positive | negative | both
    uv_channel: str | None = None  # name of UV chromatogram to use; None = first


@dataclass
class Method:
    name: str = "default"
    description: str = ""
    instrument: InstrumentModel = field(default_factory=InstrumentModel)
    acquisition: AcquisitionSettings = field(default_factory=AcquisitionSettings)
    deconvolution: DeconvSettings = field(default_factory=DeconvSettings)
    regions: RegionSettings = field(default_factory=RegionSettings)
    linking: LinkSettings = field(default_factory=LinkSettings)
    integration: dict[str, IntegrationSettings] = field(
        default_factory=lambda: {
            "tic": IntegrationSettings(slope_sensitivity=0.0, peak_width=0.1),
            "uv": IntegrationSettings(slope_sensitivity=0.0, peak_width=0.1),
            "deic": IntegrationSettings(slope_sensitivity=0.0, peak_width=0.1),
        }
    )
    quant: QuantSettings = field(default_factory=QuantSettings)
    saturation: SaturationSettings = field(default_factory=SaturationSettings)
    report: ReportSettings = field(default_factory=ReportSettings)

    # ------------------------------------------------------------- (de)serialization
    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Method:
        return _from_plain(cls, d)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> Method:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(d)

    @classmethod
    def load(cls, path_or_name: str | Path | None) -> Method:
        """Load a method from a file path or from the bundled templates by name."""
        if path_or_name is None:
            return cls()
        p = Path(path_or_name)
        if p.exists():
            return cls.from_yaml(p)
        bundled = bundled_method_path(str(path_or_name))
        if bundled is not None:
            return cls.from_yaml(bundled)
        raise FileNotFoundError(f"Method not found: {path_or_name}")

    def integration_for(self, signal: str) -> IntegrationSettings:
        if signal in self.integration:
            return self.integration[signal]
        return self.integration.get("tic", IntegrationSettings())


def bundled_method_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


def bundled_method_path(name: str) -> Path | None:
    d = bundled_method_dir()
    for cand in (d / name, d / f"{name}.yaml", d / f"{name}.yml"):
        if cand.exists():
            return cand
    return None


def bundled_methods() -> list[str]:
    d = bundled_method_dir()
    return sorted(p.stem for p in d.glob("*.yaml")) if d.exists() else []


# ------------------------------------------------------------------ helpers
def _to_plain(obj: Any) -> Any:
    if isinstance(obj, InstrumentModel):
        return obj.to_dict()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _from_plain(cls: type, d: Any) -> Any:
    if d is None:
        return cls()
    if cls is InstrumentModel:
        return InstrumentModel.from_dict(d)
    if is_dataclass(cls):
        kwargs = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            kwargs[f.name] = _convert_field(cls, f, v)
        return cls(**kwargs)
    return d


def _convert_field(cls: type, f: dataclasses.Field, v: Any) -> Any:
    name = f.name
    if cls is Method:
        mapping = {
            "instrument": InstrumentModel,
            "acquisition": AcquisitionSettings,
            "deconvolution": DeconvSettings,
            "regions": RegionSettings,
            "linking": LinkSettings,
            "quant": QuantSettings,
            "saturation": SaturationSettings,
            "report": ReportSettings,
        }
        if name in mapping:
            return _from_plain(mapping[name], v)
        if name == "integration":
            return {k: _from_plain(IntegrationSettings, sv) for k, sv in (v or {}).items()}
    if cls is DeconvSettings and name == "adducts":
        return _from_plain(AdductSettings, v)
    if cls is QuantSettings and name == "calibration":
        return _from_plain(CalibrationSettings, v)
    if cls is IntegrationSettings and name == "timed_events":
        out = []
        for ev in v or []:
            if isinstance(ev, IntegrationEvent):
                out.append(ev)
            elif isinstance(ev, dict):
                out.append(IntegrationEvent(float(ev.get("time", 0.0)), str(ev.get("event", "")), ev.get("value")))
            elif isinstance(ev, (list, tuple)) and len(ev) >= 2:
                out.append(IntegrationEvent(float(ev[0]), str(ev[1]), ev[2] if len(ev) > 2 else None))
        return out
    return v
