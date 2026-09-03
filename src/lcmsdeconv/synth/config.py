"""Configuration for the synthetic data generator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .compounds import DEFAULT_CLASSES, ClassConfig


@dataclass
class SynthConfig:
    classes: list[ClassConfig] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    polarity: int = 1
    mode: str = "rplc"  # chromatography/adduct mode
    esi_mode: str = "denatured"  # denatured | native
    # scene composition
    n_main_range: tuple[int, int] = (1, 2)
    max_impurities: int = 12
    impurity_abundance: tuple[float, float] = (1e-4, 1e-1)
    contaminant_prob: float = 0.2
    # intensity
    base_intensity_range: tuple[float, float] = (1e4, 5e6)
    # instrument
    instrument_kind: str = "tof"
    resolution_range: tuple[float, float] = (8000.0, 60000.0)
    shape_eta_range: tuple[float, float] = (0.0, 0.4)
    calibration_ppm: float = 10.0
    saturation_prob: float = 0.1
    # grid
    grid_step: float = 2e-5
    grid_mz_max: float = 10000.0
    crop_size: int = 32768
    z_max: int = 100
    # adducts
    adduct_max_lambda: float = 0.15
    adduct_max_total: int = 4
    adduct_max_per_type: int = 3


def orbitrap_config(**kw) -> SynthConfig:
    c = SynthConfig(instrument_kind="orbitrap", resolution_range=(30000.0, 240000.0), **kw)
    return c
