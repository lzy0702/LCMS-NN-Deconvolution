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
    # training-frame rendering. "profile" follows the inference path exactly: raw profile
    # scans on the instrument's own m/z axis, shot and electronic noise, vendor-style
    # thresholding, a sum over several scans (as a region's summed spectrum is), resampling to
    # the grid, and the noise level estimated from the result by the same function inference
    # uses. "grid" renders straight onto the grid with the generator's own noise level, as
    # earlier versions did; features computed that way do not match what inference sees.
    render_path: str = "profile"
    sum_frames_range: tuple[int, int] = (1, 8)
    gain_range: tuple[float, float] = (1.0, 50.0)
    electronic_sigma_range: tuple[float, float] = (0.5, 5.0)
    threshold_sigmas_range: tuple[float, float] = (0.0, 3.0)
    background_ions_range: tuple[int, int] = (0, 40)
    #: per-charge comb evidence channels appended to the four base features (0 = none)
    evidence_channels: int = 0
    # adducts: by default the load is drawn per ion, as the mean number of each adduct type an
    # ion at the apex charge carries (0..adduct_max_per_ion). Setting adduct_max_lambda instead
    # restores the older per-charge propensity, under which the load grows with charge; it is
    # kept for reproducing old benchmarks and as a heavy-adduct stress test.
    adduct_max_per_ion: float = 0.3
    adduct_max_lambda: float | None = None
    adduct_max_total: int = 4
    adduct_max_per_type: int = 3
    # impurities share their parent's charge envelope (apex scaled by mass); False draws every
    # impurity's envelope independently, as older versions did
    impurity_charge_from_parent: bool = True


def orbitrap_config(**kw) -> SynthConfig:
    c = SynthConfig(instrument_kind="orbitrap", resolution_range=(30000.0, 240000.0), **kw)
    return c
