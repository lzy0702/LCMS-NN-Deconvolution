"""Configuration for the synthetic data generator."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

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
    #: points written on either side of every point above the vendor threshold. The real Thermo
    #: runs hold no segment shorter than five points, and most are five or six: the instrument
    #: writes a point either side of each run above its threshold, but its noise is correlated
    #: over the line shape, so such runs are rarely shorter than three points. For the white
    #: noise drawn here two points either side give that five-point minimum, three gave seven.
    #: (Whole synthetic runs, ``synth.chromatography._threshold_profile``, keep three: the
    #: benchmarks were measured on them.) A setting, so that the resume fingerprint covers it.
    vendor_pad: int = 2
    background_ions_range: tuple[int, int] = (0, 40)
    #: singly charged ions (chemical background, small molecules): "keep"; "drop" - generate
    #: none, so the network never sees one; "ignore" - generate them, and give their bins no
    #: weight in the loss (``ChargeLoss(ignore_charges=(1,))``), so z = 1 is neither rewarded
    #: nor penalised while the input stays realistic
    singly_charged: str = "keep"
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
    #: the m/z range a profile frame is measured over: "sticks", the frame's own ions (+/- 2 %);
    #: "instrument", widened to a scan range drawn per frame (low end U(250, 600), high end
    #: U(1500, 3500) m/z, never narrower than the ions), as real runs are recorded, so that a
    #: species' missing charge states lie inside the grid ("seen, empty"), not outside it. The
    #: range is kept within one crop (ln(hi / lo) <= 0.98 crop_size grid_step) where the ions
    #: allow, so that the whole grid is read in one window
    scan_range: str = "sticks"
    #: share of frames whose singly charged background is 10-100 times stronger (log-uniform):
    #: single ions then reach about 0.1-300 % of the strongest component, as in real wash regions
    background_strong_prob: float = 0.0
    #: share of frames in which every oligonucleotide (dna, rna, ps_dna, ps_rna; 900 Da and up)
    #: charges as in real ion-pairing runs: apex 0.0769 M^0.445 x U(0.9, 1.1), width 0.35-0.6
    #: charges, one to three charge states (``synth.charge.sample_charge_shape``)
    oligo_real_like_prob: float = 0.0
    #: append the isotope-charge channels (charge read from resolved isotope spacing, strength,
    #: isotope limit) to the features, computed over the whole grid before the crop
    #: (``nn.features.isotope_channels``: read on Orbitrap and FT-ICR frames, empty on
    #: time-of-flight ones)
    iso_channels: bool = False
    #: append the measured channel last: 1 on the crop's bins the frame's grid covers, 0 on the
    #: zero-padded tail (a comb-view LadderNet's input; no draws, so the frames are otherwise
    #: those drawn without it)
    measured_channel: bool = False


#: settings added after resumable training runs existed, with the values every earlier run had:
#: left out of :func:`settings_repr` while they keep them
LATER_DEFAULTS = {"scan_range": "sticks", "background_strong_prob": 0.0,
                  "oligo_real_like_prob": 0.0, "iso_channels": False, "measured_channel": False}


def settings_repr(obj) -> str:
    """``repr`` of a SynthConfig, or of a mixture (lists and tuples of weights and configs), as
    the dataclass writes it, without the settings of :data:`LATER_DEFAULTS` that keep their
    defaults: the text the training run's resume fingerprint is taken over, unchanged for the
    settings an earlier version knew."""
    if isinstance(obj, SynthConfig):
        parts = [f"{f.name}={getattr(obj, f.name)!r}" for f in fields(obj)
                 if f.repr and not (f.name in LATER_DEFAULTS
                                    and getattr(obj, f.name) == LATER_DEFAULTS[f.name])]
        return f"{type(obj).__qualname__}({', '.join(parts)})"
    if isinstance(obj, list):
        return "[" + ", ".join(settings_repr(o) for o in obj) + "]"
    if isinstance(obj, tuple):
        inner = ", ".join(settings_repr(o) for o in obj)
        return f"({inner},)" if len(obj) == 1 else f"({inner})"
    return repr(obj)


def orbitrap_config(**kw) -> SynthConfig:
    c = SynthConfig(instrument_kind="orbitrap", resolution_range=(30000.0, 240000.0), **kw)
    return c
