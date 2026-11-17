"""Run-tier generator: whole LC-MS runs with UV, ESI saturation and polarity switching."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..chem.adducts import AdductLibrary
from ..chem.instrument import InstrumentModel
from ..core.model import Chromatogram, Run, Spectrum
from .adduct_sampler import member_distributions, run_adduct_draw
from .compounds import choose_class, sample_compound
from .config import SynthConfig
from .impurities import sample_impurities
from .render import ComponentInstance, Scene, build_sticks, render_profile
from .spec import Compound


def emg(t: np.ndarray, center: float, sigma: float, tau: float) -> np.ndarray:
    """Exponentially modified Gaussian elution profile, scaled to unit apex.

    The normalisation is computed from a dense reference grid over the peak rather than from
    the values passed in, so evaluating the profile at a single time gives the same answer as
    evaluating it across the whole run.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    y = _emg_raw(t, center, sigma, tau)
    ref = _emg_raw(np.linspace(center - 4 * sigma - 4 * tau, center + 6 * sigma + 8 * tau, 512),
                   center, sigma, tau)
    m = float(ref.max())
    return y / m if m > 0 else y


def _emg_raw(t: np.ndarray, center: float, sigma: float, tau: float) -> np.ndarray:
    """Exponentially modified Gaussian, up to a constant factor, without overflow or underflow.

    The textbook form ``exp(s**2/2 - d*s) * erfc((s - d)/sqrt(2))`` (``d = (t - center)/sigma``,
    ``s = sigma/tau``) multiplies a huge exponential by a vanishing erfc once the tail is short
    compared with the width. For ``s`` above about 27 both factors leave double range and the
    whole peak came out as zeros, silently removing it from the run while it stayed in the
    truth. Where the erfc argument is positive the product is rewritten with the scaled
    complementary error function, ``exp(-d**2/2) * erfcx(x)``, which is exact and bounded;
    elsewhere the original form cannot overflow.
    """
    if tau <= 0 or sigma / tau > 1e6:
        return np.exp(-0.5 * ((t - center) / sigma) ** 2)
    from scipy.special import erfc, erfcx

    s = sigma / tau
    d = (t - center) / sigma
    x = (s - d) / np.sqrt(2.0)
    y = np.empty_like(d, dtype=float)
    pos = x >= 0
    y[pos] = np.exp(-0.5 * d[pos] ** 2) * erfcx(x[pos])
    neg = ~pos
    y[neg] = np.exp(0.5 * s * s - s * d[neg]) * erfc(x[neg])
    # scale so that the tau -> 0 limit tends to the plain Gaussian rather than to zero
    y *= s * np.sqrt(np.pi / 2.0)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class PeakTruth:
    compound: Compound
    rt: float
    intensity: float
    sigma: float
    tau: float
    members: list[tuple[Compound, float]]  # (compound, relative abundance)
    charge_dists: dict[int, dict[int, float]] = field(default_factory=dict)
    saturated: bool = False
    _adducts: dict = field(default_factory=dict, repr=False)
    _counts: dict = field(default_factory=dict, repr=False)


#: Counts per ion of the simulated detector (shot-noise scale of the run tier).
RUN_GAIN = 20.0


@dataclass
class RunTruth:
    peaks: list[PeakTruth]
    polarity_schedule: list[int]
    mode: str
    esi_mode: str
    gain: float = RUN_GAIN

    def to_dict(self) -> dict:
        out = {"mode": self.mode, "esi_mode": self.esi_mode, "gain": self.gain, "peaks": []}
        for p in self.peaks:
            counts = getattr(p, "_counts", {}) or {}
            out["peaks"].append({
                "main": p.compound.name,
                "mass": p.compound.mass,
                "compound_class": p.compound.compound_class,
                "rt": p.rt,
                "intensity": p.intensity,
                "saturated": p.saturated,
                "members": [
                    {"name": c.name, "mass": c.mass, "rel": rel, "kind": c.kind,
                     "delta": c.meta.get("delta", 0.0),
                     **counts.get(i, {})}
                    for i, (c, rel) in enumerate(p.members)
                ],
            })
        return out


def _expected_counts(p: PeakTruth, profile: np.ndarray, gain: float,
                     points_per_fwhm: float = 5.0) -> dict[int, dict]:
    """Expected ion counts in each member's strongest and second-strongest charge state.

    A count here is the whole envelope of one charge state (all isotopologues and adduct forms)
    summed over every frame of the chromatographic peak and over the profile points of each
    peak, divided by the detector gain. It decides whether a member can be detected at all: a
    species needs two charge states to be confirmed, so ``support_counts`` (the second-strongest
    state) is the relevant figure. A member with a few counts cannot be told from shot noise
    however good the deconvolution is, and scoring it as a miss would measure the generator,
    not the software.
    """
    # profile peaks use the apex-height convention: a Gaussian of height h sampled at
    # points_per_fwhm points per FWHM sums to h * sqrt(2 pi) * points_per_fwhm / 2.3548
    area_per_height = np.sqrt(2.0 * np.pi) * points_per_fwhm / 2.354820045
    frames_sum = float(np.sum(profile))
    out: dict[int, dict] = {}
    for idx, (_comp, rel) in enumerate(p.members):
        cd = p.charge_dists.get(idx, {})
        per_z = sorted(cd.values(), reverse=True)
        scale = p.intensity * rel * area_per_height * frames_sum / gain
        peak = per_z[0] * scale if per_z else 0.0
        support = per_z[1] * scale if len(per_z) > 1 else (peak if set(cd) == {1} else 0.0)
        out[idx] = {"peak_counts": float(peak), "support_counts": float(support)}
    return out


def generate_run(
    cfg: SynthConfig,
    rng: np.random.Generator,
    n_peaks: int = 3,
    rt_range: tuple[float, float] = (1.0, 11.0),
    scan_rate_hz: float = 3.0,
    polarity_switching: bool = False,
    esi_saturation: bool = False,
    sparse_threshold: float = 8.0,
) -> tuple[Run, RunTruth]:
    library = AdductLibrary.from_mode(cfg.mode, 0, max_per_type=cfg.adduct_max_per_type,
                                      max_total=cfg.adduct_max_total)
    props = run_adduct_draw(cfg, library, rng)
    inst = InstrumentModel(cfg.instrument_kind,
                           float(np.exp(rng.uniform(*np.log(cfg.resolution_range)))),
                           shape="gaussian")

    # build peaks
    peaks: list[PeakTruth] = []
    # keep peaks away from the run edges, but never invert the interval on a short run
    span = max(rt_range[1] - rt_range[0], 1e-6)
    margin = min(0.5, 0.25 * span)
    centers = np.sort(rng.uniform(rt_range[0] + margin, rt_range[1] - margin, n_peaks))
    for c in centers:
        cc = choose_class(rng, cfg.classes)
        main = sample_compound(rng, cc)
        intensity = float(np.exp(rng.uniform(np.log(cfg.base_intensity_range[0]),
                                             np.log(cfg.base_intensity_range[1]))))
        sigma = float(rng.uniform(0.03, 0.12))
        tau = float(rng.uniform(0.0, 2.0) * sigma)
        members = [(main, 1.0)] + sample_impurities(rng, main, 0, cfg.max_impurities, cfg.impurity_abundance)
        cdists = {}
        adists: dict[int, dict] = {}
        for idx, (cd, adducts) in enumerate(member_distributions(cfg, members, library, props, rng)):
            cdists[idx] = cd
            adists[idx] = adducts
        saturated = esi_saturation and rng.random() < 0.5
        pk = PeakTruth(main, float(c), intensity, sigma, tau, members, cdists, saturated)
        pk._adducts = adists  # cached, reused every frame
        peaks.append(pk)

    # time and polarity axes
    n_frames = int((rt_range[1] - rt_range[0]) * 60.0 * scan_rate_hz)
    times = np.linspace(rt_range[0], rt_range[1], n_frames)
    if polarity_switching:
        pol_sched = [1 if i % 2 == 0 else -1 for i in range(n_frames)]
    else:
        pol_sched = [cfg.polarity] * n_frames

    # instrument m/z axis (tight to the ions present)
    all_mz = []
    for p in peaks:
        for idx, (comp, _rel) in enumerate(p.members):
            for z in p.charge_dists[idx]:
                all_mz.append((comp.mass + z * 1.007) / z)
    mz_lo = max(cfg.classes and 100.0, min(all_mz) - 20) if all_mz else 100.0
    mz_hi = (max(all_mz) + 20) if all_mz else 3000.0
    axis = inst.profile_axis(max(100.0, mz_lo), min(cfg.grid_mz_max, mz_hi))

    spectra: list[Spectrum] = []
    uv = np.zeros(n_frames)
    esi_level = np.percentile([p.intensity for p in peaks], 60) * 3 if peaks else 1e6
    profiles = [emg(times, p.rt, p.sigma, p.tau) for p in peaks]
    for p, prof in zip(peaks, profiles):
        p._counts = _expected_counts(p, prof, RUN_GAIN, inst.points_per_fwhm)

    for fi, (t, pol) in enumerate(zip(times, pol_sched)):
        components: list[ComponentInstance] = []
        frame_conc = 0.0
        for pi, p in enumerate(peaks):
            prof = float(profiles[pi][fi])
            if prof < 1e-4:
                continue
            for idx, (comp, rel) in enumerate(p.members):
                cd = p.charge_dists[idx]
                if not cd:
                    continue
                inten = p.intensity * rel * prof
                frame_conc += inten
                adducts = p._adducts[idx]
                components.append(ComponentInstance(comp, inten, cd, adducts))
            uv[fi] += p.intensity * prof * _uv_response(p.compound.compound_class)
        if not components:
            spectra.append(Spectrum(axis[:2].copy(), np.zeros(2, np.float32), float(t), pol, 1,
                                    f"scan={fi+1}", fi, True))
            continue
        scene = Scene(components, pol, library)
        sticks = build_sticks(scene)
        prof_spec = render_profile(sticks, axis, inst)
        # ESI saturation compresses the per-frame total (TIC flat-tops; UV stays linear)
        if esi_saturation:
            comp_factor = 1.0 / (1.0 + frame_conc / max(esi_level, 1e-9))
            if any(p.saturated for p in peaks):
                prof_spec = prof_spec * comp_factor
        # shot + electronic noise
        gain = RUN_GAIN
        lam = np.clip(prof_spec / gain, 0, None)
        prof_spec = rng.poisson(lam).astype(np.float64) * gain
        prof_spec += np.abs(rng.normal(0, 1.5, axis.size))
        mz_out, int_out = _threshold_profile(axis, prof_spec, sparse_threshold)
        spectra.append(Spectrum(mz_out, int_out.astype(np.float32), float(t), pol, 1,
                                f"scan={fi+1}", fi, True))

    # UV trace with a small delay
    delay = rng.uniform(0.02, 0.15)
    uv_time = times + delay
    chroms = {"UV1": Chromatogram(uv_time, uv + np.abs(rng.normal(0, uv.max() * 0.002 + 1e-6, n_frames)),
                                  "UV1", "uv", "AU")}
    run = Run(spectra, chroms, name="synthetic", source="synthetic",
              meta={"mode": cfg.mode, "esi_mode": cfg.esi_mode})
    truth = RunTruth(peaks, pol_sched, cfg.mode, cfg.esi_mode)
    return run, truth


def _threshold_profile(axis: np.ndarray, intensity: np.ndarray, threshold: float,
                       pad: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Keep only profile points near real signal, the way vendor software thresholds data.

    A full profile axis for every frame makes files enormous without adding information; real
    instruments write the points around each peak and drop the empty baseline between them.
    """
    if threshold <= 0:
        return axis.copy(), intensity.copy()
    keep = intensity > threshold
    if not keep.any():
        return axis[:2].copy(), intensity[:2].copy()
    idx = np.flatnonzero(keep)
    lo = np.maximum(idx - pad, 0)
    hi = np.minimum(idx + pad + 1, axis.size)
    mask = np.zeros(axis.size, dtype=bool)
    for a, b in zip(lo, hi):
        mask[a:b] = True
    return axis[mask].copy(), intensity[mask].copy()


def _uv_response(cls: str) -> float:
    return {"peptide": 1.0, "dna": 3.0, "rna": 3.0, "ps_dna": 3.0, "ps_rna": 3.0,
            "glycan": 0.2, "peg": 0.05, "small_molecule": 0.8}.get(cls, 1.0)
