"""Run-tier generator: whole LC-MS runs with UV, ESI saturation and polarity switching."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..chem.adducts import AdductLibrary
from ..chem.instrument import InstrumentModel
from ..core.model import Chromatogram, Run, Spectrum
from .adduct_sampler import adduct_fractions_for_charge, sample_run_propensities
from .charge import charge_distribution
from .compounds import choose_class, sample_compound
from .config import SynthConfig
from .impurities import sample_impurities
from .render import ComponentInstance, Scene, build_sticks, render_profile
from .spec import Compound


def emg(t: np.ndarray, center: float, sigma: float, tau: float) -> np.ndarray:
    """Exponentially modified Gaussian elution profile (unit apex ~1)."""
    if tau < 1e-4:
        y = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    else:
        from scipy.special import erfc

        arg = (sigma / tau - (t - center) / sigma) / np.sqrt(2)
        z = 0.5 * (sigma / tau) ** 2 - (t - center) / tau
        z = np.clip(z, -700, 700)
        y = np.exp(z) * erfc(arg)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    m = y.max()
    return y / m if m > 0 else y


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


@dataclass
class RunTruth:
    peaks: list[PeakTruth]
    polarity_schedule: list[int]
    mode: str
    esi_mode: str

    def to_dict(self) -> dict:
        out = {"mode": self.mode, "esi_mode": self.esi_mode, "peaks": []}
        for p in self.peaks:
            out["peaks"].append({
                "main": p.compound.name,
                "mass": p.compound.mass,
                "compound_class": p.compound.compound_class,
                "rt": p.rt,
                "intensity": p.intensity,
                "saturated": p.saturated,
                "members": [
                    {"name": c.name, "mass": c.mass, "rel": rel, "kind": c.kind,
                     "delta": c.meta.get("delta", 0.0)}
                    for c, rel in p.members
                ],
            })
        return out


def generate_run(
    cfg: SynthConfig,
    rng: np.random.Generator,
    n_peaks: int = 3,
    rt_range: tuple[float, float] = (1.0, 11.0),
    scan_rate_hz: float = 3.0,
    polarity_switching: bool = False,
    esi_saturation: bool = False,
) -> tuple[Run, RunTruth]:
    library = AdductLibrary.from_mode(cfg.mode, 0, max_per_type=cfg.adduct_max_per_type,
                                      max_total=cfg.adduct_max_total)
    props = sample_run_propensities(rng, library, cfg.adduct_max_lambda)
    inst = InstrumentModel(cfg.instrument_kind,
                           float(np.exp(rng.uniform(*np.log(cfg.resolution_range)))),
                           shape="gaussian")

    # build peaks
    peaks: list[PeakTruth] = []
    centers = np.sort(rng.uniform(rt_range[0] + 0.5, rt_range[1] - 0.5, n_peaks))
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
        for idx, (comp, _rel) in enumerate(members):
            cd = charge_distribution(comp.mass, comp.compound_class, cfg.esi_mode,
                                     cfg.polarity, rng, z_max=cfg.z_max)
            cdists[idx] = cd
            adists[idx] = {z: adduct_fractions_for_charge(z, props, library, rng) for z in cd}
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

    for fi, (t, pol) in enumerate(zip(times, pol_sched)):
        components: list[ComponentInstance] = []
        frame_conc = 0.0
        for p in peaks:
            prof = float(emg(np.array([t]), p.rt, p.sigma, p.tau)[0])
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
            spectra.append(Spectrum(axis.copy(), np.zeros(axis.size, np.float32), float(t), pol, 1,
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
        gain = 20.0
        lam = np.clip(prof_spec / gain, 0, None)
        prof_spec = rng.poisson(lam).astype(np.float64) * gain
        prof_spec += np.abs(rng.normal(0, 1.5, axis.size))
        spectra.append(Spectrum(axis.copy(), prof_spec.astype(np.float32), float(t), pol, 1,
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


def _uv_response(cls: str) -> float:
    return {"peptide": 1.0, "dna": 3.0, "rna": 3.0, "ps_dna": 3.0, "ps_rna": 3.0,
            "glycan": 0.2, "peg": 0.05, "small_molecule": 0.8}.get(cls, 1.0)
