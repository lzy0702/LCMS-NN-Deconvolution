"""Frame-tier generator: scenes -> training samples (features + labels) on grid crops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chem.adducts import AdductLibrary, AdductState
from ..chem.instrument import InstrumentModel
from ..nn.features import featurize
from ..nn.grid import LogMzGrid
from .adduct_sampler import adduct_fractions_for_charge, sample_run_propensities
from .charge import charge_distribution
from .compounds import choose_class, sample_compound
from .config import SynthConfig
from .impurities import sample_contaminant, sample_impurities
from .labels import build_labels
from .noise import NoiseConfig, add_noise
from .render import ComponentInstance, Scene, build_sticks, render_grid_by_charge
from .spec import ComponentTruth


@dataclass
class FrameSample:
    features: np.ndarray  # [C, L]
    topk_z: np.ndarray  # [L, k]
    topk_w: np.ndarray  # [L, k]
    heat: np.ndarray  # [L]
    b0: int
    truths: list[ComponentTruth]
    meta: dict


def sample_instrument(cfg: SynthConfig, rng: np.random.Generator) -> InstrumentModel:
    res = float(np.exp(rng.uniform(np.log(cfg.resolution_range[0]), np.log(cfg.resolution_range[1]))))
    eta = float(rng.uniform(*cfg.shape_eta_range))
    shape = "mixed" if eta > 0 else "gaussian"
    sat = None
    return InstrumentModel(cfg.instrument_kind, res, shape=shape, eta=eta, saturation_level=sat)


def sample_scene(cfg: SynthConfig, rng: np.random.Generator, library: AdductLibrary) -> Scene:
    n_main = int(rng.integers(cfg.n_main_range[0], cfg.n_main_range[1] + 1))
    props = sample_run_propensities(rng, library, cfg.adduct_max_lambda)
    components: list[ComponentInstance] = []
    cid = 0
    for _ in range(n_main):
        cc = choose_class(rng, cfg.classes)
        main = sample_compound(rng, cc)
        main_intensity = float(np.exp(rng.uniform(np.log(cfg.base_intensity_range[0]),
                                                  np.log(cfg.base_intensity_range[1]))))
        members = [(main, 1.0)]
        members += sample_impurities(rng, main, cid, cfg.max_impurities, cfg.impurity_abundance)
        for comp, rel in members:
            cd = charge_distribution(comp.mass, comp.compound_class, cfg.esi_mode, cfg.polarity,
                                     rng, z_max=cfg.z_max)
            adducts = {z: adduct_fractions_for_charge(z, props, library, rng) for z in cd}
            components.append(ComponentInstance(comp, main_intensity * rel, cd, adducts))
        cid = len(components)
    if rng.random() < cfg.contaminant_prob:
        comp, rel = sample_contaminant(rng, cfg.classes)
        cd = charge_distribution(comp.mass, comp.compound_class, cfg.esi_mode, cfg.polarity, rng, z_max=cfg.z_max)
        adducts = {z: {AdductState(): 1.0} for z in cd}
        base = components[0].intensity if components else 1e5
        components.append(ComponentInstance(comp, base * rel, cd, adducts))
    return Scene(components, cfg.polarity, library, meta={"propensities": props})


def _choose_crop(scene: Scene, grid: LogMzGrid, cfg: SynthConfig, rng: np.random.Generator) -> int:
    L = cfg.crop_size
    if rng.random() < 0.7 and scene.components:
        ci = scene.components[int(rng.integers(len(scene.components)))]
        if ci.charges:
            z = int(rng.choice(list(ci.charges)))
            b = grid.mass_charge_to_bin(ci.compound.mass, z)
            b0 = b - int(rng.integers(L // 8, 7 * L // 8))
        else:
            b0 = int(rng.integers(0, max(1, grid.size - L)))
    else:
        b0 = int(rng.integers(0, max(1, grid.size - L)))
    return int(np.clip(b0, 0, max(0, grid.size - L)))


def generate_frame(cfg: SynthConfig, rng: np.random.Generator) -> FrameSample:
    library = AdductLibrary.from_mode(
        cfg.mode, cfg.polarity, max_per_type=cfg.adduct_max_per_type, max_total=cfg.adduct_max_total
    )
    grid = LogMzGrid(50.0, cfg.grid_mz_max, cfg.grid_step, cfg.polarity)
    inst = sample_instrument(cfg, rng)
    scene = sample_scene(cfg, rng, library)
    sticks = build_sticks(scene)
    b0 = _choose_crop(scene, grid, cfg, rng)
    b1 = b0 + cfg.crop_size
    cmat, charges, _comp = render_grid_by_charge(sticks, grid, inst, b0, b1)
    signal = cmat.sum(axis=0) if cmat.shape[0] else np.zeros(cfg.crop_size)

    gain = float(rng.uniform(1.0, 50.0))
    esig = float(rng.uniform(0.5, 5.0))
    peak_max = signal.max() if signal.size else 0.0
    chem_max = max(5.0, 0.02 * peak_max) if peak_max > 0 else 10.0
    ncfg = NoiseConfig(gain=gain, electronic_sigma=esig, chemical_density=rng.uniform(0.0, 0.5),
                       chemical_max=chem_max)
    observed, noise = add_noise(signal, ncfg, rng)

    if rng.random() < cfg.saturation_prob and peak_max > 0:
        sat_level = peak_max * float(rng.uniform(0.3, 0.8))
        observed = np.minimum(observed, sat_level)

    noise_sigma = max(esig, 1e-6)
    feats = featurize(observed, noise_sigma, cfg.grid_step, z_max=min(cfg.z_max, 64))
    labels = build_labels(cmat, charges, noise)
    return FrameSample(feats, labels["topk_z"], labels["topk_w"], labels["heat"], b0,
                       scene.truths(), {"resolution": inst.resolution, "kind": inst.kind})
