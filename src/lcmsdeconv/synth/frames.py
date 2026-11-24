"""Frame-tier generator: scenes -> training samples (features + labels) on grid crops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chem.adducts import AdductLibrary, AdductState
from ..chem.instrument import InstrumentModel
from ..nn.features import featurize, isotope_channels
from ..nn.grid import LogMzGrid
from .adduct_sampler import member_distributions, run_adduct_draw
from .charge import charge_distribution
from .compounds import choose_class, sample_compound
from .config import SynthConfig
from .impurities import sample_contaminant, sample_impurities
from .labels import build_labels
from .noise import NoiseConfig, add_noise
from .render import (
    ComponentInstance,
    Scene,
    build_sticks,
    render_grid_by_charge,
    stick_parts,
    sticks_from_parts,
)
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
    # the real-like ion-pairing charge model, for every oligonucleotide of the frame (one draw,
    # and only when the option is on: frames without it are drawn exactly as before)
    real_like = bool(cfg.oligo_real_like_prob > 0 and rng.random() < cfg.oligo_real_like_prob)
    n_main = int(rng.integers(cfg.n_main_range[0], cfg.n_main_range[1] + 1))
    props = run_adduct_draw(cfg, library, rng)
    components: list[ComponentInstance] = []
    cid = 0
    for _ in range(n_main):
        cc = choose_class(rng, cfg.classes)
        main = sample_compound(rng, cc)
        main_intensity = float(np.exp(rng.uniform(np.log(cfg.base_intensity_range[0]),
                                                  np.log(cfg.base_intensity_range[1]))))
        members = [(main, 1.0)]
        members += sample_impurities(rng, main, cid, cfg.max_impurities, cfg.impurity_abundance)
        for (comp, rel), (cd, adducts) in zip(members,
                                              member_distributions(cfg, members, library, props, rng,
                                                                   real_like=real_like)):
            components.append(ComponentInstance(comp, main_intensity * rel, cd, adducts))
        cid = len(components)
    if rng.random() < cfg.contaminant_prob:
        comp, rel = sample_contaminant(rng, cfg.classes)
        cd = charge_distribution(comp.mass, comp.compound_class, cfg.esi_mode, cfg.polarity, rng,
                                 z_max=cfg.z_max, real_like=real_like)
        adducts = {z: {AdductState(): 1.0} for z in cd}
        base = components[0].intensity if components else 1e5
        components.append(ComponentInstance(comp, base * rel, cd, adducts))
    meta = {"propensities": props}
    if cfg.oligo_real_like_prob > 0:
        meta["oligo_real_like"] = real_like
    return Scene(components, cfg.polarity, library, meta=meta)


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


def instrument_scan_range(cfg: SynthConfig, rng: np.random.Generator, lo: float,
                          hi: float) -> tuple[float, float]:
    """The scan range of a frame measured as an instrument records it (``scan_range =
    "instrument"``): the ions' range ``lo .. hi`` widened to a range drawn per frame, low end
    U(250, 600) and high end U(1500, 3500) m/z, capped at ``0.995 grid_mz_max``; where the ions
    allow, the widening is shortened (both ends in proportion) so that ``ln(hi / lo)`` stays
    within ``0.98 crop_size grid_step`` and the frame's grid fits one crop."""
    new_lo = min(lo, float(rng.uniform(250.0, 600.0)))
    new_hi = min(max(hi, float(rng.uniform(1500.0, 3500.0))), cfg.grid_mz_max * 0.995)
    span = 0.98 * cfg.crop_size * cfg.grid_step
    if np.log(new_hi / new_lo) <= span:
        return new_lo, new_hi
    base = float(np.log(hi / lo))
    if base >= span:  # the ions themselves span more than a crop: measure just them
        return lo, hi
    below, above = float(np.log(lo / new_lo)), float(np.log(new_hi / hi))
    keep = (span - base) / (below + above)
    return float(lo * np.exp(-below * keep)), float(hi * np.exp(above * keep))


def _without_singly_charged(sticks):
    """The sticks of every charge state but z = 1."""
    keep = sticks.z != 1
    return type(sticks)(sticks.mz[keep], sticks.intensity[keep], sticks.comp_id[keep],
                        sticks.z[keep], sticks.adduct_mass[keep])


def generate_frame(cfg: SynthConfig, rng: np.random.Generator) -> FrameSample:
    library = AdductLibrary.from_mode(
        cfg.mode, cfg.polarity, max_per_type=cfg.adduct_max_per_type, max_total=cfg.adduct_max_total
    )
    if cfg.render_path == "profile":
        return generate_frame_profile(cfg, rng, library)
    if cfg.iso_channels or cfg.measured_channel:
        raise ValueError("isotope-charge and measured channels are computed on the whole measured "
                         "grid, which only the profile render path builds (render_path='profile')")
    grid = LogMzGrid(50.0, cfg.grid_mz_max, cfg.grid_step, cfg.polarity)
    inst = sample_instrument(cfg, rng)
    scene = sample_scene(cfg, rng, library)
    sticks = build_sticks(scene)
    if cfg.singly_charged == "drop":
        sticks = _without_singly_charged(sticks)
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
    feats = featurize(observed, noise_sigma, cfg.grid_step, z_max=min(cfg.z_max, 64),
                      evidence_channels=cfg.evidence_channels)
    labels = build_labels(cmat, charges, noise)
    return FrameSample(feats, labels["topk_z"], labels["topk_w"], labels["heat"], b0,
                       scene.truths(), {"resolution": inst.resolution, "kind": inst.kind})


# ----------------------------------------------------------------- inference-parity rendering
def _background_ions(rng: np.random.Generator, n: int, mz_lo: float, mz_hi: float, scale: float,
                     polarity: int) -> list[ComponentInstance]:
    """Singly charged chemical background (solvent clusters, plasticisers, column bleed)."""
    from ..chem.classes import class_isotope_pattern
    from .spec import Compound

    out = []
    # the same bounds every ion (computed once), the same draws in the same order
    log_lo, log_hi = np.log(max(mz_lo, 100.0)), np.log(max(mz_hi, 101.0))
    log_a, log_b = np.log(1e-4), np.log(3e-2)
    average: dict[int, tuple] = {}  # pattern -> its average mass (the pattern kept alive)
    for _ in range(n):
        mz = float(np.exp(rng.uniform(log_lo, log_hi)))
        mass = mz - float(polarity) * 1.00727646688
        if mass <= 50:
            continue
        pat = class_isotope_pattern(mass, "small_molecule")
        if id(pat) not in average:
            average[id(pat)] = (pat, pat.average_mass)
        comp = Compound(mass, "small_molecule", pat.shifted(mass - average[id(pat)][1]),
                        name="background", kind="contaminant")
        inten = scale * float(np.exp(rng.uniform(log_a, log_b)))
        out.append(ComponentInstance(comp, inten, {1: 1.0}, {1: {AdductState(): 1.0}}))
    return out


def scan_spectra(scans, polarity: int) -> list:
    """The thresholded scans ``[(mz, intensity), ...]`` of a training frame as profile spectra,
    as a region hands its scans to ``spectrum_to_grid(..., frames=...)`` at inference."""
    from ..core.model import Spectrum

    return [Spectrum(np.asarray(mz, dtype=float), np.asarray(y, dtype=np.float32), 0.0,
                     polarity, is_profile=True) for mz, y in scans]


def generate_frame_profile(cfg: SynthConfig, rng: np.random.Generator,
                           library: AdductLibrary) -> FrameSample:
    """One training sample rendered along the same path inference takes.

    Inference deconvolves the summed spectrum of a chromatographic region: vendor-thresholded
    profile scans, summed on their own m/z axis, resampled onto the logarithmic grid, with the
    noise level estimated from that grid spectrum. Training features computed any other way
    (rendered directly on the grid, with the generator's noise parameter standing in for the
    estimate) describe a different input distribution, and a network trained on them is
    evaluated on data it has never seen. Labels come from per-charge renders of the same scans
    through the same resampling, so they stay consistent with the features.
    """
    from .fast import ProfileRows, featurize, noisy_scan, positive_points, spectrum_to_grid
    from .render import render_profile

    inst = sample_instrument(cfg, rng)
    scene = sample_scene(cfg, rng, library)
    parts = stick_parts(scene.components, scene.polarity)
    sticks = sticks_from_parts(parts)
    polarity = cfg.polarity
    if len(sticks):
        lo = max(60.0, float(sticks.mz.min()) * 0.98)
        hi = min(cfg.grid_mz_max * 0.995, float(sticks.mz.max()) * 1.02)
    else:
        lo, hi = 300.0, 3000.0
    if hi <= lo * 1.05:
        lo, hi = max(60.0, lo / 1.5), min(cfg.grid_mz_max * 0.995, hi * 1.5)
    if cfg.scan_range == "instrument":
        lo, hi = instrument_scan_range(cfg, rng, lo, hi)
    elif cfg.scan_range != "sticks":
        raise ValueError(f"scan_range must be 'sticks' or 'instrument', not {cfg.scan_range!r}")
    # chemical background across the measured range
    n_bg = int(rng.integers(cfg.background_ions_range[0], cfg.background_ions_range[1] + 1))
    if cfg.singly_charged == "drop":
        n_bg = 0
        sticks = _without_singly_charged(sticks)
    if n_bg:
        base = max((c.intensity for c in scene.components), default=1e5)
        if cfg.background_strong_prob > 0 and rng.random() < cfg.background_strong_prob:
            # strong background: ions at about 0.1-300 % of the strongest component
            base *= float(np.exp(rng.uniform(np.log(10.0), np.log(100.0))))
        bg = _background_ions(rng, n_bg, lo, hi, base, polarity)
        if bg:
            # the sticks of the larger scene are those already built followed by the ions'
            parts += stick_parts(bg, polarity, first_id=len(scene.components))
            scene = Scene(scene.components + bg, polarity, library, meta=scene.meta)
            sticks = sticks_from_parts(parts)

    axis = inst.profile_axis(lo, hi)
    clean = render_profile(sticks, axis, inst) if len(sticks) else np.zeros(axis.size)

    # scans summed over part of an elution peak: each scan sees a fraction of the signal
    n_sum = int(rng.integers(cfg.sum_frames_range[0], cfg.sum_frames_range[1] + 1))
    weights = np.exp(-0.5 * np.linspace(-1.5, 1.5, n_sum) ** 2) if n_sum > 1 else np.ones(1)
    weights = weights / weights.sum()
    gain = float(rng.uniform(*cfg.gain_range))
    esig = float(rng.uniform(*cfg.electronic_sigma_range))
    thr = esig * float(rng.uniform(*cfg.threshold_sigmas_range))
    saturate = rng.random() < cfg.saturation_prob
    sat_level = float(clean.max() * weights.max() * rng.uniform(0.3, 0.8)) if saturate else None
    # each scan: shot noise on its share of the signal, electronic noise, saturation, and vendor
    # thresholding, which writes every point above the threshold together with its neighbours,
    # as segments of at least five points (cfg.vendor_pad), never noise points one by one
    frames_int = [noisy_scan(rng, clean, w, gain, esig, sat_level, thr, cfg.vendor_pad)
                  for w in weights]

    # exactly the grid view inference builds for this summed spectrum: each scan resampled on
    # its own points (what it kept after thresholding) and added, as for a region's scans
    total = _sum_scans(frames_int)
    raw = positive_points(axis, total)  # (axis[total > 0], total[total > 0])
    scans = [positive_points(axis, s) for s in frames_int]
    grid, observed, noise_sigma = spectrum_to_grid(raw[0], raw[1], polarity,
                                                   cfg.grid_step, cfg.grid_mz_max,
                                                   frames=scan_spectra(scans, polarity))
    g_lo = grid.mz_min
    # isotope-charge channels: absolute m/z and the instrument's resolving power, over the
    # whole grid as the predictor computes them (nn.features.isotope_channels), then cropped
    iso = isotope_channels(grid, observed, noise_sigma, inst) if cfg.iso_channels else None

    L = cfg.crop_size
    b0 = _choose_crop(scene, grid, cfg, rng) if grid.size > L else 0
    b1 = b0 + L
    window = np.zeros(L)
    seg = observed[b0:min(b1, grid.size)]
    window[: seg.size] = seg

    # per-charge noise-free signal through the same resampling, on the crop only. (Labels stay
    # unclipped: saturation does not change the charge.) The sticks near the crop, grouped by
    # charge in their own order, are what ``(sticks.z == z) & near`` selects for each charge, and
    # ProfileRows is ``grid.resample_profile(sub_axis, prof)[b0:b1]`` with the axis work done once.
    crop_lo = float(grid.bin_to_mz(b0)) * 0.999
    crop_hi = float(grid.bin_to_mz(min(b1, grid.size) - 1)) * 1.001
    charges = np.unique(sticks.z) if len(sticks) else np.array([], dtype=int)
    rows, zs = [], []
    a0 = int(np.searchsorted(axis, crop_lo))
    a1 = int(np.searchsorted(axis, crop_hi))
    sub_axis = axis[max(0, a0 - 2):min(axis.size, a1 + 2)]
    if charges.size and sub_axis.size >= 2:
        near = np.flatnonzero((sticks.mz > crop_lo * 0.995) & (sticks.mz < crop_hi * 1.005))
        near = near[np.argsort(sticks.z[near], kind="stable")]
        z_near = sticks.z[near]
        resampled = ProfileRows(grid, sub_axis, b0, L)
        for z in charges:
            sel = near[np.searchsorted(z_near, z, "left"):np.searchsorted(z_near, z, "right")]
            if not sel.size:
                continue
            sub = type(sticks)(sticks.mz[sel], sticks.intensity[sel], sticks.comp_id[sel],
                               sticks.z[sel], sticks.adduct_mass[sel])
            row = resampled.row(render_profile(sub, sub_axis, inst))
            if row.max() > 0:
                rows.append(row)
                zs.append(int(z))
    cmat = np.array(rows) if rows else np.zeros((0, L))
    # shares of the observed intensity: y * c_z / max(y, sum c), the remainder is non-ion
    signal = cmat.sum(axis=0) if cmat.shape[0] else np.zeros(L)
    noise_part = np.clip(window - signal, 0.0, None)
    feats = featurize(window, noise_sigma, cfg.grid_step, z_max=min(cfg.z_max, 64),
                      evidence_channels=cfg.evidence_channels)
    if iso is not None:
        part = np.zeros((iso.shape[0], L), dtype=np.float32)
        piece = iso[:, b0:min(b1, grid.size)]
        part[:, : piece.shape[1]] = piece
        feats = np.concatenate([feats, part])
    if cfg.measured_channel:
        # the crop's bins the grid covers (a comb-view network reads them as measured)
        measured = np.zeros((1, L), dtype=np.float32)
        measured[0, : seg.size] = 1.0
        feats = np.concatenate([feats, measured])
    labels = build_labels(cmat, np.array(zs, dtype=int), noise_part, noise_level=noise_sigma,
                          observed=window)
    meta = {"resolution": inst.resolution, "kind": inst.kind, "noise_sigma": noise_sigma,
            "n_sum": n_sum, "gain": gain, "grid_lo": g_lo, "raw": raw, "scans": scans}
    if cfg.scan_range != "sticks":
        meta["scan_range"] = (lo, hi)
    if "oligo_real_like" in scene.meta:
        meta["oligo_real_like"] = scene.meta["oligo_real_like"]
    return FrameSample(feats, labels["topk_z"], labels["topk_w"], labels["heat"], b0,
                       scene.truths(), meta)


def _sum_scans(scans: list) -> np.ndarray:
    """``np.sum(scans, axis=0)`` without stacking them: numpy adds the rows of a stack one after
    another, element by element, which is what this does."""
    total = scans[0].copy()
    for s in scans[1:]:
        total += s
    return total
