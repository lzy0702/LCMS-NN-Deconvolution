"""Training frames as commit 2a5fcb5 made them, before the label builder, the label rows, the
scans and the comb were made faster, for tests/test_synth_speed.py and the script that recorded
digests of that commit's frames (tests/data/make_ref_2a5fcb5.py, which imports this module while
running the package as it was).

:func:`reference_frame` is 2a5fcb5's ``frames.generate_frame``, line for line, with every step
made faster since replaced by the reference it stands for: the labels by
``synth.labels.build_labels``, each charge's label row by ``LogMzGrid.resample_profile`` of a
fresh ``render.render_profile``, the scans by numpy's own lines (:func:`reference_scan`) and the
features by ``nn.features.featurize``. Every other step is the package's, called as it was.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def reference_scan(rng, clean, w, gain, esig, sat_level, thr, pad):
    """One scan of a training frame, numpy's own lines (``synth.fast.noisy_scan``)."""
    lam = np.clip(clean * w / gain, 0.0, None)
    scan = rng.poisson(lam).astype(np.float64) * gain + np.abs(rng.normal(0.0, esig, clean.size))
    if sat_level is not None:
        scan = np.minimum(scan, sat_level)
    if thr > 0:
        keep = np.convolve(scan >= thr, np.ones(2 * pad + 1), mode="same") > 0
        scan[~keep] = 0.0
    return scan


def reference_frame(cfg, rng):
    """``frames.generate_frame`` as it was at 2a5fcb5 (see the module)."""
    from lcmsdeconv.chem.adducts import AdductLibrary
    from lcmsdeconv.synth import frames

    library = AdductLibrary.from_mode(cfg.mode, cfg.polarity, max_per_type=cfg.adduct_max_per_type,
                                      max_total=cfg.adduct_max_total)
    if cfg.render_path != "profile":
        return frames.generate_frame(cfg, rng)  # the grid path: its references throughout
    return _reference_frame_profile(cfg, rng, library)


def _reference_frame_profile(cfg, rng, library):
    from lcmsdeconv.nn.features import featurize, isotope_channels
    from lcmsdeconv.synth.fast import positive_points, spectrum_to_grid
    from lcmsdeconv.synth.frames import (
        FrameSample,
        _background_ions,
        _choose_crop,
        _sum_scans,
        _without_singly_charged,
        instrument_scan_range,
        sample_instrument,
        sample_scene,
        scan_spectra,
    )
    from lcmsdeconv.synth.labels import build_labels
    from lcmsdeconv.synth.render import Scene, render_profile, stick_parts, sticks_from_parts

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
    n_bg = int(rng.integers(cfg.background_ions_range[0], cfg.background_ions_range[1] + 1))
    if cfg.singly_charged == "drop":
        n_bg = 0
        sticks = _without_singly_charged(sticks)
    if n_bg:
        base = max((c.intensity for c in scene.components), default=1e5)
        if cfg.background_strong_prob > 0 and rng.random() < cfg.background_strong_prob:
            base *= float(np.exp(rng.uniform(np.log(10.0), np.log(100.0))))
        bg = _background_ions(rng, n_bg, lo, hi, base, polarity)
        if bg:
            parts += stick_parts(bg, polarity, first_id=len(scene.components))
            scene = Scene(scene.components + bg, polarity, library, meta=scene.meta)
            sticks = sticks_from_parts(parts)

    axis = inst.profile_axis(lo, hi)
    clean = render_profile(sticks, axis, inst) if len(sticks) else np.zeros(axis.size)
    n_sum = int(rng.integers(cfg.sum_frames_range[0], cfg.sum_frames_range[1] + 1))
    weights = np.exp(-0.5 * np.linspace(-1.5, 1.5, n_sum) ** 2) if n_sum > 1 else np.ones(1)
    weights = weights / weights.sum()
    gain = float(rng.uniform(*cfg.gain_range))
    esig = float(rng.uniform(*cfg.electronic_sigma_range))
    thr = esig * float(rng.uniform(*cfg.threshold_sigmas_range))
    saturate = rng.random() < cfg.saturation_prob
    sat_level = float(clean.max() * weights.max() * rng.uniform(0.3, 0.8)) if saturate else None
    frames_int = [reference_scan(rng, clean, w, gain, esig, sat_level, thr, cfg.vendor_pad)
                  for w in weights]

    total = _sum_scans(frames_int)
    raw = positive_points(axis, total)
    scans = [positive_points(axis, s) for s in frames_int]
    grid, observed, noise_sigma = spectrum_to_grid(raw[0], raw[1], polarity,
                                                   cfg.grid_step, cfg.grid_mz_max,
                                                   frames=scan_spectra(scans, polarity))
    g_lo = grid.mz_min
    iso = isotope_channels(grid, observed, noise_sigma, inst) if cfg.iso_channels else None

    L = cfg.crop_size
    b0 = _choose_crop(scene, grid, cfg, rng) if grid.size > L else 0
    b1 = b0 + L
    window = np.zeros(L)
    seg = observed[b0:min(b1, grid.size)]
    window[: seg.size] = seg

    crop_lo = float(grid.bin_to_mz(b0)) * 0.999
    crop_hi = float(grid.bin_to_mz(min(b1, grid.size) - 1)) * 1.001
    charges = np.unique(sticks.z) if len(sticks) else np.array([], dtype=int)
    rows, zs = [], []
    a0 = int(np.searchsorted(axis, crop_lo))
    a1 = int(np.searchsorted(axis, crop_hi))
    sub_axis = axis[max(0, a0 - 2):min(axis.size, a1 + 2)]
    if charges.size and sub_axis.size >= 2:
        for z in charges:
            sel = (sticks.z == z) & (sticks.mz > crop_lo * 0.995) & (sticks.mz < crop_hi * 1.005)
            if not sel.any():
                continue
            sub = type(sticks)(sticks.mz[sel], sticks.intensity[sel], sticks.comp_id[sel],
                               sticks.z[sel], sticks.adduct_mass[sel])
            res = grid.resample_profile(sub_axis, render_profile(sub, sub_axis, inst))
            row = np.zeros(L)
            res = res[b0:min(b1, grid.size)]
            row[:res.size] = res
            if row.max() > 0:
                rows.append(row)
                zs.append(int(z))
    cmat = np.array(rows) if rows else np.zeros((0, L))
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


def reference_item(mixture, idx: int, base_seed: int = 0):
    """``FrameDataset(mixture, ..., base_seed).frame(idx)`` with :func:`reference_frame`."""
    rng = np.random.default_rng(base_seed + idx)
    w = np.array([float(wc[0]) for wc in mixture])
    cfg = mixture[int(rng.choice(len(mixture), p=w / w.sum()))][1]
    return cfg, reference_frame(cfg, rng)


# ------------------------------------------------------------------------------- recipes
@lru_cache(maxsize=4)
def train_script(root: Path = ROOT):
    """``scripts/train_charge_net.py`` of ``root`` as a module (its mixtures)."""
    spec = importlib.util.spec_from_file_location("train_charge_net",
                                                  root / "scripts" / "train_charge_net.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mixture(recipe: str, crop: int, script=None):
    """The training mixture of ``recipe`` with ``crop``-bin crops, set as training sets it:

    ``default:<singly charged>:<background ions>[:<evidence channels>]`` - the default mixture
    (``mixture``, 40 background ions unless 200); ``grid:<singly charged>:<background ions>`` -
    the same rendered on the grid; ``v3|v4:<window|comb>:<iso|noiso>`` - ``mixture_v3`` or
    ``mixture_v4`` as ``nn.train`` sets it for a LadderNet with that view (singly charged ions
    kept, the measured channel in the comb view, with or without isotope-charge channels)."""
    script = script or train_script()
    parts = recipe.split(":")
    if parts[0] in ("default", "grid"):
        bg = int(parts[2])
        mix = script.mixture(crop=crop, background_ions=None if bg == 40 else bg)
        for _w, c in mix:
            c.singly_charged = parts[1]
            c.evidence_channels = int(parts[3]) if len(parts) > 3 else 0
            if parts[0] == "grid":
                c.render_path = "grid"
        return mix
    make = {"v3": script.mixture_v3, "v4": script.mixture_v4}[parts[0]]
    mix = make(crop)
    for _w, c in mix:
        c.evidence_channels = 0
        c.singly_charged = "keep"
        c.iso_channels = parts[2] == "iso"
        c.measured_channel = parts[1] == "comb"
        c.crop_size = crop
    return mix


# ------------------------------------------------------------------------- label inputs
def label_cases(rng, n):
    """Label inputs with exact ties between rows (duplicated rows, equal values), many exact
    zeros, unseen bins, shares summing to 1 and past it, nz = 0..7, k = 1..3, odd lengths."""
    for _ in range(n):
        L = int(rng.integers(3, 300))
        nz = int(rng.integers(0, 8))
        k = int(rng.integers(1, 4))
        values = rng.choice([0.0, 0.0, 0.0, 1.0, 2.0, 0.5, 3.0], size=(nz, L))
        cm = values * (1.0 if rng.random() < 0.5 else rng.random((nz, L)))
        if nz >= 2 and rng.random() < 0.5:
            cm[1] = cm[0]  # exact ties between two rows
        if nz >= 3 and rng.random() < 0.3:
            cm[2, ::2] = cm[1, ::2]  # and in every other bin of a third
        obs = rng.choice([0.0, 0.0, 1.0, 2.0, 5.0], size=L) * (
            1.0 if rng.random() < 0.5 else rng.random(L) * 4)
        if nz and rng.random() < 0.2:
            obs = cm.sum(axis=0)  # shares summing to 1, to rounding
        noise = np.clip(obs - (cm.sum(axis=0) if nz else 0.0), 0.0, None)
        charges = rng.choice(np.arange(1, 60), size=nz, replace=False)
        yield cm, charges, noise, dict(k=k, noise_level=float(rng.choice([0.0, 0.5, 1.0])),
                                       observed=obs)


# ------------------------------------------------------------------------------- digests
def _feed(h, a) -> None:
    a = np.ascontiguousarray(a)
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())


def frame_digest(fs) -> str:
    """Everything a training frame carries: features, labels, crop, every value of its meta
    (whole spectrum and scans included) and its truths."""
    h = hashlib.sha256()
    for a in (fs.features, fs.topk_z, fs.topk_w, fs.heat):
        _feed(h, a)
    meta = dict(fs.meta)
    raw, scans = meta.pop("raw", None), meta.pop("scans", None)
    h.update(repr((int(fs.b0), sorted((k, repr(v)) for k, v in meta.items()))).encode())
    for a in raw or ():
        _feed(h, a)
    for mz, y in scans or ():
        _feed(h, mz)
        _feed(h, y)
    h.update(json.dumps([t.to_dict() for t in fs.truths], sort_keys=True).encode())
    return h.hexdigest()[:24]


def sort_canary() -> str:
    """How numpy's argsort orders tied values on this machine (it differs between its sorts:
    AVX-512, AVX2, baseline). Frame digests recorded with one sort hold only where it agrees."""
    rng = np.random.default_rng(3)
    h = hashlib.sha256()
    for n in (2, 3, 4, 7, 16, 17, 40, 101):
        cols = np.zeros((n, 24))
        for j in range(24):
            k = int(rng.integers(0, 3))
            cols[rng.choice(n, size=k, replace=False), j] = rng.random(k) + 0.1
        h.update(np.argsort(-cols, axis=0).astype(np.int64).tobytes())
    return h.hexdigest()[:24]
