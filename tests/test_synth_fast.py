"""The compiled synthesis paths against the code they replace, bit for bit.

``synth.fast``, the stick builder (``synth.render._fill_sticks``) and the adduct sampler
(``synth.adduct_sampler.adduct_fractions_by_charge``) must give exactly what the original code
gives - training frames are deterministic per seed, and a paused run resumes on the same
spectra - so each is compared here with its reference on random inputs (edge cases included),
and whole training frames with the original generator, kept below as
``_reference_frame_profile``.
"""

import subprocess
import sys

import numpy as np
import pytest

from lcmsdeconv.chem.adducts import AdductLibrary, AdductState
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core import profile as core_profile
from lcmsdeconv.nn import features as F
from lcmsdeconv.nn.grid import LogMzGrid
from lcmsdeconv.nn.grid import spectrum_to_grid as reference_spectrum_to_grid
from lcmsdeconv.synth import adduct_sampler, fast, frames, labels, render
from lcmsdeconv.synth.compounds import DEFAULT_CLASSES, ClassConfig
from lcmsdeconv.synth.config import SynthConfig

needs_numba = pytest.mark.skipif(not fast.HAVE_NUMBA, reason="numba is not installed")


def same(a, b):
    """Equal bit for bit: dtype, shape and bytes (so -0.0 is not 0.0 and NaN payloads count)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def configs():
    """The training mixture's four kinds of spectra (scripts/train_charge_net.py)."""
    proteins = SynthConfig(classes=[ClassConfig("peptide", (2000.0, 90000.0), 3.0),
                                    ClassConfig("peg", (1000.0, 15000.0), 1.0)],
                           mode="rplc", polarity=1)
    oligos = SynthConfig(classes=[ClassConfig(c, (2000.0, 25000.0))
                                  for c in ("dna", "rna", "ps_dna", "ps_rna")],
                         mode="iprp", polarity=-1)
    orbitrap = SynthConfig(classes=[ClassConfig(c, (1500.0, 12000.0)) for c in ("ps_dna", "ps_rna")],
                           mode="iprp", polarity=-1, instrument_kind="orbitrap",
                           resolution_range=(35000.0, 280000.0))
    general = SynthConfig(classes=list(DEFAULT_CLASSES), mode="rplc", polarity=1)
    return [proteins, oligos, orbitrap, general]


# --------------------------------------------------------------------------- adduct states
@pytest.mark.parametrize("mode,polarity", [("rplc", 1), ("iprp", -1), ("ip-hilic", -1),
                                           ("hilic", 1), ("none", 1)])
def test_adduct_fractions_equal_the_state_by_state_construction(mode, polarity):
    rng = np.random.default_rng(1)
    for max_per_type, max_total in [(3, 4), (2, 3), (1, 1), (3, 0), (4, 6)]:
        lib = AdductLibrary.from_mode(mode, polarity, max_per_type=max_per_type,
                                      max_total=max_total)
        for trial in range(12):
            props = {n: float(rng.choice([0.0, rng.uniform(0, 0.5), 1e-300, 0.999, 1.0]))
                     for n in lib.names()}
            if trial == 0:
                props = {}
            zs = [int(z) for z in rng.permutation(np.arange(1, 101))[:15]]
            with np.errstate(all="ignore"):
                got = adduct_sampler.adduct_fractions_by_charge(zs, props, lib)
                assert list(got) == zs
                for z in zs:
                    ref = adduct_sampler._adduct_fractions_reference(z, props, lib, rng)
                    one = adduct_sampler.adduct_fractions_for_charge(z, props, lib, rng)
                    for d in (got[z], one):
                        assert list(d) == list(ref)  # the same states in the same order
                        assert [type(v) for v in d.values()] == [type(v) for v in ref.values()]
                        same(np.array(list(d.values()), dtype=float),
                             np.array(list(ref.values()), dtype=float))


# ------------------------------------------------------------------------------- sticks
def scene_with_background(cfg, seed, n_bg=25):
    rng = np.random.default_rng(seed)
    lib = AdductLibrary.from_mode(cfg.mode, cfg.polarity, max_per_type=cfg.adduct_max_per_type,
                                  max_total=cfg.adduct_max_total)
    scene = frames.sample_scene(cfg, rng, lib)
    bg = frames._background_ions(rng, n_bg, 150.0, 3000.0, 1e5, cfg.polarity)
    return render.Scene(scene.components + bg, cfg.polarity, lib)


def same_sticks(a, b):
    for f in ("mz", "intensity", "comp_id", "z", "adduct_mass"):
        same(getattr(a, f), getattr(b, f))


def test_sticks_equal_the_per_state_construction():
    for k, cfg in enumerate(configs()):
        for seed in range(4):
            scene = scene_with_background(cfg, 100 * k + seed)
            ref = render._build_sticks_reference(scene)
            same_sticks(render.build_sticks(scene), ref)
            # a scene's parts followed by those of components appended to it
            cut = len(scene.components) // 2
            parts = render.stick_parts(scene.components[:cut], scene.polarity)
            parts += render.stick_parts(scene.components[cut:], scene.polarity, first_id=cut)
            same_sticks(render.sticks_from_parts(parts), ref)
    assert len(render.build_sticks(render.Scene([], 1))) == 0
    same_sticks(render.build_sticks(render.Scene([], 1)), render._build_sticks_reference(render.Scene([], 1)))


def test_sticks_of_components_the_compiled_loop_does_not_take():
    scene = scene_with_background(configs()[0], 7)
    ci = scene.components[0]
    pat = ci.compound.pattern
    odd = render.ComponentInstance(
        type(ci.compound)(ci.compound.mass, ci.compound.compound_class,
                          type(pat)(pat.masses.astype(np.float32), pat.abundances)),
        ci.intensity, ci.charges, ci.adducts)
    tiny = render.ComponentInstance(ci.compound, 1e-12, ci.charges, ci.adducts)  # below min_stick
    quiet = render.ComponentInstance(ci.compound, ci.intensity, {3: 0.0, 4: 1.0},
                                     {4: {AdductState(): 1}})
    comps = [odd, *scene.components[1:4], tiny, quiet, odd]
    s = render.Scene(comps, scene.polarity, scene.library)
    same_sticks(render.build_sticks(s), render._build_sticks_reference(s))


# ------------------------------------------------------------------------ resampling
def axis_cases(rng):
    """Profile axes as training frames and the pipeline meet them: dense, thresholded into
    segments, with repeated points, unsorted, a single point, none inside the grid."""
    inst = InstrumentModel("tof", float(rng.uniform(8000, 60000)))
    dense = inst.profile_axis(400.0, 1600.0)
    y = rng.exponential(1.0, dense.size) * (rng.random(dense.size) < 0.4)
    keep = np.convolve(y > 0.8, np.ones(5), mode="same") > 0
    thresholded = dense[keep]
    repeated = np.sort(np.concatenate([thresholded, thresholded[::97]]))
    unsorted = rng.permutation(thresholded[:5000])
    return [dense, thresholded, repeated, unsorted, dense[:2], dense[:66], dense[:131],
            np.array([700.0]), np.array([5000.0, 5100.0])]


def test_sample_widths_equal_reference():
    rng = np.random.default_rng(2)
    for _ in range(3):
        for mz in axis_cases(rng):
            u = np.log(np.asarray(mz) - 1.00727646688)
            same(fast.sample_widths(u), core_profile.sample_widths(u))
    for n in (0, 1, 2, 3, 64, 65, 66, 129, 130, 131, 1000):
        u = np.cumsum(rng.random(n) + 0.01)
        same(fast.sample_widths(u), core_profile.sample_widths(u))


def outcome(f, *args):
    """What a call gives: its value, or the type of the exception it raises."""
    try:
        return f(*args)
    except Exception as e:  # the reference's own failures must be repeated, not hidden
        return type(e)


def test_scan_resampling_and_label_rows_equal_resample_profile():
    rng = np.random.default_rng(3)
    for trial in range(3):
        grid = LogMzGrid(390.0, 1650.0, 2e-5, 1 if trial % 2 else -1)
        for mz in axis_cases(rng):
            for y in (rng.exponential(5.0, mz.size), np.zeros(mz.size),
                      rng.exponential(5.0, mz.size) * (rng.random(mz.size) < 0.3)):
                ref = outcome(grid.resample_profile, mz, y)
                got = outcome(fast.resample_profile, grid, mz, y)
                if isinstance(ref, type):  # (one point inside the grid: numpy 2.5 refuses it)
                    assert got is ref
                    continue
                same(got, ref)
                for b0, length in ((0, grid.size), (1000, 4096), (grid.size - 100, 512)):
                    want = np.zeros(length)
                    seg = ref[b0:min(b0 + length, grid.size)]
                    want[:seg.size] = seg
                    same(fast.ProfileRows(grid, mz, b0, length).row(y), want)


def test_training_spectrum_to_grid_equals_reference():
    rng = np.random.default_rng(4)
    for trial in range(6):
        inst = InstrumentModel("orbitrap" if trial % 2 else "tof", float(rng.uniform(1e4, 1e5)))
        axis = inst.profile_axis(float(rng.uniform(300, 600)), float(rng.uniform(1500, 3000)))
        clean = rng.exponential(30.0, axis.size) * (rng.random(axis.size) < 0.2)
        scans = []
        for _ in range(int(rng.integers(1, 5))):
            s = rng.poisson(clean / 3).astype(float) * 3 + np.abs(rng.normal(0, 2.0, axis.size))
            s[~(np.convolve(s >= 4.0, np.ones(5), mode="same") > 0)] = 0.0
            scans.append(s)
        total = np.sum(scans, axis=0)
        same(frames._sum_scans(scans), total)
        mz, y = fast.positive_points(axis, total)
        same(mz, axis[total > 0])
        same(y, total[total > 0])
        points = [fast.positive_points(axis, s) for s in scans]
        spectra = frames.scan_spectra(points, 1)
        for got, ref in zip(fast.spectrum_to_grid(mz, y, 1, 2e-5, 10000.0, frames=spectra),
                            reference_spectrum_to_grid(mz, y, 1, 2e-5, 10000.0, True,
                                                       frames=spectra)):
            if isinstance(ref, LogMzGrid):
                assert got == ref and got.size == ref.size
            else:
                same(got, ref)


# ------------------------------------------------------------------------------ scans
def reference_scan(rng, clean, w, gain, esig, sat_level, thr, pad):
    lam = np.clip(clean * w / gain, 0.0, None)
    scan = rng.poisson(lam).astype(np.float64) * gain + np.abs(rng.normal(0.0, esig, clean.size))
    if sat_level is not None:
        scan = np.minimum(scan, sat_level)
    if thr > 0:
        above = scan >= thr
        keep = np.convolve(above, np.ones(2 * pad + 1), mode="same") > 0
        scan[~keep] = 0.0
    return scan


def test_noisy_scans_equal_reference_and_leave_the_generator_where_it_did():
    assert fast._poisson_skips_zeros()
    rng = np.random.default_rng(5)
    for trial in range(40):
        n = int(rng.integers(10, 20000))
        clean = rng.exponential(float(rng.choice([0.1, 10.0, 1e4])), n)
        clean *= rng.random(n) < rng.random()
        args = (float(rng.uniform(0.05, 1.0)), float(rng.uniform(1, 50)), float(rng.uniform(0.5, 5)),
                None if trial % 3 else float(rng.uniform(1, 100)),
                0.0 if trial % 5 == 0 else float(rng.uniform(0, 10)), int(rng.integers(0, 4)))
        seed = int(rng.integers(1 << 62))
        g1, g2 = np.random.default_rng(seed), np.random.default_rng(seed)
        same(fast.noisy_scan(g1, clean, *args), reference_scan(g2, clean, *args))
        assert g1.bit_generator.state == g2.bit_generator.state


# ------------------------------------------------------------------------------ features
@pytest.mark.parametrize("evidence_channels", [0, 8])
def test_featurize_equals_reference(evidence_channels):
    rng = np.random.default_rng(6)
    L = 6000
    windows = [np.zeros(L), rng.exponential(20.0, L),
               rng.exponential(20.0, L) * (rng.random(L) < 0.2), np.full(L, 1e300)]
    nan = rng.exponential(20.0, L)
    nan[[5, 400, 3000]] = np.nan
    inf = rng.exponential(20.0, L)
    inf[[7, 2000]] = np.inf
    windows += [nan, inf]
    with np.errstate(all="ignore"):
        for window in windows:
            for z_max in (1, 2, 30, 64):
                for step in (2e-5, 7e-5):
                    args = (window, float(rng.uniform(0.1, 10)), step, z_max, evidence_channels)
                    same(fast.featurize(*args), F.featurize(*args))


@needs_numba
def test_comb_kernels_equal_the_evidence_they_stand_for():
    """In float64, before the features are rounded to float32: the evidence rows, and the best
    charge and its value at every bin."""
    rng = np.random.default_rng(8)
    L = 5000
    for trial in range(6):
        x = rng.uniform(0.0, 5.0, L) * (rng.random(L) < 0.5)
        if trial == 5:
            x[[3, 999, 4000]] = np.nan
        for z_max, step in ((64, 2e-5), (2, 2e-5), (30, 7e-5)):
            ev = F.comb_evidence(x, step, z_max)
            table, count = fast._comb_table(step, z_max)
            rows = np.zeros((z_max, L))
            fast._comb_rows(x, table, count, z_max, rows)
            same(rows, ev)
            val, idx = np.empty(L), np.empty(L, dtype=np.int64)
            fast._comb_best(x, table, count, z_max, val, idx)
            want = np.argmax(ev[1:z_max], axis=0)
            same(idx, want.astype(np.int64))
            same(val, ev[1:z_max][want, np.arange(L)])


# ------------------------------------------------------------------------------ labels
def test_heat_envelope_equals_the_maximum_loop(monkeypatch):
    rng = np.random.default_rng(7)
    for _ in range(10):
        L = 3000
        nz = int(rng.integers(1, 6))
        cmat = rng.exponential(1.0, (nz, L)) * (rng.random((nz, L)) < 0.3)
        args = (cmat, np.arange(2, 2 + nz), rng.exponential(0.1, L))
        kw = dict(noise_level=float(rng.uniform(0, 0.5)), observed=cmat.sum(axis=0) + 0.01)
        got = labels.build_labels(*args, **kw)
        with monkeypatch.context() as m:
            m.setattr(fast, "HAVE_NUMBA", False)
            ref = labels.build_labels(*args, **kw)
        for k in ("topk_z", "topk_w", "heat"):
            same(got[k], ref[k])


# ------------------------------------------------------------------------- whole frames
def _reference_background_ions(rng, n, mz_lo, mz_hi, scale, polarity):
    """frames._background_ions as it was."""
    from lcmsdeconv.chem.classes import class_isotope_pattern
    from lcmsdeconv.synth.spec import Compound

    out = []
    for _ in range(n):
        mz = float(np.exp(rng.uniform(np.log(max(mz_lo, 100.0)), np.log(max(mz_hi, 101.0)))))
        mass = mz - float(polarity) * 1.00727646688
        if mass <= 50:
            continue
        pat = class_isotope_pattern(mass, "small_molecule")
        comp = Compound(mass, "small_molecule", pat.shifted(mass - pat.average_mass),
                        name="background", kind="contaminant")
        inten = scale * float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-2))))
        out.append(render.ComponentInstance(comp, inten, {1: 1.0}, {1: {AdductState(): 1.0}}))
    return out


def _reference_frame_profile(cfg, rng, library):
    """frames.generate_frame_profile as it was, on the reference functions throughout."""
    from lcmsdeconv.synth.frames import (
        FrameSample,
        Scene,
        _choose_crop,
        _without_singly_charged,
        sample_instrument,
        sample_scene,
        scan_spectra,
    )
    from lcmsdeconv.synth.render import render_profile

    build_sticks = render._build_sticks_reference
    inst = sample_instrument(cfg, rng)
    scene = sample_scene(cfg, rng, library)
    sticks = build_sticks(scene)
    polarity = cfg.polarity
    if len(sticks):
        lo = max(60.0, float(sticks.mz.min()) * 0.98)
        hi = min(cfg.grid_mz_max * 0.995, float(sticks.mz.max()) * 1.02)
    else:
        lo, hi = 300.0, 3000.0
    if hi <= lo * 1.05:
        lo, hi = max(60.0, lo / 1.5), min(cfg.grid_mz_max * 0.995, hi * 1.5)
    n_bg = int(rng.integers(cfg.background_ions_range[0], cfg.background_ions_range[1] + 1))
    if cfg.singly_charged == "drop":
        n_bg = 0
        sticks = _without_singly_charged(sticks)
    if n_bg:
        base = max((c.intensity for c in scene.components), default=1e5)
        bg = _reference_background_ions(rng, n_bg, lo, hi, base, polarity)
        if bg:
            scene = Scene(scene.components + bg, polarity, library, meta=scene.meta)
            sticks = build_sticks(scene)
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
    total = np.sum(frames_int, axis=0)
    kept = total > 0
    scans = [(axis[s > 0], s[s > 0]) for s in frames_int]
    grid, observed, noise_sigma = reference_spectrum_to_grid(
        axis[kept], total[kept], polarity, cfg.grid_step, cfg.grid_mz_max, True,
        frames=scan_spectra(scans, polarity))
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
    for z in charges:
        sel = (sticks.z == z) & (sticks.mz > crop_lo * 0.995) & (sticks.mz < crop_hi * 1.005)
        if not sel.any() or sub_axis.size < 2:
            continue
        sub = type(sticks)(sticks.mz[sel], sticks.intensity[sel], sticks.comp_id[sel],
                           sticks.z[sel], sticks.adduct_mass[sel])
        prof = render_profile(sub, sub_axis, inst)
        row = np.zeros(L)
        res = grid.resample_profile(sub_axis, prof)[b0:min(b1, grid.size)]
        row[: res.size] = res
        if row.max() > 0:
            rows.append(row)
            zs.append(int(z))
    cmat = np.array(rows) if rows else np.zeros((0, L))
    signal = cmat.sum(axis=0) if cmat.shape[0] else np.zeros(L)
    noise_part = np.clip(window - signal, 0.0, None)
    feats = F.featurize(window, noise_sigma, cfg.grid_step, z_max=min(cfg.z_max, 64),
                        evidence_channels=cfg.evidence_channels)
    lab = labels.build_labels(cmat, np.array(zs, dtype=int), noise_part, noise_level=noise_sigma,
                              observed=window)
    return FrameSample(feats, lab["topk_z"], lab["topk_w"], lab["heat"], b0, scene.truths(),
                       {"noise_sigma": noise_sigma, "n_sum": n_sum, "raw": (axis[kept], total[kept]),
                        "scans": scans})


def reference_frame(monkeypatch, cfg, seed):
    """The original generator, with the state-by-state adduct sampler and build_labels' numpy
    loop."""
    def by_charge(charges, props, library):
        rng = np.random.default_rng(0)  # not used by the reference
        return {z: adduct_sampler._adduct_fractions_reference(z, props, library, rng)
                for z in charges}

    with monkeypatch.context() as m:
        m.setattr(adduct_sampler, "adduct_fractions_by_charge", by_charge)
        m.setattr(fast, "HAVE_NUMBA", False)
        library = AdductLibrary.from_mode(cfg.mode, cfg.polarity, max_per_type=cfg.adduct_max_per_type,
                                          max_total=cfg.adduct_max_total)
        return _reference_frame_profile(cfg, np.random.default_rng(seed), library)


def same_frames(a, b):
    for k in ("features", "topk_z", "topk_w", "heat"):
        same(getattr(a, k), getattr(b, k))
    assert a.b0 == b.b0
    assert a.meta["noise_sigma"] == b.meta["noise_sigma"] and a.meta["n_sum"] == b.meta["n_sum"]
    for x, y in zip(a.meta["raw"], b.meta["raw"]):
        same(x, y)
    assert len(a.meta["scans"]) == len(b.meta["scans"])
    for (x1, y1), (x2, y2) in zip(a.meta["scans"], b.meta["scans"]):
        same(x1, x2)
        same(y1, y2)
    assert [t.to_dict() for t in a.truths] == [t.to_dict() for t in b.truths]


@pytest.mark.parametrize("stage", [1, 2])
def test_training_frames_equal_the_original_generator(monkeypatch, stage):
    for k, cfg in enumerate(configs()):
        cfg.singly_charged = "keep" if stage == 2 else "ignore"
        if stage == 2:
            cfg.background_ions_range = (0, 200)
        for seed in (11 * k + stage, 1000 + 7 * k + stage):
            got = frames.generate_frame(cfg, np.random.default_rng(seed))
            same_frames(got, reference_frame(monkeypatch, cfg, seed))


def test_frames_with_evidence_channels_and_without_singly_charged_ions(monkeypatch):
    for cfg, seed in zip(configs(), (3, 4, 5, 6)):
        cfg.evidence_channels = 6
        cfg.singly_charged = "drop" if seed % 2 else "ignore"
        same_frames(frames.generate_frame(cfg, np.random.default_rng(seed)),
                    reference_frame(monkeypatch, cfg, seed))


def test_frames_without_numba_equal_the_compiled_ones(monkeypatch):
    cfg = configs()[1]
    want = frames.generate_frame(cfg, np.random.default_rng(21))
    monkeypatch.setattr(fast, "HAVE_NUMBA", False)
    monkeypatch.setattr(render, "_HAVE_NUMBA", False)
    same_frames(frames.generate_frame(cfg, np.random.default_rng(21)), want)


# ------------------------------------------------------------------------------ warm-up
@needs_numba
def test_warm_up_compiles_every_kernel_a_frame_uses():
    """Data workers must find every kernel compiled: after the training warm-up, generating
    frames compiles nothing more (checked in a fresh process)."""
    code = """
import numpy as np
from lcmsdeconv.nn.train import _warm_up
from lcmsdeconv.synth import fast, render
from lcmsdeconv.synth.frames import generate_frame
from tests.test_synth_fast import configs
kernels = [getattr(fast, n) for n in dir(fast) if hasattr(getattr(fast, n), "signatures")]
kernels += [getattr(render, n) for n in dir(render) if hasattr(getattr(render, n), "signatures")]
cfgs = configs()
_warm_up([(1.0, c) for c in cfgs])
before = [set(k.signatures) for k in kernels]
for seed in range(6):
    for cfg in cfgs:
        cfg.background_ions_range = (0, 200)
        generate_frame(cfg, np.random.default_rng(seed))
new = [k.py_func.__name__ for k, b in zip(kernels, before) if set(k.signatures) - b]
assert not new, new
print("ok", len(kernels))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.startswith("ok")


def test_a_library_listing_a_type_twice_takes_the_reference():
    from lcmsdeconv.chem.adducts import ADDUCTS, AdductLibrary
    from lcmsdeconv.synth.adduct_sampler import (
        _adduct_fractions_reference,
        adduct_fractions_by_charge,
    )

    library = AdductLibrary([ADDUCTS["Na"], ADDUCTS["Na"], ADDUCTS["K"]], 3, 4)
    props = {"Na": 0.2, "K": 0.05}
    fast = adduct_fractions_by_charge([1, 2, 5], props, library)
    for z in (1, 2, 5):
        ref = _adduct_fractions_reference(z, props, library, None)
        assert list(fast[z]) == list(ref)
        assert all(fast[z][s] == ref[s] for s in ref)
