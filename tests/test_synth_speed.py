"""Faster training-frame synthesis against what it replaces, bit for bit.

The label builder (``synth.fast.build_labels``: no column-wise sort of the share matrix), the
label rows (``ProfileRows.row_into`` into one matrix, one reused render buffer, zero samples
skipped), the scans' rates (``fast._nonzero_rates``) and the comb (``fast._comb_best_sparse``,
``fast._comb_best_tiled``) must leave every training frame exactly as it was: training is
deterministic per seed, and a paused run resumes on the same spectra. Here each is compared with
its reference on adversarial inputs (ties of every kind for the labels, under each of numpy's
sorts; sparse and dense windows for the comb), whole frames of every recipe with the generator
as it was (``tests/_synth_reference.py``), and full-size frames with the digests of commit
2a5fcb5's own frames (``tests/data/ref_2a5fcb5.json``, made by
``tests/data/make_ref_2a5fcb5.py``), where numpy computes as on the recording machine.
"""

import ctypes
import gc
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.nn import features as F
from lcmsdeconv.nn.grid import LogMzGrid
from lcmsdeconv.synth import fast, frames, labels, render
from lcmsdeconv.synth.dataset import FrameDataset
from lcmsdeconv.synth.spec import Sticks

from ._digests import canary
from ._synth_reference import (
    frame_digest,
    label_cases,
    mixture,
    reference_frame,
    reference_item,
    reference_scan,
    sort_canary,
)

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((Path(__file__).resolve().parent / "data" / "ref_2a5fcb5.json").read_text())
needs_numba = pytest.mark.skipif(not fast.HAVE_NUMBA, reason="numba is not installed")


@pytest.fixture(autouse=True, scope="module")
def _give_back_freed_memory():
    """After this module's frames (whole 131 072-bin windows among them), give the memory freed
    back to the system. glibc keeps freed heap memory for the process to reuse, so a process
    running the whole suite holds what it once used - 1.7 GB from the whole-window ONNX export
    of tests/test_recipe_v3.py alone - while tests/test_training_workers.py starts worker
    processes; under a memory cap of 3.5 GB whether they fitted was a matter of luck. Without
    glibc this does nothing."""
    yield
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc
        pass


def same(a, b):
    """Equal bit for bit: dtype, shape and bytes (so -0.0 is not 0.0 and NaN payloads count)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def outcome(f, *args, **kw):
    """What a call gives: its value, or the type of the exception it raises."""
    try:
        return f(*args, **kw)
    except Exception as e:  # the reference's own failures must be repeated, not hidden
        return type(e)


def same_labels(got, ref):
    if isinstance(ref, type):
        assert got is ref
        return
    for key in ("topk_z", "topk_w", "heat"):
        same(got[key], ref[key])


# ------------------------------------------------------------------------------ labels
@needs_numba
def test_labels_equal_the_reference_with_ties_of_every_kind(monkeypatch):
    """Bins of all three kinds: top shares distinct (written directly), fewer nonzero shares
    than k with the rest exactly zero (numpy's order per order type), shares equal where the
    order matters (numpy per bin)."""
    kinds = np.zeros(3, dtype=np.int64)
    classify = fast._classify

    def counting(kk, n, count, top_v, top_r, unsure, topk_pos, topk_w, kind, *rest):
        ng = classify(kk, n, count, top_v, top_r, unsure, topk_pos, topk_w, kind, *rest)
        kinds[:] += np.bincount(kind, minlength=3)
        return ng

    monkeypatch.setattr(fast, "_classify", counting)
    rng = np.random.default_rng(123)
    for cm, charges, noise, kw in label_cases(rng, 2500):
        same_labels(fast.build_labels(cm, charges, noise, **kw),
                    labels.build_labels(cm, charges, noise, **kw))
    assert (kinds > 1000).all(), kinds


@needs_numba
def test_labels_of_inputs_outside_the_compiled_path_are_the_reference():
    """NaN and infinity in the observed spectrum, negative, -0.0, infinite and NaN row values,
    float32 rows, rows in Fortran order, no observed spectrum, k = 4, as many classes as the
    order-type table takes and more, short crops, NaN or negative noise: the reference's result
    or its exception."""
    rng = np.random.default_rng(9)
    L = 64
    cm = rng.choice([0.0, 1.0, 2.0], size=(4, L)) * rng.random((4, L))
    charges = np.array([2, 3, 5, 9])
    obs = cm.sum(axis=0) * rng.uniform(0.5, 1.5, L)
    noise = np.clip(obs - cm.sum(axis=0), 0.0, None)
    cases = []
    for edit in ("nan_obs", "inf_obs", "negative", "negzero", "inf_row", "nan_row", "float32",
                 "order", "fortran", "none", "k4", "most_classes", "classes", "short",
                 "nan_noise", "negative_noise"):
        c, o, nn, ch, kw = cm.copy(), obs.copy(), noise.copy(), charges, {}
        if edit == "nan_obs":
            o[[3, 9]] = np.nan
        elif edit == "inf_obs":
            o[[4, 10]] = np.inf
        elif edit == "negative":
            c[1, 7] = -0.25
        elif edit == "negzero":
            c[2, 11] = -0.0
        elif edit == "inf_row":
            c[0, 5] = np.inf
        elif edit == "nan_row":
            c[3, 6] = np.nan
        elif edit == "float32":
            c = c.astype(np.float32)
        elif edit in ("order", "fortran"):
            # sums that depend on the order of additions, seen in the third share (the
            # remainder); in Fortran order numpy sums the rows pairwise
            c = np.zeros((12, L))
            c[0], c[1], c[2:] = 0.5, 0.25, 4e-17
            c = np.asfortranarray(c) if edit == "fortran" else c
            o, ch = np.full(L, 0.7), np.arange(2, 14)
            nn = np.zeros(L)
        elif edit == "none":
            o = None
        elif edit == "k4":
            kw["k"] = 4
        elif edit in ("most_classes", "classes"):  # the largest table, and one past it
            c = np.repeat(c, 80, axis=0)[:fast.LABEL_MAX_CLASSES + (-1 if edit[0] == "m" else 3)]
            ch = np.arange(2, 2 + c.shape[0])
        elif edit == "short":
            c, o, nn = c[:, :2], o[:2], nn[:2]
        elif edit == "nan_noise":
            nn[[1, 2]] = np.nan
        elif edit == "negative_noise":
            nn[[1, 2]] = -1.0
        cases.append((edit, c, ch, nn, dict(kw, noise_level=0.2, observed=o)))
    with np.errstate(all="ignore"):
        for edit, c, ch, nn, kw in cases:
            got = outcome(fast.build_labels, c, ch, nn, **kw)
            ref = outcome(labels.build_labels, c, ch, nn, **kw)
            assert type(got) is type(ref), edit
            same_labels(got, ref)


@needs_numba
@pytest.mark.parametrize("tied_bins", [1, 2, 5])
def test_labels_of_tied_bins_sum_their_rows_as_the_reference(tied_bins):
    """Bins whose top shares tie are computed again apart from the others; their sums over
    twelve rows must be taken row after row, as numpy sums the whole matrix (a single column it
    sums pairwise, and the order of additions shows in the remainder's share)."""
    L, nz = 16, 12
    cm = np.zeros((nz, L))
    for b in range(2, 2 + tied_bins):
        cm[:, b] = [0.5, 0.5] + [1e-16] * (nz - 2)  # two equal shares, and small ones
    obs = np.full(L, 0.9)
    noise = np.clip(obs - cm.sum(axis=0), 0.0, None)
    charges = np.arange(2, 2 + nz)
    same_labels(fast.build_labels(cm, charges, noise, observed=obs),
                labels.build_labels(cm, charges, noise, observed=obs))


_SORT_CHECK = """
import numpy as np
from lcmsdeconv.synth import fast, labels
from tests._synth_reference import label_cases, sort_canary
rng = np.random.default_rng(321)
for cm, charges, noise, kw in label_cases(rng, 600):
    a = fast.build_labels(cm, charges, noise, **kw)
    b = labels.build_labels(cm, charges, noise, **kw)
    for key in ("topk_z", "topk_w", "heat"):
        assert a[key].tobytes() == b[key].tobytes(), key
print("ok", sort_canary())
"""


@needs_numba
@pytest.mark.skipif(platform.machine().lower() not in ("x86_64", "amd64"),
                    reason="numpy's sorts are told apart on x86 processors")
def test_labels_equal_the_reference_under_each_of_numpys_sorts():
    """numpy orders tied values differently in its AVX-512, AVX2 and baseline sorts, and picks
    one by the processor; the labels must be the reference's under each (a processor without
    AVX-512, as most desktop processors are, sorts as the second)."""
    from numpy._core._multiarray_umath import __cpu_dispatch__ as dispatch

    groups = [g for g in ("X86_V4", "AVX512_ICL", "AVX512_SPR") if g in dispatch]
    settings = ["", " ".join(groups), " ".join(["X86_V3", *groups])]
    seen = []
    for disable in settings:
        out = subprocess.run([sys.executable, "-c", _SORT_CHECK], capture_output=True, text=True,
                             cwd=str(ROOT), env={**os.environ, "NPY_DISABLE_CPU_FEATURES": disable})
        if out.returncode != 0 and "NPY_DISABLE_CPU_FEATURES" in out.stderr:
            continue  # numpy cannot turn those off on this processor
        assert out.returncode == 0, (disable, out.stderr[-3000:])
        seen.append(out.stdout.split()[-1])
    assert seen


# -------------------------------------------------------------------------------- comb
def windows(rng):
    """Log-intensity channels as the comb reads them: lengths around the tile, shares of
    nonzero bins either side of the sparse threshold, runs of zeros between peaks."""
    out = []
    for L in (5, 2047, 2048, 2049, 6000):
        for share in (0.0, 0.002, 0.05, 0.29, 0.31, 0.6, 1.0):
            x = rng.uniform(0.0, 5.0, L) * (rng.random(L) < share)
            out.append(x)
        runs = np.zeros(L)
        for start in rng.integers(0, L, 12):
            runs[start:start + 9] = rng.uniform(0.5, 5.0)
        out.append(runs)
    out.append(np.full(3000, 5.0))
    return out


@needs_numba
def test_sparse_and_tiled_comb_equal_the_comb_they_stand_for():
    rng = np.random.default_rng(10)
    for x in windows(rng):
        n = x.size
        for z_max, step in ((64, 2e-5), (2, 2e-5), (3, 7e-5), (30, 7e-5)):
            table, count = fast._comb_table(step, z_max)
            want_v, want_i = np.empty(n), np.empty(n, dtype=np.int64)
            fast._comb_best(x, table, count, z_max, want_v, want_i)
            for kernel, extra in ((fast._comb_best_sparse, ()),
                                  (fast._comb_best_tiled, (fast.COMB_TILE,)),
                                  (fast._comb_best_tiled, (777,))):
                val, idx = np.full(n, np.nan), np.full(n, -7, dtype=np.int64)
                assert kernel(x, table, count, z_max, val, idx, *extra)
                same(val, want_v)
                same(idx, want_i)


@needs_numba
def test_comb_kernels_refuse_inputs_they_do_not_equal_the_reference_on():
    """The sparse comb takes finite, non-negative values without -0.0; the tiled comb values
    bounded well inside the float range; both every charge with a shift. The rest goes to the
    reference kernel, through which featurize still equals nn.features.featurize."""
    table, count = fast._comb_table(2e-5, 64)
    base = np.zeros(4000)
    base[::37] = 2.0
    for edit, sparse_ok, tiled_ok in (("negzero", False, True), ("negative", False, True),
                                      ("nan", False, False), ("inf", False, False),
                                      ("huge", False, False)):
        x = base.copy()
        x[100] = {"negzero": -0.0, "negative": -1.0, "nan": np.nan, "inf": np.inf,
                  "huge": 1e301}[edit]
        val, idx = np.empty(x.size), np.empty(x.size, dtype=np.int64)
        assert fast._comb_best_sparse(x, table, count, 64, val, idx) is sparse_ok, edit
        assert fast._comb_best_tiled(x, table, count, 64, val, idx, 2048) is tiled_ok, edit
    wide, wcount = fast._comb_table(1.0, 64)  # shifts that round to zero: charges without any
    assert (wcount[2:] == 0).any()
    val, idx = np.empty(base.size), np.empty(base.size, dtype=np.int64)
    assert not fast._comb_best_sparse(base, wide, wcount, 64, val, idx)
    assert not fast._comb_best_tiled(base, wide, wcount, 64, val, idx, 2048)
    same(fast.featurize(base, 1.0, 1.0, 64), F.featurize(base, 1.0, 1.0, 64))


@needs_numba
@pytest.mark.parametrize("share", [0.0, 2.0])  # every window tiled, every one sparse
def test_featurize_equals_the_reference_through_either_comb(monkeypatch, share):
    rng = np.random.default_rng(11)
    monkeypatch.setattr(fast, "COMB_SPARSE_SHARE", share)
    for L in (300, 4097):
        for keep in (0.02, 0.3, 1.0):
            window = rng.exponential(30.0, L) * (rng.random(L) < keep)
            for z_max in (2, 64):
                args = (window, float(rng.uniform(0.5, 5.0)), 2e-5, z_max)
                same(fast.featurize(*args), F.featurize(*args))


# ------------------------------------------------------------------------------- scans
@needs_numba
def test_nonzero_rates_are_numpys():
    """``np.clip(clean * w / gain, 0, None)`` and its nonzero positions: negative values, -0.0,
    NaN and infinity included, the factors given as numpy scalars of other types."""
    rng = np.random.default_rng(12)
    clean = rng.exponential(10.0, 5000) * (rng.random(5000) < 0.3)
    clean[[3, 30, 300]] = -0.0
    clean[[4, 40]] = -2.5
    clean[[5, 50]] = np.nan
    clean[[6, 60]] = np.inf
    for w, gain in ((0.3, 7.0), (np.float32(0.2), np.float64(3.5)), (1, 2), (0.5, np.int64(4))):
        lam = np.clip(clean * w / gain, 0.0, None)
        want = np.flatnonzero(lam != 0)
        idx, vals = np.empty(clean.size, dtype=np.int64), np.empty(clean.size)
        m = fast._nonzero_rates(clean, float(w), float(gain), idx, vals)
        same(idx[:m], want.astype(np.int64))
        same(vals[:m], lam[want])


@needs_numba
def test_noisy_scans_equal_numpys_lines_and_draw_the_same():
    rng = np.random.default_rng(13)
    for trial in range(30):
        n = int(rng.integers(10, 30000))
        clean = rng.exponential(float(rng.choice([0.01, 3.0, 3e4])), n) * (rng.random(n) < 0.4)
        if trial % 4 == 0:
            clean[rng.integers(0, n, 5)] = -0.0
        args = (float(rng.uniform(0.05, 1.0)), float(rng.uniform(1, 50)),
                float(rng.uniform(0.5, 5)), None if trial % 3 else float(rng.uniform(1, 100)),
                0.0 if trial % 5 == 0 else float(rng.uniform(0, 10)), int(rng.integers(0, 4)))
        seed = int(rng.integers(1 << 62))
        g1, g2 = np.random.default_rng(seed), np.random.default_rng(seed)
        same(fast.noisy_scan(g1, clean, *args), reference_scan(g2, clean, *args))
        assert g1.bit_generator.state == g2.bit_generator.state
    bad = np.array([1.0, np.nan, 2.0])  # numpy refuses a NaN rate; so does the scan
    args = (0.5, 2.0, 1.0, None, 0.0, 2)
    assert outcome(fast.noisy_scan, np.random.default_rng(0), bad, *args) is \
        outcome(reference_scan, np.random.default_rng(0), bad, *args)


# ---------------------------------------------------------------------------- label rows
@needs_numba
def test_label_rows_written_in_place_equal_resample_profile():
    """Rows of an axis that runs past the grid at both ends (the points inside one slice of
    it), of profiles with exact zeros, -0.0, negative values, NaN and infinity, written into
    rows of one matrix: the crop of ``grid.resample_profile``, also without skipping zeros."""
    rng = np.random.default_rng(14)
    grid = LogMzGrid(500.0, 1500.0, 2e-5, 1)
    inst = InstrumentModel("orbitrap", 120000.0)
    axis = inst.profile_axis(450.0, 1700.0)
    assert axis[0] < grid.mz_min and axis[-1] > grid.mz_max
    profiles = []
    for _ in range(4):
        y = rng.exponential(5.0, axis.size) * (rng.random(axis.size) < 0.05)
        profiles.append(y)
    odd = profiles[0].copy()
    odd[rng.integers(0, axis.size, 50)] = -0.0
    odd[rng.integers(0, axis.size, 50)] = -3.0
    odd[rng.integers(0, axis.size, 3)] = np.nan
    odd[rng.integers(0, axis.size, 3)] = np.inf
    profiles += [odd, np.zeros(axis.size), profiles[1].astype(np.float32)]
    for b0, length in ((0, grid.size), (12000, 8192), (grid.size - 300, 1024)):
        rows = fast.ProfileRows(grid, axis, b0, length)
        assert rows.run is not None and rows.skip_zeros
        plain = fast.ProfileRows(grid, axis, b0, length)
        plain.skip_zeros = False
        mat = np.zeros((len(profiles), length))
        with np.errstate(all="ignore"):
            for k, y in enumerate(profiles):
                want = np.zeros(length)
                seg = grid.resample_profile(axis, y)[b0:b0 + length]
                want[:seg.size] = seg
                row = mat[k]
                assert rows.row_into(y, row) is row
                same(mat[k], want)
                same(plain.row(y), want)


def reach_cases(rng):
    """Sticks and axes for profile_reach: dense and thresholded axes (segments with gaps), with
    sticks clustered, spread, on axis points, in gaps (drawn from their far insertion point),
    near and beyond both ends, at line shapes narrower than the spacing and wide ones."""
    for kind, res, shape, eta in (("tof", 20000.0, "gaussian", 0.0), ("tof", 9000.0, "mixed", 0.3),
                                  ("orbitrap", 240000.0, "gaussian", 0.0),
                                  ("orbitrap", 40000.0, "mixed", 0.2)):
        inst = InstrumentModel(kind, res, shape=shape, eta=eta)
        dense = inst.profile_axis(500.0, 1500.0)
        keep = np.convolve(rng.random(dense.size) < 0.02, np.ones(9), mode="same") > 0
        for axis in (dense, dense[keep], dense[:3]):
            lo, hi = float(axis[0]), float(axis[-1])
            for spread, n in ((0.5, 1), (2.0, 30), (300.0, 200), (1200.0, 50)):
                centre = rng.uniform(lo - 50.0, hi + 50.0)
                mz = np.maximum(centre + rng.uniform(-spread, spread, n), 60.0)
                if n > 1:
                    mz[0] = axis[int(rng.integers(0, axis.size))]  # on a point
                yield inst, axis, mz
            yield inst, axis, np.array([lo - 30.0, hi + 30.0])  # beyond both ends
            yield inst, axis, np.array([hi + 1e-9, hi + 0.01])


def test_rendering_on_a_profiles_reach_equals_rendering_on_the_whole_axis():
    rng = np.random.default_rng(16)
    for inst, axis, mz in reach_cases(rng):
        n = mz.size
        sticks = Sticks(mz, rng.exponential(1e4, n), np.zeros(n, dtype=np.int64),
                        np.full(n, 2, dtype=np.int64), np.zeros(n))
        whole = render.render_profile(sticks, axis, inst)
        lo, hi = render.profile_reach(sticks, axis, inst)
        assert 0 <= lo <= hi <= axis.size
        assert not whole[:lo].any() and not whole[hi:].any()
        same(render.render_profile(sticks, axis[lo:hi], inst), whole[lo:hi])
    empty = Sticks(np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64),
                   np.zeros(0, dtype=np.int64), np.zeros(0))
    assert render.profile_reach(empty, axis, inst) == (0, 0)


@needs_numba
def test_label_rows_of_part_of_a_profile_equal_rows_of_the_whole():
    """``row_into(y[lo:hi], out, start=lo)`` for a profile that is zero outside ``lo:hi``,
    with the part inside the grid's run of the axis, across either end of it or outside it;
    on an axis with repeated points (no run) the whole profile is rebuilt."""
    rng = np.random.default_rng(17)
    grid = LogMzGrid(500.0, 1500.0, 2e-5, -1)
    axis = InstrumentModel("tof", 25000.0).profile_axis(470.0, 1600.0)
    repeated = np.sort(np.concatenate([axis, axis[::50]]))
    for mz in (axis, repeated):
        rows = fast.ProfileRows(grid, mz, 3000, 32768)
        assert (rows.run is not None and rows.order is None) == (mz is axis)
        for lo, hi in ((0, mz.size), (0, 40), (mz.size - 40, mz.size), (500, 9000),
                       (mz.size // 2, mz.size // 2 + 1), (700, 700)):
            y = np.zeros(mz.size)
            y[lo:hi] = rng.exponential(3.0, hi - lo) * (rng.random(hi - lo) < 0.5)
            out = np.zeros(32768)
            assert rows.row_into(y[lo:hi], out, start=lo) is out
            same(out, rows.row(y))
        with pytest.raises(ValueError):
            rows.row_into(np.ones(10), np.zeros(32768), start=mz.size - 5)


def test_render_into_a_used_buffer_equals_a_fresh_render():
    rng = np.random.default_rng(15)
    inst = InstrumentModel("tof", 30000.0, shape="mixed", eta=0.3)
    axis = inst.profile_axis(400.0, 2000.0)
    buf = np.full(axis.size, np.nan)
    for n in (0, 1, 40, 400):
        sticks = Sticks(np.sort(rng.uniform(380.0, 2050.0, n)), rng.exponential(1e4, n),
                               np.zeros(n, dtype=np.int64), np.full(n, 3, dtype=np.int64),
                               np.zeros(n))
        want = render.render_profile(sticks, axis, inst)
        assert render.render_profile(sticks, axis, inst, out=buf) is buf
        same(buf, want)


# ------------------------------------------------------------------------- whole frames
def same_frames(a, b):
    for key in ("features", "topk_z", "topk_w", "heat"):
        same(getattr(a, key), getattr(b, key))
    assert a.b0 == b.b0
    assert a.meta.keys() == b.meta.keys()
    for key in a.meta:
        if key == "raw":
            for x, y in zip(a.meta["raw"], b.meta["raw"], strict=True):
                same(x, y)
        elif key == "scans":
            for (x1, y1), (x2, y2) in zip(a.meta["scans"], b.meta["scans"], strict=True):
                same(x1, x2)
                same(y1, y2)
        else:
            assert repr(a.meta[key]) == repr(b.meta[key]), key
    assert [t.to_dict() for t in a.truths] == [t.to_dict() for t in b.truths]


#: (recipe, crop, seeds per mixture component, training items): every kind of spectrum of the
#: default mixture with singly charged ions ignored, kept (200 background ions) and dropped,
#: with evidence channels and on the grid; recipes v3 and v4 in the window view (isotope
#: channels or not) and the comb view (isotope and measured channels), at 16 384 bins
LIVE = [("default:ignore:40", 32768, (1, 2), (0, 1)), ("default:keep:200", 16384, (3, 4), (2,)),
        ("default:drop:40", 16384, (5,), (3,)), ("default:ignore:40:6", 8192, (6,), ()),
        ("grid:keep:200", 16384, (7,), ()),
        ("v3:window:iso", 16384, (1, 2), (0, 1)), ("v3:window:noiso", 16384, (3,), (2,)),
        ("v3:comb:iso", 16384, (4,), (3,)),
        ("v4:window:iso", 16384, (1, 2, 3), (0, 1, 2)), ("v4:comb:iso", 16384, (4, 5), (3,))]


@needs_numba
@pytest.mark.parametrize("recipe,crop,seeds,items", LIVE, ids=[r[0] for r in LIVE])
def test_frames_of_every_recipe_equal_the_generator_as_it_was(monkeypatch, recipe, crop, seeds,
                                                              items):
    paths = set()
    for name in ("_comb_best_sparse", "_comb_best_tiled"):
        kernel = getattr(fast, name)

        def counted(*args, _kernel=kernel, _name=name):
            paths.add(_name)
            return _kernel(*args)

        monkeypatch.setattr(fast, name, counted)
    mix = mixture(recipe, crop)
    kinds = set()
    for _w, cfg in mix:
        for seed in seeds:
            got = frames.generate_frame(cfg, np.random.default_rng(seed))
            same_frames(got, reference_frame(cfg, np.random.default_rng(seed)))
            kinds.add(got.meta["kind"])
    ds = FrameDataset(mix, 10**9, base_seed=0)
    for item in items:
        _cfg, got = ds.frame(item)
        same_frames(got, reference_item(mix, item)[1])
    assert kinds == {"tof", "orbitrap"}
    if recipe.startswith(("v3", "v4")):  # thresholded Orbitrap spectra are mostly zeros
        assert paths == {"_comb_best_sparse", "_comb_best_tiled"}



# ---------------------------------------------------------------------- 2a5fcb5's frames
def _same_arithmetic() -> bool:
    env = REF["env"]
    return (env["machine"] == platform.machine() and env["numpy"] == np.__version__
            and env["canary"] == canary() and env["sort_canary"] == sort_canary())


@pytest.mark.skipif(not _same_arithmetic(),
                    reason="digests were recorded where numpy computes or sorts differently")
@pytest.mark.parametrize("recipe", sorted({r["recipe"] for r in REF["frames"]}))
def test_frames_are_those_of_2a5fcb5(recipe):
    """Full-size frames (131 072-bin windows for v3 and v4) as the commit before the speedups
    made them, by their digests."""
    records = [r for r in REF["frames"] if r["recipe"] == recipe]
    mix = mixture(recipe, records[0]["crop"])
    ds = FrameDataset(mix, 10**9, base_seed=0)
    for r in records:
        assert frame_digest(ds.frame(r["item"])[1]) == r["digest"], r
