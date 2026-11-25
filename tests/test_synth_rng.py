"""numpy's draws compiled (``synth.rng``), the scans drawn with them (``synth.fast.frame_scans``)
and the isotope limit in a few passes (``nn.features.isotope_limit``), bit for bit.

The compiled draws must be numpy's, draw for draw, and leave the generator where numpy leaves it:
training is deterministic per seed, and a paused run resumes on the same spectra. Here they are
compared with numpy's Generator on many seeds and rates (long streams included), the scans with
the lines they replace on random inputs, whole frames of every recipe with the generator as it
was (``tests/_synth_reference.py``), and the isotope limit with the loop over the charges on
random grids. Where the compiled scans do not apply (inputs numpy refuses or treats otherwise, a
self-check that fails, no numba, turned off) the scans must be numpy's own, and why must be
told.
"""

import numpy as np
import pytest

from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.nn import features as F
from lcmsdeconv.nn.grid import LogMzGrid
from lcmsdeconv.synth import fast, frames
from lcmsdeconv.synth import rng as R
from lcmsdeconv.synth.dataset import FrameDataset

from ._synth_reference import mixture, reference_frame, reference_item, reference_scan, train_script

needs_numba = pytest.mark.skipif(not fast.HAVE_NUMBA, reason="numba is not installed")


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


def same_generators(a, b):
    """The same state (the buffered 32-bit half included) and the same next draws."""
    assert a.bit_generator.state == b.bit_generator.state
    same(a.integers(0, 2**31, size=3, dtype=np.uint32), b.integers(0, 2**31, size=3,
                                                                     dtype=np.uint32))
    same(a.random(3), b.random(3))


def pair(seed, uint32_first=0):
    """Two generators of one seed, after ``uint32_first`` 32-bit draws (a buffered half)."""
    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
    a.integers(0, 2**31, size=uint32_first, dtype=np.uint32)
    b.integers(0, 2**31, size=uint32_first, dtype=np.uint32)
    return a, b


@pytest.fixture
def fresh_check(monkeypatch):
    """This process's decision on the compiled draws forgotten, so that it is made again (and
    restored afterwards)."""
    monkeypatch.setattr(R, "_STATE", {})
    monkeypatch.delenv(R.ENV_SWITCH, raising=False)


# ------------------------------------------------------------------------------ the draws
@needs_numba
def test_the_self_check_passes_with_this_numpy(fresh_check):
    """numpy may change a distribution's algorithm in another version; then the self-check
    fails, frames are drawn by numpy (slower, still exact), and this test says so."""
    assert R.self_check() is None
    assert R.available()
    st = R.status()
    assert st["draws"] == "compiled" and st["reason"] is None and st["check_seconds"] > 0


@needs_numba
@pytest.mark.parametrize("seed", [0, 1, 7, 2026, 31337, 271828])
def test_normal_draws_are_numpys(seed):
    """The raw PCG64 stream, then normal draws at several scales (zero and a tiny one among
    them) and a location, the ziggurat's tail drawn too, after a buffered 32-bit half: the same
    values, state and next draws."""
    a, b = pair(seed, seed % 3)
    st = R.get_state(b)
    raw = np.empty(2000, dtype=np.uint64)
    R._raw(st, raw)
    R.set_state(b, st)
    same(raw, a.bit_generator.random_raw(2000))
    tail = 0
    for loc, scale, n in ((0.0, 1.0, 1_000_000), (0.0, 2.5, 200_000), (0.0, 0.0, 1000),
                          (0.0, 1e-300, 1000), (-3.5, 0.7, 50_000), (0.0, 37.5, 100_000)):
        want = a.normal(loc, scale, n)
        same(R.normal(b, loc, scale, n), want)
        if scale == 1.0:
            tail = int(np.count_nonzero(np.abs(want) > R.NOR_R))
    assert tail > 100  # the ziggurat's tail path, about one draw in 3 600
    same_generators(a, b)


@needs_numba
def test_a_long_stream_of_normal_draws_is_numpys():
    a, b = pair(20261125)
    for _ in range(5):
        same(R.normal(b, 0.0, 1.7, 1_000_000), a.normal(0.0, 1.7, 1_000_000))
    same_generators(a, b)


def poisson_rates(rng):
    """Rates of every regime, shuffled: zero (no draw), below 10 (the multiplication method)
    down to 1e-300, exactly 10 and the number just below it, 10 to 40 (PTRS, where numba's
    own Generator.poisson parts from numpy's), up to :data:`RATE_MAX`."""
    lam = np.concatenate([np.zeros(500), np.geomspace(1e-300, 9.999, 5000), rng.random(20_000) * 10,
                          np.full(500, 10.0), np.full(500, np.nextafter(10.0, 0.0)),
                          10.0 + rng.random(60_000) * 30.0,
                          np.exp(rng.uniform(np.log(40.0), np.log(1e3), 10_000)),
                          np.exp(rng.uniform(np.log(1e3), np.log(R.RATE_MAX), 5000)),
                          [R.RATE_MAX]])
    return lam[rng.permutation(lam.size)]


@needs_numba
@pytest.mark.parametrize("seed", range(8))
def test_poisson_draws_are_numpys_at_rates_of_every_size(seed):
    lam = poisson_rates(np.random.default_rng(1000 + seed))
    a, b = pair(seed, seed % 2)
    same(R.poisson_counts(b, lam), a.poisson(lam))
    rows = lam[:100_000].reshape(-1, 5)  # rates in another shape
    same(R.poisson_counts(b, rows), a.poisson(rows))
    same_generators(a, b)


@needs_numba
@pytest.mark.parametrize("seed", [77, 7654321])
def test_a_long_stream_of_poisson_draws_at_rates_of_10_to_40_is_numpys(seed):
    """A million draws at rates of 10 to 40, where PTRS rejects most often and numba's own
    Generator.poisson, which truncates where numpy floors, parted from numpy's (seed
    7654321 at draw 77 299)."""
    lam = np.linspace(10.0, 40.0, 1_000_000)
    a, b = pair(seed)
    same(R.poisson_counts(b, lam), a.poisson(lam))
    same_generators(a, b)


@needs_numba
def test_the_compiled_draws_refuse_what_numpy_refuses():
    for lam in ([1.0, np.nan], [1.0, -2.0], [R.POISSON_LAM_MAX * 2]):
        a, b = pair(5)
        assert outcome(R.poisson_counts, b, np.array(lam)) is outcome(a.poisson, np.array(lam))
        same_generators(a, b)
    for scale in (-1.0, -0.0):
        a, b = pair(5)
        assert outcome(R.normal, b, 0.0, scale, 10) is outcome(a.normal, 0.0, scale, 10)
        same_generators(a, b)
    with pytest.raises(TypeError):
        R.get_state(np.random.Generator(np.random.MT19937(3)))


# ------------------------------------------------------------------------------- the scans
def numpy_scans(rng, clean, weights, gain, esig, sat_level, thr, pad, axis):
    """The scans as numpy's own lines make them (``_synth_reference.reference_scan``), summed
    as ``frames._sum_scans`` sums them, with their positive points."""
    frames_int = [reference_scan(rng, clean, w, gain, esig, sat_level, thr, pad) for w in weights]
    total = frames._sum_scans(frames_int)
    return (axis[total > 0], total[total > 0]), [(axis[s > 0], s[s > 0]) for s in frames_int]


def same_scans(got, want):
    if isinstance(want, type):
        assert got is want
        return
    (gm, gy), gs = got
    (wm, wy), ws = want
    same(gm, wm)
    same(gy, wy)
    assert len(gs) == len(ws)
    for (a, b), (c, d) in zip(gs, ws, strict=True):
        same(a, c)
        same(b, d)


@pytest.fixture
def compiled_calls(monkeypatch):
    """Per call of :func:`lcmsdeconv.synth.rng.scan_frame` from here on, whether it drew the
    scans (False: it refused a rate). The self-check's own scans count too: a test decides
    (``R.available()``) and clears the list before what it counts."""
    calls = []
    kernel = R.scan_frame

    def counted(*args):
        out = kernel(*args)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(R, "scan_frame", counted)
    return calls


def scan_cases(rng, count):
    """Scans of every shape: one point to twelve thousand, windows of 0 to 6 points (also
    longer than the scan), negative profile values (rates clipped to 0), -0.0, large rates
    (PTRS), a NaN (numpy refuses it), all zeros, a negative gain, no noise, saturation or none,
    no threshold or one, and a buffered 32-bit half in the generator."""
    for trial in range(count):
        n = int(rng.choice([1, 2, 3, 4, 5, 7, 16, 33, 200, 1500, 12000]))
        kind = trial % 7
        clean = rng.gamma(0.5, 30.0, n) * (rng.random(n) < rng.uniform(0.05, 1.0))
        if kind == 1:
            clean[rng.random(n) < 0.1] *= -1.0
        elif kind == 2:
            clean *= 10 ** rng.uniform(0, 6)
        elif kind == 3:
            clean[rng.integers(0, n)] = np.nan
        elif kind == 4:
            clean[:] = 0.0
        elif kind == 5:
            clean[rng.random(n) < 0.3] = -0.0
        weights = rng.dirichlet(np.ones(int(rng.integers(1, 6))))
        gain = float(10 ** rng.uniform(-1, 1.7)) * (-1.0 if kind == 6 else 1.0)
        esig = float(rng.choice([0.0, rng.uniform(0, 5)]))
        sat = None if rng.random() < 0.5 else float(10 ** rng.uniform(0, 4))
        thr = float(rng.choice([0.0, -1.0, rng.uniform(0, 20), rng.uniform(0, 20)]))
        pad = int(rng.choice([0, 1, 2, 3, 6]))
        axis = np.sort(rng.uniform(100, 2000, n))
        yield (int(rng.integers(0, 2**63)), int(rng.integers(0, 3)),
               (clean, weights, gain, esig, sat, thr, pad, axis))


@needs_numba
def test_frame_scans_equal_numpys_lines_on_random_inputs(compiled_calls):
    """Values, positive points, generator state and next draws, or the same exception: against
    the lines frames used to call (``fast._frame_scans_numpy``) and numpy's own lines."""
    R.available()
    compiled_calls.clear()
    rng = np.random.default_rng(20261125)
    for seed, pre, args in scan_cases(rng, 700):
        a, b = pair(seed, pre)
        got = outcome(fast.frame_scans, a, *args)
        same_scans(got, outcome(fast._frame_scans_numpy, b, *args))
        same_generators(a, b)
        a, b = pair(seed, pre)
        got = outcome(fast.frame_scans, a, *args)
        if not isinstance(got, type):  # (numpy's lines raise as those, but not always alike)
            same_scans(got, outcome(numpy_scans, b, *args))
            same_generators(a, b)
    # most scans drawn compiled, the kernel refusing the NaN rates
    assert len(compiled_calls) > 1000 and sum(compiled_calls) > 0.8 * len(compiled_calls)
    assert False in compiled_calls


def fallback_cases():
    """Inputs the compiled scans leave to numpy's lines: ``(name, change, compiled)`` with
    ``compiled`` whether the kernel starts (and refuses a rate); the last, None, the inputs as
    they are, which it draws."""
    return [("negative noise", {"esig": -1.0}, False), ("noise -0.0", {"esig": -0.0}, False),
            ("NaN noise", {"esig": np.nan}, False), ("infinite noise", {"esig": np.inf}, False),
            ("NaN saturation", {"sat": np.nan}, False), ("zero saturation", {"sat": 0.0}, False),
            ("saturation -0.0", {"sat": -0.0}, False),
            ("negative saturation", {"sat": -3.0}, False),
            ("infinite saturation", {"sat": np.inf}, False), ("zero gain", {"gain": 0.0}, False),
            ("NaN gain", {"gain": np.nan}, False), ("infinite gain", {"gain": np.inf}, False),
            ("NaN weight", {"weights": np.array([0.5, np.nan])}, False),
            ("float32 weights", {"weights": np.array([0.4, 0.6], dtype=np.float32)}, False),
            ("weights as a list", {"weights": [0.4, 0.6]}, False),
            ("no weights", {"weights": np.zeros(0)}, False),
            ("float32 profile", {"clean": "float32"}, False), ("2-D profile", {"clean": "2d"}, False),
            ("float32 axis", {"axis": "float32"}, False), ("short axis", {"axis": "short"}, False),
            ("window longer than the scan", {"n": 4, "pad": 2}, False),
            ("negative window", {"pad": -1}, False), ("float window", {"pad": 2.0}, False),
            ("MT19937", {"gen": lambda s: np.random.Generator(np.random.MT19937(s))}, False),
            ("PCG64DXSM", {"gen": lambda s: np.random.Generator(np.random.PCG64DXSM(s))}, False),
            ("Philox", {"gen": lambda s: np.random.Generator(np.random.Philox(s))}, False),
            ("RandomState", {"gen": lambda s: np.random.RandomState(s % 2**32)}, False),
            ("a Generator subclass", {"gen": lambda s: Sub(np.random.PCG64(s))}, False),
            ("NaN rate", {"nan": True}, True), ("rate numpy refuses", {"rate": 1e19}, True),
            ("rate above RATE_MAX", {"rate": 3e15}, True), ("as they are", {}, None)]


class Sub(np.random.Generator):
    pass


@needs_numba
@pytest.mark.parametrize("name,change,compiled", fallback_cases(),
                         ids=[c[0] for c in fallback_cases()])
def test_frame_scans_leave_to_numpy_what_numpy_refuses_or_treats_otherwise(
        compiled_calls, name, change, compiled):
    R.available()
    compiled_calls.clear()
    rng = np.random.default_rng(31)
    n = change.get("n", 400)
    clean = rng.gamma(0.5, 30.0, n) * (rng.random(n) < 0.5)
    if change.get("nan"):
        clean[n // 2] = np.nan
    if "rate" in change:
        clean[n // 3] = change["rate"] * 8.0  # w = 0.5, gain 4
    if change.get("clean") == "float32":
        clean = clean.astype(np.float32)
    elif change.get("clean") == "2d":
        clean = clean.reshape(2, -1)
    axis = np.sort(rng.uniform(100, 2000, n))
    if change.get("axis") == "float32":
        axis = axis.astype(np.float32)
    elif change.get("axis") == "short":
        axis = axis[:-1]
    args = (clean, change.get("weights", np.array([0.5, 0.5])), change.get("gain", 4.0),
            change.get("esig", 2.0), change.get("sat", 30.0), 5.0, change.get("pad", 2), axis)
    make = change.get("gen", np.random.default_rng)
    a, b = make(8), make(8)
    with np.errstate(all="ignore"):
        got = outcome(fast.frame_scans, a, *args)
        same_scans(got, outcome(fast._frame_scans_numpy, b, *args))
    same(a.random(5), b.random(5))  # the generators where the lines leave them
    if compiled is None:  # the inputs as they are: drawn compiled
        assert compiled_calls == [True]
    else:  # not drawn compiled; or the kernel started and refused a rate, the generator untouched
        assert compiled_calls == ([False] if compiled else [])


@needs_numba
def test_frame_scans_with_compiled_draws_off_or_without_numba_are_numpys(monkeypatch,
                                                                         compiled_calls):
    rng = np.random.default_rng(32)
    clean = rng.gamma(0.5, 30.0, 3000) * (rng.random(3000) < 0.5)
    args = (clean, np.array([0.3, 0.7]), 3.0, 1.5, None, 4.0, 2, np.sort(rng.random(3000)))
    for setting in ("env", "disable", "numba"):
        monkeypatch.setattr(R, "_STATE", {})
        if setting == "env":
            monkeypatch.setenv(R.ENV_SWITCH, "off")
        elif setting == "disable":
            R.disable("a test turned them off")
        else:
            monkeypatch.setattr(R, "HAVE_NUMBA", False)
        assert not R.available()
        a, b = pair(9)
        same_scans(fast.frame_scans(a, *args), fast._frame_scans_numpy(b, *args))
        same_generators(a, b)
        st = R.status()
        assert st["draws"] == "numpy" and st["reason"]
        assert {"env": "turned off", "disable": "a test",
                "numba": "numba is not installed"}[setting] in st["reason"]
        monkeypatch.delenv(R.ENV_SWITCH, raising=False)
        monkeypatch.setattr(R, "HAVE_NUMBA", True)
    assert compiled_calls == []


# ------------------------------------------------------------------ the self-check failing
def perturbed(f, at=0):
    """``f`` with one value of its array result moved by one ulp."""
    def g(*args, **kw):
        out = np.array(f(*args, **kw))
        flat = out.reshape(-1)
        flat[at] = np.nextafter(flat[at], np.inf) if out.dtype.kind == "f" else flat[at] + 1
        return out
    return g


@needs_numba
@pytest.mark.parametrize("failure,reason", [
    ("exp", "compiled exp differs"), ("log1p", "compiled log1p differs"),
    ("normal", "normal draws differ"), ("poisson", "Poisson draws differ"),
    ("raw", "PCG64 stream differs"), ("state", "another state"),
    ("scans", "scans differ"), ("raises", "the self-check failed: RuntimeError"),
    ("tables", "ziggurat tables")])
def test_a_failing_self_check_draws_with_numpy_and_says_why(monkeypatch, fresh_check,
                                                           compiled_calls, failure, reason):
    """Each part of the self-check failing (the C runtime's functions, the raw stream, the
    normal and Poisson draws, the generator's state, whole scans; an error in the check; numba
    without numpy's tables): numpy draws the scans, frames are as they were, and the status
    tells why."""
    if failure in ("exp", "log1p"):
        kernel = R._crt_values
        code = {"exp": 0, "log1p": 2}[failure]

        def crt(which, x, nxt, out):
            kernel(which, x, nxt, out)
            if which == code:
                out[5] = np.nextafter(out[5], np.inf)

        monkeypatch.setattr(R, "_crt_values", crt)
    elif failure == "normal":
        monkeypatch.setattr(R, "normal", perturbed(R.normal, 1234))
    elif failure == "poisson":
        monkeypatch.setattr(R, "poisson_counts", perturbed(R.poisson_counts, 77))
    elif failure == "raw":
        kernel = R._raw

        def raw(st, out):
            kernel(st, out)
            out[3] ^= np.uint64(1)

        monkeypatch.setattr(R, "_raw", raw)
    elif failure == "state":  # the draws right, the buffered 32-bit half lost
        put = R.set_state

        def put_losing_the_half(gen, st):
            put(gen, st)
            s = gen.bit_generator.state
            s["has_uint32"], s["uinteger"] = 0, 0
            gen.bit_generator.state = s

        monkeypatch.setattr(R, "set_state", put_losing_the_half)
    elif failure == "scans":
        monkeypatch.setattr(R, "_numpy_scans", lambda *a: ((np.zeros(1), np.zeros(1)), []))
    elif failure == "raises":
        def broken(*a):
            raise RuntimeError("no inputs")

        monkeypatch.setattr(R, "_math_inputs", broken)
    else:
        monkeypatch.setattr(R, "_TABLES_MISSING", "not found (ImportError)")
    assert not R.available()
    st = R.status()
    assert st["draws"] == "numpy" and reason in st["reason"], st
    compiled_calls.clear()
    mix = mixture("v4:comb:iso", 8192)
    for _w, cfg in mix[:4]:
        same_frames(frames.generate_frame(cfg, np.random.default_rng(3)),
                    reference_frame(cfg, np.random.default_rng(3)))
    assert compiled_calls == []


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


def recipe_mixture(recipe: str, crop: int):
    """``_synth_reference.mixture``, and recipe v5 set as it sets v3 and v4."""
    if not recipe.startswith("v5"):
        return mixture(recipe, crop)
    _name, view, iso = recipe.split(":")
    mix = train_script().mixture_v5(crop)
    for _w, c in mix:
        c.evidence_channels = 0
        c.singly_charged = "keep"
        c.iso_channels = iso == "iso"
        c.measured_channel = view == "comb"
        c.crop_size = crop
    return mix


#: every recipe: the default mixture with singly charged ions ignored, kept (200 background
#: ions) and dropped, with evidence channels and on the grid; v3, v4 and v5 in both views
RECIPES = ["default:ignore:40", "default:keep:200", "default:drop:40", "default:ignore:40:6",
           "grid:keep:200", "v3:window:iso", "v3:comb:iso", "v4:window:noiso", "v4:comb:iso",
           "v5:window:iso", "v5:comb:iso"]


@needs_numba
@pytest.mark.parametrize("recipe", RECIPES)
def test_frames_of_every_recipe_equal_the_generator_as_it_was(compiled_calls, recipe):
    """Each kind of spectrum of the recipe (seeded as a test seeds it, and items as training
    reads them) against the generator as it was, whose scans are numpy's own lines; every
    profile frame's scans drawn compiled."""
    R.available()
    compiled_calls.clear()
    mix = recipe_mixture(recipe, 16384)
    made = 0
    for k, (_w, cfg) in enumerate(mix):
        seed = 101 + k
        same_frames(frames.generate_frame(cfg, np.random.default_rng(seed)),
                    reference_frame(cfg, np.random.default_rng(seed)))
        made += cfg.render_path == "profile"
    ds = FrameDataset(mix, 10**9, base_seed=0)
    for item in (50, 51):
        _cfg, got = ds.frame(item)
        same_frames(got, reference_item(mix, item)[1])
        made += _cfg.render_path == "profile"
    assert compiled_calls == [True] * made


# --------------------------------------------------------------------------- isotope limit
def isotope_limit_loop(grid, instrument, z_max=F.ISOTOPE_Z_MAX,
                       resolution_factor=F.ISOTOPE_RESOLUTION_FACTOR, bins=None):
    """``nn.features.isotope_limit`` as it was (f6b8186): the reader's test at every charge."""
    b = np.arange(grid.size) if bins is None else np.asarray(bins)
    zlim = np.zeros(b.size, dtype=np.int16)
    if b.size == 0:
        return zlim
    mz = grid.bin_to_mz(b)
    fwhm = mz / np.asarray(instrument.resolving_power(mz), dtype=float)
    off = np.abs(mz - grid.carrier)
    width_bins = np.maximum(np.log1p(fwhm / off) / grid.step, 2.0)
    for z in range(1, z_max + 1):
        spacing = 1.0033548 / z
        n = np.log1p(spacing / off) / grid.step
        resolved = n >= resolution_factor * width_bins
        if not resolved.any():
            break
        zlim[resolved] = z
    return zlim


def test_isotope_limit_equals_the_loop_over_charges_on_random_grids():
    """Grids of every instrument kind, resolving power (1e3 - 5e6), m/z range, step and
    polarity, charges up to 1 - 120, resolution factors 0.3 - 8, all bins or some (sorted, or
    unsorted with repeats)."""
    assert F._settling_is_exact()
    rng = np.random.default_rng(11)
    for trial in range(240):
        inst = InstrumentModel(str(rng.choice(["orbitrap", "fticr", "tof"])),
                               float(np.exp(rng.uniform(np.log(1e3), np.log(5e6)))))
        lo = float(rng.uniform(60, 800))
        hi = float(min(lo * rng.uniform(1.2, 20), 9900))
        grid = LogMzGrid(lo, hi, float(rng.choice([2e-5, 1e-5, 4e-5])), int(rng.choice([1, -1])))
        z_max = int(rng.choice([120, 60, 60, 30, 5, 1]))
        rf = float(rng.choice([2.5, 2.5, 1.0, 4.0, 0.3, 8.0]))
        bins = [None, np.sort(rng.choice(grid.size, size=min(grid.size, 999), replace=False)),
                rng.integers(0, grid.size, 1500), np.arange(0)][trial % 4]
        same(F.isotope_limit(grid, inst, z_max, rf, bins),
             isotope_limit_loop(grid, inst, z_max, rf, bins))


class _Grid:
    """A grid with any step, offsets and size, for inputs a LogMzGrid does not make."""

    def __init__(self, mz, step, carrier=1.00727646688):
        self.mz, self.step, self.carrier, self.size = np.asarray(mz, float), step, carrier, len(mz)

    def bin_to_mz(self, b):
        return self.mz[np.asarray(b)]


class _Instrument:
    def __init__(self, rp):
        self.rp = rp

    def resolving_power(self, mz):
        return np.broadcast_to(np.asarray(self.rp, dtype=float), np.shape(mz))


def test_isotope_limit_outside_what_the_estimate_covers_is_the_loop():
    """Resolving powers of 0, NaN or infinity, a bin at the carrier's m/z, a negative or zero
    step, no charge (z_max 0) or a charge count that is no int: the loop's answer or its
    error."""
    mz = np.geomspace(150.0, 3000.0, 400)
    cases = [(_Grid(mz, 2e-5), _Instrument(0.0), 60), (_Grid(mz, 2e-5), _Instrument(np.nan), 60),
             (_Grid(mz, 2e-5), _Instrument(np.inf), 60),
             (_Grid(np.append(mz, 1.00727646688), 2e-5), _Instrument(1e5), 60),
             (_Grid(mz, -2e-5), _Instrument(1e5), 60), (_Grid(mz, 0.0), _Instrument(1e5), 60),
             (_Grid(mz, 2e-5), _Instrument(1e5), 0), (_Grid(mz, 2e-5), _Instrument(1e5), 60.0),
             (_Grid(mz, 2e-5), _Instrument(1e5), np.int64(25))]
    with np.errstate(all="ignore"):
        for grid, inst, z_max in cases:
            got = outcome(F.isotope_limit, grid, inst, z_max)
            want = outcome(isotope_limit_loop, grid, inst, z_max)
            if isinstance(want, type):
                assert got is want
            else:
                same(got, want)


def test_isotope_limit_uses_the_loop_where_its_check_fails(monkeypatch):
    """The once-per-process check of the estimate (numpy's log1p giving an element the same
    bits wherever it sits, the estimate against the loop) failing: the loop answers."""
    monkeypatch.setattr(F, "_SETTLING", [])
    settled = F._settled_limit

    def off_by_one(*args):  # wrong by a charge on the check's grids
        z = settled(*args)
        return None if z is None else z + (z.size > 2000)

    monkeypatch.setattr(F, "_settled_limit", off_by_one)
    assert not F._settling_is_exact()
    grid, inst = LogMzGrid(300.0, 2600.0, 2e-5, -1), InstrumentModel("orbitrap", 120000.0)
    same(F.isotope_limit(grid, inst), isotope_limit_loop(grid, inst))
