"""numpy's normal and Poisson draws, compiled, equal to numpy's draw for draw.

A training frame's noisy scans (:func:`lcmsdeconv.synth.fast.frame_scans`) draw a Poisson count
for every nonzero rate and a normal value for every point of the instrument's m/z axis, often
millions of draws a frame, and numpy's Generator spends most of each draw's time in its calling
overhead. Here numpy's own algorithms run inside the compiled scan kernel (:func:`scan`),
operation for operation, on the generator's own state:

* the bit generator, PCG64 (``numpy.random.PCG64``, XSL-RR 128/64): ``state = state * M + inc``
  modulo 2**128 in LLVM's 128-bit integers, the output ``rotr64(hi ^ lo, hi >> 58)``, and
  ``next_double = (r >> 11) * 2**-53``: integer arithmetic and one exact product;
* ``random_standard_normal``, the 256-layer ziggurat with numpy's tables (numba ships a copy of
  them), and ``random_normal``, ``loc + scale * z``;
* ``random_poisson``: the multiplication method below a rate of 10, the transformed rejection
  (PTRS) from 10, with ``floor`` as numpy has it, and ``random_loggam``.

These are numpy 2.5's algorithms (``numpy/random/src/distributions/distributions.c``). They are
compiled without fast-math, so LLVM neither reorders the arithmetic nor contracts it into fused
multiply-adds, with IEEE division (``error_model="numpy"``), and their rare paths call ``exp``,
``log``, ``log1p`` and ``sqrt``, which numba binds to the C runtime CPython's C code calls, as
numpy's does. numba's own ``Generator.poisson`` is not numpy's: it truncates where numpy floors,
and now and then its draws part from numpy's.

numpy does not promise that a distribution's algorithm stays the same between its versions, and
C runtimes differ between platforms, so nothing here is taken on trust: the first use in a
process runs :func:`self_check`, which compares the compiled ``exp``, ``log``, ``log1p`` and
``sqrt`` with Python's (the C runtime's) on a million inputs over the ranges the draws use, then
the compiled draws and scans with numpy's own, draw for draw. On any difference, without numba,
or with ``LCMSDECONV_COMPILED_DRAWS=0`` in the environment, training frames are drawn by numpy
itself, as before, and :func:`status` says so and why. The scan kernel is kept in this file
because numba's cache tracks a kernel's own source file only: a kernel elsewhere that inlined
these draws would keep running an older copy of them after they changed.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

try:  # optional acceleration, as in fast.py
    from numba import njit
    from numba.extending import intrinsic

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(f):
            return f
        return wrap if not args else args[0]

    def intrinsic(f):  # type: ignore
        return f

try:  # numpy's ziggurat tables (numba's copy of numpy's ziggurat_constants.h)
    from numba.np.random import _constants as _ziggurat

    KI = np.asarray(_ziggurat.ki_double, dtype=np.uint64)
    WI = np.asarray(_ziggurat.wi_double, dtype=np.float64)
    FI = np.asarray(_ziggurat.fi_double, dtype=np.float64)
    NOR_R = float(_ziggurat.ziggurat_nor_r)
    NOR_INV_R = float(_ziggurat.ziggurat_nor_inv_r)
    _TABLES_MISSING = None if KI.size == WI.size == FI.size == 256 else "of another size"
except Exception as exc:  # pragma: no cover
    KI, WI, FI = np.zeros(256, dtype=np.uint64), np.zeros(256), np.zeros(256)
    NOR_R = NOR_INV_R = 0.0
    _TABLES_MISSING = f"not found ({type(exc).__name__})"

#: PCG64's multiplier (numpy's ``PCG_DEFAULT_MULTIPLIER_128``)
PCG_MULTIPLIER = (2549297995355413924 << 64) | 4865540595714422341
_MASK64 = (1 << 64) - 1
#: numpy's largest Poisson rate (``POISSON_LAM_MAX`` of numpy/random/_common.pyx, computed as
#: there); a larger one, or a NaN, makes ``Generator.poisson`` raise
POISSON_LAM_MAX = float(np.iinfo("int64").max) - np.sqrt(np.iinfo("int64").max) * 10
#: the largest rate a compiled scan draws; a scan with a larger one is drawn by numpy. Every
#: count the draws can make, and that count plus one, are exact float64 values below it, and the
#: self-check covers it
RATE_MAX = 1e15
#: the environment variable that turns the compiled draws off ("0", "false", "no" or "off")
ENV_SWITCH = "LCMSDECONV_COMPILED_DRAWS"


# --------------------------------------------------------------------------- the generator
@intrinsic
def _pcg_step(typingctx, s_hi, s_lo, i_hi, i_lo):
    """One PCG64 step on the state ``(s_hi, s_lo)`` with the increment ``(i_hi, i_lo)``:
    ``(new_hi, new_lo, output)``, numpy's ``pcg_setseq_128_step_r`` (``state = state * M + inc``
    modulo 2**128) followed by ``pcg_output_xsl_rr_128_64`` (``rotr64(hi ^ lo, hi >> 58)``)."""
    from llvmlite import ir
    from numba import types

    u64 = types.uint64
    sig = types.UniTuple(u64, 3)(u64, u64, u64, u64)

    def codegen(context, builder, signature, args):
        i64, i128 = ir.IntType(64), ir.IntType(128)
        sh, sl, ih, il = args
        c64 = ir.Constant(i128, 64)

        def wide(hi, lo):
            return builder.or_(builder.shl(builder.zext(hi, i128), c64), builder.zext(lo, i128))

        s = builder.add(builder.mul(wide(sh, sl), ir.Constant(i128, PCG_MULTIPLIER)),
                        wide(ih, il))
        hi = builder.trunc(builder.lshr(s, c64), i64)
        lo = builder.trunc(s, i64)
        rot = builder.lshr(hi, ir.Constant(i64, 58))
        x = builder.xor(hi, lo)
        left = builder.and_(builder.sub(ir.Constant(i64, 0), rot), ir.Constant(i64, 63))
        out = builder.or_(builder.lshr(x, rot), builder.shl(x, left))
        return context.make_tuple(builder, signature.return_type, [hi, lo, out])

    return sig, codegen


@njit(inline="always", error_model="numpy")
def next_uint64(st):
    """numpy's ``next_uint64`` of PCG64 on ``st`` = ``[state_hi, state_lo, inc_hi, inc_lo]``
    (uint64), which it advances."""
    hi, lo, out = _pcg_step(st[0], st[1], st[2], st[3])
    st[0] = hi
    st[1] = lo
    return out


@njit(inline="always", error_model="numpy")
def next_double(st):
    """numpy's ``next_double`` of PCG64: ``(next_uint64 >> 11) * 2**-53``."""
    return np.float64(next_uint64(st) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(inline="always", error_model="numpy")
def standard_normal(st):
    """numpy's ``random_standard_normal``, operation for operation."""
    while True:
        r = next_uint64(st)
        idx = np.int64(r & np.uint64(0xFF))
        r = r >> np.uint64(8)
        sign = r & np.uint64(1)
        rabs = (r >> np.uint64(1)) & np.uint64(0x000FFFFFFFFFFFFF)
        x = np.float64(rabs) * WI[idx]
        if sign & np.uint64(1):
            x = -x
        if rabs < KI[idx]:
            return x  # 99.3 % of the draws
        if idx == 0:
            while True:
                xx = -NOR_INV_R * math.log1p(-next_double(st))
                yy = -math.log1p(-next_double(st))
                if yy + yy > xx * xx:
                    if (rabs >> np.uint64(8)) & np.uint64(1):
                        return -(NOR_R + xx)
                    return NOR_R + xx
        elif (FI[idx - 1] - FI[idx]) * next_double(st) + FI[idx] < math.exp(-0.5 * x * x):
            return x


@njit(inline="always", error_model="numpy")
def loggam(x):
    """numpy's ``random_loggam``, operation for operation."""
    if x == 1.0 or x == 2.0:
        return 0.0
    n = np.int64(7 - x) if x < 7.0 else np.int64(0)
    x0 = x + n
    x2 = (1.0 / x0) * (1.0 / x0)
    gl0 = -1.39243221690590e+00
    gl0 *= x2
    gl0 += 1.796443723688307e-01
    gl0 *= x2
    gl0 += -2.955065359477124e-02
    gl0 *= x2
    gl0 += 6.410256410256410e-03
    gl0 *= x2
    gl0 += -1.917526917526918e-03
    gl0 *= x2
    gl0 += 8.417508417508418e-04
    gl0 *= x2
    gl0 += -5.952380952380952e-04
    gl0 *= x2
    gl0 += 7.936507936507937e-04
    gl0 *= x2
    gl0 += -2.777777777777778e-03
    gl0 *= x2
    gl0 += 8.333333333333333e-02
    gl = gl0 / x0 + 0.5 * 1.8378770664093453e+00 + (x0 - 0.5) * math.log(x0) - x0
    if x < 7.0:
        for _k in range(n):
            gl -= math.log(x0 - 1.0)
            x0 -= 1.0
    return gl


@njit(inline="always", error_model="numpy")
def poisson(st, lam):
    """numpy's ``random_poisson`` (an int64 count), operation for operation: no draw for a
    rate of zero, the multiplication method below 10, PTRS from 10."""
    if lam >= 10:
        slam = math.sqrt(lam)
        loglam = math.log(lam)
        b = 0.931 + 2.53 * slam
        a = -0.059 + 0.02483 * b
        invalpha = 1.1239 + 1.1328 / (b - 3.4)
        vr = 0.9277 - 3.6224 / (b - 2)
        while True:
            u = next_double(st) - 0.5
            v = next_double(st)
            us = 0.5 - math.fabs(u)
            k = np.int64(math.floor((2 * a / us + b) * u + lam + 0.43))
            if us >= 0.07 and v <= vr:
                return k
            if k < 0 or (us < 0.013 and v > us):
                continue
            if ((math.log(v) + math.log(invalpha) - math.log(a / (us * us) + b))
                    <= (-lam + np.float64(k) * loglam - loggam(np.float64(k) + 1))):
                return k
    elif lam == 0:
        return np.int64(0)
    else:
        enlam = math.exp(-lam)
        count = np.int64(0)
        prod = 1.0
        while True:
            prod *= next_double(st)
            if prod > enlam:
                count += 1
            else:
                return count


# --------------------------------------------------------------------------- state transfer
def get_state(gen: np.random.Generator) -> np.ndarray:
    """The PCG64 state of ``gen`` as ``[state_hi, state_lo, inc_hi, inc_lo]`` (uint64)."""
    s = gen.bit_generator.state
    if s.get("bit_generator") != "PCG64":
        raise TypeError("the compiled draws follow numpy's PCG64 bit generator only")
    st, inc = s["state"]["state"], s["state"]["inc"]
    return np.array([st >> 64, st & _MASK64, inc >> 64, inc & _MASK64], dtype=np.uint64)


def set_state(gen: np.random.Generator, st: np.ndarray) -> None:
    """Write the state ``st`` back to ``gen``, leaving its increment and its buffered 32-bit
    half (``has_uint32``, ``uinteger``) as they are."""
    s = gen.bit_generator.state
    s["state"]["state"] = (int(st[0]) << 64) | int(st[1])
    gen.bit_generator.state = s


def is_pcg64_generator(gen) -> bool:
    """Whether ``gen`` is a plain ``numpy.random.Generator`` on a plain ``PCG64``, whose
    ``normal`` and ``poisson`` the compiled draws stand for."""
    return type(gen) is np.random.Generator and type(gen.bit_generator) is np.random.PCG64


@njit(cache=True, error_model="numpy")
def _raw(st, out):
    for i in range(out.shape[0]):
        out[i] = next_uint64(st)


@njit(cache=True, error_model="numpy")
def _normals(st, loc, scale, out):
    for i in range(out.shape[0]):
        out[i] = loc + scale * standard_normal(st)


@njit(cache=True, error_model="numpy")
def _poissons(st, lam, out):
    for i in range(lam.shape[0]):
        out[i] = poisson(st, lam[i])


def normal(gen: np.random.Generator, loc: float, scale: float, size: int) -> np.ndarray:
    """``gen.normal(loc, scale, size)`` by the compiled draws, for a finite ``scale`` that is
    not negative (and not -0.0, which numpy refuses)."""
    scale = float(scale)
    if not (math.isfinite(scale) and scale >= 0.0 and math.copysign(1.0, scale) > 0):
        raise ValueError("scale must be finite and not negative")
    st = get_state(gen)
    out = np.empty(int(size))
    _normals(st, float(loc), scale, out)
    set_state(gen, st)
    return out


def poisson_counts(gen: np.random.Generator, lam) -> np.ndarray:
    """``gen.poisson(lam)`` (int64, ``lam``'s shape) by the compiled draws, for rates numpy
    takes (it refuses NaN, negative and too large ones; so does this)."""
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    if not np.all(lam >= 0):
        raise ValueError("lam < 0 or lam is NaN")
    if not np.all(lam <= POISSON_LAM_MAX):
        raise ValueError("lam value too large")
    st = get_state(gen)
    out = np.empty(lam.size, dtype=np.int64)
    _poissons(st, lam.ravel(), out)
    set_state(gen, st)
    return out.reshape(lam.shape)


# ------------------------------------------------------------------------------- one scan
@njit(cache=True, error_model="numpy")
def scan(st, clean, w, gain, esig, sat, thr, pad, first, last, values, above, total):
    """One noisy scan of a training frame into ``values``, drawn from the PCG64 state ``st``
    (advanced): the values, draws and their order of ``fast.noisy_scan``'s reference lines::

        lam = np.clip(clean * w / gain, 0.0, None)
        scan = rng.poisson(lam).astype(np.float64) * gain + np.abs(rng.normal(0.0, esig, n))
        scan = np.minimum(scan, sat)
        scan[~(np.convolve(scan >= thr, np.ones(2 * pad + 1), mode="same") > 0)] = 0.0  # thr > 0

    and the scan added to ``total`` (which the ``first`` scan sets: ``frames._sum_scans``'s
    additions). Returns the numbers of positive values of the scan and, in the ``last`` scan,
    of ``total``, or ``(-1, 0)`` - ``st`` then to be discarded - at a rate that is NaN or above
    :data:`RATE_MAX` (numpy refuses a NaN one or one above ``POISSON_LAM_MAX``).

    For the caller to see to: ``w`` and ``gain`` finite and ``gain`` not zero (so that a zero of
    ``clean`` is a zero rate: no draw, a count of 0, as numpy's ``random_poisson`` has it),
    ``esig`` finite and not negative or -0.0, ``sat`` positive (``inf``: no saturation; a
    scan's values are never NaN here, so ``np.minimum`` is the comparison below), ``pad >= 0``
    and ``n >= 2 * pad + 1`` when ``thr > 0`` (the ``mode="same"`` window; ``pad`` 0 without
    a threshold), ``values`` and ``total`` ``n`` float64 and ``above`` ``n`` booleans.

    Pass 1 draws the Poisson counts of the nonzero rates in order (``values = count * gain``).
    Pass 2 draws one normal per point in order, ``noise = 0.0 + esig * z`` as numpy's
    ``random_normal`` computes it, ``value = count * gain + abs(noise)``, saturates it and
    marks whether it reaches the threshold. The convolution counts the marks within ``pad``
    points either side, here a running count, so point ``j`` is final ``pad`` points later (or
    at the end of the scan): zeroed where its window holds no mark, and added to ``total``."""
    n = clean.shape[0]
    for i in range(n):
        c = np.int64(0)
        x = clean[i]
        if x != 0.0:
            x = x * w / gain
            if x < 0.0:
                x = 0.0
            if x != 0.0:
                if not (x <= RATE_MAX):
                    return np.int64(-1), np.int64(0)
                c = poisson(st, x)
        values[i] = np.float64(c) * gain
    m = np.int64(0)
    r = np.int64(0)
    threshold = thr > 0.0
    cnt = 0  # marks in the window of the point made final, [j - pad, j + pad]
    for i in range(n + pad):
        if i < n:
            noise = 0.0 + esig * standard_normal(st)
            v = values[i] + abs(noise)
            if v > sat:
                v = sat
            values[i] = v
            if threshold:
                mark = v >= thr
                above[i] = mark
                if mark:
                    cnt += 1
        j = i - pad
        if j < 0:
            continue
        v = values[j]
        if threshold:
            if cnt == 0:
                v = 0.0
                values[j] = v
            if j >= pad and above[j - pad]:
                cnt -= 1  # leaves the window of the next point
        m += 1 if v > 0.0 else 0
        t = v if first else total[j] + v
        total[j] = t
        if last:
            r += 1 if t > 0.0 else 0
    return m, r


@njit(cache=True)
def _positive_into(axis, y, mz, out):
    """``axis[y > 0]`` and ``y[y > 0]`` into ``mz`` and ``out``, one element longer than their
    number: every point written and the position advanced for a positive one, as
    ``fast._compress_positive`` does."""
    k = 0
    for i in range(y.shape[0]):
        mz[k] = axis[i]
        out[k] = y[i]
        k += 1 if y[i] > 0 else 0


def _points(axis, y, count):
    mz, out = np.empty(count + 1), np.empty(count + 1)
    _positive_into(axis, y, mz, out)
    return mz[:count], out[:count]


def scan_frame(gen, clean, weights, gain, esig, sat, thr, pad, axis):
    """``(raw, scans)`` of a training frame's scans by :func:`scan` (see
    :func:`lcmsdeconv.synth.fast.frame_scans`, which checks the arguments; ``sat`` is ``inf``
    without saturation), leaving ``gen`` where numpy's draws leave it; None, with ``gen``
    untouched, when a scan has a rate :func:`scan` does not draw."""
    st = get_state(gen)
    n = clean.size
    values, total = np.empty(n), np.empty(n)
    above = np.empty(n, dtype=np.bool_)
    scans = []
    r = 0
    for k, w in enumerate(weights):
        m, r = scan(st, clean, float(w), gain, esig, sat, thr, pad, k == 0,
                    k == len(weights) - 1, values, above, total)
        if m < 0:
            return None
        scans.append(_points(axis, values, m))
    set_state(gen, st)
    return _points(axis, total, r), scans


# ------------------------------------------------------------------------------ the check
@njit(cache=True, error_model="numpy")
def _crt_values(which, x, nxt, out):
    """``exp`` (0), ``log`` (1), ``log1p`` (2) or ``sqrt`` (3) of every ``x``, called as the
    draws call them: one value at a time, visited in the order of the cycle ``nxt`` (a load per
    step that the next depends on, so that no compiler can turn the calls into vector ones)."""
    j = 0
    for _ in range(x.shape[0]):
        v = x[j]
        if which == 0:
            out[j] = math.exp(v)
        elif which == 1:
            out[j] = math.log(v)
        elif which == 2:
            out[j] = math.log1p(v)
        else:
            out[j] = math.sqrt(v)
        j = nxt[j]


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _math_inputs(rng: np.random.Generator, n: int) -> list:
    """``(name, Python function, kernel code, inputs)`` for the C runtime functions the draws
    call, about ``n`` inputs in all over the ranges the draws call them on: ``log1p(-u)`` of the
    normal's tail (``u`` a ``next_double``), ``exp`` of its wedges (``-x*x/2``, ``|x|`` up to
    the tail's start) and of the multiplication method (``-lam``, ``lam`` below 10), ``log`` and
    ``sqrt`` of the PTRS rates (10 up to :data:`RATE_MAX`), ``log`` of its uniforms, of its
    ``invalpha`` and ``a / us**2 + b`` and of ``loggam``'s arguments."""
    q = max(n // 4, 8)
    u = rng.random(q)  # next_double values, multiples of 2**-53 in [0, 1)
    edge = np.arange(1, 65) * 2.0**-53
    log1p_x = np.concatenate([-u, -edge, -(1.0 - edge), [-0.0]])
    exp_x = np.concatenate([-0.5 * (rng.random(q // 2) * NOR_R) ** 2,
                            -np.exp(rng.uniform(np.log(1e-300), np.log(10.0), q // 2)), [-0.0]])
    rates = np.exp(rng.uniform(np.log(10.0), np.log(RATE_MAX), q // 2))
    uniforms = rng.random(q // 4)
    log_x = np.concatenate([rates, uniforms[uniforms > 0], edge,
                            1.1239 + rng.random(q // 8) * 0.25,  # invalpha
                            np.exp(rng.uniform(0.0, np.log(1e40), q // 8)),  # a / us**2 + b
                            np.arange(1.0, 64.0)])  # loggam's small arguments
    sqrt_x = np.concatenate([rates, 10.0 + rng.random(q // 2) * 1000.0])
    return [("exp", math.exp, 0, exp_x), ("log", math.log, 1, log_x),
            ("log1p", math.log1p, 2, log1p_x), ("sqrt", math.sqrt, 3, sqrt_x)]


def _numpy_scans(gen, clean, weights, gain, esig, sat, thr, pad, axis):
    """The scans of :func:`scan_frame` by numpy's own lines (``sat`` None: no saturation)."""
    frames = []
    for w in weights:
        lam = np.clip(clean * w / gain, 0.0, None)
        s = gen.poisson(lam).astype(np.float64) * gain + np.abs(gen.normal(0.0, esig, clean.size))
        if sat is not None:
            s = np.minimum(s, sat)
        if thr > 0:
            s[~(np.convolve(s >= thr, np.ones(2 * pad + 1), mode="same") > 0)] = 0.0
        frames.append(s)
    total = frames[0].copy()
    for s in frames[1:]:
        total += s
    return (axis[total > 0], total[total > 0]), [(axis[s > 0], s[s > 0]) for s in frames]


def self_check(n_math: int = 1_000_000, seed: int = 20261125) -> str | None:
    """None if the compiled draws are numpy's on this machine, else why not.

    1. The C runtime functions the draws call (:func:`_math_inputs`): the compiled ``exp``,
       ``log``, ``log1p`` and ``sqrt`` against Python's, bit for bit, on ``n_math`` inputs.
    2. Draw for draw against a numpy Generator of the same seed: the raw PCG64 stream, normal
       draws (the tail and wedge paths included), Poisson draws at rates of every regime (zero,
       below 10, exactly 10 and just below, up to :data:`RATE_MAX`), with the generator's
       buffered 32-bit half kept, its final state and its next draws.
    3. Whole scans (:func:`scan_frame` against numpy's lines): saturation, threshold, zero and
       large rates.
    """
    if _TABLES_MISSING:
        return f"numba's copy of numpy's ziggurat tables is {_TABLES_MISSING}"
    rng = np.random.default_rng(seed)
    for name, fn, code, x in _math_inputs(rng, n_math):
        x = np.ascontiguousarray(x, dtype=np.float64)
        got = np.empty(x.size)
        _crt_values(code, x, np.roll(np.arange(x.size, dtype=np.int64), -1), got)
        want = np.fromiter(map(fn, x.tolist()), dtype=np.float64, count=x.size)
        if not _same(got, want):
            bad = int(np.count_nonzero(got.view(np.int64) != want.view(np.int64)))
            return (f"the compiled {name} differs from the C runtime's (Python's math.{name}) on "
                    f"{bad} of {x.size} inputs")
    a, b = np.random.default_rng(seed + 1), np.random.default_rng(seed + 1)
    st = get_state(b)
    raw = np.empty(1000, dtype=np.uint64)
    _raw(st, raw)
    set_state(b, st)
    if not _same(a.bit_generator.random_raw(1000), raw):
        return "the compiled PCG64 stream differs from numpy's"
    a.integers(0, 2**31, size=3, dtype=np.uint32)  # a buffered 32-bit half, to be kept
    b.integers(0, 2**31, size=3, dtype=np.uint32)
    if not _same(a.normal(0.0, 2.5, 200_000), normal(b, 0.0, 2.5, 200_000)):
        return "the compiled normal draws differ from numpy's"
    lam = np.concatenate([np.zeros(2000), np.geomspace(1e-300, 9.999, 20_000),
                          np.full(500, 10.0), np.full(500, np.nextafter(10.0, 0.0)),
                          10.0 + rng.random(30_000) * 30.0,
                          np.exp(rng.uniform(np.log(40.0), np.log(RATE_MAX), 20_000))])
    lam = lam[rng.permutation(lam.size)]
    if not _same(a.poisson(lam), poisson_counts(b, lam)):
        return "the compiled Poisson draws differ from numpy's"
    if a.bit_generator.state != b.bit_generator.state or not _same(
            a.integers(0, 2**31, size=5, dtype=np.uint32),
            b.integers(0, 2**31, size=5, dtype=np.uint32)):
        return "the compiled draws leave the generator in another state than numpy's"
    for trial in range(3):
        n = 3001 + 997 * trial
        clean = np.exp(rng.uniform(np.log(1e-3), np.log(5e4), n)) * (rng.random(n) < 0.6)
        axis = np.cumsum(rng.uniform(0.5, 1.5, n)) + 100.0
        weights = rng.dirichlet(np.ones(trial + 2))
        gain, esig = float(rng.uniform(1.0, 30.0)), float(rng.uniform(0.5, 5.0))
        sat = float(np.max(clean) * weights.max() * 0.5) if trial != 1 else None
        thr = esig * 3.0 if trial != 2 else 0.0
        a, b = np.random.default_rng(seed + 2 + trial), np.random.default_rng(seed + 2 + trial)
        want = _numpy_scans(a, clean, weights, gain, esig, sat, thr, 2, axis)
        got = scan_frame(b, clean, weights, gain, esig, np.inf if sat is None else sat, thr,
                         2 if thr > 0 else 0, axis)  # as fast.frame_scans calls it
        if got is None or not (_same(got[0][0], want[0][0]) and _same(got[0][1], want[0][1])
                               and len(got[1]) == len(want[1])
                               and all(_same(p[0], q[0]) and _same(p[1], q[1])
                                       for p, q in zip(got[1], want[1], strict=True))
                               and a.bit_generator.state == b.bit_generator.state):
            return "the compiled scans differ from numpy's lines"
    return None


# ----------------------------------------------------------------------------- the switch
_STATE: dict = {}


def available() -> bool:
    """Whether this process draws training scans with the compiled draws: numba is there, they
    are not turned off (:data:`ENV_SWITCH`), and :func:`self_check` found them equal to numpy's
    here (run at the first call, once per process)."""
    if not _STATE:
        started = time.perf_counter()
        reason = None
        if not HAVE_NUMBA:
            reason = "numba is not installed"
        elif os.environ.get(ENV_SWITCH, "").strip().lower() in ("0", "false", "no", "off"):
            reason = f"turned off ({ENV_SWITCH}={os.environ[ENV_SWITCH]})"
        else:
            try:
                reason = self_check()
            except Exception as exc:  # whatever goes wrong, numpy draws
                reason = f"the self-check failed: {type(exc).__name__}: {exc}"
        _STATE.update(ok=reason is None, reason=reason,
                      seconds=time.perf_counter() - started)
    return bool(_STATE["ok"])


def disable(reason: str) -> None:
    """Draw with numpy from now on in this process, for ``reason`` (kept for :func:`status`)."""
    _STATE.update(ok=False, reason=str(reason), seconds=_STATE.get("seconds", 0.0))


def status() -> dict:
    """How this process draws training scans: ``{"draws": "compiled" | "numpy", "reason":
    why numpy (None when compiled), "check_seconds": time the decision took}``, deciding first
    if it has not (:func:`available`)."""
    available()
    return {"draws": "compiled" if _STATE["ok"] else "numpy", "reason": _STATE["reason"],
            "check_seconds": round(float(_STATE["seconds"]), 3)}
