"""Digests of training frames, predictor outputs and inputs, shared by tests/test_recipe_v3.py
and the script that recorded the reference values (tests/data/make_ref_1917c37.py). Only numpy
here, so that the script can import it while running the package as it was."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def _feed(h, a) -> None:
    a = np.ascontiguousarray(a)
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())


def frame_digest(fs) -> str:
    """Everything a training frame carries: features, labels, crop, noise, scans, truths."""
    h = hashlib.sha256()
    for a in (fs.features, fs.topk_z, fs.topk_w, fs.heat):
        _feed(h, a)
    m = fs.meta
    h.update(repr((int(fs.b0), m["noise_sigma"], m["n_sum"], m["gain"], m["grid_lo"],
                   m["resolution"], m["kind"])).encode())
    for a in m["raw"]:
        _feed(h, a)
    for mz, y in m["scans"]:
        _feed(h, mz)
        _feed(h, y)
    h.update(json.dumps([t.to_dict() for t in fs.truths], sort_keys=True).encode())
    return h.hexdigest()[:24]


def arrays_digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        _feed(h, a)
    return h.hexdigest()[:24]


def canary() -> str:
    """The transcendental functions and random draws the frames rest on. Digests recorded on
    one machine can be compared on another only where these agree bit for bit."""
    x = np.linspace(1e-3, 50.0, 4099)
    rng = np.random.default_rng(7)
    draws = [rng.uniform(0.1, 3.0, 64), rng.normal(0.0, 2.0, 64),
             rng.poisson(np.linspace(0.0, 40.0, 64)).astype(np.float64),
             rng.integers(0, 1000, 64).astype(np.float64)]
    return arrays_digest(np.exp(-x), np.log(x), np.log1p(x), np.log10(x), np.sqrt(x),
                         np.cos(x), x ** 0.8, x ** 0.445, *draws)


def ladder_input(length: int = 32768) -> np.ndarray:
    """A fixed LadderNet input ``[1, 4, length]``: 3 000 random peaks on an empty spectrum."""
    rng = np.random.default_rng(0)
    x = np.zeros((1, 4, length), dtype=np.float32)
    idx = rng.integers(0, length, 3000)
    x[0, 0, idx] = rng.uniform(0.0, 5.0, idx.size)
    x[0, 1] = rng.uniform(0.0, 1.0, length)
    x[0, 2] = rng.uniform(0.0, 1.0, length)
    x[0, 3] = rng.uniform(0.0, 1.0, length)
    return x
