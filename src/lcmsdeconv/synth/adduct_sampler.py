"""Per-run adduct propensities and per-ion adduct-state fractions."""

from __future__ import annotations

import numpy as np

from ..chem.adducts import AdductLibrary, AdductState


def sample_run_propensities(
    rng: np.random.Generator, library: AdductLibrary, max_lambda: float = 0.15
) -> dict[str, float]:
    """A propensity in [0, max_lambda] for each adduct type in the library."""
    props = {}
    for a in library.adducts:
        # some runs have essentially none of a given adduct
        lam = 0.0 if rng.random() < 0.3 else rng.uniform(0.0, max_lambda)
        props[a.name] = float(lam)
    return props


def adduct_fractions_for_charge(
    z: int,
    propensities: dict[str, float],
    library: AdductLibrary,
    rng: np.random.Generator,
) -> dict[AdductState, float]:
    """Fractions over adduct states for one charge state.

    The number of each adduct is drawn from Binomial(z, lambda) truncated at max_per_type; the
    resulting states' fractions are the products of per-type binomial weights.
    """
    from math import comb

    names = library.names()
    per_type_dist: dict[str, np.ndarray] = {}
    for name in names:
        lam = propensities.get(name, 0.0)
        kmax = min(library.max_per_type, z)
        ks = np.arange(0, kmax + 1)
        if lam <= 0:
            w = np.zeros(kmax + 1)
            w[0] = 1.0
        else:
            w = np.array([comb(z, int(k)) * lam**k * (1 - lam) ** (z - k) for k in ks])
            w = w / w.sum()
        per_type_dist[name] = w

    # Build states up to max_total adducts by combining independent per-type counts.
    states: dict[AdductState, float] = {AdductState(): 1.0}
    for name in names:
        w = per_type_dist[name]
        new: dict[AdductState, float] = {}
        for st, p in states.items():
            for k, wk in enumerate(w):
                if wk <= 0:
                    continue
                if st.total + k > library.max_total:
                    continue
                counts = tuple(st.counts)
                if k > 0:
                    counts = counts + ((name, k),)
                ns = AdductState(tuple(sorted(counts, key=lambda t: names.index(t[0]))))
                new[ns] = new.get(ns, 0.0) + p * wk
        states = new
    total = sum(states.values())
    if total <= 0:
        return {AdductState(): 1.0}
    return {k: v / total for k, v in states.items() if v / total > 1e-4}
