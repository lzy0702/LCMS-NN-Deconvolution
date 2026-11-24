"""Per-run adduct propensities and per-ion adduct-state fractions."""

from __future__ import annotations

from functools import lru_cache

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


def sample_run_adduct_load(
    rng: np.random.Generator, library: AdductLibrary, max_per_ion: float = 0.3
) -> dict[str, float]:
    """Mean number of each adduct type per ion at the apex charge, in [0, max_per_ion].

    This is the quantity a spectrum shows directly: a load of 0.1 sodium means the +Na peak is
    about a tenth of the base peak at the envelope apex. It replaces a per-charge propensity,
    which made the load grow with charge without bound: at 0.1 per charge a z = 40 protein ion
    carries four sodium ions on average and its un-adducted form all but disappears, which is
    not what intact-protein spectra look like.
    """
    return sample_run_propensities(rng, library, max_per_ion)


def propensities_for_component(load: dict[str, float], charges: dict[int, float],
                               cap: float = 0.5) -> dict[str, float]:
    """Per-charge propensities that give ``load`` adducts per ion at the mean charge."""
    from .charge import mean_charge

    zbar = max(mean_charge(charges), 1.0)
    return {name: float(min(cap, mu / zbar)) for name, mu in load.items()}


def adduct_fractions_for_charge(
    z: int,
    propensities: dict[str, float],
    library: AdductLibrary,
    rng: np.random.Generator,
) -> dict[AdductState, float]:
    """Fractions over adduct states for one charge state.

    The number of each adduct is drawn from Binomial(z, lambda) truncated at max_per_type; the
    resulting states' fractions are the products of per-type binomial weights. (``rng`` is not
    used: the fractions are expected values, not a draw.)
    """
    return adduct_fractions_by_charge([z], propensities, library)[z]


def adduct_fractions_by_charge(charges, propensities: dict[str, float],
                               library: AdductLibrary) -> dict[int, dict[AdductState, float]]:
    """:func:`adduct_fractions_for_charge` for every charge of ``charges``, in their order.

    Equal, value for value and in the same dict order, to the step-by-step construction of
    :func:`_adduct_fractions_reference`: every state is reached along exactly one path (its count
    of each adduct type), so its fraction is the product of its per-type weights taken in the
    library's order, and the states come in the lexicographic order of their counts. The states
    allowed for a library and charge are enumerated once (:func:`_state_table`); per charge only
    the per-type weights (computed as the reference computes them, the powers shared between
    charges) and one product per state remain.
    """
    names = library.names()
    if not names:
        return {z: {AdductState(): 1.0} for z in charges}
    if len(set(names)) != len(names):
        # a type listed twice reaches some states along two paths, which the reference adds up;
        # the one-path construction below does not, so such a library takes the reference
        return {z: _adduct_fractions_reference(z, propensities, library, None) for z in charges}
    weights = [_TypeWeights(propensities.get(name, 0.0)) for name in names]
    out = {}
    for z in charges:
        kmax = min(library.max_per_type, z)
        ks = np.arange(0, kmax + 1)
        ws = [tw.weights(z, ks, kmax) for tw in weights]
        counts, states = _state_table(tuple(names), int(kmax), int(library.max_total))
        p = ws[0][counts[:, 0]]
        ok = ~(p <= 0)
        for j in range(1, len(names)):
            wj = ws[j][counts[:, j]]
            ok &= ~(wj <= 0)
            p = p * wj
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            out[z] = {AdductState(): 1.0}
            continue
        p = p[idx]
        # the reference's sum(states.values()): its values are numpy floats, so plain additions
        # in order, which an accumulation repeats exactly
        total = np.add.accumulate(p)[-1]
        if total <= 0:
            out[z] = {AdductState(): 1.0}
            continue
        frac = p / total
        keep = frac > 1e-4
        out[z] = dict(zip([states[i] for i in idx[keep]], frac[keep]))
    return out


class _TypeWeights:
    """Binomial weights of one adduct type for one propensity at any charge, each computed with
    exactly the reference's expression ``comb(z, k) * lam**k * (1 - lam) ** (z - k)`` (numpy
    integer exponents); the powers, which do not depend on the charge, are computed once."""

    __slots__ = ("lam", "lam_pow", "rest_pow")

    def __init__(self, lam: float):
        self.lam = lam
        self.lam_pow: dict[int, object] = {}
        self.rest_pow: dict[int, object] = {}

    def weights(self, z, ks: np.ndarray, kmax: int) -> np.ndarray:
        from math import comb

        lam = self.lam
        if lam <= 0:
            w = np.zeros(kmax + 1)
            w[0] = 1.0
            return w
        terms = []
        for k in ks:
            lk = self.lam_pow.get(int(k))
            if lk is None:
                lk = self.lam_pow[int(k)] = lam**k
            e = z - k
            rest = self.rest_pow.get(int(e))
            if rest is None:
                rest = self.rest_pow[int(e)] = (1 - lam) ** e
            terms.append(comb(z, int(k)) * lk * rest)
        w = np.array(terms)
        return w / w.sum()


@lru_cache(maxsize=256)
def _state_table(names: tuple[str, ...], kmax: int, max_total: int):
    """Every adduct state with at most ``kmax`` of each type and ``max_total`` in all: the count
    of each type, ``[n_states, n_types]``, and the states, in the order in which the reference
    construction inserts them (lexicographic in the counts, the last type fastest)."""
    rows: list[tuple[int, ...]] = []

    def grow(prefix: tuple[int, ...], left: int) -> None:
        if len(prefix) == len(names):
            rows.append(prefix)
            return
        for k in range(0, min(kmax, left) + 1):
            grow(prefix + (k,), left - k)

    grow((), max_total)
    counts = np.array(rows, dtype=np.int64).reshape(len(rows), len(names))
    states = [AdductState(tuple((n, k) for n, k in zip(names, r) if k > 0)) for r in rows]
    return counts, states


def _adduct_fractions_reference(
    z: int,
    propensities: dict[str, float],
    library: AdductLibrary,
    rng: np.random.Generator,
) -> dict[AdductState, float]:
    """The original state-by-state construction, which :func:`adduct_fractions_by_charge`
    reproduces exactly (kept for the tests)."""
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


def member_distributions(cfg, members, library: AdductLibrary, run_adducts: dict[str, float],
                         rng: np.random.Generator,
                         real_like: bool = False) -> list[tuple[dict[int, float], dict]]:
    """Charge and adduct distributions for a main compound followed by its impurities.

    ``run_adducts`` is either a per-ion adduct load (default) or, when ``cfg.adduct_max_lambda``
    is set, the legacy per-charge propensities. Impurities inherit their parent's charge
    envelope unless ``cfg.impurity_charge_from_parent`` is off. ``real_like``: oligonucleotides
    charge as in real ion-pairing runs (``synth.charge.sample_charge_shape``).
    """
    from .charge import distribution_from_shape, related_shape, sample_charge_shape

    legacy_adducts = cfg.adduct_max_lambda is not None
    out = []
    parent_shape = None
    parent_mass = 0.0
    for idx, (comp, _rel) in enumerate(members):
        if (idx > 0 and parent_shape is not None and cfg.impurity_charge_from_parent
                and comp.kind == "impurity"):
            shape = related_shape(parent_shape, parent_mass, comp.mass, comp.compound_class, rng,
                                  real_like=real_like)
        else:
            shape = sample_charge_shape(comp.mass, comp.compound_class, cfg.esi_mode, rng,
                                        real_like=real_like)
            if idx == 0:
                shape = _single_state(shape, comp, cfg, rng, real_like)
                parent_shape, parent_mass = shape, comp.mass
        cd = distribution_from_shape(shape, cfg.z_max)
        props = run_adducts if legacy_adducts else propensities_for_component(run_adducts, cd)
        out.append((cd, adduct_fractions_by_charge(cd, props, library)))
    return out


def _single_state(shape, comp, cfg, rng: np.random.Generator, real_like: bool):
    """A real-like oligonucleotide main compound's envelope reduced to one charge state, with
    probability ``cfg.oligo_single_state_prob`` (``SynthConfig``): its apex charge times
    U(0.8, 1.25), rounded. Draws nothing unless the option is on and applies."""
    from dataclasses import replace

    from .charge import OLIGO_CLASSES

    p = getattr(cfg, "oligo_single_state_prob", 0.0)
    if not (p > 0 and real_like and comp.compound_class in OLIGO_CLASSES and comp.mass >= 900
            and shape.single is None):
        return shape
    if rng.random() >= p:
        return shape
    z = int(round(shape.centre * rng.uniform(0.8, 1.25)))
    return replace(shape, single={min(max(1, z), cfg.z_max): 1.0})


def run_adduct_draw(cfg, library: AdductLibrary, rng: np.random.Generator) -> dict[str, float]:
    """Per-run adduct load (default) or legacy per-charge propensities."""
    if cfg.adduct_max_lambda is not None:
        return sample_run_propensities(rng, library, cfg.adduct_max_lambda)
    return sample_run_adduct_load(rng, library, cfg.adduct_max_per_ion)
