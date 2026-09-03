"""Exact isotope distributions by polynomial (convolution) expansion.

The distribution of each element is raised to its atom count by binary exponentiation of
probability/mass-moment array pairs (Rockwood's method), then elements are convolved. Each
entry of the result is one nominal-mass isotopologue cluster with its intensity-weighted
centroid mass, which is the appropriate representation for resolving powers below ~1e6.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .elements import ISOTOPES, lightest_mass
from .formula import Formula


@dataclass(frozen=True)
class IsotopePattern:
    """Isotopologue centroid masses and abundances (abundances sum to 1)."""

    masses: np.ndarray
    abundances: np.ndarray

    @property
    def mono_mass(self) -> float:
        return float(self.masses[0])

    @property
    def average_mass(self) -> float:
        return float(np.sum(self.masses * self.abundances) / np.sum(self.abundances))

    @property
    def most_abundant_index(self) -> int:
        return int(np.argmax(self.abundances))

    @property
    def most_abundant_mass(self) -> float:
        return float(self.masses[self.most_abundant_index])

    def normalized(self, mode: str = "max") -> np.ndarray:
        if mode == "max":
            return self.abundances / self.abundances.max()
        return self.abundances / self.abundances.sum()

    def truncated(self, threshold: float) -> IsotopePattern:
        keep = self.abundances >= threshold * self.abundances.max()
        first = int(np.argmax(keep))
        last = len(keep) - int(np.argmax(keep[::-1]))
        return IsotopePattern(self.masses[first:last], self.abundances[first:last])

    def shifted(self, delta_mass: float) -> IsotopePattern:
        return IsotopePattern(self.masses + delta_mass, self.abundances)

    def __len__(self) -> int:
        return len(self.masses)


class _Dist:
    """Probability (p) and mass-moment (q) arrays on a nominal-mass index grid."""

    __slots__ = ("p", "q")

    def __init__(self, p: np.ndarray, q: np.ndarray):
        self.p = p
        self.q = q

    @staticmethod
    def delta() -> _Dist:
        return _Dist(np.array([1.0]), np.array([0.0]))

    def convolve(self, other: _Dist, prune: float) -> _Dist:
        p = np.convolve(self.p, other.p)
        q = np.convolve(self.q, other.p) + np.convolve(self.p, other.q)
        # prune negligible tail (keep index 0 anchored to the lightest isotopologue)
        keep = p > prune * p.max()
        last = len(p) - int(np.argmax(keep[::-1]))
        return _Dist(p[:last], q[:last])

    def power(self, n: int, prune: float) -> _Dist:
        result = _Dist.delta()
        base = self
        while n > 0:
            if n & 1:
                result = result.convolve(base, prune)
            n >>= 1
            if n:
                base = base.convolve(base, prune)
        return result


def _element_dist(el: str) -> _Dist:
    isos = ISOTOPES[el]
    m0 = lightest_mass(el)
    offsets = [int(round(i.mass - m0)) for i in isos]
    length = max(offsets) + 1
    p = np.zeros(length)
    q = np.zeros(length)
    for iso, k in zip(isos, offsets):
        p[k] += iso.abundance
        q[k] += iso.abundance * (iso.mass - m0)
    return _Dist(p, q)


@lru_cache(maxsize=4096)
def _pattern_cached(key: tuple[tuple[str, int], ...], threshold: float) -> IsotopePattern:
    prune = min(1e-12, threshold * 1e-3)
    total = _Dist.delta()
    base_mass = 0.0
    for el, n in key:
        if n <= 0:
            continue
        total = total.convolve(_element_dist(el).power(n, prune), prune)
        base_mass += n * lightest_mass(el)
    p = total.p
    q = total.q
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.where(p > 0, q / np.where(p > 0, p, 1.0), 0.0)
    masses = base_mass + centroid + np.arange(len(p)) * 0.0
    # entries with p == 0 (e.g. gaps) get the nominal position
    gaps = p <= 0
    if gaps.any():
        masses[gaps] = base_mass + np.arange(len(p))[gaps]
    keep = p >= threshold * p.max()
    last = len(p) - int(np.argmax(keep[::-1]))
    masses = masses[:last]
    p = p[:last]
    return IsotopePattern(masses.copy(), (p / p.sum()).copy())


def isotope_pattern(formula: Formula | str, threshold: float = 1e-6) -> IsotopePattern:
    """Isotope pattern of a formula. Fractional counts are rounded to integers.

    ``threshold`` is the relative abundance (to the most abundant isotopologue) below which
    trailing isotopologues are dropped.
    """
    f = Formula(formula) if isinstance(formula, str) else formula
    f = f.rounded()
    key = tuple(sorted((el, int(round(n))) for el, n in f.items() if n > 0))
    if any(n < 0 for _, n in f.items()):
        raise ValueError("Isotope pattern requires non-negative element counts")
    if not key:
        return IsotopePattern(np.array([0.0]), np.array([1.0]))
    return _pattern_cached(key, float(threshold))
