"""Charge-state distributions for ESI of macromolecules."""

from __future__ import annotations

import numpy as np

from ..chem.classes import get_class


def apex_charge(mass: float, cls: str, mode: str, rng: np.random.Generator | None = None) -> float:
    """Mean apex charge z ~ a * mass**b, with class- and mode-dependent coefficients."""
    c = get_class(cls)
    a, b = c.charge_a, c.charge_b
    if mode == "native":
        a, b = 0.0778, 0.5
    z = a * mass**b
    if rng is not None:
        z *= rng.uniform(0.85, 1.2)
    return max(1.0, z)


def charge_distribution(
    mass: float,
    cls: str,
    mode: str,
    polarity: int,
    rng: np.random.Generator,
    z_max: int = 100,
    min_fraction: float = 1e-3,
) -> dict[int, float]:
    """Sample a normalized charge-state distribution {z: fraction}."""
    if cls == "small_molecule" or mass < 900:
        # mostly singly charged, occasional 2+
        if rng.random() < 0.9 or mass < 400:
            return {1: 1.0}
        return {1: 0.8, 2: 0.2}
    zc = apex_charge(mass, cls, mode, rng)
    if mode == "native":
        width = zc * rng.uniform(0.05, 0.10)
    else:
        width = zc * rng.uniform(0.12, 0.25)
    width = max(width, 0.6)
    skew = rng.uniform(-0.4, 0.4)
    zs = np.arange(1, z_max + 1)
    g = np.exp(-0.5 * ((zs - zc) / width) ** 2) * (1.0 + skew * (zs - zc) / (width * 3))
    g = np.clip(g, 0.0, None)
    if rng.random() < 0.15:  # bimodal
        zc2 = zc * rng.uniform(0.5, 0.85)
        g = g + 0.6 * np.exp(-0.5 * ((zs - zc2) / (width * 0.8)) ** 2)
    s = g.sum()
    if s <= 0:
        return {max(1, int(round(zc))): 1.0}
    g = g / s
    keep = g >= min_fraction
    zs, g = zs[keep], g[keep]
    g = g / g.sum()
    return {int(z): float(f) for z, f in zip(zs, g)}
