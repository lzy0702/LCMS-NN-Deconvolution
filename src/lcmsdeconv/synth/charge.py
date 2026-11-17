"""Charge-state distributions for ESI of macromolecules."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..chem.classes import get_class


@dataclass(frozen=True)
class ChargeShape:
    """Parameters of one charge-state envelope.

    ``centre`` is the apex charge, ``width`` the Gaussian width in charge units, ``skew`` a linear
    tilt and ``second`` an optional second mode ``(centre, width, weight)``. ``single`` marks the
    small-molecule case, whose distribution is fixed rather than shaped.
    """

    centre: float
    width: float
    skew: float = 0.0
    second: tuple[float, float, float] | None = None
    single: dict[int, float] | None = None


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


def sample_charge_shape(mass: float, cls: str, mode: str, rng: np.random.Generator) -> ChargeShape:
    """Draw the envelope parameters for one analyte."""
    if cls == "small_molecule" or mass < 900:
        # mostly singly charged, occasional 2+
        if rng.random() < 0.9 or mass < 400:
            return ChargeShape(1.0, 0.0, single={1: 1.0})
        return ChargeShape(1.0, 0.0, single={1: 0.8, 2: 0.2})
    zc = apex_charge(mass, cls, mode, rng)
    if mode == "native":
        width = zc * rng.uniform(0.05, 0.10)
    else:
        width = zc * rng.uniform(0.12, 0.25)
    width = max(width, 0.6)
    skew = rng.uniform(-0.4, 0.4)
    second = None
    if rng.random() < 0.15:  # bimodal
        second = (zc * rng.uniform(0.5, 0.85), width * 0.8, 0.6)
    return ChargeShape(zc, width, skew, second)


def distribution_from_shape(shape: ChargeShape, z_max: int = 100,
                            min_fraction: float = 1e-3) -> dict[int, float]:
    """Normalized ``{z: fraction}`` for an envelope shape."""
    if shape.single is not None:
        return dict(shape.single)
    zc, width, skew = shape.centre, shape.width, shape.skew
    zs = np.arange(1, z_max + 1)
    g = np.exp(-0.5 * ((zs - zc) / width) ** 2) * (1.0 + skew * (zs - zc) / (width * 3))
    g = np.clip(g, 0.0, None)
    if shape.second is not None:
        zc2, w2, a2 = shape.second
        g = g + a2 * np.exp(-0.5 * ((zs - zc2) / w2) ** 2)
    s = g.sum()
    if s <= 0:
        return {max(1, int(round(zc))): 1.0}
    g = g / s
    keep = g >= min_fraction
    zs, g = zs[keep], g[keep]
    g = g / g.sum()
    return {int(z): float(f) for z, f in zip(zs, g)}


def related_shape(parent: ChargeShape, parent_mass: float, mass: float, cls: str,
                  rng: np.random.Generator, jitter: float = 0.03) -> ChargeShape:
    """Envelope of a product-related impurity, derived from its parent's envelope.

    Oxidation, deamidation, a clipped lysine or an n-1 deletion barely change how a molecule
    charges, so an impurity's envelope is its parent's, with the apex moved by the mass ratio
    (z ~ M**b) and a few percent of jitter. Sampling it independently instead would give a 1 %
    impurity charge states its parent never shows, which no real spectrum does.
    """
    if parent.single is not None:
        return parent
    b = get_class(cls).charge_b
    scale = (max(mass, 1.0) / max(parent_mass, 1.0)) ** b * rng.uniform(1.0 - jitter, 1.0 + jitter)
    second = None
    if parent.second is not None:
        zc2, w2, a2 = parent.second
        second = (zc2 * scale, w2 * scale, a2)
    return replace(parent, centre=max(1.0, parent.centre * scale),
                   width=max(0.6, parent.width * scale), second=second)


def mean_charge(distribution: dict[int, float]) -> float:
    if not distribution:
        return 1.0
    zs = np.array(list(distribution.keys()), dtype=float)
    w = np.array(list(distribution.values()), dtype=float)
    return float((zs * w).sum() / max(w.sum(), 1e-12))


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
    return distribution_from_shape(sample_charge_shape(mass, cls, mode, rng), z_max, min_fraction)
