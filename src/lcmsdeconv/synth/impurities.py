"""Sample product-related impurities and unrelated contaminants for a main compound."""

from __future__ import annotations

import numpy as np

from ..chem.classes import class_isotope_pattern
from .compounds import ClassConfig, sample_compound
from .spec import Compound

# Class-specific mass deltas (Da) that generate a realistic impurity.
_DELTAS: dict[str, list[tuple[str, float]]] = {
    "peptide": [
        ("oxidation", 15.9949), ("dioxidation", 31.9898), ("deamidation", 0.9840),
        ("water loss", -18.0106), ("C-term Lys loss", -128.0949), ("+Hex (glycation)", 162.0528),
        ("-Hex", -162.0528), ("-Fuc", -146.0579), ("succinimide", -18.0106),
        ("pyroGlu", -17.0265), ("+Na-H clip", 21.9819),
    ],
    "dna": [
        ("n-1 (dT)", -304.0460), ("n-1 (dC)", -289.0464), ("n+1 (dA)", 313.0576),
        ("depurination", -135.0545), ("+cyanoethyl", 53.0266), ("-H2O", -18.0106),
    ],
    "rna": [
        ("n-1 (U)", -306.0253), ("n-1 (C)", -305.0413), ("n+1 (A)", 329.0525),
        ("PS->PO", -15.9772), ("+cyanoethyl", 53.0266), ("2'-OMe", 14.0157),
    ],
    "ps_dna": [
        ("PS->PO", -15.9772), ("n-1 (dT)", -304.0460), ("depurination", -135.0545),
        ("+cyanoethyl", 53.0266), ("2x PS->PO", -31.9544),
    ],
    "ps_rna": [
        ("PS->PO", -15.9772), ("n-1 (U)", -306.0253), ("+cyanoethyl", 53.0266),
    ],
    "glycan": [
        ("-Hex", -162.0528), ("+Hex", 162.0528), ("+HexNAc", 203.0794),
        ("+NeuAc", 291.0954), ("sulfation", 79.9568),
    ],
    "peg": [("+unit", 44.0262), ("-unit", -44.0262), ("+2 units", 88.0524)],
    "ppg": [("+unit", 58.0419), ("-unit", -58.0419)],
    "plga": [("+lactide", 72.0211), ("+glycolide", 58.0055)],
    "small_molecule": [
        ("oxidation", 15.9949), ("+H2O", 18.0106), ("-H2O", -18.0106),
        ("demethylation", -14.0157), ("+2H", 2.0157),
    ],
}


def _class_deltas(cls: str) -> list[tuple[str, float]]:
    return _DELTAS.get(cls, _DELTAS["peptide"])


def sample_impurities(
    rng: np.random.Generator,
    parent: Compound,
    parent_id: int,
    max_impurities: int = 12,
    abundance_range: tuple[float, float] = (1e-4, 1e-1),
) -> list[tuple[Compound, float]]:
    """Return (impurity compound, relative abundance) pairs for ``parent``.

    Abundances are relative to the parent (=1.0), log-uniform in ``abundance_range``.
    """
    deltas = _class_deltas(parent.compound_class)
    n = int(min(rng.poisson(4), max_impurities))
    out: list[tuple[Compound, float]] = []
    used: set[float] = set()
    for _ in range(n):
        name, dm = deltas[int(rng.integers(len(deltas)))]
        if abs(dm) < 1e-6:
            continue
        mult = 1 if rng.random() > 0.25 else int(rng.integers(2, 4))
        dm_total = dm * mult
        new_mass = parent.mass + dm_total
        if new_mass < 100 or round(dm_total, 3) in used:
            continue
        used.add(round(dm_total, 3))
        pat = class_isotope_pattern(new_mass, parent.compound_class).shifted(
            new_mass - class_isotope_pattern(new_mass, parent.compound_class).average_mass
        )
        abundance = float(np.exp(rng.uniform(np.log(abundance_range[0]), np.log(abundance_range[1]))))
        label = f"{name}" if mult == 1 else f"{mult}x {name}"
        comp = Compound(new_mass, parent.compound_class, pat, name=f"{parent.name} [{label}]",
                        parent_id=parent_id, kind="impurity", meta={"delta": dm_total})
        out.append((comp, abundance))
    return out


def sample_contaminant(
    rng: np.random.Generator, classes: list[ClassConfig]
) -> tuple[Compound, float]:
    """An unrelated co-eluting compound at low abundance."""
    from .compounds import choose_class

    comp = sample_compound(rng, choose_class(rng, classes))
    comp.kind = "contaminant"
    ab = float(np.exp(rng.uniform(np.log(1e-3), np.log(1.0))))
    return comp, ab
