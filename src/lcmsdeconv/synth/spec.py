"""Data structures shared by the synthetic generator (single source of truth for labels)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..chem.isotopes import IsotopePattern


@dataclass
class Compound:
    """A neutral analyte (base species): its class, mass and isotope pattern."""

    mass: float
    compound_class: str
    pattern: IsotopePattern
    name: str = ""
    parent_id: int = -1  # index of the compound this is an impurity of (-1 = main)
    kind: str = "main"  # main | impurity | contaminant
    meta: dict = field(default_factory=dict)

    @property
    def mono_mass(self) -> float:
        return self.pattern.mono_mass

    @property
    def average_mass(self) -> float:
        return self.pattern.average_mass


@dataclass
class ComponentTruth:
    """Ground truth for one analyte in one frame."""

    id: int
    mass: float
    mono_mass: float
    average_mass: float
    compound_class: str
    intensity: float
    charges: dict[int, float]  # z -> relative fraction
    adducts: dict[str, float]  # adduct-state label -> relative fraction
    kind: str = "main"
    parent_id: int = -1
    name: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mass": self.mass,
            "mono_mass": self.mono_mass,
            "average_mass": self.average_mass,
            "compound_class": self.compound_class,
            "intensity": self.intensity,
            "charges": {int(k): float(v) for k, v in self.charges.items()},
            "adducts": {k: float(v) for k, v in self.adducts.items()},
            "kind": self.kind,
            "parent_id": self.parent_id,
            "name": self.name,
        }


@dataclass
class Sticks:
    """Flattened isotopologue sticks for a scene, with provenance for labelling."""

    mz: np.ndarray  # observed m/z
    intensity: np.ndarray
    comp_id: np.ndarray  # int
    z: np.ndarray  # int charge
    adduct_mass: np.ndarray  # adduct mass delta

    def __len__(self) -> int:
        return self.mz.size

    def concat(self, other: Sticks) -> Sticks:
        return Sticks(
            np.concatenate([self.mz, other.mz]),
            np.concatenate([self.intensity, other.intensity]),
            np.concatenate([self.comp_id, other.comp_id]),
            np.concatenate([self.z, other.z]),
            np.concatenate([self.adduct_mass, other.adduct_mass]),
        )

    @staticmethod
    def empty() -> Sticks:
        z = np.zeros(0)
        return Sticks(z.copy(), z.copy(), z.astype(int), z.astype(int), z.copy())
