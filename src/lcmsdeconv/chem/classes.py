"""Compound classes: average compositions ("averagine" family) and class-dependent priors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .formula import Formula
from .isotopes import IsotopePattern, isotope_pattern


@dataclass(frozen=True)
class CompoundClass:
    name: str
    unit: Formula
    description: str = ""
    default_polarity: int = 1
    #: Typical apex charge model z = a * M**b for denaturing ESI (used as a prior only).
    charge_a: float = 0.17
    charge_b: float = 0.5

    @property
    def unit_avg_mass(self) -> float:
        return self.unit.avg_mass

    @property
    def unit_mono_mass(self) -> float:
        return self.unit.mono_mass

    @property
    def mass_defect_per_da(self) -> float:
        """Expected (mono - nominal) mass defect per Da of mass for this class."""
        m = self.unit.mono_mass
        return (m - round(m)) / m if m > 0 else 0.0

    def average_formula(self, mass: float, average: bool = True) -> Formula:
        return self.unit.scaled_to_mass(mass, adjust="H", average=average)


# Average residue compositions.
_CLASSES: dict[str, CompoundClass] = {}


def register_class(cls: CompoundClass) -> CompoundClass:
    _CLASSES[cls.name] = cls
    return cls


register_class(
    CompoundClass(
        "peptide",
        Formula({"C": 4.9384, "H": 7.7583, "N": 1.3577, "O": 1.4773, "S": 0.0417}),
        "Averagine (Senko 1995): peptides and proteins",
        1,
        0.17,
        0.5,
    )
)
# Average DNA nucleotide residue: mean of dAMP, dCMP, dGMP, dTMP (as phosphodiester residues).
register_class(
    CompoundClass(
        "dna",
        Formula({"C": 9.75, "H": 12.25, "N": 3.75, "O": 6.0, "P": 1.0}),
        "Average DNA nucleotide residue (phosphodiester)",
        -1,
        0.02,
        0.63,
    )
)
register_class(
    CompoundClass(
        "rna",
        Formula({"C": 9.5, "H": 11.75, "N": 3.75, "O": 7.0, "P": 1.0}),
        "Average RNA nucleotide residue (phosphodiester)",
        -1,
        0.02,
        0.63,
    )
)
register_class(
    CompoundClass(
        "ps_dna",
        Formula({"C": 9.75, "H": 12.25, "N": 3.75, "O": 5.0, "P": 1.0, "S": 1.0}),
        "Phosphorothioate DNA residue",
        -1,
        0.02,
        0.63,
    )
)
register_class(
    CompoundClass(
        "ps_rna",
        Formula({"C": 9.5, "H": 11.75, "N": 3.75, "O": 6.0, "P": 1.0, "S": 1.0}),
        "Phosphorothioate RNA residue",
        -1,
        0.02,
        0.63,
    )
)
register_class(
    CompoundClass(
        "glycan",
        Formula({"C": 6.3, "H": 10.4, "N": 0.15, "O": 5.0}),
        "Hexose-rich oligosaccharide residue (Hex/HexNAc mix)",
        1,
        0.05,
        0.5,
    )
)
register_class(CompoundClass("peg", Formula({"C": 2, "H": 4, "O": 1}), "Poly(ethylene glycol) repeat", 1, 0.0007, 1.0))
register_class(CompoundClass("ppg", Formula({"C": 3, "H": 6, "O": 1}), "Poly(propylene glycol) repeat", 1, 0.0007, 1.0))
register_class(CompoundClass("plga", Formula({"C": 2.5, "H": 3, "O": 2}), "PLGA repeat (50:50)", 1, 0.0007, 1.0))
register_class(
    CompoundClass(
        "small_molecule",
        Formula({"C": 4.9384, "H": 7.7583, "N": 1.3577, "O": 1.4773, "S": 0.0417}),
        "Small molecules (averagine used when no formula is given)",
        1,
        0.0,
        0.0,
    )
)
register_class(
    CompoundClass(
        "generic",
        Formula({"C": 4.9384, "H": 7.7583, "N": 1.3577, "O": 1.4773, "S": 0.0417}),
        "Unknown class (averagine)",
        1,
        0.17,
        0.5,
    )
)


def get_class(name: str) -> CompoundClass:
    try:
        return _CLASSES[name]
    except KeyError as e:
        raise KeyError(f"Unknown compound class {name!r}; known: {sorted(_CLASSES)}") from e


def class_names() -> list[str]:
    return sorted(_CLASSES)


def register_repeat_unit(name: str, formula: str, description: str = "", polarity: int = 1) -> CompoundClass:
    """Register a user-defined polymer/class from a repeat-unit formula."""
    return register_class(CompoundClass(name, Formula(formula), description, polarity, 0.0007, 1.0))


def _quantize(mass: float, rel: float = 0.002) -> int:
    return int(round(math.log(max(mass, 1.0)) / rel))


@lru_cache(maxsize=8192)
def _class_pattern_cached(name: str, qmass: int, threshold: float, rel: float) -> IsotopePattern:
    cls = get_class(name)
    mass = math.exp(qmass * rel)
    f = cls.average_formula(mass, average=True)
    return isotope_pattern(f, threshold)


def class_isotope_pattern(mass: float, cls: str | CompoundClass, threshold: float = 1e-4) -> IsotopePattern:
    """Isotope pattern of an average composition of the given class near ``mass``.

    The returned pattern belongs to a quantized mass (0.2 % bins); use offsets relative to
    ``pattern.average_mass`` or ``pattern.mono_mass`` when placing it at an exact mass.
    """
    name = cls.name if isinstance(cls, CompoundClass) else cls
    if name not in _CLASSES:
        raise KeyError(f"Unknown compound class {name!r}")
    return _class_pattern_cached(name, _quantize(mass), float(threshold), 0.002)


def expected_mass_defect(mass: float, cls: str | CompoundClass) -> float:
    """Expected fractional part of the *monoisotopic* mass for the class at ``mass``."""
    c = cls if isinstance(cls, CompoundClass) else get_class(cls)
    return (mass * c.mass_defect_per_da) % 1.0


def mono_to_average_offset(mass: float, cls: str | CompoundClass, threshold: float = 1e-4) -> float:
    p = class_isotope_pattern(mass, cls, threshold)
    return p.average_mass - p.mono_mass
