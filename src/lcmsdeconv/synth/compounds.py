"""Sample analytes and their impurities for each compound class."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..chem.classes import class_isotope_pattern, get_class
from ..chem.formula import Formula
from ..chem.isotopes import isotope_pattern
from .spec import Compound

# Monoisotopic residue masses (as added to a growing chain; water added once at the ends).
_AA_MONO = {
    "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
    "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
    "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
    "H": 137.05891, "F": 147.06841, "R": 156.10111, "Y": 163.06333, "W": 186.07931,
}
# Approximate amino-acid frequencies (UniProt average).
_AA_FREQ = {
    "A": 8.25, "R": 5.53, "N": 4.06, "D": 5.46, "C": 1.38, "Q": 3.93, "E": 6.72,
    "G": 7.07, "H": 2.27, "I": 5.91, "L": 9.65, "K": 5.80, "M": 2.41, "F": 3.86,
    "P": 4.74, "S": 6.65, "T": 5.36, "W": 1.10, "Y": 2.92, "V": 6.86,
}
_WATER = Formula("H2O").mono_mass


@dataclass
class ClassConfig:
    """Sampling ranges for one compound class."""

    name: str
    mass_range: tuple[float, float]
    weight: float = 1.0
    extra: dict = field(default_factory=dict)


DEFAULT_CLASSES = [
    ClassConfig("peptide", (600.0, 150000.0), 1.4),
    ClassConfig("dna", (1500.0, 40000.0), 1.0),
    ClassConfig("rna", (1500.0, 40000.0), 1.0),
    ClassConfig("ps_dna", (1500.0, 40000.0), 0.8),
    ClassConfig("glycan", (600.0, 8000.0), 0.7),
    ClassConfig("peg", (500.0, 20000.0), 0.7),
    ClassConfig("small_molecule", (150.0, 1500.0), 0.9),
]


def _pattern_for_mass(mass: float, cls: str, threshold: float = 1e-4):
    return class_isotope_pattern(mass, cls, threshold)


def sample_peptide(rng: np.random.Generator, mass_range: tuple[float, float]) -> Compound:
    mass = float(np.exp(rng.uniform(np.log(mass_range[0]), np.log(mass_range[1]))))
    # for small peptides build an explicit sequence; for large proteins use averagine
    if mass < 6000:
        residues = list(_AA_FREQ)
        probs = np.array([_AA_FREQ[r] for r in residues])
        probs = probs / probs.sum()
        target = mass - _WATER
        seq = []
        acc = 0.0
        while acc < target:
            r = rng.choice(residues, p=probs)
            seq.append(r)
            acc += _AA_MONO[r]
        formula = Formula("H2O")
        # build composition from residues
        comp = {"G": "C2H3NO", "A": "C3H5NO", "S": "C3H5NO2", "P": "C5H7NO", "V": "C5H9NO",
                "T": "C4H7NO2", "C": "C3H5NOS", "L": "C6H11NO", "I": "C6H11NO", "N": "C4H6N2O2",
                "D": "C4H5NO3", "Q": "C5H8N2O2", "K": "C6H12N2O", "E": "C5H7NO3", "M": "C5H9NOS",
                "H": "C6H7N3O", "F": "C9H9NO", "R": "C6H12N4O", "Y": "C9H9NO2", "W": "C11H10N2O"}
        for r in seq:
            formula = formula + Formula(comp[r])
        pat = isotope_pattern(formula, 1e-4)
        return Compound(pat.average_mass, "peptide", pat, name=f"peptide {len(seq)}aa", kind="main")
    pat = _pattern_for_mass(mass, "peptide")
    return Compound(pat.average_mass, "peptide", pat, name=f"protein {mass/1000:.0f}kDa", kind="main")


def sample_mab(rng: np.random.Generator) -> Compound:
    base = rng.uniform(144000, 150000)
    pat = _pattern_for_mass(base, "peptide")
    return Compound(pat.average_mass, "peptide", pat, name="mAb", kind="main",
                    meta={"glycoform_pairs": True})


def sample_oligo(rng: np.random.Generator, cls: str, mass_range: tuple[float, float]) -> Compound:
    residue = get_class(cls).unit_avg_mass
    n = int(rng.integers(5, max(6, int(mass_range[1] / residue))))
    n = min(n, 120)
    mass = n * residue + 18.0
    pat = _pattern_for_mass(mass, cls)
    return Compound(pat.average_mass, cls, pat, name=f"{cls} {n}mer", kind="main", meta={"n": n})


def sample_polymer(rng: np.random.Generator, cls: str, mass_range: tuple[float, float]) -> Compound:
    unit = get_class(cls).unit_avg_mass
    mean_dp = rng.uniform(mass_range[0] / unit, mass_range[1] / unit)
    mean_dp = max(3.0, mean_dp)
    # Return the centroid oligomer; the run tier expands the DP distribution.
    mass = mean_dp * unit + 18.0
    pat = _pattern_for_mass(mass, cls)
    return Compound(pat.average_mass, cls, pat, name=f"{cls} DP~{mean_dp:.0f}", kind="main",
                    meta={"mean_dp": mean_dp, "unit": unit})


def sample_glycan(rng: np.random.Generator, mass_range: tuple[float, float]) -> Compound:
    mass = float(np.exp(rng.uniform(np.log(mass_range[0]), np.log(mass_range[1]))))
    pat = _pattern_for_mass(mass, "glycan")
    return Compound(pat.average_mass, "glycan", pat, name="glycan", kind="main")


def sample_small_molecule(rng: np.random.Generator, mass_range: tuple[float, float]) -> Compound:
    mass = float(rng.uniform(*mass_range))
    pat = _pattern_for_mass(mass, "small_molecule")
    return Compound(pat.average_mass, "small_molecule", pat, name="small molecule", kind="main")


def sample_compound(rng: np.random.Generator, cls_config: ClassConfig) -> Compound:
    cls = cls_config.name
    mr = cls_config.mass_range
    if cls == "peptide":
        # antibodies only when the configured range actually reaches antibody masses
        if mr[0] <= 144000.0 and mr[1] >= 150000.0 and rng.random() < 0.1:
            return sample_mab(rng)
        return sample_peptide(rng, mr)
    if cls in ("dna", "rna", "ps_dna", "ps_rna"):
        return sample_oligo(rng, cls, mr)
    if cls in ("peg", "ppg", "plga"):
        return sample_polymer(rng, cls, mr)
    if cls == "glycan":
        return sample_glycan(rng, mr)
    if cls == "small_molecule":
        return sample_small_molecule(rng, mr)
    pat = _pattern_for_mass(float(rng.uniform(*mr)), cls)
    return Compound(pat.average_mass, cls, pat, name=cls, kind="main")


def choose_class(rng: np.random.Generator, classes: list[ClassConfig]) -> ClassConfig:
    w = np.array([c.weight for c in classes])
    w = w / w.sum()
    return classes[int(rng.choice(len(classes), p=w))]
