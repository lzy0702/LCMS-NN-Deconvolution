"""Element and isotope data.

Masses and abundances follow the IUPAC/NIST 2013 compilation (Meija et al., Pure Appl. Chem.
2016). Only the isotopes relevant to organic, bio- and synthetic polymers are listed.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTON_MASS = 1.00727646688
ELECTRON_MASS = 0.000548579909
NEUTRON_MASS = 1.00866491588
C13_C12_DIFF = 1.0033548378
#: Mean spacing between successive isotopologue centroids of a peptide-like composition.
AVERAGINE_ISOTOPE_SPACING = 1.00235


@dataclass(frozen=True)
class Isotope:
    mass: float
    abundance: float


ISOTOPES: dict[str, tuple[Isotope, ...]] = {
    "H": (Isotope(1.00782503223, 0.999885), Isotope(2.01410177812, 0.000115)),
    "D": (Isotope(2.01410177812, 1.0),),
    "C": (Isotope(12.0, 0.9893), Isotope(13.00335483507, 0.0107)),
    "N": (Isotope(14.00307400443, 0.99636), Isotope(15.00010889888, 0.00364)),
    "O": (
        Isotope(15.99491461957, 0.99757),
        Isotope(16.99913175650, 0.00038),
        Isotope(17.99915961286, 0.00205),
    ),
    "S": (
        Isotope(31.9720711744, 0.9499),
        Isotope(32.9714589098, 0.0075),
        Isotope(33.967867004, 0.0425),
        Isotope(35.96708071, 0.0001),
    ),
    "P": (Isotope(30.97376199842, 1.0),),
    "F": (Isotope(18.99840316273, 1.0),),
    "Cl": (Isotope(34.968852682, 0.7576), Isotope(36.965902602, 0.2424)),
    "Br": (Isotope(78.9183376, 0.5069), Isotope(80.9162897, 0.4931)),
    "I": (Isotope(126.9044719, 1.0),),
    "Na": (Isotope(22.9897692820, 1.0),),
    "K": (
        Isotope(38.9637064864, 0.932581),
        Isotope(39.963998166, 0.000117),
        Isotope(40.9618252579, 0.067302),
    ),
    "Li": (Isotope(6.0151228874, 0.0759), Isotope(7.0160034366, 0.9241)),
    "Si": (
        Isotope(27.97692653465, 0.92223),
        Isotope(28.97649466490, 0.04685),
        Isotope(29.973770136, 0.03092),
    ),
    "Se": (
        Isotope(73.922475934, 0.0089),
        Isotope(75.919213704, 0.0937),
        Isotope(76.919914154, 0.0763),
        Isotope(77.91730928, 0.2377),
        Isotope(79.9165218, 0.4961),
        Isotope(81.9166995, 0.0873),
    ),
    "Fe": (
        Isotope(53.93960899, 0.05845),
        Isotope(55.93493633, 0.91754),
        Isotope(56.93539284, 0.02119),
        Isotope(57.93327443, 0.00282),
    ),
    "Ca": (
        Isotope(39.962590863, 0.96941),
        Isotope(41.95861783, 0.00647),
        Isotope(42.95876644, 0.00135),
        Isotope(43.95548156, 0.02086),
        Isotope(45.9536890, 0.00004),
        Isotope(47.95252276, 0.00187),
    ),
    "Mg": (
        Isotope(23.985041697, 0.7899),
        Isotope(24.985836976, 0.1000),
        Isotope(25.982592968, 0.1101),
    ),
    "Zn": (
        Isotope(63.92914201, 0.4917),
        Isotope(65.92603381, 0.2773),
        Isotope(66.92712775, 0.0404),
        Isotope(67.92484455, 0.1845),
        Isotope(69.9253192, 0.0061),
    ),
    "Cu": (Isotope(62.92959772, 0.6915), Isotope(64.92778970, 0.3085)),
    "B": (Isotope(10.01293695, 0.199), Isotope(11.00930536, 0.801)),
    "Mn": (Isotope(54.93804391, 1.0),),
    "Co": (Isotope(58.93319429, 1.0),),
    "Ni": (
        Isotope(57.93534241, 0.68077),
        Isotope(59.93078588, 0.26223),
        Isotope(60.93105557, 0.011399),
        Isotope(61.92834537, 0.036346),
        Isotope(63.92796682, 0.009255),
    ),
    "Pt": (
        Isotope(189.9599297, 0.00012),
        Isotope(191.9610387, 0.00782),
        Isotope(193.9626809, 0.3286),
        Isotope(194.9647917, 0.3378),
        Isotope(195.96495209, 0.2521),
        Isotope(197.9678949, 0.07356),
    ),
    "Gd": (
        Isotope(151.9197995, 0.0020),
        Isotope(153.9208741, 0.0218),
        Isotope(154.9226305, 0.1480),
        Isotope(155.9221312, 0.2047),
        Isotope(156.9239686, 0.1565),
        Isotope(157.9241123, 0.2484),
        Isotope(159.9270624, 0.2186),
    ),
}


def monoisotopic_mass(element: str) -> float:
    """Mass of the most abundant isotope (the conventional 'monoisotopic' mass)."""
    isos = ISOTOPES[element]
    return max(isos, key=lambda i: i.abundance).mass


def lightest_mass(element: str) -> float:
    return min(i.mass for i in ISOTOPES[element])


def average_mass(element: str) -> float:
    isos = ISOTOPES[element]
    return sum(i.mass * i.abundance for i in isos) / sum(i.abundance for i in isos)


def known_elements() -> list[str]:
    return sorted(ISOTOPES)
