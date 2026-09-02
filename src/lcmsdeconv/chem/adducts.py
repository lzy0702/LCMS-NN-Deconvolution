"""Charge carriers and adduct library.

Convention: the neutral mass M refers to the fully protonated/deprotonated species. An
"adduct state" adds a mass delta at constant charge: cation-for-proton exchanges (Na-H, K-H,
NH4-H) and neutral adducts (amines, acids, solvents). Multiply-charged polymer ions carried by
z sodium ions are therefore represented as the z-fold Na-H adduct state of a proton-charged
ion, which keeps a single m/z convention: m/z = (M + delta + z * polarity * m_p) / z.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

from .elements import PROTON_MASS
from .formula import Formula


@dataclass(frozen=True)
class Adduct:
    name: str
    formula: Formula
    category: str = "cation"  # cation | amine | acid | solvent | other
    polarity: int = 0  # 0 = both, +1 positive only, -1 negative only
    aliases: tuple[str, ...] = ()

    @property
    def mass(self) -> float:
        return self.formula.mono_mass


def _f(s: str) -> Formula:
    return Formula(s)


ADDUCTS: dict[str, Adduct] = {}


def register_adduct(a: Adduct) -> Adduct:
    ADDUCTS[a.name] = a
    for al in a.aliases:
        ADDUCTS.setdefault(al, a)
    return a


for _a in [
    Adduct("Na", _f("Na") - _f("H"), "cation", 0, ("sodium",)),
    Adduct("K", _f("K") - _f("H"), "cation", 0, ("potassium",)),
    Adduct("NH4", _f("NH3"), "cation", 0, ("ammonium",)),
    Adduct("Li", _f("Li") - _f("H"), "cation", 0),
    Adduct("Ca", _f("Ca") - _f("H2"), "cation", 0),
    Adduct("Fe", _f("Fe") - _f("H3"), "cation", 0),
    Adduct("TEA", _f("C6H15N"), "amine", -1, ("triethylamine", "HA", "hexylamine", "DMBA")),
    Adduct("DIPEA", _f("C8H19N"), "amine", -1, ("DBA", "dibutylamine", "octylamine")),
    Adduct("DEA", _f("C4H11N"), "amine", -1, ("diethylamine",)),
    Adduct("TBA", _f("C12H27N"), "amine", -1, ("tributylamine",)),
    Adduct("DMCHA", _f("C8H17N"), "amine", -1),
    Adduct("HFIP", _f("C3H2F6O"), "solvent", -1),
    Adduct("TFA", _f("C2HF3O2"), "acid", 0, ("trifluoroacetate", "trifluoroacetic acid")),
    Adduct("formate", _f("CH2O2"), "acid", 0, ("HCOOH", "FA")),
    Adduct("acetate", _f("C2H4O2"), "acid", 0, ("HOAc", "AcOH")),
    Adduct("HCl", _f("HCl"), "acid", 0),
    Adduct("H3PO4", _f("H3PO4"), "acid", 0),
    Adduct("H2O", _f("H2O"), "solvent", 0, ("water",)),
    Adduct("ACN", _f("C2H3N"), "solvent", 1, ("acetonitrile",)),
    Adduct("MeOH", _f("CH4O"), "solvent", 1, ("methanol",)),
]:
    register_adduct(_a)


#: Default adduct sets per chromatography mode.
MODE_DEFAULTS: dict[str, list[str]] = {
    "rplc": ["Na", "K", "NH4"],
    "rplc_tfa": ["Na", "K", "TFA"],
    "iprp": ["TEA", "DIPEA", "HFIP", "Na", "K"],
    "ip-rplc": ["TEA", "DIPEA", "HFIP", "Na", "K"],
    "hilic": ["NH4", "Na", "K", "formate", "acetate"],
    "ip-hilic": ["TEA", "DIPEA", "NH4", "Na", "K"],
    "nplc": ["Na", "K", "NH4"],
    "sec": ["Na", "K", "NH4"],
    "native": ["Na", "K", "NH4"],
    "polymer": ["Na", "K", "NH4"],
    "none": [],
}


@dataclass(frozen=True)
class AdductState:
    """A combination of adducts, e.g. ((Na, 2), (K, 1))."""

    counts: tuple[tuple[str, int], ...] = ()

    @property
    def mass(self) -> float:
        return sum(ADDUCTS[n].mass * k for n, k in self.counts)

    @property
    def total(self) -> int:
        return sum(k for _, k in self.counts)

    @property
    def label(self) -> str:
        if not self.counts:
            return ""
        return "".join(f"+{k if k > 1 else ''}{n}" for n, k in self.counts)

    def __str__(self) -> str:
        return self.label or "base"

    def as_dict(self) -> dict[str, int]:
        return {n: k for n, k in self.counts}


def carrier_mass(polarity: int) -> float:
    """Signed mass of the charge carrier: +proton for positive, -proton for negative."""
    return PROTON_MASS if polarity >= 0 else -PROTON_MASS


def mz_from_mass(mass: float, z: int, polarity: int = 1, adduct_mass: float = 0.0) -> float:
    return (mass + adduct_mass + z * carrier_mass(polarity)) / z


def mass_from_mz(mz: float, z: int, polarity: int = 1, adduct_mass: float = 0.0) -> float:
    return z * (mz - carrier_mass(polarity)) - adduct_mass


@dataclass
class AdductLibrary:
    adducts: list[Adduct] = field(default_factory=list)
    max_per_type: int = 3
    max_total: int = 4

    @classmethod
    def from_mode(
        cls,
        mode: str = "rplc",
        polarity: int = 0,
        include: tuple[str, ...] | list[str] = (),
        exclude: tuple[str, ...] | list[str] = (),
        max_per_type: int = 3,
        max_total: int = 4,
    ) -> AdductLibrary:
        names = list(MODE_DEFAULTS.get(mode.lower(), MODE_DEFAULTS["rplc"]))
        for n in include:
            if n not in names:
                names.append(n)
        names = [n for n in names if n not in set(exclude)]
        adducts = []
        for n in names:
            a = ADDUCTS.get(n)
            if a is None:
                # allow ad-hoc formulas such as "C7H17N"
                a = Adduct(n, Formula(n), "other", 0)
            if polarity and a.polarity and a.polarity != polarity:
                continue
            if a.name not in [x.name for x in adducts]:
                adducts.append(a)
        return cls(adducts, max_per_type, max_total)

    def names(self) -> list[str]:
        return [a.name for a in self.adducts]

    def states(self, max_total: int | None = None, max_types_mixed: int = 2) -> list[AdductState]:
        """All adduct states up to ``max_total`` adducts (mixing at most ``max_types_mixed`` types)."""
        max_total = self.max_total if max_total is None else max_total
        states = {AdductState()}
        names = self.names()
        for total in range(1, max_total + 1):
            for combo in combinations_with_replacement(names, total):
                types = set(combo)
                if len(types) > max_types_mixed:
                    continue
                counts = tuple(sorted(((n, combo.count(n)) for n in types), key=lambda t: names.index(t[0])))
                if any(k > self.max_per_type for _, k in counts):
                    continue
                states.add(AdductState(counts))
        return sorted(states, key=lambda s: (s.total, s.mass))

    def deltas(self) -> dict[str, float]:
        return {a.name: a.mass for a in self.adducts}
