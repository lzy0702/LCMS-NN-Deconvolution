"""Known mass deltas of product-related impurities, used to annotate deconvolved species."""

from __future__ import annotations

from dataclasses import dataclass

from .elements import PROTON_MASS  # noqa: F401  (re-exported for convenience)
from .formula import Formula


@dataclass(frozen=True)
class Modification:
    name: str
    delta: float
    classes: tuple[str, ...] = ("any",)
    description: str = ""


def _m(name: str, formula: str | float, classes: tuple[str, ...] = ("any",), desc: str = "") -> Modification:
    if isinstance(formula, str):
        pos, _, neg = formula.partition("-")
        d = (Formula(pos).mono_mass if pos else 0.0) - (Formula(neg).mono_mass if neg else 0.0)
    else:
        d = float(formula)
    return Modification(name, d, classes, desc)


MODIFICATIONS: list[Modification] = [
    _m("oxidation", "O", ("peptide", "small_molecule", "any"), "+O (Met/Trp oxidation)"),
    _m("dioxidation", "O2", ("peptide", "any")),
    _m("deamidation", "O-NH", ("peptide",), "Asn/Gln deamidation"),
    _m("pyro-Glu (Gln)", "-NH3", ("peptide",)),
    _m("pyro-Glu (Glu)", "-H2O", ("peptide",)),
    _m("water loss", "-H2O", ("any",)),
    _m("ammonia loss", "-NH3", ("any",)),
    _m("C-terminal Lys loss", "-C6H12N2O", ("peptide",)),
    _m("N-terminal Met loss", "-C5H9NOS", ("peptide",)),
    _m("acetylation", "C2H2O", ("peptide", "glycan", "any")),
    _m("formylation", "CO", ("peptide", "any")),
    _m("carbamylation", "CHNO", ("peptide",)),
    _m("glycation (+Hex)", "C6H10O5", ("peptide", "glycan")),
    _m("+HexNAc", "C8H13NO5", ("peptide", "glycan")),
    _m("+Fuc", "C6H10O4", ("peptide", "glycan")),
    _m("+NeuAc", "C11H17NO8", ("peptide", "glycan")),
    _m("-Hex", "-C6H10O5", ("peptide", "glycan")),
    _m("-Fuc (afucosylation)", "-C6H10O4", ("peptide", "glycan")),
    _m("phosphorylation", "HPO3", ("peptide", "glycan", "any")),
    _m("sulfation", "SO3", ("peptide", "glycan", "any")),
    _m("disulfide reduction", "H2", ("peptide",)),
    _m("disulfide formation", "-H2", ("peptide",)),
    _m("succinimide", "-H2O", ("peptide",)),
    _m("PS -> PO (desulfurization)", "O-S", ("dna", "rna", "ps_dna", "ps_rna", "any")),
    _m("depurination (dA loss, abasic)", -135.0545, ("dna", "ps_dna")),
    _m("depurination (dG loss, abasic)", -151.0494, ("dna", "ps_dna")),
    _m("+cyanoethyl", "C3H3N", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("+isobutyryl", "C4H6O", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("+benzoyl", "C7H4O", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("+DMT", "C21H18O2", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("5'-phosphate", "HPO3", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("n-1 (dA)", -313.0576, ("dna", "ps_dna")),
    _m("n-1 (dC)", -289.0464, ("dna", "ps_dna")),
    _m("n-1 (dG)", -329.0525, ("dna", "ps_dna")),
    _m("n-1 (dT)", -304.0460, ("dna", "ps_dna")),
    _m("n-1 (A)", -329.0525, ("rna", "ps_rna")),
    _m("n-1 (C)", -305.0413, ("rna", "ps_rna")),
    _m("n-1 (G)", -345.0474, ("rna", "ps_rna")),
    _m("n-1 (U)", -306.0253, ("rna", "ps_rna")),
    _m("n-1 (2'-OMe A)", -343.0682, ("rna", "ps_rna")),
    _m("n-1 (2'-OMe C)", -319.0570, ("rna", "ps_rna")),
    _m("n-1 (2'-OMe G)", -359.0631, ("rna", "ps_rna")),
    _m("n-1 (2'-OMe U)", -320.0410, ("rna", "ps_rna")),
    _m("n-1 (2'-F U)", -308.0209, ("rna", "ps_rna")),
    _m("n-1 (2'-F C)", -307.0369, ("rna", "ps_rna")),
    _m("+PEG unit", "C2H4O", ("peg", "any")),
    _m("-PEG unit", "-C2H4O", ("peg", "any")),
    _m("dimer", 0.0, ("any",), "M x 2 (handled by ratio, not delta)"),
]


def annotate_delta(delta: float, cls: str, tolerance: float = 0.05, mass: float | None = None) -> list[Modification]:
    """Return modifications whose delta matches ``delta`` within ``tolerance`` (Da).

    Deltas that are integer multiples (1..3) of a modification are also reported as
    ``2x name``. Unit-repeat matches for polymers are reported when the class is polymeric.
    """
    hits: list[Modification] = []
    for m in MODIFICATIONS:
        if m.delta == 0.0:
            continue
        if "any" not in m.classes and cls not in m.classes:
            continue
        for k in (1, 2, 3):
            if abs(delta - k * m.delta) <= tolerance:
                hits.append(m if k == 1 else Modification(f"{k}x {m.name}", k * m.delta, m.classes, m.description))
                break
    hits.sort(key=lambda m: abs(m.delta - delta))
    return hits
