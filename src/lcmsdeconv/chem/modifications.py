"""Known mass deltas of product-related impurities, used to annotate deconvolved species.

Each modification carries its monoisotopic delta and its average-mass delta. A difference of two
average masses (the deconvolved species' masses) is compared with the average delta, one of two
monoisotopic masses with the monoisotopic delta: PS -> PO is -15.977 Da monoisotopic and
-16.066 Da on average masses, and a residue of an oligonucleotide weighs 0.13-0.25 Da more on
average masses.
"""

from __future__ import annotations

from dataclasses import dataclass

from .elements import PROTON_MASS  # noqa: F401  (re-exported for convenience)
from .formula import Formula


@dataclass(frozen=True)
class Modification:
    name: str
    delta: float  # monoisotopic
    classes: tuple[str, ...] = ("any",)
    description: str = ""
    #: the delta of average masses; None where only a number is known (``delta`` serves both)
    avg_delta: float | None = None
    #: whether one molecule carries it at several sites, so that 2x and 3x are read
    #: (:data:`REPEATING`)
    repeats: bool = False

    def delta_for(self, mass_type: str = "mono") -> float:
        """The delta to compare with a difference of two ``mass_type`` masses ("mono" or
        "average")."""
        if mass_type == "average" and self.avg_delta is not None:
            return self.avg_delta
        return self.delta


#: The modifications one molecule commonly carries at several equivalent sites (methionines,
#: asparagines, lysines, linkages, nucleobases, glycosylation sites, the two heavy chains of an
#: antibody, repeat units): the only ones read as 2x and 3x multiples. A multiple of each of the
#: 120-odd entries would make a chance match likely: three cyanoethyl groups (159.19 Da on
#: average masses) named a tripropylamine N-oxide adduct of an oligonucleotide (159.27 Da). Not
#: repeated: synthesis by-products and protecting groups (cyanoethyl, isobutyryl, benzoyl, DMT),
#: end groups (5'-phosphate, pyro-Glu, N-terminal Met loss), depurination, losses and the n-1 /
#: n+1 residues (a shortmer two residues short may lack any two, and 2x and 3x of 80-odd residue
#: deltas would cover much of the mass axis).
REPEATING = frozenset({
    "oxidation", "dioxidation", "deamidation", "deamination (C -> U, A -> I)",
    "PS -> PO (desulfurization)", "+PEG unit", "-PEG unit", "phosphorylation", "sulfation",
    "acetylation", "formylation", "carbamylation", "glycation (+Hex)", "-Hex", "+HexNAc", "+Fuc",
    "-Fuc (afucosylation)", "+NeuAc", "C-terminal Lys loss", "disulfide reduction",
})


def _m(name: str, formula: str | float, classes: tuple[str, ...] = ("any",), desc: str = "") -> Modification:
    """A modification by formula ("O", "O-NH", "-H2O": what is gained, then what is lost) or by
    its delta where only a number is known."""
    if isinstance(formula, str):
        pos, _, neg = formula.partition("-")
        gained = Formula(pos) if pos else Formula()
        lost = Formula(neg) if neg else Formula()
        d, avg = gained.mono_mass - lost.mono_mass, gained.avg_mass - lost.avg_mass
    else:
        d, avg = float(formula), None
    return Modification(name, d, classes, desc, avg, name in REPEATING)


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
    # an abasic site: the base lost (adenine, guanine)
    _m("depurination (dA loss, abasic)", "-C5H5N5", ("dna", "ps_dna")),
    _m("depurination (dG loss, abasic)", "-C5H5N5O", ("dna", "ps_dna")),
    _m("+cyanoethyl", "C3H3N", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("+isobutyryl", "C4H6O", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("+benzoyl", "C7H4O", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("+DMT", "C21H18O2", ("dna", "rna", "ps_dna", "ps_rna")),
    _m("5'-phosphate", "HPO3", ("dna", "rna", "ps_dna", "ps_rna")),
    # a residue lost: the nucleoside monophosphate less water
    _m("n-1 (dA)", "-C10H12N5O5P", ("dna", "ps_dna")),
    _m("n-1 (dC)", "-C9H12N3O6P", ("dna", "ps_dna")),
    _m("n-1 (dG)", "-C10H12N5O6P", ("dna", "ps_dna")),
    _m("n-1 (dT)", "-C10H13N2O7P", ("dna", "ps_dna")),
    _m("n-1 (A)", "-C10H12N5O6P", ("rna", "ps_rna")),
    _m("n-1 (C)", "-C9H12N3O7P", ("rna", "ps_rna")),
    _m("n-1 (G)", "-C10H12N5O7P", ("rna", "ps_rna")),
    _m("n-1 (U)", "-C9H11N2O8P", ("rna", "ps_rna")),
    _m("n-1 (2'-OMe A)", "-C11H14N5O6P", ("rna", "ps_rna")),
    _m("n-1 (2'-OMe C)", "-C10H14N3O7P", ("rna", "ps_rna")),
    _m("n-1 (2'-OMe G)", "-C11H14N5O7P", ("rna", "ps_rna")),
    _m("n-1 (2'-OMe U)", "-C10H13N2O8P", ("rna", "ps_rna")),
    _m("n-1 (2'-F U)", "-C9H10FN2O7P", ("rna", "ps_rna")),
    _m("n-1 (2'-F C)", "-C9H11FN3O6P", ("rna", "ps_rna")),
    _m("n-1 (2'-F A)", "-C10H11FN5O5P", ("rna", "ps_rna")),
    _m("n-1 (2'-F G)", "-C10H11FN5O6P", ("rna", "ps_rna")),
    _m("deamination (C -> U, A -> I)", "O-NH", ("dna", "rna", "ps_dna", "ps_rna"),
       "NH2 -> OH on a nucleobase"),
    _m("+PEG unit", "C2H4O", ("peg", "any")),
    _m("-PEG unit", "-C2H4O", ("peg", "any")),
    _m("dimer", 0.0, ("any",), "M x 2 (handled by ratio, not delta)"),
]
# a phosphorothioate residue weighs S - O more than its phosphodiester
_S_O = Formula("S").mono_mass - Formula("O").mono_mass
_S_O_AVG = Formula("S").avg_mass - Formula("O").avg_mass
MODIFICATIONS += [Modification(m.name[:-1] + ", PS)", m.delta - _S_O,
                               tuple(c for c in m.classes if c.startswith("ps_")), m.description,
                               None if m.avg_delta is None else m.avg_delta - _S_O_AVG)
                  for m in list(MODIFICATIONS) if m.name.startswith("n-1 (")]
# a longmer carries one extra residue: the n-1 deltas with the sign reversed
MODIFICATIONS += [Modification("n+1" + m.name[3:], -m.delta, m.classes, m.description,
                               None if m.avg_delta is None else -m.avg_delta)
                  for m in list(MODIFICATIONS) if m.name.startswith("n-1 ")]


def annotate_delta(delta: float, cls: str, tolerance: float = 0.05, mass: float | None = None,
                   mass_type: str = "mono", multiple_tolerance: float = 0.5) -> list[Modification]:
    """Return modifications whose delta matches ``delta`` within ``tolerance`` (Da), the
    closest first.

    ``delta`` is a difference of two ``mass_type`` masses ("mono", or "average" for the
    deconvolved species' average masses) and is compared with the modifications' deltas of that
    kind. A modification that one molecule carries at several sites (:data:`REPEATING`) is also
    matched two or three times over, within ``multiple_tolerance`` of the tolerance (half, to
    name a species; a check whether a modification could explain a difference passes 1), and
    reported as ``2x name``.
    """
    hits: list[tuple[float, Modification]] = []
    for m in MODIFICATIONS:
        if m.delta == 0.0:
            continue
        if "any" not in m.classes and cls not in m.classes:
            continue
        d = m.delta_for(mass_type)
        for k in (1, 2, 3) if m.repeats else (1,):
            if abs(delta - k * d) <= (tolerance if k == 1 else multiple_tolerance * tolerance):
                hits.append((abs(delta - k * d), m if k == 1 else Modification(
                    f"{k}x {m.name}", k * m.delta, m.classes, m.description,
                    None if m.avg_delta is None else k * m.avg_delta, m.repeats)))
                break
    hits.sort(key=lambda h: h[0])
    return [m for _, m in hits]
