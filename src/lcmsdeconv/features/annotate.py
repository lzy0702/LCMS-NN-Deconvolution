"""Annotate deconvolved species relative to the main species of their region."""

from __future__ import annotations

from ..chem.modifications import annotate_delta
from ..core.model import Species


def annotate_species(
    species: list[Species],
    main: Species | None = None,
    tolerance_da: float = 0.15,
    relative_tolerance_ppm: float = 40.0,
) -> list[Species]:
    """Label each species with its mass delta to the main species and matching modifications."""
    if not species:
        return species
    if main is None:
        main = max(species, key=lambda s: s.base_intensity)
    main.annotations.append("main component")
    main.name = main.name or "main"
    for s in species:
        if s is main:
            continue
        delta = s.mass - main.mass
        tol = max(tolerance_da, main.mass * relative_tolerance_ppm * 1e-6)
        hits = annotate_delta(delta, s.compound_class, tolerance=tol)
        s.annotations.append(f"{delta:+.3f} Da vs main")
        if hits:
            s.annotations.append(hits[0].name)
            s.name = s.name or hits[0].name
        elif abs(delta - main.mass) < tol:
            s.annotations.append("dimer of main")
            s.name = s.name or "dimer"
        else:
            s.name = s.name or f"unknown {delta:+.1f} Da"
    return species


def impurity_table(species: list[Species], main: Species | None = None,
                   floor_pct: float = 0.01) -> list[dict]:
    """Rows of impurity mass, delta, annotation and percentage of the deconvolved signal.

    A species' amount ("area", and the percentage) is its base form, without adducts, and its
    total over every adduct form resolved is given beside it ("area_all_forms"). Which adduct
    forms are resolved depends on the region: each region's decoder fits the adducts it sees
    next to its strongest species, and fitting every adduct everywhere makes near-coincident
    forms of co-eluting species trade signal (PO + K lies 0.09 Da from FLP + Na). All-forms
    totals are therefore not comparable between species of different regions; base forms are
    measured for every species alike. Where the adduct load differs between species - under
    ion pairing with acetate it falls at the main species' own apex - base-form shares
    understate an impurity by the ratio of the two species' adduct loads (up to 1.7 on the
    IP-HILIC NC run, docs/benchmarks.md).
    """
    if not species:
        return []
    if main is None:
        main = max(species, key=lambda s: s.base_intensity)
    total = sum(s.base_intensity for s in species)
    rows = []
    for s in sorted(species, key=lambda x: -x.base_intensity):
        pct = 100.0 * s.base_intensity / total if total > 0 else 0.0
        if pct < floor_pct and s is not main:
            continue
        rows.append({
            "id": s.id,
            "name": s.name or ("main" if s is main else ""),
            "mass": s.mass,
            "delta_vs_main": s.mass - main.mass,
            "rt": s.rt_apex,
            "area": s.base_intensity,
            "area_all_forms": s.total_intensity,
            "percent": pct,
            "annotation": "; ".join(s.annotations),
            "adducts": s.adduct_fractions(),
            "charges": f"{min(s.charges)}-{max(s.charges)}" if s.charges else "",
            "mass_spread_ppm": s.mass_spread_ppm,
            "flags": list(s.flags),
        })
    return rows
