"""Annotate deconvolved species relative to the main species of their region."""

from __future__ import annotations

import numpy as np

from ..chem.adducts import AdductLibrary, AdductState
from ..chem.modifications import annotate_delta
from ..core.model import Species

#: Tolerance (Da, ppm of the mass, whichever is larger) of a mass difference between two
#: species of one run, by mass type. The difference cancels most of the run's calibration error,
#: so it is tighter than a mass's own: 0.15 Da on the average masses of 5-10 kDa species. The
#: floor keeps the deaminated form of a co-eluting 6.8 kDa oligonucleotide, whose envelope
#: coincides with its parent's but for a dalton and reads 0.12 Da light (docs/benchmarks.md).
DELTA_TOLERANCE = {"mono": (0.05, 12.0), "average": (0.15, 15.0)}

#: A species whose charge states disagree is read less exactly: where its and the main species'
#: mass spreads (combined) exceed the tolerance, a modification is named within them, up to this
#: many ppm (the tolerance before the like-with-like comparison): a PS-to-PO impurity whose
#: envelope overlaps its parent's at time-of-flight resolving power can read a few tenths of a
#: dalton off, and its charge states disagree as much.
SPREAD_CAP_PPM = 40.0

#: the flag a species carries when adduct forms of it are listed apart (mark_adduct_forms)
ADDUCT_FORMS_FLAG = "adduct forms listed apart, counted in its all-forms amount: "


def delta_tolerance(mass: float, mass_type: str = "average", tolerance_da: float | None = None,
                    tolerance_ppm: float | None = None) -> float:
    """The tolerance of a mass difference to a species of ``mass`` (:data:`DELTA_TOLERANCE`
    unless given)."""
    da, ppm = DELTA_TOLERANCE["mono" if mass_type == "mono" else "average"]
    da = da if tolerance_da is None else tolerance_da
    ppm = ppm if tolerance_ppm is None else tolerance_ppm
    return max(da, abs(mass) * ppm * 1e-6)


def _like(a: Species, b: Species) -> str:
    """The mass type to compare two species' masses as."""
    return a.mass_type if a.mass_type == b.mass_type else "average"


def adduct_forms(species: list[Species]) -> dict[int, list[Species]]:
    """The species marked as adduct forms (:func:`mark_adduct_forms`), by ``id()`` of their
    parent; a form whose parent is not in ``species`` (or is itself a form) is not one here."""
    by_id = {s.id: s for s in species}
    out: dict[int, list[Species]] = {}
    for s in species:
        if s.adduct_form_of is None:
            continue
        p = by_id.get(s.adduct_form_of)
        if p is None or p is s or p.adduct_form_of is not None:
            continue
        out.setdefault(id(p), []).append(s)
    return out


def listed_species(species: list[Species]) -> list[Species]:
    """The species that are not another listed species' adduct form: the impurity table's."""
    forms = {id(f) for fl in adduct_forms(species).values() for f in fl}
    return [s for s in species if id(s) not in forms]


def _half_height_span(s: Species) -> tuple[float, float] | None:
    y = np.asarray(s.intensity, dtype=float)
    if not y.size or not float(y.max()) > 0:
        return None
    idx = np.flatnonzero(y >= 0.5 * float(y.max()))
    return float(s.time[idx[0]]), float(s.time[idx[-1]])


def _co_elute(p: Species, s: Species, max_apex_min: float) -> bool:
    """Apexes within ``max_apex_min``, or one region and each apex inside the other's elution
    band at half height. Overlapping bands alone are not enough: a 4522 Da species of the IP-RPLC
    NC run peaking 0.26 min before a 4264 Da one, at the start of their region, lies two DIPEA
    above it, and its band overlaps the other's at half height."""
    if abs(p.rt_apex - s.rt_apex) <= max_apex_min:
        return True
    if p.region_id < 0 or p.region_id != s.region_id:
        return False
    a, b = _half_height_span(p), _half_height_span(s)
    return (a is not None and b is not None and a[0] <= s.rt_apex <= a[1]
            and b[0] <= p.rt_apex <= b[1])


def mark_adduct_forms(species: list[Species], library: AdductLibrary, max_n: int = 4,
                      max_apex_min: float = 0.05, tolerance_da: float | None = None,
                      tolerance_ppm: float | None = None) -> list[Species]:
    """Mark the species that are adduct forms of a stronger species they co-elute with.

    A region's decoder fits a species' adduct forms as adduct columns of it only for the adducts
    it sees next to the region's strongest species, and up to the method's ``max_per_type`` of
    a kind; :func:`lcmsdeconv.features.link.fold_adduct_forms` folds a form reported apart only
    for cation adducts and adducts some region evidenced, where the two traces correlate. The
    rest stay species of their own and read as impurities: under tripropylamine-HFIP ion pairing
    a species can carry a ladder of tripropylamine N-oxides (159.27 Da each), and one of them was
    named "3x +cyanoethyl".

    A species is an adduct form of a stronger species (base form) of its polarity when its mass is
    the other's plus ``n`` (1 to ``max_n``) times one adduct of ``library`` that applies to the
    polarity (average-mass shifts for average masses, monoisotopic for monoisotopic ones;
    within :func:`delta_tolerance` of the parent's mass), the two co-elute (apexes within
    ``max_apex_min``, or one region and each apex inside the other's band at half height), and no
    modification of the species' class explains the difference within that tolerance as well (a
    report too many is the lesser error). Where several species qualify, the strongest is the
    parent; a parent is never itself an adduct form.

    The form keeps its trace and stays in the list, with ``adduct_form_of`` (the parent's id) and
    ``adduct_form`` (the adduct state, "+2TPAO") set; the parent's flags name its forms.
    :func:`impurity_table` and :func:`lcmsdeconv.quant.purity.ms_purity` leave forms out of the
    species they compare and count them in their parents' all-forms amounts, and
    :func:`annotate_species` names them. Returns ``species``.
    """
    for s in species:
        s.adduct_form_of, s.adduct_form = None, ""
        s.flags = [f for f in s.flags if not f.startswith(ADDUCT_FORMS_FLAG)]
    adducts = list(library.adducts) if library is not None else []
    if not adducts or len(species) < 2:
        return species
    base = {id(s): s.base_intensity for s in species}
    reach = max_n * max(max(a.mass, a.avg_mass) for a in adducts) + 1.0
    parents: list[Species] = []
    forms: dict[int, list[Species]] = {}
    for s in sorted(species, key=lambda x: -base[id(x)]):
        found = None
        for p in parents:
            if p.polarity != s.polarity or not base[id(p)] > base[id(s)]:
                continue
            delta = s.mass - p.mass
            if not 0 < delta <= reach:
                continue
            mt = _like(p, s)
            tol = delta_tolerance(p.mass, mt, tolerance_da, tolerance_ppm)
            best = None
            for a in adducts:
                if a.polarity and a.polarity != s.polarity:
                    continue
                shift = a.avg_mass if mt == "average" else a.mass
                n = int(round(delta / shift)) if shift > 0 else 0
                if 1 <= n <= max_n and abs(delta - n * shift) <= tol and (
                        best is None or abs(delta - n * shift) < best[0]):
                    best = (abs(delta - n * shift), a.name, n)
            if best is None or not _co_elute(p, s, max_apex_min):
                continue
            cls = s.compound_class or p.compound_class
            if any(cls in m.classes for m in annotate_delta(delta, cls, tolerance=tol,
                                                            mass_type=mt, multiple_tolerance=1.0)):
                continue
            found = (p, AdductState(((best[1], best[2]),)).label)
            break
        if found is None:
            parents.append(s)
            continue
        p, label = found
        s.adduct_form_of, s.adduct_form = p.id, label
        forms.setdefault(id(p), []).append(s)
    for p in parents:
        if id(p) in forms:
            p.flags.append(ADDUCT_FORMS_FLAG + ", ".join(
                f"{f.adduct_form} ({f.mass:.2f} Da)"
                for f in sorted(forms[id(p)], key=lambda x: x.mass)))
    return species


def annotate_species(
    species: list[Species],
    main: Species | None = None,
    tolerance_da: float | None = None,
    relative_tolerance_ppm: float | None = None,
) -> list[Species]:
    """Label each species with its mass delta to the main species and matching modifications.

    Differences of average masses are compared with the modifications' average-mass deltas,
    of monoisotopic masses with their monoisotopic deltas, within :func:`delta_tolerance` of the
    main species' mass, or the two species' combined mass spread where that is larger, up to
    :data:`SPREAD_CAP_PPM` (``tolerance_da`` and ``relative_tolerance_ppm`` set the tolerance
    instead); where several modifications match, all are named, the closest first (a
    phosphorothioate A and a phosphodiester G residue differ by 0.07 Da). A species marked as
    another's adduct form (:func:`mark_adduct_forms`) is named as that.
    """
    if not species:
        return species
    forms = adduct_forms(species)
    listed = listed_species(species)
    if main is None:
        main = max(listed or species, key=lambda s: s.base_intensity)
    main.annotations.append("main component")
    main.name = main.name or "main"
    for s in listed:
        if s is main:
            continue
        delta = s.mass - main.mass
        mt = _like(main, s)
        tol = delta_tolerance(main.mass, mt, tolerance_da, relative_tolerance_ppm)
        if tolerance_da is None and relative_tolerance_ppm is None:
            spread = float(np.hypot(s.mass_spread_ppm or 0.0, main.mass_spread_ppm or 0.0))
            tol = max(tol, min(spread, SPREAD_CAP_PPM) * 1e-6 * main.mass)
        hits = annotate_delta(delta, s.compound_class, tolerance=tol, mass_type=mt)
        s.annotations.append(f"{delta:+.3f} Da vs main")
        if hits:
            s.annotations.append(" or ".join(h.name for h in hits[:3]))
            s.name = s.name or hits[0].name
        elif abs(delta - main.mass) < max(tol, 40e-6 * main.mass):
            # a dimer is read on other ions than its monomer, at twice the mass: 40 ppm, as before
            s.annotations.append("dimer of main")
            s.name = s.name or "dimer"
        else:
            s.name = s.name or f"unknown {delta:+.1f} Da"
    for p in listed:
        for f in forms.get(id(p), ()):
            f.annotations.append(f"{f.mass - main.mass:+.3f} Da vs main")
            f.annotations.append(f"adduct form of {p.name or 'species'} ({p.mass:.2f} Da): "
                                 f"{f.adduct_form}")
            f.name = f.name or f"{p.name or f'{p.mass:.2f} Da'} {f.adduct_form}"
    return species


def _label_names(label: str) -> list[str]:
    """The adducts a state label names, in the label's order ("+2Na+K": Na, K)."""
    out = []
    for tok in (label or "").split("+"):
        if tok and tok != "base":
            i = 0
            while i < len(tok) and tok[i].isdigit():
                i += 1
            out.append(tok[i:])
    return out


def _all_forms(s: Species, forms: list[Species]) -> tuple[float, dict[str, float]]:
    """``s``'s amount over its adduct forms and those listed apart, and each form's share: a
    form's own adduct columns count as the combined state (+TPAO with a +Na column: +Na+TPAO)."""
    if not forms:
        return s.total_intensity, s.adduct_fractions()
    own = s.adduct_fractions() or {"base": 1.0}
    # labels name their adducts in the library's order: keep it, the parent's own labels as they are
    order: list[str] = []
    for lab in sorted(own, key=lambda x: -len(_label_names(x))):
        order += [n for n in _label_names(lab) if n not in order]
    for f in forms:
        for lab in [f.adduct_form, *f.adduct_fractions()]:
            order += [n for n in _label_names(lab) if n not in order]
    as_written = {frozenset(AdductState.from_label(lab).as_dict().items()): lab for lab in own}
    amounts: dict[str, float] = {}

    def add(state: AdductState, v: float) -> None:
        key = as_written.get(frozenset(state.as_dict().items())) or state.label or "base"
        amounts[key] = amounts.get(key, 0.0) + v

    for lab, v in own.items():
        add(AdductState.from_label(lab, order), v * s.total_intensity)
    for f in forms:
        st = AdductState.from_label(f.adduct_form, order)
        for lab, v in (f.adduct_fractions() or {"base": 1.0}).items():
            add(st.plus(AdductState.from_label(lab, order), order), v * f.total_intensity)
    total = s.total_intensity + sum(f.total_intensity for f in forms)
    tot = sum(amounts.values())
    return total, ({k: v / tot for k, v in amounts.items()} if tot > 0 else {})


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

    A species marked as a co-eluting species' adduct form (:func:`mark_adduct_forms`) is no
    impurity: it has no row and no share of the base-form signal, and its amount (all its forms)
    counts in its parent's "area_all_forms" and adduct shares ("adducts").
    """
    if not species:
        return []
    forms = adduct_forms(species)
    listed = listed_species(species)
    if main is None:
        main = max(listed, key=lambda s: s.base_intensity)
    total = sum(s.base_intensity for s in listed)
    rows = []
    for s in sorted(listed, key=lambda x: -x.base_intensity):
        pct = 100.0 * s.base_intensity / total if total > 0 else 0.0
        if pct < floor_pct and s is not main:
            continue
        all_forms, fractions = _all_forms(s, forms.get(id(s), []))
        rows.append({
            "id": s.id,
            "name": s.name or ("main" if s is main else ""),
            "mass": s.mass,
            "delta_vs_main": s.mass - main.mass,
            "rt": s.rt_apex,
            "area": s.base_intensity,
            "area_all_forms": all_forms,
            "percent": pct,
            "annotation": "; ".join(s.annotations),
            "adducts": fractions,
            "charges": f"{min(s.charges)}-{max(s.charges)}" if s.charges else "",
            "mass_spread_ppm": s.mass_spread_ppm,
            "flags": list(s.flags),
        })
    return rows
