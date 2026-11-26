"""Annotation of deconvolved species: like-with-like modification deltas, a tight delta
tolerance, multiples only of repeating modifications, and co-eluting adduct ladders counted with
their parent (synthetic species lists)."""

import numpy as np
import pytest

from lcmsdeconv.chem.adducts import ADDUCTS, AdductLibrary
from lcmsdeconv.chem.formula import Formula
from lcmsdeconv.chem.modifications import MODIFICATIONS, REPEATING, annotate_delta
from lcmsdeconv.core.model import Species
from lcmsdeconv.features.annotate import (
    ADDUCT_FORMS_FLAG,
    annotate_species,
    delta_tolerance,
    impurity_table,
    mark_adduct_forms,
)
from lcmsdeconv.features.link import fold_adduct_forms
from lcmsdeconv.quant.purity import ms_purity

T = np.linspace(9.0, 11.0, 81)
TPAO = Formula("C9H21NO").avg_mass  # 159.27 Da on average masses


def _sp(i, mass, height, apex=10.0, width=0.08, cls="ps_rna", region=3, forms=None,
        mass_type="average"):
    y = height * np.exp(-0.5 * ((T - apex) / width) ** 2)
    adducts = {k: v * y for k, v in (forms or {}).items()}
    return Species(i, mass, mass_type, -1, cls, T.copy(), y, adducts, {6: height},
                   region_id=region)


def _ladder(main_mass=7500.0, errors=(0.05, -0.08, 0.10, -0.02)):
    """A main species with sodium columns, a ladder of four tripropylamine N-oxides reported
    apart, a co-eluting cyanoethyl impurity and an n-1 shortmer."""
    main = _sp(0, main_mass, 100.0, forms={"base": 0.85, "+Na": 0.15})
    ladder = [_sp(n, main_mass + n * TPAO + e, h)
              for n, h, e in zip((1, 2, 3, 4), (12.0, 4.0, 1.0, 0.3), errors)]
    cyano = _sp(5, main_mass + Formula("C3H3N").avg_mass + 0.02, 3.0)
    short = _sp(6, main_mass - 329.2, 2.0, apex=9.7, region=2)
    return main, ladder, cyano, short


def test_tripropylamine_and_its_n_oxide_are_library_adducts():
    tpa, tpao = ADDUCTS["TPA"], ADDUCTS["TPAO"]
    assert tpa.formula == Formula("C9H21N") and tpao.formula == Formula("C9H21NO")
    assert tpao.mass == pytest.approx(159.1623, abs=1e-4)
    assert tpao.avg_mass == pytest.approx(159.27, abs=0.01)
    assert ADDUCTS["tripropylamine"] is tpa
    assert ADDUCTS["TPA N-oxide"] is tpao and ADDUCTS["tripropylamine N-oxide"] is tpao
    assert (tpa.category, tpa.polarity) == (tpao.category, tpao.polarity) == ("amine", -1)
    assert {"TPA", "TPAO"} <= set(AdductLibrary.from_mode("iprp", -1, include=("TPA", "TPAO"))
                                  .names())
    # negative mode only, like the other ion-pairing amines; not a default of any mode
    assert "TPAO" not in AdductLibrary.from_mode("iprp", 1, include=("TPAO",)).names()
    for mode in ("iprp", "ip-rplc", "ip-hilic"):
        assert not {"TPA", "TPAO"} & set(AdductLibrary.from_mode(mode, -1).names())


def test_average_mass_differences_are_compared_with_average_deltas():
    po = next(m for m in MODIFICATIONS if m.name.startswith("PS -> PO"))
    assert po.delta == pytest.approx(-15.977, abs=1e-3)
    assert po.delta_for("average") == pytest.approx(-16.066, abs=1e-3)

    def names(d, mt):
        return [m.name for m in annotate_delta(d, "ps_rna", tolerance=0.05, mass_type=mt)]

    assert "PS -> PO (desulfurization)" in names(-16.06, "average")
    assert "PS -> PO (desulfurization)" not in names(-16.06, "mono")
    assert "PS -> PO (desulfurization)" in names(-15.98, "mono")
    assert "PS -> PO (desulfurization)" not in names(-15.98, "average")
    # residues have average deltas too (they were numbers): n-1 (A) 329.05 mono, 329.21 average
    assert names(-329.21, "average") == ["n-1 (A)"] and names(-329.21, "mono") == []


def test_a_tpao_adduct_is_not_three_cyanoethyl_groups():
    # a TPAO adduct read a little off (159.22-159.29 Da): three cyanoethyl groups are 159.08
    # (mono) and 159.19 Da (average), but a cyanoethyl group does not repeat, whatever the
    # tolerance
    for d in (TPAO - 0.05, TPAO + 0.02):
        for mt in ("mono", "average"):
            for tol in (0.15, 0.32):
                assert not any("cyanoethyl" in m.name
                               for m in annotate_delta(d, "ps_rna", tolerance=tol, mass_type=mt))
    # one cyanoethyl group is still named: 53.03 Da monoisotopic, 53.06 Da on average masses
    assert annotate_delta(53.03, "ps_rna", tolerance=0.05, mass_type="mono")[0].name \
        == "+cyanoethyl"
    assert annotate_delta(53.07, "ps_rna", tolerance=0.15, mass_type="average")[0].name \
        == "+cyanoethyl"


def test_only_repeating_modifications_are_read_as_multiples_within_half_the_tolerance():
    assert REPEATING <= {m.name for m in MODIFICATIONS}
    assert "+cyanoethyl" not in REPEATING and not any(n.startswith("n-1") for n in REPEATING)
    two_po = 2 * Formula("O").avg_mass - 2 * Formula("S").avg_mass
    hits = annotate_delta(two_po + 0.05, "ps_rna", tolerance=0.15, mass_type="average")
    assert hits[0].name == "2x PS -> PO (desulfurization)"
    assert hits[0].delta_for("average") == pytest.approx(two_po)
    # 0.1 Da off: within the tolerance, not within half of it; a check whether a modification
    # could explain a difference (the adduct fold's) takes the whole tolerance
    assert not annotate_delta(two_po + 0.1, "ps_rna", tolerance=0.15, mass_type="average")
    assert annotate_delta(two_po + 0.1, "ps_rna", tolerance=0.15, mass_type="average",
                          multiple_tolerance=1.0)[0].name == "2x PS -> PO (desulfurization)"
    # two isobutyryl groups (140.18 Da) are not read
    ibu = Formula("C4H6O").avg_mass
    assert not annotate_delta(2 * ibu, "ps_rna", tolerance=0.15, mass_type="average")


def test_the_documented_oligonucleotide_annotations_are_kept():
    """The IP-HILIC conjugate run's species as measured (average masses, docs/benchmarks.md):
    n+1 and n-1 differ from FLP by a C10H12N5O5PS residue (345.27 Da: A or dG with a
    phosphorothioate), which a G with a phosphodiester (345.21 Da) matches within the
    tolerance as well; both are named, the closer first."""
    t = {"FLP": 6794.00, "n+1": 7139.28, "PO": 6778.00, "n-1": 6448.74, "DA": 6794.87}
    sp = {k: _sp(i, m, 100.0 if k == "FLP" else 3.0) for i, (k, m) in enumerate(t.items())}
    annotate_species(list(sp.values()))
    ann = {k: s.annotations[-1] for k, s in sp.items()}
    assert ann["FLP"] == "main component"
    assert ann["n+1"] == "n+1 (A, PS) or n+1 (G)" and sp["n+1"].name == "n+1 (A, PS)"
    assert ann["n-1"] == "n-1 (A, PS) or n-1 (G)"
    assert ann["PO"] == "PS -> PO (desulfurization)"
    # the deaminated form reads 0.12 Da light (its envelope coincides with FLP's but for a
    # dalton): the 0.15 Da floor of the tolerance keeps it
    assert ann["DA"] == "deamination (C -> U, A -> I)"
    assert delta_tolerance(6794.0) == pytest.approx(0.15)
    assert delta_tolerance(30000.0) == pytest.approx(0.45)
    assert delta_tolerance(6794.0, "mono") == pytest.approx(0.0815, abs=1e-4)


def test_a_coeluting_adduct_ladder_is_counted_with_its_parent():
    main, ladder, cyano, short = _ladder()
    species = [main, *ladder, cyano, short]
    lib = AdductLibrary.from_mode("iprp", 0, include=("TPAO",))
    mark_adduct_forms(species, lib)
    assert [s.adduct_form_of for s in ladder] == [0, 0, 0, 0]
    assert [s.adduct_form for s in ladder] == ["+TPAO", "+2TPAO", "+3TPAO", "+4TPAO"]
    assert cyano.adduct_form_of is None and short.adduct_form_of is None
    assert any(f.startswith(ADDUCT_FORMS_FLAG) and "+4TPAO" in f for f in main.flags)

    annotate_species(species)
    assert ladder[1].annotations[-1] == "adduct form of main (7500.00 Da): +2TPAO"
    assert ladder[1].name == "main +2TPAO"
    assert cyano.annotations[-1] == "+cyanoethyl"

    rows = impurity_table(species)
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {0, 5, 6}  # the ladder is no impurity
    listed_base = main.base_intensity + cyano.base_intensity + short.base_intensity
    assert by_id[0]["area"] == pytest.approx(main.base_intensity)  # base form, as before
    assert by_id[0]["area_all_forms"] == pytest.approx(
        main.total_intensity + sum(s.total_intensity for s in ladder))
    assert by_id[5]["percent"] == pytest.approx(100 * cyano.base_intensity / listed_base)
    fr = by_id[0]["adducts"]
    assert sum(fr.values()) == pytest.approx(1.0)
    assert set(fr) == {"base", "+Na", "+TPAO", "+2TPAO", "+3TPAO", "+4TPAO"}
    assert fr["+TPAO"] / fr["base"] == pytest.approx(12.0 / 85.0)

    pur = ms_purity(species)
    assert pur["n_species"] == 3
    assert pur["main_percent"] == pytest.approx(100 * main.base_intensity / listed_base)
    assert pur["total_intensity"] == pytest.approx(sum(s.total_intensity for s in species))

    # marking again gives the same marks and one flag
    mark_adduct_forms(species, lib)
    assert sum(f.startswith(ADDUCT_FORMS_FLAG) for f in main.flags) == 1
    assert [s.adduct_form for s in ladder] == ["+TPAO", "+2TPAO", "+3TPAO", "+4TPAO"]


def test_process_run_counts_a_coeluting_adduct_form_with_its_parent(monkeypatch):
    """process_run marks a co-eluting species two tripropylamine N-oxides above a stronger one,
    reported apart by its region, as that species' adduct form (mark_adduct_forms, with the RDa
    method's library, after the cross-region steps): the form names its parent, has no row in
    the impurity table and no place among the MS purity's species, and counts in its parent's
    all-forms area and adduct fractions."""
    from lcmsdeconv.core.method import Method
    from lcmsdeconv.core.model import Run, Spectrum

    from .test_region_pieces import _process

    method = Method.load("rda_iprp_neg_oligo")
    method.acquisition.lock_mass_mz = None
    main = _sp(0, 7000.0, 100.0)
    form = _sp(1, 7000.0 + 2 * TPAO + 0.05, 25.0, apex=10.01)  # within 0.15 Da
    short = _sp(2, 6670.8, 5.0, apex=9.7, region=2)
    totals = {s.id: (s.base_intensity, s.total_intensity) for s in (main, form, short)}
    tic = 1e4 * np.exp(-0.5 * ((T - 10.0) / 0.08) ** 2) + 10.0
    run = Run([Spectrum(np.array([1000.0]), np.array([y]), rt=float(t), polarity=-1)
               for t, y in zip(T, tic)])
    result, _calls = _process(monkeypatch, run, method, [main, form, short])
    by_id = {s.id: s for s in result.species}
    assert sorted(by_id) == [0, 1, 2]  # the form stays in the species list...
    assert (by_id[1].adduct_form_of, by_id[1].adduct_form) == (0, "+2TPAO")
    assert by_id[1].annotations[-1] == "adduct form of main (7000.00 Da): +2TPAO"
    assert any(f.startswith(ADDUCT_FORMS_FLAG) and "+2TPAO" in f for f in by_id[0].flags)
    # ...but is no impurity, nor a species of the MS purity
    rows = {r["id"]: r for r in result.impurities}
    assert set(rows) == {0, 2}
    ms = result.purity["ms"]
    assert ms["n_species"] == 2
    assert ms["main_percent"] == pytest.approx(100.0 * totals[0][0] / (totals[0][0] + totals[2][0]))
    # its amount counts in its parent's all forms and adduct fractions
    all_forms = totals[0][1] + totals[1][1]
    assert rows[0]["area"] == pytest.approx(totals[0][0])
    assert rows[0]["area_all_forms"] == pytest.approx(all_forms)
    assert rows[0]["adducts"]["+2TPAO"] == pytest.approx(totals[1][1] / all_forms)
    assert rows[2]["percent"] == pytest.approx(100.0 * totals[2][0] / (totals[0][0] + totals[2][0]))


def test_a_form_with_adduct_columns_of_its_own_counts_as_the_combined_state():
    main = _sp(0, 7500.0, 100.0, forms={"base": 0.8, "+Na": 0.1, "+Na+TPAO": 0.1})
    form = _sp(1, 7500.0 + TPAO, 20.0, forms={"base": 0.5, "+Na": 0.5})
    mark_adduct_forms([main, form], AdductLibrary.from_mode("iprp", 0, include=("TPAO",)))
    fr = impurity_table([main, form])[0]["adducts"]
    # the form's +Na column is the parent's +Na+TPAO state, as the parent's own label writes it
    assert set(fr) == {"base", "+Na", "+Na+TPAO", "+TPAO"}
    assert fr["+Na+TPAO"] == pytest.approx((10.0 + 10.0) / 120.0)


def test_without_tpao_in_the_library_the_ladder_stays_an_unknown():
    main, ladder, cyano, short = _ladder()
    species = [main, *ladder, cyano, short]
    mark_adduct_forms(species, AdductLibrary.from_mode("iprp", 0))
    assert all(s.adduct_form_of is None for s in species)
    annotate_species(species)
    assert ladder[0].name == "unknown +159.3 Da"
    assert not any("cyanoethyl" in a for a in ladder[0].annotations)
    assert len(impurity_table(species)) == 7


def test_an_adduct_form_co_elutes_with_a_stronger_parent():
    lib = AdductLibrary.from_mode("iprp", 0, include=("TPAO",))

    def marked(mass=7500.0 + TPAO, height=10.0, polarity=-1, **kw):
        main = _sp(0, 7500.0, 100.0, width=0.1)
        form = _sp(1, mass, height, **{"width": 0.1, **kw})
        form.polarity = polarity
        mark_adduct_forms([main, form], lib)
        return form.adduct_form_of is not None

    assert marked()
    # apexes 0.075 min apart: one region, each apex inside the other's band at half height
    assert marked(apex=10.075)
    assert not marked(apex=10.075, region=4)
    # 0.2 min apart: the bands overlap at half height, but neither apex lies in the other's
    assert not marked(apex=10.2)
    # past the tolerance (0.15 Da); the stronger of the two; the other polarity
    assert marked(mass=7500.0 + TPAO + 0.14)
    assert not marked(mass=7500.0 + TPAO + 0.2)
    assert not marked(height=150.0)
    assert not marked(polarity=1)


def test_a_modification_of_the_class_explaining_the_difference_keeps_the_species():
    # a PEG oligomer one unit up (44.05 Da) lies 0.09 Da from two sodium exchanges
    lib = AdductLibrary.from_mode("polymer", 0)
    main = _sp(0, 5000.0, 100.0, cls="peg")
    up = _sp(1, 5000.0 + Formula("C2H4O").avg_mass, 30.0, cls="peg")
    mark_adduct_forms([main, up], lib)
    assert up.adduct_form_of is None
    # an oligonucleotide 2Na above its parent is an adduct form
    main, na2 = _sp(0, 5000.0, 100.0), _sp(1, 5000.0 + 2 * ADDUCTS["Na"].avg_mass, 30.0)
    mark_adduct_forms([main, na2], lib)
    assert na2.adduct_form == "+2Na"


def test_monoisotopic_masses_use_monoisotopic_shifts():
    lib = AdductLibrary.from_mode("iprp", 0, include=("TPAO",))
    tpao = ADDUCTS["TPAO"]
    main = _sp(0, 7500.0, 100.0, mass_type="mono")
    mono = _sp(1, 7500.0 + tpao.mass + 0.03, 10.0, mass_type="mono")
    avg = _sp(2, 7500.0 + 2 * tpao.avg_mass, 5.0, mass_type="mono")
    mark_adduct_forms([main, mono, avg], lib)
    assert mono.adduct_form == "+TPAO"
    # two average shifts are 0.21 Da from two monoisotopic ones: beyond max(0.05 Da, 12 ppm)
    assert avg.adduct_form_of is None


def test_a_coeluting_tpao_form_is_folded_not_kept_as_three_cyanoethyl_groups():
    """The adduct fold asks whether a modification of the class explains the difference as well;
    three cyanoethyl groups (159.08 Da monoisotopic, within its 0.6 Da) kept a TPAO form apart."""
    main = _sp(0, 7500.0, 100.0, forms={"base": 1.0})
    form = _sp(1, 7500.0 + TPAO + 0.02, 15.0, forms={"base": 1.0})
    out = fold_adduct_forms([main, form],
                            AdductLibrary.from_mode("iprp", -1, include=("C9H21NO",)))
    assert len(out) == 1
    assert set(out[0].adduct_intensity) == {"base", "+C9H21NO"}
    # an n+1 (a 2'-OMe G residue, 359.23 Da on average masses) 0.45 Da from +TEA+2DIPEA stays:
    # a modification of its class explains the difference
    n1 = _sp(2, 7500.0 + Formula("C11H14N5O7P").avg_mass, 5.0, forms={"base": 1.0})
    lib = AdductLibrary.from_mode("iprp", -1)
    assert any(abs(st.avg_mass - Formula("C11H14N5O7P").avg_mass) < 0.6 for st in lib.states())
    assert len(fold_adduct_forms([_sp(0, 7500.0, 100.0, forms={"base": 1.0}), n1], lib)) == 2


def test_a_species_whose_charge_states_disagree_is_named_within_its_spread():
    """A PS-to-PO impurity whose envelope overlaps its parent's reads a little off; its charge
    states disagree as much. The tolerance grows to the two species' combined spread, up to the
    40 ppm of before."""
    po = Formula("O").avg_mass - Formula("S").avg_mass

    def name(offset, spread):
        main, imp = _sp(0, 11000.0, 100.0), _sp(1, 11000.0 + po + offset, 2.0)
        main.mass_spread_ppm, imp.mass_spread_ppm = 6.0, spread
        annotate_species([main, imp])
        return imp.name

    assert name(0.20, 2.0) == "unknown -15.9 Da"  # well read: 15 ppm, 0.165 Da
    assert name(0.20, 32.0) == "PS -> PO (desulfurization)"  # 32.6 ppm: 0.36 Da
    assert name(0.40, 200.0) == "PS -> PO (desulfurization)"  # capped at 40 ppm: 0.44 Da
    assert name(0.50, 200.0) == "unknown -15.6 Da"
    # tolerances given set it exactly
    main, imp = _sp(0, 11000.0, 100.0), _sp(1, 11000.0 + po + 0.20, 2.0)
    imp.mass_spread_ppm = 32.0
    annotate_species([main, imp], tolerance_da=0.15, relative_tolerance_ppm=15.0)
    assert imp.name == "unknown -15.9 Da"
