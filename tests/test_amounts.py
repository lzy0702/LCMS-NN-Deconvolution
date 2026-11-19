"""Amounts that mean the same thing for every species: template positions, adduct forms, the
base form, and species that cross region boundaries."""

import numpy as np

from lcmsdeconv.chem.adducts import ADDUCTS, AdductLibrary, AdductState
from lcmsdeconv.chem.classes import class_isotope_pattern
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core.model import Component, FrameResult, Species
from lcmsdeconv.deconv.decode import Candidate
from lcmsdeconv.deconv.refine import _merge_same_mass, _set_mass, align_template
from lcmsdeconv.deconv.templates import build_template
from lcmsdeconv.features.link import fold_adduct_forms, link_species, merge_adjacent_species
from lcmsdeconv.nn.grid import LogMzGrid

MP = 1.00727646688
ORBI = InstrumentModel("orbitrap", 140000.0, mz_ref=200.0)


def test_a_new_mass_leaves_the_templates_where_the_fit_put_them():
    c = Component(mass=6794.10, intensity=1.0, template_shift=2e-5)
    where = c.mass * np.exp(c.template_shift)
    _set_mass(c, 6794.05)
    assert c.mass == 6794.05
    assert np.isclose(c.mass * np.exp(c.template_shift), where, rtol=1e-12)
    # two copies of one species: the stronger one's templates, everyone's columns
    a = Component(mass=6794.0, intensity=3.0, charges={4: 3.0}, adducts={"base": 3.0},
                  columns={(4, ""): 3.0}, template_shift=1e-5)
    b = Component(mass=6794.2, intensity=1.0, charges={4: 1.0}, adducts={"base": 1.0},
                  columns={(4, ""): 1.0})
    where = a.mass * np.exp(a.template_shift)
    comps = [b, a]
    _merge_same_mass(comps)
    assert len(comps) == 1 and np.isclose(comps[0].mass, 6794.05)
    assert np.isclose(comps[0].mass * np.exp(comps[0].template_shift), where, rtol=1e-12)
    assert comps[0].columns[(4, "")] == 4.0


def test_alignment_finds_the_offset_between_its_steps():
    """At Orbitrap resolving power a peak is narrower than a grid bin: a quarter-bin step leaves
    a template up to an eighth of a bin off, worth about 2 % of the fitted amount."""
    mass = 6794.0
    grid = LogMzGrid(1000.0, 2500.0, 2e-5, -1)
    for true_bins in (0.375, -0.6):
        spectrum = np.zeros(grid.size)
        for z in (3, 4, 5):
            t = build_template(mass * np.exp(true_bins * grid.step), z, grid, ORBI, "ps_rna")
            spectrum[t.bins] += 1e6 * t.values
        s = align_template(Candidate(mass, 1.0, {3, 4, 5}), spectrum, grid, ORBI, "ps_rna")
        assert abs(s / grid.step - true_bins) < 0.06


def _profile(mass, charges, height, fwhm_at, lo=1000.0, hi=2500.0, step=0.002, shift=0.0,
             potassium=0):
    """Raw Orbitrap profile of an oligonucleotide's envelopes (optionally with k potassium ions,
    whose 41K moves 6.7 % of each peak 2 Da up), every isotopologue ``shift`` Da heavier."""
    pat = class_isotope_pattern(mass, "ps_rna")
    masses, ab = pat.masses - pat.average_mass + mass + shift, pat.abundances
    for _ in range(potassium):
        masses = np.concatenate([masses + ADDUCTS["K"].mass, masses + ADDUCTS["K"].mass + 1.998])
        ab = np.concatenate([0.9326 * ab, 0.0673 * ab])
    x = np.arange(lo, hi, step)
    y = np.zeros_like(x)
    for z in charges:
        mz = (masses - z * MP) / z
        sig = fwhm_at(mz.mean()) / 2.3548
        for c, h in zip(mz, ab / ab.max()):
            a, b = np.searchsorted(x, [c - 5 * sig, c + 5 * sig])
            y[a:b] += height * h * np.exp(-0.5 * ((x[a:b] - c) / sig) ** 2)
    return x, y


def test_a_potassium_form_is_evidence_at_its_average_mass_shift():
    """Potassium is 6.7 % 41K: a K form's average mass is 0.13 Da above the base's plus the
    monoisotopic K - H shift, half the evidence tolerance; with a small measurement error on
    top, comparing against the monoisotopic shift rejected FLP's +K form on a real run."""
    from lcmsdeconv.deconv.pipeline import DeconvParams, evidenced_adducts

    mass = 6794.0
    fwhm = ORBI.fwhm
    x, y = _profile(mass, (3, 4, 5), 1e6, fwhm)
    xk, yk = _profile(mass, (3, 4, 5), 4e5, fwhm, potassium=1, shift=0.2)
    y = y + yk
    lib = AdductLibrary.from_mode("iprp", -1, max_per_type=2, max_total=3)
    assert abs(ADDUCTS["K"].avg_mass - ADDUCTS["K"].mass - 0.134) < 0.005
    params = DeconvParams(compound_class="ps_rna", adduct_mode="iprp", coelution_r=0.0)
    kept = evidenced_adducts([Candidate(mass, 1.0, {3, 4, 5})], lib, x, y, -1, 100.0, params,
                             "ps_rna", ORBI, True)
    assert "K" in kept and "Na" not in kept


def test_adduct_labels_combine():
    order = ["TEA", "Na", "K"]
    k = AdductState.from_label("+K", order)
    assert k.plus(AdductState.from_label("+2Na", order), order).label == "+2Na+K"
    assert AdductState.from_label("base", order) == AdductState()
    assert np.isclose(AdductState.from_label("+Na+2K").avg_mass,
                      ADDUCTS["Na"].avg_mass + 2 * ADDUCTS["K"].avg_mass)


def _species(sid, mass, t, y, adducts=None):
    y = np.asarray(y, dtype=float)
    return Species(sid, mass, "average", -1, "ps_rna", np.asarray(t, dtype=float), y,
                   adducts if adducts is not None else {"base": y.copy()}, {4: float(y.sum())})


def test_adduct_forms_reported_apart_count_with_their_species():
    t = np.linspace(16.0, 16.8, 21)
    prof = np.exp(-0.5 * ((t - 16.4) / 0.08) ** 2)
    flp = _species(0, 6794.00, t, 100 * prof)
    plus_k = _species(1, 6794.00 + ADDUCTS["K"].avg_mass + 0.1, t, 35 * prof)
    # a +Na form with its own +K column, i.e. FLP + Na + K
    na_base, na_k = 12 * prof, 4 * prof
    plus_na = _species(2, 6815.90, t, na_base + na_k, {"base": na_base, "+K": na_k})
    # the right mass for +Na, but it elutes on its own
    elsewhere = _species(3, 6815.98, t, 20 * np.exp(-0.5 * ((t - 16.1) / 0.05) ** 2))
    # co-elutes, but its mass is no adduct state's
    other = _species(4, 6794.00 + 50.0, t, 30 * prof)
    lib = AdductLibrary.from_mode("iprp", -1, max_per_type=2, max_total=3)
    base_before = flp.base_intensity
    out = fold_adduct_forms([plus_k, other, flp, plus_na, elsewhere], lib)
    assert sorted(s.id for s in out) == [0, 3, 4]
    parent = next(s for s in out if s.id == 0)
    assert np.isclose(parent.base_intensity, base_before)
    assert set(parent.adduct_intensity) == {"base", "+K", "+Na", "+Na+K"}
    assert np.isclose(parent.total_intensity, (100 + 35 + 16) / 100 * base_before, rtol=1e-9)
    assert np.allclose(parent.adduct_intensity["+Na+K"], na_k)
    assert any("adduct forms" in f for f in parent.flags)


def test_a_weaker_species_never_takes_a_stronger_one_as_its_adduct():
    t = np.linspace(16.0, 16.8, 21)
    prof = np.exp(-0.5 * ((t - 16.4) / 0.08) ** 2)
    weak = _species(0, 6772.02, t, 5 * prof)  # FLP - Na: FLP would be its "+Na" form
    flp = _species(1, 6794.00, t, 100 * prof)
    out = fold_adduct_forms([weak, flp], AdductLibrary.from_mode("iprp", -1))
    assert len(out) == 2


def test_impurity_shares_compare_base_forms():
    from lcmsdeconv.features.annotate import impurity_table
    from lcmsdeconv.quant.purity import ms_purity

    t = np.linspace(0, 1, 11)
    prof = np.exp(-0.5 * ((t - 0.5) / 0.1) ** 2)
    # the main species' adduct forms were not resolved, the impurity's were
    main = _species(0, 6794.0, t, 100 * prof)
    imp = _species(1, 6448.7, t, 5 * prof, {"base": 2.5 * prof, "+K": 2.5 * prof})
    rows = impurity_table([main, imp])
    pct = {r["id"]: r["percent"] for r in rows}
    assert np.isclose(pct[1] / pct[0], 2.5 / 100)
    assert np.isclose(rows[1]["area_all_forms"], imp.total_intensity)
    assert np.isclose(ms_purity([main, imp])["main_percent"], 100 * 100 / 102.5)


def test_the_edge_rule_spares_a_boundary_shared_with_the_next_region():
    """A species eluting at the valley where two regions meet peaks on their shared boundary."""
    rts = 1.0 + 0.04 * np.arange(20)
    main = np.exp(-0.5 * ((rts - 1.3) / 0.08) ** 2)
    late = np.exp(-0.5 * ((rts - rts[-1]) / 0.1) ** 2)
    frames = [FrameResult(rt, -1, [
        Component(5162.25, 100 * a, charges={4: 100 * a}, adducts={"base": 100 * a}),
        Component(4832.98, 10 * b, charges={4: 10 * b}, adducts={"base": 10 * b})], 1.0, 0.0)
        for rt, a, b in zip(rts, main, late)]
    kw = dict(min_frames=2, flat_test=False)
    assert [round(s.mass, 2) for s in link_species(frames, **kw)] == [5162.25]
    kept = link_species(frames, shared_edges=(False, True), **kw)
    assert sorted(round(s.mass, 2) for s in kept) == [4832.98, 5162.25]
    # the boundary on the other side is still judged
    assert [round(s.mass, 2) for s in link_species(frames, shared_edges=(True, False), **kw)] \
        == [5162.25]


def test_a_scan_two_regions_share_is_counted_once():
    t1, t2 = np.linspace(1.0, 1.4, 11), np.linspace(1.4, 1.8, 11)
    a = _species(0, 4833.0, t1, np.linspace(1, 10, 11))  # peaks on the shared scan, 10
    b = _species(1, 4833.1, t2, np.linspace(2, 0.1, 11))  # a weak copy there, 2
    out = merge_adjacent_species([a, b])
    assert len(out) == 1
    m = out[0]
    assert m.time.size == 21
    assert m.intensity[np.searchsorted(m.time, 1.4)] == 10.0
    assert np.isclose(m.adduct_intensity["base"].sum(), m.intensity.sum())


def test_a_candidate_is_recentred_onto_its_own_base_form_only():
    """Stepping a candidate down by an adduct must land on an envelope centred there, not on the
    flank of a stronger species' envelope that an aligned template would also fit."""
    from lcmsdeconv.deconv.nnls_solve import fit_weights
    from lcmsdeconv.deconv.refine import _fit_candidate, _recenter_base
    from lcmsdeconv.nn.grid import spectrum_to_grid

    nc = 5162.25
    k = ADDUCTS["K"].mass
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP"), max_per_type=2,
                                  max_total=2)
    res = float(ORBI.resolving_power(1000.0))

    def recentre(cand_mass, other_mass, other_height):
        x, y = _profile(nc, (3, 4, 5), 1e6, ORBI.fwhm)
        _x, y2 = _profile(other_mass, (3, 4, 5), other_height, ORBI.fwhm)
        y = y + y2
        keep = y > 1.0
        grid, obs, sigma = spectrum_to_grid(x[keep], y[keep], -1, instrument=ORBI)
        w = fit_weights(obs, sigma, grid.step, res)
        cand = Candidate(cand_mass, 1.0, {3, 4, 5})
        cand.template_shift = align_template(cand, obs, grid, ORBI, "ps_rna")
        full = _fit_candidate(cand, obs, w, grid, ORBI, "ps_rna", lib.states(),
                              mz_range=(1000.0, 2500.0))
        moved, _f = _recenter_base(cand, full, obs, w, grid, ORBI, "ps_rna", lib.states(), lib,
                                   (1000.0, 2500.0), raw_mz=x[keep], raw_int=y[keep],
                                   noise_sigma=10.0, resolution=res)
        return moved.mass

    # the main species' own potassium form, found as a candidate first: it moves onto the base
    assert abs(recentre(nc + k, nc + k, 3e5) - nc) < 0.3
    # a weak species one potassium above a point 3 Da below the main species stays where it is
    weak = nc - 3.0 + k
    assert abs(recentre(weak, weak, 2e4) - weak) < 1e-6


def test_a_mass_difference_a_modification_explains_is_not_folded():
    """Oxidation (+15.99 Da) is an ammonium adduct (+17.03 Da) within a dalton: at 31 kDa a
    measured difference of 16.9 Da is either, and the impurity must not disappear."""
    t = np.linspace(2.4, 3.0, 16)
    prof = np.exp(-0.5 * ((t - 2.7) / 0.06) ** 2)
    parent = Species(0, 31445.72, "average", 1, "peptide", t, 100 * prof, {"base": 100 * prof},
                     {20: 100.0})
    ox = Species(1, 31462.60, "average", 1, "peptide", t, 12 * prof, {"base": 12 * prof},
                 {20: 12.0})
    out = fold_adduct_forms([parent, ox], AdductLibrary.from_mode("rplc", 1))
    assert len(out) == 2


def test_longmers_and_deaminated_oligonucleotides_are_named():
    from lcmsdeconv.chem.modifications import annotate_delta

    # the IP-HILIC conjugate run's n+1 and deaminated forms, as measured (average masses)
    assert annotate_delta(7139.28 - 6794.00, "ps_rna", tolerance=0.27)[0].name == "n+1 (G)"
    assert annotate_delta(6448.74 - 6794.00, "ps_rna", tolerance=0.27)[0].name == "n-1 (G)"
    assert annotate_delta(6794.86 - 6794.00, "ps_rna", tolerance=0.27)[0].name.startswith(
        "deamination")


def test_a_form_reported_apart_in_one_region_counts_once_with_its_species():
    """Region A fitted the species with a +Na column, region B fitted it base only and reported
    its +Na form apart; they share one scan. Folding before merging (as process_run does)
    gives one species whose +Na form is counted once on the shared scan."""
    na = ADDUCTS["Na"].avg_mass
    ta, tb = np.linspace(10.0, 10.4, 11), np.linspace(10.4, 10.8, 11)
    prof = lambda t: np.exp(-0.5 * ((t - 10.44) / 0.1) ** 2)  # noqa: E731
    a = _species(0, 5162.25, ta, 100 * prof(ta) * 1.1,
                 {"base": 100 * prof(ta), "+Na": 10 * prof(ta)})
    b = _species(1, 5162.27, tb, 100 * prof(tb))
    b_na = _species(2, 5162.27 + na, tb, 10 * prof(tb))
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP"))
    out = merge_adjacent_species(fold_adduct_forms([a, b, b_na], lib))
    assert len(out) == 1
    s = out[0]
    shared = np.searchsorted(s.time, 10.4)
    assert np.isclose(s.adduct_intensity["+Na"][shared], 10 * prof(10.4))
    assert np.isclose(s.base_intensity / s.total_intensity, 1 / 1.1, rtol=0.02)


def test_the_apex_of_a_species_made_of_adduct_columns_is_read_on_all_its_forms():
    t = np.linspace(16.0, 16.8, 21)
    prof = np.exp(-0.5 * ((t - 16.44) / 0.08) ** 2)
    base = np.zeros(t.size)
    base[5] = 0.1  # one stray base reading
    s = _species(0, 6854.23, t, 100 * prof + base, {"base": base, "+K": 100 * prof})
    assert abs(s.rt_apex - 16.44) < 0.03
    # a species with a material base form is read on its base trace
    s2 = _species(1, 5146.2, t, 100 * prof, {"base": 60 * np.roll(prof, 3), "+Na": 40 * prof})
    assert abs(s2.rt_apex - t[int(np.argmax(np.roll(prof, 3)))]) < 1e-9


def test_alignment_refines_small_offsets_around_no_shift():
    mass = 5162.25
    grid = LogMzGrid(1000.0, 2500.0, 2e-5, -1)
    spectrum = np.zeros(grid.size)
    for z in (3, 4, 5):
        t = build_template(mass * np.exp(0.10 * grid.step), z, grid, ORBI, "ps_rna")
        spectrum[t.bins] += 1e6 * t.values
    s = align_template(Candidate(mass, 1.0, {3, 4, 5}), spectrum, grid, ORBI, "ps_rna")
    assert abs(s / grid.step - 0.10) < 0.04


def test_the_envelope_centre_ignores_a_charge_state_on_another_species():
    from lcmsdeconv.deconv.refine import _envelope_centre

    x, y = _profile(4000.0, (3, 4, 5), 1e5, ORBI.fwhm)
    # a stronger species whose z = 5 envelope sits 0.35 Th above this one's z = 4
    _x, y2 = _profile(5001.75, (5,), 2e6, ORBI.fwhm)
    centre = _envelope_centre(4000.0, (3, 4, 5), x, y + y2, -1, "ps_rna",
                              float(ORBI.resolving_power(1000.0)))
    assert centre is not None and abs(centre - 4000.0) < 0.3


def test_a_potassium_ladder_is_walked_down_by_average_mass_shifts():
    from lcmsdeconv.deconv.nnls_solve import fit_weights
    from lcmsdeconv.deconv.refine import _fit_candidate, _recenter_base
    from lcmsdeconv.nn.grid import spectrum_to_grid

    nc = 5162.25
    x, y = _profile(nc, (3, 4, 5), 1e6, ORBI.fwhm)
    _x, y1 = _profile(nc, (3, 4, 5), 6e5, ORBI.fwhm, potassium=1)
    _x, y2 = _profile(nc, (3, 4, 5), 4e5, ORBI.fwhm, potassium=2)
    y = y + y1 + y2
    keep = y > 1.0
    grid, obs, sigma = spectrum_to_grid(x[keep], y[keep], -1, instrument=ORBI)
    res = float(ORBI.resolving_power(1000.0))
    w = fit_weights(obs, sigma, grid.step, res)
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP"), max_per_type=2,
                                  max_total=2)
    cand = Candidate(nc + 2 * ADDUCTS["K"].avg_mass, 1.0, {3, 4, 5})  # the +2K form, as measured
    cand.template_shift = align_template(cand, obs, grid, ORBI, "ps_rna")
    full = _fit_candidate(cand, obs, w, grid, ORBI, "ps_rna", lib.states(), mz_range=(1000., 2500.))
    moved, _f = _recenter_base(cand, full, obs, w, grid, ORBI, "ps_rna", lib.states(), lib,
                               (1000.0, 2500.0), raw_mz=x[keep], raw_int=y[keep],
                               noise_sigma=10.0, resolution=res)
    assert abs(moved.mass - nc) < 0.05


def test_an_adduct_form_merged_into_its_base_keeps_its_amount_through_coincident_drops():
    from lcmsdeconv.deconv.refine import _merge_adduct_components, drop_coincident_adducts

    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP"))
    flp = Component(6793.98, 100.0, charges={4: 100.0}, adducts={"base": 100.0},
                    columns={(4, ""): 100.0})
    po = Component(6777.92, 10.0, charges={4: 10.0}, adducts={"base": 7.0, "+K": 3.0},
                   columns={(4, ""): 7.0, (4, "+K"): 3.0})
    po_na = Component(6777.92 + ADDUCTS["Na"].avg_mass, 2.0, charges={4: 2.0},
                      adducts={"base": 2.0}, columns={(4, ""): 2.0})
    comps = [flp, po, po_na]
    _merge_adduct_components(comps, lib)
    assert len(comps) == 2
    out = drop_coincident_adducts(comps, lib)
    po_out = next(c for c in out if abs(c.mass - 6777.92) < 0.1)
    # PO + K is FLP + Na: that column goes; PO's own +Na form stays
    assert po_out.adducts == {"base": 7.0, "+Na": 2.0} and po_out.intensity == 9.0


def test_isomers_either_side_of_a_boundary_stay_apart():
    t1, t2 = np.linspace(12.0, 13.14, 30), np.linspace(13.14, 13.9, 20)
    a = _species(0, 4841.90, t1, 50 * np.exp(-0.5 * ((t1 - 12.80) / 0.08) ** 2))
    b = _species(1, 4841.78, t2, 70 * np.exp(-0.5 * ((t2 - 13.30) / 0.08) ** 2))
    assert len(merge_adjacent_species([a, b])) == 2
    # one band cut by the boundary is joined
    c = _species(2, 4841.90, t1, 50 * np.exp(-0.5 * ((t1 - 13.10) / 0.15) ** 2))
    d = _species(3, 4841.78, t2, 50 * np.exp(-0.5 * ((t2 - 13.10) / 0.15) ** 2))
    assert len(merge_adjacent_species([c, d])) == 1
