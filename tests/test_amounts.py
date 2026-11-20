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


def test_alignment_does_not_slide_onto_a_neighbour_half_a_dalton_away():
    """Templates move 0.3 Da at most, even where the overlap keeps rising beyond: following it
    to half an isotope spacing let candidates beside the IP-RPLC main species slide onto its
    adduct forms' peaks, and the main species was lost."""
    mass = 6831.56
    grid = LogMzGrid(1000.0, 2500.0, 2e-5, -1)
    spectrum = np.zeros(grid.size)
    for z in (3, 4, 5):
        t = build_template(mass + 0.46, z, grid, ORBI, "ps_rna")
        spectrum[t.bins] += 1e6 * t.values
    s = align_template(Candidate(mass, 1.0, {3, 4, 5}), spectrum, grid, ORBI, "ps_rna")
    assert abs(mass * np.expm1(s)) <= 0.3 + 1e-9


def test_scans_without_a_reading_do_not_split_one_band():
    """The linking leaves zeros where it bridged a gap, and three frames whose fit gave a
    species nothing end its track: neither is a valley between two elution bands."""
    t1, t2 = np.linspace(12.0, 13.14, 30), np.linspace(13.14, 13.9, 20)
    band = lambda t: 50 * np.exp(-0.5 * ((t - 13.0) / 0.15) ** 2)  # noqa: E731
    ya = band(t1)
    ya[-4] = 0.0  # a bridged gap near the boundary
    a = _species(0, 4841.90, t1, ya)
    b = _species(1, 4841.78, t2, band(t2))
    assert len(merge_adjacent_species([a, b])) == 1
    t = np.round(np.arange(9.0, 10.2, 0.04), 6)
    y = 100 * np.exp(-0.5 * ((t - 9.5) / 0.12) ** 2)
    k = int(np.argmax(y))
    ya = y[:k + 5].copy()
    ya[k + 3:] = 0.0  # the track bridged two of the three empty frames, then ended
    a = _species(0, 3208.52, t[:k + 5], ya)
    b = _species(1, 3208.52, t[k + 6:], y[k + 6:])
    out = merge_adjacent_species([a, b])
    assert len(out) == 1 and abs(out[0].rt_apex - 9.5) < 0.03


def test_combined_adduct_labels_beyond_the_library_are_fitted_and_judged_at_their_mass():
    """A component merged with its +2K form carries a +3K column where the library lists two
    of a type at most: that column is judged at PO + 3K, and fitted frame by frame there."""
    from lcmsdeconv.chem.adducts import carrier_mass
    from lcmsdeconv.deconv.refine import _merge_adduct_components, drop_coincident_adducts
    from lcmsdeconv.features.quantify import TemplateBank

    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP"), max_per_type=2,
                                  max_total=3)
    k, na = ADDUCTS["K"].avg_mass, ADDUCTS["Na"].avg_mass
    assert "+3K" not in {s.label for s in lib.states()}
    assert abs(lib.state_of("+3K").avg_mass - 3 * k) < 1e-9
    # a stronger species whose ladder meets PO + 3K (at its +2Na) but not PO + K or PO + 2K
    x = Component(6777.92 + 3 * k - 2 * na, 50.0, charges={4: 50.0}, adducts={"base": 50.0},
                  columns={(4, ""): 50.0})
    po = Component(6777.92, 10.0, charges={4: 10.0}, adducts={"base": 10.0},
                   columns={(4, ""): 10.0})
    po2k = Component(6777.92 + 2 * k, 3.0, charges={4: 3.0}, adducts={"base": 2.0, "+K": 1.0},
                     columns={(4, ""): 2.0, (4, "+K"): 1.0})
    comps = [x, po, po2k]
    _merge_adduct_components(comps, lib)
    po_m = next(c for c in comps if abs(c.mass - 6777.92) < 0.1)
    assert (4, "+3K") in po_m.columns and (4, "+2K") in po_m.columns
    out = drop_coincident_adducts(comps, lib)
    po_out = next(c for c in out if abs(c.mass - 6777.92) < 0.1)
    assert "+3K" not in po_out.adducts and "+2K" in po_out.adducts
    # the bank's +3K column sits 3 (K - H) above the base column
    grid = LogMzGrid(1000.0, 2500.0, 2e-5, -1)
    comp = Component(6777.92, 11.0, charges={4: 11.0}, adducts={"base": 10.0, "+3K": 1.0},
                     columns={(4, ""): 10.0, (4, "+3K"): 1.0}, compound_class="ps_rna")
    bank = TemplateBank([comp], grid, ORBI, lib, compound_class="ps_rna")
    centre = {}
    for j, key in enumerate(bank.keys):
        w = bank.A[:, j]
        mz = float(np.sum(grid.bin_to_mz(bank.support) * w) / w.sum())
        centre[key[1]] = 4 * (mz - carrier_mass(-1))
    assert abs(centre["+3K"] - centre["base"] - 3 * ADDUCTS["K"].mass) < 0.05

def test_a_fold_on_the_weaker_region_piece_is_still_reported():
    """The fold flag of the piece that took a form apart survives the merge of the pieces."""
    na = ADDUCTS["Na"].avg_mass
    ta, tb = np.linspace(10.0, 10.4, 11), np.linspace(10.4, 10.8, 11)
    prof = lambda t: np.exp(-0.5 * ((t - 10.30) / 0.1) ** 2)  # noqa: E731
    a = _species(0, 5162.25, ta, 100 * prof(ta))
    b = _species(1, 5162.27, tb, 100 * prof(tb))
    b_na = _species(2, 5162.27 + na, tb, 10 * prof(tb))
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP"))
    out = merge_adjacent_species(fold_adduct_forms([a, b, b_na], lib))
    assert len(out) == 1
    assert any(f.startswith("adduct forms reported apart counted with it") for f in out[0].flags)


def test_isomers_seen_in_opposite_polarities_at_different_times_stay_apart():
    from lcmsdeconv.features.link import merge_species_across_polarity

    t = np.linspace(3.0, 11.0, 201)

    def sp(sid, mass, pol, apex):
        y = 100 * np.exp(-0.5 * ((t - apex) / 0.1) ** 2)
        return Species(sid, mass, "average", pol, "small_molecule", t, y, {"base": y.copy()},
                       {1: float(y.sum())})

    assert len(merge_species_across_polarity([sp(0, 302.115, -1, 4.0),
                                              sp(1, 302.118, 1, 10.0)])) == 2
    both = merge_species_across_polarity([sp(0, 302.115, -1, 4.0), sp(1, 302.118, 1, 4.02)])
    assert len(both) == 1 and "also seen in" in both[0].flags[-1]


def test_an_adduct_given_as_a_formula_has_states():
    lib = AdductLibrary.from_mode("iprp", -1, include=("C7H17N",), max_per_type=2, max_total=2)
    st = lib.state_of("+2C7H17N")
    assert st is not None and abs(st.mass - 2 * 115.1361) < 0.01
    assert "+2C7H17N" in {s.label for s in lib.states()}


def test_isomers_seen_in_a_scan_or_two_a_minute_apart_stay_apart():
    """Empty scans are read across only in short runs: a long stretch without a reading is a
    valley, however high the few readings either side."""
    t = np.round(np.arange(12.0, 14.0, 0.04), 6)
    ya, yb = np.zeros(t.size), np.zeros(t.size)
    ia, ib = int(np.searchsorted(t, 12.52)), int(np.searchsorted(t, 13.04))
    ya[ia:ia + 2] = [40803.0, 41210.0]
    yb[ib:ib + 2] = [40704.0, 40357.0]
    a = _species(0, 4841.90, t, ya)
    b = _species(1, 4841.90, t, yb)
    assert len(merge_adjacent_species([a, b])) == 2


def test_merged_pieces_keep_only_flags_that_still_hold():
    from lcmsdeconv.features.link import FOLD_FLAG

    ta, tb = np.linspace(10.0, 10.4, 11), np.linspace(10.4, 10.8, 11)
    prof = lambda t: np.exp(-0.5 * ((t - 10.30) / 0.1) ** 2)  # noqa: E731
    a = _species(0, 5162.25, ta, 100 * prof(ta))
    a.charges = {3: 10.0, 4: 60.0, 5: 5.0}
    a.flags = [FOLD_FLAG + "+Na (5184.21 Da)"]
    b = _species(1, 5162.27, tb, 40 * prof(tb))
    b.charges = {4: 20.0}
    b.flags = ["one charge state: charge read from isotope spacing",
               FOLD_FLAG + "+Na (5184.27 Da), +K (5200.33 Da)"]
    out = merge_adjacent_species([a, b])
    assert len(out) == 1
    flags = out[0].flags
    assert not any(f.startswith("one charge state") for f in flags)
    fold = [f for f in flags if f.startswith(FOLD_FLAG)]
    assert fold == [FOLD_FLAG + "+Na (5184.21 Da), +K (5200.33 Da)"]


def test_an_adduct_formula_must_add_mass_and_not_repeat_one_listed():
    import pytest

    with pytest.raises(ValueError):
        AdductLibrary.from_mode("hilic", 1, include=("Mg-H2",))
    assert AdductLibrary.from_mode("hilic", 1, include=("NH3",)).names() == \
        AdductLibrary.from_mode("hilic", 1).names()
    assert "C6H15N" not in AdductLibrary.from_mode("iprp", -1, include=("C6H15N",)).names()


def _comb_profile(mass, charges, shift, height=1e6, lo=400.0, hi=2600.0, step=0.002):
    pat = class_isotope_pattern(mass, "ps_rna")
    masses, ab = pat.masses - pat.average_mass + mass + shift, pat.abundances
    x = np.arange(lo, hi, step)
    y = np.zeros_like(x)
    for z in charges:
        mz = (masses - z * MP) / z
        sig = ORBI.fwhm(mz.mean()) / 2.3548
        for c, h in zip(mz, ab / ab.max()):
            a, b = np.searchsorted(x, [c - 5 * sig, c + 5 * sig])
            y[a:b] += height * h * np.exp(-0.5 * ((x[a:b] - c) / sig) ** 2)
    return x, y


def test_the_comb_phase_reads_where_a_species_isotopologues_sit():
    """The phase of a candidate's own resolved isotopologues gives the template's offset
    directly, up to half an isotope spacing either way, whatever its average mass was read as."""
    from lcmsdeconv.deconv.refine import comb_phase_shift

    rng = np.random.default_rng(1)
    for mass, zs in ((1884.27, (2, 3)), (4504.02, (3, 4)), (6794.0, (3, 4, 5))):
        for true in (0.0, 0.356, -0.40, 0.48):
            x, y = _comb_profile(mass, zs, true)
            y = y + np.abs(rng.normal(0.0, 2e3, y.size))
            s, coherence, n = comb_phase_shift(mass, zs, x, y, -1, ORBI, "ps_rna")
            assert n == len(zs) and abs(s - true) < 0.01 and coherence > 0.8
    # two species of different phase in one window: the agreement drops
    x, y = _comb_profile(4504.02, (3, 4), 0.356)
    _x, y2 = _comb_profile(4504.02 + 0.98, (3, 4), 0.0)
    _s, coherence, _n = comb_phase_shift(4504.02, (3, 4), x, y + y2, -1, ORBI, "ps_rna")
    assert coherence < 0.5


def test_the_gate_keeps_the_charge_states_that_passed_apart():
    """A charge state voted for but contradicted by its own isotopes stays in the candidate's set
    (the fit may use it), and is not among the states whose envelopes passed."""
    from lcmsdeconv.deconv.decode import refine_candidate_masses

    x, y = _comb_profile(4503.71, (3, 4), 0.0, lo=400.0, hi=2000.0)
    # an unrelated singly charged peak where z = 9 of the candidate would sit
    mz9 = (4503.71 - 9 * MP) / 9
    y = y + 5e6 * np.exp(-0.5 * ((x - mz9) / 0.003) ** 2)
    out = refine_candidate_masses([Candidate(4503.71, 1.0, {3, 4, 9})], x, y, -1,
                                  noise_sigma=1e3, compound_class="ps_rna", resolution=70000.0,
                                  instrument=ORBI, profile=True)
    assert len(out) == 1
    assert out[0].gate_charges == {3, 4} and 9 in out[0].charges
    assert out[0].gate_failed == {9}


def test_a_candidate_renamed_to_a_lighter_adduct_partner_loses_its_gate_charges():
    from lcmsdeconv.deconv.decode import dedupe_adduct_candidates

    strong = Candidate(6794.08, 1.34e6, {3, 4, 5}, gate_charges={3, 4, 5})
    weak = Candidate(6750.58, 1.2e5, {3, 4}, gate_charges={3, 4})
    kept = dedupe_adduct_candidates([strong, weak], {"Na": ADDUCTS["Na"].avg_mass})
    assert len(kept) == 1 and abs(kept[0].mass - 6750.58) < 1e-9
    # its gate measured FLP's envelopes, not this mass's
    assert kept[0].gate_charges is None and kept[0].gate_failed is None


def _edge_case(mass, zs, true, foreign=None):
    """A grid spectrum of a species whose isotopologues sit ``true`` Da from its read mass, plus
    an optional foreign species ``(mass, charges, shift, height)``."""
    grid = LogMzGrid(300.0, 2500.0, 2e-5, -1)
    spectrum = np.zeros(grid.size)
    for z in zs:
        t = build_template(mass + true, z, grid, ORBI, "ps_rna")
        spectrum[t.bins] += 1e6 * t.values
    if foreign is not None:
        fm, fzs, fs, fh = foreign
        for z in fzs:
            t = build_template(fm + fs, z, grid, ORBI, "ps_rna")
            spectrum[t.bins] += fh * t.values
    return grid, spectrum


def test_an_edge_pinned_template_takes_the_candidates_own_comb():
    """1.9 kDa shortmer read 0.34 Da light: the overlap stops at the 0.3 Da edge, the comb of the
    gate-passed states puts the templates on the isotopologues. Not for a candidate without gate
    charges or for an adduct form of an accepted species."""
    from lcmsdeconv.deconv.refine import align_candidate

    mass, true = 1884.27, 0.356
    grid, spectrum = _edge_case(mass, (2, 3), true)
    edge = align_template(Candidate(mass, 1.0, {2, 3}), spectrum, grid, ORBI, "ps_rna")
    assert abs(mass * np.expm1(edge) - 0.292) < 0.01  # pinned at the window's edge
    c = Candidate(mass, 1.0, {2, 3}, gate_charges={2, 3})
    s = align_candidate(c, spectrum, grid, ORBI, "ps_rna")
    assert abs(mass * np.expm1(s) - true) < 0.01
    no_gate = Candidate(mass, 1.0, {2, 3})
    assert align_candidate(no_gate, spectrum, grid, ORBI, "ps_rna") == edge
    form = align_candidate(c, spectrum, grid, ORBI, "ps_rna", forms_of=[mass - 21.98],
                           form_shifts=[21.98])
    assert form == edge


def test_a_mass_read_more_than_half_a_spacing_off_takes_the_better_tooth():
    """Its nearest comb tooth is on the wrong side: the template goes to the tooth that fits the
    envelope better, not to the window's edge between them (which fits nothing)."""
    from lcmsdeconv.deconv.refine import align_candidate

    for mass, zs, true in ((1884.27, (2, 3), -0.60), (4503.16, (3, 4), 0.62)):
        grid, spectrum = _edge_case(mass, zs, true)
        c = Candidate(mass, 1.0, set(zs), gate_charges=set(zs))
        s = align_candidate(c, spectrum, grid, ORBI, "ps_rna")
        assert abs(mass * np.expm1(s) - true) < 0.02


def test_the_comb_needs_its_charge_states_to_agree():
    """Two gate states whose windows hold combs half a spacing apart, or one state alone, do not
    move the template off the window's edge; a state that failed the gate is not read."""
    from lcmsdeconv.deconv.refine import align_candidate

    mass = 1884.27
    grid = LogMzGrid(400.0, 2500.0, 2e-5, -1)
    spectrum = np.zeros(grid.size)
    for z, shift in ((2, 0.35), (3, -0.35)):
        t = build_template(mass + shift, z, grid, ORBI, "ps_rna")
        spectrum[t.bins] += 1e6 * t.values
    c = Candidate(mass, 1.0, {2, 3}, gate_charges={2, 3})
    edge = align_template(c, spectrum, grid, ORBI, "ps_rna")
    assert abs(abs(mass * np.expm1(edge)) - 0.292) < 0.01
    assert align_candidate(c, spectrum, grid, ORBI, "ps_rna") == edge
    grid, spectrum = _edge_case(mass, (2, 3), 0.356)
    one = Candidate(mass, 1.0, {2, 3}, gate_charges={2})
    assert align_candidate(one, spectrum, grid, ORBI, "ps_rna") == align_template(
        one, spectrum, grid, ORBI, "ps_rna")
    # a second state under a tenth of the weight cannot veto, but counts towards the two only
    # where it agrees: contradicting z = 2 by 0.7 Da it leaves the edge, agreeing it is a check
    for z3_shift, placed in ((-0.35, False), (0.356, True)):
        spectrum = np.zeros(grid.size)
        for z, shift, h in ((2, 0.356, 1e6), (3, z3_shift, 1e5)):
            t = build_template(mass + shift, z, grid, ORBI, "ps_rna")
            spectrum[t.bins] += h * t.values
        two = Candidate(mass, 1.0, {2, 3}, gate_charges={2, 3}, gate_failed=set())
        s = align_candidate(two, spectrum, grid, ORBI, "ps_rna")
        assert (s != align_template(two, spectrum, grid, ORBI, "ps_rna")) == placed
        if placed:
            assert abs(mass * np.expm1(s) - 0.356) < 0.02
    # another species' comb under z = 5, a state that failed the gate: read by the overlap
    # search it would pull the template onto that comb, 0.2 Da light; read by the comb it would
    # break the agreement. The template follows 2 and 3
    grid, spectrum = _edge_case(mass, (2, 3), 0.356, foreign=(mass, (5,), -0.2, 3e5))
    c5 = Candidate(mass, 1.0, {2, 3, 5}, gate_charges={2, 3}, gate_failed={5})
    s = align_candidate(c5, spectrum, grid, ORBI, "ps_rna")
    assert abs(mass * np.expm1(s) - 0.356) < 0.01


def test_the_adduct_walk_reads_only_the_candidates_own_envelopes():
    """n-2 of the NC product voted on z = 3, 4 and 9, with no envelope of its own at z = 9: the
    gate passes 3 and 4 and cannot measure 9, which stays in the candidate's set. An unrelated
    peak 30 times taller under z = 9 one sodium lower must not walk it down to a phantom - read
    with the verdicts the gate gives, or with a duplicate merged in that brought z = 9 along."""
    from lcmsdeconv.deconv.decode import refine_candidate_masses
    from lcmsdeconv.deconv.nnls_solve import fit_weights
    from lcmsdeconv.deconv.pipeline import _dedupe_by_mass
    from lcmsdeconv.deconv.refine import _fit_candidate, _recenter_base

    mass = 4503.71
    grid = LogMzGrid(400.0, 2000.0, 2e-5, -1)
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP", "K"),
                                  max_per_type=2, max_total=2)
    x, y = _comb_profile(mass, (3, 4), 0.0, height=1e5, lo=400.0, hi=2000.0)
    # another species' z = 9 envelope where n-2 less a sodium would put its z = 9, and weak
    # signal one sodium lower on z = 3 and 4, as a crowded region has
    _x, y_9 = _comb_profile(mass - ADDUCTS["Na"].avg_mass, (9,), 0.0, height=3e6,
                            lo=400.0, hi=2000.0)
    _x, y_low = _comb_profile(mass - ADDUCTS["Na"].avg_mass, (3, 4), 0.0, height=1e4,
                              lo=400.0, hi=2000.0)
    y = y + y_9 + y_low
    observed = grid.resample_profile(x, y)
    weights = fit_weights(observed, 50.0, grid.step, 70000.0)
    states = lib.states()

    def walk(cand):
        full = _fit_candidate(cand, observed.copy(), weights, grid, ORBI, "ps_rna", states,
                              mz_range=(400.0, 2000.0))
        moved, _ = _recenter_base(cand, full, observed.copy(), weights, grid, ORBI, "ps_rna",
                                  states, lib, (400.0, 2000.0), raw_mz=x, raw_int=y,
                                  noise_sigma=50.0, resolution=70000.0)
        return moved.mass

    (gated,) = refine_candidate_masses([Candidate(mass, 1.0, {3, 4, 9})], x, y, -1,
                                       noise_sigma=50.0, compound_class="ps_rna",
                                       resolution=70000.0, instrument=ORBI, profile=True)
    assert gated.gate_charges == {3, 4} and 9 in gated.charges and 9 not in gated.gate_failed
    assert abs(walk(gated) - mass) < 0.5  # stays
    # a weaker pick of the same mass that voted for z = 9 too, merged in by mass
    strong = Candidate(mass, 2.0, {3, 4}, gate_charges={3, 4}, gate_failed=set())
    (merged,) = _dedupe_by_mass([strong, Candidate(mass + 0.05, 1.0, {3, 4, 9},
                                                    gate_charges={3, 4}, gate_failed=set())])
    assert 9 in merged.charges and abs(walk(merged) - mass) < 0.5
    # without a gate record every voted state is read, and the walk takes the phantom
    lost = walk(Candidate(mass, 1.0, {3, 4, 9}))
    assert abs(lost - (mass - ADDUCTS["Na"].avg_mass)) < 0.5


def test_a_comb_placed_candidate_on_an_adduct_form_still_walks_to_its_base():
    """The candidate sits on the +Na form, read 0.36 Da light: placed by its comb it must be
    compared with a lighter trial placed the same way, or the walk is always refused."""
    from lcmsdeconv.deconv.refine import refine_frame

    base = 2549.80
    na = ADDUCTS["Na"].avg_mass
    grid = LogMzGrid(400.0, 2500.0, 2e-5, -1)
    x = np.arange(400.0, 2500.0, 0.002)
    y = np.zeros_like(x)
    for m, h in ((base, 1e6), (base + na, 6e5)):
        _x, yy = _comb_profile(m, (2, 3), 0.0, height=h, lo=400.0, hi=2500.0)
        y += yy
    observed = grid.resample_profile(x, y)
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP", "K"),
                                  max_per_type=2, max_total=2)
    cand = Candidate(base + na - 0.36, 10.0, {2, 3}, gate_charges={2, 3}, gate_failed=set())
    comps, _frac, _res = refine_frame([cand], grid, observed, 100.0, ORBI, "ps_rna", lib,
                                      raw_mz=x, raw_int=y)
    assert comps and abs(comps[0].mass - base) < 0.6


def test_a_single_charge_candidate_walks_on_its_other_observed_states():
    """A candidate gated on one charge state also reads the states observed_charges finds beside
    it (each with its isotope comb, at its mass), not only that one."""
    from lcmsdeconv.deconv.nnls_solve import fit_weights
    from lcmsdeconv.deconv.refine import _fit_candidate, _recenter_base, observed_charges

    base = 5162.25
    na = ADDUCTS["Na"].avg_mass
    grid = LogMzGrid(400.0, 2500.0, 2e-5, -1)
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP", "K"),
                                  max_per_type=2, max_total=2)
    x = np.arange(400.0, 2500.0, 0.002)
    y = np.zeros_like(x)
    for m, h in ((base, 1e6), (base + na, 4e5)):
        _x, yy = _comb_profile(m, (3, 4, 5), 0.0, height=h, lo=400.0, hi=2500.0)
        y += yy
    observed = grid.resample_profile(x, y)
    weights = fit_weights(observed, 50.0, grid.step, 70000.0)
    states = lib.states()
    cand = Candidate(base + na, 1.0, {4}, gate_charges={4}, gate_failed=set(),
                     single_charge=True)
    # what refine_frame does before the walk
    cand.charges = observed_charges(cand.mass, cand.charges, x, y, 150.0, -1, "ps_rna", 70000.0,
                                    (400.0, 2500.0), instrument=ORBI)
    cand.observed_extra = cand.charges - {4}
    assert cand.charges == {3, 4, 5}
    full = _fit_candidate(cand, observed.copy(), weights, grid, ORBI, "ps_rna", states,
                          mz_range=(400.0, 2500.0))
    moved, _ = _recenter_base(cand, full, observed.copy(), weights, grid, ORBI, "ps_rna", states,
                              lib, (400.0, 2500.0), raw_mz=x, raw_int=y, noise_sigma=50.0,
                              resolution=70000.0)
    assert abs(moved.mass - base) < 0.5


def test_a_duplicate_merged_by_mass_brings_its_gate_verdicts():
    """Within the by-mass tolerance a duplicate's gate measured the same windows: the states it
    passed are the kept candidate's own and those it rejected are not; where the two disagree
    the kept candidate's verdict stands."""
    from lcmsdeconv.deconv.pipeline import _dedupe_by_mass
    from lcmsdeconv.deconv.refine import own_charges

    weak = Candidate(6752.50, 1.0, {3, 4, 15}, gate_charges={3, 4}, gate_failed={15})
    strong = Candidate(6752.35, 2.0, {4, 5}, gate_charges={4, 5}, gate_failed=set())
    (kept,) = _dedupe_by_mass([strong, weak])
    assert kept.charges == {3, 4, 5, 15}
    assert kept.gate_charges == {3, 4, 5} and kept.gate_failed == {15}
    assert own_charges(kept) == {3, 4, 5}
    strong = Candidate(6752.35, 2.0, {4, 5, 15}, gate_charges={4, 5, 15}, gate_failed=set())
    (kept,) = _dedupe_by_mass([strong, weak])
    assert kept.gate_charges == {3, 4, 5, 15} and not kept.gate_failed


def test_a_lower_charge_comb_read_at_twice_its_charge_contradicts_it():
    """The singly and doubly charged ions of a 915 Da species sit where charges 2 and 4 of an
    1831 Da one would, and fill two of its three major isotopologue positions - every other
    one. They are not that species' envelopes; its own are."""
    from lcmsdeconv.deconv.decode import isotopes_contradict_charge

    half, mass = 915.84, 1831.68
    x, y_half = _comb_profile(half, (1, 2), 0.0, lo=400.0, hi=1000.0)
    _x, y_own = _comb_profile(mass, (2, 4), 0.0, lo=400.0, hi=1000.0)
    for z in (2, 4):
        centre = (mass - z * MP) / z
        fwhm = centre / float(ORBI.resolving_power(centre))
        lo, hi = np.searchsorted(x, [centre - 1.5, centre + 1.5])
        for y, contradicts in ((y_half, True), (y_own, False)):
            assert isotopes_contradict_charge(mass, z, centre, x, y, 1e3, -1, "ps_rna", fwhm,
                                              height=float(y[lo:hi].max())) == contradicts


def test_a_mass_walked_from_anchors_the_adduct_forms_above_it():
    """NC + 3Na walked onto NC in an earlier pass: NC's +5Na form, read 0.5 Da light and beyond
    the three sodium a fit allows, keeps its window's edge as a form of the walked mass instead
    of taking its comb - placed on it, it is fitted whole, and NC's adduct forms were reported
    as a 3.3 % impurity."""
    from lcmsdeconv.deconv.refine import refine_frame

    na = ADDUCTS["Na"].avg_mass
    nc = 5162.25
    read = nc + 5 * na - 0.5
    grid, spectrum = _edge_case(read, (4, 5), 0.5)
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP", "K"),
                                  max_per_type=2, max_total=2)

    def placed(walked):
        cand = Candidate(read, 1.0, {4, 5}, gate_charges={4, 5}, gate_failed=set())
        comps, _frac, _res = refine_frame([cand], grid, spectrum, 100.0, ORBI, "ps_rna", lib,
                                          known_masses=[nc], walked=walked)
        return read * np.expm1(comps[0].template_shift)

    assert abs(placed([]) - 0.5) < 0.05  # its comb
    assert abs(placed([nc + 3 * na - 0.6]) - 0.29) < 0.03  # the window's edge
