"""Reported masses: a comb-placed species at its templates' position."""

import numpy as np

from lcmsdeconv.chem.adducts import AdductLibrary
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core.model import Component
from lcmsdeconv.deconv.decode import Candidate
from lcmsdeconv.deconv.refine import (
    ISOTOPE_DA,
    align_candidate,
    align_template,
    refine_frame,
    report_comb_masses,
)
from lcmsdeconv.deconv.templates import build_template
from lcmsdeconv.nn.grid import LogMzGrid

ORBI = InstrumentModel("orbitrap", 140000.0, mz_ref=200.0)


def _grid_spectrum(mass, zs, true):
    """A grid spectrum of a species whose isotopologues sit ``true`` Da from its read mass."""
    grid = LogMzGrid(300.0, 2500.0, 2e-5, -1)
    spectrum = np.zeros(grid.size)
    for z in zs:
        t = build_template(mass + true, z, grid, ORBI, "ps_rna")
        spectrum[t.bins] += 1e6 * t.values
    return grid, spectrum


def _comp(read, tooth, placed=True):
    c = Component(mass=read, intensity=1.0, template_shift=float(np.log(tooth / read)))
    c.comb_placed = placed
    return c


def test_the_alignment_says_whether_the_comb_placed_the_templates():
    """Placed by the comb on the tooth nearest the read mass or on the far tooth (a mass read
    more than half a spacing off), or by the overlap search inside its window."""
    for mass, zs, true, comb in ((1884.27, (2, 3), 0.356, True), (1884.27, (2, 3), 0.10, False),
                                 (4503.16, (3, 4), 0.62, True)):
        grid, spectrum = _grid_spectrum(mass, zs, true)
        c = Candidate(mass, 1.0, set(zs), gate_charges=set(zs))
        align_candidate(c, spectrum, grid, ORBI, "ps_rna")
        assert c.comb_placed is comb
    # a candidate without gate charges keeps the overlap's shift, never the comb's
    grid, spectrum = _grid_spectrum(1884.27, (2, 3), 0.356)
    c = Candidate(1884.27, 1.0, {2, 3}, comb_placed=True)
    assert align_candidate(c, spectrum, grid, ORBI, "ps_rna") == align_template(
        c, spectrum, grid, ORBI, "ps_rna")
    assert c.comb_placed is False


def test_a_comb_placed_species_is_reported_where_its_templates_sit():
    """Where the templates' tooth is the measured mass's nearest (within half a spacing of it),
    the template position is the mass, the templates do not move: n-2 of the NC product read
    0.49 Da light by its mean shift and 0.27 Da heavy, 1883.3 0.34 Da light, 3527.5 of the
    IP-HILIC NC run 0.33 Da heavy (its templates on the far tooth of its gate's mass, 3528.39)."""
    for read, tooth in ((4503.22, 4503.65), (4503.98, 4503.65), (1884.27, 1884.63),
                        (3528.32, 3527.87)):
        c = _comp(read, tooth)
        report_comb_masses([c])
        assert abs(c.mass - tooth) < 1e-9
        assert abs(c.mass * np.exp(c.template_shift) - tooth) < 1e-9
        # placed by the overlap search: the mean shift stands
        c = _comp(read, tooth, placed=False)
        report_comb_masses([c])
        assert c.mass == read
    # a tooth the measured mass does not point to: the measured mass stands and the templates
    # stay where they were placed - FLP + K of the IP-RPLC conjugate run (the overlap chose the
    # far tooth, one spacing light), and a 13 kDa impurity of a synthetic run measured 0.27 Da
    # heavy whose gate mass, read a spacing light, put its templates on the tooth below
    for read, tooth in ((6831.61, 6831.03), (13158.325, 13157.050)):
        c = _comp(read, tooth)
        assert abs(tooth - read) > 0.5 * ISOTOPE_DA
        report_comb_masses([c])
        assert c.mass == read and abs(c.mass * np.exp(c.template_shift) - tooth) < 1e-9
    # a component built without a record of its placement is left alone
    c = Component(mass=1884.27, intensity=1.0, template_shift=float(np.log(1884.63 / 1884.27)))
    report_comb_masses([c])
    assert c.mass == 1884.27


def test_a_comb_placed_species_keeps_its_measured_mass_inside_the_passes():
    """Inside refine_frame the measured mass stands and the placement is recorded; the mass moves
    to the template position only in report_comb_masses, after the passes. The next pass's
    filter of masses already found, the adduct-form exemption, the cation-ladder rule and the
    adduct merges and drops read the measured mass, as before (moved inside the pass, a later
    pass found the same envelope at its gate mass again, more than the filter's half dalton away,
    and fitted it on the other tooth: 4486.64 Da of the IP-RPLC NC run). A 1.9 kDa species read
    light by a companion two spacings lighter on its comb, as 1883.3 is."""
    truth = 1884.61
    grid = LogMzGrid(400.0, 2500.0, 2e-5, -1)
    x = np.arange(400.0, 2500.0, 0.002)
    y = np.zeros_like(x)
    for m, h in ((truth, 1.0), (truth - 2 * ISOTOPE_DA, 0.5)):
        for z in (2, 3):
            t = build_template(m, z, grid, ORBI, "ps_rna")
            y += np.interp(x, grid.bin_to_mz(t.bins), 1e6 * h * t.values, left=0.0, right=0.0)
    observed = grid.resample_profile(x, y)
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP", "K", "Na"),
                                  max_per_type=2, max_total=2)
    weak = Candidate(truth - 0.34, 10.0, {2, 3}, gate_charges={2, 3}, gate_failed=set())
    companion = Candidate(truth - 2 * ISOTOPE_DA, 5.0, {2, 3}, gate_charges={2, 3},
                          gate_failed=set())
    comps, _frac, _res = refine_frame([weak, companion], grid, observed, 100.0, ORBI, "ps_rna",
                                      lib, raw_mz=x, raw_int=y)
    (c,) = [k for k in comps if abs(k.mass - truth) < 1.0]
    tpos = c.mass * np.exp(c.template_shift)
    assert c.comb_placed and abs(tpos - truth) < 0.02
    assert tpos - c.mass > 0.3  # the measured mass, pulled light by the companion
    other = next(k for k in comps if k is not c)
    assert not other.comb_placed
    report_comb_masses(comps)
    assert abs(c.mass - truth) < 0.02 and abs(c.mass * np.exp(c.template_shift) - tpos) < 1e-9
    assert abs(other.mass - (truth - 2 * ISOTOPE_DA)) < 0.05


def test_the_pipeline_reports_a_comb_placed_species_where_its_templates_sit():
    """deconvolve_spectrum end to end, on one spectrum: a 1.9 kDa species with a companion two
    spacings lighter on its comb (1883.3's geometry) is measured 0.46 Da light, placed by its
    comb, and reported where its templates sit, within 0.05 Da of the truth."""
    from lcmsdeconv.chem.classes import class_isotope_pattern
    from lcmsdeconv.core.model import Spectrum
    from lcmsdeconv.deconv import pipeline
    from lcmsdeconv.deconv.classical import CombPredictor

    truth = 1884.61
    x = np.arange(400.0, 2500.0, 0.001)
    y = np.abs(np.random.default_rng(0).normal(0.0, 100.0, x.size))
    for m, h in ((truth, 1.0), (truth - 2 * ISOTOPE_DA, 0.5)):
        pat = class_isotope_pattern(m, "ps_rna")
        for z in (2, 3):
            mz = (pat.masses - pat.average_mass + m - z * 1.00727646688) / z
            sig = ORBI.fwhm(float(mz.mean())) / 2.3548
            for c, v in zip(mz, pat.abundances / pat.abundances.max()):
                if v > 1e-4:
                    a, b = np.searchsorted(x, [c - 5 * sig, c + 5 * sig])
                    y[a:b] += 1e6 * h * v * np.exp(-0.5 * ((x[a:b] - c) / sig) ** 2)
    keep = y > 600.0
    spec = Spectrum(x[keep], y[keep], rt=5.0, polarity=-1, is_profile=True)
    params = pipeline.DeconvParams(compound_class="ps_rna", adduct_mode="iprp",
                                   adduct_exclude=("TEA", "DIPEA", "HFIP"), charge_range=(2, 40),
                                   mass_range=(1000.0, 60000.0), isotope_charges=True)
    fr = pipeline.deconvolve_spectrum(spec, CombPredictor(z_max=40, instrument=ORBI), params, ORBI)
    (c,) = [k for k in fr.components if abs(k.mass - truth) < 1.0]
    assert c.comb_placed
    assert abs(c.mass - c.mass * np.exp(c.template_shift)) < 0.05
    assert abs(c.mass - truth) < 0.05


def test_an_edge_optimum_on_a_tooth_inside_the_window_takes_the_comb():
    """3207.4 of the IP-RPLC NC run, gate mass 3208.40: the nearest tooth of its comb lay 0.005 Da
    inside the 0.289 Da window, nearer the window's edge than the search's step before it, and
    the overlap stopped at the edge. That edge is the tooth, cut short by the window: the
    templates go on the tooth and the species is comb-placed, so it is reported there and not at
    its mean-shift mass, 0.30 Da light. An edge that a charge state the gate did not pass pulls
    off the species' tooth takes the comb within ``off_comb_da`` (0.15 Da) of the tooth, the
    tolerance that puts an optimum inside the window on the comb, and is kept further from it.
    The same at the window's other edge, for a mass read heavy."""
    mass, zs = 3208.40, (3, 4)
    for sign in (1, -1):
        true = sign * 0.284
        grid, spectrum = _grid_spectrum(mass, zs, true)
        c = Candidate(mass, 1.0, set(zs), gate_charges=set(zs))
        edge = align_template(c, spectrum, grid, ORBI, "ps_rna")
        assert 0.288 < sign * mass * np.expm1(edge) < 0.290  # pinned at the window's edge
        s = align_candidate(c, spectrum, grid, ORBI, "ps_rna")
        assert c.comb_placed and abs(mass * np.expm1(s) - true) < 0.005
        comp = _comp(mass, mass * np.exp(s))
        report_comb_masses([comp])
        assert abs(comp.mass - (mass + true)) < 0.005
        # a foreign comb under z = 5, a state the fit added but the gate did not pass, pulls the
        # overlap to the edge, 0.09 and 0.17 Da from the species' own tooth on z = 3 and 4
        for own, placed in ((0.20, True), (0.12, False)):
            grid, spectrum = _grid_spectrum(mass, zs, sign * own)
            t = build_template(mass + sign * 0.30, 5, grid, ORBI, "ps_rna")
            spectrum[t.bins] += 1e7 * t.values
            c = Candidate(mass, 1.0, {3, 4, 5}, gate_charges={3, 4}, gate_failed=set(),
                          observed_extra={5})
            edge = align_template(c, spectrum, grid, ORBI, "ps_rna")
            assert 0.288 < sign * mass * np.expm1(edge) < 0.290
            s = align_candidate(c, spectrum, grid, ORBI, "ps_rna")
            assert c.comb_placed is placed
            if placed:
                assert abs(mass * np.expm1(s) - sign * own) < 0.005
            else:
                assert s == edge


def test_an_edge_optimum_takes_the_tooth_it_stops_beside():
    """The edge is the tooth it stops beside, and that tooth takes the templates, not the better
    of the two either side of the measured mass: 3207.4's geometry read heavy, with an envelope
    three times as strong two spacings heavier on its comb. Fitted jointly, that neighbour sits
    one spacing past the other tooth, beside the pair, and makes the other tooth carry more
    (:func:`_better_tooth` takes it, a spacing beyond the window's other edge, where the overlap
    search never looked); the species is placed on its own tooth."""
    from lcmsdeconv.deconv.refine import _align_charges, _better_tooth, comb_phase_shift

    mass, zs, true = 3208.40, (3, 4), -0.284
    grid = LogMzGrid(300.0, 2500.0, 2e-5, -1)
    spectrum = np.zeros(grid.size)
    for m, h in ((mass + true, 1e6), (mass + true + 2 * ISOTOPE_DA, 3e6)):
        for z in zs:
            t = build_template(m, z, grid, ORBI, "ps_rna")
            spectrum[t.bins] += h * t.values
    c = Candidate(mass, 1.0, set(zs), gate_charges=set(zs), gate_failed=set())
    edge = align_template(c, spectrum, grid, ORBI, "ps_rna")
    assert 0.288 < -mass * np.expm1(edge) < 0.290  # pinned at the window's edge
    comb, _coh, _n = comb_phase_shift(mass, set(zs), grid.bin_to_mz(np.arange(grid.size)),
                                      spectrum, -1, ORBI, "ps_rna")
    assert abs(comb - true) < 0.005
    other = _better_tooth(mass, comb, _align_charges(mass, set(zs), grid, ORBI), spectrum, grid,
                          ORBI, "ps_rna")
    assert abs(other - (comb + ISOTOPE_DA)) < 1e-9  # the joint fit's choice
    s = align_candidate(c, spectrum, grid, ORBI, "ps_rna")
    assert c.comb_placed and abs(mass * np.expm1(s) - true) < 0.005


def test_an_edge_optimum_is_measured_against_every_tooth_in_the_window():
    """Above about 17 kDa the window (1.5 bins, 3e-5 x mass) holds the comb's teeth either side
    of the measured mass. A 20 kDa species read 0.52 Da off has its own tooth on the other side
    of the one nearest the measured mass, inside the 0.6 Da window; a foreign comb under z = 17,
    a state the gate did not pass, pulls the overlap to the edge 0.08 Da past that tooth. The
    edge's distance from the comb is wrapped to the nearest tooth, as inside the window, and
    the species' own tooth takes the templates, read light and read heavy."""
    from lcmsdeconv.deconv.refine import comb_phase_shift

    mass, gate = 20000.0, (14, 15, 16)
    grid = LogMzGrid(300.0, 2500.0, 2e-5, -1)
    for sign in (1, -1):
        true = sign * 0.52
        spectrum = np.zeros(grid.size)
        for z in gate:
            t = build_template(mass + true, z, grid, ORBI, "ps_rna")
            spectrum[t.bins] += 1e6 * t.values
        t = build_template(mass + sign * 0.66, 17, grid, ORBI, "ps_rna")
        spectrum[t.bins] += 1e7 * t.values
        c = Candidate(mass, 1.0, set(gate) | {17}, gate_charges=set(gate), gate_failed=set(),
                      observed_extra={17})
        edge = align_template(c, spectrum, grid, ORBI, "ps_rna")
        assert abs(mass * np.expm1(edge) - sign * 0.6) < 1e-3  # pinned at the window's edge
        comb, _coh, _n = comb_phase_shift(mass, set(gate), grid.bin_to_mz(np.arange(grid.size)),
                                          spectrum, -1, ORBI, "ps_rna")
        assert sign * comb < 0 and abs(comb - true) > 0.9  # the nearest tooth: the other side
        s = align_candidate(c, spectrum, grid, ORBI, "ps_rna")
        assert c.comb_placed and abs(mass * np.expm1(s) - true) < 0.01
