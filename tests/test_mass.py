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
