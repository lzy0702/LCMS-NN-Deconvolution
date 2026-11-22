"""Pieces of one species that region boundaries leave apart: a chain of tail pieces, and a
species found again later in the run (its tail beyond its band, the column wash), named in its
flags, not reported."""

import numpy as np

from lcmsdeconv.chem.classes import class_isotope_pattern
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core.model import Run, Species, Spectrum
from lcmsdeconv.deconv.refine import tooth_position
from lcmsdeconv.features.link import LATER_FLAG, merge_adjacent_species, set_aside_later_elutions
from lcmsdeconv.process import _same_molecule

MP = 1.00727646688
ORBI = InstrumentModel("orbitrap", 140000.0)
STEP = 0.04


def _species(sid, mass, t, y, charges=None):
    y = np.asarray(y, dtype=float)
    return Species(sid, mass, "average", -1, "ps_rna", np.asarray(t, dtype=float), y,
                   {"base": y.copy()}, charges or {4: float(y.sum())})


def _scan(rt, parts, lo=750.0, hi=1800.0, step=0.002):
    """A profile scan holding the envelopes ``parts`` = [(average mass, charges, height)]."""
    x = np.arange(lo, hi, step)
    y = np.zeros_like(x)
    for mass, charges, height in parts:
        pat = class_isotope_pattern(mass, "ps_rna")
        masses = pat.masses - pat.average_mass + mass
        ab = pat.abundances / pat.abundances.max()
        for z in charges:
            mz = (masses - z * MP) / z
            sig = float(ORBI.fwhm(mz.mean())) / 2.3548
            for c, h in zip(mz, ab):
                a, b = np.searchsorted(x, [c - 5 * sig, c + 5 * sig])
                y[a:b] += height * h * np.exp(-0.5 * ((x[a:b] - c) / sig) ** 2)
    keep = y > 1e-3 * max(y.max(), 1e-300)  # thresholded, as vendor software writes profiles
    return Spectrum(x[keep], y[keep], rt=rt, polarity=-1, is_profile=True)


def test_merging_repeats_until_a_chain_of_pieces_is_joined():
    """Pieces are joined strongest first: a tail piece that touches only a weaker piece of the
    band, not yet joined, stayed apart (2998.30 Da of the IP-RPLC conjugate run)."""
    def band(t):  # a peak with a slow tail
        peak = 100 * np.exp(-0.5 * ((t - 0.5) / 0.25) ** 2)
        return np.where(t <= 1.0, peak, 100 * np.exp(-0.5 * 2.0 ** 2) * np.exp(-(t - 1.0) / 1.5))

    ta, tb, tc = np.linspace(0.0, 1.0, 26), np.linspace(1.0, 1.24, 7), np.linspace(1.24, 2.4, 30)
    a = _species(0, 2998.45, ta, band(ta))
    b = _species(1, 2998.32, tb, band(tb))
    c = _species(2, 2998.30, tc, band(tc))
    assert c.total_intensity > b.total_intensity  # taken before b, while a's trace ends at 1.0
    out = merge_adjacent_species([a, b, c])
    assert len(out) == 1 and out[0].mass == 2998.45
    assert np.isclose(out[0].time.max(), 2.4)


def test_the_tooth_of_the_isotope_comb_tells_a_deaminated_form_from_its_parent():
    m = 4832.98
    one = [_scan(1.0 + 0.04 * i, [(m, (4, 5, 6), 1e6 * (1 + i))]) for i in range(3)]
    weak = [_scan(2.0, [(m, (4, 5, 6), 2e4)])]
    deam = [_scan(3.0, [(m + 0.984, (4, 5, 6), 5e5)])]
    # read from a mass 0.4 Da off, as a region's blend puts it
    t_one = tooth_position(m + 0.4, (4, 5, 6), one, ORBI, "ps_rna")
    t_weak = tooth_position(m + 0.4, (4, 5, 6), weak, ORBI, "ps_rna")
    t_deam = tooth_position(m + 0.4, (4, 5, 6), deam, ORBI, "ps_rna")
    assert abs(t_one[0] - m) < 0.05 and abs(t_weak[0] - m) < 0.05
    assert abs(t_deam[0] - (m + 0.984)) < 0.05
    assert min(t_one[1], t_weak[1], t_deam[1]) > 0.02
    # isotopes not resolved, or no signal in the windows: no reading
    tof = InstrumentModel("tof", 4000.0)
    assert tooth_position(m, (4, 5, 6), one, tof, "ps_rna") is None
    assert tooth_position(m - 600.0, (4, 5, 6), one, ORBI, "ps_rna") is None


def test_the_isotope_judgement_reads_the_scans_either_side():
    """``_same_molecule`` reads the scans of each band: n-1 on both sides of a boundary is one
    molecule; a deaminated form eluting after it is another."""
    m = 4832.98
    scans = [_scan(13.02 + STEP * i, [(m, (4, 5, 6), 1e6)]) for i in range(8)]
    scans += [_scan(14.02 + STEP * i, [(m + 0.984, (4, 5, 6), 1e6)]) for i in range(4)]
    run = Run(scans)
    same = _same_molecule(run, ORBI, "ps_rna")
    a = _species(0, 4833.86, [13.06], [1.0], {4: 3.0, 5: 1.0, 6: 0.5})
    b = _species(1, 4833.14, [13.22], [1.0], {4: 3.0, 5: 1.0, 6: 0.5})
    assert same(b, (13.17, 13.27), a, (13.01, 13.11))
    c = _species(2, 4833.90, [14.08], [1.0], {4: 3.0, 5: 1.0, 6: 0.5})
    assert not same(c, (14.01, 14.15), a, (13.01, 13.11))
    assert not same(c, (15.0, 15.1), a, (13.01, 13.11))  # no scans there
    assert not _same_molecule(run, InstrumentModel("tof", 4000.0), "ps_rna")(
        b, (13.17, 13.27), a, (13.01, 13.11))  # isotopes not resolved


def test_a_species_found_again_later_is_named_not_reported():
    t0 = np.arange(17.0, 18.74 + 1e-9, STEP)
    main = _species(0, 5162.28, t0, 1e8 * np.exp(-0.5 * ((t0 - 18.28) / 0.12) ** 2))
    t1, t2 = np.arange(19.89, 22.55 + 1e-9, STEP), np.arange(24.98, 28.57 + 1e-9, STEP)
    tail = _species(1, 5162.23, t1, 3e5 * np.exp(-(t1 - 19.89) / 2.0))
    ret = _species(2, 5162.13, t2, 4e6 * np.exp(-0.5 * ((t2 - 25.42) / 0.06) ** 2))
    other = _species(3, 5146.20, t1, 3e5 * np.exp(-(t1 - 19.89) / 2.0))
    before = _species(4, 5162.21, np.linspace(15.0, 15.76, 20), np.full(20, 3e5))
    t3 = np.arange(19.0, 19.5, STEP)
    big = _species(5, 5162.30, t3, 2e7 * np.exp(-0.5 * ((t3 - 19.25) / 0.08) ** 2))
    t4 = np.arange(18.74, 19.0, STEP)  # touching the main band: merge_adjacent_species' business
    near = _species(6, 5162.25, t4, 2e5 * np.exp(-(t4 - 18.74) / 0.2))
    base = main.base_intensity
    out = set_aside_later_elutions([main, tail, ret, other, before, big, near],
                                   lambda *a: True, {-1: STEP})
    assert sorted(s.id for s in out) == [0, 3, 4, 5, 6]
    assert main.base_intensity == base
    flag = next(f for f in main.flags if f.startswith(LATER_FLAG))
    assert "5162.23 Da at 19.89 min" in flag and "5162.13 Da at 25.42 min" in flag
    assert not big.flags
    # not the same molecule by its isotopes: reported
    main.flags.clear()
    out = set_aside_later_elutions([main, tail], lambda *a: False, {-1: STEP})
    assert len(out) == 2 and not main.flags


def test_only_the_main_species_and_forms_leaving_with_it_are_set_aside():
    """A weaker species' later piece is set aside only where it leaves the column with one of
    the main species' later elutions (NC's +201.4 Da form in the wash); elsewhere it may be an
    isomer, and it is reported."""
    g = lambda t, c, h: h * np.exp(-0.5 * ((t - c) / 0.08) ** 2)  # noqa: E731
    t0, tw = np.arange(17.5, 19.0, STEP), np.arange(25.02, 26.0, STEP)
    main = _species(0, 5162.28, t0, g(t0, 18.28, 1e8))
    wash = _species(1, 5162.13, tw, g(tw, 25.42, 1.2e6))
    form = _species(2, 5363.53, t0, g(t0, 18.28, 7e6))
    form_wash = _species(3, 5363.31, tw, g(tw, 25.42, 2e5))
    t1, t2 = np.arange(14.5, 15.5, STEP), np.arange(21.5, 22.5, STEP)
    minor = _species(4, 4000.10, t1, g(t1, 15.0, 2e6))
    isomer = _species(5, 4000.15, t2, g(t2, 22.0, 5e4))  # no main-species elution with it
    out = set_aside_later_elutions([main, wash, form, form_wash, minor, isomer],
                                   lambda *a: True, {-1: STEP})
    assert sorted(s.id for s in out) == [0, 2, 4, 5]
    assert any(f.startswith(LATER_FLAG) and "5162.13 Da at 25.42 min" in f for f in main.flags)
    assert any(f.startswith(LATER_FLAG) and "5363.31 Da at 25.42 min" in f for f in form.flags)
    assert not minor.flags


def test_a_slice_of_the_main_species_tail_carries_nothing_off_the_column():
    """The tail beyond the main band is set aside, but its highest reading lies wherever a region
    boundary cut it: a weaker species peaking there is not taken for a form leaving with it."""
    g = lambda t, c, w, h: h * np.exp(-0.5 * ((t - c) / w) ** 2)  # noqa: E731
    t0, t1 = np.arange(17.08, 18.74, STEP), np.arange(19.89, 20.86, STEP)
    main = _species(0, 5162.28, t0, g(t0, 18.28, 0.12, 1e8))
    tail = _species(1, 5162.23, t1, 3e5 * np.exp(-(t1 - 19.89) / 2.0))
    t2, t3 = np.arange(17.2, 17.9, STEP), np.arange(19.7, 20.3, STEP)
    minor = _species(2, 4832.97, t2, g(t2, 17.54, 0.08, 4e6))
    isomer = _species(3, 4833.05, t3, g(t3, 19.93, 0.06, 1.2e5))  # peaks with the tail slice
    out = set_aside_later_elutions([main, tail, minor, isomer], lambda *a: True, {-1: STEP})
    assert sorted(s.id for s in out) == [0, 2, 3]
    assert any(f.startswith(LATER_FLAG) and "5162.23 Da at 19.89 min" in f for f in main.flags)
    assert not minor.flags
