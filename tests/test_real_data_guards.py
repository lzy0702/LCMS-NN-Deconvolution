"""Guards added after validating on real Orbitrap oligonucleotide runs."""

import numpy as np

from lcmsdeconv.chem.classes import class_isotope_pattern
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.deconv.decode import envelope_discontinuous, isotopes_contradict_charge
from lcmsdeconv.features.link import _is_flat, merge_adjacent_species

MP = 1.00727646688


def _profile(centres, heights, fwhm, lo, hi, step=None):
    step = step or fwhm / 6.0
    x = np.arange(lo, hi, step)
    y = np.zeros_like(x)
    sig = fwhm / 2.3548
    for c, h in zip(centres, heights):
        y += h * np.exp(-0.5 * ((x - c) / sig) ** 2)
    return x, y


def _envelope(mass, z, cls="dna", polarity=-1):
    pat = class_isotope_pattern(mass, cls)
    avg = float((pat.abundances * pat.masses).sum() / pat.abundances.sum())
    mz = (pat.masses - avg + mass + z * polarity * MP) / z
    return mz, pat.abundances / pat.abundances.max(), (mass + z * polarity * MP) / z


def test_isotope_comb_accepts_the_true_charge():
    mz, ab, centre = _envelope(5162.0, 4)
    fwhm = centre / 50000.0
    x, y = _profile(mz, 1000 * ab, fwhm, centre - 3, centre + 3)
    assert not isotopes_contradict_charge(5162.0, 4, centre, x, y, 10.0, -1, "dna", fwhm)
    # a mass off by a third of a dalton still passes: the comb may slide
    assert not isotopes_contradict_charge(5162.3, 4, centre + 0.075, x, y, 10.0, -1, "dna", fwhm)


def test_isotope_comb_rejects_singly_charged_background_read_at_high_charge():
    # a z = 1 ion with its isotopes every 1.003 Th, read as a 17 kDa species at z = 30
    fwhm = 600.0 / 60000.0
    x, y = _profile([600.0, 601.0034, 602.0067], [1000, 300, 60], fwhm, 595, 605)
    mass = 30 * (600.0 + MP)
    assert isotopes_contradict_charge(mass, 30, 600.0, x, y, 10.0, -1, "dna", fwhm)


def test_isotope_comb_rejects_a_doubled_charge():
    # a z = 3 envelope read as its own dimer at z = 6: half the comb positions are empty
    mz, ab, centre = _envelope(5162.0, 3)
    fwhm = centre / 50000.0
    x, y = _profile(mz, 1000 * ab, fwhm, centre - 4, centre + 4)
    assert isotopes_contradict_charge(2 * 5162.0 + 0.0, 6, centre, x, y, 10.0, -1, "dna", fwhm)


def test_continuity_separates_a_hump_from_spikes():
    fwhm = 0.04
    # a continuous unresolved envelope
    x, y = _profile(np.linspace(999.6, 1000.4, 40), np.ones(40), fwhm, 999, 1001)
    assert not envelope_discontinuous(1000.0, 0.3, x, y, fwhm)
    # two narrow peaks 0.25 Th apart where a 0.6 Th-wide envelope should be
    x, y = _profile([999.85, 1000.1], [1.0, 0.8], fwhm, 999, 1001)
    assert envelope_discontinuous(1000.0, 0.3, x, y, fwhm)


def test_flatness_looks_at_the_ends_of_the_trace():
    t = np.linspace(-1, 1, 30)
    broad_peak = np.exp(-0.5 * (t / 0.45) ** 2)  # fills its window: max/median below 3
    assert broad_peak.max() / np.median(broad_peak) < 3
    assert not _is_flat(broad_peak, 3.0)
    assert _is_flat(np.full(30, 5.0) + 0.1 * np.sin(np.arange(30)), 3.0)


def test_species_split_over_adjacent_regions_are_merged():
    from lcmsdeconv.core.model import Species

    t1, t2 = np.linspace(1.0, 1.5, 11), np.linspace(1.55, 2.0, 10)
    a = Species(0, 5162.27, "average", -1, "dna", t1, np.linspace(1, 10, 11),
                {"base": np.linspace(1, 10, 11)}, {4: 10.0})
    b = Species(1, 5162.30, "average", -1, "dna", t2, np.linspace(9, 1, 10),
                {"base": np.linspace(9, 1, 10)}, {4: 8.0})
    c = Species(2, 5363.50, "average", -1, "dna", t2, np.ones(10), {"base": np.ones(10)}, {3: 1.0})
    out = merge_adjacent_species([a, b, c])
    assert len(out) == 2
    merged = next(s for s in out if abs(s.mass - 5162.27) < 0.1)
    assert merged.time.size == 21 and merged.charges[4] == 18.0


def test_comb_reads_charge_from_resolved_isotopes():
    from lcmsdeconv.deconv.classical import CombPredictor
    from lcmsdeconv.nn.grid import spectrum_to_grid

    inst = InstrumentModel("orbitrap", 140000.0)
    centres, heights = [], []
    for z in (3, 4):
        mz, ab, _c = _envelope(5162.0, z)
        centres += list(mz)
        heights += list(1e5 * ab)
    x, y = _profile(centres, heights, 1300.0 / inst.resolving_power(1300.0), 1200, 1800,
                    step=0.002)
    keep = y > 1.0
    grid, obs, sigma = spectrum_to_grid(x[keep], y[keep], -1, instrument=inst)
    with_iso = CombPredictor(z_max=40, instrument=inst).predict_grid(grid, obs, max(sigma, 1.0))
    for z in (3, 4):
        _mz, ab, _c = _envelope(5162.0, z)
        apex = _mz[int(np.argmax(ab))]
        b = int(grid.mz_to_bin(apex))
        j = b - 3 + int(np.argmax(obs[b - 3:b + 4]))
        assert with_iso.top1_z[j] == z


def _single_charge_spectrum(mass=6448.7, z=4, height=1e5, extra=()):
    """One isotope-resolved charge state of an oligonucleotide (Orbitrap, 140 000 at m/z 200)."""
    inst = InstrumentModel("orbitrap", 140000.0, mz_ref=200.0)
    mz, ab, centre = _envelope(mass, z)
    fwhm = centre / inst.resolving_power(centre)
    centres, heights = list(mz), list(height * ab)
    for c, h in extra:
        centres.append(c)
        heights.append(h)
    x, y = _profile(centres, heights, fwhm, centre - 4, centre + 4, step=fwhm / 8)
    return inst, x, y, centre, fwhm


def test_isotope_ladder_reads_the_charge_of_one_envelope():
    from lcmsdeconv.deconv.decode import isotope_charge_unique, isotope_ladder_spacing

    inst, x, y, centre, fwhm = _single_charge_spectrum()
    d = isotope_ladder_spacing(centre, 1.0, x, y, 100.0, fwhm)
    assert d is not None and abs(1.00336 / d - 4) < 0.1
    assert isotope_charge_unique(6448.7, 4, centre, x, y, 100.0, -1, "dna", fwhm)
    for wrong in (2, 3, 5, 8):  # the same envelope read at another charge
        m = wrong * (centre + MP)
        assert not isotope_charge_unique(m, wrong, centre, x, y, 100.0, -1, "dna", fwhm), wrong


def test_gate_keeps_a_single_charge_species_only_when_asked():
    from lcmsdeconv.deconv.decode import Candidate, refine_candidate_masses

    inst, x, y, centre, fwhm = _single_charge_spectrum()
    kw = dict(noise_sigma=100.0, compound_class="dna", resolution=50000.0, instrument=inst,
              profile=True)
    assert refine_candidate_masses([Candidate(6448.7, 1.0, {4})], x, y, -1, **kw) == []
    out = refine_candidate_masses([Candidate(6448.7, 1.0, {4})], x, y, -1,
                                  single_charge_ok=True, **kw)
    assert len(out) == 1 and out[0].single_charge and out[0].charges == {4}
    assert abs(out[0].mass - 6448.7) < 0.3


def test_gate_pulls_a_candidate_from_the_edge_of_its_envelope():
    """A strong unrelated peak on the isotope ladder can put the histogram peak 4 Da off."""
    from lcmsdeconv.deconv.decode import Candidate, refine_candidate_masses

    inst = InstrumentModel("orbitrap", 140000.0, mz_ref=200.0)
    centres, heights = [], []
    for z in (3, 4):
        mz, ab, _c = _envelope(6448.7, z)
        centres += list(mz)
        heights += list(1e5 * ab)
    x, y = _profile(centres, heights, 0.035, 1600, 2160, step=0.004)
    kw = dict(noise_sigma=100.0, compound_class="dna", resolution=50000.0, instrument=inst,
              profile=True)
    assert refine_candidate_masses([Candidate(6444.3, 1.0, {3, 4})], x, y, -1, **kw) == []
    out = refine_candidate_masses([Candidate(6444.3, 1.0, {3, 4})], x, y, -1,
                                  envelope_move=True, **kw)
    assert len(out) == 1 and abs(out[0].mass - 6448.7) < 0.3


def test_linking_keeps_the_mass_of_vanishingly_weak_components():
    """Tiny per-frame intensities must not drag a track's mass toward zero."""
    from lcmsdeconv.core.model import Component, FrameResult
    from lcmsdeconv.features.link import link_species

    frames = [FrameResult(1.0 + 0.01 * i, -1,
                          [Component(mass=6578.66, intensity=1e-14, charges={4: 1e-14},
                                     adducts={"base": 1e-14}, compound_class="dna")])
              for i in range(12)]
    species = link_species(frames, min_frames=1, flat_test=False)
    assert len(species) == 1
    assert abs(species[0].mass - 6578.66) < 1e-6
