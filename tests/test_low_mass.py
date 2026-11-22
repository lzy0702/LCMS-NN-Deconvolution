"""Species below 1.2 kDa, seen within m/z 400-2500 only as singly and doubly charged ions."""

import numpy as np

from lcmsdeconv.chem.classes import class_isotope_pattern
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core.model import Spectrum
from lcmsdeconv.deconv.decode import singly_charged_contradicted

MP = 1.00727646688
ORBI = InstrumentModel("orbitrap", 140000.0, mz_ref=200.0)
TRIMER, NC = 915.80, 5162.25
SINGLE = 1802.5  # an ion seen at z = 1 alone


def _profile(envelopes, lo=400.0, hi=2500.0, step=0.001, noise=0.0, seed=0):
    """Resolved isotopologue envelopes [(mass, z, height)] on one profile axis."""
    x = np.arange(lo, hi, step)
    y = np.abs(np.random.default_rng(seed).normal(0.0, noise, x.size)) if noise else np.zeros_like(x)
    for mass, z, h in envelopes:
        pat = class_isotope_pattern(mass, "ps_rna")
        mz = (pat.masses - pat.average_mass + mass - z * MP) / z
        sig = ORBI.fwhm(float(mz.mean())) / 2.3548
        for c, v in zip(mz, pat.abundances / pat.abundances.max()):
            if v > 1e-4:
                a, b = np.searchsorted(x, [c - 5 * sig, c + 5 * sig])
                y[a:b] += h * v * np.exp(-0.5 * ((x[a:b] - c) / sig) ** 2)
    return x, y


def test_an_envelope_of_higher_charge_is_not_singly_charged():
    """Every envelope has an isotopologue each 1.0034 Th, so it fills a singly charged comb;
    only a singly charged one has nothing between those positions. NC's non-covalent dimer
    [2M-5H]5- of the IP-RPLC NC run, read at z = 1, was a 2065 Da species at 0.8 % of NC."""
    for mass, z, contradicted in ((TRIMER, 1, False), (2240.3, 1, False), (4128.6, 2, True),
                                  (6193.3, 3, True), (2 * NC, 5, True)):
        centre = (mass - z * MP) / z
        x, y = _profile([(mass, z, 1e6)], lo=centre - 5.0, hi=centre + 5.0)
        assert singly_charged_contradicted(centre, 1.5, x, y, 1e3, float(y.max())) == contradicted


def _params(**kw):
    from lcmsdeconv.core.method import Method
    from lcmsdeconv.process import _deconv_params

    params = _deconv_params(Method.load("orbitrap_iprp_neg_oligo"))
    for k, v in kw.items():
        setattr(params, k, v)
    return params


def _deconvolve(spectrum, params):
    from lcmsdeconv.deconv.classical import make_predictor
    from lcmsdeconv.deconv.pipeline import deconvolve_spectrum

    pred = make_predictor(None, z_max=40, instrument=ORBI)
    return [c.mass for c in deconvolve_spectrum(spectrum, pred, params, ORBI).components]


def test_a_trimer_is_found_on_its_singly_and_doubly_charged_ions():
    """The oligonucleotide method reaches down to 800 Da: below 1.2 kDa a species' z = 3 falls
    below m/z 400, and z = 1 counts as one of its two charge states (the IP-RPLC and IP-HILIC
    NC runs hold a 915 Da trimer at 0.4 and 1.1 % of NC)."""
    x, y = _profile([(TRIMER, 1, 2e6), (TRIMER, 2, 2e6)], noise=300.0)
    keep = y > 1500.0
    spec = Spectrum(x[keep], y[keep], rt=2.48, polarity=-1, is_profile=True)
    params = _params()
    assert params.mass_range[0] <= 800.0 and params.charge_range[0] == 1
    assert params.singly_charged_species is False
    assert any(abs(m - TRIMER) < 0.3 for m in _deconvolve(spec, params))
    # the method before: from 1 kDa, at charges 2 and up
    old = _params(mass_range=(1000.0, 60000.0), charge_range=(2, 40))
    assert not any(abs(m - TRIMER) < 2.0 for m in _deconvolve(spec, old))


def test_singly_charged_ions_alone_are_no_oligonucleotide(monkeypatch):
    """A species of the oligonucleotide method needs two charge states, z = 1 among them only
    where its envelope is singly charged: a singly charged ion alone, or a higher charge read
    at z = 1 (NC's dimer [2M-5H]5- at m/z 2064, beside NC's own z = 5 read at z = 2), is no
    species."""
    from lcmsdeconv.deconv import decode

    x, y = _profile([(SINGLE, 1, 2e6), (NC, 3, 4e6), (NC, 4, 5e6), (NC, 5, 1e6),
                     (2 * NC, 5, 6e5), (2 * NC, 6, 2e5), (2 * NC, 7, 1e5)], noise=300.0)
    keep = y > 1500.0
    spec = Spectrum(x[keep], y[keep], rt=18.2, polarity=-1, is_profile=True)

    def dimer_read_at_z1(found):
        return any(2060.0 < m < 2070.0 for m in found)

    def single(found):
        return any(abs(m - SINGLE) < 2.0 for m in found)

    found = _deconvolve(spec, _params())
    assert any(abs(m - NC) < 0.3 for m in found)
    assert not single(found) and not dimer_read_at_z1(found)
    # where singly charged species count, the ion alone is one (read 0.8 Da light), and so is the
    # dimer's z = 5 read at z = 1
    old = _deconvolve(spec, _params(singly_charged_species=True))
    assert single(old) and dimer_read_at_z1(old)
    # and so it is, beside NC's z = 5 read at z = 2, without the test of what lies between the
    # isotopologues of a z = 1 state
    monkeypatch.setattr(decode, "singly_charged_contradicted", lambda *a, **k: False)
    assert dimer_read_at_z1(_deconvolve(spec, _params()))


def test_a_mass_voted_for_by_z1_alone_is_picked_only_where_singly_charged_species_count():
    """The mass histogram's peak that z = 1 alone voted for is no candidate of the
    oligonucleotide method; beside z = 2 it is."""
    from lcmsdeconv.deconv.decode import ChargeSupport, pick_candidates

    lm = np.log(800.0) + np.arange(40000) * 5e-6
    hist = np.zeros(lm.size)
    p = 20000
    hist[p - 3:p + 4] = [1.0, 3.0, 6.0, 10.0, 6.0, 3.0, 1.0]

    def picked(zs, ok):
        support = ChargeSupport(np.full(len(zs), p), np.array(zs, dtype=np.int32), lm.size)
        return [sorted(c.charges) for c in pick_candidates(lm, hist, support,
                                                           singly_charged_ok=ok)]

    assert picked([1], False) == [] and picked([1], True) == [[1]]
    assert picked([1, 2], False) == [[1, 2]]
    assert picked([2], True) == []  # one charge state of two or more is no candidate either way


def test_the_gate_takes_z1_alone_only_where_singly_charged_species_count():
    """A candidate whose only envelope is its z = 1 is unsupported in the oligonucleotide method,
    like one seen at any other single charge state."""
    from lcmsdeconv.deconv.decode import Candidate, refine_candidate_masses

    x, y = _profile([(TRIMER, 1, 2e6)])
    for ok in (False, True):
        out = refine_candidate_masses([Candidate(TRIMER, 1.0, {1})], x, y, -1, noise_sigma=100.0,
                                      compound_class="ps_rna", resolution=140000.0,
                                      instrument=ORBI, profile=True, singly_charged_ok=ok)
        assert [sorted(c.charges) for c in out] == ([[1]] if ok else [])


def _z1_window(envelopes):
    """The arguments ``singly_charged_contradicted`` gets for the trimer's z = 1 window."""
    x, y = _profile(envelopes)
    centre = TRIMER - MP
    lo, hi = np.searchsorted(x, [centre - 1.5, centre + 1.5])
    return centre, 1.5, x, y, 3e3, float(y[lo:hi].max())


def test_z1_joins_a_species_only_where_its_envelope_is_singly_charged():
    """``observed_charges`` grows a species' charge states from its z = 2 down to z = 1 where an
    envelope sits there; without singly charged species that envelope must have no peaks between
    its isotopologues. A dimer's z = 2 on the trimer's z = 1 position is no z = 1 of the
    trimer."""
    from lcmsdeconv.deconv.refine import observed_charges

    def grown(envelopes, ok):
        x, y = _profile(envelopes)
        return observed_charges(TRIMER, {2}, x, y, 3e3, -1, "ps_rna", 140000.0, (400.0, 2500.0),
                                instrument=ORBI, singly_charged_ok=ok)

    own = [(TRIMER, 2, 2e6), (TRIMER, 1, 2e6)]
    dimer = [(TRIMER, 2, 2e6), (2 * TRIMER, 2, 2e6)]  # the dimer's z = 2 on m/z 914.8
    assert singly_charged_contradicted(*_z1_window(dimer)) is True
    assert singly_charged_contradicted(*_z1_window(own)) is False
    assert grown(own, False) == {1, 2} and grown(own, True) == {1, 2}
    assert grown(dimer, False) == {2}
    assert grown(dimer, True) == {1, 2}  # any envelope fills a z = 1 comb


def test_the_fit_needs_more_than_z1_unless_singly_charged_species_count():
    """A candidate whose fit holds z = 1 alone (its z = 2 explains nothing) is no species of the
    oligonucleotide method; where singly charged species count it is one."""
    from lcmsdeconv.chem.adducts import AdductLibrary
    from lcmsdeconv.deconv.decode import Candidate
    from lcmsdeconv.deconv.refine import refine_frame
    from lcmsdeconv.deconv.templates import build_template
    from lcmsdeconv.nn.grid import LogMzGrid

    grid = LogMzGrid(400.0, 2500.0, 2e-5, -1)
    spectrum = np.zeros(grid.size)
    t = build_template(TRIMER, 1, grid, ORBI, "ps_rna")
    spectrum[t.bins] += 1e6 * t.values
    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP", "K", "Na"),
                                  max_per_type=2, max_total=2)
    for ok, outcome in ((False, "charge_support"), (True, "accepted")):
        trace = []
        cand = Candidate(TRIMER, 1.0, {1, 2}, gate_charges={1, 2}, gate_failed=set())
        comps, _f, _r = refine_frame([cand], grid, spectrum.copy(), 100.0, ORBI, "ps_rna", lib,
                                     singly_charged_ok=ok, trace=trace)
        assert [r["outcome"] for r in trace if r.get("stage") == "fit"] == [outcome]
        assert [sorted(c.charges) for c in comps] == ([[1]] if ok else [])
