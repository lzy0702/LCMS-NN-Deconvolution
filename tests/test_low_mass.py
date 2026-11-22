"""Species below 1.2 kDa, seen within m/z 400-2500 only as singly and doubly charged ions."""

import numpy as np

from lcmsdeconv.chem.classes import class_isotope_pattern
from lcmsdeconv.chem.instrument import InstrumentModel
from lcmsdeconv.core.model import Spectrum
from lcmsdeconv.deconv.decode import singly_charged_contradicted

MP = 1.00727646688
ORBI = InstrumentModel("orbitrap", 140000.0, mz_ref=200.0)
TRIMER, NC = 915.80, 5162.25


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

    x, y = _profile([(1500.4, 1, 2e6), (NC, 3, 4e6), (NC, 4, 5e6), (NC, 5, 1e6),
                     (2 * NC, 5, 6e5), (2 * NC, 6, 2e5), (2 * NC, 7, 1e5)], noise=300.0)
    keep = y > 1500.0
    spec = Spectrum(x[keep], y[keep], rt=18.2, polarity=-1, is_profile=True)

    def dimer_read_at_z1(found):
        return any(2060.0 < m < 2070.0 for m in found)

    found = _deconvolve(spec, _params())
    assert any(abs(m - NC) < 0.3 for m in found)
    assert not any(abs(m - 1500.4) < 2.0 for m in found) and not dimer_read_at_z1(found)
    # where singly charged species count, the dimer's z = 5 read at z = 1 is one
    assert dimer_read_at_z1(_deconvolve(spec, _params(singly_charged_species=True)))
    # and so it is, beside NC's z = 5 read at z = 2, without the test of what lies between the
    # isotopologues of a z = 1 state
    monkeypatch.setattr(decode, "singly_charged_contradicted", lambda *a, **k: False)
    assert dimer_read_at_z1(_deconvolve(spec, _params()))
