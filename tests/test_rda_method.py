"""The bundled method for the Waters RDa (BioAccord): the time-of-flight oligonucleotide method
with the RDa's resolving power, its lockspray's lock mass and the tripropylamine adducts."""

import numpy as np
import pytest

from lcmsdeconv.chem.adducts import AdductLibrary
from lcmsdeconv.core.method import Method, bundled_methods


def test_the_rda_method():
    assert "rda_iprp_neg_oligo" in bundled_methods()
    rda, tof = Method.load("rda_iprp_neg_oligo"), Method.load("iprp_neg_oligo")
    inst = rda.instrument
    assert (inst.kind, inst.resolution, inst.mz_ref, inst.mz_exponent) == ("tof", 15000.0,
                                                                           800.0, 0.0)
    assert inst.resolving_power(554.0) == pytest.approx(15000.0)
    assert rda.acquisition.lock_mass_mz == 554.2615 and rda.acquisition.polarity == "negative"
    assert rda.deconvolution.adducts.mode == "iprp"
    assert rda.deconvolution.adducts.include == ["TPAO"]
    # TPAO is for tripropylamine: the description says to remove it for TEA/HFIP
    assert "tripropylamine/HFIP" in rda.description
    assert "for TEA/HFIP remove TPAO from deconvolution.adducts.include" in rda.description
    assert rda.regions.rt_window is None
    # tripropylamine N-oxide joins the ion-pairing library by name; the amine is registered too,
    # for methods that list it
    lib = AdductLibrary.from_mode("iprp", -1, include=rda.deconvolution.adducts.include)
    masses = {a.name: a.mass for a in lib.adducts}
    assert masses["TPAO"] == pytest.approx(159.1623, abs=1e-3) and "TPA" not in masses
    tpa = {a.name: a.mass for a in AdductLibrary.from_mode("iprp", -1, include=["TPA"]).adducts}
    assert tpa["TPA"] == pytest.approx(143.1674, abs=1e-3)
    # otherwise the time-of-flight oligonucleotide method
    a, b = rda.to_dict(), tof.to_dict()
    for d in (a, b):
        for key in ("name", "description", "instrument"):
            d.pop(key)
        d["acquisition"].pop("lock_mass_mz")
        d["deconvolution"]["adducts"].pop("include")
    assert a == b


def test_the_rda_method_corrects_m_z_with_the_lockspray(monkeypatch, tmp_path):
    """A Waters run read +30 ppm high at its start and +20 ppm at its end, as its lockspray
    shows: processed with the RDa method, every scan's m/z is scaled by one factor (the scans
    keep a shared axis), the ratio the lockspray reads at the TIC-weighted mean time of the MS
    scans, where the error falls under a ppm; the drift leaves about 5 ppm at the run's ends."""
    from lcmsdeconv.io.mzml import read_mzml

    from .test_io_mzmlb import _write
    from .test_lockmass import (
        _error_at,
        _errors_ppm,
        _one_factor,
        _processed,
        _tic_weighted_rt,
        _waters_scans,
        offset_ppm,
    )

    run = read_mzml(_write(tmp_path / "rda.mzMLb", _waters_scans(), chromatograms=False))
    result, seen = _processed(monkeypatch, run, Method.load("rda_iprp_neg_oligo"))
    when = _tic_weighted_rt(run)
    factor = _one_factor(seen[0], run)
    assert abs(_error_at(seen[0], when)) < 1.0
    assert np.abs(_errors_ppm(seen[0])).max() < 6.0
    assert result.lock_mass["applied_ppm"] == pytest.approx((factor - 1.0) * 1e6, abs=0.01)
    assert result.lock_mass["applied_ppm"] == pytest.approx(-offset_ppm(when), abs=0.3)
    assert result.lock_mass["drift_ppm"] == pytest.approx(10.0, abs=0.5)
