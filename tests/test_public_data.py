"""Sanity checks on public spectra (skipped unless scripts/fetch_public_data.py has run).

These are agreement checks on real measurements, not training data: the deconvolved masses of
well-known standards must land where the literature says they do.
"""

from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data" / "public"

pytestmark = pytest.mark.skipif(
    not (DATA / "unidec" / "BSA.txt").exists(),
    reason="run scripts/fetch_public_data.py to enable public-data checks",
)


def _deconvolve_text(path, expected_mass, tol_rel, charge_range=(5, 60), mz_range=None):
    from lcmsdeconv.chem.instrument import InstrumentModel
    from lcmsdeconv.deconv.oracle import OraclePredictor
    from lcmsdeconv.deconv.pipeline import DeconvParams, deconvolve_spectrum
    from lcmsdeconv.io.text import read_text_spectrum
    from lcmsdeconv.nn.grid import LogMzGrid
    from lcmsdeconv.synth.spec import Sticks

    spec = read_text_spectrum(path)
    if mz_range:
        spec = spec.sliced(*mz_range)
    inst = InstrumentModel("tof", 12000.0)
    grid = LogMzGrid(50.0, 10000.0, 2e-5, 1)
    # charge hypotheses come from the expected mass: this checks the fitting chain, not the net
    sticks = Sticks(
        mz=np.array([(expected_mass + z * 1.00728) / z for z in range(*charge_range)]),
        intensity=np.ones(charge_range[1] - charge_range[0]),
        comp_id=np.zeros(charge_range[1] - charge_range[0], dtype=int),
        z=np.arange(*charge_range),
        adduct_mass=np.zeros(charge_range[1] - charge_range[0]),
    )
    params = DeconvParams(compound_class="peptide", adduct_mode="native",
                          mass_range=(5000.0, 800000.0), charge_range=charge_range,
                          adduct_max_total=1, min_charge_support=3)
    fr = deconvolve_spectrum(spec, OraclePredictor(sticks, grid, window_bins=200), params, inst)
    assert fr.components, f"no components found in {path.name}"
    top = max(fr.components, key=lambda c: c.intensity)
    rel = abs(top.mass - expected_mass) / expected_mass
    assert rel < tol_rel, f"{path.name}: got {top.mass:.1f}, expected {expected_mass} ({rel*100:.2f} %)"
    return top


def test_bsa_monomer_mass():
    # Bovine serum albumin, average mass about 66.4 kDa (native spectra carry salt adducts,
    # so the measured mass runs slightly high).
    top = _deconvolve_text(DATA / "unidec" / "BSA.txt", 66430.0, 0.01, (10, 30))
    assert top.mass_spread_ppm < 3000


def test_adh_tetramer_mass():
    # Yeast alcohol dehydrogenase tetramer, about 147.6 kDa.
    _deconvolve_text(DATA / "unidec" / "ADHclean.txt", 147600.0, 0.01, (20, 40))


def test_orbitrap_mzml_parses_and_has_resolved_isotopes():
    from lcmsdeconv.chem.instrument import centroid_profile
    from lcmsdeconv.io.mzml import read_mzml

    run = read_mzml(DATA / "ms_deisotope" / "three_test_scans.mzML")
    assert run.spectra
    s = max(run.spectra, key=lambda x: x.intensity.size)
    mz, apex, fwhm = centroid_profile(s.mz, s.intensity)
    assert mz.size > 20
    # isotope spacing of a multiply charged peptide is a small fraction of 1 Da
    d = np.diff(np.sort(mz))
    d = d[(d > 0.005) & (d < 1.05)]
    assert d.size > 5
