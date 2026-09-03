"""End-to-end tests: synthesize a run, process it, and check the outputs."""

import numpy as np
import pytest

from lcmsdeconv.core.method import Method
from lcmsdeconv.deconv.classical import CombPredictor, make_predictor
from lcmsdeconv.deconv.nnls_solve import weighted_nnls
from lcmsdeconv.process import process_run
from lcmsdeconv.synth.chromatography import generate_run
from lcmsdeconv.synth.compounds import ClassConfig
from lcmsdeconv.synth.config import SynthConfig


@pytest.fixture(scope="module")
def small_run():
    cfg = SynthConfig(classes=[ClassConfig("peptide", (12000.0, 20000.0))], mode="rplc",
                      polarity=1, adduct_max_lambda=0.03)
    run, truth = generate_run(cfg, np.random.default_rng(5), n_peaks=1, rt_range=(0.5, 1.3),
                              scan_rate_hz=1.0)
    return run, truth


def test_nnls_normal_equations_matches_direct():
    from scipy.optimize import nnls

    rng = np.random.default_rng(0)
    A = np.abs(rng.random((3000, 40)))
    b = A @ np.abs(rng.random(40))
    x1 = weighted_nnls(A, b)
    x2, _ = nnls(A, b, maxiter=400)
    assert np.abs(x1 - x2).max() < 1e-6


def test_nnls_respects_weights_and_bounds():
    rng = np.random.default_rng(1)
    A = np.abs(rng.random((500, 5)))
    b = A @ np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    x = weighted_nnls(A, b, w=np.ones(500))
    assert (x >= 0).all()
    assert abs(x[0] - 1.0) < 0.05 and abs(x[4] - 3.0) < 0.05


def test_comb_predictor_finds_charges():
    from lcmsdeconv.chem.adducts import AdductLibrary, AdductState
    from lcmsdeconv.chem.classes import class_isotope_pattern
    from lcmsdeconv.chem.instrument import InstrumentModel, estimate_noise_sigma
    from lcmsdeconv.nn.grid import LogMzGrid
    from lcmsdeconv.synth.render import ComponentInstance, Scene, build_sticks, render_profile
    from lcmsdeconv.synth.spec import Compound

    mass = 20000.0
    pat = class_isotope_pattern(mass, "peptide")
    comp = Compound(mass, "peptide", pat.shifted(mass - pat.average_mass), name="p")
    charges = {z: 1.0 / 11 for z in range(15, 26)}
    inst = InstrumentModel("tof", 30000)
    ci = ComponentInstance(comp, 1e6, charges, {z: {AdductState(): 1.0} for z in charges})
    sticks = build_sticks(Scene([ci], 1, AdductLibrary.from_mode("rplc", 1)))
    ax = inst.profile_axis(600, 2000)
    prof = render_profile(sticks, ax, inst)
    g = LogMzGrid(500.0, 2500.0, 2e-5, 1)
    obs = g.resample_profile(ax, prof)
    pred = CombPredictor(z_max=40).predict_grid(g, obs, max(estimate_noise_sigma(obs), 1e-6))
    # at the apex of each charge state's envelope the comb should name that charge
    hits = 0
    for z in charges:
        b = g.mass_charge_to_bin(mass, z)
        window = pred.top1_z[max(0, b - 30):b + 30]
        if window.size and z in set(window.tolist()):
            hits += 1
    assert hits >= 6, f"comb identified {hits} of {len(charges)} charge states"


def test_make_predictor_selects_comb():
    assert isinstance(make_predictor("comb"), CombPredictor)
    assert isinstance(make_predictor("none"), CombPredictor)


def test_process_run_end_to_end(small_run):
    run, truth = small_run
    method = Method.load("rplc_pos_protein")
    method.deconvolution.max_components = 12
    result = process_run(run, method, model_path="comb")

    assert result.species, "no species found"
    assert result.peak_tables, "no signal was integrated"
    assert any(k.startswith("TIC") for k in result.peak_tables)
    assert result.impurities
    assert result.purity.get("rows")

    true_mass = truth.peaks[0].compound.mass
    best = min(result.species, key=lambda s: abs(s.mass - true_mass))
    # candidate identification under adduction is approximate; the mass must at least be the
    # right species rather than a harmonic or an unrelated envelope
    assert abs(best.mass - true_mass) / true_mass < 0.02
    assert 0.4 < best.rt_apex / truth.peaks[0].rt < 1.6

    summary = result.summary()
    assert summary["n_species"] == len(result.species)
    assert "saturation" in summary


def test_report_and_csv_written(tmp_path, small_run):
    import pandas as pd

    from lcmsdeconv.cli import _write_csvs
    from lcmsdeconv.io.results import save_json
    from lcmsdeconv.report.html import build_report

    run, _ = small_run
    method = Method.load("rplc_pos_protein")
    method.deconvolution.max_components = 8
    result = process_run(run, method, model_path="comb")

    save_json(result.summary(), tmp_path / "results.json")
    _write_csvs(result, tmp_path)
    path = build_report(result, tmp_path / "report.html")
    html = path.read_text()
    assert "<title>" in html and "Peak table" in html
    assert (tmp_path / "species.csv").exists()
    df = pd.read_csv(tmp_path / "species.csv")
    assert {"mass", "percent", "annotation"} <= set(df.columns)
