"""Diagnostics, scoring and the generator fixes they exposed."""

import subprocess
import sys

import numpy as np
import pytest

from lcmsdeconv.diagnostics import (
    STAGES,
    attribute_losses,
    classify_false_species,
    confuse_charges,
    match_species,
    oracle_predictor,
    truth_components,
)
from lcmsdeconv.evaluate import score_species
from lcmsdeconv.synth.chromatography import emg, generate_run
from lcmsdeconv.synth.compounds import ClassConfig
from lcmsdeconv.synth.config import SynthConfig
from lcmsdeconv.synth.spec import Sticks


def test_emg_survives_short_tails():
    """A tail much shorter than the peak width must give the Gaussian, not zeros."""
    t = np.linspace(2.0, 3.0, 201)
    for tau in (0.0, 1e-9, 4.6e-4, 2e-3):
        y = emg(t, 2.5, 0.08, tau)
        assert np.isfinite(y).all()
        assert abs(y.max() - 1.0) < 0.02, tau
        g = np.exp(-0.5 * ((t - 2.5) / 0.08) ** 2)
        assert np.abs(y - g).max() < 0.05, tau


def test_emg_matches_textbook_form_where_it_is_finite():
    from scipy.special import erfc

    t = np.linspace(2.0, 3.5, 400)
    c, s, tau = 2.5, 0.08, 0.05
    ref = np.exp(0.5 * (s / tau) ** 2 - (t - c) / tau) * erfc((s / tau - (t - c) / s) / np.sqrt(2))
    ref /= ref.max()
    assert np.abs(emg(t, c, s, tau) - ref).max() < 1e-3


def test_inference_path_does_not_import_torch():
    code = ("import sys; sys.modules['torch'] = None\n"
            "from lcmsdeconv.process import process_run\n"
            "from lcmsdeconv.deconv.classical import make_predictor\n"
            "from lcmsdeconv.nn.infer import ChargePredictor\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_impurities_share_their_parents_charge_envelope():
    from lcmsdeconv.synth.charge import (
        distribution_from_shape,
        mean_charge,
        related_shape,
        sample_charge_shape,
    )

    rng = np.random.default_rng(3)
    for _ in range(20):
        parent = sample_charge_shape(30000.0, "peptide", "denatured", rng)
        child = related_shape(parent, 30000.0, 30016.0, "peptide", rng)
        zp = mean_charge(distribution_from_shape(parent))
        zc = mean_charge(distribution_from_shape(child))
        assert abs(zc - zp) / zp < 0.05


def test_per_ion_adduct_load_keeps_the_base_form_visible():
    """At a load of mu adducts per ion the base form keeps ~exp(-sum mu) of the apex charge."""
    from lcmsdeconv.chem.adducts import AdductLibrary, AdductState
    from lcmsdeconv.synth.adduct_sampler import (
        adduct_fractions_for_charge,
        propensities_for_component,
    )

    lib = AdductLibrary.from_mode("rplc", 1, max_per_type=3, max_total=4)
    rng = np.random.default_rng(0)
    load = {"Na": 0.3, "K": 0.3, "NH4": 0.3}
    for zc in (10, 40, 60):
        charges = {z: 1.0 for z in range(zc - 3, zc + 4)}
        props = propensities_for_component(load, charges)
        base = adduct_fractions_for_charge(zc, props, lib, rng)[AdductState()]
        assert 0.3 < base < 0.5, (zc, base)  # exp(-0.9) = 0.41


def test_generator_records_detectability():
    cfg = SynthConfig(classes=[ClassConfig("peptide", (12000.0, 20000.0))], mode="rplc")
    _run, truth = generate_run(cfg, np.random.default_rng(1), n_peaks=1, rt_range=(0.5, 1.5),
                               scan_rate_hz=1.0)
    members = truth.to_dict()["peaks"][0]["members"]
    assert all("support_counts" in m and m["support_counts"] >= 0 for m in members)
    main = members[0]
    assert main["support_counts"] > max(m["support_counts"] for m in members[1:] or [main]) * 0.99


def test_confused_charges_relabel_whole_envelopes():
    z = np.array([10] * 5 + [11] * 5 + [12] * 5)
    sticks = Sticks(np.linspace(500, 900, 15), np.ones(15), np.zeros(15, dtype=int), z.copy(),
                    np.zeros(15))
    same = confuse_charges(sticks, 1.0, np.random.default_rng(0))
    assert (same.z == z).all()
    wrong = confuse_charges(sticks, 0.0, np.random.default_rng(0), kernel="harmonic")
    for old, block in ((10, wrong.z[:5]), (11, wrong.z[5:10]), (12, wrong.z[10:])):
        assert len(set(block.tolist())) == 1  # an envelope is relabelled as a whole
        assert block[0] in {2 * old, old // 2, old + 1, old - 1}
        assert block[0] != old


def test_matching_is_one_to_one():
    truths = [{"mass": 10000.0, "rt": 1.0}, {"mass": 10000.3, "rt": 1.0}]
    species = [{"mass": 10000.1, "rt_apex": 1.0}]
    m = match_species(species, truths)
    assert len(m) == 1 and m[0].truth_index == 0
    truth = {"peaks": [{"rt": 1.0, "members": [
        {"mass": 10000.0, "rel": 1.0, "name": "a", "kind": "main"},
        {"mass": 10000.3, "rel": 0.05, "name": "b", "kind": "impurity"}]}]}
    sc = score_species(species, truth)
    assert sc["n_matched"] == 1 and sc["precision"] == 1.0
    found = sum(round(t["recall"] * t["n_true"]) for t in sc["tiers"].values())
    assert found == 1  # one reported species cannot account for two true ones


def test_matching_serves_the_more_abundant_truth_first():
    """An unresolvable impurity one dalton away must not take the main component's report."""
    truths = [{"mass": 74757.5, "rt": 1.0, "rel": 0.99}, {"mass": 74758.5, "rt": 1.0, "rel": 0.003}]
    species = [{"mass": 74758.4, "rt_apex": 1.0}]
    m = match_species(species, truths)
    assert len(m) == 1 and m[0].truth_index == 0


def test_false_species_are_classified():
    truths = [{"mass": 20000.0, "rt": 1.0, "name": "x"}]
    species = [{"mass": 10000.0}, {"mass": 20021.98}, {"mass": 20030.0}, {"mass": 5555.0}]
    kinds = [f["kind"] for f in classify_false_species(species, truths, set(), {"Na": 21.98194})]
    assert kinds == ["harmonic", "adduct", "mass_error", "unexplained"]


@pytest.fixture(scope="module")
def traced_run():
    from lcmsdeconv.core.method import Method
    from lcmsdeconv.process import process_run

    cfg = SynthConfig(classes=[ClassConfig("peptide", (12000.0, 20000.0))], mode="rplc",
                      polarity=1)
    run, truth = generate_run(cfg, np.random.default_rng(5), n_peaks=1, rt_range=(0.5, 1.3),
                              scan_rate_hz=1.0)
    method = Method.load("rplc_pos_protein")
    trace: list = []
    result = process_run(run, method, predictor=oracle_predictor(truth, 1), trace=trace)
    return result, truth, trace


def test_trace_records_every_stage(traced_run):
    result, truth, trace = traced_run
    assert trace and trace[0]["stages"]
    stages = {s["stage"] for s in trace[0]["stages"]}
    assert {"votes", "picked", "gated", "detected", "fit", "components"} <= stages
    assert set(stages) - {"not_yet_found", "pre_merge", "adducts"} <= set(STAGES)


def test_loss_attribution_names_a_stage_for_every_miss(traced_run):
    result, truth, trace = traced_run
    species = [s.to_dict() for s in result.species]
    recs = attribute_losses(trace, truth.to_dict(), species)
    assert len(recs) == len(truth_components(truth.to_dict()))
    main = recs[0]
    assert main["lost_at"] is None, main  # an oracle-charged main species is found
    for r in recs:
        assert r["lost_at"] in (None, "no_region", *STAGES)
