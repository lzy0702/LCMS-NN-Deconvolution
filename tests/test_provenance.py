"""Provenance (``process_run(..., provenance=True)``, ``lcmsdeconv process --provenance``): where
each species was read from, written as provenance.json, provenance.html and evidence figures.

Every number and name here is made up: 123456 plays the project number to hide.
"""

import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import lcmsdeconv
from lcmsdeconv.aliases import ENV_VAR, Aliases
from lcmsdeconv.cli import _check_outputs, main
from lcmsdeconv.core.method import Method
from lcmsdeconv.core.model import Species
from lcmsdeconv.features.link import (
    fold_adduct_forms,
    merge_adjacent_species,
    merge_species_across_polarity,
    set_aside_later_elutions,
)
from lcmsdeconv.io.mzml_writer import write_mzml
from lcmsdeconv.io.results import write_outputs
from lcmsdeconv.process import process_run
from lcmsdeconv.provenance import (
    MAX_FIGURES,
    MERGE_STEPS,
    SCHEMA,
    SCHEMA_VERSION,
    figure_name,
    write_provenance,
)
from lcmsdeconv.synth.chromatography import generate_run
from lcmsdeconv.synth.compounds import ClassConfig
from lcmsdeconv.synth.config import SynthConfig

SECRET = "123456"
NEW = "保密项目1"
PNG = b"\x89PNG\r\n\x1a\n"
MODEL = Path(lcmsdeconv.__file__).parent / "models" / "charge_unet_proof.onnx"


def _run():
    """Two 12-20 kDa peptides over 72 scans: two regions, species linked in pieces that the
    processing joins (neighbouring pieces, an adduct form reported apart)."""
    cfg = SynthConfig(classes=[ClassConfig("peptide", (12000.0, 20000.0))], mode="rplc",
                      polarity=1)
    run, _truth = generate_run(cfg, np.random.default_rng(2), n_peaks=2, rt_range=(0.5, 1.7),
                               scan_rate_hz=1.0)
    return run


def _method():
    method = Method.load("rplc_pos_protein")
    method.deconvolution.max_components = 12
    method.deconvolution.model = "comb"
    return method


@pytest.fixture(scope="module")
def run():
    return _run()


@pytest.fixture(scope="module")
def processed(run):
    return process_run(run, _method(), provenance=True)


@pytest.fixture(scope="module")
def written(processed, tmp_path_factory):
    out = tmp_path_factory.mktemp("provenance")
    files = write_outputs(processed, out)
    data = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    return out, files, data


def _stable(summary: dict) -> str:
    return json.dumps(summary | {"timings": {}}, sort_keys=True)


# ---------------------------------------------------------------------- the flag
#: the hooks provenance reads, by the function given them: (module, function, keyword)
HOOKS = [("lcmsdeconv.process", "deconvolve_spectrum", "keep"),
         ("lcmsdeconv.process", "link_species", "members"),
         ("lcmsdeconv.process", "fold_adduct_forms", "log"),
         ("lcmsdeconv.process", "merge_adjacent_species", "log"),
         ("lcmsdeconv.process", "merge_boundary_pieces", "log"),
         ("lcmsdeconv.process", "set_aside_later_elutions", "log"),
         ("lcmsdeconv.process", "merge_species_across_polarity", "log"),
         ("lcmsdeconv.deconv.pipeline", "drop_multimers", "dropped"),
         ("lcmsdeconv.deconv.pipeline", "drop_harmonic_readings", "dropped"),
         ("lcmsdeconv.deconv.pipeline", "drop_shared_evidence", "dropped"),
         ("lcmsdeconv.deconv.pipeline", "drop_coincident_adducts", "dropped"),
         ("lcmsdeconv.deconv.pipeline", "refine_frame", "parts"),
         ("lcmsdeconv.deconv.refine", "_merge_same_mass", "parts"),
         ("lcmsdeconv.deconv.refine", "_merge_adduct_components", "parts")]


def _switching_run():
    """Small molecules in a polarity-switching run: both polarities' regions."""
    cfg = SynthConfig(classes=[ClassConfig("small_molecule", (300.0, 900.0))], mode="rplc",
                      polarity=1)
    run, _truth = generate_run(cfg, np.random.default_rng(1), n_peaks=3, rt_range=(0.5, 3.5),
                               scan_rate_hz=2.0, polarity_switching=True)
    return run


def test_without_the_flag_nothing_is_kept_or_written(run, tmp_path, monkeypatch):
    """No provenance is collected (no recorder; no grid view, residual or estimator output kept
    for it; no members, merge, drop or fitted-parts log) and none is written. Every step with a
    hook is run: the method sets later elutions aside, and a polarity-switching run joins its
    polarities."""
    import importlib

    import lcmsdeconv.provenance as provenance

    calls = []

    def watched(module, name, keyword):
        f = getattr(module, name)

        def call(*args, **kwargs):
            calls.append((name, kwargs.get(keyword)))
            return f(*args, **kwargs)

        return call

    for module, name, keyword in HOOKS:
        module = importlib.import_module(module)
        monkeypatch.setattr(module, name, watched(module, name, keyword))

    def refused(*args, **kwargs):
        raise AssertionError("a recorder was made without provenance")

    monkeypatch.setattr(provenance, "Recorder", refused)
    method = _method()
    method.linking.set_aside_later_elutions = True
    result = process_run(run, method)
    switching = Method.load("small_molecule_switching")
    switching.deconvolution.model = "comb"
    assert process_run(_switching_run(), switching).provenance is None
    assert result.provenance is None
    assert {name for name, _value in calls} == {name for _m, name, _k in HOOKS}
    assert [(name, value) for name, value in calls if value is not None] == []
    files = write_outputs(result, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in files)
    assert not [p for p in files if "provenance" in p.name or p.suffix == ".png"]
    assert "provenance" not in result.summary()
    with pytest.raises(ValueError, match="provenance=True"):
        write_outputs(result, tmp_path / "again", provenance=True)


def test_provenance_changes_no_other_output(run, processed, tmp_path):
    """Collecting provenance changes nothing the run reports: every other output is byte for
    byte the same as without it (but the processing time)."""
    plain = process_run(run, _method())
    assert _stable(plain.summary()) == _stable(processed.summary())
    write_outputs(plain, tmp_path / "plain")
    write_outputs(processed, tmp_path / "prov", provenance=False)
    names = sorted(p.name for p in (tmp_path / "plain").iterdir())
    assert names == sorted(p.name for p in (tmp_path / "prov").iterdir())
    for name in names:
        a = (tmp_path / "plain" / name).read_text(encoding="utf-8")
        b = (tmp_path / "prov" / name).read_text(encoding="utf-8")
        if name == "results.json":
            a, b = (json.dumps(json.loads(x) | {"timings": {}}) for x in (a, b))
        if name == "report.html":
            a, b = (re.sub(r"<b>Processing</b>[0-9.]+ s", "", x) for x in (a, b))
        assert a == b, name


# ---------------------------------------------------------------------- provenance.json
def test_schema_regions_and_estimator(written, processed):
    out, files, data = written
    assert (data["schema"], data["version"], data["lcmsdeconv"]) == (SCHEMA, SCHEMA_VERSION,
                                                                     lcmsdeconv.__version__)
    assert (data["run"], data["method"]) == (processed.run_name, processed.method.name)
    assert data["estimator"] == {"kind": "comb", "z_max": 60, "isotope_charges": False}
    assert set(data["merge_steps"]) == set(MERGE_STEPS)
    assert [p.name for p in files[:1]] == ["results.json"]
    assert [p.name for p in files if p.parent == out][-2:] == ["provenance.json",
                                                               "provenance.html"]
    assert len(data["regions"]) == len(processed.regions) == 2
    for rec, region in zip(data["regions"], processed.regions):
        assert (rec["id"], rec["polarity"], rec["start"], rec["end"]) == (
            region.id, region.polarity, region.start, region.end)
        assert rec["scans"]["ids"] == [f.scan_id for f in region.frames]
        assert rec["scans"]["n"] == len(region.frames)
        assert rec["deconvolution"]["engine"] == "grid"
        assert rec["deconvolution"]["components"] >= len(rec["species"]) > 0
    # the boundary the two regions share
    assert data["regions"][0]["shared_boundaries"]["end"]
    assert data["regions"][1]["shared_boundaries"]["start"]


def test_masses_agree_with_results_json(written):
    out, _files, data = written
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    keys = ("id", "name", "mass", "mass_type", "polarity", "rt_apex", "flags", "region_id")
    reported = [{k: s[k] for k in keys} for s in results["species"]]
    traced = [{k: s[k] for k in keys[:-1]} for s in data["species"]]
    assert traced == [{k: v for k, v in r.items() if k != "region_id"} for r in reported]
    for s, r in zip(data["species"], results["species"]):
        assert s["amount"]["base"] == r["base_intensity"]
        assert s["amount"]["all_forms"] == r["total_intensity"]
        assert {str(c["z"]) for c in s["charges"]} == set(r["charges"])
        assert abs(sum(c["share"] for c in s["charges"]) - 1.0) < 1e-9
        # the region results.json names is the one of the piece the species was linked as
        assert s["pieces"][0]["region"]["id"] == r["region_id"]
        # the mass is its piece's, read from its component(s)
        piece = s["pieces"][s["mass_from"]["piece"]]
        assert piece["mass"] == s["mass"]
        comps = piece["evidence"]["components"]
        assert piece["evidence"]["kind"] == "grid_components" and comps
        if len(comps) == 1:
            assert abs(comps[0]["mass"] - s["mass"]) <= 1e-9 * s["mass"]
        assert s["mass_from"]["how"]


def test_rt_ranges_and_scans_agree_with_the_regions_and_the_trace(written, processed):
    _out, _files, data = written
    regions = {(r["id"], r["polarity"]): r for r in data["regions"]}
    for s, species in zip(data["species"], processed.species):
        eic = s["eic"]
        assert eic["per_scan"]
        assert eic["rt"] == species.time.tolist() and eic["intensity"] == species.intensity.tolist()
        assert set(eic["forms"]) == set(species.adduct_intensity)
        mine = [regions[(r["id"], r["polarity"])] for r in s["regions"]]
        for entry, region in zip(s["regions"], mine):
            assert (entry["start"], entry["end"], entry["scans"]) == (
                region["start"], region["end"], region["scans"]["n"])
            assert (entry["first_scan"], entry["last_scan"]) == (region["scans"]["ids"][0],
                                                                 region["scans"]["ids"][-1])
            assert entry["shared_boundaries"] == region["shared_boundaries"]
        ids = {i for r in mine for i in r["scans"]["ids"]}
        assert None not in eic["scan_ids"] and set(eic["scan_ids"]) <= ids
        assert len(set(eic["scan_ids"])) == len(eic["scan_ids"])
        assert s["rt_range"] == [min(eic["rt"]), max(eic["rt"])]
        assert min(r["start"] for r in mine) <= s["rt_range"][0] <= s["rt_range"][1] <= max(
            r["end"] for r in mine)
        assert s["rt_range"][0] <= s["rt_signal"][0] <= s["rt_apex"] <= s["rt_signal"][1]
        for p in s["pieces"]:
            region = regions[(p["region"]["id"], p["region"]["polarity"])]
            assert region["start"] <= p["rt_range"][0] <= p["rt_range"][1] <= region["end"]
            assert p["scans"] == region["scans"]["n"]  # linked over the region's scans
            assert s["id"] in region["species"] or not p["counted"]


def test_every_charge_window_holds_observed_signal(written):
    _out, _files, data = written
    checked = 0
    for s in data["species"]:
        for p in s["pieces"]:
            for c in p["evidence"]["components"]:
                assert c["template_mass"] == pytest.approx(
                    c["mass"] * np.exp(c["template_shift"]), rel=1e-12)
                shares = [st["share"] for st in c["charge_states"]]
                assert abs(sum(shares) - 1.0) < 1e-9
                for st in c["charge_states"] + c["adduct_columns"]:
                    lo, hi = st["window_mz"]
                    assert lo < hi
                    if st["fitted"] <= 0:
                        continue  # a charge state fitted on its adduct forms only
                    checked += 1
                    seen = st["observed"]
                    assert seen["points"] > 0 and seen["sum"] > 0 and seen["max"] > 0
                    assert lo <= seen["centroid_mz"] <= hi
                    g = st["grid"]
                    assert g["observed"] > 0 and g["component_model"] > 0
                    assert g["model"] >= g["component_model"] * (1 - 1e-9)
                    assert g["unexplained"] >= 0 and g["overshoot"] >= 0
                    assert g["unexplained"] - g["overshoot"] == pytest.approx(
                        g["observed"] - g["model"], rel=1e-9, abs=1e-6 * g["observed"])
                    assert 0 <= g["fit_residual"] <= g["observed"] * (1 + 1e-9)
                    # the window lies within the whole support the fit read
                    s_lo, s_hi = st["support_mz"]
                    assert s_lo <= lo < hi <= s_hi
                for st in c["charge_states"]:
                    lo, hi = st["window_mz"]
                    assert lo <= st["mz_centroid"] <= hi  # the theoretical centroid
                    cm = st["charge_map"]
                    assert cm is not None and cm["pass"] == c["pass"]
                    if cm["bins"]:
                        assert 0.0 <= cm["p_assigned"] <= 1.0 and 0.0 <= cm["top1_share"] <= 1.0
                        assert cm["alternative"] is None or cm["alternative"]["z"] != st["z"]
    assert checked > 50


def test_the_fit_residual_and_the_dropped_columns_are_the_fits_own(run):
    """A window's fit_residual is what deconvolve_spectrum's fit left there (its residual,
    summed over the window's bins), read here by wrapping the fit itself; the columns fitted
    and then dropped after the fit are listed with their step and the component they were
    attributed to, as the trace records the drops."""
    import lcmsdeconv.deconv.pipeline as pipeline
    import lcmsdeconv.process as process

    residuals, fit, deconvolve = [], pipeline.refine_frame, process.deconvolve_spectrum

    def watched_fit(*args, **kwargs):
        out = fit(*args, **kwargs)
        residuals[-1] = np.array(out[2], dtype=float)  # the last pass's is the region's
        return out

    def watched_deconvolve(*args, **kwargs):
        residuals.append(None)
        return deconvolve(*args, **kwargs)

    trace = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "refine_frame", watched_fit)
        mp.setattr(process, "deconvolve_spectrum", watched_deconvolve)
        result = process_run(run, _method(), provenance=True, trace=trace)
    prov = result.provenance
    by_region = dict(zip([(int(r.polarity), int(r.id)) for r in result.regions], residuals))
    checked = 0
    for s in prov.species:
        for p in s["pieces"]:
            residual = by_region[(p["region"]["polarity"], p["region"]["id"])]
            for c in p["evidence"]["components"]:
                for st in c["charge_states"] + c["adduct_columns"]:
                    g = st["grid"]
                    b0, b1 = g["bins"]
                    assert g["fit_residual"] == pytest.approx(float(residual[b0:b1 + 1].sum()),
                                                              rel=1e-12, abs=1e-9)
                    checked += 1
    assert checked > 100
    # the drops, as the trace records them
    for r, region in zip(trace, (prov.regions[(int(r.polarity), int(r.id))]
                                 for r in result.regions)):
        traced = sorted((st["outcome"], round(st["candidate"], 6)) for st in r["stages"]
                        if st.get("stage") == "fit" and st.get("pass") == -1)
        dropped = region["deconvolution"]["dropped"]
        assert sorted((d["step"], round(d["mass"], 6)) for d in dropped
                      if d["component_dropped"]) == traced
        for d in dropped:
            assert d["step"] in MERGE_STEPS or d["step"] in __import__(
                "lcmsdeconv.provenance", fromlist=["DROP_STEPS"]).DROP_STEPS
            assert d["attributed_to"] and d["columns"]
            assert all(a["mass"] > 0 and (a["pass"], a["id"]) != (d["pass"], d["id"])
                       for a in d["attributed_to"])
    steps = {d["step"] for r in prov.regions.values() for d in r["deconvolution"]["dropped"]}
    assert {"harmonic_reading", "shared_evidence"} <= steps
    # where a dropped component's templates lie in a reported one's window, the window says so
    shares = [st["grid"]["dropped_model"] / st["grid"]["observed"]
              for s in prov.species for p in s["pieces"] for c in p["evidence"]["components"]
              for st in c["charge_states"] if st["grid"] and st["grid"]["observed"] > 0]
    assert max(shares) > 0.05


def test_a_figure_draws_the_species_other_pieces_of_its_region_as_its_own(run, monkeypatch):
    """A piece counted in the species from the region its figure shows (an adduct form folded
    in) is drawn as the species' (purple), its windows shaded, not among the other components
    (orange). Read on every grid bin (the span not reduced): over each of its charge states'
    windows the purple curve holds its fitted templates, and the three curves add up to every
    fitted component's."""
    import lcmsdeconv.provenance as provenance

    monkeypatch.setattr(provenance, "SPAN_POINTS", 10**9)  # every bin drawn
    prov = process_run(run, _method(), provenance=True).provenance
    found = 0
    for s, fig in zip(prov.species, prov.figures):
        primary = s["pieces"][s["mass_from"]["piece"]]
        own = {c["id"] for c in primary["evidence"]["components"]}
        same = [i for i, p in enumerate(s["pieces"])
                if p["counted"] and p["region"] == primary["region"]
                and {c["id"] for c in p["evidence"]["components"]} - own]
        spec = fig["spectrum"]
        assert fig["joined_pieces"] == same
        if not same:
            assert spec["joined"] is None
            continue
        found += 1
        assert {w[3] for w in spec["windows"]} == {"own", "joined"}
        joined = {c["id"]: c for i in same for c in s["pieces"][i]["evidence"]["components"]}
        for c in joined.values():
            for st in c["charge_states"]:
                if st["window_mz"] is None:
                    continue
                lo, hi = st["window_mz"]
                assert (st["z"], lo, hi, "joined") in spec["windows"]
                inside = (spec["mz"] >= lo) & (spec["mz"] <= hi)
                assert inside.sum() == st["grid"]["bins"][1] - st["grid"]["bins"][0] + 1
                drawn = float(spec["joined"][inside].sum())
                if len(joined) == 1:
                    assert drawn == pytest.approx(st["grid"]["component_model"], rel=1e-5)
                else:
                    assert drawn >= st["grid"]["component_model"] * (1 - 1e-5)
                total = sum(float(spec[k][inside].sum()) for k in ("own", "joined", "others"))
                assert total == pytest.approx(st["grid"]["model"], rel=1e-5)
    assert found


def _oligo_run():
    """Phosphorothioate oligonucleotides (IP-RP, negative mode), three peaks: the fit of a
    region merges components it found at almost one mass, each part fitted at a template
    position of its own; a later fit pass reuses ids of components the merges took away; and a
    species gathers more pieces (adduct forms folded in) than its figure has colours."""
    cfg = SynthConfig(classes=[ClassConfig("ps_rna", (4000.0, 12000.0))], mode="iprp",
                      polarity=-1)
    run, _truth = generate_run(cfg, np.random.default_rng(1), n_peaks=3, rt_range=(0.5, 3.5),
                               scan_rate_hz=1.0)
    return run


@pytest.fixture(scope="module")
def oligo():
    """The oligonucleotide run processed with provenance, and per region with species: its
    reported and dropped components' templates added up as provenance holds them (before it
    keeps them in single precision), and what the fit subtracted from the region's grid view,
    read by wrapping numpy in the fit's module (deconv.refine)."""
    import lcmsdeconv.deconv.refine as refine
    import lcmsdeconv.process as process
    import lcmsdeconv.provenance as provenance

    subtracted, current, models = {}, [None], []

    class Subtract:  # np.subtract, whose .at also adds what it takes to the region's tally
        def __call__(self, *args, **kwargs):
            return np.subtract(*args, **kwargs)

        def at(self, a, index, values):
            np.subtract.at(a, index, values)
            if current[0] is not None:
                np.add.at(subtracted.setdefault(current[0], np.zeros(a.size)), index, values)

    class NumPy:
        subtract = Subtract()

        def __getattr__(self, name):
            return getattr(np, name)

    deconvolve, init = process.deconvolve_spectrum, provenance._Region.__init__

    def watched_deconvolve(*args, **kwargs):
        current[0] = id(kwargs["keep"])
        subtracted.pop(current[0], None)  # a new region's (an id can be another's again)
        try:
            return deconvolve(*args, **kwargs)
        finally:
            current[0] = None

    def watched_init(self, grid, keep, *args, **kwargs):
        init(self, grid, keep, *args, **kwargs)
        dropped = 0.0 if self.dropped_model is None else self.dropped_model
        models.append((self.model + dropped, subtracted[id(keep)]))

    method = Method.load("iprp_neg_oligo")
    method.deconvolution.model = "comb"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(refine, "np", NumPy())
        mp.setattr(process, "deconvolve_spectrum", watched_deconvolve)
        mp.setattr(provenance._Region, "__init__", watched_init)
        result = process_run(_oligo_run(), method, provenance=True)
    return result, models


def test_a_component_the_fit_merged_is_placed_where_the_fit_subtracted_its_parts(oligo):
    """A region's fit merges components it found at almost one mass (or an adduct form into its
    parent) into one, having subtracted each part's templates where its own fit placed them: the
    records and the figures place them there too. Every reported component's templates and
    those dropped after the fit add up to what the fit subtracted, and a merged component lists
    its parts, whose columns add up to its own."""
    result, models = oligo
    assert models
    for model, subtracted in models:
        assert np.abs(model - subtracted).max() <= 1e-9 * subtracted.max()
    comps = [c for s in result.provenance.species for p in s["pieces"]
             for c in p["evidence"]["components"]]
    dropped = [d for r in result.provenance.regions.values()
               for d in r["deconvolution"]["dropped"]]
    merged = [c for c in comps if c["fitted_parts"]]
    assert merged and [c for c in comps if c["fitted_parts"] is None]
    assert [c for c in merged if len({p["template_mass"] for p in c["fitted_parts"]}) > 1]
    for c in merged:
        parts = c["fitted_parts"]
        assert len(parts) > 1 or parts[0]["template_mass"] != pytest.approx(c["template_mass"])
        fitted: dict = {}
        for p in parts:
            for col in p["columns"]:
                key = (col["z"], col["adduct"])
                fitted[key] = fitted.get(key, 0.0) + col["fitted"]
        own = {(st["z"], ""): st["fitted"] for st in c["charge_states"] if st["fitted"] > 0}
        own |= {(a["z"], a["adduct"]): a["fitted"] for a in c["adduct_columns"]}
        assert fitted.keys() == own.keys()
        for key, v in own.items():
            assert fitted[key] == pytest.approx(v, rel=1e-9)
    for d in dropped:  # the same of a merged component dropped after the fit
        if d["fitted_parts"]:
            fitted = {}
            for p in d["fitted_parts"]:
                for col in p["columns"]:
                    key = (col["z"], col["adduct"])
                    fitted[key] = fitted.get(key, 0.0) + col["fitted"]
            assert fitted == pytest.approx({(c["z"], c["adduct"]): c["fitted"]
                                            for c in d["columns"]}, rel=1e-9)


def test_the_fits_merges_carry_each_parts_templates():
    """The fit's merges (of almost one mass, then an adduct form into its parent) carry each
    merged component's fitted parts into the one they keep: each part at its own template mass,
    its columns counted in the kept component's columns of the combined adduct label, so that
    the parts' columns add up to the kept component's."""
    from lcmsdeconv.chem.adducts import ADDUCTS, AdductLibrary
    from lcmsdeconv.core.model import Component
    from lcmsdeconv.deconv.refine import _merge_adduct_components, _merge_same_mass
    from lcmsdeconv.provenance import fitted_parts, placements

    lib = AdductLibrary.from_mode("iprp", -1, exclude=("TEA", "DIPEA", "HFIP"))
    na = ADDUCTS["Na"].avg_mass
    base = Component(6793.98, 100.0, charges={4: 100.0}, adducts={"base": 100.0},
                     columns={(4, ""): 100.0})
    twin = Component(6794.10, 5.0, charges={4: 5.0}, adducts={"base": 5.0},
                     columns={(4, ""): 5.0}, template_shift=1e-5)
    form = Component(6793.98 + na, 10.0, charges={4: 10.0}, adducts={"base": 8.0, "+K": 2.0},
                     columns={(4, ""): 8.0, (4, "+K"): 2.0})
    form_twin = Component(6794.02 + na, 1.0, charges={4: 1.0}, adducts={"base": 1.0},
                          columns={(4, ""): 1.0})
    comps = [base, twin, form, form_twin]
    where = {id(c): c.mass * float(np.exp(c.template_shift)) for c in comps}
    parts = {id(c): (c, [(where[id(c)], [(z, lab, (z, lab), v)
                                         for (z, lab), v in c.columns.items()])])
             for c in comps}
    _merge_same_mass(comps, parts=parts)
    _merge_adduct_components(comps, lib, parts=parts)
    assert comps == [base] and list(parts) == [id(base)]
    held = parts[id(base)][1]
    assert [tm for tm, _cols in held] == [where[id(c)] for c in (base, twin, form, form_twin)]
    added: dict = {}
    for _tm, cols in held:
        for _z, _lab, key, v in cols:
            added[key] = added.get(key, 0.0) + v
    assert added == pytest.approx(base.columns) and len(base.columns) == 3
    # the form's own base-form templates are counted in the +Na column, its +K in +Na's +K
    form_cols = {lab: key for _z, lab, key, _v in held[2][1]}
    assert form_cols[""] == (4, "+Na") and form_cols["+K"][1] not in ("", "+K", "+Na")
    assert held[3][1] == [(4, "", (4, "+Na"), 1.0)]
    record = fitted_parts(base, base.columns, parts)
    assert [p["template_mass"] for p in record] == [tm for tm, _cols in held]
    assert {(c["adduct"], c["template_adduct"]) for c in record[2]["columns"]} == {
        ("+Na", ""), (form_cols["+K"][1], "+K")}
    placed = placements(base, base.columns, parts)
    assert sorted(placed[(4, "")]) == sorted([(where[id(base)], "", 100.0),
                                              (where[id(twin)], "", 5.0)])
    # a component fitted as one has none
    alone = Component(5000.0, 1.0, columns={(4, ""): 1.0})
    single = {id(alone): (alone, [(5000.0, [(4, "", (4, ""), 1.0)])])}
    assert placements(alone, alone.columns, single) is None
    assert fitted_parts(alone, alone.columns, single) is None


def test_a_component_is_named_by_its_fit_pass_and_id(oligo):
    """A later fit pass numbers its components on from the number the earlier passes kept, so
    an id can recur in a region: wherever a record names a component (its evidence, what was
    dropped after the fit and what it was attributed to), the fit pass and the id name one, and
    the reported mass's origin says both."""
    result, _models = oligo
    prov = result.provenance
    by_id: dict = {}
    by_pass: dict = {}

    def note(region, rec):
        by_id.setdefault((region, rec["id"]), set()).add(rec["mass"])
        by_pass.setdefault((region, rec["pass"], rec["id"]), set()).add(rec["mass"])

    for s in prov.species:
        for p in s["pieces"]:
            for c in p["evidence"]["components"]:
                note((p["region"]["polarity"], p["region"]["id"]), c)
    for key, r in prov.regions.items():
        for d in r["deconvolution"]["dropped"]:
            note(key, d)
            for a in d["attributed_to"]:
                note(key, a)
    assert [k for k, masses in by_id.items() if len(masses) > 1]  # an id naming two
    assert all(len(masses) == 1 for masses in by_pass.values())
    for s in prov.species:
        piece = s["pieces"][s["mass_from"]["piece"]]
        for c in piece["evidence"]["components"]:
            assert f"{c['id']} (fit pass {c['pass']})" in s["mass_from"]["how"]


def test_a_figure_draws_every_piece_of_a_species_of_many(oligo, tmp_path):
    """A species joined from more pieces than its figure has colours: every piece's trace is
    drawn, the first ones named in the legend with how each was joined and the others grey
    under one entry; the spectrum's legend names its other counted pieces by number only."""
    import matplotlib.axes

    from lcmsdeconv.provenance import PIECE_COLOURS, _draw

    def numbers(values):  # consecutive whole numbers as a run (1-8)
        values = sorted(values)
        assert values == list(range(values[0], values[-1] + 1))
        return f"{values[0]}-{values[-1]}"

    result, _models = oligo
    prov = result.provenance
    i = max(range(len(prov.species)), key=lambda k: len(prov.species[k]["pieces"]))
    rec, data = prov.species[i], prov.figures[i]
    n = len(rec["pieces"])
    assert n > len(PIECE_COLOURS)
    drawn, plot = [], matplotlib.axes.Axes.plot

    def record(self, *args, **kwargs):
        drawn.append((self, kwargs.get("label"), kwargs.get("color")))
        return plot(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(matplotlib.axes.Axes, "plot", record)
        _draw(result, rec, data, tmp_path / "figure.png")
    spectrum, trace = drawn[0][0], drawn[-1][0]
    lines = [(label, colour) for ax, label, colour in drawn
             if ax is trace and label not in ("all forms", "base form")]
    assert len(lines) == n  # every piece
    named = [label for label, _c in lines if label and label.startswith("piece ")]
    assert len(named) == len(PIECE_COLOURS) - 1
    for k, label in enumerate(named[1:], start=1):
        assert label.endswith(")") and rec["pieces"][k]["joined_by"] is not None
    rest = list(range(len(named), n))
    assert [label for label, _c in lines if label and label.startswith("pieces ")] == [
        f"pieces {numbers(rest)}: grey, listed in provenance.json"]
    joined = [label for ax, label, _c in drawn
              if ax is spectrum and label and label.startswith("its other counted pieces")]
    assert joined == [f"its other counted pieces of this region (pieces "
                      f"{numbers(data['joined_pieces'])})"]
    assert len(data["joined_pieces"]) > len(PIECE_COLOURS) - 1


def test_window_sums_read_all_scans_at_once():
    """``observed`` adds up every scan's own points in a window with a few searches over all of
    the region's points at once, not two per scan, so that a region of many scans costs no more
    per window; the sums, points and scans with signal are those of the scans one by one."""
    from types import SimpleNamespace

    import lcmsdeconv.provenance as provenance
    from lcmsdeconv.nn.grid import LogMzGrid

    rng = np.random.default_rng(3)
    grid = LogMzGrid(400.0, 2000.0, 1e-4, -1)
    calls, search = [], np.searchsorted

    def counted(*args, **kwargs):
        calls.append(1)
        return search(*args, **kwargs)

    for n in (4, 400):
        frames = [SimpleNamespace(mz=np.sort(rng.uniform(500.0, 1500.0, 300)),
                                  intensity=rng.exponential(1.0, 300)) for _ in range(n)]
        mz = np.concatenate([f.mz for f in frames])
        summed = SimpleNamespace(mz=np.sort(mz), intensity=np.ones(mz.size),
                                 meta={"frames": frames})
        region = provenance._Region(grid, {"observed": np.zeros(grid.size)}, summed,
                                    SimpleNamespace(components=[], noise_sigma=1.0), None, 3.0)
        calls.clear()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(provenance.np, "searchsorted", counted)
            seen = region.raw_signal(900.0, 905.0)
        assert len(calls) <= 4
        inside = [(f.mz >= 900.0) & (f.mz <= 905.0) for f in frames]
        sums = [float(f.intensity[m].sum()) for f, m in zip(frames, inside)]
        assert seen["sum"] == pytest.approx(sum(sums), rel=1e-12)
        assert seen["points"] == sum(int(m.sum()) for m in inside)
        assert seen["scans"] == sum(v > 0 for v in sums)
        assert seen["centroid_mz"] == pytest.approx(
            sum(float((f.mz[m] * f.intensity[m]).sum()) for f, m in zip(frames, inside))
            / sum(sums), rel=1e-12)
    assert 0 < seen["scans"] < 400  # some scans have no point there


def test_observed_is_counted_on_each_scans_own_points(run):
    """``observed`` adds up the region's scans' own points in a window (the instrument's
    counts), also where each scan has an m/z axis of its own (scans calibrated one by one, as
    an Orbitrap's profile scans are), whose summed spectrum interpolates every scan onto the
    union of their points."""
    import copy

    shifted = copy.deepcopy(run)
    rng = np.random.default_rng(7)
    for sp in shifted.spectra:
        sp.mz = sp.mz * (1.0 + rng.normal(0.0, 2e-6))
    assert all(sp.is_profile for sp in shifted.spectra)
    result = process_run(shifted, _method(), provenance=True)
    regions = {(r.polarity, r.id): r for r in result.regions}
    checked = 0
    for s in result.provenance.species:
        for p in s["pieces"]:
            region = regions[(p["region"]["polarity"], p["region"]["id"])]
            summed = region.summed(shifted)
            for c in p["evidence"]["components"]:
                for st in c["charge_states"]:
                    if st["fitted"] <= 0:
                        continue
                    lo, hi = st["window_mz"]
                    inside = [(f.mz >= lo) & (f.mz <= hi) for f in region.frames]
                    sums = [float(np.asarray(f.intensity[m], dtype=float).sum())
                            for f, m in zip(region.frames, inside)]
                    seen = st["observed"]
                    assert seen["sum"] == pytest.approx(sum(sums), rel=1e-12)
                    assert seen["points"] == sum(int(m.sum()) for m in inside)
                    assert seen["scans"] == sum(v > 0 for v in sums)
                    window = (summed.mz >= lo) & (summed.mz <= hi)
                    assert seen["max"] == pytest.approx(float(summed.intensity[window].max()))
                    checked += 1
    assert checked > 50


def test_merged_species_list_their_pieces(written, processed):
    _out, _files, data = written
    reported = {s.id for s in processed.species}
    steps = [m["step"] for s in data["species"] for m in s["merges"]]
    assert "adjacent_pieces" in steps and "adduct_fold" in steps
    for s in data["species"]:
        pieces, merges = s["pieces"], s["merges"]
        assert pieces[0]["species_id"] == s["id"] and pieces[0]["joined_by"] is None
        # every piece but the species' own is joined by one step of the list (it comes first in
        # that step's list; a step joining a species that had joined others lists them after it)
        assert sorted(m["joined_pieces"][0] for m in merges) == list(range(1, len(pieces)))
        assert {i for m in merges for i in m["joined_pieces"]} == set(range(1, len(pieces)))
        for m in merges:
            assert m["step"] in MERGE_STEPS and 0 <= m["kept_piece"] < len(pieces)
            first = pieces[m["joined_pieces"][0]]
            assert first["joined_by"] == m["step"]
            assert first["species_id"] == m["joined"]["species_id"]
            assert first["mass"] == m["joined"]["mass"]
            assert m["joined"]["species_id"] not in reported  # no longer reported apart
            if m["step"] == "adduct_fold":
                assert m["adduct_state"] in s["adduct_fractions"]
        if len(pieces) > 1:
            assert any(f.startswith("adduct forms reported apart") for f in s["flags"]) or \
                "adduct_fold" not in [m["step"] for m in merges]


# ---------------------------------------------------------------------- figures and index
def test_evidence_figures_are_png_files_named_by_id_and_mass(written):
    import matplotlib.image as mpimg

    out, files, data = written
    images = [p for p in files if p.suffix == ".png"]
    assert len(images) == len(data["species"]) == data["figures"]["drawn"] <= MAX_FIGURES
    for s in data["species"]:
        assert s["figure"] == f"evidence/{figure_name(s)}"
        assert re.fullmatch(r"evidence/species-\d+-\d+\.\d\d\.png", s["figure"])
        path = out / s["figure"]
        assert path.read_bytes()[:8] == PNG
        image = mpimg.imread(path)
        assert image.ndim == 3 and image.shape[0] > 400 and image.shape[1] > 600
    page = (out / "provenance.html").read_text(encoding="utf-8")
    for s in data["species"]:
        assert f'href="{s["figure"]}"' in page and f"{s['mass']:,.2f}" in page
    assert 'href="provenance.json"' in page and "file:" not in page and str(out) not in page


def test_the_figures_are_capped(processed, tmp_path):
    files = write_provenance(processed, tmp_path, max_figures=2)
    data = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    drawn = [s for s in data["species"] if s["figure"]]
    assert [s["id"] for s in drawn] == [s["id"] for s in data["species"][:2]]
    assert len([p for p in files if p.suffix == ".png"]) == 2
    assert data["figures"]["drawn"] == 2 and data["figures"]["max"] == 2
    assert "not drawn" in (tmp_path / "provenance.html").read_text(encoding="utf-8")


def test_provenance_names_the_results_it_was_written_with(processed, tmp_path):
    """provenance.json (and provenance.html) name the results.json written with it by its
    SHA-256, so provenance left from an earlier run in the folder can be told apart; written
    on its own, the hash is that of the results.json write_outputs writes."""
    files = write_outputs(processed, tmp_path)
    data = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256((tmp_path / "results.json").read_bytes()).hexdigest()
    assert data["results"] == {"file": "results.json", "sha256": sha}
    assert sha in (tmp_path / "provenance.html").read_text(encoding="utf-8")
    write_provenance(processed, tmp_path / "alone")
    alone = json.loads((tmp_path / "alone" / "provenance.json").read_text(encoding="utf-8"))
    assert alone["results"]["sha256"] == sha
    assert files[0].name == "results.json"


def test_figures_left_from_an_earlier_run_are_removed(processed, tmp_path):
    """Writing the provenance again into a folder removes the evidence figures it does not
    draw, which no provenance.html links any more."""
    write_provenance(processed, tmp_path)
    first = sorted(p.name for p in (tmp_path / "evidence").iterdir())
    assert len(first) == len(processed.species)
    (tmp_path / "evidence" / "notes.txt").write_text("mine", encoding="utf-8")
    files = write_provenance(processed, tmp_path, max_figures=2)
    left = sorted(p.name for p in (tmp_path / "evidence").iterdir())
    assert left == sorted([p.name for p in files if p.suffix == ".png"] + ["notes.txt"])
    assert len(left) == 3
    write_provenance(processed, tmp_path, max_figures=0)
    assert [p.name for p in (tmp_path / "evidence").iterdir()] == ["notes.txt"]


def test_mz_after_a_lock_mass_correction_says_so(processed):
    """Every m/z of the provenance is read on the corrected scans: the units say so where a
    lock-mass correction was applied, with how to go back to the file's m/z."""
    import dataclasses

    plain = processed.provenance.to_dict(processed)
    assert plain["units"]["mz"] == "m/z" and plain["lock_mass"] is None
    corrected = dataclasses.replace(processed, lock_mass={"reference_mz": 554.2615,
                                                          "applied_ppm": -29.98})
    units = processed.provenance.to_dict(corrected)["units"]
    assert "lock-mass correction" in units["mz"]
    assert "m/z / (1 + applied_ppm * 1e-6)" in units["mz"]


def test_polarity_switching_figures_and_index_name_each_region_with_its_polarity(tmp_path):
    """Region ids restart in each polarity: where the run's regions have both, the figures (the
    spectrum's title, each piece's legend entry with its species id and apex) and the index name
    each region with its polarity; a region whose piece is not counted (the other polarity's
    reading) is marked so in the index and in species.regions."""
    import matplotlib.text

    method = Method.load("small_molecule_switching")
    method.deconvolution.model = "comb"
    result = process_run(_switching_run(), method, provenance=True)
    assert {int(r.polarity) for r in result.regions} == {-1, 1}
    joined = [s for s in result.provenance.species
              if any(p["joined_by"] == "other_polarity" for p in s["pieces"])]
    assert joined
    drawn, set_text = [], matplotlib.text.Text.set_text

    def record(self, text):
        drawn.append(str(text))
        return set_text(self, text)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(matplotlib.text.Text, "set_text", record)
        write_provenance(result, tmp_path)
    data = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    page = (tmp_path / "provenance.html").read_text(encoding="utf-8")
    for s in data["species"]:
        for r in s["regions"]:
            pieces = [s["pieces"][i] for i in r["pieces"]]
            assert r["counted"] == any(p["counted"] for p in pieces)
            sign = "+" if r["polarity"] > 0 else "-"
            if not r["counted"]:
                assert {p["joined_by"] for p in pieces} == {"other_polarity"}
                assert f"{r['id']}{sign} (other polarity)" in page
        if s["figure"]:
            primary = s["pieces"][s["mass_from"]["piece"]]["region"]
            sign = "+" if primary["polarity"] > 0 else "-"
            assert [t for t in drawn if f"Summed spectrum of region {primary['id']}{sign}," in t]
            for i, p in enumerate(s["pieces"]):
                sign = "+" if p["region"]["polarity"] > 0 else "-"
                label = (f"piece {i}: species {p['species_id']}, region {p['region']['id']}{sign}"
                         f", {p['mass']:,.2f} Da, apex {p['rt_apex']:.2f} min")
                if len(s["pieces"]) > 1:
                    assert [t for t in drawn if t.startswith(label)], label


def test_a_set_aside_region_is_marked_in_the_index():
    """species.regions marks a region whose pieces are not counted, and the index says why."""
    from lcmsdeconv.provenance import _index

    t = np.linspace(0.0, 1.0, 11)
    main_, later = _sp(0, 7000.0, t, 100 * np.ones(11)), _sp(1, 7000.0, t + 3.0, np.ones(11))
    rec = _recorder([main_, later])
    rec._merge([("set_aside", main_, later, {})])
    record = rec._species(main_, rec.trails[id(main_)])
    assert [(r["id"], r["pieces"], r["counted"]) for r in record["regions"]] == [
        (0, [0], True), (1, [1], False)]
    page = _index({"species": [record], "regions": [{"polarity": -1}],
                   "estimator": {"kind": "comb"}, "run": "run", "method": "method",
                   "lcmsdeconv": lcmsdeconv.__version__, "results": {"sha256": "0" * 64},
                   "figures": {"species": 1, "drawn": 0, "rule": "none"}})
    assert "<td>0, 1 (set aside)</td>" in page


def test_matplotlib_notes_on_font_weights_are_left_out_while_drawing(caplog):
    """The CJK fallback has no face of the weight asked for; matplotlib's note of it (a log line
    on stderr that reads like an error) is left out while the figures are drawn, and only then."""
    import logging

    from lcmsdeconv.provenance import _quiet_font_weights

    log = logging.getLogger("matplotlib.font_manager")
    note = "findfont: Failed to find font weight %s for %s, now using %s."
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        with _quiet_font_weights():
            log.warning(note, "normal", "WenQuanYi Zen Hei", "medium")
            log.warning("findfont: Font family %s not found.", "X")
        log.warning(note, "normal", "WenQuanYi Zen Hei", "medium")
    assert [r.getMessage()[:30] for r in caplog.records] == [
        "findfont: Font family X not fo", "findfont: Failed to find font "]


def test_a_network_is_named_with_its_checksum_and_read_on_the_envelopes(run):
    result = process_run(run, _method(), model_path=str(MODEL), provenance=True)
    est = result.provenance.estimator
    assert est["kind"] == "network" and est["file"] == MODEL.name
    assert est["sha256"] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    maps = [st["charge_map"] for s in result.provenance.species for p in s["pieces"]
            for c in p["evidence"]["components"] for st in c["charge_states"]]
    assert maps and all(m is not None for m in maps)
    read = [m for m in maps if m["bins"]]
    assert read and all(0.0 <= m["p_assigned"] <= 1.0 for m in read)


def test_a_method_quantifying_the_summed_spectrum_only(run, tmp_path):
    """With ``quantify_frames: false`` a species is read once, from its region's summed spectrum:
    its trace has one point, at the scans' mean time, which no single scan has."""
    method = _method()
    method.deconvolution.quantify_frames = False
    method.linking.min_frames = 1  # one reading per region
    result = process_run(run, method, provenance=True)
    assert result.species
    write_provenance(result, tmp_path, max_figures=1)
    data = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    for s in data["species"]:
        assert not s["eic"]["per_scan"]
        assert len(s["eic"]["rt"]) == 1 and s["eic"]["scan_ids"] == [None]
        assert s["pieces"][0]["evidence"]["components"]


def _recorder(species):
    """A Recorder holding each of ``species`` as one piece of its own region, as
    :meth:`~lcmsdeconv.provenance.Recorder.region` records them."""
    from lcmsdeconv.provenance import Recorder, _Trail, piece_charges

    rec = Recorder.__new__(Recorder)
    rec.per_scan, rec._scans, rec.contexts = True, {s.polarity: {} for s in species}, {}
    rec.trails = {id(s): _Trail(s, [{"species_id": s.id, "mass": s.mass, "joined_by": None,
                                      "counted": True, "region": {"id": s.id, "polarity": -1},
                                      "charges": piece_charges(s),
                                      "evidence": {"components": [{"id": 10 + s.id}]}}],
                                [None], [(s.time, s.intensity)]) for s in species}
    return rec


def test_nested_merges_and_a_mass_taken_from_a_boundary_piece():
    """A species set aside after it joined a neighbouring piece brings both along, not counted;
    a boundary piece whose mass the species takes becomes the piece its mass is from. A merge
    names the kept species' own piece as the one kept, wherever its mass is from, and the mass
    is said to come from a boundary piece only where the piece it is from was joined as one."""
    t = np.linspace(0.0, 1.0, 11)
    a, b, c, d = (_sp(i, 5000.0 + 0.1 * i, t, (4 - i) * np.ones(11)) for i in range(4))
    rec = _recorder([a, b, c, d])
    rec._merge([("adjacent_pieces", b, c, {}), ("set_aside", a, b, {}),
                ("boundary_piece", a, d, {"boundary_rt": 0.5, "mass_from_piece": True})])
    trail = rec.trails[id(a)]
    assert list(rec.trails) == [id(a)]
    assert [p["species_id"] for p in trail.pieces] == [0, 1, 2, 3]
    assert [p["joined_by"] for p in trail.pieces] == [None, "set_aside", "adjacent_pieces",
                                                      "boundary_piece"]
    assert [p["counted"] for p in trail.pieces] == [True, False, False, True]
    steps = sorted(trail.merges, key=lambda m: m["seq"])
    assert [(m["step"], m["kept_piece"], m["joined_pieces"], m["counted"]) for m in steps] == [
        ("adjacent_pieces", 1, [2], True), ("set_aside", 0, [1, 2], False),
        ("boundary_piece", 0, [3], True)]
    assert trail.primary == 3 and steps[-1]["mass_from_piece"]
    assert "boundary piece" in rec._species(a, trail)["mass_from"]["how"]

    # a species that took a boundary piece's mass, then set aside: the one it is set aside for
    # keeps its own mass, and says so
    a, b, d = (_sp(i, 5000.0 + 0.1 * i, t, (4 - i) * np.ones(11)) for i in (0, 1, 3))
    rec = _recorder([a, b, d])
    rec._merge([("boundary_piece", b, d, {"boundary_rt": 0.5, "mass_from_piece": True}),
                ("set_aside", a, b, {})])
    record = rec._species(a, rec.trails[id(a)])
    assert record["mass_from"]["piece"] == 0
    assert record["mass_from"]["how"] == "the mass of component 10 of region 0's summed spectrum"

    # a later merge into a species whose mass a boundary piece gave: kept is its own piece
    k, s, el = (_sp(i, 6000.0 + 0.1 * i, t, (4 - i) * np.ones(11)) for i in range(3))
    rec = _recorder([k, s, el])
    rec._merge([("boundary_piece", k, s, {"boundary_rt": 0.5, "mass_from_piece": True}),
                ("set_aside", k, el, {})])
    trail = rec.trails[id(k)]
    steps = sorted(trail.merges, key=lambda m: m["seq"])
    assert [(m["step"], m["kept_piece"], m["joined_pieces"]) for m in steps] == [
        ("boundary_piece", 0, [1]), ("set_aside", 0, [2])]
    assert trail.primary == 1


def test_counted_pieces_add_up_to_the_species_charges(written):
    """Each piece records the charges it was linked with: a species' charges (results.json's)
    are those of its counted pieces added up, for pieces joined as neighbours and as adduct
    forms."""
    out, _files, data = written
    results = {s["id"]: s for s in json.loads((out / "results.json").read_text("utf-8"))["species"]}
    steps = set()
    for s in data["species"]:
        added: dict[str, float] = {}
        for p in s["pieces"]:
            if p["counted"]:
                steps.add(p["joined_by"])
                for c in p["charges"]:
                    added[str(c["z"])] = added.get(str(c["z"]), 0.0) + c["value"]
        expected = results[s["id"]]["charges"]
        assert set(added) == set(expected)
        for z, v in expected.items():
            assert added[z] == pytest.approx(v, rel=1e-12, abs=1e-9)
        assert {str(c["z"]): c["value"] for c in s["charges"]} == {
            z: v for z, v in expected.items() if v > 0}
    assert {"adjacent_pieces", "adduct_fold"} <= steps


def test_a_boundary_piece_adds_its_charges_and_a_set_aside_one_does_not():
    """The bookkeeping of :func:`test_counted_pieces_add_up_to_the_species_charges` for a
    boundary piece (merge_boundary_pieces, as the processing runs it) and a later elution set
    aside (not counted)."""
    from lcmsdeconv.features.link import merge_boundary_pieces

    step = 0.02
    band = lambda t, c, w: np.exp(-0.5 * ((t - c) / w) ** 2)  # noqa: E731
    tk, ts = np.round(np.arange(1.0, 1.6001, step), 6), np.round(np.arange(0.8, 1.0001, step), 6)
    k = _sp(0, 5000.0, tk, 100 * band(tk, 1.2, 0.15))
    s = _sp(1, 5000.8, ts, 50 * band(ts, 1.0, 0.08))
    k.charges, s.charges = {4: 10.0, 5: 20.0}, {5: 3.0, 6: 1.0}
    later = _sp(2, 5000.0, tk + 3.0, band(tk, 1.2, 0.1))
    later.charges = {4: 0.5}
    rec = _recorder([k, s, later])
    log = []
    assert merge_boundary_pieces([k, s, later], [(-1, 1.0)], lambda *a: True, {-1: step},
                                 log=log) == [k, later]
    assert [(e[0], e[1] is k, e[2] is s) for e in log] == [("boundary_piece", True, True)]
    log.append(("set_aside", k, later, {}))  # as set_aside_later_elutions logs it
    rec._merge(log)
    added: dict[int, float] = {}
    for p in rec.trails[id(k)].pieces:
        if p["counted"]:
            for c in p["charges"]:
                added[c["z"]] = added.get(c["z"], 0.0) + c["value"]
    assert added == k.charges == {4: 10.0, 5: 23.0, 6: 1.0}


# ---------------------------------------------------------------------- the merge steps' log
def _sp(sid, mass, t, y, polarity=-1, cls="ps_rna"):
    y = np.asarray(y, dtype=float)
    return Species(sid, mass, "average", polarity, cls, np.asarray(t, dtype=float), y,
                   {"base": y.copy()}, {4: float(y.sum())})


def test_the_merge_steps_log_what_they_join():
    band = lambda t: 100 * np.exp(-0.5 * ((t - 0.9) / 0.2) ** 2)  # noqa: E731
    ta, tb = np.linspace(0.0, 1.0, 26), np.linspace(1.0, 1.3, 8)
    a, b = _sp(0, 2998.45, ta, band(ta)), _sp(1, 2998.40, tb, band(tb))
    log = []
    out = merge_adjacent_species([a, b], log=log)
    assert out == [a] and log == [("adjacent_pieces", a, b, {})]

    # an adduct form folded into its parent
    from lcmsdeconv.chem.adducts import AdductLibrary

    t = np.linspace(0.0, 1.0, 26)
    parent, form = _sp(0, 6000.0, t, band(t)), _sp(1, 6000.0 + 21.98, t, 0.2 * band(t))
    log = []
    assert fold_adduct_forms([parent, form], AdductLibrary.from_mode("rplc", -1), log=log) == [
        parent]
    assert log == [("adduct_fold", parent, form, {"adduct_state": "+Na"})]

    # the other polarity's reading, and a later elution set aside
    pos, neg = _sp(0, 500.0, t, band(t), polarity=1), _sp(1, 500.0, t, 0.5 * band(t))
    log = []
    assert merge_species_across_polarity([pos, neg], log=log) == [pos]
    assert log == [("other_polarity", pos, neg, {})]
    main_, later = _sp(0, 7000.0, t, 1e4 * band(t)), _sp(1, 7000.0, t + 3.0, band(t))
    log = []
    assert set_aside_later_elutions([main_, later], lambda *a: True, {-1: 0.04}, log=log) == [
        main_]
    assert log == [("set_aside", main_, later, {})]


# ---------------------------------------------------------------------- the command line
@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """The run as an mzML file named for a project number, with that number in the run's id,
    its UV trace's name, and the method's file name, name and description; and the bundled
    network copied to a file named for it."""
    d = tmp_path_factory.mktemp("planted")
    run = _run()
    uv = run.chromatograms.pop("UV1")
    uv.name = f"UV {SECRET}"
    run.chromatograms[uv.name] = uv
    run.name = f"LC-{SECRET}-S"
    mzml = write_mzml(run, d / f"LC-{SECRET}-S.mzML")
    method = _method()
    method.name, method.description = f"M-{SECRET}", f"for project {SECRET}"
    method.acquisition.uv_channel = f"UV {SECRET}"
    path = d / f"method-{SECRET}.yaml"
    method.to_yaml(path)
    model = d / f"net-{SECRET}.onnx"
    shutil.copyfile(MODEL, model)
    shutil.copyfile(MODEL.with_suffix(".json"), model.with_suffix(".json"))
    return {"dir": d, "mzml": mzml, "method": path, "model": model}


def _process(planted, out, *extra):
    args = ["process", str(planted["mzml"]), "-m", str(planted["method"]), "-o", str(out), *extra]
    return CliRunner().invoke(main, args, env={ENV_VAR: None})


def test_the_command_line_flag(planted):
    out = planted["dir"] / "with"
    res = _process(planted, out, "--provenance")
    assert res.exit_code == 0, res.output
    assert (out / "provenance.json").is_file() and (out / "provenance.html").is_file()
    assert list((out / "evidence").glob("species-*.png"))
    assert f"provenance: {out / 'provenance.html'}" in res.stdout
    plain = _process(planted, planted["dir"] / "without")
    assert plain.exit_code == 0, plain.output
    assert not list((planted["dir"] / "without").glob("provenance*"))
    assert not (planted["dir"] / "without" / "evidence").exists()
    assert "provenance" not in plain.stdout
    # the other outputs are the same with the flag as without it
    for p in (planted["dir"] / "without").iterdir():
        a, b = p.read_text(encoding="utf-8"), (out / p.name).read_text(encoding="utf-8")
        if p.name == "results.json":
            a, b = (json.dumps(json.loads(x) | {"timings": {}}) for x in (a, b))
        if p.name == "report.html":
            a, b = (re.sub(r"<b>Processing</b>[0-9.]+ s", "", x) for x in (a, b))
        assert a == b, p.name


def test_aliases_reach_every_provenance_file(planted):
    """With an alias, every provenance file shows NEW: in its name, its text and every text its
    figures draw (the run's name in the titles, drawn with the CJK fallback where one is
    installed), the network's file name included; the leak check finds nothing left."""
    import matplotlib.text
    from matplotlib import font_manager

    from lcmsdeconv.aliases import _file_text

    drawn, set_text = [], matplotlib.text.Text.set_text

    def record(self, s):
        drawn.append(str(s))
        return set_text(self, s)

    out = planted["dir"] / f"out-{SECRET}"
    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mp.setattr(matplotlib.text.Text, "set_text", record)
        res = _process(planted, out, "--provenance", "--model", str(planted["model"]),
                       "--alias", f"{SECRET}={NEW}")
    assert res.exit_code == 0, res.output
    assert SECRET not in res.stdout and SECRET not in res.stderr and "WARNING" not in res.stderr
    texts = {}
    for p in sorted(out.rglob("*")):
        name = p.relative_to(out).as_posix()
        assert SECRET not in name
        if p.suffix in (".json", ".html"):
            text = p.read_text(encoding="utf-8")
            if p.suffix == ".json":
                text = json.dumps(json.loads(text), ensure_ascii=False)
            texts[name] = text
    assert {"provenance.json", "provenance.html"} <= set(texts)
    figures = sorted((out / "evidence").glob("species-*.png"))
    assert figures
    for p in figures:  # the title each holds as text, as the leak check reads it
        texts[p.relative_to(out).as_posix()] = _file_text(p)
        assert f"LC-{NEW}-S: species" in texts[p.relative_to(out).as_posix()]
    for name, text in texts.items():
        assert SECRET not in text, name
    data = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert data["run"] == f"LC-{NEW}-S" and data["method"] == f"M-{NEW}"
    assert data["alias_pairs"] == 1 and data["estimator"]["file"] == f"net-{NEW}.onnx"
    assert data["estimator"]["sha256"] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert f"Provenance: LC-{NEW}-S" in texts["provenance.html"]
    assert drawn and not [t for t in drawn if SECRET in t]
    assert [t for t in drawn if f"LC-{NEW}-S: species" in t]
    if {f.name for f in font_manager.fontManager.ttflist} & {"WenQuanYi Zen Hei",
                                                             "Noto Sans CJK SC"}:
        assert not [w for w in caught if "missing from font" in str(w.message)]


def test_aliases_reach_the_network_name_of_a_run_aliased_already(planted, tmp_path):
    """A run and a method aliased already, processed without ``aliases=`` as docs/usage.md
    says: write_outputs(..., aliases=...) shows NEW in the network's file name too, and the
    leak check finds nothing in the provenance files."""
    from lcmsdeconv.io.mzml import read_mzml

    aliases = Aliases.parse(f"{SECRET}={NEW}")
    run = read_mzml(planted["mzml"])
    method = Method.load(str(planted["method"]))
    run, method = aliases.alias_run(run), aliases.alias_method(method, run)
    result = process_run(run, method, model_path=str(planted["model"]), provenance=True)
    out = tmp_path / "out"
    written = write_outputs(result, out, aliases=aliases)
    data = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert data["estimator"]["file"] == f"net-{NEW}.onnx"
    assert data["run"] == f"LC-{NEW}-S"
    texts = [p for p in written if p.suffix in (".json", ".html")]
    assert {p.name for p in texts} >= {"provenance.json", "provenance.html"}
    assert aliases.leaks(texts, out) == []
    assert aliases.leaks([p for p in written if p.suffix == ".png"], out, text=False) == []
    assert aliases.leaks_in(out, exclude=written) == []


def test_the_leak_check_reads_provenance_texts_and_figure_titles(tmp_path, capsys):
    """process's check of its outputs covers the provenance files: provenance.json and .html by
    their text, a figure by its name and the title its PNG file holds as text (its pixels are
    not read), and an earlier run's provenance files and figures left in --out likewise."""
    import matplotlib.image

    aliases = Aliases.parse(f"{SECRET}={NEW}")
    (tmp_path / "evidence").mkdir()
    page = tmp_path / "provenance.json"
    page.write_text(json.dumps({"run": f"LC-{SECRET}-S"}), encoding="utf-8")
    named = tmp_path / "evidence" / f"species-1-{SECRET}x.png"
    named.write_bytes(PNG + b"no text")
    inside = tmp_path / "evidence" / "species-2-100.00.png"
    inside.write_bytes(PNG + f"tEXt {SECRET} ".encode())  # no text chunk: not read
    titled = tmp_path / "evidence" / "species-3-200.00.png"
    matplotlib.image.imsave(titled, np.zeros((2, 2)),
                            metadata={"Title": f"LC-{SECRET}-S: species 3, 200.00 Da"})
    _check_outputs(aliases, [page, named, inside, titled], tmp_path)
    err = capsys.readouterr().err
    assert "WARNING: provenance.json still holds 1 occurrence" in err
    assert "WARNING: evidence/species-1-***x.png still holds 1 occurrence" in err
    assert "WARNING: evidence/species-3-200.00.png still holds 1 occurrence" in err
    assert "species-2" not in err and SECRET not in err
    earlier = tmp_path / "earlier"
    (earlier / "evidence").mkdir(parents=True)
    (earlier / "provenance.html").write_text(f"<p>LC-{SECRET}-S</p>", encoding="utf-8")
    shutil.copyfile(titled, earlier / "evidence" / "species-4-300.00.png")
    shutil.copyfile(titled, earlier / "notes.png")  # no figure of lcmsdeconv's: by its name
    assert aliases.leaks_in(tmp_path, exclude=[page, named, inside, titled]) == [
        ("earlier/evidence/species-4-300.00.png", 1), ("earlier/provenance.html", 1)]


def test_an_earlier_runs_figures_left_in_the_folder_are_flagged(planted):
    """Figures an earlier run drew without aliases show its names in their titles. A later run
    with an alias into the same --out, which leaves them there (it draws none), warns of each,
    by the title each holds as text; so does one into a folder holding such an output in a
    folder of its own, whose own figures (aliased) it does not warn of."""
    first = planted["dir"] / "first"
    res = _process(planted, first, "--provenance")
    assert res.exit_code == 0, res.output
    figures = sorted(p.relative_to(first).as_posix()
                     for p in (first / "evidence").glob("species-*.png"))
    assert figures
    second = planted["dir"] / "second"
    shutil.copytree(first, second / "earlier")
    res = _process(planted, first, "--alias", f"{SECRET}={NEW}")
    assert res.exit_code == 0, res.output
    for name in figures:
        assert f"WARNING: {name}, which this run did not write, holds 1 occurrence" in res.stderr
    res = _process(planted, second, "--provenance", "--alias", f"{SECRET}={NEW}")
    assert res.exit_code == 0, res.output
    for name in figures:
        assert (f"WARNING: earlier/{name}, which this run did not write, holds 1 occurrence"
                in res.stderr)
    assert not [line for line in res.stderr.splitlines() if line.startswith("WARNING: evidence/")]
    assert list((second / "evidence").glob("species-*.png"))
    assert SECRET not in res.stderr


# ---------------------------------------------------------------------- the version
def test_the_package_and_its_metadata_agree_on_the_version():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    found = re.search(r'^version = "([^"]+)"$', pyproject.read_text(encoding="utf-8"), re.M)
    assert found and found.group(1) == lcmsdeconv.__version__ == "0.1.2"
