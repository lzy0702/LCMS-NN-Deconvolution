"""The analysis window (regions.rt_window): regions, TIC and UV integration, purity and content
within it only, on a synthetic run of two proteins eluting about a minute apart; and its YAML
form, with the lock mass's."""

import math

import numpy as np
import pytest

from lcmsdeconv.core.method import Method, RegionSettings, analysis_window, rt_window_setting
from lcmsdeconv.process import process_run
from lcmsdeconv.synth.chromatography import generate_run
from lcmsdeconv.synth.compounds import ClassConfig
from lcmsdeconv.synth.config import SynthConfig


@pytest.fixture(scope="module")
def two_bands():
    """A run of 0.5-3.5 min holding two proteins, at 1.38 and 2.46 min, and its UV trace; the
    run processed whole; and the time between the two bands."""
    cfg = SynthConfig(classes=[ClassConfig("peptide", (12000.0, 20000.0))], mode="rplc",
                      polarity=1, adduct_max_lambda=0.03)
    run, truth = generate_run(cfg, np.random.default_rng(2), n_peaks=2, rt_range=(0.5, 3.5),
                              scan_rate_hz=1.0)
    first, second = truth.peaks
    assert second.rt - first.rt > 1.0
    return run, truth, _process(run, None), 0.5 * (first.rt + second.rt)


def _process(run, window):
    method = Method.load("rplc_pos_protein")
    method.deconvolution.max_components = 12
    method.regions.rt_window = window
    return process_run(run, method, model_path="comb")


def _near(species, mass, rel=0.002):
    return [s for s in species if abs(s.mass - mass) <= rel * mass]


def _detector_peaks(result):
    """The TIC and UV peak tables' peaks (not those of the deconvolved traces)."""
    return [p for name, table in result.peak_tables.items() if not name.startswith("m ")
            for p in table.peaks]


@pytest.mark.parametrize("side", ["after", "before"])
def test_species_and_peaks_outside_the_window_are_left_out(two_bands, side):
    run, truth, whole, mid = two_bands
    inside, outside = truth.peaks[::-1] if side == "after" else truth.peaks
    window = [mid, None] if side == "after" else [None, mid]
    lo, hi = analysis_window(Method(regions=RegionSettings(rt_window=window)))
    # the whole run reports both proteins and integrates both bands
    for p in truth.peaks:
        assert _near(whole.species, p.compound.mass)
        assert any(abs(q.rt - p.rt) < 0.15 for q in whole.peak_tables["TIC+"].peaks)
    result = _process(run, window)
    assert _near(result.species, inside.compound.mass)
    assert not _near(result.species, outside.compound.mass)
    assert result.species and all(lo <= s.rt_apex <= hi for s in result.species)
    assert all(lo <= r.start and r.end <= hi for r in result.regions)
    # TIC and UV peaks: only those within the window are integrated
    peaks = _detector_peaks(result)
    assert peaks and all(lo <= q.start and q.end <= hi for q in peaks)
    assert any(abs(q.rt - inside.rt) < 0.15 for q in result.peak_tables["TIC+"].peaks)
    # purity and content: from the window's peaks and species only
    assert result.purity["rows"] and all(lo <= r["rt"] <= hi for r in result.purity["rows"])
    assert result.purity["ms"]["n_species"] == len(result.species)
    assert sum(r["area_pct"] for r in result.purity["rows"]) == pytest.approx(100.0)
    assert all(lo <= r["rt"] <= hi for r in result.impurities)
    # the whole traces are kept for display, and the summary records the window
    assert np.array_equal(result.chromatograms["TIC+"].time, whole.chromatograms["TIC+"].time)
    assert result.summary()["rt_window"] == window == result.rt_window


def test_a_window_covering_the_run_changes_nothing(two_bands):
    run, _truth, whole, _mid = two_bands
    expected = whole.summary()
    expected.pop("timings")
    for window in ([0.0, 100.0], [None, 100.0], [0.0, None]):
        got = _process(run, window).summary()
        got.pop("timings")
        assert got.pop("rt_window") == window
        assert got == {k: v for k, v in expected.items() if k != "rt_window"}
    # a window open at both ends is no window
    got = _process(run, [None, None]).summary()
    got.pop("timings")
    assert got == expected and got["rt_window"] is None


def test_a_window_without_scans_warns(two_bands):
    run, *_ = two_bands
    result = _process(run, [10.0, 20.0])
    assert not result.species and not result.regions and not _detector_peaks(result)
    assert any("the analysis window (from 10 min to 20 min) holds no MS scans; the run's scans "
               "span 0.50-3.50 min" in w for w in result.warnings)


def test_the_report_names_the_window(two_bands, tmp_path):
    from lcmsdeconv.report.html import build_report

    run, _truth, _whole, mid = two_bands
    result = _process(run, [round(mid, 2), None])
    html = build_report(result, tmp_path / "report.html").read_text()
    assert f"<b>Analysis window</b>from {round(mid, 2):g} min to the end" in html


def test_re_integration_keeps_to_the_window(two_bands):
    """Re-integrating a windowed result's trace (the GUI's re-integrate button calls
    process.reintegrate) integrates the TIC and UV traces within the window, as processing did,
    where the whole traces kept for display hold the band left out; the area-% purity follows
    the purity signal's new table; a deconvolved trace is integrated whole."""
    import copy

    from lcmsdeconv.chrom.integrate import integrate_chromatogram
    from lcmsdeconv.process import reintegrate
    from lcmsdeconv.quant.purity import area_percent_purity

    run, _truth, _whole, mid = two_bands
    result = _process(run, [mid, None])
    lo, hi = analysis_window(result.method)
    uv = next(k for k, ch in result.chromatograms.items() if ch.kind == "uv")
    assert result.method.quant.purity_signal == "uv" and result.purity["rows"]
    purity, ms = copy.deepcopy(result.purity), result.purity["ms"]
    for name, kind in (("TIC+", "tic"), (uv, "uv")):
        settings = result.method.integration_for(kind)
        whole = integrate_chromatogram(result.chromatograms[name], settings, name)
        assert any(p.rt < lo for p in whole.peaks)  # what integrating the whole trace gives
        before = [(p.rt, p.area) for p in result.peak_tables[name].peaks]
        table = reintegrate(result, name)
        assert result.peak_tables[name] is table
        assert [(p.rt, p.area) for p in table.peaks] == before
        assert table.peaks and all(lo <= p.start and p.end <= hi for p in table.peaks)
    assert result.purity == purity
    # other settings for the purity signal: the purity is read from its new table
    method = copy.deepcopy(result.method)
    method.integration_for("uv").area_reject = 2000.0
    table = reintegrate(result, uv, method)
    assert 0 < len(table) < len(purity["rows"])
    assert all(p.area >= 2000.0 and lo <= p.start for p in table.peaks)
    assert result.purity == {**area_percent_purity(table).to_dict(), "ms": ms}
    assert sum(r["area_pct"] for r in result.purity["rows"]) == pytest.approx(100.0)
    # not the purity signal: the purity stays
    reintegrate(result, "TIC+", method)
    assert result.purity == {**area_percent_purity(table).to_dict(), "ms": ms}
    # a deconvolved trace, whole; and no such trace
    deic = next(k for k, ch in result.chromatograms.items() if ch.kind == "deic")
    before = [(p.rt, p.area) for p in result.peak_tables[deic].peaks]
    assert [(p.rt, p.area) for p in reintegrate(result, deic).peaks] == before
    assert reintegrate(result, "no such trace") is None


# ------------------------------------------------------------------ the method's YAML
@pytest.mark.parametrize("window, bounds", [
    ([2, 30], (2.0, 30.0)),
    ([2.5, None], (2.5, math.inf)),
    ([None, 30], (-math.inf, 30.0)),
    (None, None),
])
def test_the_window_round_trips_through_yaml(tmp_path, window, bounds):
    method = Method.load("iprp_neg_oligo")
    method.regions.rt_window = window
    method.acquisition.lock_mass_mz = 554.2615
    path = tmp_path / "m.yaml"
    method.to_yaml(path)
    back = Method.from_yaml(path)
    assert back.regions.rt_window == (None if window is None else
                                      [None if v is None else float(v) for v in window])
    assert back.acquisition.lock_mass_mz == 554.2615
    assert back.to_dict() == Method.from_dict(method.to_dict()).to_dict()
    assert analysis_window(back) == bounds


def test_a_method_file_without_the_new_settings_loads_with_their_defaults():
    method = Method.from_dict({"name": "old", "regions": {"min_snr": 3.0},
                               "acquisition": {"polarity": "negative"}})
    assert method.regions.rt_window is None and method.acquisition.lock_mass_mz is None
    assert analysis_window(method) is None


@pytest.mark.parametrize("bad", [[30, 2], [2, 2], [2], [2, 10, 15], "2-30", [2, "x"],
                                 [True, 10], [float("nan"), 10], 2.0])
def test_a_malformed_window_is_refused(bad):
    with pytest.raises(ValueError, match="rt_window"):
        rt_window_setting(bad)
    with pytest.raises(ValueError, match="rt_window"):
        Method.from_dict({"regions": {"rt_window": bad}})


@pytest.mark.parametrize("bad", ["554.26", -1.0, 0, True])
def test_a_malformed_lock_mass_is_refused(bad):
    with pytest.raises(ValueError, match="lock_mass_mz"):
        Method.from_dict({"acquisition": {"lock_mass_mz": bad}})


# ------------------------------------------------------------------ the regions' noise floor
def _crowded_run(weak_height, seed=0):
    """A TIC-only run of 60 min, a scan a second: a quiet baseline (noise of 5 counts), a stretch
    crowded with bands from 3 to 27 min round a main band at 15 min, and a weak band at 28.5 min."""
    from lcmsdeconv.core.model import Run, Spectrum

    rng = np.random.default_rng(seed)
    t = np.arange(3600) / 60.0
    tic = 1000.0 + rng.normal(0.0, 5.0, t.size)
    for c in np.arange(3.0, 27.0, 0.3):
        tic += rng.uniform(300.0, 3000.0) * np.exp(-0.5 * ((t - c) / 0.08) ** 2)
    tic += 2e4 * np.exp(-0.5 * ((t - 15.0) / 0.1) ** 2)
    tic += weak_height * np.exp(-0.5 * ((t - 28.5) / 0.06) ** 2)
    return Run([Spectrum(np.array([1000.0]), np.array([y]), rt=float(x), polarity=-1)
                for x, y in zip(t, tic)])


def test_the_regions_noise_floor_is_the_whole_runs(monkeypatch):
    """A window that leaves out the run's quiet baseline and is crowded with bands reads a noise
    on its own trace many times the run's, and three times that lies above a weak band near the
    floor. The noise is read on the whole run, so the band makes the same region with the window
    as without it; the relative floors are the window's, so no region of the run within the
    window is lost."""
    from lcmsdeconv import process
    from lcmsdeconv.core.model import FrameResult
    from lcmsdeconv.features.regions import source_noise

    monkeypatch.setattr(process, "deconvolve_spectrum",
                        lambda spec, *a, **k: FrameResult(spec.rt, spec.polarity, []))
    run, window, height = _crowded_run(175.0), [2.0, 30.0], 175.0
    view = run.windowed(*window)
    method = Method.load("rda_iprp_neg_oligo")
    method.acquisition.lock_mass_mz = None
    assert source_noise(view, "tic", -1) > 5.0 * source_noise(run, "tic", -1)
    assert method.regions.min_snr * source_noise(view, "tic", -1) > height
    found = {}
    for w in (None, window):
        method.regions.rt_window = w
        found[w is None] = process.process_run(run, method, predictor=object()).regions
    whole, windowed = found[True], found[False]

    def band(regions):
        return [(r.start, r.end) for r in regions if r.start <= 28.5 <= r.end]

    assert len(band(whole)) == 1 and band(windowed) == band(whole)
    inside = {(r.start, r.end) for r in whole if window[0] <= r.start and r.end <= window[1]}
    assert inside <= {(r.start, r.end) for r in windowed}


# ------------------------------------------------------------------ flatness judged in context
def test_a_flat_species_is_judged_with_the_scans_beyond_the_window(monkeypatch):
    """A window drawn round one band: the region fills it, and whether a species' trace falls
    away is judged with a region width of the run's scans on either side, beyond the window. A
    chemical background species, the same in every scan (and the region's strongest, so the
    edge rule spares it), is dropped as without the window, and the band's species is kept."""
    from lcmsdeconv import process
    from lcmsdeconv.core.model import Component, FrameResult, Run, Spectrum
    from lcmsdeconv.features.regions import Region

    times = np.round(np.arange(4.0, 7.0 + 1e-9, 0.02), 4)
    run = Run([Spectrum(np.array([1000.0]), np.array([1.0]), rt=float(t), polarity=1)
               for t in times])

    def band(t):
        return 400.0 * np.exp(-0.5 * ((t - 5.3) / 0.05) ** 2)

    class Bank:
        """What each scan quantifies to: the background everywhere, the band in its scans."""

        def __init__(self, *args, **kwargs):
            pass

        def quantify(self, spec):
            comps = [Component(5000.0, 100.0)]
            if band(spec.rt) > 1.0:
                comps.append(Component(6000.0, float(band(spec.rt))))
            return FrameResult(spec.rt, spec.polarity, comps)

    monkeypatch.setattr(process, "TemplateBank", Bank)
    monkeypatch.setattr(process, "deconvolve_spectrum", lambda spec, *a, **k: FrameResult(
        spec.rt, spec.polarity, [Component(5000.0, 1.0), Component(6000.0, 1.0)]))
    monkeypatch.setattr(process, "find_regions", lambda r, *a, **k: [
        Region(0, 5.0, 5.6, 1, r.frames(1, (5.0, 5.6)))])
    method = Method.load("rplc_pos_protein")
    for window in (None, [5.0, 5.6]):
        method.regions.rt_window = window
        result = process.process_run(run, method, predictor=object())
        assert [round(s.mass) for s in result.species] == [6000], window
