"""The lock-mass correction of Waters runs (lcmsdeconv.io.lockmass) on synthetic mzMLb and mzML
files written as ProteoWizard writes a Waters run: function 1 the MS scans (profile, negative),
function 2 the lockspray (an 11 m/z window round Leu-enkephalin [M-H]- 554.2615) with a few
scans near the start and the end of the run. The instrument reads +30 ppm high at the first
lockspray scans, drifting to +20 ppm at the last, in both functions alike. The correction divides
every scan's m/z by one ratio, read at the TIC-weighted mean retention time of the run's MS scans
(0.625 min here, +24.7 ppm), so the scans keep one shared m/z axis and the drift leaves up to
about 5 ppm at the run's ends."""

import json
import re

import numpy as np
import pytest

from lcmsdeconv.core.method import Method
from lcmsdeconv.core.model import Run, Spectrum
from lcmsdeconv.io.lockmass import lock_mass_corrected, reference_peak
from lcmsdeconv.io.mzml import read_mzml

from .test_io_mzmlb import _write

LOCK = 554.2615
LOCK_WINDOW = (549.0, 560.0)
MS_WINDOW = (400.0, 2000.0)
PEAKS = (700.0, 1000.0, 1300.0)  # true m/z of the MS scans' peaks
T_LOCK = (0.2, 1.0)  # the lockspray scans' times (minutes)
PPM_LOCK = (30.0, 20.0)  # what the instrument reads at those times
MS_TIMES = np.round(np.arange(0.05, 1.21, 0.05), 4)
BASE = 5.0  # the MS scans' baseline


def offset_ppm(t):
    """The instrument's m/z error at time t: as the lockspray reads it, linear between its
    readings and constant beyond them."""
    return float(np.interp(t, T_LOCK, PPM_LOCK))


def _gauss(x, centre, fwhm, height):
    return height * np.exp(-4.0 * np.log(2.0) * ((x - centre) / fwhm) ** 2)


def _ms_scan(t, ppm=None):
    """Peaks at PEAKS read offset_ppm(t) high (``ppm``, where given), R 15 000, finely sampled and
    thresholded to the peaks as vendor software writes profiles."""
    f = 1.0 + (offset_ppm(t) if ppm is None else ppm) * 1e-6
    xs, ys = [], []
    for k, mz in enumerate(PEAKS):
        x = np.arange(mz - 0.4, mz + 0.4, 0.002)
        xs.append(x * f)
        ys.append(_gauss(x, mz, mz / 15000.0, 1e5 * (1 + k)) + BASE)
    return np.concatenate(xs), np.concatenate(ys)


def _lock_scan(t, rng, ppm=None, peak=True):
    """A lockspray scan: Leu-enkephalin at R 10 000 sampled every 0.018 m/z (three points a
    half-height width) on a baseline of about 100 counts."""
    ppm = offset_ppm(t) if ppm is None else ppm
    x = np.arange(LOCK_WINDOW[0] + rng.uniform(0.0, 0.018), LOCK_WINDOW[1], 0.018)
    y = rng.poisson(100.0, x.size).astype(float)
    if peak:
        y += _gauss(x, LOCK * (1.0 + ppm * 1e-6), LOCK / 10000.0, 1.6e5)
    return x, y


def _waters_scans(lock=True, peak=True, seed=3):
    """Waters scans in file order: function 1 at MS_TIMES, three lockspray scans at each of
    T_LOCK (``lock``; without the reference peak unless ``peak``)."""
    rng = np.random.default_rng(seed)
    scans, n_lock = [], 0
    for k, t in enumerate(MS_TIMES):
        mz, it = _ms_scan(t)
        scans.append((f"function=1 process=0 scan={k + 1}", float(t), mz, it, MS_WINDOW))
        for tl in T_LOCK if lock else ():
            if abs(t - tl) < 1e-9:
                for _ in range(3):
                    n_lock += 1
                    x, y = _lock_scan(tl, rng, peak=peak)
                    scans.append((f"function=2 process=0 scan={n_lock}", float(tl) + 0.001, x, y,
                                  LOCK_WINDOW))
    return scans


def _centre(mz, it, true_mz):
    """A finely sampled peak's centre: its intensity-weighted mean above the baseline."""
    sel = np.abs(mz - true_mz) < 0.3
    x, y = mz[sel], it[sel] - BASE
    return float(np.sum(x * y) / np.sum(y))


def _errors_ppm(run):
    """Each MS scan's peaks' m/z error (ppm) against PEAKS."""
    return np.array([[(_centre(s.mz, s.intensity, mz) / mz - 1.0) * 1e6 for mz in PEAKS]
                     for s in run.spectra])


def _tic_weighted_rt(run):
    """The TIC-weighted mean retention time of the run's MS scans."""
    w = np.array([s.tic for s in run.spectra])
    return float(np.sum(w * np.array([s.rt for s in run.spectra])) / np.sum(w))


def _one_factor(corrected, run):
    """The factor every scan's m/z was multiplied by, checked to be one for all of them."""
    factor = corrected.spectra[0].mz[0] / run.spectra[0].mz[0]
    assert len(corrected.spectra) == len(run.spectra)
    for c, s in zip(corrected.spectra, run.spectra):
        np.testing.assert_allclose(c.mz, s.mz * factor, rtol=1e-13, atol=0.0)
    return factor


def _error_at(run, when):
    """The MS scans' m/z error (ppm) at time ``when``, interpolated between scans."""
    return float(np.interp(when, [s.rt for s in run.spectra], _errors_ppm(run).mean(axis=1)))


def _method(lock=LOCK):
    """The time-of-flight oligonucleotide method with the lock mass ``lock``."""
    method = Method.load("iprp_neg_oligo")
    method.acquisition.lock_mass_mz = lock
    return method


@pytest.fixture(scope="module")
def waters(tmp_path_factory):
    d = tmp_path_factory.mktemp("lockmass")
    scans = _waters_scans()
    return _write(d / "run.mzMLb", scans), _write(d / "run.mzML", scans)


# ------------------------------------------------------------------ reading the reference
@pytest.mark.parametrize("phase", np.linspace(0.0, 0.018, 8, endpoint=False))
def test_the_reference_peak_is_read_at_the_centroid_of_its_top_half(phase):
    """Sampled at three points a half-height width, a Gaussian peak is read within 0.1 ppm at
    any sampling phase, on a noisy baseline (the midpoint of its half-height points errs by up
    to 0.7 ppm, a centroid of the sampled points above half height by up to 9 ppm)."""
    rng = np.random.default_rng(int(phase * 1e5))
    x = np.arange(549.0, 560.0, 0.018) + phase
    true = LOCK * (1.0 + 30e-6)
    y = _gauss(x, true, LOCK / 10000.0, 1.6e5) + rng.poisson(100.0, x.size)
    assert abs(reference_peak(x, y, LOCK) / true - 1.0) * 1e6 < 0.1


def test_a_scan_without_the_reference_peak_gives_no_reading():
    rng = np.random.default_rng(0)
    x = np.arange(549.0, 560.0, 0.018)
    noise = rng.poisson(100.0, x.size).astype(float)
    assert reference_peak(x, noise, LOCK) is None  # the lockspray off: noise only
    assert reference_peak(x[x > 556.0], noise[x > 556.0], LOCK) is None  # no point in the window
    assert reference_peak(x, np.zeros_like(x), LOCK) is None
    # a peak cut by the window's edge: no half-height point on its far side within 0.5 m/z
    edge = _gauss(x, LOCK + 0.5, 0.2, 1e5) + noise
    assert reference_peak(x, edge, LOCK) is None
    # a centroided scan: one point per peak, read where it lies
    sticks_x = np.array([553.1, 554.27, 555.27, 556.0])
    sticks_y = np.array([300.0, 1.6e5, 5e4, 200.0])
    assert reference_peak(sticks_x, sticks_y, LOCK) == 554.27


# ------------------------------------------------------------------ the reader
def test_the_reader_keeps_the_lockspray_scans_apart(waters):
    for path in waters:
        run = read_mzml(path)
        assert run.meta["waters_functions"] == {1: MS_TIMES.size, 2: 6}
        assert run.meta["function"] == 1 and len(run.spectra) == MS_TIMES.size
        lock = run.meta["lockspray"]
        assert [round(rt, 3) for rt, _mz, _it in lock] == [0.201] * 3 + [1.001] * 3
        for _rt, mz, it in lock:
            assert mz.size == it.size and LOCK_WINDOW[0] <= mz[0] and mz[-1] <= LOCK_WINDOW[1]
            assert np.all(np.diff(mz) > 0)
        # reading the lockspray function itself: function 1 spans 1600 m/z, it is no lockspray
        assert "lockspray" not in read_mzml(path, function=2).meta


def test_a_wide_function_not_read_is_no_lockspray(tmp_path):
    """MSE: the high-energy function scans the full window as often as the low-energy one; it is
    not kept (only the narrow lockspray function is)."""
    scans = []
    for k, (sid, rt, mz, it, window) in enumerate(_waters_scans()):
        scans.append((sid, rt, mz, it, window))
        if sid.startswith("function=1"):
            scans.append((f"function=3 process=0 scan={k + 1}", rt + 0.002, mz, it, MS_WINDOW))
    run = read_mzml(_write(tmp_path / "mse.mzML", scans, chromatograms=False))
    assert run.meta["waters_functions"] == {1: MS_TIMES.size, 2: 6, 3: MS_TIMES.size}
    assert len(run.meta["lockspray"]) == 6


# ------------------------------------------------------------------ the correction
def test_one_ratio_corrects_the_run_at_its_tic_weighted_time(waters):
    """The readings are interpolated to the TIC-weighted mean retention time of the MS scans and
    every scan's m/z is divided by that one ratio: the error there goes, every scan is scaled by
    the same factor (scans that shared one m/z axis still do), and the drift is left, up to about
    5 ppm at the run's ends."""
    for path in waters:
        run = read_mzml(path)
        before = _errors_ppm(run)
        expected = np.array([[offset_ppm(s.rt)] * len(PEAKS) for s in run.spectra])
        assert np.abs(before - expected).max() < 0.2  # the run as written
        corrected, record, warnings = lock_mass_corrected(run, LOCK)
        assert warnings == []
        when = _tic_weighted_rt(run)
        assert when == pytest.approx(0.625) and record["tic_weighted_rt"] == pytest.approx(when)
        factor = _one_factor(corrected, run)
        assert (factor - 1.0) * 1e6 == pytest.approx(-offset_ppm(when), abs=0.3)
        assert abs(_error_at(corrected, when)) < 1.0
        after = _errors_ppm(corrected)
        assert np.abs(after - (expected - offset_ppm(when))).max() < 0.5
        assert 4.0 < np.abs(after).max() < 6.0  # +5.3 ppm at the first scan, -4.7 at the last
        # the readings, the scans read, the scaling applied and the readings' spread
        assert [r["rt"] for r in record["readings"]] == [0.201] * 3 + [1.001] * 3
        assert all(abs(r["ppm"] - offset_ppm(r["rt"])) < 0.5 for r in record["readings"])
        assert (record["scans"], record["skipped"], record["reference_mz"]) == (6, 0, LOCK)
        assert record["applied_ppm"] == pytest.approx((factor - 1.0) * 1e6, abs=0.01)
        assert record["drift_ppm"] == pytest.approx(10.0, abs=0.5)
        assert corrected.meta["lock_mass"] is record
        # the run read is left as it was
        assert np.array_equal(_errors_ppm(run), before)
        # and a corrected run is not corrected again
        again, record2, warnings2 = lock_mass_corrected(corrected, LOCK)
        assert again is corrected and record2 is record and warnings2 == []
        _, _, other = lock_mass_corrected(corrected, 556.2766)
        assert len(other) == 1 and "corrected with 554.2615 already" in other[0]


def test_scans_on_one_axis_keep_it_and_the_ratio_follows_the_ion_current():
    """MS scans that share one m/z axis (as a time-of-flight's bins) share it after the
    correction, so a region's scans sum point by point; and the ratio is read where the run's
    ion current lies (a band at 1.0 min), not at the middle of its scans' times."""
    rng = np.random.default_rng(1)
    axis = np.arange(990.0, 1010.0, 0.01)
    band = lambda t: np.exp(-0.5 * ((t - 1.0) / 0.1) ** 2)  # noqa: E731
    spectra = [Spectrum(axis, 1.0 + _gauss(axis, 1000.0 * (1.0 + offset_ppm(t) * 1e-6),
                                           1000.0 / 15000.0, 1e4 * band(t)),
                        rt=float(t), polarity=-1, is_profile=True) for t in MS_TIMES]
    lock = [(tl, *_lock_scan(tl, rng)) for tl in T_LOCK for _ in range(3)]
    run = Run(spectra, {}, "x", "", {"lockspray": lock})
    corrected, record, _ = lock_mass_corrected(run, LOCK)
    assert all(np.array_equal(s.mz, corrected.spectra[0].mz) for s in corrected.spectra)
    assert not np.array_equal(corrected.spectra[0].mz, axis)  # corrected all the same
    assert corrected.sum_frames(corrected.spectra).mz.size == axis.size
    when = _tic_weighted_rt(run)
    assert when > 0.9 and record["tic_weighted_rt"] == pytest.approx(when, abs=1e-4)
    assert record["applied_ppm"] == pytest.approx(-offset_ppm(when), abs=0.3)
    assert (_one_factor(corrected, run) - 1.0) * 1e6 == pytest.approx(record["applied_ppm"],
                                                                     abs=0.01)


def test_the_ratio_is_read_within_the_analysis_window():
    """With an analysis window, the ratio's time is weighted over the MS scans within it: a heavy
    column wash after the window does not pull it away from the band analysed (and with no scan
    within the window the whole run's time is taken)."""
    rng = np.random.default_rng(2)
    axis = np.arange(990.0, 1010.0, 0.01)
    current = lambda t: (np.exp(-0.5 * ((t - 0.4) / 0.1) ** 2)  # noqa: E731
                         + 5.0 * np.exp(-0.5 * ((t - 1.15) / 0.05) ** 2))
    spectra = [Spectrum(axis, 1.0 + _gauss(axis, 1000.0 * (1.0 + offset_ppm(t) * 1e-6),
                                           1000.0 / 15000.0, 1e4 * current(t)),
                        rt=float(t), polarity=-1, is_profile=True) for t in MS_TIMES]
    lock = [(tl, *_lock_scan(tl, rng)) for tl in T_LOCK for _ in range(3)]
    run = Run(spectra, {}, "x", "", {"lockspray": lock})
    whole = lock_mass_corrected(run, LOCK)[1]
    window = (0.1, 0.8)
    corrected, record, _ = lock_mass_corrected(run, LOCK, rt_range=window)
    within = [s for s in run.spectra if window[0] <= s.rt <= window[1]]
    when = _tic_weighted_rt(Run(within, {}, "x", "", {}))
    assert whole["tic_weighted_rt"] > 0.9
    assert record["tic_weighted_rt"] == pytest.approx(when, abs=1e-4)
    assert 0.3 < when < 0.5
    # the band at 0.4 min keeps under half a ppm with the window, several ppm without it
    assert abs(offset_ppm(0.4) + record["applied_ppm"]) < 0.5
    assert abs(offset_ppm(0.4) + whole["applied_ppm"]) > 3.0
    assert (_one_factor(corrected, run) - 1.0) * 1e6 == pytest.approx(record["applied_ppm"],
                                                                     abs=0.01)
    empty = lock_mass_corrected(run, LOCK, rt_range=(5.0, 6.0))[1]
    assert empty["tic_weighted_rt"] == pytest.approx(whole["tic_weighted_rt"], abs=1e-9)


def test_scans_without_the_reference_are_skipped(tmp_path):
    """A lockspray scan that does not show the reference is skipped; where none shows it, the
    run is left uncorrected with a warning."""
    rng = np.random.default_rng(5)
    run = read_mzml(_write(tmp_path / "x.mzML", _waters_scans(), chromatograms=False))
    run.meta["lockspray"].insert(0, (0.0, *_lock_scan(0.0, rng, peak=False)))
    corrected, record, warnings = lock_mass_corrected(run, LOCK)
    assert warnings == [] and (record["scans"], record["skipped"]) == (7, 1)
    assert len(record["readings"]) == 6
    assert abs(_error_at(corrected, _tic_weighted_rt(run))) < 1.0
    run = read_mzml(_write(tmp_path / "y.mzML", _waters_scans(peak=False), chromatograms=False))
    same, record, warnings = lock_mass_corrected(run, LOCK)
    assert same is run and record is None
    assert warnings == ["lock mass 554.2615: none of the 6 lockspray scans shows it within 0.5 "
                        "m/z: m/z not corrected"]


def test_no_lockspray_scans_warn_and_change_nothing(tmp_path):
    run = read_mzml(_write(tmp_path / "nolock.mzMLb", _waters_scans(lock=False),
                           chromatograms=False))
    assert "lockspray" not in run.meta
    same, record, warnings = lock_mass_corrected(run, LOCK)
    assert same is run and record is None
    assert warnings == ["lock mass 554.2615 is set, but the run has no lockspray scans: m/z not "
                        "corrected"]


# ------------------------------------------------------------------ readings not of the reference
def _ms_run(lock, ppm=20.0):
    """A run of MS scans at MS_TIMES reading ``ppm`` high throughout, with the lockspray scans
    ``lock`` = [(rt, m/z, intensity)]."""
    spectra = [Spectrum(*_ms_scan(t, ppm), rt=float(t), polarity=-1, is_profile=True)
               for t in MS_TIMES]
    return Run(spectra, {}, "x", "", {"lockspray": lock})


def _background_scan(rng):
    """A lockspray scan with the lockspray off: noise, and a background ion 0.30 m/z below the
    lock mass."""
    x = np.arange(LOCK_WINDOW[0] + rng.uniform(0.0, 0.018), LOCK_WINDOW[1], 0.018)
    return x, (rng.poisson(100.0, x.size).astype(float)
               + _gauss(x, LOCK - 0.30, LOCK / 10000.0, 3000.0))


def test_a_background_ion_or_a_stray_point_is_no_reading_of_the_reference():
    """The lockspray off at the end of the run: a background ion 0.30 m/z below the lock mass, or
    a thresholded scan's single stray point in the window, is read where the reference would be
    (541 and 381 ppm low). More than 100 ppm off, such readings are skipped, counted and named in
    a warning, and the run is corrected by the readings that remain; with none left, it is not
    corrected."""
    rng = np.random.default_rng(7)
    good = [(0.2, *_lock_scan(0.2, rng, ppm=20.0)) for _ in range(3)]
    bg = _background_scan(rng)
    stray = (np.array([550.1, 554.05, 558.3]), np.array([40.0, 35.0, 50.0]))
    ppm_bg, ppm_stray = ((reference_peak(*scan, LOCK) / LOCK - 1.0) * 1e6 for scan in (bg, stray))
    assert ppm_bg == pytest.approx(-541.0, abs=5.0) and ppm_stray == pytest.approx(-381.0, abs=1.0)
    off = [(1.0, *bg)] * 3 + [(1.1, *stray)]
    named = f"{ppm_bg:+.0f} ppm at 1.00 min (3 scans), {ppm_stray:+.0f} ppm at 1.10 min"
    run = _ms_run(good + off)
    corrected, record, warnings = lock_mass_corrected(run, LOCK)
    assert (record["scans"], record["skipped"], len(record["readings"])) == (7, 4, 3)
    assert record["applied_ppm"] == pytest.approx(-20.0, abs=0.3)
    assert record["drift_ppm"] == 0.0
    assert np.abs(_errors_ppm(corrected)).max() < 1.0
    assert warnings == [f"lock mass 554.2615: 4 lockspray readings more than 100 ppm off skipped "
                        f"as another ion or a stray point ({named})"]
    # a single one is named alike
    _, record, warnings = lock_mass_corrected(_ms_run(good + off[3:]), LOCK)
    assert record["skipped"] == 1 and record["applied_ppm"] == pytest.approx(-20.0, abs=0.3)
    assert warnings == [f"lock mass 554.2615: 1 lockspray reading more than 100 ppm off skipped "
                        f"as another ion or a stray point ({ppm_stray:+.0f} ppm at 1.10 min)"]
    # no reading within 100 ppm: nothing is corrected
    run = _ms_run(off)
    same, record, warnings = lock_mass_corrected(run, LOCK)
    assert same is run and record is None
    assert warnings == [f"lock mass 554.2615: none of the 4 lockspray scans shows it within 100 "
                        f"ppm (the peaks read nearest it: {named}): m/z not corrected"]
    # a bound of 1000 ppm takes them
    _, record, warnings = lock_mass_corrected(run, LOCK, max_ppm=1000.0)
    assert record["skipped"] == 0 and len(record["readings"]) == 4


def test_readings_at_one_time_are_taken_at_their_median():
    """Three scans at one time reading +20, +21 and +35 ppm: their median (+21 ppm) is used, not
    their mean (+25.3), and a warning says how far they spread; within 5 ppm nothing is said."""
    rng = np.random.default_rng(11)
    lock = [(0.2, *_lock_scan(0.2, rng, ppm=p)) for p in (20.0, 21.0, 35.0)]
    corrected, record, warnings = lock_mass_corrected(_ms_run(lock, ppm=21.0), LOCK)
    assert record["applied_ppm"] == pytest.approx(-21.0, abs=0.3)
    assert np.abs(_errors_ppm(corrected)).max() < 1.0
    assert len(warnings) == 1 and re.fullmatch(
        r"lock mass 554\.2615: the 3 readings at 0\.20 min spread 1[45]\.\d ppm; their median is "
        r"used", warnings[0]), warnings
    lock = [(0.2, *_lock_scan(0.2, rng, ppm=p)) for p in (20.0, 21.0, 23.0)]
    _, record, warnings = lock_mass_corrected(_ms_run(lock, ppm=21.0), LOCK)
    assert warnings == [] and record["applied_ppm"] == pytest.approx(-21.0, abs=0.3)


# ------------------------------------------------------------------ in processing
def _processed(monkeypatch, run, method):
    """process_run with no regions found; returns the result and the runs find_regions was
    handed (the scans processing reads)."""
    from lcmsdeconv import process

    seen = []

    def regions(run, *args, **kw):
        seen.append(run)
        return []

    monkeypatch.setattr(process, "find_regions", regions)
    return process.process_run(run, method, predictor=object()), seen


def test_process_run_corrects_once_and_records_it(monkeypatch, waters, tmp_path):
    from lcmsdeconv.report.html import build_report

    method = _method()
    run = read_mzml(waters[0])
    before = _errors_ppm(run)
    when = _tic_weighted_rt(run)
    for _ in range(2):  # the same run processed twice: corrected once each time
        result, seen = _processed(monkeypatch, run, method)
        factor = _one_factor(seen[0], run)
        assert abs(_error_at(seen[0], when)) < 1.0
        assert result.summary()["lock_mass"]["applied_ppm"] == pytest.approx(
            (factor - 1.0) * 1e6, abs=0.01)
        assert not any("lock mass" in w for w in result.warnings)
        assert np.array_equal(_errors_ppm(run), before)
    # the report's header says what was applied
    lock = result.lock_mass
    html = build_report(result, tmp_path / "report.html", plots=False).read_text()
    assert (f"<b>Lock mass 554.2615</b>m/z scaled by {lock['applied_ppm']:+.1f} ppm (readings "
            f"spread {lock['drift_ppm']:.1f} ppm over the run; 6 of 6 lockspray scans "
            "read)") in html
    # a run corrected before processing keeps its correction, also under a method without one
    corrected, record, _ = lock_mass_corrected(run, LOCK)
    for lock in (LOCK, None):
        method.acquisition.lock_mass_mz = lock
        result, seen = _processed(monkeypatch, corrected, method)
        assert seen[0].spectra[3] is corrected.spectra[3] and result.lock_mass is record


def test_process_run_without_lockspray_warns_and_corrects_nothing(monkeypatch, tmp_path):
    run = read_mzml(_write(tmp_path / "nolock.mzML", _waters_scans(lock=False),
                           chromatograms=False))
    result, seen = _processed(monkeypatch, run, _method())
    assert seen[0] is run and result.lock_mass is None
    assert result.summary()["lock_mass"] is None
    assert any("no lockspray scans: m/z not corrected" in w for w in result.warnings)
    # no lock mass in the method: nothing said, nothing done
    result, seen = _processed(monkeypatch, run, _method(None))
    assert seen[0] is run and result.lock_mass is None
    assert not any("lock" in w for w in result.warnings)


def test_the_command_line_applies_the_correction_once(monkeypatch, waters, tmp_path):
    from click.testing import CliRunner

    from lcmsdeconv import process
    from lcmsdeconv.cli import main

    seen = []

    def regions(run, *args, **kw):
        seen.append(run)
        return []

    monkeypatch.setattr(process, "find_regions", regions)
    out = tmp_path / "out"
    with_lock, without = tmp_path / "lock.yaml", tmp_path / "nolock.yaml"
    _method().to_yaml(with_lock)
    _method(None).to_yaml(without)
    res = CliRunner().invoke(main, ["process", str(waters[0]), "-m", str(with_lock),
                                    "--model", "comb", "--no-report", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert "6 lockspray scans kept" in res.output
    run = read_mzml(waters[0])
    factor = _one_factor(seen[0], run)
    summary = json.loads((out / "results.json").read_text())
    lock = summary["lock_mass"]
    assert lock["applied_ppm"] == pytest.approx((factor - 1.0) * 1e6, abs=0.01)
    assert lock["applied_ppm"] == pytest.approx(-offset_ppm(_tic_weighted_rt(run)), abs=0.3)
    assert lock["drift_ppm"] == pytest.approx(10.0, abs=0.5)
    assert (f"lock mass 554.2615: m/z scaled by {lock['applied_ppm']:+.1f} ppm (readings spread "
            f"{lock['drift_ppm']:.1f} ppm over the run; 6 of 6 lockspray scans read)") in res.output
    assert summary["rt_window"] is None
    # a method without a lock mass: the scans are kept, said unused, and nothing is corrected
    res = CliRunner().invoke(main, ["process", str(waters[1]), "-m", str(without),
                                    "--model", "comb", "--no-report", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert "6 lockspray scans kept (not used: the method sets no acquisition.lock_mass_mz)" \
        in res.output
    assert np.abs(_errors_ppm(seen[-1]) - _errors_ppm(read_mzml(waters[1]))).max() == 0.0
    assert json.loads((out / "results.json").read_text())["lock_mass"] is None
