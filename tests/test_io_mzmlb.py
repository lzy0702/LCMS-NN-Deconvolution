"""mzMLb files, Waters functions and chromatogram kinds (lcmsdeconv.io.mzml).

The files are written with psims, as ProteoWizard would write a Waters run: function 1 scans
400-5000 m/z (profile, negative), function 2 is the lockspray reference (549-560 m/z round
Leu-enkephalin at 554.26) with a few scans interleaved, and the chromatograms are a TIC, a 260 nm
UV trace and the pump pressure.
"""

import warnings

import h5py
import numpy as np
import pytest

from lcmsdeconv.core.model import Chromatogram, Run, Spectrum
from lcmsdeconv.io.mzml import (
    UnsafeXMLError,
    _check_mzmlb_xml,
    _chromatogram_kind,
    _read_chromatograms,
    read_mzml,
)
from lcmsdeconv.io.mzml_writer import write_mzml

FULL = (400.0, 5000.0)
LOCK = (549.0, 560.0)
T = np.linspace(0.0, 1.0, 31)  # chromatogram times, minutes


def _scans(n=20, lock_after=(4, 10, 16)):
    """Waters scans in file order: n of function 1, a lockspray scan after the listed ones."""
    rng = np.random.default_rng(1)
    out = []
    for k in range(n):
        mz = np.linspace(*FULL, 600)
        it = rng.random(600) * 50 + 1000 * np.exp(-0.5 * ((mz - 1200.0 - 10 * k) / 2.0) ** 2)
        out.append((f"function=1 process=0 scan={k + 1}", 0.05 * (k + 1), mz, it, FULL))
        if k in lock_after:
            lmz = np.linspace(*LOCK, 80)
            lit = 2000 * np.exp(-0.5 * ((lmz - 554.26) / 0.05) ** 2) + 1
            out.append((f"function=2 process=0 scan={lock_after.index(k) + 1}",
                        0.05 * (k + 1) + 0.02, lmz, lit, LOCK))
    return out


def _write(path, scans, chromatograms=True, seconds=False):
    """An mzML (psims MzMLWriter) or, by its suffix, mzMLb (MzMLbWriter) file of `scans`."""
    from psims.mzml import MzMLWriter
    from psims.mzmlb.writer import MzMLbWriter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if path.suffix.lower() == ".mzmlb":
            writer = MzMLbWriter(str(path), close=True)
        else:
            writer = MzMLWriter(open(path, "wb"), close=True)
        with writer as out:
            out.controlled_vocabularies()
            out.file_description(file_contents=["MS1 spectrum"])
            out.software_list([{"id": "psims", "params": ["custom unreleased software tool"]}])
            with out.run(id="run"):
                with out.spectrum_list(count=len(scans)):
                    for sid, rt, mz, it, window in scans:
                        start = ({"name": "scan start time", "value": rt * 60.0, "unitName": "second"}
                                 if seconds else rt)
                        out.write_spectrum(mz, it.astype(np.float32), id=sid,
                                           polarity="negative scan", centroided=False,
                                           scan_start_time=start, scan_window_list=[window],
                                           params=["MS1 spectrum", {"ms level": 1}])
                if chromatograms:
                    with out.chromatogram_list(count=3):
                        out.write_chromatogram(T, 1e6 * np.exp(-(((T - 0.5) / 0.1) ** 2)), id="TIC",
                                               chromatogram_type="total ion current chromatogram")
                        # the UV trace's times in seconds
                        out.write_chromatogram(T * 60.0, 0.2 * np.exp(-(((T - 0.48) / 0.1) ** 2)),
                                               id="TUV 260",
                                               chromatogram_type="electromagnetic radiation chromatogram",
                                               time_unit="second", intensity_unit="absorbance unit")
                        out.write_chromatogram(T, 5000.0 + 100.0 * T, id="QSM System Pressure",
                                               chromatogram_type="pressure chromatogram",
                                               intensity_unit="pascal")
    return path


@pytest.fixture(scope="module")
def waters(tmp_path_factory):
    d = tmp_path_factory.mktemp("waters")
    scans = _scans()
    return _write(d / "run.mzMLb", scans), _write(d / "run.mzML", scans)


def test_mzmlb_and_mzml_read_function_1_alike(waters):
    runs = [read_mzml(p) for p in waters]
    for run in runs:
        assert [s.scan_id for s in run.spectra] == [f"function=1 process=0 scan={k}"
                                                     for k in range(1, 21)]
        assert [s.index for s in run.spectra] == list(range(20))
        assert np.allclose(run.rts, 0.05 * np.arange(1, 21))
        assert run.polarities == [-1]
        assert all(s.is_profile and s.ms_level == 1 for s in run.spectra)
        assert np.array_equal(run.spectra[0].mz, np.linspace(*FULL, 600))
        assert run.meta == {"waters_functions": {1: 20, 2: 3}, "function": 1}
    for a, b in zip(*(run.spectra for run in runs)):
        assert np.array_equal(a.mz, b.mz) and np.array_equal(a.intensity, b.intensity)


def test_function_2_is_the_lockspray(waters):
    for path in waters:
        run = read_mzml(path, function=2)
        assert [s.scan_id for s in run.spectra] == [f"function=2 process=0 scan={j}" for j in (1, 2, 3)]
        assert [s.index for s in run.spectra] == [0, 1, 2]
        assert all(LOCK[0] <= s.mz[0] and s.mz[-1] <= LOCK[1] for s in run.spectra)
        assert run.meta == {"waters_functions": {1: 20, 2: 3}, "function": 2}
        with pytest.raises(ValueError, match="no MS1 scans of Waters function 3"):
            read_mzml(path, function=3)


def test_most_scans_win_and_a_tie_goes_to_the_lowest_function(tmp_path):
    mz, it = np.linspace(*FULL, 50), np.ones(50)

    def scan(f, s):
        return f"function={f} process=0 scan={s}", 0.1 * s + 0.01 * f, mz, it, FULL

    # MSE: high and low energy alternate, the high-energy scan first in the file
    run = read_mzml(_write(tmp_path / "mse.mzMLb",
                           [scan(f, s) for s in range(1, 6) for f in (2, 1, 3)][:-1],
                           chromatograms=False))
    assert run.meta == {"waters_functions": {1: 5, 2: 5, 3: 4}, "function": 1}
    assert [s.scan_id for s in run.spectra] == [f"function=1 process=0 scan={s}" for s in range(1, 6)]
    run = read_mzml(_write(tmp_path / "more.mzML", [scan(1, 1), scan(1, 2)]
                           + [scan(3, s) for s in range(1, 4)], chromatograms=False))
    assert run.meta == {"waters_functions": {1: 2, 3: 3}, "function": 3}


def test_a_run_without_functions_is_read_whole(tmp_path):
    mz, it = np.linspace(400.0, 2000.0, 40), np.arange(40.0)
    scans = [(f"controllerType=0 controllerNumber=1 scan={k + 1}", 0.1 * k, mz, it, (400.0, 2000.0))
             for k in range(5)]
    for name in ("thermo.mzML", "thermo.MZMLB"):
        path = _write(tmp_path / name, scans, chromatograms=False, seconds=True)
        run = read_mzml(path)
        assert run.meta == {}
        assert [s.scan_id for s in run.spectra] == [sid for sid, *_ in scans]
        assert [s.index for s in run.spectra] == list(range(5))
        assert np.allclose(run.rts, 0.1 * np.arange(5))  # written in seconds
        assert run.chromatograms == {}
        with pytest.raises(ValueError, match="carry no Waters function numbers"):
            read_mzml(path, function=1)


def test_chromatogram_kinds_and_units(waters):
    for path in waters:
        chroms = _read_chromatograms(path)
        assert {n: c.kind for n, c in chroms.items()} == {
            "TIC": "tic", "TUV 260": "uv", "QSM System Pressure": "pressure"}
        assert np.allclose(chroms["TUV 260"].time, T)  # seconds in the file
        assert np.allclose(chroms["TIC"].time, T)
        assert np.allclose(chroms["QSM System Pressure"].intensity, 5000.0 + 100.0 * T)
        # the pressure is no detector trace: UV-based purity and regions take the TUV trace
        assert [c.name for c in read_mzml(path).uv_traces()] == ["TUV 260"]


@pytest.mark.parametrize("terms, name, kind", [
    ({"pressure chromatogram": ""}, "QSM System Pressure", "pressure"),
    ({"electromagnetic radiation chromatogram": ""}, "TUV 260", "uv"),
    ({"absorption chromatogram": ""}, "PDA 254 nm", "uv"),
    # a sensor a converter typed as a detector trace is still a sensor
    ({"electromagnetic radiation chromatogram": ""}, "QSM System Pressure", "pressure"),
    ({"total ion current chromatogram": ""}, "BSM Flow", "flow"),
    ({"total ion current chromatogram": ""}, "TIC", "tic"),
    ({"basepeak chromatogram": ""}, "BPC", "bpc"),
    ({"selected ion current chromatogram": ""}, "SIC 554.26", "eic"),
    ({}, "Column Temperature", "temperature"),
    ({}, "PDA 254 nm", "uv"),
    ({}, "TIC", "tic"),
])
def test_chromatogram_kind(terms, name, kind):
    assert _chromatogram_kind({"id": name, **terms}, name) == kind


def test_written_kinds_read_back(tmp_path):
    t = np.linspace(0.0, 1.0, 10)
    run = Run([Spectrum(np.linspace(500.0, 600.0, 5), np.ones(5), rt=0.1, index=0)],
              {"UV1": Chromatogram(t, t, "UV1", "uv", "AU"),
               "Pump": Chromatogram(t, t + 1.0, "Pump", "pressure", "bar")})
    back = read_mzml(write_mzml(run, tmp_path / "w.mzML"))
    assert {n: c.kind for n, c in back.chromatograms.items()} == {"UV1": "uv", "Pump": "pressure"}


def _rewrite_xml(path, edit):
    """Replace the mzML document of an mzMLb file by edit(document)."""
    with h5py.File(path, "r+") as h5:
        xml = np.asarray(h5["mzML"][:]).tobytes()
        attrs = dict(h5["mzML"].attrs)
        del h5["mzML"]
        ds = h5.create_dataset("mzML", data=np.frombuffer(edit(xml), dtype=np.int8))
        for k, v in attrs.items():
            ds.attrs[k] = v
    return path


@pytest.mark.parametrize("edit, match", [
    (lambda x: x.replace(b"?>", b'?>\n<!DOCTYPE mzML [<!ENTITY a "aaaaaaaaaa">]>', 1), "DOCTYPE"),
    (lambda x: x.replace(b"</mzML>", b"<!entity a 'a'></mzML>"), "ENTITY"),
    (lambda x: x.replace(b"encoding='utf-8'", b"encoding='UTF-7'", 1), "not UTF-8 or Latin-1"),
    (lambda x: x.replace(b"encoding='utf-8'", b"encoding='utf-16'", 1).decode().encode("utf-16"),
     "not UTF-8 or Latin-1"),
])
def test_mzmlb_xml_declaring_entities_is_refused(tmp_path, edit, match):
    path = _rewrite_xml(_write(tmp_path / "x.mzMLb", _scans(3, ())), edit)
    with pytest.raises(UnsafeXMLError, match=match):
        read_mzml(path)
    with pytest.raises(UnsafeXMLError, match=match):
        _read_chromatograms(path)


def test_the_check_reads_across_its_blocks(tmp_path):
    clean = _write(tmp_path / "clean.mzMLb", _scans(2, ()), chromatograms=False)
    bad = _rewrite_xml(_write(tmp_path / "bad.mzMLb", _scans(2, ()), chromatograms=False),
                       lambda x: x.replace(b"?>", b"?><!DOCTYPE mzML>", 1))
    for block in (1, 2, 5, 8, 9, 64):
        _check_mzmlb_xml(clean, block=block)
        with pytest.raises(UnsafeXMLError, match="DOCTYPE"):
            _check_mzmlb_xml(bad, block=block)


def test_cli_passes_waters_function_to_the_reader(waters, tmp_path, monkeypatch):
    from click.testing import CliRunner

    import lcmsdeconv.process
    from lcmsdeconv.cli import main

    class Read(Exception):
        pass

    seen = {}

    def stop(run, method, **kw):
        seen["run"] = run
        raise Read

    monkeypatch.setattr(lcmsdeconv.process, "process_run", stop)
    runner = CliRunner()
    res = runner.invoke(main, ["process", str(waters[0]), "--waters-function", "2", "-o",
                               str(tmp_path / "out")])
    assert isinstance(res.exception, Read), res.output
    assert seen["run"].meta["function"] == 2 and len(seen["run"].spectra) == 3
    assert "Waters function 2 read (MS1 scans by function: {1: 20, 2: 3})" in res.output
    res = runner.invoke(main, ["process", str(waters[1]), "-o", str(tmp_path / "out")])
    assert isinstance(res.exception, Read) and seen["run"].meta["function"] == 1
    res = runner.invoke(main, ["process", str(waters[1]), "--waters-function", "5"])
    assert res.exit_code == 1 and "no MS1 scans of Waters function 5" in res.output
    raw = tmp_path / "x.raw"
    raw.write_bytes(b"")
    res = runner.invoke(main, ["process", str(raw), "--waters-function", "1"])
    assert res.exit_code == 2 and "--waters-function is for Waters runs" in res.output
    res = runner.invoke(main, ["gui", "--waters-function", "1"])
    assert res.exit_code == 2 and "--waters-function applies to the --run file" in res.output
