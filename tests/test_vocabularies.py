"""The vocabularies saved with the package, and mzML read and written without the network.

pyteomics and psims fetch psi-ms.obo and uo.obo from purl.obolibrary.org whenever they read or
write mzML unless given a vocabulary; lcmsdeconv gives them the copies in lcmsdeconv/data/cv
(lcmsdeconv.io.vocabularies). The network checks run in a fresh interpreter whose audit hook
records, and refuses, every network call: an earlier test's psims writer may already hold
vocabularies in this process, and a refused call cannot stall the test.
"""

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from pyteomics.version import VersionInfo, version

import lcmsdeconv
from lcmsdeconv.cli import main
from lcmsdeconv.core.model import Chromatogram, Run, Spectrum
from lcmsdeconv.io import vocabularies
from lcmsdeconv.io.mzml import read_mzml
from lcmsdeconv.io.mzml_writer import write_mzml
from lcmsdeconv.io.vocabularies import (
    PSI_MS,
    UO,
    SavedVocabularyResolver,
    VocabularyError,
    vocabulary,
)

#: the saved copies (data/cv/README.md): data-version and sha256 of the uncompressed file
SAVED = {
    PSI_MS: ("psi-ms.obo.gz", "4.2.18",
             "dc44e4de316d35f1ad62d018b921647c08ba9c36b5aae8cb970d733cd174a221"),
    UO: ("uo.obo.gz", "releases/2026-10-12",
         "ade7be5a515551d3d653bda8ad256563df322e83c96b41da4f848e8a470c2846"),
}


@pytest.mark.parametrize("uri", sorted(SAVED))
def test_saved_vocabularies_load_with_their_versions(uri):
    name, version, sha256 = SAVED[uri]
    data = gzip.decompress((vocabularies.CV_DIR / name).read_bytes())
    assert hashlib.sha256(data).hexdigest() == sha256
    cv = vocabulary(uri)
    assert cv.version == version and cv is vocabulary(uri)
    assert SavedVocabularyResolver().load(uri) is cv


def test_saved_vocabularies_hold_the_terms_mzml_needs():
    ms = vocabulary()
    assert ms["MS:1000016"].name == "scan start time"
    assert ms["UO:0000031"].name == "minute"  # PSI-MS carries the units it uses
    assert vocabulary(UO)["UO:0000010"].name == "second"
    assert ms["MS:1000627"].name == "selected ion current chromatogram"


def test_other_vocabularies_are_refused_not_fetched():
    with pytest.raises(ValueError, match="no copy of this vocabulary"):
        vocabulary("http://purl.obolibrary.org/obo/pato.obo")
    resolver = SavedVocabularyResolver()
    with pytest.raises(ValueError):
        resolver.load("http://www.unimod.org/obo/unimod.obo")
    assert resolver.fallback("http://purl.obolibrary.org/obo/pato.obo") is None
    assert resolver.use_remote is False
    with resolver.resolve(UO) as fh:
        assert fh.readline().startswith(b"format-version")


def test_threads_asking_at_once_share_one_parse(monkeypatch):
    monkeypatch.setattr(vocabularies, "_parsed", {})
    parse, calls = vocabularies._parse, []

    def slow_parse(uri):
        calls.append(uri)
        time.sleep(0.2)  # the other threads arrive while the first parses
        return parse(uri)

    monkeypatch.setattr(vocabularies, "_parse", slow_parse)
    start, got = threading.Barrier(6), []

    def ask():
        start.wait()
        got.append(vocabulary(UO))

    threads = [threading.Thread(target=ask) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [UO] and len(got) == 6 and all(cv is got[0] for cv in got)


_OFFLINE = r'''
import json, re, sys, warnings

events = []


def hook(name, args):
    if name in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "urllib.Request"):
        events.append(name)
        raise OSError("no network in this test")


sys.addaudithook(hook)
import numpy as np
from psims.controlled_vocabulary.controlled_vocabulary import obo_cache
from psims.mzml.components import default_cv_list
from psims.mzmlb.writer import MzMLbWriter

from lcmsdeconv.core.model import Chromatogram, Run, Spectrum
from lcmsdeconv.io.mzml import read_mzml
from lcmsdeconv.io.mzml_writer import write_mzml
from lcmsdeconv.io.vocabularies import SavedVocabularyResolver

out = sys.argv[1]
before = (obo_cache.use_remote, obo_cache.enabled, obo_cache.cache_path, dict(obo_cache.resolvers))
mz = np.linspace(500.0, 600.0, 50)
spectra = [Spectrum(mz, np.arange(50.0) + i, rt=0.1 * i, polarity=-1, index=i, is_profile=True)
           for i in range(4)]
t, y = np.linspace(0.0, 0.3, 12), np.linspace(0.0, 1.0, 12)
path = write_mzml(Run(spectra, {"UV 260": Chromatogram(t, y, "UV 260", "uv", "AU")}, name="r"),
                  out + "/r.mzML")
# psims' module-level default entries, which a later plain psims writer would use: not loaded
psims_defaults = [cv._vocabulary is None for cv in default_cv_list]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    writer = MzMLbWriter(out + "/r.mzMLb", close=True,
                         vocabulary_resolver=SavedVocabularyResolver())
    with writer as w:
        w.controlled_vocabularies()
        w.file_description(file_contents=["MS1 spectrum"])
        w.software_list([{"id": "t", "params": ["custom unreleased software tool"]}])
        with w.run(id="r"):
            with w.spectrum_list(count=len(spectra)):
                for i, s in enumerate(spectra):
                    w.write_spectrum(s.mz, s.intensity.astype(np.float32), id=f"scan={i + 1}",
                                     polarity="negative scan", centroided=False,
                                     scan_start_time=s.rt,
                                     params=["MS1 spectrum", {"ms level": 1}])
            with w.chromatogram_list(count=1):
                w.write_chromatogram(t, y.astype(np.float32), id="UV 260", time_unit="minute",
                                     chromatogram_type="electromagnetic radiation chromatogram")
runs = {p: read_mzml(p) for p in (path, out + "/r.mzMLb")}
seen = list(events)
after = (obo_cache.use_remote, obo_cache.enabled, obo_cache.cache_path, dict(obo_cache.resolvers))
from pyteomics import mzml  # control: left to itself, pyteomics 5 has psims fetch the vocabulary
from pyteomics.version import VersionInfo, version

with mzml.read(str(path)) as r:
    next(r)
print(json.dumps({
    "events": seen, "control": events[len(seen):], "obo_cache_unchanged": before == after,
    "pyteomics_5": VersionInfo(version) >= VersionInfo("5.0"), "psims_defaults": psims_defaults,
    "cvlist": re.findall(r'<cv [^>]*version="([^"]*)"', open(path).read()),
    "runs": [{"rt": [s.rt for s in run.spectra], "polarity": [s.polarity for s in run.spectra],
              "mz_equal": all(np.array_equal(s.mz, mz) for s in run.spectra),
              "intensity_equal": all(np.array_equal(s.intensity, np.arange(50.0) + i)
                                     for i, s in enumerate(run.spectra)),
              "chromatograms": {k: [c.kind, np.allclose(c.time, t), np.allclose(c.intensity, y)]
                                for k, c in run.chromatograms.items()}}
             for run in runs.values()]}))
'''


def test_mzml_and_mzmlb_are_written_and_read_without_the_network(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(lcmsdeconv.__file__).resolve().parents[1]))
    done = subprocess.run([sys.executable, "-c", _OFFLINE, str(tmp_path)], env=env, timeout=300,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["events"] == []
    if got["pyteomics_5"]:  # pyteomics before 5 reads with no vocabulary, and fetches nothing
        assert got["control"]  # the hook does see psims fetching for pyteomics left to itself
    assert got["obo_cache_unchanged"]
    assert got["psims_defaults"] == [True, True]
    assert got["cvlist"] == ["4.2.18", "releases/2026-10-12"]
    for run in got["runs"]:  # the mzML, then the mzMLb
        assert run["rt"] == pytest.approx([0.0, 0.1, 0.2, 0.3])
        assert run["polarity"] == [-1] * 4 and run["mz_equal"] and run["intensity_equal"]
        assert run["chromatograms"] == {"UV 260": ["uv", True, True]}


def _probe_files(tmp_path: Path) -> dict[str, Path]:
    """A small mzML file as written, and copies holding one cvParam the saved PSI-MS lacks: in a
    spectrum, in the chromatogram, and a unit given only by an accession neither vocabulary
    holds; and one whose unit is given only by an accession UO holds but PSI-MS does not
    (UO:0000015, centimeter)."""
    mz = np.linspace(500.0, 600.0, 50)
    spectra = [Spectrum(mz, np.arange(50.0) + i, rt=0.1 * i, polarity=-1, index=i,
                        is_profile=True) for i in range(4)]
    t, y = np.linspace(0.0, 0.3, 12), np.linspace(0.0, 1.0, 12)
    plain = write_mzml(Run(spectra, {"UV 260": Chromatogram(t, y, "UV 260", "uv", "AU")},
                           name="r"), tmp_path / "plain.mzML")
    text = plain.read_text()
    unknown = '<cvParam cvRef="MS" accession="MS:9999901" name="new term" value="5"/>'

    def unit(accession):
        return ('<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="0.0" '
                f'unitCvRef="UO" unitAccession="{accession}"/>')

    files = {"plain": plain}
    for name, element, param in (("spectrum", "spectrum", unknown),
                                 ("chromatogram", "chromatogram", unknown),
                                 ("unit", "spectrum", unit("UO:9999999")),
                                 ("uo_unit", "spectrum", unit("UO:0000015"))):
        m = re.search(rf"<{element} [^>]*>", text)
        files[name] = tmp_path / f"{name}.mzML"
        files[name].write_text(text[:m.end()] + param + text[m.end():])
    return files


def test_terms_the_saved_vocabulary_lacks_are_read_with_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(vocabularies, "_unknown", {})
    files = _probe_files(tmp_path)
    plain = read_mzml(files["plain"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = {k: read_mzml(files[k]) for k in ("spectrum", "chromatogram", "unit", "uo_unit")}
        again = read_mzml(files["spectrum"])
    for run in [*runs.values(), again]:
        assert [s.rt for s in run.spectra] == [s.rt for s in plain.spectra]
        assert list(run.chromatograms) == ["UV 260"]  # kept, not dropped with the read failing
    told = [str(w.message) for w in caught if "saved with lcmsdeconv" in str(w.message)]
    if VersionInfo(version) >= VersionInfo("5.0"):  # pyteomics before 5 looks up no term at all
        assert len(told) == 2  # once each, however often met
        assert told[0].startswith("MS:9999901 ") and told[1].startswith("UO:9999999 ")
        assert "scripts/refresh_vocabularies.py" in told[0]
    reader = vocabularies.reader_vocabulary()
    assert reader["UO:0000015"].name == "centimeter"  # a unit PSI-MS lacks is found in UO
    assert reader["MS:9999901"].relationship == [] and reader["MS:9999901"].name == "MS:9999901"
    assert reader["MS:1000016"] is vocabulary()["MS:1000016"]


@pytest.mark.parametrize("damage", ["missing", "truncated", "not_gzip", "no_terms"])
def test_a_damaged_saved_vocabulary_is_named_not_taken_for_a_bad_run(damage, tmp_path,
                                                                     monkeypatch):
    cv_dir = tmp_path / "cv"
    shutil.copytree(vocabularies.CV_DIR, cv_dir)
    target = cv_dir / "psi-ms.obo.gz"
    if damage == "missing":
        target.unlink()
    elif damage == "truncated":
        target.write_bytes(target.read_bytes()[:50000])
    elif damage == "not_gzip":
        target.write_bytes(b"not gzip\n" * 100)
    else:
        target.write_bytes(gzip.compress(b"format-version: 1.2\ndata-version: x\n"))
    monkeypatch.setattr(vocabularies, "CV_DIR", cv_dir)
    monkeypatch.setattr(vocabularies, "_parsed", {})
    with pytest.raises(VocabularyError, match="psi-ms.obo.gz.*reinstall lcmsdeconv"):
        vocabulary()
    assert not issubclass(VocabularyError, ValueError)  # psims would take it for "not available"
    run = tmp_path / "run.mzML"
    run.write_text("<mzML/>")
    for args in (["process", str(run), "--out", str(tmp_path / "out"), "--quiet"],
                 ["synth", "--preset", "oligo", "--peaks", "1", "--minutes", "0.2",
                  "--out", str(tmp_path / "synth")]):
        done = CliRunner().invoke(main, args)
        assert done.exit_code == 1 and "reinstall lcmsdeconv" in done.output, done.output
        assert "Aborted" not in done.output
    assert not (tmp_path / "synth" / "run.mzML").exists()  # no file half written


def test_a_file_that_cannot_be_written_whole_is_removed(tmp_path, monkeypatch):
    cv_dir = tmp_path / "cv"
    shutil.copytree(vocabularies.CV_DIR, cv_dir)
    (cv_dir / "uo.obo.gz").unlink()  # reading needs PSI-MS only, writing UO too
    monkeypatch.setattr(vocabularies, "CV_DIR", cv_dir)
    monkeypatch.setattr(vocabularies, "_parsed", {})
    spectra = [Spectrum(np.linspace(1.0, 2.0, 5), np.ones(5), rt=0.1, polarity=-1, index=0)]
    with pytest.raises(VocabularyError, match="uo.obo.gz"):
        write_mzml(Run(spectra, {}, name="r"), tmp_path / "r.mzML")
    assert not (tmp_path / "r.mzML").exists()
