"""Name aliases (lcmsdeconv.aliases, --alias, --alias-file, LCMSDECONV_ALIAS_FILE).

Every number and name here is made up: 123456 plays the project number to hide.
"""

import dataclasses
import json
import logging
import re
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest
from click.testing import CliRunner

from lcmsdeconv.aliases import ENV_VAR, AliasError, Aliases, unique_names
from lcmsdeconv.cli import main
from lcmsdeconv.core.method import IntegrationEvent, Method
from lcmsdeconv.core.model import Chromatogram, Run, Spectrum

SECRET = "123456"
NEW = "保密项目1"
DATA_URI = re.compile(r"data:[\w.+/-]+;base64,[A-Za-z0-9+/=]*")


# ---------------------------------------------------------------------- parsing
def test_parse_pairs():
    a = Aliases.parse([f"{SECRET}={NEW}", "  A B = C=D  ", ("x", " y ")])
    assert a.pairs == {SECRET: NEW, "A B": "C=D", "x": "y"}
    assert len(a) == 3 and a and not Aliases()
    assert repr(a) == "Aliases(3 pairs)" and SECRET not in repr(a)
    assert Aliases.parse(f"{SECRET}={NEW}").pairs == {SECRET: NEW}


@pytest.mark.parametrize("item, reason", [(f"PRJ-{SECRET}", "no '='"),
                                         (f" ={SECRET}", "nothing before '='"),
                                         (f"PRJ-{SECRET} =  ", "nothing after '='")])
def test_parse_errors_never_quote_the_pair(item, reason):
    with pytest.raises(AliasError) as err:
        Aliases.parse(["a=b", item])
    assert str(err.value).startswith("--alias #2: ") and reason in str(err.value)
    assert SECRET not in str(err.value)


@pytest.mark.parametrize("item, side, code", [
    (f"{SECRET}\u200b={NEW}", "OLD", "U+200B"),  # a zero-width space pasted with the number
    (f"\ufeff{SECRET}={NEW}", "OLD", "U+FEFF"),
    (f"{SECRET}\x00={NEW}", "OLD", "U+0000"),
    (f"{SECRET}=\u200e{NEW}", "NEW", "U+200E"),
    (f"{SECRET}=A\u2028B", "NEW", "U+2028"),
])
def test_invisible_characters_are_refused(item, side, code):
    """An OLD holding an invisible or control character would match nothing, silently: refused,
    naming the character's code point, never the pair."""
    with pytest.raises(AliasError) as err:
        Aliases.parse(["a=b", item])
    assert str(err.value) == f"--alias #2: {side} holds {code}, an invisible or control character"
    with pytest.raises(AliasError, match="^" + re.escape(f"alias pair 1: {side} holds {code},")):
        Aliases([tuple(item.split("=", 1))])


def test_alias_file_format(tmp_path):
    lines = ["# project codes", "", f"  {SECRET} = {NEW}  ", "   # an indented comment",
             "PRJ-A=first", "PRJ-A=second", "OLD = NEW = with equals", "\t"]
    path = tmp_path / "aliases.txt"
    path.write_bytes(b"\xef\xbb\xbf" + "\r\n".join(lines).encode("utf-8") + b"\r\n")
    a = Aliases.from_file(path)
    assert a.pairs == {SECRET: NEW, "PRJ-A": "second", "OLD": "NEW = with equals"}
    lf = tmp_path / "lf.txt"
    lf.write_text("\n".join(lines), encoding="utf-8")
    assert Aliases.from_file(lf).pairs == a.pairs
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing yet\n", encoding="utf-8")
    assert not Aliases.from_file(empty)


@pytest.mark.parametrize("content, reason, line", [
    (f"QQ=RR\n# fine\nPRJ {SECRET} no equals\n", "no '='", 3),
    (f"QQ=RR\n={SECRET}\n", "nothing before '='", 2),
    (f"\n\nPRJ-{SECRET}=\n", "nothing after '='", 3),
    (b"QQ=RR\r\nQZ=RZ\r\nPRJ-\xff\xfe\r\n", "not UTF-8", 3),
])
def test_alias_file_errors_name_file_and_line_but_not_the_line(tmp_path, content, reason, line):
    path = tmp_path / "secret-aliases.txt"
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    with pytest.raises(AliasError) as err:
        Aliases.from_file(path)
    msg = str(err.value)
    assert msg == f"alias file {path}, line {line}: {reason}" + (
        " (expected OLD=NEW)" if reason == "no '='" else "")
    assert SECRET not in msg and "PRJ" not in msg


@pytest.mark.parametrize("content, line, reason", [
    (f"QQ=RR\n{SECRET}\u200b={NEW}\n".encode(), 2,
     "OLD holds U+200B, an invisible or control character"),
    (f"QQ=RR\nQZ=RZ\n{SECRET}=\u200e{NEW}\n".encode(), 3,
     "NEW holds U+200E, an invisible or control character"),
    (f"{SECRET}=A\u2028B\nPRJ\n".encode(), 1,  # U+2028 ends no line: an editor shows one
     "NEW holds U+2028, an invisible or control character"),
    (f"QQ=RR\n{SECRET}={NEW}\n".encode("utf-16-le"), 1,
     "not UTF-8 (it holds NUL bytes, as UTF-16 text does)"),
    (b"QQ=RR\rQZ=RZ\rPRJ-\xff=x\r", 3, "not UTF-8"),  # a lone CR ends a line
])
def test_alias_file_with_invisible_characters_or_utf16(tmp_path, content, line, reason):
    path = tmp_path / "aliases.txt"
    path.write_bytes(content)
    with pytest.raises(AliasError) as err:
        Aliases.from_file(path)
    assert str(err.value) == f"alias file {path}, line {line}: {reason}"


def test_byte_order_marks_where_files_were_joined(tmp_path):
    """`cat a.txt b.txt` of two files saved with a byte order mark puts one at the start of a
    line: taken as one, not as part of the OLD (or of a comment)."""
    joined = tmp_path / "joined.txt"
    joined.write_bytes(b"\xef\xbb\xbf999999=X\r\n\xef\xbb\xbf# the second file\r\n\xef\xbb\xbf"
                       + f"{SECRET}={NEW}\r\n".encode())
    a = Aliases.from_file(joined)
    assert a.pairs == {"999999": "X", SECRET: NEW} and a.apply(f"LC-{SECRET}") == f"LC-{NEW}"


def test_alias_file_missing_or_a_directory(tmp_path):
    with pytest.raises(AliasError, match="not found$"):
        Aliases.from_file(tmp_path / "missing.txt")
    with pytest.raises(AliasError, match="is a directory$"):
        Aliases.from_file(tmp_path)


def test_alias_file_errors_show_its_path_aliased(tmp_path):
    """A failing file's path is shown with the pairs known so far: those of earlier sources and
    of the file's lines before the one that failed."""
    path = tmp_path / f"aliases-{SECRET}.txt"
    path.write_text(f"{SECRET}=X\nbroken\n", encoding="utf-8")
    with pytest.raises(AliasError) as err:
        Aliases.from_file(path)
    assert "aliases-X.txt, line 2" in str(err.value) and SECRET not in str(err.value)
    with pytest.raises(AliasError) as err:
        Aliases.from_file(tmp_path / f"gone-{SECRET}.txt", known=Aliases.parse(f"{SECRET}=Y"))
    assert "gone-Y.txt: not found" in str(err.value)


def test_a_failing_file_is_shown_with_the_alias_pairs_too(tmp_path):
    """The --alias pairs are read first, so a failing --alias-file's or LCMSDECONV_ALIAS_FILE's
    path is shown with them, wherever they come on the command line."""
    folder = tmp_path / f"sec-{SECRET}"
    folder.mkdir()
    bad = folder / f"bad-{SECRET}.txt"
    bad.write_text("no equals here\n", encoding="utf-8")
    gone = str(folder / f"gone-{SECRET}.txt")
    for files, environ in [([folder / f"missing-{SECRET}.txt"], {}), ([bad], {}),
                           ([], {ENV_VAR: gone})]:
        with pytest.raises(AliasError) as err:
            Aliases.collect(files, [f"{SECRET}={NEW}"], environ=environ)
        assert f"sec-{NEW}" in str(err.value) and SECRET not in str(err.value)
    for args, env in [(["--alias-file", str(bad), "--alias", f"{SECRET}={NEW}"], None),
                      (["--alias", f"{SECRET}={NEW}", "--alias-file", str(bad)], None),
                      (["--alias", f"{SECRET}={NEW}"], gone)]:
        res = CliRunner().invoke(main, ["report", str(tmp_path), *args], env={ENV_VAR: env})
        assert res.exit_code == 1 and "Traceback" not in res.output
        assert res.stderr.startswith("Error: ") and f"sec-{NEW}" in res.stderr
        assert SECRET not in res.output


def test_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert not Aliases.from_env()
    path = tmp_path / "env.txt"
    path.write_text(f"{SECRET}={NEW}\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(path))
    assert Aliases.from_env().pairs == {SECRET: NEW}
    assert Aliases.from_env({ENV_VAR: ""}).pairs == {}
    with pytest.raises(AliasError, match=f"^{ENV_VAR}: alias file .*: not found$"):
        Aliases.from_env({ENV_VAR: str(tmp_path / "missing.txt")})


def test_precedence_env_then_files_then_pairs(tmp_path):
    env = tmp_path / "env.txt"
    env.write_text(f"{SECRET}=ENV\n777777=E7\n", encoding="utf-8")
    first, second = tmp_path / "1.txt", tmp_path / "2.txt"
    first.write_text(f"{SECRET}=FILE1\n888888=F8\n", encoding="utf-8")
    second.write_text(f"{SECRET}=FILE2\n", encoding="utf-8")
    environ = {ENV_VAR: str(env)}
    assert Aliases.collect(environ=environ).pairs == {SECRET: "ENV", "777777": "E7"}
    got = Aliases.collect([first, second], environ=environ).pairs
    assert got == {"777777": "E7", "888888": "F8", SECRET: "FILE2"}
    got = Aliases.collect([first, second], [f"{SECRET}=CLI", "888888=C8"], environ=environ)
    assert got.pairs == {"777777": "E7", SECRET: "CLI", "888888": "C8"}
    assert (Aliases.parse("a=1") | Aliases.parse(["a=2", "b=3"])).pairs == {"a": "2", "b": "3"}


# ---------------------------------------------------------------------- matching
@pytest.mark.parametrize("name, aliased", [
    ("LC-123456-S", f"LC-{NEW}-S"), ("LC123456", f"LC{NEW}"),
    ("123456_2", f"{NEW}_2"), ("123456.2", f"{NEW}.2"), ("1234567", "1234567"),
    ("0123456", "0123456"), ("123456123456", "123456123456"), ("a123456b", f"a{NEW}b"),
])
def test_digit_rules(name, aliased):
    assert Aliases.parse(f"{SECRET}={NEW}").apply(name) == aliased


def test_longest_old_wins_in_one_pass():
    a = Aliases.parse([f"{SECRET}=QQ", f"LC-{SECRET}=RR", "QQ=never"])
    assert a.apply(f"LC-{SECRET}-S X-{SECRET}") == "RR-S X-QQ"  # NEW never matched again
    assert Aliases.parse([f"{SECRET}=ABC", "ABC=DEF"]).apply(f"{SECRET} ABC") == "ABC DEF"
    # two OLDs starting at one position: the longer wins, whichever pair comes first
    for pairs in ([f"{SECRET}=A", f"{SECRET}-X=B"], [f"{SECRET}-X=B", f"{SECRET}=A"]):
        assert Aliases.parse(pairs).apply(f"LC-{SECRET}-X") == "LC-B"


def test_case_sensitive_and_nfc():
    assert Aliases.parse("prj=X").apply("PRJ prj Prj") == "PRJ X Prj"
    composed, decomposed = "Café-1", "Café-1"
    assert Aliases.parse(f"{decomposed}=X").apply(f"a {composed}") == "a X"
    assert Aliases.parse(f"{composed}=X").apply(f"a {decomposed}") == "a X"
    other = "Café"  # no alias in it: comes back as it was, not recomposed
    assert Aliases.parse("zzz=X").apply(other) is other


def test_digit_rule_applies_to_digit_ends_only():
    assert Aliases.parse("SAMPLE=S").apply("2SAMPLE3") == "2S3"
    a = Aliases.parse(["A1=X", "1B=Y"])
    assert a.apply("A12 A1B 21B C1B") == "A12 XB 21B CY"


def test_apply_leaves_anything_but_text():
    a = Aliases.parse(f"{SECRET}={NEW}")
    assert a.apply(None) is None and a.apply(123456) == 123456 and a.apply("") == ""
    text = "no alias here"
    assert a.apply(text) is text


def test_unique_names():
    assert unique_names(["UV", "UV", "TIC", "UV"]) == ["UV", "UV (2)", "TIC", "UV (3)"]
    assert unique_names(["X", "X", "X (2)"]) == ["X", "X (3)", "X (2)"]


# ---------------------------------------------------------------------- objects
def _run():
    t = np.linspace(0.0, 1.0, 11)
    spectra = [Spectrum(np.array([500.0, 501.0]), np.array([1.0, 2.0]), 0.1 * i,
                        scan_id=f"file={SECRET} scan={i}" if i == 0 else f"scan={i}", index=i)
               for i in range(3)]
    chroms = {f"UV {SECRET}": Chromatogram(t, np.ones(11), f"UV {SECRET}", "uv", f"mAU {SECRET}",
                                           {"source": f"{SECRET}.raw", "nm": 260.0}),
              f"UV {NEW}": Chromatogram(t, 2 * np.ones(11), f"UV {NEW}", "uv", "mAU"),
              "TIC": Chromatogram(t, 3 * np.ones(11), "TIC", "tic")}
    meta = {"function": 2, "waters_functions": {1: 123456, 2: 1234560},
            "lockspray": [(0.5, np.array([554.26]), np.array([1e4]))],
            "instrument": f"QTOF {SECRET}", "files": [f"a-{SECRET}.raw", 123456.7]}
    return Run(spectra, chroms, f"LC-{SECRET}-S", f"/data/LC-{SECRET}-S.mzML", meta)


def test_alias_run_returns_an_aliased_copy():
    run = _run()
    keys_before = list(run.chromatograms)
    got = Aliases.parse(f"{SECRET}={NEW}").alias_run(run)
    assert got is not run
    assert got.name == f"LC-{NEW}-S" and got.source == f"/data/LC-{NEW}-S.mzML"
    # the names aliasing made equal stay apart: no trace is lost
    assert list(got.chromatograms) == [f"UV {NEW}", f"UV {NEW} (2)", "TIC"]
    assert [c.name for c in got.chromatograms.values()] == list(got.chromatograms)
    assert [float(c.intensity[0]) for c in got.chromatograms.values()] == [1.0, 2.0, 3.0]
    uv = got.chromatograms[f"UV {NEW}"]
    assert uv.unit == f"mAU {NEW}" and uv.meta == {"source": f"{NEW}.raw", "nm": 260.0}
    assert uv.intensity is run.chromatograms[f"UV {SECRET}"].intensity  # arrays shared
    # strings aliased, numbers and arrays left as they are
    assert got.meta["instrument"] == f"QTOF {NEW}"
    assert got.meta["files"] == [f"a-{NEW}.raw", 123456.7]
    assert got.meta["waters_functions"] == {1: 123456, 2: 1234560}
    assert got.meta["lockspray"] is run.meta["lockspray"]
    assert got.spectra[0].scan_id == f"file={NEW} scan=0" and got.spectra[1] is run.spectra[1]
    # the caller's run is unchanged
    assert run.name == f"LC-{SECRET}-S" and list(run.chromatograms) == keys_before
    assert run.chromatograms[f"UV {SECRET}"].name == f"UV {SECRET}"
    assert run.meta["instrument"] == f"QTOF {SECRET}" and run.spectra[0].scan_id.startswith("file=1")


def test_alias_method_returns_an_aliased_copy():
    method = Method.load("rplc_pos_protein")
    method.name, method.description = f"M-{SECRET}", f"for project {SECRET}"
    method.acquisition.uv_channel = f"UV {SECRET}"
    method.quant.purity_signal = f"UV {SECRET}"
    method.quant.calibration.amount_unit = f"mg/mL {SECRET}"
    method.quant.sample_amount = 123456.7
    method.quant.main_species = "mass:123456.7"
    method.report.title = f"Report {SECRET}"
    method.integration["uv"].area_unit = f"AU*s {SECRET}"
    method.deconvolution.model = f"models/{SECRET}.onnx"
    method.saturation.detector_level = 123456.0
    method.instrument.kind = f"tof {SECRET}"  # the report's Instrument
    method.quant.calibration.weighting = f"1/x {SECRET}"  # results.json's potency
    method.integration["uv"].timed_events = [  # the desktop application's events table
        IntegrationEvent(1.0, f"note {SECRET}", f"by {SECRET}"),
        IntegrationEvent(2.0, "Baseline now", None), IntegrationEvent(3.0, "peak_width", "0.2"),
        IntegrationEvent(4.0, "integration", "off"), IntegrationEvent(5.0, "area_reject", SECRET)]
    got = Aliases.parse(f"{SECRET}={NEW}").alias_method(method)
    assert got is not method
    assert (got.name, got.description) == (f"M-{NEW}", f"for project {NEW}")
    assert got.acquisition.uv_channel == got.quant.purity_signal == f"UV {NEW}"
    assert got.quant.calibration.amount_unit == f"mg/mL {NEW}"
    assert got.report.title == f"Report {NEW}" and got.integration["uv"].area_unit == f"AU*s {NEW}"
    assert got.instrument.kind == f"tof {NEW}" and got.quant.calibration.weighting == f"1/x {NEW}"
    # event names and values: free text aliased; keywords, numbers and switches as they are
    assert [(e.event, e.value) for e in got.integration["uv"].timed_events] == [
        (f"note {NEW}", f"by {NEW}"), ("Baseline now", None), ("peak_width", "0.2"),
        ("integration", "off"), ("area_reject", SECRET)]
    # numbers, settings and the model's path are left as they are
    assert got.quant.sample_amount == 123456.7 and got.quant.main_species == "mass:123456.7"
    assert got.deconvolution.model == f"models/{SECRET}.onnx"
    assert got.saturation.detector_level == 123456.0
    keyword = Method.load("rplc_pos_protein")
    assert Aliases.parse("uv=X").alias_method(keyword).quant.purity_signal == "uv"
    # a keyword setting is never aliased, nor is free text aliased into a keyword: the method
    # works as before
    keyword.instrument.kind, keyword.quant.calibration.weighting = "orbitrap", "1/x"
    keyword.integration["uv"].timed_events = [IntegrationEvent(1.0, "baseline_now", "on"),
                                              IntegrationEvent(2.0, "tof 1", "tof 1")]
    odd = Aliases.parse(["orbitrap=X", "1/x=Y", "baseline=Z", "on=W", "tof 1=split peak"])
    got = odd.alias_method(keyword)
    assert (got.instrument.kind, got.quant.calibration.weighting) == ("orbitrap", "1/x")
    assert [(e.event, e.value) for e in got.integration["uv"].timed_events] == [
        ("baseline_now", "on"), ("tof 1", "split peak")]
    # the caller's method is unchanged
    assert method.name == f"M-{SECRET}" and method.integration["uv"].area_unit == f"AU*s {SECRET}"


#: the method settings alias_method leaves as they are: keywords and names lcmsdeconv looks up
#: (an adduct, a compound class, an execution provider), the model's path; none reaches an output
NOT_ALIASED = {"instrument.shape", "instrument.saturation_kind", "acquisition.polarity",
               "deconvolution.compound_class", "deconvolution.class_candidates",
               "deconvolution.mass_type", "deconvolution.adducts.mode",
               "deconvolution.adducts.include", "deconvolution.adducts.exclude",
               "deconvolution.model", "deconvolution.providers", "regions.source",
               "integration.*.shoulders", "integration.*.tangent_skim_mode", "quant.main_species",
               "quant.calibration.mode"}


def _texts(value, path=""):
    """(setting path, text) for every text of a method's to_dict(); an integration setting's
    signal and list positions read as "*"."""
    if isinstance(value, dict):
        for k, v in value.items():
            key = "*" if path == "integration." else k
            yield from _texts(v, f"{path}{key}.")
    elif isinstance(value, list):
        for v in value:
            yield from _texts(v, path)
    elif isinstance(value, str):
        yield path[:-1], value


def test_every_method_text_is_aliased_or_listed():
    """OLD planted in every text setting of a method: after alias_method, only the settings of
    NOT_ALIASED still hold it. A text setting added to Method fails here until it is aliased or
    listed."""
    d = Method.load("rplc_pos_protein").to_dict()

    def plant(value, path=""):
        if isinstance(value, dict):
            return {k: plant(v, f"{path}{k}.") for k, v in value.items()}
        if isinstance(value, list):
            return [plant(v, path) for v in value]
        return f"{value} {SECRET}" if isinstance(value, str) else value

    d = plant(d)  # and the text settings that are null or empty lists here
    d["acquisition"]["uv_channel"] = f"UV {SECRET}"
    d["deconvolution"]["model"] = f"models/{SECRET}.onnx"
    d["deconvolution"]["adducts"]["include"] = [f"Na {SECRET}"]
    d["deconvolution"]["adducts"]["exclude"] = [f"K {SECRET}"]
    d["report"]["title"] = f"Report {SECRET}"
    for settings in d["integration"].values():
        settings["timed_events"] = [{"time": 1.0, "event": f"note {SECRET}", "value": SECRET}]
    method = Method.from_dict(d)
    planted = {path for path, text in _texts(method.to_dict()) if SECRET in text}
    assert {"name", "description", "instrument.kind", "report.title"} <= planted
    left = {path for path, text in _texts(Aliases.parse(f"{SECRET}={NEW}").alias_method(
        method).to_dict()) if SECRET in text}
    # a timed event's value that is a number ("123456") stays as it is, as numbers do
    assert left == NOT_ALIASED | {"integration.*.timed_events.value"}


def test_no_pairs_change_nothing():
    run, method, none = _run(), Method(), Aliases()
    assert none.alias_run(run) is run and none.alias_method(method) is method
    assert none.alias_chromatograms(run.chromatograms) is run.chromatograms
    assert none.apply(f"LC-{SECRET}") == f"LC-{SECRET}" and none.count(SECRET) == 0


def _uv(name, height):
    t = np.linspace(0.0, 1.0, 11)
    return Chromatogram(t, height * np.ones(11), name, "uv", "mAU")


def test_trace_references_name_the_same_traces_after_aliasing():
    """Given the run read, the method's UV channel and purity signal name the traces they
    named, under the names aliasing gave them, also where it made names equal (two OLDs for one
    NEW, a trace named NEW already); one naming no trace still names none, and a deconvolved
    trace's table ("m <mass>", named after aliasing) is named as before."""
    t = np.linspace(0.0, 1.0, 11)
    chroms = {f"UV {NEW}": _uv(f"UV {NEW}", 1.0), f"UV {SECRET}": _uv(f"UV {SECRET}", 2.0),
              "UV 654321": _uv("UV 654321", 3.0), "TIC": Chromatogram(t, np.ones(11), "TIC", "tic")}
    run = Run([], chroms, "r", "r.mzML", {})
    two = Aliases.parse([f"{SECRET}={NEW}", f"654321={NEW}"])
    assert list(two.alias_run(run).chromatograms) == [f"UV {NEW}", f"UV {NEW} (2)",
                                                      f"UV {NEW} (3)", "TIC"]
    method = Method()
    method.acquisition.uv_channel = method.quant.purity_signal = "UV 654321"
    got = two.alias_method(method, run)
    assert got.acquisition.uv_channel == got.quant.purity_signal == f"UV {NEW} (3)"
    as_text = two.alias_method(method)  # without the run: the first trace so named
    assert as_text.acquisition.uv_channel == f"UV {NEW}"
    method.acquisition.uv_channel = "UV 777777"  # no such trace: still none (the first UV)
    method.quant.purity_signal = "m 12345.67"  # a deconvolved trace's table
    got = Aliases.parse([f"{SECRET}={NEW}", "12345=X"]).alias_method(method, run)
    assert (got.acquisition.uv_channel, got.quant.purity_signal) == (None, "m 12345.67")
    # a signal naming no table, which aliasing would make name the UV trace's: still none
    lone = Run([], {f"UV {SECRET}": _uv(f"UV {SECRET}", 2.0)}, "r", "r.mzML", {})
    method.acquisition.uv_channel, method.quant.purity_signal = None, f"UV {NEW}"
    got = two.alias_method(method, lone)
    assert got.acquisition.uv_channel is None and got.quant.purity_signal is None
    method.quant.purity_signal = "uv"
    assert two.alias_method(method, lone).quant.purity_signal == "uv"
    no_uv = Run([], {"TIC": chroms["TIC"]}, "r", "r.mzML", {})
    method.acquisition.uv_channel, method.quant.purity_signal = f"UV {SECRET}", "TIC+"
    got = two.alias_method(method, no_uv)
    assert (got.acquisition.uv_channel, got.quant.purity_signal) == (None, "TIC+")


def _numbers(value):
    """The numbers of a results.json summary, in order (times and the pair count left out)."""
    if isinstance(value, dict):
        return [n for k, v in value.items() if k not in ("timings", "alias_pairs")
                for n in _numbers(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _numbers(v)]
    return [value] if isinstance(value, (int, float)) else []


def test_aliasing_quantifies_the_same_trace(planted):
    """A run with a trace named NEW already and two more whose OLDs alias to that NEW, the method
    naming the last: aliased, every number of the result is the one without aliases."""
    from lcmsdeconv.io.mzml import read_mzml
    from lcmsdeconv.process import process_run

    run = read_mzml(planted["mzml"])
    uv = run.chromatograms[f"UV {SECRET}"]
    run.chromatograms = {
        f"UV {NEW}": dataclasses.replace(uv, name=f"UV {NEW}", intensity=3.0 * uv.intensity),
        **run.chromatograms,
        "UV 654321": dataclasses.replace(uv, name="UV 654321", intensity=0.5 * uv.intensity)}
    method = Method.load(planted["method"])
    method.acquisition.uv_channel = method.quant.purity_signal = "UV 654321"
    plain = process_run(run, method)
    two = Aliases.parse([f"{SECRET}={NEW}", f"654321={NEW}"])
    aliased = process_run(run, method, aliases=two)
    assert plain.purity["signal"] == "UV 654321"
    assert aliased.purity["signal"] == f"UV {NEW} (3)"
    assert json.dumps(_numbers(aliased.summary())) == json.dumps(_numbers(plain.summary()))


def test_the_command_line_resolves_the_trace_references(planted, tmp_path, monkeypatch):
    """`process` aliases the method with the run as read: given a run whose two UV traces alias
    to one name, the method it processes names the trace the method file names."""
    import lcmsdeconv.process
    from lcmsdeconv.io.mzml import read_mzml
    from lcmsdeconv.io.mzml_writer import write_mzml

    run = read_mzml(planted["mzml"])
    uv = run.chromatograms[f"UV {SECRET}"]
    run.chromatograms["UV 654321"] = dataclasses.replace(uv, name="UV 654321")
    mzml = write_mzml(run, tmp_path / f"LC-{SECRET}-S2.mzML")
    method = Method.load(planted["method"])
    method.acquisition.uv_channel = method.quant.purity_signal = "UV 654321"
    method_file = tmp_path / "method.yaml"
    method.to_yaml(method_file)
    seen = {}

    def stop(run, method, **kw):
        seen.update(run=run, method=method)
        raise RuntimeError("stop here")

    monkeypatch.setattr(lcmsdeconv.process, "process_run", stop)
    res = CliRunner().invoke(main, ["process", str(mzml), "-m", str(method_file), "-o",
                                    str(tmp_path / "out"), "--alias", f"{SECRET}={NEW}",
                                    "--alias", f"654321={NEW}"], env={ENV_VAR: None})
    assert res.exit_code == 1 and "RuntimeError: stop here" in res.stderr
    assert list(seen["run"].chromatograms)[-2:] == [f"UV {NEW}", f"UV {NEW} (2)"]
    references = seen["method"].acquisition.uv_channel, seen["method"].quant.purity_signal
    assert references == (f"UV {NEW} (2)", f"UV {NEW} (2)")


# ---------------------------------------------------------------------- the leak check
def test_count_skips_numbers():
    a = Aliases.parse(f"{SECRET}={NEW}")
    text = f"x {SECRET} y 0.{SECRET} {SECRET}.7 1{SECRET} {SECRET}0 {SECRET},5 1,{SECRET} {SECRET}, z"
    assert a.count(text) == 2  # "x 123456 y" and "123456, z"
    assert a.mask(f"peaks_{SECRET}.5.csv") == "peaks_***.5.csv"


def test_leaks_reads_json_html_and_names(tmp_path):
    a = Aliases.parse([f"{SECRET}=X", "R&D-7=Y", "项目甲=Z"])
    (tmp_path / "results.json").write_text(json.dumps({"run": "项目甲", "area": 123456.7}),
                                           encoding="utf-8")  # written \uXXXX
    (tmp_path / "report.html").write_text(
        f'<p>R&amp;D-7</p><img src="data:image/png;base64,AAA{SECRET}AAA=">', encoding="utf-8")
    (tmp_path / f"peaks_{SECRET}.csv").write_text("rt,area\n1.0,123456.0\n", encoding="utf-8")
    (tmp_path / "clean.csv").write_text("rt,area\n1.0,1234560.0\n", encoding="utf-8")
    files = sorted(tmp_path.iterdir())
    assert sorted(a.leaks(files, tmp_path)) == [("peaks_***.csv", 1), ("report.html", 1),
                                                ("results.json", 1)]


# ---------------------------------------------------------------------- the console
def test_console_aliases_warnings_logs_and_tracebacks(capsys, caplog):
    a = Aliases.parse(f"{SECRET}={NEW}")
    log = logging.getLogger("lcmsdeconv.test_aliases")
    hooks = (warnings.showwarning, logging.lastResort, sys.excepthook)
    with a.console(), caplog.at_level(logging.WARNING):
        warnings.warn(f"LC-{SECRET}.mzML: its chromatograms were not read", UserWarning,
                      stacklevel=1)
        log.warning("reading %s", f"LC-{SECRET}.mzML")
        try:
            raise ValueError(f"no scans in LC-{SECRET}.mzML")
        except ValueError:
            sys.excepthook(*sys.exc_info())
    err = capsys.readouterr().err
    assert f"LC-{NEW}.mzML: its chromatograms were not read" in err
    assert f"ValueError: no scans in LC-{NEW}.mzML" in err and SECRET not in err
    assert caplog.records[-1].getMessage() == f"reading LC-{NEW}.mzML"
    assert (warnings.showwarning, logging.lastResort, sys.excepthook) == hooks


def _fail_in_a_thread():
    raise ValueError(f"bad LC-{SECRET}.mzML")


def test_console_covers_handlers_and_threads_started_within(capsys):
    """A log handler added within the context (logging.basicConfig, say) and an exception in a
    thread print aliased too; the hooks are restored after."""
    a = Aliases.parse(f"{SECRET}={NEW}")
    log = logging.getLogger("lcmsdeconv.test_aliases.within")
    hooks = (threading.excepthook, logging.getLogRecordFactory())
    with a.console():
        handler = logging.StreamHandler(sys.stderr)
        log.addHandler(handler)
        try:
            try:
                raise ValueError(f"bad LC-{SECRET}.mzML")
            except ValueError:
                log.exception("failed on %s", f"LC-{SECRET}.mzML")
        finally:
            log.removeHandler(handler)
        worker = threading.Thread(target=_fail_in_a_thread, name=f"reader-{SECRET}")
        worker.start()
        worker.join()
    err = capsys.readouterr().err
    assert f"failed on LC-{NEW}.mzML" in err and f"ValueError: bad LC-{NEW}.mzML" in err
    assert f"Exception in thread reader-{NEW}:" in err and SECRET not in err
    assert (threading.excepthook, logging.getLogRecordFactory()) == hooks


def test_an_error_leaving_the_console_is_printed_aliased(tmp_path):
    """An exception leaving the context (a missing run, say) is printed by Python after it,
    where no one catches it: aliased, as is one raised from it; any other is printed as
    Python prints it."""
    code = ("from lcmsdeconv.aliases import Aliases\n"
            f"a = Aliases.parse('{SECRET}=X')\n"
            "try:\n"
            "    with a.console():\n"
            f"        open('run-{SECRET}.mzML')\n"
            "except FileNotFoundError as exc:\n"
            "    raise RuntimeError('could not read the run') from exc\n")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=tmp_path, timeout=300)
    assert res.returncode == 1
    assert "No such file or directory: 'run-X.mzML'" in res.stderr
    assert "RuntimeError: could not read the run" in res.stderr and SECRET not in res.stderr
    other = code.replace("raise RuntimeError('could not read the run') from exc",
                         "pass") + f"raise KeyError('{SECRET}')\n"
    res = subprocess.run([sys.executable, "-c", other], capture_output=True, text=True,
                         cwd=tmp_path, timeout=300)
    assert res.returncode == 1 and f"KeyError: '{SECRET}'" in res.stderr


def test_console_does_not_alias_twice(capsys):
    a = Aliases.parse(f"{SECRET}=P-{SECRET}")  # a NEW holding its OLD: aliased twice it grows
    with a.console(), a.console():
        warnings.warn(f"run {SECRET}", UserWarning, stacklevel=1)
    assert f"run P-{SECRET}\n" in capsys.readouterr().err


# ---------------------------------------------------------------------- the command line
@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """A small synthetic run with 123456 in every name lcmsdeconv reads: the run's file name and
    mzML run id, its UV trace's name, the --uv-csv file's name, and a method's file name, name,
    description, UV channel, amount unit and UV area unit."""
    from lcmsdeconv.io.mzml_writer import write_mzml
    from lcmsdeconv.synth.chromatography import generate_run
    from lcmsdeconv.synth.compounds import ClassConfig
    from lcmsdeconv.synth.config import SynthConfig

    d = tmp_path_factory.mktemp("planted")
    cfg = SynthConfig(classes=[ClassConfig("peptide", (12000.0, 20000.0))], mode="rplc",
                      polarity=1, adduct_max_lambda=0.03)
    run, _truth = generate_run(cfg, np.random.default_rng(5), n_peaks=1, rt_range=(0.5, 1.3),
                               scan_rate_hz=1.0)
    uv = run.chromatograms.pop("UV1")
    uv.name = f"UV {SECRET}"
    run.chromatograms[uv.name] = uv
    run.name = f"LC-{SECRET}-S"  # the mzML run id
    mzml = write_mzml(run, d / f"LC-{SECRET}-S.mzML")
    uv_csv = d / f"uv-{SECRET}.csv"
    uv_csv.write_text("time,absorbance\n" + "".join(f"{t:.4f},{y:.6f}\n" for t, y in
                                                     zip(uv.time, 0.5 * uv.intensity)),
                      encoding="utf-8")
    method = Method.load("rplc_pos_protein")
    method.name, method.description = f"M-{SECRET}", f"for project {SECRET}"
    method.acquisition.uv_channel = f"UV {SECRET}"
    method.deconvolution.model = "comb"
    method.deconvolution.max_components = 8
    method.quant.calibration.mode = "response_factor"
    method.quant.calibration.amount_unit = f"mg/mL {SECRET}"
    method.integration["uv"].area_unit = f"AU*s {SECRET}"
    meth = d / f"method-{SECRET}.yaml"
    method.to_yaml(meth)
    return {"dir": d, "mzml": mzml, "uv_csv": uv_csv, "method": meth}


def _process(planted, out, *extra, env=None):
    args = ["process", str(planted["mzml"]), "-m", str(planted["method"]), "-o", str(out),
            "--uv-csv", str(planted["uv_csv"]), *extra]
    return CliRunner().invoke(main, args, env={ENV_VAR: env})


def _outputs(out, images=False):
    """Each output file's name (relative to ``out``) and its text (a JSON file's decoded), the
    report's embedded images left out unless ``images``."""
    found = {}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            if p.suffix == ".json":
                text = json.dumps(json.loads(text), ensure_ascii=False)
            found[p.relative_to(out).as_posix()] = text if images else DATA_URI.sub("", text)
    return found


@pytest.fixture(scope="module")
def aliased(planted):
    """`process` with the pairs from all three sources (a later pair for one OLD wins), the
    texts every plot drew recorded."""
    import matplotlib.text

    d = planted["dir"]
    env_file, alias_file = d / "env-aliases.txt", d / "aliases.txt"
    env_file.write_text(f"{SECRET}=FROM-ENV\n", encoding="utf-8")
    alias_file.write_text(f"# test\r\n{SECRET} = FROM-FILE\r\n", encoding="utf-8")
    drawn, set_text = [], matplotlib.text.Text.set_text

    def record(self, s):
        drawn.append(str(s))
        return set_text(self, s)

    out = d / f"out-{SECRET}"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(matplotlib.text.Text, "set_text", record)
        res = _process(planted, out, "--alias-file", str(alias_file), "--alias", f"{SECRET}={NEW}",
                       env=str(env_file))
    return res, out, drawn


def test_process_shows_no_old_text_anywhere(planted, aliased):
    res, out, drawn = aliased
    assert res.exit_code == 0, res.output
    assert out.is_dir()  # --out keeps its name on disk
    files = _outputs(out)
    assert {"results.json", "report.html", "species.csv", f"peaks_UV_{NEW}.csv"} <= set(files)
    for name, text in files.items():
        assert SECRET not in name and SECRET not in text, name
    assert SECRET not in res.stdout and SECRET not in res.stderr
    assert f"reading {str(planted['mzml']).replace(SECRET, NEW)}" in res.stdout
    assert res.stdout.rstrip().endswith(f"out-{NEW}/results.json")
    assert "WARNING" not in res.stderr  # the check found nothing left
    # the plots drew the aliased names, none of the old
    assert drawn and not [t for t in drawn if SECRET in t]
    assert f"UV {NEW}" in drawn


def test_process_outputs_carry_the_aliased_names(aliased):
    _res, out, _drawn = aliased
    summary = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert summary["run"] == f"LC-{NEW}-S" and summary["method"] == f"M-{NEW}"
    assert list(summary)[:3] == ["run", "method", "alias_pairs"] and summary["alias_pairs"] == 1
    # the method's UV channel still names the run's UV trace, which the purity is read from
    assert f"UV {NEW}" in summary["peak_tables"] and summary["purity"]["signal"] == f"UV {NEW}"
    assert summary["potency"]["unit"] == f"mg/mL {NEW}"
    page = (out / "report.html").read_text(encoding="utf-8")
    for text in (f"<title>LC-MS deconvolution report: LC-{NEW}-S</title>",
                 f"Method M-{NEW} &middot; for project {NEW}", f"Peak table: UV {NEW}",
                 f"Area (AU*s {NEW})", f"mg/mL {NEW}"):
        assert text in page, text


def _stable(files):
    """Output texts without what differs from run to run (the processing time)."""
    out = dict(files)
    summary = json.loads(out["results.json"])
    summary["timings"] = {}
    out["results.json"] = json.dumps(summary)
    out["report.html"] = re.sub(r"<b>Processing</b>[0-9.]+ s", "", out["report.html"])
    return out


def test_without_aliases_nothing_changes(planted, monkeypatch):
    plain = _process(planted, planted["dir"] / "plain")
    assert plain.exit_code == 0, plain.output
    files = _outputs(planted["dir"] / "plain", images=True)
    summary = json.loads(files["results.json"])
    assert summary["run"] == f"LC-{SECRET}-S" and "alias_pairs" not in summary
    assert f"peaks_UV_{SECRET}.csv" in files and f"reading {planted['mzml']}" in plain.stdout
    # an alias file without pairs changes nothing either, byte for byte (but the times)
    empty = planted["dir"] / "no-pairs.txt"
    empty.write_text("# none yet\n", encoding="utf-8")
    again = _process(planted, planted["dir"] / "plain2", "--alias-file", str(empty))
    assert again.exit_code == 0, again.output
    assert _stable(_outputs(planted["dir"] / "plain2", images=True)) == _stable(files)
    assert again.stdout.replace("plain2", "plain") == plain.stdout


def test_numbers_are_never_aliased(planted):
    """Aliases act on names before anything is formatted: an area of 123456.7 or 1234560 is
    written as it is."""
    from lcmsdeconv.cli import _write_csvs
    from lcmsdeconv.io.mzml import read_mzml
    from lcmsdeconv.process import process_run
    from lcmsdeconv.report.html import build_report

    run = read_mzml(planted["mzml"])
    method = Method.load(planted["method"])
    result = process_run(run, method, aliases=Aliases.parse(f"{SECRET}={NEW}"))
    assert run.name == f"LC-{SECRET}-S" and method.name == f"M-{SECRET}"  # not changed
    assert result.run_name == f"LC-{NEW}-S" and result.alias_pairs == 1
    # the method is processed aliased too: no OLD in the result
    m = result.method
    assert (m.name, m.description) == (f"M-{NEW}", f"for project {NEW}")
    assert m.integration["uv"].area_unit == f"AU*s {NEW}"
    summary = json.dumps(result.summary(), ensure_ascii=False)
    assert SECRET not in summary and f'"unit": "mg/mL {NEW}"' in summary
    table = result.peak_tables[f"UV {NEW}"]
    table.peaks[0].area, table.peaks[0].height = 123456.7, 1234560.0
    out = planted["dir"] / "numbers"
    out.mkdir()
    _write_csvs(result, out)
    page = build_report(result, out / "report.html", plots=False).read_text(encoding="utf-8")
    assert "123,456.7" in page and "1,234,560.0" in page and SECRET not in page
    csv = (out / f"peaks_UV_{NEW}.csv").read_text(encoding="utf-8")
    assert "123456.7" in csv and "1234560.0" in csv


def test_leftover_old_text_is_warned_of_without_quoting_it(planted):
    """A NEW that holds its OLD leaves it in every output: each file is named once on stderr,
    with the OLD masked in its name, and the command still succeeds."""
    out = planted["dir"] / "leaky"
    res = _process(planted, out, "--alias", f"{SECRET}=P{SECRET}X", "--quiet")
    assert res.exit_code == 0, res.output
    warned = [line for line in res.stderr.splitlines() if line.startswith("WARNING: ")]
    named = {line.split(" still holds ")[0][len("WARNING: "):] for line in warned}
    assert {"results.json", "report.html", "peaks_UV_P***X.csv"} <= named
    assert len(named) == len(warned) and SECRET not in res.stderr


def test_what_else_out_holds_is_checked_too(planted):
    """An earlier run's outputs in --out are not rewritten: each name there holding an OLD, and
    the text of each file named as an output, is warned of, masked; the text of other files is
    not read, and nothing is deleted."""
    out = planted["dir"] / "reused"
    out.mkdir()
    (out / f"peaks_UV_{SECRET}.csv").write_text(f"name\nUV {SECRET}\n", encoding="utf-8")
    (out / f"earlier-{SECRET}").mkdir()
    (out / "notes.txt").write_text(f"project {SECRET}\n", encoding="utf-8")
    res = _process(planted, out, "--alias", f"{SECRET}={NEW}", "--quiet", "--no-report")
    assert res.exit_code == 0, res.output
    warned = sorted(line for line in res.stderr.splitlines() if line.startswith("WARNING: "))
    assert warned == [
        "WARNING: earlier-***, which this run did not write, holds 1 occurrence of an alias's "
        "OLD text",
        "WARNING: peaks_UV_***.csv, which this run did not write, holds 2 occurrences of an "
        "alias's OLD text"]
    assert (out / f"peaks_UV_{SECRET}.csv").is_file() and (out / f"peaks_UV_{NEW}.csv").is_file()


def test_reader_warnings_are_aliased(planted, monkeypatch):
    """The reader's warning quotes the file's name; printed while aliases are set, it shows the
    aliased name."""
    import lcmsdeconv.io.mzml as mzml

    def broken(record, array):
        raise RuntimeError("damaged chromatogram")

    monkeypatch.setattr(mzml, "_array_unit", broken)
    res = _process(planted, planted["dir"] / "warned", "--alias", f"{SECRET}={NEW}",
                   "--no-report")
    assert res.exit_code == 0, res.output
    assert f"LC-{NEW}-S.mzML: its chromatograms after the first 0 were not read" in res.stderr
    assert SECRET not in res.stderr and SECRET not in res.stdout


def test_errors_are_aliased_and_clean(planted, tmp_path):
    alias = ["--alias", f"{SECRET}={NEW}"]
    missing = CliRunner().invoke(main, ["process", str(tmp_path / f"run-{SECRET}.mzML"), *alias],
                                 env={ENV_VAR: None})
    assert missing.exit_code == 2 and f"run-{NEW}.mzML" in missing.stderr
    assert SECRET not in missing.output
    no_method = _process(planted, tmp_path / "o", "-m", str(tmp_path / f"m-{SECRET}.yaml"), *alias)
    assert no_method.exit_code == 1 and f"Method not found: {tmp_path / ('m-' + NEW + '.yaml')}" \
        in no_method.stderr and SECRET not in no_method.output
    bad = tmp_path / "bad.txt"
    bad.write_text(f"{SECRET}={NEW}\nPRJ {SECRET} without equals\n", encoding="utf-8")
    broken = _process(planted, tmp_path / "o", "--alias-file", str(bad))
    assert broken.exit_code == 1 and "Traceback" not in broken.output
    assert broken.stderr.strip() == f"Error: alias file {bad}, line 2: no '=' (expected OLD=NEW)"
    env = CliRunner().invoke(main, ["report", str(tmp_path)],
                             env={ENV_VAR: str(tmp_path / "gone.txt")})
    assert env.exit_code == 1 and env.stderr.startswith(f"Error: {ENV_VAR}: alias file ")


def test_an_alias_file_error_shows_its_path_aliased_once(tmp_path):
    """A failing --alias-file's path is shown with the pairs read before it, once (a NEW holding
    its OLD would grow if they were applied twice)."""
    env = tmp_path / "env.txt"
    env.write_text(f"{SECRET}=P{SECRET}X\n", encoding="utf-8")
    bad = tmp_path / f"files-{SECRET}" / "bad.txt"
    bad.parent.mkdir()
    bad.write_text("broken\n", encoding="utf-8")
    res = CliRunner().invoke(main, ["report", str(tmp_path), "--alias-file", str(bad)],
                             env={ENV_VAR: str(env)})
    shown = str(bad).replace(SECRET, f"P{SECRET}X")
    assert res.exit_code == 1
    assert res.stderr.strip() == f"Error: alias file {shown}, line 1: no '=' (expected OLD=NEW)"


def test_report_synth_and_evaluate_show_aliased_paths(aliased, tmp_path):
    _res, out, _drawn = aliased
    shown = CliRunner().invoke(main, ["report", str(out), "--alias", f"{SECRET}={NEW}"])
    assert shown.exit_code == 0 and shown.stdout.strip() == str(out / "report.html").replace(
        SECRET, NEW) and not shown.stderr
    old = CliRunner().invoke(main, ["report", str(out), "--alias", f"{NEW}={SECRET}"])
    assert old.exit_code == 0 and "WARNING: report.html still holds" in old.stderr
    runs = tmp_path / f"runs-{SECRET}"
    made = CliRunner().invoke(main, ["synth", "--out", str(runs), "--peaks", "1", "--minutes",
                                     "0.3", "--rate", "1", "--alias", f"{SECRET}={NEW}"])
    assert made.exit_code == 0, made.output
    assert f"runs-{NEW}/run.mzML" in made.stdout and SECRET not in made.output
    gone = CliRunner().invoke(main, ["evaluate", str(tmp_path / f"none-{SECRET}"), "--alias",
                                     f"{SECRET}={NEW}"])
    assert gone.exit_code == 2 and f"none-{NEW}" in gone.stderr and SECRET not in gone.output
