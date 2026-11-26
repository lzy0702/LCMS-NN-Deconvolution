"""Names in the HTML report and the peak-table CSV files are shown as written.

Run, method and chromatogram names carry project numbers and whatever else a lab puts in them
('&', '#', '/', '<', quotes, '$'): the report escapes every such text, so it shows them verbatim
and its element structure is the same as with a plain name, and its plots draw them literally.
"""

import unicodedata
import warnings
from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lcmsdeconv.chrom.integrate import integrate_chromatogram
from lcmsdeconv.cli import _csv_names, _write_csvs
from lcmsdeconv.core.method import IntegrationSettings, Method
from lcmsdeconv.core.model import Chromatogram, Peak, PeakTable, Species
from lcmsdeconv.process import ProcessResult
from lcmsdeconv.quant.saturation import SaturationReport
from lcmsdeconv.report import html as report

NAMES = ["PRJ-000000-SAMPLE+18", "PRJ#2026/001 & Co", "<b>x</b> \"q\" 'a'", "$\\alpha$_1"]


def _result(name: str) -> ProcessResult:
    """A processed run whose run, method, trace, species and other texts all carry ``name``."""
    method = Method.load("iprp_neg_oligo")
    method.name, method.description = f"M {name}", f"for {name}"
    t = np.linspace(0.0, 4.0, 401)
    y = (100.0 * np.exp(-0.5 * ((t - 1.5) / 0.04) ** 2)
         + 40.0 * np.exp(-0.5 * ((t - 2.5) / 0.04) ** 2))
    traces = [Chromatogram(t, y, f"UV {name}", "uv", "AU"),
              Chromatogram(t, 1e4 * y, f"TIC {name}", "tic", "counts")]
    settings = IntegrationSettings(area_unit=f"AU*s {name}")
    tables = {c.name: integrate_chromatogram(c, settings) for c in traces}
    for table in tables.values():
        for p in table.peaks:
            p.code = f"BB {name}"
    species = Species(1, 7000.0, "average", -1, "ps_rna", t, y, name=f"API {name}")
    impurity = {"name": f"n-1 {name}", "mass": 6700.1, "delta_vs_main": -299.9, "rt": 1.5,
                "area": 10.0, "area_all_forms": 12.0, "percent": 1.0, "charges": f"5-9 {name}",
                "mass_spread_ppm": 3.0, "adducts": {f"TEA {name}": 0.1},
                "annotation": f"main component {name}", "flags": [f"flag {name}"]}
    return ProcessResult(
        run_name=f"R {name}", method=method, species=[species], peak_tables=tables,
        chromatograms={c.name: c for c in traces}, impurities=[impurity],
        purity={"rows": [{"id": 1, "rt": 1.5, "area": 10.0, "area_pct": 71.4,
                          "code": f"BB {name}", "is_main": True}]},
        potency={"area": 10.0, "amount": 2.0, "unit": f"mg {name}", "percent_of_nominal": 99.0,
                 "calibration": {"mode": f"linear {name}", "r_squared": 0.999}},
        saturation=SaturationReport(peak_flags={1: [f"saturated {name}"]}),
        warnings=[f"warning about {name}"], rt_window=[0.5, None])


class _Outline(HTMLParser):
    """The elements of a page, with their attribute names, and its text."""

    def __init__(self, page: str):
        super().__init__(convert_charrefs=True)
        self.elements, self.text = [], []
        self.feed(page)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, tuple(k for k, _v in attrs)))

    def handle_endtag(self, tag):
        self.elements.append(("/" + tag,))

    def handle_data(self, data):
        self.text.append(data)


@pytest.mark.parametrize("name", NAMES)
def test_names_show_as_written_in_the_same_page_structure(name, tmp_path):
    plain = _Outline(report.build_report(_result("plain"), tmp_path / "plain.html",
                                         plots=False).read_text(encoding="utf-8"))
    page = report.build_report(_result(name), tmp_path / "r.html", plots=False)
    got = _Outline(page.read_text(encoding="utf-8"))
    assert got.elements == plain.elements
    shown = set(got.text)
    for text in (f"LC-MS deconvolution report: R {name}", f"R {name}", f"M {name}",
                 f"Peak table: UV {name}", f"Peak table: TIC {name}", f"Area (AU*s {name})",
                 f"BB {name}", f"n-1 {name}", f"5-9 {name}", f"TEA {name} 10.0%",
                 f"main component {name}", f" warning about {name}", f" saturated {name}",
                 "Peak 1.", f"2.0000 mg {name}", f"linear {name} r²=0.99900"):
        assert text in shown, text
    assert f"Method M {name} · for {name}" in shown


def test_ordinary_names_come_out_unchanged(tmp_path):
    name = "PRJ-000000-SAMPLE+18"
    page = report.build_report(_result(name), tmp_path / "r.html", plots=False).read_text()
    assert f"<title>LC-MS deconvolution report: R {name}</title>" in page
    assert f"<div><b>Run</b>R {name}</div>" in page and f"<h2>Peak table: UV {name}</h2>" in page


@pytest.mark.parametrize("name", ["$\\alpha$_1", "$x_$"])  # mathtext, and none matplotlib parses
def test_plots_draw_names_as_written(name, tmp_path, monkeypatch):
    figures, to_uri = [], report._fig_to_uri

    def keep(fig):
        figures.append(fig)
        return to_uri(fig)

    monkeypatch.setattr(report, "_fig_to_uri", keep)
    page = report.build_report(_result(name), tmp_path / "r.html").read_text()
    assert page.count('<img src="data:image/png;base64,') == 3
    labels = [ax.yaxis.label for ax in figures[0].axes]
    assert sorted(label.get_text() for label in labels) == [f"TIC {name}", f"UV {name}"]
    assert not any(label.get_parse_math() for label in labels)
    legend = figures[1].axes[0].get_legend().get_texts()
    assert [t.get_text() for t in legend] == [f"7,000.0 Da (API {name})"]
    assert not legend[0].get_parse_math()


def test_plots_fall_back_to_a_cjk_font_for_names_in_chinese(tmp_path):
    from matplotlib import font_manager

    if not {f.name for f in font_manager.fontManager.ttflist} & set(report.CJK_FONTS):
        pytest.skip("no CJK font of report.CJK_FONTS is installed")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report.build_report(_result("测试样品1"), tmp_path / "r.html")
    assert not [w for w in caught if "missing from font" in str(w.message)]


def test_peak_table_csv_names_are_unique():
    names = ["TIC", "UV 260", "UV_260", "uv 260", "UV_260_2", "UV/260", "TUV 260 nm"]
    parts = _csv_names(names)
    assert parts["TIC"] == "TIC" and parts["UV 260"] == "UV_260"  # as before where unique
    assert parts["UV_260_2"] == "UV_260_2" and parts["TUV 260 nm"] == "TUV_260_nm"
    assert len({p.casefold() for p in parts.values()}) == len(names)
    assert _csv_names(["TIC", "UV1"]) == {"TIC": "TIC", "UV1": "UV1"}


def test_peak_table_csv_names_are_composed_and_short():
    nfc = "UV é 한"
    nfd = unicodedata.normalize("NFD", nfc)  # the same name, decomposed: one file on macOS
    long_a, long_b = "UV " + "x" * 300, "UV " + "x" * 299 + "y"
    parts = _csv_names([nfc, nfd, long_a, long_b, "UV " + "保" * 60])
    assert parts[nfc] == "UV_é_한" and parts[nfd] == "UV_é_한_2"
    assert all(len(p.encode("utf-8")) <= 100 for p in parts.values())
    assert parts[long_a] != parts[long_b] and parts[long_a].startswith("UV_xxx")
    assert parts["UV " + "保" * 60].startswith("UV_保保")


def test_each_peak_table_gets_its_own_csv(tmp_path):
    names = ["UV 260", "UV_260", "uv/260", "UV " + "x" * 300]
    tables = {n: PeakTable([Peak(rt=1.0 + i, start=0.9 + i, end=1.1 + i, area=1.0, height=1.0)], n)
              for i, n in enumerate(names)}
    _write_csvs(SimpleNamespace(impurities=[], peak_tables=tables), tmp_path)
    parts = _csv_names(names)
    assert len(list(tmp_path.glob("peaks_*.csv"))) == 4
    for i, n in enumerate(names):
        assert pd.read_csv(tmp_path / f"peaks_{parts[n]}.csv")["rt"].tolist() == [1.0 + i]
