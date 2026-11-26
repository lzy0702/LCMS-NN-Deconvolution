"""Offscreen smoke tests for the desktop application (QT_QPA_PLATFORM=offscreen)."""

import os
import re

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from lcmsdeconv.core.method import IntegrationEvent  # noqa: E402
from lcmsdeconv.core.model import Chromatogram, Run, Spectrum  # noqa: E402


@pytest.fixture
def demo_run():
    rng = np.random.default_rng(0)
    spectra = []
    mz = np.linspace(500, 2000, 500)
    for i in range(20):
        prof = np.exp(-0.5 * ((np.arange(20) - 10) / 3) ** 2)[i]
        y = 1000 * prof * np.exp(-0.5 * ((mz - 1000) / 2) ** 2) + rng.random(500)
        spectra.append(Spectrum(mz, y, rt=0.1 * i, polarity=1, index=i, is_profile=True))
    t = np.linspace(0, 1.9, 40)
    uv = Chromatogram(t, np.exp(-0.5 * ((t - 1.0) / 0.3) ** 2) * 50, "UV1", "uv", "AU")
    return Run(spectra, {"UV1": uv}, name="demo")


def test_window_builds(qtbot):
    from lcmsdeconv.gui.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    assert win.method.name
    assert win.event_type.count() > 20
    assert win.species_model.columnCount() == 11


def test_load_run_and_plot(qtbot, demo_run):
    from lcmsdeconv.gui.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.run = demo_run
    win._plot_raw_chromatograms()
    assert win.rt_line.value() > 0
    win.rt_line.setValue(1.0)
    win.on_rt_moved()
    assert "Raw spectrum" in win.spectrum_plot.plotItem.titleLabel.text


def test_names_show_as_written_in_the_legend_and_error_boxes(qtbot, demo_run, monkeypatch,
                                                            tmp_path):
    """pyqtgraph's legend and Qt's message boxes take text for markup: names holding '<' or '&'
    are escaped for the one and shown as plain text in the other."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox

    from lcmsdeconv.gui.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    names = ["UV <260> PRJ#1 & Co", "UV <b>x</b> \"q\""]
    t = np.linspace(0, 1.9, 40)
    win.run = Run(demo_run.spectra, {n: Chromatogram(t, np.ones(40), n, "uv") for n in names},
                  name="r")
    win._plot_raw_chromatograms()
    legend = win.chrom_plot.plotItem.legend
    assert [label.item.toPlainText() for _sample, label in legend.items] == ["TIC", *names]

    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box) or 0)
    win.load_run(str(tmp_path / "no <b>such</b> & run.mzML"))
    assert len(shown) == 1 and shown[0].textFormat() == Qt.TextFormat.PlainText
    assert "no <b>such</b> & run.mzML" in shown[0].text()


def test_event_table_edits(qtbot):
    from lcmsdeconv.gui.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    n0 = win.events_model.rowCount()
    win.event_type.setCurrentText("integration_off")
    win.on_add_event()
    assert win.events_model.rowCount() == n0 + 1
    idx = win.events_model.index(n0, 0)
    win.events_model.setData(idx, "3.5")
    assert abs(win.events_model.events()[n0].time - 3.5) < 1e-9
    win.events_model.remove_row(n0)
    assert win.events_model.rowCount() == n0


def test_save_and_load_integration_settings(qtbot):
    from lcmsdeconv.gui.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.width_spin.setValue(0.42)
    win.events_model.add_event(IntegrationEvent(1.0, "baseline_now", None))
    win._save_integration_settings()
    settings = win.method.integration_for("tic")
    assert abs(settings.peak_width - 0.42) < 1e-9
    assert any(e.event == "baseline_now" for e in settings.timed_events)
    win._load_integration_settings()
    assert abs(win.width_spin.value() - 0.42) < 1e-9


def test_table_models_format_values(qtbot):
    from lcmsdeconv.gui.models import DictTableModel

    m = DictTableModel([("mass", "Mass"), ("adducts", "Adducts")], formats={"mass": ",.2f"})
    m.set_rows([{"mass": 12345.678, "adducts": {"base": 0.9, "+Na": 0.1}}])
    from PySide6.QtCore import Qt

    assert m.data(m.index(0, 0), Qt.DisplayRole) == "12,345.68"
    assert "base 90.0%" in m.data(m.index(0, 1), Qt.DisplayRole)


def test_gui_displays_processed_result(qtbot):
    """The window must render a real ProcessResult: tables, plots and warnings."""
    import numpy as np

    from lcmsdeconv.core.method import Method
    from lcmsdeconv.gui.app import MainWindow
    from lcmsdeconv.process import process_run
    from lcmsdeconv.synth.chromatography import generate_run
    from lcmsdeconv.synth.compounds import ClassConfig
    from lcmsdeconv.synth.config import SynthConfig

    cfg = SynthConfig(classes=[ClassConfig("peptide", (12000.0, 20000.0))], mode="rplc",
                      polarity=1, adduct_max_lambda=0.03)
    run, _ = generate_run(cfg, np.random.default_rng(5), n_peaks=1, rt_range=(0.5, 1.3),
                          scan_rate_hz=1.0)
    method = Method.load("rplc_pos_protein")
    method.deconvolution.max_components = 8
    result = process_run(run, method, model_path="comb")

    win = MainWindow()
    qtbot.addWidget(win)
    win.run = run
    win.on_processed(result)
    assert win.species_model.rowCount() > 0
    assert win.signal_box.count() > 0
    assert win.peaks_model.rowCount() >= 0
    assert win.report_btn.isEnabled()
    # selecting a species draws its deconvolved chromatogram
    win.on_species_clicked(win.species_model.index(0, 0))
    assert "Deconvolved EIC" in win.mass_plot.plotItem.titleLabel.text


def test_aliases_show_in_the_window_and_its_exports(qtbot, tmp_path, monkeypatch):
    """With alias pairs (here from $LCMSDECONV_ALIAS_FILE), the window shows NEW for OLD in its
    status bar, legend, tables and dialogs, the method it loads is aliased, it quantifies the
    trace the method names, and the report it exports holds no OLD (123456 and 654321 play
    project numbers, both aliased to one NEW)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from lcmsdeconv.aliases import ENV_VAR
    from lcmsdeconv.core.method import Method
    from lcmsdeconv.gui.app import MainWindow
    from lcmsdeconv.io.mzml_writer import write_mzml
    from lcmsdeconv.process import process_run
    from lcmsdeconv.synth.chromatography import generate_run
    from lcmsdeconv.synth.compounds import ClassConfig
    from lcmsdeconv.synth.config import SynthConfig

    cfg = SynthConfig(classes=[ClassConfig("peptide", (12000.0, 20000.0))], mode="rplc",
                      polarity=1, adduct_max_lambda=0.03)
    run, _ = generate_run(cfg, np.random.default_rng(5), n_peaks=1, rt_range=(0.5, 1.3),
                          scan_rate_hz=1.0)
    uv = run.chromatograms.pop("UV1")
    run.chromatograms["UV 123456"] = Chromatogram(uv.time, uv.intensity, "UV 123456", "uv", "mAU")
    run.chromatograms["UV 654321"] = Chromatogram(uv.time, 0.5 * uv.intensity, "UV 654321",
                                                  "uv", "mAU")
    path = write_mzml(run, tmp_path / "LC-123456-S.mzML")
    method = Method.load("rplc_pos_protein")
    method.name, method.description = "M-123456", "for project 123456"
    method.instrument.kind = "tof 123456"
    method.deconvolution.model, method.deconvolution.max_components = "comb", 8
    method.acquisition.uv_channel = method.quant.purity_signal = "UV 654321"
    method.integration["tic"].timed_events = [IntegrationEvent(0.2, "note 123456", "by 123456")]
    method_file = tmp_path / "method-123456.yaml"
    method.to_yaml(method_file)
    pairs = tmp_path / "aliases.txt"
    pairs.write_text("123456=保密项目1\n654321=保密项目1\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(pairs))

    win = MainWindow(method_name=str(method_file))
    qtbot.addWidget(win)
    assert len(win.aliases) == 2 and win.method.name == "M-保密项目1"
    events = win.events_model
    assert [events.data(events.index(0, c), Qt.DisplayRole) for c in (1, 2)] == [
        "note 保密项目1", "by 保密项目1"]
    win.load_run(str(path))
    assert win.statusBar().currentMessage().startswith("LC-保密项目1-S.mzML: ")
    legend = win.chrom_plot.plotItem.legend
    assert [label.item.toPlainText() for _sample, label in legend.items] == [
        "TIC", "UV 保密项目1", "UV 保密项目1 (2)"]

    # processed as the Process button processes: the method names the second trace still
    win.on_processed(process_run(win.run, win._method_for_run()))
    assert win.result.purity["signal"] == "UV 保密项目1 (2)"
    quantified = win.result.chromatograms["UV 保密项目1 (2)"]
    assert quantified.intensity is win._run_read.chromatograms["UV 654321"].intensity
    shown = [win.windowTitle(), win.statusBar().currentMessage(), win.warn_text.toPlainText()]
    shown += [win.signal_box.itemText(i) for i in range(win.signal_box.count())]
    for model in (win.species_model, win.peaks_model, win.events_model):
        shown += [str(model.data(model.index(r, c), Qt.DisplayRole))
                  for r in range(model.rowCount()) for c in range(model.columnCount())]
    assert "UV 保密项目1 (2)" in shown and not [s for s in shown if "123456" in s]

    out = tmp_path / "report-123456.html"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kw: (str(out), ""))
    win.on_export_report()
    assert win.statusBar().currentMessage() == (
        "report written to " + str(out).replace("123456", "保密项目1"))
    page = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]*", "", out.read_text(encoding="utf-8"))
    for text in ("LC-保密项目1-S", "M-保密项目1", "for project 保密项目1", "TOF 保密项目1",
                 "UV 保密项目1 (2)"):
        assert text in page, text
    assert "123456" not in page and "654321" not in page

    boxes = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: boxes.append(box) or 0)
    win.load_run(str(tmp_path / "missing-123456.mzML"))
    win.on_method_changed(str(tmp_path / "gone-123456.yaml"))
    win.on_failed('Traceback (most recent call last):\n  File "run-123456.py"\nValueError: 123456')
    assert [box.windowTitle() for box in boxes] == ["Could not read run", "Method",
                                                    "Processing failed"]
    assert "missing-保密项目1.mzML" in boxes[0].text() and "gone-保密项目1.yaml" in boxes[1].text()
    assert 'File "run-保密项目1.py"' in boxes[2].text()
    assert not [box.text() for box in boxes if "123456" in box.text()]
    # a method chosen later is aliased as it is loaded too
    win.on_method_changed(str(method_file))
    assert (win.method.name, win.method.instrument.kind) == ("M-保密项目1", "tof 保密项目1")


def test_gui_command_and_launch_take_the_aliases(qtbot, tmp_path, monkeypatch, capsys):
    """`lcmsdeconv gui --alias ...` hands its pairs to the application, and launch() prints an
    error that stops it from starting (a method that is not there) aliased."""
    import sys

    from click.testing import CliRunner

    import lcmsdeconv.gui.app as app
    from lcmsdeconv.aliases import ENV_VAR, Aliases
    from lcmsdeconv.cli import main

    given = {}
    monkeypatch.setattr(app, "launch", lambda *args, **kw: given.update(kw) or 0)
    res = CliRunner().invoke(main, ["gui", "--alias", "123456=保密项目1"], env={ENV_VAR: None})
    assert res.exit_code == 0, res.output
    assert given["aliases"].pairs == {"123456": "保密项目1"}
    monkeypatch.undo()

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # restored after the test
    missing = tmp_path / "nowhere-123456" / "method-123456.yaml"
    with pytest.raises(FileNotFoundError) as err:
        app.launch(None, str(missing), aliases=Aliases.parse("123456=保密项目1"))
    sys.excepthook(err.type, err.value, err.tb)  # as Python does, where no one catches it
    printed = capsys.readouterr().err
    assert "nowhere-保密项目1/method-保密项目1.yaml" in printed and "123456" not in printed
