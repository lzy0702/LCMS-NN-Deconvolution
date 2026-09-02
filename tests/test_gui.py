"""Offscreen smoke tests for the desktop application (QT_QPA_PLATFORM=offscreen)."""

import os

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
    assert win.species_model.columnCount() == 10


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
