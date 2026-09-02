"""PySide6 desktop application: chromatograms, spectra, species tables, integration events."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.method import IntegrationEvent, Method, bundled_methods
from ..core.model import Run
from ..process import ProcessResult, process_run
from .models import DictTableModel, EventTableModel

pg.setConfigOptions(antialias=True)


def _apply_theme(app) -> None:
    """Match plot colours to the desktop palette (chromatography software is usually light)."""
    palette = app.palette()
    dark = palette.window().color().lightness() < 128
    pg.setConfigOption("background", "#16181a" if dark else "#ffffff")
    pg.setConfigOption("foreground", "#e8e8e8" if dark else "#1a1a1a")


class ProcessWorker(QObject):
    """Runs processing off the UI thread."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, run: Run, method: Method, model_path: str | None = None):
        super().__init__()
        self.run = run
        self.method = method
        self.model_path = model_path

    def run_processing(self):
        try:
            result = process_run(self.run, self.method, model_path=self.model_path,
                                 progress=lambda m: self.progress.emit(m))
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self, run_file: str | None = None, method_name: str | None = None):
        super().__init__()
        self.setWindowTitle("lcmsdeconv")
        self.resize(1280, 860)
        self.run: Run | None = None
        self.result: ProcessResult | None = None
        self.method: Method = Method.load(method_name) if method_name else Method.load("rplc_pos_protein")
        self._thread: QThread | None = None
        self._worker: ProcessWorker | None = None

        self._build_ui()
        if run_file:
            self.load_run(run_file)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        bar = QHBoxLayout()
        self.open_btn = QPushButton("Open run…")
        self.open_btn.clicked.connect(self.on_open)
        self.method_box = QComboBox()
        self.method_box.addItems(bundled_methods())
        idx = self.method_box.findText(self.method.name)
        if idx >= 0:
            self.method_box.setCurrentIndex(idx)
        self.method_box.currentTextChanged.connect(self.on_method_changed)
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.on_process)
        self.process_btn.setEnabled(False)
        self.report_btn = QPushButton("Export report…")
        self.report_btn.clicked.connect(self.on_export_report)
        self.report_btn.setEnabled(False)
        bar.addWidget(self.open_btn)
        bar.addWidget(QLabel("Method:"))
        bar.addWidget(self.method_box)
        bar.addWidget(self.process_btn)
        bar.addWidget(self.report_btn)
        bar.addStretch(1)
        outer.addLayout(bar)

        splitter = QSplitter(Qt.Vertical)
        # chromatogram + spectrum
        top = QSplitter(Qt.Horizontal)
        self.chrom_plot = pg.PlotWidget(title="Chromatogram")
        self.chrom_plot.setLabel("bottom", "Retention time", units="min")
        self.chrom_plot.addLegend(offset=(-10, 10))
        self.rt_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#b45309", width=1))
        self.rt_line.sigPositionChanged.connect(self.on_rt_moved)
        self.chrom_plot.addItem(self.rt_line)
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(27, 110, 194, 40))
        self.region.setZValue(-10)
        self.chrom_plot.addItem(self.region)
        top.addWidget(self.chrom_plot)

        right = QSplitter(Qt.Vertical)
        self.spectrum_plot = pg.PlotWidget(title="Raw spectrum")
        self.spectrum_plot.setLabel("bottom", "m/z")
        self.mass_plot = pg.PlotWidget(title="Deconvolved mass spectrum")
        self.mass_plot.setLabel("bottom", "Neutral mass", units="Da")
        right.addWidget(self.spectrum_plot)
        right.addWidget(self.mass_plot)
        top.addWidget(right)
        top.setSizes([700, 560])
        splitter.addWidget(top)

        tabs = QTabWidget()
        self.species_model = DictTableModel(
            [("name", "Species"), ("mass", "Mass (Da)"), ("delta_vs_main", "Δ vs main"),
             ("rt", "RT (min)"), ("area", "Area"), ("percent", "% of ions"),
             ("charges", "Charges"), ("mass_spread_ppm", "Spread (ppm)"),
             ("adducts", "Adducts"), ("annotation", "Annotation")],
            formats={"mass": ",.2f", "delta_vs_main": "+.3f", "rt": ".3f", "area": ",.0f",
                     "percent": ".4f", "mass_spread_ppm": ".1f"})
        self.species_view = QTableView()
        self.species_view.setModel(self.species_model)
        self.species_view.setSelectionBehavior(QTableView.SelectRows)
        self.species_view.clicked.connect(self.on_species_clicked)
        tabs.addTab(self.species_view, "Species and impurities")

        self.peaks_model = DictTableModel(
            [("id", "#"), ("rt", "RT (min)"), ("start", "Start"), ("end", "End"),
             ("area", "Area"), ("height", "Height"), ("width_half", "Width 50%"),
             ("area_pct", "Area %"), ("tailing", "Tailing"), ("code", "Code")],
            formats={"rt": ".3f", "start": ".3f", "end": ".3f", "area": ",.1f",
                     "height": ",.1f", "width_half": ".4f", "area_pct": ".3f", "tailing": ".2f"})
        peak_tab = QWidget()
        pl = QVBoxLayout(peak_tab)
        self.signal_box = QComboBox()
        self.signal_box.currentTextChanged.connect(self.on_signal_changed)
        pl.addWidget(self.signal_box)
        self.peaks_view = QTableView()
        self.peaks_view.setModel(self.peaks_model)
        self.peaks_view.setSelectionBehavior(QTableView.SelectRows)
        pl.addWidget(self.peaks_view)
        tabs.addTab(peak_tab, "Peaks")

        tabs.addTab(self._build_events_tab(), "Integration events")
        self.warn_text = QTextEdit()
        self.warn_text.setReadOnly(True)
        tabs.addTab(self.warn_text, "Warnings")
        splitter.addWidget(tabs)
        splitter.setSizes([520, 320])
        outer.addWidget(splitter)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Open an mzML run to begin")

    def _build_events_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setRange(0.0, 1e12)
        self.slope_spin.setDecimals(4)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.001, 60.0)
        self.width_spin.setDecimals(4)
        self.width_spin.setSingleStep(0.01)
        self.area_reject_spin = QDoubleSpinBox()
        self.area_reject_spin.setRange(0.0, 1e12)
        self.height_reject_spin = QDoubleSpinBox()
        self.height_reject_spin.setRange(0.0, 1e12)
        self.signal_events_box = QComboBox()
        self.signal_events_box.addItems(["tic", "uv", "deic"])
        self.signal_events_box.currentTextChanged.connect(self._load_integration_settings)
        form.addRow("Signal", self.signal_events_box)
        form.addRow("Slope sensitivity (0 = auto)", self.slope_spin)
        form.addRow("Peak width (min)", self.width_spin)
        form.addRow("Area reject", self.area_reject_spin)
        form.addRow("Height reject", self.height_reject_spin)
        lay.addLayout(form)

        self.events_model = EventTableModel()
        self.events_view = QTableView()
        self.events_view.setModel(self.events_model)
        lay.addWidget(self.events_view)

        row = QHBoxLayout()
        self.event_type = QComboBox()
        from ..chrom.events import ALL_EVENTS

        self.event_type.addItems(ALL_EVENTS)
        add = QPushButton("Add event")
        add.clicked.connect(self.on_add_event)
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self.on_remove_event)
        reint = QPushButton("Re-integrate")
        reint.clicked.connect(self.on_reintegrate)
        row.addWidget(self.event_type)
        row.addWidget(add)
        row.addWidget(rm)
        row.addStretch(1)
        row.addWidget(reint)
        lay.addLayout(row)
        self._load_integration_settings()
        return w

    # ------------------------------------------------------------------ actions
    def _load_integration_settings(self):
        s = self.method.integration_for(self.signal_events_box.currentText())
        self.slope_spin.setValue(float(s.slope_sensitivity))
        self.width_spin.setValue(float(s.peak_width))
        self.area_reject_spin.setValue(float(s.area_reject))
        self.height_reject_spin.setValue(float(s.height_reject))
        self.events_model.set_events(s.timed_events)

    def _save_integration_settings(self):
        s = self.method.integration_for(self.signal_events_box.currentText())
        s.slope_sensitivity = self.slope_spin.value()
        s.peak_width = self.width_spin.value()
        s.area_reject = self.area_reject_spin.value()
        s.height_reject = self.height_reject_spin.value()
        s.timed_events = self.events_model.events()

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open mzML", "", "mzML (*.mzML *.mzml);;All files (*)")
        if path:
            self.load_run(path)

    def load_run(self, path: str):
        from ..io.mzml import read_mzml

        self.statusBar().showMessage(f"reading {path}…")
        QApplication.processEvents()
        try:
            self.run = read_mzml(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not read run", str(exc))
            return
        self.process_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"{Path(path).name}: {len(self.run.spectra)} frames, polarities {self.run.polarities}")
        self._plot_raw_chromatograms()

    def on_method_changed(self, name: str):
        try:
            self.method = Method.load(name)
        except Exception as exc:
            QMessageBox.warning(self, "Method", str(exc))
            return
        self._load_integration_settings()

    def on_process(self):
        if self.run is None:
            return
        self._save_integration_settings()
        self.process_btn.setEnabled(False)
        self.statusBar().showMessage("processing…")
        self._thread = QThread()
        self._worker = ProcessWorker(self.run, self.method)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run_processing)
        self._worker.finished.connect(self.on_processed)
        self._worker.failed.connect(self.on_failed)
        self._worker.progress.connect(lambda m: self.statusBar().showMessage(m))
        self._thread.start()

    def on_processed(self, result):
        self.result = result
        self._thread.quit()
        self.process_btn.setEnabled(True)
        self.report_btn.setEnabled(True)
        self.species_model.set_rows(result.impurities)
        self.signal_box.blockSignals(True)
        self.signal_box.clear()
        self.signal_box.addItems(list(result.peak_tables))
        self.signal_box.blockSignals(False)
        if result.peak_tables:
            self.on_signal_changed(next(iter(result.peak_tables)))
        self.warn_text.setPlainText("\n".join(result.warnings) or "No warnings.")
        self._plot_result()
        self.statusBar().showMessage(
            f"{len(result.species)} species in {result.timings.get('total_s', 0):.1f} s")

    def on_failed(self, message: str):
        self._thread.quit()
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Processing failed", message)
        self.statusBar().showMessage("processing failed")

    def on_signal_changed(self, name: str):
        if not self.result or name not in self.result.peak_tables:
            return
        table = self.result.peak_tables[name]
        self.peaks_model.set_rows([p.to_dict() for p in table.peaks])

    def on_species_clicked(self, index):
        if not self.result:
            return
        row = self.species_model.row_dict(index.row())
        sid = row.get("id")
        sp = next((s for s in self.result.species if s.id == sid), None)
        if sp is None:
            return
        self.mass_plot.clear()
        self.mass_plot.plot(sp.time, sp.intensity, pen=pg.mkPen("#1b6ec2", width=2))
        self.mass_plot.setTitle(f"Deconvolved EIC {sp.mass:,.2f} Da")
        self.rt_line.setValue(sp.rt_apex)

    def on_rt_moved(self):
        if self.run is None:
            return
        rt = float(self.rt_line.value())
        frames = self.run.spectra
        if not frames:
            return
        i = int(np.argmin([abs(s.rt - rt) for s in frames]))
        s = frames[i]
        self.spectrum_plot.clear()
        self.spectrum_plot.plot(s.mz, s.intensity, pen=pg.mkPen("#1b6ec2", width=1))
        self.spectrum_plot.setTitle(f"Raw spectrum at {s.rt:.3f} min ({'+' if s.polarity > 0 else '-'})")

    def on_add_event(self):
        t = float(self.rt_line.value()) if self.run else 0.0
        self.events_model.add_event(IntegrationEvent(t, self.event_type.currentText(), None))

    def on_remove_event(self):
        rows = self.events_view.selectionModel().selectedRows()
        for r in sorted((r.row() for r in rows), reverse=True):
            self.events_model.remove_row(r)

    def on_reintegrate(self):
        if not self.result:
            return
        from ..chrom.integrate import integrate_chromatogram

        self._save_integration_settings()
        name = self.signal_box.currentText()
        ch = self.result.chromatograms.get(name)
        if ch is None:
            return
        kind = "uv" if ch.kind == "uv" else ("deic" if ch.kind == "deic" else "tic")
        table = integrate_chromatogram(ch, self.method.integration_for(kind), name=name)
        self.result.peak_tables[name] = table
        self.on_signal_changed(name)
        self._plot_result()
        self.statusBar().showMessage(f"re-integrated {name}: {len(table)} peaks")

    def on_export_report(self):
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save report", "report.html", "HTML (*.html)")
        if not path:
            return
        from ..report.html import build_report

        build_report(self.result, path)
        self.statusBar().showMessage(f"report written to {path}")

    # ------------------------------------------------------------------ plots
    def _plot_raw_chromatograms(self):
        self.chrom_plot.clear()
        self.chrom_plot.addItem(self.rt_line)
        self.chrom_plot.addItem(self.region)
        if self.run is None:
            return
        tic = self.run.tic()
        self.chrom_plot.plot(tic.time, tic.intensity, pen=pg.mkPen("#1b6ec2", width=1),
                             name="TIC")
        for uv in self.run.uv_traces():
            self.chrom_plot.plot(uv.time, uv.intensity, pen=pg.mkPen("#b45309", width=1),
                                 name=uv.name)
        if tic.time.size:
            self.rt_line.setValue(float(tic.time[int(np.argmax(tic.intensity))]))
            self.region.setRegion((float(tic.time.min()), float(tic.time.max())))
            self.on_rt_moved()

    def _plot_result(self):
        if not self.result:
            return
        self._plot_raw_chromatograms()
        name = self.signal_box.currentText()
        table = self.result.peak_tables.get(name)
        ch = self.result.chromatograms.get(name)
        if table and ch is not None:
            for p in table.peaks:
                self.chrom_plot.plot([p.start, p.end], [p.baseline_start, p.baseline_end],
                                     pen=pg.mkPen("#c2410c", width=1, style=Qt.DashLine))
        self.mass_plot.clear()
        if self.result.species:
            masses = np.array([s.mass for s in self.result.species])
            inten = np.array([s.total_intensity for s in self.result.species])
            for m, i in zip(masses, inten):
                self.mass_plot.plot([m, m], [0, i], pen=pg.mkPen("#1b6ec2", width=2))
            self.mass_plot.setTitle("Deconvolved mass spectrum")


def launch(run_file: str | None = None, method: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _apply_theme(app)
    win = MainWindow(run_file, method)
    win.show()
    return app.exec()
