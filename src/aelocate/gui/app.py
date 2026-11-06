"""Real-time AE monitor: live stream, hits, location map and a live method comparison.

Run with ``aelocate gui`` (or ``python -m aelocate.gui.app``). The acquisition and processing
loop runs in a worker thread (:mod:`aelocate.gui.worker`). The GUI thread only buffers frames
and redraws at ~15 fps, so the plotting rate never throttles acquisition.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..benchmark import calibrate, train_ml_locator
from ..pickers import AICPicker, ThresholdPicker
from ..presets import PRESETS, Setup
from ..sources import PLB
from ..units import dbae_to_volts
from .worker import AcquisitionWorker, Frame

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"standard": "#2a78d6", "aic": "#eb6834", "ml": "#1baf7a"}
METHOD_NAMES = {"standard": "Standard ΔT", "aic": "AIC picker", "ml": "Neural network"}
STREAM_SECONDS = 0.5
TABLE_ROWS = 200
TOLERANCE = 0.025


class _Trainer(QObject):
    """Trains the ML locator on a background Python thread; reports back through Qt signals."""

    done = Signal(object, object)
    failed = Signal(str)

    def start(self, setup: Setup) -> None:
        def run():
            try:
                ml, velocities = train_ml_locator(setup, n=2500, seed=0)
                self.done.emit(ml, velocities)
            except Exception as exc:  # pragma: no cover - shown in the UI
                self.failed.emit(str(exc))

        threading.Thread(target=run, daemon=True, name="ml-training").start()


def _plot(title: str, xlabel: str, ylabel: str) -> pg.PlotWidget:
    w = pg.PlotWidget()
    p = w.getPlotItem()
    p.setTitle(f"<span style='color:{INK};font-weight:600'>{title}</span>")
    p.setLabel("bottom", xlabel)
    p.setLabel("left", ylabel)
    p.showGrid(x=True, y=True, alpha=0.25)
    return w


class MainWindow(QMainWindow):
    def __init__(self, setup: Setup, model_path: str | None = None, seed: int = 1, train: bool = True,
                 start_thread: bool = True) -> None:
        super().__init__()
        self.setup = setup
        self.setWindowTitle(f"aelocate: {setup.name}")
        self.n_ch = setup.n_channels

        if setup.is_linear:
            self.velocities = calibrate(setup, np.random.default_rng(0))
        else:
            v = setup.group_velocity("S0")
            self.velocities = {"standard": v, "aic": v}
        locators = {
            "standard": setup.locator(self.velocities["standard"], ThresholdPicker()),
            "aic": setup.locator(self.velocities["aic"], AICPicker()),
        }
        self.worker = AcquisitionWorker(setup, locators, seed=seed)
        n_frames = int(round(STREAM_SECONDS * setup.fs / self.worker.block))
        self.env_t: deque[np.ndarray] = deque(maxlen=n_frames)
        self.env: deque[np.ndarray] = deque(maxlen=n_frames)
        self.hit_t: deque[float] = deque(maxlen=6000)
        self.hit_amp: deque[float] = deque(maxlen=6000)
        self.positions = {m: deque(maxlen=20000) for m in SERIES}
        self.errors = {m: deque(maxlen=400) for m in SERIES}
        self.n_located = dict.fromkeys(SERIES, 0)
        self.pending_hits: list = []
        self.last_event = None
        self.frame_info: Frame | None = None
        self.n_hits = np.zeros(self.n_ch, dtype=int)
        self.dirty = True

        self._build_ui()

        self.trainer = _Trainer()
        self.trainer.done.connect(self._on_trained)
        self.trainer.failed.connect(lambda msg: self.model_label.setText(f"training failed: {msg}"))
        if setup.is_linear:
            if model_path:
                from ..ml import MLLinearLocator

                self._on_trained(MLLinearLocator.load(model_path), None)
            elif train:
                self.model_label.setText("Training neural network on simulated\nlocation pulses + PLBs (~10 s)…")
                self.trainer.start(setup)
        else:
            self.model_label.setText("Neural network: linear set-ups only")

        self.thread = None
        if start_thread:
            self.thread = QThread(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.frame.connect(self.on_frame)
            self.thread.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.redraw)
        self.timer.start(66)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        s = self.setup
        controls = QWidget()
        controls.setFixedWidth(290)
        cl = QVBoxLayout(controls)

        acq = QGroupBox("Acquisition")
        f = QFormLayout(acq)
        self.run_btn = QPushButton("Pause")
        self.run_btn.setCheckable(True)
        self.run_btn.toggled.connect(self._toggle_pause)
        f.addRow(self.run_btn)
        self.speed = QComboBox()
        self.speed.addItems(["1× real time", "2×", "5×", "As fast as possible"])
        self.speed.currentIndexChanged.connect(self._set_speed)
        f.addRow("Speed", self.speed)
        self.threshold = QDoubleSpinBox(minimum=25, maximum=90, value=s.detector.threshold_db, suffix=" dB_AE")
        self.threshold.valueChanged.connect(self.worker.set_threshold)
        f.addRow("Threshold", self.threshold)
        rates = {e.name: e.rate_hz for e in self.worker.daq.emitters}
        self.crack_rate = QDoubleSpinBox(minimum=0, maximum=100, value=rates.get("crack", 0), suffix=" ev/s")
        self.noise_rate = QDoubleSpinBox(minimum=0, maximum=100, value=rates.get("background", 0), suffix=" ev/s")
        for w in (self.crack_rate, self.noise_rate):
            w.valueChanged.connect(lambda _: self.worker.set_rates(self.crack_rate.value(), self.noise_rate.value()))
        f.addRow("Crack AE", self.crack_rate)
        f.addRow("Background AE", self.noise_rate)
        cl.addWidget(acq)

        plb = QGroupBox("Pencil-lead break (Hsu-Nielsen)")
        f = QFormLayout(plb)
        (x_lo, x_hi), *rest = s.specimen.bounds
        self.plb_x = QDoubleSpinBox(minimum=x_lo, maximum=x_hi, value=(x_lo + x_hi) / 2, singleStep=0.05,
                                    decimals=3, suffix=" m")
        f.addRow("x", self.plb_x)
        self.plb_y = None
        if rest:
            (y_lo, y_hi), = rest
            self.plb_y = QDoubleSpinBox(minimum=y_lo, maximum=y_hi, value=(y_lo + y_hi) / 2, singleStep=0.05,
                                        decimals=3, suffix=" m")
            f.addRow("y", self.plb_y)
        btn = QPushButton("Break lead (95 dB)")
        btn.clicked.connect(self._break_lead)
        f.addRow(btn)
        cl.addWidget(plb)

        loc = QGroupBox("Location")
        f = QFormLayout(loc)
        self.method = QComboBox()
        for key in ("standard", "aic", "ml"):
            self.method.addItem(METHOD_NAMES[key], key)
        self.method.model().item(2).setEnabled(False)
        self.method.currentIndexChanged.connect(lambda _: self._mark_dirty())
        f.addRow("Map shows", self.method)
        clear = QPushButton("Clear map and statistics")
        clear.clicked.connect(self._clear)
        f.addRow(clear)
        f.addRow(QLabel(f"v = {self.velocities['standard']:.0f} m/s (threshold)\n"
                        f"v = {self.velocities['aic']:.0f} m/s (AIC)"))
        self.model_label = QLabel("")
        self.model_label.setWordWrap(True)
        f.addRow(self.model_label)
        cl.addWidget(loc)

        status = QGroupBox("Status")
        v = QVBoxLayout(status)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-family: Consolas, monospace;")
        v.addWidget(self.status_label)
        cl.addWidget(status)
        cl.addStretch(1)

        # --- plots
        self.stream = pg.GraphicsLayoutWidget()
        self.stream_curves, self.stream_plots = [], []
        thr_mv = float(dbae_to_volts(s.detector.threshold_db)) * 1e3
        self.thr_lines = []
        for ch in range(self.n_ch):
            p = self.stream.addPlot(row=ch, col=0)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setLabel("left", f"S{ch + 1}", units=None)
            p.setYRange(-8 * thr_mv, 8 * thr_mv)
            p.setMouseEnabled(x=False, y=True)
            if ch:
                p.setXLink(self.stream_plots[0])
            if ch < self.n_ch - 1:
                p.hideAxis("bottom")
            curve = p.plot(pen=pg.mkPen(INK2, width=1))
            for sign in (1, -1):
                line = pg.InfiniteLine(sign * thr_mv, angle=0, pen=pg.mkPen(MUTED, width=1, style=Qt.DotLine))
                p.addItem(line)
                self.thr_lines.append((sign, line))
            self.stream_curves.append(curve)
            self.stream_plots.append(p)
        self.stream_plots[0].setTitle(f"<span style='color:{INK};font-weight:600'>Live stream, mV "
                                      f"(last {STREAM_SECONDS:.1f} s, dotted = threshold)</span>")
        self.stream_plots[-1].setLabel("bottom", "Time, s")
        self.threshold.valueChanged.connect(self._update_threshold_lines)

        if s.is_linear:
            self.loc_plot = _plot("Location map: located events", "Position, m", "Events per cm")
            self.loc_curve = self.loc_plot.plot(stepMode="center", fillLevel=0, pen=pg.mkPen(SERIES["standard"], width=1.5),
                                                brush=pg.mkBrush(SERIES["standard"] + "55"))
            self.loc_plot.setXRange(0, s.specimen.length)
            sx = np.asarray(s.specimen.sensor_x)
            self.loc_plot.addItem(pg.ScatterPlotItem(sx, np.zeros_like(sx), symbol="t1", size=14,
                                                     brush=pg.mkBrush(INK2), pen=None))
        else:
            self.loc_plot = _plot("Location map: located events", "x, m", "y, m")
            self.loc_plot.setAspectLocked(True)
            self.loc_plot.setRange(xRange=(-0.05, s.specimen.width + 0.05), yRange=(-0.05, s.specimen.height + 0.05))
            w, h = s.specimen.width, s.specimen.height
            self.loc_plot.plot([0, w, w, 0, 0], [0, 0, h, h, 0], pen=pg.mkPen(MUTED, width=1))
            self.loc_scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(SERIES["standard"] + "88"))
            self.loc_plot.addItem(self.loc_scatter)
            sxy = np.asarray(s.specimen.sensor_xy)
            self.loc_plot.addItem(pg.ScatterPlotItem(sxy[:, 0], sxy[:, 1], symbol="t1", size=14,
                                                     brush=pg.mkBrush(INK2), pen=None))
        for e in self.worker.daq.emitters:
            if e.name == "crack":
                if s.is_linear:
                    self.loc_plot.addItem(pg.InfiniteLine(e.center[0], pen=pg.mkPen(MUTED, width=1, style=Qt.DashLine),
                                                          label="true defect", labelOpts={"color": INK2, "position": 0.9}))
                else:
                    self.loc_plot.addItem(pg.ScatterPlotItem([e.center[0]], [e.center[1]], symbol="o", size=26, brush=None,
                                                             pen=pg.mkPen(INK, width=1.5, style=Qt.DashLine)))

        self.event_plot = _plot("Last located event (blue: threshold, orange: AIC)", "Time from first arrival, µs",
                                "Sensor")
        self.amp_plot = _plot("Hits: amplitude vs time", "Time, s", "Amplitude, dB_AE")
        self.amp_scatter = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(INK2 + "99"))
        self.amp_plot.addItem(self.amp_scatter)
        self.amp_plot.setYRange(40, 100)
        self.amp_thr = pg.InfiniteLine(s.detector.threshold_db, angle=0,
                                       pen=pg.mkPen(MUTED, width=1, style=Qt.DotLine))
        self.amp_plot.addItem(self.amp_thr)

        self.acc_table = QTableWidget(3, 4)
        self.acc_table.setHorizontalHeaderLabels(["Method", "Located", "Median error", f"Within ±{TOLERANCE*1e3:.0f} mm"])
        self.acc_table.verticalHeader().setVisible(False)
        self.acc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.acc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.acc_table.setFixedHeight(118)
        for r, key in enumerate(SERIES):
            item = QTableWidgetItem("● " + METHOD_NAMES[key])
            item.setForeground(pg.mkColor(SERIES[key]))
            self.acc_table.setItem(r, 0, item)

        self.hits_table = QTableWidget(0, 7)
        self.hits_table.setHorizontalHeaderLabels(["Time, s", "Ch", "Amp, dB", "Rise, µs", "Dur, µs",
                                                   "Counts", "Energy, aJ"])
        self.hits_table.verticalHeader().setVisible(False)
        self.hits_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.hits_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        acc_box = QGroupBox("Live accuracy vs ground truth (all methods run in parallel)")
        QVBoxLayout(acc_box).addWidget(self.acc_table)
        hits_box = QGroupBox("Hits (newest first)")
        QVBoxLayout(hits_box).addWidget(self.hits_table)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(acc_box)
        rv.addWidget(hits_box, 1)

        center = QWidget()
        g = QGridLayout(center)
        g.addWidget(self.stream, 0, 0, 1, 2)
        g.addWidget(self.loc_plot, 1, 0)
        g.addWidget(self.event_plot, 1, 1)
        g.addWidget(self.amp_plot, 2, 0)
        g.addWidget(right, 2, 1)
        g.setRowStretch(0, 2 + self.n_ch // 2)
        g.setRowStretch(1, 3)
        g.setRowStretch(2, 3)

        root = QWidget()
        lay = QGridLayout(root)
        lay.addWidget(controls, 0, 0)
        lay.addWidget(center, 0, 1)
        lay.setColumnStretch(1, 1)
        self.setCentralWidget(root)
        self.resize(1600, 980)

    # -------------------------------------------------------------- actions
    def _toggle_pause(self, paused: bool) -> None:
        self.worker.set_paused(paused)
        self.run_btn.setText("Resume" if paused else "Pause")

    def _set_speed(self, i: int) -> None:
        self.worker.speed = (1.0, 2.0, 5.0, float("inf"))[i]

    def _break_lead(self) -> None:
        pos = [self.plb_x.value()] + ([self.plb_y.value()] if self.plb_y is not None else [])
        self.worker.inject(PLB, pos, 95.0)

    def _update_threshold_lines(self, db: float) -> None:
        thr_mv = float(dbae_to_volts(db)) * 1e3
        for sign, line in self.thr_lines:
            line.setValue(sign * thr_mv)
        self.amp_thr.setValue(db)

    def _clear(self) -> None:
        for m in SERIES:
            self.positions[m].clear()
            self.errors[m].clear()
            self.n_located[m] = 0
        self.hit_t.clear()
        self.hit_amp.clear()
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self.dirty = True

    def _on_trained(self, ml, velocities) -> None:
        self.worker.set_locator("ml", ml)
        self.method.model().item(2).setEnabled(True)
        self.model_label.setText("Neural network ready (MLP, 22 hit features,\ntrained on pulser + PLB events)")

    # --------------------------------------------------------------- frames
    def on_frame(self, fr: Frame) -> None:
        self.env_t.append(fr.env_t)
        self.env.append(fr.envelope)
        for h in fr.hits:
            self.hit_t.append(h.arrival_time)
            self.hit_amp.append(h.amplitude_db)
            self.n_hits[h.channel] += 1
        self.pending_hits.extend(fr.hits)
        method = self.method.currentData()
        for m, locs in fr.located.items():
            self.n_located[m] += len(locs)
            for loc in locs:
                self.positions[m].append(loc.position)
            self.errors[m].extend(fr.errors.get(m, []))
            if m == method and locs:
                self.last_event = locs[-1]
        self.frame_info = fr
        self.dirty = True

    def redraw(self) -> None:
        if not self.dirty or self.frame_info is None:
            return
        self.dirty = False
        fr = self.frame_info
        if self.env:
            t = np.concatenate(list(self.env_t))
            env = np.concatenate(list(self.env), axis=1) * 1e3
            for ch, curve in enumerate(self.stream_curves):
                curve.setData(t, env[ch])
            self.stream_plots[0].setXRange(t[0], t[-1], padding=0)

        method = self.method.currentData()
        color = SERIES[method]
        pos = list(self.positions[method])
        if self.setup.is_linear:
            edges = np.arange(0, self.setup.specimen.length + 0.01, 0.01)
            counts, _ = np.histogram([p[0] for p in pos], bins=edges)
            self.loc_curve.setData(edges, counts)
            self.loc_curve.setPen(pg.mkPen(color, width=1.5))
            self.loc_curve.setBrush(pg.mkBrush(color + "55"))
        else:
            arr = np.asarray(pos[-4000:]) if pos else np.zeros((0, 2))
            self.loc_scatter.setData(arr[:, 0], arr[:, 1], brush=pg.mkBrush(color + "88"))

        if self.hit_t:
            ht = np.fromiter(self.hit_t, float)
            self.amp_scatter.setData(ht, np.fromiter(self.hit_amp, float))
        self.amp_plot.setXRange(max(0.0, fr.t_end - 60), max(10.0, fr.t_end), padding=0.02)

        self._draw_last_event()
        self._update_tables()
        rms = " ".join(f"{r:4.0f}" for r in fr.rms_db) if fr.rms_db is not None else ""
        rate = self.n_hits / max(fr.t_end, 1e-9)
        self.status_label.setText(
            f"t        {fr.t_end:8.1f} s\n"
            f"CPU load {100 * fr.load:8.0f} % of real time\n"
            f"RMS      {rms} dB\n"
            f"hits/s   {' '.join(f'{r:4.1f}' for r in rate)}\n"
            f"located  {self.n_located[method]:8d}")

    def _draw_last_event(self) -> None:
        loc = self.last_event
        if loc is None:
            return
        p = self.event_plot.getPlotItem()
        p.clear()
        aic = AICPicker()
        t_ref = loc.event.first.arrival_time
        step = max(float(np.abs(h.waveform).max()) for h in loc.event.hits) * 1e3 * 2.2
        ticks = []
        for k, h in enumerate(sorted(loc.event.hits, key=lambda h: h.channel)):
            off = -k * step
            t = ((h.waveform_start + np.arange(h.waveform.size)) / h.fs - t_ref) * 1e6
            p.plot(t, h.waveform * 1e3 + off, pen=pg.mkPen(INK2, width=1))
            for tt, col in ((h.arrival_time, SERIES["standard"]), (aic.pick(h), SERIES["aic"])):
                x = (tt - t_ref) * 1e6
                p.plot([x, x], [off - step * 0.45, off + step * 0.45], pen=pg.mkPen(col, width=2))
            ticks.append((off, f"S{h.channel + 1}  {h.amplitude_db:.0f} dB"))
        p.getAxis("left").setTicks([ticks])
        p.setXRange(-150, 450, padding=0)
        where = f"x = {loc.x:.3f} m" if loc.y is None else f"x = {loc.x:.2f}, y = {loc.y:.2f} m"
        p.setTitle(f"<span style='color:{INK};font-weight:600'>Last located event, {where} "
                   f"(<span style='color:{SERIES['standard']}'>threshold</span>, "
                   f"<span style='color:{SERIES['aic']}'>AIC</span>)</span>")

    def _update_tables(self) -> None:
        for r, m in enumerate(SERIES):
            e = np.asarray(self.errors[m])
            vals = [f"{self.n_located[m]}",
                    f"{np.median(e) * 1e3:.1f} mm" if e.size else "–",
                    f"{100 * np.mean(e <= TOLERANCE):.0f} %" if e.size else "–"]
            for c, v in enumerate(vals, start=1):
                self.acc_table.setItem(r, c, QTableWidgetItem(v))
        new = self.pending_hits[-TABLE_ROWS:]
        self.pending_hits.clear()
        if not new:
            return
        tbl = self.hits_table
        tbl.setUpdatesEnabled(False)
        for h in new:
            tbl.insertRow(0)
            for c, v in enumerate((f"{h.arrival_time:.6f}", f"{h.channel + 1}", f"{h.amplitude_db:.1f}",
                                   f"{h.rise_time * 1e6:.0f}", f"{h.duration * 1e6:.0f}", f"{h.counts}",
                                   f"{h.energy_aj:.3g}")):
                tbl.setItem(0, c, QTableWidgetItem(v))
        while tbl.rowCount() > TABLE_ROWS:
            tbl.removeRow(tbl.rowCount() - 1)
        tbl.setUpdatesEnabled(True)

    # ------------------------------------------------------------- shutdown
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.timer.stop()
        self.worker.stop()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(2000)
        super().closeEvent(event)


def _configure(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    pg.setConfigOptions(antialias=True, background=SURFACE, foreground=INK2)


def render_screenshot(preset: str, path: str | Path, seconds: float = 20.0, train: bool = True) -> Path:
    """Run the monitor headless for ``seconds`` of stream time and save a PNG of the window."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    _configure(app)
    setup = PRESETS[preset]()
    win = MainWindow(setup, train=False, start_thread=False)
    if train and setup.is_linear:
        ml, _ = train_ml_locator(setup, n=2500, seed=0)
        win._on_trained(ml, None)
        win.method.setCurrentIndex(2)
    n_blocks = int(seconds * setup.fs / win.worker.block)
    for i in range(n_blocks):
        if i == n_blocks // 2:
            pos = [setup.specimen.sensor_x[0] + 0.2] if setup.is_linear else [0.7, 0.3]
            win.worker.inject(PLB, pos, 95.0)
        win.on_frame(win.worker.step())
    win.show()
    win.redraw()
    app.processEvents()
    path = Path(path)
    win.grab().save(str(path))
    win.close()
    return path


def main(preset: str = "plate-linear", model: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    _configure(app)
    win = MainWindow(PRESETS[preset](), model_path=model)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
