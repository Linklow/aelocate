import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.mark.parametrize("preset", ["plate-linear", "plate-planar"])
def test_monitor_processes_frames_headless(preset):
    from PySide6.QtWidgets import QApplication

    from aelocate.gui.app import MainWindow
    from aelocate.presets import PRESETS
    from aelocate.sources import PLB

    app = QApplication.instance() or QApplication([])
    setup = PRESETS[preset]()
    win = MainWindow(setup, train=False, start_thread=False)
    win.worker.set_threshold(42.0)
    pos = [0.6] if setup.is_linear else [0.5, 0.5]
    win.worker.inject(PLB, pos, 95.0)
    for _ in range(30):
        win.on_frame(win.worker.step())
    win.redraw()
    app.processEvents()
    assert win.n_located["standard"] >= 1
    assert win.hits_table.rowCount() > 0
    assert win.worker.config.threshold_db == 42.0
    win.close()
