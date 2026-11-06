# -*- coding: utf-8 -*-
"""
Smoke test of the desktop application (#243): load files, plot every tab, fit a Gumbel distribution and build the
settings dialog, offscreen and without errors.

It also guards the Qt modules the application needs: they must all be in PySide6 Essentials, which QATS depends on
instead of the full PySide6.
"""

import os
import sys
import time
import unittest

from . import requires_qt
from .test_reader_snapshot import DATA

# every Qt module the application may load, with matplotlib's Qt backend; all are in PySide6 Essentials
ESSENTIALS = {"QtCore", "QtGui", "QtWidgets", "QtOpenGL", "QtOpenGLWidgets", "QtSvg"}


@requires_qt
class TestGuiSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def settle(self, window, timeout=60.0):
        """Let the worker threads finish and their results reach the main thread, which may start new workers."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            window.threadpool.waitForDone()
            for _ in range(10):
                self.app.processEvents()
            if window.threadpool.activeThreadCount() == 0:
                return
        self.fail("worker threads did not finish")

    def check_series(self, window, rows):
        from qtpy.QtCore import Qt

        model = window.db_proxy_model
        for row in rows:
            item = window.db_source_model.itemFromIndex(model.mapToSource(model.index(row, 0)))
            item.setCheckState(Qt.Checked)

    def test_load_plot_and_fit(self):
        from qats.app.gui import Qats, SettingsDialog

        window = Qats()
        files = [str(DATA / "mooring.ts"), str(DATA / "example.csv")]
        with self.assertNoLogs(level="ERROR"):
            window.load_files(files)
            self.settle(window)
            self.assertGreater(window.db_source_model.rowCount(), 3)

            # time history, power spectrum, statistics, maxima/minima CDF and cycle distribution
            self.check_series(window, range(3))
            window.on_display()
            self.settle(window)
            for name in ("history_fig", "spectrum_fig", "weibull_fig", "cycles_fig"):
                with self.subTest(name):
                    fig = getattr(window, name)
                    self.assertTrue(any(ax.lines or ax.collections or ax.patches for ax in fig.axes), name)
            self.assertGreater(window.stats_table.rowCount(), 0)

            # a Gumbel fit to the extremes of the checked series opens a new tab
            ntabs = window.tabs.count()
            window.on_create_gumbel_plot()
            self.settle(window)
            self.assertEqual(window.tabs.count(), ntabs + 1)

            dialog = SettingsDialog(False, 512, 50, 1, parent=window)
            dialog.close()

        loaded = {m.split(".")[1] for m in sys.modules if m.startswith("PySide6.Qt")}
        self.assertLessEqual(loaded, ESSENTIALS | {"QtDataVisualization"})  # qtpy tries the last one, if installed


if __name__ == "__main__":
    unittest.main()
