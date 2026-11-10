# -*- coding: utf-8 -*-
"""
The Weibull fit settings of the desktop application (#154): the Statistics and Maxima/Minima CDF tabs use the same
fit, the settings dialog can't produce an invalid combination, and changed settings recalculate the shown series.
"""

import logging
import os
import tempfile
import time
import unittest
import unittest.mock

import numpy as np

from . import realisations, requires_qt
from .test_reader_snapshot import DATA

NAMES = ("Mooring line 4", "Mooring line 5")


@requires_qt
class TestGuiMaxima(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from qats.app.gui import Qats

        self.window = Qats()
        # the window logs to its own widget, which is deleted with the window
        self.addCleanup(logging.getLogger().removeHandler, self.window.logger)
        # never touch the user's settings file
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.window.settings_file = os.path.join(tmp.name, "qats.settings")
        self.window.settings = dict()
        self.window.load_files([str(DATA / "mooring.ts")])
        self.settle()

    def settle(self, timeout=60.0):
        """Let the worker threads finish and their results reach the main thread, which may start new workers."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.window.threadpool.waitForDone()
            for _ in range(10):
                self.app.processEvents()
            if self.window.threadpool.activeThreadCount() == 0:
                return
        self.fail("worker threads did not finish")

    def display(self):
        """Check the series in NAMES and display them."""
        from qtpy.QtCore import Qt

        model = self.window.db_source_model
        for row in range(model.rowCount()):
            item = model.item(row)
            if os.path.basename(item.text()) in NAMES:
                item.setCheckState(Qt.Checked)
        self.window.on_display()
        self.settle()

    def set_fit(self, fit, plotting_positions="median"):
        self.window.settings.update(
            weibull_method=fit.get("method", "pwm"),
            weibull_tail=fit.get("tail"),
            weibull_tail_model=fit.get("tail_model", "censored"),
            weibull_loc=fit.get("loc"),
            plotting_positions=plotting_positions,
        )

    def expected(self, name, fit, minima=False):
        ts = self.window.db.get(name=name)
        # an lse fit is fitted to the plotting positions of the Maxima/Minima CDF tab
        lse = dict(plotting_positions=self.window.plotting_positions()) if fit.get("method") == "lse" else dict()
        return ts.stats(
            twin=self.window.time_window(),
            filterargs=self.window.filter_settings(),
            statsdur=10800.0,
            quantiles=(0.37, 0.57, 0.9),
            is_minima=minima,
            include_sample=True,
            **fit,
            **lse,
        )

    def table(self):
        """Statistics table as {name: {key: text}}."""
        from qats.app.gui import STATS_ORDER

        t = self.window.stats_table
        rows = dict()
        for r in range(t.rowCount()):
            if t.item(r, 0) is None:
                continue
            rows[os.path.basename(t.item(r, 0).text())] = {k: t.item(r, c).text() for c, k in enumerate(STATS_ORDER)}
        return rows

    def check_same_fit(self, fit, minima=False):
        """The table and the maxima/minima CDF plot show the library's fit with these settings."""
        from qats.stats.paper import probability_paper
        from qats.stats.weibull import Weibull

        table = self.table()
        self.assertEqual(set(table), set(NAMES))
        # the plotted lines in the order of the series: fitted points, other points, fitted line (and u)
        lines = [ln for ln in self.window.weibull_axes.get_lines() if ln.get_linestyle() != "--"]
        self.assertEqual(len(lines), 3 * len(NAMES))
        for i, name in enumerate(table):
            s = self.expected(name, fit, minima=minima)
            for key in ("wloc", "wscale", "wshape", "wthreshold", "wn", "gn", "p_90.00"):
                value = s[key]
                self.assertEqual(table[name][key], "" if value is None else f"{value:12.5g}", (name, key))

            p = probability_paper(
                s["sample"],
                Weibull(s["wloc"], s["wscale"], s["wshape"]),
                tail=fit.get("tail"),
                tail_model=fit.get("tail_model", "censored"),
                is_minima=minima,
            )
            fitted, other, line = lines[3 * i : 3 * i + 3]
            np.testing.assert_array_equal(fitted.get_xdata(), p.x[p.fitted])
            np.testing.assert_array_equal(fitted.get_ydata(), p.y[p.fitted])
            np.testing.assert_array_equal(other.get_xdata(), p.x[~p.fitted])
            np.testing.assert_array_equal(line.get_xdata(), p.fit_x)
            np.testing.assert_array_equal(line.get_ydata(), p.fit_y)

    def test_same_fit_in_table_and_plot(self):
        from qats.app.gui import fit_description

        for fit in (
            dict(),
            dict(method="lse", tail=0.2),
            dict(method="mle", tail=0.2, loc="mean"),
            dict(method="msm", tail=0.2, tail_model="exceedances"),
            dict(method="mle", tail=0.2, tail_model="exceedances", loc="threshold"),
        ):
            with self.subTest(**fit):
                self.set_fit(fit)
                self.display()
                self.check_same_fit(fit)
                full = dict(dict(method="pwm", tail=None, tail_model="censored", loc=None), **fit)
                self.assertEqual(self.window.stats_fit_label.text(), fit_description(full, "median"))
                self.assertEqual(self.window.weibull_axes.get_title(), fit_description(full, "median"))

    def test_minima(self):
        fit = dict(method="msm", tail=0.2, tail_model="exceedances")
        self.set_fit(fit)
        self.window.minima.setChecked(True)
        self.display()
        self.check_same_fit(fit, minima=True)
        self.assertEqual(self.window.weibull_axes.get_xlabel(), "Minima")

    def test_plot_draws_tail(self):
        fit = dict(method="lse", tail=0.2)
        self.set_fit(fit)
        self.display()
        axes = self.window.weibull_axes
        # one legend entry per series, with u and n, then one entry for each kind of point
        labels = [t.get_text() for t in axes.get_legend().get_texts()]
        self.assertEqual(len(labels), 4)
        self.assertTrue(all("u = " in label and "n = " in label for label in labels[:2]))
        self.assertEqual(labels[2:], ["Fitted", "Not fitted"])
        # fitted points opaque, the others faded and smaller, in the series' colour
        lines = axes.get_lines()
        fitted = [ln for ln in lines if ln.get_marker() == "o" and ln.get_alpha() is None]
        faded = [ln for ln in lines if ln.get_marker() == "o" and ln.get_alpha() is not None]
        self.assertEqual(len(fitted), 2)
        self.assertEqual(len(faded), 2)
        for f, h in zip(fitted, faded):
            self.assertEqual(f.get_color(), h.get_color())
            self.assertLess(h.get_alpha(), 1.0)
            self.assertLess(h.get_markersize(), f.get_markersize())
        # the threshold value u of each series
        s = self.expected(NAMES[0], fit)
        verticals = [ln for ln in lines if ln.get_linestyle() == "--"]
        self.assertEqual(len(verticals), 2)
        self.assertIn(s["wthreshold"], [ln.get_xdata()[0] for ln in verticals])

    def test_changed_settings_recalculate(self):
        from qats.app.gui import SettingsDialog

        self.display()
        before = self.table()[NAMES[0]]["wloc"]
        fit = dict(method="msm", tail=0.2, tail_model="exceedances", loc=None)
        result = (False, 20000, 256, 2, fit, "median", dict(method="pwm", plotting_positions="median"), True)
        with unittest.mock.patch.object(SettingsDialog, "settings", return_value=result):
            self.window.on_open_settings()
        self.settle()
        self.assertNotEqual(self.table()[NAMES[0]]["wloc"], before)
        self.check_same_fit(fit)

    def test_gumbel_settings(self):
        # stored on OK, used by the next Extremes CDF tab, and the shown statistics are not recalculated
        from qats.app.gui import SettingsDialog

        self.display()
        gumbel = dict(method="mle", plotting_positions="gringorten")
        result = (False, 20000, 256, 2, self.window.weibull_fit(), self.window.plotting_positions(), gumbel, True)
        with unittest.mock.patch.object(SettingsDialog, "settings", return_value=result):
            with unittest.mock.patch.object(self.window, "start_times_series_processing_thread") as recalculate:
                self.window.on_open_settings()
        recalculate.assert_not_called()
        self.assertEqual(self.window.settings["gumbel_method"], "mle")
        self.assertEqual(self.window.settings["gumbel_plotting_positions"], "gringorten")
        self.assertEqual(self.window.gumbel_fit(), gumbel)
        # independent of the plotting positions of the Maxima/Minima CDF tab
        self.assertEqual(self.window.plotting_positions(), "median")

    def test_cancelled_settings_keep_table(self):
        from qats.app.gui import SettingsDialog

        self.display()
        before = self.table()
        fit = dict(method="msm", tail=0.2, tail_model="exceedances", loc=None)
        with unittest.mock.patch.object(
            SettingsDialog,
            "settings",
            return_value=(
                False,
                20000,
                256,
                2,
                fit,
                "median",
                dict(method="mle", plotting_positions="gringorten"),
                False,
            ),
        ):
            self.window.on_open_settings()
        self.settle()
        self.assertEqual(self.table(), before)
        self.assertNotIn("weibull_method", self.window.settings)
        self.assertNotIn("gumbel_method", self.window.settings)
        self.assertNotIn("gumbel_plotting_positions", self.window.settings)

    def test_invalid_stored_settings(self):
        from qats.app.gui import WEIBULL_FIT_DEFAULTS

        for settings in (
            dict(weibull_method="pwm2"),
            dict(weibull_method="pwm", weibull_tail=0.2),  # no censored pwm fit
            dict(weibull_tail=2.0, weibull_method="lse"),
            dict(weibull_loc="threshold"),
        ):
            with self.subTest(**settings):
                self.window.settings = dict(settings)
                with self.assertLogs(level="WARNING"):
                    self.assertEqual(self.window.weibull_fit(), WEIBULL_FIT_DEFAULTS)
        self.window.settings = dict(plotting_positions="hazen")
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.window.plotting_positions(), "median")
        for settings, expected in (
            (dict(gumbel_method="gumbel"), dict(method="pwm", plotting_positions="median")),
            (
                dict(gumbel_method="PWM", gumbel_plotting_positions="beard"),
                dict(method="pwm", plotting_positions="beard"),
            ),
            (
                dict(gumbel_method=None, gumbel_plotting_positions="mle"),
                dict(method="pwm", plotting_positions="median"),
            ),
            (
                dict(gumbel_method="lse", gumbel_plotting_positions="hazen"),
                dict(method="lse", plotting_positions="median"),
            ),
        ):
            with self.subTest(**settings):
                self.window.settings = dict(settings)
                with self.assertLogs(level="WARNING"):
                    self.assertEqual(self.window.gumbel_fit(), expected)
        self.window.settings = dict()
        self.assertEqual(self.window.gumbel_fit(), dict(method="pwm", plotting_positions="median"))

    def container(self):
        return {name: self.window.db.get(name=name) for name in NAMES}

    def extremes_axes(self):
        """Axes of the last Extremes CDF tab."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        tabs = self.window.tabs
        index = max(i for i in range(tabs.count()) if tabs.tabText(i) == "Extremes CDF")
        return tabs.widget(index).findChild(FigureCanvasQTAgg).figure.axes[0]

    def test_extremes_cdf(self):
        # the tab follows the maxima/minima selection and its own Gumbel fit method and plotting positions, and draws
        # what the library returns, for the extremes of 12 realisations of the same process
        from qats.app.funcs import calculate_gumbel_fit

        container = realisations()
        self.window.settings["plotting_positions"] = "symmetrical"  # of the Maxima/Minima CDF tab, not used here
        self.window.settings["gumbel_plotting_positions"] = "mean"
        for minima, method in ((False, "pwm"), (True, "pwm"), (False, "mle"), (True, "lse")):
            with self.subTest(minima=minima, method=method):
                self.window.settings["gumbel_method"] = method
                self.window.minima.setChecked(minima)
                self.window.maxima.setChecked(not minima)
                self.window.start_gumbel_fit_thread(container)
                self.settle()
                axes = self.extremes_axes()
                g = calculate_gumbel_fit(
                    container,
                    self.window.time_window(),
                    self.window.filter_settings(),
                    minima=minima,
                    plotting_positions="mean",
                    method=method,
                )
                self.assertIn(f"Gumbel fit: {method},", axes.get_title())
                p = g["paper"]
                points, line = axes.get_lines()
                np.testing.assert_array_equal(points.get_xdata(), p.x)
                np.testing.assert_array_equal(points.get_ydata(), p.y)
                np.testing.assert_array_equal(line.get_xdata(), p.fit_x)
                np.testing.assert_array_equal(line.get_ydata(), p.fit_y)
                self.assertEqual(points.get_color(), line.get_color())
                np.testing.assert_array_equal(axes.get_yticks(), p.yticks)
                self.assertEqual([t.get_text() for t in axes.get_xticklabels()], p.xticklabels)
                self.assertIn("minima" if minima else "maxima", axes.get_title())
                self.assertIn("mean plotting positions", axes.get_title())
                self.assertGreater(p.fit_x.size, 0)

    def test_extremes_cdf_failed_fit(self):
        # the same realisation twice: equal extremes, so the Gumbel scale is zero: the points without a line
        ts = realisations()["seed 1"]
        with self.assertLogs(level="WARNING") as logs:
            self.window.start_gumbel_fit_thread({"a": ts, "b": ts})
            self.settle()
        self.assertTrue(any("could not be fitted" in m for m in logs.output))
        points, line = self.extremes_axes().get_lines()
        self.assertEqual(points.get_xdata().size, 2)
        self.assertEqual(line.get_xdata().size, 0)

    def test_cycle_distribution(self):
        # the bars are the library's cycle counts, also with a single bin
        from qats.app.funcs import calculate_rfc

        for nbins in (1, 16):
            with self.subTest(nbins=nbins):
                rfc = calculate_rfc(self.container(), None, None, nbins)
                self.window.plot_rfc(rfc)
                bars = self.window.cycles_axes.patches
                self.assertEqual(len(bars), nbins * len(NAMES))
                for i, name in enumerate(NAMES):
                    r, c, w = rfc[name]
                    series = bars[i * nbins : (i + 1) * nbins]
                    np.testing.assert_allclose([b.get_x() + 0.5 * b.get_width() for b in series], r, rtol=1e-12)
                    np.testing.assert_array_equal([b.get_height() for b in series], c)
                    np.testing.assert_allclose([b.get_width() for b in series], w, rtol=1e-12)


@requires_qt
class TestGuiOptionsMatchLibrary(unittest.TestCase):
    """The fit options the GUI offers are the library's, so they can't drift apart."""

    def test_options(self):
        from qats.app.gui import GUMBEL_DEFAULTS, GUMBEL_METHODS, PLOTTING_POSITIONS, TAIL_MODELS, WEIBULL_METHODS
        from qats.stats.empirical import _PLOTTING_POSITIONS, _TAIL_MODELS
        from qats.stats.gumbel import _FIT_METHODS as GUMBEL_FIT_METHODS
        from qats.stats.weibull import _FIT_FUNCTIONS

        self.assertEqual(set(TAIL_MODELS), set(_TAIL_MODELS))
        self.assertEqual(set(PLOTTING_POSITIONS), set(_PLOTTING_POSITIONS))
        self.assertLessEqual(set(WEIBULL_METHODS), set(_FIT_FUNCTIONS))
        self.assertEqual(set(GUMBEL_METHODS), set(GUMBEL_FIT_METHODS))
        self.assertIn(GUMBEL_DEFAULTS["method"], GUMBEL_METHODS)
        self.assertIn(GUMBEL_DEFAULTS["plotting_positions"], PLOTTING_POSITIONS)


@requires_qt
class TestSettingsDialogFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def dialog(self, fit=None, plotting_positions="median"):
        from qats.app.gui import SettingsDialog

        d = SettingsDialog(False, 512, 50, 1, fit=fit, plotting_positions=plotting_positions)
        self.addCleanup(d.close)
        return d

    def enabled(self, combo):
        model = combo.model()
        return {combo.itemData(i): model.item(i).isEnabled() for i in range(combo.count())}

    def test_defaults(self):
        d = self.dialog()
        self.assertEqual(d.get_fit(), dict(method="pwm", tail=None, tail_model="censored", loc=None))
        self.assertEqual(d.get_plotting_positions(), "median")
        self.assertEqual(d.get_gumbel(), dict(method="pwm", plotting_positions="median"))
        self.assertFalse(d.tailspinbox.isEnabled())
        self.assertFalse(d.tailmodelcombo.isEnabled())
        self.assertFalse(d.locspinbox.isEnabled())

    def test_gumbel_round_trip(self):
        from qats.app.gui import SettingsDialog

        for gumbel in (
            dict(method="pwm", plotting_positions="gringorten"),
            dict(method="msm", plotting_positions="median"),
            dict(method="lse", plotting_positions="mean"),
            dict(method="mle", plotting_positions="beard"),
        ):
            with self.subTest(**gumbel):
                # independent of the plotting positions of the maxima/minima distribution
                d = SettingsDialog(False, 512, 50, 1, plotting_positions="symmetrical", gumbel=gumbel)
                self.addCleanup(d.close)
                self.assertEqual(d.get_gumbel(), gumbel)
                self.assertEqual(d.get_plotting_positions(), "symmetrical")
                self.assertTrue(d.gumbelmethodcombo.isEnabled())
                self.assertTrue(d.gumbelplottingpositionscombo.isEnabled())

    def test_censored_tail_disables_pwm_and_msm(self):
        d = self.dialog()
        d.tailcheckbox.setChecked(True)
        self.assertEqual(self.enabled(d.methodcombo), dict(pwm=False, msm=False, lse=True, mle=True))
        self.assertEqual(d.methodcombo.currentData(), "lse")  # moved from pwm
        self.assertIn("no censored tail fit", d.methodcombo.model().item(0).toolTip())
        self.assertEqual(d.get_fit(), dict(method="lse", tail=0.2, tail_model="censored", loc=None))

        # every method fits the exceedances
        d.tailmodelcombo.setCurrentIndex(d.tailmodelcombo.findData("exceedances"))
        self.assertTrue(all(self.enabled(d.methodcombo).values()))

    def test_threshold_only_for_exceedances(self):
        d = self.dialog(fit=dict(method="mle", tail=0.2, tail_model="exceedances", loc="threshold"))
        self.assertTrue(self.enabled(d.loccombo)["threshold"])
        self.assertEqual(d.get_fit(), dict(method="mle", tail=0.2, tail_model="exceedances", loc="threshold"))
        d.tailmodelcombo.setCurrentIndex(d.tailmodelcombo.findData("censored"))
        self.assertFalse(self.enabled(d.loccombo)["threshold"])
        self.assertIsNone(d.get_fit()["loc"])  # moved back to fitted
        d.tailcheckbox.setChecked(False)
        self.assertFalse(self.enabled(d.loccombo)["threshold"])

    def test_round_trip(self):
        from qats.ts import _check_weibull_options

        for fit, plotting_positions in (
            (dict(method="msm", tail=0.15, tail_model="exceedances", loc=None), "mean"),
            (dict(method="lse", tail=0.3, tail_model="censored", loc="mean"), "gringorten"),
            (dict(method="mle", tail=None, tail_model="censored", loc=120.5), "beard"),
        ):
            with self.subTest(**fit):
                d = self.dialog(fit=fit, plotting_positions=plotting_positions)
                self.assertEqual(d.get_fit(), fit)
                self.assertEqual(d.get_plotting_positions(), plotting_positions)
                _check_weibull_options(**d.get_fit())

    def test_value_location(self):
        d = self.dialog()
        d.loccombo.setCurrentIndex(d.loccombo.findData("value"))
        self.assertTrue(d.locspinbox.isEnabled())
        d.locspinbox.setValue(-35.25)
        self.assertEqual(d.get_fit()["loc"], -35.25)


if __name__ == "__main__":
    unittest.main()
