# -*- coding: utf-8 -*-
"""
Module for testing gumbel class
"""

import unittest
import unittest.mock

import numpy as np
from scipy.optimize import curve_fit

from qats._validation import QatsValueError
from qats.stats.empirical import empirical_cdf
from qats.stats.gumbel import Gumbel, _euler_masceroni, bootstrap, lse, mle, msm, pwm
from qats.stats.paper import probability_paper


class EulerMascheroniTest(unittest.TestCase):
    def test_eulermascheroni(self):
        self.assertAlmostEqual(_euler_masceroni(), 0.57721566490153286060651209008240243104215933593992, places=12)


class GumbelTestCases(unittest.TestCase):
    def setUp(self):
        self.loc = 1000.0
        self.scale = 150.0

        gd = Gumbel(loc=self.loc, scale=self.scale)
        self.x = gd.rnd(size=50, seed=13)

    def test_pwm(self):
        a, b = pwm(self.x)
        self.assertLessEqual((self.loc - a) / self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)

    def test_msm(self):
        a, b = msm(self.x)
        self.assertLessEqual((self.loc - a) / self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)

    def test_lse(self):
        a, b = lse(self.x)
        self.assertLessEqual((self.loc - a) / self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)

    def test_mle(self):
        a, b = mle(self.x)
        self.assertLessEqual((self.loc - a) / self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)

    def test_fit(self):
        gumb = Gumbel.fit(self.x, method="msm")
        a, b = msm(self.x)
        self.assertEqual(gumb.loc, a)
        self.assertEqual(gumb.scale, b)


class GumbelFitInfoTestCases(unittest.TestCase):
    def setUp(self):
        self.x = Gumbel(loc=1000.0, scale=150.0).rnd(size=50, seed=13)

    def test_fit_info(self):
        for method in ("msm", "lse", "pwm", "mle"):
            with self.subTest(method=method):
                plotting_positions = (
                    "median" if method == "lse" else None
                )  # the plotting positions of an lse fit, default median
                self.assertEqual(
                    Gumbel.fit(self.x, method=method).fit_info,
                    dict(method=method, n=50, plotting_positions=plotting_positions),
                )

    def test_not_fitted(self):
        self.assertIsNone(Gumbel(1000.0, 150.0).fit_info)

    def test_method_not_case_sensitive(self):
        # a method name that isn't lowercase used to raise KeyError
        for method, f in (("PWM", pwm), ("Msm", msm), ("LSE", lse), ("mLe", mle)):
            with self.subTest(method=method):
                g = Gumbel.fit(self.x, method=method)
                self.assertEqual((g.loc, g.scale), f(self.x))
                self.assertEqual(g.fit_info["method"], method.lower())
        m, cv = bootstrap(1000.0, 150.0, 20, 5, method="MSM")
        self.assertEqual(m.shape, (2,))

    def test_invalid_method(self):
        for method in ("gumbel", None):
            with self.subTest(method=method), self.assertRaisesRegex(QatsValueError, "Method must be either"):
                Gumbel.fit(self.x, method=method)


class GumbelLseKindTestCases(unittest.TestCase):
    """
    lse() fitted to chosen plotting positions, against SciPy's least squares fit of the cdf to the same positions.
    """

    def setUp(self):
        self.x = Gumbel(loc=1000.0, scale=150.0).rnd(size=12, seed=13)

    def check(self, params, plotting_positions):
        """
        The parameters equal SciPy's least squares fit of the cdf to the given plotting positions. The objective is
        flat around the optimum, so the two solvers stop up to about 1e-5 apart; the sum of squares must not be larger.
        """
        z = np.sort(self.x)
        f = empirical_cdf(z.size, kind=plotting_positions)

        def cdf(v, a, b):
            return np.exp(-np.exp(-(v - a) / b))

        reference, _ = curve_fit(cdf, z, f, p0=(1000.0, 150.0))
        np.testing.assert_allclose(params, reference, rtol=2e-5)
        self.assertLessEqual(np.sum((cdf(z, *params) - f) ** 2), np.sum((cdf(z, *reference) - f) ** 2) * (1 + 1e-9))

    def test_kinds(self):
        for plotting_positions in ("median", "gringorten", "mean", "beard", "symmetrical"):
            with self.subTest(plotting_positions=plotting_positions):
                self.check(lse(self.x, plotting_positions=plotting_positions), plotting_positions)

    def test_default_unchanged(self):
        # the default is the median ranks lse() always used
        self.assertEqual(lse(self.x), lse(self.x, plotting_positions="median"))
        self.check(lse(self.x), "median")

    def test_kind_matters(self):
        # with 12 values the plotting positions move the fit noticeably
        self.assertGreater(
            abs(lse(self.x, plotting_positions="gringorten")[1] / lse(self.x, plotting_positions="mean")[1] - 1.0), 0.01
        )

    def test_fit_and_fit_info(self):
        g = Gumbel.fit(self.x, method="LSE", plotting_positions="gringorten")
        self.assertEqual((g.loc, g.scale), lse(self.x, plotting_positions="gringorten"))
        self.assertEqual(g.fit_info, dict(method="lse", n=12, plotting_positions="gringorten"))

    def test_invalid(self):
        with self.assertRaisesRegex(QatsValueError, "plotting positions"):
            lse(self.x, plotting_positions="hazen")
        for method in ("msm", "mle", "pwm"):
            with self.subTest(method=method), self.assertRaisesRegex(QatsValueError, "only for the method lse"):
                Gumbel.fit(self.x, method=method, plotting_positions="median")

    def test_plot_uses_fitted_positions(self):
        import matplotlib.pyplot as plt

        backend = plt.get_backend()
        plt.switch_backend("Agg")
        self.addCleanup(plt.switch_backend, backend)
        self.addCleanup(plt.close, "all")
        g = Gumbel.fit(self.x, method="lse", plotting_positions="gringorten")
        data = g.plot(ax=plt.figure().gca()).get_lines()[0]
        np.testing.assert_allclose(data.get_ydata(), empirical_cdf(12, kind="gringorten"), rtol=1e-14)


class GumbelPlotTestCases(unittest.TestCase):
    """Gumbel.plot() draws with qats.stats.paper, run with the Agg backend."""

    def setUp(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        backend = plt.get_backend()
        plt.switch_backend("Agg")
        self.addCleanup(plt.switch_backend, backend)
        self.addCleanup(plt.close, "all")
        self.g = Gumbel.fit(Gumbel(loc=1000.0, scale=150.0).rnd(size=50, seed=13), method="mle")

    def test_default_as_before(self):
        # data against median ranks, as Gumbel.ecdf, and the fitted cdf, on linear scales
        with unittest.mock.patch("matplotlib.pyplot.show") as show:
            ax = self.g.plot()
        show.assert_called_once()
        data, line = ax.get_lines()
        np.testing.assert_array_equal(data.get_xdata(), np.sort(self.g.data))
        np.testing.assert_allclose(data.get_ydata(), empirical_cdf(50, kind="median"), rtol=1e-14)
        np.testing.assert_allclose(data.get_ydata(), self.g.ecdf, rtol=1e-14)
        np.testing.assert_allclose(line.get_ydata(), self.g.cdf(line.get_xdata()), rtol=1e-9, atol=1e-12)

    def test_gumbel_paper(self):
        ax = self.g.plot(paper="gumbel", ax=self.plt.figure().gca())
        p = probability_paper(self.g.data, self.g, paper="gumbel", plotting_positions="median")
        data, line = ax.get_lines()
        np.testing.assert_array_equal(data.get_ydata(), p.y)
        np.testing.assert_array_equal(line.get_ydata(), p.fit_y)
        # a straight line on its own paper
        slope, intercept = np.polyfit(line.get_xdata(), line.get_ydata(), 1)
        np.testing.assert_allclose(slope * line.get_xdata() + intercept, line.get_ydata(), atol=1e-9)

    def test_ax_and_no_data(self):
        ax = self.plt.figure().gca()
        with unittest.mock.patch("matplotlib.pyplot.show") as show:
            result = Gumbel(1000.0, 150.0).plot(ax=ax)
        show.assert_not_called()
        self.assertIs(result, ax)
        self.assertEqual(len(ax.get_lines()), 1)

    def test_plot_linear_deprecated(self):
        with unittest.mock.patch("matplotlib.pyplot.show"):
            with self.assertWarnsRegex(DeprecationWarning, r"Gumbel.plot_linear\(\) is deprecated.*Gumbel.plot\("):
                self.g.plot_linear()


if __name__ == "__main__":
    unittest.main()
