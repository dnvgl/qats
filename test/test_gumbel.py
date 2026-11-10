# -*- coding: utf-8 -*-
"""
Module for testing gumbel class
"""

import unittest
import unittest.mock

import numpy as np

from qats.stats.empirical import empirical_cdf
from qats.stats.gumbel import Gumbel, _euler_masceroni, lse, mle, msm, pwm
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
                self.assertEqual(Gumbel.fit(self.x, method=method).fit_info, dict(method=method, n=50))

    def test_not_fitted(self):
        self.assertIsNone(Gumbel(1000.0, 150.0).fit_info)


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
        p = probability_paper(self.g.data, self.g, paper="gumbel", kind="median")
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
