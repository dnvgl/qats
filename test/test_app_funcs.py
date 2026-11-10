import os
import unittest

import numpy as np

from qats import TsDB
from qats.fatigue.rainflow import rebin
from qats.stats.empirical import empirical_cdf
from qats.stats.gumbel import Gumbel
from qats.stats.gumbel import pwm as gumbel_pwm
from qats.stats.paper import probability_paper
from qats.stats.weibull import Weibull

from . import realisations, requires_qt

FIT_OPTIONS = [
    dict(),
    dict(method="lse", tail=0.2),
    dict(method="mle", tail=0.2, loc="mean"),
    dict(method="msm", tail=0.2, tail_model="exceedances"),
    dict(method="mle", tail=0.2, tail_model="exceedances", loc="threshold"),
]


@requires_qt
class TestCalculateStats(unittest.TestCase):
    """The statistics and maxima/minima distribution shown in the GUI are those of the library."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "mooring.ts")
        cls.ts = TsDB.fromfile(path).get(name="mooring line 4")

    def check(self, fit, minima=False, plotting_positions="median"):
        from qats.app.funcs import calculate_stats

        s = calculate_stats({"a": self.ts}, None, None, minima=minima, fit=fit, plotting_positions=plotting_positions)[
            "a"
        ]
        # an lse fit is fitted to the plotting positions of the plot
        lse = dict(plotting_positions=plotting_positions) if fit.get("method") == "lse" else dict()
        expected = self.ts.stats(
            statsdur=10800.0, quantiles=(0.37, 0.57, 0.9), is_minima=minima, include_sample=True, **fit, **lse
        )
        for key, value in expected.items():
            if key == "sample":
                np.testing.assert_array_equal(s[key], value)
            elif isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(s[key]), key)
            else:
                self.assertEqual(s[key], value, key)

        p = probability_paper(
            expected["sample"],
            Weibull(expected["wloc"], expected["wscale"], expected["wshape"]),
            plotting_positions=plotting_positions,
            tail=fit.get("tail"),
            tail_model=fit.get("tail_model", "censored"),
            is_minima=minima,
        )
        for attr in ("x", "y", "fitted", "fit_x", "fit_y", "yticks"):
            np.testing.assert_array_equal(getattr(s["paper"], attr), getattr(p, attr), err_msg=attr)
        self.assertEqual(s["paper"].paper, "rayleigh")
        self.assertEqual(s["paper"].plotting_positions, plotting_positions)

    def test_fit_options(self):
        for fit in FIT_OPTIONS:
            with self.subTest(**fit):
                self.check(fit)

    def test_minima(self):
        for fit in (dict(), dict(method="msm", tail=0.2, tail_model="exceedances")):
            with self.subTest(**fit):
                self.check(fit, minima=True)

    def test_plotting_positions(self):
        self.check(dict(), plotting_positions="mean")

    def test_failed_fit(self):
        # mle to the exceedances with the location fitted doesn't converge on these maxima: NaN, points without a line
        from qats.app.funcs import calculate_stats

        s = calculate_stats({"a": self.ts}, None, None, fit=dict(method="mle", tail=0.2, tail_model="exceedances"))["a"]
        self.assertTrue(np.isnan(s["wloc"]))
        self.assertGreater(s["paper"].x.size, 0)
        self.assertEqual(s["paper"].fit_x.size, 0)


@requires_qt
class TestCalculateGumbelFit(unittest.TestCase):
    """
    The extremes and Gumbel distribution on the Extremes CDF tab are those of the library, for the extremes of 12
    realisations of the same process.
    """

    @classmethod
    def setUpClass(cls):
        cls.container = realisations()

    def check(self, minima, plotting_positions="median"):
        from qats.app.funcs import calculate_gumbel_fit

        g = calculate_gumbel_fit(self.container, None, None, minima=minima, plotting_positions=plotting_positions)
        extremes = np.sort([ts.x.min() if minima else ts.x.max() for ts in self.container.values()])
        np.testing.assert_array_equal(g["sample"], extremes)

        # the parameters of a pwm fit to the maxima, or to the minima multiplied by -1
        z = -extremes if minima else extremes
        self.assertEqual((g["loc"], g["scale"]), gumbel_pwm(z))

        # the sample and fitted line on Gumbel paper
        p = probability_paper(
            extremes,
            Gumbel(g["loc"], g["scale"]),
            paper="gumbel",
            plotting_positions=plotting_positions,
            is_minima=minima,
        )
        for attr in ("x", "y", "fit_x", "fit_y", "xticks", "yticks"):
            np.testing.assert_array_equal(getattr(g["paper"], attr), getattr(p, attr), err_msg=attr)
        self.assertEqual(g["paper"].xticklabels, p.xticklabels)
        self.assertEqual(
            (g["paper"].paper, g["paper"].plotting_positions, g["plotting_positions"], g["minima"]),
            ("gumbel", plotting_positions, plotting_positions, minima),
        )

        # hand calculation: the sorted (mirrored) extremes against -ln(-ln F), the fitted line (x - loc) / scale
        f = empirical_cdf(extremes.size, kind=plotting_positions)
        np.testing.assert_allclose(g["paper"].x, np.sort(z), rtol=1e-14)
        np.testing.assert_allclose(g["paper"].y, -np.log(-np.log(f)), rtol=1e-14)
        line = (g["paper"].fit_x - g["loc"]) / g["scale"]
        np.testing.assert_allclose(g["paper"].fit_y, line, rtol=1e-9, atol=1e-9)

    def test_maxima(self):
        self.check(minima=False)

    def test_minima(self):
        self.check(minima=True)

    def test_plotting_positions(self):
        self.check(minima=False, plotting_positions="gringorten")

    def test_minima_labelled_in_own_units(self):
        from qats.app.funcs import calculate_gumbel_fit

        g = calculate_gumbel_fit(self.container, None, None, minima=True)
        # the mirrored paper: x ticks at -value, labelled with the value
        labels = np.array([float(v) for v in g["paper"].xticklabels])
        np.testing.assert_allclose(g["paper"].xticks, -labels, rtol=1e-12)

    def test_lse_fitted_to_plotting_positions(self):
        # an lse fit is fitted to the plotting positions of the plot, so they change the fit
        from qats.app.funcs import calculate_gumbel_fit

        fits = dict()
        for plotting_positions in ("median", "gringorten"):
            with self.subTest(plotting_positions=plotting_positions):
                g = calculate_gumbel_fit(
                    self.container, None, None, plotting_positions=plotting_positions, method="lse"
                )
                expected = Gumbel.fit(g["sample"], method="lse", plotting_positions=plotting_positions)
                self.assertEqual((g["loc"], g["scale"]), (expected.loc, expected.scale))
                fits[plotting_positions] = (g["loc"], g["scale"])
        self.assertNotEqual(fits["median"], fits["gringorten"])
        # the other methods don't use them
        a = calculate_gumbel_fit(self.container, None, None, plotting_positions="median", method="mle")
        b = calculate_gumbel_fit(self.container, None, None, plotting_positions="gringorten", method="mle")
        self.assertEqual((a["loc"], a["scale"]), (b["loc"], b["scale"]))

    def test_methods(self):
        # each method gives the parameters of Gumbel.fit(), on the maxima or the minima multiplied by -1
        from qats.app.funcs import calculate_gumbel_fit

        for method in ("pwm", "msm", "lse", "mle"):
            for minima in (False, True):
                with self.subTest(method=method, minima=minima):
                    g = calculate_gumbel_fit(self.container, None, None, minima=minima, method=method)
                    z = -g["sample"] if minima else g["sample"]
                    expected = Gumbel.fit(z, method=method)
                    self.assertEqual((g["loc"], g["scale"]), (expected.loc, expected.scale))
                    self.assertEqual(g["method"], method)
                    np.testing.assert_allclose(
                        g["paper"].fit_y, (g["paper"].fit_x - g["loc"]) / g["scale"], rtol=1e-9, atol=1e-9
                    )

    def test_invalid_method(self):
        # an invalid method raises, and is not shown as a failed fit
        from qats._validation import QatsValueError
        from qats.app.funcs import calculate_gumbel_fit

        with self.assertRaisesRegex(QatsValueError, "must be one of"):
            calculate_gumbel_fit(self.container, None, None, method="gumbel")
        self.assertEqual(calculate_gumbel_fit(self.container, None, None, method="MLE")["method"], "mle")

    def test_failed_fit(self):
        # the same realisation twice: equal extremes, zero scale, NaN parameters and the points without a line
        from qats.app.funcs import calculate_gumbel_fit

        ts = self.container["seed 1"]
        for minima in (False, True):
            with self.subTest(minima=minima):
                g = calculate_gumbel_fit({"a": ts, "b": ts}, None, None, minima=minima)
                self.assertTrue(np.isnan(g["loc"]) and np.isnan(g["scale"]))
                self.assertEqual(g["paper"].x.size, 2)
                self.assertEqual(g["paper"].fit_x.size, 0)

    def test_too_few_series(self):
        from qats.app.funcs import calculate_gumbel_fit

        with self.assertRaises(ValueError):
            calculate_gumbel_fit({"seed 1": self.container["seed 1"]}, None, None)


@requires_qt
class TestCalculateRfc(unittest.TestCase):
    """The cycle distribution on the Cycles tab is the library's rainflow count, grouped by rebin()."""

    @classmethod
    def setUpClass(cls):
        # each series is counted on its own, so two different mooring lines
        db = TsDB.fromfile(os.path.join(os.path.dirname(__file__), "..", "data", "mooring.ts"))
        cls.container = {f"line {i}": db.get(name=f"mooring line {i}") for i in (1, 2)}

    def test_counts_and_widths(self):
        from qats.app.funcs import calculate_rfc

        for nbins in (1, 10, 256):
            out = calculate_rfc(self.container, None, None, nbins)
            for name, ts in self.container.items():
                with self.subTest(nbins=nbins, name=name):
                    r, c, w = out[name]
                    cycles = ts.rfc()
                    expected = rebin(cycles, binby="range", n=nbins)
                    np.testing.assert_array_equal(r, expected[:, 0])
                    np.testing.assert_array_equal(c, expected[:, 2])
                    # equal bins from zero to the largest range
                    np.testing.assert_allclose(w, np.full(nbins, cycles[:, 0].max() / nbins), rtol=1e-12)
                    self.assertEqual(c.sum(), cycles[:, 2].sum())

    def test_no_cycles(self):
        from qats import TimeSeries
        from qats.app.funcs import calculate_rfc

        ts = TimeSeries("flat", np.arange(10.0), np.ones(10))
        r, c, w = calculate_rfc({"flat": ts}, None, None, 10)["flat"]
        self.assertEqual((r.size, c.size, w.size), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
