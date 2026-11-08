import os
import unittest

import numpy as np

from qats import TsDB
from qats._validation import QatsTypeError, QatsValueError
from qats.stats.gumbel import Gumbel
from qats.stats.paper import paper_coordinates, probability_paper
from qats.stats.weibull import Weibull


def mooring_series():
    """'mooring line 4' in data/mooring.ts."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "mooring.ts")
    return TsDB.fromfile(path).get(name="mooring line 4")


def residual(x, y):
    """Sum of squared residuals of a straight line fitted to x, y."""
    return np.polyfit(x, y, 1, full=True)[1][0]


class PaperCoordinatesTestCases(unittest.TestCase):
    """Paper coordinates of a small sample, against hand calculations."""

    def setUp(self):
        self.x = np.array([1.0, 2.0, 4.0])
        self.f = np.array([0.7, 1.7, 2.7]) / 3.4  # median ranks (i - 0.3) / (n + 0.4)

    def test_rayleigh(self):
        p = probability_paper(self.x)
        np.testing.assert_array_equal(p.x, self.x)
        np.testing.assert_allclose(p.y, np.sqrt(-np.log(1.0 - self.f)), rtol=1e-14)

    def test_weibull(self):
        p = probability_paper(self.x, paper="weibull", loc=0.5)
        np.testing.assert_allclose(p.x, np.log([0.5, 1.5, 3.5]), rtol=1e-14)
        np.testing.assert_allclose(p.y, np.log(-np.log(1.0 - self.f)), rtol=1e-14)

    def test_gumbel(self):
        p = probability_paper(self.x, paper="gumbel")
        np.testing.assert_array_equal(p.x, self.x)
        np.testing.assert_allclose(p.y, -np.log(-np.log(self.f)), rtol=1e-14)

    def test_mean_ranks(self):
        p = probability_paper(self.x, kind="mean")
        np.testing.assert_allclose(p.y, np.sqrt(-np.log(1.0 - np.array([0.25, 0.5, 0.75]))), rtol=1e-14)

    def test_unsorted_sample(self):
        p = probability_paper([4.0, 1.0, 2.0])
        np.testing.assert_array_equal(p.sample, self.x)
        np.testing.assert_array_equal(p.x, self.x)

    def test_paper_coordinates(self):
        px, py = paper_coordinates(self.x, self.f, paper="weibull", loc=0.5)
        p = probability_paper(self.x, paper="weibull", loc=0.5)
        np.testing.assert_array_equal(px, p.x)
        np.testing.assert_array_equal(py, p.y)

    def test_paper_coordinates_below_loc(self):
        px, _ = paper_coordinates(self.x, self.f, paper="weibull", loc=2.0)
        self.assertTrue(np.all(np.isnan(px[:2])))
        self.assertAlmostEqual(px[2], np.log(2.0), delta=1e-14)

    def test_no_fit(self):
        p = probability_paper(self.x)
        self.assertEqual(p.fit_x.size, 0)
        self.assertEqual(p.fit_y.size, 0)
        self.assertTrue(np.all(p.fitted))
        self.assertEqual(p.excluded.size, 0)


class PaperStraightLineTestCases(unittest.TestCase):
    """Each paper's own model plots as a straight line, other models as curves."""

    def test_rayleigh(self):
        w = Weibull(0.0, 3.0, 2.0)
        p = probability_paper(w.rnd(200, seed=1), w)
        self.assertEqual(p.fit_x.size, 500)
        self.assertLess(residual(p.fit_x, p.fit_y), 1e-20)

    def test_rayleigh_with_location(self):
        w = Weibull(10.0, 3.0, 2.0)
        p = probability_paper(w.rnd(200, seed=1), w)
        self.assertLess(residual(p.fit_x, p.fit_y), 1e-20)

    def test_weibull(self):
        w = Weibull(10.0, 3.0, 1.4)
        p = probability_paper(w.rnd(200, seed=2), w, paper="weibull")
        self.assertEqual(p.loc, 10.0)
        self.assertLess(residual(p.fit_x, p.fit_y), 1e-20)

    def test_gumbel(self):
        g = Gumbel(5.0, 2.0)
        p = probability_paper(g.rnd(100, seed=3), g, paper="gumbel")
        self.assertLess(residual(p.fit_x, p.fit_y), 1e-20)

    def test_curved_on_other_papers(self):
        w = Weibull(0.0, 3.0, 1.4)
        s = w.rnd(200, seed=4)
        for paper in ("rayleigh", "gumbel"):
            with self.subTest(paper=paper):
                p = probability_paper(s, w, paper=paper)
                self.assertGreater(residual(p.fit_x, p.fit_y), 1.0)

    def test_fitted_line_values(self):
        # the line is the fitted quantile function on the paper's axes
        w = Weibull(1.0, 3.0, 1.4)
        p = probability_paper(w.rnd(50, seed=5), w, paper="gumbel")
        np.testing.assert_allclose(p.fit_y, -np.log(-np.log(w.cdf(p.fit_x))), rtol=1e-10)

    def test_line_spans_points(self):
        w = Weibull(0.0, 3.0, 2.0)
        p = probability_paper(w.rnd(200, seed=1), w)
        self.assertAlmostEqual(p.fit_y[0], p.y[0], delta=1e-12)
        self.assertAlmostEqual(p.fit_y[-1], p.yticks[-1], delta=1e-12)
        self.assertGreaterEqual(p.fit_y[-1], p.y[-1])


class PaperTailTestCases(unittest.TestCase):
    """Fitted points of tail fits, and a fit to the exceedances drawn through the whole sample."""

    def setUp(self):
        # mean ranks i/11: the 5 values below 0.5 are outside the tail 0.5, u = 3, tied with the next value
        self.x = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_censored(self):
        p = probability_paper(self.x, tail=0.5)
        np.testing.assert_array_equal(p.fitted, np.arange(10) >= 5)
        self.assertEqual(p.x.size, 10)  # the values outside the tail are shown too

    def test_exceedances(self):
        w = Weibull(3.0, 2.0, 1.5)
        p = probability_paper(self.x, w, tail=0.5, tail_model="exceedances")
        np.testing.assert_array_equal(p.fitted, self.x > 3.0)
        self.assertEqual(p.x.size, 10)

    def test_exceedances_line(self):
        # hand calculation: F(x) = 1 - p exp(-((x - u)/b)^c) with p = 4/10, the actual fraction above u = 3
        u, b, c, frac = 3.0, 2.0, 1.5, 0.4
        p = probability_paper(self.x, Weibull(u, b, c), tail=0.5, tail_model="exceedances")
        f = 1.0 - frac * np.exp(-(((p.fit_x - u) / b) ** c))
        np.testing.assert_allclose(p.fit_y, np.sqrt(-np.log(1.0 - f)), rtol=1e-10)
        self.assertAlmostEqual(p.fit_x[0], u, delta=1e-12)  # the line starts at u

    def test_exceedances_line_starts_at_u(self):
        # with the location fitted below u, the line still starts at u
        u = 3.0
        w = Weibull(2.0, 2.5, 1.5)
        p = probability_paper(self.x, w, tail=0.5, tail_model="exceedances")
        self.assertAlmostEqual(p.fit_x[0], u, delta=1e-9)
        f = 1.0 - 0.4 * (1.0 - w.cdf(p.fit_x))
        np.testing.assert_allclose(p.fit_y, np.sqrt(-np.log(1.0 - f)), rtol=1e-10)

    def test_censored_line_not_mapped(self):
        w = Weibull(0.0, 4.0, 2.0)
        p = probability_paper(self.x, w, tail=0.5)
        np.testing.assert_allclose(p.fit_y, np.sqrt(-np.log(1.0 - w.cdf(p.fit_x))), rtol=1e-10)

    def test_failed_fit(self):
        # a fit that failed has NaN parameters: no fitted line
        for tail_model in ("censored", "exceedances"):
            with self.subTest(tail_model=tail_model):
                p = probability_paper(self.x, Weibull(np.nan, np.nan, np.nan), tail=0.5, tail_model=tail_model)
                self.assertEqual(p.fit_x.size, 0)
                self.assertEqual(p.x.size, 10)

    def test_whole_sample_in_tail(self):
        p = probability_paper(self.x, Weibull(0.0, 4.0, 2.0), tail=1.0, tail_model="exceedances")
        self.assertTrue(np.all(p.fitted))

    def test_same_selection_as_stats(self):
        ts = mooring_series()
        for method, tail_model in (("lse", "censored"), ("msm", "exceedances")):
            with self.subTest(tail_model=tail_model):
                s = ts.stats(method=method, tail=0.2, tail_model=tail_model, include_sample=True)
                w = Weibull(s["wloc"], s["wscale"], s["wshape"])
                p = probability_paper(s["sample"], w, tail=0.2, tail_model=tail_model)
                self.assertEqual(np.count_nonzero(p.fitted), s["wn"])
                self.assertEqual(p.sample[~p.fitted].max(), s["wthreshold"])


class PaperExcludedTestCases(unittest.TestCase):
    """Values that can't be shown on Weibull paper."""

    def test_excluded(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = probability_paper(x, paper="weibull", loc=2.0)
        np.testing.assert_array_equal(p.excluded, [1.0, 2.0])
        np.testing.assert_array_equal(p.sample, [3.0, 4.0, 5.0])
        # the shown values keep their plotting positions in the whole sample
        f = (np.arange(3, 6) - 0.3) / 5.4
        np.testing.assert_allclose(p.y, np.log(-np.log(1.0 - f)), rtol=1e-14)

    def test_location_from_fit(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = probability_paper(x, Weibull(2.5, 2.0, 1.5), paper="weibull")
        self.assertEqual(p.loc, 2.5)
        np.testing.assert_array_equal(p.excluded, [1.0, 2.0])
        self.assertTrue(np.all(p.fit_x > -np.inf))

    def test_location_not_weibull(self):
        p = probability_paper([1.0, 2.0, 3.0], Gumbel(2.0, 1.0), paper="weibull")
        self.assertEqual(p.loc, 0.0)

    def test_other_papers_show_all(self):
        p = probability_paper([-1.0, 0.0, 1.0], Weibull(2.5, 2.0, 1.5))
        self.assertEqual(p.excluded.size, 0)
        self.assertEqual(p.loc, 0.0)


class PaperMinimaTestCases(unittest.TestCase):
    """Minima are drawn as the minima multiplied by -1."""

    def test_minima(self):
        g = Gumbel(5.0, 2.0)
        s = g.rnd(100, seed=3)
        pmax = probability_paper(s, g, paper="gumbel")
        pmin = probability_paper(-s, g, paper="gumbel", is_minima=True)
        np.testing.assert_array_equal(pmin.x, pmax.x)
        np.testing.assert_array_equal(pmin.y, pmax.y)
        np.testing.assert_array_equal(pmin.fit_x, pmax.fit_x)
        np.testing.assert_array_equal(pmin.xticks, pmax.xticks)
        np.testing.assert_array_equal(pmin.sample, -pmax.sample)
        self.assertEqual(pmin.xticklabels, [f"{-float(v) + 0.0:g}" for v in pmax.xticklabels])
        self.assertEqual(pmin.yticklabels, pmax.yticklabels)

    def test_minima_excluded(self):
        p = probability_paper([-1.0, -2.0, -3.0], paper="weibull", loc=1.5, is_minima=True)
        np.testing.assert_array_equal(p.excluded, [-1.0])
        np.testing.assert_array_equal(p.sample, [-2.0, -3.0])


class PaperTicksTestCases(unittest.TestCase):
    """Tick positions and labels."""

    def test_probability_ticks(self):
        p = probability_paper([1.0, 2.0, 4.0])  # median ranks 0.206 ... 0.794
        self.assertEqual(p.yticklabels, ["0.1", "0.3", "0.5", "0.7", "0.8"])
        np.testing.assert_allclose(p.yticks, np.sqrt(-np.log(1.0 - np.array([0.1, 0.3, 0.5, 0.7, 0.8]))))
        p = probability_paper([1.0, 2.0, 4.0], paper="gumbel")
        self.assertEqual(p.yticklabels, ["0.2", "0.3", "0.5", "0.7", "0.8"])
        np.testing.assert_allclose(p.yticks, -np.log(-np.log(np.array([0.2, 0.3, 0.5, 0.7, 0.8]))))

    def test_rayleigh_starts_at_zero(self):
        p = probability_paper(np.arange(1.0, 101.0))  # lowest median rank 0.007
        self.assertEqual(p.yticklabels[:3], ["0", "0.1", "0.3"])
        self.assertEqual(p.yticks[0], 0.0)

    def test_probability_ticks_span_points(self):
        w = Weibull(0.0, 3.0, 2.0)
        for paper in ("rayleigh", "weibull", "gumbel"):
            with self.subTest(paper=paper):
                p = probability_paper(w.rnd(500, seed=6), w, paper=paper)
                self.assertLessEqual(p.yticks[0], p.y.min())
                self.assertGreaterEqual(p.yticks[-1], p.y.max())
                np.testing.assert_allclose(p.yticks, paper_coordinates([], [float(v) for v in p.yticklabels], paper)[1])

    def test_value_ticks(self):
        p = probability_paper([1.0, 2.0, 4.0])
        np.testing.assert_array_equal(p.xticks, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(p.xticklabels, ["1", "2", "3", "4"])

    def test_value_ticks_weibull(self):
        p = probability_paper([1.0, 2.0, 4.0], paper="weibull", loc=0.5)
        np.testing.assert_allclose(p.xticks, np.log(np.array([1.0, 2.0, 3.0, 4.0]) - 0.5))
        self.assertEqual(p.xticklabels, ["1", "2", "3", "4"])

    def test_value_ticks_decimals(self):
        p = probability_paper([0.1, 0.2, 0.35])
        self.assertEqual(p.xticklabels, ["0.1", "0.15", "0.2", "0.25", "0.3", "0.35"])

    def test_single_value(self):
        p = probability_paper([2.0])
        self.assertEqual(p.xticklabels, ["2"])


class PaperLinearTestCases(unittest.TestCase):
    """The linear paper: cumulative probability on a linear scale."""

    def setUp(self):
        self.x = np.array([1.0, 2.0, 4.0])
        self.f = np.array([0.7, 1.7, 2.7]) / 3.4  # median ranks (i - 0.3) / (n + 0.4)

    def test_coordinates(self):
        p = probability_paper(self.x, paper="linear")
        np.testing.assert_array_equal(p.x, self.x)
        np.testing.assert_allclose(p.y, self.f, rtol=1e-14)
        px, py = paper_coordinates(self.x, self.f, paper="linear")
        np.testing.assert_array_equal(px, self.x)
        np.testing.assert_array_equal(py, self.f)

    def test_ticks(self):
        # steps of 0.1, from the tick at or below the lowest point (0.206) to the tick at or above the highest (0.794)
        p = probability_paper(self.x, paper="linear")
        np.testing.assert_allclose(p.yticks, np.linspace(0.2, 0.8, 7), atol=1e-15)
        self.assertEqual(p.yticklabels, ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"])
        # the whole axis for a larger sample
        p = probability_paper(np.arange(1.0, 101.0), paper="linear")
        self.assertEqual(p.yticklabels, ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])

    def test_fitted_line(self):
        w = Weibull(0.5, 2.0, 1.5)
        p = probability_paper(self.x, w, paper="linear")
        np.testing.assert_allclose(p.fit_y, w.cdf(p.fit_x), rtol=1e-9, atol=1e-12)


class PaperNoSampleTestCases(unittest.TestCase):
    """Only a fitted distribution, without a sample."""

    def test_fitted_only(self):
        w = Weibull(0.5, 2.0, 1.5)
        for paper in ("rayleigh", "weibull", "gumbel", "linear"):
            with self.subTest(paper=paper):
                p = probability_paper(None, w, paper=paper)
                for a in (p.sample, p.x, p.y, p.fitted, p.excluded):
                    self.assertEqual(a.size, 0)
                self.assertGreater(p.fit_x.size, 0)
                self.assertTrue(np.all(np.isfinite(p.fit_x)))
                # from the lowest to the highest tick around the probabilities 0.001 and 0.999
                self.assertAlmostEqual(p.fit_y[0], p.yticks[0], delta=1e-12)
                self.assertLessEqual(float(p.yticklabels[0]), 0.001)
                self.assertGreaterEqual(float(p.yticklabels[-1]), 0.999)
                self.assertGreater(len(p.xticks), 1)

    def test_invalid(self):
        with self.assertRaisesRegex(QatsValueError, "Give a sample, a fitted distribution or both"):
            probability_paper(None)
        with self.assertRaisesRegex(QatsValueError, "give the sample"):
            probability_paper(None, Weibull(0.5, 2.0, 1.5), tail=0.2)


class PaperValidationTestCases(unittest.TestCase):
    """Invalid arguments raise."""

    def test_invalid(self):
        x = [1.0, 2.0, 3.0]
        w = Weibull(0.0, 2.0, 1.5)
        cases = [
            (QatsValueError, dict(paper="normal")),
            (QatsValueError, dict(kind="weibull")),
            (QatsValueError, dict(tail=0.0)),
            (QatsValueError, dict(tail=1.5)),
            (QatsTypeError, dict(tail="0.2")),
            (QatsTypeError, dict(tail=True)),
            (QatsValueError, dict(tail=0.2, tail_model="peaks")),
            (QatsValueError, dict(fitted=w, tail_model="exceedances")),
            (QatsValueError, dict(tail=0.2, tail_model="exceedances")),
            (QatsTypeError, dict(fitted=(0.0, 2.0, 1.5))),
            (QatsValueError, dict(loc=0.5)),
            (QatsTypeError, dict(paper="weibull", loc="mean")),
        ]
        for error, kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(error):
                    probability_paper(x, **kwargs)

    def test_invalid_sample(self):
        for sample in ([], [1.0, np.nan], [1.0, np.inf]):
            with self.subTest(sample=sample):
                with self.assertRaises(QatsValueError):
                    probability_paper(sample)

    def test_invalid_paper_coordinates(self):
        with self.assertRaises(QatsValueError):
            paper_coordinates([1.0], [0.5], paper="normal")
