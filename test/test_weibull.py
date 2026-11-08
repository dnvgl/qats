import os
import tempfile
import unittest
import unittest.mock

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import gamma

from qats import TsDB
from qats._validation import QatsTypeError, QatsValueError
from qats.stats.empirical import empirical_cdf
from qats.stats.paper import probability_paper
from qats.stats.weibull import Weibull, lse, mle, msm, plot_fit, pwm, pwm2, weibull2gumbel

# todo: more test cases for weibull class and functions


def mooring_maxima():
    """Maxima of 'mooring line 4' in data/mooring.ts."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "mooring.ts")
    return TsDB.fromfile(path).get(name="mooring line 4").maxima()


def on_curve(loc, scale, shape, n):
    """Sample of size n whose mean rank empirical cdf lies exactly on the Weibull cdf."""
    return loc + scale * (-np.log(1.0 - empirical_cdf(n, kind="mean"))) ** (1.0 / shape)


def censored_nll(params, fitted, r, xc):
    """Negative log-likelihood with r values censored at xc."""
    loc, scale, shape = params
    d = stats.weibull_min(shape, loc, scale)
    return -(np.sum(d.logpdf(fitted)) + r * d.logcdf(xc))


class WeibullTestCases(unittest.TestCase):
    """
    Test the general case where the loc parameter is non-zero.
    """

    def setUp(self):
        self.loc = 66000.0
        self.scale = 410.0
        self.shape = 2.0

        wd = Weibull(loc=self.loc, scale=self.scale, shape=self.shape)
        self.x = wd.rnd(size=500, seed=3)

    def test_pwm(self):
        a, b, c = pwm(self.x)
        self.assertLessEqual((self.loc - a) / self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_msm(self):
        a, b, c = msm(self.x)
        self.assertLessEqual((self.loc - a) / self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_lse(self):
        a, b, c = lse(self.x)
        self.assertLessEqual((self.loc - a) / self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_mle(self):
        a, b, c = mle(self.x)
        self.assertLessEqual((self.loc - a) / self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_fit(self):
        weib = Weibull.fit(self.x, method="pwm")
        a, b, c = pwm(self.x)
        self.assertEqual(weib.loc, a)
        self.assertEqual(weib.scale, b)
        self.assertEqual(weib.shape, c)


class Weibull2PTestCases(unittest.TestCase):
    """
    Test the case where the loc parameter is zero
    """

    def setUp(self):
        self.loc = 0.0
        self.scale = 410.0
        self.shape = 3.5

        wd = Weibull(loc=self.loc, scale=self.scale, shape=self.shape)
        self.x = wd.rnd(size=500, seed=3)

    def test_pwm2(self):
        b, c = pwm2(self.x)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_fit_2p(self):
        weib = Weibull.fit(self.x, method="pwm2")
        b, c = pwm2(self.x)
        self.assertEqual(weib.loc, 0.0)
        self.assertEqual(weib.scale, b)
        self.assertEqual(weib.shape, c)


class WeibullUnchangedTestCases(unittest.TestCase):
    """
    Fits without `loc` and censoring give the results of the implementation before 5.6.0 (QATS 5.5.0).
    """

    # parameters (loc, scale, shape) from QATS 5.5.0
    expected = {
        ("x", "pwm"): (66002.54030480506, 401.36552999396787, 2.0557943856492313),
        ("x", "msm"): (66007.13674423081, 396.10058465175837, 2.026173528009048),
        ("x", "lse"): (65992.32735940871, 411.6523542260263, 2.137790552908745),
        ("x", "mle"): (66000.82406436153, 403.3346582083379, 2.066697284811693),
        ("x", "lse87"): (66011.19350514917, 382.65851944626553, 1.909343385265083),
        ("exceedances", "pwm"): (66285.43870632947, 201.14662283644626, 1.3167362503798936),
        ("exceedances", "msm"): (66270.89260447028, 220.5114626688839, 1.454774878725967),
        ("exceedances", "lse"): (66290.95173532389, 196.53881716053598, 1.2094741965561897),
    }

    def setUp(self):
        x = Weibull(66000.0, 410.0, 2.0).rnd(size=500, seed=3)
        self.samples = {"x": x, "exceedances": x[x > 66300.0]}
        self.functions = {
            "pwm": pwm,
            "msm": msm,
            "lse": lse,
            "mle": mle,
            "lse87": lambda s: lse(s, threshold=0.87),
        }

    def test_unchanged(self):
        for (sample, method), expected in self.expected.items():
            # the least squares solver (MINPACK) gives results that differ by up to about 2e-7 between platforms
            rtol = 1e-6 if method.startswith("lse") else 1e-12
            with self.subTest(sample=sample, method=method):
                np.testing.assert_allclose(self.functions[method](self.samples[sample]), expected, rtol=rtol)

    def test_pwm2_unchanged(self):
        y = Weibull(0.0, 410.0, 3.5).rnd(size=500, seed=3)
        np.testing.assert_allclose(pwm2(y), (406.74938152575425, 3.6343090334500556), rtol=1e-12)

    def test_mle_tail_without_censored_values(self):
        # a tail that includes the smallest value censors nothing
        x = self.samples["x"]
        self.assertEqual(mle(x, tail=1.0), mle(x))
        self.assertEqual(mle(x, tail=1.0 - 0.5 / (x.size + 1)), mle(x))

    def test_lse_tail_is_threshold(self):
        x = self.samples["x"]
        for p in (0.13, 0.5, 1.0):
            with self.subTest(tail=p):
                self.assertEqual(lse(x, tail=p), lse(x, threshold=1 - p))
                self.assertEqual(lse(x, tail=p, loc=66000.0), lse(x, threshold=1 - p, loc=66000.0))


class WeibullFixedLocTestCases(unittest.TestCase):
    """
    Fits with fixed location parameter, against hand calculations and SciPy.
    """

    def test_pwm_hand_calculation(self):
        # x - loc = 1, 2, 4: M100 = 7/3, M110 = (0 * 1 + 1/2 * 2 + 1 * 4) / 3 = 5/3,
        # shape = ln 2 / ln(M100 / (2 * (M100 - M110))) = ln 2 / ln(7/4), scale = M100 / gamma(1 + 1/shape)
        shape = np.log(2.0) / np.log(7.0 / 4.0)
        scale = 7.0 / 3.0 / gamma(1.0 + 1.0 / shape)
        np.testing.assert_allclose(pwm([14.0, 11.0, 12.0], loc=10.0), (10.0, scale, shape), rtol=1e-12)

    def test_pwm_loc_zero_is_pwm2(self):
        x = Weibull(0.0, 410.0, 3.5).rnd(size=500, seed=3)
        a, b, c = pwm(x, loc=0)
        self.assertEqual(a, 0.0)
        self.assertEqual((b, c), pwm2(x))

    def test_msm_hand_calculation(self):
        # x - loc = 1, 1, 4 + 3 sqrt(2) has mean 2 + sqrt(2) and standard deviation equal to the mean, so the
        # coefficient of variation is 1: an exponential distribution, shape 1 and scale equal to the mean
        x = 10.0 + np.array([1.0, 4.0 + 3.0 * np.sqrt(2.0), 1.0])
        np.testing.assert_allclose(msm(x, loc=10.0), (10.0, 2.0 + np.sqrt(2.0), 1.0), rtol=1e-9)

    def test_lse_on_curve(self):
        # the empirical cdf lies on the Weibull cdf, so the least squares fit is exact
        x = on_curve(10.0, 3.0, 1.7, 20)
        np.testing.assert_allclose(lse(x, loc=10.0), (10.0, 3.0, 1.7), rtol=1e-7)

    def test_lse_tail_on_curve(self):
        # values outside the tail do not affect the fit, also not when they are below the fixed location
        x = on_curve(10.0, 3.0, 1.7, 20)
        x[:10] = np.linspace(0.0, 10.5, 10)
        np.testing.assert_allclose(lse(x, tail=0.5, loc=10.0), (10.0, 3.0, 1.7), rtol=1e-7)

    def test_lse_tail_fitted_loc_on_curve(self):
        x = on_curve(10.0, 3.0, 1.7, 20)
        x[:10] = 10.0 + 0.5 * (x[:10] - 10.0)
        np.testing.assert_allclose(lse(x, tail=0.5), (10.0, 3.0, 1.7), rtol=1e-6)

    def test_value_close_to_loc(self):
        # a value just above the fixed location, as for an exceedance close to the threshold, must not prevent the fit
        x = Weibull(10.0, 3.0, 1.2).rnd(size=200, seed=1)
        x[0] = 10.0 + 1e-12
        y = np.sort(x - 10.0)
        f = empirical_cdf(y.size, kind="mean")
        (scale, shape), _ = curve_fit(lambda z, b, c: 1.0 - np.exp(-((z / b) ** c)), y, f, p0=(3.0, 1.2))
        np.testing.assert_allclose(lse(x, loc=10.0), (10.0, scale, shape), rtol=1e-6)
        shape, _, scale = stats.weibull_min.fit(x, floc=10.0)
        a, b, c = mle(x, loc=10.0)
        np.testing.assert_allclose((b, c), (scale, shape), rtol=1e-4)
        loglik = lambda b, c: np.sum(stats.weibull_min.logpdf(x, c, 10.0, b))  # noqa: E731
        self.assertGreaterEqual(loglik(b, c), loglik(scale, shape) - 1e-9)

    def test_mle(self):
        x = Weibull(66000.0, 410.0, 2.0).rnd(size=500, seed=3)
        shape, loc, scale = stats.weibull_min.fit(x, floc=66000.0)
        a, b, c = mle(x, loc=66000.0)
        np.testing.assert_allclose((a, b, c), (loc, scale, shape), rtol=1e-6)
        # the scale parameter solves the likelihood equation for the scale: scale^shape = mean((x - loc)^shape)
        self.assertAlmostEqual(b, np.mean((x - a) ** c) ** (1.0 / c), delta=1e-9 * b)


class WeibullExceedancesTestCases(unittest.TestCase):
    """
    Fits to the exceedances of a threshold u with the location parameter at u.
    """

    def setUp(self):
        self.u = 66300.0
        x = Weibull(66000.0, 410.0, 2.0).rnd(size=500, seed=3)
        self.x = x[x > self.u]

    def test_pwm(self):
        a, b, c = pwm(self.x, loc=self.u)
        self.assertEqual(a, self.u)
        self.assertEqual((b, c), pwm2(self.x - self.u))

    def test_msm(self):
        a, b, c = msm(self.x, loc=self.u)
        y = self.x - self.u
        # the fitted distribution has the sample mean and coefficient of variation
        mean, var = stats.weibull_min.stats(c, loc=0.0, scale=b, moments="mv")
        self.assertAlmostEqual(mean, y.mean(), delta=1e-9 * y.mean())
        self.assertAlmostEqual(np.sqrt(var) / mean, y.std() / y.mean(), delta=1e-9)

    def test_lse(self):
        # reference: SciPy's least squares fit of the cdf to the mean rank empirical cdf
        y = np.sort(self.x - self.u)
        f = empirical_cdf(y.size, kind="mean")
        (scale, shape), _ = curve_fit(lambda z, b, c: 1.0 - np.exp(-((z / b) ** c)), y, f, p0=(200.0, 1.5))
        np.testing.assert_allclose(lse(self.x, loc=self.u), (self.u, scale, shape), rtol=1e-6)

    def test_mle(self):
        shape, loc, scale = stats.weibull_min.fit(self.x, floc=self.u)
        np.testing.assert_allclose(mle(self.x, loc=self.u), (loc, scale, shape), rtol=1e-6)


@unittest.skipUnless(hasattr(stats, "CensoredData"), "requires scipy>=1.11")
class WeibullCensoredMLETestCases(unittest.TestCase):
    """
    Censored maximum likelihood against SciPy's fit to censored data.
    """

    def setUp(self):
        self.x = Weibull(66000.0, 410.0, 2.0).rnd(size=500, seed=3)

    # tail and number of censored values r among the 500: the values with mean rank i / 501 < 1 - tail
    tails = ((0.5, 250), (0.13, 435))

    def check(self, tail, r, loc):
        xs = np.sort(self.x)
        fitted, xc = xs[r:], xs[r - 1]
        data = stats.CensoredData(uncensored=fitted, left=np.full(r, xc))
        kwargs = dict() if loc is None else dict(floc=loc)
        shape, scale_loc, scale = stats.weibull_min.fit(data, **kwargs)
        reference = (scale_loc, scale, shape)
        params = mle(self.x, tail=tail, loc=loc)
        np.testing.assert_allclose(params, reference, rtol=1e-6)
        # the likelihood is at least as large as at SciPy's optimum
        self.assertLessEqual(censored_nll(params, fitted, r, xc), censored_nll(reference, fitted, r, xc) + 1e-9)

    def test_fixed_loc(self):
        for tail, r in self.tails:
            with self.subTest(tail=tail):
                self.check(tail, r, 66000.0)

    def test_fitted_loc(self):
        for tail, r in self.tails:
            with self.subTest(tail=tail):
                self.check(tail, r, None)


class WeibullFailedFitTestCases(unittest.TestCase):
    """
    Fits that fail raise QatsValueError.
    """

    def setUp(self):
        self.x = Weibull(66000.0, 410.0, 2.0).rnd(size=500, seed=3)

    def test_mle_not_converged(self):
        # the solver reaches the maximum number of function calls on these samples
        with self.assertRaisesRegex(QatsValueError, "did not converge"):
            mle(mooring_maxima())
        with self.assertRaisesRegex(QatsValueError, "did not converge"):
            mle(self.x[self.x > 66300.0])

    def test_lse_not_converged(self):
        def leastsq(*args, **kwargs):
            return np.array([np.nan, np.nan, np.nan]), None, {}, "Number of calls has reached maxfev", 5

        with unittest.mock.patch("qats.stats.weibull.leastsq", leastsq):
            with self.assertRaisesRegex(QatsValueError, "did not converge"):
                lse(self.x)
            with self.assertRaisesRegex(QatsValueError, "did not converge"):
                lse(self.x, loc=66000.0)

    def test_loc_above_smallest_value(self):
        # 3-parameter fit to the mooring maxima gives a location parameter above the smallest maximum
        with self.assertRaisesRegex(QatsValueError, "location parameter .* not below"):
            pwm(mooring_maxima())

    def test_nan(self):
        x = self.x.copy()
        x[10] = np.nan
        with self.assertRaisesRegex(QatsValueError, "not finite"):
            pwm(x)

    def test_msm_no_shape(self):
        # zero coefficient of variation
        with self.assertRaisesRegex(QatsValueError, "msm fit failed"):
            msm(np.full(10, 5.0), loc=0.0)

    def test_fixed_loc_not_below_sample(self):
        for f in (pwm, msm, lse, mle):
            with self.subTest(method=f.__name__), self.assertRaisesRegex(QatsValueError, "must be below"):
                f(self.x, loc=self.x.min())
        with self.assertRaisesRegex(QatsValueError, "must be below"):
            pwm2(self.x - self.x.min())
        with self.assertRaisesRegex(QatsValueError, "must be below"):
            mle(self.x, tail=0.5, loc=np.median(self.x))

    def test_threshold_out_of_range(self):
        for q in (-0.1, 1.0):
            with self.subTest(threshold=q), self.assertRaises(QatsValueError):
                lse(self.x, threshold=q)

    def test_tail_out_of_range(self):
        for f in (lse, mle):
            for p in (-0.1, 0.0, 1.1):
                with self.subTest(method=f.__name__, tail=p), self.assertRaises(QatsValueError):
                    f(self.x, tail=p)
            with self.subTest(method=f.__name__, tail=1e-4), self.assertRaisesRegex(QatsValueError, "No values"):
                f(self.x, tail=1e-4)

    def test_tail_and_threshold(self):
        with self.assertRaisesRegex(QatsValueError, "not both"):
            lse(self.x, threshold=0.87, tail=0.13)


class WeibullFitTestCases(unittest.TestCase):
    """
    Weibull.fit() passes loc and tail on to the fit functions.
    """

    def setUp(self):
        self.x = Weibull(66000.0, 410.0, 2.0).rnd(size=500, seed=3)

    def test_loc_and_tail(self):
        cases = [
            ("pwm", dict(loc=66000.0), pwm(self.x, loc=66000.0)),
            ("pwm2", dict(loc=66000.0), pwm(self.x, loc=66000.0)),
            ("msm", dict(loc=66000.0), msm(self.x, loc=66000.0)),
            ("lse", dict(loc=66000.0), lse(self.x, loc=66000.0)),
            ("lse", dict(tail=0.13), lse(self.x, tail=0.13)),
            ("lse", dict(loc=66000.0, tail=0.13), lse(self.x, tail=0.13, loc=66000.0)),
            ("mle", dict(loc=66000.0), mle(self.x, loc=66000.0)),
            ("mle", dict(tail=0.13), mle(self.x, tail=0.13)),
            ("mle", dict(loc=66000.0, tail=0.13), mle(self.x, tail=0.13, loc=66000.0)),
        ]
        for method, kwargs, expected in cases:
            with self.subTest(method=method, **kwargs):
                self.assertEqual(Weibull.fit(self.x, method=method, **kwargs).params, expected)

    def test_tail_not_available(self):
        for method in ("pwm", "pwm2", "msm"):
            with self.subTest(method=method), self.assertRaisesRegex(QatsValueError, "only available"):
                Weibull.fit(self.x, method=method, tail=0.13)

    def test_method_case_insensitive(self):
        self.assertEqual(Weibull.fit(self.x, method="PWM").params, pwm(self.x))


class WeibullPlotLinearTestCases(unittest.TestCase):
    """Weibull.plot_linear() draws the same as before it used qats.stats.paper, run with the Agg backend."""

    def setUp(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        backend = plt.get_backend()
        plt.switch_backend("Agg")
        self.addCleanup(plt.switch_backend, backend)
        self.addCleanup(plt.close, "all")

    def test_unchanged(self):
        # data below and at the location parameter can't be shown on Weibull paper
        w = Weibull(1.0, 2.0, 1.5, data=[4.0, 0.5, 1.5, 1.0, 6.0, 2.5])
        with unittest.mock.patch("matplotlib.pyplot.show"), unittest.mock.patch("matplotlib.pyplot.close"):
            with self.assertWarnsRegex(DeprecationWarning, r"Weibull.plot_linear\(\) is deprecated.*Weibull.plot\("):
                w.plot_linear()
        ax = self.plt.gcf().axes[0]
        data, fitted = ax.get_lines()

        # the coordinates as calculated before 5.6.0
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(np.sort(w.data - w.loc))
        y = np.log(-np.log(1.0 - empirical_cdf(6, kind="mean")))
        shown = np.isfinite(x)
        np.testing.assert_allclose(data.get_xdata(), x[shown], rtol=1e-14)
        np.testing.assert_allclose(data.get_ydata(), y[shown], rtol=1e-14)
        self.assertEqual(data.get_xdata().size, 4)

        yy = np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
        x = np.log(w.invcdf(p=yy) - w.loc)
        y = np.log(-np.log(1.0 - yy))
        np.testing.assert_allclose(fitted.get_xdata(), x, rtol=1e-14)
        np.testing.assert_allclose(fitted.get_ydata(), y, rtol=1e-14)
        np.testing.assert_allclose(ax.get_xlim(), (x[0], x[-1]), rtol=1e-14)
        np.testing.assert_allclose(ax.get_ylim(), (y[0], y[-1]), rtol=1e-14)
        np.testing.assert_allclose(ax.get_yticks(), y, rtol=1e-14)
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], [str(v) for v in yy])
        xloc = np.linspace(x[0], x[-1], 5)
        np.testing.assert_allclose(ax.get_xticks(), xloc, rtol=1e-14)
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()], [str(v) for v in np.around(np.exp(xloc) + w.loc, 2)]
        )

    def test_plot_fit_deprecated(self):
        x = Weibull(1.0, 2.0, 1.5).rnd(size=20, seed=1)
        with unittest.mock.patch("matplotlib.pyplot.show"):
            with self.assertWarnsRegex(DeprecationWarning, r"plot_fit\(\) is deprecated.*\.plot\(paper='weibull'"):
                plot_fit(x, (0.5, 2.0, 1.5))


class WeibullFitTailModelTestCases(unittest.TestCase):
    """
    Weibull.fit() with the tail models, and the fit settings it keeps in fit_info.
    """

    def setUp(self):
        self.x = Weibull(66000.0, 410.0, 2.0).rnd(size=500, seed=3)
        self.tail = 0.2
        # values outside the tail: mean rank i / 501 below 1 - tail, and the threshold value u, the largest of them
        self.r = int(np.ceil((1.0 - self.tail) * (self.x.size + 1))) - 1
        self.u = np.sort(self.x)[self.r - 1]
        self.exceedances = self.x[self.x > self.u]

    def test_exceedances_same_as_complete_fit(self):
        # the fit to the exceedances is the complete fit of the values above u
        for method in ("pwm", "pwm2", "msm", "lse", "mle"):
            for loc in (None, self.u, 66000.0):
                if method == "mle" and loc is None:
                    continue  # does not converge on this sample
                with self.subTest(method=method, loc=loc):
                    kwargs = dict() if loc is None else dict(loc=loc)
                    w = Weibull.fit(self.x, method=method, tail=self.tail, tail_model="exceedances", **kwargs)
                    self.assertEqual(w.params, Weibull.fit(self.exceedances, method=method, **kwargs).params)
                    # the whole sample is kept
                    np.testing.assert_array_equal(w.data, self.x)

    def test_exceedances_whole_sample_in_tail(self):
        w = Weibull.fit(self.x, method="msm", tail=1.0, tail_model="exceedances")
        self.assertEqual(w.params, Weibull.fit(self.x, method="msm").params)
        self.assertIsNone(w.fit_info["threshold"])

    def test_censored_unchanged(self):
        w = Weibull.fit(self.x, method="mle", tail=self.tail, tail_model="censored")
        self.assertEqual(w.params, mle(self.x, tail=self.tail))

    def test_fit_info(self):
        complete = dict(tail=None, tail_model=None, threshold=None, n=500)
        cases = [
            (dict(method="msm"), dict(loc_from="fitted", **complete)),
            (dict(method="pwm2"), dict(loc_from="value", **complete)),
            (dict(method="lse", loc=66000.0), dict(loc_from="value", **complete)),
            (
                dict(method="LSE", tail=self.tail),
                dict(tail=self.tail, tail_model="censored", loc_from="fitted", threshold=self.u, n=500 - self.r),
            ),
            (
                dict(method="mle", tail=self.tail, tail_model="exceedances", loc=self.u),
                dict(
                    tail=self.tail,
                    tail_model="exceedances",
                    loc_from="value",
                    threshold=self.u,
                    n=self.exceedances.size,
                ),
            ),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                info = Weibull.fit(self.x, **kwargs).fit_info
                self.assertEqual(info, dict(method=kwargs["method"].lower(), **expected))

    def test_fit_info_not_fitted(self):
        self.assertIsNone(Weibull(1.0, 2.0, 1.5).fit_info)
        self.assertIsNone(Weibull(1.0, 2.0, 1.5, data=self.x).fit_info)

    def test_fit_info_is_a_copy(self):
        w = Weibull.fit(self.x, method="lse", tail=self.tail)
        w.fit_info["tail"] = 0.5
        self.assertEqual(w.fit_info["tail"], self.tail)

    def test_invalid(self):
        cases = [
            (QatsTypeError, dict(method="lse", loc="mean")),
            (QatsTypeError, dict(method="lse", loc="threshold")),
            (QatsTypeError, dict(method="lse", tail="0.2")),
            (QatsValueError, dict(method="lse", tail_model="exceedances")),
            (QatsValueError, dict(method="lse", tail=0.2, tail_model="peaks")),
            (QatsValueError, dict(method="msm", tail=0.2)),
            (QatsValueError, dict(method="weibull")),
        ]
        for error, kwargs in cases:
            with self.subTest(**kwargs), self.assertRaises(error):
                Weibull.fit(self.x, **kwargs)

    def test_gumbel_parameters_exceedances(self):
        # the distribution describes the exceedances, so the default n is their number
        w = Weibull.fit(self.x, method="msm", tail=self.tail, tail_model="exceedances")
        self.assertEqual(w.gumbel_parameters(), weibull2gumbel(*w.params, self.exceedances.size))
        # unchanged for the other fits: all values
        for kwargs in (dict(method="msm"), dict(method="lse", tail=self.tail)):
            with self.subTest(**kwargs):
                w = Weibull.fit(self.x, **kwargs)
                self.assertEqual(w.gumbel_parameters(), weibull2gumbel(*w.params, self.x.size))

    def test_mse_exceedances(self):
        w = Weibull.fit(self.x, method="msm", tail=self.tail, tail_model="exceedances")
        z = np.sort(self.exceedances)
        expected = np.mean((empirical_cdf(z.size, kind="mean") - w.cdf(z)) ** 2.0)
        self.assertAlmostEqual(w.mse, expected, delta=1e-15)
        # unchanged for a fit to all values
        w = Weibull.fit(self.x, method="msm")
        z = np.sort(self.x)
        self.assertAlmostEqual(w.mse, np.mean((w.ecdf - w.cdf(z)) ** 2.0), delta=1e-15)


class WeibullPlotTestCases(unittest.TestCase):
    """Weibull.plot() draws with qats.stats.paper, run with the Agg backend."""

    def setUp(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        backend = plt.get_backend()
        plt.switch_backend("Agg")
        self.addCleanup(plt.switch_backend, backend)
        self.addCleanup(plt.close, "all")
        self.x = Weibull(66000.0, 410.0, 2.0).rnd(size=200, seed=3)

    def axes(self):
        return self.plt.figure().gca()

    def lines(self, ax):
        """Data points (fitted, then not fitted), the fitted line and the dashed threshold value line."""
        lines = ax.get_lines()
        points = [ln for ln in lines if ln.get_marker() == "o"]
        fitted_line = [ln for ln in lines if ln.get_marker() != "o" and ln.get_linestyle() == "-"]
        dashed = [ln for ln in lines if ln.get_linestyle() == "--"]
        return points, fitted_line, dashed

    def test_default_as_before(self):
        # data against mean ranks and the fitted cdf, on linear scales
        w = Weibull.fit(self.x, method="msm")
        with unittest.mock.patch("matplotlib.pyplot.show") as show:
            ax = w.plot()
        show.assert_called_once()
        (data,), (line,), dashed = self.lines(ax)
        np.testing.assert_array_equal(data.get_xdata(), np.sort(self.x))
        np.testing.assert_allclose(data.get_ydata(), empirical_cdf(self.x.size, kind="mean"), rtol=1e-14)
        np.testing.assert_allclose(line.get_ydata(), w.cdf(line.get_xdata()), rtol=1e-9, atol=1e-12)
        self.assertEqual(dashed, [])
        self.assertEqual(ax.get_xlabel(), "X")
        self.assertEqual(ax.get_ylabel(), "Cumulative probability")

    def test_ax(self):
        ax = self.axes()
        with unittest.mock.patch("matplotlib.pyplot.show") as show:
            result = Weibull.fit(self.x, method="msm").plot(ax=ax)
        show.assert_not_called()
        self.assertIs(result, ax)
        self.assertIn(ax.figure.number, self.plt.get_fignums())

    def test_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weibull.png")
            with unittest.mock.patch("matplotlib.pyplot.show") as show:
                Weibull.fit(self.x, method="msm").plot(filename=path)
            show.assert_not_called()
            self.assertTrue(os.path.isfile(path))
        # a figure that is only saved is closed
        self.assertEqual(self.plt.get_fignums(), [])

    def test_paper_and_kind(self):
        w = Weibull.fit(self.x, method="msm")
        for paper in ("rayleigh", "weibull", "gumbel", "linear"):
            with self.subTest(paper=paper):
                ax = w.plot(paper=paper, kind="median", ax=self.axes())
                p = probability_paper(self.x, w, paper=paper, kind="median")
                (data,), (line,), _ = self.lines(ax)
                np.testing.assert_array_equal(data.get_xdata(), p.x)
                np.testing.assert_array_equal(data.get_ydata(), p.y)
                np.testing.assert_array_equal(line.get_xdata(), p.fit_x)
                np.testing.assert_array_equal(ax.get_yticks(), p.yticks)
                self.assertEqual([t.get_text() for t in ax.get_yticklabels()], p.yticklabels)
                self.assertEqual([t.get_text() for t in ax.get_xticklabels()], p.xticklabels)

    def test_censored_tail(self):
        w = Weibull.fit(self.x, method="lse", tail=0.2)
        ax = w.plot(paper="rayleigh", ax=self.axes())
        (fitted, other), (line,), (u,) = self.lines(ax)
        info = w.fit_info
        self.assertEqual(fitted.get_xdata().size, info["n"])
        self.assertTrue(np.all(fitted.get_xdata() > info["threshold"]))
        self.assertIsNone(fitted.get_alpha())
        self.assertLess(other.get_alpha(), 1.0)
        self.assertLess(other.get_markersize(), fitted.get_markersize())
        self.assertEqual(u.get_xdata()[0], info["threshold"])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels[:3], ["Data", "Data not fitted", "Fitted"])

    def test_exceedances_tail(self):
        w = Weibull.fit(self.x, method="msm", tail=0.2, tail_model="exceedances")
        ax = w.plot(paper="rayleigh", ax=self.axes())
        (fitted, _), (line,), (u,) = self.lines(ax)
        u_value = w.fit_info["threshold"]
        self.assertEqual(fitted.get_xdata().size, np.count_nonzero(self.x > u_value))
        # the fitted distribution describes the values above u, so the line starts at u
        self.assertAlmostEqual(line.get_xdata()[0], u_value, delta=1e-6 * u_value)

    def test_no_data(self):
        w = Weibull(1.0, 2.0, 1.5)
        ax = w.plot(ax=self.axes())
        points, (line,), _ = self.lines(ax)
        self.assertEqual(points, [])
        np.testing.assert_allclose(line.get_ydata(), w.cdf(line.get_xdata()), rtol=1e-9, atol=1e-12)
        self.assertGreaterEqual(line.get_ydata().min(), 0.0)
        self.assertLess(line.get_ydata().max(), 1.0)


if __name__ == "__main__":
    unittest.main()
