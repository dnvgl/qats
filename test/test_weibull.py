import os
import unittest
import unittest.mock

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import gamma

from qats import TsDB
from qats._validation import QatsValueError
from qats.stats.empirical import empirical_cdf
from qats.stats.weibull import Weibull, lse, mle, msm, pwm, pwm2

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


if __name__ == "__main__":
    unittest.main()
