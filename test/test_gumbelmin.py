# -*- coding: utf-8 -*-
"""
Module for testing GumbelMin class (deprecated since 5.4.0, removed in 6.0.0)
"""

import subprocess
import sys
import unittest
import warnings

import numpy as np
from scipy.stats import gumbel_l

from qats.stats.gumbelmin import GumbelMin, lse, mle, msm


class GumbelMinTestCase(unittest.TestCase):
    def setUp(self):
        # GumbelMin is deprecated; the deprecation warning itself is tested in GumbelMinDeprecationTestCases
        self.enterContext(warnings.catch_warnings())
        warnings.simplefilter("ignore", DeprecationWarning)

        self.loc = 10.0
        self.scale = 2.0
        self.dist = GumbelMin(loc=self.loc, scale=self.scale)


class GumbelMinBootstrapTestCases(GumbelMinTestCase):
    """GumbelMin.bootstrap() (#157)."""

    def test_bootstrap_equals_mean_of_fits_to_resamples(self):
        """
        Reference: the bootstrap mean parameters and coefficients of variation equal those calculated by hand from
        fits to the same random samples (same seed).
        """
        size, n = 200, 20
        for method, fit in (("msm", msm), ("lse", lse), ("mle", mle)):
            with self.subTest(method):
                np.random.seed(7)
                m, cv = self.dist.bootstrap(size=size, method=method, N=n)

                np.random.seed(7)
                par = np.array([fit(self.dist.rnd(size=size)) for _ in range(n)])
                np.testing.assert_allclose(m, par.mean(axis=0))
                np.testing.assert_allclose(cv, par.std(axis=0, ddof=1) / par.mean(axis=0))

    def test_bootstrap_recovers_distribution_parameters(self):
        """Bootstrapping a distribution with known parameters gives mean parameters close to them."""
        np.random.seed(1)
        for method in ("msm", "lse", "mle"):
            with self.subTest(method):
                m, cv = self.dist.bootstrap(size=1000, method=method, N=50)
                np.testing.assert_allclose(m, [self.loc, self.scale], rtol=0.02)
                self.assertTrue(np.all(cv > 0.0))

    def test_bootstrap_sample_size_defaults_to_data_size(self):
        data = self.dist.rnd(size=150, seed=3)
        dist = GumbelMin(loc=self.loc, scale=self.scale, data=data)
        np.random.seed(5)
        m, _ = dist.bootstrap(N=10)
        np.random.seed(5)
        m_ref, _ = dist.bootstrap(size=150, N=10)
        np.testing.assert_allclose(m, m_ref)


class GumbelMinMLETestCases(GumbelMinTestCase):
    """Maximum likelihood fit (#160)."""

    def test_mle_equals_scipy_gumbel_l_fit(self):
        """Reference: the maximum likelihood estimates of scipy.stats.gumbel_l."""
        for loc, scale, size, seed in ((10.0, 2.0, 2000, 3), (-5.0, 0.5, 50, 1), (1000.0, 150.0, 500, 13)):
            with self.subTest(loc=loc, scale=scale, size=size):
                x = GumbelMin(loc=loc, scale=scale).rnd(size=size, seed=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)  # no overflow or convergence warnings
                    a, b = mle(x)
                np.testing.assert_allclose([a, b], gumbel_l.fit(x), rtol=1e-6)

    def test_fit_mle(self):
        x = self.dist.rnd(size=2000, seed=3)
        dist = GumbelMin()
        dist.fit(x, method="mle")
        np.testing.assert_allclose([dist.location, dist.scale], gumbel_l.fit(x), rtol=1e-6)


class GumbelMinDistributionTestCases(GumbelMinTestCase):
    """cdf() and pdf() without x (#161), compared with scipy.stats.gumbel_l."""

    def test_cdf_without_x(self):
        x = np.linspace(self.loc, self.loc - 3.0 * self.dist.std, 100)  # default range, see GumbelMin.cdf()
        np.testing.assert_allclose(self.dist.cdf(), gumbel_l.cdf(x, loc=self.loc, scale=self.scale))

    def test_pdf_without_x(self):
        x = np.linspace(self.loc, self.loc - 3.0 * self.dist.std, 100)  # default range, see GumbelMin.pdf()
        np.testing.assert_allclose(self.dist.pdf(), gumbel_l.pdf(x, loc=self.loc, scale=self.scale))


class GumbelMinDeprecationTestCases(unittest.TestCase):
    """GumbelMin and the module functions are deprecated since 5.4.0."""

    def test_class_warns(self):
        with self.assertWarnsRegex(DeprecationWarning, r"qats\.stats\.gumbelmin\.GumbelMin is deprecated"):
            GumbelMin(loc=0.0, scale=1.0)

    def test_functions_warn(self):
        x = np.random.default_rng(1).gumbel(size=50)
        for fit in (lse, mle, msm):
            with self.subTest(fit.__name__):
                with self.assertWarnsRegex(DeprecationWarning, rf"qats\.stats\.gumbelmin\.{fit.__name__}\(\)"):
                    fit(-x)

    def test_warning_is_shown_in_user_scripts(self):
        """A script using GumbelMin shows the warning with Python's default warning filters."""
        code = "from qats.stats.gumbelmin import GumbelMin\nGumbelMin(loc=0.0, scale=1.0)\n"
        # fixed command: the current interpreter running the code above, without -W options
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: S603
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DeprecationWarning: qats.stats.gumbelmin.GumbelMin is deprecated", result.stderr)

    def test_methods_do_not_warn_again(self):
        """Only creating the instance warns, not each method that uses the (deprecated) module functions."""
        with self.assertWarns(DeprecationWarning):
            dist = GumbelMin(loc=0.0, scale=1.0)
        x = dist.rnd(size=100, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            dist.fit(x, method="mle")
            dist.bootstrap(size=50, N=3)
            dist.cdf()
            dist.pdf()


if __name__ == "__main__":
    unittest.main()
