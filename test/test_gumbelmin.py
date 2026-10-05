# -*- coding: utf-8 -*-
"""
Module for testing GumbelMin class
"""

import unittest
import warnings

import numpy as np

from qats.stats.gumbelmin import GumbelMin, lse, mle, msm


class GumbelMinBootstrapTestCases(unittest.TestCase):
    """GumbelMin.bootstrap() (#157)."""

    def setUp(self):
        self.loc = 10.0
        self.scale = 2.0
        self.dist = GumbelMin(loc=self.loc, scale=self.scale)

    def test_bootstrap_equals_mean_of_fits_to_resamples(self):
        """
        Reference: the bootstrap mean parameters and coefficients of variation equal those calculated by hand from
        fits to the same random samples (same seed).
        """
        size, n = 200, 20
        for method, fit in (("msm", msm), ("lse", lse), ("mle", mle)):
            with self.subTest(method), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # mle may warn about slow convergence
                np.random.seed(7)
                m, cv = self.dist.bootstrap(size=size, method=method, N=n)

                np.random.seed(7)
                par = np.array([fit(self.dist.rnd(size=size)) for _ in range(n)])
                np.testing.assert_allclose(m, par.mean(axis=0))
                np.testing.assert_allclose(cv, par.std(axis=0, ddof=1) / par.mean(axis=0))

    def test_bootstrap_recovers_distribution_parameters(self):
        """Bootstrapping a distribution with known parameters gives mean parameters close to them."""
        np.random.seed(1)
        for method in ("msm", "lse"):
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


if __name__ == "__main__":
    unittest.main()
