# -*- coding: utf-8 -*-
"""
Empirical distribution: plotting positions, and the tail of a sample shared by the tail fits and probability papers.
"""

import unittest

import numpy as np

from qats._validation import QatsTypeError, QatsValueError
from qats.stats.empirical import _PLOTTING_POSITIONS, _check_tail_model, empirical_cdf, tail_threshold


class TailThresholdTestCases(unittest.TestCase):
    def test_hand_calculation(self):
        # n = 9, mean ranks i / 10: 0.1 ... 0.9. tail 0.2: ranks at or above 0.8 are in the tail, so 7 values are
        # outside it and u is the 7th smallest value. tail 0.5: ranks 0.1 ... 0.4 are outside, u is the 4th
        x = np.array([9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0])
        self.assertEqual(tail_threshold(x, 0.2), 7.0)
        self.assertEqual(tail_threshold(x, 0.5), 4.0)

    def test_rank_rule(self):
        # the rule of TimeSeries.stats(): r = ceil((1 - tail) * (n + 1)) - 1 values outside the tail
        x = np.random.default_rng(3).normal(size=371)
        for tail in (0.05, 0.13, 0.2, 0.5, 0.9):
            with self.subTest(tail=tail):
                r = int(np.ceil((1.0 - tail) * (x.size + 1))) - 1
                self.assertEqual(tail_threshold(x, tail), np.sort(x)[r - 1])

    def test_whole_sample_in_tail(self):
        self.assertIsNone(tail_threshold([1.0, 2.0, 3.0], 1.0))
        self.assertIsNone(tail_threshold([1.0, 2.0, 3.0], 0.9))

    def test_invalid_tail(self):
        cases = ((0.0, QatsValueError), (1.5, QatsValueError), ("0.2", QatsTypeError), (True, QatsTypeError))
        for tail, error in cases:
            with self.subTest(tail=tail), self.assertRaises(error):
                tail_threshold([1.0, 2.0, 3.0], tail)


class TailModelTestCases(unittest.TestCase):
    def test_invalid(self):
        with self.assertRaisesRegex(QatsValueError, "must be one of censored, exceedances"):
            _check_tail_model("peaks", 0.2)
        with self.assertRaisesRegex(QatsValueError, "needs the tail fraction"):
            _check_tail_model("exceedances", None)

    def test_valid(self):
        for tail_model, tail in (("censored", None), ("censored", 0.2), ("exceedances", 0.2)):
            with self.subTest(tail_model=tail_model, tail=tail):
                _check_tail_model(tail_model, tail)


class PlottingPositionsTestCases(unittest.TestCase):
    def test_formulas(self):
        # the formulas listed in _PLOTTING_POSITIONS, for n = 4
        i, n = np.arange(1.0, 5.0), 4.0
        expected = dict(
            mean=i / (n + 1),
            median=(i - 0.3) / (n + 0.4),
            symmetrical=(i - 0.5) / n,
            beard=(i - 0.31) / (n + 0.38),
            gringorten=(i - 0.44) / (n + 0.12),
        )
        self.assertEqual(set(expected), set(_PLOTTING_POSITIONS))
        for kind, f in expected.items():
            with self.subTest(kind=kind):
                np.testing.assert_allclose(empirical_cdf(4, kind=kind), f, rtol=1e-14)

    def test_invalid_kind(self):
        with self.assertRaises(ValueError):
            empirical_cdf(4, kind="weibull")


if __name__ == "__main__":
    unittest.main()
