# -*- coding: utf-8 -*-
"""
Module for testing rainflow algorithm
"""

import itertools
import random
import unittest
import numpy as np
from qats import rainflow


class TestRainflowCounting(unittest.TestCase):
    # Load series and corresponding cycle counts from ASTM E1049-85
    series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    # raw cycles
    cycles = [(3, -0.5, 0.5), (4, -1.0, 0.5), (4, 1.0, 1.0), (6, 1.0, 0.5), (8, 0.0, 0.5), (8, 1.0, 0.5), (9, 0.5, 0.5)]
    # cycles grouped in 2 bins
    cycles_n2 = [(2.25, 0.125, 2.0), (6.75, 0.625, 2.0)]
    # cycles grouped in bins of width 2
    cycles_bw2 = [(1.0, np.nan, 0.0), (3.0, 0.125, 2.0), (5.0, 1.0, 0.5), (7.0, 0.5, 1.0), (9.0, 0.5, 0.5)]
    # cycles grouped in bins of width 5
    cycles_bw5 = [(2.5, 0.125, 2.0), (7.5, 0.625, 2.0)]

    def test_rainflow_counting(self):
        """
        Standard test
        """
        self.assertEqual(self.cycles, rainflow.count_cycles(self.series))

    def test_rainflow_ndigits(self):
        """
        Add noise to test series. Test that the noise is ignored when specifying fewer significant digits
        """
        series = [x + 0.01 * random.random() for x in self.series]
        self.assertNotEqual(self.cycles, rainflow.count_cycles(series))
        self.assertEqual(self.cycles, rainflow.count_cycles(series, ndigits=1))

    def test_series_with_zero_derivatives(self):
        """
        Duplicate values in series to create zero derivatives (platou). Test that these are ignored by the cycle
        counting.
        """
        series = itertools.chain(*([x, x] for x in self.series))
        self.assertEqual(self.cycles, rainflow.count_cycles(series))

    def test_rainflow_rebinning_binwidth2(self):
        """
        Test that values are correctly gathered to new bins of width 2
        """
        self.assertEqual(self.cycles_bw2, rainflow.count_cycles(self.series, binwidth=2.))

    def test_rainflow_rebinning_binwidth5(self):
        """
        Test that values are correctly gathered to new bins of width 5
        """
        self.assertEqual(self.cycles_bw5, rainflow.count_cycles(self.series, binwidth=5.))

    def test_rainflow_rebinning_nbin2(self):
        """
        Test that values are correctly gathered to 2 bins
        """
        self.assertEqual(self.cycles_n2, rainflow.count_cycles(self.series, nbins=2))

    def test_rainflow_rebin_exceptions(self):
        """
        Test that rebinning to bins which upper bound is lower than the maximum magnitude in the cycle
        distribution raises a ValueError
        """
        try:
            _ = rainflow.rebin_cycles(self.cycles, [0, 2, 4, 6, 8])
        except ValueError:
            pass
        else:
            self.fail("Did not raise ValueError when specifying bins which do not cover all cycle magnitudes.")


if __name__ == '__main__':
    unittest.main()
