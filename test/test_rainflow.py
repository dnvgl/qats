# -*- coding: utf-8 -*-
"""
Module for testing rainflow algorithm
"""

import itertools
import random
import unittest

from qats import rainflow


class TestRainflowCounting(unittest.TestCase):
    # Load series and corresponding cycle counts from ASTM E1049-85
    series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    cycles = [(3, 0.5), (4, 1.5), (6, 0.5), (8, 1.0), (9, 0.5)]
    cycles_n2 = [(2.25, 2.0), (6.75, 2.0)]  # cycles grouped in 2 bins
    cycles_bw2 = [(1, 0.0), (3, 2.0), (5, 0.5), (7, 1.0), (9, 0.5)]  # cycles grouped in bins of width 2
    cycles_bw5 = [(2.5, 2.0), (7.5, 2.0)]   # cycles grouped in bins of width 5

    def test_rainflow_counting(self):
        """
        Standard test
        """
        self.assertEqual(rainflow.count_cycles(self.series), self.cycles)

    def test_rainflow_ndigits(self):
        """
        Add noise to test series. Test that the noise is ignored when specifying fewer significant digits
        """
        series = [x + 0.01 * random.random() for x in self.series]
        self.assertNotEqual(rainflow.count_cycles(series), self.cycles)
        self.assertEqual(rainflow.count_cycles(series, ndigits=1), self.cycles)

    def test_series_with_zero_derivatives(self):
        """
        Duplicate values in series to create zero derivatives (platou). Test that these are ignored by the cycle
        counting.
        """
        series = itertools.chain(*([x, x] for x in self.series))
        self.assertEqual(rainflow.count_cycles(series), self.cycles)

    def test_rainflow_rebinning_binwidth2(self):
        """
        Test that values are correctly gathered to new bins of width 2
        """
        self.assertEqual(rainflow.count_cycles(self.series, binwidth=2.), self.cycles_bw2)

    def test_rainflow_rebinning_binwidth5(self):
        """
        Test that values are correctly gathered to new bins of width 5
        """
        self.assertEqual(rainflow.count_cycles(self.series, binwidth=5.), self.cycles_bw5)

    def test_rainflow_rebinning_nbin2(self):
        """
        Test that values are correctly gathered to 2 bins
        """
        self.assertEqual(rainflow.count_cycles(self.series, nbins=2), self.cycles_n2)

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
