# -*- coding: utf-8 -*-
"""
Module for testing rainflow algorithm
"""

import itertools
import unittest
import numpy as np
from qats.fatigue import rainflow


class TestRainflowCounting(unittest.TestCase):
    # Load series and corresponding cycle counts from ASTM E1049-85
    series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    # reversals
    reversals = series[1:-1]
    # raw cycles
    cycles = [(3, -0.5, 0.5), (4, -1.0, 0.5), (4, 1.0, 1.0), (6, 1.0, 0.5), (8, 0.0, 0.5), (8, 1.0, 0.5), (9, 0.5, 0.5)]
    # raw cycles if end points are included
    '''
    Note: 
    until qats version 4.6.1, matching cycles where aggregated in the count. For later versions, this is not done by 
    count_cycles(), hence for the series used here we get two identical entries instead; (2., -1., 0.5) x 2
    -- old code: --
    # (first and last half cycles match to form a full cycle -> (2, -1.0, 1.0))
    cycles_endpoints = [(2, -1.0, 1.0), (3, -0.5, 0.5), (4, -1.0, 0.5), (4, 1.0, 1.0), (6, 1.0, 0.5), (8, 0.0, 0.5),
                        (8, 1.0, 0.5), (9, 0.5, 0.5)]
    '''
    cycles_endpoints = [(2, -1.0, 0.5), (2, -1.0, 0.5), (3, -0.5, 0.5), (4, -1.0, 0.5), (4, 1.0, 1.0), (6, 1.0, 0.5),
                        (8, 0.0, 0.5), (8, 1.0, 0.5), (9, 0.5, 0.5)]
    # cycles grouped in 2 bins
    cycles_n2 = [(2.25, 0.125, 2.0), (6.75, 0.625, 2.0)]
    # cycles grouped in bins of width 2
    cycles_bw2 = [(1.0, np.nan, 0.0), (3.0, 0.125, 2.0), (5.0, 1.0, 0.5), (7.0, 0.5, 1.0), (9.0, 0.5, 0.5)]
    # cycles grouped in bins of width 5
    cycles_bw5 = [(2.5, 0.125, 2.0), (7.5, 0.625, 2.0)]

    def test_reversals(self):
        """
        Test that reversals returns expected points
        """
        self.assertEqual(self.reversals, list(rainflow.reversals(self.series)))

    def test_reversals_recursive(self):
        """
        Test that reversals are unchanged if passed to reversals() with endpoints=True
        """
        reversals = self.reversals[:]
        for _ in range(3):
            reversals = list(rainflow.reversals(reversals, endpoints=True))
            self.assertEqual(self.reversals, reversals)

    def test_rainflow_counting(self):
        """
        Standard test
        """
        # self.assertEqual(np.array(self.cycles), rainflow.count_cycles(self.series))
        # np.testing module ensures that the list `self.cycles` may be compared to the array returned from count_cycles
        np.testing.assert_array_equal(self.cycles, rainflow.count_cycles(self.series))

    def test_rainflow_counting_with_endpoints(self):
        """
        Test cycle counting when end points are included.
        """
        # self.assertEqual(self.cycles_endpoints, rainflow.count_cycles(self.series, endpoints=True))
        np.testing.assert_array_equal(self.cycles_endpoints, rainflow.count_cycles(self.series, endpoints=True))

    def test_rainflow_counting_using_reversals(self):
        """
        Test that cycle counting using reversals works if endpoints=True
        """
        reversals = list(rainflow.reversals(self.series))
        # self.assertEqual(self.cycles, rainflow.count_cycles(reversals, endpoints=True))
        np.testing.assert_array_equal(self.cycles, rainflow.count_cycles(reversals, endpoints=True))

    def test_series_with_zero_derivatives(self):
        """
        Duplicate values in series to create zero derivatives (platou). Test that these are ignored by the cycle
        counting.
        """
        series = itertools.chain(*([x, x] for x in self.series))
        # self.assertEqual(self.cycles, rainflow.count_cycles(series))
        np.testing.assert_array_equal(self.cycles, rainflow.count_cycles(series))

    def test_rainflow_rebinning_binwidth2(self):
        """
        Test that values are correctly gathered to new bins of width 2
        """
        self.assertEqual(self.cycles_bw2, rainflow.rebin(rainflow.count_cycles(self.series), w=2.))

    def test_rainflow_rebinning_binwidth5(self):
        """
        Test that values are correctly gathered to new bins of width 5
        """
        self.assertEqual(self.cycles_bw5, rainflow.rebin(rainflow.count_cycles(self.series), w=5.))

    def test_rainflow_rebinning_nbin2(self):
        """
        Test that values are correctly gathered to 2 bins
        """
        self.assertEqual(self.cycles_n2, rainflow.rebin(rainflow.count_cycles(self.series), n=2))

    def test_rainflow_rebin_exceptions(self):
        """
        Test that rebinning raises errors as it should do
        """
        try:
            _ = rainflow.rebin(self.cycles, binby='nothing')
        except ValueError:
            pass
        else:
            self.fail("Did not raise ValueError when binby was not equal to neither 'mean' nor 'range'.")

        try:
            _ = rainflow.rebin(self.cycles)
        except ValueError:
            pass
        else:
            self.fail("Did not raise ValueError when neither `n` nor `w` were specified.")


if __name__ == '__main__':
    unittest.main()
