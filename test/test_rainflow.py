# -*- coding: utf-8 -*-
"""
Module for testing rainflow algorithm
"""

import itertools
import unittest
import numpy as np
import os
from qats.fatigue import rainflow
from qats import TsDB


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
    # (not affected by change in behaviour after version 4.6.1)
    cycles_n2 = [(2.25, 0.125, 2.0), (6.75, 0.625, 2.0)]

    # cycles grouped in bins of width 2
    # (the following was the solution for version <= 4.6.1):
    # cycles_bw2 = [(1.0, np.nan, 0.0), (3.0, 0.125, 2.0), (5.0, 1.0, 0.5), (7.0, 0.5, 1.0), (9.0, 0.5, 0.5)]
    # (solution for version > 4.6.1, placing cycles at bin edges in the upper bin):
    cycles_bw2 = [(1.0, np.nan, 0.0), (3.0, -0.5, 0.5), (5.0, 0.333333, 1.5), (7.0, 1.0, 0.5), (9.0, 0.5, 1.5)]

    # cycles grouped in bins of width 5
    # (not affected by change in behaviour after version 4.6.1)
    cycles_bw5 = [(2.5, 0.125, 2.0), (7.5, 0.625, 2.0)]

    def setUp(self):
        """
        Set up for some of the tests.
        """
        # load irregular 3-hour time series test rebin and mesh
        tsfile = os.path.join(os.path.dirname(__file__), '..', 'data', 'simo_p_out.ts')
        self.irreg_series = TsDB.fromfile(tsfile).get(name='Tension_2_qs').x

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
        # self.assertEqual(self.cycles_bw2, rainflow.rebin(rainflow.count_cycles(self.series), w=2.))
        # np.testing.assert_array_equal(self.cycles_bw2, rainflow.rebin(rainflow.count_cycles(self.series), w=2.))
        np.testing.assert_array_almost_equal(self.cycles_bw2, rainflow.rebin(rainflow.count_cycles(self.series), w=2.),
                                             decimal=6)

    def test_rainflow_rebinning_binwidth5(self):
        """
        Test that values are correctly gathered to new bins of width 5
        """
        # self.assertEqual(self.cycles_bw5, rainflow.rebin(rainflow.count_cycles(self.series), w=5.))
        np.testing.assert_array_equal(self.cycles_bw5, rainflow.rebin(rainflow.count_cycles(self.series), w=5.))

    def test_rainflow_rebinning_nbin2(self):
        """
        Test that values are correctly gathered to 2 bins
        """
        # self.assertEqual(self.cycles_n2, rainflow.rebin(rainflow.count_cycles(self.series), n=2))
        np.testing.assert_array_equal(self.cycles_n2, rainflow.rebin(rainflow.count_cycles(self.series), n=2))

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

    def test_rebin_sum_of_counts(self):
        """
        Test that rebinning does not alter total number of counts.
        """
        cycles = rainflow.count_cycles(self.irreg_series)
        cycles_rebinned_range = rainflow.rebin(cycles, binby='range', n=50)
        self.assertEqual(cycles[:, 2].sum(), cycles_rebinned_range[:, 2].sum(),
                         msg="Total cycle counts changed after rebinning.")

    def test_mesh(self):
        """
        Multiple tests to ensure meshgrid returned from mesh() is correct.
        """
        # no. of range and mean bins, respectively (should be different to avoid hiding errors caused by transposing)
        nr, nm = 100, 50
        # raw cycles
        cycles = rainflow.count_cycles(self.irreg_series)
        # rebinned cycles - will be used to verify mesh
        cycles_rebinned_range = rainflow.rebin(cycles, binby='range', n=nr)
        cycles_rebinned_mean = rainflow.rebin(cycles, binby='mean', n=nm)
        # generate mesh
        rmesh, mmesh, cmesh = rainflow.mesh(cycles, nr=nr, nm=nm)

        # tests
        # (shape)
        np.testing.assert_equal(cmesh.shape, (nm, nr), err_msg=f"Shape of mesh is wrong, should be {(nm, nr)}")
        np.testing.assert_equal(cmesh.shape, rmesh.shape, err_msg="Shapes do not match: 'cmesh' and 'rmesh'")
        np.testing.assert_equal(cmesh.shape, mmesh.shape, err_msg="Shapes do not match: 'cmesh' and 'mmesh'")
        # (sum of counts; total and along each axis)
        np.testing.assert_equal(cycles[:, 2].sum(), cmesh.sum(), err_msg="Sum of counts and mesh not equal")
        np.testing.assert_array_equal(cycles_rebinned_range[:, 2], cmesh.sum(axis=0),
                                      err_msg=f"Sum of counts along mean axis (constant ranges) are wrong")
        np.testing.assert_array_equal(cycles_rebinned_mean[:, 2], cmesh.sum(axis=1),
                                      err_msg=f"Sum of counts along range axis (constant means) are wrong")
        # (bins along respective axes should match bins obtained by rebinning by 'range' and 'mean', respectively)
        np.testing.assert_array_equal(rmesh[0, :], cycles_rebinned_range[:, 0],
                                      err_msg="Range mesh error (transposed by 'accident'?)")
        np.testing.assert_array_equal(mmesh[:, 0], cycles_rebinned_mean[:, 1],
                                      err_msg="Mean mesh error (transposed by 'accident'?)")


if __name__ == '__main__':
    unittest.main()
