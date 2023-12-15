# -*- coding: utf-8 -*-
"""
Module for testing signal processing functions
"""
import os
import unittest

import numpy as np

from qats import TsDB
from qats.signal import find_reversals
from qats.fatigue.rainflow import reversals


class TestReversals(unittest.TestCase):
    def setUp(self):
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.peaks_file = "example.tension.ts"
        self.peaks_path = os.path.join(self.data_directory, self.peaks_file)

    def test_find_reversals(self):
        """ 
        The function `qats.signal.find_reversals()` identifies all turning points of 
        a the specified signal. This is similar to what `qats.fatigue.rainflow.reversals()`
        does, but in some cases they produce slightly different arrays.
        However, when the turning points from `find_reversals()` are passed through
        `reversals()` (with `endpoints=True`), the outcome should be the same as if 
        the signal itself was passed through `reversals()`.
        """
        db = TsDB.fromfile(self.peaks_path)
        ts = db.get("Tension")

        # find reversals by fatigue.rainflow.reversals()
        rev_rainflow = np.fromiter(reversals(ts.x), dtype=float)
        # find reversals by signal.find_maxima()
        rev_signal, _ = find_reversals(ts.x)

        # for this time series, these two method does not result in the same turning points
        #   note:   this is not a requirement, but we ensure that this is the case now for 
        #           the value of the next check 
        self.assertNotEqual(rev_signal.size, rev_rainflow.size)

        # however, passing the turning points from `find_reversals()` through 
        # `reversals()` (with `endpoints=True`) should result in identical arrays
        rev_combined = np.fromiter(reversals(rev_signal, endpoints=True), dtype=float)

        # now check that this is actually the case
        self.assertEqual(rev_combined.size, rev_rainflow.size)
        np.testing.assert_array_equal(rev_combined, rev_rainflow)
        

if __name__ == '__main__':
    unittest.main()
