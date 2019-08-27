# -*- coding: utf-8 -*-
"""
Module for testing readers.
The module utilizes TsDB.fromfile and .get() to read at least one time series from the file, to check that this does not
generate any exceptions.
"""
from qats import TsDB
import unittest
import os

# todo: add test class for matlab


class TestAllReaders(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        # file name, number of (time series) keys
        self.files = [
            ("results_SIMA341.h5", 768),
            ("results_SIMA35.h5", 774),
            ("results_SIMA36.h5", 774),
            ("results_SIMA37dev.h5", 774),
            ("n_elmfor.bin", 27),
            ("n_elmtra.bin", 162),
            ("mooring.ts", 14),
            ("simo_p.ts", 22),
            ("simo_p_out.ts", 22),
            ("simo_r1.ts", 22),
            ("simo_r2.ts", 22),
            ("simo_trans.ts", 12),
            ("simo_n.tda", 6),
            ("decay.tda", 6),
            ("example.csv", 6),
            ("model_test_data.dat", 39),
        ]

    def test_correct_number_of_timeseries(self):
        for filename, nts in self.files:
            db = TsDB.fromfile(os.path.join(self.data_directory, filename))
            self.assertEqual(nts, db.n, f"Failed to identify correct number of time series on '{filename}'")

    def test_correct_timeseries_size(self):
        for filename, _ in self.files:
            db = TsDB.fromfile(os.path.join(self.data_directory, filename))
            ts = db.get(ind=0)  # should not fail
            self.assertTrue(ts.t.size > 1 and ts.t.size == ts.x.size, f"Reading of time series from '{filename}' failed")


if __name__ == '__main__':
    unittest.main()
