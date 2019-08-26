# -*- coding: utf-8 -*-
"""
Module for testing readers.
The module utilizes TsDB.fromfile and .get() to read at least one time series from the file, to check that this does not
generate any exceptions.
"""
from qats import TsDB
import unittest
import os

# todo: add test class for csv
# todo: add test class for direct_access
# todo: add test class for matlab
# todo: add test class for sima (.bin)
# todo: add test class for other (ascii)


class TestReadersSimaH5(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env for conda build
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'sima_h5_versions')
        # .h5 files used are from example "Riflex Simple Flexible Riser" available in SIMA, exported from various
        # SIMA versions
        self.h5_files = [
            # file name, number of (time series) keys
            ("results_SIMA341.h5", 768),
            ("results_SIMA35.h5", 774),
            ("results_SIMA36.h5", 774),
            ("results_SIMA37dev.h5", 774),
        ]

    def test_h5_sima_load_correct_number_of_keys(self):
        for h5_file, nts in self.h5_files:
            db = TsDB.fromfile(os.path.join(self.data_directory, h5_file))
            self.assertEqual(db.n, nts, f"Failed to identify correct number of time series on '{h5_file}'")

    def test_h5_sima_timeseries_correctly_read(self):
        for h5_file, _ in self.h5_files:
            db = TsDB.fromfile(os.path.join(self.data_directory, h5_file))
            ts = db.get(ind=0)  # should not fail
            self.assertTrue(ts.t.size > 1 and ts.t.size == ts.x.size, f"Reading of time series from '{h5_file}' failed")


if __name__ == '__main__':
    unittest.main()