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
# todo: add test class for simo (.tda)
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


class TestReadersCSV(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        # time series arranged by column on .csv files
        self.files = [
            # file name, number of (time series) keys
            ("example.csv", 6),
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


class TestReadersDirectAccess(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        # time series arranged by column on .csv files
        self.files = [
            # file name, number of (time series) keys
            ("mooring.ts", 14),
            ("simo_p.ts", 22),
            ("simo_p_out.ts", 22),
            ("simo_r1.ts", 22),
            ("simo_r2.ts", 22),
            ("simo_trans.ts", 12),
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


class TestReadersSimaBin(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        # time series arranged by column on .csv files
        self.files = [
            # file name, number of (time series) keys
            ("n_elmfor.bin", 27),
            ("n_elmtra.bin", 162),
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
