# -*- coding: utf-8 -*-
"""
Module for testing readers.
The module utilizes TsDB.load() and .get() to read at least one time series from the file, to check that this does not
generate any exceptions.
"""
from qats import TsDB
import unittest
import os


class TestReadersSimaH5(unittest.TestCase):
    def setUp(self):
        self.db = TsDB()
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env for conda build
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data', 'sima_h5_versions')

    def test_h5_sima_timeseries_correctly_read(self):
        fn = 'example_h5_output.h5'
        self.db.load(os.path.join(self.data_directory, fn))
        t, x = self.db.get(name='TotalforceElement1')
        self.assertTrue(t.size > 1 and t.size == x.size,
                        "Reading of time series from .h5 file '%s' failed" % fn)


if __name__ == '__main__':
    unittest.main()
