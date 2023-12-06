# -*- coding: utf-8 -*-
"""
Module for testing io.
The module utilizes TsDB.fromfile and .get() to read at least one time series from the file, to check that this does not
generate any exceptions.
"""
import os
import sys
import unittest
from pathlib import Path

from qats import TsDB
from qats.io.other import (read_ascii_names, read_ascii_data, read_dat_names, read_dat_data)

# todo: add test class for matlab

ROOT = Path(__file__).resolve().parent


class TestAllReaders(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env
        self.data_directory = os.path.join(ROOT, '..', 'data')
        # file name, number of (time series) keys
        self.files = [
            # sima h5 files
            ("results_SIMA341.h5", 768),
            ("results_SIMA35.h5", 774),
            ("results_SIMA36.h5", 774),
            ("results_SIMA37dev.h5", 774),
            # sima bin files (riflex)
            ("n_elmfor.bin", 27),
            ("n_elmtra.bin", 162),
            ("wt_blresp.bin", 21),
            ("wt_witurb.bin", 23),
            ("sima_witurb.bin", 23),
            # direct access files
            ("mooring.ts", 14),
            ("simo_p.ts", 22),
            ("simo_p_out.ts", 22),
            ("simo_r1.ts", 22),
            ("simo_r2.ts", 22),
            ("simo_trans.ts", 12),
            ("simo_n.tda", 6),
            ("decay.tda", 6),
            # csv files
            ("example.csv", 6),
            ("integer.csv", 1),
            ("negative_time.csv", 1),
            # # ascii files
            ("model_test_data.dat", 39),
            # # tdms files
            ("data.tdms", 4),
        ]

    def test_correct_number_of_timeseries(self):
        """ Read key file, check number of keys (data not loaded) """
        failed = []
        for filename, nts in self.files:
            db = TsDB.fromfile(os.path.join(self.data_directory, filename))
            if not nts == db.n:
                failed.append(f"{filename} ({nts} != {db.n})")
        self.assertTrue(len(failed) == 0,
                        f"Failed to identify correct number of time series on {len(failed)} file(s):\n   *** " +
                        f"\n   *** ".join(failed))

    def test_correct_timeseries_size(self):
        """ Load time series: check that it loads and that t.size matches x.size """
        failed = []
        for filename, _ in self.files:
            try:
                db = TsDB.fromfile(os.path.join(self.data_directory, filename))
                ts = db.get(ind=0)  # should not fail
                self.assertTrue(ts.t.size > 1 and ts.t.size == ts.x.size,
                                f"Did not read time series correctly (t.size = {ts.t.size}, x.size = {ts.x.size})")
            except Exception:
                exctype, excvalue, _ = sys.exc_info()
                exctypestr = str(exctype).lstrip("<class '").rstrip("'>")  # e.g. <class 'IndexError'>  =>  IndexError
                failed.append(f"{filename}: {exctypestr}: {excvalue}")
        self.assertTrue(len(failed) == 0,
                        f"Failed to read time series from {len(failed)} file(s):\n   *** " +
                        f"\n   *** ".join(failed))


class TestASCIIReaders(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env
        self.data_directory = os.path.join(ROOT, '..', 'data')
        # file name, number of (time series) keys
        self.standard_file = os.path.join(self.data_directory, "model_test_data.dat")
        self.file_with_meta = os.path.join(self.data_directory, "model_test_tab.txt")
        
    def test_names_from_standard_file(self):
        names0 = read_dat_names(self.standard_file)
        names1 = read_ascii_names(self.standard_file)
        self.assertEqual(len(names0), len(names1), "The two ascii readers return different number of time series names")
        self.assertEqual(names0, names1, "The two ascii readers return different time series names")
    
    def test_data_from_standard_file(self):
        data0 = read_dat_data(self.standard_file)
        data1 = read_ascii_data(self.standard_file)
        
        # check that first data row is found correctly
        self.assertEqual(data0[0, 0], data1[0, 0], "The two ascii readers return different t0.")
        
        # check dimensions
        self.assertEqual(data0.shape, data1.shape, "The two ascii readers return data of different dimensions.")
    
    def test_names_from_file_with_meta(self):
        names0 = read_ascii_names(self.file_with_meta)
        names1 = "X Y Z Roll Pitch Yaw T1 T2 T3 F1 F2 F3 cable1 cable2 cable3".split()
        self.assertEqual(len(names0), len(names1), "The two ascii readers return different number of time series names")
        self.assertEqual(names0, names1, "The two ascii readers return different time series names")

    def test_data_from_file_with_meta(self):
        data = read_ascii_data(self.file_with_meta)
        
        # check that first data row is found correctly
        self.assertEqual(data[0, 0], 0., "The two ascii readers return different t[0].")
        self.assertEqual(data[0, 1], 0.031623, "The two ascii readers return different t[1].")
        
        # check dimensions
        self.assertEqual(data.shape, (16, 19141), "The two ascii readers return data of different dimensions.")

if __name__ == '__main__':
    unittest.main()
