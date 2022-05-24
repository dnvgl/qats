# -*- coding: utf-8 -*-
"""
Module for testing io.
The module utilizes TsDB.fromfile and .get() to read at least one time series from the file, to check that this does not
generate any exceptions.
"""
from qats import TsDB
import unittest
import os
import sys

# todo: add test class for matlab


class TestAllReaders(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
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
        failed = []
        for filename, nts in self.files:
            db = TsDB.fromfile(os.path.join(self.data_directory, filename))
            if not nts == db.n:
                failed.append(f"{filename} ({nts} != {db.n})")
        self.assertTrue(len(failed) == 0,
                        f"Failed to identify correct number of time series on {len(failed)} file(s):\n   *** " +
                        f"\n   *** ".join(failed))

    def test_correct_timeseries_size(self):
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


if __name__ == '__main__':
    unittest.main()
