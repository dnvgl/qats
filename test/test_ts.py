# -*- coding: utf-8 -*-
"""
Module for testing TimeSeries class
"""

import os
import unittest
from datetime import datetime
import numpy as np
from qats import TimeSeries, TsDB
import copy


class TestTs(unittest.TestCase):
    def setUp(self):
        """
        Common setup for all tests
        """
        self.db = TsDB()
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env for conda build
        self.tsfile = os.path.join(os.path.dirname(__file__), '..', 'data', 'mooring.ts')
        self.db.load(self.tsfile)
        self.ts = self.db.get(name="Mooring line 4")
        # add datetime reference to ts for later testing
        self.ts._dtg_ref = datetime.now()

    def test_tsdb_getone_returns_timeseries_object(self):
        """
        Test that the input ts is a TimeSeries object
        """
        self.assertTrue(isinstance(self.ts, TimeSeries), "self.ts not TimeSeries object.")

    def test_dtg_start_is_correct_type(self):
        """
        Test that the dtg_start attribute has correct type
        """
        self.assertTrue(isinstance(self.ts._dtg_ref, datetime) or self.ts._dtg_ref is None,
                        "Expected 'dtg_start' datetime object or None")

    def test_time_dtg_equals_time(self):
        """
        Test that the time_as_datetime property is identical to the original time array (floats)
        """
        dt0 = self.ts.dtg_time[0]
        # subract start time as datetime and convert to float seconds
        time_from_datetime = [dt.total_seconds() for dt in [dti - dt0 for dti in self.ts.dtg_time]]

        # test equality with relative tolerance, there are minor round off errors with datetime objects which
        # breaks numpy.array_equal()
        self.assertTrue(np.allclose(np.array(time_from_datetime), self.ts.t, rtol=1.e-6, atol=0.), "The date-times are "
                                                                                                   "not equal to the "
                                                                                                   "time array.")

    def test_specifying_both_resample_and_twin_raises_assertionerror(self):
        """
        Test that an AssertionError is raised if one tries to specify both resampling to new time array and
        cropping to a time window at the same time.
        """
        try:
            _, _ = self.ts.get(twin=(0., 100.), resample=np.arange(0., 300., 0.01))
        except AssertionError:
            pass
        else:
            self.fail("The TimeSeries.get() method does not raise AssertionError if one tries to specify both "
                      "resampling to new time array and cropping to a time window at the same time.")

    def test_resampling_beyond_original_time_array_raises_valueerror(self):
        """
        Test that a ValueError is raised if one tries to resample the time series beyond the original time array.
        """
        try:
            _, _ = self.ts.get(resample=np.arange(-5., 300., 0.01))
        except ValueError:
            pass
        else:
            self.fail("The TimeSeries.get() method does not raise ValueError if one tries to resample/extrapolate "
                      "beyond the original time array.")

    def test_resampling_beyond_original_time_array_raises_valueerror_2(self):
        """
        Test that a ValueError is raised if one tries to resample the time series beyond the original time array.
        """
        try:
            _, _ = self.ts.get(resample=np.arange(0., 1.e6, 1000.))    # large step to avoid MemoryError
        except ValueError:
            pass
        else:
            self.fail("The TimeSeries.get() method does not raise ValueError if one tries to resample/extrapolate "
                      "beyond the original time array.")

    def test_filter_lp_hp(self):
        """
        Test that filter() method uses get() as intended, and that sum of lp and hp components equals original signal.

        Note: Test is setup to suit the exact signal tested - this is not a generic test.
        """
        twin = (1000, 1e12)
        freq = 0.03
        _, xtot = self.ts.get(twin=twin)
        # check 1: should not raise error
        _, xlo = self.ts.filter('lp', freq, twin=twin)
        _, xhi = self.ts.filter('hp', freq, twin=twin)
        # check 2: sum of components (almots) equals total signal
        deviation = np.max((xlo + xhi - xtot) / xtot)
        self.assertLessEqual(deviation, 0.02, "Sum of low- and high-pass components does not equal total signal")

    def test_data_is_dict_type(self):
        """
        Test that the data property returns dictionary
        """
        # todo: invoke type check below when the quality of the data property is assured
        # self.assertIsInstance(self.ts.data, dict, "Data property does not return dictionary.")
        try:
            self.ts.data
        except NotImplementedError:
            pass
        else:
            self.fail("Use of data property does not raise NotImplemented")

    def test_ts_copy_returns_unique_ts(self):
        """
        Test that the object returned from TimeSeries.copy is correct type and is not identical to the original
        """
        new_ts = self.ts.copy()

        self.assertIsInstance(new_ts, TimeSeries, "TimeSeries.copy() does not return a TimeSeries object, "
                                                  "but type '%s'." % type(new_ts))

        self.assertIsNot(self.ts.t, new_ts.t, "TimeSeries.copy() returns TimeSeries with time array which is bound "
                                              "to the time array of the original TimeSeries object.")

        self.assertIsNot(self.ts.x, new_ts.x, "TimeSeries.copy() returns TimeSeries with data array which is bound"
                                              "to the data array of the original TimeSeries object.")

    def test_copycopy_returns_unique_ts(self):
        """
        Test that the object returned from copy.copy(TimeSeries) is correct type and is not identical to the original
        """
        new_ts = copy.copy(self.ts)

        self.assertIsInstance(new_ts, TimeSeries, "copy.copy(TimeSeries) does not return a TimeSeries object, "
                                                  "but type '%s'." % type(new_ts))
        self.assertIsNot(self.ts.t, new_ts.t, "copy.copy(TimeSeries) returns TimeSeries with time array which is bound "
                                              "to the time array of the original TimeSeries object.")

        self.assertIsNot(self.ts.x, new_ts.x, "copy.copy(TimeSeries) returns TimeSeries with data array which is bound "
                                              "to the data array of the original TimeSeries object.")

    def test_max_equals_largest_maxima(self):
        """
        Test that the value returned from max() method equals the largest value in the array returned from maxima() method
        """
        twin = (500, 1.e12)
        self.assertEqual(np.max(self.ts.maxima(twin=twin)), self.ts.max(twin=twin),
                         "Method max() returns value which is different from the largest value from maxima() method")

    def test_min_equals_smallest_minima(self):
        """
        Test that the value returned from min() method equals the smalles value in the array returned from minima() method
        """
        twin = (500, 1.e12)
        self.assertEqual(np.min(self.ts.minima(twin=twin)), self.ts.min(twin=twin),
                         "Method min() returns value which is different from the smallest value from minima() method")


if __name__ == '__main__':
    unittest.main()
