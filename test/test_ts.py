# -*- coding: utf-8 -*-
"""
Module for testing TimeSeries class
"""

import copy
import os
import unittest
from datetime import datetime

import numpy as np

from qats import TimeSeries, TsDB
from qats._validation import QatsTypeError, QatsValueError
from qats.stats.weibull import Weibull, _pwm3, lse, weibull2gumbel


class TestTs(unittest.TestCase):
    def setUp(self):
        """
        Common setup for all tests
        """
        self.db = TsDB()
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env for conda build
        self.tsfile = os.path.join(os.path.dirname(__file__), "..", "data", "mooring.ts")
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
        self.assertTrue(
            isinstance(self.ts._dtg_ref, datetime) or self.ts._dtg_ref is None,
            "Expected 'dtg_start' datetime object or None",
        )

    def test_time_dtg_equals_time(self):
        """
        Test that the time_as_datetime property is identical to the original time array (floats)
        """
        dt0 = self.ts.dtg_time[0]
        # subract start time as datetime and convert to float seconds
        time_from_datetime = [dt.total_seconds() for dt in [dti - dt0 for dti in self.ts.dtg_time]]

        # test equality with relative tolerance, there are minor round off errors with datetime objects which
        # breaks numpy.array_equal()
        self.assertTrue(
            np.allclose(np.array(time_from_datetime), self.ts.t, rtol=1.0e-6, atol=0.0),
            "The date-times are not equal to the time array.",
        )

    def test_specifying_both_resample_and_twin_raises_assertionerror(self):
        """
        Test that an AssertionError is raised if one tries to specify both resampling to new time array and
        cropping to a time window at the same time.
        """
        try:
            _, _ = self.ts.get(twin=(0.0, 100.0), resample=np.arange(0.0, 300.0, 0.01))
        except AssertionError:
            pass
        else:
            self.fail(
                "The TimeSeries.get() method does not raise AssertionError if one tries to specify both "
                "resampling to new time array and cropping to a time window at the same time."
            )

    def test_resampling_beyond_original_time_array_raises_valueerror(self):
        """
        Test that a ValueError is raised if one tries to resample the time series beyond the original time array.
        """
        try:
            _, _ = self.ts.get(resample=np.arange(-5.0, 300.0, 0.01))
        except ValueError:
            pass
        else:
            self.fail(
                "The TimeSeries.get() method does not raise ValueError if one tries to resample/extrapolate "
                "beyond the original time array."
            )

    def test_resampling_beyond_original_time_array_raises_valueerror_2(self):
        """
        Test that a ValueError is raised if one tries to resample the time series beyond the original time array.
        """
        try:
            _, _ = self.ts.get(resample=np.arange(0.0, 1.0e6, 1000.0))  # large step to avoid MemoryError
        except ValueError:
            pass
        else:
            self.fail(
                "The TimeSeries.get() method does not raise ValueError if one tries to resample/extrapolate "
                "beyond the original time array."
            )

    def test_filter_lp_hp(self):
        """
        Test that filter() method uses get() as intended, and that sum of lp and hp components equals original signal.

        Note: Test is setup to suit the exact signal tested - this is not a generic test.
        """
        twin = (1000, 1e12)
        freq = 0.03
        _, xtot = self.ts.get(twin=twin)
        # check 1: should not raise error
        _, xlo = self.ts.filter("lp", freq, twin=twin)
        _, xhi = self.ts.filter("hp", freq, twin=twin)
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

        self.assertIsInstance(
            new_ts, TimeSeries, "TimeSeries.copy() does not return a TimeSeries object, but type '%s'." % type(new_ts)
        )

        self.assertIsNot(
            self.ts.t,
            new_ts.t,
            "TimeSeries.copy() returns TimeSeries with time array which is bound "
            "to the time array of the original TimeSeries object.",
        )

        self.assertIsNot(
            self.ts.x,
            new_ts.x,
            "TimeSeries.copy() returns TimeSeries with data array which is bound"
            "to the data array of the original TimeSeries object.",
        )

    def test_copycopy_returns_unique_ts(self):
        """
        Test that the object returned from copy.copy(TimeSeries) is correct type and is not identical to the original
        """
        new_ts = copy.copy(self.ts)

        self.assertIsInstance(
            new_ts,
            TimeSeries,
            "copy.copy(TimeSeries) does not return a TimeSeries object, but type '%s'." % type(new_ts),
        )
        self.assertIsNot(
            self.ts.t,
            new_ts.t,
            "copy.copy(TimeSeries) returns TimeSeries with time array which is bound "
            "to the time array of the original TimeSeries object.",
        )

        self.assertIsNot(
            self.ts.x,
            new_ts.x,
            "copy.copy(TimeSeries) returns TimeSeries with data array which is bound "
            "to the data array of the original TimeSeries object.",
        )

    def test_max_equals_largest_maxima(self):
        """
        Test that the value returned from max() method equals the largest value in the array returned from maxima()
        method
        """
        twin = (500, 1.0e12)
        self.assertEqual(
            np.max(self.ts.maxima(twin=twin)),
            self.ts.max(twin=twin),
            "Method max() returns value which is different from the largest value from maxima() method",
        )

    def test_min_equals_smallest_minima(self):
        """
        Test that the value returned from min() method equals the smalles value in the array returned from minima()
        method
        """
        twin = (500, 1.0e12)
        self.assertEqual(
            np.min(self.ts.minima(twin=twin)),
            self.ts.min(twin=twin),
            "Method min() returns value which is different from the smallest value from minima() method",
        )

    def test_correct_number_of_maxima(self):
        """
        Test that the correct number of maxima/peaks are found
        """
        self.assertEqual(139, len(self.ts.maxima(twin=None, threshold=None)))
        self.assertEqual(128, len(self.ts.maxima(twin=(500, 1.0e12), threshold=None)))
        self.assertEqual(128, len(self.ts.maxima(twin=(500, 1.0e12), threshold=500.0)))

    def test_correct_number_of_minima(self):
        """
        Test that the correct number of minima/troughs are found
        """
        # flip the time series, then the minima shall be equal to maxima in the original timeseries multiplied with -1
        tsflip = copy.copy(self.ts)
        tsflip.x *= -1
        self.assertEqual(139, len(tsflip.minima(twin=None, threshold=None)))
        self.assertEqual(128, len(tsflip.minima(twin=(500, 1.0e12), threshold=None)))
        self.assertEqual(128, len(tsflip.minima(twin=(500, 1.0e12), threshold=-500.0)))

    def test_stats_weibull_parameters(self):
        """
        Test that stats() reports the Weibull and Gumbel parameters of QATS 5.5.0, also when the Weibull location
        parameter is above the smallest maximum (where qats.stats.weibull.pwm() raises)
        """
        s = self.ts.stats()
        np.testing.assert_allclose(
            (s["wloc"], s["wscale"], s["wshape"], s["gloc"], s["gscale"]),
            (628.6977387879958, 16.075909034042294, 0.7713810419072378, 772.9552088937547, 34.41691465498858),
            rtol=1e-12,
        )

    def test_fit_weibull_unchanged(self):
        """
        Test that fit_weibull() without the arguments added in 5.6.0 gives the parameters of QATS 5.5.0
        """
        np.testing.assert_allclose(
            self.ts.fit_weibull().params, (625.538004079687, 20.93291628966276, 0.9115119725895221), rtol=1e-12
        )


class TestTsWeibullFit(unittest.TestCase):
    """Weibull fit options of TimeSeries.stats() and fit_weibull()."""

    def setUp(self):
        # narrow-banded process around the mean level 100 for three hours
        rng = np.random.default_rng(1)
        t = np.arange(0.0, 10800.0, 0.5)
        f = np.linspace(0.08, 0.12, 60)
        x = 100.0 + np.sum(np.cos(2.0 * np.pi * np.outer(t, f) + rng.uniform(0.0, 2.0 * np.pi, f.size)), axis=1)
        self.ts = TimeSeries("x", t, x)
        self.mx = self.ts.stats(include_sample=True)["sample"]
        self.tail = 0.2
        # tail maxima: mean rank i / (n + 1) at or above 1 - tail, threshold value: the largest maximum below them
        n = self.mx.size
        self.r = int(np.ceil((1.0 - self.tail) * (n + 1))) - 1
        self.u = np.sort(self.mx)[self.r - 1]

    def check_cell(self, stats_kwargs, sample, fit_kwargs, ndist):
        """stats() and fit_weibull() give the parameters of Weibull.fit(), and Gumbel parameters from ndist maxima"""
        expected = Weibull.fit(sample, **fit_kwargs).params
        s = self.ts.stats(**stats_kwargs)
        self.assertEqual((s["wloc"], s["wscale"], s["wshape"]), expected)
        self.assertEqual(self.ts.fit_weibull(**stats_kwargs).params, expected)
        gn = round(10800.0 / (self.ts.t[-1] - self.ts.t[0]) * ndist)
        self.assertEqual(s["gn"], gn)
        self.assertEqual((s["gloc"], s["gscale"]), weibull2gumbel(*expected, gn))

    def test_threshold_value(self):
        s = self.ts.stats(method="lse", tail=self.tail)
        self.assertEqual(s["wthreshold"], self.u)
        self.assertEqual(s["wn"], self.mx.size - self.r)
        # the maxima above u are the tail maxima
        self.assertEqual(np.count_nonzero(self.mx > self.u), self.mx.size - self.r)

    def test_all_maxima(self):
        for method in ("pwm", "pwm2", "msm", "lse", "mle"):
            for loc, fixed in ((None, None), ("mean", self.ts.mean()), (95.0, 95.0)):
                if method == "pwm" and loc is None:
                    continue  # the unchecked default fit, see test_default_unchanged
                with self.subTest(method=method, loc=loc):
                    fit_kwargs = dict(method=method) if fixed is None else dict(method=method, loc=fixed)
                    self.check_cell(dict(method=method, loc=loc), self.mx, fit_kwargs, self.mx.size)

    def test_censored(self):
        for method in ("lse", "mle"):
            for loc, fixed in ((None, None), ("mean", self.ts.mean()), (95.0, 95.0)):
                with self.subTest(method=method, loc=loc):
                    fit_kwargs = dict(method=method, tail=self.tail)
                    if fixed is not None:
                        fit_kwargs["loc"] = fixed
                    self.check_cell(dict(method=method, tail=self.tail, loc=loc), self.mx, fit_kwargs, self.mx.size)

    def test_censored_lse_is_lse_threshold(self):
        s = self.ts.stats(method="lse", tail=self.tail)
        self.assertEqual((s["wloc"], s["wscale"], s["wshape"]), lse(self.mx, threshold=1 - self.tail))

    def test_exceedances(self):
        exceedances = self.mx[self.mx > self.u]
        for method in ("pwm", "pwm2", "msm", "lse", "mle"):
            for loc, fixed in ((None, None), ("threshold", self.u), ("mean", self.ts.mean()), (95.0, 95.0)):
                if method == "mle" and loc is None:
                    continue  # does not converge on this sample, see test_failed_fit
                with self.subTest(method=method, loc=loc):
                    fit_kwargs = dict(method=method) if fixed is None else dict(method=method, loc=fixed)
                    self.check_cell(
                        dict(method=method, tail=self.tail, tail_model="exceedances", loc=loc),
                        exceedances,
                        fit_kwargs,
                        exceedances.size,
                    )

    def test_reported_options(self):
        s = self.ts.stats(method="MLE", tail=self.tail, tail_model="exceedances", loc="threshold")
        self.assertEqual(
            {k: s[k] for k in ("wmethod", "wtail", "wtail_model", "wloc_from", "wthreshold", "wn")},
            dict(
                wmethod="mle",
                wtail=self.tail,
                wtail_model="exceedances",
                wloc_from="threshold",
                wthreshold=self.u,
                wn=self.mx.size - self.r,
            ),
        )
        s = self.ts.stats()
        self.assertEqual(
            {k: s[k] for k in ("wmethod", "wtail", "wtail_model", "wloc_from", "wthreshold", "wn")},
            dict(wmethod="pwm", wtail=None, wtail_model=None, wloc_from="fitted", wthreshold=None, wn=self.mx.size),
        )
        self.assertEqual(self.ts.stats(method="pwm2")["wloc_from"], "value")
        self.assertEqual(self.ts.stats(method="lse", loc=95.0)["wloc_from"], "value")

    def test_fit_info(self):
        # fit_weibull() keeps the settings that stats() reports, with the location given as a number
        cases = [
            dict(method="msm"),
            dict(method="lse", tail=self.tail),
            dict(method="lse", tail=self.tail, loc="mean"),
            dict(method="msm", tail=self.tail, tail_model="exceedances"),
            dict(method="mle", tail=self.tail, tail_model="exceedances", loc="threshold"),
        ]
        for kwargs in cases:
            with self.subTest(**kwargs):
                s = self.ts.stats(**kwargs)
                info = self.ts.fit_weibull(**kwargs).fit_info
                loc_from = "value" if s["wloc_from"] in ("mean", "threshold") else s["wloc_from"]
                expected = dict(
                    method=s["wmethod"],
                    tail=s["wtail"],
                    tail_model=s["wtail_model"],
                    loc_from=loc_from,
                    threshold=s["wthreshold"],
                    n=s["wn"],
                )
                self.assertEqual(info, expected)

    def test_fit_keeps_all_maxima(self):
        w = self.ts.fit_weibull(method="msm", tail=self.tail, tail_model="exceedances")
        np.testing.assert_array_equal(np.sort(w.data), np.sort(self.mx))

    def test_default_unchanged(self):
        s = self.ts.stats()
        self.assertEqual((s["wloc"], s["wscale"], s["wshape"]), _pwm3(self.mx))
        self.assertEqual(s["gn"], round(10800.0 / (self.ts.t[-1] - self.ts.t[0]) * self.mx.size))

    def test_minima(self):
        s = self.ts.stats(is_minima=True, include_sample=True, method="lse", tail=self.tail, loc="mean")
        mn = -s["sample"]
        expected = Weibull.fit(mn, method="lse", tail=self.tail, loc=-self.ts.mean()).params
        self.assertEqual((s["wloc"], s["wscale"], s["wshape"]), expected)

    def test_failed_fit(self):
        # the fit to the exceedances by mle with the location parameter fitted does not converge on this sample
        kwargs = dict(method="mle", tail=self.tail, tail_model="exceedances")
        s = self.ts.stats(**kwargs)
        for key in ("wloc", "wscale", "wshape", "gn", "gloc", "gscale", "p_90.00"):
            with self.subTest(key=key):
                self.assertTrue(np.isnan(s[key]))
        self.assertEqual(s["wthreshold"], self.u)
        with self.assertRaisesRegex(QatsValueError, "did not converge"):
            self.ts.fit_weibull(**kwargs)
        # a fixed location parameter at the smallest fitted maximum
        loc = np.sort(self.mx)[self.r]
        self.assertTrue(np.isnan(self.ts.stats(method="lse", tail=self.tail, loc=loc)["wloc"]))
        with self.assertRaisesRegex(QatsValueError, "must be below"):
            self.ts.fit_weibull(method="lse", tail=self.tail, loc=loc)

    def test_not_possible(self):
        cases = [
            (dict(method="pwm", tail=0.2), "no censored tail fit"),
            (dict(method="pwm2", tail=0.2), "no censored tail fit"),
            (dict(method="msm", tail=0.2), "no censored tail fit"),
            (dict(method="lse", tail=0.2, loc="threshold"), "only in a fit to the exceedances"),
            (dict(method="lse", loc="threshold"), "only in a fit to the exceedances"),
            (dict(method="lse", tail_model="exceedances"), "needs the tail fraction"),
        ]
        for kwargs, message in cases:
            for f in (self.ts.stats, self.ts.fit_weibull):
                with self.subTest(fn=f.__name__, **kwargs), self.assertRaisesRegex(QatsValueError, message):
                    f(**kwargs)
        # the message names the alternatives
        with self.assertRaisesRegex(QatsValueError, "'lse' or 'mle'.*tail_model='exceedances'"):
            self.ts.stats(tail=0.2)

    def test_invalid_options(self):
        cases = [
            (dict(method="gumbel"), QatsValueError),
            (dict(method=None), QatsValueError),
            (dict(method="lse", tail=0.0), QatsValueError),
            (dict(method="lse", tail=1.5), QatsValueError),
            (dict(method="lse", tail="0.2"), QatsTypeError),
            (dict(method="lse", tail=0.2, tail_model="peaks"), QatsValueError),
            (dict(method="lse", loc="median"), QatsValueError),
            (dict(method="lse", loc=[95.0]), QatsTypeError),
        ]
        for kwargs, error in cases:
            for f in (self.ts.stats, self.ts.fit_weibull):
                with self.subTest(fn=f.__name__, **{k: repr(v) for k, v in kwargs.items()}):
                    with self.assertRaises(error):
                        f(**kwargs)


class TestTsPlots(unittest.TestCase):
    """Plot methods, run with the non-interactive Agg backend."""

    def setUp(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        backend = plt.get_backend()
        plt.switch_backend("Agg")
        self.addCleanup(plt.switch_backend, backend)
        self.addCleanup(plt.close, "all")
        t = np.linspace(0.0, 1000.0, 20001)
        self.ts = TimeSeries("x", t, np.sin(0.3 * t) + 0.2 * np.sin(2.1 * t))

    def test_plot_cycle_rangemean3d(self):
        """The 3D cycle range-mean plot is drawn on 3D axes (#150)."""
        self.ts.plot_cycle_rangemean3d(nr=10, nm=10, show=False)
        fig = self.plt.gcf()
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(fig.axes[0].name, "3d")

    def test_plot_cycle_rangemean3d_reuses_figure_axes(self):
        """Plotting twice to the same figure number reuses the 3D axes instead of adding new ones."""
        self.ts.plot_cycle_rangemean3d(nr=10, nm=10, show=False, num=1)
        self.ts.plot_cycle_rangemean3d(nr=10, nm=10, show=False, num=1)
        self.assertEqual(len(self.plt.figure(num=1).axes), 1)


if __name__ == "__main__":
    unittest.main()
