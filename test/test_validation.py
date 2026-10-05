# -*- coding: utf-8 -*-
"""
Module for testing input validation (#148).

Invalid input raises ValueError or TypeError, which are still subclasses of AssertionError (until 6.0), and the
checks also work when Python runs with -O.
"""

import os
import re
import subprocess
import sys
import unittest

import numpy as np

from qats import TimeSeries, TsDB
from qats.fatigue.corrections import goodman_haigh
from qats.fatigue.sn import SNCurve, minersum
from qats.io.direct_access import read_ts_data
from qats.io.tdms import read_data as read_tdms_data
from qats.motions import transform_motion, velocity
from qats.stats import gumbel, weibull
from qats.stats.gumbel import Gumbel
from qats.stats.gumbelmin import GumbelMin
from qats.stats.weibull import Weibull

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

METHOD_GUMBEL = "Method must be either msm or lse or pwm or mle"
METHOD_GUMBELMIN = "Method must be either msm or lse or mle"
SCALE = "The scale parameter must be larger than 0."
REBIN = "Cycles must be rebinned for this plot - either 'n' or 'w' must be different from None"


def _ts():
    t = np.linspace(0.0, 100.0, 1001)
    return TimeSeries("x", t, np.sin(t))


def _gumbel(scale):
    g = Gumbel(loc=0.0, scale=1.0)
    g.scale = scale
    return g


def _gumbelmin(scale):
    g = GumbelMin(loc=0.0, scale=1.0)
    g.scale = scale
    return g


def _bilinear():
    return SNCurve("bilinear", m1=4.0, m2=5.0, loga1=15.117, nswitch=1e7)


def _db():
    db = TsDB()
    db.load(os.path.join(DATA, "mooring.ts"))
    return db


# (id, call, expected exception type, expected message: str for exact match or compiled regex)
CASES = [
    # qats.ts.TimeSeries
    (
        "TimeSeries: t and x lengths",
        lambda: TimeSeries("x", np.arange(3.0), np.arange(4.0)),
        ValueError,
        "Time and data must be of equal length.",
    ),
    (
        "TimeSeries: dtg_ref type",
        lambda: TimeSeries("x", np.arange(3.0), np.arange(3.0), dtg_ref="2020"),
        TypeError,
        "Expected 'dtg_ref' datetime object or None",
    ),
    (
        "TimeSeries: data type",
        lambda: TimeSeries("x", np.arange(2.0), np.array(["a", "b"])),
        TypeError,
        "Data (x) must be integers or floats not '<class 'numpy.str_'>'.",
    ),
    (
        "TimeSeries.get: resample and twin",
        lambda: _ts().get(twin=(0.0, 10.0), resample=np.arange(0.0, 10.0, 0.5)),
        ValueError,
        "Cannot specify both resampling to `newt` and cropping to time window `twin`.",
    ),
    (
        "TimeSeries.get: filterargs type",
        lambda: _ts().get(filterargs="lp"),
        TypeError,
        "Parameter filter should be either a list or tuple, not <class 'str'>.",
    ),
    (
        "TimeSeries.get: lowpass args",
        lambda: _ts().get(filterargs=("lp",)),
        ValueError,
        "Excepted 2 values in filterargs but got 1.",
    ),
    (
        "TimeSeries.get: highpass args",
        lambda: _ts().get(filterargs=("hp", 0.1, 0.2)),
        ValueError,
        "Excepted 2 values in filterargs but got 3.",
    ),
    (
        "TimeSeries.get: bandpass args",
        lambda: _ts().get(filterargs=("bp", 0.1)),
        ValueError,
        "Excepted 3 values in filterargs but got 2.",
    ),
    (
        "TimeSeries.get: bandblock args",
        lambda: _ts().get(filterargs=("bs", 0.1)),
        ValueError,
        "Excepted 3 values in filterargs but got 2.",
    ),
    (
        "TimeSeries.get: thresholdpass args",
        lambda: _ts().get(filterargs=["tp"]),
        ValueError,
        "Excepted 2 values in filterargs but got 1.",
    ),
    (
        "TimeSeries.plot_cycle_range: no rebinning",
        lambda: _ts().plot_cycle_range(n=None, w=None, show=False),
        ValueError,
        REBIN,
    ),
    (
        "TimeSeries.resample: no dt or t",
        lambda: _ts().resample(),
        ValueError,
        "Either new time step 'dt' or new time array 't' has to be specified.",
    ),
    (
        "TimeSeries.resample: extrapolation",
        lambda: _ts().resample(t=np.arange(0.0, 200.0, 1.0)),
        ValueError,
        "The new specified time array exceeds the original time array. Extrapolation is not allowed.",
    ),
    (
        "TimeSeries.resample: dt <= 0",
        lambda: _ts().resample(dt=0.0),
        ValueError,
        "The specified time step is to small.",
    ),
    # qats.tsdb.TsDB
    (
        "TsDB.plot_cycle_range: no rebinning",
        lambda: _db().plot_cycle_range(names="Surge", n=None, w=None, show=False),
        ValueError,
        REBIN,
    ),
    (
        "TsDB._read: retkeys length",
        lambda: _db()._read(_db().list(names="Surge"), retkeys=[]),
        ValueError,
        "The number of 'retkeys' is different from the number of 'keys'.",
    ),
    # qats.stats.gumbel
    ("Gumbel: loc None", lambda: Gumbel(loc=None, scale=1.0), ValueError, "Location parameter must be finite."),
    (
        "Gumbel: scale <= 0",
        lambda: Gumbel(loc=0.0, scale=0.0),
        ValueError,
        "Scale parameter must be finit and larger than 0.",
    ),
    (
        "Gumbel: scale nan",
        lambda: Gumbel(loc=0.0, scale=np.nan),
        ValueError,
        "Scale parameter must be finit and larger than 0.",
    ),
    (
        "Gumbel.ecdf: no data",
        lambda: Gumbel(loc=0.0, scale=1.0).ecdf(),
        ValueError,
        "Requires data/sample to be specified.",
    ),
    ("Gumbel.cdf: scale <= 0", lambda: _gumbel(-1.0).cdf(1.0), ValueError, SCALE),
    ("Gumbel.cdf: scale nan", lambda: _gumbel(np.nan).cdf(1.0), ValueError, SCALE),
    ("Gumbel.invcdf: scale <= 0", lambda: _gumbel(-1.0).invcdf(0.5), ValueError, SCALE),
    ("Gumbel.pdf: scale <= 0", lambda: _gumbel(-1.0).pdf(1.0), ValueError, SCALE),
    (
        "Gumbel.fit: method",
        lambda: Gumbel.fit(np.random.default_rng(1).gumbel(size=50), method="x"),
        ValueError,
        METHOD_GUMBEL,
    ),
    ("gumbel.bootstrap: method", lambda: gumbel.bootstrap(0.0, 1.0, 10, 2, method="x"), ValueError, METHOD_GUMBEL),
    (
        "gumbel.plot_fits: method",
        lambda: gumbel.plot_fits(np.arange(1.0, 20.0), methods=["x"]),
        ValueError,
        METHOD_GUMBEL,
    ),
    # qats.stats.gumbelmin
    (
        "GumbelMin.bootstrap: method",
        lambda: GumbelMin(loc=0.0, scale=1.0).bootstrap(size=10, method="x"),
        ValueError,
        METHOD_GUMBELMIN,
    ),
    (
        "GumbelMin.bootstrap: no data",
        lambda: GumbelMin(loc=0.0, scale=1.0).bootstrap(),
        ValueError,
        "Either size has to be specified or a sample has to be specified.",
    ),
    ("GumbelMin.fit: method", lambda: GumbelMin().fit(np.arange(1.0, 20.0), method="x"), ValueError, METHOD_GUMBELMIN),
    ("GumbelMin.cdf: scale <= 0", lambda: _gumbelmin(-1.0).cdf(1.0), ValueError, SCALE),
    ("GumbelMin.invcdf: scale <= 0", lambda: _gumbelmin(-1.0).invcdf(0.5), ValueError, SCALE),
    ("GumbelMin.pdf: scale <= 0", lambda: _gumbelmin(np.nan).pdf(1.0), ValueError, SCALE),
    # qats.stats.weibull
    (
        "Weibull.cdf: x below loc",
        lambda: Weibull(loc=1.0, scale=1.0, shape=2.0).cdf(0.5),
        ValueError,
        "The location parameter must be less than all items in data set",
    ),
    (
        "Weibull.pdf: x below loc",
        lambda: Weibull(loc=1.0, scale=1.0, shape=2.0).pdf(0.5),
        ValueError,
        "The location parameter must be less than all items in data set",
    ),
    (
        "Weibull.fit: method",
        lambda: Weibull.fit(np.arange(1.0, 20.0), method="x"),
        ValueError,
        "Method must be either msm or lse or mle or pwm or pwm2",
    ),
    (
        "weibull.bootstrap: method",
        lambda: weibull.bootstrap(0.0, 1.0, 2.0, 10, 2, method="x"),
        ValueError,
        METHOD_GUMBEL,
    ),
    # qats.fatigue
    (
        "SNCurve.a: bilinear",
        lambda: _bilinear().a,
        ValueError,
        "For bi-linear curves, use `a1` and `a2` instead of `a`",
    ),
    (
        "SNCurve.loga: bilinear",
        lambda: _bilinear().loga,
        ValueError,
        "For bi-linear curves, use `loga1` and `loga2` instead of `loga`",
    ),
    (
        "SNCurve.m: bilinear",
        lambda: _bilinear().m,
        ValueError,
        "For bi-linear curves, use `m1` and `m2` instead of `m`",
    ),
    (
        "minersum: th with callable",
        lambda: minersum([10.0], [1.0], lambda s: 1e6, th=0.03),
        ValueError,
        "Parameter 'th' is only accepted if 'sn' is a dict or an SNCurve instance. "
        "For other cases, use parameter 'args' or 'kwds'.",
    ),
    (
        "minersum: sn not callable",
        lambda: minersum([10.0], [1.0], 5.0),
        TypeError,
        "Parameter 'sn' must be dict, callable or class instance with callable method 'n'",
    ),
    (
        "goodman_haigh: cycles shape",
        lambda: goodman_haigh(np.zeros(3), 100.0),
        ValueError,
        "Cycles must be specified as 2D array or shape (n, 2) (or: list of 2-tuples)",
    ),
    # qats.motions
    (
        "transform_motion: dofs",
        lambda: transform_motion(np.zeros((5, 10)), (0.0, 0.0, 0.0)),
        ValueError,
        "Motion must be of shape (6, nt) (6-dof motion), got (5, 10)",
    ),
    (
        "transform_motion: newref size",
        lambda: transform_motion(np.zeros((6, 10)), (0.0, 0.0)),
        ValueError,
        "Specified position must be list/tuple with three values, got 2",
    ),
    (
        "velocity: x ndim",
        lambda: velocity(np.zeros((2, 2, 2)), 0.1),
        ValueError,
        "Input signal 'x' must be 1- or 2-D array",
    ),
    (
        "velocity: t size",
        lambda: velocity(np.zeros(10), np.arange(5.0)),
        ValueError,
        "If time array is specified, it must match number of time steps in input signal(s)",
    ),
    # qats.io
    (
        "read_ts_data: index",
        lambda: read_ts_data(os.path.join(DATA, "mooring.ts"), ind=[9999]),
        ValueError,
        re.compile(r"Requested time series no\. 9999, but there are only \d+ time series on file"),
    ),
    (
        "tdms read_data: name",
        lambda: read_tdms_data(os.path.join(DATA, "data.tdms"), names=["no_group"]),
        ValueError,
        "Unable to parse group name and channel name from no_group.",
    ),
]

# checks that cannot be reached yet, because the method fails before them
UNREACHABLE = {
    "GumbelMin.bootstrap: method": "GumbelMin.bootstrap() raises AttributeError before its checks (#157)",
    "GumbelMin.bootstrap: no data": "GumbelMin.bootstrap() raises AttributeError before its checks (#157)",
}


class TestValidation(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot as plt

        backend = plt.get_backend()
        plt.switch_backend("Agg")
        self.addCleanup(plt.switch_backend, backend)
        self.addCleanup(plt.close, "all")

    def test_invalid_input(self):
        """Each check raises the documented exception type, still an AssertionError, with the same message."""
        for name, call, exc_type, message in CASES:
            with self.subTest(name):
                if name in UNREACHABLE:
                    self.skipTest(UNREACHABLE[name])
                with self.assertRaises(exc_type) as cm:
                    call()
                self.assertIsInstance(cm.exception, AssertionError, "must stay an AssertionError until 6.0")
                if isinstance(message, re.Pattern):
                    self.assertRegex(str(cm.exception), message)
                else:
                    self.assertEqual(str(cm.exception), message)

    def test_existing_code_catching_assertionerror_still_works(self):
        try:
            Gumbel(loc=0.0, scale=-1.0)
        except AssertionError:
            pass
        else:
            self.fail("invalid input is no longer caught by `except AssertionError`")

    def test_gui_logging_level(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy.QtWidgets import QApplication

        from qats.app.gui import Qats

        _ = QApplication.instance() or QApplication([])
        with self.assertRaises(ValueError) as cm:
            Qats(logging_level="bad")
        self.assertIsInstance(cm.exception, AssertionError)
        self.assertEqual(str(cm.exception), "invalid logging level: 'bad'")

    def test_checks_with_python_optimizations(self):
        """With `python -O`, assert statements are removed; the input checks must still raise."""
        code = (
            "import numpy as np\n"
            "from qats import TimeSeries\n"
            "from qats.stats.gumbel import Gumbel\n"
            "from qats.fatigue.sn import minersum\n"
            "assert False, 'assert statements should be removed by -O'\n"
            "for call, exc in (\n"
            "    (lambda: Gumbel(loc=0.0, scale=-1.0), ValueError),\n"
            "    (lambda: TimeSeries('x', np.arange(3.0), np.arange(4.0)), ValueError),\n"
            "    (lambda: minersum([10.0], [1.0], 5.0), TypeError),\n"
            "):\n"
            "    try:\n"
            "        call()\n"
            "    except exc:\n"
            "        pass\n"
            "    else:\n"
            "        raise SystemExit('check did not raise with -O')\n"
            "print('ok')\n"
        )
        # fixed command: the current interpreter with -O and the code above
        result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)  # noqa: S603
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
