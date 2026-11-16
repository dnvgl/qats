import os
import unittest

import numpy as np

from qats import TsDB
from qats.stats.paper import probability_paper
from qats.stats.weibull import Weibull

from . import requires_qt

FIT_OPTIONS = [
    dict(),
    dict(method="lse", tail=0.2),
    dict(method="mle", tail=0.2, loc="mean"),
    dict(method="msm", tail=0.2, tail_model="exceedances"),
    dict(method="mle", tail=0.2, tail_model="exceedances", loc="threshold"),
]


@requires_qt
class TestCalculateStats(unittest.TestCase):
    """The statistics and maxima/minima distribution shown in the GUI are those of the library."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "mooring.ts")
        cls.ts = TsDB.fromfile(path).get(name="mooring line 4")

    def check(self, fit, minima=False, kind="median", view="exceedances"):
        from qats.app.funcs import calculate_stats

        s = calculate_stats({"a": self.ts}, None, None, minima=minima, fit=fit, kind=kind, view=view)["a"]
        expected = self.ts.stats(
            statsdur=10800.0, quantiles=(0.37, 0.57, 0.9), is_minima=minima, include_sample=True, **fit
        )
        for key, value in expected.items():
            if key == "sample":
                np.testing.assert_array_equal(s[key], value)
            elif isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(s[key]), key)
            else:
                self.assertEqual(s[key], value, key)

        p = probability_paper(
            expected["sample"],
            Weibull(expected["wloc"], expected["wscale"], expected["wshape"]),
            kind=kind,
            tail=fit.get("tail"),
            tail_model=fit.get("tail_model", "censored"),
            view=view if fit.get("tail_model") == "exceedances" else None,
            is_minima=minima,
        )
        for attr in ("x", "y", "fitted", "fit_x", "fit_y", "yticks"):
            np.testing.assert_array_equal(getattr(s["paper"], attr), getattr(p, attr), err_msg=attr)
        self.assertEqual(s["paper"].paper, "rayleigh")
        self.assertEqual(s["paper"].kind, kind)

    def test_fit_options(self):
        for fit in FIT_OPTIONS:
            with self.subTest(**fit):
                self.check(fit)

    def test_minima(self):
        for fit in (dict(), dict(method="msm", tail=0.2, tail_model="exceedances")):
            with self.subTest(**fit):
                self.check(fit, minima=True)

    def test_plotting_positions(self):
        self.check(dict(), kind="mean")

    def test_views(self):
        # the view applies to a fit to the exceedances; the other fits are drawn with all maxima in either view
        for view in ("exceedances", "full-sample"):
            for fit in FIT_OPTIONS:
                with self.subTest(view=view, **fit):
                    self.check(fit, view=view)
            with self.subTest(view=view, minima=True):
                self.check(dict(method="msm", tail=0.2, tail_model="exceedances"), minima=True, view=view)

    def test_failed_fit(self):
        # mle to the exceedances with the location fitted doesn't converge on these maxima: NaN, points without a line
        from qats.app.funcs import calculate_stats

        s = calculate_stats({"a": self.ts}, None, None, fit=dict(method="mle", tail=0.2, tail_model="exceedances"))["a"]
        self.assertTrue(np.isnan(s["wloc"]))
        self.assertGreater(s["paper"].x.size, 0)
        self.assertEqual(s["paper"].fit_x.size, 0)


if __name__ == "__main__":
    unittest.main()
