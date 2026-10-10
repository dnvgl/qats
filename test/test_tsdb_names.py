# -*- coding: utf-8 -*-
"""
TsDB name lookup and common path behave the same on every platform: names are matched ignoring case, with '/' and
'\\' equal, and the common path never includes a series name (#208).
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from qats import TimeSeries, TsDB

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MOORING = os.path.join(DATA, "mooring.ts")


def series(name):
    t = np.arange(0.0, 10.0, 0.5)
    return TimeSeries(name, t, np.sin(t))


class TestCaseInsensitiveNames(unittest.TestCase):
    def test_any_case_finds_the_series(self):
        db = TsDB.fromfile(MOORING)
        for name in ("Surge", "surge", "SURGE", "sUrGe"):
            with self.subTest(name):
                self.assertEqual(db.get(name=name).name, "Surge")

    def test_names_differing_only_in_case_are_not_unique(self):
        db = TsDB.fromfile(MOORING)
        db.add(series("surge"))
        self.assertEqual(db.list(names="surge", display=False, relative=True), ["Surge", "surge"])
        with self.assertRaisesRegex(ValueError, "not unique"):
            db.get(name="surge")

    def test_full_key_in_other_case(self):
        db = TsDB.fromfile(MOORING)
        key = os.path.join(MOORING, "Surge")
        self.assertEqual(db.list(names=key.upper(), display=False), [key])

    def test_order_kept(self):
        db = TsDB.fromfile(MOORING)
        self.assertEqual(list(db.getm(names=["SWAY", "surge"], store=False)), ["Sway", "Surge"])

    def test_wildcards(self):
        db = TsDB.fromfile(MOORING)
        self.assertEqual(len(db.list(names="mooring line*", display=False)), 8)
        self.assertEqual(db.list(names="?URGE", display=False, relative=True), ["Surge"])


class TestSeparators(unittest.TestCase):
    def test_slash_and_backslash_equal(self):
        db = TsDB.fromfile(os.path.join(DATA, "results_SIMA36.h5"))
        relative = db.list(display=False, relative=True)[0]  # group\...\name
        self.assertIn("\\", relative)
        expected = db.list(names=relative, display=False)
        self.assertEqual(len(expected), 1)
        self.assertEqual(db.list(names=relative.replace("\\", "/").upper(), display=False), expected)


class TestCommonPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_never_includes_a_series_name(self):
        """One series left from a file, and one added with the same name in another case."""
        db = TsDB.fromfile(MOORING)
        db.clear(names=db.list(display=False)[1:], display=False)
        db.add(series("surge"))
        self.assertEqual(db.common, MOORING)
        self.assertEqual(db.list(display=False, relative=True), ["Surge", "surge"])

    def test_separators_in_square_brackets(self):
        path = os.path.join(self.tmp, "acc.dat")
        with open(path, "w") as f:
            f.write("Time Acc[m/s^2] Acc[m/s]\n0.0 1.0 2.0\n0.5 1.5 2.5\n")
        db = TsDB.fromfile(path)
        self.assertEqual(db.common, path)
        self.assertEqual(db.list(display=False, relative=True), ["Acc[m/s^2]", "Acc[m/s]"])

    def test_unchanged_for_files_and_groups(self):
        """The common path is as before when no name differs only in case."""
        db = TsDB.fromfile([MOORING, os.path.join(DATA, "example.csv")])
        self.assertEqual(db.common, DATA)
        h5 = TsDB.fromfile(os.path.join(DATA, "results_SIMA36.h5"))
        self.assertEqual(h5.common, os.path.commonpath(h5.register_keys))


if __name__ == "__main__":
    unittest.main()
