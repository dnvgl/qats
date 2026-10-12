# -*- coding: utf-8 -*-
"""
TsDB keys are paths (file path + series name), where '/' and '\\' inside square brackets (e.g. units, "Acc[m/s^2]")
are part of a name, not separators (#201).
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from qats import TsDB

SEP = os.sep


class TestPathDirname(unittest.TestCase):
    def test_cases(self):
        root = os.path.abspath(os.sep)  # e.g. "C:\\" or "/"
        cases = {
            # key: expected dirname
            os.path.join("data", "one.dat", "surge"): os.path.join("data", "one.dat"),
            os.path.join("data", "one.dat", "surge[m]"): os.path.join("data", "one.dat"),
            os.path.join("data", "one.dat", "Acc[m/s^2]"): os.path.join("data", "one.dat"),
            os.path.join("data", "one.dat", "Acc[m\\s]"): os.path.join("data", "one.dat"),
            os.path.join("data", "run[1].dat", "surge"): os.path.join("data", "run[1].dat"),
            os.path.join("data", "one.dat", "group[a/b]", "x[m]"): os.path.join("data", "one.dat", "group[a/b]"),
            os.path.join(root, "surge[m]"): root,
            "surge[m]": "",
            "surge": "",
        }
        for key, expected in cases.items():
            with self.subTest(key):
                self.assertEqual(TsDB._path_dirname(key), expected)

    def test_relpath(self):
        base = os.path.join(os.path.abspath(os.sep), "data")
        cases = [
            # key, start, expected
            (os.path.join(base, "one.dat", "surge[m]"), os.path.join(base, "one.dat"), "surge[m]"),
            (os.path.join(base, "one.dat", "Acc[m/s^2]"), os.path.join(base, "one.dat"), "Acc[m/s^2]"),
            (os.path.join(base, "run[1].dat", "surge"), os.path.join(base, "run[1].dat"), "surge"),
            (os.path.join(base, "run[1].dat", "surge[m]"), base, os.path.join("run[1].dat", "surge[m]")),
            (
                os.path.join(base, "one.dat", "grp[a/b]", "x"),
                os.path.join(base, "one.dat"),
                os.path.join("grp[a/b]", "x"),
            ),
            (os.path.join(base, "one.dat", "surge"), "", os.path.join(base, "one.dat", "surge")),
        ]
        for key, start, expected in cases:
            with self.subTest(key=key, start=start):
                self.assertEqual(TsDB._path_relpath(key, start), expected)

    def test_same_as_os_path_dirname_without_brackets(self):
        for key in (os.path.join("a", "b", "c"), os.path.join(os.path.abspath(os.sep), "a"), "a", ""):
            with self.subTest(key):
                self.assertEqual(TsDB._path_dirname(key), os.path.dirname(key))


class TestBracketsInNames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, filename, text):
        path = os.path.join(self.tmp, filename)
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_single_series(self):
        path = self.write("one.dat", "Time surge[m]\n0.0 1.0\n0.5 2.0\n")
        db = TsDB.fromfile(path)
        self.assertEqual(db.common, path)
        self.assertEqual(db.list(display=False, relative=True), ["surge[m]"])
        np.testing.assert_array_equal(db.get(name="surge[m]").x, [1.0, 2.0])

    def test_rename(self):
        path = self.write("two.dat", "Time surge[m] heave[m]\n0.0 1.0 10.0\n0.5 2.0 20.0\n")
        db = TsDB.fromfile(path)
        db.rename("surge[m]", "renamed[m]")
        self.assertEqual(db.register_keys[0], os.path.join(path, "renamed[m]"))
        self.assertEqual(db.list(display=False, relative=True), ["renamed[m]", "heave[m]"])
        ts = db.get(name="renamed[m]")
        self.assertEqual(ts.name, "renamed[m]")
        np.testing.assert_array_equal(ts.x, [1.0, 2.0])

    def test_export_to_file_name_with_brackets(self):
        source = self.write("two.dat", "Time surge heave\n0.0 1.0 10.0\n0.5 2.0 20.0\n")
        target = os.path.join(self.tmp, "out", "run[1].dat")
        TsDB.fromfile(source).export(target, verbose=False)
        self.assertTrue(os.path.isfile(target))
        self.assertEqual(sorted(os.listdir(self.tmp)), ["out", "two.dat"])  # no stray directory such as "out[1].dat"
        # a list skips the wildcard expansion, which reads "[1]" as a pattern (#204)
        self.assertEqual(TsDB.fromfile([target]).list(display=False, relative=True), ["surge", "heave"])


if __name__ == "__main__":
    unittest.main()
