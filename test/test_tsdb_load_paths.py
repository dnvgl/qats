# -*- coding: utf-8 -*-
"""
TsDB.load() with file names that contain wildcard characters, and with wildcard patterns (#204).
"""

import os
import shutil
import tempfile
import unittest

from qats import TsDB


class TestLoadPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, filename, series):
        path = os.path.join(self.tmp, filename)
        with open(path, "w") as f:
            f.write(f"Time {series}\n0.0 1.0\n0.5 2.0\n")
        return path

    def names(self, db):
        return sorted(os.path.basename(k) for k in db.list(display=False))

    def test_file_name_with_brackets(self):
        path = self.write("run[1].dat", "surge")
        for db in (TsDB.fromfile(path), TsDB.fromfile([path])):
            self.assertEqual(self.names(db), ["surge"])
        db = TsDB()
        db.load(path)
        self.assertEqual(db.register_parent[db.list(display=False)[0]], os.path.abspath(path))

    def test_existing_file_is_not_expanded(self):
        """A str naming an existing file loads that file, not the files its name matches as a pattern."""
        bracketed = self.write("run[1].dat", "bracketed")
        self.write("run1.dat", "plain")
        self.assertEqual(self.names(TsDB.fromfile(bracketed)), ["bracketed"])

    def test_wildcards_still_work(self):
        self.write("run1.dat", "a")
        self.write("run2.dat", "b")
        self.write("other.dat", "c")
        self.assertEqual(self.names(TsDB.fromfile(os.path.join(self.tmp, "run*.dat"))), ["a", "b"])
        self.assertEqual(self.names(TsDB.fromfile(os.path.join(self.tmp, "run?.dat"))), ["a", "b"])
        # a character class, when no file has that literal name
        self.assertEqual(self.names(TsDB.fromfile(os.path.join(self.tmp, "run[12].dat"))), ["a", "b"])

    def test_no_match(self):
        with self.assertRaises(FileExistsError):
            TsDB.fromfile(os.path.join(self.tmp, "missing*.dat"))


if __name__ == "__main__":
    unittest.main()
