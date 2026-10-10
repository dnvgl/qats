# -*- coding: utf-8 -*-
"""
The rules for .dat and .csv files in the documentation page "Writing files QATS can read"
(docs/source/file_formats.rst, #90). Each test checks one documented rule, so the page can't drift from the readers.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qats import TsDB

FORMATS = Path(__file__).resolve().parent / ".." / "docs" / "source" / "formats"


class FileFormatTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, filename, text, encoding="utf-8", newline="\n"):
        path = os.path.join(self.tmp, filename)
        with open(path, "w", encoding=encoding, newline=newline) as f:
            f.write(text)
        return path

    def read(self, path):
        """Series names and (t, x) arrays from a file."""
        series = TsDB.fromfile(path).getm(names="*", store=False)
        return {name: (ts.t, ts.x) for name, ts in series.items()}

    def assert_series(self, got, expected):
        self.assertEqual(list(got), list(expected))
        for name, (t, x) in expected.items():
            np.testing.assert_array_equal(got[name][0], t, err_msg=f"{name} (t)")
            np.testing.assert_array_equal(got[name][1], x, err_msg=f"{name} (x)")


class TestDocumentedExamples(FileFormatTestCase):
    def test_example_dat(self):
        got = self.read(str(FORMATS / "example.dat"))
        self.assertEqual(list(got), ["Heave[m]", "Pitch[deg]"])
        np.testing.assert_array_equal(got["Heave[m]"][0], [0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        np.testing.assert_array_equal(got["Pitch[deg]"][1], [1.5, 1.426, 1.213, 0.882, 0.464, 0.0])

    def test_example_csv(self):
        got = self.read(str(FORMATS / "example.csv"))
        self.assertEqual(list(got), ["Wave elevation", "Surge"])
        np.testing.assert_array_equal(got["Surge"][1], [1.2, 1.21, 1.23, 1.26, 1.3, 1.35])


class TestDat(FileFormatTestCase):
    expected = {"surge": ([0.0, 0.5], [1.0, 2.0]), "heave": ([0.0, 0.5], [10.0, 20.0])}

    def test_comments_anywhere(self):
        path = self.write("a.dat", "# first\n# second\nTime surge heave\n0.0 1.0 10.0\n# middle\n0.5 2.0 20.0\n")
        self.assert_series(self.read(path), self.expected)

    def test_spaces_and_tabs(self):
        path = self.write("a.dat", "Time\t surge   heave\n0.0\t1.0    10.0\n0.5  \t 2.0 20.0\n")
        self.assert_series(self.read(path), self.expected)

    def test_time_name(self):
        for name in ("Time", "time", "Time[s]", "time_s"):
            with self.subTest(name):
                path = self.write("a.dat", f"{name} surge heave\n0.0 1.0 10.0\n0.5 2.0 20.0\n")
                self.assert_series(self.read(path), self.expected)

    def test_no_time_name(self):
        path = self.write("a.dat", "t surge\n0.0 1.0\n0.5 2.0\n")
        with self.assertRaisesRegex(KeyError, "does not contain a time vector"):
            TsDB.fromfile(path)

    def test_blank_line_before_header(self):
        path = self.write("a.dat", "\nTime surge\n0.0 1.0\n0.5 2.0\n")
        with self.assertRaisesRegex(KeyError, "does not contain a time vector"):
            TsDB.fromfile(path)

    def test_two_time_names(self):
        path = self.write("a.dat", "Time surge Timestamp\n0.0 1.0 5\n0.5 2.0 6\n")
        with self.assertRaisesRegex(KeyError, "duplicate time vectors"):
            TsDB.fromfile(path)

    def test_name_with_space(self):
        path = self.write("a.dat", "Time wave height\n0.0 1.0\n0.5 2.0\n")
        with self.assertRaisesRegex(ValueError, "invalid column index"):
            TsDB.fromfile(path).getm(names="*", store=False)

    def test_unit_kept_in_name(self):
        # two series: with one, the relative name is wrong (#201)
        path = self.write("a.dat", "Time surge[m] heave[m]\n0.0 1.0 10.0\n0.5 2.0 20.0\n")
        db = TsDB.fromfile(path)
        self.assertEqual(db.list(display=False, relative=True), ["surge[m]", "heave[m]"])
        self.assertIsNone(db.get(name="surge[m]").unit)

    def test_numbers(self):
        path = self.write("a.dat", "Time surge\n0.0 1e-3\n5.0E-1 nan\n1.0 -2.5e+2\n")
        got = self.read(path)
        np.testing.assert_array_equal(got["surge"][0], [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(got["surge"][1], [0.001, np.nan, -250.0])

    def test_line_endings(self):
        path = self.write("a.dat", "Time surge heave\n0.0 1.0 10.0\n0.5 2.0 20.0\n", newline="\r\n")
        self.assert_series(self.read(path), self.expected)

    @unittest.expectedFailure  # #199: the first column is read as time without an error; remove when fixed
    def test_time_not_first(self):
        path = self.write("a.dat", "surge Time heave\n1.0 0.0 10.0\n2.0 0.5 20.0\n")
        with self.assertRaises(Exception):
            TsDB.fromfile(path).getm(names="*", store=False)


class TestCsv(FileFormatTestCase):
    expected = {"surge": ([0.0, 0.5], [1.0, 2.0]), "heave": ([0.0, 0.5], [10.0, 20.0])}

    def test_separators(self):
        for label, sep in (("comma", ","), ("semicolon", ";"), ("tab", "\t"), ("space", " ")):
            with self.subTest(label):
                text = sep.join(["Time", "surge", "heave"]) + "\n"
                text += sep.join(["0.0", "1.0", "10.0"]) + "\n" + sep.join(["0.5", "2.0", "20.0"]) + "\n"
                self.assert_series(self.read(self.write("a.csv", text)), self.expected)

    def test_time_any_name(self):
        path = self.write("a.csv", "t,surge,heave\n0.0,1.0,10.0\n0.5,2.0,20.0\n")
        self.assert_series(self.read(path), self.expected)

    def test_names_with_spaces_and_quotes(self):
        path = self.write("a.csv", 'Time,wave height,"surge, x"\n0.0,1.0,10.0\n0.5,2.0,20.0\n')
        self.assertEqual(list(self.read(path)), ["wave height", "surge, x"])

    def test_empty_cell_is_nan(self):
        path = self.write("a.csv", "Time,surge,heave\n0.0,1.0,\n0.5,2.0,20.0\n")
        np.testing.assert_array_equal(self.read(path)["heave"][1], [np.nan, 20.0])

    def test_utf8_with_and_without_bom(self):
        for encoding in ("utf-8", "utf-8-sig"):
            with self.subTest(encoding):
                path = self.write("a.csv", "Tid,bølge\n0.0,1.0\n0.5,2.0\n", encoding=encoding)
                self.assertEqual(list(self.read(path)), ["bølge"])

    def test_line_endings(self):
        path = self.write("a.csv", "Time,surge,heave\n0.0,1.0,10.0\n0.5,2.0,20.0\n", newline="\r\n")
        self.assert_series(self.read(path), self.expected)

    # limitations documented on the page; when #200 changes them, update the page too

    def test_windows_encoding_fails(self):
        path = self.write("a.csv", "Tid,bølge\n0.0,1.0\n0.5,2.0\n", encoding="cp1252")
        with self.assertRaises(UnicodeDecodeError):
            TsDB.fromfile(path)

    def test_decimal_comma_fails(self):
        path = self.write("a.csv", "Time;surge\n0,0;1,5\n0,5;2,5\n")
        with self.assertRaisesRegex(TypeError, "must be integers or floats"):
            TsDB.fromfile(path).getm(names="*", store=False)

    def test_comment_line_fails(self):
        path = self.write("a.csv", "# comment\nTime,surge\n0.0,1.0\n0.5,2.0\n")
        with self.assertRaisesRegex(TypeError, "time must be given as array of floats"):
            TsDB.fromfile(path).getm(names="*", store=False)


class TestBothFormats(FileFormatTestCase):
    def test_slash_in_name(self):
        """Outside square brackets, / separates a group and a series name."""
        for filename, text in (
            ("a.dat", "Time Acc[m/s^2] Acc/x\n0.0 1.0 2.0\n0.5 1.5 2.5\n"),
            ("a.csv", "Time,Acc[m/s^2],Acc/x\n0.0,1.0,2.0\n0.5,1.5,2.5\n"),
        ):
            with self.subTest(filename):
                db = TsDB.fromfile(self.write(filename, text))
                self.assertEqual(db.list(display=False, relative=True), ["Acc[m/s^2]", os.path.join("Acc", "x")])
                np.testing.assert_array_equal(db.get(name="Acc[m/s^2]").x, [1.0, 1.5])
                np.testing.assert_array_equal(db.get(name="Acc/x").x, [2.0, 2.5])

    def test_python_export_round_trip(self):
        """The snippet in "From Python": export to .ts or .dat and read back."""
        from qats import TimeSeries

        t = np.arange(0.0, 100.0, 0.05)
        db = TsDB()
        db.add(TimeSeries("Heave", t, 0.1 * np.exp(-0.05 * t) * np.cos(t), unit="m"))
        db.add(TimeSeries("Pitch", t, 1.5 * np.exp(-0.05 * t) * np.cos(t), unit="deg"))
        for ext in (".ts", ".dat"):
            with self.subTest(ext):
                path = os.path.join(self.tmp, "decay" + ext)
                db.export(path, verbose=False)
                back = TsDB.fromfile(path)
                self.assertEqual(back.list(display=False, relative=True), ["Heave", "Pitch"])
                np.testing.assert_allclose(back.get(name="Pitch").x, db.get(name="Pitch").x, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
