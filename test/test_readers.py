# -*- coding: utf-8 -*-
"""
Module for testing io.
The module utilizes TsDB.fromfile and .get() to read at least one time series from the file, to check that this does not
generate any exceptions.
"""

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pymatreader import read_mat

from qats import TsDB
from qats._validation import QatsValueError
from qats.io.direct_access import read_dis_names, read_tda_names, read_ts_names
from qats.io.openfast import read_out_units, read_outb_data, read_outb_names, read_outb_units
from qats.io.sintef_mat import read_units

# todo: add test class for matlab

ROOT = Path(__file__).resolve().parent


class TestAllReaders(unittest.TestCase):
    def setUp(self):
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env
        self.data_directory = os.path.join(ROOT, "..", "data")

        # make pickle file
        self.make_pickle_file_multiindex()

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
            ("simo_p.ts", 21),  # truncated: the last of the 22 series in the key file is incomplete (#179)
            ("simo_p_out.ts", 22),
            ("simo_r1.ts", 21),  # truncated, as simo_p.ts
            ("simo_r2.ts", 21),  # truncated, as simo_p.ts
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
            # # pickle files
            ("df_multiindex.pkl", 4),
            # openfast files
            ("openfast_MinimalExample.out", 21),
            ("openfast_MinimalExample.outb", 21),
            ("openfast_FASTOutBin.outb", 10),
            ("openfast_5MW_OC4Semi_Linear.outb", 15),
            ("openfast_bd_driver.out", 6),
            ("openfast_ifw_primary.out", 3),
            ("openfast_md_driver.MD.out", 1),
            ("openfast_FASTOut_Hydro.out", 1),
        ]

        # files to delete after testing is completed
        # (because they were generated for this test)
        self.files_to_delete = ["df_multiindex.pkl"]

    def tearDown(self) -> None:
        # cleanup after test completion
        for fn in self.files_to_delete:
            fp = os.path.join(self.data_directory, fn)
            if os.path.exists(fp):
                os.remove(fp)

    def make_pickle_file_multiindex(self):
        """Create pickle file with multiindex columns to be used in reader testing"""

        # Generate range of seconds
        seconds = np.linspace(0, 100, 1000)

        # Generate random data
        data = np.random.randn(1000, 4)

        # Create MultiIndex for columns
        arrays = [
            ["Category A", "Category A", "Category B", "Category B"],
            ["Subcategory 1", "Subcategory 2", "Subcategory 1", "Subcategory 2"],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=("Category", "Subcategory"))

        # Create the DataFrame
        df = pd.DataFrame(data, index=seconds, columns=index)

        # Save the DataFrame to a pickle file
        df.to_pickle(os.path.join(self.data_directory, "df_multiindex.pkl"))

    def test_correct_number_of_timeseries(self):
        """Read key file, check number of keys (data not loaded)"""
        failed = []
        for filename, nts in self.files:
            db = TsDB.fromfile(os.path.join(self.data_directory, filename))
            if not nts == db.n:
                failed.append(f"{filename} ({nts} != {db.n})")
        self.assertTrue(
            len(failed) == 0,
            f"Failed to identify correct number of time series on {len(failed)} file(s):\n   *** "
            + "\n   *** ".join(failed),
        )

    def test_correct_timeseries_size(self):
        """Load time series: check that it loads and that t.size matches x.size"""
        failed = []
        for filename, _ in self.files:
            try:
                db = TsDB.fromfile(os.path.join(self.data_directory, filename))
                ts = db.get(ind=0)  # should not fail
                self.assertTrue(
                    ts.t.size > 1 and ts.t.size == ts.x.size,
                    f"Did not read time series correctly (t.size = {ts.t.size}, x.size = {ts.x.size})",
                )
            except Exception:
                exctype, excvalue, _ = sys.exc_info()
                exctypestr = str(exctype).lstrip("<class '").rstrip("'>")  # e.g. <class 'IndexError'>  =>  IndexError
                failed.append(f"{filename}: {exctypestr}: {excvalue}")
        self.assertTrue(
            len(failed) == 0,
            f"Failed to read time series from {len(failed)} file(s):\n   *** " + "\n   *** ".join(failed),
        )


# one or two files per format; the series positions in these files are what the readers must get right
SELECTIVE_FILES = [
    "mooring.ts",
    "simo_p_out.ts",
    "decay.tda",
    "n_elmtra.bin",
    "wt_witurb.bin",
    "results_SIMA36.h5",
    "example_sima_h5_output_shrinked.h5",
    "example.csv",
    "model_test_data.dat",
    "data.tdms",
    # MATLAB, one per MAT file version: v7.3 (HDF5), v7 (compressed), v6, v4
    "test4210.mat",
    "test4210_v7.mat",
    "test4210_v6.mat",
    "test20320_ntnu.mat",
    "openfast_MinimalExample.out",
    "openfast_MinimalExample.outb",
]


class TestSelectiveReading(unittest.TestCase):
    """Reading some series of a file gives the same arrays as reading all of them (#175)."""

    def setUp(self):
        self.data_directory = os.path.join(ROOT, "..", "data")

    def assert_subsets_equal_full_read(self, path):
        full_db = TsDB.fromfile(path)
        keys = full_db.list()
        n = len(keys)
        full = full_db.getm(ind=list(range(n)), store=False, fullkey=True)
        subsets = [[0], [n // 2], [n - 1], [n - 1, n // 2, 0], list(range(n - 1, max(n - 6, -1), -1))]
        for ind in subsets:
            db = TsDB.fromfile(path)  # nothing read yet
            got = db.getm(ind=ind, store=False, fullkey=True)
            self.assertEqual(list(got), [keys[i] for i in ind], f"order of {ind}")
            for i in ind:
                ts, ref = got[keys[i]], full[keys[i]]
                np.testing.assert_array_equal(ts.t, ref.t, err_msg=f"{keys[i]} (t, subset {ind})")
                np.testing.assert_array_equal(ts.x, ref.x, err_msg=f"{keys[i]} (x, subset {ind})")

    def test_subsets_equal_full_read(self):
        for filename in SELECTIVE_FILES:
            path = os.path.abspath(os.path.join(self.data_directory, filename))
            with self.subTest(filename):
                self.assert_subsets_equal_full_read(path)

    def test_pickle_subsets_equal_full_read(self):
        """No pickle file in data/, so one is written here (#190)."""
        t = np.linspace(0.0, 10.0, 101)
        df = pd.DataFrame({f"s{j}": (j + 1) * np.sin(t + j) for j in range(6)}, index=t)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "series.pkl")
            df.to_pickle(path)
            self.assert_subsets_equal_full_read(path)
            got = TsDB.fromfile(path).get(name="s4", store=False)
            np.testing.assert_array_equal(got.x, df["s4"].to_numpy())

    def test_csv_out_of_file_order(self):
        path = os.path.abspath(os.path.join(self.data_directory, "example.csv"))
        full = TsDB.fromfile(path).getm(names="*", store=False)
        got = TsDB.fromfile(path).getm(ind=[5, 3, 0], store=False)
        self.assertEqual(list(got), ["yaw", "roll", "surge"])
        for name, ts in got.items():
            np.testing.assert_array_equal(ts.x, full[name].x, err_msg=name)

    def test_subset_by_name_equals_full_read(self):
        path = os.path.abspath(os.path.join(self.data_directory, "mooring.ts"))
        full = TsDB.fromfile(path).getm(names="*", store=False)
        got = TsDB.fromfile(path).getm(names=["Mooring line 8", "Surge"], store=False)
        self.assertEqual(list(got), ["Mooring line 8", "Surge"])
        for name, ts in got.items():
            np.testing.assert_array_equal(ts.x, full[name].x, err_msg=name)


class TestIndependentReading(unittest.TestCase):
    """QATS reads the same values as a direct read without QATS (#175)."""

    def setUp(self):
        self.data_directory = os.path.join(ROOT, "..", "data")

    def test_csv_equals_pandas(self):
        path = os.path.join(self.data_directory, "example.csv")
        df = pd.read_csv(path, sep="\t")
        db = TsDB.fromfile(path)
        self.assertEqual([os.path.basename(k) for k in db.list()], list(df.columns[1:]))
        for name in df.columns[1:]:
            ts = db.get(name=name, store=False)
            np.testing.assert_array_equal(ts.t, df.iloc[:, 0].to_numpy(), err_msg=f"{name} (t)")
            np.testing.assert_array_equal(ts.x, df[name].to_numpy(), err_msg=f"{name} (x)")

    def test_mat_versions_equal(self):
        """The same recording stored as MAT v7.3 (HDF5), v7 and v6 gives identical series."""
        reference = TsDB.fromfile(os.path.join(self.data_directory, "test4210.mat")).getm(names="*", store=False)
        self.assertEqual(len(reference), 40)
        for filename in ("test4210_v7.mat", "test4210_v6.mat"):
            with self.subTest(filename):
                other = TsDB.fromfile(os.path.join(self.data_directory, filename)).getm(names="*", store=False)
                self.assertEqual(list(other), list(reference))
                for name, ts in reference.items():
                    np.testing.assert_array_equal(other[name].t, ts.t, err_msg=f"{name} (t)")
                    np.testing.assert_array_equal(other[name].x, ts.x, err_msg=f"{name} (x)")

    def test_mat_units(self):
        """Units are read from chan_units, as stored on file, in every MAT version (#184)."""
        path = os.path.join(self.data_directory, "test4210.mat")
        raw = read_mat(path)  # read directly, without QATS
        expected = dict(zip(raw["chan_names"], raw["chan_units"]))
        self.assertEqual(expected["Time"], "s")
        self.assertEqual(expected["WAVE1"], "m")
        self.assertIn("m/s²", expected.values())

        reference = TsDB.fromfile(path).getm(names="*", store=False)
        self.assertEqual({name: ts.unit for name, ts in reference.items()}, {n: expected[n] for n in reference})
        self.assertEqual(read_units(path), expected)

        for filename, lost in (("test4210_v7.mat", {}), ("test4210_v6.mat", {"²": "�"})):
            with self.subTest(filename):
                other = TsDB.fromfile(os.path.join(self.data_directory, filename)).getm(names="*", store=False)
                for name, ts in reference.items():
                    # the v6 file was written with "²" already replaced by U+FFFD; QATS returns what is on file
                    unit = ts.unit
                    for char, stored in lost.items():
                        unit = unit.replace(char, stored)
                    self.assertEqual(other[name].unit, unit, name)

        ntnu = TsDB.fromfile(os.path.join(self.data_directory, "test20320_ntnu.mat")).getm(names="*", store=False)
        self.assertEqual({ts.unit for ts in ntnu.values()}, {None})  # one variable per series, no units on file

    def test_ts_equals_numpy(self):
        """Direct access .ts: a header record, the time record, then one record per series in key file order."""
        path = os.path.join(self.data_directory, "mooring.ts")
        ndat = int(np.fromfile(path, dtype="<i4", count=1)[0])
        records = np.fromfile(path, dtype="<f4").reshape(-1, ndat)
        with open(os.path.join(self.data_directory, "mooring.key")) as f:
            names = [line.strip() for line in f if line.strip() and not line.startswith(("**", "'"))]
        # the first name in the key file is the time array
        names = [name for name in names if name.upper() != "END"][1:]
        db = TsDB.fromfile(path)
        self.assertEqual([os.path.basename(k) for k in db.list()], names)
        self.assertEqual(records.shape[0], len(names) + 2)  # header and time records
        for i, name in enumerate(names):
            ts = db.get(name=name, store=False)
            np.testing.assert_array_equal(ts.t, records[1], err_msg=f"{name} (t)")
            np.testing.assert_array_equal(ts.x, records[i + 2], err_msg=f"{name} (x)")


class TestNameReaders(unittest.TestCase):
    """The direct-access name readers accept relative paths (#180)."""

    def setUp(self):
        self.data_directory = os.path.abspath(os.path.join(ROOT, "..", "data"))
        self.cwd = os.getcwd()
        os.chdir(self.data_directory)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_relative_path(self):
        for reader, filename in (
            (read_ts_names, "mooring.key"),
            (read_tda_names, "decay.txt"),
            (read_dis_names, "mooring.key"),
        ):
            with self.subTest(reader.__name__):
                names = reader(filename)
                self.assertGreater(len(names), 0)
                self.assertEqual(names, reader(os.path.join(self.data_directory, filename)))


SQRT_EPS32 = float(np.sqrt(np.finfo(np.float32).eps))


def int16_step(x):
    """
    Resolution of a channel stored as int16 in an OpenFAST binary file: its range divided by 65535 (a range of at
    least sqrt(eps) of float32, as OpenFAST's WrBinFAST scales nearly constant channels), plus the float32 rounding of
    the values, which OpenFAST scales in single precision.
    """
    x = np.asarray(x)
    rng = float(np.max(x) - np.min(x))
    return max(rng, SQRT_EPS32) / 65535.0 + float(np.finfo(np.float32).eps) * float(np.max(np.abs(x)))


def write_outb(path, fileid, t, x, names, units, trailing=b""):
    """
    Write an OpenFAST binary output file as OpenFAST's WrBinFAST does (NWTC_IO.f90): `x` has one row per channel,
    `names` and `units` include time.
    """
    nchannels, nsteps = x.shape
    lenname = max(len(s) for s in names + units) if fileid == 4 else 10
    header = struct.pack("<h", fileid)
    if fileid == 4:
        header += struct.pack("<h", lenname)
    header += struct.pack("<ii", nchannels, nsteps)
    if fileid == 1:
        # time packed in [-65536, 65535]
        timescale = 131071.0 / (t[-1] - t[0])
        timeoffset = -65536.0 - timescale * t[0]
        header += struct.pack("<dd", timescale, timeoffset)
    else:
        header += struct.pack("<dd", t[0], t[1] - t[0])
    # int16 scaling of each channel to its range (not stored in uncompressed files)
    xmin, xmax = x.min(axis=1).astype(np.float32), x.max(axis=1).astype(np.float32)
    rng = xmax - xmin
    scale = np.float32(65535.0) / np.where(np.abs(rng) < SQRT_EPS32, np.float32(SQRT_EPS32), rng)
    scale = scale.astype(np.float32)
    offset = (np.float32(-32768.0) - scale * xmin).astype(np.float32)
    if fileid != 3:
        header += scale.tobytes() + offset.tobytes()
    description = b"Written by the QATS tests"
    header += struct.pack("<i", len(description)) + description
    header += "".join(s.ljust(lenname)[:lenname] for s in names).encode("latin-1")
    header += "".join(s.ljust(lenname)[:lenname] for s in units).encode("latin-1")
    if fileid == 1:
        packed = np.rint(np.clip(timescale * t + timeoffset, -65536.0, 65535.0)).astype("<i4")
        header += packed.tobytes()
    if fileid == 3:
        data = x.T.astype("<f8")
    else:
        scaled = (scale[:, np.newaxis] * x.astype(np.float32) + offset[:, np.newaxis]).astype(np.float32)
        data = np.rint(np.clip(scaled, -32768.0, 32767.0)).T.astype("<i2")
    with open(path, "wb") as f:
        f.write(header + data.tobytes() + trailing)


class TestOpenFast(unittest.TestCase):
    """OpenFAST .out and .outb files (#169)."""

    def setUp(self):
        self.data_directory = os.path.join(ROOT, "..", "data")
        t = np.arange(0.0, 20.0, 0.05)
        self.t = t
        self.x = np.vstack([10.0 * np.sin(t), 1e5 + 3.0 * np.cos(0.3 * t), np.full(t.size, 2.5), -(t**2)])
        self.names = ["Time", "Wave1Elev", "FairTen1", "Const", "LongChannelName"]
        self.units = ["(s)", "(m)", "(kN)", "(-)", "(kN-m)"]

    def path(self, filename):
        return os.path.join(self.data_directory, filename)

    def test_published_value(self):
        """The last GenPwr value in a file with constant time step (file format identifier 2), as in the OpenFAST
        Python toolbox's own test of this file."""
        db = TsDB.fromfile(self.path("openfast_FASTOutBin.outb"))
        ts = db.get(name="GenPwr")
        self.assertEqual(ts.unit, "kW")
        self.assertEqual(ts.x[-1], 40.57663190807828)
        self.assertEqual(ts.t.size, 201)
        self.assertAlmostEqual(ts.t[-1], 1.0, places=12)

    def test_text_equals_binary(self):
        """The text and binary file from the same run hold the same channels, to the int16 resolution."""
        text = TsDB.fromfile(self.path("openfast_MinimalExample.out")).getm(names="*", store=False)
        binary = TsDB.fromfile(self.path("openfast_MinimalExample.outb")).getm(names="*", store=False)
        self.assertEqual(list(text), list(binary))
        self.assertEqual(len(text), 21)
        for name, ts in text.items():
            other = binary[name]
            self.assertEqual(other.unit, ts.unit, name)
            np.testing.assert_allclose(other.t, ts.t, rtol=0, atol=1e-12, err_msg=name)
            np.testing.assert_allclose(other.x, ts.x, rtol=0, atol=int16_step(ts.x), err_msg=name)

    def test_text_equals_numpy(self):
        """The text file read directly: a header of 6 lines, names, units in parentheses, tab-separated values."""
        path = self.path("openfast_MinimalExample.out")
        with open(path) as f:
            lines = f.readlines()
        names = lines[6].rstrip("\n").split("\t")
        units = lines[7].rstrip("\n").split("\t")
        values = np.loadtxt(path, skiprows=8, unpack=True)
        expected = {name: unit[1:-1] for name, unit in zip(names, units)}
        self.assertEqual(expected["Time"], "s")
        self.assertEqual(expected["RootMyc1"], "kN-m")
        self.assertEqual(read_out_units(path), expected)
        db = TsDB.fromfile(path)
        self.assertEqual([os.path.basename(k) for k in db.list()], names[1:])
        for i, name in enumerate(names[1:]):
            ts = db.get(name=name, store=False)
            self.assertEqual(ts.unit, expected[name])
            np.testing.assert_array_equal(ts.t, values[0], err_msg=f"{name} (t)")
            np.testing.assert_array_equal(ts.x, values[i + 1], err_msg=f"{name} (x)")

    def test_text_variants(self):
        """Header lines, padding and units vary between OpenFAST modules and versions."""
        for filename, first, unit, n, t0 in (
            ("openfast_bd_driver.out", "TipTDxr", "m", 10, 0.0),  # names padded with spaces before the tabs
            ("openfast_ifw_primary.out", "Wind1VelX", "m/s", 10, 30.0),  # '##' header, space-separated
            ("openfast_md_driver.MD.out", "FAIRTEN1", "N", 599, 0.1),  # 'These predictions were generated by'
            ("openfast_FASTOut_Hydro.out", "Wave1Elev", "m", 5, 0.0),  # 'This output file was closed' at the end
        ):
            with self.subTest(filename):
                ts = TsDB.fromfile(self.path(filename)).get(ind=0, store=False)
                self.assertEqual(ts.name, first)
                self.assertEqual(ts.unit, unit)
                self.assertEqual(ts.t.size, n)
                self.assertEqual(ts.t[0], t0)
        self.assertEqual(read_out_units(self.path("openfast_FASTOut_Hydro.out"))["Time"], "sec")
        np.testing.assert_array_equal(
            TsDB.fromfile(self.path("openfast_FASTOut_Hydro.out")).get(ind=0).x, [0.0, 10.0, 0.0, -10.0, 0.0]
        )

    def test_uncompressed(self):
        """Uncompressed files (file format identifier 3) hold float64 values, also with a single time step."""
        for filename, n, nsteps in (
            ("openfast_5MW_OC4Semi_Linear.outb", 15, 101),
            ("openfast_5MW_OC3Mnpl_Linear.outb", 44, 1),
        ):
            with self.subTest(filename):
                data = read_outb_data(self.path(filename))
                self.assertEqual(data.shape, (n + 1, nsteps))
                self.assertEqual(data[0, 0], 0.0)

    def test_written_files(self):
        """Files written as OpenFAST does, in every binary format, give back the series to their resolution."""
        with tempfile.TemporaryDirectory() as tmp:
            for fileid in (1, 2, 3, 4):
                with self.subTest(fileid=fileid):
                    path = os.path.join(tmp, f"fileid{fileid}.outb")
                    write_outb(path, fileid, self.t, self.x, self.names, self.units)
                    lenname = 15 if fileid == 4 else 10  # names cut to 10 characters before file format 4
                    self.assertEqual(read_outb_names(path), [s[:lenname] for s in self.names[1:]])
                    self.assertEqual(list(read_outb_units(path).values()), ["s", "m", "kN", "-", "kN-m"])
                    data = read_outb_data(path)
                    # time: stored to half its resolution (file format 1), or from first time and step
                    atol = 0.5 * (self.t[-1] - self.t[0]) / 131071.0 if fileid == 1 else 1e-12
                    np.testing.assert_allclose(data[0], self.t, rtol=0, atol=atol)
                    for x, y in zip(self.x, data[1:]):
                        atol = 0.0 if fileid == 3 else int16_step(x)
                        np.testing.assert_allclose(y, x, rtol=0, atol=atol)
                    # a subset, out of file order and with time last
                    np.testing.assert_array_equal(read_outb_data(path, ind=[3, 1, 0]), data[[3, 1, 0]])

    def test_trailing_bytes_ignored(self):
        """Data on file after the last time step is ignored, as in files found in practice."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trailing.outb")
            write_outb(path, 4, self.t, self.x, self.names, self.units, trailing=b"\x01\x02\x03")
            self.assertEqual(read_outb_data(path).shape, (5, self.t.size))

    def test_cut_off_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cut.outb")
            write_outb(path, 4, self.t, self.x, self.names, self.units)
            with open(path, "rb") as f:
                content = f.read()
            with open(path, "wb") as f:
                f.write(content[:-10])
            with self.assertRaisesRegex(QatsValueError, "requires"):
                read_outb_data(path)

    def test_fortran_exponent(self):
        """Fortran writes exponents with three digits without the 'E', e.g. '1.234-100'."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "small.out")
            with open(path, "w") as f:
                f.write("\nPredictions were generated on 01-Jan-2026 using OpenFAST\n\n\n\n\n")
                f.write("Time\tX\n(s)\t(m)\n0.0\t1.5E+00\n0.1\t1.234-100\n0.2\t2.0+100\n")
            ts = TsDB.fromfile(path).get(name="X")
            np.testing.assert_array_equal(ts.x, [1.5, 1.234e-100, 2.0e100])


if __name__ == "__main__":
    unittest.main()
