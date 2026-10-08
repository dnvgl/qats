# -*- coding: utf-8 -*-
"""
Module for testing io.
The module utilizes TsDB.fromfile and .get() to read at least one time series from the file, to check that this does not
generate any exceptions.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from qats import TsDB
from qats.io.direct_access import read_dis_names, read_tda_names, read_ts_names

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
]


class TestSelectiveReading(unittest.TestCase):
    """Reading some series of a file gives the same arrays as reading all of them (#175)."""

    def setUp(self):
        self.data_directory = os.path.join(ROOT, "..", "data")

    def test_subsets_equal_full_read(self):
        for filename in SELECTIVE_FILES:
            path = os.path.abspath(os.path.join(self.data_directory, filename))
            with self.subTest(filename):
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


if __name__ == "__main__":
    unittest.main()
