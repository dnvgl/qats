# -*- coding: utf-8 -*-
"""
Module for testing TsDB class
"""

import contextlib
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from qats import TimeSeries, TsDB
from qats.io import registry

from .test_registry import RegistryTestCase, make_reader

# todo: add tests for listing subset(s) based on specifying parameter `names` (with and wo param. `keys`)
# todo: add test for getm() with fullkey=False (similar to test_get_many_correct_key, but with shorter key)


class TestTsDB(unittest.TestCase):
    def setUp(self):
        self.db = TsDB()
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env for conda build
        self.data_directory = os.path.join(os.path.dirname(__file__), "..", "data")

    def test_exception_load_numeric(self):
        try:
            self.db.load(223334)  # numeric values should throw an exception
        except TypeError:
            pass
        else:
            self.fail("Did not throw exception on numeric file name")

    def test_exception_load_dict(self):
        try:
            self.db.load({})  # dictionary should throw an exception
        except TypeError:
            pass
        else:
            self.fail("Did not throw exception on dictionary of file names.")

    def test_exception_load_directory(self):
        try:
            self.db.load(self.data_directory)
        except FileExistsError:
            pass
        else:
            self.fail("Did not throw exception when trying to load a directory.")

    def test_exception_load_nonexistingfile(self):
        try:
            self.db.load(os.path.join(self.data_directory, "donotexist.ts"))
        except FileExistsError:
            pass
        else:
            self.fail("Did not throw exception when trying to load a non-existing file.")

    def test_exception_load_unsupportedfile(self):
        try:
            self.db.load(os.path.join(self.data_directory, "unsupportedfile.out"))
        except NotImplementedError:
            pass
        else:
            self.fail("Did not throw exception when trying to load a file type which is not yet supported.")

    def test_list_all(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        k = self.db.list(display=False)
        self.assertEqual(14, len(k), "Deviating number of listed keys = %d" % len(k))

    def test_list_subset(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        k = self.db.list(names="Mooring line*", display=False)
        self.assertEqual(8, len(k), "Deviating number of listed keys = %d" % len(k))

    def test_list_subset_misc_criteria(self):
        for tsfile in ("mooring.ts", "simo_p_out.ts"):
            self.db.load(os.path.join(self.data_directory, tsfile))
        # test 1
        k = self.db.list(names="Tension*", display=False)
        self.assertEqual(10, len(k), "Deviating number of listed keys = %d" % len(k))
        # test 2
        k = self.db.list(names="simo_p_out.ts*line*", display=False)
        self.assertEqual(2, len(k), "Deviating number of listed keys = %d" % len(k))

    def test_truncated_ts_lists_only_complete_series(self):
        """A .ts file with an incomplete last record (#179): warn and list only the series that can be read."""
        path = os.path.join(self.data_directory, "simo_p.ts")
        with self.assertWarnsRegex(UserWarning, "holds 21 complete series.*lists 22.*Skipping hc_line_2"):
            self.db.load(path)
        names = [os.path.basename(k) for k in self.db.list(display=False)]
        self.assertEqual(21, len(names))
        self.assertNotIn("hc_line_2", names)
        self.assertEqual(21, len(self.db.getm(names="*", store=False)))

    def test_list_subset_keep_specified_order(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        names_reversed = list(reversed([os.path.basename(k) for k in self.db.register_keys]))
        namelist = [os.path.basename(_) for _ in self.db.list(names=names_reversed)]
        self.assertEqual(names_reversed, namelist, "Failed to keep specified order")

    def test_list_subset_special_characters(self):
        self.db.load(os.path.join(self.data_directory, "model_test_data.dat"))
        # should return exactly one key
        self.assertEqual(1, len(self.db.list(names="RW1[m]")), "TsDB.list() returned wrong number of keys")

    def test_list_subset_special_characters_2(self):
        self.db.load(os.path.join(self.data_directory, "model_test_data.dat"))
        # should return exactly one key
        self.assertEqual(1, len(self.db.list(names="Acc-X[m/s^2]")), "TsDB.list() returned wrong number of keys")

    def test_list_prepended_wildcard_1_3(self):
        """
        Test that wildcard is prepended in a reasonable manner. Test cases:
            1. Specifying 'XG' should not return 'vel_XG'
            2. Specifying '*XG' should return both 'XG' and 'vel_XG'
            3. Specifying full key should be possible
            4. If multiple files are loaded, specifying 'XG' should return all occurrences (across files)

        The first three are tested here, while the fourth is tested in `test_list_prepended_wildcard_4()`
        """
        path = os.path.join(self.data_directory, "simo_r1.ts")
        db = self.db
        db.load(path)
        k1 = db.list(names="XG")  # should return 1 key
        k2 = db.list(names="*XG")  # should return 2 keys
        k3 = db.list(names=os.path.abspath(os.path.join(path, "XG")))  # should return 1 key
        # test of the cases described in docstring
        self.assertEqual(len(k1), 1, "TsDB.list() failed to return correct number of keys for names='XG'")
        self.assertEqual(len(k2), 2, "TsDB.list() failed to return correct number of keys for names='*XG'")
        self.assertEqual(len(k3), 1, "TsDB.list() failed to return correct number of keys when specifying full path")

    def test_list_prepended_wildcard_4(self):
        """
        See description of `test_list_prepended_wildcard_1_3()`
        """
        db = self.db
        db.load(os.path.join(self.data_directory, "simo_r1.ts"))
        db.load(os.path.join(self.data_directory, "simo_r2.ts"))
        k1 = db.list(names="XG")  # should return 2 keys
        k2 = db.list(names="*XG")  # should return 4 keys
        # test of the cases described in docstring
        self.assertEqual(len(k1), 2, "TsDB.list() failed to return correct number of keys for names='XG'")
        self.assertEqual(len(k2), 4, "TsDB.list() failed to return correct number of keys for names='*XG'")

    def test_clear_all(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        self.db.clear(display=False)
        k = self.db.list(display=False)
        self.assertEqual([], k, "Did not clear all registered keys.")

    def test_clear_subset(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        self.db.clear(names="*Mooring line*", display=False)
        k = self.db.list(display=False)
        self.assertEqual(6, len(k), "Did not clear subset of registered keys correctly. %d keys remaining" % len(k))

    def test_getda_correct_key(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        rk = self.db.list(names="Heave", display=False)
        container = self.db.getda(names="Heave", fullkey=True)
        self.assertEqual(rk, list(container.keys()), "db list method and get_many method returns different keys.")

    def test_getda_correct_number_of_arrays(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        rk = self.db.list(names="Heave", display=False)  # should be only 1 key returned in this case
        container = self.db.getda(names="Heave", fullkey=True)
        self.assertEqual(2, len(container[rk[0]]), "Got more than 2 arrays (time and data) in return from get_many().")

    def test_gets_none(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        container = self.db.getda(names=[])
        n = len(container)
        self.assertEqual(0, n, "Should have received empty container (OrderedDict) from getda()")

    def test_getl_correct_key(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        rk = self.db.list(names="Heave", display=False, relative=True)
        tslist = self.db.getl(names="Heave")
        self.assertEqual(rk, [ts.name for ts in tslist], "db list method and getl returns different keys.")

    def test_getm_correct_key(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        rk = self.db.list(names="Heave", display=False)
        container = self.db.getm(names="Heave", fullkey=True)
        self.assertEqual(rk, list(container.keys()), "db list method and getm method returns different keys.")

    def test_getm_correct_key_by_ind(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        rk = self.db.list(names="Heave", display=False)
        container = self.db.getm(ind=2, fullkey=True)
        self.assertEqual(rk, list(container.keys()), "db list method and getm method returns different keys.")

    def test_getd_equals_getm(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        container1 = self.db.getm(names="*", fullkey=True)
        container2 = self.db.getd(names="*", fullkey=True)
        for name, ts in container1.items():
            self.assertTrue(
                name in container2 and container2[name] is container1[name],
                "container returned by getd is not identical to container returned by getm",
            )

    def test_geta(self):
        tsfile = os.path.join(self.data_directory, "simo_p.ts")
        self.db.load(tsfile)
        tsname = "Tension_2_qs"
        keys = self.db.list(names=tsname, display=False)
        _, data1 = self.db.geta(name=keys[0])

        # test 1: geta() when ts is already loaded
        _, data2 = self.db.geta(name=tsname)
        self.assertTrue(
            np.array_equal(data1, data2), "Did not get correct data time series using get() (ts pre-loaded)"
        )
        # test 2: geta() when ts is not already loaded
        db2 = TsDB()
        db2.load(tsfile)
        _, data3 = db2.geta(name=tsname)
        self.assertTrue(
            np.array_equal(data1, data3), "Did not get correct data time series using get() (ts not pre-loaded)"
        )

    def test_get_by_name(self):
        tsfile = os.path.join(self.data_directory, "simo_p.ts")
        self.db.load(tsfile)
        tsname = "Tension_2_qs"
        keys = self.db.list(names=tsname, display=False)
        key = keys[0]
        ts1 = self.db.getm(names=key, fullkey=True)[key]
        # test 1: get_ts() when ts is already loaded
        ts2 = self.db.get(name=tsname)
        self.assertIs(ts1, ts2, "Did not get correct TimeSeries  using get_ts() (ts pre-loaded)")
        # test 2: get_ts() when ts is not already loaded
        db2 = TsDB.fromfile(tsfile)
        ts3 = db2.get(name=tsname)
        self.assertTrue(
            np.array_equal(ts1.x, ts3.x), "Did not get correct TimeSeries using get_ts() (ts not pre-loaded)"
        )

    def test_get_by_index(self):
        tsfile = os.path.join(self.data_directory, "simo_p.ts")
        self.db.load(tsfile)
        tsname = "Tension_2_qs"
        key = self.db.list(names=tsname, display=False)[0]
        ts1 = self.db.get(name=tsname)
        ind = self.db.register_keys.index(key)
        # test 1: get_ts() using index when ts is already loaded
        ts2 = self.db.get(ind=ind)
        self.assertIs(ts1, ts2, "Did not get correct TimeSeries using get_ts() and specifying index (ts pre-loaded)")

        # test 2: get_ts() using index when ts is not already loaded
        db2 = TsDB.fromfile(tsfile)
        ts3 = db2.get(ind=ind)
        self.assertTrue(
            np.array_equal(ts1.x, ts3.x),
            "Did not get correct TimeSeries using get_ts() and specifying index (ts not pre-loaded)",
        )

    def test_get_by_index_0(self):
        """Should not fail when index 0 is specified"""
        tsfile = os.path.join(self.data_directory, "simo_p.ts")
        self.db.load(tsfile)
        _ = self.db.get(ind=0)
        # should not fail

    def test_get_exceptions(self):
        self.db.load(os.path.join(self.data_directory, "simo_p.ts"))
        # test 1: no match
        try:
            _ = self.db.geta(name="nonexisting_key")
        except LookupError:
            pass
        else:
            self.fail("Did not raise LookupError when no match was found")
        # test 2: more than one match
        try:
            _ = self.db.geta(name="Tension*")
        except ValueError:
            pass
        else:
            self.fail("Did not raise ValueError when multiple matches were found")

    def test_get_correct_number_of_timesteps(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        rk = self.db.list(names="Heave", display=False)  # should be only 1 key returned in this case
        container = self.db.getda(names="Heave", fullkey=True)
        self.assertEqual(65536, len(container[rk[0]][0]), "Deviating number of time steps.")

    def test_add_raises_keyerror_on_nonunique_key(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        container = self.db.getm(names="Surge", fullkey=True)
        for k, v in container.items():
            try:
                self.db.add(v)
            except KeyError:
                pass
            else:
                self.fail("Did not raise KeyError when trying to add time series with non-unique name to db.")

    def test_add_does_not_raise_error(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        ts = TimeSeries("quiteuniquekeyiguess", np.arange(0.0, 100.0, 0.01), np.sin(np.arange(0.0, 100.0, 0.01)))
        self.db.add(ts)
        # should not raise errors

    def test_rename(self):
        tsfile = os.path.abspath(os.path.join(self.data_directory, "simo_p.ts"))
        self.db.load(tsfile)
        oldname = "Tension_2_qs"
        newname = "mooringline"
        #
        oldkey = os.path.join(tsfile, oldname)
        newkey = os.path.join(tsfile, newname)
        # get data before rename()
        _, data1 = self.db.geta(name=oldname)
        parent1 = self.db.register_parent[oldkey]
        index1 = self.db.register_indices[oldkey]
        # rename
        self.db.rename(oldname, newname)
        # get data after rename()
        _, data2 = self.db.geta(name=newname)
        parent2 = self.db.register_parent[newkey]
        index2 = self.db.register_indices[newkey]
        # checks
        self.assertTrue(newkey in self.db.register_keys, "register_keys not updated by rename()")
        self.assertEqual(parent1, parent2, "register_parent not correctly updated")
        self.assertEqual(index1, index2, "register_indices not correctly updated")
        self.assertTrue(np.array_equal(data1, data2), "register not correctly updated")

    def test_rename_execption(self):
        tsfile = os.path.join(self.data_directory, "simo_p.ts")
        self.db.load(tsfile)
        oldname = "Tension_2_qs"
        newname = "Tension_3_qs"
        try:
            self.db.rename(oldname, newname)
        except ValueError:
            pass
        else:
            self.fail("Did not throw ValueError when attempting renaming to non-unique name.")

    def test_maxima_minima(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        container = self.db.getm(names="Surge")
        for k, ts in container.items():
            _ = ts.maxima()
            _, _ = ts.maxima(rettime=True)
            _ = ts.minima()
            _, _ = ts.minima(rettime=True)
            # currently only testing that no error are thrown

    def test_types_in_container_from_get_many(self):
        """
        Test correct types
        """
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        container = self.db.getda(names="Surge")
        for key, ts in container.items():
            self.assertIsInstance(key, str, "Key should be type string.")
            self.assertIsInstance(ts, tuple, "Time series container should be type tuple.")
            self.assertIsInstance(ts[0], np.ndarray, "First item of time series container should be type numpy array.")
            self.assertIsInstance(ts[1], np.ndarray, "Second item of time series container should be type numpy array.")

    def test_types_in_container_from_get_many_ts(self):
        """
        Test correct types
        """
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        container = self.db.getm(names="Surge")
        for key, ts in container.items():
            self.assertIsInstance(key, str, "Key should be type string.")
            self.assertIsInstance(ts, TimeSeries, "Time series container should be type TimeSeries.")
            self.assertIsInstance(ts.t, np.ndarray, "Attribute t of time series should be type numpy array.")
            self.assertIsInstance(ts.x, np.ndarray, "Attribute x of time series should be type numpy array.")

    def test_create_dataframe(self):
        # create dataframe, check that time series data is the same
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        name = "Sway"
        ts = self.db.get(name=name)
        # create dataframe
        df = self.db.to_dataframe()
        # check that dataframe index equals time array
        self.assertTrue((df.index == ts.t).all(), "time array of dataframe is incorrect")
        # check that data array is the same
        self.assertTrue((df[name] == ts.x).all(), "time series data array of dataframe is incorrect")

    def test_copy(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        name = "Surge"
        ts1 = self.db.get(name=name)
        db2 = self.db.copy()
        ts2 = db2.get(name=name)
        self.assertIsNot(ts1, ts2, "Copy with shallow=False kept binding on ts to source database")
        self.assertTrue(np.array_equal(ts1.x, ts2.x), "Copy did returned TimeSeries with different value array")

    def test_copy_shallow(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        name = "Surge"
        ts1 = self.db.get(name=name)
        db2 = self.db.copy(shallow=True)
        ts2 = db2.get(name=name)
        self.assertIs(ts1, ts2, "Copy with shallow=True did not return source instance")

    def test_update(self):
        self.skipTest("todo: update db2 name and ts names")
        """
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        n_before = self.db.n
        db2 = TsDB()
        db2.load(os.path.join(self.data_directory, ' ... '))
        self.db.update(db2, names="*")
        n_after = self.db.n
        ts1 = self.db.get_ts(name="")
        ts2 = db2.get_ts(name="")
        self.assertEqual(n_before + 3, n_after, "Did not update with correct number of keys")
        self.assertIsNot(ts1, ts2, "Update with shallow=False kept binding on ts to source database")
        """

    def test_update_shallow(self):
        self.skipTest("todo: update db2 name and ts names")
        """
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        n_before = self.db.n
        db2 = TsDB()
        db2.load(os.path.join(self.data_directory, '....ts'))
        self.db.update(db2, names="JACKET*motion", shallow=True)
        n_after = self.db.n
        ts1 = self.db.get_ts(name="...")
        ts2 = db2.get_ts(name="...")
        self.assertEqual(n_before + 3, n_after, "Did not update with correct number of keys")
        self.assertIs(ts1, ts2, "Update with shallow=True did not return source instance")
        """

    def test_is_common_time_false(self):
        self.skipTest("todo: update db2 name and ts names")
        """
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        self.db.load(os.path.join(self.data_directory, '....ts'))
        names = "Surge", "..."
        is_common = self.db.is_common_time(names=names)
        self.assertFalse(is_common, "'is_common_time()' did not report False")
        """

    def test_is_common_time_true(self):
        self.skipTest("todo: update db2 name and ts names")
        """
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        self.db.load(os.path.join(self.data_directory, '....ts'))
        names = "Surge", "Sway"
        is_common = self.db.is_common_time(names=names)
        self.assertTrue(is_common, "'is_common_time()' did not report True")
        """

    def test_export_uncommon_timearray_error(self):
        self.skipTest("todo: update db2 name and ts names")
        """
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        self.db.load(os.path.join(self.data_directory, '....ts'))
        names = "Surge", "..."
        keys = self.db.list(names=names, display=False)
        fnout = os.path.join(self.data_directory, '_test_export.ts')
        try:
            self.db.export(fnout, keys=keys)
        except ValueError:
            pass
        else:
            # clean exported files (in the event is was exported though it should not)
            os.remove(fnout)
            os.remove(os.path.splitext(fnout)[0] + ".key")
            self.fail("Did not throw exception when exporting un-common time arrays to .ts")
        """

    def test_export(self):
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        names = "Surge", "Sway"
        keys = self.db.list(names=names, display=False)
        fnout = os.path.join(self.data_directory, "_test_export.ts")
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, "w")
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, names=keys)
        finally:
            # reset sys.stdout
            sys.stdout = was_stdout
            f.close()
            # clean (remove exported files)
            with contextlib.suppress(FileNotFoundError):
                os.remove(fnout)
                os.remove(os.path.splitext(fnout)[0] + ".key")
        # should not raise errors

    def test_export_reload(self):
        # export and reload .ts, check that time series data is the same
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        name = "Sway"
        fnout = os.path.join(self.data_directory, "_test_export.ts")
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, "w")
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, names=name)
        finally:
            # reset sys.stdout
            sys.stdout = was_stdout
            f.close()
        # reload
        db2 = TsDB()
        db2.load(fnout)
        # compare ts
        ts1 = self.db.get(name=name)
        ts2 = db2.get(name=name)
        # clean exported files
        with contextlib.suppress(FileNotFoundError):
            os.remove(fnout)
            os.remove(os.path.splitext(fnout)[0] + ".key")

        # check arrays
        self.assertTrue(np.array_equal(ts1.x, ts2.x), "Export/reload of .ts did not yield same arrays")

    def test_export_ascii(self):
        self.db.load(os.path.join(self.data_directory, "model_test_data.dat"))
        names = "WaveC[m]", "Wave-S[m]", "Surge[m]"
        fnout = os.path.join(self.data_directory, "_test_export.dat")
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, "w")
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, names=names, verbose=False)
        finally:
            # clean exported files and route screen dump back
            os.remove(fnout)
            sys.stdout = was_stdout
            f.close()
        # should not raise errors

    def test_export_reload_ascii(self):
        self.db.load(os.path.join(self.data_directory, "model_test_data.dat"))
        name = "Wave-S[m]"
        fnout = os.path.join(self.data_directory, "_test_export.dat")
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, "w")
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, names=name)
        finally:
            sys.stdout = was_stdout
            f.close()
        # reload
        db2 = TsDB()
        db2.load(fnout)
        # compare ts
        ts1 = self.db.get(name=name)
        ts2 = db2.get(name=name)

        # clean exported files
        os.remove(fnout)

        # check arrays
        np.testing.assert_array_almost_equal(ts1.x, ts2.x, 6, "Export/reload of ascii (.dat) did not yield same arrays")

    def test_export_reload_pickle(self):
        # export and reload .pkl, check that time series data is the same
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))
        name = "Sway"
        fnout = os.path.join(self.data_directory, "_test_export.pkl")
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, "w")
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, names=name)
        finally:
            # reset sys.stdout
            sys.stdout = was_stdout
            f.close()
        # reload
        db2 = TsDB()
        db2.load(fnout)
        # compare ts
        ts1 = self.db.get(name=name)
        ts2 = db2.get(name=name)
        # clean exported files
        with contextlib.suppress(FileNotFoundError):
            os.remove(fnout)

        # check arrays
        self.assertTrue(np.array_equal(ts1.x, ts2.x), "Export/reload of pickle (.pkl) did not yield same arrays")

    def test_export_h5(self):
        self.db.load(os.path.join(self.data_directory, "model_test_data.dat"))
        names = "WaveC[m]", "Wave-S[m]", "Surge[m]"
        fnout = os.path.join(self.data_directory, "_test_export.h5")
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, "w")
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, names=names, verbose=False)
        finally:
            # clean exported files and route screen dump back
            os.remove(fnout)
            sys.stdout = was_stdout
            f.close()
        # should not raise errors

    def test_stats_dataframe(self):
        """Test that stats dataframe is correctly constructed"""
        fn = os.path.join(self.data_directory, "mooring.ts")
        keys = ["Surge", "Sway", "Heave"]
        db = TsDB.fromfile(fn)
        stats = db.stats(names=keys)  # type: dict
        df = db.stats_dataframe(names=keys)
        # check that statistics for each time series is correctly stored in columns
        self.assertListEqual(keys, list(df.keys()), "Statistics dataframe does not have time series stats in columns")
        # check that the values are correctly fetched from dataframe
        failed = []
        for k in keys:
            for kstat, val in stats[k].items():
                if not df[k][kstat] == val:
                    failed.append((k, kstat))
        self.assertFalse(failed, "Statistics dataframe values don't match the statistics dict")


class TestTsDBWithoutCommonPath(unittest.TestCase):
    """
    Keys without a common path, e.g. files loaded from different drives (#136). Mixing a relative key (from
    `TsDB.add()` on an empty database) with absolute keys (from `TsDB.load()`) gives the same situation on any OS.
    """

    def setUp(self):
        self.data_directory = os.path.join(os.path.dirname(__file__), "..", "data")
        self.db = TsDB()
        self.db.add(TimeSeries("added", np.arange(10.0), np.zeros(10)))
        self.db.load(os.path.join(self.data_directory, "mooring.ts"))

    def test_no_common_path(self):
        self.assertEqual(self.db.common, "")

    def test_list_relative_returns_full_keys(self):
        self.assertEqual(self.db.list(relative=True), self.db.list())

    def test_getm_without_fullkey_returns_full_keys(self):
        keys = self.db.list(names="*Surge")
        self.assertEqual(list(self.db.getm(names=keys, fullkey=False, store=False).keys()), keys)


@unittest.skipUnless(sys.platform == "win32", "drives only exist on Windows")
class TestTsDBDifferentDrives(unittest.TestCase):
    """Files loaded from two different drives (#136). Skipped unless the temp directory is on another drive."""

    def setUp(self):
        import shutil
        import tempfile

        data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        if os.path.splitdrive(tmp)[0].lower() == os.path.splitdrive(data_directory)[0].lower():
            self.skipTest("the temp directory is on the same drive as the test data")
        for f in ("mooring.ts", "mooring.key"):
            shutil.copy(os.path.join(data_directory, f), tmp)
        self.db = TsDB()
        self.db.load(os.path.join(data_directory, "mooring.ts"))
        self.db.load(os.path.join(tmp, "mooring.ts"))
        self.export_file = os.path.join(tmp, "export.dat")

    def test_list_relative(self):
        self.assertEqual(self.db.common, "")
        self.assertEqual(self.db.list(relative=True), self.db.list())

    def test_export_without_basename(self):
        keys = self.db.list(names="*Surge")
        self.assertEqual(len(keys), 2)
        self.db.export(self.export_file, names=keys, basename=False, verbose=False)
        names = TsDB.fromfile(self.export_file).list(relative=True)
        self.assertEqual(len(names), 2)
        self.assertFalse(any(":" in n for n in names), names)


class TestTsDBBookkeeping(unittest.TestCase):
    """Lazy reading still finds the right data after the register is changed (#175)."""

    @classmethod
    def setUpClass(cls):
        data_directory = os.path.join(os.path.dirname(__file__), "..", "data")
        cls.source = os.path.abspath(os.path.join(data_directory, "mooring.ts"))
        cls.reference = TsDB.fromfile(cls.source).getm(names="*", store=False)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def assert_same_data(self, ts, name):
        np.testing.assert_array_equal(ts.t, self.reference[name].t, err_msg=f"{name} (t)")
        np.testing.assert_array_equal(ts.x, self.reference[name].x, err_msg=f"{name} (x)")

    def copy_source(self, folder, key_file=True):
        os.makedirs(os.path.join(self.tmp, folder))
        shutil.copy(self.source, os.path.join(self.tmp, folder))
        if key_file:
            shutil.copy(self.source.replace(".ts", ".key"), os.path.join(self.tmp, folder))
        return os.path.join(self.tmp, folder, "mooring.ts")

    def test_rename_before_reading(self):
        db = TsDB.fromfile(self.source)
        db.rename("Surge", "Surge renamed")
        self.assert_same_data(db.get(name="Surge renamed"), "Surge")

    def test_copy_before_reading(self):
        for shallow in (False, True):
            with self.subTest(shallow=shallow):
                copied = TsDB.fromfile(self.source).copy(shallow=shallow)
                self.assert_same_data(copied.get(name="Sway"), "Sway")

    def test_copy_registers_match_keys(self):
        """The registers of a copy have exactly the copy's keys, with the source file of each series (#192)."""
        copied = TsDB.fromfile(self.source).copy(names=["Surge", "Sway"])
        self.assertEqual(list(copied.register), copied.register_keys)
        self.assertEqual(list(copied.register_parent), copied.register_keys)
        self.assertEqual(list(copied._register_indices), copied.register_keys)
        self.assertEqual(set(copied.register_parent.values()), {self.source})
        self.assertEqual(list(copied._register_indices.values()), [1, 2])  # Surge and Sway on mooring.ts

    def test_add_returns_key(self):
        db = TsDB.fromfile(self.source)
        ts = self.reference["Surge"].copy()
        ts.name = "added"
        key = db.add(ts)
        self.assertEqual(key, db.register_keys[-1])
        self.assertIs(db.register[key], ts)

    def test_update_from_unread_db(self):
        db = TsDB()
        db.update(TsDB.fromfile(self.source))
        self.assertEqual(db.n, len(self.reference))
        self.assert_same_data(db.get(name="Heave"), "Heave")

    def test_clear_then_read_others(self):
        db = TsDB.fromfile(self.source)
        db.clear(names="Surge", display=False)
        self.assertEqual(db.n, len(self.reference) - 1)
        self.assert_same_data(db.get(name="Sway"), "Sway")
        self.assert_same_data(db.get(name="Mooring line 8"), "Mooring line 8")

    def test_same_name_in_two_files(self):
        first, second = self.copy_source("first"), self.copy_source("second")
        db = TsDB()
        db.load([first, second])
        self.assertEqual(db.n, 2 * len(self.reference))
        with self.assertRaises(ValueError):
            db.get(name="Surge")  # not unique
        for path in (first, second):
            key = os.path.join(path, "Surge")
            self.assert_same_data(db.getm(names=key, store=False, fullkey=True)[key], "Surge")

    def test_missing_key_file(self):
        path = self.copy_source("nokey", key_file=False)
        with self.assertRaises(FileNotFoundError) as cm:
            TsDB.fromfile(path)
        self.assertIn("mooring.key", str(cm.exception))


class TestExportRoundTrip(unittest.TestCase):
    """Exported files read back with the same names and values (#175)."""

    def setUp(self):
        self.data_directory = os.path.join(os.path.dirname(__file__), "..", "data")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def round_trip(self, source, extension, names):
        db = TsDB.fromfile(os.path.join(self.data_directory, source))
        original = db.getm(names=names, store=False)
        path = os.path.join(self.tmp, "export" + extension)
        db.export(path, names=names, verbose=False)
        reloaded = TsDB.fromfile(path).getm(names="*", store=False)
        return original, reloaded

    def test_ts(self):
        """.ts stores single precision, as do the source files used here."""
        original, reloaded = self.round_trip("mooring.ts", ".ts", ["Surge", "Sway", "Mooring line 8"])
        self.assertEqual(list(reloaded), list(original))
        for name, ts in original.items():
            np.testing.assert_allclose(reloaded[name].t, ts.t, rtol=1e-6, err_msg=f"{name} (t)")
            np.testing.assert_allclose(reloaded[name].x, ts.x, rtol=1e-6, err_msg=f"{name} (x)")

    def test_h5(self):
        names = ["WaveC[m]", "Wave-S[m]", "Surge[m]"]
        original, reloaded = self.round_trip("model_test_data.dat", ".h5", names)
        self.assertEqual(sorted(reloaded), sorted(original))
        for name, ts in original.items():
            # .h5 stores the start time and time step, not the time array: the reloaded time is start + i * step,
            # while the .dat source holds the time rounded to 8 decimals
            np.testing.assert_allclose(reloaded[name].t, ts.t, rtol=0, atol=1e-6, err_msg=f"{name} (t)")
            np.testing.assert_allclose(reloaded[name].x, ts.x, rtol=1e-12, err_msg=f"{name} (x)")


class TestTsDBReaders(RegistryTestCase):
    """TsDB reads files through the reader registry (#168)."""

    def setUp(self):
        super().setUp()
        self.data_directory = os.path.join(os.path.dirname(__file__), "..", "data")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_reader_by_name(self):
        """A file whose name matches no reader is read with the reader given by name."""
        path = os.path.join(self.tmp, "example.txt")
        shutil.copyfile(os.path.join(self.data_directory, "example.csv"), path)
        with self.assertRaises(NotImplementedError):
            TsDB.fromfile(path)
        db = TsDB.fromfile(path, reader="csv")
        ref = TsDB.fromfile(os.path.join(self.data_directory, "example.csv"))
        self.assertEqual([os.path.basename(k) for k in db.list()], [os.path.basename(k) for k in ref.list()])
        np.testing.assert_array_equal(db.get(name="yaw").x, ref.get(name="yaw").x)

    def test_reader_instance_not_registered(self):
        db = TsDB()
        path = os.path.join(self.tmp, "data.xyz")
        open(path, "w").close()
        db.load(path, reader=make_reader("unregistered")())
        self.assertNotIn("unregistered", [r.name for r in registry.readers()])
        ts = db.get(name="a")
        self.assertEqual(ts.unit, "m")  # units from the reader reach the TimeSeries
        np.testing.assert_array_equal(ts.x, [0.0, 1.0, 2.0])
        self.assertIsNone(db.get(name="b").unit)

    def test_reader_registered_in_script(self):
        path = os.path.join(self.tmp, "data.xyz")
        open(path, "w").close()
        with self.assertRaises(NotImplementedError):
            TsDB.fromfile(path)
        registry.register(make_reader("script"))
        db = TsDB.fromfile(path)
        self.assertEqual([os.path.basename(k) for k in db.list()], ["a", "b"])
        np.testing.assert_array_equal(db.get(name="b").x, [0.0, 1.0, 2.0])

    def test_invalid_reader(self):
        with self.assertRaises(TypeError):
            TsDB.fromfile(os.path.join(self.data_directory, "mooring.ts"), reader=42)
        with self.assertRaisesRegex(ValueError, "No reader named"):
            TsDB.fromfile(os.path.join(self.data_directory, "mooring.ts"), reader="nope")

    def test_register_indices_deprecated(self):
        db = TsDB.fromfile(os.path.join(self.data_directory, "mooring.ts"))
        with self.assertWarnsRegex(DeprecationWarning, r"TsDB\.register_indices is deprecated since QATS 5\.5\.0"):
            indices = db.register_indices
        self.assertEqual(list(indices.values()), list(range(1, 15)))
        db = TsDB.fromfile(os.path.join(self.data_directory, "results_SIMA36.h5"))
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(set(db.register_indices.values()), {None})

    def test_rename_before_reading_by_name(self):
        """Formats read by name (here .h5) can be read after renaming a series that has not been read yet."""
        path = os.path.join(self.data_directory, "results_SIMA36.h5")
        ref = TsDB.fromfile(path)
        key = ref.list()[3]
        name = key[len(os.path.abspath(path)) + 1 :]
        db = TsDB.fromfile(path)
        db.rename(name, "renamed")
        ts = db.get(name="renamed")
        self.assertEqual(ts.name, os.path.join(os.path.dirname(name), "renamed"))  # .h5 names include the groups
        np.testing.assert_array_equal(ts.x, ref.get(name=key).x)

    def test_store_false_keeps_nothing(self):
        db = TsDB.fromfile(os.path.join(self.data_directory, "mooring.ts"))
        for _ in range(2):
            db.getm(names="*", store=False)
            self.assertTrue(all(ts is None for ts in db.register.values()))


if __name__ == "__main__":
    unittest.main()
