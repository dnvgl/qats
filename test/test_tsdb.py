# -*- coding: utf-8 -*-
"""
Module for testing TsDB class
"""
from qats import TimeSeries, TsDB
import unittest
import os
import numpy as np
import sys


# todo: add tests for listing subset(s) based on specifying parameter `names` (with and wo param. `keys`)
# todo: more tests for TsDB.get_many_ts()
# todo: more tests for .h5 files (use flexout.h5 and example_h5_output.h5)
# todo: add test for get_many() with fullkey=False (similar to test_get_many_correct_key, but with shorter key)


class TestTsDB(unittest.TestCase):
    def setUp(self):
        self.db = TsDB()
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env for conda build
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')

    def test_exception_load_numeric(self):
        try:
            self.db.load(223334)    # numeric values should throw an exception
        except TypeError:
            pass
        else:
            self.fail("Did not throw exception on numeric file name")

    def test_exception_load_dict(self):
        try:
            self.db.load({})    # dictionary should throw an exception
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
            self.db.load(os.path.join(self.data_directory, 'donotexist.ts'))
        except FileExistsError:
            pass
        else:
            self.fail("Did not throw exception when trying to load a non-existing file.")

    def test_exception_load_unsupportedfile(self):
        try:
            self.db.load(os.path.join(self.data_directory, 'unsupportedfile.out'))
        except NotImplementedError:
            pass
        else:
            self.fail("Did not throw exception when trying to load a file type which is not yet supported.")

    def test_h5_sima_load_correct_number_of_keys(self):
        self.db.load(os.path.join(self.data_directory, 'example_sima_h5_output_shrinked.h5'))
        self.assertEqual(self.db.n, 34, "Should identify 34 time series datasets on the sima .h5 file")

    ''' temporarily disabled due to failure when building - to be debugged later
    def test_h5_sima_timeseries_correctly_read(self):
        fn = 'example_h5_output.h5'
        self.db.load(os.path.join(self.data_directory, fn))
        t, x = self.db.get(name='TotalforceElement1')
        self.assertTrue(t.size > 1 and t.size == x.size,
                        "Reading of time series from .h5 file '%s' failed" % fn)
    '''

    def test_list_all(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        k = self.db.list(display=False)
        self.assertEqual(14, len(k), "Deviating number of listed keys = %d" % len(k))

    def test_list_subset(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        k = self.db.list(names="Mooring line*", display=False)
        self.assertEqual(8, len(k), "Deviating number of listed keys = %d" % len(k))

    def test_list_subset_misc_criteria(self):
        for tsfile in ('mooring.ts', 'simo_p.ts'):
            self.db.load(os.path.join(self.data_directory, tsfile))
        # test 1
        k = self.db.list(names="Tension*", display=False)
        self.assertEqual(10, len(k), "Deviating number of listed keys = %d" % len(k))
        # test 2
        k = self.db.list(names="simo_p.ts*line*", display=False)
        self.assertEqual(2, len(k), "Deviating number of listed keys = %d" % len(k))

    def test_list_subset_keep_specified_order(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        names_reversed = list(reversed([os.path.basename(k) for k in self.db.register_keys]))
        namelist = [os.path.basename(_) for _ in self.db.list(names=names_reversed)]
        self.assertEqual(names_reversed, namelist, "Failed to keep specified order")

    def test_list_subset_special_characters(self):
        self.db.load(os.path.join(self.data_directory, 'model_test_data.dat'))
        # should return exactly one key
        self.assertEqual(1, len(self.db.list(names="RW1[m]")), "TsDB.list() returned wrong number of keys")

    def test_list_subset_special_characters_2(self):
        self.db.load(os.path.join(self.data_directory, 'model_test_data.dat'))
        # should return exactly one key
        self.assertEqual(1, len(self.db.list(names="Acc-X[m/s^2]")), "TsDB.list() returned wrong number of keys")

    def test_clear_all(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        self.db.clear(display=False)
        k = self.db.list(display=False)
        self.assertEqual([], k, "Did not clear all registered keys.")

    def test_clear_subset(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        self.db.clear(names="*Mooring line*", display=False)
        k = self.db.list(display=False)
        self.assertEqual(6, len(k), "Did not clear subset of registered keys correctly. %d keys remaining" % len(k))

    def test_getd_correct_key(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        rk = self.db.list(names="Heave", display=False)
        container = self.db.getd(names="Heave", fullkey=True)
        self.assertEqual(rk, list(container.keys()), "db list method and get_many method returns different keys.")

    def test_getd_correct_number_of_arrays(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        rk = self.db.list(names="Heave", display=False)  # should be only 1 key returned in this case
        container = self.db.getd(names="Heave", fullkey=True)
        self.assertEqual(2, len(container[rk[0]]), "Got more than 2 arrays (time and data) in return from get_many().")

    def test_gets_none(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        container = self.db.getd(names=[])
        n = len(container)
        self.assertEqual(0, n, "Should have received empty container (OrderedDict) from get_many()")

    def test_getm_correct_key(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        rk = self.db.list(names="Heave", display=False)
        container = self.db.getm(names="Heave", fullkey=True)
        self.assertEqual(rk, list(container.keys()), "db list method and get_many_ts method returns different keys.")

    def test_getm_correct_key_by_ind(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        rk = self.db.list(names="Heave", display=False)
        container = self.db.getm(ind=2, fullkey=True)
        self.assertEqual(rk, list(container.keys()), "db list method and get_many_ts method returns different keys.")

    def test_geta(self):
        tsfile = os.path.join(self.data_directory, 'simo_p.ts')
        self.db.load(tsfile)
        tsname = "Tension_2_qs"
        keys = self.db.list(names=tsname, display=False)
        _, data1 = self.db.geta(name=keys[0])

        # test 1: geta() when ts is already loaded
        _, data2 = self.db.geta(name=tsname)
        self.assertTrue(np.array_equal(data1, data2), "Did not get correct data time series using get() "
                                                      "(ts pre-loaded)")
        # test 2: geta() when ts is not already loaded
        db2 = TsDB()
        db2.load(tsfile)
        _, data3 = db2.geta(name=tsname)
        self.assertTrue(np.array_equal(data1, data3), "Did not get correct data time series using get() "
                                                      "(ts not pre-loaded)")

    def test_get_by_name(self):
        tsfile = os.path.join(self.data_directory, 'simo_p.ts')
        self.db.load(tsfile)
        tsname = "Tension_2_qs"
        keys = self.db.list(names=tsname, display=False)
        key = keys[0]
        ts1 = self.db.getm(names=key, fullkey=True)[key]
        # test 1: get_ts() when ts is already loaded
        ts2 = self.db.get(name=tsname)
        self.assertIs(ts1, ts2, "Did not get correct TimeSeries  using get_ts()"
                                " (ts pre-loaded)")
        # test 2: get_ts() when ts is not already loaded
        db2 = TsDB.fromfile(tsfile)
        ts3 = db2.get(name=tsname)
        self.assertTrue(np.array_equal(ts1.x, ts3.x), "Did not get correct TimeSeries using get_ts()"
                                                      " (ts not pre-loaded)")

    def test_get_by_index(self):
        tsfile = os.path.join(self.data_directory, 'simo_p.ts')
        self.db.load(tsfile)
        tsname = "Tension_2_qs"
        key = self.db.list(names=tsname, display=False)[0]
        ts1 = self.db.get(name=tsname)
        ind = self.db.register_keys.index(key)
        # test 1: get_ts() using index when ts is already loaded
        ts2 = self.db.get(ind=ind)
        self.assertIs(ts1, ts2, "Did not get correct TimeSeries using get_ts() and specifying index"
                                " (ts pre-loaded)")

        # test 2: get_ts() using index when ts is not already loaded
        db2 = TsDB.fromfile(tsfile)
        ts3 = db2.get(ind=ind)
        self.assertTrue(np.array_equal(ts1.x, ts3.x), "Did not get correct TimeSeries using get_ts() and specifying"
                                                      " index (ts not pre-loaded)")

    def test_get_exceptions(self):
        self.db.load(os.path.join(self.data_directory, 'simo_p.ts'))
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
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        rk = self.db.list(names="Heave", display=False)  # should be only 1 key returned in this case
        container = self.db.getd(names="Heave", fullkey=True)
        self.assertEqual(65536, len(container[rk[0]][0]), "Deviating number of time steps.")

    def test_add_raises_keyerror_on_nonunique_key(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        container = self.db.getm(names="Surge", fullkey=True)
        for k, v in container.items():
            try:
                self.db.add(v)
            except KeyError:
                pass
            else:
                self.fail("Did not raise KeyError when trying to add time series with non-unique name to db.")

    def test_add_does_not_raise_error(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        ts = TimeSeries("quiteuniquekeyiguess", np.arange(0., 100., 0.01), np.sin(np.arange(0., 100., 0.01)))
        self.db.add(ts)
        # should not raise errors

    def test_rename(self):
        tsfile = os.path.abspath(os.path.join(self.data_directory, 'simo_p.ts'))
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
        self.db.rename(newname, name=oldname)
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
        tsfile = os.path.join(self.data_directory, 'simo_p.ts')
        self.db.load(tsfile)
        oldname = "Tension_2_qs"
        newname = "Tension_3_qs"
        try:
            self.db.rename(newname, name=oldname)
        except ValueError:
            pass
        else:
            self.fail("rename() did not raise ValueError when specifying existing key/name")

    def test_maxima_minima(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
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
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        container = self.db.getd(names="Surge")
        for key, ts in container.items():
            self.assertIsInstance(key, str, "Key should be type string.")
            self.assertIsInstance(ts, tuple, "Time series container should be type tuple.")
            self.assertIsInstance(ts[0], np.ndarray, "First item of time series container should be type numpy array.")
            self.assertIsInstance(ts[1], np.ndarray, "Second item of time series container should be type numpy array.")

    def test_types_in_container_from_get_many_ts(self):
        """
        Test correct types
        """
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        container = self.db.getm(names="Surge")
        for key, ts in container.items():
            self.assertIsInstance(key, str, "Key should be type string.")
            self.assertIsInstance(ts, TimeSeries, "Time series container should be type TimeSeries.")
            self.assertIsInstance(ts.t, np.ndarray, "Attribute t of time series should be type numpy array.")
            self.assertIsInstance(ts.x, np.ndarray, "Attribute x of time series should be type numpy array.")

    def test_copy(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        name = "Surge"
        db1 = self.db
        ts1 = self.db.get(name=name)
        db2 = self.db.copy()
        ts2 = db2.get(name=name)
        self.assertIsNot(ts1, ts2, "Copy with shallow=False kept binding on ts to source database")
        self.assertTrue(np.array_equal(ts1.x, ts2.x), "Copy did returned TimeSeries with different value array")

    def test_copy_shallow(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        name = "Surge"
        db1 = self.db
        ts1 = self.db.get(name=name)
        db2 = self.db.copy(shallow=True)
        ts2 = db2.get(name=name)
        self.assertIs(ts1, ts2, "Copy with shallow=True did not return source instance")

    def test_update(self):
        pass
        # todo: update db2 name and ts names
        '''
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
        '''

    def test_update_shallow(self):
        pass
        # todo: update db2 name and ts names
        '''
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
        '''

    def test_is_common_time_false(self):
        pass
        # todo: update db2 name and ts names
        '''
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        self.db.load(os.path.join(self.data_directory, '....ts'))
        names = "Surge", "..."
        is_common = self.db.is_common_time(names=names)
        self.assertFalse(is_common, "'is_common_time()' did not report False")
        '''

    def test_is_common_time_true(self):
        pass
        # todo: update db2 name and ts names
        '''
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        self.db.load(os.path.join(self.data_directory, '....ts'))
        names = "Surge", "Sway"
        is_common = self.db.is_common_time(names=names)
        self.assertTrue(is_common, "'is_common_time()' did not report True")
        '''

    def test_export_uncommon_timearray_error(self):
        pass
        # todo: update db2 name and ts names
        '''
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
        '''

    def test_export(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        names = "Surge", "Sway"
        keys = self.db.list(names=names, display=False)
        fnout = os.path.join(self.data_directory, '_test_export.ts')
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, keys=keys)
        finally:
            # reset sys.stdout
            sys.stdout = was_stdout
            f.close()
            # clean (remove exported files)
            try:
                os.remove(fnout)
                os.remove(os.path.splitext(fnout)[0] + ".key")
            except FileNotFoundError:
                pass
        # should not raise errors

    def test_export_reload(self):
        self.db.load(os.path.join(self.data_directory, 'mooring.ts'))
        name = "Sway"
        keys = self.db.list(names=name, display=False)
        fnout = os.path.join(self.data_directory, '_test_export.ts')
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, keys=keys)
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
        try:
            os.remove(fnout)
            os.remove(os.path.splitext(fnout)[0] + ".key")
        except FileNotFoundError:
            pass

        # check arrays
        self.assertTrue(np.array_equal(ts1.x, ts2.x), "Export/reload did not yield same arrays")

    def test_export_ascii(self):
        self.db.load(os.path.join(self.data_directory, 'model_test_data.dat'))
        names = "WaveC[m]", "Wave-S[m]", "Surge[m]"
        keys = self.db.list(names=names, display=False)
        fnout = os.path.join(self.data_directory, '_test_export.dat')
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, keys=keys, verbose=False)
        finally:
            # clean exported files and route screen dump back
            os.remove(fnout)
            sys.stdout = was_stdout
            f.close()
        # should not raise errors

    def test_export_reload_ascii(self):
        self.db.load(os.path.join(self.data_directory, 'model_test_data.dat'))
        name = "Wave-S[m]"
        keys = self.db.list(names=name, display=False)
        fnout = os.path.join(self.data_directory, '_test_export.dat')
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
            # export, should not raise errors
            self.db.export(fnout, keys=keys)
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
        np.testing.assert_array_almost_equal(ts1.x, ts2.x, 6, "Export/reload did not yield same arrays")


if __name__ == '__main__':
    unittest.main()
