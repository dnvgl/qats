# -*- coding: utf-8 -*-
"""
Module for testing io operations directly
"""
from qats import TsDB
from qats.io import sima
import unittest
import os
import sys


class TestTsDB(unittest.TestCase):
    def setUp(self):
        self.db = TsDB()
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env for conda build
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')

    def test_write_simo_file(self):
        self.db.load(os.path.join(self.data_directory, 'integer.csv'))
        name = "random"
        ts = self.db.get(name=name, store=False)

        fnout = os.path.join(self.data_directory, '_test_export.asc')
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
            # export, should not raise errors
            sima.write_simo_file(fnout, ts.t, ts.x, dt=0.5)
        finally:
            # reset sys.stdout
            sys.stdout = was_stdout
            f.close()
            # clean (remove exported files)
            try:
                os.remove(fnout)
            except FileNotFoundError:
                pass
        # should not raise errors

    def test_write_simo_file_negative_time(self):
        self.db.load(os.path.join(self.data_directory, 'negative_time.csv'))
        name = "random"
        ts = self.db.get(name=name, store=False)

        fnout = os.path.join(self.data_directory, '_test_export.asc')
        try:
            # route screen dump from export to null
            was_stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
            # export, should not raise errors
            sima.write_simo_file(fnout, ts.t, ts.x, dt=0.5)
        finally:
            # reset sys.stdout
            sys.stdout = was_stdout
            f.close()
            # clean (remove exported files)
            try:
                os.remove(fnout)
            except FileNotFoundError:
                pass
        # should not raise errors


if __name__ == '__main__':
    unittest.main()
