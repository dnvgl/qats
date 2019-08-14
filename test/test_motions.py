# -*- coding: utf-8 -*-
"""
Module for testing functions in operations module
"""
from qats import TsDB
from qats.motions import transform_motion, velocity, acceleration
import unittest
import os
import numpy as np


class TestOperations(unittest.TestCase):
    def setUp(self):
        self.db = TsDB()
        # the data directory used in the test relative to this module
        # necessary to do it like this for the tests to work both locally and in virtual env for conda build
        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')

    def test_transform_motion(self):
        self.db.load(os.path.join(self.data_directory, 'simo_trans.ts'))
        db = self.db
        motionkeys = db.list(relative=True)[:6]  # names of 6-dof motion time series
        motion = [ts.x for ts in db.getl(motionkeys)]
        # check transformation 1, new ref = (74.61, 0., 0.)
        xyz1 = transform_motion(motion, newref=(74.61, 0., 0.))  # default rotunit ('deg') is used
        np.testing.assert_allclose(xyz1[0], db.get("XG_trans1").x, rtol=0, atol=1e-6,
                                   err_msg="XG differs in transformation 1")
        np.testing.assert_allclose(xyz1[1], db.get("YG_trans1").x, rtol=0, atol=1e-6,
                                   err_msg="YG differs in transformation 1")
        np.testing.assert_allclose(xyz1[2], db.get("ZG_trans1").x, rtol=0, atol=1e-6,
                                   err_msg="YG differs in transformation 1")

    def test_velocity(self):
        """ Check that numerical differentiation of sin(x) is approximately equal to cos(x) """
        x = np.linspace(0., 4 * np.pi, num=1000)
        y = np.sin(x)
        dydt = velocity(y, x)
        # check vs. analytical answer, disregard first and last values (inaccurate at bounds)
        np.testing.assert_allclose(dydt[1:-1], np.cos(x)[1:-1], rtol=1e-4, atol=0,
                                   err_msg="Numerical diff. of sin(x) differs from cos(x)")

    def test_velocity_n_signals(self):
        """ Check that velocity handles 2-D input array (more than one signal) """
        x = np.linspace(0., 4 * np.pi, num=1000)
        y1 = np.sin(x)
        y2 = np.sin(x + np.pi / 4)
        dy = velocity([y1, y2], x)
        dy1 = velocity(y1, x)
        np.testing.assert_array_equal(dy[0, :], dy1, err_msg="velocity fails to handle multiple signals properly")

    def test_acceleration(self):
        """ Check that acceleration is same as time differentiation (velocity) twice """
        x = np.linspace(0., 4 * np.pi, num=1000)
        y = np.sin(x)
        acc = acceleration(y, x)
        dydt2 = velocity(velocity(y, x), x)
        np.testing.assert_array_equal(acc, dydt2, err_msg="acceleration differs from double time differentiation")


if __name__ == '__main__':
    unittest.main()
