# -*- coding: utf-8 -*-
"""
Module for testing stats functions
"""
import unittest
import numpy as np
from qats.signal import smooth, average_frequency
from qats.stats.gumbel import _euler_masceroni


class TestFunStats(unittest.TestCase):
    def test_eulermascheroni(self):
        self.assertAlmostEqual(_euler_masceroni(), 0.57721566490153286060651209008240243104215933593992, places=12)


class TestSignal(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0., 1000., num=10000)
        self.x1 = np.sin(2.*np.pi*0.05*self.t)
        self.x2 = 0.15*np.sin(2. * np.pi * 0.15 * self.t)
        self.x3 = self.x1 + self.x2
        self.noise = self.x3 + 0.1*np.random.randn(np.size(self.x3))

    def test_average_frequency(self):
        self.assertAlmostEqual(average_frequency(self.t, self.x1, up=True), 0.05, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x1, up=False), 0.05, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x2, up=True), 0.15, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x2, up=False), 0.15, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x3, up=True), 0.05, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x3, up=False), 0.05, places=3)

    def test_smooth(self):
        # fails if not smoothed
        self.assertAlmostEqual(average_frequency(self.t, smooth(self.noise, window_len=11), up=True), 0.05, places=3)


if __name__ == '__main__':
    unittest.main()
