# -*- coding: utf-8 -*-
"""
Module for testing signal processing functions
"""
import unittest
import numpy as np
from qats.signal import smooth, average_frequency, taper, lowpass, highpass, bandblock, bandpass


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

    def test_taper(self):
        tapered, _ = taper(self.noise, alpha=0.01)
        self.assertAlmostEqual(tapered[0], 0., places=3)
        self.assertAlmostEqual(tapered[-1], 0., places=3)

    def test_lp_hp(self):
        dt = self.t[1] - self.t[0]
        xlp = lowpass(self.noise, dt, 0.1)
        xhp = highpass(self.noise, dt, 0.1)
        self.assertTrue(np.allclose(self.noise, xlp + xhp))

    def test_bandstop_bandpass(self):
        dt = self.t[1] - self.t[0]
        band = bandpass(self.noise, dt, 0.1, 0.2)
        rest = bandblock(self.noise, dt, 0.1, 0.2)
        self.assertTrue(np.allclose(self.noise, band + rest))


if __name__ == '__main__':
    unittest.main()
