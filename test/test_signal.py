# -*- coding: utf-8 -*-
"""
Module for testing signal processing functions
"""
import unittest
import numpy as np
import os
from qats.signal import smooth, average_frequency, taper, lowpass, highpass, bandblock, bandpass, psd, find_maxima
from qats import TsDB


class TestSignal(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0., 100000., num=1000000)
        self.x1 = 10 + np.sin(2. * np.pi * 0.05 * self.t)
        self.x2 = 0.15 * np.sin(2. * np.pi * 0.20 * self.t)
        self.x = self.x1 + self.x2
        self.xnoise = self.x + 0.1 * np.random.randn(np.size(self.x))

        self.data_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.peaks_file = "example.peaks.ts"
        self.peaks_path = os.path.join(self.data_directory, self.peaks_file)

    def test_average_frequency(self):
        """Check that the average frequency is correct."""
        self.assertAlmostEqual(average_frequency(self.t, self.x1, up=True), 0.05, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x1, up=False), 0.05, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x2, up=True), 0.20, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x2, up=False), 0.20, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x, up=True), 0.05, places=3)
        self.assertAlmostEqual(average_frequency(self.t, self.x, up=False), 0.05, places=3)

    def test_smooth(self):
        """Check that the noise is removed and that the average mean crossing frequency equals that of the base signal."""
        self.assertAlmostEqual(average_frequency(self.t, smooth(self.xnoise, window_len=31), up=True), 0.05, places=3)

    def test_taper(self):
        """Check that the signal is tapered to zero in both ends."""
        tapered, _ = taper(self.xnoise, alpha=0.02)
        self.assertAlmostEqual(tapered[0], 0., delta=0.01)
        self.assertAlmostEqual(tapered[-1], 0., delta=0.01)

    def test_reconstruct_signal_from_lowpass_and_higpass(self):
        """Check that the sum of the lowpassed signal and the highpassed signal equals the original signal."""
        dt = self.t[1] - self.t[0]
        xlp = lowpass(self.xnoise, dt, 0.1)
        xhp = highpass(self.xnoise, dt, 0.1)
        deviation = max((xlp + xhp - self.xnoise) / self.xnoise)
        self.assertLessEqual(deviation, 0.05)

    def test_reconstruct_signal_from_bandstop_and_bandpass(self):
        """Check that the sum of the lowpassed signal and the highpassed signal equals the original signal."""
        dt = self.t[1] - self.t[0]
        band = bandpass(self.xnoise, dt, 0.15, 0.25)
        rest = bandblock(self.xnoise, dt, 0.15, 0.25)
        deviation = max((band + rest - self.xnoise) / self.xnoise)
        self.assertLessEqual(deviation, 0.05)

    def test_statistics_of_lowpassed_signal(self):
        """Check statistics against analytical solution."""
        dt = self.t[1] - self.t[0]
        x = lowpass(self.xnoise, dt, 0.1)
        self.assertAlmostEqual(np.mean(x), 10., delta=0.001)
        self.assertAlmostEqual(np.var(x), 0.5, delta=0.001)     # variance of sinoid = amplitude ** 2 / 2

    def test_statistics_of_highpassed_signal(self):
        """Check statistics against analytical solution."""
        dt = self.t[1] - self.t[0]
        x = highpass(self.xnoise, dt, 0.1)
        self.assertAlmostEqual(np.mean(x), 0., delta=0.001)

        # variance of sinoid = amplitude ** 2 / 2
        # variance of random uniform distributed number is amplitude squared
        # the processes are independent
        self.assertAlmostEqual(np.var(x), 0.15 ** 2. / 2. + 0.1 ** 2., delta=0.001)

    def test_statistics_of_bandpassed_signal(self):
        """Check statistics against analytical solution."""
        dt = self.t[1] - self.t[0]
        x = bandpass(self.xnoise, dt, 0.1, 0.25)
        self.assertAlmostEqual(np.mean(x), 0., delta=0.001)

        # variance of sinoid = amplitude ** 2 / 2
        self.assertAlmostEqual(np.var(x), 0.15 ** 2. / 2., delta=0.001)

    def test_statistics_of_bandblocked_signal(self):
        """Check statistics against analytical solution."""
        dt = self.t[1] - self.t[0]
        x = bandblock(self.xnoise, dt, 0.1, 0.25)
        self.assertAlmostEqual(np.mean(x), 10., delta=0.001)

        # variance of sinoid = amplitude ** 2 / 2
        # variance of random uniform distributed number is amplitude squared
        # the processes are independent
        self.assertAlmostEqual(np.var(x), 0.5 + 0.1 ** 2., delta=0.001)

    def test_psd_area_moment_0(self):
        """Check zero area moment of the spectral density."""
        dt = self.t[1] - self.t[0]
        f, p = psd(self.x, dt, nperseg=1000)
        df = f[1] - f[0]
        self.assertAlmostEqual(df * np.sum(p), np.var(self.x), delta=1.e-3)

    def test_psd_area_moment_2(self):
        """Check second area moment of the spectral density."""
        dt = self.t[1] - self.t[0]
        f, p = psd(self.x2, dt, nperseg=1000)
        df = f[1] - f[0]
        m0 = df * np.sum(p)
        m2 = df * np.sum(f ** 2. * p)
        self.assertAlmostEqual(np.sqrt(m2 / m0), 0.2, delta=1.e-3)

    def test_psd_nyquist_frequency(self):
        """Check that the maximum psd frequency equals the Nyquist frequency."""
        dt = self.t[1] - self.t[0]
        f, _ = psd(self.x, dt)
        self.assertAlmostEqual(np.max(f), 0.5 * 1. / dt, delta=1.e-6)

    def test_psd_zero_frequency(self):
        """Check that the maximum psd frequency equals the Nyquist frequency."""
        dt = self.t[1] - self.t[0]
        f, _ = psd(self.x, dt)
        self.assertAlmostEqual(np.min(f), 0., delta=1.e-6)

    def test_find_maxima_global(self):
        """ 
        Check that correct number of global maxima is found
        * end points shoult not be included
        * if down-crossing after last up-crossing, peak in-between should be included
        
        This test would have caught issue #106: https://github.com/dnvgl/qats/issues/106
        """
        db = TsDB.fromfile(self.peaks_path)
        ts = db.get(ind=0)

        # this time series has 842 global maxima
        # * the last global maximum is between an up-crossing and a down-crossing
        # * the last up-crossing is between the last two points in the series - this 
        #   previously lead find_maxima() to erroneously identify the end point as an
        #   additional global maximum (=> 843 maxima), ref. issue 106.
        x1 = ts.x
        peaks1, _ = find_maxima(x1, local=False)
        npeaks1 = peaks1.size
        self.assertEqual(npeaks1, 842)
        
        # also check that if the last mean-level crossing is a down-crossing, the
        # global maximum of this last half-cycle is included (otherwise, only 841 peaks
        # will be found)
        x2 = x1[:-10]
        peaks2, _ = find_maxima(x2, local=False)
        npeaks2 = peaks2.size
        self.assertEqual(npeaks2, 842)

if __name__ == '__main__':
    unittest.main()
