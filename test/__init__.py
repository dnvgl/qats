# -*- coding: utf-8 -*-
import functools
import importlib.util
import unittest

# the desktop application needs qtpy and a Qt binding (the extra "gui"); checked without importing them, so that
# collecting the tests doesn't load Qt
HAS_QT = importlib.util.find_spec("qtpy") is not None and any(
    importlib.util.find_spec(binding) is not None for binding in ("PySide6", "PyQt6")
)
requires_qt = unittest.skipUnless(HAS_QT, 'needs Qt: python -m pip install "qats[gui]"')


@functools.cache
def realisations(n=12, duration=10800.0, dt=0.5):
    """
    `n` independent realisations of the same stationary Gaussian process, as from simulations with different seeds.

    A narrow-banded process around the mean level 100 with standard deviation 5: a sum of cosines with equal
    amplitudes at every frequency 1/duration apart between 0.08 and 0.12 Hz, with random phases. The frequency spacing
    1/duration keeps each realisation from repeating within the record. The extremes of the realisations are a sample
    for a Gumbel fit.

    Returns
    -------
    dict
        TimeSeries by name, 'seed 1' ... 'seed n'. Cached: don't change the series.
    """
    import numpy as np

    from qats import TimeSeries

    t = np.arange(0.0, duration, dt)
    k = np.flatnonzero((np.fft.rfftfreq(t.size, dt) >= 0.08) & (np.fft.rfftfreq(t.size, dt) <= 0.12))
    amplitude = 5.0 * np.sqrt(2.0 / k.size)  # variance: k.size * amplitude**2 / 2 = 25
    out = dict()
    for seed in range(1, n + 1):
        phases = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, k.size)
        spectrum = np.zeros(t.size // 2 + 1, dtype=complex)
        spectrum[k] = 0.5 * t.size * amplitude * np.exp(1j * phases)  # irfft divides by the number of points
        x = 100.0 + np.fft.irfft(spectrum, n=t.size)
        out[f"seed {seed}"] = TimeSeries(f"seed {seed}", t, x)
    return out
