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


@functools.cache
def riser_seeds():
    """
    20 realisations of one process from simulations: the top tension of a catenary riser in the same sea state with
    different wave seeds, 3 hours each at 0.1 s, from `data/riser_tension_seeds.ts` (see `data/README.md`).

    Unlike :func:`realisations`, a non-Gaussian response: the largest maxima are short peaks after near-slack.

    Returns
    -------
    dict
        TimeSeries by name, 'seed01' ... 'seed20'. Cached: don't change the series.
    """
    import os

    from qats import TsDB

    db = TsDB.fromfile(os.path.join(os.path.dirname(__file__), "..", "data", "riser_tension_seeds.ts"))
    return {name: db.get(name=name) for name in db.list(names="seed*", display=False)}
