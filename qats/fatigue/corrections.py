#!/usr/bin/env python
# coding: utf-8
"""
Fatigue related corrections.
"""
import numpy as np


def goodman_haigh(cycles, uts):
    """
    Effective alternating stress using the Goodman-Haigh mean stress correction for fully reversed loading (R=-1).

    Parameters
    ----------
    cycles : np.ndarray or list
        Array of cycle ranges and mean values, shape `(n, 2)`. First column is cycle ranges, second column is cycle
        mean values.
    uts : float
        Material ultimate tensile strength, in same unit as cycle ranges and mean values.

    Returns
    -------
    corrected_ranges: np.ndarray
        Corrected stress ranges, shape: `(n,)` (1D array).

    Notes
    -----
    In materials science and fatigue, the Goodman relation is an equation used to quantify the interaction of mean and
    alternating stresses on the fatigue life of a material.

    A Goodman diagram,[1][2] sometimes called a Haigh diagram or a Haigh-Soderberg diagram,[3] is a graph of (linear)
    mean stress vs. (linear) alternating stress, showing when the material fails at some given number of cycles.

    A scatterplot of experimental data shown on such a plot can often be approximated by a parabola known as the Gerber
    line, which can in turn be (conservatively) approximated by a straight line called the Goodman line.

    Correcting the stress ranges like this can only be applied with an SN-curve with a stress ratio (R) of -1.

    References
    ----------
    1. Herbert J. Sutherland and John F. Mandell. "Optimized Goodman diagram for the analysis of fiberglass composites
       used in wind turbine blades"
    2. David Roylance. "Fatigue", Archived 2011-06-29 at the Wayback Machine.. 2001
    3. Tapany Udomphol. "Fatigue of metals". 2007.

    Examples
    --------

    >>> from qats.fatigue.rainflow import count_cycles
    >>> from qats.fatigue.corrections import goodman_haigh
    >>> # assuming a series has been established, and that uts is defined
    >>> cycles = count_cycles(series)
    >>> corrected_ranges = goodman_haigh(cycles[:, :2], uts)

    The array obtained is then a 1D array of same size as the first dimension of `cycles`:

    >>> cycles.shape
    (1350, 3)
    >>> corrected_ranges.shape
    (1350,)

    """
    # ensure array and assert 2d
    cycles = np.asarray(cycles)
    assert cycles.ndim == 2 and cycles.shape[1] == 2, \
        "Cycles must be specified as 2D array or shape (n, 2) (or: list of 2-tuples)"

    # unpack
    ranges, means = cycles.T

    # calculate effective alternating stress
    corrected_ranges = ranges * (uts / (uts - means))

    return corrected_ranges

