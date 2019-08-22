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
    cycles : list
        Pairs of cycle range and mean
    uts : float
        Material ultimate tensile strength

    Returns
    -------
    list
        Corrected stress ranges

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

    """
    ranges, means = zip(*cycles)

    # calculate effective alternating stress
    corrected_ranges = np.array(ranges) * (uts / (uts - np.array(means)))

    return list(corrected_ranges)

