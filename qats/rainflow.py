#!/usr/bin/env python
# coding: utf-8
"""
Rainflow cycle counting algorithm according to ASTM E1049-85 (2011), section 5.4.4.

.. todo:: Include example of typical use in ``qats.rainflow`` docstring.
"""

from collections import deque, defaultdict
import numpy as np


def reversals(series):
    """
    A generator function which iterates over the reversals in the iterable *series*.

    Parameters
    ----------
    series : array_like
        data series

    Returns
    -------
    generator
        yields floats

    Notes
    -----
    Reversals are the points at which the first derivative on the series changes sign. The generator never yields
    the first and the last points in the series.

    """
    series = iter(series)

    x_last, x = next(series), next(series)
    d_last = (x - x_last)

    for x_next in series:
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0:
            yield x
        x_last, x = x, x_next
        d_last = d_next


def cycles_fromto(series):
    """
    Returns start and end points for full cycles and half-cycles from *series* using the Rainflow algorithm.

    Parameters
    ----------
    series : array_like
        data series

    Returns
    -------
    list
        full cycles
    list
        half cycles

    Notes
    -----
    The unsymmetrical From-To cycle counting keeps more information about the loading cycles than Range and Range-Mean
    counting does, because it also stores the "orientation" of each load cycle. With this you can reconstruct a time
    history from a Rainflow matrix.

    The cycles are extracted from the iterable *series* according to section 5.4.4 in ASTM E1049 (2011).
    """
    points = deque()
    full, half = [], []

    for r in reversals(series):
        points.append(r)
        while len(points) >= 3:
            # Form ranges X and Y from the three most recent points
            x = abs(points[-2] - points[-1])
            y = abs(points[-3] - points[-2])
            fromto = (points[-2], points[-3])   # cycle start and end points

            if x < y:
                # Read the next point
                break
            elif len(points) == 3:
                # Y contains the starting point
                # Count Y as one-half cycle and discard the first point
                half.append(fromto)
                points.popleft()
            else:
                # Count Y as one cycle and discard the peak and the valley of Y
                full.append(fromto)
                last = points.pop()
                points.pop()
                points.pop()
                points.append(last)
    else:
        # Count the remaining ranges as one-half cycles
        while len(points) > 1:
            half.append((points[-1], points[-2]))
            points.pop()
    return full, half


def cycles_rangemean(series):
    """
    Returns range and mean value of all full cycles and half-cycles from *series* using the Rainflow algorithm.

    Parameters
    ----------
    series : array_like
        data series

    Returns
    -------
    list
        full cycles
    list
        half cycles

    Notes
    -----
    The cycles are extracted from the iterable *series* according to section 5.4.4 in ASTM E1049 (2011).

    Examples
    --------
    Extract range and mean value for all full and half cycles.
    >>> from qats.rainflow import cycles_rangemean
    >>> series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    >>> full, half = cycles_rangemean(series)
    >>> full
    [(4, 1)]
    >>> half
    [(3, -0.5), (4, -1.0), (8, 1.0), (6, 1.0), (8, 0.0), (9, 0.5)]

    """
    # extract cycle start and end points
    full, half = cycles_fromto(series)

    # calculate cycle range and mean values
    full = [(abs(_[1] - _[0]), 0.5 * (_[0] + _[1])) for _ in full]
    half = [(abs(_[1] - _[0]), 0.5 * (_[0] + _[1])) for _ in half]

    return full, half


def count_cycles(series, ndigits=None, nbins=None, binwidth=None):
    """
    Returns a sorted list containing pairs of cycle magnitude and count.

    Parameters
    ----------
    series : array_like
        Sata series.
    ndigits : int, optional
        If `ndigits` is given the cycles will be rounded to the given number of digits before counting.
    nbins : int, optional
        If `nbins` is given the cycle count per cycle magnitude is rebinned to `nbins` bins.
    binwidth : float, optional
        If `binwidth` is given the cycle count per cycle magnitude is rebinned to bins with a width of `binwidth`.

    Returns
    -------
    list
        List of tuples with cycle magnitude and count, sorted by increasing magnitude::

            [(magn1, count1), (magn2, count2), ...]


    Notes
    -----
    One-half cycles are counted as 0.5, so the returned counts may not be whole numbers. The cycles are extracted
    from the iterable series using the extract_cycles function.

    Rebinning is not applied if specified `nbins` is larger than the original number of bins.

    `nbins` override `binwidth` if both are specified.

    Examples
    --------
    Assuming `series` is a time series array:

    >>> from qats.rainflow import count_cycles
    >>> cycles = count_cycles(series, nbins=200)

    The sorted list of tuples (pairs) may be unpacked into separate lists of magnitude and count by the command:

    >>> magnitude, count = zip(*cycles)

    """
    full, half = extract_cycles(series)

    # Round the cycle magnitudes if requested
    if ndigits is not None:
        full = (round(x, ndigits) for x in full)
        half = (round(x, ndigits) for x in half)

    # Count cycles
    cycles = defaultdict(float)
    for x in full:
        cycles[x] += 1.0
    for x in half:
        cycles[x] += 0.5

    # sorted pairs (tuples) of cycle magnitude and count in list
    cycles = sorted(cycles.items())

    if ((nbins is not None) and (nbins < len(cycles))) or (binwidth is not None):
        if nbins is not None:
            # rebin to specified number of bins
            cycles_rebinned = rebin_cycles_nbins(cycles, nbins, ndigits=ndigits)
        else:
            # rebin to bins of specified width
            cycles_rebinned = rebin_cycles_binwidth(cycles, binwidth, ndigits=ndigits)

        return cycles_rebinned

    else:
        return cycles


def rebin_cycles_binwidth(cycles, binwidth, ndigits=None):
    """
    Rebin cycle distribution to bins of specified width.

    Parameters
    ----------
    cycles : list
        Sorted list of tuples of cycle magnitude versus count.
    binwidth : float, optional
        Bin width in rebinned distribution.
    ndigits : int, optional
        If `ndigits` is given the cycles will be rounded to the given number of digits before counting.

    Returns
    -------
    list
        Sorted list of tuples of rebinned cycle magnitude versus count.

    Notes
    -----
    Cycles are gathered into a specific bin if ``lower_bound < cycle_magnitude <= upper_bound``. The bin magnitude
    (after rebinning) is represented by its mid-point value ``0.5 * (lower_bound + upper_bound)``.

    """
    # extract maximum magnitude
    max_magnitude = max([m for m, _ in cycles])

    # create bins ranging from 0 to maximum magnitude, stepping with specified bin width
    bins = np.arange(0., max_magnitude + binwidth, binwidth)

    # rebin
    cycles_rebinned = rebin_cycles(cycles, bins, ndigits=ndigits)

    return cycles_rebinned


def rebin_cycles_nbins(cycles, nbins, ndigits=None):
    """
    Rebin cycle distribution to a specified number of equidistant bins

    Parameters
    ----------
    cycles : list
        Sorted list of tuples of cycle magnitude versus count.
    nbins : int, optional
        Number of equidistant bins in rebinned distribution.
    ndigits : int, optional
        If `ndigits` is given the cycles will be rounded to the given number of digits before counting.

    Returns
    -------
    list
        sorted list of tuples of rebinned cycle magnitude versus count

    Notes
    -----
    Cycles are gathered into a specific bin if ``lower_bound < cycle_magnitude <= upper_bound``. The bin magnitude
    (after rebinning) is represented by its mid-point value ``0.5 * (lower_bound + upper_bound)``.

    """
    # extract maximum magnitude
    max_magnitude = max([m for m, _ in cycles])

    # create specified number of bins ranging from 0 to maximum magnitude
    bins = np.linspace(0., max_magnitude, nbins + 1)

    # rebin
    cycles_rebinned = rebin_cycles(cycles, bins, ndigits=ndigits)

    return cycles_rebinned


def rebin_cycles(cycles, bins, ndigits=None):
    """
    Rebin cycle distribution to specified bins

    Parameters
    ----------
    cycles : list
        Sorted list of tuples of cycle magnitude versus count.
    bins : array_like
        Bins specified by bin boundaries e.g. first bin ranges from the first to the second value in `bins`.
    ndigits : int, optional
        If `ndigits` is given the cycles will be rounded to the given number of digits before counting.

    Returns
    -------
    list
        Sorted list of tuples of rebinned cycle magnitude versus count.

    Notes
    -----
    Cycles are gathered into a specific bin if ``lower_bound < cycle_magnitude <= upper_bound``. The bin magnitude
    (after rebinning) is represented by its mid-point value ``0.5 * (lower_bound + upper_bound)``.

    You should not rebin from bins with relatively low resolution to higher resolution because that may cause
    shifting a significant amount of the cycle count to bins with a different midpoint (magnitude). This will
    not raise an error but you should avoid it.

    See Also
    --------
    rainflow.rebin_cycles_nbins
    rainflow.rebin_cycles_binwidth

    """
    # check that the specified bins cover the maximum magnitude
    max_magnitude = max([m for m, _ in cycles])
    max_bins = max(bins)

    if max_magnitude > max_bins:
        raise ValueError("The maximum magnitude '%5.3g' exceeds the upper bound of the specified bins '%5.3g'."
                         "Increase the upper boundary of the specified bins." % (max_magnitude, max_bins))

    cycles_rebinned = defaultdict(float)
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        center = 0.5 * lo + 0.5 * hi
        if ndigits is not None:
            center = round(center, ndigits)
        cycles_rebinned[center] = np.sum([c for m, c in cycles if (m > lo) and (m <= hi)])

    return sorted(cycles_rebinned.items())

