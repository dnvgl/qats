#!/usr/bin/env python
# coding: utf-8
"""
Rainflow cycle counting algorithm according to ASTM E1049-85 (2011), section 5.4.4.
"""
from collections import deque, defaultdict
import numpy as np

# TODO: plot functions for 2D (range, count) and 3D (range, mean, count)


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

    Examples
    --------
    Extract start and end points for all full and half cycles.
    >>> from qats.rainflow import cycles_fromto
    >>> series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    >>> full, half = cycles_fromto(series)
    >>> full
    [(3, -1)]
    >>> half
    [(1, -2), (-3, 1), (5, -3), (-2, 4), (4, -4), (-4, 5)]

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
    [(4, 1.0)]
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
    Returns a sorted list containing triplets of cycle range, mean and count.

    Parameters
    ----------
    series : array_like
        Data series.
    ndigits : int, optional
        If `ndigits` is given the cycles will be rounded to the given number of digits before counting.
    nbins : int, optional
        If `nbins` is given the cycle count per cycle magnitude is rebinned to `nbins` bins.
    binwidth : float, optional
        If `binwidth` is given the cycle count per cycle magnitude is rebinned to bins with a width of `binwidth`.

    Returns
    -------
    list
        List of tuples with cycle range, mean and count, sorted by increasing range::

            [(range1, mean1, count1), (range2, mean2, count2), ...]


    Notes
    -----
    The cycles are extracted from the iterable series using the `cycles_rangemean` function.

    Half cycles are counted as 0.5, so the returned counts may not be whole numbers.

    Rebinning is not applied if specified `nbins` is larger than the original number of bins. `nbins` override
    `binwidth` if both are specified.

    Examples
    --------
    Extract raw cycle range, mean and count:

    >>> from qats.rainflow import count_cycles
    >>> series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    >>> count_cycles(series)
    [(3, -0.5, 0.5), (4, -1.0, 0.5), (4, 1.0, 1.0), (6, 1.0, 0.5), (8, 0.0, 0.5), (8, 1.0, 0.5), (9, 0.5, 0.5)]

    Extract cycle range, mean and count and downsample to 3 bins:

    >>> count_cycles(series, nbins=3)
    [(1.5, -0.5, 0.5), (4.5, 0.5, 2.0), (7.5, 0.5, 1.5)]

    The sorted list of cycles may be unpacked into separate lists of cycle range, mean and count as:

    >>> cycles = count_cycles(series)
    >>> r, m, c = zip(*cycles)

    See Also
    --------
    rainflow.rebin_cycles

    """
    full, half = cycles_rangemean(series)

    # Round the cycle range and mean value if requested
    if ndigits is not None:
        full = ((round(_[0], ndigits), round(_[1], ndigits)) for _ in full)
        half = ((round(_[0], ndigits), round(_[1], ndigits)) for _ in half)

    # Count cycles
    cycles = defaultdict(float)
    for x in full:
        cycles[x] += 1.0
    for x in half:
        cycles[x] += 0.5

    # sorted triplets (tuples) of cycle range, mean and count in list
    cycles = sorted([rm + tuple([c]) for rm, c in cycles.items()])

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
        Sorted list of tuples of cycle range, mean and count.
    binwidth : float, optional
        Bin width in rebinned distribution.
    ndigits : int, optional
        If `ndigits` is given the cycles will be rounded to the given number of digits before counting.

    Returns
    -------
    list
        Sorted list of tuples of rebinned cycle range, mean and count.

    Notes
    -----
    Cycles are gathered into a specific bin if ``lower_bound < cycle_range <= upper_bound``. The rebinned distribution
    is discretized by the bin mid-points ``0.5 * (lower_bound + upper_bound)``.

    """
    # extract maximum range
    max_range = max([m for m, _, _ in cycles])

    # create bins ranging from 0 to maximum range, stepping with specified bin width
    bins = np.arange(0., max_range + binwidth, binwidth)

    # rebin
    rebinned_cycles = rebin_cycles(cycles, bins, ndigits=ndigits)

    return rebinned_cycles


def rebin_cycles_nbins(cycles, nbins, ndigits=None):
    """
    Rebin cycle distribution to a specified number of equidistant bins

    Parameters
    ----------
    cycles : list
        Sorted list of tuples of cycle range, mean and count.
    nbins : int, optional
        Number of equidistant bins in rebinned distribution.
    ndigits : int, optional
        If `ndigits` is given the cycles will be rounded to the given number of digits before counting.

    Returns
    -------
    list
        sorted list of tuples of rebinned cycle range, mean and count

    Notes
    -----
    Cycles are gathered into a specific bin if ``lower_bound < cycle_range <= upper_bound``. The rebinned distribution
    is discretized by the bin mid-points ``0.5 * (lower_bound + upper_bound)``.

    """
    # extract maximum magnitude
    max_range = max([m for m, _, _ in cycles])

    # create specified number of bins ranging from 0 to maximum magnitude
    bins = np.linspace(0., max_range, nbins + 1)

    # rebin
    rebinned_cycles = rebin_cycles(cycles, bins, ndigits=ndigits)

    return rebinned_cycles


def rebin_cycles(cycles, bins, ndigits=None):
    """
    Rebin cycle distribution to specified bins

    Parameters
    ----------
    cycles : list
        Sorted list of tuples of cycle range, mean and count.
    bins : array_like
        Bins specified by bin boundaries e.g. first bin ranges from the first to the second value in `bins`, the next
        bin from the second value to the third, and so on.
    ndigits : int, optional
        If `ndigits` is given the cycles will be rounded to the given number of digits before counting.

    Returns
    -------
    list
        Sorted list of tuples of rebinned cycle range, mean and count.

    Notes
    -----
    Cycles are gathered into a specific bin if ``lower_bound < cycle_range <= upper_bound``. The rebinned distribution
    is discretized by cycle range as the bin mid-points ``0.5 * (lower_bound + upper_bound)``. The rebinned cycle mean
    value is the weighted average of all cycle means in the bin.

    Note that rebinning may shift a significant amount of the cycle count to bins with a different midpoint
    (cycle range).

    See Also
    --------
    rainflow.rebin_cycles_nbins
    rainflow.rebin_cycles_binwidth

    """
    # check that the specified bins cover the maximum range
    max_range = max([m for m, _, _ in cycles])
    max_bins = max(bins)

    if max_range > max_bins:
        raise ValueError(f"The maximum range {max_range} exceeds the upper bound of the specified bins {max_bins}."
                         "Increase the upper boundary of the specified bins.")

    rebinned_cycles = list()
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]

        # count cycles in bin
        bin_n = np.sum([c for r, _, c in cycles if (r > lo) and (r <= hi)])

        # cycle range as bin midpoint
        bin_range = round(0.5 * (lo + hi), ndigits) if ndigits is not None else 0.5 * (lo + hi)

        # cycle mean as weighted average of all cycle mean values in bin (defaults to nan)
        bin_mean = sum([c * m for r, m, c in cycles if (r > lo) and (r <= hi)]) / bin_n if bin_n > 0. else np.nan

        rebinned_cycles.append(tuple([bin_range, bin_mean, bin_n]))

    return sorted(rebinned_cycles)

