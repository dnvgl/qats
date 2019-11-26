#!/usr/bin/env python
# coding: utf-8
"""
Rainflow cycle counting algorithm according to ASTM E1049-85 (2011), section 5.4.4.
"""
from collections import deque, defaultdict
from itertools import chain
import numpy as np

# TODO: Evaluate from-to counting which stores the "orientation" of each cycle. Enables reconstruction of a time history


def reversals(series, endpoints=False):
    """
    A generator function which iterates over the reversals in the iterable *series*.

    Parameters
    ----------
    series : array_like
        data series
    endpoints : bool, optional
        If True, first and last points in `series` are included. Default is False.
        Note that in general, inclusion of end points is only relevant if the series passed is already an array of
        reversal points.

    Returns
    -------
    generator
        yields floats

    Notes
    -----
    Reversals are the points at which the first derivative on the series changes sign. The generator never yields
    the first and the last points in the series, unless `endpoints` is set to True (in which case they are always
    included).

    """
    series = iter(series)

    x_last, x = next(series), next(series)
    d_last = (x - x_last)

    if endpoints is True:
        yield x_last

    for x_next in series:
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0:
            yield x
        x_last, x = x, x_next
        d_last = d_next

    if endpoints is True:
        yield x


def cycles(series, endpoints=False):
    """
    Find full cycles and half-cycles range and mean value from *series* and count the number of occurrences using the
    Rainflow algorithm.

    Parameters
    ----------
    series : array_like
        data series
    endpoints : bool, optional
        If True, first and last points in `series` are included as cycle start/end points. This is convenient if the
        series given is already an array of reversal points. Default is False.

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
    Extract start and end points for all full and half cycles.

    >>> from qats.fatigue.rainflow import cycles
    >>> series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    >>> full, half = cycles(series)
    >>> full
    [(4, 1.0)]
    >>> half
    [(3, -0.5), (4, -1.0), (8, 1.0), (6, 1.0), (8, 0.0), (9, 0.5)]

    """
    points = deque()
    full, half = list(), list()

    for r in reversals(series, endpoints=endpoints):
        points.append(r)
        while len(points) >= 3:
            # Form ranges X and Y from the three most recent points
            x = abs(points[-2] - points[-1])
            y = abs(points[-3] - points[-2])        # cycle range
            m = 0.5 * (points[-2] + points[-3])     # cycle mean
            # fromto = (points[-2], points[-3])     # cycle start and end points (not currently in use)

            if x < y:
                # Read the next point
                break
            elif len(points) == 3:
                # Y contains the starting point
                # Count Y as half cycle and discard the first point
                half.append((y, m))
                points.popleft()
            else:
                # Count Y as one cycle and discard the peak and the valley of Y
                full.append((y, m))
                last = points.pop()
                points.pop()
                points.pop()
                points.append(last)
    else:
        # Count the remaining ranges as half cycles
        while len(points) > 1:
            half.append((abs(points[-2] - points[-1]), 0.5 * (points[-1] + points[-2])))
            points.pop()

    return full, half


def count_cycles(series, endpoints=False):
    """
    Count number of occurrences of cycle range and mean combinations.

    Parameters
    ----------
    series : array_like
        Data series.
    endpoints : bool, optional
        If True, first and last points in `series` are included as cycle start/end points. This is convenient if the
        series given is already an array of reversal points, but should in general not be used otherwise.
        Default is False.

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) where `n` is number of cycle ranges. Each row consists of three values; the cycle range,
        mean and count. Counts are either 1.0 (for full cycles) or 0.5 (for half cycles).
        The array is sorted by increasing cycle range.

    Notes
    -----
    The cycles are extracted from the iterable series using the `cycles()` function.

    Since half cycles are counted as 0.5, the returned counts are not necessarily whole numbers.

    Examples
    --------
    Extract raw cycle range, mean and count:

    >>> from qats.fatigue.rainflow import count_cycles
    >>> series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    >>> count_cycles(series)
    array([[ 3. , -0.5,  0.5],
           [ 4. , -1. ,  0.5],
           [ 4. ,  1. ,  1. ],
           [ 6. ,  1. ,  0.5],
           [ 8. ,  0. ,  0.5],
           [ 8. ,  1. ,  0.5],
           [ 9. ,  0.5,  0.5]])

    The array may be unpacked into separate arrays of cycle range, mean and count as:

    >>> r, m, c = count_cycles(series).T

    The following will also work, but is slower than the example above:

    >>> r, m, c = zip(*count_cycles(series))

    See Also
    --------
    rainflow.reversals, rainflow.cycles

    """
    full, half = cycles(series, endpoints=endpoints)

    # number of cycles (full, half, total)
    nf = len(full)
    nh = len(half)
    n = nf + nh

    # initiate and populate array with cycle counts
    counts = np.zeros((n, 3))
    counts[:, :2] = full + half  # full and half are lists
    counts[:nf, 2] = 1.0  # full cycles count 1.0
    counts[nf:, 2] = 0.5  # half cycles count 0.5

    # sort by increasing range, then mean
    counts = _sort_cycles(counts, copy=False)

    return counts


def mesh(cycles, nr=100, nm=100):
    """
    Mesh range-mean distribution.

    Parameters
    ----------
    cycles : list
        Cycle ranges, mean values and count.
    nr : int, optional
        Number of equidistant bins for cycle ranges.
    nm : int, optional
        Number of equidistant bins for cycle means.

    Returns
    -------
    array
        Cycle ranges.
    array
        Cycle mean value.
    array
        Cycle count.

    Examples
    --------
    Rebin the cycle distribution onto a 10x10 mesh suitable for surface plotting.

    >>> from qats.fatigue.rainflow import count_cycles
    >>> series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    >>> count_cycles(series)
    >>> r, m, c = mesh(cycles, nr=10, nm=10)

    """
    # create mesh
    maxrange = max([r for r, _, _ in cycles])
    maxmean = max([m for _, m, _ in cycles])
    minmean = min([m for _, m, _ in cycles])
    ri = np.linspace(0., 1.1 * maxrange, nr)
    mj = np.linspace(0.9 * minmean, 1.1 * maxmean, nm)
    rij, mij = np.meshgrid(ri, mj)
    cij = np.zeros(np.shape(mij))

    # rebin distribution
    for i in range(nr - 1):
        for j in range(nm - 1):
            for r, m, c in cycles:
                if (ri[i] <= r < ri[i + 1]) and (mj[j] <= m < mj[j + 1]):
                    cij[i, j] += c

    print(f"Number of cycles {sum([c for _, _, c in cycles])} / {cij.sum()}.")

    return rij, mij, cij


def rebin(cycles, binby='range', n=None, w=None):
    """
    Rebin cycles in specified bins by range or mean value

    Parameters
    ----------
    cycles : list
        Cycle ranges, mean values and count.
    bins : list
        Bins specified by bin boundaries e.g. first bin ranges from the first to the second value in `bins`, the next
        bin from the second value to the third, and so on.
    binby : str, optional
        'range' - Rebin by cycle range (default)
        'mean'  - Rebin by cycle mean
    n : int, optional
        Number of equidistant bins for cycle ranges and cycle mean values.
    w : float, optional
        Width of equidistant bins for cycle ranges and cycle mean values. Overrules `n` is specified.

    Returns
    -------
    list
        Rebinned cycles in ascending order (cycle ranges or mean values).

    Notes
    -----
    Cycles are gathered into a specific bin if the primary measure (range of mean) is within that bin's boundaries. The
    primary measure is represented by the bin mid point. The secondary measure (range or mean) is represented by its
    weighted average (weighted by number of occurrences) in the bin.

    Note that rebinning may shift a significant amount of the cycles to bins which midpoint differs notably from
    the original cycle value (range or mean). We advice to rebin for plotting but when calculating e.g. fatigue damage
    we advice to use the raw unbinned values.

    See Also
    --------
    rainflow.create_bins

    """

    def create_bins(start, stop, n=None, w=None):
        """
        Create equidistant bins.

        Parameters
        ----------
        start : float
            Lowest bin value
        stop : float
            Largest bin value
        n : int, optional
            Number of equidistant bins
        w : float, optional
            Width of equidistant bins. Overrules `n` is specified.

        Returns
        -------
        list
            Bins specified by bin boundaries e.g. first bin ranges from the first to the second value in `bins`, the next
            bin from the second value to the third, and so on.

        """
        if (not n) and (not w):
            raise ValueError('Specify either the number of bins `n` or the bin width `w`.')

        if w is not None:
            # crate bins with specified w
            return np.arange(start, stop + w, w)

        else:
            # create specified number of bins
            return np.linspace(start, stop, n + 1)

    if binby not in ('range', 'mean'):
        raise ValueError(f"Unable to bin by '{binby}'. Must be either 'range' or 'mean'.")

    # unpack
    ranges, means, counts = zip(*cycles)

    # rebin
    if binby == 'range':
        bins = create_bins(0., max(ranges), n=n, w=w)
        bin_primary = [0.5 * (lo + hi) for lo, hi in zip(bins[:-1], bins[1:])]  # bin mid points
        bin_secondary = list()
        bin_n = list()
        for lo, hi in zip(bins[:-1], bins[1:]):
            # number of cycles which range is within bin boundaries
            n = sum([c for r, c in zip(ranges, counts) if (r > lo) and (r <= hi)])
            bin_n.append(n)
            # weighted average of cycle means
            bin_secondary.append(sum([c * m for r, m, c in zip(ranges, means, counts) if (r > lo) and (r <= hi)]) / n if n > 0. else np.nan)

        return list(zip(bin_primary, bin_secondary, bin_n))
    else:
        bins = create_bins(min(means), max(means), n=n, w=w)
        bin_primary = [0.5 * (lo + hi) for lo, hi in zip(bins[:-1], bins[1:])]  # bin mid points
        bin_secondary = list()
        bin_n = list()
        for lo, hi in zip(bins[:-1], bins[1:]):
            # number of cycles which mean is within bin boundaries
            n = sum([c for m, c in zip(means, counts) if (m > lo) and (m <= hi)])
            bin_n.append(n)
            # weighted average of cycle ranges
            bin_secondary.append(sum([c * r for r, m, c in zip(ranges, means, counts) if (m > lo) and (m <= hi)]) / n if n > 0. else np.nan)

        return list(zip(bin_secondary, bin_primary, bin_n))


def _toarray(longlist) -> np.ndarray:
    """
    Convert (potentially long) list to numpy array.

    Parameters
    ----------
    longlist

    Returns
    -------
    np.ndarray

    Notes
    -----
    Code is taken from:
    https://stackoverflow.com/questions/17973507/why-is-converting-a-long-2d-list-to-numpy-array-so-slow

    Using ipython %timeit command, this function performed around 0.7s per loop for a list of length ~4500000
    (with 3 items in each row).

    The following alternatives performed at around 2.1s per loop:
    >>> # alternative 1
    >>> arr = np.array(longlist)
    >>> # alternative 2
    >>> arr = np.zeros((len(longlist), len(longlist[0])))
    >>> arr[:] = longlist

    ... and this alternative performed at around 2.5s per loop:
    >>> arr = np.array([[for row[i] in longlist] for i in range(len(longlist[0]))])

    """
    if isinstance(longlist, np.ndarray):
        # return quickly if input is already a numpy array
        return longlist
    flat = np.fromiter(chain.from_iterable(longlist), np.array(longlist[0][0]).dtype, -1)
    return flat.reshape((len(longlist), -1))


def _sort_cycles(arr, copy=False):
    """
    Sort cycles array wrt. 1st column then 2nd column.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (n, 3).
    copy : bool, optional
        If True, return a copy instead of sorting the array inplace.

    Returns
    -------
    np.ndarray
    """
    if copy:
        # make a copy
        arr = np.array(arr)
    # using numpy.argsort(), ref. https://stackoverflow.com/a/38194077
    arr = arr[arr[:, 1].argsort()]  # first sort doesn't need to be stable
    arr = arr[arr[:, 0].argsort(kind='mergesort')]
    return arr

