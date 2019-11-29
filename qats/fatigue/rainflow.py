#!/usr/bin/env python
# coding: utf-8
"""
Rainflow cycle counting algorithm according to ASTM E1049-85 (2011), section 5.4.4.
"""
from collections import deque
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
    Find full cycles and half-cycles range and mean value from `series` and count the number of occurrences using the
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
    full : list
        full cycles (range and mean).
    half : list
        half cycles (range and mean).

    Notes
    -----
    The cycles are extracted from the iterable `series` according to section 5.4.4 in ASTM E1049 (2011).

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
    cycles : np.ndarray
        Array of shape (n, 3) where `n` is number of cycle ranges. Each row consists of three values; the cycle
        range, mean and count. Counts are either 1.0 (for full cycles) or 0.5 (for half cycles).
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
    reversals, cycles

    """
    full, half = cycles(series, endpoints=endpoints)

    # number of cycles (full, half, total)
    nf = len(full)
    nh = len(half)
    n = nf + nh

    # initiate and populate array with cycle counts
    cycles_ = np.zeros((n, 3))
    cycles_[:, :2] = full + half  # full and half are lists
    cycles_[:nf, 2] = 1.0  # full cycles count 1.0
    cycles_[nf:, 2] = 0.5  # half cycles count 0.5

    # sort by increasing range, then mean
    cycles_ = _sort_cycles(cycles_, copy=False)

    return cycles_


def mesh(cycles, nr=100, nm=100):
    """
    Mesh range-mean distribution.

    Parameters
    ----------
    cycles : array_like
        Array of shape (n, 3). Columns should be: cycle range, cycle mean, count. See description of output
        from :func:`count_cycles`.
    nr : int, optional
        Number of equidistant bins for cycle ranges.
    nm : int, optional
        Number of equidistant bins for cycle means.

    Returns
    -------
    rangemesh : np.ndarray
        Cycle ranges meshgrid, shape: `(nm, nr)`.
    meanmesh : np.ndarray
        Cycle mean value meshgrid, shape: `(nm, nr)`.
    countmesh : np.ndarray
        Cycle count 2D histogram, shape: `(nm, nr)`.

    See Also
    --------
    numpy.histogram2d
    numpy.meshgrid

    Notes
    -----
    .. versionadded :: 4.7.0

    This function has been re-written for version 4.7.0. For versions <= 4.6.1, the mesh established was not correct.

    Shape of the returned arrays is consistent with :func:`numpy.meshgrid`: ``(nm, nr)``, i.e. number of rows is `nm` and
    number of columns is `nr`. This means that the array is transposed compared the output from
    :func:`numpy.histogram2d`, which is a 2D histogram of shape ``(nr, nm)``.

    The cycle count mesh is consistent with the cycles returned from :func:`rebin`, such that the sum of the cycle count
    mesh along each of its axes is equals the counts for cycles rebinned by 'mean' or 'range', respectively:

    >>> cycles = count_cycles(series)
    >>> _, _, cmesh = mesh(cycles, nr=200, nm=100)
    >>> cycles_rebinned_range = rebin(cycles, binby='range', n=200)
    >>> # sum of `cmesh` along cycle mean axis (constant cycle range) vs. counts from rebinned cycles
    >>> (cmesh.sum(axis=0) == cycles_rebinned_range[:, 2]).all()
    True
    >>> cmesh.sum(axis=0).shape
    (200,)
    >>> cmesh.sum(axis=1).shape
    (100,)


    Examples
    --------
    Rebin the cycle distribution onto a 10 x 15 mesh:

    >>> from qats.fatigue.rainflow import count_cycles, mesh
    >>> # (obtain some series from e.g. a simulation)
    >>> count_cycles(series)
    >>> rangemesh, meanmesh, countmesh = mesh(cycles, nr=15, nm=10)
    >>> countmesh.shape  # (same shape for all three arrays)
    (10, 15)

    The mesh returned is suitable for plotting with `matplotlib` 3D plots, for instance:

    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.plot_surface(rangemesh, meanmesh, countmesh, cmap=mpl.cm.coolwarm)
    >>> ax.set_xlabel('Cycle range')
    >>> ax.set_ylabel('Cycle mean')
    >>> ax.set_zlabel('Cycle count')
    >>> plt.show()

    """
    # unpack
    cycles = _toarray(cycles)  # ensure array, not list
    ranges, means, counts = cycles.T

    # create mesh
    maxrange = ranges.max()
    maxmean = means.max()
    minmean = means.min()

    # xyrange = ([0., maxrange], [minmean, maxmean])
    xyrange = ([0., maxrange], [minmean, maxmean])
    hist2d, r_edges, m_edges = np.histogram2d(ranges, means, bins=[nr, nm], range=xyrange, weights=counts)

    # 2D histogram from np.histogram2d must be transposed for consistency with np.meshgrid
    cmesh = hist2d.T

    # arrays of bin mid points (sizes `nr` and `rm`)
    rbins = 0.5 * (r_edges[:-1] + r_edges[1:])
    mbins = 0.5 * (m_edges[:-1] + m_edges[1:])

    # mesh grids
    rmesh, mmesh = np.meshgrid(rbins, mbins)

    return rmesh, mmesh, cmesh


def rebin(cycles, binby='range', n=None, w=None):
    """
    Rebin cycles in specified bins by range or mean value

    Parameters
    ----------
    cycles : array_like
        Array of shape (n, 3). Columns should be: cycle range, cycle mean, count. See description of output
        from `rainflow.count_cycles()`.
    binby : str, optional
        'range' - Rebin by cycle range (default)
        'mean'  - Rebin by cycle mean
    n : int, optional
        Number of equidistant bins for cycle ranges and cycle mean values.
    w : float, optional
        Width of equidistant bins for cycle ranges and cycle mean values. Overrides `n` if specified.

    Returns
    -------
    cycles : np.ndarray
        Array with shape `(nbins, 3)`, where `nbins` is number of bins. Columns are: cycle range, mean, count.

    Notes
    -----
    Cycles are gathered into a specific bin if the primary measure (range or mean) is within that bin's boundaries. The
    primary measure is represented by the bin mid point. The secondary measure (range or mean) is represented by its
    weighted average (weighted by number of occurrences) in the bin.

    Note that rebinning may shift a significant amount of the cycles to bins which midpoint differs notably from
    the original cycle value (range or mean). We advice to rebin for plotting, but when calculating e.g. fatigue damage
    we advice to use the raw unbinned values.

    .. versionadded :: 4.7.0

    If a cycle is at the edge between two bins, it is placed in the 'highest' bin. E.g. If bin edges are [0, 1, 2, 3],
    a cycle with value 1 will be counted in the bin [1, 2) - see documentation for np.histogram for details. This
    behaviour is contrary to version <= 4.6.1, for which the value 1 was counted in bin (0, 1]. The new way is slightly
    more conservative, since cycle ranges (or means) that end up on an edge is shifted towards a higher value.
    However; in most cases, there will be no difference at all since very few cycles coincide with bin edges.

    Examples
    --------
    Rebinning by range with bin width 1.0. Number of bins is then determined from the max cycle range.
    The second column (which is here the secondary measure, since we are binning by range) is the weighted average
    of the mean value for the cycles that fall within each bin:

    >>> from qats.fatigue.rainflow import count_cycles
    >>> series = [0, -2, 1, -3, 1, -1, 3, -2, 2, -2, 0, 1, -2, 3, -1, 2, -3, 0, -1, 0]
    >>> cycles = count_cycles(series)
    >>> rebin(cycles, binby='range', w=2.0)
    array([[ 1.  , -0.5 ,  0.5 ],
           [ 3.  , -0.25,  4.  ],
           [ 5.  ,  0.  ,  3.5 ]])

    See Also
    --------
    numpy.histogram
    count_cycles

    """
    if binby not in ('range', 'mean'):
        raise ValueError(f"Invalid option for `binby`: '{binby}'. Must be either 'range' or 'mean'")

    # unpack
    cycles = _toarray(cycles)  # ensure array, not list
    ranges, means, counts = cycles.T

    # rebin
    if binby == 'range':
        # establish bin edges
        bins = _create_bins(0., ranges.max(), n=n, w=w)
        # nbins = bins.size - 1

        # establish bin mid points (size -1 compared to array with bin edges)
        bin_primary = 0.5 * (bins[:-1] + bins[1:])

        # calculate sum of counts for each bin using np.histogram(). `range` has no effect since bins are given
        # explicitly, and is therefore passed as None
        bin_n, _ = np.histogram(ranges, bins=bins, range=None, weights=counts)

        # weighted average of cycle means is established in a similar way, however; dividing by total count needs to be
        # done afterwards. For bins with no count, the weighted average is set to nan.
        weights = counts * means
        bin_secondary, _ = np.histogram(ranges, bins=bins, range=None, weights=weights)
        bin_secondary[bin_n > 0] *= 1. / bin_n[bin_n > 0]
        bin_secondary[bin_n == 0] = np.nan

        return np.array([bin_primary, bin_secondary, bin_n]).T

    else:  # i.e. binby == 'mean'
        # create bins
        bins = _create_bins(means.min(), means.max(), n=n, w=w)
        # nbins = bins.size - 1

        # establish bin mid points (size -1 compared to array with bin edges)
        bin_primary = 0.5 * (bins[:-1] + bins[1:])

        # calculate sum of counts for each bin using np.histogram(). `range` has no effect since bins are given
        # explicitly, and is therefore passed as None
        bin_n, _ = np.histogram(means, bins=bins, range=None, weights=counts)

        # weighted average of cycle range is established in a similar way, however; dividing by total count needs to be
        # done afterwards. For bins with no count, the weighted average is set to nan.
        weights = counts * ranges
        bin_secondary, _ = np.histogram(means, bins=bins, range=None, weights=weights)
        bin_secondary[bin_n > 0] *= 1. / bin_n[bin_n > 0]
        bin_secondary[bin_n == 0] = np.nan

        return np.array([bin_secondary, bin_primary, bin_n]).T


def _create_bins(start, stop, n=None, w=None):
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
    np.ndarray
        Bins specified by bin boundaries e.g. first bin ranges from the first to the second value in `bins`,
        the next bin from the second value to the third, and so on.

    """
    if (not n) and (not w):
        raise ValueError('Specify either the number of bins `n` or the bin width `w`.')

    if w is not None:
        # crate bins with specified w
        return np.arange(start, stop + w, w)
    else:
        # create specified number of bins
        return np.linspace(start, stop, n + 1)


def _toarray(longlist) -> np.ndarray:
    """
    Convert (potentially long) list to 2D numpy array.

    Parameters
    ----------
    longlist : list or np.ndarray
        List that may be broadcast into 2D numpy array.

    Returns
    -------
    np.ndarray

    Notes
    -----
    Code is based on:
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
        2D array of shape (n, 3).
    copy : bool, optional
        If True, return a copy instead of sorting the array inplace.
        Note that this increases time consumption, in particular for large arrays.

    Returns
    -------
    np.ndarray

    Raises
    ------
    IndexError
        If input array is 1D or if second dimension is less than 2.
    """
    if copy:
        # make a copy
        arr = np.array(arr)
    # using numpy.argsort(), ref. https://stackoverflow.com/a/38194077
    arr = arr[arr[:, 1].argsort()]  # first sort doesn't need to be stable
    arr = arr[arr[:, 0].argsort(kind='mergesort')]
    return arr

