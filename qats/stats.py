# encoding: utf8
"""
Basic functions for statistical inference and signal processing.
"""
import numpy as np


def autocorrelation(series):
    """
    Estimation of the auto-correlation coefficients of *series*

    Parameters
    ----------
    series : array_like
        data series

    Returns
    -------
    list
        arrays of autocorrelation coefficients for the entire *series* for lags in the range [dt, dt, duration]

    Notes
    -----
    I took a part of code from pandas autocorrelation_plot() function. I checked the answers and the values are
    matching exactly.
    The auto-correlation coefficients can be plotted against the time vector associated with series.

    References
    ----------
    1. Wikipedia, http://en.wikipedia.org/wiki/Autocorrelation

    """
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        """
        Calculation of autocorrelation coefficients for lag *h*

        Parameters
        ----------
        h : float
            lag

        Returns
        -------
        array
           autocorrelation coefficients for lag *h*

        """
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)

    x = np.arange(n)  # Avoiding lag 0 calculation
    acf = map(r, x)
    return acf


def average_frequency(t, x, up=True):
    """
    Average frequency of mean level crossings.

    Parameters
    ----------
    t : array_like
        Time (seconds).
    x : array_like
        Signal.
    up : bool, optional
        - True: Period based on average time between up-crossings
        - False: Period based on average time between down-crossings


    Returns
    -------
    float
        Average frequency of mean level crossings (Hz)

    """
    # remove mean value from time series
    x_ = x - np.mean(x)

    if up:
        crossings = 1 * (x_ > 0.)
        indicator = 1
    else:
        crossings = 1 * (x_ < 0.)
        indicator = -1

    crossings = np.diff(crossings)  # array with value=1 at position of each up-crossing and -1 at each down-crossing
    crossings[crossings != indicator] = 0   # remove crossings with opposite direction
    i = np.where(crossings == indicator)[0] + 1  # indices for crossings
    d = (t[i[-1]] - t[i[0]]) / (np.abs(np.sum(crossings)) - 1)  # duration between first and last crossing
    return 1./d


def empirical_cdf(n, kind='mean'):
    """
    Empirical cumulative distribution function given a sample size.

    Parameters
    ----------
    n : int
        sample size
    kind : str, optional
        - 'mean':  ``i/(n+1)`` (aka. Weibull method)
        - 'median' ``(i-0.3)/(n+0.4)``
        - 'symmetrical': ``(i-0.5)/n``
        - 'beard': ``(i - 0.31)/(n + 0.38)`` (Jenkinson's/Beard's method)
        - 'gringorten': ``(i - 0.44)/(n + 0.12)`` (Gringorten's method)

    Returns
    -------
    array
        Empirical cumulative distribution function

    Notes
    -----
    Gumbel recommended the following quantile formulation ``Pi = i/(n+1)``.
    This formulation produces a symmetrical CDF in the sense that the same
    plotting positions will result from the data regardless of
    whether they are assembled in ascending or descending order.

    Jenkinson's/Beard's method is based on the "idea that a natural
    estimate for the plotting position is the median of its probability
    density distribution".

    A more sophisticated formulation ``Pi = (i-0.3)/(n+0.4)`` approximates the
    median of the distribution free estimate of the sample variate to about
    0.1% and, even for small values of `n`, produces parameter estimations
    comparable to the result obtained by maximum likelihood estimations
    (Bury, 1999, p43)

    The probability corresponding to the unbiased plotting position can be
    approximated by the Gringorten formula in the case of type 1 Extreme
    value distribution.

    References
    ----------
    1. `Plotting positions <http://en.wikipedia.org/wiki/Q%E2%80%93Q_plot>`_, About plotting positions
    """

    n = float(n)
    i = np.arange(n) + 1
    if kind == 'mean':
        f = i / (n + 1.)
    elif kind == 'median':
        f = (i - 0.3) / (n + 0.4)
    elif kind == 'symmetrical':
        f = (i - 0.5) / n
    elif kind == 'beard':
        f = (i - 0.31) / (n + 0.38)
    elif kind == 'gringorten':
        f = (i - 0.44) / (n + 0.12)
    else:
        raise ValueError("Distribution type must be either 'mean','median','symmetrical','beard','gringorten'")

    return f


def find_maxima(x, local=False, threshold=None, up=True, retind=False):
    """
    Return sorted maxima

    Parameters
    ----------
    x : array
        Signal.
    local : bool, optional
        If True, local maxima are also included (see notes below). Default is to include only global maxima.
    threshold : float, optional
        Include only maxima larger than specified treshold. Default is mean value of signal.
    up : bool, optional
        If True (default), identify maxima between up-crossings. If False, identify maxima between down-crossings.
    retind : bool, optional
        If True, return (maxima, indices), where indices is positions of maxima in input signal array.

    Returns
    -------
    array
        Signal maxima, sorted from smallest to largest.
    array
        Only returned if `retind` is True.
        Indices of signal maxima.

    Notes
    -----
    By default only 'global' maxima are considered, i.e. the largest maximum between each mean-level up-crossing.
    If ``local=True``, local maxima are also included (first derivative is zero, second derivative is negative).

    Examples
    --------
    Extract global maxima from time series signal `x`:

    >>> maxima = find_maxima(x)

    Extract global maxima and corresponding indices:

    >>> maxima, indices = find_maxima(x, retind=True)

    Assuming `time` is the time vector (numpy array) for signal `x`, the following example will provide an array of
    time instants associated with the maxima sample:

    >>> maxima, indices = find_maxima(x, retind=True)
    >>> time_maxima = time[indices]

    """
    # remove mean value from time series to identify crossings
    x_ = x - np.mean(x)

    if up:
        crossings = 1 * (x_ > 0.)
        indicator = 1
    else:
        crossings = 1 * (x_ < 0.)
        indicator = -1

    crossings = np.diff(crossings)          # array with 1 at position of each up-crossing and -1 at each down-crossing
    crossings[crossings != indicator] = 0   # remove crossings with opposite direction
    n_crossings = np.sum(crossings)         # number of crossings

    # no global or local maxima if the signal crosses mean only once
    if n_crossings < 2:
        if retind:
            return None, None
        else:
            return None

    crossing_indices = np.where(crossings == indicator)[0] + 1  # indices for crossings

    # add first and last index in time series to avoid loosing important peaks, particularly important for problems
    # with low-frequent oscillations
    if crossing_indices[-1] < (x_.size - 1):
        crossing_indices = np.append(crossing_indices, [x_.size - 1])

    # find maxima
    if not local:
        # global
        maxima = np.zeros(n_crossings)
        maxima_indices = np.zeros(n_crossings, dtype=int)

        # loop to find max. between each up-crossing:
        for j, start in enumerate(crossing_indices[:-1]):
            stop = crossing_indices[j + 1]
            maxima[j] = x[start:stop].max()
            maxima_indices[j] = start + np.argmax(x[start:stop])

    else:
        # local
        ds = 1 * (np.diff(x) < 0)     # zero while ascending (positive derivative) and 1 while descending
        ds = np.append(ds, [0])       # lost data points when differentiating, close cycles by adding 0 at end
        d2s = np.diff(ds)             # equal to +/-1 at each turning point, +1 indicates maxima
        d2s = np.insert(d2s, 0, [0])  # lost data points when differentiating, close cycles by adding 0 at start

        maxima_indices = np.where(d2s == 1)[0]  # unpack tuple returned from np.where
        maxima = x[maxima_indices]

    # discard maxima lower than specified threshold
    if threshold is not None:
        if isinstance(threshold, float):
            above_threshold = (maxima >= threshold)
            maxima = maxima[above_threshold]
            maxima_indices = maxima_indices[above_threshold]
        else:
            raise TypeError("Specified threshold is wrong type, should be float: %s" % type(threshold))
    else:
        pass

    # sort ascending
    ascending = np.argsort(maxima)
    maxima = maxima[ascending]
    maxima_indices = maxima_indices[ascending]

    if retind:
        return maxima, maxima_indices
    else:
        return maxima

