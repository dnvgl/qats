# encoding: utf8
"""
Basic functions for statistical inference.
"""
import numpy as np


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

