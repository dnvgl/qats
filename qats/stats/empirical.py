# encoding: utf8
"""
Basic functions for statistical inference.
"""

import numbers

import numpy as np

from .._validation import QatsTypeError, QatsValueError

# plotting positions of empirical_cdf(): kind -> formula
_PLOTTING_POSITIONS = {
    "mean": "i/(n+1)",
    "median": "(i-0.3)/(n+0.4)",
    "symmetrical": "(i-0.5)/n",
    "beard": "(i-0.31)/(n+0.38)",
    "gringorten": "(i-0.44)/(n+0.12)",
}

# tail models of a fit to the tail of a sample: the fitted distribution describes the whole sample (censored fit), or
# only the values above the threshold value u (fit to the exceedances)
_TAIL_MODELS = ("censored", "exceedances")


def empirical_cdf(n, kind="mean"):
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

    if kind not in _PLOTTING_POSITIONS:
        raise QatsValueError("Distribution type must be either 'mean','median','symmetrical','beard','gringorten'")
    n = float(n)
    i = np.arange(n) + 1
    if kind == "mean":
        f = i / (n + 1.0)
    elif kind == "median":
        f = (i - 0.3) / (n + 0.4)
    elif kind == "symmetrical":
        f = (i - 0.5) / n
    elif kind == "beard":
        f = (i - 0.31) / (n + 0.38)
    else:  # gringorten
        f = (i - 0.44) / (n + 0.12)

    return f


def tail_threshold(x, tail):
    """
    Threshold value u of the tail of a sample: the largest value outside the tail.

    The tail is the fraction `tail` of the largest values, selected as in the tail fits of
    :mod:`qats.stats.weibull` and :meth:`qats.TimeSeries.stats`: the values whose mean-rank cumulative probability
    ``i/(n+1)`` is at least ``1 - tail``. A fit to the exceedances fits the values above u.

    .. versionadded :: 5.6.0

    Parameters
    ----------
    x : array_like
        Sample.
    tail : float
        Tail fraction <0, 1], so 0.2 is the 20% largest values.

    Returns
    -------
    float or None
        The threshold value u, or None if the tail holds the whole sample.

    Raises
    ------
    ValueError
        If `tail` is not in <0, 1].
    TypeError
        If `tail` is not a number.

    Examples
    --------
    Fit a Weibull distribution to the 20% largest maxima, with the location parameter at u

    >>> from qats.stats.empirical import tail_threshold
    >>> from qats.stats.weibull import Weibull
    >>> u = tail_threshold(maxima, 0.2)
    >>> weib = Weibull.fit(maxima, method="mle", tail=0.2, tail_model="exceedances", loc=u)
    """
    _check_tail(tail)
    x = np.sort(np.asarray(x, dtype=float).ravel())
    r = _tail_count(x.size, tail)
    return float(x[r - 1]) if r > 0 else None


def _tail_count(n, tail):
    """
    Number of values outside the tail of a sorted sample of size `n`: the values whose mean-rank empirical cumulative
    probability is below ``1 - tail``. The tail fits in :mod:`qats.stats.weibull` select the fitted values by the same
    rule.
    """
    return int(np.count_nonzero(empirical_cdf(n, kind="mean") < 1.0 - tail))


def _check_tail(tail):
    """Raise if the tail fraction `tail` is not a number in <0, 1]."""
    if isinstance(tail, bool) or not isinstance(tail, numbers.Real):
        raise QatsTypeError(f"The tail must be a number, got {type(tail).__name__}")
    if not 0.0 < tail <= 1.0:
        raise QatsValueError(f"The tail must be larger than 0 and at most 1, got {tail}")


def _check_tail_model(tail_model, tail):
    """Raise if `tail_model` is not a tail model, or is 'exceedances' without a tail fraction."""
    if tail_model not in _TAIL_MODELS:
        raise QatsValueError(f"The tail model must be one of {', '.join(_TAIL_MODELS)}, got {tail_model!r}")
    if tail is None and tail_model == "exceedances":
        raise QatsValueError("A fit to the exceedances (tail_model='exceedances') needs the tail fraction, tail=")
