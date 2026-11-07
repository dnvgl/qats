# -*- coding: utf-8 -*-
"""
Coordinates for plotting a sample and a fitted distribution on probability paper.

On probability paper, the cumulative probability axis is transformed so that one distribution plots as a straight
line. The functions return the coordinates, tick positions and tick labels, and leave the drawing to the caller.

============  ============  ===================  ==========================================================
Paper         x axis        y axis               Straight line for
============  ============  ===================  ==========================================================
``rayleigh``  x             √(−ln(1−F))          Rayleigh: Weibull with shape 2 (any location)
``weibull``   ln(x − loc)   ln(−ln(1−F))         Weibull with location `loc`
``gumbel``    x             −ln(−ln F)           Gumbel (extremes)
============  ============  ===================  ==========================================================

.. versionadded :: 5.6.0
"""

import numbers
from dataclasses import dataclass

import numpy as np

from .._validation import QatsTypeError, QatsValueError
from .empirical import _tail_count, empirical_cdf
from .weibull import Weibull

PAPERS = ("rayleigh", "weibull", "gumbel")
_KINDS = ("mean", "median", "symmetrical", "beard", "gringorten")
_TAIL_MODELS = ("censored", "exceedances")

# candidate probabilities for the ticks on the probability axis of each paper, spread evenly enough on that paper
_UPPER = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
_PROBABILITIES = dict(
    rayleigh=np.array([0.0, 0.1, 0.3] + _UPPER),
    weibull=np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3] + _UPPER),
    gumbel=np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3] + _UPPER),
)

# number of points on the fitted line
_NFIT = 500


@dataclass(frozen=True, eq=False)
class ProbabilityPaper:
    """
    A sample and a fitted distribution on probability paper, returned by :func:`probability_paper`.

    .. versionadded :: 5.6.0

    Attributes
    ----------
    paper : str
        Probability paper: 'rayleigh', 'weibull' or 'gumbel'.
    kind : str
        Plotting positions, see :func:`qats.stats.empirical.empirical_cdf`.
    loc : float
        Location of the Weibull paper, the x axis is ln(x - loc). Zero on the other papers.
    sample : array
        The sample values shown on the paper, in the original units, in the same order as `x`.
    x, y : array
        The sample on the paper's axes.
    fitted : array of bool
        The sample values the fit used, in the same order as `x`: all values, or the values in the tail of a tail
        fit.
    excluded : array
        Sample values, in the original units, that can't be shown on this paper: the values at or below the location
        of a Weibull paper.
    fit_x, fit_y : array
        The fitted distribution on the paper's axes, a straight line on its own paper and a curve on the others.
        Empty without a fitted distribution.
    xticks : array
        Tick positions on the x axis.
    xticklabels : list of str
        Tick labels on the x axis, in data units.
    yticks : array
        Tick positions on the y axis.
    yticklabels : list of str
        Tick labels on the y axis, cumulative probabilities.
    """

    paper: str
    kind: str
    loc: float
    sample: np.ndarray
    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    excluded: np.ndarray
    fit_x: np.ndarray
    fit_y: np.ndarray
    xticks: np.ndarray
    xticklabels: list[str]
    yticks: np.ndarray
    yticklabels: list[str]


def paper_coordinates(x, f, paper="rayleigh", loc=0.0):
    """
    Transform values and cumulative probabilities to the axes of a probability paper.

    .. versionadded :: 5.6.0

    Parameters
    ----------
    x : array_like
        Values.
    f : array_like
        Cumulative probabilities, between 0 and 1.
    paper : str, optional
        Probability paper: 'rayleigh' (default), 'weibull' or 'gumbel'. See :mod:`qats.stats.paper`.
    loc : float, optional
        Location of the Weibull paper, the x axis is ln(x - loc). Used only on Weibull paper.

    Returns
    -------
    array
        Values on the x axis. On Weibull paper, values at or below `loc` give NaN.
    array
        Cumulative probabilities on the y axis.

    Examples
    --------
    >>> from qats.stats.paper import paper_coordinates
    >>> px, py = paper_coordinates([1.0, 2.0, 4.0], [0.2, 0.5, 0.8], paper="weibull", loc=0.5)
    """
    _check_paper(paper)
    return _paper_x(x, paper, loc), _paper_y(f, paper)


def probability_paper(
    sample,
    fitted=None,
    *,
    paper="rayleigh",
    kind="median",
    tail=None,
    tail_model="censored",
    loc=None,
    is_minima=False,
):
    """
    Sample and fitted distribution on probability paper: everything needed to draw a probability plot.

    .. versionadded :: 5.6.0

    Parameters
    ----------
    sample : array_like
        Sample, for example the maxima of a time series. With `is_minima`, the minima in their original units.
    fitted : object, optional
        Fitted distribution with methods ``cdf()`` and ``invcdf()``, for example
        :class:`~qats.stats.weibull.Weibull` or :class:`~qats.stats.gumbel.Gumbel`. With `is_minima`, the
        distribution fitted to the minima multiplied by -1.
    paper : str, optional
        Probability paper: 'rayleigh' (default), 'weibull' or 'gumbel'. See :mod:`qats.stats.paper`.
    kind : str, optional
        Plotting positions, see :func:`qats.stats.empirical.empirical_cdf`. Default 'median', ``(i-0.3)/(n+0.4)``.
    tail : float, optional
        Tail fraction of a tail fit: the fit used only the largest values, a fraction `tail` of the sample. The
        values in the tail are marked in `fitted` of the result. See Notes.
    tail_model : {'censored', 'exceedances'}, optional
        How `fitted` describes the sample, with `tail`:

        - 'censored' (default): `fitted` describes the whole sample, as a censored fit does.
        - 'exceedances': `fitted` describes only the values above the threshold value u, as a fit to the
          exceedances does. Its conditional cumulative distribution F_u is drawn as ``F = 1 - p (1 - F_u)``, where p
          is the fraction of the sample above u, so the fitted line runs through the upper values of the whole
          sample from u upwards.

    loc : float, optional
        Location of the Weibull paper, the x axis is ln(x - loc). Default the location of `fitted` if it is a
        :class:`~qats.stats.weibull.Weibull` distribution, otherwise 0. Only for Weibull paper. With `is_minima`,
        the location of the minima multiplied by -1.
    is_minima : bool, optional
        The sample is minima. The paper is drawn for the minima multiplied by -1, with x tick labels in the original
        units.

    Returns
    -------
    ProbabilityPaper
        Coordinates of the sample and the fitted line, which sample values the fit used and which can't be shown,
        and the tick positions and labels.

    Raises
    ------
    ValueError
        If an option is invalid, or the sample is empty or has values that are not finite.
    TypeError
        If `tail` or `loc` is not a number, or `fitted` has no ``invcdf()``.

    Notes
    -----
    The plotting positions are computed for the whole sample, also with a tail, and before excluding the values a
    Weibull paper can't show, so these don't shift the other values.

    The values in the tail are selected as in the tail fits of :mod:`qats.stats.weibull` and
    :meth:`qats.TimeSeries.stats`, independently of `kind`: the values whose mean-rank cumulative probability
    ``i/(n+1)`` is at least ``1 - tail``. The threshold value u is the largest value outside the tail. With the
    'censored' tail model, the values in the tail are marked as fitted. With 'exceedances', the values above u are.

    A fit to the exceedances describes only the values above u. Without `tail`, it is drawn with only the
    exceedances as its own complete sample: pass the exceedances as `sample`.

    Examples
    --------
    Draw the maxima of a time series and a Weibull distribution fitted to them on Rayleigh paper

    >>> import matplotlib.pyplot as plt
    >>> from qats.stats.paper import probability_paper
    >>> from qats.stats.weibull import Weibull
    >>> mx = ts.maxima()
    >>> p = probability_paper(mx, Weibull.fit(mx, method="mle", tail=0.2), tail=0.2)
    >>> plt.plot(p.x[p.fitted], p.y[p.fitted], "o")
    >>> plt.plot(p.x[~p.fitted], p.y[~p.fitted], "o", mfc="none")
    >>> plt.plot(p.fit_x, p.fit_y, "-")
    >>> plt.xticks(p.xticks, p.xticklabels)
    >>> plt.yticks(p.yticks, p.yticklabels)
    >>> plt.show()
    """
    _check_paper(paper)
    if kind not in _KINDS:
        raise QatsValueError(f"The plotting positions (kind) must be one of {', '.join(_KINDS)}, got {kind!r}")
    if tail is not None:
        if isinstance(tail, bool) or not isinstance(tail, numbers.Real):
            raise QatsTypeError(f"The tail must be a number, got {type(tail).__name__}")
        if not 0.0 < tail <= 1.0:
            raise QatsValueError(f"The tail must be larger than 0 and at most 1, got {tail}")
    if tail_model not in _TAIL_MODELS:
        raise QatsValueError(f"The tail model must be one of {', '.join(_TAIL_MODELS)}, got {tail_model!r}")
    exceedances = tail is not None and tail_model == "exceedances"
    if tail is None and tail_model == "exceedances":
        raise QatsValueError("A fit to the exceedances (tail_model='exceedances') needs the tail fraction, tail=")
    if exceedances and fitted is None:
        raise QatsValueError("The tail model 'exceedances' describes a fitted distribution, give fitted=")
    if fitted is not None and not callable(getattr(fitted, "invcdf", None)):
        raise QatsTypeError(f"The fitted distribution must have a method invcdf(), got {type(fitted).__name__}")
    if loc is not None:
        if paper != "weibull":
            raise QatsValueError(f"The location (loc=) is only for Weibull paper, not {paper} paper")
        if isinstance(loc, bool) or not isinstance(loc, numbers.Real):
            raise QatsTypeError(f"The location must be a number, got {type(loc).__name__}")

    z = np.asarray(sample, dtype=float).ravel()
    if z.size == 0:
        raise QatsValueError("The sample is empty")
    if not np.all(np.isfinite(z)):
        raise QatsValueError("The sample has values that are not finite (NaN or infinite)")
    sign = -1.0 if is_minima else 1.0
    z = np.sort(sign * z)
    n = z.size
    f = empirical_cdf(n, kind=kind)

    # values used in the fit
    r = _tail_count(n, tail) if tail is not None else 0
    u = z[r - 1] if r > 0 else None
    if exceedances and u is not None:
        used = z > u
    else:
        used = np.arange(n) >= r
    p = np.count_nonzero(used) / n if exceedances else 1.0

    # location of the Weibull paper, and the values it can show
    if paper == "weibull":
        if loc is None:
            loc = float(fitted.loc) if isinstance(fitted, Weibull) else 0.0
        shown = z > loc
    else:
        loc = 0.0
        shown = np.ones(n, dtype=bool)
    x, y = _paper_x(z[shown], paper, loc), _paper_y(f[shown], paper)

    # probability axis: from the tick at or below the lowest point to the tick at or above the highest
    fs = f[shown] if np.any(shown) else f
    f_lo = fs[0]
    if exceedances and u is not None:
        # the fitted line runs from u upwards
        f_lo = 1.0 - p * (1.0 - _cdf_at(fitted, u))
    candidates = _PROBABILITIES[paper]
    ptick = _tick_probabilities(candidates, min(fs[0], f_lo), fs[-1])

    # fitted line on a grid uniform on the probability axis
    fit_x, fit_y = np.array([]), np.array([])
    xfit = np.array([])
    if fitted is not None:
        fit_y = np.linspace(_paper_y(f_lo, paper), _paper_y(ptick[-1], paper), _NFIT)
        fgrid = _paper_f(fit_y, paper)
        if exceedances:
            fgrid = np.clip(1.0 - (1.0 - fgrid) / p, 0.0, 1.0)
        xfit = np.asarray(fitted.invcdf(fgrid), dtype=float)
        keep = np.isfinite(xfit)
        if paper == "weibull":
            keep &= xfit > loc
        xfit, fit_y = xfit[keep], fit_y[keep]
        fit_x = _paper_x(xfit, paper, loc)

    # x axis ticks in data units
    values = np.concatenate([z[shown], xfit])
    xtick = _nice_ticks(values.min(), values.max()) if values.size else np.array([])
    if paper == "weibull":
        xtick = xtick[xtick > loc]

    return ProbabilityPaper(
        paper=paper,
        kind=kind,
        loc=loc,
        sample=sign * z[shown],
        x=x,
        y=y,
        fitted=used[shown],
        excluded=sign * z[~shown],
        fit_x=fit_x,
        fit_y=fit_y,
        xticks=_paper_x(xtick, paper, loc),
        xticklabels=[_format(sign * v) for v in xtick],
        yticks=_paper_y(ptick, paper),
        yticklabels=[_format(v) for v in ptick],
    )


def _check_paper(paper):
    if paper not in PAPERS:
        raise QatsValueError(f"The probability paper must be one of {', '.join(PAPERS)}, got {paper!r}")


def _paper_x(x, paper, loc):
    """Values on the x axis of a probability paper, NaN at or below the location of a Weibull paper."""
    x = np.asarray(x, dtype=float)
    if paper == "weibull":
        d = x - loc
        return np.log(np.where(d > 0.0, d, np.nan))
    return x.copy()


def _paper_y(f, paper):
    """Cumulative probabilities on the y axis of a probability paper."""
    f = np.asarray(f, dtype=float)
    if paper == "rayleigh":
        return np.sqrt(-np.log1p(-f))
    elif paper == "weibull":
        return np.log(-np.log1p(-f))
    else:
        return -np.log(-np.log(f))


def _paper_f(y, paper):
    """Cumulative probabilities from the y axis of a probability paper, the inverse of :func:`_paper_y`."""
    y = np.asarray(y, dtype=float)
    if paper == "rayleigh":
        return -np.expm1(-(y**2))
    elif paper == "weibull":
        return -np.expm1(-np.exp(y))
    else:
        return np.exp(-np.exp(-y))


def _cdf_at(fitted, x):
    """
    Cumulative probability of the fitted distribution at x, zero at or below the lower bound of a Weibull distribution
    (and with a location that is NaN, where the fitted line is empty).
    """
    if isinstance(fitted, Weibull) and not x > fitted.loc:
        return 0.0
    return float(fitted.cdf(np.array([x]))[0])


def _tick_probabilities(candidates, lo, hi):
    """
    Tick probabilities from the largest candidate at or below `lo` to the smallest candidate at or above `hi`, or the
    smallest and largest candidates.
    """
    below = candidates[candidates <= lo]
    above = candidates[candidates >= hi]
    first = below[-1] if below.size else candidates[0]
    last = above[0] if above.size else candidates[-1]
    return candidates[(candidates >= first) & (candidates <= last)]


def _nice_ticks(lo, hi, n=5):
    """About `n` ticks between `lo` and `hi`, at steps of 1, 2, 2.5 or 5 times a power of 10."""
    if not hi > lo:
        return np.array([lo])
    raw = (hi - lo) / n
    mag = 10.0 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * mag >= raw * (1.0 - 1e-12))
    ticks = np.arange(np.ceil(lo / step - 1e-9), np.floor(hi / step + 1e-9) + 1.0) * step
    decimals = max(0, int(-np.floor(np.log10(step))) + 1)
    return np.round(ticks, decimals) + 0.0  # + 0.0 turns -0.0 into 0.0


def _format(v):
    return f"{v + 0.0:g}"
