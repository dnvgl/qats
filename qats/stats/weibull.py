#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:class:`Weibull` class and functions related to Weibull distribution.
"""

import numbers

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, fsolve, leastsq
from scipy.special import binom, gamma

from .._deprecation import warn_deprecated
from .._validation import QatsTypeError, QatsValueError
from ..signal import find_maxima
from .empirical import (
    _check_plotting_positions,
    _check_tail,
    _check_tail_model,
    _tail_count,
    empirical_cdf,
    tail_threshold,
)
from .paper import _plot_distribution, paper_coordinates

# todo: build documentation and check that docstrings behave as intended
# todo: create examples
# todo: consider replacing latex with python code or pseudo code (ref. link below)
# link: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#other-points-to-keep-in-mind


class Weibull(object):
    """
    The Weibull class offers miscellaneous functions for working with the Weibull
    distribution, defined as (cumulative distribution function)::

        F(x) = 1 - exp{-[(x-a)/b]^c}

    where `a` is location parameter, `b` is scale parameter and `c` is shape parameter.


    .. versionchanged :: 5.4.0
        Invalid input raises :class:`ValueError` or :class:`TypeError` instead of :class:`AssertionError`, also when
        Python runs with ``-O``. Both are still subclasses of :class:`AssertionError` until 6.0.

    Parameters
    ----------
    loc: float
        Weibull location parameter.
    scale: float
        Weibull scale parameter.
    shape: float
        Weibull shape parameter.
    data: array_like, optional
        Sample data, used to establish empirical cdf and is included in plots.
        To fit the Weibull distribution to the sample data, use :meth:`Weibull.fit()`.

    Attributes
    ----------
    loc: float
        Weibull location parameter.
    scale: float
        Weibull scale parameter.
    shape: float
        Weibull shape parameter.
    data: array_like
        Sample data. Exists only if distribution parameters are estimated from a sample. After a fit to the
        exceedances, the whole sample, of which only the values above the threshold value were fitted.
    fit_info: dict or None
        How the distribution was fitted, see :attr:`Weibull.fit_info`.

    Notes
    -----
    For a Weibull 2-parameter distribution, specify location parameter 0 (zero).


    Examples
    --------
    To initiate an instance based on parameters, use:

    >>> from qats.stats.weibull import Weibull
    >>> weib = Weibull(loc, scale, shape)

    If you need to establish a Weibull instance based on a sample data set, use:

    >>> weib = Weibull.fit(data, method='pwm')


    References
    ----------
    1. Moment estimators for Weibull parameters and their asymptotic efficiencies, Waloddi Weibull, April 1969, Lausanne
       Switzerland, Technical report AFML-TR-69-135

    2. Continuous univariate distributions, Volume 1, N.L.Johnson, S.Kotz and N.Balakrishnan, 1994,  John Wiley and sons
       inc.

    3. `weibull.com <http://www.weibull.com/hotwire/issue15/relbasics15.htm>`_, About location parameter

    4. `Plotting positions <http://en.wikipedia.org/wiki/Q%E2%80%93Q_plot>`_, About plotting positions

    5. `Bootstrapping <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>`_, Bootstrapping statistics

    6. Estimation of the generalized extreme value distribution by the method of probability weighted moments, Hosking,
       J. R. M., Wallis, J. R. and Wood, E. F., 1985, Technometrics, 27, pp. 251-261

    7. Estimating the three-parameter Weibull distribution by the method of probability weighted moments with
       application to medical survival data, Bortolucci, A. A. et.al.

    8. Theory and derivation for Weibull parameter probability weighted moment estimators, Grender, J.M., Dell, T.R.,
       Reich, R.M., 1991 United Sates Department of Agriculture

    9. Probability weighted moments, Greenwood, J. A.; Landwehr, J.M.; Matalas, N.C.; Wallis, J.R., 1979, Water
       Resources Research. 15(5): 1049-1054.

    10. Probability weighted moments compared with some traditional techniques in estimating gumbel parameters and
        quantiles., Landwehr, J.M.; Matalas, N.C.; Wallis, J.R., 1979., Water Resources Research. 15(5): 1063-1064.

    """

    def __init__(self, loc, scale, shape, data=None):
        self.loc = loc
        self.scale = scale
        self.shape = shape

        if data is not None:
            self.data = np.array(data)
        else:
            self.data = None
        self._fit_info = None  # set by Weibull.fit()

    @property
    def fit_info(self):
        """
        How the distribution was fitted by :meth:`Weibull.fit`, None if it was not.

        .. versionadded :: 5.6.0

        Returns
        -------
        dict or None
            A copy of the fit settings, with the meanings of the ``w*`` keys of :meth:`qats.TimeSeries.stats`:

            ==================  ======================================================================================
            method              Method of the fit
            tail                Tail fraction of a tail fit, None if all values were fitted
            tail_model          'censored' or 'exceedances', None if all values were fitted
            loc_from            Where the location parameter is from: 'fitted', or 'value' (given, or zero for ``pwm2``)
            threshold           Threshold value u of a tail fit, the largest value outside the tail (None if none)
            n                   Number of values fitted
            plotting_positions  Plotting positions of an ``lse`` fit, None for the other methods
            ==================  ======================================================================================
        """
        return None if self._fit_info is None else dict(self._fit_info)

    def _described(self):
        """
        The sample values the distribution describes: all of `data`, or only the values above the threshold value
        after a fit to the exceedances.
        """
        info = self._fit_info
        if info is not None and info["tail_model"] == "exceedances" and info["threshold"] is not None:
            return self.data[self.data > info["threshold"]]
        return self.data

    @property
    def cov(self):
        """
        Distribution coefficient of variation (C.O.V.)

        Returns
        -------
        float
            distribution c.o.v.
        """
        return self.std / self.mean

    @property
    def ecdf(self):
        """
        Empirical cumulative distribution function associated with the sample.

        Returns
        -------
        array
            Empirical cumulative distribution function.

        Notes
        -----
        A mean rank method is chosen to approximate the mean of the distribution [2].

        The empirical cdf is also used as plotting positions when plotting the sample
        on probability paper.

        """
        p = empirical_cdf(self.data.size, kind="mean")
        return p

    @property
    def kurt(self):
        """
        Distribution kurtosis.

        Returns
        -------
        float
            distribution kurtosis
        """
        r1 = gamma(1.0 + 1.0 / self.shape)
        r2 = gamma(1.0 + 2.0 / self.shape)
        r3 = gamma(1.0 + 3.0 / self.shape)
        r4 = gamma(1.0 + 4.0 / self.shape)

        k = (r4 + r1 * (-4.0 * r3 + r1 * (6.0 * r2 - 3.0 * r1**2))) / (r2 - r1 * r1)
        return k

    @property
    def mean(self):
        """
        Distribution mean value

        Returns
        -------
        float
            distribution mean value
        """
        return self.loc + self.scale * gamma(1.0 + 1.0 / self.shape)

    @property
    def mse(self):
        """
        Mean squared error of fitted cumulative distribution (a,b,c) and empirical distribution

        .. versionchanged :: 5.6.0
            After a fit to the exceedances, the error is computed for the values above the threshold value, which
            the distribution describes, with their own empirical distribution.

        Returns
        -------
        float
            mean squared error

        """
        x = np.sort(self._described())
        e = np.sum((empirical_cdf(x.size, kind="mean") - self.cdf(x)) ** 2.0) / x.size
        return e

    @property
    def params(self):
        """
        Distribution parameters.

        Returns
        -------
        tuple
            Distribution parameters: (loc, scale, shape).
        """
        return self.loc, self.scale, self.shape

    @property
    def skew(self):
        """
        Distribution skewness

        Returns
        -------
        float
            distribution skewness
        """
        r1 = gamma(1.0 + 1.0 / self.shape)
        r2 = gamma(1.0 + 2.0 / self.shape)
        r3 = gamma(1.0 + 3.0 / self.shape)
        s = (r3 + r1 * (2.0 * r1**2.0 - 3.0 * r2)) / (r2 - r1**2.0) ** 1.5
        return s

    @property
    def std(self):
        """
        Distribution standard deviation

        Returns
        -------
        float
            distribution standard deviation
        """
        r1 = gamma(1.0 + 1.0 / self.shape)
        r2 = gamma(1.0 + 2.0 / self.shape)
        s = self.scale * np.sqrt(r2 - r1**2.0)
        return s

    def gumbel_parameters(self, n=None):
        """
        Calculate parameters of the asymptotic Gumbel extreme value distribution (Type 1) for the extreme value of N
        independent, Weibull distributed variables.

        .. versionchanged :: 5.6.0
            After a fit to the exceedances, the default `n` is the number of values above the threshold value, which
            the distribution describes.

        Parameters
        ----------
        n : int
            number of independent weibull distributed variables, default equal to number of peaks (self.data.size),
            or the number of peaks above the threshold value after a fit to the exceedances

        Returns
        -------
        tuple
            Gumbel location and scale parameters


        See Also
        --------
        qats.stats.weibull.weibull2gumbel


        Notes
        -----
        If the sample `x` is based on lets say a 30-hour simulation but you seek an estimate of the e.g. 3-hour extreme
        value then `n` should be calculated as the nearest integer to::

            n = (3 / 30) * nx

        where `nx` is the total number of maxima during 30 hour.


        References
        ----------
        1. Bury, K.V. (1975), "Statistical models in applied science"

        """
        if n is None:
            n = self._described().size
        gloc, gscale = weibull2gumbel(self.loc, self.scale, self.shape, n)

        return gloc, gscale

    def cdf(self, x=None):
        """
        Cumulative distribution function (cumulative probability) for specified values x

        Parameters
        ----------
        x : array_like
            values

        Returns
        -------
        array
            cumulative probabilities for specified values x

        Notes
        -----
        A range of x values are applied if x is not specified.
        """
        if x is None:
            x = np.linspace(self.loc, self.loc + 3.0 * self.std, 100)
        else:
            x = np.array(x)

        if not (np.all(x >= self.loc)):
            raise QatsValueError("The location parameter must be less than all items in data set")

        p = 1.0 - np.exp(-(((x - self.loc) / self.scale) ** self.shape))
        return p

    @classmethod
    def fit(
        cls, data, method="msm", verbose=False, *, loc=None, tail=None, tail_model="censored", plotting_positions=None
    ):
        """
        Establish Weibull class instance by fit to sample.

        .. versionchanged :: 5.6.0
            New arguments `loc`, `tail`, `tail_model` and `plotting_positions`. The fit settings are kept in
            :attr:`fit_info`. A fit that fails raises :class:`ValueError`, see the fit functions.

        Parameters
        ----------
        data : array_like
            Sample.
        method : str, optional
            Method of fit. Available options:

            - ``msm`` = method of sample moments (default)
            - ``lse`` = least-square estimation
            - ``mle`` = maximum likelihood estimation
            - ``pwm`` = probability weighted moments
            - ``pwm2`` = probability weighted moments, 2-parameter distribution

        verbose : bool
            If True, fitted parameters are printed to screen.
        loc : float, optional
            Fixed location parameter. Only the scale and shape parameters are fitted, to the fitted values minus
            `loc`. By default the location parameter is fitted too, except with ``pwm2`` where it is zero.
        tail : float, optional
            Fit to the tail of the sample: the fraction <0, 1] of the largest values, so 0.2 fits the 20% largest
            values. The values are selected as in :func:`lse` and :func:`mle`, see also
            :func:`qats.stats.empirical.tail_threshold`.
        tail_model : {'censored', 'exceedances'}, optional
            How a fit to the tail treats the other values:

            - 'censored' (default): the distribution describes the whole sample, and the values outside the tail
              count only by their number. Only with ``lse`` and ``mle``, see :func:`lse` and :func:`mle`.
            - 'exceedances': the values above the threshold value u, the largest value outside the tail, are fitted as
              a complete sample with any method, and the distribution describes only those values.

        plotting_positions : str, optional
            Plotting positions an ``lse`` fit is fitted to, see :func:`lse` and
            :func:`qats.stats.empirical.empirical_cdf`. Default 'mean'. Only with ``lse``.

        Returns
        -------
        Weibull
            Weibull class instance. It keeps the whole sample as `data`, also after a fit to the exceedances, and the
            fit settings in :attr:`fit_info`.

        Raises
        ------
        ValueError
            If an option is invalid or the options can't be combined, or the fit fails.
        TypeError
            If `loc` or `tail` is not a number.

        See Also
        --------
        Weibull.fromsignal, qats.stats.empirical.tail_threshold

        Examples
        --------
        Assuming `data` is a sample array/list:

        >>> from qats.stats.weibull import Weibull
        >>> weib = Weibull.fit(data, method="msm")

        Fit the largest 13% of the sample by censored maximum likelihood:

        >>> weib = Weibull.fit(data, method="mle", tail=0.13)

        Fit the scale and shape parameters to the exceedances of the threshold value u of the 13% largest values,
        with the location parameter at u:

        >>> from qats.stats.empirical import tail_threshold
        >>> u = tail_threshold(data, 0.13)
        >>> weib = Weibull.fit(data, method="mle", tail=0.13, tail_model="exceedances", loc=u)

        """
        if not isinstance(method, str) or method.lower() not in _FIT_FUNCTIONS:
            raise QatsValueError("Method must be either %s" % (" or ".join(_FIT_FUNCTIONS)))
        method = method.lower()
        if loc is not None and (isinstance(loc, bool) or not isinstance(loc, numbers.Real)):
            raise QatsTypeError(f"The location must be a number or None, got {type(loc).__name__}")
        if tail is not None:
            _check_tail(tail)
        _check_tail_model(tail_model, tail)
        exceedances = tail is not None and tail_model == "exceedances"
        if tail is not None and not exceedances and method not in ("lse", "mle"):
            raise QatsValueError(
                f"A tail fit is only available with the methods lse and mle, not {method}. Or fit the exceedances "
                f"with tail_model='exceedances'."
            )
        if plotting_positions is not None:
            if method != "lse":
                raise QatsValueError(f"plotting_positions= is only for the method lse, not {method}")
            _check_plotting_positions(plotting_positions)

        data = np.array(data)  # ensure numpy array

        # values fitted: all, or the exceedances of the threshold value u (a complete sample, without tail)
        threshold = tail_threshold(data, tail) if tail is not None else None
        sample = data
        fit_tail = tail
        if exceedances:
            fit_tail = None
            if threshold is not None:
                sample = data[data > threshold]

        if method == "pwm2" and loc is not None:
            loc_fit, scale, shape = pwm(sample, loc=loc)
        elif method == "pwm2":  # 2-parameter distribution
            loc_fit = 0
            scale, shape = pwm2(sample)
        else:  # 3-parameter distribution
            kwargs = dict()
            if loc is not None:
                kwargs["loc"] = loc
            if fit_tail is not None:
                kwargs["tail"] = fit_tail
            if plotting_positions is not None:
                kwargs["plotting_positions"] = plotting_positions
            loc_fit, scale, shape = _FIT_FUNCTIONS[method](sample, **kwargs)

        if verbose:
            print("Fitted parameters:\nlocation = %5.3g\nscale = %5.3g\nshape = %5.3g" % (loc_fit, scale, shape))

        # number of values fitted: in a censored fit the values in the tail, selected by rank as in lse() and mle()
        nfit = sample.size if fit_tail is None else data.size - _tail_count(data.size, tail)

        weib = cls(loc_fit, scale, shape, data=data)
        weib._fit_info = dict(
            method=method,
            tail=tail,
            tail_model=tail_model if tail is not None else None,
            loc_from="value" if (loc is not None or method == "pwm2") else "fitted",
            threshold=threshold,
            n=nfit,
            plotting_positions=(plotting_positions or "mean") if method == "lse" else None,
        )
        return weib

    @classmethod
    def fromsignal(cls, x, method="msm", verbose=False):
        """
        Establish Weibull class instance by fit to global maxima from time series signal.

        Parameters
        ----------
        x : array_like
            Time series signal.
        method : str, optional
            Method of fit. See :meth:`Weibull.fit()` for description of options.
        verbose : bool, optional
            If True, fitted parameters are printed to screen.

        Returns
        -------
        Weibull
            Class instance.

        See Also
        --------
        Weibull.fit
        qats.stats.find_maxima

        Examples
        --------
        Assuming `x` is a time series signal:

        >>> from qats.stats.weibull import Weibull
        >>> weib = Weibull.fromsignal(x, method='msm')

        Note that the example above is equivalent to:

        >>> from qats.signal import find_maxima
        >>> sample, _ = find_maxima(x, local=False, threshold=None, up=True)
        >>> weib = Weibull.fit(sample, method='msm')
        """
        sample, _ = find_maxima(x, local=False, threshold=None, up=True)
        return Weibull.fit(sample, method=method, verbose=verbose)

    def invcdf(self, p=None):
        """
        Inverse cumulative distribution function for specified quantiles p

        Parameters
        ----------
        p : array_like
            quantiles (or. cumulative probabilities if you like)

        Returns
        -------
        array
            values corresponding to the specified quantiles

        Notes
        -----
        A range of quantiles from 0 to 1 are applied if quantiles are not specified

        """
        if p is None:
            p = np.linspace(0.001, 0.999, 100)
        else:
            p = np.array(p)

        x = np.zeros(np.shape(p))

        x[p == 1.0] = np.inf  # asymptotic
        x[(p < 0.0) | (p > 1.0)] = np.nan  # probabilities out of bounds
        x[(p >= 0.0) & (p < 1.0)] = self.loc + self.scale * (-np.log(1.0 - p[(p >= 0.0) & (p < 1.0)])) ** (
            1.0 / self.shape
        )
        return x

    def pdf(self, x=None):
        """
        Probability density function for specified values x

        Parameters
        ----------
        x : array_like
            values

        Returns
        -------
        array
            probability density function for specified values x

        Notes
        -----
        A range of x values are applied if x is not specified.
        """
        if x is None:
            x = np.linspace(self.loc, self.loc + 3.0 * self.std, 100)
        else:
            x = np.array(x)

        if not (np.all(x >= self.loc)):
            raise QatsValueError("The location parameter must be less than all items in data set")

        p = (
            self.shape
            / self.scale
            * ((x - self.loc) / self.scale) ** (self.shape - 1.0)
            * np.exp(-(((x - self.loc) / self.scale) ** self.shape))
        )
        return p

    def plot(self, filename=None, *, paper=None, plotting_positions=None, ax=None):
        """
        Plot the distribution and the sample it was fitted to on probability paper

        .. versionchanged :: 5.6.0
            New arguments `paper`, `plotting_positions` and `ax`, and the axes are returned. The plot is drawn with
            :func:`qats.stats.paper.probability_paper`, and shows a tail fit as :attr:`fit_info` describes it: the
            values outside the fit faded, and the threshold value u dashed. With the default arguments the data and
            the fitted distribution are drawn as before, but the fitted line spans the probability axis, from the
            lowest data point to the top tick.

        Parameters
        ----------
        filename : str, optional
            Save plot as `filename`, default is to show plot on screen
        paper : str, optional
            Probability paper: 'linear' (cumulative probability on a linear scale), 'rayleigh', 'weibull' or
            'gumbel'. See :mod:`qats.stats.paper`. Default 'linear'; will be 'rayleigh' from 6.0.
        plotting_positions : str, optional
            Plotting positions of the data, see :func:`qats.stats.empirical.empirical_cdf`. Default those an
            ``lse`` fit was fitted to (see :attr:`fit_info`), otherwise 'mean', ``i/(n+1)``; will be 'median' from 6.0.
        ax : matplotlib.axes.Axes, optional
            Draw on these axes. The figure is then neither shown nor closed, only saved if `filename` is given.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        Examples
        --------
        Plot distribution and show the figure

        >>> from qats.stats.weibull import Weibull
        >>> distribution = Weibull(100., 15., 2.5)
        >>> distribution.plot()

        Plot distribution and save the figure as png

        >>> distribution.plot(filename="plot.png")

        Plot a fit to the 20% largest maxima on Rayleigh paper with median plotting positions, in a figure of your
        own

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> Weibull.fit(maxima, method="mle", tail=0.2).plot(paper="rayleigh", plotting_positions="median", ax=ax)
        >>> plt.show()

        """
        if plotting_positions is None:
            # the plotting positions an lse fit was fitted to
            plotting_positions = (self._fit_info or dict()).get("plotting_positions") or "mean"
        return _plot_distribution(self, filename, "linear" if paper is None else paper, plotting_positions, ax)

    def plot_linear(self, filename=None):
        """
        Plot data on Weibull paper (linearized scales))

        .. deprecated :: 5.6.0
            Use :meth:`Weibull.plot` with ``paper="weibull"`` instead. ``plot_linear`` will be removed in 6.0.

        .. versionchanged :: 5.6.0
            The coordinates come from :func:`qats.stats.paper.paper_coordinates`, and data at or below the location
            parameter, which can't be shown on Weibull paper, are left out explicitly. The figure is unchanged.

        Parameters
        ----------
        filename : str, optional
            Save plot as `filename`, default is to show plot on screen

        Examples
        --------
        Plot distribution and show the figure

        >>> from qats.stats.weibull import Weibull
        >>> distribution = Weibull(100., 15., 2.5)
        >>> distribution.plot_linear()

        Plot distribution and save the figure as png

        >>> from qats.stats.weibull import Weibull
        >>> distribution = Weibull(100., 15., 2.5)
        >>> distribution.plot_linear(filename="plot.png")

        References
        ----------
        1. `Continuous univariate distributions, Volume 1`, N.L.Johnson, S.Kotz and N.Balakrishnan, 1994, \
            John Wiley and sons inc.
        """
        warn_deprecated(
            "qats.stats.weibull.Weibull.plot_linear()",
            "5.6.0",
            "6.0.0",
            alternative="Weibull.plot(paper='weibull', plotting_positions='mean')",
        )
        plt.figure()

        # plot data if that exists, except values at or below the location parameter
        if self.data is not None:
            x, y = paper_coordinates(np.sort(self.data), self.ecdf, paper="weibull", loc=self.loc)
            shown = np.isfinite(x)
            plt.plot(x[shown], y[shown], "ko", label="Data")

        # plot fitted distribution
        yy = np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
        x, y = paper_coordinates(self.invcdf(p=yy), yy, paper="weibull", loc=self.loc)
        plt.plot(x, y, label="Fitted")

        plt.xlim(x[0], x[-1])
        plt.ylim(y[0], y[-1])
        plt.xlabel("X")
        plt.ylabel("Cumulative probability")
        plt.yticks(y, yy)
        xloc = np.linspace(x[0], x[-1], 5)
        xlab = np.around(np.exp(xloc) + self.loc, decimals=2)
        plt.xticks(xloc, xlab)
        plt.legend(loc="upper left")
        plt.grid(True)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        # close figure
        plt.close()

    def rnd(self, size=None, seed=None):
        """
        Draw random samples from probability distribution

        Parameters
        ----------
        size : int|numpy shape, optional
            sample size (default 1 random value is returned)
        seed : int
            seed for random number generator (default seed is random)

        Returns
        -------
        array
            random sample

        """

        if seed is not None:
            np.random.seed(seed)
        r = np.random.random_sample(size)
        x = self.invcdf(p=r)

        return x


def bootstrap(loc, scale, shape, size, repetitions, method="pwm"):
    """
    Quantify mean and coefficient of variation of Weibull distribution parameters using parametric bootstrapping

    Parameters
    ----------
    loc : float
        Source distribution location parameter
    scale : float
        Source distribution scale parameter
    shape : float
        Source distribution shape parameter
    size : int
        Size of bootstrapped sample
    method : str, optional
        Method of fit. Available options:

        - ``msm`` = method of sample moments
        - ``lse`` = least-square estimation
        - ``mle`` = maximum likelihood estimation
        - ``pwm`` = probability weighted moments (default)

    repetitions : int, optional
        Number of bootstrap samples. default equal to 100

    Returns
    -------
    array
        Mean distribution parameters
    array
        Coefficient of variation of distribution parameter

    Notes
    -----
    In statistics, bootstrapping is a method for assigning measures of accuracy
    to sample estimates (variance,quantiles). This technique allows estimation of the
    sampling distribution  of almost any statistic using only very simple methods. Generally,
    it falls in the broader class of resampling methods. In this case a parametric model is fitted
    to the data, and samples of random numbers with the same size as the original data,
    are drawn from this fitted model. Then the quantity, or estimate, of interest is
    calculated from these data. This sampling process is repeated many times as for other
    bootstrap methods. If the results really matter, as many samples as is reasonable,
    given available computing power and time, should be used. Increasing the number of
    samples cannot increase the amount of information in the original data, it can only
    reduce the effects of random sampling errors which can arise from a bootstrap procedure itself.
    See [5] about bootstrapping.

    Examples
    --------
    To quantify the uncertainty (coefficient of variation) of a Weibull distribution fitted to a sample with 5 values
    (using 100 repetition):

    >>> from qats.stats.weibull import bootstrap
    >>> m, cv = bootstrap(10., 5., 2.5, 5, 100)

    """
    options = {"msm": msm, "lse": lse, "pwm": pwm, "mle": mle}
    if method.lower() not in options.keys():
        raise QatsValueError("Method must be either %s" % (" or ".join(options.keys())))

    # initiate distribution
    distribution = Weibull(loc, scale, shape)

    i = 0
    par = np.zeros((repetitions, 2))
    while i < repetitions:
        # sample random numbers from source distribution
        x = distribution.rnd(size=size)

        # estimate distribution parameters to random sample
        par[i, :] = options[method](x)
        i += 1

    # calculate mean value and coefficient of variation of the distribution parameters
    m = par.mean(axis=0)
    cv = par.std(axis=0, ddof=1) / m

    return m, cv


def lse(x, threshold: float = None, tail: float = None, loc: float = None, plotting_positions: str = "mean"):
    """
    Fit Weibull distribution parameters to sample by method of least square fit to empirical cdf.

    .. versionchanged :: 5.6.0
        New arguments `tail`, `loc` and `plotting_positions`. A fit that fails raises :class:`ValueError` instead of
        returning wrong parameters.

    Parameters
    ----------
    x : array_like
        sample data
    threshold : float, optional
        Fit distribution to data points above this threshold. The threshold is defined as value <0, 1> in the
        empirical CDF. So with threshold=0.87 the distribution is fitted to the values exceeding the 0.87-quantile of
        the empirical cumulative distribution function. Kept for compatibility, use `tail` in new code.
    tail : float, optional
        Fit distribution to the tail of the sample: the fraction <0, 1] of the largest values, so 0.13 fits the 13%
        largest values. ``tail=p`` is the same as ``threshold=1 - p``. Do not give both.
    loc : float, optional
        Fixed location parameter. Only the scale and shape parameters are fitted, as a 2-parameter Weibull
        distribution of the fitted values minus `loc`. Must be below the smallest fitted value.
    plotting_positions : str, optional
        Plotting positions the distribution is fitted to, see :func:`qats.stats.empirical.empirical_cdf`. Default
        'mean', ``i/(n+1)``.

    Returns
    -------
    tuple (floats)
        Distribution parameters ``(loc, scale, shape)``.

    Raises
    ------
    QatsValueError
        If `plotting_positions` is not one of the plotting positions of :func:`qats.stats.empirical.empirical_cdf`,
        or the fit fails: the solver does not converge, a parameter is not finite, the scale or shape parameter is not
        positive, or the location parameter is not below the smallest fitted value.

    Notes
    -----
    The cumulative distribution function is fitted to the given plotting positions of the sample, by default mean
    rank estimates. In a fit to the tail, the plotting positions are those of the whole sample, and the tail is
    selected by mean rank whatever the plotting positions: the values with ``i/(n+1)`` at or above ``1 - tail`` (or
    `threshold`).
    """
    _check_plotting_positions(plotting_positions)
    x = np.sort(x)  # sorted dataset
    f = empirical_cdf(x.size, kind=plotting_positions)  # plotting positions the cdf is fitted to
    rank = empirical_cdf(x.size, kind="mean")  # mean rank, selects the tail
    weights = np.ones(np.shape(x))  # by default include all data points
    threshold = _tail_level(rank, threshold=threshold, tail=tail)
    if threshold is not None:
        # exclude data points below threshold (fit only to the tail)
        weights[rank < threshold] = 0.0
    xmin = x[weights > 0.0][0]  # smallest fitted value

    if loc is not None:
        _check_loc(loc, xmin)
        z, y = x[weights > 0.0], f[weights > 0.0]

        def e2(v):
            # error function to be minimized, 2-parameter weibull function of the fitted values minus loc
            return 1.0 - np.exp(-(((z - loc) / v[0]) ** v[1])) - y

        for v0 in _initial_guesses(z - loc, y, censored=threshold is not None and z.size < x.size):
            p, cov, info, msg, ier = leastsq(e2, np.asarray(v0), full_output=True)
            if ier in (1, 2, 3, 4):
                break
        else:
            raise QatsValueError(f"The lse fit did not converge: {msg}")
        return _check_params((float(loc), p[0], p[1]), xmin, "lse")

    # define error function
    # v: distribution parameters
    # z: data points
    # y: empirical cdf defined at data points x (z)
    # w: weights to exclude data points below threshold if tail fitting is requested
    def fp(v, z):
        # parametric weibull function
        return 1.0 - np.exp(-(((z - v[0]) / v[1]) ** v[2]))

    def e(v, z, y, w):
        # error function to be minimized
        return w * (fp(v, z) - y)

    a0, b0, c0 = _msm3(x)  # initial guess based on method of moments

    # least square fit (x corresponds to z in lambdas, f corresponds to y and w corresponds to weights)
    p, cov, info, msg, ier = leastsq(e, np.asarray([a0, b0, c0]), args=(x, f, weights), full_output=True)
    if ier not in (1, 2, 3, 4):
        raise QatsValueError(f"The lse fit did not converge: {msg}")

    return _check_params((p[0], p[1], p[2]), xmin, "lse")


def mle(x, tail: float = None, loc: float = None):
    """
    Fit Weibull distribution parameters to sample by maximum likelihood estimation

    .. versionchanged :: 5.6.0
        New arguments `tail` (censored fit) and `loc`. A fit that fails raises :class:`ValueError` instead of
        returning wrong parameters.

    Parameters
    ----------
    x : array_like
        sample data
    tail : float, optional
        Fit distribution to the tail of the sample by censored maximum likelihood: the fraction <0, 1] of the largest
        values is fitted, so 0.13 fits the 13% largest values. The values are selected as in :func:`lse`, those with
        empirical CDF (mean rank) at or above ``1 - tail``. The other values are censored.
    loc : float, optional
        Fixed location parameter. Only the scale and shape parameters are fitted. Must be below the smallest value in
        the likelihood, which is the censoring value in a tail fit.

    Returns
    -------
    tuple (floats)
        Distribution parameters ``(loc, scale, shape)``.

    Raises
    ------
    QatsValueError
        If the fit fails: the solver does not converge, a parameter is not finite, the scale or shape parameter is not
        positive, or the location parameter is not below the smallest value in the likelihood.

    Notes
    -----
    In a tail fit, the `r` values with empirical CDF (mean rank) below ``1 - tail`` are censored at the largest of
    them, ``xc = x(r)``. They enter the likelihood only through the probability that a value is at or below `xc`::

        L = F(xc)^r * f(x(r+1)) * ... * f(x(n))

    The likelihood equations [2, p. 656 eq. 21.73-21.75] are then summed over the n - r fitted values and each gets
    the derivative of ``r * ln F(xc)``. With no censored values (`r` = 0) the fit is the same as without `tail`.

    With a fixed location parameter the equation for the location parameter is left out.

    """
    r = 0  # number of censored values
    xc = None  # censoring value

    def mle_eq(p, z):
        """
        MLE equation set according to [2, p. 656 eq. 21.73-21.75]

        Parameters
        ----------
        p : array_like
            distribution parameters a, b, c
        z : array_like
            data sample (fitted values)

        Returns
        -------
        tuple (floats)
            Estimates of distribution parameters ``(loc, scale, shape)``.
        """
        loc, scale, shape = p  # unpack parameters
        n = z.size

        f1 = (
            n / shape
            + np.sum(np.log((z - loc) / scale))
            - np.sum(((z - loc) / scale) ** shape * np.log((z - loc) / scale))
        )
        f2 = -n * shape / scale + shape / scale * np.sum(((z - loc) / scale) ** shape)
        f3 = -(shape - 1.0) * np.sum(1.0 / (z - loc)) + shape / scale * np.sum(((z - loc) / scale) ** (shape - 1.0))

        if r > 0:
            # derivatives of r * ln F(xc)
            zc = (xc - loc) / scale
            h = zc**shape
            d = r / np.expm1(h)
            f1 += d * h * np.log(zc)
            f2 -= d * shape / scale * h
            f3 -= d * shape / scale * zc ** (shape - 1.0)

        return f1, f2, f3

    x = np.array(x)
    z = x  # fitted values
    if tail is not None:
        f = empirical_cdf(x.size, kind="mean")
        threshold = _tail_level(f, tail=tail)
        r = int(np.count_nonzero(f < threshold))
        if r > 0:
            z = np.sort(x)
            xc = z[r - 1]
            z = z[r:]
    xmin = xc if r > 0 else z.min()  # smallest value in the likelihood

    if loc is not None:
        _check_loc(loc, xmin)
        zs = np.sort(z)
        for v0 in _initial_guesses(zs - loc, empirical_cdf(x.size, kind="mean")[r:], censored=r > 0):
            p, info, ier, msg = fsolve(lambda v: mle_eq((loc, v[0], v[1]), zs)[:2], np.asarray(v0), full_output=True)
            if ier == 1:
                break
        params = (float(loc), p[0], p[1])
    else:
        a0, b0, c0 = _msm3(x)  # initial guess
        p, info, ier, msg = fsolve(mle_eq, np.asarray([a0, b0, c0]), args=z, full_output=True)
        params = (p[0], p[1], p[2])

    if ier != 1:
        raise QatsValueError(f"The mle fit did not converge: {msg}")

    return _check_params(params, xmin, "mle")


def mlj(sample, l, j):  # noqa: E741 (`l` is a public keyword argument)
    """
    Probability weighted moment `Mljk` of observation order l, order of cdf j, with emphasize on the
    right/upper tail (k=0).

    Parameters
    ----------
    sample : array
        Sample.
    l : int
        Order of observation (sample).
    j : int
        Order of cumulative distribution function.

    Returns
    -------
    float
        Probability weighted moment

    Notes
    -----
    The probability weighted moment `Mljk` is defined by Greenwood and others (1979) as::

        M_{l,j,k} = E[X^l * F^j * (1-F)^k]

    where `X(F)` is the inverse form of the distribution and `F` is the cumulative distribution function.
    When `j=k=0` and `l` is a non-negative integer, then `M_{l,0,0}` represents the conventional moment of order `l`
    about the origin.

    PWMs can be applied either when the small observations are more important than the large observations (k=0), as in
    strength properties of materials, or when the large observations should have more influence than the smaller
    observations (k=0) as with three diameter distribution modelling. Here we have chosen the latter and derived
    unbiased estimators for moments `M_{l,j,0}(k=0)`, see eq. 32 in [8]::

        M_{l,j,0} = (1 / n) * sum(x[i]^l * binom(i-1, j) / binom(n-1, j))

    where `i` is a counter from `j+1` to `n` and `binom()` is the binomial coefficient.

    """
    # todo: include pseudo-code (or Sphinx-friendly LaTex code) for M_{l,j,0} as included below
    """
    .. math:: M_{l,j,k} = E[X^l F^j (1-F)^k]
    .. math:: M_{l,j,0} = \\frac{1}{n}\\sum_{i=j+1}^{n}{X_{(i)}^l\\frac{\\binom{i-1}{j}}{\\binom{n-1}{j}}}
    """
    n = float(sample.size)
    xi = np.sort(sample)[j:]  # (j+1)th subsample of sorted sample
    ii = np.arange(j + 1, n + 1)

    # ratio of binomial coefficients, probability that xi is the largest observation in a subsample of size (j+1)
    # drawn from n.
    bc = np.array([binom(i - 1.0, j) / binom(n - 1.0, j) for i in ii])

    return (1.0 / n) * np.sum(xi**l * bc)  # eq. 32 in [8]


def msm(x, loc: float = None):
    """
    Fit Weibull distribution parameters to sample by method of sample moments

    .. versionchanged :: 5.6.0
        New argument `loc`. A fit that fails raises :class:`ValueError` instead of returning wrong parameters.

    Parameters
    ----------
    x : array_like
        sample data
    loc : float, optional
        Fixed location parameter. Only the scale and shape parameters are fitted, as a 2-parameter Weibull
        distribution of the sample minus `loc`. Must be below the smallest value in the sample.

    Returns
    -------
    floats
        The `loc`, `scale` and `shape` distribution parameters

    Raises
    ------
    QatsValueError
        If the fit fails: no shape parameter matches the sample moments, a parameter is not finite, the scale or shape
        parameter is not positive, or the location parameter is not below the smallest value in the sample.

    Notes
    -----
    See description in [1]. With a fixed location parameter, the shape parameter is the one that gives the sample
    coefficient of variation of ``x - loc``, and the scale parameter follows from the sample mean of ``x - loc``.
    """
    x = np.array(x)
    if loc is None:
        return _check_params(_msm3(x), x.min(), "msm")

    _check_loc(loc, x.min())
    b, c = _msm2(x - loc)

    return _check_params((float(loc), b, c), x.min(), "msm")


def _msm2(y):
    """2-parameter fit (location zero) by method of sample moments, without checking the parameters."""
    m1 = y.mean()  # first sample raw moment aka. mean
    cv = y.std() / m1  # coefficient of variation

    def f(shape):
        return np.sqrt(gamma(1.0 + 2.0 / shape) / gamma(1.0 + 1.0 / shape) ** 2.0 - 1.0) - cv

    c = _shape_root(f, "coefficient of variation", cv)
    b = m1 / gamma(1.0 + 1.0 / c)

    return b, c


def _msm3(x):
    """3-parameter fit by method of sample moments, without checking the parameters."""
    a1 = x.mean()  # first sample raw moment aka. mean
    m2 = x.var()  # second sample central moment aka. variance
    m3 = np.mean((x - a1) ** 3.0)  # third sample central moment
    c1 = m3 / m2 ** (3.0 / 2.0)  # coefficient of sample skewness

    # solve for shape parameter using root search, see [1, eq. 68 on p.11]
    def f(shape, shape0):
        eq = (
            gamma((shape + 3.0) / shape)
            - 3.0 * gamma((shape + 1.0) / shape) * gamma((shape + 2.0) / shape)
            + 2 * gamma((shape + 1.0) / shape) ** 3.0
        ) / (gamma((shape + 2.0) / shape) - gamma((shape + 1.0) / shape) ** 2.0) ** (3.0 / 2.0) - shape0
        return eq

    c = _shape_root(lambda shape: f(shape, c1), "skewness", c1)

    # calculate location and scale parameters, see [1, eq. 67 on p. 11]
    g1 = gamma((c + 1.0) / c)
    g2 = gamma((c + 2.0) / c)
    b = np.sqrt(m2 / (g2 - g1**2.0))
    a = a1 - g1 * b

    return a, b, c


def plot_fit(x: np.ndarray, params: tuple, path: str = None):
    """
    Plot data sample versus empirical and fitted cumulative distribution function on linearized Weibull scales

    .. deprecated :: 5.6.0
        Use ``Weibull(loc, scale, shape, data=x).plot(paper="weibull")`` instead, see :meth:`Weibull.plot`.
        ``plot_fit`` will be removed in 6.0.

    Parameters
    ----------
    x : array_like
        Data sample
    params : tuple
        location, scale and shape parameter of the Weibull distribution
    path : str, optional
        Save figure to file instead of displaying it.

    """
    warn_deprecated(
        "qats.stats.weibull.plot_fit()",
        "5.6.0",
        "6.0.0",
        alternative="Weibull(loc, scale, shape, data=x).plot(paper='weibull', plotting_positions='mean')",
    )
    # unpack distribution parameters
    a, b, c = params

    # normalize data and empirical CDF
    x = np.sort(x)
    x_norm = np.log(x - a)
    y_norm = np.log(-np.log(1.0 - (np.arange(x.size) + 1.0) / (x.size + 1.0)))

    # labels and tick positions for weibull paper plot
    p = np.array([0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
    p_norm = np.log(-np.log(1.0 - p))

    # normalize fitted distribution
    q_norm = np.log(b * (-np.log(1.0 - p)) ** (1.0 / c))

    # plot
    plt.figure()
    plt.plot(x_norm, y_norm, "ko", label="Data")
    plt.plot(q_norm, p_norm, label="Fit")
    plt.yticks(p_norm, p)
    xticks = np.linspace(q_norm[0], q_norm[-1], 5)
    xlabel = np.around(np.exp(xticks) + a, decimals=2)
    plt.xticks(xticks, xlabel)
    plt.title(f"a={a:5.3f}, b={b:5.3f}, c={c:5.3f}")
    plt.xlabel("X")
    plt.ylabel("Cumulative probability")
    plt.legend(loc="upper left")
    plt.grid(True)
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


def pwm(x, loc: float = None):
    """
    Fit distribution parameters to sample by method of probability weighted
    moments

    .. versionchanged :: 5.6.0
        New argument `loc`. A fit that fails raises :class:`ValueError` instead of returning wrong parameters.

    Parameters
    ----------
    x : array_like
        sample data
    loc : float, optional
        Fixed location parameter. Only the scale and shape parameters are fitted, with the formulas of :func:`pwm2`
        applied to the sample minus `loc`. Must be below the smallest value in the sample.

    Returns
    -------
    floats
        The `loc`, `scale` and `shape` distribution parameters

    Raises
    ------
    QatsValueError
        If the fit fails: a parameter is not finite, the scale or shape parameter is not positive, or the location
        parameter is not below the smallest value in the sample.

    Notes
    -----
    Details on probability weighted moments are provided in [8].

    See Also
    --------
    mlj()

    """
    x = np.asarray(x)
    if loc is None:
        params = _pwm3(x)
    else:
        _check_loc(loc, x.min())
        b, c = _pwm2(x - loc)
        params = (float(loc), b, c)

    return _check_params(params, x.min(), "pwm")


def _pwm3(x):
    """3-parameter fit by probability weighted moments, without checking the parameters."""
    # probability weighted moments of order 0,1,2,3
    m100 = mlj(x, 1, 0)
    m110 = mlj(x, 1, 1)
    m120 = mlj(x, 1, 2)
    m130 = mlj(x, 1, 3)

    # parameter estimates
    c = np.log(2.0) / np.log((2.0 * m110 - m100) / (2.0 * (5.0 * m110 - m100 - 6.0 * m120 + 2.0 * m130)))
    a = 4.0 * (m100 * (3.0 * m120 - m130 - m110) - m110**2.0) / (m100 - 8.0 * m110 + 12.0 * m120 - 4.0 * m130)
    b = (m100 - a) / gamma(1.0 + 1.0 / c)

    return a, b, c


def pwm2(x):
    """
    Fit distribution parameters to sample by method of probability weighted moments assuming the location parameter
    is zero.

    .. versionchanged :: 5.6.0
        A fit that fails raises :class:`ValueError` instead of returning wrong parameters.

    Parameters
    ----------
    x : array_like
        sample data

    Returns
    -------
    floats
        The `scale` and `shape` distribution parameters

    Raises
    ------
    QatsValueError
        If the fit fails: a parameter is not finite or not positive, or the sample has values at or below zero.

    Notes
    -----
    Details on probability weighted moments are provided on p.14-15 in [8]. Note that only the scale and parameters are
    estimated, the location parameter is assumed zero. The result is the same as from ``pwm(x, loc=0)``.

    See Also
    --------
    mlj(), pwm()

    """
    _, b, c = pwm(x, loc=0.0)
    return b, c


def _pwm2(x):
    """2-parameter fit by probability weighted moments, without checking the parameters."""
    # probability weighted moments of order 0,1
    m100 = mlj(x, 1, 0)
    m110 = mlj(x, 1, 1)

    # parameter estimates
    c = np.log(2.0) / np.log(m100 / (2.0 * (m100 - m110)))
    b = m100 / gamma(1.0 + 1.0 / c)

    return b, c


def weibull2gumbel(loc, scale, shape, n):
    """
    Calculate parameters of the asymptotic Gumbel extreme value distribution (Type 1) for the extreme value of N
    independent,Weibull distributed variables.

    Parameters
    ----------
    loc : float
        Weibull distribution location parameter
    scale : float
        Weibull distribution scale parameter
    shape : float
        Weibull distribution shape parameter
    n : int
        Number of independent weibull distributed variables

    Returns
    -------
    tuple
        Gumbel location and scale parameters

    Notes
    -----
    If the sample `x` is based on lets say a 30-hour simulation but you seek
    an estimate of the e.g. 3-hour extreme value then `n` should be calculated as
    the nearest integer to::

        n = (3 / 30) * nx

    where `nx` is the total number of maxima during 30 hour.

    References
    ----------
    1. Bury, K.V. (1975), "Statistical models in applied science"

    """
    gloc = loc + scale * np.log(n) ** (1.0 / shape)
    gscale = 1.0 / (shape / scale * np.log(n) ** ((shape - 1.0) / shape))

    return gloc, gscale


def _tail_level(f, threshold=None, tail=None):
    """
    Level of the empirical cdf `f` at and above which values are fitted, from either `threshold` (that level) or
    `tail` (the fraction of the largest values, the level is ``1 - tail``). None if neither is given.
    """
    if threshold is not None and tail is not None:
        raise QatsValueError("Give either tail or threshold, not both")
    if tail is not None:
        _check_tail(tail)
        threshold = 1.0 - tail
    elif threshold is None:
        return None
    elif not 0.0 <= threshold < 1.0:
        raise QatsValueError(f"The threshold must be at least 0 and less than 1, got {threshold}")
    if not np.any(f >= threshold):
        what = f"tail {tail}" if tail is not None else f"threshold {threshold}"
        raise QatsValueError(f"No values of the sample are in the {what}")
    return threshold


def _check_loc(loc, xmin):
    """Check that a fixed location parameter is below the smallest fitted value."""
    if not loc < xmin:
        raise QatsValueError(f"The location parameter ({loc:g}) must be below the smallest fitted value ({xmin:g})")


def _check_params(params, xmin, method):
    """Raise if fitted parameters are not finite, scale or shape is not positive, or loc is not below `xmin`."""
    loc, scale, shape = params
    if not np.all(np.isfinite(params)):
        raise QatsValueError(
            f"The {method} fit failed, the parameters are not finite: loc={loc:g}, scale={scale:g}, shape={shape:g}"
        )
    if not (scale > 0.0 and shape > 0.0):
        raise QatsValueError(f"The {method} fit failed, the scale ({scale:g}) and shape ({shape:g}) must be positive")
    if not loc < xmin:
        raise QatsValueError(
            f"The {method} fit failed, the location parameter ({loc:g}) is not below the smallest fitted value "
            f"({xmin:g})"
        )
    return params


def _initial_guesses(y, f, censored):
    """
    Initial guesses of scale and shape of a 2-parameter Weibull distribution of the sorted values `y` > 0 with
    empirical cdf `f`, in the order to try them.

    - A straight line fit on Weibull paper uses the positions of the values in the sample, so it suits a censored
      sample, but a single value close to zero can pull it far off.
    - The method of moments isn't disturbed by values close to zero, but it treats `y` as a complete sample. If no
      shape parameter matches the moments, an exponential distribution with the mean of `y` is used instead.
    """
    shape, intercept = np.polyfit(np.log(y), np.log(-np.log(1.0 - f)), 1)
    paper = (np.exp(-intercept / shape), shape)
    try:
        moments = _msm2(y)
    except QatsValueError:
        moments = (y.mean(), 1.0)
    return [paper, moments] if censored else [moments, paper]


def _shape_root(f, statistic, value):
    """Root of `f` between shape parameters 0.1 and 1000."""
    try:
        return brentq(f, a=0.1, b=1000.0, maxiter=1000)
    except (ValueError, RuntimeError) as err:
        raise QatsValueError(
            f"The msm fit failed, no shape parameter between 0.1 and 1000 matches the sample {statistic} ({value:g})"
        ) from err


# fit functions of Weibull.fit() by method, defined once for the library: TimeSeries.fit_weibull() and stats() offer the
# same methods
_FIT_FUNCTIONS = {"msm": msm, "lse": lse, "mle": mle, "pwm": pwm, "pwm2": pwm2}
_FIT_METHODS = tuple(_FIT_FUNCTIONS)
