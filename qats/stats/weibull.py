#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:class:`Weibull` class and functions related to Weibull distribution.
"""

import numpy as np
from scipy.special import gamma, binom
from scipy.optimize import leastsq, fsolve, brentq
import matplotlib.pyplot as plt
from .empirical import empirical_cdf
from ..signal import find_maxima


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
        Sample data. Exists only if distribution parameters are estimated from a sample.

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
    1. Moment estimators for Weibull parameters and their asymptotic efficiencies, Waloddi Weibull, April 1969, Lausanne Switzerland, Technical report AFML-TR-69-135

    2. Continuous univariate distributions, Volume 1, N.L.Johnson, S.Kotz and N.Balakrishnan, 1994,  John Wiley and sons inc.

    3. `weibull.com <http://www.weibull.com/hotwire/issue15/relbasics15.htm>`_, About location parameter

    4. `Plotting positions <http://en.wikipedia.org/wiki/Q%E2%80%93Q_plot>`_, About plotting positions

    5. `Bootstrapping <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>`_, Bootstrapping statistics

    6. Estimation of the generalized extreme value distribution by the method of probability weighted moments, Hosking, J. R. M., Wallis, J. R. and Wood, E. F., 1985, Technometrics, 27, pp. 251-261

    7. Estimating the three-parameter Weibull distribution by the method of probability weighted moments with application to medical survival data, Bortolucci, A. A. et.al.

    8. Theory and derivation for Weibull parameter probability weighted moment estimators, Grender, J.M., Dell, T.R., Reich, R.M., 1991 United Sates Department of Agriculture

    9. Probability weighted moments, Greenwood, J. A.; Landwehr, J.M.; Matalas, N.C.; Wallis, J.R., 1979, Water Resources Research. 15(5): 1049-1054.

    10. Probability weighted moments compared with some traditional techniques in estimating gumbel parameters and quantiles., Landwehr, J.M.; Matalas, N.C.; Wallis, J.R., 1979., Water Resources Research. 15(5): 1063-1064.

    """

    def __init__(self, loc, scale, shape, data=None):
        self.loc = loc
        self.scale = scale
        self.shape = shape

        if data is not None:
            self.data = np.array(data)
        else:
            self.data = None

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
        p = empirical_cdf(self.data.size, kind='mean')
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
        r1 = gamma(1. + 1. / self.shape)
        r2 = gamma(1. + 2. / self.shape)
        r3 = gamma(1. + 3. / self.shape)
        r4 = gamma(1. + 4. / self.shape)

        k = (r4 + r1 * (-4.0 * r3 + r1 * (6.0 * r2 - 3.0 * r1 ** 2))) / (r2 - r1 * r1)
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
        return self.loc + self.scale * gamma(1. + 1. / self.shape)

    @property
    def mse(self):
        """
        Mean squared error of fitted cumulative distribution (a,b,c) and empirical distribution

        Returns
        -------
        float
            mean squared error

        """
        e = np.sum((self.ecdf - self.cdf(np.sort(self.data))) ** 2.)/self.data.size
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
        r1 = gamma(1. + 1. / self.shape)
        r2 = gamma(1. + 2. / self.shape)
        r3 = gamma(1. + 3. / self.shape)
        s = (r3 + r1 * (2. * r1 ** 2. - 3. * r2)) / (r2 - r1 ** 2.) ** 1.5
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
        r1 = gamma(1. + 1. / self.shape)
        r2 = gamma(1. + 2. / self.shape)
        s = self.scale * np.sqrt(r2 - r1 ** 2.)
        return s

    def gumbel_parameters(self, n=None):
        """
        Calculate parameters of the asymptotic Gumbel extreme value distribution (Type 1) for the extreme value of N
        independent, Weibull distributed variables.

        Parameters
        ----------
        n : int
            number of independent weibull distributed variables, default equal to number of peaks (self.data.size)

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
            n = self.data.size
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
            x = np.linspace(self.loc, self.loc + 3. * self.std, 100)
        else:
            x = np.array(x)

        assert np.all(x >= self.loc), "The location parameter must be less than all items in data set"

        p = 1. - np.exp(-((x - self.loc) / self.scale) ** self.shape)
        return p

    @classmethod
    def fit(cls, data, method='msm', verbose=False):
        """
        Establish Weibull class instance by fit to sample.

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

        Returns
        -------
        Weibull
            Weibull class instance

        See Also
        --------
        Weibull.fromsignal

        Examples
        --------
        Assuming `data` is a sample array/list:

        >>> from qats.stats.weibull import Weibull
        >>> weib = Weibull.fit(data, method="msm")

        """
        options = {'msm': msm, 'lse': lse, 'mle': mle, 'pwm': pwm, 'pwm2': pwm2}
        assert method.lower() in options.keys(), "Method must be either %s" % (' or '.join(options.keys()))

        data = np.array(data)  # ensure numpy array

        params = options[method](data)
        if len(params) == 2:  # 2-parameter distribution
            loc = 0
            scale, shape = params
        else:                 # 3-parameter distribution
            loc, scale, shape = params

        if verbose:
            print("Fitted parameters:\n"
                  "location = %5.3g\n"
                  "scale = %5.3g\n"
                  "shape = %5.3g" % (loc, scale, shape)
                  )

        weib = cls(loc, scale, shape, data=data)
        return weib

    @classmethod
    def fromsignal(cls, x, method='msm', verbose=False):
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
        >>> sample = find_maxima(x, local=False, threshold=None, up=True, retind=False)
        >>> weib = Weibull.fit(sample, method='msm')
        """
        sample = find_maxima(x, local=False, threshold=None, up=True, retind=False)
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

        x[p == 1.] = np.inf  # asymptotic
        x[(p < 0.) | (p > 1.)] = np.nan  # probabilities out of bounds
        x[(p >= 0.) & (p < 1.)] = self.loc + self.scale * \
                                  (-np.log(1. - p[(p >= 0.) & (p < 1.)])) ** (1. / self.shape)
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
            x = np.linspace(self.loc, self.loc + 3. * self.std, 100)
        else:
            x = np.array(x)

        assert np.all(x >= self.loc), "The location parameter must be less than all items in data set"

        p = self.shape / self.scale * ((x - self.loc) / self.scale) ** (self.shape - 1.) * np.exp(
            -((x - self.loc) / self.scale) ** self.shape)
        return p

    def plot(self, filename=None):
        """
        Plot data on regular scales

        Parameters
        ----------
        filename : str, optional
            Save plot as `filename`, default is to show plot on screen

        Examples
        --------
        Plot distribution and show the figure

        >>> from qats.stats.weibull import Weibull
        >>> distribution = Weibull(100., 15., 2.5)
        >>> distribution.plot()

        Plot distribution and save the figure as png

        >>> from qats.stats.weibull import Weibull
        >>> distribution = Weibull(100., 15., 2.5)
        >>> distribution.plot(filename="plot.png")

        """
        plt.figure()

        # plot data if that exist
        if self.data is not None:
            plt.plot(np.sort(self.data), self.ecdf, 'ko', label='Data')

        # plot fitted/specified distribution
        y = np.linspace(0.001, 0.9999, 1000)
        x = self.invcdf(p=y)
        plt.plot(x, y, '-r', label='Fitted')
        plt.xlabel('X')
        plt.ylabel('Cumulative probability')
        plt.legend(loc='upper left')
        plt.grid(True)

        # show figure
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_linear(self, filename=None):
        """
        Plot data on Weibull paper (linearized scales))

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
        plt.figure()

        # plot data if that exists
        if self.data is not None:
            x = np.log(np.sort(self.data - self.loc))
            y = np.log(-np.log(1. - self.ecdf))
            plt.plot(x, y, 'ko', label='Data')

        # plot fitted distribution
        yy = np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
        y = np.log(-np.log(1. - yy))
        x = np.log(self.invcdf(p=yy) - self.loc)
        plt.plot(x, y, label='Fitted')

        plt.xlim(x[0], x[-1])
        plt.ylim(y[0], y[-1])
        plt.xlabel('X')
        plt.ylabel('Cumulative probability')
        plt.yticks(y, yy)
        xloc = np.linspace(x[0], x[-1], 5)
        xlab = np.around(np.exp(xloc) + self.loc, decimals=2)
        plt.xticks(xloc, xlab)
        plt.legend(loc='upper left')
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


def bootstrap(loc, scale, shape, size, repetitions, method='pwm'):
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
    options = {'msm': msm, 'lse': lse, 'pwm': pwm, 'mle': mle}
    assert method.lower() in options.keys(), "Method must be either %s" % (' or '.join(options.keys()))

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


def lse(x):
    """
    Fit Weibull distribution parameters to sample by method of least square fit to empirical cdf.

    Parameters
    ----------
    x : array_like
        sample data

    Returns
    -------
    tuple (floats)
        Distribution parameters ``(loc, scale, shape)``.

    Notes
    -----
    Uses what are known as (approximate) mean rank estimates for the empirical cdf.
    """
    x = np.sort(x)
    f = empirical_cdf(x.size, kind='mean')  # mean rank empirical cdf according to Weibull [1]
    fp = lambda v, z: 1. - np.exp(-((z - v[0]) / v[1]) ** v[2])  # parametric weibull function
    e = lambda v, z, y: (fp(v, z) - y)  # error function to be minimized
    a0, b0, c0 = msm(x)  # initial guess based on method of moments

    # least square fit
    p, cov, info, msg, ier = leastsq(e, np.asarray([a0, b0, c0]), args=(x, f), full_output=True)

    return p[0], p[1], p[2]


def mle(x):
    """
    Fit Weibull distribution parameters to sample by maximum likelihood estimation

    Parameters
    ----------
    x : array_like
        sample data

    Returns
    -------
    tuple (floats)
        Distribution parameters ``(loc, scale, shape)``.

    """

    def mle_eq(p, z):
        """
        MLE equation set according to [2, p. 656 eq. 21.73-21.75]

        Parameters
        ----------
        p : array_like
            distribution parameters a, b, c
        z : array_like
            data sample

        Returns
        -------
        tuple (floats)
            Estimates of distribution parameters ``(loc, scale, shape)``.
        """
        loc, scale, shape = p  # unpack parameters
        n = z.size

        f1 = n / shape + np.sum(np.log((z - loc) / scale)) - np.sum(
            ((z - loc) / scale) ** shape * np.log((z - loc) / scale))
        f2 = -n * shape / scale + shape / scale * np.sum(((z - loc) / scale) ** shape)
        f3 = -(shape - 1.) * np.sum(1. / (z - loc)) + shape / scale * np.sum(((z - loc) / scale) ** (shape - 1.))

        return f1, f2, f3

    x = np.array(x)
    a0, b0, c0 = msm(x)  # initial guess
    a, b, c = fsolve(mle_eq, np.asarray([a0, b0, c0]), args=x)

    return a, b, c


def mlj(sample, l, j):
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

    , where `X(F)` is the inverse form of the distribution and `F` is the cumulative distribution function.
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
    '''
    .. math:: M_{l,j,k} = E[X^l F^j (1-F)^k]
    .. math:: M_{l,j,0} = \frac{1}{n}\sum_{i=j+1}^{n}{X_{(i)}^l\frac{\binom{i-1}{j}}{\binom{n-1}{j}}}
    '''
    n = float(sample.size)
    xi = np.sort(sample)[j:]  # (j+1)th subsample of sorted sample
    ii = np.arange(j + 1, n + 1)

    # ratio of binomial coefficients, probability that xi is the largest observation in a subsample of size (j+1)
    # drawn from n.
    bc = np.array([binom(i - 1., j) / binom(n - 1., j) for i in ii])

    return (1. / n) * np.sum(xi ** l * bc)  # eq. 32 in [8]


def msm(x):
    """
    Fit Weibull distribution parameters to sample by method of sample moments

    Parameters
    ----------
    x : array_like
        sample data

    Returns
    -------
    floats
        The `loc`, `scale` and `shape` distribution parameters

    Notes
    -----
    See description in [1].
    """
    x = np.array(x)
    a1 = x.mean()  # first sample raw moment aka. mean
    m2 = x.var()  # second sample central moment aka. variance
    m3 = np.mean((x - a1) ** 3.)  # third sample central moment
    c1 = m3 / m2 ** (3. / 2.)  # coefficient of sample skewness

    # solve for shape parameter using root search, see [1, eq. 68 on p.11]
    def f(shape, shape0):
        eq = (gamma((shape + 3.) / shape) - 3. * gamma((shape + 1.) / shape) * gamma((shape + 2.) / shape) +
              2 * gamma((shape + 1.) / shape) ** 3.) / (gamma((shape + 2.) / shape) -
                                                        gamma((shape + 1.) / shape) ** 2.) ** (3. / 2.) - shape0
        return eq

    c = brentq(f, args=(c1,), a=0.1, b=1000., maxiter=1000)

    # calculate location and scale parameters, see [1, eq. 67 on p. 11]
    g1 = gamma((c + 1.) / c)
    g2 = gamma((c + 2.) / c)
    b = np.sqrt(m2 / (g2 - g1 ** 2.))
    a = a1 - g1 * b

    return a, b, c


def plot_fits(data, filename=None, method="pwm"):
    """
    Plot data sample versus empirical and fitted cumulative distribution function on linearized Weibull scales

    Parameters
    ----------
    data : array_like
        Data sample
    filename : str, optional
        Save plot as `filename`, default is to show plot on sc
    method : {'pwm', 'msm', 'lse', 'mle'}, optional
            Method of fit, default is 'pwm'
    """
    # fit distributions and plot
    options = {'msm': msm, 'lse': lse, 'pwm': pwm, 'mle': mle}
    assert method.lower() in options.keys(), "Method must be either %s" % (' or '.join(options.keys()))

    # estimate location and scale parameter
    loc, scale, shape = options[method](data)

    plt.figure()

    # sort data, create empirical distribution function and plot
    x = np.log(np.sort(data - loc))
    y = np.log(-np.log(1. - empirical_cdf(data.size, kind='mean')))
    plt.plot(x, y, 'ko', label='Data')

    # plot fitted distribution
    yy = np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
    y = np.log(-np.log(1. - yy))
    x = np.log(scale * (-np.log(1. - yy)) ** (1. / shape))
    plt.plot(x, y, label=method)

    # adjust figure
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.xlabel('X')
    plt.ylabel('Cumulative probability')
    plt.yticks(y, yy)
    xloc = np.linspace(x[0], x[-1], 5)
    xlab = np.around(np.exp(xloc) + loc, decimals=2)
    plt.xticks(xloc, xlab)
    plt.legend(loc='upper left')
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    # close figure to avoid high memory consumption
    plt.close()


def pwm(x):
    """
    Fit distribution parameters to sample by method of probability weighted
    moments

    Parameters
    ----------
    x : array_like
        sample data

    Returns
    -------
    floats
        The `loc`, `scale` and `shape` distribution parameters

    Notes
    -----
    Details on probability weighted moments are provided in [8].

    See Also
    --------
    mlj()

    """
    # probability weighted moments of order 0,1,2,3
    m100 = mlj(x, 1, 0)
    m110 = mlj(x, 1, 1)
    m120 = mlj(x, 1, 2)
    m130 = mlj(x, 1, 3)

    # parameter estimates
    c = np.log(2.) / np.log((2. * m110 - m100) / (2. * (5. * m110 - m100 - 6. * m120 + 2. * m130)))
    a = 4. * (m100 * (3. * m120 - m130 - m110) - m110 ** 2.) / (m100 - 8. * m110 + 12. * m120 - 4. * m130)
    b = (m100 - a) / gamma(1. + 1. / c)

    return a, b, c


def pwm2(x):
    """
    Fit distribution parameters to sample by method of probability weighted moments assuming the location parameter
    is zero.

    Parameters
    ----------
    x : array_like
        sample data

    Returns
    -------
    floats
        The `scale` and `shape` distribution parameters

    Notes
    -----
    Details on probability weighted moments are provided on p.14-15 in [8]. Note that only the scale and parameters are
    estimated, the location parameter is assumed zero.

    See Also
    --------
    mlj(), pwm()

    """
    # probability weighted moments of order 0,1
    m100 = mlj(x, 1, 0)
    m110 = mlj(x, 1, 1)

    # parameter estimates
    c = np.log(2.) / np.log(m100 / (2.*(m100 - m110)))
    b = m100 / gamma(1. + 1. / c)

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
    gloc = loc + scale * np.log(n) ** (1. / shape)
    gscale = 1. / (shape / scale * np.log(n) ** ((shape - 1.) / shape))

    return gloc, gscale
