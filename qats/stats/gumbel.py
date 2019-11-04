#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:class:`Gumbel` class and functions related to Gumbel distribution.
"""
import numpy as np
from scipy.special import zetac, binom
from scipy.optimize import leastsq, fsolve
import matplotlib.pyplot as plt
from .empirical import empirical_cdf


# todo: build documentation and check that docstrings behave as intended
# todo: create examples
# todo: consider replacing latex with python code or pseudo code (ref. link below)
# link: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#other-points-to-keep-in-mind


class Gumbel(object):
    """
    The Gumbel maxima distribution.

    The cumulative distribution function is defined as::

        F(x) = exp{-exp[-(x-a)/b]}

    where `a` is location parameter and `b` is the scale parameter.

    Parameters
    ----------
    loc: float
        Gumbel location parameter.
    scale: float
        Gumbel scale parameter.
    data: array_like, optional
        Sample data, used to establish empirical cdf and is included in plots.
        To fit the Gumbel distribution to the sample data, use :meth:`Gumbel.fit`.

    Attributes
    ----------
    loc : float
        Gumbel location parameter.
    scale : float
        Gumbel scale parameter.
    data : array_like
        Sample data.

    Examples
    --------
    To initiate an instance based on parameters, use:

    >>> from qats.stats.gumbel import Gumbel
    >>> gumb = Gumbel(loc, scale)

    If you need to establish a Gumbel instance based on a sample data set, use:

    >>> gumb = Gumbel.fit(data, method='msm')


    References
    ----------
    1. Statistical models in applied science., Bury, K.V. (1975), Wiley, New York

    2. Bruk av asymptotiske ekstremverdifordelinger, Haver, S. (2007)

    3. `Plotting positions <http://en.wikipedia.org/wiki/Q%E2%80%93Q_plot>`_, About plotting positions

    4. `Usable estimators for parameters in Gumbel distribution
       <http://stats.stackexchange.com/questions/71197/usable-estimators-for-parameters-in-gumbel-distribution>`_

    5. `Bootstrapping statistics <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>`_

    6. Probability weighted moments, Greenwood, J. A.; Landwehr, J.M.; Matalas, N.C.; Wallis, J.R., 1979,
       Water Resources Research. 15(5): 1049-1054.

    7. Probability weighted moments compared with some traditional techniques in estimating gumbel parameters and
       quantiles., Landwehr, J.M.; Matalas, N.C.; Wallis, J.R., 1979., Water Resources Research. 15(5): 1063-1064.

    """

    def __init__(self, loc, scale, data=None):
        assert loc is not None, "Location parameter must be finite."
        assert scale is not None and scale > 0, "Scale parameter must be finit and larger than 0."
        self.loc = loc
        self.scale = scale

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
            Distribution c.o.v.
        """
        return self.std / self.mean

    @property
    def ecdf(self):
        """
        Median rank empirical cumulative distribution function associated with the sample

        Returns
        -------
        array
            Empirical cumulative distribution function

        Notes
        -----
        Requires data/sample to be specified.

        Gumbel recommended the following mean rank quantile formulation ``Pi = i/(n+1)``.
        This formulation produces a symmetrical CDF in the sense that the
        same plotting positions will result from the data regardless of
        whether they are assembled in ascending or descending order.

        A more sophisticated median rank formulation ``Pi = (i-0.3)/(n+0.4)`` approximates the
        median of the distribution free estimate of the sample variate to about
        0.1% and, even for small values of n, produces parameter estimations
        comparable to the result obtained by maximum likelihood estimations (Bury, 1999, p43).
        A median rank method, ``pi=(i-0.3)/(n+0.4)``, is chosen to approximate the mean
        of the distribution [2].

        The empirical cdf is also used as plotting positions when plotting the sample
        on probability paper.
        """
        assert self.data is not None, "Requires data/sample to be specified."
        return empirical_cdf(self.data.size, kind='median')

    @property
    def kurt(self):
        """
        Distribution kurtosis

        Returns
        -------
        float
            Distribution kurtosis
        """
        return 12. / 5.

    @property
    def mean(self):
        """
        Distribution mean value

        Returns
        -------
        float
            Distribution mean value
        """
        return self.loc + self.scale * _euler_masceroni()

    @property
    def median(self):
        """
        Distribution median value

        Returns
        -------
        float
            Distribution median value
        """
        return self.loc - self.scale * np.log(np.log(2.))

    @property
    def mode(self):
        """
        Distribution mode value

        Returns
        -------
        float
            Distribution mode value
        """
        return self.loc

    @property
    def mse(self):
        """
        Mean squared error of fitted cumulative distribution (a,b,c) and empirical distribution

        Returns
        -------
        float
            mean squared error

        Notes
        -----
        Requires data/sample to be specified.

        """
        return np.sum((self.ecdf - self.cdf(np.sort(self.data))) ** 2.) / self.data.size

    @property
    def params(self):
        """
        Distribution parameters.

        Returns
        -------
        tuple
            Distribution parameters: (loc, scale).
        """
        return self.loc, self.scale

    @property
    def std(self):
        """
        Distribution standard deviation

        Returns
        -------
        float
            Distribution standard deviation
        """
        return np.pi * self.scale / np.sqrt(6)

    @property
    def skew(self):
        """
        Distribution skewness

        Returns
        -------
        float
            Distribution skewness

        Notes
        -----
        `zetac` is the complementary Riemann zeta function (zeta function minus 1).
        See http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zetac.html

        """
        return 12. * np.sqrt(6.) * (1. + zetac(3)) / np.pi ** 3

    def cdf(self, x=None):
        """
        Cumulative distribution function (cumulative probability) for specified values x

        Parameters
        ----------
        x : array_like, optional
            Calculate cumulative probability for these values

        Returns
        -------
        array
            Cumulative probabilities for specified values x

        Notes
        -----
        A range of x values [loc, loc+3*std] are applied if x is not specified.

        """
        assert self.scale > 0., "The scale parameter must be larger than 0."

        if x is None:
            x = np.linspace(self.loc, self.loc + 3. * self.std, 100)
        else:
            x = np.array(x)

        z = (x - self.loc) / self.scale
        p = np.exp(-np.exp(-z))
        return p

    @classmethod
    def fit(cls, data, method='msm', verbose=False):
        """
        Determine distribution parameters by fit to sample.

        Parameters
        ----------
        data : array_like
            Sample
        method : str, optional
            Method of fit. Options:

            - ``msm`` = method of sample moments (default)
            - ``lse`` = least-square estimation
            - ``mle`` = maximum likelihood estimation
            - ``pwm`` = probability weighted moments

        verbose : bool, optional
            If true, fitted parameters are written to screen.

        Examples
        --------
        Assuming `data` is a sample array/list:

        >>> from qats.stats.gumbel import Gumbel
        >>> gumb = Gumbel.fit(data, method="msm")

        """
        options = {'msm': msm, 'lse': lse, 'pwm': pwm, 'mle': mle}
        assert method.lower() in options.keys(), "Method must be either %s" % (' or '.join(options.keys()))

        data = np.array(data)  # ensure numpy array

        loc, scale = options[method](data)
        if verbose:
            print("Fitted parameters:\nloc = %5.3g\nscale = %5.3g" % (loc, scale))

        gumb = cls(loc, scale, data=data)

        return gumb

    @classmethod
    def fit_from_weibull_parameters(cls, wa, wb, wc, n, verbose=False):
        """
        Calculate Gumbel distribution parameters from n independent Weibull distributed variables.

        Parameters
        ----------
        wa : float
            Weibull loc parameter
        wb : float
            Weibull scale parameter
        wc : float
            Weibull shape parameter
        n : int
            Number independently distributed variables
        verbose : bool
            Print fitted parameters

        Notes
        -----
        A warning is issued if Weibull shape parameter less than 1. In this case,
        the convergence towards asymptotic extreme value distribution is slow
        , and the asymptotic distribution will be non-conservative relative
        to the exact distribution. The asymptotic distribution is correct with Weibull
        shape equal to 1 and conservative with Weibull shape larger than 1.
        These deviations diminish with larger samples. See [1, p. 380].

        References
        ----------
        1. Bury, Karl V., 1975, "Statistical Models in Applied Science", University of British Columbia, John Wiley & Sons

        Examples
        --------
        >>> from qats.stats.gumbel import Gumbel
        >>> gumb = Gumbel.fit_from_weibull_parameters(wa, wb, wc, n)
        """

        loc = wa + wb * np.log(n) ** (1. / wc)  # eq. 11.41 on page 380 in [1]
        scale = (wb / wc) * np.log(n) ** ((1.-wc) / wc)  # eq. 11.41 on page 380 in [1]

        if verbose:
            print("Fitted parameters:\nloc = %5.3g\nscale = %5.3g" % (loc, scale))

        gumb = cls(loc, scale)

        return gumb

    def invcdf(self, p=None):
        """
        Inverse cumulative distribution function for specified probabilities

        Parameters
        ----------
        p : array_like, optional
            Calculate the inverse cumulative distribution function for these probabilities

        Returns
        -------
        array
            Values corresponding to the specified quantiles

        Notes
        -----
        A range of quantiles from 0.001 to 0.999 are applied if quantiles are not specified

        """
        assert self.scale > 0., "The scale parameter must be larger than 0."

        if p is None:
            p = np.linspace(0.001, 0.999, 100)
        else:
            p = np.array(p)

        x = np.zeros(np.shape(p))

        x[p == 1.] = np.inf  # asymptotic
        x[(p < 0.) | (p > 1.)] = np.nan  # probabilities out of bounds
        z = (p >= 0.) & (p < 1.)  # valid quantile range
        x[z] = self.loc - self.scale * np.log(-np.log(p[z]))

        return x

    def pdf(self, x=None):
        """
        Probability density function for specified values x

        Parameters
        ----------
        x : array_like, optional
            Cumulative probabilities for specified values x

        Returns
        -------
        array
            Calculate probability density for these values x

        Notes
        -----
        A range of x values [loc, loc+3*std] are applied if x is not specified.

        """
        assert self.scale > 0., "The scale parameter must be larger than 0."

        if x is None:
            x = np.linspace(self.loc, self.loc + 3. * self.std, 100)
        else:
            x = np.array(x)

        z = (x - self.loc) / self.scale
        p = (1. / self.scale) * np.exp(-z - np.exp(-z))
        return p

    def plot(self, filename=None):
        """
        Plot cumulative distribution function

        Parameters
        ----------
        filename : str, optional
            Save plot as `filename`, default is to show plot on screen

        """
        plt.figure()

        # plot sample versus empirical distribution
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

        if filename is not None:
            plt.savefig(filename)

        else:
            plt.show()

        # close figure to avoid high memory consumption
        plt.close()

    def plot_linear(self, filename=None):
        """
        Plot cumulative distribution function on linearized Gumbel scales

        Parameters
        ----------
        filename : str, optional
            Save plot as `filename`, default is to show plot on screen

        """
        plt.figure()

        # plot sample versus empirical distribution function
        if self.data is not None:
            x = np.sort(self.data)
            z = -np.log(-np.log(self.ecdf))
            plt.plot(x, z, 'ko', label='Data')

        # plot distribution
        x = self.invcdf(p=np.linspace(0.001, 0.9999, 1000))
        z = (x - self.loc) / self.scale
        plt.plot(x, z, '-r', label='Fitted')

        # plotting positions and plot configurations
        p = np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
        y = -np.log(-np.log(p))
        plt.yticks(y, p)
        plt.legend(loc='upper left')
        plt.ylim(y[0], y[-1])
        plt.xlabel('X')
        plt.ylabel('Cumulative probability')
        plt.grid(True)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        # close figure to avoid high memory consumption
        plt.close()

    def rnd(self, size=None, seed=None):
        """
        Draw random samples from probability distribution

        Parameters
        ----------
        size : int|numpy shape, optional
            Sample size (default 1 random value is returned)
        seed : int, optional
            Seed for random number generator (default seed is random)

        Returns
        -------
        array
            Random sample

        Examples
        --------
        Pick 1000 values randomly from a Gumbel distribution

        >>> from qats.stats.gumbel import Gumbel
        >>> g = Gumbel(loc, scale)
        >>> sample = g.rnd(size=1000)

        If you want to preset the seed for the random sampling (to be able to repeat the sampling)

        >>> from qats.stats.gumbel import Gumbel
        >>> g = Gumbel(loc, scale)
        >>> sample = g.rnd(size=1000, seed=3)

        """
        if seed is not None:
            np.random.seed(seed)
        r = np.random.random_sample(size)
        x = self.invcdf(p=r)

        return x


def _euler_masceroni():
    """
    The Euler-Masceroni constant

    Returns
    -------
    float
        The Euler-Mascheroni constant

    Notes
    -----
    The Euler–Masceroni constant (also called Euler's constant) is a
    mathematical constant recurring in analysis and number theory. It
    is defined as the limiting difference between the harmonic series and
    the natural logarithm.

    """
    return 0.57721566490153286060651209008240243104215933593992


def bootstrap(loc, scale, size, repetitions, method='pwm'):
    """
    Quantify mean and coefficient of variation of Gumbel distribution parameters using parametric bootstrapping

    Parameters
    ----------
    loc : float
        Source distribution location parameter
    scale : float
        Source distribution scale parameter
    size : int
        Size of bootstrapped sample
    method : str, optional
        method of fit, optional
        'msm' = method of sample moments
        'lse' = least-square estimation
        'mle' = maximum likelihood estimation
        'pwm' = probability weighted moments (default)
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
    to sample estimates (variance, quantiles). This technique allows estimation of the
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
    To quantify the uncertainty (coefficient of variation) of a Gumbel distribution fitted to a sample with 5 values
    (using 100 repetition):

    >>> from qats.stats.gumbel import bootstrap
    >>> m, cv = bootstrap(10, 2.5, 5, 100)

    """
    options = {'msm': msm, 'lse': lse, 'pwm': pwm, 'mle': mle}
    assert method.lower() in options.keys(), "Method must be either %s" % (' or '.join(options.keys()))

    # initiate distribution
    distribution = Gumbel(loc, scale)

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
    Fit Gumbel distribution parameters to sample by method of least square fit to empirical cdf

    Parameters
    ----------
    x : array_like
        data sample

    Returns
    -------
    floats
        distribution `loc` and `scale` parameters

    Notes
    -----
    Uses an approximate median rank estimate for the empirical cdf.
    """
    x = np.sort(x)
    f = empirical_cdf(x.size, kind='median')  # median rank cdf
    fp = lambda v, z: np.exp(-np.exp(-(z - v[0]) / v[1]))  # parametric Gumbel function
    e = lambda v, z, y: (fp(v, z) - y)  # error function to be minimized
    a0, b0 = msm(x)  # initial guess based on method of moments

    # least square fit
    p, cov, info, msg, ier = leastsq(e, np.array([a0, b0]), args=(x, f), full_output=True)

    return p[0], p[1]


def mle(x):
    """
    Fit distribution parameters to sample by maximum likelihood estimation

    Parameters
    ----------
    x : array_like
        data sample

    Returns
    -------
    floats
        distribution `loc` and `scale` parameters

    Notes
    -----
    MLE equation set is given in 'Statistical Distributions' by Forbes et.al. (2010) and referred
    at [4]
    """

    def mle_eq(p, z):
        """
        MLE equation set

        Parameters
        ----------
        p : list
            distribution parameters a, b, c
        z : array_like
            data sample
        """
        loc, scale = p  # unpack parameters
        n = z.size

        out = [loc + scale * np.log(1. / n * np.sum(np.exp(-z / scale))),
               z.mean() - np.sum(z * np.exp(-z / scale)) / np.sum(np.exp(-z / scale)) - scale]

        return out

    x = np.array(x)
    a0, b0 = msm(x)  # initial guess
    a, b = fsolve(mle_eq, np.array([a0, b0]), args=x)

    return a, b


def msm(x):
    """
    Fit Gumbel distribution parameters to sample by method of sample moments

    Parameters
    ----------
    x : array_like
        data sample

    Returns
    -------
    floats
        distribution `loc` and `scale` parameters

    Notes
    -----
    See description in [1] and [2].
    """
    x = np.array(x)

    b = np.sqrt(6.) * np.std(x, ddof=1) / np.pi  # using the unbiased sample standard deviation
    a = x.mean() - _euler_masceroni() * b

    return a, b


def plot_fits(data, filename=None, methods=None):
    """
    Plot data sample versus empirical and fitted cumulative distribution function on linearized Gumbel scales

    Parameters
    ----------
    data : array_like
        Data sample
    filename : str, optional
        Save plot as `filename`, default is to show plot on sc
    methods : tuple, optional
        Methods of fit. Options (default all):

        - ``msm`` = method of sample moments
        - ``lse`` = least-square estimation
        - ``mle`` = maximum likelihood estimation
        - ``pwm`` = probability weighted moments
    """
    plt.figure()

    # sort data, create empirical distribution function and plot
    x = np.sort(data)
    z = -np.log(-np.log(empirical_cdf(x.size, kind='median')))
    plt.plot(x, z, 'ko', label='Data')

    # fit distributions and plot
    options = {'msm': msm, 'lse': lse, 'pwm': pwm, 'mle': mle}

    # default if not specified
    if not methods:
        methods = options.keys()

    for method in methods:
        assert method.lower() in options.keys(), "Method must be either %s" % (' or '.join(options.keys()))

        # estimate location and scale parameter
        loc, scale = options[method](x)

        # linarize and plot
        z = (x - loc) / scale
        plt.plot(x, z, label=method)

    # adjust plotting positions and plot configurations
    p = np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
    y = -np.log(-np.log(p))
    plt.yticks(y, p)
    plt.legend(loc='upper left')
    plt.ylim(y[0], y[-1])
    plt.xlabel('data')
    plt.ylabel('Cumulative probability')
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    # close figure to avoid high memory consumption
    plt.close()


def pwm(x):
    """
    Fit Gumbel distribution parameters to sample by method of probability weighted
    moments [7].

    Parameters
    ----------
    x : array_like
        data sample

    Returns
    -------
    tuple
        distribution parameters `location` and `scale`

    """

    def mk(z, k):
        """
        Probability weighted moments with emphasize on the left/lower tail (j=0).

        Parameters
        ----------
        z   :   array
            sample
        k   :   order of complementary cumulative distribution function

        Notes
        -----
        The probability weighted moment Mljk is defined by Greenwood and others (1979)

        .. math::

            M_{l,j,k} = E[X^l F^j (1-F)^k]

        , where X(F) is the inverse form of the distribution and F is the cumulative distribution function.
        When j=k=0 and l is a non-negative integer then M_{l,0,0} represents the conventional moment of order l about
        the origin.

        PWMs can be applied either when the small observations are more important than the large observations (k=0), as in
        strength properties of materials, or when the large observations should have more influence than the smaller
        observations (k=0) as with three diameter distribution modelling. Here we have chose the former and derived
        unbiased estimators for moments M_{1,0,k} (j=0), see eq. 16 in [6].

        .. math::

            M_{1,0,k} = \frac{1}{n}\sum_{i=1}^{n}{X_{(i)}\frac{\binom{n-i}{k}}{\binom{n-1}{k}}}

        """
        n = float(z.size)
        xi = np.sort(z)
        ii = np.arange(1, n + 1)

        # ratio of binomial coefficients
        bc = np.array([binom(n - i, k) / binom(n - 1., k) for i in ii])

        return (1. / n) * np.sum(xi * bc)  # eq. 16 in [6]

    m0 = mk(x, 0.)
    m1 = mk(x, 1.)

    b = (m0 - 2. * m1) / np.log(2.)
    a = m0 - _euler_masceroni() * b

    return a, b

