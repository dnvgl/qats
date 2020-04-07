#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:class:`GumbelMin` class and functions related to Gumbel (minima) distribution.
"""
import numpy as np
from scipy.special import zetac
from scipy.optimize import leastsq, fsolve
from matplotlib.pyplot import figure, ylabel, yticks, plot, legend, grid, show, xlabel, ylim, savefig
from .empirical import empirical_cdf
from .gumbel import _euler_masceroni as em


# todo: move fit methods e.g. _msm from class to standalone functions (importable)
# todo: check fit procedures (read up once more and check implementation)
# todo: create unit tests


class GumbelMin(object):
    """
    The Gumbel minima distribution.

    The cumulative distribution function is defined as::

        F(x) = 1 - exp{-exp[(x-a)/b]}

    where `a` is location parameter and `b` is the scale parameter.

    Parameters
    ----------
    loc : float
        Gumbel location parameter.
    scale : float
        Gumbel scale parameter.
    data : array_like, optional
        Sample data, used to establish empirical cdf and is included in plots.
        To fit the Gumbel distribution to the sample data, use :meth:`GumbelMin.fit`.

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

    >>> from qats.stats.gumbelmin import GumbelMin
    >>> gumb = GumbelMin(loc, scale)

    If you need to establish a Gumbel instance based on a sample data set, use:

    >>> gumb = GumbelMin.fit(data, method='msm')

    References
    ----------
    1. Bury, K.V. (1975) Statistical models in applied science. Wiley, New York

    2. Haver, S. (2007), "Bruk av asymptotiske ekstremverdifordelinger"

    3. `Plotting positions <http://en.wikipedia.org/wiki/Q%E2%80%93Q_plot>`_, About plotting positions

    4. `Usable estimators for parameters in Gumbel distribution
       <http://stats.stackexchange.com/questions/71197/usable-estimators-for-parameters-in-gumbel-distribution>`_

    5. `Bootstrapping statistics <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>`_

    """

    def __init__(self, loc=None, scale=None, data=None):
        self.location = loc
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
        c : float
            distribution c.o.v.
        """
        return self.std / self.mean

    @property
    def ecdf(self):
        """
        Median rank empirical cumulative distribution function associated with the sample

        Notes
        -----
        Gumbel recommended the following mean rank quantile formulation Pi = i/(n+1).
        This formulation produces a symmetrical CDF in the sense that the
        same plotting positions will result from the data regardless of
        whether they are assembled in ascending or descending order.

        A more sophisticated median rank formulation Pi = (i-0.3)/(n+0.4) approximates the
        median of the distribution free estimate of the sample variate to about
        0.1% and, even for small values of n, produces parameter estimations
        comparable to the result obtained by maximum likelihood estimations (Bury, 1999, p43)
        A median rank method, pi=(i-0.3)/(n+0.4), is chosen to approximate the mean of the distribution [2]

        The empirical cdf is also used as plotting positions when plotting the sample
        on probability paper.
        """
        try:
            p = empirical_cdf(self.data.size, kind='median')
            return p
        except TypeError:
            print("The sample is not defined.")

    @property
    def kurt(self):
        """
        Distribution kurtosis

        Returns
        -------
        k : float
            distribution kurtosis
        """
        try:
            k = 12. / 5.
            return k
        except TypeError:
            print("Distribution parameters are not defined.")

    @property
    def mean(self):
        """
        Distribution mean value

        Returns
        -------
        m : float
            distribution mean value
        """
        try:
            m = self.location - self.scale * em()
            return m
        except TypeError:
            print("Distribution parameters are not defined.")

    @property
    def median(self):
        """
        Distribution median value

        Returns
        -------
        m : float
            distribution median value
        """
        try:
            m = self.location + self.scale * np.log(np.log(2.))
            return m
        except TypeError:
            print("Distribution parameters are not defined.")

    @property
    def mode(self):
        """
        Distribution mode value

        Returns
        -------
        m : float
            distribution mode value
        """
        try:
            m = self.location
            return m
        except TypeError:
            print("Distribution parameters are not defined.")

    @property
    def std(self):
        """
        Distribution standard deviation

        Returns
        -------
        s : float
            distribution standard deviation
        """
        try:
            s = np.pi * self.scale / np.sqrt(6)
            return s
        except TypeError:
            print("Distribution parameters are not defined.")

    @property
    def skew(self):
        """
        Distribution skewness

        Returns
        -------
        s : float
            distribution skewness
        """
        try:
            # zetac is the complementary Riemann zeta function (zeta function minus 1)
            # http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.zetac.html
            s = -12. * np.sqrt(6.) * (1. + zetac(3)) / np.pi ** 3
            return s
        except TypeError:
            print("Distribution parameters are not defined.")

    def bootstrap(self, size=None, method='msm', N=100):
        """
        Parametric bootstrapping of source distribution

        Parameters
        ----------
        size : int
            bootstrap sample size. default equal to source sample size
        method : {'msm','lse','mle'}
            method of fit, optional
            'msm' = method of sample moments
            'lse' = least-square estimation
            'mle' = maximum likelihood estimation
        N : int
            number of bootstrap samples. default equal to 10

        Returns
        -------
        array-like
            m  - mean distribution parameters
        array_like
            cv - coefficient of variation of distribution parameter

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

        """
        options = {'msm': self._msm, 'lse': self._lse, 'mle': self._mle}
        assert method.lower() in options.keys(), "Method must be either %s" % (' or '.join(options.keys()))

        if size is None:
            assert self.data is not None, "Either size has to be specified or a sample has to be specified."
            size = np.size(self.data)

        i = 0
        par = np.zeros((N, 2))
        while (i < N):
            x = self.rnd(size=size)
            par[i, :] = options[method](x)
            i += 1

        m = par.mean(axis=0)
        cv = par.std(axis=0, ddof=1) / m

        return m, cv

    def cdf(self, x=None):
        """
        Cumulative distribution function (cumulative probability) for specified values x

        Parameters
        ----------
        x : array_like
            values

        Returns
        -------
        cdf : array
            cumulative probabilities for specified values x

        Notes
        -----
        A range of x values [location, location+3*std] are applied if x is not specified.
        """
        try:
            if x is None:
                x = np.linspace(self.loc, self.loc - 3. * self.std, 100)
            else:
                x = np.array(x)

            assert self.scale > 0., "The scale parameter must be larger than 0."

            z = (x - self.location) / self.scale
            p = 1. - np.exp(-np.exp(z))
            return p
        except TypeError:
            print("Distribution parameters are not defined")

    def fit(self, data=None, method='msm', verbose=False):
        """
        Determine distribution parameters by fit to sample.

        Parameters
        ----------
        data : array_like
            sample, optional
        method : {'msm','lse','mle'}
            method of fit, optional
            'msm' = method of sample moments
            'lse' = least-square estimation
            'mle' = maximum likelihood estimation
        verbose : bool
            turn on output of fitted parameters

        Notes
        -----
        If data is not input any data stored in object (self.data) will be used.

        """

        options = {'msm': msm, 'lse': lse, 'mle': mle}
        assert method.lower() in options.keys(), "Method must be either %s" % (' or '.join(options.keys()))

        if data is not None:
            # update sample data
            self.data = np.array(data).reshape(np.shape(data))  # make vector shaped

        try:
            self.location, self.scale = options[method](self.data)
            if verbose:
                print("Fitted parameters:\nlocation = %(location)5.3g\nscale = %(scale)5.3g" % self.__dict__)
        except TypeError:
            print("The sample data is not defined.")

    def fit_from_weibull_parameters(self, wa, wb, wc, n, verbose=False):
        """
        Calculate Gumbel distribution parameters from n independent Weibull distributed variables.

        Parameters
        ----------
        wa : float
            Weibull location parameter
        wb : float
            Weibull scale parameter
        wc : float
            Weibull shape parameter
        n : int
            Number independently distributed variables
        verbose : bool
            print fitted parameters

        Notes
        -----
        A warning is issued if Weibull shape parameter less than 1. In this case,
        the convergence towards asymptotic extreme value distribution is slow
        , and the asymptotic distribution will be non-conservative relative
        to the exact distribution. The asymptotic distribution is correct with Weibull
        shape equal to 1 and conservative with Weibull shape larger than 1.
        These deviations diminish with larger samples. See [1, p. 380].

        """
        # TODO:
        #  self.location = wa + wb * np.log(n) ** (1. / wc)
        #  self.scale = 1. / (wc / wb * np.log(n) ** ((wc - 1.) / wc))
        #  if verbose:
        #     print "Fitted parameters:\nlocation = %(location)5.3g\nscale = %(scale)5.3g" % self.__dict__
        raise NotImplementedError("Formula for deriving Gumbel minimum distribution parameter is not implemented.")

    def gp_plot(self, showfig=True, save=None):
        """
        Plot data on Gumbel paper (linearized scales))

        Parameters
        ----------
        showfig : bool
            show figure immediately on screen, default True
        save : filename
            save figure to file, default None

        """

        figure()

        x = np.sort(self.data)

        # sample
        z_data = -np.log(-np.log(self.ecdf))
        plot(x, z_data, 'ko', label='Data')

        # fit distributions
        a_msm, b_msm = msm(self.data)
        a_mle, b_mle = mle(self.data)
        a_lse, b_lse = lse(self.data)

        z_msm = (x - a_msm) / b_msm
        z_mle = (x - a_mle) / b_mle
        z_lse = (x - a_lse) / b_lse

        plot(x, z_msm, '-r', label='MSM')
        plot(x, z_mle, '--g', label='MLE')
        plot(x, z_lse, ':b', label='LSE')

        # plotting positions and plot configurations
        p = np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
        y = -np.log(-np.log(p))
        yticks(y, p)
        legend(loc='upper left')
        ylim(y[0], y[-1])
        xlabel('X')
        ylabel('Cumulative probability')
        grid(True)

        if showfig:
            show()
        elif save is not None:
            savefig(save)
        else:
            pass

    def invcdf(self, p=None):
        """
        Inverse cumulative distribution function for specified quantiles p

        Parameters
        ----------
        p : array_like
            quantiles (or. cumulative probabilities if you like)

        Returns
        -------
        x : array
            values corresponding to the specified quantiles

        Notes
        -----
        A range of quantiles from 0.001 to 0.999 are applied if quantiles are not specified
        """
        try:
            if p is None:
                p = np.linspace(0.001, 0.999, 100)
            else:
                p = np.array(p)

            assert self.scale > 0., "The scale parameter must be larger than 0."

            x = np.zeros(np.shape(p))

            x[p == 1.] = np.inf  # asymptotic
            x[(p < 0.) | (p > 1.)] = np.nan  # probabilities out of bounds
            z = (p >= 0.) & (p < 1.)  # valid quantile range
            x[z] = self.location + self.scale * np.log(-np.log(1.-p[z]))
            return x
        except TypeError:
            print("Distribution parameters are not defined")

    def pdf(self, x=None):
        """
        Probability density function for specified values x

        Parameters
        ----------
        x : array_like
            values

        Returns
        -------
        pdf : array
            probability density function for specified values x

        Notes
        -----
        A range of x values [location, location+3*std] are applied if x is not specified.
        """
        try:
            if x is None:
                x = np.linspace(self.loc, self.loc - 3. * self.std, 100)
            else:
                x = np.array(x)

            assert self.scale > 0., "The scale parameter must be larger than 0."

            z = (x - self.location) / self.scale
            p = (1. / self.scale) * np.exp(z - np.exp(z))
            return p
        except TypeError:
            print("Distribution parameters are not defined")

    def plot(self, showfig=True, save=None):
        """
        Plot data on regular scales

        Parameters
        ----------
        showfig : bool
            show figure immediately on screen, default True
        save : filename including suffix
            save figure to file, default None

        """

        figure()

        if self.data is not None:
            plot(np.sort(self.data), self.ecdf, 'ko', label='Data')

        y = np.linspace(0.001, 0.9999, 1000)
        x = self.invcdf(p=y)

        plot(x, y, '-r', label='Fitted')
        xlabel('X')
        ylabel('Cumulative probability')
        legend(loc='upper left')
        grid(True)

        if showfig:
            show()
        elif save is not None:
            savefig(save)
        else:
            pass

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
        x : array
            random sample

        """

        if seed is not None:
            np.random.seed(seed)
        r = np.random.random_sample(size)
        x = self.invcdf(p=r)

        return x


def lse(x):
    """
    Fit distribution parameters to sample by method of least square fit to empirical cdf

    Parameters
    ----------
    x : array_like
        sample

    Notes
    -----
    Uses an approximate median rank estimate for the empirical cdf.
    """
    x = np.sort(x)
    f = empirical_cdf(x.size, kind='median')
    fp = lambda v, z: 1. - np.exp(-np.exp((z - v[0]) / v[1]))  # parametric Gumbel function
    e = lambda v, z, y: (fp(v, z) - y)  # error function to be minimized
    a0, b0 = msm(x)  # initial guess based on method of moments

    # least square fit
    p, cov, info, msg, ier = leastsq(e, [a0, b0], args=(x, f), full_output=1)

    return p[0], p[1]


def mle(x):
    """
    Fit distribution parameters to sample by maximum likelihood estimation

    Parameters
    ----------
    x : array_like
        sample

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
            data
        """
        loc, scale = p  # unpack parameters
        n = z.size

        out = [loc + scale * np.log(1. / n * np.sum(np.exp(z / scale))),
               z.mean() - np.sum(z * np.exp(z / scale)) / np.sum(np.exp(z / scale)) - scale]

        return out

    x = np.array(x)
    a0, b0 = msm(x)  # initial guess
    a, b = fsolve(mle_eq, [a0, b0], args=x)

    return a, b


def msm(x):
    """
    Fit distribution parameters to sample by method of sample moments

    Parameters
    ----------
    x : array_like
        sample

    Notes
    -----
    See description in [1] and [2].
    """
    x = np.array(x)

    b = np.sqrt(6.) * np.std(x, ddof=1) / np.pi  # using the unbiased sample standard deviation
    a = x.mean() + em() * b

    return a, b
