#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides :class:`TimeSeries` class.
"""
import copy
import os
from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew, tstd
import matplotlib.pyplot as plt
from matplotlib import cm
from .fatigue.rainflow import count_cycles, rebin as rebin_cycles, mesh
from .signal import lowpass, highpass, bandblock, bandpass, threshold as thresholdpass, smooth, taper, \
    average_frequency, find_maxima, psd
from .stats.weibull import Weibull, weibull2gumbel, pwm
from .stats.gumbel import Gumbel


# todo: weibull and gumbel + plotting (self.pd = Weibull(), self.evd = Gumbel())
# todo: autocorrelation (see signal module)
# todo: turning-points
# todo: peaks(narrow_band=True)
# todo: valleys(narrow_band=True)
# todo: stats2dict
# todo: write (different file formats, chosen based on suffix e.g. .csv, .ts, .bin, resample as necessary)
# todo: plot (subplots: history=True, histogram=False, psd=False, peaks=False, level_crossing=False, peak_cdf=False)
# todo: level crossings (see Moan&Naess book)
# todo: cross spectrum(scipy.signal.csd)
# todo: coherence (scipy.signal.coherence)
# todo: smarter sizing of scatter dots in plot_cycle_rangemean()

class TimeSeries(object):
    """
    A class for storage, processing and presentation of time series.

    Parameters
    ----------
    name : str
        time series name/identifier
    t : array_like
        time array; floats (seconds) or datetime objects
    x : array_like
        data points corresponding to time
    parent : str, optional
        Path to file it originates from, absolute or relative. Should not be specified for new (calculated)
        time series.
    dtg_ref : datetime.datetime, optional
        Date-time referring to ``t=0`` (time equal zero). Enables output of entire time array as list of
        datetime.datetime objects.
    kind : str, optional
        Kind/type of signal e.g. 'force' or 'acceleration'
    unit : str, optional
        Unit of measure e.g. 'N' (Newton), 'kN' (kilo Newton), 'm/s2'

    Attributes
    ----------
    name : str
        Time series name/identifier
    x : array_like
        Data points corresponding to time, see property :attr:`~TimeSeries.t`.
    parent : str
        Absolute or relative path to file it originates from.
    kind : str
        Kind of signal e.g. 'force' or 'acceleration'.
    unit : str
        Unit of measure e.g. 'N' (Newton) or 'm/s2'.

    Notes
    -----
    If time is given as array of datetime objects and `dtg_ref` is not specified, `dtg_ref` will be set to time
    array start value.
    """

    def __init__(self, name, t, x, parent=None, dtg_ref=None, kind=None, unit=None):
        # TODO: diagnose t series on initiation. check for nans, infs etc. before storing data on self.

        # check input parameters
        assert t.size == x.size, "Time and data must be of equal length."
        assert isinstance(dtg_ref, (datetime, np.datetime64)) or dtg_ref is None, \
            "Expected 'dtg_ref' datetime object or None"
        assert isinstance(x[0], (int, np.int32, np.int64, float, np.float32, np.float64)), \
            f"Data (x) must be integers or floats not '{type(x[0])}'."

        self.name = name
        self.kind = kind
        self.unit = unit
        self.parent = parent

        if isinstance(t[0], np.datetime64):
            # convert from numpy datetime (pandas, nptdms)
            t = np.array(t).astype(datetime)

        if isinstance(dtg_ref, np.datetime64):
            # tolist() returns a single date time
            # https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64/13753918#13753918
            dtg_ref = dtg_ref.tolist()

        if isinstance(t[0], datetime):
            # handle time specified as datetime
            self._dtg_ref = dtg_ref if dtg_ref is not None else t[0]
            self._dtg_time = t  # time as datetime
            self._t = np.array([(_ - self._dtg_ref).total_seconds() for _ in t])    # time as floats
        elif isinstance(t[0], (int, np.int32, np.int64, float, np.float32, np.float64)):
            # time as integer and floats
            self._t = np.array(t).flatten().astype(float)
            self._dtg_ref = dtg_ref
            self._dtg_time = None
        else:
            raise TypeError(f"time must be given as array of floats or datetime objects, not '{type(t[0])}'.")

        self.x = np.array(x).flatten().astype(float)

    def __copy__(self):
        """
        Return copy current TimeSeries object without binding to the original

        Returns
        -------
        TimeSeries
            Copy of current TimeSeries

        """
        # copy used to avoid bindings between copied instances and the original instances
        name = copy.copy(self.name)
        t = copy.copy(self.t)
        x = copy.copy(self.x)
        parent = copy.copy(self.parent)
        dt_ref = copy.copy(self._dtg_ref)
        new = TimeSeries(name, t, x, parent=parent, dtg_ref=dt_ref)
        return new

    def __iter__(self):
        """
        Generator yielding pairs of time and data

        Returns
        -------
        tuple
            (t, x)
        """
        i = 0
        while i < self.n:
            yield (self._t[i], self.x[i])
            i += 1

    def __repr__(self):
        return str('<TimeSeries "%s">') % self.name

    # todo: consider including self.__deepcopy__

    @property
    def average_frequency(self):
        """
        Average frequency of mean level crossings

        Returns
        -------
        float
            average frequency of mean level crossings
        """
        return average_frequency(self.t, self.x, up=True)

    @property
    def average_period(self):
        """
        Average period between mean level crossings

        Returns
        -------
        float
            average period between mean level crossings
        """
        return 1. / self.average_frequency

    @property
    def data(self):
        """
        Return dictionary of attributes and properties

        Returns
        -------
        dict
            Attributes and properties

        """
        # todo: QA/debugging on TimeSeries.data (dictionary)
        '''
        # make dict with all non callable items from class dir(), skip private '__<>__' and itself ("data")
        d = OrderedDict()
        # d.update(self.__dict__)
        for prop in dir(self):
            if prop.startswith("__"):
                continue
            if prop == "data":
                continue
            attr = self.__getattribute__(prop)
            if not callable(attr):
                d[prop] = copy.copy(attr)
        # todo: ts.data (dict): consider to include output from selected methods (e.g. mean, std, skewness, kurtosis)

        return d
        '''
        raise NotImplementedError("data property is not yet implemented")

    @property
    def dt(self):
        """
        Average time step

        Returns
        -------
        float
            average time step

        Notes
        -----
        The TimeSeries class does not require equidistant time vector.

        """
        return np.mean(np.diff(self.t))

    @property
    def dtg_ref(self):
        return self._dtg_ref

    @property
    def dtg_time(self):
        """
        Time array formatted as datetime.datetime objects.

        Returns
        -------
        array
            Time array formatted as array of datetime.datetime objects, provided that `dtg_ref` is defined.
            Otherwise, returns None.
        """
        if self._dtg_ref is not None:
            if self._dtg_time is None:
                # create array of datetime objects (reference is dtg_ref given on class initiation)
                # assuming time array is provided in seconds
                self._dtg_time = np.array([self._dtg_ref + timedelta(seconds=t) for t in self.t])
            else:
                pass
            return self._dtg_time
        else:
            # dtg_ref not provided
            return None

    @property
    def fullname(self):
        """
        Full name.

        Returns
        -------
        str
            parent and name joined to single string
        """
        if self.parent is None:
            return self.name
        else:
            return os.path.join(self.parent, self.name)

    @property
    def n(self):
        """
        Number of time steps

        Returns
        -------
        int
            number of time steps
        """
        return self.t.size

    @property
    def is_constant_dt(self):
        """
        Boolean telling whether the time series is sampled at constant intervals or not.

        Returns
        -------
        bool
            Constant sampling interval or not
        """
        dt = np.diff(self.t)
        dt_avg = np.ones(np.shape(dt)) * self.dt

        if np.allclose(dt, dt_avg, rtol=1.e-5, atol=0.):
            return True
        else:
            return False

    @property
    def start(self):
        """
        Start time.

        Returns
        -------
        float
            start time
        """
        return self.t[0]

    @property
    def t(self):
        """
        Time array
        """
        return self._t

    @property
    def end(self):
        """
        End time.

        Returns
        -------
        float
            end time
        """
        return self.t[-1]

    @property
    def dtg_start(self):
        """
        Start time as datetime object.

        Returns
        -------
        datetime
            Start time as datetime.datetime instanc, provided that `dtg_ref` is defined.
            Otherwise, returns None.
        """
        if self._dtg_ref is not None:
            return self._dtg_ref + timedelta(seconds=self.start)
        else:
            return None

    @property
    def dtg_end(self):
        """
        End time as datetime object.

        Returns
        -------
        datetime
            End time as datetime.datetime instance, provided that `dtg_ref` is defined.
            Otherwise, returns None.
        """
        if self._dtg_ref is not None:
            return self._dtg_ref + timedelta(seconds=self.end)
        else:
            return None

    @property
    def duration(self):
        """
        Duration of time series.

        Returns
        -------
        float
            time series duration
        """
        return self.end - self.start

    def copy(self, newname=None):
        """
        Return copy of this TimeSeries object

        Parameters
        ----------
        newname: str, optional
            Name to assign to copy.

        Returns
        -------
        TimeSeries
            Copy of this TimeSeries object

        Notes
        -----
        Does not support parameters for the get() method. It is recommended to modify the copied object
        instead of the original.

        """
        newts = self.__copy__()
        if newname is not None:
            newts.name = newname
        return newts

    def filter(self, filtertype, freq, twin=None, taperfrac=None):
        """
        Apply FFT filter

        Parameters
        ----------
        filtertype : str
            Type of filter to apply. Available options:

            - 'lp' - low-pass filter time series
            - 'hp' - high-pass filter time series
            - 'bp' - band-pass filter time series
            - 'bs' - band-stop filter time series
            - 'tp' - threshold filter time series

        freq : float or tuple
            Filter frequency [Hz]. One frequency for 'lp', 'hp' and 'tp' filters, two frequencies for 'bp' and 'bs'.
        twin : tuple, optional
            Time window (start, stop) to consider (removes away signal outside time window).
        taperfrac : float, optional
            Fraction of time domain signal to be tapered. For details, see `get()`.

        Returns
        -------
        tuple
            Tuple with to arrays: time array and filtered signal.

        See Also
        --------
        get()

        Notes
        -----
        filter() is a special case of the get() method. For instance; assuming `ts` is a TimeSeries instance,  the code:

        >>> t, xlo = ts.filter('lp', 0.03, twin=(1000, 11800))

        ... is equivalent to:

        >>> t, xlo = ts.get(twin=(1000, 11800), filterargs=('lp', 0.03))

        Examples
        --------
        Low-pass filter the time series at period of 30 [s] (assuming `ts` is a TimeSeries instance):

        >>> t, xlo = ts.filter('lp', 1/30)

        High-pass filter at period of 30 [s]:

        >>> t, xhi = ts.filter('hp', 1/30)

        Band-pass filter, preserving signal between 0.01 and 0.1 [Hz]:

        >>> t, xband = ts.filter('bp', (0.01, 0.1))

        If the signal includes a significant transient, this may be omitted by specifying time window to consider:

        >>> t, xlo = ts.filter('lp', 0.03, twin=(1000, 1e12))

        """
        # make 'freq' a tuple if float is given
        if isinstance(freq, float):
            freq = freq,

        # check input parameters
        if filtertype in ("lp", "hp", "tp"):
            if len(freq) != 1:
                raise ValueError("One frequency is expected for filter types 'lp', 'hp' and 'tp'")
        elif filtertype in ("bp", "bs"):
            if len(freq) != 2:
                raise ValueError("Two frequencies are expected for filter types 'bp' and 'bs'")
        else:
            raise ValueError("Invalid filter type: '%s'" % filtertype)

        # perform filtering
        if isinstance(freq, float):
            fargs = (filtertype, freq)
        elif isinstance(freq, tuple):
            fargs = (filtertype,) + freq
        else:
            raise TypeError("Unexpected type '%s' of freq." % type(freq))

        t, x = self.get(twin=twin, filterargs=fargs, taperfrac=taperfrac)

        return t, x

    def fit_weibull(self, twin=None, method='msm'):
        """
        Fit Weibull distribution to sample of global maxima.

        Parameters
        ----------
        twin : tuple, optional
            Time window to consider.
        method : {'msm', 'lse', 'mle', 'pwm', 'pwm2'}, optional
            See `qats.weibull.Weibull.fit` for description of options.

        Returns
        -------
        Weibull
            Class instance.

        See Also
        --------
        maxima()
        qats.signal.find_maxima
        qats.stats.weibull.Weibull.fit
        qats.stats.weibull.Weibull.fromsignal

        Examples
        --------
        >>> weib = ts.fit_weibull(twin=(200., 1e12), method='msm')
        >>> print(weib.params)
        (372.90398322024936, 5.2533589509069714, 0.8516538181105509)

        The fit in example above is equivalent to:

        >>> from qats.stats.weibull import Weibull
        >>> maxima = ts.maxima(local=False, threshold=None, twin=(200., 1e12), rettime=False)
        >>> weib = Weibull.fit(maxima, method='msm')

        """
        maxima = self.maxima(local=False, threshold=None, twin=twin, rettime=False)
        return Weibull.fit(maxima, method=method)

    def get(self, twin=None, resample=None, window_len=None, filterargs=None, window="rectangular", taperfrac=None):
        """
        Return time series processed according to parameters

        Parameters
        ----------
        twin : tuple, optional
            time window, cuts away time series beyond time window
        filterargs : tuple, optional
            Apply filtering. Argument options:

            - ('lp', f)           - Low-pass filter at frequency f
            - ('hp', f)           - High-pass filter at frequency f
            - ('bp', flo, fhi)    - Band-pass filter between frequencies flo and fhi
            - ('bs', flo, fhi)    - Band-stop filter between frequencies flo and fhi
            - ('tp', a)           - Threshold-pass filter at amplitude a

        resample : float or ndarray or list, optional
            Resample time series:

            - to specified constant time step if `resample` is a float
            - to specified time if `resample` is an array or list

        window_len : int, optional
            Smooth time serie based on convoluting the time series with a window of specified length. An odd
            number is recommended.
        window : str, optional
            Type of window function used for smoothing, default 'rectangular', see `qats.signal.smooth` for options.
        taperfrac : float, optional
            Fraction of time domain signal to be tapered. A taper of 0.001-0.01 is recommended before calculation of
            the power spectral density.

        Returns
        -------
        tuple
            Tuple of two numpy arrays; ``(time, data)``

        Notes
        -----
        The data is modified in the following order
            1. Windowing
            2. Resampling
            3. Tapering
            4. Filtering
            5. Smoothing

        Mean value of signal is subtracted before step 3, and then re-added after step 4 (except if high-pass filter is
        specified).

        Resampling is achieved by specifying either `dt` or `t`. If both are specified `t` overrules. Resampling to a
        a specified time array `t` cannot be specified at the same time as a time window `twin`.

        Filtering is achieved by FFT truncation.

        When smoothing the central data point is calculated at the weighted average of the windowed data points. In the
        case of the rectangular window all windowed data points are equally weighted. Other window functions are
        available on request. Note that windowing have some similarities to filtering.

        Windowing is achieved by direct truncation of time series within specified time range.

        Data tapering is performed by multiplying the time series with a Tukey window (tapered cosine window).

        See Also
        --------
        resample, interpolate, qats.signal.smooth

        """
        assert not ((isinstance(resample, np.ndarray)) and (twin is not None)), \
            "Cannot specify both resampling to `newt` and cropping to time window `twin`."

        def new_timearray(t0, t1, d):
            """ Establish time array from specified start (t0), end (t1) and time step (d) """
            n = int(round((t1 - t0) / d)) + 1
            t_ = np.linspace(t0, t1, n, retstep=False)
            return t_

        # copy time series
        t = copy.copy(self.t)
        x = copy.copy(self.x)

        # windowing
        if twin is not None:
            t_start, t_end = twin
            i = (t >= t_start) & (t <= t_end)
            t, x = t[i], x[i]

        # resampling
        if resample is not None:
            if isinstance(resample, (np.ndarray, list)):
                # specified time array
                t = resample
            elif isinstance(resample, (float, np.float32)):
                # specified time step
                t = new_timearray(t[0], t[-1], resample)
            else:
                raise TypeError("Parameter resample should be either a float or a numpy.ndarray type, not %s." % type(resample))

            x = self.interpolate(t)

        elif filterargs is not None and not self.is_constant_dt:
            # filtering is specified but the time step is not constant (use the average time step)
            t = new_timearray(t[0], t[-1], self.dt)
            x = self.interpolate(t)
        else:
            pass

        # data tapering
        if (taperfrac is not None) and (isinstance(taperfrac, float)) and (taperfrac > 0.) and (taperfrac < 1.):
            x, _ = taper(x, window='tukey', alpha=taperfrac)

        # filtering
        if filterargs is not None:
            assert isinstance(filterargs, tuple) or isinstance(filterargs, list), \
                "Parameter filter should be either a list or tuple, not %s." % type(filterargs)

            # time step for the filter calculation
            _dt = t[1] - t[0]

            if filterargs[0] == 'lp':
                assert len(filterargs) == 2, "Excepted 2 values in filterargs but got %d." % len(filterargs)
                x = lowpass(x, _dt, filterargs[1])

            elif filterargs[0] == 'hp':
                assert len(filterargs) == 2, "Excepted 2 values in filterargs but got %d." % len(filterargs)
                x = highpass(x, _dt, filterargs[1])

            elif filterargs[0] == 'bp':
                assert len(filterargs) == 3, "Excepted 3 values in filterargs but got %d." % len(filterargs)
                x = bandpass(x, _dt, filterargs[1], filterargs[2])

            elif filterargs[0] == 'bs':
                assert len(filterargs) == 3, "Excepted 3 values in filterargs but got %d." % len(filterargs)
                x = bandblock(x, _dt, filterargs[1], filterargs[2])

            elif filterargs[0] == 'tp':
                assert len(filterargs) == 2, "Excepted 2 values in filterargs but got %d." % len(filterargs)
                x = thresholdpass(x, filterargs[1])
            else:
                # invalid filter type
                raise ValueError(f"Invalid filter type: {filterargs[0]}")

        # smoothing
        if (window_len is not None) and (isinstance(window_len, int)) and (window_len > 0):
            x = smooth(x, window_len=window_len, window=window, mode='same')

        return t, x

    def interpolate(self, time):
        """
        Interpolate linearly in data to values a specified time

        Parameters
        ----------
        time : array_like
            time at which data is interpolated

        Returns
        -------
        array_like
            interpolated data values

        Raises
        ------
        ValueError
            If interpolation is attempted on values outside the range of `t`.

        Notes
        -----
        Extrapolation outside the range of `t` is not allowed since that does not make sense when analysing generally
        irregular timeseries.

        """
        f = interp1d(self.t, self.x, bounds_error=True)
        return f(time)

    def kurtosis(self, fisher=False, bias=False, **kwargs):
        """
        Kurtosis of time series

        Parameters
        ----------
        fisher : bool, optional
            If True, Fisher’s definition is used (normal ==> 0.0). If False (default), Pearson’s definition
            is used (normal ==> 3.0).
        bias : bool, optional
            If False (default), then the calculations are corrected for statistical bias.
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        float
            Sample kurtosis

        See also
        --------
        scipy.stats.kurtosis

        """
        # get data array, time does not matter
        _, x = self.get(**kwargs)
        return kurtosis(x, fisher=fisher, bias=bias)

    def max(self, **kwargs):
        """
        Maximum value of time series.

        Parameters
        ----------
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        float
            Maximum value of time series

        Examples
        --------
        Get maximum of entire time series:

        >>> xmax = ts.max()

        Get maximum within a specified time window:

        >>> xmax = ts.max(twin=(1200, 12000))
        """
        _, x = self.get(**kwargs)
        return x.max()

    def maxima(self, twin=None, local=False, threshold=None, rettime=False, **kwargs):
        """
        Return sorted maxima

        Parameters
        ----------
        twin : tuple, optional
            Time window (start, end) to consider.
        local : bool, optional
            return local maxima also, default only global maxima are considered
        threshold : float, optional
            consider only maxima larger than specified treshold. Default mean value.
        rettime : bool, optional
            If True, (maxima, time_maxima), where `time_maxima` is an array of time instants associated with the
            maxima sample.
        **kwargs : optional
                Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        array
            Signal maxima, sorted from smallest to largest.
        array
            Only returned if `rettime` is True.
            Time instants of signal maxima.

        See Also
        --------
        qats.signal.find_maxima

        Notes
        -----
        By default only 'global' maxima are considered, i.e. the largest maximum between each mean-level up-crossing.
        If ``local=True``, local maxima are also included (first derivative is zero, second derivative is negative).

        """
        # get time and data
        t, x = self.get(twin=twin, **kwargs)

        # find maxima (and associated time, if specified)
        if rettime is True:
            m, ind = find_maxima(x, local=local, threshold=threshold, retind=True)
            return m, t[ind]
        else:
            m = find_maxima(x, local=local, threshold=threshold, retind=False)
            return m

    def mean(self, **kwargs):
        """
        Mean value of time series.

        Parameters
        ----------
        **kwargs : optional
            Keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        float
            Sample mean

        See also
        --------
        numpy.mean

        """
        # get data array
        _, x = self.get(**kwargs)
        return np.mean(x)

    def min(self, **kwargs):
        """
        Minimum value of time series.

        Parameters
        ----------
        **kwargs : optional
            Keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        float
            Minimum value of time series
        """
        _, x = self.get(**kwargs)
        return x.min()

    def minima(self, twin=None, local=False, threshold=None, rettime=False, **kwargs):
        """
        Return sorted minima

        Parameters
        ----------
        twin : tuple, optional
            Time window (start, end) to consider.
        local : bool, optional
            return local minima also, default only global minima are considered
        threshold : float, optional
            consider only minima smaller (considering the sign) than specified threshold. Default mean value.
        rettime : bool, optional
            If True, (maxima, time_maxima), where `time_maxima` is an array of time instants associated with the
            maxima sample.
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        array
            Signal minima, sorted from smallest to largest.
        array
            Only returned if `rettime` is True.
            Time instants of signal minima.

        Notes
        -----
        Minima are found by multiplying the time series with -1, finding the maxima using the maxima() method and then
        multiplying the maxima with -1 again.

        By default only 'global' minima are considered, that is the smallest minimum between each mean-level up-crossing.
        If local=True local minima are also considered.

        See Also
        --------
        maxima()
        """
        # get time and data
        t, x = self.get(twin=twin, **kwargs)

        # flip the time series to that minima becomes maxima
        x *= -1.

        # find minima (and associated time, if specified)
        if rettime is True:
            m, ind = find_maxima(x, local=local, threshold=threshold, retind=True)
            return -1. * m, t[ind]  # reverse flip
        else:
            m = find_maxima(x, local=local, threshold=threshold, retind=False)
            return -1. * m          # reverse flip

    def modify(self, **kwargs):
        """
        Modify TimeSeries object

        Parameters
        ----------
        **kwargs : optional
            Keyword arguments are passed to :meth:`get()`.

        Notes
        -----
        Modifies ´t´ and ´x´ on current TimeSeries object. This is irreversible.
        """
        self._t, self.x = self.get(**kwargs)

    def plot(self, figurename=None, show=None, num=1, **kwargs):
        """
        Plot time series trace.

        Parameters
        ----------
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`get()`.

        """
        # dict with numpy arrays: time and data
        t, x = self.get(**kwargs)

        plt.figure(num=num)
        plt.plot(t, x, label=self.name)
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def plot_psd(self, figurename=None, show=None, num=1, **kwargs):
        """
        Plot time series power spectral density.

        Parameters
        ----------
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`psd()`.

        """
        # dict with TimeSeries objects
        plt.figure(num=num)
        f, p = self.psd(**kwargs)
        plt.plot(f, p, label=self.name)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.grid()
        plt.legend()
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def plot_cycle_range(self, n=200, w=None, bw=1., figurename=None, show=None, num=1, **kwargs):
        """
        Plot cycle range versus number of occurrences.

        Parameters
        ----------
        n : int, optional
            Group by cycle range in `n` equidistant bins.
        w : float, optional
            Group by cycle range in `w` wide equidistant bins. Overrides `n`.
        bw : float, optional
            Bar width, expressed as ratio of bin width.
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`rfc()`.

        See Also
        --------
        rfc, qats.fatigue.rainflow.count_cycles, qats.fatigue.rainflow.rebin
        """
        # rebin cycles
        assert (n is not None) or (w is not None), "Cycles must be rebinned for this plot - either 'n' or 'w' must " \
                                                   "be different from None"

        cycles = self.rfc(**kwargs)
        cycles = rebin_cycles(cycles, binby='range', n=n, w=w)

        r, _, c = cycles.T   # unpack cycle range and count, ignore mean value
        dr = r[1] - r[0]     # bin width, used as basis for bar width
        plt.figure(num=num)
        plt.bar(r, c, dr * bw, label=self.name)
        plt.xlabel('Cycle range')
        plt.ylabel('Cycle count (-)')
        plt.grid()
        plt.legend()
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def plot_cycle_rangemean(self, n=None, w=None, figurename=None, show=None, num=1, **kwargs):
        """
        Plot cycle range-mean versus number of occurrences.

        Parameters
        ----------
        n : int, optional
            Group by cycle range in *n* equidistant bins.
        w : float, optional
            Group by cycle range in *w* wide equidistant bins. Overrides `n`.
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`rfc()`.

        Notes
        -----
        Cycle means are represented by weighted averages in each bin.

        See Also
        --------
        rfc, plot_cycle_range,
        qats.fatigue.rainflow.count_cycles, qats.fatigue.rainflow.rebin
        """
        cycles = self.rfc(**kwargs)

        # rebin cycles
        if (n is not None) or (w is not None):
            cycles = rebin_cycles(cycles, binby='range', n=n, w=w)

        ranges, means, counts = cycles.T  # unpack cycle range, mean and count

        # the scatter plot (with double marker size for improved readability)
        plt.figure(num=num)
        plt.scatter(means, ranges, s=[2. * c for c in counts], alpha=0.4, label=self.name)
        plt.xlabel('Cycle mean')
        plt.ylabel('Cycle range')
        plt.grid()
        plt.legend()
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def plot_cycle_rangemean3d(self, nr=100, nm=100, figurename=None, show=None, num=1, **kwargs):
        """
        Plot cycle range-mean versus number of occurrences as 3D surface.

        Parameters
        ----------
        nr : int, optional
            Group by cycle range in *nr* equidistant bins.
        nm : int, optional
            Group by cycle mean in *nm* equidistant bins.
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`rfc()`.

        """
        # This import registers the 3D projection, but is otherwise unused.
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        cycles = self.rfc(**kwargs)
        ranges, means, counts = mesh(cycles, nr=nr, nm=nm)

        fig = plt.figure(num=num)
        ax = fig.gca(projection='3d')
        ax.plot_surface(ranges, means, counts, cmap=cm.coolwarm)
        ax.set_xlabel('Cycle range')
        ax.set_ylabel('Cycle mean')
        ax.set_zlabel('Cycle count')
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def psd(self, nperseg=None, noverlap=None, detrend='constant', nfft=None, normalize=False, **kwargs):
        """
        Estimate power spectral density using Welch’s method.

        Parameters
        ----------
        nperseg : int, optional
            Length of each segment. Default 1/4 of the signal length.
        noverlap : int, optional
            Number of points to overlap between segments. Default noverlap = nperseg / 2.
        nfft : int, optional
            Length of the FFT used, if a zero padded FFT is desired. Default the FFT length is nperseg.
        detrend : str or function, optional
            Specifies how to detrend each segment. If detrend is a string, it is passed as the type argument to
            detrend. If it is a function, it takes a segment and returns a detrended segment. Defaults to ‘constant’.
        normalize : bool, optional
            Normalize power spectral density on maxium density.
        kwargs : optional
            Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        tuple
            Two arrays: sample frequencies and corresponding power spectral density

        Notes
        -----
        Welch’s method [1] computes an estimate of the power spectral density by dividing the data into overlapping
        segments, computing a modified periodogram for each segment and averaging the periodograms. Welch method is
        chosen over the periodogram as the spectral density is smoothed by adjusting the `nperseg` parameter. The
        periodogram returns a raw spectrum which requires additional smoothing to get a readable spectral density plot.

        An appropriate amount of overlap will depend on the choice of window and on your requirements. For the default
        ‘hanning’ window an overlap of 50% is a reasonable trade off between accurately estimating the signal power,
        while not over counting any of the data. Narrower windows may require a larger overlap.

        If noverlap is 0, this method is equivalent to Bartlett’s method [2].

        References
        ----------
        1. P. Welch, “The use of the fast Fourier transform for the estimation of power spectra: A method based on
           time averaging over short, modified periodograms”, IEEE Trans. Audio Electroacoust. vol. 15, pp. 70-73, 1967.
        2. M.S. Bartlett, “Periodogram Analysis and Continuous Spectra”, Biometrika, vol. 37, pp. 1-16, 1950.

        See also
        --------
        qats.signal.psd
        scipy.signal.welch
        scipy.signal.periodogram

        """
        # get time and data arrays
        t, x = self.get(**kwargs)

        if nperseg is None:
            nperseg = int(0.25 * x.size)

        # ensure constant time step
        _dt = np.diff(t)
        # (use atol not zero to avoid false positives for zero values)
        if not np.isclose(min(_dt), max(_dt), rtol=1.e-2, atol=1.e-6):
            raise ValueError(f"The time step of '{self.name}' varies with more than 1%. A constant time step is "
                             f"required when estimating power spectral density using FFT. Resample to constant "
                             f"time step.")

        # average time step for requested series
        dt = float(np.mean(_dt))

        # estimate psd using qats.signal.psd (which uses welch's definition)
        f, p = psd(x, dt, nperseg=nperseg, noverlap=noverlap, detrend=detrend, nfft=nfft, scaling='density',
                   return_onesided=True, axis=-1)

        if normalize:
            p = p / np.max(p)

        return f, p

    def resample(self, dt=None, t=None):
        """
        Resample with constant sampling interval dt

        Parameters
        ----------
        dt : float, optional
            Constant sampling interval
        t : array_like, optional
            Time array to which the data is resampled

        Returns
        -------
        array_like
            Resampled data

        Notes
        -----
        Either `dt` or `t` have to specified. If both are specified `t` overrules.

        If `dt` is used the resampled data cover the period given by the objects `t` attribute, in other words from
        `start` to `end`.

        """
        assert dt is not None or t is not None, "Either new time step 'dt' or new time array 't' has to be specified."
        if t is not None:
            assert t.min() >= self.start and t.max() <= self.end, "The new specified time array exceeds the original " \
                                                                  "time array. Extrapolation is not allowed."
            return self.interpolate(t)
        else:
            assert dt > 0., "The specified time step is to small."
            return self.interpolate(np.arange(self.start, self.end, step=dt))

    def rfc(self, **kwargs):
        """
        Returns a sorted list containing with cycle range, mean and count.

        Parameters
        ----------
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        np.array
            Cycle range, mean and count sorted ascending by cycle range.

        Examples
        --------
        Unpack cycle ranges, means and counts as 1D arrays:

        >>> cycles = ts.rfc()
        >>> ranges, means, counts = cycles.T

        Notes
        -----
        Half cycles are counted as 0.5, so the returned counts may not be whole numbers.

        See Also
        --------
        qats.fatigue.rainflow.count_cycles

        """
        # get data array
        _, x = self.get(**kwargs)

        cycles = count_cycles(x)

        return cycles

    def set_dtg_ref(self, new_ref=None):
        """
        Set or adjust the dtg reference.

        If there is no pre-defined dtg reference for the time series, dtg reference (`dtg_ref`) is set to the specified
        value.

        If the time series already has a dtg reference, time array is updated so that t=0 now refers to the new
        dtg reference. This implies that time array is shifted as follows::

            t += (dtg_ref - new_ref).total_seconds()

        If no new value is specified, `dtg_ref` is adjusted so that time array starts at zero (if it doesn't already).

        Parameters
        ----------
        new_ref: datetime
            New dtg reference.

        Raises
        ------
        ValueError
            If `new_ref` is not a `datetime.datetime` instance, or if `dtg_ref` is not pre-defined and `new_dtg` is not
            given.
        """
        if new_ref is not None and not isinstance(new_ref, datetime):
            raise ValueError("`new_ref` must be a datetime.datetime instance")
        elif new_ref is None and self._dtg_ref is None:
            raise ValueError("New dtg reference must be given if attribute `dtg_ref` is not pre-defined")

        if new_ref is not None and self._dtg_ref is None:
            # set dtg_ref to new_ref, then return (no basis to adjust time array)
            self._dtg_ref = new_ref
        elif new_ref is not None and self._dtg_ref is not None:
            # adjust time array so that t=0 refers to new ref
            delta = (self._dtg_ref - new_ref).total_seconds()
            self._dtg_ref = new_ref
            self._t += delta
            self._dtg_time = None  # reset, no need to initiate new array until requested
        elif new_ref is None and self._dtg_ref is not None:
            # adjust dtg_ref and time array so that dtg_ref refers to t=0
            delta = -self.start
            self._dtg_ref = self.dtg_start
            self._t += delta
            self._dtg_time = None  # reset, no need to initiate new array until requested

    def stats(self, statsdur=10800., quantiles=(0.37, 0.57, 0.9), is_minima=False, include_sample=False, **kwargs):
        """
        Returns dictionary with time series properties and statistics

        Parameters
        ----------
        statsdur : float
            Duration in seconds for estimation of extreme value distribution (Gumbel) from peak distribution (Weibull).
            Default is 10800 seconds (3 hours).
        quantiles : tuple, optional
            Quantiles in the Gumbel distribution used for extreme value estimation, defaults to (0.37, 0.57, 0.90).
        is_minima : bool, optional
            Fit to sample of minima instead of maxima. The sample is multiplied by -1 prior to parameter estimation.
        include_sample : bool, optional
            Return sample of maxima or minima (minima=True).
        **kwargs
            Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        OrderedDict
            Time series statistics (for details, see Notes below).

        Notes
        -----
        The returned dictionary contains::

            Key         Description
            --------    -----------
            start       First value of time array [s]
            end         Last value of time array [s]
            duration    end - start [s]
            dtavg       Average time step [s]
            mean        Mean value of signal array
            std         Unbiased standard deviation of signal array
            skew        Unbiased skewness of signal array (=0 for normal distribution)
            kurt        Unbiased kurtosis of signal array (=3 for normal distribution)
            min         Minimum value of signal array
            max         Maximum value of signal array
            tz          Average mean crossing period [s]
            wloc        Weibull distribution location parameter fitted to sample of global maxima
            wscale      Weibull distribution scale parameter fitted to sample of global maxima
            wshape      Weibull distribution shape parameter fitted to sample of global maxima
            gloc        Gumbel distribution location parameter estimated from Weibull distribution and `statsdur`
            gscale      Gumbel distribution scale parameter estimated from Weibull distribution and `statsdur`
            p_* ..      Extreme values estimated from the Gumbel distribution, e.g. p_90 is the 0.9 quantile


        See Also
        --------
        qats.stats.weibull.weibull2gumbel, qats.stats.weibull.pwm

        Examples
        --------
        Generate statistics for entire time series, unfiltered:

        >>> stats = ts.stats()

        To get statistics for filtered time series, one may specify the filter to apply:

        >>> stats_lf = ts.stats(filterargs=('lp', 0.3))
        >>> stats_hf = ts.stats(filterargs=('hp', 0.3))

        To ignore the transient part of the time series, time window may be specified:

        >>> stats = ts.stats(twin=(500., 1e12))
        """
        # get time series as array
        t, x = self.get(**kwargs)

        try:
            tz = 1. / average_frequency(t, x)
        except IndexError:
            # too few maxima, tz may not calculated (keep it as None)
            tz = np.nan

        # find global maxima or minima
        if not is_minima:
            f = 1.
        else:
            f = -1.

        mx = find_maxima(f * x)
        if np.size(mx) <= 1:
            wloc = wscale = wshape = gloc = gscale = np.nan
            pvalues = {f"p_{100 * q:.2f}": np.nan for q in quantiles}
        else:
            wloc, wscale, wshape = pwm(mx)
            if any(np.isnan([wloc, wscale, wshape])):
                # force all parameters to nan if one (typically scale or shape) is
                wloc = wscale = wshape = np.nan
            n = round(statsdur / (t[-1] - t[0]) * np.size(mx))
            try:
                gloc, gscale = weibull2gumbel(wloc, wscale, wshape, n)
            except (AssertionError, ZeroDivisionError) as e:
                # invalid distribution parameters or bad combinations
                gloc = gscale = np.nan

            try:
                g = Gumbel(loc=gloc, scale=gscale)
            except AssertionError as e:
                # invalid distribution parameters
                values = np.nan * np.ones(np.shape(quantiles))
            else:
                values = g.invcdf(p=quantiles)
            finally:
                pvalues = {f"p_{100 * q:.2f}": f * v for q, v in zip(quantiles, values)}

        # establish output dictionary
        d = OrderedDict(
            start=t[0], end=t[-1], duration=t[-1] - t[0], dtavg=np.mean(np.diff(t)),
            mean=x.mean(), std=tstd(x), skew=skew(x, bias=False),
            kurt=kurtosis(x, fisher=False, bias=False), min=x.min(), max=x.max(), tz=tz,
            wloc=wloc, wscale=wscale, wshape=wshape, gloc=gloc, gscale=gscale, is_minima=is_minima,
            sample=f * mx if include_sample else None, **pvalues
        )
        return d

    def std(self, **kwargs):
        """
        Unbiased standard deviation of time series.

        Parameters
        ----------
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        float
            Sample standard deviation

        Notes
        -----
        Computes the unbiased sample standard deviation, i.e. it uses a correction factor n / (n - ddof).

        See also
        --------
        scipy.stats.tstd

        """
        # get data array, time does not matter
        _, x = self.get(**kwargs)
        return tstd(x)

    def skew(self, bias=False, **kwargs):
        """
        Skewness of time series

        Parameters
        ----------
        bias : bool, optional
            If False (default), then the calculations are corrected for statistical bias.
        **kwargs : optional
            Additional keyword arguments are passed to :meth:`get()`.

        Returns
        -------
        float
            Sample skewness

        See also
        --------
        scipy.stats.skew

        """
        # get data array, time does not matter
        _, x = self.get(**kwargs)
        return skew(x, bias=bias)
