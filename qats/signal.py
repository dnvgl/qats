#!/usr/bin/env python
# encoding: utf8
"""
Module with functions for signal processing.

"""
import numpy as np
from scipy.fftpack import fft, ifft, rfft, irfft
from scipy.signal import welch, butter, filtfilt, sosfiltfilt


def extend_signal_ends(x, n):
    """Extend the signal ends with `n` values to mitigate the edge effect.

    Parameters
    ----------
    x : array_like
        Signal
    n : int
        Number of values prepended and appended to signal.

    Notes
    -----
    At each end of the signal `n` values of the signal are replicated, flipped and joined with the signal to maintain
    continuity in the signal level and slope at the joining points. This should mitigate end effects when filterin
    the signal.

    The original signal is retrieved as `x[n:-n:1]`.
    """
    start = 2. * x[0] - 1. * x[n:0:-1]
    end = 2. * x[-1] - 1. * x[-2:-(n + 2):-1]
    return np.concatenate((start, x, end))


def smooth(x, window_len=11, window='rectangular', mode='same'):
    """
    Smooth time serie based on convolution of a window function and the time serie.

    Parameters
    ----------
    x : array
        The input signal.
    window_len : int, optional
        The dimension of the smoothing window.
    window : {'rectangular', 'hanning', 'hamming', 'bartlett', 'blackman'}, optional
        The type of window. Rectangular window will produce a moving average smoothing.
    mode : {‘same’, ‘valid’, ‘full’}, optional
        full:
           This returns the convolution at each point of overlap, with an output
           shape of (N+M-1,). At the end-points of the convolution, the signals
           do not overlap completely, and boundary effects may be seen.

        same:
           By default mode is 'same' which returns output of length max(M, N).
           Boundary effects are still visible.

        valid:
           Mode valid returns output of length max(M, N) - min(M, N) + 1. The
           convolution product is only given for points where the signals overlap
           completely. Values outside the signal boundary have no effect

    Returns
    -------
    array
        The smoothed signal.

    Notes
    -----
    This method is based on the convolution of a scaled window with the signal. The signal is prepared by introducing
    reflected copies of the signal (with the window size) in both ends so that transient parts are minimized
    in the beginning and end of the output signal.

    Examples
    --------
    >>> from numpy import linspace
    >>> from numpy.random import randn
    >>> t = linspace(-2,2,0.1)
    >>> x = sin(t)+randn(len(t))*0.1
    >>> y = smooth(x)

    References
    ----------
    1. Wikipedia, http://en.wikipedia.org/wiki/Convolution

    See Also
    --------
    numpy.convolve

    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ('rectangular', 'hanning', 'hamming', 'bartlett', 'blackman'):
        raise ValueError("Window is not one of '{0}', '{1}', '{2}', '{3}', '{4}'".format(
                   *('rectangular', 'hanning', 'hamming', 'bartlett', 'blackman')))

    if window == 'rectangular':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    if mode == 'valid':
        s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        y = np.convolve(w / w.sum(), s, mode=mode)
    else:
        s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
        y = np.convolve(w / w.sum(), s, mode=mode)
        y = y[window_len - 1:-window_len + 1]

    return y


def taper(x, window='tukey', alpha=0.001):
    """
    Taper the input time serie using a window function

    Parameters
    ----------
    x : array
        Time series (without time vector), dimension `n*1`.
    window : {'tukey','cosine','hanning', 'flat', ....}
       Window function type. See numpy documentation for more windows
    alpha : float, optional
        Fraction of time domain signal to be tapered. Applies only
        to tukey and kaiser windows.

    Returns
    -------
    array
        Tapered time domain signal
    float
        correction factor to prevent FFT components from diminishing after the windowing.

    Notes
    -----
    All FFT based measurements assume that the signal is periodic in the time frame. When the measured signal is
    not periodic then leakage occurs. Leakage results in misleading information about the spectral amplitude and
    frequency. A window is shaped so that it is exactly zero at the beginning and end of the data block and has
    some special shape in between. This function is then multiplied with the time data block forcing the signal to be
    periodic and ultimately reduces the effects of leakage. There are many windows to choose from, each with advantages
    for specific applications. You must understand the effects of leakage and know the tradeoffs and advantages of the
    various windowing functions to accurately interpret frequency domain measurements.

    The cosine window is also known as the sine window.

    The Tukey window is also known as the tapered cosine window.

    See Also
    --------
    numpy.bartlett, numpy.blackman, numpy.hamming, numpy.hanning, numpy.kaiser

    References
    ----------
    1. Wikipedia, http://en.wikipedia.org/wiki/Window_function
    2. Melbourne G. Briscoe (1972), Energy loss in surface wave spectra due to data windowing, North Atlantic Treaty Organization (NATO), Saclant ASW Research Centre,

    """
    window_len = np.size(x)
    window = window.lower()

    # choice of window function
    if window == 'rectangular':
        w = np.ones(window_len)
    elif window == 'tukey':
        # alpha = 0 - rectangular window, alpha - Hann window
        w = np.zeros(window_len)
        for i in range(window_len):
            if (i >= 0) & (i < alpha * window_len / 2):
                w[i] = 0.5 * (1 + np.cos(np.pi * (2 * i / (alpha * window_len) - 1)))
            if (i >= alpha * window_len / 2) & (i <= window_len * (1 - alpha / 2)):
                w[i] = 1
            if (i > window_len * (1 - alpha / 2)) & (i <= window_len):
                w[i] = 0.5 * (1 + np.cos(np.pi * (2 * i / (alpha * window_len) - 2 / alpha + 1)))
    elif window == 'cosine':
        # also known as sine window
        n = np.arange(window_len)
        w = np.sin(np.pi * n / (window_len - 1))
    elif window == 'kaiser':
        w = eval('np.' + window + '(window_len,alpha)')
    else:
        w = eval('np.' + window + '(window_len)')

    # calculate tapered time series
    y = x * w

    # calculate weighting factor that should be applied so that the correct FFT signal amplitude level is recovered
    # after the windowing.
    wcorr = np.sum(w ** 2) / window_len

    return y, wcorr


def lowpass(x, dt, fc, order=5):
    """
    Low pass filter data signal x at cut off frequency fc, blocking harmonic content above fc.
    
    Parameters
    ----------
    x : array_like
        Signal
    dt : float
        Signal sampling rate (s)
    fc : float
        Cut off frequency (Hz)
    order : int, optional
        Butterworth filter order. Default 5.
    
    Returns
    -------
    array
        Filtered signal

    See Also
    --------
    scipy.signal.butter, scipy.signal.filtfilt
    """
    nyq = 0.5 * 1. / dt         # nyquist frequency
    normal_cutoff = fc / nyq    # normalized cut off frequency
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = filtfilt(b, a, x)
    return y


def highpass(x, dt, fc, order=5):
    """
    High pass filter data signal x at cut off frequency fc, blocking harmonic content below fc.
    
    Parameters
    ----------
    x : array_like
        Signal
    dt : float
        Signal sampling rate (s)
    fc : float
        Cut off frequency (Hz)
    order : int, optional
        Butterworth filter order. Default 5.
    
    Returns
    -------
    array
        Filtered signal

    See Also
    --------
    scipy.signal.butter, scipy.signal.filtfilt
    """
    nyq = 0.5 * 1. / dt         # nyquist frequency
    normal_cutoff = fc / nyq    # normalized cut off frequency
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    y = filtfilt(b, a, x)
    return y


def bandpass(x, dt, flow, fupp, order=5):
    """
    Band pass filter data signal x at cut off frequencies flow and fupp, blocking harmonic content outside the
    frequency band [flow, fupp]
    
    Parameters
    ----------
    x : array_like
        Signal
    dt : float
        Signal sampling rate (s)
    flow, fupp : float
        Passing frequency band (Hz)
    order : int, optional
        Butterworth filter order. Default 5.
    
    Returns
    -------
    array
        Filtered signal

    See Also
    --------
    scipy.signal.butter, scipy.signal.sosfiltfilt
    """
    nyq = 0.5 * 1. / dt  # nyquist frequency
    normal_cutoff = (flow / nyq, fupp / nyq)  # normalized cut off frequencies
    sos = butter(order, normal_cutoff, btype='bandpass', analog=False, output='sos')
    y = sosfiltfilt(sos, x)
    return y


def bandblock(x, dt, flow, fupp, order=5):
    """
    Band block filter data signal x at cut off frequencies flow and fupp, blocking harmonic content inside the
    frequency band [flow, fupp]
    
    Parameters
    ----------
    x : array_like
        Signal
    dt : float
        Signal sampling rate (s)
    flow, fupp : float
        Blocked frequency band (Hz)
    order : int, optional
        Butterworth filter order. Default 5.
    
    Returns
    -------
    array
        Filtered signal

    Notes
    -----
    SciPy bandpass/bandstop filters designed with b, a are unstable and may result in erroneous filters at higher
    filter orders. Here we use sos (second-order sections) output of filter design instead.

    See Also
    --------
    scipy.signal.butter, scipy.signal.sosfiltfilt
    """
    nyq = 0.5 * 1. / dt                         # nyquist frequency
    normal_cutoff = (flow / nyq, fupp / nyq)    # normalized cut off frequencies
    sos = butter(order, normal_cutoff, btype='bandstop', analog=False, output='sos')
    y = sosfiltfilt(sos, x)
    return y


def threshold(x, thresholds):
    """
    Allow only frequency components whose amplitudes are between the lower threshold value and the upper threshold
    value to pass.
    
    Parameters
    ----------
    x : array_like
        input data signal
    thresholds : tuple
        passing amplitude range, thresholds as fraction of maximum frequency component amplitude
    
    Returns
    -------
    array
        filtered data signal

    Notes
    -----
    FFT filter.

    See Also
    --------
    scipy.fftpack
    """
    real_signal = np.all(np.isreal(x))
    n = x.size
    nfft = int(pow(2, np.ceil(np.log(n) / np.log(2))))
    lth, uth = thresholds   # unpack lower and upper thresholds of passing range

    if real_signal:
        fa = rfft(x, nfft)
        h = np.zeros(np.shape(fa))
        h[(lth*max(abs(fa)) < abs(fa)) & (abs(fa) <= uth*max(abs(fa)))] = 1.0
        x1 = irfft(fa * h, nfft)
    else:
        fa = fft(x, nfft)
        h = np.zeros(np.shape(fa))
        h[(lth*max(abs(fa)) < abs(fa)) & (abs(fa) <= uth*max(abs(fa)))] = 1.0
        x1 = ifft(fa * h, nfft)

    return x1[:n]


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
            return np.array([]), np.array([], dtype=int)
        else:
            return np.array([])

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


def psd(x, dt, **kwargs):
    """
    Estimate power spectral density using Welch’s method.

    Parameters
    ----------
    x : array_like
        Input data signal.
    dt : float
        Time step.
    kwargs : optional
        See `scipy.signal.welch` documentation for available options.

    Returns
    -------
    tuple
        Two arrays: sample frequencies and corresponding power spectral density

    Notes
    -----
    This function basically wraps `scipy.signal.welch` to control defaults etc.

    See also
    --------
    scipy.signal.welch, scipy.signal.periodogram

    """
    x = np.asarray(x)

    # estimate psd using welch's definition
    f, p = welch(x, fs=1./dt, **kwargs)

    return f, p
