#!/usr/bin/env python
# encoding: utf8
"""
Module with functions for signal processing.

"""
import sys
import numpy as np
from scipy.fftpack import fft, ifft, rfft, irfft, fftfreq, rfftfreq


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


def lowpass(x, dt, fc):
    """
    Low pass filter data signal x at cut off frequency fc, blocking harmonic content above fc.
    
    Parameters
    ----------
    x : array_like
        input data signal
    dt : float
        time step
    fc : float
        cut off frequency (Hz)
    
    Returns
    -------
    array
        filtered data signal

    Notes
    -----
    FFT filter.
    
    """

    if fc == 0:
        fc = sys.float_info.epsilon
    real_signal = np.all(np.isreal(x))
    n = x.size
    nfft = int(pow(2, np.ceil(np.log(n) / np.log(2))))
    
    if real_signal:
        fa = rfft(x, nfft)
        f = rfftfreq(nfft, d=dt)
        h = np.zeros(np.shape(f))
        h[f.__abs__() <= fc] = 1.
        x1 = irfft(fa * h, nfft)
    else:
        fa = fft(x, nfft)
        f = fftfreq(nfft, d=dt)
        h = np.zeros(np.shape(f))
        h[f.__abs__() <= fc] = 1.
        x1 = ifft(fa * h, nfft)

    return x1[:n]


def highpass(x, dt, fc):
    """
    High pass filter data signal x at cut off frequency fc, blocking harmonic content below fc.
    
    Parameters
    ----------
    x : array_like
        input data signal
    dt : float
        time step
    fc : float
        cut off frequency (Hz)
    
    Returns
    -------
    array
        filtered data signal

    Notes
    -----
    FFT filter.
       
    """

    if fc == 0:
        fc = sys.float_info.epsilon
    real_signal = np.all(np.isreal(x))
    n = x.size
    nfft = int(pow(2, np.ceil(np.log(n) / np.log(2))))

    if real_signal:
        fa = rfft(x, nfft)
        f = rfftfreq(nfft, d=dt)
        h = np.zeros(np.shape(f))
        h[f.__abs__() >= fc] = 1.
        x1 = irfft(fa * h, nfft)
    else:
        fa = fft(x, nfft)
        f = fftfreq(nfft, d=dt)
        h = np.zeros(np.shape(f))
        h[f.__abs__() >= fc] = 1.
        x1 = ifft(fa * h, nfft)

    return x1[:n]


def bandpass(x, dt, flow, fupp):
    """
    Band pass filter data signal x at cut off frequencies flow and fupp, blocking harmonic content outside the
    frequency band [flow, fupp]
    
    Parameters
    ----------
    x : array_like
        input data signal
    dt : float
        time step
    flow, fupp : float
        passing frequency band (Hz)
    
    Returns
    -------
    array
        filtered data signal

    Notes
    -----
    FFT filter.
    
    """
    real_signal = np.all(np.isreal(x))
    n = x.size
    nfft = int(pow(2, np.ceil(np.log(n) / np.log(2))))
    if flow == 0:
        flow = sys.float_info.epsilon
    if fupp == 0:
        fupp = sys.float_info.epsilon
    
    if real_signal:
        fa = rfft(x, nfft)
        f = rfftfreq(nfft, d=dt)
        h = np.zeros(np.shape(f))
        h[(f.__abs__() >= flow) & (f.__abs__() <= fupp)] = 1.
        x1 = irfft(fa * h, nfft)
    else:
        fa = fft(x, nfft)
        f = fftfreq(nfft, d=dt)
        h = np.zeros(np.shape(f))
        h[(f.__abs__() >= flow) & (f.__abs__() <= fupp)] = 1.
        x1 = ifft(fa * h, nfft)

    return x1[:n]


def bandblock(x, dt, flow, fupp):
    """
    Band block filter data signal x at cut off frequencies flow and fupp, blocking harmonic content inside the
    frequency band [flow, fupp]
    
    Parameters
    ----------
    x : array_like
        input data signal
    dt : float
        time step
    flow, fupp : float
        blocked frequency band (Hz)
    
    Returns
    -------
    array
        filtered data signal

    Notes
    -----
    FFT filter.
       
    """
    real_signal = np.all(np.isreal(x))
    n = x.size
    nfft = int(pow(2, np.ceil(np.log(n) / np.log(2))))
    if flow == 0:
        flow = sys.float_info.epsilon
    if fupp == 0:
        fupp = sys.float_info.epsilon

    if real_signal:
        fa = rfft(x, nfft)
        f = rfftfreq(nfft, d=dt)
        h = np.ones(np.shape(f))
        h[(f.__abs__() >= flow) & (f.__abs__() <= fupp)] = 0.
        x1 = irfft(fa * h, nfft)
    else:
        fa = fft(x, nfft)
        f = fftfreq(nfft, d=dt)
        h = np.ones(np.shape(f))
        h[(f.__abs__() >= flow) & (f.__abs__() <= fupp)] = 0.
        x1 = ifft(fa * h, nfft)

    return x1[:n]


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


