"""
Module with functions for handling file operations and calculations. Made for multithreading.
"""
import numpy as np
from ..tsdb import TsDB
from ..stats.gumbel import pwm as gumbel_pwm
from ..fatigue.rainflow import rebin as rebin_cycles


def calculate_psd(container, twin, fargs, nperseg, normalize):
    """
    Calculate power spectral density and return as numpy arrays

    Parameters
    ----------
    container : dict
        TimeSeries objects
    twin : tuple
        Time window. Time series are cropped to time window before estimating psd.
    fargs : tuple
        Filter arguments. Time series are filtered before estimating psd.
    nperseg : int
        Size of segments used for ustimating PSD using Welch's method.
    normalize : bool
        Normalize power spectral density on maximum density.

    Returns
    -------
    dict
        Frequency versus power spectral density
    """
    container_out = dict()

    for name, ts in container.items():
        # resampling to average time step for robustness (necessary for series with varying time step)
        f, s = ts.psd(twin=twin, filterargs=fargs, resample=ts.dt, taperfrac=0.1, nperseg=nperseg, normalize=normalize)
        container_out[name] = tuple([f, s])

    return container_out


def calculate_rfc(container, twin, fargs, nbins):
    """
    Count cycles using Rainflow counting method

    Parameters
    ----------
    container : dict
        TimeSeries objects
    twin : tuple
        Time window. Time series are cropped to time window before estimating psd.
    fargs : tuple
        Filter arguments. Time series are filtered before estimating psd.
    nbins : int
        Number of bins in cycle distribution.

    Returns
    -------
    dict
        Cycle magnitude versus count
    """
    container_out = dict()

    for name, ts in container.items():
        # rainflow counting
        cycles = ts.rfc(twin=twin, filterargs=fargs)

        # rebin
        if nbins is not None:
            cycles = rebin_cycles(cycles, binby='range', n=nbins)

        # unpack pairs (tuples) in list
        r, _, c = zip(*cycles)
        container_out[name] = tuple([r, c])

    return container_out


def calculate_trace(container, twin, fargs):
    """
    Calculate trace and peaks/troughs of filtered time series

    Parameters
    ----------
    container : dict
        TimeSeries objects
    twin : tuple
        Time window. Time series are cropped to time window before extracting maxima.
    fargs : tuple
        Filter arguments. Time series are filtered before extracting maxima.

    Returns
    -------
    dict
        Filtered and windowed time series and peaks/throughs
    """
    container_out = dict()

    for name, ts in container.items():
        t, x = ts.get(twin=twin, filterargs=fargs)
        xmin, tmin = ts.minima(twin=twin, filterargs=fargs, rettime=True)
        xmax, tmax = ts.maxima(twin=twin, filterargs=fargs, rettime=True)

        container_out[name] = dict(t=t, x=x, tmin=tmin, xmin=xmin,
                                   tmax=tmax, xmax=xmax)

    return container_out


def calculate_stats(container, twin, fargs, minima=False):
    """
    Calculate time series statistics

    Parameters
    ----------
    container : dict
        TimeSeries objects
    twin : tuple
        Time window. Time series are cropped to time window before extracting maxima.
    fargs : tuple
        Filter arguments. Time series are filtered before extracting maxima.
    minima : bool, optional
        Fit to sample of minima instead of maxima. The sample is multiplied by -1 prior to parameter estimation.

    Returns
    -------
    dict
        Filtered and windowed time series and peaks/throughs
    """
    return {name: ts.stats(twin=twin, filterargs=fargs, statsdur=10800., quantiles=(0.37, 0.57, 0.9),
                           is_minima=minima, include_sample=True) for name, ts in container.items()}


def calculate_gumbel_fit(container, twin, fargs, minima=False):
    """
    Calculate time series extremes and fit Gumbel distribution

    Parameters
    ----------
    container : dict
        TimeSeries objects
    twin : tuple
        Time window. Time series are cropped to time window before extracting maxima.
    fargs : tuple
        Filter arguments. Time series are filtered before extracting maxima.
    minima : bool, optional
        Fit to sample of minima instead of maxima. The sample is multiplied by -1 prior to parameter estimation.

    Returns
    -------
    dict
        Sample and fitted gumbel distribution parameters
    """
    if len(container.keys()) < 2:
        raise ValueError(f"Select more than 1 time series to fit Gumbel CDF to extremes sample.")

    # create sample of extremes
    sample = list()
    for name, ts in container.items():
        _, x = ts.get(twin=twin, filterargs=fargs)

        if minima:
            # sample minimum value
            sample.append(np.min(x))
        else:
            # sample maximum value
            sample.append(np.max(x))

    # numpy array sorted ascending
    sample = np.sort(np.array(sample))

    # flip sample of minima to enable Gumbel fit
    if minima:
        sample *= -1

    # estimate gumbel distribution parameters
    loc, scale = gumbel_pwm(sample)
    return dict(loc=loc, scale=scale, sample=sample, minima=minima)


def export_to_file(filename, db, names, twin, fargs):
    """
    Export selected time series to file

    Parameters
    ----------
    filename : str
        File name
    db : TsDB
        Time series data base
    names : list
        Name of time series to export
    twin : tuple
        Time window. Time series are cropped to time window before export.
    fargs : tuple
        Filter arguments. Time series are filtered before export
    """
    db.export(filename, names=names, exist_ok=True, basename=False, twin=twin, filterargs=fargs)


def import_from_file(files):
    """
    Import time series files

    Parameters
    ----------
    files : list
        File names

    Returns
    -------
    TsDB
        Time series data base
    """
    db = TsDB()
    db.load(files, read=False)
    return db


def read_timeseries(db, names):
    """
    Read time series from files

    Parameters
    ----------
    db : TsDB
        Time series data base
    names : list
        Name of time series to read

    Returns
    -------
    dict
        Container with TimeSeries objects
    """
    return db.getm(names=names, store=False)

