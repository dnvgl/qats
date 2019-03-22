"""
Module with functions for handling file operations and calculations. Made for multithreading.
"""
from qats import TsDB
from qats.weibull import pwm as weibull_pwm

# todo: Create calculate gumbel fit (meant for tools->create extreme value distribution)


def calculate_psd(container, twin, fargs):
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

    Returns
    -------
    dict
        Frequency versus power spectral density
    """
    container_out = dict()

    for name, ts in container.items():
        # resampling to average time step for robustness (necessary for series with varying time step)
        f, s = ts.psd(twin=twin, filterargs=fargs, resample=ts.dt)
        container_out[name] = tuple([f, s])

    return container_out


def calculate_rfc(container, twin, fargs, nbins=256):
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
    nbins : int, optional
        Number of bins in cycle distribution.

    Returns
    -------
    dict
        Cycle magnitude versus count
    """
    container_out = dict()

    for name, ts in container.items():
        # Do Rainflow counting
        pairs = ts.rfc(nbins=nbins, twin=twin, filterargs=fargs)

        # unpack pairs (tuples) in list
        m, c = zip(*pairs)
        container_out[name] = tuple([m, c])

    return container_out


def calculate_weibull_fit(container, twin, fargs, minima=False):
    """
    Calculate time series maxima and fit Weibull distribution

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
        The returned sample is

    Returns
    -------
    dict
        Frequency versus power spectral density
    """
    container_out = dict()

    for name, ts in container.items():
        if not minima:
            # global maxima
            m = ts.maxima(twin=twin, rettime=False, filterargs=fargs)
            f = 1
        else:
            # global minima
            m = ts.minima(twin=twin, rettime=False, filterargs=fargs)
            f = -1     # flip sample for weibull fit

        if (m is not None) and (len(m) > 1):
            # fit weibull distribution parameter
            a, b, c = weibull_pwm(f * m)
        else:
            # no sample or too small sample for estimation of parameters
            m = a = b = c = None

        container_out[name] = dict(sample=m, loc=a, scale=b, shape=c, minima=minima)

    return container_out


def export_to_file(name, db, keys, twin, fargs):
    """
    Export selected time series to file

    Parameters
    ----------
    name : str
        File name
    db : TsDB
        Time series data base
    keys : list
        Name of time series to export
    twin : tuple
        Time window. Time series are cropped to time window before export.
    fargs : tuple
        Filter arguments. Time series are filtered before export
    """
    db.export(name, keys=keys, exist_ok=True, basename=False, twin=twin, filterargs=fargs)


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


def read_timeseries(db, keys):
    """
    Read time series from files

    Parameters
    ----------
    db : TsDB
        Time series data base
    keys : list
        Name of time series to read

    Returns
    -------
    dict
        Container with TimeSeries objects
    """
    return db.get_many_ts(keys=keys, store=False)

