"""
Module with functions for handling file operations and calculations. Made for multithreading.
"""

import os

import numpy as np

from .._validation import QatsValueError
from ..fatigue.rainflow import _create_bins
from ..fatigue.rainflow import rebin as rebin_cycles
from ..stats.gumbel import _FIT_METHODS as GUMBEL_FIT_METHODS
from ..stats.gumbel import Gumbel
from ..stats.paper import probability_paper
from ..stats.weibull import Weibull
from ..tsdb import TsDB


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
        Size of segments used for estimating PSD using Welch's method.
        NOTE: the minimum of specified `nperseg` and signal length is used.
    normalize : bool
        Normalize power spectral density on maximum density.

    Returns
    -------
    dict
        Frequency versus power spectral density
    """
    container_out = dict()

    for name, ts in container.items():
        # NOTE: nperseg passed on as minimum of specified value and signal input length
        #       this is done in scipy anyway, which also issues a UserWArnings stating:
        #       "nperseg = {nperseg} is greater than input length = {input_length:d}, using nperseg = {input_length:d}"
        t, _ = ts.get(twin=twin, resample=ts.dt)
        nperseg_ = np.min((nperseg, t.size))
        # calculate power spectral density
        # (resampling to average time step for robustness, necessary for series with varying time step)
        f, s = ts.psd(twin=twin, filterargs=fargs, resample=ts.dt, taperfrac=0.1, nperseg=nperseg_, normalize=normalize)
        container_out[name] = tuple([f, s])

    return container_out


def calculate_rfc(container, twin, fargs, nbins):
    """
    Count cycles using Rainflow counting method, and group them by range

    .. versionchanged :: 5.6.0
        Returns the bar widths with the ranges and counts.

    Parameters
    ----------
    container : dict
        TimeSeries objects
    twin : tuple
        Time window. Time series are cropped to time window before counting cycles.
    fargs : tuple
        Filter arguments. Time series are filtered before counting cycles.
    nbins : int
        Number of equal bins of cycle range, from zero to the largest range, see
        :func:`qats.fatigue.rainflow.rebin`.

    Returns
    -------
    dict
        Cycle range, count and bar width of each time series, as a tuple of arrays: the bin mid points, the number of
        cycles in each bin and the bin width. Empty arrays if no cycles were found.
    """
    container_out = dict()

    for name, ts in container.items():
        # rainflow counting, grouped by range in bins from zero to the largest range, as rebin() does
        cycles = np.asarray(ts.rfc(twin=twin, filterargs=fargs), dtype=float).reshape(-1, 3)
        if cycles.shape[0] == 0:
            container_out[name] = (np.array([]), np.array([]), np.array([]))
            continue

        edges = _create_bins(0.0, cycles[:, 0].max(), n=nbins)
        r, _, c = rebin_cycles(cycles, binby="range", n=nbins).T
        container_out[name] = (r, c, np.diff(edges))

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

        container_out[name] = dict(t=t, x=x, tmin=tmin, xmin=xmin, tmax=tmax, xmax=xmax)

    return container_out


def calculate_stats(container, twin, fargs, minima=False, fit=None, plotting_positions="median"):
    """
    Calculate time series statistics, and the maxima/minima with the fitted Weibull distribution on Rayleigh paper

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
    fit : dict, optional
        Weibull fit options for :meth:`qats.TimeSeries.stats`: `method`, `tail`, `tail_model` and `loc`. Default
        the defaults of :meth:`qats.TimeSeries.stats`.
    plotting_positions : str, optional
        Plotting positions on the probability paper, see :func:`qats.stats.empirical.empirical_cdf`. An ``lse`` fit
        is fitted to these plotting positions, so the fitted line is drawn against the points it was fitted to.

    Returns
    -------
    dict
        Statistics of each time series, see :meth:`qats.TimeSeries.stats`, with the sample and the fitted
        distribution on Rayleigh paper as :class:`qats.stats.paper.ProbabilityPaper` under the key 'paper'
    """
    fit = dict() if fit is None else dict(fit)
    if fit.get("method", "").lower() == "lse":
        fit["plotting_positions"] = plotting_positions
    container_out = dict()
    for name, ts in container.items():
        s = ts.stats(
            twin=twin,
            filterargs=fargs,
            statsdur=10800.0,
            quantiles=(0.37, 0.57, 0.9),
            is_minima=minima,
            include_sample=True,
            **fit,
        )
        s["paper"] = None
        if np.size(s["sample"]) > 0:
            s["paper"] = probability_paper(
                s["sample"],
                Weibull(s["wloc"], s["wscale"], s["wshape"]),
                plotting_positions=plotting_positions,
                tail=s["wtail"],
                tail_model=s["wtail_model"] or "censored",
                is_minima=minima,
            )
        container_out[name] = s
    return container_out


def calculate_gumbel_fit(container, twin, fargs, minima=False, plotting_positions="median", method="pwm"):
    """
    Sample the extreme of each time series, fit a Gumbel distribution, and put both on Gumbel paper

    .. versionchanged :: 5.6.0
        New arguments `plotting_positions` and `method`. The sample and the fitted distribution are returned on
        Gumbel paper, from
        :func:`qats.stats.paper.probability_paper`, and the sample in its original units, also for minima.

    Parameters
    ----------
    container : dict
        TimeSeries objects, at least 2.
    twin : tuple
        Time window. Time series are cropped to time window before extracting the extremes.
    fargs : tuple
        Filter arguments. Time series are filtered before extracting the extremes.
    minima : bool, optional
        Sample the minimum of each time series instead of the maximum. The Gumbel distribution is fitted to the
        minima multiplied by -1, and the paper is mirrored.
    plotting_positions : str, optional
        Plotting positions on the Gumbel paper, see :func:`qats.stats.empirical.empirical_cdf`. An ``lse`` fit is
        fitted to these plotting positions.
    method : str, optional
        Method of the Gumbel fit, see :meth:`qats.stats.gumbel.Gumbel.fit`. Default 'pwm'.

    Returns
    -------
    dict
        - 'sample': the extremes sorted ascending, in their original units
        - 'loc', 'scale': the Gumbel parameters, of the minima multiplied by -1 for minima. NaN if the fit fails.
        - 'method': the method of the fit
        - 'minima', 'plotting_positions': as given
        - 'paper': the sample and the fitted distribution on Gumbel paper,
          :class:`qats.stats.paper.ProbabilityPaper`. Without a fitted line if the fit fails.
    """
    if len(container.keys()) < 2:
        raise ValueError("Select more than 1 time series to fit Gumbel CDF to extremes sample.")
    # an invalid method raises here, not as a failed fit below
    if not isinstance(method, str) or method.lower() not in GUMBEL_FIT_METHODS:
        raise QatsValueError(f"The Gumbel fit method must be one of {', '.join(GUMBEL_FIT_METHODS)}, got {method!r}")
    method = method.lower()

    # the extreme of each time series, in its original units
    sample = list()
    for name, ts in container.items():
        _, x = ts.get(twin=twin, filterargs=fargs)
        sample.append(np.min(x) if minima else np.max(x))
    sample = np.sort(np.array(sample))

    # fit to the maxima, or to the minima multiplied by -1. A fit that fails (e.g. equal extremes, so the scale is
    # zero) gives NaN parameters and the sample without a fitted line.
    try:
        fitargs = dict(plotting_positions=plotting_positions) if method == "lse" else dict()
        gumbel = Gumbel.fit(-sample if minima else sample, method=method, **fitargs)
    except QatsValueError:
        gumbel = None
    paper = probability_paper(sample, gumbel, paper="gumbel", plotting_positions=plotting_positions, is_minima=minima)
    loc, scale = (np.nan, np.nan) if gumbel is None else (gumbel.loc, gumbel.scale)
    return dict(
        loc=loc,
        scale=scale,
        method=method,
        sample=sample,
        minima=minima,
        plotting_positions=plotting_positions,
        paper=paper,
    )


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


def import_files(files):
    """
    Import time series files one by one, so that a file that cannot be read does not stop the others.

    Parameters
    ----------
    files : list
        File names

    Returns
    -------
    TsDB
        Time series data base with the files that could be read
    dict
        Reason by file name, for the files that could not be read

    Notes
    -----
    .. versionadded :: 5.5.0
    """
    db = TsDB()
    failed = dict()
    for file in files:
        if os.path.isdir(file):
            failed[file] = "it is a folder, not a file"
            continue
        try:
            db.load(file, read=False)
        except Exception as err:  # e.g. no reader for the file, or a damaged file: report it and load the others
            failed[file] = str(err)
    return db, failed


def file_dialog_filters(readers):
    """
    Name filters for a file dialog, from file format readers.

    Parameters
    ----------
    readers : list of qats.io.base.Reader
        Readers, e.g. from `qats.io.registry.readers`.

    Returns
    -------
    str
        Filters separated by ';;': all supported files first, then one per reader sorted by description, then all
        files.

    Notes
    -----
    .. versionadded :: 5.5.0
    """
    patterns = []  # of all readers, without repeats
    filters = []
    for reader in sorted(readers, key=lambda r: r.description.lower()):
        filters.append(f"{reader.description} ({' '.join(reader.patterns)})")
        patterns.extend(p for p in reader.patterns if p not in patterns)
    return ";;".join([f"All supported files ({' '.join(patterns)})", *filters, "All files (*)"])


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
