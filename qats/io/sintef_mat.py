"""
Readers for SINTEF Ocean test data exhange format based on the Matlab .mat file.

Works for MAT file versions 4, 6, 7 and 7.3 (tested with the sample files in data/).
"""

import fnmatch
from datetime import datetime, timedelta
from typing import List, Tuple, Union

import numpy as np
from pymatreader import read_mat

from .base import Reader, SeriesData, SeriesInfo, SourceFile


def read_names(path):
    """
    Read time series names from SINTEF Ocean test data exhange format based on the Matlab .mat file.

    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    str
        Name of the time array
    list
        Time series names

    Examples
    --------
    >>> tname, names = read_names('data.mat')

    """
    data = read_data(path)

    # identify time key, check that there is only one
    _tn = fnmatch.filter(data.keys(), "[Tt]ime*")
    if len(_tn) < 1:
        raise KeyError("File does not contain a time vector: %s" % path)
    elif len(_tn) > 1:
        raise KeyError("Duplicate time vectors on file: %s" % path)
    else:
        timename = _tn[0]

    # Keep only arrays of same size as time array
    tsize = data[timename].size
    names = [k for k, v in data.items() if (isinstance(v, np.ndarray) and v.size == tsize) and k != timename]

    return timename, names


def read_data(path: str, names: Union[List[str], Tuple[str]] = None):
    """
    Read time series data from SINTEF Ocean test data exhange format based on the Matlab .mat file.

    Parameters
    ----------
    path : str
        File path
    names : Union[List[str], Tuple[str]], optional
        Names of the requested time series incl. the time array itself. Defaults to all time series on the file.

    Returns
    -------
    dict
        Time and data

    Examples
    --------
    >>> tname, names = read_names('data.mat')
    >>> data = read_data('data.mat', [tname, *names])
    >>> t = data[tname]   # time
    >>> x1 = data[names[0]]  # first data series

    """
    # ignore the data field (if it exists) which contains the time series data in
    # latest file format
    data = read_mat(path)

    if "chan_names" in data.keys():
        # latest exhange format based on v.7.3 mat files
        data = dict(zip(data["chan_names"], np.transpose(data["data"])))
    else:
        # exhange format based on v.7.2 mat files
        ignored = ["comment", "fs", "test_num", "test_date", "__header__", "__version__", "__globals__"]
        data = {k: v for k, v in data.items() if k not in ignored}

    if names is not None:
        return {k: v for k, v in data.items() if k in names}
    else:
        return data


def _datenums_to_datetime(timearr):
    """
    Convert array of MATLAB datenum floats to array of datetime objects.

    Parameters
    ----------
    timearr: array_like or float
        Time array.

    Returns
    -------
    array
        Array of datetime objects, same shape as input array.
    """

    def convert(dn):
        # ref: https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
        _dtg = datetime.fromordinal(int(dn)) + timedelta(days=dn % 1) - timedelta(days=366)
        return _dtg

    # return quickly if float (not array) is given
    if np.ndim(timearr) == 0:
        return convert(timearr)

    # convert array
    timearr = np.asarray(timearr)
    was_shape = timearr.shape
    timearr = timearr.flatten()
    dtarray = np.array([convert(t) for t in timearr])
    return dtarray.reshape(was_shape)


class _MatSource(SourceFile):
    def __init__(self, path):
        super().__init__(path)
        self._timename = None
        self._names = None

    def _scan(self):
        if self._names is None:
            self._timename, self._names = read_names(self.path)

    def series(self):
        self._scan()
        return [SeriesInfo(name) for name in self._names]

    def read(self, names):
        self._scan()
        data = read_data(self.path, [self._timename, *names])
        return [SeriesData(name, data[self._timename], data[name]) for name in names]

    def _legacy_index(self, name):
        """Position of the series on file, as stored in `TsDB.register_indices` before 5.5.0."""
        return None


class MatReader(Reader):
    """
    Reader for MATLAB files (``.mat``, versions 4, 6, 7 and 7.3) in the SINTEF Ocean test data exchange format.

    .. versionadded :: 5.5.0
    """

    name = "matlab"
    description = "MATLAB (SINTEF Ocean test data exchange format)"
    patterns = ("*.mat",)

    def open(self, path):
        return _MatSource(path)
