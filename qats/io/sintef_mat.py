"""
Readers for SINTEF Ocean test data exhange format based on the Matlab .mat file.

Works for MAT file versions 4, 6, 7 and 7.3 (tested with the sample files in data/).
"""

import fnmatch
from datetime import datetime, timedelta
from typing import List, Tuple, Union

import h5py
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

    .. versionchanged :: 5.5.0
        Files with the names in ``chan_names`` are no longer read in full to list the names.
    """
    channels = _read_channels(path)
    if channels is None:
        return _series_names(read_data(path), path)
    keys = list(dict.fromkeys(channels[0]))
    timename = _time_name(keys, path)
    return timename, [k for k in keys if k != timename]


def read_units(path):
    """
    Read the units of the time series from SINTEF Ocean test data exhange format based on the Matlab .mat file.

    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    dict
        Unit by name, for the time array and the time series that have a unit on file. Empty if the file holds no
        units.

    Notes
    -----
    Units are read from the variable ``chan_units`` (one unit per name in ``chan_names``). Files with one variable
    per series hold no units.

    The units are returned as stored. Non-ASCII characters lost when a file was written (e.g. ``²`` replaced by
    ``\\ufffd`` in a version 6 file) cannot be restored.

    .. versionadded :: 5.5.0

    Examples
    --------
    >>> units = read_units('data.mat')
    >>> units['Time']
    's'
    """
    channels = _read_channels(path)
    return {} if channels is None else channels[1]


def _read_channels(path):
    """
    Names (in file order, possibly repeated) and units of a file with the series as rows of one array (``data``,
    with the names in ``chan_names``), read without reading the data. None for files with one variable per series.
    """
    data = read_mat(path, variable_names=["chan_names", "chan_units"])
    if "chan_names" not in data:
        return None
    names = _as_list(data["chan_names"])
    units = dict()
    if "chan_units" in data:
        for name, unit in zip(names, _as_list(data["chan_units"])):
            unit = str(unit).strip()
            if unit:
                units[name] = unit
    return names, units


def _read_channel_rows(path, names, requested):
    """
    The rows of the channel array for the requested names, by name. The last of repeated names wins, as in
    `_read_file`. Files of MAT version 7.3 are HDF5 files, where each row is read on its own; other versions are read
    in full.
    """
    position = {name: i for i, name in enumerate(names)}
    if h5py.is_hdf5(path):
        # MATLAB stores the transpose of an array in HDF5: a row here is a column of `data` in MATLAB, i.e. a series
        with h5py.File(path, "r") as f:
            dset = f["data"]
            return {name: dset[position[name], :] for name in requested}
    rows = np.transpose(read_mat(path, variable_names=["data"])["data"])
    return {name: rows[position[name]] for name in requested}


def _read_file(path):
    """Time series and units on file: (dict of arrays by name, dict of units by name)."""
    data = read_mat(path)
    units = dict()

    if "chan_names" in data.keys():
        # latest exhange format based on v.7.3 mat files
        names = _as_list(data["chan_names"])
        if "chan_units" in data.keys():
            for name, unit in zip(names, _as_list(data["chan_units"])):
                unit = str(unit).strip()
                if unit:
                    units[name] = unit
        data = dict(zip(names, np.transpose(data["data"])))
    else:
        # exhange format based on v.7.2 mat files
        ignored = ["comment", "fs", "test_num", "test_date", "__header__", "__version__", "__globals__"]
        data = {k: v for k, v in data.items() if k not in ignored}

    return data, units


def _as_list(value):
    """A cell array of strings as a list; pymatreader returns a single string for a cell array of one."""
    return [value] if isinstance(value, str) else list(value)


def _time_name(keys, path):
    """Name of the time array among the names on a file; there must be exactly one."""
    _tn = fnmatch.filter(keys, "[Tt]ime*")
    if len(_tn) < 1:
        raise KeyError("File does not contain a time vector: %s" % path)
    elif len(_tn) > 1:
        raise KeyError("Duplicate time vectors on file: %s" % path)
    return _tn[0]


def _series_names(data, path):
    """Name of the time array and the time series names, from the arrays on a file."""
    timename = _time_name(list(data.keys()), path)

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

    .. versionchanged :: 5.5.0
        From files of MAT version 7.3 with the names in ``chan_names``, only the requested series are read.
    """
    channels = _read_channels(path)
    if channels is not None:
        keys = list(dict.fromkeys(channels[0]))
        requested = keys if names is None else [k for k in keys if k in names]
        return _read_channel_rows(path, channels[0], requested)

    data = _read_file(path)[0]

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
        self._units = None

    def _scan(self):
        if self._names is None:
            channels = _read_channels(self.path)
            if channels is None:
                # one variable per series: names and time key from one read of the file
                data, self._units = _read_file(self.path)
                self._timename, self._names = _series_names(data, self.path)
            else:
                # names and units only, without the data
                self._units = channels[1]
                keys = list(dict.fromkeys(channels[0]))
                self._timename = _time_name(keys, self.path)
                self._names = [k for k in keys if k != self._timename]

    def series(self):
        self._scan()
        return [SeriesInfo(name, self._units.get(name)) for name in self._names]

    def read(self, names):
        self._scan()
        data = read_data(self.path, [self._timename, *names])
        return [SeriesData(name, data[self._timename], data[name], unit=self._units.get(name)) for name in names]

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
