"""
Readers for SINTEF Ocean test data exhange format based on the Matlab .mat file.

Works for matlab file format version <=7.2 and >=7.3.
"""
import fnmatch
import numpy as np
from typing import List, Tuple, Union
from datetime import datetime, timedelta
from pymatreader import read_mat


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
    _tn = fnmatch.filter(data.keys(), '[Tt]ime*')
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

