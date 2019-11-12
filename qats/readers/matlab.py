"""
Readers for SINTEF Ocean test data exhange format based on the Matlab .mat file.
"""
import os
import hdf5storage
import fnmatch
import numpy as np
from scipy.io import loadmat
from datetime import datetime, timedelta


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
    try:
        # scipy loadmat will raise an NotImplementedError if file is v7.3 or newer (HDF5)
        tname, names = read_names_v72(path)
    except NotImplementedError:
        tname, names = read_names_v73(path)

    return tname, names


def read_names_v72(path):
    """
    Read time series names from SINTEF Ocean test data exhange format based on the Matlab .mat file version < 7.3.

    Parameters
    ----------
    path : str
        MAT file path (relative or absolute)

    Returns
    -------
    str
        Name of the time array
    list
        Time series names

    Examples
    --------
    >>> tname, names = read_names('data.mat')

    Notes
    -----
    This code works for Matlab version < 7.3. For version >= 7.3, .mat is on hdf5 format
    <http://pyhogs.github.io/reading-mat-files.html>.
    """
    mat = loadmat(path)
    names = list(mat.keys())  # make list, since type dict_keys does not support remove()

    # identify time key, check that there is only one
    _tn = fnmatch.filter(names, '[Tt]ime*')
    if len(_tn) < 1:
        raise KeyError("File does not contain a time vector: %s" % path)
    elif len(_tn) > 1:
        raise KeyError("Duplicate time vectors on file: %s" % path)

    # filter names, keep only np.ndarrays of same size as time array
    timename = _tn[0]
    tsize = mat[timename].size
    names = [k for k, v in mat.items() if (isinstance(v, np.ndarray) and v.size == tsize)]

    # remove time key from list of time series
    names.remove(timename)

    return timename, names


def read_names_v73(path):
    """
    Read time series names from SINTEF Ocean test data exhange format based on the Matlab .mat file version >= 7.3.

    Parameters
    ----------
    path: str
        File path

    Returns
    -------
    str
        Name of the time array
    list
        List of time series names (datasets)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % path)

    # hdfstorage
    mat = hdf5storage.loadmat(path)
    shape = mat['chan_names'].shape
    names = list()

    if shape[0] == 1 and shape[1] >= 1:
        for name in mat['chan_names'][0]:
            names.append((str(name[0][0])))
    elif shape[0] >= 1 and shape[1] == 1:
        for name in mat['chan_names']:
            names.append((str(name[0][0][0])))
    else:
        raise KeyError(f"File does not contain a channel name vector: {path}")

    # identify time key, check that there is only one
    _tn = fnmatch.filter(names, '[Tt]ime*')
    if len(_tn) < 1:
        raise KeyError(f"File does not contain a time vector: {path}")
    elif len(_tn) > 1:
        raise KeyError(f"Duplicate time vectors on file: {path}")

    # remove time key from list of time series
    timename = _tn[0]
    names.remove(timename)

    return timename, names


def read_data(path, names):
    """
    Read time series data from SINTEF Ocean test data exhange format based on the Matlab .mat file.

    Parameters
    ----------
    path : str
        File path
    names : list
        Names of the requested time series incl. the time array itself

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
    try:
        # scipy loadmat will raise an NotImplementedError if file is v7.3 or newer (HDF5)
        data = read_data_v72(path, names)
    except NotImplementedError:
        data = read_data_v73(path, names)

    return data


def read_data_v72(path, names):
    """
    Read time series data from SINTEF Ocean test data exhange format based on the Matlab .mat file version < 7.3.

    Parameters
    ----------
    path : str
        Path (relative or absolute) to the time series file.
    names : list
        Names of the requested time series incl. the time array itself

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
    return loadmat(path, squeeze_me=True, variable_names=names)


def read_data_v73(path, names):
    """
    Read time series names from SINTEF Ocean test data exhange format based on the Matlab .mat file version >= 7.3.

    Parameters
    ----------
    path : str
        File path
    names : list
        Names of the requested time series incl. the time array itself

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
    mat_indices = []
    data = {}

    mat = hdf5storage.loadmat(path)
    shape = mat['chan_names'].shape

    if shape[0] == 1 and shape[1] >= 1:
        for name in mat['chan_names'][0]:
            mat_indices.append((str(name[0][0])))
    elif shape[0] >= 1 and shape[1] == 1:
        for name in mat['chan_names']:
            mat_indices.append((str(name[0][0][0])))
    else:
        raise KeyError("File does not contain a channel name vector: %s" % path)

    for name in names:
        index_pos = mat_indices.index(name)
        data[name] = mat['data'][:, index_pos]

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

