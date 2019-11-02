"""
Readers for binary time series files exported from MATLAB
"""
import numpy as np
import fnmatch
from datetime import datetime, timedelta
from scipy.io import loadmat
import os


def read_names(path):
    """
    Read time series names from MATLAB .mat file.

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


def read_data(path, names, verbose=False):
    """
    Read time-series from MATLAB .mat file.

    Parameters
    ----------
    path : str
        Path (relative or absolute) to the time series file.
    names : list
        Names of the requested time series incl. the time array itself
    verbose : bool, optional
        Increase verbosity

    Returns
    -------
    dict
        Time and data

    Examples
    --------
    >>> tname, names = read_names('data.mat')
    >>> data = read_data('data.mat', names)
    >>> t = data[tname]   # time
    >>> x1 = data[names[0]]  # first data series
    """
    if verbose:
        print('Reading %s ...' % path)

    data = loadmat(path, squeeze_me=True, variable_names=names)
    return data


def _datenums_to_datetime(timearr):
    """
    Convert array of MATLAB datenum floats to array of datetime objects.

    Parameters
    ----------
    timearr: array_like or float

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


if __name__ == '__main__':
    path = os.path.join('C:\\Users', 'ebg', 'OneDrive - SevanSSP AS', 'Projects', 'Cambo', 'ModelTest', 'data.mat')
    tname, names = read_names(path)
    data = read_data(path, names)
    x1 = data[names[0]]  # first data series