import numpy as np
import fnmatch
from datetime import datetime, timedelta
from scipy.io import loadmat


# TODO: Rename this module matlab_v72 when we have a use case on v. 7.3 or newer. The new module should be named matlab


def read_keys(path):
    """
    Read keys from MATLAB .mat file.

    Parameters
    ----------
    path : str
        Keyfile path

    Returns
    -------
    str
        Key of the time array
    list
        Time series keys

    Notes
    -----
    This code works for Matlab version < 7.3. For version >= 7.3, .mat is on hdf5 format
    <http://pyhogs.github.io/reading-mat-files.html>.
    """
    mat = loadmat(path)
    keys = list(mat.keys())  # make list, since type dict_keys does not support remove()

    # identify time key, check that there is only one
    timekeys = fnmatch.filter(keys, '[Tt]ime*')
    if len(timekeys) < 1:
        raise KeyError("File does not contain a time vector: %s" % path)
    elif len(timekeys) > 1:
        raise KeyError("Duplicate time vectors on file: %s" % path)

    # filter keys, keep only np.ndarrays of same size as time array
    timekey = timekeys[0]
    tsize = mat[timekey].size
    keys = [k for k, v in mat.items() if (isinstance(v, np.ndarray) and v.size == tsize)]

    # remove time key from list of time series
    keys.remove(timekey)

    return timekey, keys


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
