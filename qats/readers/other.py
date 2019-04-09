"""
Readers for various time series file formats
"""
import fnmatch
import numpy as np


def read_dat_keys(path):
    """
    Read data series keys/names from an ASCII file with data arranged column-wise. The
    keys are in the first non-commented row.

    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    list
        Name of data series (keys)
        
    Notes
    -----
    The keys are extracted from the header row (the first non-commented row). The comment
    character is '#'.
    """
    keys = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                # keys in first row that is not a comment
                keys = line.split()
                break
    
    if keys is not None:
        # identify time key, check that there is only one
        timekeys = fnmatch.filter(keys, '[Tt]ime*')
        if len(timekeys) < 1:
            raise KeyError(f"The file '{path}' does not contain a time vector")
        elif len(timekeys) > 1:
            raise KeyError(f"The file '{path}' contain duplicate time vectors")

        # remove time key from list of time series
        keys.remove(timekeys[0])

    return keys


def read_dat_data(path, ind=None):
    """
    Read time series arranged column wise on ascii formatted file.

    Parameters
    ----------
    path : str
        File path
    ind : list of integers, optional
        Defines which responses to include in returned array. Each of the indices in `ind` refer the sequence number
        of the reponses. Note that `0` is the index of the time array.

    Returns
    -------
    array
        Data

    Notes
    -----
    If ``ind`` is specified, time array is only included if `0` is included in the specified indices.

    If ``ind`` is specified, the response arrays (1-D) are stored in the returned 2-D array in the same order as
    specified. I.e. if ``ind=[0,10,2,3]``, then the response array with index `10` on .ts file is obtained from
    ``data[1,:]``.

    """
    with open(path, 'r') as f:
        # skip commmented lines at start of file
        for line in f:
            if not line.startswith("#"):
                # skip all comment lines
                break

        # load data from the remaining rows as an array, skip the header row with keys (first row after comments)
        data = np.loadtxt(f, skiprows=1, usecols=ind, unpack=True)

    return data

