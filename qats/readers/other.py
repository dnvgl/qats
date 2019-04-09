"""
Readers for various time series file formats
"""
import fnmatch
import numpy as np


def read_dat_names(path):
    """
    Read time series names from an ASCII file with data arranged column-wise. The names are in the first non-commented
    row.

    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    list
        Time series names
        
    Notes
    -----
    The names are extracted from the header row (the first non-commented row). The comment character is '#'. Time is
    assumed to be in the first column.
    """
    names = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                # names in first row that is not a comment
                names = line.split()
                break
    
    if names is not None:
        # identify time key, check that there is only one
        timekeys = fnmatch.filter(names, '[Tt]ime*')
        if len(timekeys) < 1:
            raise KeyError(f"The file '{path}' does not contain a time vector")
        elif len(timekeys) > 1:
            raise KeyError(f"The file '{path}' contain duplicate time vectors")

    # skip the time array name assumed to be in the first column
    return names[1:]


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
        Time and data

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
        data = np.loadtxt(f, skiprows=0, usecols=ind, unpack=True)

    return data

