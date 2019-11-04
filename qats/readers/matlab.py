"""
Readers for Matlab v >= 7.3 binary time series files, which ar in HDF5 format
Built on Sintef Ocean's model test export format
"""
import os
import hdf5storage
import fnmatch


def read_names(path, verbose=False):
    """
    Extracts time series names for `mat` data sets containing (or; interpreted as) time series from h5 based mat-file
    exported from Matlab v=>7.3.

    Parameters
    ----------
    path: str
        Path (relative or absolute) to h5-file
    verbose: bool, optional
        If True, print info.

    Returns
    -------
    list
        List of time series names (datasets)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % path)

    if verbose:
        print('Identifying datasets on %s ...' % path)

    names = []  # list of dataset names/keys (only for data sets that are time series)

    # hdfstorage
    mat = hdf5storage.loadmat(path)
    shape = mat['chan_names'].shape

    if shape[0] == 1 and shape[1] >= 1:
        for name in mat['chan_names'][0]:
            names.append((str(name[0][0])))
    elif shape[0] >= 1 and shape[1] == 1:
        for name in mat['chan_names']:
            names.append((str(name[0][0][0])))
    else:
        raise KeyError("File does not contain a channel name vector: %s" % path)

    # identify time key, check that there is only one
    _tn = fnmatch.filter(names, '[Tt]ime*')
    if len(_tn) < 1:
        raise KeyError("File does not contain a time vector: %s" % path)
    elif len(_tn) > 1:
        raise KeyError("Duplicate time vectors on file: %s" % path)

    # remove time key from list of time series
    timename = _tn[0]
    names.remove(timename)

    print(names)

    return timename, names


def read_data(path, names, verbose=False):
    """
        Read time-series from MATLAB .mat v => 7.3 file.

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


if __name__ == '__main__':
    path = os.path.join('C:\\Users', 'ebg', 'OneDrive - SevanSSP AS', 'Projects', 'Cambo', 'ModelTest', 'CE21130.mat')
    tname, names = read_names(path)
    data = read_data(path, names)
    x1 = data[names[0]]  # first data series