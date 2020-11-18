"""
Readers for HDF5 formatted time series files exported from SIMA
"""
import os
import h5py
import numpy as np


def read_names(path, verbose=False):
    """
    Extracts time series names for `h5` data sets containing (or; interpreted as) time series from h5-file exported
    from SIMA.

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
    n_dsets = 0
    n_groups = 0

    with h5py.File(path, "r") as f:
        allkeys = list(f.keys())
        for key in allkeys:
            if isinstance(f[key], h5py.Dataset):
                n_dsets += 1
                # check if dataset is a time series
                dset = f[key]
                timeinfo = _timearray_info(dset)

                # include data set if it is interpreted as a time series (i.e. if timeinfo is not None)
                if timeinfo is not None:
                    names.append(key)

            elif isinstance(f[key], h5py.Group):
                n_groups += 1
                # print("group: ", key)
                for i in f[key]:
                    allkeys.append('{}/{}'.format(key, i))

            else:
                raise Exception("unexpected error: %s is not a dataset or a group" % key)

    if verbose:
        print("   no. of groups      : %4d" % n_groups + "   (incl. subgroups)")
        print("   no. of datasets    : %4d" % n_dsets)
        print("   no. of time series : %4d" % len(names) + "   (datasets interpreted as time series)")

    # replace all slashes with backslash
    names = [_.replace('/', "\\") for _ in names]

    return names


def read_data(path, names=None, verbose=False):
    """
    Extracts time series data from `.h5` (or `hdf5`) file exported from SIMA.


    Parameters
    ----------
    path: str
        File name.
    names: str or list, optional
        Timeseries/dataset names. If None (default), all time series are read.
    verbose:  bool, optional
        Increase verbosity

    Returns
    -------
    list
        List of arrays with time and data

    Notes
    -----
    If `keys` is not specified, all datasets that are interpreted as time series will be read. See also `read_h5_keys`.

    Note that the hdf5 is an extremely flexible file format and this reader is tailored to the SIMA h5 file format.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % path)

    if verbose:
        print('Reading %s ...' % path)

    if isinstance(names, str):
        names = [names]
    elif type(names) in (list, tuple):
        pass
    elif names is None:
        # if names not specified, get all names
        names = read_names(path)
    else:
        raise TypeError("`names` must be str/list/tuple, got: %s" % type(names))

    # correct slashes
    names = [_.replace('\\', '/') for _ in names]
    arrays = []

    with h5py.File(path, "r") as f:
        for name in names:
            dset = f[name]

            if not isinstance(dset, h5py.Dataset):
                raise TypeError("expected to get h5py.Dataset, got %s (for name '%s')" % (type(dset), name))

            # get values (data array), check size and dimensions
            data = dset[:]  # dset.value
            # todo: consider if check of data type is really neccessary -- if not, dset.ndim, dset.size etc. may be used
            if not isinstance(data, np.ndarray):
                raise NotImplemented("only value of type np.ndarray is implemented, got: %s (for name '%s')" %
                                     (type(data), name))
            if data.ndim != 1:
                raise NotImplemented("only 1-dimensional arrays implemented, got ndim=%d (for name '%s')" %
                                     (data.ndim, name))
            data = data.flatten()   # flatten data array
            nt = data.size          # number of time steps

            # attrs = dset.attrs  # dict with dataset attributes

            # establish or extract time array
            timeinfo = _timearray_info(dset)
            if timeinfo is None:
                raise Exception("no time info extracted for dataset '%s'" % name)
            kind = timeinfo["kind"]
            if kind == "sima":
                t_start = timeinfo["start"]
                dt = timeinfo["dt"]
                t_end = t_start + (nt-1) * dt
                timearr, step = np.linspace(t_start, t_end, nt, retstep=True)
                if not np.isclose(dt, step):  # dt == step:
                    raise Exception("unexpected error: `dt` should be %s but is %s" % (dt, step))

            # todo: expand when _get_h5_timearray_info() has been expanded

            # verify that time array has correct shape (==> should be same as `data` shape)
            if not timearr.shape == data.shape:
                raise Exception("unexpected error: `time` has shape " + str(timearr.shape) + "while data has"
                                "shape " + str(data.shape) + " (should be equal)")
            # make 2-dimensional array with time and data, and append to output
            arrays.append([timearr, data])

    return arrays


def _timearray_info(dset):
    """
    Extracts time array from h5-dataset.

    Parameters
    ----------
    dset: h5py.Dataset
        Dataset to check/extract time array info on.

    Returns
    -------
    dict
        Dictionary with time array info, keys.
        Returns None if time array info is not found/understood.
    """
    if not isinstance(dset, h5py.Dataset):
        raise TypeError("expected h5py.Dataset, got: %s" % type(dset))

    attrs = dset.attrs
    if "start" in attrs and "delta" in attrs:
        # SIMA-way of defining time array
        timeinfo = {
            "kind": "sima",
            "start": float(attrs["start"]),
            "dt": float(attrs["delta"]),
        }
        return timeinfo
    else:
        return None
