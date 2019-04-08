import os
import h5py
import numpy as np


def read_keys(path, verbose=False):
    """
    Extracts keys (data sets names) for `h5` data sets containing (or; interpreted as) time series from h5-file exported
    from SIMA.

    Parameters
    ----------
    path: str
        File name.
    verbose: bool, optional
        If True, print info.

    Returns
    -------
    list
        List of dset names (keys/addresses)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % path)

    if verbose:
        print('Identifying datasets on %s ...' % path)

    dsetnames = []  # list of dataset names/keys (only for data sets that are time series)
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
                    dsetnames.append(key)

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
        print("   no. of time series : %4d" % len(dsetnames) + "   (datasets interpreted as time series)")

    return dsetnames


def read_data(path, dsetnames=None, verbose=False):
    """
    Extracts time series data from `.h5` (or `hdf5`) file exported from SIMA.


    Parameters
    ----------
    path: str
        File name.
    dsetnames: str or list, optional
        List of dset names (keys/adresses) to desired data sets. If None (default), all (time series) data
        sets are read.
    verbose:  bool, optional
        Write info to screen?

    Returns
    -------
    list
        List of arrays with time and data

    Notes
    -----
    If `dsetnames` is not specified, all datasets that are interpreted as time series will be read. See also
    `read_h5_keys`.

    Note that the hdf5 is an extremely flexible file format and this reader is tailored to the SIMA h5 file format.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % path)

    if verbose:
        print('Reading %s ...' % path)

    if isinstance(dsetnames, str):
        dsetnames = [dsetnames]
    elif type(dsetnames) in (list, tuple):
        pass
    elif dsetnames is None:
        pass
    else:
        raise TypeError("`dsetnames` must be str/list/tuple, got: %s" % type(dsetnames))

    # if dsetnames not specified, get all keys
    if dsetnames is None:
        dsetnames = read_keys(path)

    arrays = []

    with h5py.File(path, "r") as f:
        for key in dsetnames:
            dset = f[key]

            if not isinstance(dset, h5py.Dataset):
                raise TypeError("expected to get h5py.Dataset, got %s (for key '%s')" % (type(dset), key))

            # get values (data array), check size and dimensions
            data = dset[:]  # dset.value
            # todo: consider if check of data type is really neccessary -- if not, dset.ndim, dset.size etc. may be used
            if not isinstance(data, np.ndarray):
                raise NotImplemented("only value of type np.ndarray is implented, got: %s (for key '%s')" %
                                     (type(data), key))
            if data.ndim != 1:
                raise NotImplemented("only 1-dimensional arrays implemented, got ndim=%d (for key '%s')" %
                                     (data.ndim, key))
            data = data.flatten()   # flatten data array
            nt = data.size          # number of time steps

            # attrs = dset.attrs  # dict with dataset attributes

            # establish or extract time array
            timeinfo = _timearray_info(dset)
            if timeinfo is None:
                raise Exception("no time info extracted for dataset '%s'" % key)
            kind = timeinfo["kind"]
            if kind == "sima":
                t_start = timeinfo["start"]
                dt = timeinfo["dt"]
                t_end = t_start + (nt-1) * dt
                timearr, step = np.linspace(t_start, t_end, nt, retstep=True)
                if not dt == step:
                    raise Exception("unexpected error: `dt` should be %s but is %s" % (dt, step))

            # todo: expand when _get_h5_timearray_info() has been expanded

            # verify that time array has correct shape (==> should be same as `data` shape)
            if not timearr.shape == data.shape:
                raise Exception("unexpected error: `time` has shape " + str(timearr.shape) + "while data has"
                                "shape " + str(data.shape) + " (should be equal)")
            # make 2-dimensional array with time and data, and append to output
            arr = np.array([timearr, data])
            arrays.append(arr)

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
            "start": attrs["start"],
            "dt": attrs["delta"],
        }
        return timeinfo
    else:
        return None
