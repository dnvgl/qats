"""
Readers for HDF5 formatted time series files exported from SIMA
"""

import os
from collections import deque

import h5py
import numpy as np

from .base import Reader, SeriesData, SeriesInfo, SourceFile


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
        print("Identifying datasets on %s ..." % path)

    series, n_groups, n_dsets = _scan(path)

    if verbose:
        print("   no. of groups      : %4d" % n_groups + "   (incl. subgroups)")
        print("   no. of datasets    : %4d" % n_dsets)
        print("   no. of time series : %4d" % len(series) + "   (datasets interpreted as time series)")

    # groups and name joined with backslash
    return ["\\".join((*group, name)) for group, name in series]


def _decode(name):
    return name.decode("utf-8", "surrogateescape")


def _scan(path):
    """
    The time series on a file as (groups, name), the groups a tuple from the top of the file, and the number of groups
    and datasets on the file.
    """
    series = []
    n_dsets = 0
    n_groups = 0

    # breadth first through the groups, each in h5py's order: the order of creation if the file keeps it (e.g. node_2
    # before node_10), else name order. Through h5py's low-level interface, which is several times faster than
    # looking up each object through h5py.File for files with many datasets (#171).
    with h5py.File(path, "r") as f:
        groups = deque([((), h5py.h5g.open(f.id, b"/"))])
        while groups:
            parents, gid = groups.popleft()
            for name in gid:
                oid = h5py.h5o.open(gid, name)
                if isinstance(oid, h5py.h5d.DatasetID):
                    n_dsets += 1
                    # a time series has SIMA's time attributes (see _timearray_info)
                    if h5py.h5a.exists(oid, b"start") and h5py.h5a.exists(oid, b"delta"):
                        series.append((parents, _decode(name)))
                elif isinstance(oid, h5py.h5g.GroupID):
                    n_groups += 1
                    groups.append(((*parents, _decode(name)), oid))

    return series, n_groups, n_dsets


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
        print("Reading %s ..." % path)

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
    names = [_.replace("\\", "/") for _ in names]
    return _read_datasets(path, names)


def _read_datasets(path, names):
    """Time and values of the datasets at the given HDF5 paths ('/' between groups), as [t, x] lists."""
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
                raise NotImplementedError(
                    f"only value of type np.ndarray is implemented, got: {type(data)} (for name '{name}')"
                )
            if data.ndim != 1:
                raise NotImplementedError(
                    f"only 1-dimensional arrays implemented, got ndim={data.ndim} (for name '{name}')"
                )
            nt = data.size  # number of time steps

            # attrs = dset.attrs  # dict with dataset attributes

            # establish or extract time array
            timeinfo = _timearray_info(dset)
            if timeinfo is None:
                raise Exception("no time info extracted for dataset '%s'" % name)
            kind = timeinfo["kind"]
            if kind == "sima":
                t_start = timeinfo["start"]
                dt = timeinfo["dt"]
                t_end = t_start + (nt - 1) * dt
                timearr, step = np.linspace(t_start, t_end, nt, retstep=True)
                if not np.isclose(dt, step):  # dt == step:
                    raise Exception("unexpected error: `dt` should be %s but is %s" % (dt, step))

            # todo: expand when _get_h5_timearray_info() has been expanded

            # verify that time array has correct shape (==> should be same as `data` shape)
            if not timearr.shape == data.shape:
                raise Exception(
                    "unexpected error: `time` has shape " + str(timearr.shape) + "while data has"
                    "shape " + str(data.shape) + " (should be equal)"
                )
            # make 2-dimensional array with time and data, and append to output
            arrays.append([timearr, data])

    return arrays


def write_data(path, data: dict):
    """
    Write time series to SIMA HDF5 file.

    Parameters
    ----------
    path : str
        File path
    data : dict
        Time series data name vs. time and values
    """
    # todo: Pass complete time series info (yunit etc.) if this is stored on TimeSeries object
    with h5py.File(path, "w") as f:
        for name, tx in data.items():
            t, x = tx
            start = t[0]
            delta = t[1] - t[0]
            dset = f.create_dataset(name, data=x)
            dset.attrs["name"] = os.path.basename(name)
            dset.attrs["start"] = start
            dset.attrs["delta"] = delta
            dset.attrs["xunit"] = "s"
            dset.attrs["yunit"] = ""


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
        timeinfo = {"kind": "sima"}
        # timeinfo["start"] = float(attrs["start"])
        # timeinfo["dt"} =: float(attrs["delta"]),
        for keyout, keysima in {"start": "start", "dt": "delta"}.items():
            # we presume that attrs["start"] and attrs["delta"] are numpy arrays of size 1
            # (otherwise .item() will fail, but then it should because we expect start and delta
            #  to be single values not arrays)
            timeinfo[keyout] = attrs[keysima].item()
        return timeinfo
    else:
        return None


class _H5Source(SourceFile):
    def __init__(self, path):
        super().__init__(path)
        self._names = None

    def series(self):
        if self._names is None:
            self._names = _scan(self.path)[0]
        return [SeriesInfo(name, group=group) for group, name in self._names]

    def read(self, series):
        paths = ["/".join((*s.group, s.name)) for s in series]
        return [SeriesData(s.name, t, x, group=s.group) for s, (t, x) in zip(series, _read_datasets(self.path, paths))]

    def _legacy_index(self, name):
        """Position of the series on file, as stored in `TsDB.register_indices` before 5.5.0."""
        return None


def _is_sima_series(name, obj):
    """Visitor for `h5py.Group.visititems`: True (which stops the visit) for a dataset with SIMA's time attributes."""
    if isinstance(obj, h5py.Dataset) and "start" in obj.attrs and "delta" in obj.attrs:
        return True
    return None


class H5Reader(Reader):
    """
    Reader for HDF5 files exported from SIMA (``.h5``, ``.hdf5``), where each time series is a dataset with the
    attributes ``start`` and ``delta`` describing its time.

    .. versionadded :: 5.5.0
    """

    name = "sima-h5"
    description = "SIMA HDF5"
    patterns = ("*.h5", "*.hdf5")

    def can_read(self, path):
        """True if the file holds at least one dataset with SIMA's time attributes."""
        with h5py.File(path, "r") as f:
            return bool(f.visititems(_is_sima_series))

    def open(self, path):
        return _H5Source(path)
