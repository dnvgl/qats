#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides :class:`TsDB` class.
"""
import os
import glob
import copy
import fnmatch
from uuid import uuid4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from .ts import TimeSeries
from .fatigue.rainflow import rebin as rebin_cycles
from .io.sima import (
    read_names as read_sima_names,
    read_ascii_data as read_sima_ascii_data,
    read_bin_data as read_sima_bin_data,
    read_sima_wind_names
)
from .io.sima_h5 import (
    read_names as read_sima_h5_names,
    read_data as read_sima_h5_data,
    write_data as write_sima_h5_data
)
from .io.csv import (
    read_names as read_csv_names,
    read_data as read_csv_data
)
from .io.direct_access import (
    read_ts_names,
    read_tda_names,
    read_ts_data,
    read_tda_data,
    write_ts_data
)
from .io.sintef_mat import (
    read_names as read_mat_names,
    read_data as read_mat_data
)
from .io.tdms import (
    read_names as read_tdms_names,
    read_data as read_tdms_data
)
from .io.other import (
    read_dat_names,
    read_dat_data,
    write_dat_data
)

# todo: cross spectrum(scipy.signal.csd)
# todo: coherence (scipy.signal.coherence)


class TsDB(object):
    """
    A class for storage, processing and presentation of time series.

    Parameters
    ----------
    name : str, optional
        Database name.

    Attributes
    ----------
    name : str
        Database name.
    register : OrderedDict
        TimeSeries objects by unique time series id.
    register_parent : OrderedDict
        Parent file path by unique time series id.
    register_indices : OrderedDict
        Index of time series on parent file by unique time series id.
    register_keys : list
        Unique time series id.

    """

    def __init__(self, name=None):
        self.uuid = uuid4()
        self.name = name
        self.register = OrderedDict()           # dictionary of unique id and time series objects
        self.register_parent = OrderedDict()    # dictionary of unique id and parent name (source/file name)
        self.register_indices = OrderedDict()   # dictionary of unique id and the time series index on parent file
        self.register_keys = []     # register keys in the order the associated time series where loaded
        self._timekeys = dict()  # register of time keys (only relevant for .mat files)

    def __contains__(self, item):
        if isinstance(item, str):
            # item interpreted as time series name
            names = item
        elif isinstance(item, TimeSeries):
            names = item.fullname
        else:
            raise TypeError(f"Unable to check containment of type '{type(item)}' items. Item must be string or"
                            f"TimeSeries.")

        match = self.list(names=names, display=False)
        if len(match) > 0:
            return True
        else:
            return False

    def __eq__(self, other):
        return isinstance(other, TsDB) and other.uuid == self.uuid

    def __iter__(self):
        """
        Generator yielding time series in database

        Returns
        -------
        TimeSeries
            time series in database
        """
        for key in self.register_keys:
            item = self.register[key]
            if item is not None:
                yield self.register[key]

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"<TsDB id='{self.uuid}'>"

    def __str__(self):
        _ = f"type: TsDB\n" \
            f"id : {self.uuid}\n" \
            f"name : {self.name}\n" \
            f"number of time series : {self.n}\n"
        return _

    @classmethod
    def fromfile(cls, filenames, read=False, verbose=False):
        """
        Create TsDB instance from one ore more files.

        Parameters
        ----------
        filenames: str, list or tuple
            File names including suffix. Wildcards can also be used.
        read: bool, optional
            If True, all time series are read from file and stored. The default is that they are read from file
            when requested by any of the `get*()` methods.
        verbose : bool, optional
            If True, print information to screen.

        Returns
        -------
        TsDB
            TsDB instance with files loaded.

        Notes
        -----
        The purpose of the classmethod is efficient initiation of a new class instance. For example:

        >>> from qats import TsDB
        >>> tsdb = TsDB.fromfile("mooring.ts")

        ... is equivalent to:

        >>> tsdb = TsDB("")
        >>> tsdb.load("mooring.ts")


        See also notes for :meth:`~TsDB.load`.
        """
        tsdb = cls("")
        tsdb.load(filenames, read=read, verbose=verbose)
        return tsdb

    @property
    def common(self):
        """
        Common part of all keys (paths) in DB.

        Returns
        -------
        str
        """
        if self.n == 0:
            return ""
        elif self.n == 1:
            k = self.register_keys[0]
            return self._path_dirname(k)
        else:
            try:
                return os.path.commonpath(self.register_keys)
            except ValueError:
                # if paths contain both absolute and relative paths, the paths are on the different
                # drives or if paths is empty.
                return ""

    @property
    def n(self):
        """
        Number of time series in database

        Returns
        -------
        int : number of loaded time series

        """
        return len(self.register)

    @staticmethod
    def _check_time_arrays(container, **kwargs):
        """

        Parameters
        ----------
        container: dict
            Time series container obtained from `TsDB.getm()`.
        kwargs:
            Optional arguments passed to `TimeSeries.get()`

        Returns
        -------
        dict
            Attributes: 'is_common' (bool), 'dtg_defined' (bool), 'dtg_ref' (datetime or None), 'common' (tuple),
            'devations' (dict)', 'actions' (list), 'msg' (str).
        """
        # evaluate whether the time series have common time array (time steps and time windows are equal)
        _dtg_ref = []
        _dt = []
        _start_time = []
        _end_time = []

        for _, ts in container.items():
            _dtg_ref.append(ts.dtg_ref)
            _dt.append(ts.dt)
            _start_time.append(ts.start)
            _end_time.append(ts.end)

        # evaluate dtg_ref, etc.
        _same_dtg_ref = all(r == _dtg_ref[0] for r in _dtg_ref)
        dtg_defined = any(r is not None for r in _dtg_ref)
        dtg_ref = _dtg_ref[0] if (dtg_defined is True and _same_dtg_ref is True) else None

        # evaluate time step, start and end time
        _same_dt = ((max(_dt) - min(_dt)) == 0.)
        _same_start_time = ((max(_start_time) - min(_start_time)) == 0.)
        _same_end_time = ((max(_end_time) - min(_end_time)) == 0.)

        # recommended parameters for common time: latest start, earliest end, smallest avg. time step
        common_start = max(_start_time)
        common_end = min(_end_time)
        if common_start < common_end:
            common = common_start, common_end, min(_dt)
        else:
            # not possible to create common time array (earliest end time is before latest start time)
            common = None

        # evaluate and define deviations and recommended actions
        deviations = OrderedDict()

        if not _same_dt:
            deviations["dt"] = "mean time step varies from %.10g to %.10g" % (min(_dt), max(_dt))

        if not _same_start_time:
            deviations["start"] = "start time varies from %.7g to %.7g" % (min(_start_time), max(_start_time))

        if not _same_end_time:
            deviations["end"] = "end time varies from %.7g to %.7g" % (min(_end_time), max(_end_time))

        if dtg_defined and not _same_dtg_ref:
            '''
            If dtg_ref test criteria fails, none of the previous tests are relevant anymore. Therefore, the devations
            dict is reset here.
            '''
            deviations = OrderedDict()
            deviations["dtg_ref"] = "'dtg_ref' is defined for one or more of the time series, " \
                                    "but is not equal for all of them"

        # identify any deviations that are handled by kwargs
        handled = set()
        if "resample" in kwargs:
            if isinstance(kwargs.get("resample"), float):
                handled.add("dt")
            elif isinstance(kwargs.get("resample"), np.ndarray) or isinstance(kwargs.get("resample"), list):
                t_new = kwargs.get("resample")
                if np.allclose(np.min(np.diff(t_new)), np.max(np.diff(t_new))):
                    handled.add("dt")
                if t_new[0] >= common_start:
                    handled.add("start")
                if t_new[-1] >= common_end:
                    handled.add("end")
        if kwargs.get("twin") is not None:
            t_start, t_end = kwargs["twin"]
            if t_start >= common_start:
                handled.add("start")
            if t_end >= common_end:
                handled.add("end")

        # remove deviations handled by kwargs
        for h in handled:
            _ = deviations.pop(h, None)

        # evaluate if time array is common
        is_common = (len(deviations) == 0)

        # define actions
        actions = []
        msg = ""
        if not is_common:
            if "dt" in deviations:
                actions.append("Resample to constant time step by specifying `resample=%.10g` (or smaller)" % common[2])
            if "start" in deviations or "end" in deviations:
                actions.append("Crop time series to the common time window by specifying "
                               "`twin=(%.7g, %.7g)`" % (common[0], common[1]))
                actions.append("Resample to a common time array by specifying "
                               "`resample=np.arange(%.7g, %.7g, %.7g)`" % (common[0], common[1]+common[2], common[2]))
            if "dtg_ref" in deviations:
                actions.append("Ensure 'dtg_ref' is the same for all specified time series "
                               "-- see `TimeSeries.set_dtg_ref()`")

            # compile deviations and recommended actions to message string
            msg = "Time array is not common within specified parameters for specified time series.\n" \
                  "Deviations:\n"
            for dev, s in deviations.items():
                msg += "  * %s : %s\n" % (dev, s)
            if len(actions) > 0:
                msg += "Recommended actions (one or more may be necessary):\n"
                if common is not None:
                    for a in actions:
                        msg += "  * %s\n" % a
                else:
                    msg += "   (none - not possible to create common time array)\n"

        timecheck = dict(
            is_common=is_common,        # True or False
            # dtg info
            dtg_defined=dtg_defined,    # True or False
            dtg_ref=dtg_ref,            # datetime or None
            # recommended parameters for common time array
            common=common,              # tuple or None
            # descriptions of deviations and proposed actions
            deviations=deviations,      # dict
            actions=actions,            # list
            msg=msg,                    # str
        )

        return timecheck

    @staticmethod
    def _path_basename(key):
        """
        As os.path.basename, but does not split on '/' or '\\' if they are within square brackets.
        """
        if "[" in key:
            i = key.index("[")
            return os.path.basename(key[:i]) + key[i:]
        else:
            return os.path.basename(key)

    @staticmethod
    def _path_dirname(key):
        """
        As os.path.dirname, but does not split on '/' or '\\' if they are within square brackets.
        """
        if "[" in key:
            i = key.index("[")
            return os.path.dirname(key[:i]) + key[i:]
        else:
            return os.path.dirname(key)

    @staticmethod
    def _path_relpath(key, start=os.curdir):
        """
        As os.path.relpath, but does not split on '/' or '\\' if they are within square brackets.
        """
        if "[" in key:
            i = key.index("[")
            return os.path.relpath(key[:i], start) + key[i:]
        else:
            return os.path.relpath(key, start)

    @staticmethod
    def _reorder_namelist(namelist, names=None):
        """
        Re-order time series names to match order in which they were specified.

        Parameters
        ----------
        namelist: list
            List of time series names to re-order.
        names: list, optional
            Time series name (patterns)

        Returns
        -------
        list
            Re-ordered time series names

        Raises
        ------
        ValueError
            If `keep_order` is True, in combination with more than one key and more than one name.
        """

        # define function need for sorting
        def get_index(m, patterns):
            """Return index of matched time series names."""
            try:
                _ind = [fnmatch.fnmatch(m, pat) for pat in patterns].index(True)
            except ValueError:
                raise Exception("Unexpected error: could not find sorting index for key '%s'" % m)
            return _ind

        # re-order names to match specified order
        if names is not None and len(names) > 1:
            return sorted(namelist, key=lambda m: get_index(m, names))
        else:
            # names or names not sufficiently specified, there is nothing to do
            return namelist

    def _make_export_friendly_names(self, container, keep_basename=False):
        """
        Shorten keys by removing common part of it and replace path separators with underscore. Use only basename
        if specified.

        Parameters
        ----------
        container : dict
            Container with time series
        keep_basename : bool, optional
            Keep only time series names e.g. 'tension' in key 'C:\data\results.ts\tension'. Default False.

        Returns
        -------
        dict
            Time series container with modified keys

        Raises
        ------
        KeyError
            If `keep_basename` = True and several timeseries have the same name

        Notes
        -----
        To keep info but avoid path confusion when loading the exported file later
            - keys shortened by only removing non-unique part
            - remove parent file extension from key
            - replace path separators in key with underscore
            - do not replace '.' elsewhere in the key e.g. directory name or filename containing '.'

        """
        new_container = OrderedDict()

        # force keeping basename if there is only 1 ts in the container. This to avoid totally removing the name
        if len(container) == 1:
            keep_basename = True

        if keep_basename:
            key_count = defaultdict(int)
        else:
            # common part of all selected keys
            common_key = os.path.commonpath([str(k) for k in container.keys()])

        for key, ts in container.items():
            if keep_basename:
                # shorten key by using only basename
                new_k = self._path_basename(key)
                key_count[new_k] += 1
                if key_count[new_k] > 1:
                    raise KeyError("Multiple keys with basename '%s' -- "
                                   "consider using ``basename=False``" % new_k)
            else:
                # make
                relkey = self._path_relpath(key, common_key)
                name = self._path_basename(relkey)
                new_k = "_".join([os.path.splitext(self._path_dirname(relkey))[0].replace(os.path.sep, "_"), name])

                # Remove leading underscores in the name
                if new_k.startswith("_"):
                    new_k = new_k[1:]

            # put in new container
            new_container[new_k] = ts

        return new_container

    def _read(self, keys, retkeys=None, store=True):
        """
        Read time series specified by absolute keys from file

        Parameters
        ----------
        keys : list
            Unique time series identifiers/keys/path
        retkeys : list, optional
            Unique identifiers/keys used in returned container. Useful if you want to have shorter keys
            (less the common db path) in the container returned. Note that full key is used in the db register.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.

        Returns
        -------
        dict
            TimeSeries objects

        Notes
        -----
        Absolute keys are obtained using the list() method.

        """
        # handle retkeys
        if retkeys is not None:
            assert len(retkeys) == len(keys), "The number of 'retkeys' is different from the number of 'keys'."
        else:
            retkeys = copy.copy(keys)

        # correlation between full keys and specified keys for returned container
        keypairs = dict(zip(keys, retkeys))

        # initiate and pre-populate ordered dictionary (to keep order of keys)
        container = OrderedDict()
        for key in keys:
            reg = self.register[key]   # gives None if not previously read/stored
            if isinstance(reg, TimeSeries) or reg is None:
                container[keypairs[key]] = reg
            else:
                raise LookupError("Unexpected look-up error, key: %s" % key)

        # group keys by parent file
        keys_by_parent = defaultdict(list)
        for key in keys:
            # only interested in keys that are not already stored
            if container[keypairs[key]] is None:
                keys_by_parent[self.register_parent[key]].append(key)

        # read requested keys, file by file
        for parent, keys in keys_by_parent.items():
            names = [key.replace(parent, "").lstrip(os.path.sep) for key in keys]

            # extract parent file extension
            fext = os.path.splitext(parent)[-1]

            # indices of time series to be read
            indices = [0] + [self.register_indices[key] for key in keys]

            tslist = [None] * len(keys)

            if fext == '.ts':
                data = read_ts_data(parent, ind=indices)
                for i, name in enumerate(names):
                    tslist[i] = TimeSeries(name, data[0, :], data[i + 1, :], parent=parent)

            elif fext == '.tda':
                data = read_tda_data(parent, ind=indices)
                for i, name in enumerate(names):
                    tslist[i] = TimeSeries(name, data[0, :], data[i+1, :], parent=parent)

            elif fext == '.asc':
                data = read_sima_ascii_data(parent, ind=indices)
                for i, name in enumerate(names):
                    tslist[i] = TimeSeries(name, data[0, :], data[i+1, :], parent=parent)

            elif fext == '.bin':
                data = read_sima_bin_data(parent, ind=indices)
                for i, name in enumerate(names):
                    tslist[i] = TimeSeries(name, data[0, :], data[i+1, :], parent=parent)

            elif fext == '.dat':
                data = read_dat_data(parent, ind=indices)
                for i, name in enumerate(names):
                    tslist[i] = TimeSeries(name, data[0, :], data[i+1, :], parent=parent)

            elif fext == '.mat':
                _tk = self._timekeys[parent]
                data = read_mat_data(parent, [_tk, *names])
                for i, name in enumerate(names):
                    tslist[i] = TimeSeries(name, data[_tk], data[name], parent=parent)

            elif fext in ('.h5', '.hdf5'):
                data = read_sima_h5_data(parent, names=names)
                for i, name in enumerate(names):
                    timearr, arr = data[i]
                    tslist[i] = TimeSeries(name, timearr, arr, parent=parent)

            elif fext == '.csv':
                data = read_csv_data(parent, ind=indices)
                for i, name in enumerate(names):
                    tslist[i] = TimeSeries(name, data[0, :], data[i + 1, :], parent=parent)

            elif fext == '.tdms':
                data = read_tdms_data(parent, names=names)
                for i, name in enumerate(names):
                    timearr, arr = data[i]
                    tslist[i] = TimeSeries(name, timearr, arr, parent=parent)
            else:
                raise NotImplementedError("Invalid file type: %s (ext = %s)" % (parent, fext))

            # add to ordered dictionary (and store in db if specified)
            for key, ts in zip(keys, tslist):
                container[keypairs[key]] = ts

                if store:
                    self.register[key] = ts

        return container

    def add(self, ts):
        """
        Add new TimeSeries object to db

        Parameters
        ----------
        ts : TimeSeries
            added TimeSeries object

        Notes
        -----
        Key/identifier will be the name of the time series. If you want to change the key, just change the name before
        adding the TimeSeries to the db.

        """
        '''
        Note: 
        If only 'name' is used as the register key, one may encounter issues later (e.g. at export), since 
        os.path.commonpath will not accept a mix of absolute and relative paths. Therefore, the added timeseries is
        registered with a fictitious key constructed as follows: commonpath + ts.name
        '''
        if not isinstance(ts, TimeSeries):
            raise TypeError("expected TimeSeries instance, got: %s" % type(ts))

        key = os.path.join(self.common, ts.name)

        if key in self.register.keys():
            raise KeyError("The specified key is not unique: %s" % key)

        self.register[key] = ts
        self.register_parent[key] = None    # does not have a parent (file)
        self.register_indices[key] = None   # ... and therefore has no index (yet)
        self.register_keys.append(key)

    def clear(self, names=None, display=True):
        """
        Clear/remove time series from register

        Parameters
        ----------
        names : str or list or tuple, optional
            Name of time series to remove from database register, supports wildcard.
        display : bool, optional
            disable print to screen. Default True
        """
        if display:
            print("Removing:`\n")
        match = self.list(names=names, display=display, relative=False)
        for k in match:
            _ = self.register.pop(k, None)
            _ = self.register_parent.pop(k, None)
            _ = self.register_indices.pop(k, None)
            _ = self.register_keys.pop(self.register_keys.index(k))

    def copy(self, names=None, shallow=False):
        """
        Make a copy (new TsDB instance) with the specified keys/names included.

        Parameters
        ----------
        names : str or list or tuple, optional
            Time series names filter that supports regular expressions.
        shallow: bool, optional
            If False (default), the copied TimeSeries objects do not point back to the TimeSeries instances in the
            source TsDB.

        Returns
        -------
        tsdb
            New TsDB instance.

        Notes
        -----
        If shallow is True, the TimeSeries objects in the new TsDB instance point back to the original objects. This
        implies that modifications to the copied TimeSeries are routed back to the source objects.
        The advantage of using ``shallow=True`` is that memory usage is limited. However, if in doubt use
        ``shallow=False`` (the default).

        Specified timeseries that are not preloaded (stored), will be loaded during this procedure.
        """
        new = TsDB(name=self.name)
        container = self.getm(names=names, store=True, fullkey=True)
        for key, ts in container.items():
            if shallow is False:
                ts = ts.copy()
            new.add(ts)
            new.register_parent[key] = self.register_parent[key]
            new.register_indices[key] = self.register_indices[key]
        return new

    def create_common_time(self, names=None, twin=None, maxdt=None, strict=False):
        """
        Creates common time array from: latest start time, earliest end time and minimum mean time step.

        Parameters
        ----------
        names : str or list or tuple, optional
            Time series names
        twin: tuple, optional
            If specified, the common time array within specified window (start, end) is created.
        maxdt: float, optional
            Max. time step desired. If minimum mean time step is larger than 'maxdt', the latter is used.
        strict: bool, optional
            If True, ValueError is raised if obtained time step deviates from specified/recommended time step.

        Returns
        -------
        array
            New time array.

        Notes
        -----
        Be careful when specifying the maximum time step.
        """
        container = self.getm(names=names, fullkey=True, store=False)
        timecheck = self._check_time_arrays(container)
        try:
            start, end, dt = timecheck["common"]
        except TypeError:
            raise ValueError("Could not create common time array "
                             "- check if earliest end time is before latest start time")
        if maxdt is not None:
            if dt > maxdt:
                dt = maxdt
        nt = int(round((end-start)/dt)) + 1
        common_time, _dt = np.linspace(start, end, nt, retstep=True)
        if strict is True and not round(dt - _dt, 4) == 0:
            raise ValueError("obtained 'dt' (%f) deviates from specified 'dt' (%s)" % (_dt, dt))
        if twin is not None:
            t_start, t_end = twin
            ind = (common_time >= t_start) & (common_time <= t_end)
            common_time = common_time[ind]
        return common_time

    def export(self, filename, names=None, delim="\t", skip_header=False, exist_ok=True, basename=True,
               verbose=False, **kwargs):
        """
        Export time series to file

        Parameters
        ----------
        filename : str
            File name including suffix.
        names : str or list or tuple, optional
            Time series names
        delim : str, optional
            column delimiter for column wise ascii file, default "\t"
        skip_header: bool, optional
            For ascii files, skip header with keys? Default is to include them on first line. This parameter is ignored
            for other file formats.
        exist_ok : bool, optional
            if false (the default), a FileExistsError is raised if target file already exists
        basename : bool, optional
            If true (the default), basename (no path/file info) will be exported to key file. If false, the name/path
            relative to common path is used. Also see notes below.
        verbose : bool, optional
            Print information
        kwargs : optional
            see documentation of :meth:`~qats.TimeSeries.get` method for available options

        Notes
        -----
        Currently implemented file formats
         - direct access format (.ts)
         - column-wise ascii file with single header line defining the keys(.dat) (comment character is #)

        The time series are resampled to a common time vector with a constant time step (sample rate). The minimum
        average time step of all the selected time series is applied. This is done before enforcing the specified
        time window.

        If `basename` is true, an exception is raised if two or more time series share the same basename. The solution
        is then to specify ``basename=False``, so that only the common part of the paths/identifiers of the specified
        time series is removed before writing the identifiers to key file (applies to .ts format) or ascii file header.
        Note that in this case the keys are modified so that path separators ('/' and '\\') are replaced by
        underscore ('_').
        """
        # todo: export(): include `header` parameter; text string that is included on top of key file

        # assert that filename is specified as string
        if not isinstance(filename, str):
            raise TypeError("Filename should be a str, not: %s" % type(filename))

        # assert that if the file exists the user has specifically allowed overwriting it
        if os.path.isfile(filename) and exist_ok is False:
            raise FileExistsError("The file '%s' already exists." % filename)

        # create non-existing directories
        dirname = self._path_dirname(filename)
        if dirname != "" and not os.path.exists(dirname):
            os.makedirs(dirname)

        # generate time series container
        '''
        Note:
        For most of the code below, it is convenient to use container generated by `getm()`. However; to evaluate
        whether time array is common (using `_check_time_arrays()`), we need TimeSeries objects in case `dtg_ref` 
        defined. The container is therefore generated as follows:
        1) Generate container of TimeSeries objects
        2) Modify container keys to export friendly keys
        3) Perform time array check (taking `dtg_ref` and  `**kwargs` into account)
        4) Convert container to container of arrays (same as output from `getm()`, taking **kwargs into account)
        '''
        # generate container, step 1 (generate container of TimeSeries objects)
        container = self.getm(names=names, fullkey=True, store=False)
        # generate container, step 2 (create export friendly keys)
        container = self._make_export_friendly_names(container, keep_basename=basename)
        # generate container, step 3 (perform time array check
        timecheck = self._check_time_arrays(container, **kwargs)
        # generate container, step 4 (convert to container of arrays)
        container = OrderedDict((k, v.get(**kwargs)) for k, v in container.items())

        # evaluate outcome of time array check, raise error if not common (time steps and time windows are not equal)
        if not timecheck["is_common"]:
            # create error message (principle: error+advice rather than automagic solution)
            # in other words; inform the user that the time arrays are not equal and suggest how to mitigate
            raise ValueError(timecheck["msg"])
        else:
            # all time arrays are equal, just use the time array from one of the TimeSeries objects
            common_time_array = container[list(container)[0]][0]

        # get file extension
        _, ext = os.path.splitext(filename)

        if ext == ".ts":    # write direct access file
            write_ts_data(filename, common_time_array, container)

        elif ext == ".dat":     # write ascii file
            write_dat_data(filename, common_time_array, container, delim=delim, skip_header=skip_header)

        elif ext == ".h5":
            write_sima_h5_data(filename, container)

        else:
            raise NotImplementedError("File format/type '%s' is not yet implemented." % ext)

        # print information
        if verbose:
            print(50 * "=")
            print("Exported %d records to file '%s'" % (len(container), filename))
            print(50 * "-")
            for key in container.keys():
                print(key)
            print(50 * "=")
        else:
            print("Exported %d records to file '%s'." % (len(container), filename))

    def get(self, name=None, ind=None, store=True):
        """
        Get (single) TimeSeries object.

        Parameters
        ----------
        name : str, optional
            Time series name
        ind : int, optional
            Time series index in database register.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.

        Returns
        -------
        TimeSeries
            Time series

        Notes
        -----
        Either name or ind must be specified.

        Error is raised if zero or more than one match is obtained.

        Note that this method is somewhat similar to :meth:`~geta` but returns a TimeSeries object instead of arrays.
        Therefore this method does not support keyword arguments to be passed further to `TimeSeries.get()`.

        See also
        --------
        qats.TimeSeries
        geta, getda, getl, getm
        """
        # check that at least one of the required parameters is given,
        # and that non-compatible parameters are not combined
        if name is None and ind is None:
            raise TypeError("Either `name` or `ind` must be given")

        if name is not None and ind is not None:
            raise TypeError("Cannot combine parameters `ind` and `name`")

        # check type of specified parameters
        if name is not None and not isinstance(name, str):
            raise TypeError("Parameter `name` must be string if specified")

        if ind is not None and not isinstance(ind, int):
            raise TypeError("Parameter `ind` must be integer if specified")

        # return quickly if specified key (or index) matches a register entry and data is already loaded
        if name is not None:
            _ = self.list(names=name, display=False)
            if len(_) == 0:
                raise LookupError(f"No match found for '{name}'")

            elif len(_) > 1:
                raise ValueError(f"Name '{name}' is not unique. Got {len(_)} matches.")
            else:
                key = _[0]
        else:
            # ind is not None
            try:
                _ = self.register_keys[ind]
            except IndexError as err:
                raise IndexError(f"Index {ind} is not available (number of entries = {self.n})") from err
            else:
                key = _

        # check if data is read
        ts = self.register.get(key, None)
        if isinstance(ts, TimeSeries):
            return ts
        else:
            container = self.getm(names=name, ind=ind, fullkey=True, store=store)
            n = len(container)
            if n == 0:
                raise LookupError("No match found for specified name")
            elif n > 1:
                raise ValueError("More than one match found for specified name:"
                                 "\n    %s" % "\n    ".join(container.keys()))
            else:
                return container.popitem()[1]

    def geta(self, name=None, ind=None, store=True, **kwargs):
        """
        Get (single) time series as numpy arrays (tuple of time and data arrays).
        Optionally, the returned data array may be processed according to specified optional arguments (see `kwargs`).

        Parameters
        ----------
        name : str, optional
            Time series name
        ind : int, optional
            Time series index in database register.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        kwargs : optional
            See documentation of :meth:`~qats.TimeSeries.get` method for available options

        Returns
        -------
        tuple
            Time and data arrays

        Notes
        -----
        Either name or ind must be specified.

        Error is raised if zero or more than one match is obtained.

        See also
        --------
        qats.TimeSeries, qats.TimeSeries.get
        get, getda, getl, getm
        """
        # use self.get() to avoid duplicate code
        try:
            ts = self.get(name=name, ind=ind, store=store)
        except TypeError:
            raise
        except LookupError:
            raise
        except ValueError:
            raise
        return ts.get(**kwargs)

    def getd(self, names=None, ind=None, store=True, fullkey=False):
        """
        Get dict of TimeSeries objects.

        Note that this method is identical to :meth:`~qats.TimeSeries.getm`.

        Parameters
        ----------
        names : str or list or tuple, optional
            Time series name(s), supports wildcard.
        ind : int or list, optional
            Index (or indices) of desired time series (index refers to index of key in list attribute `register_keys`).
            This parameter may not be combined with `keys` or `names`.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        fullkey : bool, optional
            Use full key in returned container

        Returns
        -------
        dict
            TimeSeries objects

        See also
        --------
        qats.TimeSeries
        get, geta, getda, getl, getm
        """
        return self.getm(names=names, ind=ind, store=store, fullkey=fullkey)

    def getda(self, names=None, ind=None, store=True, fullkey=False, **kwargs):
        """
        Get dictionary of (numpy) arrays.
        Optionally, the returned data arrays may be processed according to specified optional arguments (see `kwargs`).

        Parameters
        ----------
        names : str or list or tuple, optional
            Time series names, supports wildcard.
        ind : int or list, optional
            Index (or indices) of desired time series (index refers to index of key in list attribute `register_keys`).
            This parameter may not be combined with `keys` or `names`.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        fullkey : bool, optional
            Use full key in returned container
        kwargs : optional
            see documentation of :meth:`~qats.TimeSeries.get` method for available options

        Returns
        -------
        dict
            Each entry is a tuple with 2 arrays: time and data for each time series

        Notes
        -----
        When working on a large time series database it is recommended to set ``store=False`` to avoid too high memory
        usage. Then the TimeSeries objects will not be stored in the database, only their addresses.

        See also
        --------
        qats.TimeSeries, qats.TimeSeries.get
        get, geta, getl, getm
        """
        # read time series and put in ordered dictionary (reuse getm() to avoid duplicating code)
        container = OrderedDict((k, v.get(**kwargs)) for k, v in
                                self.getm(names=names, ind=ind, store=store, fullkey=fullkey).items())

        return container

    def getl(self, names=None, ind=None, store=True):
        """
        Get list of TimeSeries objects.

        Parameters
        ----------
        names : str or list or tuple, optional
            Time series names, supports wildcard.
        ind : int or list, optional
            Index (or indices) of desired time series (index refers to index of key in list attribute `register_keys`).
            This parameter may not be combined with `keys` or `names`.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        Returns
        -------
        list
            TimeSeries objects

        Notes
        -----
        When working on a large time series database it is recommended to set ``store=False`` to avoid too high memory
        usage. Then the TimeSeries objects will not be stored in the database, only their addresses.

        See also
        --------
        qats.TimeSeries
        get, geta, getda, getm
        """
        # read time series and return in list (reuse getm() to avoid duplicating code)
        return list(self.getm(names=names, ind=ind, store=store).values())

    def getm(self, names=None, ind=None, store=True, fullkey=False):
        """
        Get (dictionary of) multiple TimeSeries objects.

        Parameters
        ----------
        names : str or list or tuple, optional
            Time series name(s), supports wildcard.
        ind : int or list, optional
            Index (or indices) of desired time series (index refers to index of key in list attribute `register_keys`).
            This parameter may not be combined with `keys` or `names`.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        fullkey : bool, optional
            Use full key in returned container

        Returns
        -------
        dict
            TimeSeries objects

        Notes
        -----
        When working on a large time series database it is recommended to set ``store=False`` to avoid too high memory
        usage. Then the TimeSeries objects will not be stored in the database, only their addresses.

        Note that this method is somewhat similar to :meth:`~getda` but returns TimeSeries objects instead of numpy
        arrays. Therefore this method does not support keyword arguments to be passed further to TimeSeries.get().

        See also
        --------
        qats.TimeSeries
        get, geta, getda, getl
        """
        # check that non-compatible parameters are not combined
        if ind is not None and names is not None:
            raise TypeError("Cannot combine parameters `ind` and `names`")

        # get absolute keys
        if ind is None:
            keys = self.list(names=names, display=False, relative=False)
        else:
            # generate keys directly from indices
            if isinstance(ind, int):
                ind = [ind]
            try:
                keys = [self.register_keys[i] for i in ind]
            except IndexError as err:
                raise IndexError("one or more index is out of range (number of entries = %d)" % self.n) from err
            except TypeError as err:
                raise TypeError("Parameter `ind` must be integer or list of integers") from err

        if fullkey:
            # use full key in returned container
            retkeys = keys
        else:
            # todo: consider including the '\\' in self.common (currently I am not sure what is the best)
            # create shorter keys (relative to db common path)
            _common = self.common + os.path.sep
            retkeys = [k.replace(_common, "") for k in keys]

        # read time series and put in ordered dictionary
        container = self._read(keys, retkeys=retkeys, store=store)

        return container

    def is_common_time(self, names=None, twin=None):
        """
        Check if time array is common.

        Parameters
        ----------
        names : str or list or tuple, optional
            Time series names
        twin: tuple, optional
            Time window (start, end) to consider.

        Returns
        -------
        bool
            True if common time array (within specified time window), otherwise False.
        """
        container = self.getm(names=names, fullkey=True, store=False)
        timecheck = self._check_time_arrays(container, twin=twin)
        return timecheck["is_common"]

    def list(self, names=None, display=False, relative=False):
        """
        List time series in database by id/key

        Parameters
        ----------
        names : str or list or tuple, optional
            Time series names filter that supports regular expressions, default all time series will be listed
        display : bool, optional
            Disable print to screen, default False
        relative : bool, optional
            Truncate time series names to unique part i.e. path relative to common path of all time series.

        Returns
        -------
        list
            Time series names

        Notes
        -----
        Full identifier/key is obtained by joining the common path of all time series in db and the unique part of the
        identifiers.
        """
        def _remove_special_characters(strings):
            """
            Remove characters with special meaning in regular expressions

            Parameters
            ----------
            strings : tuple or list
                Strings from which special characters are to be removed

            Returns
            -------
            list :
                Strings with special characters removed

            """
            out = []
            for s in copy.copy(strings):
                # handle regex special characters
                s = s.replace("[", ":[:")  # to avoid troubles when "[" is inserted to handle "]"
                s = s.replace("]", "[]]")
                s = s.replace(":[:", "[[]")
                s = s.replace("^", "[^]")
                s = s.replace("(", "[(]")
                s = s.replace(")", "[)]")
                out.append(s)
            return out

        # full names in db register
        keys_in_register = copy.copy(self.register_keys)

        # get common path
        common = self.common

        # handle types and add leading wildcard to enable pattern matching across paths
        # Note: wildcard should not be prepended:
        #           - if there is no common path
        #           - for names starting with the common path (name is then a full path)
        if names is not None:
            if common == "":
                # no common path -> prefix should not be included (and is not needed)
                _prefix = ""
            else:
                # add prefix *\\ (or */ in unix)
                _prefix = '*' + os.path.sep
            if isinstance(names, str):
                if not names.startswith(common):
                    names = _prefix + names
                names = [names]
            elif type(names) in (list, tuple):
                names = [_prefix + n if not n.startswith(common) else n for n in names]
            else:
                raise TypeError(f"Parameter `names` should be of type str/list/tuple, not {type(names)}")

        # generate list of names/keys
        if names is None:
            # all time series names if no patterns are specified
            match = keys_in_register
        else:
            # match time series keys with specified time series name patterns
            match = []
            for name in _remove_special_characters(names):
                match.extend(fnmatch.filter(keys_in_register, name))

        if relative:
            match = [self._path_relpath(_, common) for _ in match]

        if display:
            print()
            print("=============================================================================================")
            if relative:
                print(f"Common path : {common}")
                print("---------------------------------------------------------------------------------------------")
            if len(match) > 0:
                print("\n".join(match))
            else:
                print("(no match found for specified names/names)")
            print("=============================================================================================")

        return match

    def load(self, filenames, read=False, verbose=False):
        """
        Load time series from files

        Parameters
        ----------
        filenames : str or list or tuple
            File names including suffix. Wildcards can also be used.
        read: bool, optional
            If True, all time series are read from file and stored. The default is that they are read from file
            when requested by any of the `get` methods.
        verbose : bool, optional
            If True, print information to screen.

        Notes
        -----
        `read=True` may be time consuming and require high memory usage if applied for large files with many
        time series. However, if you will work with all the time series on files of moderate size, `read=True` can
        provide efficiency as you only access the file(s) once.

        If time series names contain path separators ('/' or '\\'), these must be enclosed within square brackets (e.g.
        as the slash in "response[m/s]").
        """
        if isinstance(filenames, list) or isinstance(filenames, tuple):
            # expect that iterable contains set of file names, stored with absolute path
            files = [os.path.abspath(f) for f in filenames]
        elif isinstance(filenames, str):
            # string is interpreted as a filename, possibly with wildcards, stored with absolute path
            files = [os.path.abspath(f) for f in glob.glob(filenames)]
        else:
            raise TypeError("files should be either str/tuple/list, not: %s" % type(filenames))

        # If the specified filenames does not exist glob.glob returns an empty list
        if len(files) < 1:
            raise FileExistsError("Path does not exist: %s" % filenames)

        # read time series names and possibly also the data
        for thefile in files:
            fext = os.path.splitext(thefile)[-1]
            basename = os.path.basename(thefile)
            dirname = os.path.dirname(thefile)

            if not os.path.isfile(thefile):
                raise FileExistsError("Object is not a file: %s" % thefile)

            if fext == '.ts':
                # direct access format without info array
                names = read_ts_names(thefile.replace(fext, '.key'))

            elif fext == '.tda':
                # simo s2x direct access format (with info array)
                names = read_tda_names(thefile.replace(fext, '.txt'))

            elif fext == '.asc':
                # simo-riflex, sima ascii
                names = read_sima_names(os.path.join(dirname, 'key_' + basename.replace(fext, '.txt')))

            elif fext == '.bin':
                _ = os.path.join(dirname, 'key_' + basename.replace(fext, '.txt'))
                if thefile.endswith('witurb.bin') or thefile.endswith('blresp.bin'):
                    # wind turbine data and blade response is stored on .bin files but the corresponding
                    # key file has a different structure than the ones associated with 'elmfor' and 'noddis'
                    # .bin files
                    names = read_sima_wind_names(_)
                else:
                    # riflex/simo, sima direct access format
                    names = read_sima_names(_)

            elif fext == '.dat':
                # plain column wise ascii format
                names = read_dat_names(thefile)

            elif fext == '.mat':
                # SINTEF Ocean test data export format based on Matlab .mat files.
                _tk, names = read_mat_names(thefile)
                self._timekeys[thefile] = _tk   # remember the name of the time array

            elif fext in ('.h5', '.hdf5'):
                # sima h5
                names = read_sima_h5_names(thefile)

            elif fext == '.csv':
                # column wise csv
                names = read_csv_names(thefile)

            elif fext == '.tdms':
                # National Instrument structured binary file format
                names = read_tdms_names(thefile)

            else:
                raise NotImplementedError("Invalid file type: %s" % thefile)

            # update database register: unique id (key) consists of filename and time series name
            # before the time series is loaded the register store the time series index on the file
            # j +1 since record 0 is the time vector
            for j, name in enumerate(names):
                key = os.path.join(thefile, name)
                # None until the time series is read and stored
                self.register[key] = None
                # Parent, i.e. source file
                self.register_parent[key] = thefile
                # Time series index on file, to speed up reading the time series, dummy for .mat files, +1 to skip time
                ind = j + 1 if fext not in ('.h5', '.hdf5', '.mat') else None
                self.register_indices[key] = ind
                # time series names in the order the associated time series where loaded
                self.register_keys.append(key)

            if read is True:
                keys = [os.path.join(thefile, name) for name in names]
                _ = self._read(keys, store=True)

            if verbose:
                print("Loaded %d records from file '%s'." % (len(names), thefile))
                print('\n'.join(names))

    def plot(self, names=None, figurename=None, show=None, num=1, store=True, **kwargs):
        """
        Plot time series traces.

        Parameters
        ----------
        names : str/list/tuple, optional
            Time series names
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        kwargs : optional
            See documentation of TimeSeries.get() method for available options

        Notes
        -----
        When working on a large time series database it is recommended to set ``store=False`` to avoid too high memory
        usage. Then the TimeSeries objects will not be stored in the database, only their addresses.

        See also
        --------
        qats.TimeSeries

        """
        # todo: add possibility for subplots in plot method. nsub=None (int), sharex=True.
        # todo: consider need for `keep_order` parameter when plotting
        # dict with numpy arrays: time and data
        container = self.getda(names=names, store=store, **kwargs)

        plt.figure(num=num)
        for k, v in container.items():
            label = k  # todo: more readable label, e.g. remove commonpath
            plt.plot(v[0], v[1], label=label)

        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def plot_psd(self, names=None, figurename=None, show=None, num=1, store=True, **kwargs):
        """
        Plot time series power spectral density.

        Parameters
        ----------
        names : str/list/tuple, optional
            Time series names
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        kwargs : optional
            see documentation of TimeSeries.get() method for available options

        Notes
        -----
        When working on a large time series database it is recommended to set ``store=False`` to avoid too high memory
        usage. Then the TimeSeries objects will not be stored in the database, only their addresses.

        See also
        --------
        qats.TimeSeries

        """
        # dict with TimeSeries objects
        container = self.getm(names=names, store=store)

        plt.figure(num=num)
        for k, v in container.items():
            f, p = v.psd(**kwargs)
            plt.plot(f, p, label=k)

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        plt.grid()
        plt.legend()
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def plot_cycle_range(self, names=None, n=200, w=None, bw=1., figurename=None, show=None, num=1, store=True,
                         **kwargs):
        """
        Plot cycle range versus number of occurrences.

        Parameters
        ----------
        names : str/list/tuple, optional
            Time series names
        n : int, optional
            Group by cycle range in *n* equidistant bins.
        w : float, optional
            Group by cycle range in *w* wide equidistant bins. Overrides *n*.
        bw : float, optional
            Bar width, expressed as ratio of bin width.
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        kwargs : optional
            see documentation of TimeSeries.get() method for available options

        Notes
        -----
        When working on a large time series database it is recommended to set store=False to avoid too high memory
        usage. Then the TimeSeries objects will not be stored in the database, only their addresses.

        See Also
        --------
        TsDB.rfc, TsDB.plot_cycle_rangemean, TimeSeries.rfc, rainflow.count_cycles, rainflow.rebin_cycles

        """
        # dict with TimeSeries objects
        container = self.getm(names=names, store=store)

        # ensure that rebinning will be done
        assert (n is not None) or (w is not None), "Cycles must be rebinned for this plot - either 'n' or 'w' must " \
                                                   "be different from None"

        plt.figure(num=num)
        for k, v in container.items():
            # extract cycles
            cycles = v.rfc(**kwargs)

            # rebin cycles
            cycles = rebin_cycles(cycles, binby='range', n=n, w=w)

            r, _, c = zip(*cycles)  # unpack range and count pairs, ignore mean value
            dr = r[1] - r[0]        # bin width, used as basis for bar width
            plt.bar(r, c, dr * bw, label=k, alpha=0.4)

        plt.xlabel('Cycle range')
        plt.ylabel('Cycle count (-)')
        plt.grid()
        plt.legend()
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def plot_cycle_rangemean(self, names=None, n=None, w=None, figurename=None, show=True, num=1, store=True, **kwargs):
        """
        Plot cycle range-mean versus number of occurrences.

        Parameters
        ----------
        names : str/list/tuple, optional
            Time series names
        n : int, optional
            Group by cycle range in *n* equidistant bins.
        w : float, optional
            Group by cycle range in *w* wide equidistant bins. Overrides *n*.
        figurename : str, optional
            Save figure to file 'figurename' instead of displaying on screen.
        show : bool, optional
            Show figure? Defaults to False if `figurename` is specified, otherwise True.
        num : int, optional
            Matplotlib figure number. Defaults to 1.
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        kwargs : optional
            see documentation of TimeSeries.get() method for available options

        Notes
        -----
        When working on a large time series database it is recommended to set ``store=False`` to avoid too high memory
        Cycle means are represented by weighted averages in each bin.

        When working on a large time series database it is recommended to set store=False to avoid too high memory
        usage. Then the TimeSeries objects will not be stored in the database, only their addresses.

        See also
        --------
        qats.TimeSeries

        """
        # dict with TimeSeries objects
        container = self.getm(names=names, store=store)

        plt.figure(num=num)
        for k, v in container.items():
            # extract cycles
            cycles = v.rfc(**kwargs)

            # rebin cycles
            if (n is not None) or (w is not None):
                cycles = rebin_cycles(cycles, binby='range', n=n, w=w)

            ranges, means, counts = zip(*cycles)      # unpack range and count pairs, ignore mean value
            plt.scatter(means, ranges, s=[2. * c for c in counts], label=k, alpha=0.4)  # double marker size for improved readability

        plt.xlabel('Cycle mean')
        plt.ylabel('Cycle range')
        plt.grid()
        plt.legend()
        if figurename is not None:
            plt.savefig(figurename)
        if show is True or (show is None and figurename is None):
            plt.show()

    def rename(self, name, newname):
        """
        Rename a timeseries (and update register accordingly).

        Parameters
        ----------
        name : str
            Unique time series name (pattern)
        newname : str
            New time series name (i.e. unique id/key less the path of parent file).

        Notes
        -----
        A single match must be obtained for the specified `name`.
        """
        # find matching time series names
        names = self.list(names=name, display=False, relative=False)
        n = len(names)
        if n == 0:
            raise LookupError("No match found for specified name")
        elif n > 1:
            raise ValueError("More than one match found for specified name:"
                             "\n    %s" % "\n    ".join(names))

        # define new key, and check that it doesn't exist
        # todo: consider replace (parent --> "") instead of dirname (only relevant for ts from .h5?)
        oldkey = names[0]
        newkey = os.path.join(self._path_dirname(oldkey), newname)
        if newkey in self.register_keys:
            raise ValueError(f"New key already exists: {newkey}")

        # rename (change register)
        self.register[newkey] = self.register.pop(oldkey)
        self.register_parent[newkey] = self.register_parent.pop(oldkey)
        self.register_indices[newkey] = self.register_indices.pop(oldkey)
        self.register_keys[self.register_keys.index(oldkey)] = newkey

        # rename TimeSeries instance if it is pre-loaded
        ts = self.register.get(newkey, None)
        if isinstance(ts, TimeSeries):
            ts.name = newname

        return

    def stats(self, statsdur=10800., names=None, ind=None, store=True, fullkey=False, **kwargs):
        """
        Get statistics for time series processed according to parameters

        Parameters
        ----------
        statsdur : float, optional
            Duration in seconds for estimation of extreme value distribution (Gumbel) from peak distribution (Weibull).
            Default is 3 hours.
        names : str or list or tuple, optional
            Time series names
        ind : int or list, optional
            Time series indices in database
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        fullkey : bool, optional
            Use full key in returned container
        kwargs : optional
            see documentation of TimeSeries.stats() and TimeSeries.get() methods for available options

        Returns
        -------
        dictionary
            Each entry is a dictionary with statistics

        Notes
        -----
        Full unique path/key is obtained by joining the common part of the paths/keys and the unique part of the keys

        When working on a large time series database it is recommended to set ``store=False`` to avoid too high memory
        usage. Then the TimeSeries objects will not be stored in the database, only their addresses.

        See also
        --------
        qats.TimeSeries.stats, qats.TimeSeries.get

        Examples
        --------
        To get sample statistics and 3-hour extreme (default) value statistics for time series in database

        >>> from qats import TsDB
        >>> db = TsDB.fromfile('filename')
        >>> stats = db.stats()

        To get sample statistics and 1-hour (3600 seconds) extreme value statistics for time series in database which
        names starts with 'Mooring'.

        >>> stats = db.stats(statsdur=3600., names="Mooring*")

        To get sample statistics and 3-hour extreme value statistics for time series in database which names starts with
        'Mooring' low-pass filtered at 0.025 Hz.

        >>> stats = db.stats(names="Mooring*", filterargs=("lp", 0.025))

        To ignore the transient part of the time series, time window may be specified:

        >>> stats = db.stats(names="Mooring*", twin=(1000., 1e12), filterargs=("lp", 0.025))

        """
        # todo: create entry in dictionary with meta data such as applied time window, filter etc.
        # read time series and put in ordered dictionary (reuse getm() to avoid duplicating code)
        container = OrderedDict((k, v.stats(statsdur=statsdur, **kwargs)) for k, v in
                                self.getm(names=names, ind=ind, store=store, fullkey=fullkey).items())

        return container

    def stats_dataframe(self, statsdur=10800., names=None, ind=None, store=True, fullkey=False, **kwargs):
        """
        Get Pandas Dataframe with time series statistics.

        Parameters
        ----------
        statsdur : float, optional
            Duration in seconds for estimation of extreme value distribution (Gumbel) from peak distribution (Weibull).
            Default is 3 hours.
        names : str or list or tuple, optional
            Time series names
        ind : int or list, optional
            Time series indices in database
        store : bool, optional
            Disable time series storage. Default is to store the time series objects first time it is read.
        fullkey : bool, optional
            Use full key in returned container
        kwargs : optional
            see documentation of TimeSeries.stats() and TimeSeries.get() methods for available options

        Returns
        -------
        Dataframe
            Statistics for each time series organized in columns.

        Notes
        -----

        See also
        --------
        qats.TsDB.stats, pandas.Dataframe

        Examples
        --------
        >>> from qats import TsDB
        >>> db = TsDB.fromfile('filename')
        >>> df = db.stats_dataframe()
        >>> # print first part of dataframe
        >>> print(df.head())

        >>> # transpose dataframe to get statistics for each time series in rows instead of columns
        >>> df = df.transpose(copy=True)
        """
        df = pd.DataFrame(self.stats(statsdur=statsdur, names=names, ind=ind, store=store, fullkey=fullkey, **kwargs))

        return df

    def update(self, tsdb, names=None, shallow=False):
        """
        Update TsDB with speicified keys/names from other TsDB instance.

        Parameters
        ----------
        tsdb: TsDB
            TsDB instance to update from.
        names : str or list or tuple, optional
            Time series names
        shallow: bool, optional
            If False (default), the copied TimeSeries objects do not point back to the TimeSeries instances in the
            source TsDB.

        """
        if not isinstance(tsdb, TsDB):
            raise TypeError("expected TsDB instance, got: %s" % type(tsdb))

        container = tsdb.getm(names=names, fullkey=True)
        for key, ts in container.items():
            if key in self.register.keys():
                raise KeyError("The specified key is not unique: %s" % key)

            if shallow is False:
                ts = ts.copy()

            self.register[key] = ts
            self.register_keys.append(key)
            self.register_parent[key] = tsdb.register_parent[key]
            self.register_indices[key] = tsdb.register_indices[key]

        return
