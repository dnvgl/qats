from nptdms import TdmsFile as npTdmsFile
from typing import List, Tuple, Union
import numpy as np
import os


def read_names(path):
    """
    Extracts time series names from Technical Data Management Streaming file `.tdms`.
    ref: http://www.ni.com/product-documentation/3727/en/

    Parameters
    ----------
    path: str
        Path (relative or absolute) to tdms-file

    Returns
    -------
    list
        List of time series names (datasets)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % path)

    tdms_file = npTdmsFile(path)
    names = []
    groups = tdms_file.groups()
    for group in groups:
        channels = tdms_file.group_channels(group=group)
        for channel in channels:
            if "'time'" not in channel.path.lower():
                names.append(channel.path.replace("'", ""))
    return names


def read_data(path: str, names: Union[List[str], Tuple[str]] = None):
    """
    Extracts time series data from `.tdms` (or `hdf5`) files exported from LabVIEW (normally).

    Parameters
    ----------
    path: str
        File name.
    names: str or list, optional
        Timeseries/dataset names. If None (default), all time series are read.

    Returns
    -------
    list
        List of arrays with time and data

    """
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % path)

    if isinstance(names, str):
        names = [names]
    elif type(names) in (list, tuple):
        pass
    elif names is None:
        # if names not specified, get all names
        names = read_names(path)
    else:
        raise TypeError("`names` must be str/list/tuple, got: %s" % type(names))

    arrays = []
    tdms_file = npTdmsFile(path)
    for name in names:
        group_name = name.split('/')[1]
        channel_name = name.split('/')[2]

        channel_obj = tdms_file.object(group_name, channel_name)
        data = channel_obj.data

        timearr = []

        try:
            timearr = channel_obj.time_track()
        except KeyError:
            channels = tdms_file.group_channels(group=group_name)
            for channel in channels:
                if channel.channel.lower() == 'time':
                    timearr = np.array(channel.data)
                    break

        if type(timearr) is not np.ndarray:
            raise Exception("no time info extracted for dataset '%s'" % name)

        arr = np.array([timearr, data])
        arrays.append(arr)

    return arrays
