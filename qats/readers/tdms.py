from nptdms import TdmsFile as npTdmsFile
from typing import List, Tuple, Union
from datetime import datetime
import numpy as np
import os


def read_names(path):
    """
    Read time series names from Technical Data Management Streaming (TDMS) file.

    Parameters
    ----------
    path: str
        Path (relative or absolute) to tdms-file

    Returns
    -------
    list
        List of time series names (datasets)

    References
    ----------
    1. http://www.ni.com/product-documentation/3727/en/

    """
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % path)

    tdms_file = npTdmsFile(path)
    names = []
    groups = tdms_file.groups()
    for group in groups:
        channels = tdms_file.group_channels(group=group)
        for channel in channels:
            if 'time' != channel.channel.lower():
                names.append(channel.group + '\\' + channel.channel)

    return names


def read_data(path: str, names: Union[List[str], Tuple[str]] = None):
    """
    Read time series data from Technical Data Management Streaming (TDMS) files typically exported from LabVIEW.

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
        group_name = name.split('\\')[0]
        channel_name = name.split('\\')[1]

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

        # verify that time array has correct shape (==> should be same as `data` shape)
        if not timearr.shape == data.shape:
            raise Exception("unexpected error: `time` has shape " + str(timearr.shape) + "while data has"
                            "shape " + str(data.shape) + " (should be equal)")

        if type(timearr[0]) is np.datetime64:
            arr = [timearr.astype(datetime), data]
        else:
            arr = [timearr, data]
        arrays.append(arr)

    return arrays
