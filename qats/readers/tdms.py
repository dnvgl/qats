from nptdms import TdmsFile
from typing import List, Tuple, Union
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

    f = TdmsFile(path)
    names = [f"{g.name}\\{c.name}" for g in f.groups() for c in f[g.name].channels() if c.name.lower() != 'time']

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
    f = TdmsFile.open(path)
    for name in names:
        # assuming an object hierarchy with depth 2 'group-channel'
        assert len(name.split("\\")) == 2, f"Unable to parse group name and channel name from {name}."
        group_name, channel_name = name.split("\\")
        group = f[group_name]
        channel = group[channel_name]
        data = channel[:]
        time = None

        try:
            # try to fetch time track defined by wf_start_time and wf_start_offset attributes
            time = channel.time_track()
        except KeyError:
            # wf_start_time and wf_start_offset attributes does not exist
            # check if time is a separate channel
            for _ in ["time", "Time"]:
                try:
                    time = group[_][:]
                except KeyError:
                    pass
                else:
                    break
        finally:
            if time is None:
                raise KeyError(f"Could not find time array for channel {channel_name}.")
            elif not time.shape == data.shape:
                # verify that time and data arrays have the same shape
                raise ValueError(f"Time {time.shape} and data {data.shape} are not equally shaped.")
            else:
                arrays.append([time, data])

    return arrays
