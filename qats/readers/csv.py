"""
Readers CSV formatted time series files
"""
import pandas as pd


def read_keys(path):
    """
    Read data series keys/names from a comma-separated file.

    Parameters
    ----------
    path : str
        Name of .csv file

    Returns
    -------
    list
        Name of data series

    Notes
    -----
    The series names are expected found on the header row.

    """
    # pandas will infer the format e.g. delimiter.
    df = pd.read_csv(path, nrows=1, sep=None, engine='python')
    return list(df)


def read_data(path, ind=None):
    """
    Read time series arranged column wise on a comma-separated file.

    Parameters
    ----------
    path : str
        CSV file path (relative or absolute)
    ind : list|tuple, optional
        Read only a subset of the data series specified by index , with 0 being the first. For example,
        ind = (1,4,5) will extract the 2nd, 5th and 6th columns. The default, None, results in all columns being read.

    Returns
    -------
    array
        Time and data series

    """
    df = pd.read_csv(path, usecols=ind, sep=None, engine='python')  # pandas will infer the format e.g. delimiter.
    return df.T.to_numpy()
