"""
Readers CSV formatted time series files
"""
import pandas as pd


def read_names(path):
    """
    Read time series names from a comma-separated file.

    Parameters
    ----------
    path : str
        Name of .csv file

    Returns
    -------
    list
        Time series names

    Notes
    -----
    The series names are expected found on the header row. Time is expected to be in the first column.

    """
    # pandas will infer the format e.g. delimiter.
    df = pd.read_csv(path, nrows=1, sep=None, engine='python', encoding='utf-8')
    names = list(df)
    _ = names.pop(0)    # remove time which is assumed to be in the first column
    return names


def read_data(path, ind=None):
    """
    Read time series data arranged column wise on a comma-separated file.

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
        Time and data

    """
    df = pd.read_csv(path, usecols=ind, sep=None, engine='python')  # pandas will infer the format e.g. delimiter.
    return df.T.to_numpy()
