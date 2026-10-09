"""
Readers CSV formatted time series files
"""

import pandas as pd

from .base import Reader, _RowSource


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
    df = pd.read_csv(path, nrows=1, sep=None, engine="python", encoding="utf-8")
    names = list(df)
    _ = names.pop(0)  # remove time which is assumed to be in the first column
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
        Time and data, with the columns in the order given by `ind`.

    """
    df = pd.read_csv(path, usecols=ind, sep=None, engine="python")  # pandas will infer the format e.g. delimiter.
    if ind is not None:
        # pandas returns the columns in file order, whatever the order of `usecols`
        cols = sorted(set(ind))
        df = df.iloc[:, [cols.index(i) for i in ind]]
    return df.T.to_numpy()


class _CsvSource(_RowSource):
    def _read_names(self):
        return read_names(self.path)

    def _read_rows(self, ind):
        return read_data(self.path, ind=ind)


class CsvReader(Reader):
    """
    Reader for comma-separated files with the names on the header row and time in the first column (``.csv``).

    .. versionadded :: 5.5.0
    """

    name = "csv"
    description = "Comma-separated values"
    patterns = ("*.csv",)

    def open(self, path):
        return _CsvSource(path)
