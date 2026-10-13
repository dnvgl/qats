"""
Readers CSV formatted time series files
"""

import codecs
import csv

import pandas as pd

from .._validation import QatsValueError
from .base import Reader, _RowSource

# encodings tried in turn: UTF-8 (also with a byte-order mark), then the Windows encoding of Excel's "CSV (comma
# delimited)" in Western European languages
_ENCODINGS = ("utf-8", "cp1252")
_COMMENT = "#"


def _is_utf8(path, chunk_size=2**20):
    """True if the whole file decodes as UTF-8."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                decoder.decode(chunk)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _csv_options(path):
    """
    Options for pandas.read_csv for a .csv file: its encoding, the comment lines before the header, and the decimal
    mark (a decimal comma in files with ';' as separator, as Excel writes with e.g. Norwegian regional settings).
    """
    encoding = _ENCODINGS[0] if _is_utf8(path) else _ENCODINGS[1]
    skiprows = 0
    header = data = ""
    with open(path, encoding=encoding) as f:
        for line in f:
            if line.startswith(_COMMENT) and not header:
                skiprows += 1
            elif not header:
                header = line
            else:
                data = line
                break
    decimal = "."
    try:
        # the separator as pandas finds it (sep=None): sniffed from the header line
        separator = csv.Sniffer().sniff(header).delimiter
    except csv.Error:
        separator = None  # e.g. a single column
    if separator == ";" and "," in data:
        decimal = ","
    if separator is not None and data:
        # more values than names, e.g. decimal commas with ',' as separator: pandas would shift the columns and read
        # wrong values without an error
        names, values = (next(csv.reader([line.strip()], delimiter=separator)) for line in (header, data))
        if len(values) > len(names) and any(v.strip() for v in values[len(names) :]):
            raise QatsValueError(
                f"The file '{path}' has {len(values)} values on its first data row but {len(names)} names, with "
                f"{separator!r} as separator. Decimal commas need ';' as separator."
            )
    return dict(encoding=encoding, skiprows=skiprows, decimal=decimal)


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

    .. versionchanged :: 5.5.0
        Files in the Windows-1252 encoding and files with comment lines (starting with '#') before the header are
        read.
    """
    # pandas will infer the format e.g. delimiter.
    df = pd.read_csv(path, nrows=1, sep=None, engine="python", index_col=False, **_csv_options(path))
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

    Raises
    ------
    QatsValueError
        If a column holds a value that is not a number. The message names the column, the data row and the value.

    Notes
    -----
    .. versionchanged :: 5.5.0
        Files in the Windows-1252 encoding, files with comment lines (starting with '#') before the header, and files
        with ';' as separator and a decimal comma (e.g. ``1,5``) are read. A value that is not a number raises
        QatsValueError naming the column and row; before, the error was raised later and did not say where. A
        separator at the end of the rows no longer shifts the columns, and a first data row with more values than
        names (e.g. decimal commas with ',' as separator) raises QatsValueError; before, both gave wrong values.
    """
    # pandas will infer the format e.g. delimiter. index_col=False: a separator at the end of the rows does not make
    # the first column an index, which would shift the other columns
    df = pd.read_csv(path, usecols=ind, sep=None, engine="python", index_col=False, **_csv_options(path))
    for name in df.columns:
        if not pd.api.types.is_numeric_dtype(df[name]):
            # a value that is not a number (empty cells are nan, which is a number)
            invalid = pd.to_numeric(df[name], errors="coerce").isna() & df[name].notna()
            if not invalid.any():
                continue
            row = int(invalid.to_numpy().nonzero()[0][0])
            raise QatsValueError(
                f"The file '{path}' has a value that is not a number in column '{name}', data row {row + 1}: "
                f"{df[name].iloc[row]!r}. Numbers must use a decimal point, or a decimal comma with ';' as separator."
            )
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
