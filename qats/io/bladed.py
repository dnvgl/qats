"""
Readers for Bladed (DNV) result files.

Bladed writes each output group as two files: a text header (``.%NN``, e.g. ``powprod.%04``) and the data
(``.$NN``), named in the header. Open the header file.

.. versionadded :: 5.5.0
"""

import math
import os
import re

import numpy as np

from .._validation import QatsValueError
from .base import Reader, SeriesData, SeriesInfo, SourceFile

# SI units of Bladed's dimension codes, as the Bladed Results API gives them
_UNITS = {
    "T": "s",
    "L": "m",
    "L/T": "m/s",
    "L/TT": "m/s²",
    "A": "rad",
    "A/T": "rad/s",
    "A/TT": "rad/s²",
    "F": "N",
    "F/L": "N/m",
    "F/LL": "N/m²",
    "FL": "Nm",
    "FL/L": "Nm/m",
    "MLL": "kgm²",
    "P": "W",
    "Q": "VAr",
    "N": None,  # no unit
}
_FORMATS = {"R*4": "<f4", "R*8": "<f8"}
_STATISTICS = ("ULOADS", "MAXTIME", "MINTIME", "MEAN")  # where the description of the data ends
_AXIS_KEYS = ("AXIUNIT", "AXIMETH", "AXIVAL", "AXITICK", "MIN", "STEP")
_QUOTED = re.compile(r"'([^']*)'")


def _unit(code):
    """SI unit of a Bladed dimension code; a code without a known SI unit is returned as it is."""
    code = code.strip()
    return _UNITS.get(code, code)


def read_header(path):
    """
    Read the header file of a Bladed output group (``.%NN``).

    Parameters
    ----------
    path : str
        Header file path

    Returns
    -------
    dict
        The keywords and their values as text, except: ``VARIAB`` (list of variable names), ``VARUNIT`` and ``DIMENS``
        (lists), ``GENLAB`` (the group name, without quotes) and ``axes`` (list of dicts with ``label`` and the axis
        keywords ``AXIUNIT``, ``AXIMETH``, ``AXIVAL``, ``AXITICK``, ``MIN`` and ``STEP``). The statistics at the end of
        the header are not read.

    Notes
    -----
    .. versionadded :: 5.5.0

    Examples
    --------
    >>> header = read_header('powprod.%04')
    >>> header['GENLAB']
    'Control variables'
    """
    header = {"axes": []}
    with open(path, encoding="latin-1") as f:
        for line in f:
            if not line.strip() or line[0].isspace():
                continue  # blank, or a continuation line of the statistics
            key, _, value = line.replace("\t", " ").partition(" ")
            key, value = key.strip(), value.strip()
            if key in _STATISTICS:
                break
            if key == "AXISLAB":
                header["axes"].append({"label": _QUOTED.findall(value)[0] if "'" in value else value})
            elif key in _AXIS_KEYS and header["axes"]:
                header["axes"][-1][key] = value
            elif key == "VARIAB":
                header[key] = _QUOTED.findall(value)
            elif key in ("VARUNIT", "DIMENS"):
                header[key] = value.split()
            elif key == "GENLAB":
                header[key] = _QUOTED.findall(value)[0] if "'" in value else value
            else:
                header[key] = value
    return header


def _axis_labels(axis, n):
    """
    Labels of the points on an axis other than time, e.g. stations along a blade: '16.0741 m' from listed values
    (AXIMETH 3) or a regular grid (AXIMETH 2), or the text labels of e.g. tower members (AXIMETH 1, AXITICK).
    """
    method = axis.get("AXIMETH")
    if method == "1":
        return _QUOTED.findall(axis.get("AXITICK", ""))
    if method == "2":
        values = float(axis["MIN"]) + float(axis["STEP"]) * np.arange(n)
    else:
        values = [float(v) for v in axis.get("AXIVAL", "").split()]
    unit = _unit(axis.get("AXIUNIT", "N"))
    return [f"{value:g}" + (f" {unit}" if unit else "") for value in values]


def _layout(path, header=None):
    """
    The series of a Bladed output group as (name, unit, variable index, station index or None), the data file, its
    data type and dimensions, and the time array. Raises QatsValueError if the group is not a time series that can be
    read.
    """
    header = read_header(path) if header is None else header
    group = header.get("GENLAB", "")

    def invalid(reason):
        return QatsValueError(f"The Bladed output group '{group}' in '{path}' {reason}")

    for key in ("FILE", "DIMENS", "VARIAB", "FORMAT"):
        if key not in header:
            raise QatsValueError(f"'{path}' is not a Bladed header file: no {key}")
    axes = header["axes"]
    if not axes or axes[-1]["label"] != "Time" or axes[-1].get("AXIMETH") != "2":
        last = axes[-1]["label"] if axes else "none"
        raise invalid(f"is not a time series: its last axis is '{last}' (CONFIG {header.get('CONFIG', '?')})")
    if header.get("ACCESS", "D") != "D" or header["FORMAT"] not in _FORMATS:
        raise invalid(f"has a data format QATS can't read (ACCESS {header.get('ACCESS')}, FORMAT {header['FORMAT']})")

    dims = [int(d) for d in header["DIMENS"]]
    variables = header["VARIAB"]
    units = [_unit(code) for code in header.get("VARUNIT", [])] or [None] * len(variables)
    if len(dims) not in (2, 3) or dims[0] != len(variables) or len(units) != len(variables):
        raise invalid(f"has {len(variables)} variables and {len(units)} units, but dimensions {dims}")

    series = []
    if len(dims) == 2:
        series = [(name, unit, i, None) for i, (name, unit) in enumerate(zip(variables, units))]
    else:
        labels = _axis_labels(axes[0], dims[1])
        if len(labels) != dims[1]:
            raise invalid(f"has {len(labels)} values on its axis '{axes[0]['label']}', but dimensions {dims}")
        for i, (name, unit) in enumerate(zip(variables, units)):
            for j, label in enumerate(labels):
                series.append((f"{name} @ {label}", unit, i, j))

    first, step = float(axes[-1]["MIN"]), float(axes[-1]["STEP"])
    time = first + step * np.arange(dims[-1])
    data_file = os.path.join(os.path.dirname(path), header["FILE"])
    return series, data_file, _FORMATS[header["FORMAT"]], dims, time


def read_names(path):
    """
    Read the series names of a Bladed output group from its header file (``.%NN``).

    Parameters
    ----------
    path : str
        Header file path

    Returns
    -------
    list
        Series names. For groups with a second axis, e.g. stations along a blade, one series per variable and point
        on the axis, named ``<variable> @ <point>``, e.g. ``Blade 1 Mx (Principal elastic axes) @ 16.0741 m`` or
        ``Tower Mx @ Mbr 5 End 1`` for an axis with text labels.

    Raises
    ------
    QatsValueError
        If the group is not a time series (its last axis is not time), or has a data format QATS can't read.

    Notes
    -----
    .. versionadded :: 5.5.0

    Examples
    --------
    >>> names = read_names('powprod.%04')
    """
    return [name for name, _, _, _ in _layout(path)[0]]


def read_units(path):
    """
    Read the units of the series of a Bladed output group from its header file (``.%NN``).

    Parameters
    ----------
    path : str
        Header file path

    Returns
    -------
    dict
        SI unit by series name, e.g. ``Nm`` for Bladed's ``FL``; None for series without a unit. A dimension code
        without a known SI unit is given as it is in the file.

    Notes
    -----
    .. versionadded :: 5.5.0
    """
    return {name: unit for name, unit, _, _ in _layout(path)[0]}


def read_data(path, ind=None):
    """
    Read time and series of a Bladed output group from its header file (``.%NN``) and data file (``.$NN``).

    Parameters
    ----------
    path : str
        Header file path
    ind : list of integers, optional
        Defines which series to include in the returned array, by their position in `read_names` plus one: `0` is the
        index of the time array. All series by default.

    Returns
    -------
    array
        Time and data, one series per row, in the order of `ind`.

    Notes
    -----
    Time is not stored in the data file: it is the first time and time step on the header's time axis.

    .. versionadded :: 5.5.0

    Examples
    --------
    >>> data = read_data('powprod.%04', ind=[0, 2])
    >>> t, x = data
    """
    series, data_file, dtype, dims, time = _layout(path)
    ind = list(range(len(series) + 1)) if ind is None else list(ind)
    count = math.prod(dims)
    if not os.path.isfile(data_file):
        raise QatsValueError(f"The data file '{data_file}' named in '{path}' does not exist")
    values = np.fromfile(data_file, dtype=dtype, count=count)
    if values.size < count:
        raise QatsValueError(f"The data file '{data_file}' holds {values.size} values, but '{path}' requires {count}")
    values = values.reshape(dims, order="F")  # the first index varies fastest in the file

    data = np.empty((len(ind), time.size), dtype=np.float64)
    for row, i in enumerate(ind):
        if i == 0:
            data[row] = time
        else:
            _, _, variable, station = series[i - 1]
            data[row] = values[variable] if station is None else values[variable, station]
    return data


class _BladedSource(SourceFile):
    def __init__(self, path):
        super().__init__(path)
        self._series = None
        self._group = None

    def _scan(self):
        if self._series is None:
            header = read_header(self.path)
            self._group = (header.get("GENLAB", ""),)
            self._series = _layout(self.path, header)[0]

    def series(self):
        self._scan()
        return [SeriesInfo(name, unit, group=self._group) for name, unit, _, _ in self._series]

    def read(self, series):
        self._scan()
        position = {name: i + 1 for i, (name, _, _, _) in enumerate(self._series)}  # the last of duplicate names wins
        data = read_data(self.path, ind=[0] + [position[s.name] for s in series])
        t = data[0]
        return [SeriesData(s.name, t, data[i + 1], unit=s.unit, group=s.group) for i, s in enumerate(series)]

    def _legacy_index(self, name):
        """Position of the series on file, as stored in `TsDB.register_indices` before 5.5.0."""
        return None


class BladedReader(Reader):
    """
    Reader for Bladed (DNV) result files: the header file of an output group (``.%NN``), with the data in the file it
    names (``.$NN``). Output groups whose last axis is time are read; for groups with a second axis, e.g. stations
    along a blade, each variable and station is a series. Units are read and given in SI.

    .. versionadded :: 5.5.0
    """

    name = "bladed"
    description = "Bladed output"
    patterns = ("*.%[0-9]*",)  # the group number has two or more digits, e.g. .%04 or .%150

    def can_read(self, path):
        """
        True if the file is a Bladed header, with the data file and dimensions of an output group. Output groups that
        aren't time series are accepted, so that reading them gives an error that says why.
        """
        try:
            header = read_header(path)
        except OSError:
            return False
        return "FILE" in header and "DIMENS" in header

    def open(self, path):
        return _BladedSource(path)
