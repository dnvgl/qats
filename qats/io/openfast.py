"""
Readers for OpenFAST results: text output files (``.out``) and binary output files (``.outb``).

OpenFAST and its module drivers (AeroDyn, BeamDyn, HydroDyn, MoorDyn, InflowWind, ...) write the same two formats.
Both hold the channel names and units, with time in the first channel.

.. versionadded :: 5.5.0
"""

import re
import struct
from dataclasses import dataclass

import numpy as np

from .._validation import QatsValueError
from .base import Reader, SeriesData, SeriesInfo, _RowSource

# File format identifiers of binary output files, as written by OpenFAST's NWTC Subroutine Library (WrBinFAST)
_FILEID_WITH_TIME = 1  # data as scaled int16, time as scaled int32
_FILEID_WITHOUT_TIME = 2  # data as scaled int16, time from first time and constant step, names 10 characters
_FILEID_NO_COMPRESS = 3  # data as float64, time from first time and constant step
_FILEID_CHANLEN_IN = 4  # as 2, with the length of the names on file

_NAME_LINE_MAX = 40  # the name line of a text output file is searched for among the first lines only
_CHUNK = 2**22  # values read at a time from a binary output file, to limit memory use


def _strip_unit(unit):
    """Unit without one pair of enclosing parentheses, e.g. 'kN' from '(kN)'. None if empty."""
    unit = unit.strip()
    if unit.startswith("(") and unit.endswith(")"):
        unit = unit[1:-1].strip()
    return unit or None


def _split_out_line(line):
    """Fields of a name or unit line: tab-separated (fields may be padded with spaces) or space-separated."""
    line = line.rstrip("\r\n")
    if "\t" in line:
        fields = [field.strip() for field in line.split("\t")]
        while fields and not fields[-1]:
            fields.pop()
        return fields
    return line.split()


def _read_out_header(path):
    """Lines before the name line, names and units (time included), and the number of header lines."""
    with open(path, encoding="latin-1") as f:
        lines = []
        for line in f:
            fields = line.split()
            if fields and fields[0].lower() == "time":
                names = _split_out_line(line)
                units = _split_out_line(f.readline())
                break
            lines.append(line)
            if len(lines) >= _NAME_LINE_MAX:
                raise QatsValueError(
                    f"The file '{path}' is not an OpenFAST output file: no line starting with 'Time' among the "
                    f"first {_NAME_LINE_MAX} lines"
                )
        else:
            raise QatsValueError(f"The file '{path}' is not an OpenFAST output file: no line starting with 'Time'")

    if len(units) != len(names):
        raise QatsValueError(
            f"The file '{path}' has {len(names)} channel names but {len(units)} units on the line below them"
        )
    return lines, names, [_strip_unit(unit) for unit in units], len(lines) + 2


def read_out_names(path):
    """
    Read the channel names from an OpenFAST text output file (``.out``).

    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    list
        Channel names, without time

    Notes
    -----
    The names are on the first line that starts with "Time" (in any case), among the first 40 lines. The fields are
    separated by tabs, or by spaces if the line holds no tabs.

    .. versionadded :: 5.5.0

    Examples
    --------
    >>> names = read_out_names('turbine.out')
    """
    return _read_out_header(path)[1][1:]


def read_out_units(path):
    """
    Read the channel units from an OpenFAST text output file (``.out``).

    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    dict
        Unit by channel name, time included. Units are given without their enclosing parentheses, e.g. ``kN`` for
        ``(kN)`` on file; an empty unit is None.

    Notes
    -----
    .. versionadded :: 5.5.0

    Examples
    --------
    >>> units = read_out_units('turbine.out')
    >>> units['Time']
    's'
    """
    _, names, units, _ = _read_out_header(path)
    return dict(zip(names, units))


_FORTRAN_EXPONENT = re.compile(r"(?<=[\d.])([+-]\d{3})$")


def _to_float(field):
    """Float from a Fortran number, also a three-digit exponent written without 'E' (e.g. '1.234-100')."""
    return float(_FORTRAN_EXPONENT.sub(r"E\1", field))


def read_out_data(path, ind=None):
    """
    Read time and channels from an OpenFAST text output file (``.out``).

    Parameters
    ----------
    path : str
        File path
    ind : list of integers, optional
        Defines which channels to include in the returned array. Each of the indices in `ind` refer to the position
        of the channel on file. Note that `0` is the index of the time array. All channels by default.

    Returns
    -------
    array
        Time and data, one channel per row, in the order of `ind`.

    Notes
    -----
    Lines that start with "This" after the data, such as "This output file was closed on ...", are skipped.

    .. versionadded :: 5.5.0

    Examples
    --------
    >>> data = read_out_data('turbine.out', ind=[0, 3])
    >>> t, x = data
    """
    skiprows = _read_out_header(path)[3]
    kwargs = dict(skiprows=skiprows, usecols=ind, comments="This", ndmin=2, unpack=True, encoding="latin-1")
    try:
        return np.loadtxt(path, **kwargs)
    except ValueError:
        # Fortran drops the 'E' of exponents with three digits
        return np.loadtxt(path, converters=_to_float, **kwargs)


@dataclass
class _OutbHeader:
    """Header of an OpenFAST binary output file."""

    fileid: int
    nchannels: int  # without time
    nsteps: int
    time: tuple  # (scale, offset) of the packed time (file id 1), else (first time, time step)
    scale: np.ndarray  # None for file id 3
    offset: np.ndarray  # None for file id 3
    description: str
    names: list  # time included
    units: list  # time included
    time_position: int  # file position of the packed time (file id 1)
    data_position: int  # file position of the data
    size: int  # file size needed for header and data


def _read_outb_header(path):
    """Header of an OpenFAST binary output file. Raises QatsValueError if the header is not valid."""

    def invalid(reason):
        return QatsValueError(f"The file '{path}' is not a valid OpenFAST binary output file: {reason}")

    def unpack(fmt):
        size = struct.calcsize(fmt)
        buffer = f.read(size)
        if len(buffer) < size:
            raise invalid("the header is cut off")
        return struct.unpack(fmt, buffer)

    def read_strings(n, length):
        buffer = f.read(n * length)
        if len(buffer) < n * length:
            raise invalid("the header is cut off")
        text = buffer.decode("latin-1")
        return [text[i * length : (i + 1) * length].strip() for i in range(n)]

    with open(path, "rb") as f:
        (fileid,) = unpack("<h")
        if fileid not in (_FILEID_WITH_TIME, _FILEID_WITHOUT_TIME, _FILEID_NO_COMPRESS, _FILEID_CHANLEN_IN):
            raise invalid(f"unknown file format identifier {fileid}")
        (lenname,) = unpack("<h") if fileid == _FILEID_CHANLEN_IN else (10,)
        if lenname < 1:
            raise invalid(f"channel name length {lenname}")
        nchannels, nsteps = unpack("<ii")
        if nchannels < 1 or nsteps < 0:
            raise invalid(f"{nchannels} channels and {nsteps} time steps")
        time = unpack("<dd")
        if fileid != _FILEID_WITH_TIME and not time[1] > 0.0:
            raise invalid(f"time step {time[1]}")
        if fileid != _FILEID_NO_COMPRESS:
            scale = np.array(unpack(f"<{nchannels}f"), dtype=np.float64)
            offset = np.array(unpack(f"<{nchannels}f"), dtype=np.float64)
        else:
            scale = offset = None
        (lendesc,) = unpack("<i")
        if lendesc < 0:
            raise invalid(f"description length {lendesc}")
        description = read_strings(1, lendesc)[0] if lendesc else ""
        names = read_strings(nchannels + 1, lenname)
        units = [_strip_unit(unit) for unit in read_strings(nchannels + 1, lenname)]
        time_position = f.tell()
        f.seek(0, 2)
        filesize = f.tell()

    data_position = time_position + (4 * nsteps if fileid == _FILEID_WITH_TIME else 0)
    itemsize = 8 if fileid == _FILEID_NO_COMPRESS else 2
    size = data_position + itemsize * nsteps * nchannels
    if filesize < size:
        raise invalid(f"the file holds {filesize} bytes, but its header requires {size}")

    return _OutbHeader(
        fileid, nchannels, nsteps, time, scale, offset, description, names, units, time_position, data_position, size
    )


def read_outb_names(path):
    """
    Read the channel names from an OpenFAST binary output file (``.outb``).

    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    list
        Channel names, without time

    Notes
    -----
    Files written before OpenFAST 2.4 store at most 10 characters of each name.

    .. versionadded :: 5.5.0

    Examples
    --------
    >>> names = read_outb_names('turbine.outb')
    """
    return _read_outb_header(path).names[1:]


def read_outb_units(path):
    """
    Read the channel units from an OpenFAST binary output file (``.outb``).

    Parameters
    ----------
    path : str
        File path

    Returns
    -------
    dict
        Unit by channel name, time included. Units are given without their enclosing parentheses, e.g. ``kN`` for
        ``(kN)`` on file; an empty unit is None.

    Notes
    -----
    .. versionadded :: 5.5.0

    Examples
    --------
    >>> units = read_outb_units('turbine.outb')
    >>> units['Time']
    's'
    """
    header = _read_outb_header(path)
    return dict(zip(header.names, header.units))


def read_outb_data(path, ind=None):
    """
    Read time and channels from an OpenFAST binary output file (``.outb``).

    Parameters
    ----------
    path : str
        File path
    ind : list of integers, optional
        Defines which channels to include in the returned array. Each of the indices in `ind` refer to the position
        of the channel on file. Note that `0` is the index of the time array. All channels by default.

    Returns
    -------
    array
        Time and data, one channel per row, in the order of `ind`.

    Notes
    -----
    Most files store each channel as 16-bit integers, scaled to the channel's range: the values are then accurate to
    1/65535 of the channel's range (maximum minus minimum). Files written with uncompressed output (file format
    identifier 3) store 64-bit floats, which are returned as stored.

    Time is either stored (file format identifier 1, as 32-bit integers scaled to the time range) or computed from
    the first time and the time step on file. Module drivers, such as the AeroDyn driver, store 0 as the first time.

    Data on file after the last time step is ignored.

    .. versionadded :: 5.5.0

    Examples
    --------
    >>> data = read_outb_data('turbine.outb', ind=[0, 3])
    >>> t, x = data
    """
    header = _read_outb_header(path)
    nchannels, nsteps = header.nchannels, header.nsteps
    ind = list(range(nchannels + 1)) if ind is None else list(ind)
    for i in ind:
        if not 0 <= i <= nchannels:
            raise IndexError(f"Channel index {i} is out of range on '{path}', which has {nchannels} channels and time")
    columns = [i - 1 for i in ind if i > 0]
    dtype = np.dtype("<f8") if header.fileid == _FILEID_NO_COMPRESS else np.dtype("<i2")

    with open(path, "rb") as f:
        if header.fileid == _FILEID_WITH_TIME:
            f.seek(header.time_position)
            packed = np.fromfile(f, dtype="<i4", count=nsteps).astype(np.float64)
            timescale, timeoffset = header.time
            time = (packed - timeoffset) / timescale
        else:
            first, step = header.time
            time = first + step * np.arange(nsteps)

        # the data is stored one time step after the other: read a block of time steps at a time
        values = np.empty((len(columns), nsteps), dtype=np.float64)
        f.seek(header.data_position)
        rows = max(1, _CHUNK // nchannels)
        for start in range(0, nsteps, rows):
            n = min(rows, nsteps - start)
            block = np.fromfile(f, dtype=dtype, count=n * nchannels).reshape(n, nchannels)
            values[:, start : start + n] = block[:, columns].T

    if header.fileid != _FILEID_NO_COMPRESS:
        values = (values - header.offset[columns, np.newaxis]) / header.scale[columns, np.newaxis]

    data = np.empty((len(ind), nsteps), dtype=np.float64)
    row = 0
    for j, i in enumerate(ind):
        if i == 0:
            data[j] = time
        else:
            data[j] = values[row]
            row += 1
    return data


class _OpenFastSource(_RowSource):
    """Source for OpenFAST output files, which also hold the units."""

    def __init__(self, path):
        super().__init__(path)
        self._units = None

    def _read_names_units(self):
        """Names and units in file order, time included."""
        raise NotImplementedError

    def _read_names(self):
        names, units = self._read_names_units()
        # the last of duplicate names wins, as for the positions
        self._units = dict(zip(names[1:], units[1:]))
        return names[1:]

    def series(self):
        return [SeriesInfo(s.name, self._units[s.name]) for s in super().series()]

    def read(self, series):
        data = super().read(series)
        return [SeriesData(d.name, d.t, d.x, unit=self._units[d.name]) for d in data]

    def _legacy_index(self, name):
        """Position of the series on file, as stored in `TsDB.register_indices` before 5.5.0."""
        return None


class _OutSource(_OpenFastSource):
    def _read_names_units(self):
        _, names, units, _ = _read_out_header(self.path)
        return names, units

    def _read_rows(self, ind):
        return read_out_data(self.path, ind=ind)


class _OutbSource(_OpenFastSource):
    def _read_names_units(self):
        header = _read_outb_header(self.path)
        return header.names, header.units

    def _read_rows(self, ind):
        return read_outb_data(self.path, ind=ind)


# first words of the header lines written by OpenFAST and its modules and drivers
_OUT_SIGNATURES = ("predictions were generated", "these predictions were generated", "## this file was generated")


class OpenFastReader(Reader):
    """
    Reader for OpenFAST text (``.out``) and binary (``.outb``) output files, also those written by the module
    drivers (AeroDyn, BeamDyn, HydroDyn, MoorDyn, InflowWind, ...).

    Units are read from the files.

    .. versionadded :: 5.5.0
    """

    name = "openfast"
    description = "OpenFAST output"
    patterns = ("*.out", "*.outb")

    def can_read(self, path):
        """
        True if the file has a valid OpenFAST binary header (``.outb``), or the header lines that OpenFAST and its
        modules write before the channel names (``.out``). Other tools also write ``.out`` files.
        """
        try:
            if str(path).lower().endswith(".outb"):
                _read_outb_header(path)
                return True
            lines = _read_out_header(path)[0]
        except (QatsValueError, OSError):
            return False
        return any(line.strip().lower().startswith(_OUT_SIGNATURES) for line in lines)

    def open(self, path):
        if str(path).lower().endswith(".outb"):
            return _OutbSource(path)
        return _OutSource(path)
