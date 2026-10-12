"""
Base classes for file format readers.

A reader describes a file format: its name, the file name patterns it handles and, optionally, a check of the file
content. Opening a file with a reader gives a `SourceFile`, which lists the series on the file and reads the
requested ones as plain arrays. Readers are found through the registry in `qats.io.registry`, which also loads
readers from installed plugin packages.

.. versionadded :: 5.5.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

READER_API_VERSION = 1
"""Version of the reader interface defined in this module. Plugins with another version are not loaded."""


@dataclass(frozen=True)
class SeriesInfo:
    """
    A series on a file, as listed by `SourceFile.series`, without its data.

    Parameters
    ----------
    name : str
        Series name, unique within the file.
    unit : str, optional
        Unit of the series values, if the file holds it.

    Notes
    -----
    .. versionadded :: 5.5.0
    """

    name: str
    unit: str | None = None


@dataclass
class SeriesData:
    """
    One series read from a file, as returned by `SourceFile.read`.

    Parameters
    ----------
    name : str
        Series name.
    t : array
        Time.
    x : array
        Values, same size as `t`.
    unit : str, optional
        Unit of the values, if the file holds it.
    meta : dict, optional
        Other information about the series from the file.

    Notes
    -----
    .. versionadded :: 5.5.0
    """

    name: str
    t: np.ndarray
    x: np.ndarray
    unit: str | None = None
    meta: dict = field(default_factory=dict)


class SourceFile(ABC):
    """
    An opened file: lists its series and reads the requested ones.

    Parameters
    ----------
    path : str
        File path.

    Notes
    -----
    A source file may keep information about the file that makes reading faster, such as the names and positions
    of the series, but never the series data: `read` returns the arrays to the caller and keeps no reference to them,
    so that a caller can read series when needed and drop them afterwards. It must not keep the file open between
    calls, since an open file is locked on Windows.

    .. versionadded :: 5.5.0
    """

    def __init__(self, path):
        self.path = path

    @abstractmethod
    def series(self) -> list[SeriesInfo]:
        """
        List the series on the file, in file order, without reading their data.

        Returns
        -------
        list of SeriesInfo
            Series on the file.
        """

    @abstractmethod
    def read(self, names: list[str]) -> list[SeriesData]:
        """
        Read series from the file.

        Parameters
        ----------
        names : list of str
            Names of the series to read, as listed by `series`.

        Returns
        -------
        list of SeriesData
            The requested series, in the requested order.
        """

    def __repr__(self):
        return f"{type(self).__name__}({self.path!r})"


class Reader:
    """
    A file format reader.

    Subclasses set the class attributes and implement `open`, and may override `can_read`.

    Attributes
    ----------
    name : str
        Unique name of the reader, e.g. ``"csv"``. Used to choose a reader explicitly.
    description : str
        Short description of the format, e.g. for file dialogs.
    patterns : tuple of str
        File name patterns (shell-style wildcards, e.g. ``"*.csv"``), matched case-insensitively.
    priority : int
        Readers with higher priority are tried first when several match a file. The built-in readers have 0.
    api_version : int
        Version of the reader interface the reader is written for, see `READER_API_VERSION`.

    Notes
    -----
    .. versionadded :: 5.5.0
    """

    name: str = ""
    description: str = ""
    patterns: tuple[str, ...] = ()
    priority: int = 0
    api_version: int = READER_API_VERSION

    def can_read(self, path) -> bool:
        """
        Check the file content, for formats whose file name patterns are shared with other formats.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        bool
            True if the reader can read the file. The default accepts any file matching `patterns`.
        """
        return True

    def open(self, path) -> SourceFile:
        """
        Open a file.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        SourceFile
            The opened file.
        """
        raise NotImplementedError(f"{type(self).__name__}.open() is not implemented")

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}: {', '.join(self.patterns)}>"


class _RowSource(SourceFile):
    """
    Source for formats read as a 2-D array with time in row 0 and the series in file order in the following rows,
    where a subset is read by passing row positions (``ind``) to the format's data function.

    Subclasses implement `_read_names` and `_read_rows`.
    """

    def __init__(self, path):
        super().__init__(path)
        self._names = None
        self._positions = None

    def _read_names(self):
        """Series names in file order."""
        raise NotImplementedError

    def _read_rows(self, ind):
        """2-D array with the rows at positions `ind` (0 is time), in the order of `ind`."""
        raise NotImplementedError

    def _scan(self):
        if self._names is None:
            self._names = list(self._read_names())
            # the last of duplicate names wins, as in TsDB.load()
            self._positions = {name: j + 1 for j, name in enumerate(self._names)}

    def series(self):
        self._scan()
        return [SeriesInfo(name) for name in self._names]

    def read(self, names):
        self._scan()
        unique = list(dict.fromkeys(names))  # the format functions need distinct positions
        rows = self._read_rows([0] + [self._positions[name] for name in unique])
        row = {name: i + 1 for i, name in enumerate(unique)}
        return [SeriesData(name, rows[0, :], rows[row[name], :]) for name in names]

    def _legacy_index(self, name):
        """Position of the series on file, as stored in `TsDB.register_indices` before 5.5.0."""
        self._scan()
        return self._positions[name]
