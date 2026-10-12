"""
Registry of file format readers.

The registry holds the built-in readers and the readers from installed plugin packages, and finds the reader for a
file. It is filled on first use, not on import. A plugin package adds readers through the entry point group
``qats.readers``, with each entry point referring to a `qats.io.base.Reader` subclass, e.g. in ``pyproject.toml``::

    [project.entry-points."qats.readers"]
    myformat = "qats_myformat:MyFormatReader"

A plugin that cannot be loaded is skipped with a logged warning, and never stops QATS from working.

.. versionadded :: 5.5.0
"""

import fnmatch
import importlib
import importlib.metadata
import logging
import os

from .._validation import QatsTypeError, QatsValueError
from .base import READER_API_VERSION, Reader

__all__ = ["ENTRY_POINT_GROUP", "find_reader", "get_reader", "readers", "register"]

ENTRY_POINT_GROUP = "qats.readers"
"""Entry point group where plugin packages register their readers."""

_BUILTIN_READERS = (
    "qats.io.direct_access:TsReader",
    "qats.io.direct_access:TdaReader",
    "qats.io.sima:AsciiReader",
    "qats.io.sima:BinReader",
    "qats.io.other:DatReader",
    "qats.io.sintef_mat:MatReader",
    "qats.io.sima_h5:H5Reader",
    "qats.io.csv:CsvReader",
    "qats.io.pickle_format:PickleReader",
    "qats.io.tdms:TdmsReader",
)

logger = logging.getLogger(__name__)

_readers: dict[str, Reader] = {}  # by name, in registration order
_loaded = False


def _instance(reader):
    """Reader instance from a Reader subclass or instance."""
    if isinstance(reader, type) and issubclass(reader, Reader):
        reader = reader()
    if not isinstance(reader, Reader):
        raise QatsTypeError(f"Expected a Reader subclass or instance, got {reader!r}")
    if not reader.name:
        raise QatsValueError(f"Reader {type(reader).__name__} has no name")
    return reader


def _add(reader, replace=False):
    if reader.name in _readers and not replace:
        raise QatsValueError(
            f"A reader named {reader.name!r} is already registered ({_readers[reader.name]!r}). "
            "Use replace=True to replace it."
        )
    _readers[reader.name] = reader


def _load():
    """Register the built-in readers and the readers from installed plugins, once."""
    global _loaded
    if _loaded:
        return
    _loaded = True  # set first, so that a plugin calling register() does not load again

    for path in _BUILTIN_READERS:
        module, _, cls = path.partition(":")
        _add(_instance(getattr(importlib.import_module(module), cls)), replace=True)

    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            reader = _instance(ep.load())
            if reader.api_version != READER_API_VERSION:
                raise QatsValueError(
                    f"Reader {reader.name!r} ({type(reader).__name__}) is written for reader API version "
                    f"{reader.api_version}, but this QATS supports version {READER_API_VERSION}"
                )
            if reader.name in _readers:
                raise QatsValueError(
                    f"Reader {reader.name!r} ({type(reader).__name__}) has the same name as the registered reader "
                    f"{_readers[reader.name]!r}"
                )
            _add(reader)
        except Exception as err:  # a broken plugin must not break QATS
            dist = getattr(ep, "dist", None)  # the installed package, to uninstall or upgrade
            package = f" from the package {dist.name} {dist.version}" if dist is not None else ""
            logger.warning("Skipped the file reader plugin %r (%s)%s: %s", ep.name, ep.value, package, err)


def register(reader, replace=False):
    """
    Register a file format reader.

    Use this to add a reader in a script. To add a reader from a package, use the entry point group
    ``qats.readers`` instead (see the module documentation).

    Parameters
    ----------
    reader : Reader subclass or instance
        The reader. Its `name` must be unique.
    replace : bool, optional
        Replace a registered reader with the same name. By default this raises an error.

    Returns
    -------
    Reader
        The registered reader instance.

    Raises
    ------
    QatsValueError
        If a reader with the same name is registered and `replace` is False.

    Notes
    -----
    .. versionadded :: 5.5.0
    """
    _load()
    reader = _instance(reader)
    _add(reader, replace=replace)
    return reader


def readers():
    """
    List the registered readers.

    Returns
    -------
    list of Reader
        Readers in registration order: the built-in readers first, then the readers from plugins and scripts.

    Notes
    -----
    .. versionadded :: 5.5.0
    """
    _load()
    return list(_readers.values())


def get_reader(name):
    """
    Get a registered reader by name.

    Parameters
    ----------
    name : str
        Reader name.

    Returns
    -------
    Reader
        The reader.

    Raises
    ------
    QatsValueError
        If no reader has this name.

    Notes
    -----
    .. versionadded :: 5.5.0
    """
    _load()
    try:
        return _readers[name]
    except KeyError:
        raise QatsValueError(f"No reader named {name!r}. Available readers: {', '.join(_readers)}") from None


def _matches(reader, path):
    basename = os.path.basename(path).lower()
    return any(fnmatch.fnmatchcase(basename, pattern.lower()) for pattern in reader.patterns)


def find_reader(path, name=None):
    """
    Find the reader for a file.

    Parameters
    ----------
    path : str
        File path.
    name : str, optional
        Name of the reader to use, regardless of the file name and content.

    Returns
    -------
    Reader
        The reader.

    Raises
    ------
    NotImplementedError
        If no reader can read the file.
    QatsValueError
        If `name` is given and no reader has this name.

    Notes
    -----
    Without `name`, the readers whose file name patterns match the file name (case-insensitively) are tried in
    order of decreasing `priority`, and in registration order for equal priority. The first one whose `can_read`
    accepts the file is returned. A `can_read` that raises an error counts as not accepting the file.

    .. versionadded :: 5.5.0
    """
    if name is not None:
        return get_reader(name)
    _load()
    candidates = sorted((r for r in _readers.values() if _matches(r, path)), key=lambda r: -r.priority)
    for reader in candidates:
        try:
            accepted = reader.can_read(path)
        except Exception as err:
            logger.debug("Reader %r failed to check %s: %s", reader.name, path, err)
            accepted = False
        if accepted:
            return reader
    if candidates:
        declined = ", ".join(r.name for r in candidates)
        raise NotImplementedError(
            f"No reader can read {path} (the readers {declined} match its name, but not its content)"
        )
    raise NotImplementedError(f"Invalid file type, no reader matches the file name: {path}")
