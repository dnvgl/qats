.. _reader_plugins:

Writing a reader plugin
#######################

A reader teaches QATS a file format. Put it in a small Python package that declares it as a ``qats.readers`` entry
point, and once the package is installed, QATS uses the reader in every script and in the GUI: the file open dialogue
lists the format, and dropped files are read with it. For a one-off format, you can also register a reader in a
script, without a package (see the end of this page).

.. versionadded:: 5.5.0


The reader interface
********************

A reader has two parts, both in :mod:`qats.io.base`:

* A :class:`~qats.io.base.Reader` describes the format:

  * ``name``: a unique name, e.g. ``"example-table"``, used to choose the reader explicitly
    (``TsDB.load(path, reader="example-table")``).
  * ``description``: a short description, shown in the GUI's file dialogue and by ``qats formats``.
  * ``patterns``: file name patterns, e.g. ``("*.etab",)``, matched case-insensitively.
  * ``can_read(path)`` (optional): checks the file content, for formats whose file names other tools also use. Return
    ``False`` for files the reader can't read; QATS then tries other readers.
  * ``open(path)``: returns a :class:`~qats.io.base.SourceFile` for the file.
  * ``priority`` (optional, default 0): readers with higher priority are tried first when several match a file.

* A :class:`~qats.io.base.SourceFile` is an opened file:

  * ``series()`` lists the series on the file, in file order, as :class:`~qats.io.base.SeriesInfo` (name, unit and
    group), without reading their data.
  * ``read(series)`` reads the requested series, given as the ``SeriesInfo`` objects that ``series()`` listed, and
    returns :class:`~qats.io.base.SeriesData` (name, time, values, unit and group) in the requested order. A series
    may be requested more than once.

Some rules make a reader work well with QATS and the GUI:

* **Names and groups:** ``name`` is the series name without its groups. For formats with groups, e.g. HDF5, give
  the groups from the top of the file as ``group``, a tuple of strings; ``TsDB`` names the series by its groups and
  name joined with a backslash. Group and name together must be unique within the file.
* **Units:** give the unit of the values, if the file holds it, as ``unit``; otherwise ``None``.
* **No data kept, no file left open:** a source file may keep names and positions, but never the series data, and
  must close the file before returning, since an open file is locked on Windows.
* **Clear errors:** raise a ``ValueError`` (or ``TypeError``) whose message names the file and what is wrong.


An example plugin package
*************************

A complete plugin for a text table with the series names on the first line, starting with time, the units on the
second line, and then one column per series:

.. code-block:: text

    time surge pitch
    s    m     deg
    0.0  0.00  0.50
    0.5  0.96  0.44

The package has two files:

.. code-block:: text

    qats-example-reader/
        pyproject.toml
        qats_example_reader/
            __init__.py

``pyproject.toml`` declares the reader as an entry point in the group ``qats.readers``:

.. literalinclude:: ../../test/plugins/qats-example-reader/pyproject.toml
   :language: toml

``qats_example_reader/__init__.py`` holds the reader:

.. literalinclude:: ../../test/plugins/qats-example-reader/qats_example_reader/__init__.py
   :language: python
   :linenos:

This package is part of the QATS tests: it is installed and used to read a file on every change to QATS.


Installing and checking a plugin
********************************

Install the package in the same environment as QATS, e.g. from its folder:

.. code-block:: console

    $ pip install ./qats-example-reader

Then check that QATS finds the reader:

.. code-block:: console

    $ qats formats
    File formats (reader name, file names, description):
      direct-access-ts   *.ts             Direct access time series (SIMO, RIFLEX)
      ...
      example-table      *.etab           Example text table with names and units (plugin package qats-example-reader 1.0.0)

Files matching the reader's patterns now open in scripts and in the GUI:

.. code-block:: python

    from qats import TsDB

    db = TsDB.fromfile("motions.etab")
    pitch = db.get(name="pitch")
    print(pitch.unit)  # deg

A plugin that can't be loaded, e.g. because it fails to import, is written for another version of the reader
interface or has the same name as another reader, is skipped with a warning naming the package, and never stops QATS
from working.


Registering a reader in a script
********************************

For a format used in one script only, register the reader in the script with :func:`qats.io.registry.register`
instead of making a package. See *Choose a file reader and add your own* in :ref:`examples`.
