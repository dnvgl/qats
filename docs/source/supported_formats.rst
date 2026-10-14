.. _supported_formats:

Supported file formats
######################

QATS reads the formats below, in scripts (:meth:`qats.TsDB.load`, :meth:`qats.TsDB.fromfile`) and in the GUI. The
reader is chosen from the file name, and for some formats also from the content. Formats added by installed reader
plugins work in the same way (see :ref:`reader_plugins`).

.. qats-readers::

To see the formats your installation reads, including those from plugins, run:

.. code-block:: console

    $ qats formats


Choosing a reader
*****************

For a file whose name doesn't tell its format, give the reader name:

.. code-block:: python

    from qats import TsDB

    db = TsDB.fromfile("results.txt", reader="csv")


Notes on each format
********************

.. _format-direct-access-ts:

Direct access time series (``direct-access-ts``)
================================================

SIMO and RIFLEX direct access files (``.ts``). The series names are read from the key file with the same name and the
extension ``.key``, in the same folder. If the file holds fewer complete series than the key file lists (e.g. a
simulation that stopped early), the incomplete ones are skipped with a warning.

.. _format-direct-access-tda:

Direct access time series with info array (``direct-access-tda``)
=================================================================

SIMO S2X direct access files with an info array (``.tda``). The series names are read from the file with the same name
and the extension ``.txt``.

.. _format-sima-ascii:

SIMA/RIFLEX ASCII results (``sima-ascii``)
==========================================

Results written by SIMA/RIFLEX as text (``.asc``), with a key file ``key_<name>.txt`` in the same folder. The series
names are built from the key file: line, segment, node or element, and degree of freedom, e.g.
``Riser_Seg001_No001_Xd``. The type of result is the last part of the key file name, e.g. ``noddis`` for nodal
displacements.

.. _format-sima-bin:

SIMA/RIFLEX binary results (``sima-bin``)
=========================================

As the ASCII results, in binary (``.bin``), with the same key files. Wind turbine results (``*witurb.bin``,
``*blresp.bin``) have their own key file layout, which QATS recognises from the file name.

.. _format-dat:

Column-wise ASCII time series (``dat``)
=======================================

Text files with time and one column per series (``.dat``). See :ref:`file_formats` for the exact rules.

.. _format-matlab:

MATLAB (``matlab``)
===================

MATLAB files (``.mat``, versions 4, 6, 7 and 7.3) in the SINTEF Ocean test data exchange format. Units are read when
the file holds them (``chan_units``).

.. _format-sima-h5:

SIMA HDF5 (``sima-h5``)
=======================

HDF5 files exported from SIMA (``.h5``, ``.hdf5``). Each dataset with SIMA's time attributes (``start`` and ``delta``)
is a series, named by its groups and dataset name, e.g. ``RiserLine\segment_1\node_1\Displacement in x - direction``. HDF5 files
without such datasets, e.g. MATLAB 7.3 files, are not taken for SIMA files.

.. _format-csv:

Comma-separated values (``csv``)
================================

Text files with a header line of names and time in the first column (``.csv``), also as saved by Excel. See
:ref:`file_formats` for the exact rules.

.. _format-pickle:

Pickled pandas DataFrame (``pickle``)
=====================================

A pandas ``DataFrame`` saved with ``to_pickle`` (``.pkl``, ``.pickle``), with time as the index and one column per
series. Column names with several levels are joined with spaces. Only open pickle files you trust: loading a pickle
file can run code.

.. _format-tdms:

National Instruments TDMS (``tdms``)
====================================

TDMS files, typically written by LabVIEW (``.tdms``). Each channel is a series, named by its group and channel, e.g.
``Wagon 55\Wagon Master Position``. Time is taken from the channel's waveform properties, or from a channel named ``time`` in the
same group.

.. _format-openfast:

OpenFAST output (``openfast``)
==============================

OpenFAST text (``.out``) and binary (``.outb``) output files, also those written by the module drivers, with the
channel units. Other tools also write ``.out`` files, so QATS checks the header lines before taking a ``.out`` file
for OpenFAST output. Binary files usually store each channel as 16-bit integers scaled to the channel's range, so
their values are accurate to about 1/65535 of that range. See the example in :ref:`examples`.

.. _format-bladed:

Bladed output (``bladed``)
==========================

Bladed writes each output group as a header file (``.%NN``, e.g. ``powprod.%04``) and a data file (``.$NN``) named in
the header: open the header file. Groups whose last axis is time are read; others (e.g. power curves or modes) give an
error that says why. A series is named by its group and variable, e.g. ``Control variables\Demanded power``. Groups
with a second axis give one series per variable and point on that axis, e.g.
``Blade 1 Mx (Principal elastic axes) @ 16.0741 m``. Units are given in SI. See the example in :ref:`examples`.


Other formats
*************

For a format QATS doesn't read, you can:

* write the results as ``.dat`` or ``.csv`` files, following :ref:`file_formats`;
* add a reader, in a script or as a plugin package that works in every script and in the GUI, following
  :ref:`reader_plugins`;
* read the results with the simulation tool's Python interface and add them to a :class:`~qats.TsDB` as
  :class:`~qats.TimeSeries`, as in the examples.
