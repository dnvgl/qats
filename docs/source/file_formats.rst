.. _file_formats:

Writing files QATS can read
###########################

If your software writes a format QATS doesn't read, you can often still use QATS by writing the time series to a
plain text file: either a ``.dat`` file with columns separated by spaces or tabs, or a ``.csv`` file. This page
describes exactly what QATS expects of these two formats. Both can be written by a script, exported from a
spreadsheet or edited in a text editor.

Both formats hold several time series that share one time array: time in the first column and one column per series.
If your series have different time arrays, write them to separate files, or use Python as described in
:ref:`file_formats_python`.

Both formats are read by file extension: name the files ``*.dat`` or ``*.csv``.


Column-wise text file (``.dat``)
********************************

.. literalinclude:: formats/example.dat
   :caption: example.dat

Rules:

* **Comments:** lines starting with ``#`` are ignored, anywhere in the file.
* **Header:** the first line that is not a comment holds the names. It must not be blank.
* **Names:** separated by spaces or tabs, so a name **cannot contain spaces**. Write ``Wave_elevation``, not
  ``Wave elevation``. Other characters are kept as part of the name, e.g. ``Heave[m]``.
* **Time:** must be the **first** column, and its name must start with "time" in any case, e.g. ``Time``, ``time`` or
  ``TIME[s]``. No other column name may start with ``Time`` or ``time`` (``Timestamp`` would count as a second time
  column and the file is rejected).
* **Values:** separated by spaces or tabs, any number of each. Every line must have one value per name.
* **Numbers:** decimal point, not decimal comma. Scientific notation (``1.5e-3``) and ``nan`` are accepted.
* **Line endings** may be Windows (CRLF) or Unix (LF).


Comma-separated values (``.csv``)
*********************************

.. literalinclude:: formats/example.csv
   :caption: example.csv

Rules:

* **Header:** the first line that doesn't start with ``#`` holds the names. Lines starting with ``#`` before it are
  skipped.
* **Separator:** comma, semicolon, tab or space, detected from the header line. Other separators (e.g. ``|``) are not
  supported.
* **Columns:** time and at least one series. A file with only a time column is an error.
* **Names:** may contain spaces. A name containing the separator must be in double quotes, e.g. ``"surge, x"``.
* **Time:** the first column, with any name.
* **Values:** an empty cell is read as ``nan``. A separator at the end of each line is allowed, but a line must not
  have more values than there are names.
* **Numbers:** decimal point. With ``;`` as separator, a decimal comma (e.g. ``1,5``) also works. Scientific notation
  is accepted.
* **Encoding:** UTF-8, with or without a byte-order mark, or Windows-1252 (Western European).

From Excel, both "CSV UTF-8 (comma delimited)" and the plain "CSV (comma delimited)" work, also with regional settings
that write decimal commas, such as Norwegian:

.. literalinclude:: formats/example_excel.csv
   :caption: example_excel.csv (saved by Excel with Norwegian regional settings)
   :encoding: cp1252


Rules for both formats
**********************

* **Time** should increase from row to row. This is not checked when the file is read.
* **Units** are not read from these formats. A unit written in the name, e.g. ``Heave[m]``, is kept as part of the
  name.
* **Slashes in names:** QATS treats ``/`` and ``\`` in a name as separators between a group and a series name, like
  folders, unless they are inside square brackets. ``Acc[m/s^2]`` is kept as written, while ``Acc/x`` becomes the
  series ``x`` in the group ``Acc``.


Common errors
*************

=======================================================================  ===================================================
Error                                                                    Likely cause
=======================================================================  ===================================================
``KeyError: The file ... does not contain a time vector`` (``.dat``)     No name starts with "time", or there's a blank line
                                                                         before the header
``KeyError: The time column ... must be the first column`` (``.dat``)    Move the time column to the first column
``KeyError: The file ... contains duplicate time vectors`` (``.dat``)    More than one name starts with ``Time``/``time``
``ValueError: invalid column index ...`` (``.dat``)                      A name contains a space, or a line has fewer
                                                                         values than names
``QatsValueError: ... no comma, semicolon, tab ...`` (``.csv``)          Only a time column, or another separator, e.g.
                                                                         ``|``
``QatsValueError: ... not a number in column ...`` (``.csv``)            Text in a value column, a decimal comma with
                                                                         another separator than ``;``, or a line before
                                                                         the header that doesn't start with ``#``
=======================================================================  ===================================================


.. _file_formats_python:

From Python
***********

If you can run Python, there's no need to go through a text file: create :class:`~qats.TimeSeries` objects from your
arrays, add them to a :class:`~qats.TsDB` and export it to a format QATS reads, for use in the GUI or later. This also
works for series with different time arrays.

.. code-block:: python

   import numpy as np
   from qats import TimeSeries, TsDB

   t = np.arange(0.0, 100.0, 0.05)
   db = TsDB()
   db.add(TimeSeries("Heave", t, 0.1 * np.exp(-0.05 * t) * np.cos(t), unit="m"))
   db.add(TimeSeries("Pitch", t, 1.5 * np.exp(-0.05 * t) * np.cos(t), unit="deg"))
   db.export("decay.ts")  # SIMO/RIFLEX direct access format; .dat, .h5 and .pkl also work

If you have many files in another format, a reader for that format saves the conversion. Open an
`issue <https://github.com/dnvgl/qats/issues>`_ to suggest it, or contribute one.
