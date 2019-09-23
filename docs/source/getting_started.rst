.. _getting_started:

Getting started
###############

Prerequisites
*************

You need Python version 3.6 or later. You can find it at https://www.python.org or https://www.anaconda.com.

Installation
************

.. note::
    As of version 4.2.0, you must install the desired qt binding yourself (needed for the GUI to work).
    Supported packages are: PyQt5, Pyside2, PyQt4 and Pyside. See installation instructions below.

QATS is installed from PyPI by using `pip`:

.. code-block:: console

    pip install qats

In order to use the GUI, you must also install a Python package with qt bindings (here, `PyQt5` is used as an
example):

.. code-block::

    pip install pyqt5

Supported qt bindings are: PyQt5, Pyside2, PyQt4 and Pyside.

.. warning::
    If you install QATS in a Conda environment make sure that the conda-package `pyqt` is not installed in that
    same environment as that will conflict with `PyQt5` installed from PyPI. `PyQt5` is a dependency of
    QATS.

Now you should be able to import the package in the Python console

.. code-block:: python

    >>> import qats
    >>> help(qats)

    Help on package qats:

    NAME
        qats - Library for efficient processing and visualization of time series.

    PACKAGE CONTENTS
        app (package)
        cli
        fatigue
        gumbel
        gumbelmin
        rainflow
        readers (package)
        signal
        stats
        ts
        tsdb
        weibull
    ...
    ...
    >>>

and the command line interface (CLI).

.. code-block:: console

    qats -h

    usage: qats [-h] [--version] {app,config} ...

    QATS is a library and desktop application for time series analysis

    optional arguments:
      -h, --help    show this help message and exit
      --version     Package version

    Commands:
      {app,config}
        app         Launch the desktop application
        config      Configure the package

You can also add shortcuts for the QATS GUI to your start menu and desktop.

.. code-block::

    qats config --link-app

Your first script
*****************

Import the time series database, load data to it from file and plot it all.

.. literalinclude:: examples/plot.py
   :language: python
   :linenos:
   :lines: 1-17

Take a look at :ref:`examples` and the :ref:`api` to learn how to use QATS and build it into your code.

