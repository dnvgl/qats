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

    python -m pip install qats

In order to use the GUI, you must also install a Python package with qt bindings (here, `PyQt5` is used as an
example):

.. code-block::

    python -m pip install pyqt5

Supported qt bindings are: PyQt5, Pyside2, PyQt4 and Pyside.

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



Launching the GUI
*****************

The GUI is launched via the CLI:

.. code-block::

    qats app

Or, you may add a shortcut for launching the QATS GUI to your Windows Start menu and on the Desktop by running the command:

.. code-block::

    qats config --link-app

.. note::
    As of version 4.11.0, the CLI is also available through the ``python -m`` switch, for example:

    .. code-block::

        python -m qats -h
        python -m qats app

    To add a Windows Start menu shortcut that utilizes this to launch the GUI without invoking the qats executable
    (i.e., does not call ``qats.exe``), use

    .. code-block::

        python -m qats config --link-app-no-exe


..    :code:`python -m qats config --link-app-no-exe`.


..    :code:`python -m qats -h` or :code:`python -m qats app`.


Your first script
*****************

Import the time series database, load data to it from file and plot it all.

.. literalinclude:: examples/plot.py
   :language: python
   :linenos:
   :lines: 1-17

Take a look at :ref:`examples` and the :ref:`api` to learn how to use QATS and build it into your code.

