.. _getting_started:

Getting started
###############

Prerequisites
*************

You need Python, which may be installed from for instance https://www.python.org or https://www.anaconda.com.
QATS 5.4 supports Python 3.11, 3.12, 3.13 and 3.14; see :ref:`python_version_support`.
On Python 3.8, 3.9 or 3.10, stay on QATS 5.3, which ``pip`` selects automatically.

Installation
************

QATS is installed from PyPI by using `pip`:

.. code-block:: console

    $ python -m pip install qats

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

and run the command line interface (CLI).

.. code-block:: console

    $ qats -h

    usage: qats [-h] [--version] {app,config} ...

    QATS is a library and desktop application for time series analysis

    optional arguments:
      -h, --help    show this help message and exit
      --version     Package version

    Commands:
      {app,config}
        app         Launch the desktop application
        config      Configure the package


.. note::
    The GUI uses the `Qt <https://www.qt.io>`_ binding `PySide6 <https://pypi.org/project/PySide6/>`_, which is
    installed with qats and is the binding QATS is tested with. `PyQt6 <https://pypi.org/project/PyQt6/>`_ also works,
    through `qtpy <https://github.com/spyder-ide/qtpy>`_, but is supported on a best-effort basis only. To use it,
    install it (``python -m pip install pyqt6``) and set the environment variable :code:`QT_API` to :code:`pyqt6`.
    See the `qtpy README <https://github.com/spyder-ide/qtpy/blob/master/README.md>`_ for details.

.. note::
    As of version 4.11.0, the CLI is also available through the ``python -m`` switch, for example:

    .. code-block::

        $ python -m qats -h
        $ python -m qats app

..        $ python -m qats config --link-app


..    :code:`python -m qats config --link-app-no-exe`.


..    :code:`python -m qats -h` or :code:`python -m qats app`.



Launching the GUI
*****************

The GUI is launched via the CLI:

.. code-block::

    $ qats app

If using qats on **Windows**, you may add a shortcut for launching the qats GUI to your Windows Start menu and on the Desktop by running the command:

.. code-block::

    C:\> qats config --link-app


Your first script
*****************

Import the time series database, load data to it from file and plot it all.

.. literalinclude:: examples/plot.py
   :language: python
   :linenos:
   :lines: 1-17

Take a look at :ref:`examples` and the :ref:`api` to learn how to use :code:`qats` and build it into your code.

