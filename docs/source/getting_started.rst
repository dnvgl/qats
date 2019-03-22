.. _getting_started:

Getting started
###############

Quick-start
***********

If you are familiar with Anaconda, conda packages and conda virtual environments, you may use this quick-start guide.
Otherwise, proceed to the more thorough explanation in next subsections.

Assuming you have conda installed, create a virtual Python 3.6 environment and install QATS like this

.. code-block:: console

    conda create -y -n py36 python=3.6
    conda install -y -c <channel name> -n py36 qats

You should now be able to import the package and use it.

.. code-block:: python

    >>> import qats
    >>> help(qats)


About Conda and Anaconda
************************

Today the preferred way of installing Python is through installing `Anaconda <https://www.anaconda.com/download/>`_.
Anaconda. Just select which python version you would like to have in your root installation.

Conda is the basic package manager and Anaconda is Conda + many common packages. Conda helps you get the right packages
and compatible versions.

QATS is distributed as a conda package in the same way as `numpy`, `scipy` etc., through a conda-channel. A
conda-channel is more or less a location on the web or on the network serving python packages, and you can subscribe to
the channel and install the python package. Run the command below to see which channels (URLs) you have by default.

.. code-block:: console

    conda info


Create a virtual Python 3 environment
*************************************

QATS runs on Python 3 so if your root installation is Python 2, you need to create a virtual environment on your pc.
Open the anaconda prompt and create an environment called "py36" (or whatever).

.. code-block:: console

    conda create -n py36 python=3.6

With virtual environments you may have several different versions of the same software e.g. Python installed on the
computer at the same time. You just activate the relevant environment when needed. Run the command below to see which
envs you have at the moment (root is the default environment created at anaconda installation).

.. code-block:: console

    conda info -e

Activate the "py36" environment with the activate command.

.. code-block:: console

    activate py36


Install QATS
************

As conda-package
================

QATS is hosted as conda package on the '<channel name>' channel. Run the command below to add the channel to your
channel URL list.

.. code-block:: console

    conda config --append channels <channel name>

If you run the command :code:`conda info` again to verify that the new channel is appended to the channel URL list. You
can use :code:`--prepend` instead of :code:`--append` to give the channel the highest priority. Priority meaning that
a single package is available on several channels, conda will install from the channel with highest priority.

Now you should find the available versions of QATS by running the command below.

.. code-block:: console

    conda search qats

To install the latest version you run the command below.

.. code-block:: console

    conda install qats

If you did not add the conda channel to your channel list you can directly install the latest version of QATS from the
channel using the command below.

.. code-block:: console

    conda install -c <channel name> qats

To updated qats to the latest available version, used:

.. code-block:: console

    conda update qats

Take a look at the `cheat-sheet <http://conda.pydata.org/docs/_downloads/conda-cheatsheet.pdf>`_ to learn more conda
commands.


From wheel file
===============

If a developer share a wheel (.whl file) with you, typically for testing a "hot-of-the-shelf" version, you can install
it using pip:

.. code-block:: console

    pip install qats-<x.y.z>_.whl

And uninstall it using pip:

.. code-block:: console

    pip uninstall qats


Use QATS
********

Verify the installation
=======================

To verify that things work and that you got the right version open a python session, import qats and check version number
and content.

.. code-block:: python

    >>> import qats
    >>> help(qats)


Create your first script
========================

Import the time series database, load data to it from file and plot it all.

.. literalinclude:: examples\plot.py
   :language: python
   :linenos:
   :lines: 1-17


Next steps
**********

Use the :ref:`gui` to quickly inspect and analyse time series and prepare plots for reporting.

See :ref:`examples` for more examples on how to invoke QATS in your own scripts to do more advance operations.

:ref:`api` provide information on the content of the QATS package.
