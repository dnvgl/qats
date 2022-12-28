# QATS

Python library and GUI for efficient processing and visualization of time series.

[![Build Status](https://github.com/dnvgl/qats/actions/workflows/test.yml/badge.svg)](https://github.com/dnvgl/qats/actions/workflows/test.yml)
[![PyPi Deployment Status](https://github.com/dnvgl/qats/actions/workflows/publish.yml/badge.svg)](https://github.com/dnvgl/qats/actions/workflows/publish.yml)
[![Documentation Status](https://readthedocs.org/projects/qats/badge/?version=latest)](https://qats.readthedocs.io/en/latest/?badge=latest)

## General

### About

The python library provides tools for:
- Import and export from/to various pre-defined time series file formats
- Signal processing
- Inferring statistical distributions
- Cycle counting using the Rainflow algorithm

It was originally created to handle time series files exported from [SIMO](https://www.dnvgl.com/services/complex-multibody-calculations-simo-2311) 
and [RIFLEX](https://www.dnvgl.com/services/riser-analysis-software-for-marine-riser-systems-riflex-2312). Now it also
handles [SIMA](https://www.dnvgl.com/services/marine-operations-and-mooring-analysis-software-sima-2324) hdf5 (.h5) files, 
Matlab (version < 7.3) .mat files, CSV files and more.  

QATS also features a GUI which offers efficient and low threshold processing and visualization of time series. It is
perfect for inspecting, comparing and reporting:
- time series
- power spectral density distributions
- peak and extreme distributions
- cycle distributions

### Demo

![QATS GUI](https://raw.githubusercontent.com/dnvgl/qats/master/docs/source/demo.gif)

## Python version support

QATS supports Python 3.7, 3.8, 3.9 and 3.10.

## Getting started

### Installation

Run the below command in a Python environment to install the latest QATS release:

```console
python -m pip install qats
```

To upgrade from a previous version, the command is:

```console
python -m pip install --upgrade qats
```

You may now import qats in your own scripts:

```python
from qats import TsDB, TimeSeries
```

... or use the GUI to inspect time series. Note that as of version 4.2.0 you are quite free to choose which 
[Qt](https://www.qt.io) binding you would like to use for the GUI: [PyQt5](https://pypi.org/project/PyQt5/) or 
[PySide2](https://pypi.org/project/PySide2/), or even [PyQt4](https://pypi.org/project/PyQt4/) / 
[PySide](https://pypi.org/project/PySide/).

Install the chosen binding (here PyQt5 as an example):

```console
python -m pip install pyqt5
```

If multiple Qt bindinds are installed, the one to use may be controlled by setting the environmental variable `QT_API` to the desired package. Accepted values include `pyqt5` (to use PyQt5) and `pyside2` (PySide2). For more details, see [README file for qtpy](https://github.com/spyder-ide/qtpy/blob/master/README.md).

The GUI may now be launched by:

```console
qats app
```

To create a start menu link, which you can even pin to the taskbar to ease access to the 
QATS GUI, run the following command:

```console
qats config --link-app
```

Take a look at the resources listed below to learn more.

_New in version 4.11.0._ The command line interface is also accessible by running Python with the '-m' option. The following commands are equvivalent to those above:
```console 
python -m qats app
python -m qats config --link-app
```

### Resources

* [**Source**](https://github.com/dnvgl/qats)
* [**Issues**](https://github.com/dnvgl/qats/issues)
* [**Changelog**](https://github.com/dnvgl/qats/blob/master/CHANGELOG.md)
* [**Documentation**](https://qats.readthedocs.io)
* [**Download**](https://pypi.org/project/qats/)

## Contribute

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Install Python version 3.7 or later from either https://www.python.org or https://www.anaconda.com.

### Clone the source code repository

At the desired location, run: 

```git clone https://github.com/dnvgl/qats.git```

### Installing

To get the development environment running:

... create an isolated Python environment and activate it,

```console
python -m venv /path/to/new/virtual/environment

/path/to/new/virtual/environment/Scripts/activate
```

... install the dev dependencies in [requirements.txt](requirements.txt),

```console
python -m pip install -r requirements.txt
```

.. and install the package in development ("editable") mode.

```console
python -m pip install -e .
```

_Note: This is similar to the "legacy" development installation command ``python setup.py develop``, see the [setuptools page on development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)._

You should now be able to import the package in the Python console,

```python
import qats
help(qats)
```

... and use the command line interface (CLI).

```console
qats -h
```

_New in version 4.11.0._ The CLI is also available from 

```console
python -m qats -h
```

### Running the tests

The automated tests are run using [unittest](https://docs.python.org/3/library/unittest.html/).

```console
python -m unittest discover 
```

### Building the package

Build tarball and wheel distributions by:

```console
python setup.py sdist bdist_wheel
```

The distribution file names adhere to the [PEP 0427](https://www.python.org/dev/peps/pep-0427/#file-name-convention) 
convention `{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl`.

### Building the documentation

The html documentation is built using [Sphinx](http://www.sphinx-doc.org/en/master)

```console
sphinx-build -b html docs\source docs\_build
```

To force a build to read/write all files (always read all files and don't use a saved environment), include the `-a` and `-E` options:

```console
sphinx-build -a -E -b html docs\source docs\_build
```

### Deployment
Packaging, unit testing and deployment to [PyPi](https://pypi.org/project/qats/) is automated using [GitHub Actions](https://docs.github.com/en/actions).

### Versioning

We apply the "major.minor.micro" versioning scheme defined in [PEP 440](https://www.python.org/dev/peps/pep-0440/). See also [Scheme choices](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#scheme-choices) on https://packaging.python.org/.

We cut a new version by applying a Git tag like `3.0.1` at the desired commit and then 
[setuptools_scm](https://github.com/pypa/setuptools_scm/#setup-py-usage) takes care of the rest. For the versions 
available, see the [tags on this repository](https://github.com/dnvgl/qats/tags). 

## Authors

* **Per Voie** - [tovop](https://github.com/tovop)
* **Erling Lone** - [eneelo](https://github.com/eneelo)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
