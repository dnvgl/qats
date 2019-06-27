# QATS

Python library and GUI for efficient processing and visualization of time series.

[![Build Status](https://travis-ci.com/dnvgl/qats.svg?branch=master)](https://travis-ci.com/dnvgl/qats)
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

### Getting started

Run the below command in a Python environment to install the latest QATS release.

```console
pip install qats
```

Launch the GUI...

```console
qats app
```

and create a start menu link which you can even pin to the taskbar to ease access to the QATS GUI.

```console
qats config --link-app
```

Take a look at the resources listed below to learn more.

### Resources

* [**Source**](https://github.com/dnvgl/qats)
* [**Issues**](https://github.com/dnvgl/qats/issues)
* [**Documentation**](https://qats.readthedocs.io)
* [**Download**](https://pypi.org/project/qats/)

## Contribute

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Install Python version 3.6 or later from either https://www.python.org or https://www.anaconda.com.

### Clone the source code repository

At the desired location run ```git clone https://github.com/dnvgl/qats.git```

### Installing

To get the development environment running

.. create an isolated Python environment and activate it,

```console
python -m venv /path/to/new/virtual/environment

/path/to/new/virtual/environment/Scripts/activate
```

.. install the dev dependencies in [requirements.txt](requirements.txt)

```console
pip install -r requirements.txt
```

.. and install the package in development mode.

```console
python setup.py develop
```

Now you should be able to import the package in the Python console

```python
import qats
help(qats)
```

.. and the command line interface (CLI).

```console
qats -h
```

### Running the tests

The automated tests are run using [Tox](https://tox.readthedocs.io/en/latest/).

```console
tox
```

The test automation is configured in the file `tox.ini`.

### Building the package

Build tarball and wheel distributions by 

```console
python setup.py sdist bdist_wheel
```

The distribution file names adhere to the [PEP 0427](https://www.python.org/dev/peps/pep-0427/#file-name-convention) 
convention `{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl`.

### Building the documentation

The html documentation is build using [Sphinx](http://www.sphinx-doc.org/en/master)

```console
sphinx-build -b html docs\source docs\_build
```

### Deployment
Packaging, unit testing and deployment to [PyPi](https://pypi.org/project/qats/) is automated using 
[Travis-CI](https://travis-ci.com).

### Versioning

We apply the "major.minor.micro" versioning scheme defined in [PEP 440](https://www.python.org/dev/peps/pep-0440/).

We cut a new version by applying a Git tag like `3.0.1` at the desired commit and then 
[setuptools_scm](https://github.com/pypa/setuptools_scm/#setup-py-usage) takes care of the rest. For the versions 
available, see the [tags on this repository](https://github.com/dnvgl/qats/tags). 

## Authors

* **Per Voie** - [tovop](https://github.com/tovop)
* **Erling Lone** - [eneelo](https://github.com/eneelo)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
