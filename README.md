# QATS

Python library and GUI for efficient processing and visualization of time series.

[![Python versions](https://img.shields.io/pypi/pyversions/qats)](https://pypi.org/project/qats/)
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

## Getting started

### Installation

QATS requires Python 3.11 or later. Run the below command in a Python environment to install the latest QATS release:

```console
python -m pip install qats
```

To upgrade from a previous version, the command is:

```console
python -m pip install --upgrade qats
```

You may now import qats in your own scripts:

```python
>>> from qats import TsDB, TimeSeries
```

... or use the GUI to inspect time series. 

_New in version 5.0.0._ 
The [Qt](https://www.qt.io) binding [PySide6](https://pypi.org/project/PySide6/) is installed with `qats`. 
If you would rather like to use [PyQt6](https://pypi.org/project/PyQt6/), run

```console
python -m pip install pyqt6
```

If multiple Qt bindinds are installed, the one to use may be controlled by setting the environmental variable `QT_API` to the desired package. Accepted values include `pyqt6` (to use PyQt6) and `pyside6` (PySide6). For more details, see [README file for qtpy](https://github.com/spyder-ide/qtpy/blob/master/README.md).

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
* [**Changelog**](https://github.com/dnvgl/qats/releases)
* [**Documentation**](https://qats.readthedocs.io)
* [**Download**](https://pypi.org/project/qats/)

## Contribute

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) with [the official installer](https://docs.astral.sh/uv/getting-started/installation/).
uv installs a suitable Python version automatically if you don't have one.

### Clone the source code repository

At the desired location, run:

```console
git clone https://github.com/dnvgl/qats.git
```

The package version is derived from Git tags (using [setuptools-scm](https://setuptools-scm.readthedocs.io)), so clone
the full repository with tags rather than downloading a source archive.

### Installing

To get the development environment running, run from the repository root:

```console
uv sync
```

This will
- create an isolated environment in `.venv`
- install the locked dependencies from `uv.lock`, including those required for development, testing and building
  documentation (the `dev` and `docs` dependency groups)
- and install the package in development ("editable") mode

To use a specific Python version, e.g. 3.12, run `uv sync --python 3.12`.

Prefix commands with `uv run` to run them in the environment (or activate `.venv` first). You should now be able to
import the package in the Python console,

```console
uv run python
```

```python
>>> import qats
>>> help(qats)
```

... and use the command line interface (CLI).

```console
uv run qats -h
```

_New in version 4.11.0._ The CLI is also available from

```console
uv run python -m qats -h
```

To add or update a dependency, use `uv add` (e.g. `uv add "numpy>=2"` or `uv add --dev pytest`) rather than editing
`pyproject.toml` by hand, so that `uv.lock` is updated as well. Commit both files.

### Running the tests

The automated tests are run using [pytest](https://docs.pytest.org).

```console
uv run pytest test/
```

### Linting and formatting

Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. CI fails if
either check finds a problem, so run them before pushing:

```console
uv run ruff check .
uv run ruff format --check .
```

To fix what can be fixed automatically and reformat the code:

```console
uv run ruff check --fix .
uv run ruff format .
```

The commit that first formatted the code base with ruff is listed in `.git-blame-ignore-revs`. To have `git blame`
skip it locally (GitHub does this automatically), run once:

```console
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

### Building the package

Build tarball and wheel distributions by

```console
uv build
```

The builds appear in the `dist` folder. The distributions adhere to the [PEP 0427](https://www.python.org/dev/peps/pep-0427/#file-name-convention)
convention `{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl`.

### Building the documentation

The html documentation is built using [Sphinx](http://www.sphinx-doc.org/en/master)

```console
uv run sphinx-build -b html docs/source docs/_build
```

To force a build to read/write all files (always read all files and don't use a saved environment), include the `-a` and `-E` options:

```console
uv run sphinx-build -a -E -b html docs/source docs/_build
```

### Versioning

We apply the "major.minor.micro" versioning scheme defined in [PEP 440](https://www.python.org/dev/peps/pep-0440/). See also [Scheme choices](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#scheme-choices) on https://packaging.python.org/.

Versions are tagged with Git tags like `v3.0.1`. See the [tags on this repository](https://github.com/dnvgl/qats/tags). 

### Release and deployment

We use the [release cycle in GitHub](https://github.com/dnvgl/qats/releases) to cut new releases.

Once a new release is published, [GitHub Actions](https://docs.github.com/en/actions) takes care of the packaging and deployment to [PyPi](https://pypi.org/project/qats/). Unit tests run on every push and pull request.

The workflows for continuous integration and deployment are found in [.github/workflows](.github/workflows/).

## Authors

* **Per Voie** - [tovop](https://github.com/tovop)
* **Erling Lone** - [eneelo](https://github.com/eneelo)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
