# QATS

Python library and Qt GUI for processing and visualising time series from marine and offshore simulations (SIMO, RIFLEX, SIMA and others): file I/O, signal processing, extreme-value statistics, rainflow counting and fatigue. The users are experienced engineers; correctness of numerical methods comes first.

Plan and rationale: [docs/roadmap.md](docs/roadmap.md) and [docs/decisions/](docs/decisions/). Read the relevant decision record before changing anything it covers.

## Layout

- `qats/ts.py` – `TimeSeries`: one series, with statistics, filtering, PSD, rainflow counting and plotting
- `qats/tsdb.py` – `TsDB`: a database of series loaded from files. Reader dispatch is currently an `if fext == ...` chain in both `TsDB.load` (registering keys) and `TsDB._read` (reading data); a new format must be added in both places until the 5.5 reader registry replaces them.
- `qats/io/` – one module per file format (SIMO/RIFLEX direct-access, SIMA `.h5`, Matlab, CSV, TDMS, pickle, ...)
- `qats/signal.py` – filters, reversals, maxima, PSD, CSD, coherence, transfer function (`tfe`)
- `qats/stats/` – Weibull, Gumbel, Gumbel-min, empirical distributions
- `qats/fatigue/` – `rainflow.py` (ASTM E1049-85), `sn.py` (`SNCurve`, Miner sums), `corrections.py`
- `qats/app/` – Qt GUI (`gui.py`), via qtpy; `qats/cli.py` – the `qats` command
- `test/` – unittest-style tests run with pytest, using data files from `data/`
- `docs/source/` – Sphinx docs (furo theme, myst), examples in `docs/source/examples/`

## Commands

Today the project uses Poetry with `poetry-dynamic-versioning`. Release 5.4 moves to PEP 621 + uv with setuptools-scm (PR #135) and then to ruff; update this section in those PRs.

- Install for development: `poetry install`
- Run tests: `pytest test/`
- Run one test file: `pytest test/test_rainflow.py`
- Check the CLI: `qats -h`, `qats app -h`, `python -m qats -h`
- Build docs: `sphinx-build -b html docs/source docs/_build`
- Build the package: `poetry build`

## Conventions

- Follow the style of the surrounding code. Lines up to about 120 characters are common.
- Docstrings use the NumPy format (Parameters / Returns / Notes / Examples). Mark new public API with `.. versionadded :: X.Y.Z` and changed behaviour with `.. versionchanged :: X.Y.Z`.
- Type hints must work on the oldest supported Python. Until 5.4 that is 3.8 (`Tuple`, `Optional` from `typing`); from 5.4 the minimum is 3.11, so use `list[int]` and `X | None` ([0004](docs/decisions/0004-python-version-support-policy.md)).
- The version comes from git tags. Never edit `qats/_version.py` by hand.
- Release notes go in GitHub Releases; `CHANGELOG.md` is frozen.

## Rules

- **API stability:** 5.x releases are additive only. Anything that would break user code goes into 6.0, behind a `DeprecationWarning` in 5.x first.
- **Numerical changes need a reference test:** a change to statistics, rainflow counting or fatigue must include a test against a known result (a published value, a hand calculation or the previous implementation's output).
- **Qt:** import Qt only through `qtpy`, never `PySide6` or `PyQt6` directly. CI runs with PySide6; PyQt6 is best effort ([0001](docs/decisions/0001-pyside6-tested-pyqt6-best-effort.md)).
- **Compiled code:** Numba is an optional extra (`qats[fast]`) with a pure-Python fallback that gives identical results; QATS must import and work without Numba. No Rust or C extensions without a new decision record ([0002](docs/decisions/0002-numba-before-rust.md)).
- **S-N curves:** built-in DNV-RP-C203 presets carry the edition they come from and are tested against the standard's tables ([0003](docs/decisions/0003-include-c203-sn-curves.md)).
- **Decisions:** when a PR settles a design question, add a short record to `docs/decisions/` (next number, same structure) and update this file if it adds a standing rule.

## Workflow

- One GitHub issue per piece of work, assigned to a release milestone; the roadmap's "Done when" line is the acceptance criterion.
- Work on a branch, open a PR against `master`; CI (lint, tests, CLI, package and docs build) must pass.
- Start larger tasks in plan mode and get the plan reviewed before implementing.
