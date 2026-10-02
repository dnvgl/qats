# 0004 – Support the officially supported Python versions

- **Status:** Accepted
- **Date:** 2026-10-02
- **Release:** 5.4 (Installable again)

## Context

QATS 5.3.1 requires `python >=3.8.1,<3.13`. It doesn't install on Python 3.13 or 3.14, yet still supports 3.8 and 3.9, which are past end of life. Supporting old versions has a real cost: Python-conditional dependency pins in `pyproject.toml`, old tool versions in CI (Poetry 1.8 for 3.8/3.9), and older type-hint syntax. Supporting new versions late blocks users.

PR #133 proposed a policy: support the Python versions that are officially supported upstream, that is, versions with status *bugfix* or *security* on the [Status of Python versions](https://devguide.python.org/versions/) page. Python 3.10 reaches end of life in October 2026, before 5.4 ships in December 2026.

## Decision

- QATS supports the Python versions that are officially supported upstream at the time of each release.
- 5.4 supports **Python 3.11–3.14**: `requires-python = ">=3.11"`, with no upper bound.
- Support for a new Python version is added in the first QATS release after its final release. Support for an end-of-life version is dropped in the first QATS release after its end-of-life date.
- The docs state the policy and point to the PyPI classifiers/badge for the exact versions, instead of listing version numbers in several places (wording from PR #133).

## Consequences

- Dependency specs get simpler: no Python-conditional pins for numpy, pandas, scipy or PySide6, and no `importlib-resources` backport.
- Code may use Python 3.11 features and modern type hints (`list[int]`, `X | None`).
- Users on Python 3.8–3.10 stay on QATS 5.3.x, which remains installable from PyPI.
- The CI matrix and classifiers must be updated at each Python release (yearly, in October). Add this to the release checklist.
