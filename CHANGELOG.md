## Changelog
All notable changes to the project will be documented in this file, with format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

We apply the *"major.minor.micro"* versioning scheme defined in [PEP 440](https://www.python.org/dev/peps/pep-0440).


<!-- Notes:
- Header links are defined at the bottom of this markdown file.
- Versions are level 3 headings (###) (no difference between major/minor/micro versions)
- Change categories are level 4 headings (####).
- The following change categories should be used (and they should be grouped): 
  Added, Changed, Deprecated, Removed, Fixed.
  For details and guidance, see: https://keepachangelog.com/en/1.0.0/#how
-->


### [Unreleased]

Click link to see all [unreleased] changes to the master branch of the repository. 
For comparison to specific branches, use the [GitHub compare](https://github.com/dnvgl/qats/compare) page.


### [4.0.1] // 23.08.2019

#### Fixed
- SIMA .h5 reader bug that occured if numpy 1.16 is used.
- `qats.cli.launch_app()` did not connect `sys.excepthook` with custom error traceback dialogue.

### [4.0.0] // 22.08.2019

This release is not backwards compatible. Main updates are related to fatigue calculation capabilities.

#### Added
- New sub-package `qats.fatigue` for fatigue-related modules. 
  Modules in this sub-package: `corrections` (new), `rainflow` (moved), `sncalc` (new)
- New sub-package `qats.stats` for modules with statistical distributions. 
  Modules in this sub-package: `empirical` (moved + new name), `gumbel` (moved), `gumbelmin` (moved), `weibull` (moved)
- Rainflow counting algorithm now also extracts mean value of cycles (see `qats.fatigue.rainflow.cycles()` and `.count_cycles()`).
- Class method `TsDB.getl()` ("get list").
- New module `motions.py`, for transformations and operations related to motion time series.
- Changelog (this file)

#### Changed
- Sub-module `rainflow` moved into sub-package `qats.fatigue` => `qats.fatigue.rainflow`
- Sub-modules `gumbel`, `gumbelmin` and `weibull` moved into sub-package `qats.stats` => `qats.fatigue.`
- Sub-module `stats` renamed to `empirical` and moved into sub-package `qats.stats` => `qats.stats.empirical`
- Documentation updated with new API reference structure.
- Changed `TsDB.getd()` method ("get dict") to return dict of `TimeSeries` objects, which is identical to `TsDB.getm()`. 
  For previous functionality of `.getd()`, use `TsDB.getda()` ("get dict of arrays").

#### Fixed
- Wildcard prepending to names specified to `TsDB.list()` and `TsDB.get*()` methods. 
  See [this summary](https://github.com/dnvgl/qats/pull/28#issue-296526900) for details.

### [3.0.6] // 2019-06-27
#### Fixed
- Issues related to proper deployment to PyPI and Read the Docs.
- Bug when using band-stop filter in GUI.


### [3.0.5] // 2019-06-26
First proper release to [PyPI](https://pypi.org/project/qats/).



<!-- Links to be defined below here -->

[Unreleased]: https://github.com/dnvgl/qats/compare/4.0.1...HEAD
[4.0.1]: https://github.com/dnvgl/qats/compare/4.0.0...4.0.1
[4.0.0]: https://github.com/dnvgl/qats/compare/3.0.6...4.0.0
[3.0.6]: https://github.com/dnvgl/qats/compare/3.0.5...3.0.6
[3.0.5]: https://github.com/dnvgl/qats/compare/3.0.0...3.0.5
