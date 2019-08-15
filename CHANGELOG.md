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

#### Added
- Class method `TsDB.getl()` ("get list").
- New module `motions.py`, for transformations and operations related to motion time series.
- Changelog (this file)

#### Changed
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

[Unreleased]: https://github.com/dnvgl/qats/compare/3.0.6...HEAD
[3.0.6]: https://github.com/dnvgl/qats/compare/3.0.5...3.0.6
[3.0.5]: https://github.com/dnvgl/qats/compare/3.0.0...3.0.5
