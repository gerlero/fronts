# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.8] - Unreleased

### Added

- Add new automatic mode to `solve()`. Removes the need for users to tune the `dS_dob_bracket` parameter until the function succeeds. In practice, `solve()` can now be expected to return the solution to a problem upon the first call with no parameter tuning required.
- Add optional `dS_dob_hint` parameter to `solve()`. Allows users to pass an optional hint to the new automatic mode, which may accelerate convergence in some scenarios.
- Add `__version__` attribute to the main package, which stores the current version as a string.
- Support extra options during install: e.g., ```$ pip install fronts[examples]``` also installs Matplotlib, which is required to run the examples. Other options: ``[symbolic]``, ``[doc]``, ``[publish]``, and ``[dev]`` (the latter installs all extras).

### Changed

- Update `solve()` to employ the new automatic mode by default. `dS_dob_bracket` is still available but now defaults to `None`, which triggers the new behavior.
- Change default `Si_tol` in `solve()` to 1e-3.
- Update examples to use the new features of `solve()`.
- Update verbose output of `solve()`. Ambiguously named column "Evaluations" replaced with "Calls to D"; now counts all invocations of `D`.
- Improve error messages in Python 3 by suppressing internal exceptions from the exception context.

### Fixed

- Fix floating point warnings that sometimes appeared during calls to `solve()` and `solve_from_guess()`.
- Remove unnecessary restriction on the `l` parameter of `D.van_genuchten()`.
- Fix list of examples in README file.

## [0.9.7] - 2020-02-11

### Fixed

- Fix a problem related to the encoding of README.md that prevented installation with some versions of _pip_.

## [0.9.6] - 2020-02-10

### Added

- Add optional intrinsic permeability, kinematic viscosity, and gravitational acceleration parameters to `D.van_genuchten()` and `D.richards()`, which can be used in place of the hydraulic conductivity parameter.
- Update _examples/HF135_ to use intrinsic permeability.
- Add new _radial_ example to _examples/HF135_.
- Add scripts to plot the diffusivities in all example cases.
- Add units of measurement to _examples/1INFILTR_ and _examples/HF135_.
- Add mention of porousMultiphaseFoam software version used to validate _examples/HF135_. 
- Add this changelog file to the project.

### Changed

- Rename `SemiInfiniteSolution` class to just `Solution` (and the old `Solution` class to `BaseSolution`).
- Update Hydrus-1D validation data for _examples/1INFILTR_.
- Change "water content" to "saturation" in _examples/HF135_.
- Improve example plots.
- Rename "API documentation" to "reference documentation".
- Improve README.md file and reference documentation.

### Removed

- Remove _examples/powerlaw/radial_ example. Replaced by new _radial_ example in _examples/HF135_.

## [0.9.5] - 2019-12-26

### Changed

- Rename parameter of `D.constant()` function to `D0`.
- Improve documentation.

### Fixed

- Fix `D.constant`, which was previously broken due to a bug.
- Fix use of the terms "saturation" and "water content" throughout the project [Note: fixed in _examples/HF135_ in 0.9.6].

## [0.9.4] - 2019-10-09

### Changed

- Improve README file and reference documentation.

## [0.9.3] - 2019-09-27

### Added

- Add links to the online reference documentation for functionality listed in the README file.
- Add GSaM logo to README file.

### Changed

- Make parameter max_nodes in function `solve_from_guess()` default to 1000.
- Make functions raise `TypeError` instead of `ValueError` on illegal combinations of arguments.

### Fixed

- Fix typo in name of _examples/1INFILTR_.
- Fix wrong name used in mention of _examples/refine.py_ in the README file.

## [0.9.2] - 2019-09-16

First public pre-release version.

[0.9.8]: https://github.com/gerlero/fronts/compare/v0.9.7...HEAD
[0.9.7]: https://github.com/gerlero/fronts/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/gerlero/fronts/compare/v0.9.5...v0.9.6
[0.9.5]: https://github.com/gerlero/fronts/compare/v0.9.4...v0.9.5
[0.9.4]: https://github.com/gerlero/fronts/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/gerlero/fronts/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/gerlero/fronts/releases/tag/v0.9.2
