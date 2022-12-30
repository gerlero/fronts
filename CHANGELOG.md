# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.12] - 2022-12-30

### Added

- Add ``letxs()`` and ```letd()``` diffusivity models based on the LET correlations to the `fronts.D` module.

## [0.9.11] - 2022-12-01

### Fixed

- Fix plotting errors in example cases with newer versions of Matplotlib.

### Removed

- Drop support for Python versions older than 3.7 (including Python 2.7).
- Drop support for SciPy versions older than 1.4.0.
- Remove optional dependencies needed to publish the package, including the install option ``[publish]``. The project now uses GitHub Actions to automatically build and upload releases to PyPI.

## [0.9.10] - 2020-10-30

### Fixed

- Fix possible NumPy warning (emitted before the expected `ValueError` is raised) when `solve()` is invoked with a `D` that is not valid.
- Revert "saturation" to "water content" in _examples/HF135_ case.

## [0.9.9] - 2020-10-21

### Added

- Add an automated test suite for validation of the software. Tests are executed automatically on Travis CI whenever changes are pushed to GitHub. Install with new option ``[test]`` (or ``[dev]``) and use ``pytest`` to run the tests locally.

### Changed

- Check that `D` is well behaved at the expected extrema of the solution when invoking `solve()`. If `solve()` determines that the problem cannot be solved, the function will now raise a `ValueError` immediately without performing any iterations.
- Improve error message when attempting to use the 'explicit' method of `solve()` and `solve_flowrate()` with SciPy older than 1.4.0.
- Improve reference documentation.

### Fixed

- Fix _examples/1INFILTR_ case with NumPy versions lower than 1.16.
- Fix _examples/HF135/inverse2.py_ failure in some environments.
- Fix problem statement in 'Usage' section of README file.

## [0.9.8] - 2020-06-11

### Added

- Add new solver `solve_flowrate()` to solve radial problems (cylindrical or polar) with a fixed-flowrate boundary condition.
- Add new automatic mode to `solve()`. Removes the need for users to tune the `d_dob_bracket` (formerly `dS_dob_bracket`) parameter until the function succeeds. In practice, `solve()` can now be expected to return the solution to a problem upon the first call with no parameter tuning required.  The new `solve_flowrate()` also works in a similar manner.
- Allow expressions of _D_ to be passed directly to the solvers (e.g.: ``solve(D="c**2", ...)``). Removes the need for users to provide the derivatives of custom functions. This functionality uses _SymPy_, which is now installed by default.
- Add optional explicit integration method to `solve()` and `solve_flowrate()`. Using it requires SciPy 1.4.0 or later (Python 3 only).
- Add Brooks and Corey moisture diffusivity model to the `fronts.D` module.
- Add new properties `b`, `d_dob`, `i` and `ob`, and methods `d_drb()`, `d_dtb()` and `fluxb()` to the `Solution` class.
- Add optional `d_dob_hint` parameter to `solve()`. Allows users to pass an optional hint to the new automatic mode, which may accelerate convergence in some scenarios.
- Add execution time to the verbose output of solvers. Measures total CPU time in Python 3, or wall-clock time in Python 2.7.
- Add `catch_errors` option to `ode()` that converts _D_ domain errors to invalid values for easier compatibility with SciPy code.
- Add ``'polar'`` as a valid value for the `radial` parameter of `ode()` and solvers.
- Add `from_expr()` to the `fronts.D` module, which transforms expressions into callable _D_ functions.
- Add `__version__` attribute to the main package, which stores the current version as a string.
- Support extra options during install: e.g., ```$ pip install fronts[examples]``` also installs Matplotlib, which is required to run the examples. Other options: ``[doc]``, ``[publish]``, and ``[dev]`` (the latter installs all extras).

### Changed

- Drop the letter `S` from the names of function parameters and methods. For more generality, the library now does not name the solution field in user-facing code. As a consequence of this, `Solution` objects are now callable.
- Update `solve()` to employ the new automatic mode by default. `d_dob_bracket` (formerly `dS_dob_bracket`) is still available but now defaults to `None`, which triggers the new behavior.
- Change default tolerance (`itol`, formerly `Si_tol`) in `solve()` to 1e-3.
- Replace use of `S` in documentation with the Greek letter theta.
- Update examples with the new function signatures and method names.
- Update examples to use the new features of `solve()`.
- Update _examples/HF135/radial.py_ to use the new solver `solve_flowrate()`.
- Update _examples/exact_ case to use an expression for _D_.
- Update verbose output of `solve()`. Ambiguously named column "Evaluations" replaced with "Calls to D"; now counts all invocations of `D`.
- Improve error messages in Python 3 by suppressing internal exceptions from the exception context.
- Improve README file and reference documentation.

### Fixed

- Fix floating point warnings that sometimes appeared during calls to `solve()` and `solve_from_guess()`.
- Remove unnecessary restriction on the `l` parameter of `D.van_genuchten()`.
- Fix list of examples in README file.
- Fix _examples/exact_ case in Python 2.7.
- Fix encoding error when installed in editable mode in Python 2.7.

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
- Change "water content" to "saturation" in _examples/HF135_ [Note: fixed in 0.9.10].
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
- Fix use of the terms "saturation" and "water content" throughout the project [Note: fixed in _examples/HF135_ in 0.9.10].

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

[0.9.13]: https://github.com/gerlero/fronts/compare/v0.9.12...HEAD
[0.9.12]: https://github.com/gerlero/fronts/compare/v0.9.11...v0.9.12
[0.9.11]: https://github.com/gerlero/fronts/compare/v0.9.10...v0.9.11
[0.9.10]: https://github.com/gerlero/fronts/compare/v0.9.9...v0.9.10
[0.9.9]: https://github.com/gerlero/fronts/compare/v0.9.8...v0.9.9
[0.9.8]: https://github.com/gerlero/fronts/compare/v0.9.7...v0.9.8
[0.9.7]: https://github.com/gerlero/fronts/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/gerlero/fronts/compare/v0.9.5...v0.9.6
[0.9.5]: https://github.com/gerlero/fronts/compare/v0.9.4...v0.9.5
[0.9.4]: https://github.com/gerlero/fronts/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/gerlero/fronts/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/gerlero/fronts/releases/tag/v0.9.2
