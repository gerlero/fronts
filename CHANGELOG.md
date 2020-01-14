# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.9.6-dev - Unreleased

### Added

- Add optional intrinsic permeability, kinematic viscosity, and gravitational acceleration parameters to `D.van_genuchten()` and `D.richards()`, which can be used in place of the hydraulic conductivity parameter.
- Update _examples/HF135_ to use intrinsic permeability.
- Add units of measurement to _examples/1INFILTR_ and _examples/HF135_.
- Add mention of porousMultiphaseFoam software version used to validate _examples/HF135_. 
- Add this changelog file to the project.

### Changed

- Update Hydrus-1D validation data for _examples/1INFILTR_.
- Change "water content" to "saturation" in _examples/HF135_.
- Clarify whether the Darcy velocity or true velocity is being plotted in the examples.
- Rename "API documentation" to "reference documentation".
- Improve README.md file.

## 0.9.5 - 2019-12-26

### Changed

- Rename parameter of `D.constant()` function to `D0`.
- Improve documentation.

### Fixed

- Fix `D.constant`, which was previously broken due to a bug.
- Fix use of the terms "saturation" and "water content" throughout the project [Note: fixed in _examples/HF135_ in 0.9.6].

## 0.9.4 - 2019-10-09

### Changed

- Improve README file and reference documentation.

## 0.9.3 - 2019-09-27

### Added

- Add links to the online reference documentation for functionality listed in the README file.
- Add GSaM logo to README file.

### Changed

- Make parameter max_nodes in function `solve_from_guess()` default to 1000.
- Make functions raise `TypeError` instead of `ValueError` on illegal combinations of arguments.

### Fixed

- Fix typo in name of _examples/1INFILTR_.
- Fix wrong name used in mention of _examples/refine.py_ in the README file.

## 0.9.2 - 2019-09-16

First public pre-release version.
