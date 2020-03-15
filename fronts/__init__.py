"""
Numerical library for nonlinear diffusion problems based on the Boltzmann
transformation.
"""

from __future__ import division, absolute_import, print_function

from ._boltzmann import ode, BaseSolution, o, do_dr, do_dt, r, t, as_o
from ._semiinfinite import solve, solve_from_guess, Solution, inverse

__all__ = ['solve', 'solve_from_guess', 'Solution', 'inverse',
           'ode', 'BaseSolution', 'o', 'do_dr', 'do_dt', 'r', 't', 'as_o']

__version__ = '0.9.8-dev'  # Single source of the package's version
