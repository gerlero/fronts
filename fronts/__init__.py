"""
Numerical library for nonlinear diffusion problems based on the Boltzmann
transformation.
"""

__version__ = '1.2.0'

from ._boltzmann import ode, BaseSolution, o, do_dr, do_dt, r, t, as_o
from ._semiinfinite import solve, solve_flowrate, solve_from_guess, Solution
from ._inverse import inverse, sorptivity

__all__ = ['solve', 'solve_flowrate', 'solve_from_guess', 'Solution',
           'inverse', 'sorptivity' 'ode', 'BaseSolution', 'o', 'do_dr',
           'do_dt', 'r', 't', 'as_o']
