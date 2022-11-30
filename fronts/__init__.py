"""
Numerical library for nonlinear diffusion problems based on the Boltzmann
transformation.
"""

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # No importlib.metadada in Python < 3.8
    import importlib_metadata

try:
    __version__ = importlib_metadata.version('fronts')
except Exception:
    __version__ = 'unknown'

from ._boltzmann import ode, BaseSolution, o, do_dr, do_dt, r, t, as_o
from ._semiinfinite import (solve, solve_flowrate, solve_from_guess, Solution, 
                            inverse)

__all__ = ['solve', 'solve_flowrate', 'solve_from_guess', 'Solution',
           'inverse', 'ode', 'BaseSolution', 'o', 'do_dr', 'do_dt', 'r', 't',
           'as_o']


