"""Numerical library for nonlinear diffusion problems based on the Boltzmann transformation."""

__version__ = "1.2.3"

from ._boltzmann import BaseSolution, as_o, do_dr, do_dt, o, ode, r, t
from ._inverse import inverse, sorptivity
from ._semiinfinite import Solution, solve, solve_flowrate, solve_from_guess

__all__ = [
    "solve",
    "solve_flowrate",
    "solve_from_guess",
    "Solution",
    "inverse",
    "sorptivity",
    "ode",
    "BaseSolution",
    "o",
    "do_dr",
    "do_dt",
    "r",
    "t",
    "as_o",
]
