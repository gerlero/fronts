"""Numerical library for nonlinear diffusion problems based on the Boltzmann transformation."""

__version__ = "1.2.7"

from ._boltzmann import BaseSolution, as_o, do_dr, do_dt, o, ode, r, t
from ._inverse import inverse, sorptivity
from ._semiinfinite import Solution, solve, solve_flowrate, solve_from_guess

__all__ = [
    "BaseSolution",
    "Solution",
    "as_o",
    "do_dr",
    "do_dt",
    "inverse",
    "o",
    "ode",
    "r",
    "solve",
    "solve_flowrate",
    "solve_from_guess",
    "sorptivity",
    "t",
]
