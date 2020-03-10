#!/usr/bin/env python

"""
This example solves a problem that has an exact solution (using
`fronts.solve_from_guess`) and compares the solutions.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve_from_guess


def D(S, derivatives=0):

    D = 0.5*(1 - np.log(S)) # Exact solution: S(o) = np.exp(-o)
    # Reference: Philip (1960) Table 1, No. 13
    # https://doi.org/10.1071/PH600001

    if derivatives == 0: return D

    dD_dS = -0.5/S

    if derivatives == 1: return D, dD_dS

    d2D_dS2 = -dD_dS/S

    if derivatives == 2: return D, dD_dS, d2D_dS2

    raise ValueError("derivatives must be 0, 1, or 2")


o = np.linspace(0, 20, 100)

epsilon = 1e-5

solution = solve_from_guess(D, Si=epsilon, Sb=1, o_guess=o, S_guess=0.5,
                            verbose=2)


fig = plt.figure()
fig.canvas.set_window_title("S plot")

plt.title("S(o)")
plt.plot(o, solution.S(o=o), color='steelblue', label="Fronts")
plt.plot(o, np.exp(-o), color='sandybrown', label="Exact")
plt.xlabel("o")
plt.ylabel("S")
plt.grid(which='both')
plt.legend()

plt.show()
