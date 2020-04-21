#!/usr/bin/env python

"""
This example solves a problem that has an exact solution (using
`fronts.solve_from_guess`) and compares the solutions.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve_from_guess


def D(theta, derivatives=0):

    D = 0.5*(1 - np.log(theta))  # Exact solution: theta(o) = np.exp(-o)
    # Reference: Philip (1960) Table 1, No. 13
    # https://doi.org/10.1071/PH600001

    if derivatives == 0: return D

    dD_dtheta = -0.5/theta

    if derivatives == 1: return D, dD_dtheta

    d2D_dtheta2 = -dD_dtheta/theta

    if derivatives == 2: return D, dD_dtheta, d2D_dtheta2

    raise ValueError("derivatives must be 0, 1, or 2")


o = np.linspace(0, 20, 100)

epsilon = 1e-5

theta = solve_from_guess(D, i=epsilon, b=1, o_guess=o, guess=0.5,
                         verbose=2)


fig = plt.figure()
fig.canvas.set_window_title("theta plot")

plt.title("theta(o)")
plt.plot(o, theta(o=o), color='steelblue', label="Fronts")
plt.plot(o, np.exp(-o), color='sandybrown', label="Exact")
plt.xlabel("o")
plt.ylabel("theta")
plt.grid(which='both')
plt.legend()

plt.show()
