#!/usr/bin/env python

"""Plot of D for the 'exact' validation case."""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt


def D(theta, derivatives=0):

    D = 0.5*(1 - np.log(theta))  # Exact solution: theta(o) = np.exp(-o)
    # Reference: Philip (1960) Table 1, No. 13
    # https://doi.org/10.1071/PH600001

    if derivatives == 0: return D

    dD_dtheta = -0.5/theta

    if derivatives == 1: return D, dD_dtheta

    d2D_dtheta2 = -dD_dtheta/theta

    if derivatives == 2: return D, dD_dtheta, d2D_dtheta2

    raise ValueError("derivatives must be 0, 1, or 2")


epsilon = 1e-6

theta = np.linspace(epsilon, 1, 200)

fig = plt.figure()
fig.canvas.set_window_title("D plot")

plt.title("D(theta)")
plt.plot(theta, D(theta))
plt.xlabel("theta")
plt.ylabel("D")
plt.grid(which='both')

plt.show()
