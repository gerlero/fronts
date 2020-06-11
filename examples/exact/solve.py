#!/usr/bin/env python

"""
This example solves a problem that has an exact solution (using `fronts.solve`)
and compares the solutions.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve

theta = solve(D="0.5*(1 - log(theta))", i=0, b=1, verbose=2)

o = np.linspace(0, 20, 200)

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
