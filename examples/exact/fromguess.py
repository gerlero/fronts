#!/usr/bin/env python3

"""
This example solves a problem that has an exact solution (using
`fronts.solve_from_guess`) and compares the solutions.
"""

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve_from_guess

o = np.linspace(0, 20, 100)

theta = solve_from_guess(D="0.5*(1 - log(theta))", i=0, b=1,
                         o_guess=o, guess=0.5, verbose=2)


fig = plt.figure()
fig.canvas.manager.set_window_title("theta plot")

plt.title("theta(o)")
plt.plot(o, theta(o=o), color='steelblue', label="Fronts")
plt.plot(o, np.exp(-o), color='sandybrown', label="Exact")
plt.xlabel("o")
plt.ylabel("theta")
plt.grid(which='both')
plt.legend()

plt.show()
