#!/usr/bin/env python

"""Solve a radial case with a moving boundary with `fronts.solve`."""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve
from fronts.D import power_law


k = 4.0
Si = 0.1
Sb = 1.0

solution = solve(D=power_law(k=k), Si=Si, Sb=Sb, 
				 radial='cylindrical', ob=0.25)


r = np.linspace(0, 10, 200)


fig = plt.figure()
fig.canvas.set_window_title("S plot")

plt.title("S field")
plt.plot(r, solution.S(r,t=30), label="t=30")
plt.plot(r, solution.S(r,t=60), label="t=60")
plt.xlabel("r")
plt.ylabel("S")
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.set_window_title("Flux plot")

plt.title("Flux field")
plt.plot(r, solution.flux(r,t=30), label="t=30")
plt.plot(r, solution.flux(r,t=60), label="t=60")
plt.xlabel("r")
plt.ylabel("flux")
plt.grid(which='both')
plt.legend()

plt.show()
