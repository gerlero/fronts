#!/usr/bin/env python3

"""Example of usage of `fronts.solve`."""

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve
from fronts.D import power_law


k = 4.0
ci = 0.1
cb = 1.0

c = solve(D=power_law(k=k), i=ci, b=cb, verbose=2)

r = np.linspace(0, 10, 200)


fig = plt.figure()
fig.canvas.manager.set_window_title("c plot")

plt.title("c field")
plt.plot(r, c(r,t=30), label="t=30")
plt.plot(r, c(r,t=60), label="t=60")
plt.xlabel("r")
plt.ylabel("S")
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.manager.set_window_title("Flux plot")

plt.title("Flux field")
plt.plot(r, c.flux(r,t=30), label="t=30")
plt.plot(r, c.flux(r,t=60), label="t=60")
plt.xlabel("r")
plt.ylabel("flux")
plt.grid(which='both')
plt.legend()

plt.show()
