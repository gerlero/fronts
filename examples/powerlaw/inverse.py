#!/usr/bin/env python3

"""Examples of usage of `fronts.solve` and `fronts.inverse`."""

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve, inverse
from fronts.D import power_law

k = 4.0
D = power_law(k=k)

c1i = 0.1
c1b = 1.0

c1 = solve(D, i=c1i, b=c1b, verbose=2)

c2i = 1.0
c2b = 0.1

c2 = solve(D, i=c2i, b=c2b, verbose=2)
# Reverse Sb and Si

r = np.linspace(0, 30, 200)
t = 60


fig = plt.figure()
fig.canvas.manager.set_window_title("c plot")

plt.title(f"c field at t={t}")
plt.plot(r, c1(r,t), label="Case 1")
plt.plot(r, c2(r,t), label="Case 2")
plt.xlabel("r")
plt.ylabel("c")
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.manager.set_window_title("Flux plot")

plt.title(f"Flux field at t={t}")
plt.plot(r, c1.flux(r,t), label="Case 1")
plt.plot(r, c2.flux(r,t), label="Case 2")
plt.xlabel("r")
plt.ylabel("flux")
plt.grid(which='both')
plt.legend()

D1 = inverse(o=c1.o, samples=c1(o=c1.o))
D2 = inverse(o=c2.o, samples=c2(o=c2.o))

c = np.linspace(0.1, 1.0, 200)

fig = plt.figure()
fig.canvas.manager.set_window_title("D plot")

plt.title("D")
plt.plot(c, D1(c), label="inverse() of case 1")
plt.plot(c, D2(c), label="inverse() of case 2")
plt.plot(c, D(c), label="Analytical")
plt.xlabel("D")
plt.ylabel("c")
plt.yscale('log')
plt.grid(which='both')
plt.legend()

plt.show()
