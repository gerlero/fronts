#!/usr/bin/env python3

"""
Plot of D in the powerlaw case.
"""

import numpy as np
import matplotlib.pyplot as plt

from fronts.D import power_law

k = 4.0

D = power_law(k=k)

c = np.linspace(0, 1, 200)

fig = plt.figure()
fig.canvas.manager.set_window_title("D plot")

plt.title("D(c)")
plt.plot(c, D(c))
plt.xlabel("c")
plt.ylabel("D")
plt.grid(which='both')
plt.yscale('log')

plt.show()
