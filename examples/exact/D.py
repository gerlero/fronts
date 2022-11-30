#!/usr/bin/env python3

"""Plot of D for the 'exact' validation case."""

import numpy as np
import matplotlib.pyplot as plt


def D(theta):

    return 0.5*(1 - np.log(theta)) # Exact solution: theta(o) = np.exp(-o)
    # Reference: Philip (1960) Table 1, No. 13
    # https://doi.org/10.1071/PH600001


epsilon = 1e-6

theta = np.linspace(epsilon, 1, 200)

fig = plt.figure()
fig.canvas.manager.set_window_title("D plot")

plt.title("D(theta)")
plt.plot(theta, D(theta))
plt.xlabel("theta")
plt.ylabel("D")
plt.grid(which='both')

plt.show()
