#!/usr/bin/env python

"""
1INFILTR case from Hydrus-1D, horizontal.

Diffusivity plot.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts.D import van_genuchten

from validation import r_unit, t_unit

epsilon = 1e-6

Ks = 25  # cm/h
alpha = 0.01433  # 1/cm
n = 1.506
theta_range = (0, 0.3308)

D = van_genuchten(n=n, alpha=alpha, Ks=Ks, theta_range=theta_range)

theta = np.linspace(theta_range[0]+epsilon, theta_range[1]-epsilon, 200)


fig = plt.figure()
fig.canvas.set_window_title("Diffusivity plot")

plt.title("Diffusivity function")
plt.plot(theta, D(theta))
plt.xlabel("water content [-]")
plt.ylabel("diffusivity [{}**2/{}]".format(r_unit, t_unit))
plt.yscale('log')
plt.grid(which='both')

plt.show()
