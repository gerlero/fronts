#!/usr/bin/env python

"""
Lateral flow in a HF135 nitrocellulose membrane.

Diffusivity plot.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts.D import van_genuchten

from validation import r_unit, t_unit

epsilon = 1e-7

# Wetting of an HF135 membrane, Van Genuchten model
# Data from Buser (PhD thesis, 2016)
# http://hdl.handle.net/1773/38064
theta_range = (0.0473, 0.945)
k = 5.50e-13  # m^2
alpha = 0.2555  # 1/m
n = 2.3521
theta_i = 0.102755  # Computed from P0

D = van_genuchten(n=n, alpha=alpha, k=k, theta_range=theta_range)

theta = np.linspace(theta_range[0]+epsilon, theta_range[1]-epsilon, 200)


fig = plt.figure()
fig.canvas.manager.set_window_title("Diffusivity plot")

plt.title("Diffusivity function")
plt.plot(theta, D(theta))
plt.xlabel("water content [-]")
plt.ylabel("diffusivity [{}**2/{}]".format(r_unit, t_unit))
plt.yscale('log')
plt.grid(which='both')

plt.show()
