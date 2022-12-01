#!/usr/bin/env python3

"""
Lateral flow in a HF135 nitrocellulose membrane.

This example solves the problem with Fronts and compares the solution with
the one obtained using porousMultiphaseFoam (OpenFOAM)
"""

import matplotlib.pyplot as plt

from fronts import solve
from fronts.D import van_genuchten

import validation

epsilon = 1e-7

# Wetting of an HF135 membrane, Van Genuchten model
# Data from Buser (PhD thesis, 2016)
# http://hdl.handle.net/1773/38064
theta_range = (0.0473, 0.945)
k = 5.50e-13  # m**2
alpha = 0.2555  # 1/m
n = 2.3521
theta_i = 0.102755  # Computed from P0

theta_b = theta_range[1] - epsilon

D = van_genuchten(n=n, alpha=alpha, k=k, theta_range=theta_range)


theta = solve(D=D, i=theta_i, b=theta_b, verbose=2)


fig = plt.figure()
fig.canvas.manager.set_window_title("Water content plot")

plt.title(f"Water content field at t={validation.t} {validation.t_unit}")
plt.plot(validation.r, theta(validation.r,validation.t),
         color='steelblue', label="Fronts")
plt.plot(validation.r, validation.theta, color='sandybrown', label=validation.name)
plt.xlabel(f"position [{validation.r_unit}]")
plt.ylabel("water content [-]")
plt.grid(which='both')
plt.legend()


fig = plt.figure()
fig.canvas.manager.set_window_title("Velocity plot")

plt.title(f"Velocity field at t={validation.t} {validation.t_unit}")
plt.plot(validation.r, theta.flux(validation.r,validation.t),
         color='steelblue', label="Fronts")
plt.plot(validation.r, validation.velocity,
         color='sandybrown', label=validation.name)
plt.xlabel(f"position [{validation.r_unit}]")
plt.ylabel(f"Darcy velocity [{validation.r_unit}/{validation.t_unit}]")
plt.grid(which='both')
plt.legend()

plt.show()
