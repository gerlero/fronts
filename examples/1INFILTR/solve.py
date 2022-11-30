#!/usr/bin/env python3

"""1INFILTR case from Hydrus-1D, horizontal."""

import matplotlib.pyplot as plt

from fronts import solve
from fronts.D import van_genuchten

import validation

epsilon = 1e-6

Ks = 25  # cm/h
alpha = 0.01433  # 1/cm
n = 1.506
theta_range = (0, 0.3308)

D = van_genuchten(n=n, alpha=alpha, Ks=Ks, theta_range=theta_range)

theta = solve(D=D, i=0.1003, b=0.3308-epsilon, verbose=2)


fig = plt.figure()
fig.canvas.manager.set_window_title("Water content plot")

plt.title("Water content fields")
for t, theta_ in zip(validation.t, validation.theta):
    plt.plot(validation.r, theta(validation.r,t),
             label=f"Fronts, t={t} {validation.t_unit}")
    plt.plot(validation.r, theta_, label=f"{validation.name}, t={t} {validation.t_unit}")
plt.xlabel(f"r [{validation.r_unit}]")
plt.ylabel("water content [-]")
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.manager.set_window_title("Velocity plot")

plt.title("Velocity fields")
for t, velocity in zip(validation.t, validation.velocity):
    plt.plot(validation.r, theta.flux(validation.r,t),
             label=f"Fronts, t={t} {validation.t_unit}")
    plt.plot(validation.r, velocity, label=f"{validation.name}, t={t} {validation.t_unit}")
plt.xlabel(f"r [{validation.r_unit}]")
plt.ylabel(f"Darcy velocity [{validation.r_unit}/{validation.t_unit}]")
plt.grid(which='both')
plt.legend()

plt.show()
