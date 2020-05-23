#!/usr/bin/env python

"""1INFILTR case from Hydrus-1D, horizontal."""

from __future__ import division, absolute_import, print_function

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
fig.canvas.set_window_title("Water content plot")

plt.title("Water content fields")
for t, theta_ in zip(validation.t, validation.theta):
    plt.plot(validation.r, theta(validation.r,t),
             label="Fronts, t={} {}".format(t, validation.t_unit))
    plt.plot(validation.r, theta_, label="{}, t={} {}"
             .format(validation.name, t, validation.t_unit))
plt.xlabel("r [{}]".format(validation.r_unit))
plt.ylabel("water content [-]")
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.set_window_title("Velocity plot")

plt.title("Velocity fields")
for t, velocity in zip(validation.t, validation.velocity):
    plt.plot(validation.r, theta.flux(validation.r,t),
             label="Fronts, t={} {}".format(t, validation.t_unit))
    plt.plot(validation.r, velocity, label="{}, t={} {}"
             .format(validation.name, t, validation.t_unit))
plt.xlabel("r [{}]".format(validation.r_unit))
plt.ylabel("Darcy velocity [{}/{}]".format(validation.r_unit, validation.t_unit))
plt.grid(which='both')
plt.legend()

plt.show()
