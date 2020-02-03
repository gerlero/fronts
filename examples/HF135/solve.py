#!/usr/bin/env python

"""
Lateral flow in a HF135 nitrocellulose membrane.

This example solves the problem with Fronts and compares the solution with
the one obtained using porousMultiphaseFoam (OpenFOAM)
"""

from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt

from fronts import solve
from fronts.D import van_genuchten

import validation

epsilon = 1e-7

# Wetting of an HF135 membrane, Van Genuchten model
# Data from Buser (PhD thesis, 2016)
# http://hdl.handle.net/1773/38064
S_range = (0.0473, 0.945)
k = 5.50e-13  # m**2
alpha = 0.2555  # 1/m
n = 2.3521
Si = 0.102755  # Computed from P0 

Sb = S_range[1] - epsilon

D = van_genuchten(n=n, alpha=alpha, k=k, S_range=S_range)


solution = solve(D=D, Si=Si, Sb=Sb, Si_tol=1e-3, verbose=2)


fig = plt.figure()
fig.canvas.set_window_title("Saturation plot")

plt.title("Saturation field at t={} {}".format(validation.t, validation.t_unit))
plt.plot(validation.r, solution.S(validation.r,validation.t), 
         color='steelblue', label="Fronts")
plt.plot(validation.r, validation.S, color='sandybrown', label=validation.name)
plt.xlabel("position [{}]".format(validation.r_unit))
plt.ylabel("saturation [-]")
plt.grid(which='both')
plt.legend()


fig = plt.figure()
fig.canvas.set_window_title("Velocity plot")

plt.title("Velocity field at t={} {}".format(validation.t, validation.t_unit))
plt.plot(validation.r, solution.flux(validation.r,validation.t),
         color='steelblue', label="Fronts")
plt.plot(validation.r, validation.velocity, 
         color='sandybrown', label=validation.name)
plt.xlabel("position [{}]".format(validation.r_unit))
plt.ylabel("true velocity [{}/{}]".format(validation.r_unit, validation.t_unit))
plt.grid(which='both')
plt.legend()

plt.show()