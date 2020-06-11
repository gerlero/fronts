#!/usr/bin/env python

"""
Radial flow in a HF135 nitrocellulose membrane.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from math import pi

from fronts import solve_flowrate
from fronts.D import van_genuchten

from validation import r_unit, t_unit

epsilon = 1e-7

# Wetting of an HF135 membrane, Van Genuchten model
# Data from Buser (PhD thesis, 2016)
# http://hdl.handle.net/1773/38064
S_range = (0.0473, 0.945)
k = 5.50e-13  # m**2
alpha = 0.2555  # 1/m
n = 2.3521
Si = 0.102755  # Computed from P0
h = 1.60e-4  # m -- thickness
eps = 0.76  # porosity

flowrate = 1e-9  # m**3/s

D = van_genuchten(n=n, alpha=alpha, k=k, theta_range=S_range)


S = solve_flowrate(D=D, i=Si, Qb=flowrate/eps, radial='cylindrical', height=h,
	               verbose=2)


r = np.linspace(0, 5e-2, 200)  # m
t = (60, 120)  # s

fig = plt.figure()
fig.canvas.set_window_title("Saturation plot")

plt.title("Saturation fields")
plt.plot(r, S(r,t[0]), label="t={} {}".format(t[0], t_unit))
plt.plot(r, S(r,t[1]), label="t={} {}".format(t[1], t_unit))
plt.xlabel("position [{}]".format(r_unit))
plt.ylabel("saturation [-]")
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.set_window_title("Velocity plot")

plt.title("Velocity fields")
plt.plot(r, S.flux(r,t[0]), label="t={} {}".format(t[0], t_unit))
plt.plot(r, S.flux(r,t[1]), label="t={} {}".format(t[1], t_unit))
plt.xlabel("position [{}]".format(r_unit))
plt.ylabel("true velocity [{}/{}]".format(r_unit, t_unit))
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.set_window_title("Flow rate plot")

plt.title("Flow rate fields")
plt.plot(r, S.flux(r,t[0]) * eps * (2*pi*r) * h,
         label="t={} {}".format(t[0], t_unit))
plt.plot(r, S.flux(r,t[1]) * eps * (2*pi*r) * h,
         label="t={} {}".format(t[1], t_unit))
plt.xlabel("position [{}]".format(r_unit))
plt.ylabel("flow rate [{}**3/{}]".format(r_unit, t_unit))
plt.grid(which='both')
plt.legend()

plt.show()
