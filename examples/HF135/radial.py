#!/usr/bin/env python3

"""
Radial flow in a HF135 nitrocellulose membrane.
"""

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
theta_range = (0.0473, 0.945)
k = 5.50e-13  # m**2
alpha = 0.2555  # 1/m
n = 2.3521
theta_i = 0.102755  # Computed from P0
h = 1.60e-4  # m -- thickness

flowrate = 1e-9  # m**3/s

D = van_genuchten(n=n, alpha=alpha, k=k, theta_range=theta_range)


theta = solve_flowrate(D=D, i=theta_i, Qb=flowrate, radial='cylindrical', height=h,
	               verbose=2)


r = np.linspace(0, 5e-2, 200)  # m
t = (60, 120)  # s

fig = plt.figure()
fig.canvas.manager.set_window_title("Water content plot")

plt.title("Water content fields")
plt.plot(r, theta(r,t[0]), label=f"t={t[0]} {t_unit}")
plt.plot(r, theta(r,t[1]), label=f"t={t[1]} {t_unit}")
plt.xlabel(f"position [{r_unit}]")
plt.ylabel("water content [-]")
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.manager.set_window_title("Velocity plot")

plt.title("Velocity fields")
plt.plot(r, theta.flux(r,t[0]), label=f"t={t[0]} {t_unit}")
plt.plot(r, theta.flux(r,t[1]), label=f"t={t[1]} {t_unit}")
plt.xlabel(f"position [{r_unit}]")
plt.ylabel(f"Darcy velocity [{r_unit}/{t_unit}]")
plt.grid(which='both')
plt.legend()

fig = plt.figure()
fig.canvas.manager.set_window_title("Flow rate plot")

plt.title("Flow rate fields")
plt.plot(r, theta.flux(r,t[0]) * (2*pi*r) * h,
         label=f"t={t[0]} {t_unit}")
plt.plot(r, theta.flux(r,t[1]) * (2*pi*r) * h,
         label=f"t={t[1]} {t_unit}")
plt.xlabel(f"position [{r_unit}]")
plt.ylabel(f"flow rate [{r_unit}**3/{t_unit}]")
plt.grid(which='both')
plt.legend()

plt.show()
