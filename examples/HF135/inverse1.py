#!/usr/bin/env python3

"""
Lateral flow in a HF135 nitrocellulose membrane.

This example shows how to use `fronts.inverse` to extract the diffusivity
function D from a solution. Here, the solution is obtained with
`fronts.solve`. The extracted D is then used with `fronts.solve` and the
same conditions to verify that an equivalent solution is obtained.

Warning: this example takes ~30 seconds to run to completion.
"""

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve, inverse
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

D_analytical = van_genuchten(n=n, alpha=alpha, k=k, theta_range=theta_range)

analytical = solve(D=D_analytical, i=theta_i, b=theta_b)

o = np.linspace(analytical.o[0], analytical.o[-1], 2000)

theta = analytical(o=o)

D_inverse = inverse(o=o, samples=theta)


fig = plt.figure()
fig.canvas.manager.set_window_title("Diffusivity plot")

plt.title("Diffusivity function")
plt.plot(theta, D_analytical(theta), label="Analytical")
plt.plot(theta, D_inverse(theta), label="Obtained with inverse()")
plt.xlabel("water content [-]")
plt.ylabel(f"diffusivity [{validation.r_unit}**2/{validation.t_unit}]")
plt.yscale('log')
plt.grid(which='both')
plt.legend()


inverse = solve(D=D_inverse, i=theta_i, b=theta_b, verbose=2)

fig = plt.figure()
fig.canvas.manager.set_window_title("Water content plot")

plt.title("Water content in terms of o")
plt.plot(o, analytical(o=o), label="Using analytical diffusivities")
plt.plot(o, inverse(o=o), label="Using diffusivities obtained with inverse()")
plt.xlabel(f"o [{validation.r_unit}/{validation.t_unit}**0.5]")
plt.ylabel("water content [-]")
plt.grid(which='both')
plt.legend()

plt.show()
