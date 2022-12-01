#!/usr/bin/env python3

"""
Lateral flow in a HF135 nitrocellulose membrane.

In this example, we use `fronts.inverse` to obtain the diffusivity function D
from the validation case and then use it to solve the same problem.

Warning: this example takes ~70 seconds to run to completion.
"""

import matplotlib.pyplot as plt

from fronts import solve, inverse, o

import validation

D_inverse = inverse(o=o(validation.r, validation.t)[::5], samples=validation.theta[::5])
# Using only a fifth of the points so that it does not run too slow

theta = solve(D=D_inverse, i=validation.theta[-1], b=validation.theta[0], itol=5e-3, verbose=2)


fig = plt.figure()
fig.canvas.manager.set_window_title("Water content plot")

plt.title(f"Water content field at t={validation.t} {validation.t_unit}")
plt.plot(validation.r, validation.theta,
         label=f"Original ({validation.name})")
plt.plot(validation.r, theta(validation.r, validation.t),
         label="Reconstructed with inverse() and solve()")
plt.xlabel(f"r [{validation.r_unit}]")
plt.ylabel("water content [-]")
plt.grid(which='both')
plt.legend()

plt.show()
