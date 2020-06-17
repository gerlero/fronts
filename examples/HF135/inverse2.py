#!/usr/bin/env python

"""
Lateral flow in a HF135 nitrocellulose membrane.

In this example, we use `fronts.inverse` to obtain the diffusivity function D
from the validation case and then use it to solve the same problem.

Warning: this example takes ~70 seconds to run to completion.
"""

from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt

from fronts import solve, inverse, o

import validation

D_inverse = inverse(o=o(validation.r, validation.t)[::5], samples=validation.S[::5])
# Using only a fifth of the points so that it does not run too slow

S = solve(D=D_inverse, i=validation.S[-1], b=validation.S[0], itol=5e-3, verbose=2)


fig = plt.figure()
fig.canvas.set_window_title("Saturation plot")

plt.title("Saturation field at t={} {}".format(validation.t, validation.t_unit))
plt.plot(validation.r, validation.S,
         label="Original ({})".format(validation.name))
plt.plot(validation.r, S(validation.r, validation.t),
         label="Reconstructed with inverse() and solve()")
plt.xlabel("r [{}]".format(validation.r_unit))
plt.ylabel("saturation [-]")
plt.grid(which='both')
plt.legend()

plt.show()
