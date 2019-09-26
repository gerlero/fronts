#!/usr/bin/env python

"""
Lateral flow in a HF135 nitrocellulose membrane.

In this example, we use `fronts.inverse` to obtain the diffusivity function D 
from the validation case and then use it to solve the same problem.

Warning: this example takes ~70 seconds to run to completion.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve, inverse, o
from fronts.D import van_genuchten

import validation

D_inverse = inverse(o=o(validation.r, validation.t)[::5], S=validation.S[::5])
# Using only a fifth of the points so that it does not run too slow

sol = solve(D=D_inverse, Si=validation.S[-1], Sb=validation.S[0], 
               Si_tol=1e-3, verbose=2)

plt.title("Solution at t={}".format(validation.t))
plt.plot(validation.r, validation.S, 
         label="Original ({})".format(validation.name))
plt.plot(validation.r, sol.S(validation.r, validation.t), 
         label="Reconstructed with inverse and solve")
plt.xlabel("r")
plt.ylabel("saturation")
plt.grid(which='both')
plt.legend()
plt.show()
