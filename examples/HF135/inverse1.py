#!/usr/bin/env python

"""
Lateral flow in a HF135 nitrocellulose membrane.

This example shows how to use `fronts.inverse` to extract the diffusivity
function D from a solution. Here, the solution is obtained with 
`fronts.solve`. The extracted D is then used with `fronts.solve` and the
same conditions to verify that an equivalent solution is obtained.

Warning: this example takes ~30 seconds to run to completion.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve, inverse
from fronts.D import van_genuchten

epsilon = 1e-7

# Wetting of an HF135 membrane, Van Genuchten model
# Data from Buser (PhD thesis, 2016)
# http://hdl.handle.net/1773/38064
S_range = (0.0473, 0.945)
k = 5.50e-13
alpha = 0.2555
n = 2.3521
Si = 0.102755  # Computed from P0 

Sb = S_range[1] - epsilon

D_analytical = van_genuchten(n=n, alpha=alpha, k=k, S_range=S_range)

analytical = solve(D=D_analytical, Si=Si, Sb=Sb)

o = np.linspace(analytical.o[0], analytical.o[-1], 500)

S = analytical.S(o=o)

D_inverse = inverse(o=o, S=S)

plt.title("Diffusivities")
plt.plot(S, D_analytical(S), label="Analytical")
plt.plot(S, D_inverse(S), label="Obtained with inverse")
plt.xlabel("water content")
plt.ylabel("D")
plt.yscale('log')
plt.grid(which='both')
plt.legend()
plt.show()


inverse = solve(D=D_inverse, Si=Si, Sb=Sb, Si_tol=1e-3, verbose=2)

plt.title("Solution in terms of o")
plt.plot(o, analytical.S(o=o), label="Using analytical diffusivities")
plt.plot(o, inverse.S(o=o), label="Using diffusivities obtained from inverse")
plt.xlabel("o")
plt.ylabel("water content")
plt.grid(which='both')
plt.legend()
plt.show()
