#!/usr/bin/env python

"""Examples of usage of `fronts.solve` and `fronts.inverse`."""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve, inverse
from fronts.D import power_law

k = 4.0
D = power_law(k=k)

Si = 0.1
Sb = 1.0

result_1 = solve(D, Si=Si, Sb=Sb)

Si = 1.0
Sb = 0.1

result_2 = solve(D, Si=Si, Sb=Sb, dS_dob_bracket=(1e3, 1e4))
# Reverse Sb and Si

r = np.linspace(0, 30, 200)
t = 60

plt.title("Solution at t={}".format(t))
plt.plot(r, result_1.S(r,t), label="Case 1")
plt.plot(r, result_2.S(r,t), label="Case 2")
plt.xlabel("r")
plt.ylabel("S")
plt.grid(which='both')
plt.legend()
plt.show()

plt.title("Solution at t={}".format(t))
plt.plot(r, result_1.flux(r,t), label="Case 1")
plt.plot(r, result_2.flux(r,t), label="Case 2")
plt.xlabel("r")
plt.ylabel("flux")
plt.grid(which='both')
plt.legend()
plt.show()

D1 = inverse(o=result_1.o, S=result_1.S(o=result_1.o))
D2 = inverse(o=result_2.o, S=result_2.S(o=result_2.o))

S = np.linspace(0.1, 1.0, 200)

plt.title("Diffusivities".format(t))
plt.plot(S, D1(S), label="Case 1")
plt.plot(S, D2(S), label="Case 2")
plt.plot(S, D(S), label="Analytical")
plt.xlabel("D")
plt.ylabel("S")
plt.yscale('log')
plt.grid(which='both')
plt.legend()
plt.show()
