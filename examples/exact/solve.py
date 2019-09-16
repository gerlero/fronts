#!/usr/bin/env python

"""
This example solves a problem that has an exact solution (using `fronts.solve`) 
and compares the solutions
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve


def D(S, derivatives=0):

    D = 0.5*(1 - np.log(S))  # Exact solution: S(o) = np.exp(-o)
    # Reference: Philip (1960) Table 1, No. 13
    # https://doi.org/10.1071/PH600001
    
    if derivatives == 0: return D
    
    dD_dS = -0.5/S

    if derivatives == 1: return D, dD_dS
            
    d2D_dS2 = -dD_dS/S

    if derivatives == 2: return D, dD_dS, d2D_dS2

    raise ValueError("derivatives must be 0, 1, or 2")


epsilon = 1e-6

solution = solve(D, Si=epsilon, Sb=1, dS_dob_bracket=(-1.1, -0.9), Si_tol=1e-3,
			verbose=2)

o = np.linspace(0, 20, 200)

plt.title("Solution in terms of o")
plt.plot(o, solution.S(o=o), color='steelblue', label="Fronts")
plt.plot(o, np.exp(-o), color='sandybrown', label="Exact") 
plt.xlabel("o")
plt.ylabel("S")
plt.grid(which='both')
plt.legend()
plt.show()
