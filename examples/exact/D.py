#!/usr/bin/env python

"""
Plot of D in the exact case.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

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

S = np.linspace(epsilon, 1, 200)


fig = plt.figure()
fig.canvas.set_window_title("D plot")

plt.title("D(S)")
plt.plot(S, D(S)) 
plt.xlabel("S")
plt.ylabel("D")
plt.grid(which='both')

plt.show()
