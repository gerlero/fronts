#!/usr/bin/env python

"""
Plot of D in the powerlaw case.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts.D import power_law

k = 4.0

D = power_law(k=k)

S = np.linspace(0, 1, 200)

plt.title("D")
plt.plot(S, D(S)) 
plt.xlabel("S")
plt.ylabel("D")
plt.grid(which='both')
plt.show()
