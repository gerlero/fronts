#!/usr/bin/env python

"""1INFILTR case from Hydrus-1D, horizontal."""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

from fronts import solve
from fronts.D import van_genuchten

import validation

epsilon = 1e-6

D = van_genuchten(n=1.506, alpha=0.01433, Ks=25, S_range=(0, 0.3308))

solution = solve(D=D, Si=0.1003, Sb=0.3308-epsilon, verbose=2)

plt.title("Solutions")
for t, S in zip(validation.t, validation.S):
    plt.plot(validation.r, solution.S(validation.r,t), 
             label="Fronts, t={}".format(t))
    plt.plot(validation.r, S, label="{}, t={}".format(validation.name,t))
plt.xlabel("r")
plt.ylabel("water content")
plt.grid(which='both')
plt.legend()
plt.show()

plt.title("Solutions")
for t, velocity in zip(validation.t, validation.velocity):
    plt.plot(validation.r, solution.flux(validation.r,t), 
             label="Fronts, t={}".format(t))
    plt.plot(validation.r, velocity, 
             label="{}, t={}".format(validation.name,t))
plt.xlabel("r")
plt.ylabel("velocity")
plt.grid(which='both')
plt.legend()
plt.show()