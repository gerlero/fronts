#!/usr/bin/env python

"""
Lateral flow in a HF135 nitrocellulose membrane.

This example shows a possible usage of `fronts.solve_from_guess` (and also
`fronts.solve`) to refine a solution previously obtained with `fronts.solve`
"""

from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt

from fronts import solve, solve_from_guess
from fronts.D import van_genuchten

import validation

epsilon = 1e-7

# Wetting of an HF135 membrane, Van Genuchten model
# Data from Buser (PhD thesis, 2016)
# http://hdl.handle.net/1773/38064
S_range = (0.0473, 0.945)
k = 5.50e-13  # m**2
alpha = 0.2555  # 1/m
n = 2.3521
Si = 0.102755  # Computed from P0 

Sb = S_range[1] - epsilon

D = van_genuchten(n=n, alpha=alpha, k=k, S_range=S_range)

print("----Starting solution----")
coarse = solve(D=D, Si=Si, Sb=Sb, Si_tol=5e-2, verbose=2)

print()
print("----Refined with solve----")
fine = solve(D=D, Si=Si, Sb=Sb, dS_dob_bracket=coarse.dS_dob_bracket, verbose=2)


o_guess = fine.o
S_guess = coarse.S(o=o_guess)

print()
print("----Refined with solve_from_guess----")
from_guess = solve_from_guess(D=D, Si=Si, Sb=Sb, o_guess=o_guess, 
                              S_guess=S_guess, verbose=2)


fig = plt.figure()
fig.canvas.set_window_title("Saturation plot")

plt.title("Saturation field at t={} {}".format(validation.t, validation.t_unit))
plt.plot(validation.r, coarse.S(validation.r,validation.t), 
	     label="Starting solution (solve, high tolerance)")
plt.plot(validation.r, fine.S(validation.r,validation.t), 
	     label="Refined with solve")
plt.plot(validation.r, from_guess.S(validation.r,validation.t), 
	     label="Refined with solve_from_guess")
plt.xlabel("position [{}]".format(validation.r_unit))
plt.ylabel("saturation [-]")
plt.grid(which='both')
plt.legend()


fig = plt.figure()
fig.canvas.set_window_title("Velocity plot")

plt.title("Velocity field at t={} {}".format(validation.t, validation.t_unit))
plt.plot(validation.r, coarse.flux(validation.r,validation.t), 
	     label="Starting solution (solve, high tolerance)")
plt.plot(validation.r, fine.flux(validation.r,validation.t), 
	     label="Refined with solve")
plt.plot(validation.r, from_guess.flux(validation.r,validation.t), 
	     label="Refined with solve_from_guess")
plt.xlabel("position [{}]".format(validation.r_unit))
plt.ylabel("true velocity [{}/{}]".format(validation.r_unit, validation.t_unit))
plt.grid(which='both')
plt.legend()

plt.show()
