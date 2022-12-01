#!/usr/bin/env python3

"""
Lateral flow in a HF135 nitrocellulose membrane.

This example shows a possible usage of `fronts.solve_from_guess` (and also
`fronts.solve`) to refine a solution previously obtained with `fronts.solve`
"""

import matplotlib.pyplot as plt

from fronts import solve, solve_from_guess
from fronts.D import van_genuchten

import validation

epsilon = 1e-7

# Wetting of an HF135 membrane, Van Genuchten model
# Data from Buser (PhD thesis, 2016)
# http://hdl.handle.net/1773/38064
theta_range = (0.0473, 0.945)
k = 5.50e-13  # m**2
alpha = 0.2555  # 1/m
n = 2.3521
theta_i = 0.102755  # Computed from P0

theta_b = theta_range[1] - epsilon

D = van_genuchten(n=n, alpha=alpha, k=k, theta_range=theta_range)

print("----Starting solution----")
coarse = solve(D=D, i=theta_i, b=theta_b, itol=5e-2, d_dob_bracket=(-1,0),
               verbose=2)

print()
print("----Refined with solve()----")
fine = solve(D=D, i=theta_i, b=theta_b, d_dob_bracket=coarse.d_dob_bracket,
             verbose=2)


o_guess = fine.o
guess = coarse(o=o_guess)

print()
print("----Refined with solve_from_guess()----")
from_guess = solve_from_guess(D=D, i=theta_i, b=theta_b, o_guess=o_guess, guess=guess,
                              verbose=2)


fig = plt.figure()
fig.canvas.manager.set_window_title("Water content plot")

plt.title(f"Water content field at t={validation.t} {validation.t_unit}")
plt.plot(validation.r, coarse(validation.r,validation.t),
         label="Starting solution (solve(), higher tolerance)")
plt.plot(validation.r, fine(validation.r,validation.t),
         label="Refined with solve()")
plt.plot(validation.r, from_guess(validation.r,validation.t),
         label="Refined with solve_from_guess()")
plt.xlabel(f"position [{validation.r_unit}]")
plt.ylabel("water content [-]")
plt.grid(which='both')
plt.legend()


fig = plt.figure()
fig.canvas.manager.set_window_title("Velocity plot")

plt.title(f"Velocity field at t={validation.t} {validation.t_unit}")
plt.plot(validation.r, coarse.flux(validation.r,validation.t),
         label="Starting solution (solve(), higher tolerance)")
plt.plot(validation.r, fine.flux(validation.r,validation.t),
         label="Refined with solve()")
plt.plot(validation.r, from_guess.flux(validation.r,validation.t),
         label="Refined with solve_from_guess()")
plt.xlabel(f"position [{validation.r_unit}]")
plt.ylabel(f"Darcy velocity [{validation.r_unit}/{validation.t_unit}]")
plt.grid(which='both')
plt.legend()

plt.show()
