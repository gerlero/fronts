#!/usr/bin/env python3

"""
Script that derives the expressions for the LET-based diffusivity functions.
"""

import sympy

from generate import functionstr

Swir = sympy.Symbol('Swir', real=True, positive=True)

L, E, T  = sympy.symbols('L, E, T', real=True)
Dwt = sympy.Symbol('Dwt', real=True)

theta_range = sympy.symbols('theta_range[0], theta_range[1]', real=True)
theta = sympy.Symbol('theta', real=True)

################################
# Reference: Gerlero et al. (2022)
# https://doi.org/10.1007/s11242-021-01724-w


Swp = (theta - theta_range[0])/(theta_range[1] - theta_range[0])
D = Dwt*Swp**L/(Swp**L + E*(1 - Swp)**T)

print("D={}".format(D))

print(functionstr(theta, D.simplify()))