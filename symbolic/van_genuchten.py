#!/usr/bin/env python

"""
Script that derives the expressions for the bundled Van Genuchten diffusivity
function

Used only in development. Running this script requires SymPy.
"""

from __future__ import division, absolute_import, print_function

import sympy as sp

alpha, m, Ks, l = sp.symbols('alpha, m, Ks, l', real=True, positive=True)
theta_range = sp.symbols('theta_range[0], theta_range[1]', real=True)

theta = sp.Symbol('theta', real=True)

################################
Se = sp.Symbol('Se', real=True, positive=True)

D = (1-m)*Ks/(alpha*m*(theta_range[1] - theta_range[0])) * Se**l*Se**(-1/m) * ((1-Se**(1/m))**(-m) + (1-Se**(1/m))**m - 2)

D = D.subs(Se, (theta - theta_range[0])/(theta_range[1] - theta_range[0]))
# Reference: Van Genuchten (1980) Equation 11
# https://doi.org/10.2136/sssaj1980.03615995004400050002x
################################

x, (D, dD_dtheta, d2D_dtheta2) = sp.cse([sp.diff(D, theta, n) for n in range(3)], optimizations='basic')

for var, expr in x:
    print("{} = {}".format(var, expr))

print("D = {}".format(D))
print("dD_dtheta = {}".format(dD_dtheta))
print("d2D_dtheta2 = {}".format(d2D_dtheta2))
