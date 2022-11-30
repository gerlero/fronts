#!/usr/bin/env python3

"""
Script that derives the expressions for the bundled Van Genuchten diffusivity
function

Used only in development. Running this script requires SymPy.
"""

import sympy

from generate import functionstr

alpha, m, Ks, l = sympy.symbols('alpha, m, Ks, l', real=True, positive=True)
theta_range = sympy.symbols('theta_range[0], theta_range[1]', real=True)

theta = sympy.Symbol('theta', real=True)

################################
Se = sympy.Symbol('Se', real=True, positive=True)

D = (1-m)*Ks/(alpha*m*(theta_range[1] - theta_range[0])) * Se**l*Se**(-1/m) * ((1-Se**(1/m))**(-m) + (1-Se**(1/m))**m - 2)

D = D.subs(Se, (theta - theta_range[0])/(theta_range[1] - theta_range[0]))
# Reference: Van Genuchten (1980) Equation 11
# https://doi.org/10.2136/sssaj1980.03615995004400050002x
################################

D = D.simplify()

print(functionstr(theta, D))
