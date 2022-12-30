#!/usr/bin/env python

"""
Script that derives the expressions for the LET-based diffusivity functions.
"""

import sympy

from generate import functionstr

Ks = sympy.Symbol('Ks', real=True, positive=True)
Lw, Ew, Tw  = sympy.symbols('Lw, Ew, Tw', real=True)

alpha = sympy.Symbol('alpha', real=True, positive=True)
Ls, Es, Ts  = sympy.symbols('Ls, Es, Ts', real=True)

theta_range = sympy.symbols('theta_range[0], theta_range[1]', real=True)
theta = sympy.Symbol('theta', real=True)

################################
# References:
# Lomeland (2018)
# http://www.jgmaas.com/SCA/2018/SCA2018-056.pdf
# Gerlero et al. (2022)
# https://doi.org/10.1007/s11242-021-01724-w

Dwt = Ks/alpha
Swir = theta_range[0]/theta_range[1]

Swp = (theta - theta_range[0])/(theta_range[1] - theta_range[0])

D = Es*Dwt/theta_range[1]*Swp**Lw*Swp**Ts*(1 - Swp)**Ls*(Ls*Swp - Swp*Ts + Ts)/(Swp*(Swir - 1)*(Swp - 1)*(Es*Swp**Ts + (1 - Swp)**Ls)**2*(Ew*(1 - Swp)**Tw + Swp**Lw))

################################

print("D={}".format(D))

print(functionstr(theta, D))
