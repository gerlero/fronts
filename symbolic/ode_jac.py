#!/usr/bin/env python3

"""
Script that derives the ODE's Jacobian matrix

Used only in development. Running this script requires SymPy.
"""

import sympy

o = sympy.Symbol('o', real=True)

D_ = sympy.Symbol('D_', real=True, positive=True)
dD_dtheta, d2D_dtheta2 = sympy.symbols('dD_dtheta, d2D_dtheta2', real=True)
theta, dtheta_do, d2theta_do2 = sympy.symbols('theta, dtheta_do, d2theta_do2', real=True)

k = sympy.Symbol('k', integer=True, nonnegative=True)
k_o = sympy.Symbol('k_o', real=True)

D = sympy.Function('D', real=True, positive=True)

subs = [(D_, D(theta)),
        (dD_dtheta, D(theta).diff(theta)),
        (d2D_dtheta2, D(theta).diff(theta, 2)),
        (k_o, sympy.Piecewise((k/o, sympy.Ne(k,0)), (0, sympy.Eq(k,0))))]

backsubs = [sub[::-1] for sub in subs[::-1]]

################################
y = theta, dtheta_do

d2theta_do2 = -((o/2 + dD_dtheta*dtheta_do)/D_ + k_o)*dtheta_do

fun = (dtheta_do, d2theta_do2)
################################


J = sympy.Matrix(fun).subs(subs).jacobian(y).subs(backsubs)

xs, [J] = sympy.cse(J, optimizations='basic')

for x in xs:
    print("{} = {}".format(*x))

for i in range(2):
    for j in range(2):
        print(f"J[{i},{j}] = {J[i, j]}")
