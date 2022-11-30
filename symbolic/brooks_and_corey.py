#!/usr/bin/env python3

"""
Script that derives the expressions for the bundled Brooks and Corey
diffusivity function.
"""

import sympy

from generate import functionstr

Hp = sympy.Symbol('Hp', real=True, negative=True)
Ks, alpha, n, Se = sympy.symbols('Ks, alpha, n, Se', real=True, positive=True)
theta_range = sympy.symbols('theta_range[0], theta_range[1]', real=True)
l = sympy.Symbol('l', real=True)
theta = sympy.Symbol('theta', real=True)

################################
Se_expr = 1/(sympy.Abs(alpha*Hp)**n)
Hp_expr, = sympy.solve(Se_expr - Se, Hp)
Se_expr2 = (theta - theta_range[0])/(theta_range[1] - theta_range[0])

C = -n/Hp*(theta_range[1] - theta_range[0])*1/(sympy.Abs(alpha*Hp)**n)
K = Ks*Se**(2/n + l + 2)

C = C.subs(Hp, Hp_expr)
C = C.subs(Se_expr, Se)
################################

D = K/C

print(f"D={D.simplify()}")

D = D.subs(Se, Se_expr2).simplify()

print(functionstr(theta, D))