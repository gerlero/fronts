#!/usr/bin/env python

"""
Script that derives the ODE's Jacobian matrix

Used only in development. Running this script requires SymPy.
"""

from __future__ import division, absolute_import, print_function

import sympy as sp

o = sp.Symbol('o', real=True, nonnegative=True)

D_ = sp.Symbol('D_', real=True, positive=True)
dD_dS, d2D_dS2 = sp.symbols('dD_dS, d2D_dS2', real=True)
S, dS_do, d2S_do2 = sp.symbols('S, dS_do, d2S_do2', real=True)

k = sp.Symbol('k', integer=True, nonnegative=True)

D = sp.Function('D', real=True, positive=True)

D_subs = [(D_, D(S)), (dD_dS, D(S).diff(S)), (d2D_dS2, D(S).diff(S, 2))]

D_backsubs = [sub[::-1] for sub in D_subs[::-1]]

################################
y = S, dS_do

d2S_do2 = -((o/2 + dD_dS*dS_do)/D_ + k/o)*dS_do

fun = (dS_do, d2S_do2)
################################

for i, fi in enumerate(fun):
    for j, yj in enumerate(y):

        element = sp.diff(fi.subs(D_subs), yj).subs(D_backsubs).simplify()

        print("jacobian[{},{}] = {}".format(i, j, element))
