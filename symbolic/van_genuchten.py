#!/usr/bin/env python3

"""
Script that derives the expressions for the bundled Van Genuchten diffusivity
function

Used only in development. Running this script requires SymPy.
"""

import sympy  # type: ignore [import-untyped]
from sympy.codegen.cfunctions import expm1, log1p  # type: ignore [import-untyped]

from .generate import functionstr

alpha, m, Ks, l = sympy.symbols("alpha, m, Ks, l", real=True, positive=True)
theta_range = sympy.symbols("theta_range[0], theta_range[1]", real=True)

theta = sympy.Symbol("theta", real=True)

################################
Se = sympy.Symbol("Se", real=True, positive=True)

# The bracketed term of Van Genuchten (1980) Equation 11,
#
#   (1 - Se**(1/m))**(-m) + (1 - Se**(1/m))**m - 2,
#
# is algebraically equal to w**(-1)*(w - 1)**2 with w = (1 - Se**(1/m))**m, but
# evaluating it directly in floating point suffers catastrophic cancellation as
# Se**(1/m) -> 0 (i.e., near the residual water content), where w - 1 -> 0. In
# that regime the direct evaluation underflows to exact zero -- e.g., with
# n=1.1, already at Se <= 0.1 -- which makes D() return 0 and breaks solvers
# that divide by D. Writing w - 1 = expm1(m*log1p(-Se**(1/m))) evaluates the
# bracket to full precision all the way down to the underflow limit of
# Se**(1/m) itself.
u = m * log1p(-(Se ** (1 / m)))

D = (
    (1 - m)
    * Ks
    / (alpha * m * (theta_range[1] - theta_range[0]))
    * Se**l
    * Se ** (-1 / m)
    * expm1(u) ** 2
    * sympy.exp(-u)
)

D = D.subs(Se, (theta - theta_range[0]) / (theta_range[1] - theta_range[0]))
# Reference: Van Genuchten (1980) Equation 11
# https://doi.org/10.2136/sssaj1980.03615995004400050002x
################################

print(functionstr(theta, D))
