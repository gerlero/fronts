import sympy

import itertools

def _derivative_names(var):
    yield "D"
    yield f"dD_d{var}"
    for n in itertools.count(start=2):
        yield f"d{n}D_d{var}{n}"

def functionstr(var, expr):
    """
    Return a string that defines a function ``D`` that can evaluate `expr` and
    its first two derivatives with respect to `var`.

    The returned definition uses common subexpression elimination (CSE).

    Parameters
    ----------
    var : `sympy.Symbol` or str
        The function's variable.
    expr : `sympy.Expression` or str or float
        SymPy-compatible expression. Any free symbols other than `var` will be
        taken as parameters that must be in scope when the returned code is
        executed. Use of special functions and constructs is not currently
        allowed.

    Returns
    -------
    str
        Python code that defines the function ``D``.
    """

    var = sympy.sympify(var)
    expr = sympy.sympify(expr)

    exprs = [expr]

    for n in range(1,3):
        exprs.append(exprs[-1].diff(var))

    xs, exprs = sympy.cse(exprs, optimizations='basic')

    appearance = {(None, expr): n for (n, expr) in enumerate(exprs)}

    for x in reversed(xs):
        appearance[x] = 2
        for other in appearance:
            if x[0] in other[1].free_symbols:
                appearance[x] = min(appearance[x], appearance[other])

    variable = {var}

    for x in xs:
        if (x[1].free_symbols & variable):
            variable.add(x[0])

    lines = \
     ["# - Code generated with functionstr() from ../symbolic/generate.py - #"]

    for x in xs:
        if x[0] not in variable:
            lines.append("{} = {}".format(*x))

    lines.append(f"def D({var}, derivatives=0):")


    deriv_names = [name for name,_ in zip(_derivative_names(var), range(3))]

    for n, (name, expr) in enumerate(zip(deriv_names, exprs)):
        for x in xs:
            if x[0] in variable and appearance[x] == n:
                lines.append("    {} = {}".format(*x))
        lines.append(f"    {name} = {expr}")
        lines.append(f"    if derivatives == {n}: return {', '.join(deriv_names[:n+1])}")

    lines.append(
        '    raise ValueError("derivatives must be 0, 1 or 2")')

    lines.append(
      "# ----------------------- End generated code ----------------------- #")

    return "\n".join(lines)