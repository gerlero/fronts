import itertools
from collections.abc import Generator

import sympy  # type: ignore [import-untyped]


def _derivative_names(var: str) -> Generator[str, None, None]:
    yield "D"
    yield f"dD_d{var}"
    for n in itertools.count(start=2):
        yield f"d{n}D_d{var}{n}"


def functionstr(var: sympy.Symbol | str, expr: sympy.Expr | str | float) -> str:
    """
    Return a string that defines a function ``D`` that can evaluate `expr` and
    its first two derivatives with respect to `var`.

    The returned definition uses common subexpression elimination (CSE).

    Parameters
    ----------
    var : `sympy.Symbol` or str
        The function's variable.
    expr : `sympy.Expr` or str or float
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

    for _ in range(1, 3):
        exprs.append(exprs[-1].diff(var))

    xs, exprs = sympy.cse(exprs, optimizations="basic")

    appearance = {(None, expr): n for (n, expr) in enumerate(exprs)}

    for x in reversed(xs):
        appearance[x] = 2
        for other, n in appearance.items():
            if x[0] in other[1].free_symbols:
                appearance[x] = min(appearance[x], n)

    variable = {var}

    for x in xs:
        if x[1].free_symbols & variable:
            variable.add(x[0])

    lines = ["# - Code generated with functionstr() from ../symbolic/generate.py - #"]

    lines.extend("{} = {}".format(*x) for x in xs if x[0] not in variable)

    lines.append(f"def D({var}, derivatives=0):")

    deriv_names = [
        name for name, _ in zip(_derivative_names(var), range(3), strict=False)
    ]

    for n, (name, expr) in enumerate(zip(deriv_names, exprs, strict=True)):
        for x in xs:
            lines.extend(
                "    {} = {}".format(*x)
                for x in xs
                if x[0] in variable and appearance[x] == n
            )
        lines.append(f"    {name} = {expr}")
        lines.append(
            f"    if derivatives == {n}: return {', '.join(deriv_names[: n + 1])}"
        )

    lines.append('    raise ValueError("derivatives must be 0, 1 or 2")')

    lines.append(
        "# ----------------------- End generated code ----------------------- #"
    )

    return "\n".join(lines)
