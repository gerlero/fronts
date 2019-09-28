from __future__ import division, absolute_import, print_function

class BadBracket(ValueError):
    """
    Exception raised by `bisect` when a given bracket does not contain a root.
    """

class IterationLimitReached(RuntimeError):
    """
    Exception raised by `bisect` when the convergence criterion is not met
    within the specified number of iterations.
    """

class _BisectResults(object):
    def __init__(self, root, residual, bracket, iterations, function_calls):
        self.root = root
        self.residual = residual
        self.bracket = bracket
        self.iterations = iterations
        self.function_calls = function_calls


def bisect(f, bracket, ftol=1e-12, maxiter=100):
    """
    Find root of a function within a bracket using the bisection method.

    The function must have opposite signs at the endpoints of the bracket.

    Compared to SciPy's `scipy.optimize.bisect` function, this implementation
    defines convergence with respect to the residual (i.e., the value of `f`
    when evaluated at the found approximate root)

    Parameters
    ----------
    f : callable
        Continuous scalar function.
    bracket: A sequence of 2 floats
        An interval bracketing a root. `f` must have different signs at the
        two endpoints, or a BadBracket exception may be raised.
    ftol : float, optional
        Absolute tolerance for the residual. Must be nonnegative.
    maxiter : int, optional
        Maximum number of iterations. Must be nonnegative.

    Returns
    -------
    Object with the following fields defined:
    root : float
        Approximated root location.
    residual : float
        Value of the function evaluated at the root.
    bracket : A sequence of 2 floats
        Subinterval of the given bracket that contains the root. May be used in
        a later call with a smaller `ftol` to improve the approximation.
    iterations : int
        Number of iterations required to find the root.
    function_calls : int
        Number of times the function was called.
    """

    if ftol < 0:
        raise ValueError("ftol cannot be negative")

    if maxiter < 0:
        raise ValueError("maxiter cannot be negative")

    a, b = bracket

    # Check that the bracket is valid
    f_a = f(a)
    f_b = f(b)

    if f_a*f_b > 0:
        raise BadBracket("f must have different signs at the bracket "
                           "endpoints")

    # Check if we can just return one of the endpoints (not required, but nice)
    if abs(f_a) <= ftol or abs(f_b) <= ftol:
        if abs(f_a) <= abs(f_b):
            root, f_root = a, f_a
        else:
            root, f_root = b, f_b

        return _BisectResults(root=root, residual=f_root, bracket=(a,b),
                              iterations=0, function_calls=2)

    # Perform the actual bisection
    for i in range(maxiter):

        m = (a + b)/2
        f_m = f(m)

        if f_m*f_a > 0:
            a, f_a = m, f_m
        else:
            b, f_b = m, f_m

        if abs(f_m) <= ftol:
            return _BisectResults(root=m, residual=f_m, bracket=(a,b),
                                  iterations=i+1, function_calls=i+3)
    else:
        raise IterationLimitReached("failed to converge after {} iterations"
                                    .format(maxiter))

