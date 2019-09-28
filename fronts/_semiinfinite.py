"""
This module uses the Boltzmann transformation to deal with initial-boundary
value problems in semi-infinite domains.
"""

from __future__ import division, absolute_import, print_function

import itertools

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import PchipInterpolator

from ._boltzmann import ode, Solution, r
from ._util import bisect, BadBracket, IterationLimitReached


class SemiInfiniteSolution(Solution):
    r"""
    Continuous solution to a semi-infinite problem.

    Its methods describe a continuous solution to a problem of finding a
    function `S` of `r` and `t` such that:

    .. math::
         \dfrac{\partial S}{\partial t} = \nabla\cdot\left[D\left(S\right)
                        \dfrac{\partial S}{\partial r}\mathbf{\hat{r}}\right]

    with `r` bounded at :math:`r_b(t)=o_b\sqrt t` on the left and unbounded to
    the right. For :math:`r<r_b(t)`, the methods will evaluate to NaNs.

    Parameters
    ----------
    sol : callable
        Solution to the corresponding ODE obtained with `ode`. For any `o` in
        the closed interval [`ob`, `oi`], ``sol(o)[0]`` is the value of
        `S` at `o`, and ``sol(o)[1]`` is the value of the derivative
        :math:`dS/do` at `o`. `sol` will only be evaluated in this interval.
    ob : float
        :math:`o_b`, which determines the behavior of the boundary.
    oi : float
        Value of the Boltzmann variable at which the solution can be considered
        to be equal to the initial condition. Must be :math:`\geq o_b`.
    D : callable
        `D` used to obtain `sol`. Must be the same function that was passed to
        `ode`.

    See also
    --------
    solve
    solve_from_guess
    ode
    """
    def __init__(self, sol, ob, oi, D):
        if ob > oi:
            raise ValueError("ob cannot be greater than oi")

        def wrapped_sol(o):
            under = np.less(o, ob)
            over = np.greater(o, oi)

            o = np.where(under|over, oi, o)

            y = sol(o)
            y[:,under] = np.nan
            y[1:,over] = 0

            return y

        super(SemiInfiniteSolution, self).__init__(sol=wrapped_sol, D=D)
        self._ob = ob
        self._oi = oi

    def rb(self, t):
        """
        :math:`r_b`, the location of the boundary.

        This is the point where the boundary condition of the problem is
        imposed.

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s). Values must be positive.

        Returns
        -------
        rb : float or numpy.ndarray
            The return is of the same type and shape as `t`.

        Notes
        -----
        Depending on :math:`o_b`, the boundary may be fixed at :math:`r=0` or
        it may move with time.
        """
        return r(o=self._ob, t=t)


def solve(D, Si, Sb, dS_dob_bracket=(-1.0, 1.0), radial=False, ob=0.0,
          Si_tol=1e-6, maxiter=100, verbose=0):
    r"""
    Solve an instance of the general problem.

    Given a positive function `D`, scalars :math:`S_i`, :math:`S_b` and
    :math:`o_b`, and coordinate unit vector :math:`\mathbf{\hat{r}}`, finds a
    function `S` of `r` and `t` such that:

    .. math:: \begin{cases} \dfrac{\partial S}{\partial t} =
        \nabla\cdot\left[D\left(S\right)\dfrac{\partial S}{\partial r}
        \mathbf{\hat{r}}\right ] & r>r_b(t),t>0\\
        S(r, 0) = S_i & r>0 \\
        S(r_b(t), t) = S_b & t>0 \\
        r_b(t) = o_b\sqrt t
        \end{cases}

    Parameters
    ----------
    D : callable
        Twice-differentiable function that maps the range of `S` to positive
        values. It can be called as ``D(S)`` to evaluate it at `S`. It can
        also be called as ``D(S, n)`` with `n` equal to 1 or 2, in which case
        the first `n` derivatives of the function evaluated at the same `S` are
        included (in order) as additional return values. While mathematically a
        scalar function, `D` operates in a vectorized fashion with the same
        semantics when `S` is a `numpy.ndarray`.
    Si : float
        :math:`S_i`, the initial value of `S` in the domain.
    Sb : float
        :math:`S_b`, the value of `S` imposed at the boundary.
    dS_dob_bracket : (float, float), optional
        Search interval that contains the value of the derivative of `S` with
        respect to the Boltzmann variable `o` (i.e., :math:`dS/do`) at the
        boundary in the solution. The interval can be made as wide as desired,
        at the cost of additional iterations required to obtain the solution.
        To refine a solution obtained previously with this same function, pass
        in that solution's final `dS_dob_bracket`. This parameter is always
        checked and a `ValueError` is raised if a `dS_dob_bracket` is found not
        to be valid for the problem.
    radial : {False, 'cylindrical', 'spherical'}, optional
        Choice of coordinate unit vector :math:`\mathbf{\hat{r}}`. Must be one
        of the following:

            * `False` (default)
                :math:`\mathbf{\hat{r}}` is any coordinate unit vector in
                rectangular (Cartesian) coordinates, or an axial unit vector in
                a cylindrical coordinate system
            * ``'cylindrical'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                cylindrical coordinate system
            * ``'spherical'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                spherical coordinate system
    ob : float, optional
        :math:`o_b`, which determines the behavior of the boundary. The default
        is zero, which implies that the boundary always exists at :math:`r=0`.
        It must be strictly positive if `radial` is not `False`. Be aware that
        a non-zero value implies a moving boundary.
    Si_tol : float, optional
        Absolute tolerance for :math:`S_i`.
    maxiter : int, optional
        Maximum number of iterations. A `RuntimeError` will be raised if the
        specified tolerance is not achieved within this number of iterations.
        Must be >= 2.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity. Must be one of the following:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    solution : SemiInfiniteSolution
        See `SemiInfiniteSolution` for a description of the solution object.
        Additional fields specific to this solver are included in the object:

            * `o` : numpy.ndarray, shape (n,)
                Final solver mesh, in terms of the Boltzmann variable `o`.
            * `niter` : int
                Number of iterations required to find the solution.
            * `dS_dob_bracket` : (float, float)
                Subinterval of `dS_dob_bracket` that contains the value of
                :math:`dS/do` at the boundary in the solution. May be used in a
                subsequent call with a smaller `Si_tol` to avoid reduntant
                iterations if wanting to refine a previously obtained solution.

    See also
    --------
    solve_from_guess

    Notes
    -----
    Given the expression of :math:`r_b` which specifies the location of the
    boundary, a fixed boundary can be had only if :math:`o_b=0`. Any other
    :math:`o_b` implies a moving boundary. This restriction affects radial
    problems in particular.

    This function works by transforming the partial differential equation with
    the Boltzmann transformation using `ode` and then solving the resulting ODE
    repeateadly using the 'Radau' method as implemented in
    `scipy.integrate.solve_ivp`. The boundary condition is satisfied exactly as
    the starting point, and the algorithm iterates with different values of
    :math:`dS/do` at the boundary (chosen from within `dS_dob_bracket` using
    bisection) until it finds the solution that also satisfies the initial
    condition with the specified tolerance. This scheme assumes that
    :math:`dS/do` at the boundary varies continuously with :math:`S_i`.
    """
    direction = np.sign(Si - Sb)

    # Clip dS_dob_bracket according to the search direction
    dS_dob_bracket = [x if x*direction>0 else 0.0 for x in dS_dob_bracket]

    if radial and ob <= 0:
        raise ValueError("ob must be positive when using a radial coordinate")

    if maxiter < 2:
        raise ValueError("maxiter must be >= 2")

    fun, jac = ode(D=D, radial=radial)

    # Integration events
    def settled(o, S):
        return S[1]
    settled.terminal = True

    def blew_past_Si(o, S):
        return S[0] - (Si + direction*Si_tol)
    blew_past_Si.terminal = True

    # Integration data
    counter = itertools.count(start=1)
    saved_integration = {}

    # Integration function - returns the Si residual
    def integrate(dS_dob, verbose=verbose):

        try:
            ivp_result = solve_ivp(fun, t_span=(ob, np.inf), y0=(Sb, dS_dob),
                                   method='Radau', jac=jac,
                                   events=(settled, blew_past_Si),
                                   dense_output=True)

        except (ValueError, ArithmeticError, UnboundLocalError):
            # Catch D domain errors. Also catch UnboundLocalError caused by
            # https://github.com/scipy/scipy/issues/10775

            Si_residual = direction*np.inf

            if verbose >= 2:
                print("{:^15}{:^15}{:^15}{:^15.5e}".format(
                       next(counter), "*",  "*", dS_dob))

        else:
            if ivp_result.success and ivp_result.t_events[0].size == 1:

                saved_integration['dS_dob'] = dS_dob
                saved_integration['o'] = ivp_result.t
                saved_integration['sol'] = ivp_result.sol

                Si_residual = ivp_result.y[0,-1] - Si

                if verbose >= 2:
                    print("{:^15}{:^15}{:^15.2e}{:^15.5e}".format(
                           next(counter), ivp_result.nfev, Si_residual,
                           dS_dob))

            else:
                Si_residual = direction*np.inf

                if verbose >= 2:
                    print("{:^15}{:^15}{:^15}{:^15.5e}".format(
                           next(counter), ivp_result.nfev, "*", dS_dob))

        return Si_residual


    if verbose >= 2:
        print("{:^15}{:^15}{:^15}{:^15}".format(
              "Iteration", "Evaluations", "Si residual", "dS/do at ob"))

    try:  # Find the dS_dob that makes the initial condition hold
        bisect_result = bisect(integrate, bracket=dS_dob_bracket, ftol=Si_tol,
                               maxiter=maxiter-2)
    except BadBracket:
        if verbose:
            print("dS_dob_bracket does not contain target dS/do at ob. Try "
                  "again with a correct dS_dob_bracket.")
        raise ValueError("dS_dob_bracket does not contain target dS/do at ob")

    except IterationLimitReached:
        if verbose:
            print("The solver did not converge after {} iterations.".format(
                  maxiter))
        raise RuntimeError("solver did not converge after {} iterations."
                           .format(maxiter))

    if verbose:
        print("Solved in {} iterations.".format(bisect_result.function_calls))
        print("Si residual: {:.2e}".format(bisect_result.residual))
        print("dS/do at ob: {:.5e} (bracket: [{:.5e}, {:.5e}])".format(
              bisect_result.root,
              bisect_result.bracket[0], bisect_result.bracket[1]))

    if saved_integration['dS_dob'] != bisect_result.root:
        integrate(bisect_result.root, verbose=0)
    assert saved_integration['dS_dob'] == bisect_result.root

    solution = SemiInfiniteSolution(sol=saved_integration['sol'],
                                    ob=saved_integration['o'][0],
                                    oi=saved_integration['o'][-1],
                                    D=D)

    solution.o = saved_integration['o']
    solution.niter = bisect_result.function_calls
    solution.dS_dob_bracket = bisect_result.bracket

    return solution


def solve_from_guess(D, Si, Sb, o_guess, S_guess, radial=False, max_nodes=1000,
                    verbose=0):
    r"""
    Solve an instance of the general problem starting from a guess of the
    solution.

    Given a positive function `D`, scalars :math:`S_i`, :math:`S_b` and
    :math:`o_b`, and coordinate unit vector :math:`\mathbf{\hat{r}}`, finds a
    function `S` of `r` and `t` such that:

    .. math:: \begin{cases} \dfrac{\partial S}{\partial t} =
        \nabla\cdot\left[D\left(S\right)\dfrac{\partial S}{\partial r}
        \mathbf{\hat{r}}\right ] & r>r_b(t),t>0\\
        S(r, 0) = S_i & r>0 \\
        S(r_b(t), t) = S_b & t>0 \\
        r_b(t) = o_b\sqrt t
        \end{cases}

    This function requires an initial mesh and guess of the solution. It is
    significantly less robust than `solve`, and will fail to converge in many
    cases that the latter can easily handle (whether it converges will usually
    depend heavily on the problem, the initial mesh and the guess of the
    solution; it will raise a `RuntimeError` on failure). However, when it
    converges it is usually faster than `solve`, which may be an advantage for
    some use cases. You should nonetheless prefer `solve` unless you have a
    particular use case for which you have found this function to be better.

    Possible use cases include refining a solution (note that `solve` can do
    that too), optimization runs in which known solutions make good first
    approximations of solutions with similar parameters and every second of
    computing time counts, and in the implementation of other solving
    algorithms. In all these cases, `solve` should probably be used as a
    fallback for when this function fails.

    Parameters
    ----------
    D : callable
        Twice-differentiable function that maps the range of `S` to positive
        values. It can be called as ``D(S)`` to evaluate it at `S`. It can
        also be called as ``D(S, n)`` with `n` equal to 1 or 2, in which case
        the first `n` derivatives of the function evaluated at the same `S` are
        included (in order) as additional return values. While mathematically a
        scalar function, `D` operates in a vectorized fashion with the same
        semantics when `S` is a `numpy.ndarray`.
    Si : float
        :math:`S_i`, the initial value of `S` in the domain.
    Sb : float
        :math:`S_b`, the value of `S` imposed at the boundary.
    o_guess : numpy.array_like, shape (n_guess,)
        Initial mesh in terms of the Boltzmann variable `o`. Must be strictly
        increasing. ``o_guess[0]`` is :math:`o_b`, which determines the
        behavior of the boundary. If zero, it implies that the boundary always
        exists at :math:`r=0`. It must be strictly positive if `radial` is not
        `False`. Be aware that a non-zero value implies a moving boundary. On
        the other end, ``o_guess[-1]`` must be large enough to contain the
        solution to the semi-infinite problem.
    S_guess : float or numpy.array_like, shape (n_guess,)
        Initial guess of `S` at the points in `o_guess`. If a single value, the
        guess is interpreted as uniform.
    radial : {False, 'cylindrical', 'spherical'}, optional
        Choice of coordinate unit vector :math:`\mathbf{\hat{r}}`. Must be one
        of the following:

            * `False` (default)
                :math:`\mathbf{\hat{r}}` is any coordinate unit vector in
                rectangular (Cartesian) coordinates, or an axial unit vector in
                a cylindrical coordinate system
            * ``'cylindrical'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                cylindrical coordinate system
            * ``'spherical'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                spherical coordinate system
    max_nodes : int, optional
        Maximum allowed number of mesh nodes.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity. Must be one of the following:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    solution : SemiInfiniteSolution
        See `SemiInfiniteSolution` for a description of the solution object.
        Additional fields specific to this solver are included in the object:

            * o : numpy.ndarray, shape (n,)
                Final solver mesh, in terms of the Boltzmann variable o.
            * niter : int
                Number of iterations required to find the solution.
            * rms_residuals : numpy.ndarray, shape (n-1,)
                RMS values of the relative residuals over each mesh interval.

    See also
    --------
    solve

    Notes
    -----
    Given that the location of the boundary is expressed in terms of the
    Boltzmann variable, a fixed boundary can be had only if ``o_guess[0]`` is
    0. Any other  ``o_guess[0]`` implies a moving boundary. This restriction
    affects radial problems in particular.

    This function works by transforming the partial differential equation with
    the Boltzmann transformation using `ode` and then solving the resulting ODE
    with SciPy's boundary value problem solver `scipy.integrate.solve_bvp` and
    a two-point Dirichlet condition that matches the boundary and initial
    conditions of the problem. Upon that solver's convergence, it runs a final
    check on whether the candidate solution also satisfies the semi-infinite
    condition (which implies :math:`dS/do\to0` as :math:`o\to\infty`).
    """

    if radial and o_guess[0] <= 0:
        raise ValueError("o_guess[0] must be positive when using a radial "
                         "coordinate")

    if np.ndim(S_guess) == 0:
        S_guess = np.full_like(o_guess, fill_value=S_guess)

    dS_do_guess = np.gradient(S_guess, o_guess)

    fun, jac = ode(D=D, radial=radial)

    # Boundary conditions
    def bc(yb, yi):
        return (yb[0]-Sb, yi[0]-Si)

    dbc_dyb = np.array(((1, 0), (0, 0)))
    dbc_dyi = np.array(((0, 0), (1, 0)))
    def bc_jac(yb, yi):
        return dbc_dyb, dbc_dyi

    if verbose >= 2:
        print("Solving with solve_bvp")

    bvp_result = solve_bvp(fun, bc=bc, x=o_guess, y=(S_guess, dS_do_guess),
                           fun_jac=jac, bc_jac=bc_jac,
                           max_nodes=max_nodes, verbose=verbose)

    if not bvp_result.success:
        raise RuntimeError("solve_bvp did not converge: {}".format(
                            bvp_result.message))

    if abs(bvp_result.y[1,-1]) > 1e-6:
        if verbose:
            print("The given mesh is too small for the problem. Try again "
                  "after extending o_guess towards the right")

        raise RuntimeError("o_guess cannot contain solution")

    solution = SemiInfiniteSolution(sol=bvp_result.sol,
                                    ob=bvp_result.x[0],
                                    oi=bvp_result.x[-1],
                                    D=D)

    solution.o = bvp_result.x
    solution.niter = bvp_result.niter
    solution.rms_residuals = bvp_result.rms_residuals

    return solution


def inverse(o, S):
    r"""
    Solve the inverse problem.

    Given a function `S` of `r` and `t`, and scalars :math:`S_i`, :math:`S_b`
    and :math:`o_b`, finds a positive function `D` of the values of `S` such
    that:

    .. math:: \begin{cases} \dfrac{\partial S}{\partial t} =
        \dfrac{\partial}{\partial r}\left(D\left(S\right)\dfrac{\partial S}
        {\partial r}\right) & r>r_b(t),t>0\\
        S(r, 0) = S_i & r>0 \\
        S(r_b(t), t) = S_b & t>0 \\
        r_b(t) = o_b\sqrt t
        \end{cases}

    `S` is input via its values on a finite set of points expressed in terms of
    the Boltzmann variable. Problems in radial coordinates are not supported.

    Parameters
    ----------
    o : numpy.array_like, shape (n,)
        Points where `S` is known, expressed in terms of the Boltzmann
        variable. Must be strictly increasing.

    S : numpy.array_like, shape (n,)
        Values of the solution at `o`. Must be monotonic (either non-increasing
        or non-decreasing) and ``S[-1]`` must be :math:`S_i`.

    Returns
    -------
    D : callable
        Twice-differentiable function that maps the range of `S` to positive
        values. It can be called as ``D(S)`` to evaluate it at `S`. It can
        also be called as ``D(S, n)`` with `n` equal to 1 or 2, in which case
        the first `n` derivatives of the function evaluated at the same `S` are
        included (in order) as additional return values. While mathematically a
        scalar function, `D` operates in a vectorized fashion with the same
        semantics when `S` is a `numpy.ndarray`.

    See also
    --------
    o

    Notes
    -----
    An `o` function of `S` is constructed by interpolating the input data with
    a PCHIP monotonic cubic spline. The function `D` is then constructed by
    applying the expressions that result from solving the Boltzmann-transformed
    equation for `D`.

    Depending on the number of points, the returned `D` may take orders of
    magnitude more time to be evaluated than an analytic function. In that
    case, you may notice that solvers work significantly slower when called
    with this `D`.
    """

    if not np.all(np.diff(o) > 0):
        raise ValueError("o must be strictly increasing")

    if not(np.all(np.diff(S) >= -1e-12) or np.all(np.diff(S) <= 1e-12)):
        raise ValueError("S must be monotonic")

    Si = S[-1]

    S, indices = np.unique(S, return_index=True)
    o = o[indices]

    o_func = PchipInterpolator(x=S, y=o)

    o_antiderivative_func = o_func.antiderivative()
    o_antiderivative_Si = o_func.antiderivative()(Si)

    o_funcs = [o_func.derivative(i) for i in range(4)]

    def D(S, derivatives=0):

        IodS = o_antiderivative_func(S) - o_antiderivative_Si

        do_dS = o_funcs[1](S)

        D = -(do_dS*IodS)/2

        if derivatives == 0: return D

        o = o_funcs[0](S)
        d2o_dS2 = o_funcs[2](S)

        dD_dS = -(d2o_dS2*IodS + do_dS*o)/2

        if derivatives == 1: return D, dD_dS

        d3o_dS3 = o_funcs[3](S)

        d2D_dS2 = -(d3o_dS3*IodS + 2*d2o_dS2*o + do_dS**2)/2

        if derivatives == 2: return D, dD_dS, d2D_dS2

        raise ValueError("derivatives must be 0, 1, or 2")

    return D


