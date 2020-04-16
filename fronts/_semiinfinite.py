"""
This module uses the Boltzmann transformation to deal with initial-boundary
value problems in semi-infinite domains.
"""

from __future__ import division, absolute_import, print_function
import six

from collections import namedtuple

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import PchipInterpolator

from ._boltzmann import ode, BaseSolution, r
from ._rootfinding import bracket_root, bisect, NotABracketError


class Solution(BaseSolution):
    r"""
    Continuous solution to a problem.

    A subclass of `BaseSolution`, its methods describe a continuous solution to
    a problem of finding a function `S` of `r` and `t` such that:

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

        super(Solution, self).__init__(sol=wrapped_sol, D=D)
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


class _Shooter(object):
    """
    Base shooter class.

    Parameters
    ----------
    D : callable
    Si : float
    radial : {False, 'cylindrical', 'polar', 'spherical'}
    ob : float
    S_direction : {-1, 0, 1}
    Si_tol : float
    max_shots : None or int
    shot_callback : None or callable

    Attributes
    ----------
    shots : int
        Number of calls to `shoot`.
    best_shot : None or Result
        Result that corresponds to the call to `shoot` that returned the lowest
        ``abs(Si_residual)``.
    """

    def __init__(self, D, Si, radial, ob, S_direction, Si_tol, max_shots,
                 shot_callback):

        assert not radial or ob > 0
        assert S_direction in {-1, 0, 1}
        assert max_shots is None or max_shots >= 0
        assert shot_callback is None or callable(shot_callback)

        self._fun, self._jac = ode(D, radial)
        self._Si = Si
        self._ob = ob
        self._S_direction = S_direction
        self._max_shots = max_shots
        self._shot_callback = shot_callback

        # Integration events
        def settled(o, y):
            return y[1]
        settled.terminal = True

        def blew_past_Si(o, y):
            return y[0] - (Si + S_direction*Si_tol)
        blew_past_Si.terminal = True

        self._events = (settled, blew_past_Si)

        self.shots = 0
        self.best_shot = None


    Result = namedtuple("Result", ['Sb', 
                                   'dS_dob',
                                   'Si_residual',
                                   'D_calls',
                                   'o',
                                   'sol'])

    def integrate(self, Sb, dS_dob):
        """
        Integrate and return the full result.

        Parameters
        ----------
        Sb : float
        dS_dob : float

        Returns
        -------
        Result
        """
        assert (self._Si-Sb)*self._S_direction >= 0
        assert dS_dob*self._S_direction >= 0

        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                ivp_result = solve_ivp(self._fun,
                                       t_span=(self._ob, np.inf),
                                       y0=(Sb, dS_dob),
                                       method='Radau',
                                       jac=self._jac,
                                       events=self._events,
                                       dense_output=True)

            except (ValueError, ArithmeticError, UnboundLocalError):
                # Catch D domain errors. Also catch UnboundLocalError caused by
                # https://github.com/scipy/scipy/issues/10775 (fixed in SciPy
                # v1.4.0; but we do not require that version because it does
                # not support Python 2.7)

                return self.Result(Sb=Sb,
                                   dS_dob=dS_dob,
                                   Si_residual=self._S_direction*np.inf,
                                   D_calls=None,
                                   o=None,
                                   sol=None)

        if ivp_result.success and ivp_result.t_events[0].size == 1:

            return self.Result(Sb=Sb,
                               dS_dob=dS_dob,
                               Si_residual=ivp_result.y[0,-1] - self._Si,
                               D_calls=ivp_result.nfev + ivp_result.njev,
                               o=ivp_result.t,
                               sol=ivp_result.sol)

        else:

            return self.Result(Sb=Sb,
                               dS_dob=dS_dob,
                               Si_residual=self._S_direction*np.inf,
                               D_calls=ivp_result.nfev + ivp_result.njev,
                               o=None,
                               sol=None)



    class ShotLimitReached(RuntimeError):
        """
        Exception raised when `shoot` is called after the maximum number of
        shots has been reached.
        """

    def shoot(self, *args, **kwargs):
        """
        Calls `integrate` and returns the result's `Si_residual`. Each call
        increments the number of shots.

        It raises a `ShotLimitReached` exception if the maximum number of
        allowed shots has been reached.

        Parameters are passed through to `result`. After the call, the callback
        provided to the constructor is invoked (if any).

        Parameters
        ----------
        *args, **kwargs
            Arguments passed through to `integrate`

        Returns
        -------
        Si_residual : float
        """
        self.shots += 1

        if self._max_shots is not None and self.shots > self._max_shots:
            raise self.ShotLimitReached

        result = self.integrate(*args, **kwargs)

        if self._shot_callback is not None:
            self._shot_callback(result)

        if (self.best_shot is None 
                or abs(result.Si_residual) < abs(self.best_shot.Si_residual)):
            self.best_shot = result

        return result.Si_residual


class _DirichletShooter(_Shooter):
    """
    Shooter for Dirichlet problems.

    Parameters
    ----------
    D : callable
    Si : float
    Sb : float
    radial : {False, 'cylindrical', 'polar', 'spherical'}
    ob : float
    Si_tol : float
    max_shots : None or int
    shot_callback : None or callable
    """

    def __init__(self, D, Si, Sb, radial, ob, Si_tol, max_shots, 
                 shot_callback):

        S_direction = np.sign(Si-Sb)

        super(_DirichletShooter, self).__init__(D=D,
                                                Si=Si,
                                                radial=radial,
                                                ob=ob,
                                                S_direction=S_direction,
                                                Si_tol=Si_tol,
                                                max_shots=max_shots,
                                                shot_callback=shot_callback)

        self._Sb = Sb


    def integrate(self, dS_dob):
        """
        Integrate and return the full result.

        Parameters
        ----------
        dS_dob : float

        Returns
        -------
        Result
        """
        return super(_DirichletShooter, self).integrate(Sb=self._Sb,
                                                        dS_dob=dS_dob)


def solve(D, Si, Sb, radial=False, ob=0.0, Si_tol=1e-3, dS_dob_hint=None,
          dS_dob_bracket=None, maxiter=100, verbose=0):
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
        values. It can be called as ``D(S)`` to evaluate it at `S`. It can also
        be called as ``D(S, n)`` with `n` equal to 1 or 2, in which case the
        first `n` derivatives of the function evaluated at the same `S` are
        included (in order) as additional return values. While mathematically a
        scalar function, `D` operates in a vectorized fashion with the same
        semantics when `S` is a `numpy.ndarray`.
    Si : float
        :math:`S_i`, the initial value of `S` in the domain.
    Sb : float
        :math:`S_b`, the value of `S` imposed at the boundary.
    radial : {False, 'cylindrical', 'polar', 'spherical'}, optional
        Choice of coordinate unit vector :math:`\mathbf{\hat{r}}`. Must be one
        of the following:

            *   `False` (default)
                :math:`\mathbf{\hat{r}}` is any coordinate unit vector in
                rectangular (Cartesian) coordinates, or an axial unit vector in
                a cylindrical coordinate system
            *   ``'cylindrical'`` or ``'polar'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                cylindrical or polar coordinate system
            *   ``'spherical'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                spherical coordinate system
    ob : float, optional
        :math:`o_b`, which determines the behavior of the boundary. The default
        is zero, which implies that the boundary always exists at :math:`r=0`.
        It must be strictly positive if `radial` is not `False`. Be aware that
        a non-zero value implies a moving boundary.
    Si_tol : float, optional
        Absolute tolerance for :math:`S_i`.
    dS_dob_hint : None or float, optional
        Optional hint to the solver. If given, it should be a number close to
        the expected value of the derivative of `S` with respect to the
        Boltzmann variable `o` (i.e., :math:`dS/do`) at the boundary in the
        solution to be found. This parameter is typically not needed.
    dS_dob_bracket : None or sequence of two floats
        Optional search interval that brackets the value of :math:`dS/do` at
        the boundary in the solution. If given, the solver will use bisection
        to find a solution in which :math:`dS/do` falls inside that interval (a
        `ValueError` will be raised for an incorrect interval). This parameter
        cannot be passed together with a `dS_dob_hint`. It is also not needed
        in typical usage.
    maxiter : int, optional
        Maximum number of iterations. A `RuntimeError` will be raised if the
        specified tolerance is not achieved within this number of iterations.
        Must be nonnegative.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity. Must be one of the following:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    solution : Solution
        See `Solution` for a description of the solution object.
        Additional fields specific to this solver are included in the object:

            *   `o` *(numpy.ndarray, shape (n,))*
                Final solver mesh, in terms of the Boltzmann variable `o`.
            *   `niter` *(int)*
                Number of iterations required to find the solution.
            *   `dS_dob_bracket` *(sequence of two floats or None)*
                If available, an interval that contains the value of
                :math:`dS/do` at the boundary in the solution. May be used as
                the input `dS_dob_bracket` in a subsequent call with a smaller
                `Si_tol` for the same problem in order to avoid reduntant
                iterations. Whether this interval is available or not depends
                on the strategy used internally by the solver; in particular,
                this field is never `None` if a `dS_dob_bracket` is passed when
                calling the function.

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
    repeatedly with the 'Radau' method as implemented in the `scipy.integrate`
    module and a custom shooting algorithm. The boundary condition is satisfied
    exactly as the starting point, and the algorithm iterates with different
    values of :math:`dS/do` at the boundary until it finds the solution that
    also satisfies the initial condition within the specified tolerance. Trial
    values of :math:`dS/do` at the boundary are selected automatically by
    default (taking into account an optional hint if
    passed by the user), or by bisecting an optional search interval. This
    scheme assumes that :math:`dS/do` at the boundary varies continuously with
    :math:`S_i`.
    """
    if radial and ob <= 0:
        raise ValueError("ob must be positive when using a radial coordinate")

    if maxiter < 0:
        raise ValueError("maxiter must not be negative")

    if dS_dob_bracket is not None:
        if dS_dob_hint is not None:
            raise TypeError("cannot pass both dS_dob_hint and dS_dob_bracket")

        dS_dob_bracket = tuple(x if np.sign(x) == np.sign(Si-Sb) else 0
                               for x in dS_dob_bracket)

    elif dS_dob_hint is None:
        dS_dob_hint = (Si-Sb)/(2*D(Sb)**0.5)

    elif np.sign(dS_dob_hint) != np.sign(Si-Sb):
        raise ValueError("sign of dS_dob_hint does not match direction given "
                         "by Sb and Si")

    if verbose >= 2:
        print("{:^15}{:^15}{:^15}{:^15}".format(
               "Iteration",
               "Si residual",
               "dS/do|b",
               "Calls to D"))

        def shot_callback(result):
            if np.isfinite(result.Si_residual):
                print("{:^15}{:^15.2e}{:^15.7e}{:^15}".format(
                       shooter.shots,
                       result.Si_residual,
                       result.dS_dob,
                       result.D_calls))
            else:
                print("{:^15}{:^15}{:^15.7e}{:^15}".format(
                       shooter.shots,
                       "*",
                       result.dS_dob,
                       result.D_calls or "*"))
    else:
        shot_callback = None

    shooter = _DirichletShooter(D=D,
                                Si=Si,
                                Sb=Sb,
                                radial=radial,
                                ob=ob,
                                Si_tol=Si_tol,
                                max_shots=maxiter,
                                shot_callback=shot_callback)

    try:

        if dS_dob_bracket is None:
            if Si == Sb:
                dS_dob = 0
                dS_dob_bracket = (0, 0)

            else:
                dS_dob_result = bracket_root(shooter.shoot,
                                             interval=(0, dS_dob_hint),
                                             f_interval=(Sb-Si, None),
                                             ftol=Si_tol,
                                             maxiter=None)

                dS_dob = dS_dob_result.root
                dS_dob_bracket = dS_dob_result.bracket
                f_bracket = dS_dob_result.f_bracket

        else:
            assert dS_dob_hint is None
            dS_dob = None
            f_bracket = tuple(Sb-Si if x==0 else None for x in dS_dob_bracket)


        if dS_dob is None:
            try:
                dS_dob_result = bisect(shooter.shoot,
                                       bracket=dS_dob_bracket,
                                       f_bracket=f_bracket,
                                       ftol=Si_tol,
                                       maxiter=None)

                dS_dob = dS_dob_result.root
                dS_dob_bracket = dS_dob_result.bracket
                f_bracket = dS_dob_result.f_bracket

            except NotABracketError:
                assert dS_dob_hint is None
                if verbose:
                    print("dS_dob_bracket does not contain target dS/do at ob."
                          " Try again with a correct interval.")
                six.raise_from(
                    ValueError("dS_dob_bracket does not contain target dS/do "
                               "at ob"),
                    None)

    except shooter.ShotLimitReached:
        if verbose:
          print("The solver did not converge after {} iterations.".format(
                maxiter))
        six.raise_from(
            RuntimeError("The solver did not converge after {} iterations."
                         .format(maxiter)),
            None)

    if shooter.best_shot is not None and shooter.best_shot.dS_dob == dS_dob:
        result = shooter.best_shot
    else:
        result = shooter.integrate(dS_dob=dS_dob)

    if verbose:
        print("Solved in {} iterations.".format(shooter.shots))
        print("Si residual: {:.2e}".format(result.Si_residual))
        if dS_dob_bracket is not None:
            print("dS/do at ob: {:.7e} (bracket: [{:.7e}, {:.7e}])".format(
                  dS_dob, min(dS_dob_bracket), max(dS_dob_bracket)))
        else:
            print("dS/do at ob: {:.7e}".format(dS_dob))

    solution = Solution(sol=result.sol,
                        ob=result.o[0],
                        oi=result.o[-1],
                        D=D)

    solution.o = result.o
    solution.niter = shooter.shots
    solution.dS_dob_bracket = dS_dob_bracket

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

    Alternative to the main `solve` function. This function requires a starting
    mesh and guess of the solution. It is significantly less robust than
    `solve`, and will fail to converge in many cases that the latter can easily
    handle (whether it converges will usually depend heavily on the problem,
    the starting mesh and the guess of the solution; it will raise a
    `RuntimeError` on failure). However, when it converges it is usually faster
    than `solve`, which may be an advantage for some use cases. You should
    nonetheless prefer `solve` unless you have a particular use case for which
    you have found this function to be better.

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
        values. It can be called as ``D(S)`` to evaluate it at `S`. It can also
        be called as ``D(S, n)`` with `n` equal to 1 or 2, in which case the
        first `n` derivatives of the function evaluated at the same `S` are
        included (in order) as additional return values. While mathematically a
        scalar function, `D` operates in a vectorized fashion with the same
        semantics when `S` is a `numpy.ndarray`.
    Si : float
        :math:`S_i`, the initial value of `S` in the domain.
    Sb : float
        :math:`S_b`, the value of `S` imposed at the boundary.
    o_guess : numpy.array_like, shape (n_guess,)
        Starting mesh in terms of the Boltzmann variable `o`. Must be strictly
        increasing. ``o_guess[0]`` is :math:`o_b`, which determines the
        behavior of the boundary. If zero, it implies that the boundary always
        exists at :math:`r=0`. It must be strictly positive if `radial` is not
        `False`. Be aware that a non-zero value implies a moving boundary. On
        the other end, ``o_guess[-1]`` must be large enough to contain the
        solution to the semi-infinite problem.
    S_guess : float or numpy.array_like, shape (n_guess,)
        Starting guess of `S` at the points in `o_guess`. If a single value,
        the guess is assumed uniform.
    radial : {False, 'cylindrical', 'polar', 'spherical'}, optional
        Choice of coordinate unit vector :math:`\mathbf{\hat{r}}`. Must be one
        of the following:

            *   `False` (default)
                :math:`\mathbf{\hat{r}}` is any coordinate unit vector in
                rectangular (Cartesian) coordinates, or an axial unit vector in
                a cylindrical coordinate system
            *   ``'cylindrical'`` or ``'polar'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                cylindrical or polar coordinate system
            *   ``'spherical'``
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
    solution : Solution
        See `Solution` for a description of the solution object.
        Additional fields specific to this solver are included in the object:

            *   `o` *(numpy.ndarray, shape (n,))*
                Final solver mesh, in terms of the Boltzmann variable o.
            *   `niter` *(int)*
                Number of iterations required to find the solution.
            *   `rms_residuals` *(numpy.ndarray, shape (n-1,))*
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
    with SciPy's collocation-based boundary value problem solver
    `scipy.integrate.solve_bvp` and a two-point Dirichlet condition that
    matches the boundary and initial conditions of the problem. Upon that
    solver's convergence, it runs a final check on whether the candidate
    solution also satisfies the semi-infinite condition (which implies
    :math:`dS/do\to0` as :math:`o\to\infty`).
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

    with np.errstate(divide='ignore', invalid='ignore'):
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

    solution = Solution(sol=bvp_result.sol,
                        ob=bvp_result.x[0],
                        oi=bvp_result.x[-1],
                        D=D)

    solution.o = bvp_result.x
    solution.niter = bvp_result.niter
    solution.rms_residuals = bvp_result.rms_residuals

    return solution


def inverse(o, S):
    r"""
    Solve an inverse problem.

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

    `S` is taken as its values on a discrete set of points expressed in terms
    of the Boltzmann variable. Problems in radial coordinates are not
    supported.

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
        values. It can be called as ``D(S)`` to evaluate it at `S`. It can also
        be called as ``D(S, n)`` with `n` equal to 1 or 2, in which case the
        first `n` derivatives of the function evaluated at the same `S` are
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

    While very fast, the scheme used by this function is somewhat limited in
    its practical precision because of the use of interpolation (see the Notes)
    and the fact that two `S` functions that differ little in their values may
    actually be the consequence of very different `D` functions. If the goal is
    to find the parameters for a parameterized `D`, you may opt to perform an
    optimization run using `solve` instead.

    Depending on the number of points, the returned `D` may take orders of
    magnitude more time to be evaluated than an analytical function. In that
    case, you may notice that solvers work significantly slower when called
    with this `D`.

    This function also works if the problem has different boundary condition,
    as long as it is compatible with the Boltzmann transformation so that
    `S` can be considered a function of `o` only.
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
