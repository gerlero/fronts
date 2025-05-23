"""
Semi-infinite domain problems with the Boltzmann transformation.

This module uses the Boltzmann transformation to deal with initial-boundary
value problems in semi-infinite domains.
"""

from __future__ import annotations

import sys
from time import process_time
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, TypeVar

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from ._boltzmann import BaseSolution, ode, r
from ._rootfinding import NotABracketError, bisect, bracket_root
from .D import _D0, _checked, _ScalarD1, _ScalarD2, _VectorizedD2, from_expr

if TYPE_CHECKING:
    import sympy  # type: ignore[import-untyped]


class Solution(BaseSolution):
    r"""
    Solution to a problem.

    Represents a continuously differentiable function :math:`\theta` of `r` and
    `t` such that:

    .. math::
        \dfrac{\partial\theta}{\partial t} = \nabla\cdot\left[D(\theta)
                    \dfrac{\partial \theta}{\partial r}\mathbf{\hat{r}}\right]

    with `r` bounded at :math:`r_b(t)=o_b\sqrt t` on the left and unbounded to
    the right. For :math:`r<r_b(t)`, the methods will evaluate to NaNs.

    Parameters
    ----------
    sol : callable
        Solution to an ODE obtained with `ode`. For any float or
        one-dimensional NumPy array ``o`` with values in the closed interval
        [`ob`, `oi`], ``sol(o)[0]`` are the values of :math:`\theta` at ``o``,
        and ``sol(o)[1]`` are the values of the derivative :math:`d\theta/do`
        at `o``. `sol` will only be evaluated in this interval.
    ob : float
        Parameter :math:`o_b`, which determines the behavior of the boundary in
        the problem.
    oi : float
        Value of the Boltzmann variable at which the solution can be considered
        to be equal to the initial condition. Cannot be less than `ob`.
    D : callable
        Function to evaluate :math:`D` at arbitrary values of the solution.
        Must be callable with a float or NumPy array as its argument.
    """

    def __init__(
        self,
        sol: Callable[
            [float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]],
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        ],
        ob: float,
        oi: float,
        D: _D0,
    ) -> None:
        if ob > oi:
            msg = "ob cannot be greater than oi"
            raise ValueError(msg)

        def wrapped_sol(
            o: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
            under = np.less(o, ob)
            over = np.greater(o, oi)

            o = np.where(under | over, oi, o)

            y = np.asarray(sol(o))
            y[:, under] = np.nan
            y[1:, over] = 0

            return y

        super().__init__(sol=wrapped_sol, D=D)
        self._ob = ob
        self._oi = oi

    @property
    def i(self) -> float:
        """float: Initial value of the solution."""
        return self(o=self._oi)  # type: ignore[return-value]

    @property
    def ob(self) -> float:
        """float: Parameter :math:`o_b`."""
        return self._ob

    @property
    def oi(self) -> float:
        """
        float: Parameter :math:`o_i`.

        The value of the Boltzmann variable at which the solution can be
        considered to be equal to the initial condition.
        """
        return self._oi

    def rb(
        self, t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Boundary location.

        Returns :math:`r_b`, the location of the boundary.

        Depending on :math:`o_b`, the boundary may be fixed at :math:`r=0` or
        it may move with time.

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s). Values must not be negative.

        Returns
        -------
        rb : float or numpy.ndarray
            The return is of the same type and shape as `t`.
        """
        return r(o=self.ob, t=t)

    @property
    def b(self) -> float:
        """float: Boundary value of the solution."""
        return self(o=self.ob)  # type: ignore[return-value]

    def d_drb(
        self, t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Spatial derivative of the solution at the boundary.

        Evaluates and returns :math:`\partial\theta/\partial r|_b`. Equivalent
        to ``self.d_dr(self.rb(t), t)``.

        Parameters
        ----------
        t : float or numpy.ndarray, shape (n,)
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray, shape (n,)
        """
        return self.d_dr(self.rb(t), t)

    def d_dtb(
        self, t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Time derivative of the solution at the boundary.

        Evaluates and returns :math:`\partial\theta/\partial t|_b`. Equivalent
        to ``self.d_dt(self.rb(t), t)``.

        Parameters
        ----------
        t : float or numpy.ndarray, shape (n,)
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray, shape (n,)
        """
        return self.d_dt(self.rb(t), t)

    def fluxb(
        self, t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Boundary flux.

        Returns the diffusive flux of :math:`\theta` at the boundary, in the
        direction :math:`\mathbf{\hat{r}}`. Equivalent to
        ``self.flux(self.rb(t), t)``.

        Parameters
        ----------
        t : float or numpy.ndarray, shape (n,)
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray, shape (n,)
        """
        return self.flux(self.rb(t), t)

    @property
    def d_dob(self) -> float:
        """
        float: boundary Boltzmann derivative.

        Derivative of the solution with respect to the Boltzmann
        variable at the boundary.
        """
        return self.d_do(o=self.ob)  # type: ignore[return-value]

    def sorptivity(
        self,
        *,
        o: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None = None,
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Sorptivity.

        Returns the sorptivity :math:`S` of :math:`\theta`, equal to
        :math:`-2D(\theta)\partial\theta/\partial o`.

        Parameters
        ----------
        o : float or numpy.ndarray, shape (n,), optional
            Value(s) of the Boltzmann variable. If not given, the method
            will return the sorptivity at the boundary.

        Returns
        -------
        float or numpy.ndarray, shape (n,)

        References
        ----------
        [1] PHILIP, J. R. The theory of infiltration: 4. Sorptivity and
        algebraic infiltration equations. Soil Science, 1957, vol. 84, no. 3,
        pp. 257-264.
        """
        if o is None:
            o = self.ob
        return super().sorptivity(o=o)


R = TypeVar("R")


class _Shooter:
    """
    Base shooter class.

    Parameters
    ----------
    D : callable
    i : float
    radial : {False, 'cylindrical', 'polar', 'spherical'}
    ob : float
    theta_direction : {-1, 0, 1}
    itol : float
    method : {'implicit', 'explicit'}
    max_shots : None or int
    shot_callback : None or callable

    Attributes
    ----------
    shots : int
        Number of calls to `shoot`.
    best_shot : None or Result
        Result that corresponds to the call to `shoot` that returned the lowest
        ``abs(i_residual)``.
    """

    @staticmethod
    def _native_float_inputs(
        f: Callable[[float, tuple[float, float]], R],
    ) -> Callable[[float, tuple[float, float]], R]:
        """Speeds up arithmetic by converting NumPy inputs to native floats."""

        def wrapper(o: float, y: tuple[float, float]) -> R:
            return f(float(o), (float(y[0]), float(y[1])))

        return wrapper

    class Result(NamedTuple):
        b: float
        d_dob: float | None
        i_residual: float
        D_calls: int | None
        o: np.ndarray[tuple[int], np.dtype[np.floating]] | None
        sol: (
            Callable[
                [float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]],
                np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            ]
            | None
        )

    def __init__(
        self,
        D: _ScalarD2 | _ScalarD1,
        i: float,
        radial: Literal[False, "cylindrical", "polar", "spherical"],
        ob: float,
        theta_direction: int,
        itol: float,
        method: Literal["implicit", "explicit"],
        max_shots: int | None,
        shot_callback: Callable[[Result], None] | None,
    ) -> None:
        assert callable(D)
        assert not radial or ob > 0
        assert theta_direction in {-1, 0, 1}
        assert method in {"implicit", "explicit"}
        assert max_shots is None or max_shots >= 0
        assert shot_callback is None or callable(shot_callback)

        self._i = i
        self._ob = ob
        self._theta_direction = theta_direction
        self._method = method
        self._max_shots = max_shots
        self._shot_callback = shot_callback

        fun, jac = ode(D, radial=radial, catch_errors=True)
        self._fun = self._native_float_inputs(fun)
        if method == "implicit":
            self._jac = self._native_float_inputs(jac)

        self._checked_D = _checked(D)

        # Integration events
        def settled(
            o: float, y: np.ndarray[tuple[int], np.dtype[np.floating]]
        ) -> float:
            return y[1]  # type: ignore[no-any-return]

        settled.terminal = True  # type: ignore[attr-defined]

        def blew_past_i(
            o: float, y: np.ndarray[tuple[int], np.dtype[np.floating]]
        ) -> float:
            return y[0] - (i + theta_direction * itol)  # type: ignore[no-any-return]

        blew_past_i.terminal = True  # type: ignore[attr-defined]

        self._events = [settled, blew_past_i]

        self.shots = 0
        self.best_shot: _Shooter.Result | None = None

    def integrate(self, b: float, d_dob: float) -> Result:
        """
        Integrate and return the full result.

        Parameters
        ----------
        b : float
        d_dob : float

        Returns
        -------
        Result
        """
        assert (self._i - b) * self._theta_direction >= 0
        assert d_dob * self._theta_direction >= 0

        with np.errstate(divide="ignore", invalid="ignore"):
            try:
                if self._method == "explicit":
                    ivp_result = solve_ivp(
                        self._fun,
                        t_span=(self._ob, np.inf),
                        y0=(b, d_dob),
                        method="DOP853",
                        events=self._events,
                        dense_output=True,
                    )  # type: ignore[call-overload]
                else:
                    ivp_result = solve_ivp(
                        self._fun,
                        t_span=(self._ob, np.inf),
                        y0=(b, d_dob),
                        method="Radau",
                        jac=self._jac,
                        events=self._events,
                        dense_output=True,
                    )  # type: ignore[call-overload]
            except ValueError as e:
                # Workaround for https://github.com/scipy/scipy/issues/17066
                if str(e) == "`ts` must be strictly increasing or decreasing.":
                    return self.Result(
                        b=b,
                        d_dob=d_dob,
                        i_residual=self._theta_direction * np.inf,
                        D_calls=None,
                        o=None,
                        sol=None,
                    )
                raise

        if ivp_result.success and ivp_result.t_events[0].size == 1:
            return self.Result(
                b=b,
                d_dob=d_dob,
                i_residual=ivp_result.y[0, -1] - self._i,
                D_calls=ivp_result.nfev + ivp_result.njev,
                o=ivp_result.t,
                sol=ivp_result.sol,
            )

        return self.Result(
            b=b,
            d_dob=d_dob,
            i_residual=self._theta_direction * np.inf,
            D_calls=ivp_result.nfev + ivp_result.njev,
            o=None,
            sol=None,
        )

    class ShotLimitReached(RuntimeError):
        """
        Shot limit reached.

        Exception raised when `shoot` is called after the maximum number of
        shots has been reached.
        """

    def shoot(self, *args: Any, **kwargs: Any) -> float:  # noqa: ANN401
        """
        Call `integrate` and returns the result's `i_residual`.

        Each call increments the number of shots.

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
        i_residual : float
        """
        self.shots += 1

        if self._max_shots is not None and self.shots > self._max_shots:
            raise self.ShotLimitReached

        result = self.integrate(*args, **kwargs)

        if self._shot_callback is not None:
            self._shot_callback(result)

        if self.best_shot is None or abs(result.i_residual) < abs(
            self.best_shot.i_residual
        ):
            self.best_shot = result

        return result.i_residual


class _DirichletShooter(_Shooter):
    """
    Shooter for Dirichlet problems.

    Parameters
    ----------
    D : callable
    i : float
    b : float
    radial : {False, 'cylindrical', 'polar', 'spherical'}
    ob : float
    itol : float
    max_shots : None or int
    shot_callback : None or callable
    """

    def __init__(
        self,
        D: _ScalarD1 | _ScalarD2,
        i: float,
        b: float,
        radial: Literal[False, "cylindrical", "polar", "spherical"],
        ob: float,
        itol: float,
        method: Literal["implicit", "explicit"],
        max_shots: int | None,
        shot_callback: Callable[[_Shooter.Result], None] | None,
    ) -> None:
        theta_direction = np.sign(i - b)

        super().__init__(
            D=D,
            i=i,
            radial=radial,
            ob=ob,
            theta_direction=theta_direction,
            itol=itol,
            method=method,
            max_shots=max_shots,
            shot_callback=shot_callback,
        )

        self._b = b

        self._checked_D(b)
        if abs(i - b) > itol:
            self._checked_D(i - theta_direction * itol)

    def integrate(self, d_dob: float) -> _Shooter.Result:  # type: ignore[override]
        """
        Integrate and return the full result.

        Parameters
        ----------
        d_dob : float

        Returns
        -------
        Result
        """
        return super().integrate(b=self._b, d_dob=d_dob)


def solve(
    D: _ScalarD1 | _ScalarD2 | str | float,
    i: float,
    b: float,
    radial: Literal[False, "cylindrical", "polar", "spherical"] = False,
    ob: float = 0.0,
    itol: float = 1e-3,
    d_dob_hint: float | None = None,
    d_dob_bracket: tuple[float, float] | None = None,
    method: Literal["implicit", "explicit"] = "implicit",
    maxiter: int = 100,
    verbose: int = 0,
) -> Solution:
    r"""
    Solve a problem with a Dirichlet boundary condition.

    Given a positive function `D`, scalars :math:`\theta_i`, :math:`\theta_b`
    and :math:`o_b`, and coordinate unit vector :math:`\mathbf{\hat{r}}`, finds
    a function :math:`\theta` of `r` and `t` such that:

    .. math:: \begin{cases} \dfrac{\partial\theta}{\partial t} =
        \nabla\cdot\left[D(\theta)\dfrac{\partial\theta}{\partial r}
        \mathbf{\hat{r}}\right ] & r>r_b(t),t>0\\
        \theta(r, 0) = \theta_i & r>0 \\
        \theta(r_b(t), t) = \theta_b & t>0 \\
        r_b(t) = o_b\sqrt t
        \end{cases}

    Parameters
    ----------
    D : callable or `sympy.Expr` or str or float
        Callable that evaluates :math:`D` and its derivatives, obtained from
        the :mod:`fronts.D` module or defined in the same manner---i.e.:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        When called by this function, ``theta`` is always a single float.
        However, calls as ``D(theta)`` should also accept a NumPy array
        argument.

        Alternatively, instead of a callable, the argument can be the
        expression of :math:`D` in the form of a string or
        :class:`sympy.Expr` with a single variable. In this case, the
        solver will differentiate and evaluate the expression as necessary.
    i : float
        Initial condition, :math:`\theta_i`.
    b : float
        Imposed boundary value, :math:`\theta_b`.
    radial : {False, 'cylindrical', 'polar', 'spherical'}, optional
        Choice of coordinate unit vector :math:`\mathbf{\hat{r}}`. Must be one
        of the following:

            *   False (default):
                :math:`\mathbf{\hat{r}}` is any coordinate unit vector in
                rectangular (Cartesian) coordinates, or an axial unit vector in
                a cylindrical coordinate system
            *   ``'cylindrical'`` or ``'polar'``:
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                cylindrical or polar coordinate system
            *   ``'spherical'``:
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                spherical coordinate system
    ob : float, optional
        Parameter :math:`o_b`, which determines the behavior of the boundary.
        The default is zero, which means that the boundary always exists at
        :math:`r=0`. It must be strictly positive if `radial` is not False. A
        non-zero value implies a moving boundary.

    Returns
    -------
    solution : Solution
        See :class:`Solution` for a description of the solution object.
        Additional fields specific to this solver are included in the object:

            *   `o` *(numpy.ndarray, shape (n,))* --
                Final solver mesh, in terms of the Boltzmann variable.
            *   `niter` *(int)* --
                Number of iterations required to find the solution.
            *   `d_dob_bracket` *(sequence of two floats or None)* --
                If available, an interval that contains the value of
                :math:`d\theta/do|_b`. May be used as the input `d_dob_bracket`
                in a subsequent call with a smaller `itol` for the same problem
                in order to avoid reduntant iterations. Whether this interval
                is available or not depends on the strategy used internally by
                the solver; in particular, this field is never `None` if a
                `d_dob_bracket` is passed when calling the function.

    Other Parameters
    ----------------
    itol : float, optional
        Absolute tolerance for the initial condition.
    d_dob_hint : None or float, optional
        Optional hint to the solver. If given, it should be a number close to
        the expected value of the derivative of :math:`\theta` with respect to
        the Boltzmann variable at the boundary (i.e., :math:`d\theta/do|_b`) in
        the solution to be found. This parameter is typically not needed.
    d_dob_bracket : None or sequence of two floats, optional
        Optional search interval that brackets the value of
        :math:`d\theta/do|_b` in the solution. If given, the solver will use
        bisection to find a solution in which :math:`d\theta/do|_b` falls
        inside that interval (a `ValueError` will be raised for an incorrect
        interval). This parameter cannot be passed together with a
        `d_dob_hint`. It is also not needed in typical usage.
    method : {'implicit', 'explicit'}, optional
        Selects the integration method used by the solver:

            *   ``'implicit'`` (default):
                uses a Radau IIA implicit method of order 5. A sensible default
                choice that will work for any problem
            *   ``'explicit'``:
                uses the DOP853 explicit method of order 8. As an explicit
                method, it trades off general solver robustness and accuracy
                for faster results in "well-behaved" cases. With this method,
                the second derivative of :math:`D` is not needed
    maxiter : int, optional
        Maximum number of iterations. A `RuntimeError` will be raised if the
        specified tolerance is not achieved within this number of iterations.
        Must be nonnegative.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity. Must be one of the following:

            * 0 (default): work silently
            * 1: display a termination report
            * 2: also display progress during iterations

    See Also
    --------
    solve_from_guess
    solve_flowrate

    Notes
    -----
    This function works by transforming the partial differential equation with
    the Boltzmann transformation using :func:`ode` and then solving the
    resulting ODE repeatedly with the chosen integration method as implemented
    in the  :mod:`scipy.integrate` module and a custom shooting algorithm. The
    boundary condition is satisfied exactly as the starting point, and the
    algorithm iterates with different values of :math:`d\theta/do|_b` until it
    finds the solution that also verifies the initial condition within the
    specified tolerance. Trial values of :math:`d\theta/do|_b` are selected
    automatically by default (using heuristics, which can also take into
    account an optional hint if passed by the user), or by bisecting an
    optional search interval. This scheme assumes that :math:`d\theta/do|_b`
    varies continuously with :math:`\theta_i`.

    References
    ----------
    [1] GERLERO, G. S.; BERLI, C. L. A.; KLER, P. A. Open-source
    high-performance software packages for direct and inverse solving of
    horizontal capillary flow. Capillarity, 2023, vol. 6, no. 2, pp. 31-40.
    """
    if verbose:
        start_time = process_time()

    if radial and ob <= 0:
        msg = "ob must be positive when using a radial coordinate"
        raise ValueError(msg)

    if maxiter < 0:
        msg = "maxiter must not be negative"
        raise ValueError(msg)

    if not callable(D):
        D = from_expr(
            D, vectorized=False, max_derivatives=(1 if method == "explicit" else 2)
        )

    if d_dob_bracket is not None:
        if d_dob_hint is not None:
            msg = "cannot pass both d_dob_hint and d_dob_bracket"
            raise TypeError(msg)

        d_dob_bracket = tuple(
            x if np.sign(x) == np.sign(i - b) else 0 for x in d_dob_bracket
        )  # type: ignore[assignment]

    elif d_dob_hint is None:
        d_dob_hint = (i - b) / (2 * _checked(D, b) ** 0.5)

    elif np.sign(d_dob_hint) != np.sign(i - b):
        msg = "sign of d_dob_hint does not match direction given by b and i"
        raise ValueError(msg)

    if verbose >= 2:
        print(f"{'Iteration':^15}{'Residual':^15}{'d/do|b':^15}{'Calls to D':^15}")

        def shot_callback(result: _Shooter.Result) -> None:
            if np.isfinite(result.i_residual):
                print(
                    f"{shooter.shots:^15}{result.i_residual:^15.2e}{result.d_dob:^15.7e}{result.D_calls:^15}"
                )

            else:
                print(
                    f"{shooter.shots:^15}{'*':^15}{result.d_dob:^15.7e}{result.D_calls or '*':^15}"
                )
    else:
        shot_callback = None  # type: ignore[assignment]

    shooter = _DirichletShooter(
        D=D,
        i=i,
        b=b,
        radial=radial,
        ob=ob,
        itol=itol,
        method=method,
        max_shots=maxiter,
        shot_callback=shot_callback,
    )

    try:
        if d_dob_bracket is None:
            if i == b:
                d_dob: float | None = 0.0
                d_dob_bracket = (0.0, 0.0)

            else:
                assert d_dob_hint is not None
                d_dob_result = bracket_root(
                    shooter.shoot,
                    interval=(0, d_dob_hint),
                    f_interval=(b - i, None),
                    ftol=itol,
                    maxiter=None,
                )

                d_dob = d_dob_result.root
                assert d_dob_result.bracket is not None
                d_dob_bracket = d_dob_result.bracket
                assert d_dob_result.f_bracket is not None
                f_bracket: tuple[float | None, float | None] = d_dob_result.f_bracket

        else:
            assert d_dob_hint is None
            d_dob = None
            f_bracket = tuple(b - i if x == 0 else None for x in d_dob_bracket)  # type: ignore[assignment]

        if d_dob is None:
            try:
                d_dob_result = bisect(
                    shooter.shoot,
                    bracket=d_dob_bracket,
                    f_bracket=f_bracket,
                    ftol=itol,
                    maxiter=None,
                )

                assert d_dob_result.root is not None
                d_dob = d_dob_result.root
                assert d_dob_result.bracket is not None
                d_dob_bracket = d_dob_result.bracket
                assert d_dob_result.f_bracket is not None
                f_bracket = d_dob_result.f_bracket

            except NotABracketError:
                assert d_dob_hint is None
                if verbose:
                    print(
                        "d_dob_bracket does not contain target d/do|b. Try "
                        "again with a correct interval."
                    )
                msg = "d_dob_bracket does not contain target d/do|b"
                raise ValueError(msg) from None

    except shooter.ShotLimitReached:
        if verbose:
            print(f"The solver did not converge after {maxiter} iterations.")
            print(f"Execution time: {process_time() - start_time:.3f} s")
        msg = f"The solver did not converge after {maxiter} iterations."
        raise RuntimeError(msg) from None

    if shooter.best_shot is not None and shooter.best_shot.d_dob == d_dob:
        result = shooter.best_shot
    else:
        result = shooter.integrate(d_dob=d_dob)

    assert result.sol is not None
    assert result.o is not None
    solution = Solution(sol=result.sol, ob=result.o[0], oi=result.o[-1], D=D)

    solution.o = result.o  # type: ignore[attr-defined]
    solution.niter = shooter.shots  # type: ignore[attr-defined]
    solution.d_dob_bracket = d_dob_bracket  # type: ignore[attr-defined]

    if verbose:
        print(f"Solved in {shooter.shots} iterations.")
        print(f"Residual: {result.i_residual:.2e}")
        if d_dob_bracket is not None:
            print(
                f"d/do|b: {d_dob:.7e} (bracket: [{min(d_dob_bracket):.7e}, {max(d_dob_bracket):.7e}])"
            )
        else:
            print(f"d/do|b: {d_dob:.7e}")
        print(f"Execution time: {process_time() - start_time:.3f} s")

    return solution


class _FlowrateShooter(_Shooter):
    """
    Shooter for problems with a fixed-flowrate boundary condition.

    Parameters
    ----------
    D : callable
    i : float
    rel_flowrate : float
        Flow rate per unit angle and (if applicable) height.
    ob : float
    itol : float
    max_shots : None or int
    shot_callback : None or callable
    """

    def __init__(
        self,
        D: _ScalarD2 | _ScalarD1,
        i: float,
        rel_flowrate: float,
        ob: float,
        itol: float,
        method: Literal["implicit", "explicit"],
        max_shots: int | None,
        shot_callback: Callable[[_Shooter.Result], None] | None,
    ) -> None:
        assert ob > 0

        theta_direction = np.sign(-rel_flowrate)

        super().__init__(
            D=D,
            i=i,
            ob=ob,
            radial="cylindrical",
            theta_direction=theta_direction,
            itol=itol,
            method=method,
            max_shots=max_shots,
            shot_callback=shot_callback,
        )

        self._D_ = D
        # Flow rate per unit angle and height
        self._rel_flowrate = rel_flowrate

    def integrate(self, b: float) -> _Shooter.Result:  # type: ignore[override]
        """
        Integrate and return the full result.

        Parameters
        ----------
        b : float

        Returns
        -------
        Result
        """
        try:
            Db = self._checked_D(b)
        except ValueError:
            return self.Result(
                b=b,
                d_dob=None,
                i_residual=-self._theta_direction * np.inf,
                D_calls=1,
                o=None,
                sol=None,
            )

        d_dob = -self._rel_flowrate / (Db * self._ob)

        result = super().integrate(b=b, d_dob=d_dob)

        D_calls = result.D_calls + 1 if result.D_calls is not None else None

        return result._replace(D_calls=D_calls)


def solve_flowrate(
    D: _ScalarD2 | _ScalarD1 | sympy.Expr | str | float,
    i: float,
    Qb: float,
    radial: Literal["cylindrical", "polar"],
    ob: float = 1e-6,
    angle: float = 2 * np.pi,
    height: float | None = None,
    itol: float = 1e-3,
    b_hint: float | None = None,
    b_bracket: tuple[float, float] | None = None,
    method: Literal["implicit", "explicit"] = "implicit",
    maxiter: int = 100,
    verbose: int = 0,
) -> Solution:
    r"""
    Solve a radial problem with a fixed-flowrate boundary condition.

    Given a positive function `D`, scalars :math:`\theta_i`, :math:`\theta_b`
    and :math:`o_b`, and coordinate unit vector :math:`\mathbf{\hat{r}}`, finds
    a function :math:`\theta` of `r` and `t` such that:

    .. math:: \begin{cases} \dfrac{\partial\theta}{\partial t} =
        \nabla\cdot\left[D(\theta)\dfrac{\partial\theta}{\partial r}
        \mathbf{\hat{r}}\right ] & r>r_b(t),t>0\\
        \theta(r,0) = \theta_i & r>0 \\
        Q(r_b(t),t) = Q_b & t>0 \\
        r_b(t) = o_b\sqrt t
        \end{cases}

    Parameters
    ----------
    D : callable or `sympy.Expr` or str or float
        Callable that evaluates :math:`D` and its derivatives, obtained from
        the :mod:`fronts.D` module or defined in the same manner---i.e.:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        where ``theta`` is always a float in the latter two cases, but it may
        be either a single float or a NumPy array when `D` is called as
        ``D(theta)``.

        Alternatively, instead of a callable, the argument can be the
        expression of :math:`D` in the form of a string or
        :class:`sympy.Expr` with a single variable. In this case, the
        solver will differentiate and evaluate the expression as necessary.
    i : float
        Initial condition, :math:`\theta_i`.
    Qb : float
        Imposed flow rate of :math:`\theta` at the boundary, :math:`Q_b`.

        The flow rate is considered in the direction of
        :math:`\mathbf{\hat{r}}`: a positive value means that :math:`\theta` is
        flowing into the domain; negative values mean that :math:`\theta` flows
        out of the domain.
    radial : {'cylindrical', 'polar'}
        Choice of coordinate unit vector :math:`\mathbf{\hat{r}}`. Must be one
        of the following:

            *   ``'cylindrical'`` :
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                cylindrical coordinate system
            *   ``'polar'`` :
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                polar coordinate system
    ob : float, optional
        Parameter :math:`o_b`, which determines the behavior of the boundary.
        It must be positive. The boundary acts as a line source or sink in the
        limit where `ob` tends to zero.
    angle : float, optional
        Total angle covered by the domain. The default is :math:`2\pi`, which
        means that :math:`\theta` may flow through the boundary in all
        directions. Must be positive and no greater than :math:`2\pi`.
    height : None or float, optional
        Axial height of the domain if ``radial=='cylindrical'``. Not allowed if
        ``radial=='polar'``.

    Returns
    -------
    solution : Solution
        See :class:`Solution` for a description of the solution object.
        Additional fields specific to this solver are included in the object:

            *   `o` *(numpy.ndarray, shape (n,))* --
                Final solver mesh, in terms of the Boltzmann variable.
            *   `niter` *(int)* --
                Number of iterations required to find the solution.
            *   `b_bracket` *(sequence of two floats or None)* --
                If available, an interval that contains the value of
                :math:`\theta_b`. May be used as the input `b_bracket` in a
                subsequent call with a smaller `itol` for the same problem in
                order to avoid reduntant iterations. Whether this interval is
                available or not depends on the strategy used internally by the
                solver; in particular, this field is never `None` if a
                `b_bracket` is passed when calling the function.

    Other Parameters
    ----------------
    itol : float, optional
        Absolute tolerance for the initial condition.
    b_hint : None or float, optional
        Optional hint to the solver. If given, it should be a number close to
        the expected value of :math:`\theta` at the boundary (i.e.
        :math:`\theta_b`) in the solution to be found.
    b_bracket : None or sequence of two floats, optional
        Optional search interval that brackets the value of :math:`\theta_b`
        in the solution. If given, the solver will use bisection to find a
        solution in which :math:`\theta_b` falls inside that interval (a
        `ValueError` will be raised for an incorrect interval). This parameter
        cannot be passed together with a `b_hint`.
    method : {'implicit', 'explicit'}, optional
        Selects the integration method used by the solver:

            *   ``'implicit'`` (default):
                uses a Radau IIA implicit method of order 5. A sensible default
                choice that will work for any problem
            *   ``'explicit'``:
                uses the DOP853 explicit method of order 8. As an explicit
                method, it trades off general solver robustness and accuracy
                for faster results in "well-behaved" cases. With this method,
                the second derivative of :math:`D` is not needed
    maxiter : int, optional
        Maximum number of iterations. A `RuntimeError` will be raised if the
        specified tolerance is not achieved within this number of iterations.
        Must be nonnegative.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity. Must be one of the following:

            * 0 (default): work silently
            * 1: display a termination report
            * 2: also display progress during iterations

    See Also
    --------
    solve

    Notes
    -----
    This function works by transforming the partial differential equation with
    the Boltzmann transformation using :func:`ode` and then solving the
    resulting ODE repeatedly with the chosen integration method as implemented
    in the :mod:`scipy.integrate` module and a custom shooting algorithm. The
    boundary condition is satisfied exactly as the starting point, and the
    algorithm iterates with different values of :math:`\theta` at the boundary
    until it finds the solution that also verifies the initial condition within
    the specified tolerance. Trial values of :math:`\theta` at the boundary are
    selected automatically by default (using heuristics, which can also take
    into account an optional hint if passed by the user), or by bisecting an
    optional search interval. This scheme assumes that :math:`\theta` at the
    boundary varies continuously with :math:`\theta_i`.

    References
    ----------
    [1] GERLERO, G. S.; BERLI, C. L. A.; KLER, P. A. Open-source
    high-performance software packages for direct and inverse solving of
    horizontal capillary flow. Capillarity, 2023, vol. 6, no. 2, pp. 31-40.
    """
    if verbose:
        start_time = process_time()

    if ob <= 0:
        msg = "ob must be positive"
        raise ValueError(msg)

    if not 0 < angle <= 2 * np.pi:
        msg = "angle must be positive and no greater than 2*pi"
        raise ValueError(msg)

    if radial == "cylindrical":
        if height is None:
            msg = (
                "must pass a height if radial == 'cylindrical' (or use radial='polar')"
            )
            raise TypeError(msg)
        if height <= 0:
            msg = "height must be positive"
            raise ValueError(msg)
    elif radial == "polar":
        if height is not None:
            msg = (
                "height parameter not allowed if radial == "
                "'polar' (use radial='cylindrical' instead)"
            )
            raise TypeError(msg)
    else:
        msg = "radial must be one of {'cylindrical', 'polar'}"
        raise ValueError(msg)

    if maxiter < 0:
        msg = "maxiter must not be negative"
        raise ValueError(msg)

    if not callable(D):
        D = from_expr(
            D, vectorized=False, max_derivatives=(1 if method == "explicit" else 2)
        )

    if b_bracket is not None:
        if b_hint is not None:
            msg = "cannot pass both b_hint and b_bracket"
            raise TypeError(msg)

        b_bracket = tuple(x if np.sign(i - x) == np.sign(-Qb) else i for x in b_bracket)  # type: ignore[assignment]

    elif b_hint is None:
        b_hint = i + np.sign(Qb)

    elif np.sign(i - b_hint) != np.sign(-Qb):
        msg = "value of b_hint disagrees with flowrate sign"
        raise ValueError(msg)

    if verbose >= 2:
        print(
            f"{'Iteration':^15}{'Residual':^15}{'Boundary value':^15}{'d/do|b':^15}{'Calls to D':^15}"
        )

        def shot_callback(result: _Shooter.Result) -> None:
            if np.isfinite(result.i_residual):
                print(
                    f"{shooter.shots:^15}{result.i_residual:^15.2e}{result.b:^15.2e}{result.d_dob:^15.7e}{result.D_calls:^15}"
                )

            elif result.d_dob is not None:
                print(
                    f"{shooter.shots:^15}{'*':^15}{result.b:^15.2e}{result.d_dob:^15.7e}{result.D_calls or '*':^15}"
                )
            else:
                print(
                    f"{shooter.shots:^15}{'*':^15}{result.b:^15.2e}{'*':^15}{result.D_calls or '*':^15}"
                )

    else:
        shot_callback = None  # type: ignore[assignment]

    shooter = _FlowrateShooter(
        D=D,
        i=i,
        rel_flowrate=Qb / (angle * (height or 1)),
        ob=ob,
        itol=itol,
        method=method,
        max_shots=maxiter,
        shot_callback=shot_callback,
    )

    try:
        if b_bracket is None:
            if Qb == 0:
                b: float | None = i
                b_bracket = (i, i)

            else:
                assert b_hint is not None
                b_result = bracket_root(
                    shooter.shoot,
                    interval=(i, b_hint),
                    f_interval=(None, None),
                    ftol=itol,
                    maxiter=None,
                )

                b = b_result.root
                assert b_result.bracket is not None
                b_bracket = b_result.bracket
                assert b_result.f_bracket is not None
                f_bracket = b_result.f_bracket

        else:
            assert b_hint is None
            b = None
            f_bracket = tuple(0 if Qb == 0 and x == i else None for x in b_bracket)  # type: ignore[assignment]

        if b is None:
            try:
                b_result = bisect(
                    shooter.shoot,
                    bracket=b_bracket,
                    f_bracket=f_bracket,
                    ftol=itol,
                    maxiter=None,
                )

                assert b_result.root is not None
                b = b_result.root
                assert b_result.bracket is not None
                b_bracket = b_result.bracket
                assert b_result.f_bracket is not None
                f_bracket = b_result.f_bracket

            except NotABracketError:
                assert b_hint is None
                if verbose:
                    print(
                        "b_bracket does not contain target boundary value. "
                        "Try again with a correct interval."
                    )
                msg = "b_bracket does not contain target bounday value"
                raise ValueError(msg) from None

    except shooter.ShotLimitReached:
        if verbose:
            print(f"The solver did not converge after {maxiter} iterations.")
            print(f"Execution time: {process_time() - start_time:.3f} s")
        msg = f"The solver did not converge after {maxiter} iterations."
        raise RuntimeError(msg) from None

    if shooter.best_shot is not None and shooter.best_shot.b == b:
        result = shooter.best_shot
    else:
        result = shooter.integrate(b=b)

    assert result.sol is not None
    assert result.o is not None
    solution = Solution(sol=result.sol, ob=result.o[0], oi=result.o[-1], D=D)

    solution.o = result.o  # type: ignore[attr-defined]
    solution.niter = shooter.shots  # type: ignore[attr-defined]
    solution.b_bracket = b_bracket  # type: ignore[attr-defined]

    if verbose:
        print(f"Solved in {shooter.shots} iterations.")
        print(f"Residual: {result.i_residual:.2e}")
        if b_bracket is not None:
            print(
                f"Boundary value: {b:.7e} (bracket: [{min(b_bracket):.7e}, {max(b_bracket):.7e}])"
            )
        else:
            print(f"Boundary value: {b:.7e}")
        print(f"Execution time: {process_time() - start_time:.3f} s")

    return solution


def solve_from_guess(
    D: _VectorizedD2 | sympy.Expr | str | float,
    i: float,
    b: float,
    o_guess: np.ndarray[tuple[int], np.dtype[np.floating]],
    guess: float | np.ndarray[tuple[int], np.dtype[np.floating]],
    radial: Literal[False, "cylindrical", "polar", "spherical"] = False,
    max_nodes: int = 1000,
    verbose: int = 0,
) -> Solution:
    r"""
    Alternative solver for problems with a Dirichlet boundary condition.

    Given a positive function `D`, scalars :math:`\theta_i`, :math:`\theta_b`
    and :math:`o_b`, and coordinate unit vector :math:`\mathbf{\hat{r}}`, finds
    a function :math:`\theta` of `r` and `t` such that:

    .. math:: \begin{cases} \dfrac{\partial\theta}{\partial t} =
        \nabla\cdot\left[D(\theta))\dfrac{\partial\theta}{\partial r}
        \mathbf{\hat{r}}\right ] & r>r_b(t),t>0\\
        \theta(r, 0) = \theta_i & r>0 \\
        \theta(r_b(t), t) = \theta_b & t>0 \\
        r_b(t) = o_b\sqrt t
        \end{cases}

    Alternative to the main :func:`solve` function. This function requires a
    starting  mesh and guess of the solution. It is significantly less robust
    than :func:`solve`, and will fail to converge in many cases that the latter
    can easily handle (whether it converges will usually depend heavily on the
    problem, the starting mesh and the guess of the solution; it will raise a
    `RuntimeError` on failure). However, when it converges it is usually faster
    than :func:`solve`, which may be an advantage for some use cases. You
    should nonetheless prefer :func:`solve` unless you have a particular use
    case for which you have found this function to be better.

    Possible use cases include refining a solution (note that :func:`solve` can
    do that too), optimization runs in which known solutions make good first
    approximations of solutions with similar parameters and every second of
    computing time counts, and in the implementation of other solving
    algorithms. In all these cases, :func:`solve` should probably be used as a
    fallback for when this function fails.

    Parameters
    ----------
    D : callable or `sympy.Expr` or str or float
        Callable that evaluates :math:`D` and its derivatives, obtained from
        the :mod:`fronts.D` module or defined in the same manner---i.e.:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        where ``theta`` may be a single float or a NumPy array.

        Alternatively, instead of a callable, the argument can be the
        expression of :math:`D` in the form of a string or
        :class:`sympy.Expr` with a single variable. In this case, the
        solver will differentiate and evaluate the expression as necessary.
    i : float
        Initial condition, :math:`\theta_i`.
    b : float
        Imposed boundary value, :math:`\theta_b`.
    o_guess : numpy.array_like, shape (n_guess,)
        Starting mesh in terms of the Boltzmann variable. Must be strictly
        increasing. ``o_guess[0]`` is taken as the value of the parameter
        :math:`o_b`, which determines the behavior of the boundary. If zero, it
        implies that the boundary always exists at :math:`r=0`. It must be
        strictly positive if `radial` is not False. A non-zero value implies
        a moving boundary.

        On the other end, ``o_guess[-1]`` must be large enough to contain the
        solution to the semi-infinite problem.
    guess : float or numpy.array_like, shape (n_guess,)
        Starting guess of the solution at the points in `o_guess`. If a single
        value, the guess is assumed uniform.
    radial : {False, 'cylindrical', 'polar', 'spherical'}, optional
        Choice of coordinate unit vector :math:`\mathbf{\hat{r}}`. Must be one
        of the following:

            *   False (default):
                :math:`\mathbf{\hat{r}}` is any coordinate unit vector in
                rectangular (Cartesian) coordinates, or an axial unit vector in
                a cylindrical coordinate system
            *   ``'cylindrical'`` or ``'polar'``:
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                cylindrical or polar coordinate system
            *   ``'spherical'``:
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                spherical coordinate system

    Returns
    -------
    solution : Solution
        See :class:`Solution` for a description of the solution object.
        Additional fields specific to this solver are included in the object:

            *   `o` *(numpy.ndarray, shape (n,))* --
                Final solver mesh, in terms of the Boltzmann variable.
            *   `niter` *(int)* --
                Number of iterations required to find the solution.
            *   `rms_residuals` *(numpy.ndarray, shape (n-1,))* --
                RMS values of the relative residuals over each mesh interval.

    Other Parameters
    ----------------
    max_nodes : int, optional
        Maximum allowed number of mesh nodes.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity. Must be one of the following:

            * 0 (default): work silently
            * 1: display a termination report
            * 2: also display progress during iterations

    See Also
    --------
    solve

    Notes
    -----
    This function works by transforming the partial differential equation with
    the Boltzmann transformation using :func:`ode` and then solving the
    resulting ODE with SciPy's collocation-based boundary value problem solver
    :func:`scipy.integrate.solve_bvp` and a two-point Dirichlet condition that
    matches the boundary and initial conditions of the problem. Upon that
    solver's convergence, it runs a final check on whether the candidate
    solution also satisfies the semi-infinite condition (which implies
    :math:`d\theta/do\to0` as :math:`o\to\infty`).
    """
    if verbose:
        start_time = process_time()

    if radial and o_guess[0] <= 0:
        msg = "o_guess[0] must be positive when using a radial coordinate"
        raise ValueError(msg)

    if not callable(D):
        D = from_expr(D)

    if np.ndim(guess) == 0:
        guess = np.full_like(o_guess, fill_value=guess)

    d_do_guess = np.gradient(guess, o_guess)

    fun, jac = ode(D=D, radial=radial)

    # Boundary conditions
    def bc(
        yb: np.ndarray[tuple[int], np.dtype[np.floating]],
        yi: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> tuple[float, float]:
        return (yb[0] - b, yi[0] - i)

    dbc_dyb = np.array(((1, 0), (0, 0)))
    dbc_dyi = np.array(((0, 0), (1, 0)))

    def bc_jac(
        yb: np.ndarray[tuple[int], np.dtype[np.floating]],
        yi: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.floating]],
        np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ]:
        return dbc_dyb, dbc_dyi  # type: ignore[return-value]

    with np.errstate(divide="ignore", invalid="ignore"):
        bvp_result = solve_bvp(
            fun,
            bc=bc,
            x=o_guess,
            y=(guess, d_do_guess),
            fun_jac=jac,
            bc_jac=bc_jac,
            max_nodes=max_nodes,
            verbose=verbose,
        )  # type: ignore[call-overload]

    if not bvp_result.success:
        if verbose:
            print(f"Execution time: {process_time() - start_time:.3f} s")
        msg = f"The solver did not converge: {bvp_result.message}"
        raise RuntimeError(msg)

    if abs(bvp_result.y[1, -1]) > 1e-6:
        if verbose:
            print(
                "The given mesh is too small for the problem. Try again "
                "after extending o_guess towards the right"
            )
            print(f"Execution time: {process_time() - start_time:.3f} s")

        msg = "o_guess cannot contain solution"
        raise RuntimeError(msg)

    solution = Solution(
        sol=bvp_result.sol, ob=bvp_result.x[0], oi=bvp_result.x[-1], D=D
    )

    solution.o = bvp_result.x  # type: ignore[attr-defined]
    solution.niter = bvp_result.niter  # type: ignore[attr-defined]
    solution.rms_residuals = bvp_result.rms_residuals  # type: ignore[attr-defined]

    if verbose:
        print(f"Execution time: {process_time() - start_time:.3f} s")

    return solution
