"""
This module uses the Boltzmann transformation to deal with initial-boundary
value problems in semi-infinite domains.
"""

from __future__ import division, absolute_import, print_function
import six

from collections import namedtuple
try:
    from time import process_time as timer
except ImportError:  # No time.process_time() in Python < 3.3
    from timeit import default_timer as timer  # Not the same, but close enough

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import PchipInterpolator

from ._boltzmann import ode, BaseSolution, r
from ._rootfinding import bracket_root, bisect, NotABracketError
from .D import from_expr


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
        `numpy.ndarray` ``o`` in the closed interval [`ob`, `oi`],
        ``sol(o)[0]`` are the values of :math:`\theta` at ``o``, and
        ``sol(o)[1]`` are the values of the derivative :math:`d\theta/do` at 
        ``o``. `sol` will only be evaluated in this interval.
    ob : float
        :math:`o_b`. Determines the behavior of the boundary in the problem.
    oi : float
        Value of the Boltzmann variable at which the solution can be considered
        to be equal to the initial condition. Cannot be less than `ob`.
    D : callable
        `D` used to obtain `sol`. Must be the same function that was passed to
        `ode`.
        
    """
    def __init__(self, sol, ob, oi, D):
        if ob > oi:
            raise ValueError("ob cannot be greater than oi")

        def wrapped_sol(o):
            under = np.less(o, ob)
            over = np.greater(o, oi)

            o = np.where(under|over, oi, o)

            y = np.asarray(sol(o))
            y[:,under] = np.nan
            y[1:,over] = 0

            return y

        super(Solution, self).__init__(sol=wrapped_sol, D=D)
        self._ob = ob
        self._oi = oi

    @property
    def i(self):
        """float: Initial value of the solution."""
        return self(o=self._oi)

    @property
    def ob(self):
        """float: :math:`o_b`"""
        return self._ob
    
    def rb(self, t):
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
    def b(self):
        """float: Boundary value of the solution."""
        return self(o=self.ob)

    def d_drb(self, t):
        r"""
        Spatial derivative of the solution at the boundary.

        Evaluates and returns :math:`\partial\theta/\partial r|_b`. Equivalent
        to ``self.d_dr(self.rb(t), t)``.

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray
            The return is of the same type and shape as `t`.
        """
        return self.d_dr(self.rb(t), t)

    def d_dtb(self, t):
        r"""
        Time derivative of the solution at the boundary.

        Evaluates and returns :math:`\partial\theta/\partial t|_b`. Equivalent
        to ``self.d_dt(self.rb(t), t)``.

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray
            The return is of the same type and shape as `t`.
        """
        return self.d_dt(self.rb(t), t)

    def fluxb(self, t):
        r"""
        Boundary flux.

        Returns the diffusive flux of :math:`\theta` at the boundary, in the
        direction :math:`\mathbf{\hat{r}}`. Equivalent to
        ``self.flux(self.rb(t), t)``.

        Parameters
        ----------
        t : float or numpy.ndarray
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray
            The return is of the same type and shape as `t`.
        """
        return self.flux(self.rb(t), t)

    @property
    def d_dob(self):
        """
        float: Derivative of the solution with respect to the Boltzmann
        variable at the boundary.
        """
        return self.d_do(o=self.ob)
    


class _Shooter(object):
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

    def __init__(self, D, i, radial, ob, theta_direction, itol, max_shots,
                 shot_callback):

        assert callable(D)
        assert not radial or ob > 0
        assert theta_direction in {-1, 0, 1}
        assert max_shots is None or max_shots >= 0
        assert shot_callback is None or callable(shot_callback)

        self._fun, self._jac = ode(D, radial)
        self._i = i
        self._ob = ob
        self._theta_direction = theta_direction
        self._max_shots = max_shots
        self._shot_callback = shot_callback

        # Integration events
        def settled(o, y):
            return y[1]
        settled.terminal = True

        def blew_past_i(o, y):
            return y[0] - (i + theta_direction*itol)
        blew_past_i.terminal = True

        self._events = (settled, blew_past_i)

        self.shots = 0
        self.best_shot = None


    Result = namedtuple("Result", ['b', 
                                   'd_dob',
                                   'i_residual',
                                   'D_calls',
                                   'o',
                                   'sol'])

    def integrate(self, b, d_dob):
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
        assert (self._i - b)*self._theta_direction >= 0
        assert d_dob*self._theta_direction >= 0

        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                ivp_result = solve_ivp(self._fun,
                                       t_span=(self._ob, np.inf),
                                       y0=(b, d_dob),
                                       method='Radau',
                                       jac=self._jac,
                                       events=self._events,
                                       dense_output=True)

            except (ValueError, ArithmeticError, UnboundLocalError):
                # Catch D domain errors. Also catch UnboundLocalError caused by
                # https://github.com/scipy/scipy/issues/10775 (fixed in SciPy
                # v1.4.0; but we do not require that version because it does
                # not support Python 2.7)

                return self.Result(b=b,
                                   d_dob=d_dob,
                                   i_residual=self._theta_direction*np.inf,
                                   D_calls=None,
                                   o=None,
                                   sol=None)

        if ivp_result.success and ivp_result.t_events[0].size == 1:

            return self.Result(b=b,
                               d_dob=d_dob,
                               i_residual=ivp_result.y[0,-1] - self._i,
                               D_calls=ivp_result.nfev + ivp_result.njev,
                               o=ivp_result.t,
                               sol=ivp_result.sol)

        else:

            return self.Result(b=b,
                               d_dob=d_dob,
                               i_residual=self._theta_direction*np.inf,
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
        Calls `integrate` and returns the result's `i_residual`. Each call
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
        i_residual : float
        """
        self.shots += 1

        if self._max_shots is not None and self.shots > self._max_shots:
            raise self.ShotLimitReached

        result = self.integrate(*args, **kwargs)

        if self._shot_callback is not None:
            self._shot_callback(result)

        if (self.best_shot is None 
                or abs(result.i_residual) < abs(self.best_shot.i_residual)):
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

    def __init__(self, D, i, b, radial, ob, itol, max_shots, shot_callback):

        theta_direction = np.sign(i - b)

        super(_DirichletShooter, self).__init__(D=D,
                                                i=i,
                                                radial=radial,
                                                ob=ob,
                                                theta_direction=theta_direction,
                                                itol=itol,
                                                max_shots=max_shots,
                                                shot_callback=shot_callback)

        self._b = b


    def integrate(self, d_dob):
        """
        Integrate and return the full result.

        Parameters
        ----------
        d_dob : float

        Returns
        -------
        Result
        """
        return super(_DirichletShooter, self).integrate(b=self._b,
                                                        d_dob=d_dob)


def solve(D, i, b, radial=False, ob=0.0, itol=1e-3, d_dob_hint=None,
          d_dob_bracket=None, maxiter=100, verbose=0):
    r"""
    Solve an instance of the general problem.

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
    D : callable or `sympy.Expression` or str or float
        Callable that evaluates :math:`D` and its derivatives, obtained from
        the :mod:`fronts.D` module or defined in the same manner.

        Alternatively, an expression for :math:`D` in the form of a string or
        `sympy.Expression` with a single variable.
    i : float
        :math:`\theta_i`, the initial value of :math:`\theta` in the domain.
    b : float
        :math:`\theta_b`, the value of :math:`\theta` imposed at the boundary.
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
    itol : float, optional
        Absolute tolerance for the initial condition.
    d_dob_hint : None or float, optional
        Optional hint to the solver. If given, it should be a number close to
        the expected value of the derivative of :math:`\theta` with respect to
        the Boltzmann variable `o` at the boundary (i.e.,
        :math:`d\theta/do|_b`) in the solution to be found. This parameter is
        typically not needed.
    d_dob_bracket : None or sequence of two floats
        Optional search interval that brackets the value of
        :math:`d\theta/do|_b` in the solution. If given, the solver will use
        bisection to find a solution in which :math:`d\theta/do|_b` falls
        inside that interval (a `ValueError` will be raised for an incorrect
        interval). This parameter cannot be passed together with a
        `d_dob_hint`. It is also not needed in typical usage.
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
            *   `d_dob_bracket` *(sequence of two floats or None)*
                If available, an interval that contains the value of
                :math:`d\theta/do|_b`. May be used as the input `d_dob_bracket`
                in a subsequent call with a smaller `itol` for the same problem
                in order to avoid reduntant iterations. Whether this interval
                is available or not depends on the strategy used internally by
                the solver; in particular, this field is never `None` if a
                `d_dob_bracket` is passed when calling the function.

    See also
    --------
    solve_from_guess
    solve_flowrate

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
    values of :math:`d\theta/do|_b` until it finds the solution that also
    satisfies the initial condition within the specified tolerance. Trial
    values of :math:`d\theta/do|_b` are selected automatically by default
    (taking into account an optional hint if passed by the user), or by
    bisecting an optional search interval. This scheme assumes that
    :math:`d\theta/do|_b` varies continuously with :math:`\theta_i`.
    """
    if verbose:
        start_time = timer()

    if radial and ob <= 0:
        raise ValueError("ob must be positive when using a radial coordinate")

    if maxiter < 0:
        raise ValueError("maxiter must not be negative")

    if not callable(D):
        D = from_expr(D)

    if d_dob_bracket is not None:
        if d_dob_hint is not None:
            raise TypeError("cannot pass both d_dob_hint and d_dob_bracket")

        d_dob_bracket = tuple(x if np.sign(x) == np.sign(i - b) else 0
                              for x in d_dob_bracket)

    elif d_dob_hint is None:
        d_dob_hint = (i - b)/(2*D(b)**0.5)

    elif np.sign(d_dob_hint) != np.sign(i - b):
        raise ValueError("sign of d_dob_hint does not match direction given by"
                         "b and i")

    if verbose >= 2:
        print("{:^15}{:^15}{:^15}{:^15}".format(
               "Iteration",
               "Residual",
               "d/do|b",
               "Calls to D"))

        def shot_callback(result):
            if np.isfinite(result.i_residual):
                print("{:^15}{:^15.2e}{:^15.7e}{:^15}".format(
                       shooter.shots,
                       result.i_residual,
                       result.d_dob,
                       result.D_calls))
            else:
                print("{:^15}{:^15}{:^15.7e}{:^15}".format(
                       shooter.shots,
                       "*",
                       result.d_dob,
                       result.D_calls or "*"))
    else:
        shot_callback = None

    shooter = _DirichletShooter(D=D,
                                i=i,
                                b=b,
                                radial=radial,
                                ob=ob,
                                itol=itol,
                                max_shots=maxiter,
                                shot_callback=shot_callback)

    try:
        if d_dob_bracket is None:
            if i == b:
                d_dob = 0
                d_dob_bracket = (0, 0)

            else:
                d_dob_result = bracket_root(shooter.shoot,
                                            interval=(0, d_dob_hint),
                                            f_interval=(b-i, None),
                                            ftol=itol,
                                            maxiter=None)

                d_dob = d_dob_result.root
                d_dob_bracket = d_dob_result.bracket
                f_bracket = d_dob_result.f_bracket

        else:
            assert d_dob_hint is None
            d_dob = None
            f_bracket = tuple(b-i if x==0 else None for x in d_dob_bracket)


        if d_dob is None:
            try:
                d_dob_result = bisect(shooter.shoot,
                                      bracket=d_dob_bracket,
                                      f_bracket=f_bracket,
                                      ftol=itol,
                                      maxiter=None)

                d_dob = d_dob_result.root
                d_dob_bracket = d_dob_result.bracket
                f_bracket = d_dob_result.f_bracket

            except NotABracketError:
                assert d_dob_hint is None
                if verbose:
                    print("d_dob_bracket does not contain target d/do|b. Try "
                          "again with a correct interval.")
                six.raise_from(
                    ValueError("d_dob_bracket does not contain target d/do|b"),
                    None)

    except shooter.ShotLimitReached:
        if verbose:
          print("The solver did not converge after {} iterations.".format(
                maxiter))
          print("Execution time: {:.3f} s".format(timer() - start_time))
        six.raise_from(
            RuntimeError("The solver did not converge after {} iterations."
                         .format(maxiter)),
            None)

    if shooter.best_shot is not None and shooter.best_shot.d_dob == d_dob:
        result = shooter.best_shot
    else:
        result = shooter.integrate(d_dob=d_dob)

    solution = Solution(sol=result.sol,
                        ob=result.o[0],
                        oi=result.o[-1],
                        D=D)

    solution.o = result.o
    solution.niter = shooter.shots
    solution.d_dob_bracket = d_dob_bracket

    if verbose:
        print("Solved in {} iterations.".format(shooter.shots))
        print("Residual: {:.2e}".format(result.i_residual))
        if d_dob_bracket is not None:
            print("d/do|b: {:.7e} (bracket: [{:.7e}, {:.7e}])".format(
                  d_dob, min(d_dob_bracket), max(d_dob_bracket)))
        else:
            print("d/do|b: {:.7e}".format(d_dob))
        print("Execution time: {:.3f} s".format(timer() - start_time))

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

    def __init__(self, D, i, rel_flowrate, ob, itol, max_shots, shot_callback):

        assert ob > 0

        theta_direction = np.sign(-rel_flowrate)

        super(_FlowrateShooter, self).__init__(D=D,
                                               i=i,
                                               ob=ob,
                                               radial='cylindrical',
                                               theta_direction=theta_direction,
                                               itol=itol,
                                               max_shots=max_shots,
                                               shot_callback=shot_callback)

        self._D_ = D
        # Flow rate per unit angle and height
        self._rel_flowrate = rel_flowrate

    class _DError(Exception):
        pass

    def _D(self, theta):
        """
        Call `D` and return its value if valid.

        Raises a `_DError` exception if the call fails or does not return a
        finite, positive value.

        Parameters
        ----------
        float

        Returns
        -------
        float
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                D = self._D_(theta)
            except (ValueError, ArithmeticError) as e:
                six.raise_from(self._DError, e)

        try:
            D = float(D)
        except TypeError as e:
            six.raise_from(self._DError, e)

        if not np.isfinite(D) or D <= 0:
            raise self._DError
        
        return D

    def integrate(self, b):
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
            Db = self._D(b)
        except self._DError:
            return self.Result(b=b,
                               d_dob=None,
                               i_residual=-self._theta_direction*np.inf,
                               D_calls=1,
                               o=None,
                               sol=None)

        d_dob = -self._rel_flowrate/(Db*self._ob)

        result = super(_FlowrateShooter, self).integrate(b=b, d_dob=d_dob)

        D_calls = result.D_calls+1 if result.D_calls is not None else None

        return result._replace(D_calls=D_calls)


def solve_flowrate(D, i, Qb, radial, ob=1e-6, angle=2*np.pi, height=None,
                   itol=1e-3, b_hint=None, b_bracket=None, maxiter=100,
                   verbose=0):
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
    D : callable or `sympy.Expression` or str or float
        Callable that evaluates :math:`D` and its derivatives, obtained from
        the :mod:`fronts.D` module or defined in the same manner.

        Alternatively, an expression for :math:`D` in the form of a string or
        `sympy.Expression` with a single variable.
    i : float
        :math:`\theta_i`, the initial value of :math:`\theta` in the domain.
    Qb : float
        :math:`Q_b`, flow rate of :math:`\theta` imposed at the boundary. A
        positive value means that :math:`\theta` is flowing into the domain;
        negative values mean that :math:`\theta` flows out of the domain.
    radial : {'cylindrical', 'polar'}
        Choice of coordinate unit vector :math:`\mathbf{\hat{r}}`. Must be one
        of the following:

            *   ``'cylindrical'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                cylindrical coordinate system
            *   ``'polar'``
                :math:`\mathbf{\hat{r}}` is the radial unit vector in a
                polar coordinate system
    ob : float, optional
        :math:`o_b`, which determines the behavior of the boundary. It must be
        positive. The boundary acts as a line source or sink in the limit where
        `ob` tends to zero.
    angle : float, optional
        Total angle of the domain. Must be positive and no greater than
        :math:`2\pi`.
    height : None or float, optional
        Axial height of the domain if ``radial=='cylindrical'``. Not allowed if
        ``radial=='polar'``.
    itol : float, optional
        Absolute tolerance for :math:`\theta_i`.
    b_hint : None or float, optional
        Optional hint to the solver. If given, it should be a number close to
        the expected value of :math:`\theta` at the boundary (i.e.
        :math:`\theta_b`) in the solution to be found.
    b_bracket : None or sequence of two floats
        Optional search interval that brackets the value of :math:`\theta_b`
        in the solution. If given, the solver will use bisection to find a
        solution in which :math:`\theta_b` falls inside that interval (a
        `ValueError` will be raised for an incorrect interval). This parameter
        cannot be passed together with a `b_hint`.
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
            *   `b_bracket` *(sequence of two floats or None)*
                If available, an interval that contains the value of
                :math:`\theta_b`. May be used as the input `b_bracket` in a
                subsequent call with a smaller `itol` for the same problem in
                order to avoid reduntant iterations. Whether this interval is
                available or not depends on the strategy used internally by the
                solver; in particular, this field is never `None` if a
                `b_bracket` is passed when calling the function.

    See also
    --------
    solve

    Notes
    -----
    This function works by transforming the partial differential equation with
    the Boltzmann transformation using `ode` and then solving the resulting ODE
    repeatedly with the 'Radau' method as implemented in the `scipy.integrate`
    module and a custom shooting algorithm. The boundary condition is satisfied
    exactly as the starting point, and the algorithm iterates with different
    values of :math:`\theta` at the boundary until it finds the solution that
    also satisfies the initial condition within the specified tolerance. Trial
    values of :math:`\theta` at the boundary are selected automatically by
    default (taking into account an optional hint if passed by the user), or by
    bisecting an optional search interval. This scheme assumes that
    :math:`\theta` at the boundary varies continuously with :math:`\theta_i`.
    """
    if verbose:
        start_time = timer()

    if ob <= 0:
        raise ValueError("ob must be positive")

    if not 0 < angle <= 2*np.pi:
        raise ValueError("angle must be positive and no greater than 2*pi")

    if radial == 'cylindrical':
        if height is None:
            raise TypeError("must pass a height if radial == 'cylindrical' "
                            "(or use radial='polar')")
        if height <= 0:
            raise ValueError("height must be positive")
    elif radial == 'polar':
        if height is not None:
            raise TypeError("height parameter not allowed if radial == "
                            "'polar' (use radial='cylindrical' instead)")
    else:
        raise ValueError("radial must be one of {'cylindrical', 'polar'}")

    if maxiter < 0:
        raise ValueError("maxiter must not be negative")

    if not callable(D):
        D = from_expr(D)

    if b_bracket is not None:
        if b_hint is not None:
            raise TypeError("cannot pass both b_hint and b_bracket")

        b_bracket = tuple(x if np.sign(i-x) == np.sign(-Qb) else i
                           for x in b_bracket)

    elif b_hint is None:
        b_hint = i + np.sign(Qb)

    elif np.sign(i-b_hint) != np.sign(-Qb):
        raise ValueError("value of b_hint disagrees with flowrate sign")

    if verbose >= 2:
        print("{:^15}{:^15}{:^15}{:^15}{:^15}".format(
               "Iteration",
               "Residual",
               "Boundary value",
               "d/do|b",
               "Calls to D"))

        def shot_callback(result):
            if np.isfinite(result.i_residual):
                print("{:^15}{:^15.2e}{:^15.2e}{:^15.7e}{:^15}".format(
                       shooter.shots,
                       result.i_residual,
                       result.b,
                       result.d_dob,
                       result.D_calls))

            elif result.d_dob is not None:
                print("{:^15}{:^15}{:^15.2e}{:^15.7e}{:^15}".format(
                       shooter.shots,
                       "*",
                       result.b,
                       result.d_dob,
                       result.D_calls or "*"))
            else:
                print("{:^15}{:^15}{:^15.2e}{:^15}{:^15}".format(
                       shooter.shots,
                       "*",
                       result.b,
                       "*",
                       result.D_calls or "*"))
    else:
        shot_callback = None

    shooter = _FlowrateShooter(D=D,
                               i=i,
                               rel_flowrate=Qb/(angle*(height or 1)),
                               ob=ob,
                               itol=itol,
                               max_shots=maxiter,
                               shot_callback=shot_callback)

    try:
        if b_bracket is None:
            if Qb == 0:
                b = i
                b_bracket = (i, i)
        
            else:
                b_result = bracket_root(shooter.shoot,
                                        interval=(i, b_hint),
                                        f_interval=(None, None),
                                        ftol=itol,
                                        maxiter=None)

                b = b_result.root
                b_bracket = b_result.bracket
                f_bracket = b_result.f_bracket

        else:
            assert b_hint is None
            b = None
            f_bracket = tuple(0 if Qb==0 and x==i else None for x in b_bracket)

        if b is None:
            try:
                b_result = bisect(shooter.shoot,
                                  bracket=b_bracket,
                                  f_bracket=f_bracket,
                                  ftol=itol,
                                  maxiter=None)

                b = b_result.root
                b_bracket = b_result.bracket
                f_bracket = b_result.f_bracket

            except NotABracketError:
                assert b_hint is None
                if verbose:
                    print("b_bracket does not contain target boundary value. "
                          "Try again with a correct interval.")
                six.raise_from(
                    ValueError(
                        "b_bracket does not contain target bounday value"
                        ),
                    None)

    except shooter.ShotLimitReached:
        if verbose:
          print("The solver did not converge after {} iterations.".format(
                maxiter))
          print("Execution time: {:.3f} s".format(timer() - start_time))
        six.raise_from(
            RuntimeError("The solver did not converge after {} iterations."
                         .format(maxiter)),
            None)

    if shooter.best_shot is not None and shooter.best_shot.b == b:
        result = shooter.best_shot
    else:
        result = shooter.integrate(b=b)

    solution = Solution(sol=result.sol,
                        ob=result.o[0],
                        oi=result.o[-1],
                        D=D)

    solution.o = result.o
    solution.niter = shooter.shots
    solution.b_bracket = b_bracket

    if verbose:
        print("Solved in {} iterations.".format(shooter.shots))
        print("Residual: {:.2e}".format(result.i_residual))
        if b_bracket is not None:
            print("Boundary value: {:.7e} (bracket: [{:.7e}, {:.7e}])".format(
                  b, min(b_bracket), max(b_bracket)))
        else:
            print("Boundary value: {:.7e}".format(b))
        print("Execution time: {:.3f} s".format(timer() - start_time))

    return solution


def solve_from_guess(D, i, b, o_guess, guess, radial=False, max_nodes=1000,
                     verbose=0):
    r"""
    Solve an instance of the general problem starting from a guess of the
    solution.

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
    D : callable or `sympy.Expression` or str or float
        Callable that evaluates :math:`D` and its derivatives, obtained from
        the :mod:`fronts.D` module or defined in the same manner.

        Alternatively, an expression for :math:`D` in the form of a string or
        `sympy.Expression` with a single variable.
    i : float
        :math:`\theta_i`, the initial value of :math:`\theta` in the domain.
    b : float
        :math:`\theta_b`, the value of :math:`\theta` imposed at the boundary.
    o_guess : numpy.array_like, shape (n_guess,)
        Starting mesh in terms of the Boltzmann variable `o`. Must be strictly
        increasing. ``o_guess[0]`` is :math:`o_b`, which determines the
        behavior of the boundary. If zero, it implies that the boundary always
        exists at :math:`r=0`. It must be strictly positive if `radial` is not
        `False`. Be aware that a non-zero value implies a moving boundary. On
        the other end, ``o_guess[-1]`` must be large enough to contain the
        solution to the semi-infinite problem.
    guess : float or numpy.array_like, shape (n_guess,)
        Starting guess of :math:`\theta` at the points in `o_guess`. If a
        single value, the guess is assumed uniform.
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
    :math:`d\theta/do\to0` as :math:`o\to\infty`).
    """
    if verbose:
        start_time = timer()

    if radial and o_guess[0] <= 0:
        raise ValueError("o_guess[0] must be positive when using a radial "
                         "coordinate")

    if not callable(D):
        D = from_expr(D)

    if np.ndim(guess) == 0:
        guess = np.full_like(o_guess, fill_value=guess)

    d_do_guess = np.gradient(guess, o_guess)

    fun, jac = ode(D=D, radial=radial)

    # Boundary conditions
    def bc(yb, yi):
        return (yb[0]-b, yi[0]-i)

    dbc_dyb = np.array(((1, 0), (0, 0)))
    dbc_dyi = np.array(((0, 0), (1, 0)))
    def bc_jac(yb, yi):
        return dbc_dyb, dbc_dyi

    with np.errstate(divide='ignore', invalid='ignore'):
        bvp_result = solve_bvp(fun, bc=bc, x=o_guess, y=(guess, d_do_guess),
                               fun_jac=jac, bc_jac=bc_jac,
                               max_nodes=max_nodes, verbose=verbose)

    if not bvp_result.success:
        if verbose:
            print("Execution time: {:.3f} s".format(timer() - start_time))
        raise RuntimeError("The solver did not converge: {}".format(
                            bvp_result.message))

    if abs(bvp_result.y[1,-1]) > 1e-6:
        if verbose:
            print("The given mesh is too small for the problem. Try again "
                  "after extending o_guess towards the right")
            print("Execution time: {:.3f} s".format(timer() - start_time))

        raise RuntimeError("o_guess cannot contain solution")

    solution = Solution(sol=bvp_result.sol,
                        ob=bvp_result.x[0],
                        oi=bvp_result.x[-1],
                        D=D)

    solution.o = bvp_result.x
    solution.niter = bvp_result.niter
    solution.rms_residuals = bvp_result.rms_residuals

    if verbose:
        print("Execution time: {:.3f} s".format(timer() - start_time))

    return solution


def inverse(o, samples):
    r"""
    Solve an inverse problem.

    Given a function :math:`\theta` of `r` and `t`, and scalars
    :math:`\theta_i`, :math:`\theta_b` and :math:`o_b`, finds a positive
    function `D` of the values of :math:`\theta` such that:

    .. math:: \begin{cases} \dfrac{\partial\theta}{\partial t} =
        \dfrac{\partial}{\partial r}\left(D(\theta)\dfrac{\partial\theta}
        {\partial r}\right) & r>r_b(t),t>0\\
        \theta(r, 0) = \theta_i & r>0 \\
        \theta(r_b(t), t) = \theta_b & t>0 \\
        r_b(t) = o_b\sqrt t
        \end{cases}

    :math:`\theta` is taken as its values on a discrete set of points expressed
    in terms of the Boltzmann variable. Problems in radial coordinates are not
    supported.

    Parameters
    ----------
    o : numpy.array_like, shape (n,)
        Points where :math:`\theta` is known, expressed in terms of the
        Boltzmann variable. Must be strictly increasing.

    samples : numpy.array_like, shape (n,)
        Values of :math:`\theta` at `o`. Must be monotonic (either
        non-increasing or non-decreasing) and ``samples[-1]`` must be
        :math:`\theta_i`.

    Returns
    -------
    D : callable
        Twice-differentiable function that maps the range of :math:`\theta` to
        positive values. It can be called as ``D(theta)`` to evaluate it at
        ``theta``. It can also be called as ``D(theta, n)`` with ``n`` equal to
        1 or 2, in which case the first ``n`` derivatives of the function
        evaluated at the same ``theta`` are included (in order) as additional
        return values. While mathematically a scalar function, `D` operates in
        a vectorized fashion with the same semantics when ``theta`` is a
        `numpy.ndarray`.

    See also
    --------
    o

    Notes
    -----
    An `o` function of :math:`\theta` is constructed by interpolating the input
    data with a PCHIP monotonic cubic spline. The function `D` is then
    constructed by applying the expressions that result from solving the
    Boltzmann-transformed equation for `D`.

    While very fast, the scheme used by this function is somewhat limited in
    its practical precision because of the use of interpolation (see the Notes)
    and the fact that two :math:`\theta` functions that differ little in their
    values may actually be the consequence of very different `D` functions. If
    the goal is to find the parameters for a parameterized `D`, you may opt to
    perform an optimization run using `solve` instead.

    Depending on the number of points, the returned `D` may take orders of
    magnitude more time to be evaluated than an analytical function. In that
    case, you may notice that solvers work significantly slower when called
    with this `D`.

    This function also works if the problem has different boundary condition,
    as long as it is compatible with the Boltzmann transformation so that
    :math:`\theta` can be considered a function of `o` only.
    """

    if not np.all(np.diff(o) > 0):
        raise ValueError("o must be strictly increasing")

    if not(np.all(np.diff(samples) >= -1e-12)
            or np.all(np.diff(samples) <= 1e-12)):
        raise ValueError("samples must be monotonic")

    i = samples[-1]

    samples, indices = np.unique(samples, return_index=True)
    o = o[indices]

    o_func = PchipInterpolator(x=samples, y=o)

    o_antiderivative_func = o_func.antiderivative()
    o_antiderivative_i = o_func.antiderivative()(i)

    o_funcs = [o_func.derivative(i) for i in range(4)]

    def D(theta, derivatives=0):

        Iodtheta = o_antiderivative_func(theta) - o_antiderivative_i

        do_dtheta = o_funcs[1](theta)

        D = -(do_dtheta*Iodtheta)/2

        if derivatives == 0: return D

        o = o_funcs[0](theta)
        d2o_dtheta2 = o_funcs[2](theta)

        dD_dtheta = -(d2o_dtheta2*Iodtheta + do_dtheta*o)/2

        if derivatives == 1: return D, dD_dtheta

        d3o_dtheta3 = o_funcs[3](theta)

        d2D_dtheta2 = -(d3o_dtheta3*Iodtheta + 2*d2o_dtheta2*o + do_dtheta**2)/2

        if derivatives == 2: return D, dD_dtheta, d2D_dtheta2

        raise ValueError("derivatives must be 0, 1, or 2")

    return D
