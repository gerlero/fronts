"""
Internal Boltzmann transformation module.

This module defines the Boltzmann variable, transformation of the partial
differential equation into the ODE in terms of that variable, and inverse
transformation of the ODE's solution into a solution to the partial
differential equation.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, overload

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from .D import _D0, _ScalarD1, _ScalarD2, _VectorizedD1, _VectorizedD2


@overload
def o(r: float, t: float) -> float: ...


@overload
def o(
    r: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    t: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


@overload
def o(
    r: float, t: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


@overload
def o(
    r: np.ndarray[tuple[int, ...], np.dtype[np.floating]], t: float
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


def o(
    r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    r"""
    Transform to the Boltzmann variable.

    Returns the Boltzmann variable at the given `r` and `t`, which is the
    result of applying the Boltzmann transformation:

    .. math::
        o(r,t) = r/\sqrt t

    Parameters
    ----------
    r : float or numpy.ndarray
        Location(s). If an array, it must have a shape broadcastable with `t`.
    t : float or numpy.ndarray
        Time(s). If an array, it must have a shape broadcastable with `r`.
        Values must be positive.

    Returns
    -------
    o : float or numpy.ndarray
        The return is a float if both `r` and `t` are floats. Otherwise it is
        an array of the shape that results from broadcasting `r` and `t`.

    See Also
    --------
    do_dr
    do_dt
    r
    t
    as_o
    """
    return r / t**0.5


@overload
def do_dr(r: object, t: float) -> float: ...


@overload
def do_dr(
    r: object, t: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


def do_dr(
    r: object, t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    r"""
    Spatial derivative of the Boltzmann transformation.

    Returns the partial derivative :math:`\partial o/\partial r` evaluated at
    (`r`, `t`).

    Parameters
    ----------
    r : float or numpy.ndarray
        Location(s). If an array, it must have a shape broadcastable with `t`.
    t : float or numpy.ndarray
        Time(s). If an array, it must have a shape broadcastable with `r`.
        Values must be positive.

    Returns
    -------
    do_dr : float or numpy.ndarray
        The return is a float if both `r` and `t` are floats. Otherwise it is
        an array of the shape that results from broadcasting `r` and `t`.

    See Also
    --------
    o
    do_dt
    """
    return 1 / t**0.5


@overload
def do_dt(r: float, t: float) -> float: ...


@overload
def do_dt(
    r: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    t: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


@overload
def do_dt(
    r: float, t: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


@overload
def do_dt(
    r: np.ndarray[tuple[int, ...], np.dtype[np.floating]], t: float
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


def do_dt(
    r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    r"""
    Time derivative of the Boltzmann transformation.

    Returns the partial derivative :math:`\partial o/\partial t` evaluated at
    (`r`, `t`).

    Parameters
    ----------
    r : float or numpy.ndarray
        Location(s). If an array, it must have a shape broadcastable with `t`.
    t : float or numpy.ndarray
        Time(s). If an array, it must have a shape broadcastable with `r`.
        Values must be positive.

    Returns
    -------
    do_dt : float or numpy.ndarray
        The return is a float if both `r` and `t` are floats. Otherwise it is
        an array of the shape that results from broadcasting `r` and `t`.

    See Also
    --------
    o
    do_dr
    """
    return -o(r, t) / (2 * t)


@overload
def r(o: float, t: float) -> float: ...


@overload
def r(
    o: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    t: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


@overload
def r(
    o: float, t: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


@overload
def r(
    o: np.ndarray[tuple[int, ...], np.dtype[np.floating]], t: float
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


def r(
    o: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    """
    Transform back from the Boltzmann variable into `r`.

    Parameters
    ----------
    o : float or numpy.ndarray
        Value(s) of the Boltzmann variable. If an array, it must have a shape
        broadcastable with `t`.
    t : float or numpy.ndarray
        Time(s). If an array, it must have a shape broadcastable with `o`.
        Values must not be negative.

    Returns
    -------
    r : float or numpy.ndarray
        The return is a float if both `o` and `t` are floats. Otherwise it is
        an array of the shape that results from broadcasting `o` and `t`.

    See Also
    --------
    o
    t
    """
    return o * t**0.5


@overload
def t(o: float, r: float) -> float: ...


@overload
def t(
    o: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    r: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


@overload
def t(
    o: float, r: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


@overload
def t(
    o: np.ndarray[tuple[int, ...], np.dtype[np.floating]], r: float
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...


def t(
    o: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    """
    Transform back from the Boltzmann variable into `t`.

    Parameters
    ----------
    o : float or numpy.ndarray
        Value(s) of the Boltzmann variable. If a NumPy array, it must have a
        shape broadcastable with `r`. Values must not be zero.
    r : float or numpy.ndarray
        Location(s). If a NumPy array, it must have a shape broadcastable with
        `o`.

    Returns
    -------
    t : float or numpy.ndarray
        The return is a float if both `o` and `r` are floats. Otherwise it is
        an array of the shape that results from broadcasting `o` and `r`.

    See Also
    --------
    o
    r
    """
    return (r / o) ** 2


_o = o


def as_o(
    r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None,
    t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None,
    o: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None,
) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    """
    Transform to the Boltzmann variable if called with `r` and `t`.

    Passes the values through if called with `o` only. On other combinations of
    arguments, it raises a `TypeError` with a message explaining valid usage.

    This function is a helper to define other functions that may be called
    either with `r` and `t`, or with just `o`.

    Parameters
    ----------
    r : None or float or numpy.ndarray, optional
        Location(s). If this parameter is used, `t` must also be given. If an
        array, it must have a shape broadcastable with `t`.
    t : None or float or numpy.ndarray, optional
        Time(s). If an array, it must have a shape broadcastable with `r`.
        Values must be positive.
    o : None or float or numpy.ndarray, optional
        Value(s) of the Boltzmann variable. If this parameter is used, neither
        `r` nor `t` can be given.

    Returns
    -------
    o : float or numpy.ndarray
        Passes `o` through if it is given. Otherwise, it calls the function
        :func:`o` and returns ``o(r,t)``.

    See Also
    --------
    o
    """
    if o is not None:
        if r is not None or t is not None:
            msg = "must pass either r and t, or just o"
            raise TypeError(msg)
        return o

    if r is None or t is None:
        msg = "must pass either r and t, or just o"
        raise TypeError(msg)
    return _o(r, t)


_k = {False: 0, "cylindrical": 1, "polar": 1, "spherical": 2}


def ode(
    D: _ScalarD1 | _ScalarD2 | _VectorizedD1 | _VectorizedD2,
    radial: Literal[False, "cylindrical", "polar", "spherical"] = False,
    catch_errors: bool = False,  # noqa: FBT001
) -> tuple[
    Callable[
        [
            float | np.ndarray[tuple[int], np.dtype[np.floating]],
            tuple[float, float]
            | np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.floating]],
        ],
        np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ],
    Callable[
        [
            float | np.ndarray[tuple[int], np.dtype[np.floating]],
            tuple[float, float]
            | np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.floating]],
        ],
        np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ],
]:
    r"""
    Transform the PDE into an ODE.

    Given a positive function `D` and coordinate unit vector
    :math:`\mathbf{\hat{r}}`, transforms the partial differential equation
    (PDE) in which :math:`\theta` is the unknown function of :math:`r` and
    :math:`t`:

    .. math:: \dfrac{\partial\theta}{\partial t} =
        \nabla\cdot\left[D(\theta)\dfrac{\partial\theta}{\partial r}
        \mathbf{\hat{r}}\right]

    into an ordinary differential equation (ODE) where :math:`\theta` is an
    unknown function of the Boltzmann variable :math:`o`.

    This function returns the `fun` and `jac` callables that may be used to
    solve the ODE with the solvers included with SciPy (:mod:`scipy.integrate`
    module). The second-order ODE is expressed as a system of first-order ODEs
    with independent variable :math:`o` where ``y[0]`` in `fun` and `jac`
    correspond to the value of the function :math:`\theta` itself and ``y[1]``
    to its first derivative :math:`d\theta/do`.

    Parameters
    ----------
    D : callable
        Callable as ``D(theta, 1)``, which must return both :math:`D` and its
        first derivative evaluated at ``theta``. If the returned `jac` will be
        used, it must also be callable as ``D(theta, 2)`` to obtain :math:`D`,
        its first derivative, and its second derivative evaluated at ``theta``.
        To allow vectorized usage of `fun` and `jac`, `D` must be able to
        accept a NumPy array as ``theta``.
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
    fun : callable
        Function that returns the right-hand side of the system. The calling
        signature is ``fun(o, y)``.
        In non-vectorized usage, ``o`` is a float and ``y`` is any sequence of
        two floats, and `fun` returns a NumPy array with shape (2,). For
        vectorized usage, ``o`` and ``y`` must be NumPy arrays with shapes (n,)
        and (2,n) respectively, and the return is a NumPy array of shape (2,n).
    jac : callable
        Function that returns the Jacobian matrix of the right-hand side of the
        system. The calling signature is ``jac(o, y)``.
        In non-vectorized usage, ``o`` is a scalar and ``y`` is an array-like
        with shape (2,n), and the return is a NumPy array with shape (2,2). In
        vectorized usage, ``o`` and ``y`` must be NumPy arrays with shapes (n,)
        and (2,n) respectively, and the return is a NumPy array of shape
        (2,2,n).

    Other Parameters
    ----------------
    catch_errors : bool, optional
        Whether to catch exceptions that may be attributed to a domain error of
        `D` and convert them to NaN (or +/-Inf) values in the returns of `fun`
        and `jac`. If True, the following exceptions will be caught as domain
        errors:

            *   `ValueError` and `ArithmeticError` (the latter includes
                `ZeroDivisionError`) raised by a call to `D`
            *   `ZeroDivisionError` when attempting to divide by a zero value
                returned by `D`
            *   `TypeError` when assigning to the return array (usually because
                Python arithmetic inside `D` caused that function to return a
                complex value)

        Returning NaN or infinite values signals the domain error to a caller
        that does not expect `fun` and `jac` to raise exceptions to indicate
        this condition (such as SciPy).

        This option is useful in non-vectorized usage, and particularly where
        the invocation of `D` might use native Python mathematical functions
        and types. It is less relevant in vectorized usage and other cases that
        involve only NumPy types and functions, as those will not cause these
        exceptions by default.

        If False (default), the exceptions will be allowed to propagate to the
        callers of `fun` and `jac`.

    See Also
    --------
    BaseSolution
    o

    Notes
    -----
    If `radial` is other than `False`, the PDE is undefined at :math:`r=0`, and
    therefore the returned ODE is also undefined for :math:`o=0`.
    """
    try:
        k = _k[radial]
    except KeyError:
        msg = f"radial must be one of {{{', '.join(repr(key) for key in _k)}}}"
        raise ValueError(msg) from None

    def fun(
        o: float | np.ndarray[tuple[int], np.dtype[np.floating]],
        y: tuple[float, float]
        | np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        theta, dtheta_do = y

        try:
            D1 = D(theta, 1)
        except (ValueError, ArithmeticError):
            if catch_errors:
                return np.array((dtheta_do, np.nan * o), float)
            raise

        D_, dD_dtheta = D1

        k_o = k / o if k else 0

        N = o / 2 + dD_dtheta * dtheta_do

        try:
            R = N / D_
        except ZeroDivisionError:
            if catch_errors:
                R = N * np.inf
            else:
                raise

        d2theta_do2 = -(R + k_o) * dtheta_do

        try:
            return np.array((dtheta_do, d2theta_do2), float)
        except TypeError:
            if catch_errors:
                return np.array((dtheta_do, np.nan * o), float)
            raise

    def jac(
        o: float | np.ndarray[tuple[int], np.dtype[np.floating]],
        y: tuple[float, float]
        | np.ndarray[tuple[int] | tuple[int, ...], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        theta, dtheta_do = y

        J = np.empty((2, 2, *np.shape(o)))
        J[0, 0] = 0
        J[0, 1] = 1

        try:
            D2 = D(theta, 2)  # type: ignore[call-overload]
        except (ValueError, ArithmeticError):
            if catch_errors:
                try:
                    # This second call to D will not be counted by solvers.
                    # However, this is a very unlikely edge case.
                    D1 = D(theta, 1)
                except (ValueError, ArithmeticError):
                    J[1, :] = np.nan
                    return J
                else:
                    D_, dD_dtheta = D1
                    d2D_dtheta2 = np.nan
            else:
                raise
        else:
            D_, dD_dtheta, d2D_dtheta2 = D2

        k_o = k / o if k else 0

        # Jacobian expressions were obtained symbolically
        # Source: ../symbolic/ode_jac.py
        try:
            x0 = 1 / D_
        except ZeroDivisionError:
            if catch_errors:
                x0 = np.inf
            else:
                raise
        x1 = dD_dtheta * dtheta_do
        x2 = x0 * (o + 2 * x1) / 2
        try:
            J[1, 0] = -dtheta_do * x0 * (d2D_dtheta2 * dtheta_do - dD_dtheta * x2)
        except TypeError:
            if catch_errors:
                J[1, 0] = np.nan
            else:
                raise
        try:
            J[1, 1] = -k_o - x0 * x1 - x2
        except TypeError:
            if catch_errors:
                J[1, 1] = np.nan
            else:
                raise

        return J

    return fun, jac


class BaseSolution:
    r"""
    Base class for solutions using the Boltzmann transformation.

    Represents a continuously differentiable function :math:`\theta` of `r` and
    `t` such that:

    .. math::
        \dfrac{\partial\theta}{\partial t} = \nabla\cdot\left[D(\theta)
                    \dfrac{\partial \theta}{\partial r}\mathbf{\hat{r}}\right]

    Parameters
    ----------
    sol : callable
        Solution to an ODE obtained with `ode`. For any float or
        `numpy.ndarray` ``o``, ``sol(o)[0]`` are the values of :math:`\theta`
        at ``o``, and ``sol(o)[1]`` are the values of the derivative
        :math:`d\theta/do` at ``o``.
    D : callable
        Function to evaluate :math:`D` at arbitrary values of the solution.
        Must be callable with a float or NumPy array as its argument.

    See Also
    --------
    ode
    """

    def __init__(
        self,
        sol: Callable[
            [float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]],
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        ],
        D: _D0,
    ) -> None:
        self._sol = sol
        self._D = D

    def __call__(
        self,
        r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None = None,
        t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None = None,
        o: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None = None,
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Evaluate the solution.

        Evaluates and returns :math:`\theta`. May be called either with
        arguments `r` and `t`, or with just `o`.

        Parameters
        ----------
        r : None or float or numpy.ndarray, shape (n,), optional
            Location(s). If this parameter is used, `t` must also be given.
        t : None or float or numpy.ndarray, optional
            Time(s). Values must be positive.
        o : None or float or numpy.ndarray, shape (n,) optional
            Value(s) of the Boltzmann variable. If this parameter is used,
            neither `r` nor `t` can be given.

        Returns
        -------
        float or numpy.ndarray, shape (n,)
        """
        return self._sol(as_o(r, t, o))[0]  # type: ignore[no-any-return]

    def d_dr(
        self,
        r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Spatial derivative of the solution.

        Evaluates and returns :math:`\partial\theta/\partial r`.

        Parameters
        ----------
        r : float or numpy.ndarray, shape (n,)
            Location(s) along the coordinate.
        t : float or numpy.ndarray, shape (n,)
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray, shape (n,)
        """
        return self.d_do(r, t) * do_dr(r, t)

    def d_dt(
        self,
        r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Time derivative of the solution.

        Evaluates and returns :math:`\partial\theta/\partial t`.

        Parameters
        ----------
        r : float or numpy.ndarray, shape (n,)
            Location(s).
        t : float or numpy.ndarray, shape (n,)
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray, shape (n,)
        """
        return self.d_do(r, t) * do_dt(r, t)

    def flux(
        self,
        r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Diffusive flux.

        Returns the diffusive flux of :math:`\theta` in the direction
        :math:`\mathbf{\hat{r}}`, equal to
        :math:`-D(\theta)\partial\theta/\partial r`.

        Parameters
        ----------
        r : float or numpy.ndarray, shape (n,)
            Location(s).
        t : float or numpy.ndarray, shape (n,)
            Time(s). Values must be positive.

        Returns
        -------
        float or numpy.ndarray, shape (n,)
        """
        return -self._D(self(r, t)) * self.d_dr(r, t)

    def d_do(
        self,
        r: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None = None,
        t: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None = None,
        o: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]] | None = None,
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Boltzmann-variable derivative of the solution.

        Evaluates and returns :math:`d\theta/do`, the derivative of
        :math:`\theta` with respect to the Boltzmann variable. May be called
        either with arguments `r` and `t`, or with just `o`.

        Parameters
        ----------
        r : None or float or numpy.ndarray, shape (n,), optional
            Location(s). If this parameter is used, `t` must also be given.
        t : None or float or numpy.ndarray, shape (n,), optional
            Time(s). Values must be positive.
        o : None or float or numpy.ndarray, shape (n,), optional
            Value(s) of the Boltzmann variable. If this parameter is used,
            neither `r` nor `t` can be given.

        Returns
        -------
        float or numpy.ndarray, shape (n,)
        """
        return self._sol(as_o(r, t, o))[1]  # type: ignore[no-any-return]

    def sorptivity(
        self, *, o: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    ) -> float | np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        r"""
        Sorptivity.

        Returns the sorptivity :math:`S` of :math:`\theta`, equal to
        :math:`-2D(\theta)\partial\theta/\partial o`.

        Parameters
        ----------
        o : float or numpy.ndarray, shape (n,)
            Value(s) of the Boltzmann variable.

        Returns
        -------
        float or numpy.ndarray, shape (n,)

        References
        ----------
        [1] PHILIP, J. R. The theory of infiltration: 4. Sorptivity and
        algebraic infiltration equations. Soil Science, 1957, vol. 84, no. 3,
        pp. 257-264.
        """
        return -2 * self._D(self(o=o)) * self.d_do(o=o)
