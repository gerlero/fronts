"""
This module defines the Boltzmann variable, transformation of the partial
differential equation into the ODE in terms of that variable, and inverse
transformation of the ODE's solution into a solution to the partial
differential equation.
"""

from __future__ import division, absolute_import, print_function
import six

import numpy as np


def o(r, t):
    r"""
    Transform to the Boltzmann variable.

    Returns the Boltzmann variable at the given `r` and `t`, which is the
    result of applying the Boltzmann transformation:

    .. math::
        o(r,t) = r/\sqrt t

    Parameters
    ----------
    r : float or numpy.ndarray
        Location(s). If a `numpy.ndarray`, it must have a shape broadcastable
        with `t`.
    t : float or numpy.ndarray
        Time(s). If a `numpy.ndarray`, it must have a shape broadcastable with
        `r`. Values must be positive.

    Returns
    -------
    o : float or numpy.ndarray
        The return is a float if both `r` and `t` are floats. Otherwise it is a
        `numpy.ndarray` of the shape that results from broadcasting `r` and
        `t`.

    See also
    --------
    do_dr
    do_dt
    r
    t
    as_o
    """
    return r/t**0.5


def do_dr(r, t):
    r"""
    Spatial derivative of the Boltzmann transformation.

    Returns the partial derivative :math:`\partial o/\partial r` evaluated at
    (`r`, `t`).

    Parameters
    ----------
    r : float or numpy.ndarray
        Location(s). If a `numpy.ndarray`, it must have a shape broadcastable
        with `t`.
    t : float or numpy.ndarray
        Time(s). If a `numpy.ndarray`, it must have a shape broadcastable with
        `r`. Values must be positive.

    Returns
    -------
    do_dr : float or numpy.ndarray
        The return is a float if both `r` and `t` are floats. Otherwise it is a
        `numpy.ndarray` of the shape that results from broadcasting `r` and
        `t`.

    See also
    --------
    o
    do_dt
    """
    return 1/t**0.5


def do_dt(r, t):
    r"""
    Time derivative of the Boltzmann transformation.

    Returns the partial derivative :math:`\partial o/\partial t` evaluated at
    (`r`, `t`).

    Parameters
    ----------
    r : float or numpy.ndarray
        Location(s). If a `numpy.ndarray`, it must have a shape broadcastable
        with `t`.
    t : float or numpy.ndarray
        Time(s). If a `numpy.ndarray`, it must have a shape broadcastable with
        `r`. Values must be positive.

    Returns
    -------
    do_dt : float or numpy.ndarray
        The return is a float if both `r` and `t` are floats. Otherwise it is a
        `numpy.ndarray` of the shape that results from broadcasting `r` and
        `t`.

    See also
    --------
    o
    do_dr
    """
    return -o(r,t)/(2*t)


def r(o, t):
    """
    Transform back from the Boltzmann variable into `r`.

    Parameters
    ----------
    o : float or numpy.ndarray
        Value(s) of the Boltzmann variable. If a `numpy.ndarray`, it must have
        a shape broadcastable with `t`.
    t : float or numpy.ndarray
        Time(s). If a `numpy.ndarray`, it must have a shape broadcastable with
        `o`. Values must not be negative.

    Returns
    -------
    r : float or numpy.ndarray
        The return is a float if both `o` and `t` are floats. Otherwise it is a
        `numpy.ndarray` of the shape that results from broadcasting `o` and
        `t`.

    See also
    --------
    o
    t
    """
    return o*t**0.5


def t(o, r):
    """
    Transform back from the Boltzmann variable into `t`.

    Parameters
    ----------
    o : float or numpy.ndarray
        Value(s) of the Boltzmann variable. If a `numpy.ndarray`, it must have
        a shape broadcastable with `r`. Values must not be zero.
    r : float or numpy.ndarray
        Location(s). If a `numpy.ndarray`, it must have a shape broadcastable
        with `o`.

    Returns
    -------
    t : float or numpy.ndarray
        The return is a float if both `o` and `r` are floats. Otherwise it is a
        `numpy.ndarray` of the shape that results from broadcasting `o` and
        `r`.

    See also
    --------
    o
    r
    """
    return (r/o)**2


_o = o
def as_o(r=None, t=None, o=None):
    """
    Transform to the Boltzmann variable if called with `r` and `t`. Passes the
    values through if called with `o` only. On other combinations of arguments,
    it raises a `TypeError` with a message explaining valid usage.

    This function is a helper to define other functions that may be called
    either with `r` and `t`, or with just `o`.

    Parameters
    ----------
    r : None or float or numpy.ndarray, optional
        Location(s). If a `numpy.ndarray`, it must have a shape broadcastable
        with `t`. If this parameter is used, you must also pass `t` and cannot
        pass `o`.
    t : None or float or numpy.ndarray, optional
        Time(s). If a `numpy.ndarray`, it must have a shape broadcastable with
        `r`. Values must be positive. If this parameter is used, you must also
        pass `r` and cannot pass `o`.
    o : None or float or numpy.ndarray, optional
        Value(s) of the Boltzmann variable. If this parameter is used, you
        cannot pass `r` or `t`.

    Returns
    -------
    o : float or numpy.ndarray
        Passes `o` through if it is given. Otherwise, it returns ``o(r,t)``.

    See also
    --------
    o
    """
    if o is not None:
        if r is not None or t is not None:
            raise TypeError("must pass either r and t, or just o")
        return o

    if r is None or t is None:
        raise TypeError("must pass either r and t, or just o")
    return _o(r, t)


_k = {False: 0,
      'cylindrical': 1,
      'polar': 1,
      'spherical': 2}

def ode(D, radial=False):
    r"""
    Transform the PDE into an ODE.

    Given a positive function `D` and coordinate unit vector
    :math:`\mathbf{\hat{r}}`, transform the partial differential equation
    (PDE) in which :math:`\theta` is the unknown function of `r` and `t`:

    .. math:: \dfrac{\partial\theta}{\partial t} =
        \nabla\cdot\left[D(\theta)\dfrac{\partial\theta}{\partial r}
        \mathbf{\hat{r}}\right]

    into an ordinary differential equation (ODE) where :math:`\theta` is an
    unknown function of the Boltzmann variable `o`.

    This function returns the `fun` and `jac` callables that may be used to
    solve the ODE with the solvers included with SciPy (`scipy.integrate`
    module). The second-order ODE is expressed as a system of first-order ODEs
    with independent variable `o` where ``y[0]`` in `fun` and `jac` correspond
    to the value of the function :math:`\theta` itself and ``y[1]`` to its
    first derivative :math:`d\theta/do`.

    `fun` and `jac` support both non-vectorized usage (where their first
    argument is a float) as well as vectorized usage (when `numpy.ndarray`
    objects are passed as both arguments).

    Parameters
    ----------
    D : callable
        Twice-differentiable function that maps the range of :math:`\theta` to
        positive values. It can be called as ``D(theta)`` to evaluate it at
        ``theta``. It can also be called as ``D(theta, n)`` with ``n`` equal to
        1 or 2, in which case the first ``n`` derivatives of the function
        evaluated at the same ``theta`` are included (in order) as additional
        return values. While mathematically a scalar function, `D` operates in
        a vectorized fashion with the same semantics when ``theta`` is a
        `numpy.ndarray`.
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

    Returns
    -------
    fun : callable
        Function that returns the right-hand side of the system. The calling
        signature is ``fun(o, y)``.
    jac : callable
        Function that returns the Jacobian matrix of the right-hand side of the
        system. The calling signature is ``jac(o, y)``.

    See also
    --------
    BaseSolution
    o

    Notes
    -----
    If `radial` is not `False`, the PDE is undefined at :math:`r=0`, and
    therefore the returned ODE is also undefined for :math:`o=0`.
    """

    try:
        k = _k[radial]
    except KeyError:
        six.raise_from(ValueError("radial must be one of {{{}}}".format(
                       ", ".join(repr(key) for key in _k))),
                       None)


    def fun(o, y):

        theta, dtheta_do = y

        D_, dD_dtheta = D(theta, 1)

        k_o = k/o if k else 0

        d2theta_do2 = -((o/2 + dD_dtheta*dtheta_do)/D_ + k_o)*dtheta_do

        return np.array((dtheta_do, d2theta_do2))


    def jac(o, y):

        theta, dtheta_do = y

        jacobian = np.empty((2,2)+np.shape(o))

        D_, dD_dtheta, d2D_dtheta2 = D(theta, 2)

        k_o = k/o if k else 0

        # The following expressions were obtained symbolically
        # (see ../symbolic/ode_jac.py)
        jacobian[0,0] = 0
        jacobian[0,1] = 1
        jacobian[1,0] = -dtheta_do*(2*D_*d2D_dtheta2*dtheta_do - dD_dtheta*(2*dD_dtheta*dtheta_do + o))/(2*D_**2)
        jacobian[1,1] = -k_o - 2*dD_dtheta*dtheta_do/D_ - o/(2*D_)

        return jacobian

    return fun, jac


class BaseSolution(object):
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
        `D` used to obtain `sol`. Must be the same function that was passed to
        `ode`.

    See also
    --------
    ode
    """
    def __init__(self, sol, D):
        self._sol = sol
        self._D = D

    def __call__(self, r=None, t=None, o=None):
        r"""
        Evaluate the solution.

        Evaluates and returns :math:`\theta`. May be called either with
        arguments `r` and `t`, or with just `o`.

        Parameters
        ----------
        r : None or float or numpy.ndarray, optional
            Location(s). If a `numpy.ndarray`, it must have a shape
            broadcastable with `t`. If this parameter is used, you must also
            pass `t` and cannot pass `o`.
        t : None or float or numpy.ndarray, optional
            Time(s). If a `numpy.ndarray`, it must have a shape broadcastable
            ith `r`. Values must be positive. If this parameter is used, you
            must also pass `r` and cannot pass `o`.
        o : None or float or numpy.ndarray, optional
            Value(s) of the Boltzmann variable. If this parameter is used, you
            cannot pass `r` or `t`.

        Returns
        -------
        float or numpy.ndarray
            If `o` is passed, the return is of the same type and shape as `o`.
            Otherwise, return is a float if both `r` and `t` are floats, or a
            `numpy.ndarray` of the shape that results from broadcasting `r` and
            `t`.
        """
        return self._sol(as_o(r,t,o))[0]

    def d_dr(self, r, t):
        r"""
        Spatial derivative of the solution.

        Evaluates and returns :math:`\partial\theta/\partial r`.

        Parameters
        ----------
        r : float or numpy.ndarray
            Location(s) along the coordinate. If a `numpy.ndarray`, it must
            have a shape broadcastable with `t`.
        t : float or numpy.ndarray
            Time(s). If a `numpy.ndarray`, it must have a shape broadcastable
            with `r`. Values must be positive.

        Returns
        -------
        float or numpy.ndarray
            The return is a float if both `r` and `t` are floats. Otherwise it
            is a `numpy.ndarray` of the shape that results from broadcasting
            `r` and `t`.
        """
        return self.d_do(r,t) * do_dr(r,t)

    def d_dt(self, r, t):
        r"""
        Time derivative of the solution.

        Evaluates and returns :math:`\partial\theta/\partial t`.

        Parameters
        ----------
        r : float or numpy.ndarray
            Location(s). If a `numpy.ndarray`, it must have a shape
            broadcastable with `t`.
        t : float or numpy.ndarray
            Time(s). If a `numpy.ndarray`, it must have a shape broadcastable
            with `r`. Values must be positive.

        Returns
        -------
        float or numpy.ndarray
            The return is a float if both `r` and `t` are floats. Otherwise it
            is a `numpy.ndarray` of the shape that results from broadcasting
            `r` and `t`.
        """
        return self.d_do(r,t) * do_dt(r,t)

    def flux(self, r, t):
        r"""
        Diffusive flux.

        Returns the diffusive flux of :math:`\theta` in the direction
        :math:`\mathbf{\hat{r}}`, equal to
        :math:`-D(\theta)\partial\theta/\partial r`.

        Parameters
        ----------
        r : float or numpy.ndarray
            Location(s). If a `numpy.ndarray`, it must have a shape
            broadcastable with `t`.
        t : float or numpy.ndarray
            Time(s). If a `numpy.ndarray`, it must have a shape broadcastable
            with `r`. Values must be positive.

        Returns
        -------
        float or numpy.ndarray
            The return is a float if both `r` and `t` are floats. Otherwise it
            is a `numpy.ndarray` of the shape that results from broadcasting
            `r` and `t`.
        """
        return -self._D(self(r,t)) * self.d_dr(r,t)

    def d_do(self, r=None, t=None, o=None):
        r"""
        Boltzmann-variable derivative of the solution.

        Evaluates and returns :math:`d\theta/do`, the derivative of
        :math:`\theta` with respect to the Boltzmann variable. May be called
        either with arguments `r` and `t`, or with just `o`.

        Parameters
        ----------
        r : None or float or numpy.ndarray, optional
            Location(s). If a `numpy.ndarray`, it must have a shape
            broadcastable with `t`. If this parameter is used, you must also
            pass `t` and cannot pass `o`.
        t : None or float or numpy.ndarray, optional
            Time(s). If a `numpy.ndarray`, it must have a shape broadcastable
            with `r`. Values must be positive. If this parameter is used, you
            must also pass `r` and cannot pass `o`.
        o : None or float or numpy.ndarray, optional
            Value(s) of the Boltzmann variable. If this parameter is used, you
            cannot pass `r` or `t`.

        Returns
        -------
        float or numpy.ndarray
            If `o` is passed, the return is of the same type and shape as `o`.
            Otherwise, the return is a float if both `r` and `t` are floats, or
            a `numpy.ndarray` of the shape that results from broadcasting `r`
            and `t`.
        """
        return self._sol(as_o(r,t,o))[1]
