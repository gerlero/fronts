"""
This module defines the Boltzmann variable, transformation of the partial
differential equation into the ODE in terms of that variable, and inverse
transformation of the ODE's solution into a solution to the partial
differential equation.
"""

from __future__ import division, absolute_import, print_function

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
        `o`. Values must be positive.

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
        a shape broadcastable with `r`.
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
    r : float or numpy.ndarray, optional
        Location(s). If a `numpy.ndarray`, it must have a shape broadcastable
        with `t`. If this parameter is used, you must also pass `t` and cannot
        pass `o`.
    t : float or numpy.ndarray, optional
        Time(s). If a `numpy.ndarray`, it must have a shape broadcastable with
        `r`. Values must be positive. If this parameter is used, you must also
        pass `r` and cannot pass `o`.
    o : float or numpy.ndarray, optional
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
      'spherical': 2}

def ode(D, radial=False):
    r"""
    Transform the PDE into an ODE.

    Given a positive function `D` and coordinate unit vector
    :math:`\mathbf{\hat{r}}`, transform the partial differential equation
    (PDE) in which `S` is the unknown function of `r` and `t`:

    .. math:: \dfrac{\partial S}{\partial t} =
        \nabla\cdot\left[D\left(S\right)\dfrac{\partial S}{\partial r}
        \mathbf{\hat{r}}\right]

    into an ordinary differential equation (ODE) where `S` is an unknown
    function of the Boltzmann variable `o`.

    This function returns the `fun` and `jac` callables that may be used to
    solve the ODE with the solvers included with SciPy (`scipy.integrate`
    module). The second-order ODE is expressed as a system of first-order ODEs
    with independent variable `o` where ``y[0]`` in `fun` and `jac` correspond
    to the value of the function `S` itself and ``y[1]`` to its first
    derivative :math:`dS/do`.

    `fun` and `jac` support both non-vectorized usage (where their first
    argument is a float) as well as vectorized usage (when `numpy.ndarray`
    objects are passed as both arguments).

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
        raise ValueError("radial must be one of {}".format(tuple(_k.keys())))


    def fun(o, y):

        S, dS_do = y

        D_, dD_dS = D(S, 1)

        k_o = k/o if k else 0

        d2S_do2 = -((o/2 + dD_dS*dS_do)/D_ + k_o)*dS_do

        return np.array((dS_do, d2S_do2))


    def jac(o, y):

        S, dS_do = y

        jacobian = np.empty((2,2)+np.shape(o))

        D_, dD_dS, d2D_dS2 = D(S, 2)

        k_o = k/o if k else 0

        # The following expressions were obtained symbolically
        # (see ../symbolic/ode_jac.py)
        jacobian[0,0] = 0
        jacobian[0,1] = 1
        jacobian[1,0] = -dS_do*(2*D_*d2D_dS2*dS_do - dD_dS*(2*dD_dS*dS_do + o))/(2*D_**2)
        jacobian[1,1] = -k_o - 2*dD_dS*dS_do/D_ - o/(2*D_)

        return jacobian

    return fun, jac



class BaseSolution(object):
    r"""
    Base class for solutions using the Boltzmann transformation.

    Its methods describe a continuous solution to any problem of finding a
    function `S` of `r` and `t` such that:

    .. math::
         \dfrac{\partial S}{\partial t} = \nabla\cdot\left[D\left(S\right)
                        \dfrac{\partial S}{\partial r}\mathbf{\hat{r}}\right]

    Parameters
    ----------
    sol : callable
        Solution to an ODE obtained with `ode`. For any float or
        `numpy.ndarray` `o`, ``sol(o)[0]`` are the values of `S` at `o`, and
        ``sol(o)[1]`` the values of the derivative `dS/do` at `o`.
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

    def S(self, r=None, t=None, o=None):
        """
        `S`, the unknown function.

        May be called either with parameters `r` and `t`, or with just `o`.

        Parameters
        ----------
        r : float or numpy.ndarray, optional
            Location(s). If a `numpy.ndarray`, it must have a shape
            broadcastable with `t`. If this parameter is used, you must also
            pass `t` and cannot pass `o`.
        t : float or numpy.ndarray, optional
            Time(s). If a `numpy.ndarray`, it must have a shape broadcastable
            ith `r`. Values must be positive. If this parameter is used, you
            must also pass `r` and cannot pass `o`.
        o : float or numpy.ndarray, optional
            Value(s) of the Boltzmann variable. If this parameter is used, you
            cannot pass `r` or `t`.

        Returns
        -------
        S : float or numpy.ndarray
            If `o` is passed, the return is of the same type and shape as `o`.
            Otherwise, return is a float if both `r` and `t` are floats, or a
            `numpy.ndarray` of the shape that results from broadcasting `r` and
            `t`.
        """
        return self._sol(as_o(r,t,o))[0]

    def dS_dr(self, r, t):
        r"""
        :math:`\partial S/\partial r`, spatial derivative of `S`.

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
        dS_dr : float or numpy.ndarray
            The return is a float if both `r` and `t` are floats. Otherwise it
            is a `numpy.ndarray` of the shape that results from broadcasting
            `r` and `t`.
        """
        return self.dS_do(r,t) * do_dr(r,t)

    def dS_dt(self, r, t):
        r"""
        :math:`\partial S/\partial t`, time derivative of `S`.

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
        dS_dt : float or numpy.ndarray
            The return is a float if both `r` and `t` are floats. Otherwise it
            is a `numpy.ndarray` of the shape that results from broadcasting
            `r` and `t`.
        """
        return self.dS_do(r,t) * do_dt(r,t)

    def flux(self, r, t):
        r"""
        Diffusive flux of `S`.

        Returns the diffusive flux of `S` in the direction
        :math:`\mathbf{\hat{r}}`, equal to :math:`-D(S)\partial S/\partial r`.

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
        flux : float or numpy.ndarray
            The return is a float if both `r` and `t` are floats. Otherwise it
            is a `numpy.ndarray` of the shape that results from broadcasting
            `r` and `t`.
        """
        return -self._D(self.S(r,t)) * self.dS_dr(r,t)

    def dS_do(self, r=None, t=None, o=None):
        r"""
        :math:`dS/do`, derivative of `S` with respect to the Boltzmann
        variable.

        May be called either with parameters `r` and `t`, or with just `o`.

        Parameters
        ----------
        r : float or numpy.ndarray, optional
            Location(s). If a `numpy.ndarray`, it must have a shape
            broadcastable with `t`. If this parameter is used, you must also
            pass `t` and cannot pass `o`.
        t : float or numpy.ndarray, optional
            Time(s). If a `numpy.ndarray`, it must have a shape broadcastable
            with `r`. Values must be positive. If this parameter is used, you
            must also pass `r` and cannot pass `o`.
        o : float or numpy.ndarray, optional
            Value(s) of the Boltzmann variable. If this parameter is used, you
            cannot pass `r` or `t`.

        Returns
        -------
        dS_do : float or numpy.ndarray
            If `o` is passed, the return is of the same type and shape as `o`.
            Otherwise, the return is a float if both `r` and `t` are floats, or
            a `numpy.ndarray` of the shape that results from broadcasting `r`
            and `t`.
        """
        return self._sol(as_o(r,t,o))[1]





