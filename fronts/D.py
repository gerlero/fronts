# -*- coding: utf-8 -*-

"""D functions."""

from __future__ import division, absolute_import, print_function

import numpy as np
import sympy


def constant(D0):
    r"""
    Return a constant `D` function.

    Given :math:`D_0`, returns the function `D`:

    .. math:: D(\theta) = D_0

    Parameters
    ----------
    D0 : float
        :math:`D_0`, a positive constant

    Returns
    -------
    D : callable
        Function to evaluate :math:`D` and its derivatives:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        In all cases, the argument ``theta`` may be a single float or a NumPy
        array.

    Notes
    -----
    This function is not particularly useful: a constant `D` will turn a
    diffusion problem into a linear one, which has an exact solution and no
    numerical solvers are necessary. However, it is provided here given that it
    is the simplest supported function.
    """

    if D0 <= 0:
        raise ValueError("D0 must be positive")

    def D(_, derivatives=0):

        if derivatives == 0: return D0

        return (D0,) + (0,)*derivatives

    return D


def from_expr(expr):
    """
    Create a `D` function from a SymPy-compatible expression.

    Parameters
    ----------
    expr : `sympy.Expression` or str or float
        SymPy-compatible expression containing up to one free symbol.

    Returns
    -------
    D : callable
        Function to evaluate :math:`D` and its derivatives:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``
        
        In all cases, the argument ``theta`` may be a single float or a NumPy
        array.

    Notes
    -----
    Users will rarely need to call this function, as all built-in solver
    functions already do so themselves when they receive an expression as `D`.
    """

    expr = sympy.sympify(expr)

    free = expr.free_symbols
    if len(free) == 1:
        [theta] = free
    elif not free:
        return constant(float(expr))
    else:
        raise ValueError("expression cannot contain more than one variable")

    exprs = [expr]
    for _ in range(2):
        exprs.append(exprs[-1].diff(theta))

    funcs = tuple(sympy.lambdify(theta, expr, modules=np) for expr in exprs)

    def D(theta, derivatives=0):

        try:
            # Convert scalars to NumPy scalars; avoids
            # https://github.com/sympy/sympy/issues/11306
            theta = np.float64(theta)
        except TypeError:
            pass

        if derivatives == 0:
            return funcs[0](theta)

        if derivatives == 1:
            return funcs[0](theta), funcs[1](theta)

        if derivatives == 2:
            return funcs[0](theta), funcs[1](theta), funcs[2](theta)

        raise ValueError("derivatives must be 0, 1 or 2")

    return D


def power_law(k, a=1.0, epsilon=0.0):
    r"""
    Return a power-law `D` function.

    Given the scalars `a`, `k` and :math:`\varepsilon`, returns a function `D`
    defined as:

    .. math:: D(\theta) = a\theta^k + \varepsilon

    Parameters
    ----------
    k : float
        Exponent
    a : float, optional
        Constant factor. The default is 1.
    epsilon : float, optional
        :math:`\varepsilon`, the deviation term. The default is 0.

    Returns
    -------
    D : callable
        Function to evaluate :math:`D` and its derivatives:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        In all cases, the argument ``theta`` may be a single float or a NumPy
        array.

    Notes
    -----
    Keep in mind that, depending on the parameters, the returned `D` does not
    necessarily map every value of :math:`\theta` to a positive value.
    """

    def D(theta, derivatives=0):

        D = a*theta**k + epsilon

        if derivatives == 0: return D

        dD_dtheta = k*D/theta

        if derivatives == 1: return D, dD_dtheta

        d2D_dtheta2 = (k-1)*dD_dtheta/theta

        if derivatives == 2: return D, dD_dtheta, d2D_dtheta2

        raise ValueError("derivatives must be 0, 1, or 2")

    return D


def _as_Ks(Ks=None, k=None, nu=1e-6, g=9.81):
    r"""
    Return the saturated hydraulic conductivity, computed from the instrinsic
    permeability if necessary.

    Parameters
    ----------
    Ks : None or float, optional
        :math:`K_S`, the saturated hydraulic conductivity. Must be positive. If
        neither `Ks` nor `k` are given, the saturated hydraulic conductivity is
        assumed to be 1.
    k : None or float, optional
        Intrinsic permeability of the porous medium. Can be given in place of
        `Ks`, which results in the saturated hydraulic conductivity being
        computed using :math:`K_S = kg/\nu`. Must be positive.
    nu : float, optional
        :math:`\nu`, the kinematic viscosity of the wetting fluid. Only used if
        `k` is passed instead of `Ks`. Must be positive. Defaults to 1e-6,
        approximately the kinematic viscosity of water at 20°C in SI units.
    g : float, optional
        Magnitude of the gravitational acceleration. Only used if `k` is passed
        instead of `Ks`. Must be positive. Defaults to 9.81, the gravity of
        Earth in SI units.

    Returns
    -------
    Ks : float
        :math:`K_S`, the saturated hydraulic conductivity
    """
    if Ks is not None:
        if k is not None:
            raise TypeError("cannot pass both Ks and k")
        if Ks <= 0:
            raise ValueError("Ks must be positive")
        return Ks

    elif k is not None:
        if k <= 0:
            raise ValueError("k must be positive")
        if nu <= 0:
            raise ValueError("nu must be positive")
        if g <= 0:
            raise ValueError("g must be positive")
        return g*k/nu

    else:
        return 1


def brooks_and_corey(n, l=1.0, alpha=1.0, Ks=None, k=None, nu=1e-6, g=9.81,
                     theta_range=(0.0,1.0)):
    r"""
    Return a Brooks and Corey moisture diffusivity function.

    Given the saturated hydraulic conductivity :math:`K_S` and parameters
    :math:`\alpha`, `n`, `l`, :math:`\theta_r` and :math:`\theta_s`, the Brooks
    and Corey moisture diffusivity function `D` is defined as:

    .. math:: D(\theta) = \frac{K_S S_e^{1/n + l + 1}}
                               {\alpha n (\theta_s-\theta_r)}
        

    where:

    .. math:: S_e = \frac{\theta-\theta_r}{\theta_s-\theta_r}

    and :math:`\theta` is water content.


    Parameters
    ----------
    n : float
        `n` parameter.
    l : float, optional
        `l` parameter. The default is 1.
    alpha : float, optional
        :math:`\alpha` parameter. The default is 1. Must be positive.
    Ks : None or float, optional
        :math:`K_S`, the saturated hydraulic conductivity. Must be positive. If
        neither `Ks` nor `k` are given, the saturated hydraulic conductivity is
        assumed to be 1.
    k : None or float, optional
        Intrinsic permeability of the porous medium. Can be given in place of
        `Ks`, which results in the saturated hydraulic conductivity being
        computed using :math:`K_S = kg/\nu`. Must be positive.
    nu : float, optional
        :math:`\nu`, the kinematic viscosity of the wetting fluid. Only used if
        `k` is passed instead of `Ks`. Must be positive. Defaults to 1e-6,
        approximately the kinematic viscosity of water at 20°C in SI units.
    g : float, optional
        Magnitude of the gravitational acceleration. Only used if `k` is passed
        instead of `Ks`. Must be positive. Defaults to 9.81, the gravity of
        Earth in SI units.
    theta_range : sequence of two floats, optional
        (:math:`\theta_r`, :math:`\theta_s`), where :math:`\theta_r` is the
        minimum (also known as residual) water content and :math:`\theta_s` is
        the maximum water content. The default is (0, 1). :math:`\theta_s` must
        be greater than :math:`\theta_r`.

    Returns
    -------
    D : callable
        Function to evaluate :math:`D` and its derivatives:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        In all cases, the argument ``theta`` may be a single float or a NumPy
        array.

    References
    ----------
    [1] BROOKS, R.; COREY, T. Hydraulic properties of porous media. Hydrology
    Papers, Colorado State University, 1964, vol. 24, p. 37.
    """

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    Ks = _as_Ks(Ks=Ks, k=k, nu=nu, g=g)

    if theta_range[1] <= theta_range[0]:
        raise ValueError("theta_range[1] must be greater than theta_range[0]")

    # - Code generated with functionstr() from ../symbolic/generate.py - #
    # Source: ../symbolic/brooks_and_corey.py
    x1 = 1/n
    x2 = n*(l + 2) + 1
    x3 = x1*x2
    def D(theta, derivatives=0):
        x0 = theta - theta_range[0]
        x4 = Ks*x1*(-x0/(theta_range[0] - theta_range[1]))**x3/alpha
        D = x4/x0
        if derivatives == 0: return D
        dD_dtheta = x4*(x3 - 1)/x0**2
        if derivatives == 1: return D, dD_dtheta
        d2D_dtheta2 = x4*(-3*x3 + 2 + x2**2/n**2)/x0**3
        if derivatives == 2: return D, dD_dtheta, d2D_dtheta2
        raise ValueError("derivatives must be 0, 1 or 2")
    # ----------------------- End generated code ----------------------- #

    return D


def van_genuchten(n=None, m=None, l=0.5, alpha=1.0, Ks=None, k=None, nu=1e-6,
                  g=9.81, theta_range=(0.0,1.0)):
    r"""
    Return a Van Genuchten moisture diffusivity function.

    Given the saturated hydraulic conductivity :math:`K_S` and parameters
    :math:`\alpha`, `m`, `l`, :math:`\theta_r` and :math:`\theta_s`, the Van
    Genuchten moisture diffusivity function `D` is defined as:

    .. math:: D(\theta)=\frac{(1-m)K_S}{\alpha m (\theta_s-\theta_r)}
        S_e^{l-\frac{1}{m}}\left((1-S_e^\frac{1}{m})^{-m} +
        (1-S_e^\frac{1}{m})^m - 2 \right)

    where:

    .. math:: S_e = \frac{\theta-\theta_r}{\theta_s-\theta_r}

    and :math:`\theta` is water content.

    In common usage, the `m` parameter is replaced with an `n` parameter so
    that :math:`m=1-1/n`. This function supports either parameter.

    Parameters
    ----------
    n : float, optional
        `n` parameter in the Van Genuchten model. Must be >1. Either `n` or `m`
        must be given (but not both).
    m : float, optional
        `m` parameter in the Van Genuchten model. Must be strictly between 0
        and 1. Either `n` or `m` must be given (but not both).
    l : float, optional
        Pore connectivity parameter. The default is 0.5.
    alpha : float, optional
        :math:`\alpha` parameter of the Van Genuchten model. The default is 1.
        Must be positive.
    Ks : None or float, optional
        :math:`K_S`, the saturated hydraulic conductivity. Must be positive. If
        neither `Ks` nor `k` are given, the saturated hydraulic conductivity is
        assumed to be 1.
    k : None or float, optional
        Intrinsic permeability of the porous medium. Can be given in place of
        `Ks`, which results in the saturated hydraulic conductivity being
        computed using :math:`K_S = kg/\nu`. Must be positive.
    nu : float, optional
        :math:`\nu`, the kinematic viscosity of the wetting fluid. Only used if
        `k` is passed instead of `Ks`. Must be positive. Defaults to 1e-6,
        approximately the kinematic viscosity of water at 20°C in SI units.
    g : float, optional
        Magnitude of the gravitational acceleration. Only used if `k` is passed
        instead of `Ks`. Must be positive. Defaults to 9.81, the gravity of
        Earth in SI units.
    theta_range : sequence of two floats, optional
        (:math:`\theta_r`, :math:`\theta_s`), where :math:`\theta_r` is the
        minimum (also known as residual) water content and :math:`\theta_s` is
        the maximum water content. The default is (0, 1). :math:`\theta_s` must
        be greater than :math:`\theta_r`.

    Returns
    -------
    D : callable
        Function to evaluate :math:`D` and its derivatives:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        In all cases, the argument ``theta`` may be a single float or a NumPy
        array.

    Notes
    -----
    The expression used is the one found in Van Genuchten's original paper [1],
    but with the addition of the optional `l` parameter.

    References
    ----------
    [1] VAN GENUCHTEN, M. Th. A closed-form equation for predicting the
    hydraulic conductivity of unsaturated soils. Soil Science Society of
    America Journal, 1980, vol. 44, no 5, p. 892-898.
    """

    if n is not None:
        if m is not None:
            raise TypeError("cannot pass both n and m")
        if n <= 1:
            raise ValueError("n must be greater than 1.0")
        m = 1-1/n

    elif m is None:
        raise TypeError("either n or m must be given")

    if not (0<m<1):
        raise ValueError("m must be strictly between 0.0 and 1.0")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    Ks = _as_Ks(Ks=Ks, k=k, nu=nu, g=g)

    if theta_range[1]-theta_range[0] <= 0:
        raise ValueError("theta_range[1] must be greater than theta_range[0]")

    # - Code generated with functionstr() from ../symbolic/generate.py - #
    # Source: ../symbolic/van_genuchten.py
    x1 = 1/(theta_range[0] - theta_range[1])
    x3 = 1/m
    x8 = l - x3
    def D(theta, derivatives=0):
        x0 = theta - theta_range[0]
        x2 = -x0*x1
        x4 = x2**x3
        x5 = (1 - x4)**m
        x6 = x5 - 2
        x7 = (x5*x6 + 1)/x5
        x9 = Ks*x1*x2**x8*x3*(m - 1)/alpha
        D = x7*x9
        if derivatives == 0: return D
        x10 = (x1*(-theta + theta_range[0]))**x3
        x11 = (1 - x10)**m
        x12 = x4/(x10 - 1)
        x13 = (x11*(x11 - 2) + 1)/x11
        dD_dtheta = x9*(-x12*x13 + 2*x12*(x11 - 1) + x13*x8)/x0
        if derivatives == 1: return D, dD_dtheta
        x14 = x4 - 1
        x15 = x2**(2*x3)/x14**2
        x16 = 4*x5 - 4
        x17 = x4/x14
        x18 = x7*x8
        x19 = x17*x7
        x20 = x15*x7
        x21 = x3*x5
        x22 = x3*x6
        d2D_dtheta2 = x9*(-x15*x16 + x16*x17*x8 - 2*x17*x18 + x17*(-x17*x21 - x17*x22 + 3*x17*x5 + x17*x6 + x21 + x22 - 2*x5 + 2) - x18 - x19*x3 + x19 + x20*x3 + x20 + x7*x8**2)/x0**2
        if derivatives == 2: return D, dD_dtheta, d2D_dtheta2
        raise ValueError("derivatives must be 0, 1 or 2")
    # ----------------------- End generated code ----------------------- #

    return D


def richards(C, kr, Ks=None, k=None, nu=1e-6, g=9.81):
    r"""
    Return a moisture diffusivity function for a Richards equation problem.

    Given `K_S` and the functions `C` and :math:`k_r` (whose argument is either
    water content or saturation), returns the function:

    .. math:: D(\theta) = \frac{K_S k_r(\theta)}{C(\theta)}

    This function helps transform problems of the horizontal Richards equation
    (for which :math:`K_S`, :math:`k_r`, and `C` are known parameters) into
    problems of the moisture diffusivity equation that can be solved with this
    library.

    Parameters
    ----------
    C : callable
        Capillary capacity function (also known as hydraulic capacity
        function). A twice-differentiable function that maps values of
        :math:`\theta` to positive values. It can be called as ``C(theta)`` to
        evaluate it at ``theta``. It can also be called as ``C(theta, n)`` with
        ``n`` equal to 1 or 2, in which case the first ``n`` derivatives of the
        function evaluated at the same ``theta`` are included (in order) as
        additional return values. While mathematically a scalar function, `C`
        operates in a vectorized fashion with the same semantics when ``theta``
        is a `numpy.ndarray`.
    kr : callable
        :math:`k_r`, the relative permeability function (also known as relative
        conductivity function). A twice-differentiable function that maps
        values of :math:`\theta` to positive values (usually between 0 and 1).
        It can be called as ``kr(theta)`` to evaluate it at ``theta``. It can
        also be called as ``kr(theta, n)`` with ``n`` equal to 1 or 2, in which
        case the first ``n`` derivatives of the function evaluated at the same
        ``theta`` are included (in order) as additional return values. While
        mathematically a scalar function, `kr` operates in a vectorized fashion
        with the same semantics when ``theta`` is a `numpy.ndarray`.
    Ks : None or float, optional
        :math:`K_S`, the saturated hydraulic conductivity. Must be positive. If
        neither `Ks` nor `k` are given, the saturated hydraulic conductivity is
        assumed to be 1.
    k : None or float, optional
        Intrinsic permeability of the porous medium. Can be given in place of
        `Ks`, which results in the saturated hydraulic conductivity being
        computed using :math:`K_S = kg/\nu`. Must be positive.
    nu : float, optional
        :math:`\nu`, the kinematic viscosity of the wetting fluid. Only used if
        `k` is passed instead of `Ks`. Must be positive. Defaults to 1e-6,
        approximately the kinematic viscosity of water at 20°C in SI units.
    g : float, optional
        Magnitude of the gravitational acceleration. Only used if `k` is passed
        instead of `Ks`. Must be positive. Defaults to 9.81, the gravity of
        Earth in SI units.

    Returns
    -------
    D : callable
        Function to evaluate :math:`D` and its derivatives:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its
                first derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        In all cases, the argument ``theta`` may be a single float or a NumPy
        array.
    """

    Ks = _as_Ks(Ks=Ks, k=k, nu=nu, g=g)

    def D(theta, derivatives=0):

        if derivatives == 0: return Ks*kr(theta)/C(theta)

        K_ = [Ks*kr for kr in kr(theta, derivatives)]
        C_ = C(theta, derivatives)

        D = K_[0]/C_[0]

        dD_dtheta = (K_[1]*C_[0] - K_[0]*C_[1])/C_[0]**2

        if derivatives == 1: return D, dD_dtheta

        d2D_dtheta2 = (K_[2] - 2*dD_dtheta*C_[1] - D*C_[2])/C_[0]

        if derivatives == 2: return D, dD_dtheta, d2D_dtheta2

        raise ValueError("derivatives must be 0, 1, or 2")

    return D
