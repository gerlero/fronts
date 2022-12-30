# -*- coding: utf-8 -*-

"""D functions."""

import functools

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


def from_expr(expr, vectorized=True, max_derivatives=2):
    """
    Create a `D` function from a SymPy-compatible expression.

    Parameters
    ----------
    expr : `sympy.Expression` or str or float
        SymPy-compatible expression containing up to one free symbol.
    vectorized : bool, optional
        Whether the returned `D` must be compatible with a solver that uses
        vectorized calls.

        If True (default), the first argument passed to `D` may always be
        either a float or a NumPy array. However, if False, calls as
        ``D(theta, 1)`` or ``D(theta, 2)`` will assume that ``theta`` is a
        single float, which may allow for optimizations that speed up the
        evaluations required by a solver that does not use vectorized calls.

        Note that, regardless of this setting, calls to `D` that do not ask for
        any derivatives (i.e., calls as ``D(theta)``) will always take floats
        and arrays interchangeably. This behavior ensures that `D` is always
        compatible with the solution classes.
    max_derivatives : int, optional
        Highest-order derivative of `D` that may be required. Can be 0, 1 or 2.
        The default is 2.

    Returns
    -------
    D : callable
        Function to evaluate :math:`D` and its derivatives:

            *   ``D(theta)`` evaluates and returns :math:`D` at ``theta``
            *   If `max_derivatives` >= 1, ``D(theta, 1)`` returns both the
                value of :math:`D` and its first derivative at ``theta``
            *   If `max_derivatives` is 2, ``D(theta, 2)`` returns the value of
                :math:`D`, its first derivative, and its second derivative at
                ``theta``
        
        If `vectorized` is True, the argument ``theta`` may be a single float
        or a NumPy array in all cases. If `vectorized` is False, ``theta`` may
        be either a float or an array when `D` is called as ``D(theta)``, but
        it must be a float otherwise.

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

    if max_derivatives not in {0, 1, 2}:
        raise ValueError("max_derivatives must be 0, 1 or 2")

    if max_derivatives == 0:

        func = sympy.lambdify(theta, expr, modules=np)

        def D(theta):
            try:
                # Convert scalars to NumPy scalars; avoids
                # https://github.com/sympy/sympy/issues/11306
                theta = np.float64(theta)
            except TypeError:
                pass

            return func(theta)

        return D


    exprs = [expr]
    for _ in range(max_derivatives):
        exprs.append(exprs[-1].diff(theta))


    if vectorized:
        
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

            if derivatives == 2 and max_derivatives == 2:
                return funcs[0](theta), funcs[1](theta), funcs[2](theta)

            raise ValueError(f"derivatives must be one of {{{', '.join(str(n) for n in range(max_derivatives+1))}}}")

    else:

        f0v = sympy.lambdify(theta, exprs[0], modules=np)
        f01 = sympy.lambdify(theta, exprs[:2], modules='math')
        if max_derivatives == 2:
            f2 = sympy.lambdify(theta, exprs[2], modules='math')

        def D(theta, derivatives=0):

            if derivatives == 0:
                try:
                    # Convert scalars to NumPy scalars; avoids
                    # https://github.com/sympy/sympy/issues/11306
                    theta = np.float64(theta)
                except TypeError:
                    pass

                return f0v(theta)

            if derivatives == 1:
                return f01(theta)

            if derivatives == 2 and max_derivatives == 2:
                return f01(theta) + [f2(theta)]

            raise ValueError(f"derivatives must be one of {{{', '.join(str(n) for n in range(max_derivatives+1))}}}")

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


def letxs(Lw, Ew, Tw, Ls, Es, Ts, Ks=None, k=None, nu=1e-6, g=9.81, alpha=1.0,
          theta_range=(0.0,1.0)):
    r"""
    Return a diffusivity function that combines the LETx relative permeability
    correlation and the LETs capillary pressure correlation for spontaneous
    imbibition. Both correlations are part of the LET family of hydraulic
    functions.

    Given the saturated hydraulic conductivity :math:`K_S`, irreducible water
    saturation :math:`S_{wir}`, capillary pressure :math:`P_{cir}` at 
    irreducible saturation, and shape parameters :math:`L_w`, :math:`E_w`,
    :math:`T_w` and :math:`L_s`, :math:`E_s`, :math:`T_s`; the LET-based
    diffusivity function :math:`D` is defined as:

    .. math:: D(\theta) = -K_S K_{rw}(\theta) d P_c/d \theta

    where the variable :math:`\theta` is moisture content.

    The functions :math:`K_{rw}` and :math:`P_c` are, respectively, relative
    permeability and capillary pressure, defined as:

    .. math:: K_{rw} =
                     \frac{S_{wp}^{L_w}}{S_{wp}^{L_w} + E_w (1 - S_{wp})^{T_w}}

    .. math:: P_c = 
       P_{cir} \frac{(1 - S_{ws})^{L_s}}{(1 - S_{ws})^{L_s} + E_s S_{ws}^{T_s}}
    
    with:

    .. math:: S_{wp} = S_{ws} = \frac{\theta - \theta_r}{\theta_s - \theta_r}

    and:

    .. math:: P_{cir} = \frac{\rho g}{\alpha}

    Parameters
    ----------
    Lw : float
        :math:`L_w` parameter for the LETx correlation.
    Ew : float
        :math:`E_w` parameter for the LETx correlation.
    Tw : float
        :math:`T_w` parameter for the LETx correlation.
    Ls : float
        :math:`L_s` parameter for the LETs correlation.
    Es : float
        :math:`E_s` parameter for the LETs correlation.
    Ts : float
        :math:`T_s` parameter for the LETs correlation.
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
    alpha : float, optional
        :math:`\alpha` parameter. The default is 1. Must be positive.
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
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its first
                derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        In all cases, the argument ``theta`` may be a single float or a NumPy
        array.

    References
    ----------
    [1] LOMELAND, F. Overview of the LET family of versatile correlations for
    flow functions. In: Proceedings of the International Symposium of the
    Society of Core Analysts, 2018, p. SCA2018-056.

    [2] GERLERO, G. S.; VALDEZ, A.; URTEAGA, R; KLER, P. A. Validity of
    capillary imbibition models in paper-based microfluidic applications.
    Transport in Porous Media, 2022, vol. 141, no. 7, pp. 1-20.
    """

    Ks = _as_Ks(Ks=Ks, k=k, nu=nu, g=g)
    
    # - Code generated with functionstr() from ../symbolic/generate.py - #
    x0 = theta_range[0] - theta_range[1]
    x2 = 1/x0
    x12 = 1/theta_range[1]
    x15 = -x0
    x16 = 1/x15
    x41 = Ls - Ts
    x44 = -x41
    x45 = 2*x44
    x47 = x2*x45
    x52 = Lw**2
    x53 = Ts**2
    x58 = Ls**2
    def D(theta, derivatives=0):
        x1 = theta - theta_range[0]
        x3 = x1*x2
        x4 = x3 + 1
        x5 = x4**Ls
        x6 = -x3
        x7 = x6**Lw
        x8 = Es*x6**Ts
        x9 = 1/x4
        x10 = -Ls*x3 + Ts*x3 + Ts
        x11 = x10*x9
        x13 = 1/x1
        x14 = Ks*x12*x13*x7*x8/(alpha*(theta_range[0]*x12 - 1))
        D = x0*x11*x14*x5/((x5 + x8)**2*(Ew*x4**Tw + x7))
        if derivatives == 0: return D
        x17 = x1*x16
        x18 = x17 - 1
        x19 = 1/x18
        x20 = Ls*x10
        x21 = x19*x20
        x22 = x0*x10
        x23 = x13*x22
        x24 = -x18
        x25 = Ew*x24**Tw
        x26 = x25 + x7
        x27 = 1/x26
        x28 = Lw*x13
        x29 = Tw*x25
        x30 = x19*x2
        x31 = -x28*x7 + x29*x30
        x32 = x24**Ls
        x33 = x32 + x8
        x34 = 1/x33
        x35 = Es*x17**Ts
        x36 = Ts*x13
        x37 = Ls*x32
        x38 = -x16*x37/x24 + x35*x36
        x39 = x34*x38
        x40 = 2*x22
        x42 = x33**(-2)
        x43 = x14*x27*x32*x42*x9
        dD_dtheta = x43*(Lw*x0*x10*x13 + Ts*x0*x10*x13 + x0*x10*x27*x31 - x11 - x21 - x23 - x39*x40 - x41)
        if derivatives == 1: return D, dD_dtheta
        x46 = x13*x45
        x48 = x1**(-2)
        x49 = x22*x48
        x50 = 2*x49
        x51 = 3*x49
        x54 = 2*x11
        x55 = x10*x2
        x56 = 2*x21
        x57 = x18**(-2)
        x59 = 4*x39
        x60 = x27*x31
        x61 = 2*x60
        x62 = Lw*x23
        x63 = Ts*x23
        x64 = x35*x48
        x65 = 1/(x15**2*x24**2)
        x66 = x17**Lw*x48
        d2D_dtheta2 = x43*(-Ls*x19*x47 + Ls*x30*x54 + Lw*Ts*x50 - Lw*x51 + Ts*x46 - Ts*x51 + x11*x59 + x13*x54 + x13*x56 - x2*x20*x57 + x21*x59 - x22*x27*(-Lw*x66 + Tw**2*x25*x65 - x29*x65 + x52*x66) + 6*x22*x38**2*x42 - x22*x59*x60 + x23*x59 - x23*x61 + x28*x45 - x28*x54 - x28*x56 - x34*x40*(-Ts*x64 + x32*x58*x65 - x37*x65 + x53*x64) - x36*x54 - x36*x56 - x44*x59 + x45*x60 - x46 - x47*x9 + x49*x52 + x49*x53 + x50 - x54*x60 + x55*x57*x58 - x56*x60 - x59*x62 - x59*x63 + x61*x62 + x61*x63 + 2*x55/x4**2 + x31**2*x40/x26**2)
        if derivatives == 2: return D, dD_dtheta, d2D_dtheta2
        raise ValueError("derivatives must be 0, 1 or 2")
    # ----------------------- End generated code ----------------------- #

    return D


def letd(L, E, T, Dwt=1.0, theta_range=(0.0,1.0)):
    r"""
    Return a LETd diffusivity function.

    The LETd diffusivity function :math:`D` is defined as:

    .. math:: D(\theta) =
                    D_{wt} \frac{S_{wp}^L}{S_{wp}^L + E (1 - S_{wp})^T}

    with:

    .. math:: S_{wp} = \frac{\theta - \theta_r}{\theta_s - \theta_r}

    Parameters
    ----------
    L : float
        :math:`L` parameter for the LETd correlation.
    E : float
        :math:`E` parameter for the LETd correlation.
    T : float
        :math:`T` parameter for the LETd correlation.
    Dwt : float, optional
        Constant diffusivity factor. The default is 1.
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
            *   ``D(theta, 1)`` returns both the value of :math:`D` and its first
                derivative at ``theta``
            *   ``D(theta, 2)`` returns the value of :math:`D`, its first
                derivative, and its second derivative at ``theta``

        In all cases, the argument ``theta`` may be a single float or a NumPy
        array.

    References
    ----------
    [1] GERLERO, G. S.; VALDEZ, A.; URTEAGA, R; KLER, P. A. Validity of
    capillary imbibition models in paper-based microfluidic applications.
    Transport in Porous Media, 2022, vol. 141, no. 7, pp. 1-20.
    """
    # - Code generated with functionstr() from ../symbolic/generate.py - #
    x1 = -theta_range[1]
    x2 = 1/(theta_range[0] + x1)
    x17 = L**2
    def D(theta, derivatives=0):
        x0 = theta - theta_range[0]
        x3 = (-x0*x2)**L
        x4 = theta + x1
        x5 = E*(x2*x4)**T
        x6 = x3 + x5
        x7 = 1/x6
        x8 = Dwt*x3*x7
        D = x8
        if derivatives == 0: return D
        x9 = L/x0
        x10 = -x0
        x11 = (x10*x2)**L
        x12 = L*x11
        x13 = T*x5
        x14 = x13/x4 - x12/x10
        x15 = x14*x7
        dD_dtheta = x8*(-x15 + x9)
        if derivatives == 1: return D, dD_dtheta
        x16 = x0**(-2)
        x18 = x10**(-2)
        x19 = x4**(-2)
        d2D_dtheta2 = x8*(-L*x16 + 2*x14**2/x6**2 - 2*x15*x9 + x16*x17 - x7*(T**2*x19*x5 + x11*x17*x18 - x12*x18 - x13*x19))
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


def _checked(D, theta=None):
    """
    Call `D` and return its value if valid.

    Raises a `ValueError` if the call fails or does not return a finite,
    positive value; or if its derivative is not finite.

    Parameters
    ----------
    D : callable
        Function to call.
    theta : float or None
        Evaluation value for `D`. If not given, this function works as a
        decorator.

    Returns
    -------
    D : float
        ``D(theta)``.
    """
    if theta is None:
        return functools.partial(_checked, D)

    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            D_, dD_dtheta = D(theta, 1)
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"D({theta}, 1) failed with {e.__class__.__name__}") from e

    try:
        D_ = float(D_)
        dD_dtheta = float(dD_dtheta)
    except TypeError as e:
        raise ValueError(f"D({theta}, 1) returned wrong type") from e

    if not np.isfinite(D_) or D_ <= 0 or not np.isfinite(dD_dtheta):
        raise ValueError(f"D({theta}, 1) returned invalid value")
    
    return D_

