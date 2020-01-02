"""Included D functions."""

from __future__ import division, absolute_import, print_function

def constant(D0):
    """
    Return a constant `D` function.

    Given :math:`D_0`, returns the function `D`:

    .. math:: D(S) = D_0

    Parameters
    ----------
    D0 : float
        :math:`D_0`, a positive constant

    Returns
    -------
    D : callable
        Function that maps any value of `S` to the given constant. It can be
        called as ``D(S)`` to obtain the value. It can also be called as
        ``D(S, n)`` with `n` equal to 1 or 2, in which case the first `n`
        derivatives of the function, which are always zero, are included (in
        order) as additional return values. While mathematically a scalar
        function, `D` operates in a vectorized fashion with the same semantics
        when `S` is a `numpy.ndarray`.

    Notes
    -----
    This function is not particularly useful: a constant `D` will turn a
    diffusion problem into a linear one, which has an exact solution and no
    numerical solvers are necessary. However, it is provided here given that it
    is the simplest supported function.
    """

    if D0 <= 0:
        raise ValueError("D0 must be positive")

    def D(S, derivatives=0):

        if derivatives == 0: return D0

        return (D0,) + (0,)*derivatives

    return D


def power_law(k, a=1.0, epsilon=0.0):
    r"""
    Return a power-law `D` function.

    Given the scalars `a`, `k` and :math:`\varepsilon`, returns a function `D`
    defined as:

    .. math:: D(S) = aS^k + \varepsilon

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
        Twice-differentiable function that maps `S` to values according to the
        expression. It can be called as ``D(S)`` to evaluate it at `S`. It can
        also be called as ``D(S, n)`` with `n` equal to 1 or 2, in which case
        the first `n` derivatives of the function evaluated at the same `S` are
        included (in order) as additional return values. While mathematically a
        scalar function, `D` operates in a vectorized fashion with the same
        semantics when `S` is a `numpy.ndarray`.

    Notes
    -----
    Keep in mind that, depending on the parameters, the returned `D` does not
    necessarily map every value of `S` to a positive value.
    """

    def D(S, derivatives=0):

        D = a*S**k + epsilon

        if derivatives == 0: return D

        dD_dS = k*D/S

        if derivatives == 1: return D, dD_dS

        d2D_dS2 = (k-1)*dD_dS/S

        if derivatives == 2: return D, dD_dS, d2D_dS2

        raise ValueError("derivatives must be 0, 1, or 2")

    return D


def _as_Ks(Ks=None, k=None, nu=1e-6, g=9.81):
    r"""
    Return the saturated hydraulic conductivity, computed from the instrinsic
    permeability if necessary.

    Parameters
    ----------
    Ks : float, optional
        :math:`K_S`, the saturated hydraulic conductivity. Must be positive. If
        neither `Ks` nor `k` are given, the saturated hydraulic conductivity is
        assumed to be 1.
    k : float, optional
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


def van_genuchten(n=None, m=None, l=0.5, alpha=1.0, Ks=None, k=None, nu=1e-6,
                  g=9.81, S_range=(0.0,1.0)):
    r"""
    Return a Van Genuchten moisture diffusivity function.

    Given the saturated hydraulic conductivity :math:`K_S` and parameters
    :math:`\alpha`, `m`, `l`, :math:`S_r` and :math:`S_s`, the Van Genuchten
    moisture diffusivity function `D` is defined as:

    .. math:: D(S)=\frac{(1-m)K_S}{\alpha m (S_s-S_r)}
        S_e^{l-\frac{1}{m}}\left((1-S_e^\frac{1}{m})^{-m} +
        (1-S_e^\frac{1}{m})^m - 2 \right)

    where:

    .. math:: S_e = \frac{S-S_r}{S_s-S_r}

    and `S` is either water content or saturation.

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
        Pore connectivity parameter. The default is 0.5. Must be strictly
        between 0 and 1.
    alpha : float, optional
        :math:`\alpha` parameter of the Van Genuchten model. The default is 1.
        Must be positive.
    Ks : float, optional
        :math:`K_S`, the saturated hydraulic conductivity. Must be positive. If
        neither `Ks` nor `k` are given, the saturated hydraulic conductivity is
        assumed to be 1.
    k : float, optional
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
    S_range : (float, float), optional
        the tuple (:math:`S_r`, :math:`S_s`), where :math:`S_r` is the minimum
        (also known as residual) and :math:`S_s` the maximum water content (or
        saturation, depending on the meaning given to `S`). The default is
        (0, 1). :math:`S_s` must be greater than :math:`S_r`.

    Returns
    -------
    D : callable
        Twice-differentiable function that maps values of `S` in the open
        interval (:math:`S_r`, :math:`S_s`) to positive values. It can be
        called as ``D(S)`` to evaluate it at `S`. It can also be called as
        ``D(S, n)`` with `n` equal to 1 or 2, in which case the first `n`
        derivatives of the function evaluated at the same `S` are included (in
        order) as additional return values. While mathematically a scalar
        function, `D` operates in a vectorized fashion with the same semantics
        when `S` is a `numpy.ndarray`.

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

    if not (0<l<1):
        raise ValueError("l must be strictly between 0.0 and 1.0")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    Ks = _as_Ks(Ks=Ks, k=k, nu=nu, g=g)

    if S_range[1]-S_range[0] <= 0:
        raise ValueError("S_range[1] must be greater than S_range[0]")

    # The following expressions were obtained symbolically
    # (see ../symbolic/van_genuchten.py)

    x0 = 1/m
    x1 = -S_range[0]
    x3 = 1/(S_range[0] - S_range[1])

    def D(S, derivatives=0):

        x2 = S + x1
        x4 = -x2*x3
        x5 = x4**x0
        x6 = 1/x5
        x7 = (1 - x5)**m
        x8 = 1/x7
        x9 = x6*(x7 + x8 - 2)
        x10 = x0*x9
        x11 = Ks*x3*x4**l*(m - 1)/alpha

        D = x10*x11

        if derivatives == 0: return D

        x12 = (x2/(S_range[1] + x1))**x0
        x13 = 1 - x12
        x14 = x13**m
        x15 = 1/x14
        x16 = x14 + x15 - 2
        x17 = x0*x11

        dD_dS = x17*x6*(l*x16 - x0*x16 - x12*(x14 - x15)/x13)/x2

        if derivatives == 1: return D, dD_dS

        x18 = l*x9
        x19 = 1/(x5 - 1)
        x20 = x19*(x7 - x8)
        x21 = 2*x0
        x22 = x0*x7
        x23 = x0*x8
        x24 = x19*x5

        d2D_dS2 = x17*(2*l*x20 + x10*(x0 + 1) - x18*x21 + x18*(l - 1) + x19*(-x22*x24 + x22 + x23*x24 - x23 + x24*x7 + x24*x8 - x7 + x8) - x20*x21)/x2**2

        if derivatives == 2: return D, dD_dS, d2D_dS2

        raise ValueError("derivatives must be 0, 1, or 2")

    return D


def richards(C, kr, Ks=None, k=None, nu=1e-6, g=9.81):
    r"""
    Return a moisture diffusivity function for a Richards equation problem.

    Given `K_S` and the functions `C` and :math:`k_r` of `S` (where `S` is
    either water content or saturation), returns the function:

    .. math:: D(S) = \frac{K_S k_r(S)}{C(S)}

    This function helps transform problems of the horizontal Richards equation
    (for which :math:`K_S`, :math:`k_r`, and `C` are known parameters) into
    problems of the moisture diffusivity equation that can be solved with this
    library.

    Parameters
    ----------
    C : callable
        Capillary capacity function (also known as hydraulic capacity
        function). A twice-differentiable function that maps values of `S` to
        positive values. It can be called as ``C(S)`` to evaluate it at `S`. It
        can also be called as ``C(S, n)`` with `n` equal to 1 or 2, in which
        case the first `n` derivatives of the function evaluated at the same
        `S` are included (in order) as additional return values. While
        mathematically a scalar function, `C` operates in a vectorized fashion
        with the same semantics when `S` is a `numpy.ndarray`.
    kr : callable
        :math:`k_r`, the relative permeability function (also known as relative
        conductivity function). A twice-differentiable function that maps
        values of `S` to positive values (usually between 0 and 1). It can be
        called as ``kr(S)`` to evaluate it at `S`. It can also be called as
        ``kr(S, n)`` with `n` equal to 1 or 2, in which case the first `n`
        derivatives of the function evaluated at the same `S` are included
        (in order) as additional return values. While mathematically a scalar
        function, `kr` operates in a vectorized fashion with the same semantics
        when `S` is a `numpy.ndarray`.
    Ks : float, optional
        :math:`K_S`, the saturated hydraulic conductivity. Must be positive. If
        neither `Ks` nor `k` are given, the saturated hydraulic conductivity is
        assumed to be 1.
    k : float, optional
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
        Twice-differentiable function that maps values of `S` in the domains of
        both `kr` and `C` to positive values. It can be called as ``D(S)`` to
        evaluate it at `S`. It can also be called as ``D(S, n)`` with `n` equal
        to 1 or 2, in which case the first `n` derivatives of the function
        evaluated at the same `S` are included (in order) as additional return
        values. While mathematically a scalar function, `D` operates in a
        vectorized fashion with the same semantics when `S` is a
        `numpy.ndarray`.
    """

    Ks = _as_Ks(Ks=Ks, k=k, nu=nu, g=g)

    def D(S, derivatives=0):

        if derivatives == 0: return Ks*kr(S)/C(S)

        K_ = [Ks*kr for kr in kr(S, derivatives)]
        C_ = C(S, derivatives)

        D = K_[0]/C_[0]

        dD_dS = (K_[1]*C_[0] - K_[0]*C_[1])/C_[0]**2

        if derivatives == 1: return D, dD_dS

        d2D_dS2 = (K_[2] - 2*dD_dS*C_[1] - D*C_[2])/C_[0]

        if derivatives == 2: return D, dD_dS, d2D_dS2

        raise ValueError("derivatives must be 0, 1, or 2")

    return D

