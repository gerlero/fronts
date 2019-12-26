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


def van_genuchten(n=None, m=None, l=0.5, alpha=1.0, Ks=1.0,
                  S_range=(0.0,1.0)):
    r"""
    Return a Van Genuchten moisture diffusivity function.

    Given the parameters :math:`K_s`, :math:`\alpha`, `m`, `l`, :math:`S_r` and
    :math:`S_s`, the Van Genuchten moisture diffusivity function `D`
    is defined as:

    .. math:: D(S)=\frac{(1-m)K_s}{\alpha m (S_s-S_r)}
        S_e^{(l-\frac{1}{m})}\left((1-S_e^\frac{1}{m})^{-m} +
        (1-S_e^\frac{1}{m})^m - 2 \right)

    where:

    .. math:: S_e = \frac{S-S_r}{S_s-S_r}

    and `S` is either water content or saturation.

    In common usage, the `m` parameter is replaced with an `n` parameter so
    that :math:`m=1-\tfrac{1}{n}`. This function supports either parameter.

    Parameters
    ----------
    n : float, optional
        `n` parameter in the Van Genuchten model. Must be >1. You must pass
        either `n` or `m` (but not both).
    m : float, optional
        `m` parameter in the Van Genucthen model. Must be strictly between 0
        and 1. You must pass either `n` or `m` (but not both).
    l : float, optional
        Pore connectivity parameter. The default is 0.5. Must be strictly
        between 0 and 1.
    alpha : float, optional
        :math:`\alpha` parameter of the Van Genucthen model. The default is 1.
        Must be positive.
    Ks : float, optional
        :math:`K_s`, the hydraulic conductivity when saturated. The default is
        1. Must be positive.
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
        called as ``D(S)`` to evaluate it at `S`. It can  also be called as
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
    hydraulic conductivity of unsaturated soils. Soil science society of
    America journal, 1980, vol. 44, no 5, p. 892-898.
    """

    if n is not None:
        if m is not None:
            raise TypeError("cannot pass both n and m")
        if n <= 1:
            raise ValueError("n must be greater than 1.0")
        m = 1-1/n

    elif m is None:
        raise TypeError("must pass either n or m")

    if not (0<m<1):
        raise ValueError("m must be strictly between 0.0 and 1.0")

    if not (0<l<1):
        raise ValueError("l must be strictly between 0.0 and 1.0")

    if alpha<=0:
        raise ValueError("alpha must be positive")

    if Ks<=0:
        raise ValueError("Ks must be positive")

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


def richards(K, C):
    r"""
    Return a moisture diffusivity function for a Richards equation problem.

    Given the functions `K` and `C` (where `S` is either water content or
    saturation) returns the function:

    .. math:: D(S) = \frac{K(S)}{C(S)}

    This effectively converts horizontal Richards equation problems (for which
    those two functions are parameters) into moisture diffusivity problems that
    can be solved using this library.

    Parameters
    ----------
    K : callable
        Hydraulic conductivity function. A twice-differentiable function that 
        maps values of `S` to positive values. It can be called as ``K(S)`` to
        evaluate it at `S`. It can also be called as ``K(S, n)`` with `n` equal
        to 1 or 2, in which case the first `n` derivatives of the function
        evaluated at the same `S` are included (in order) as additional return
        values. While mathematically a scalar function, `K` operates in a
        vectorized fashion with the same semantics when `S` is a
        `numpy.ndarray`.
    C : callable
        Capillary capacity function. A twice-differentiable function that maps
        values of `S` to positive values. It can be called as ``C(S)`` to
        evaluate it at `S`. It can also be called as ``C(S, n)`` with `n` equal
        to 1 or 2, in which case the first `n` derivatives of the function
        evaluated at the same `S` are included (in order) as additional return
        values. While mathematically a scalar function, `C` operates in a
        vectorized fashion with the same semantics when `S` is a
        `numpy.ndarray`.

    Returns
    -------
    D : callable
        Twice-differentiable function that maps values of `S` in the domains of
        both `K` and `C` to positive values. It can be called as ``D(S)`` to
        evaluate it at `S`. It can also be called as ``D(S, n)`` with `n` equal
        to 1 or 2, in which case the first `n` derivatives of the function
        evaluated at the same `S` are included (in order) as additional return
        values. While mathematically a scalar function, `D` operates in a
        vectorized fashion with the same semantics when `S` is a
        `numpy.ndarray`.
    """

    def D(S, derivatives=0):

        if derivatives == 0: return K(S)/C(S)

        K_ = K(S, derivatives)
        C_ = C(S, derivatives)

        D = K_[0]/C_[0]

        dD_dS = (K_[1]*C_[0] - K_[0]*C_[1])/C_[0]**2

        if derivatives == 1: return D, dD_dS

        d2D_dS2 = (K_[2] - 2*dD_dS*C_[1] - D*C_[2])/C_[0]

        if derivatives == 2: return D, dD_dS, d2D_dS2

        raise ValueError("derivatives must be 0, 1, or 2")

    return D

