from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from scipy.interpolate import PchipInterpolator

if TYPE_CHECKING:
    from .D import _VectorizedD2


def inverse(
    o: np.ndarray[tuple[int], np.dtype[np.floating]],
    samples: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> _VectorizedD2:
    r"""
    Extract `D` from samples of a solution.

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
        non-increasing or non-decreasing) and ``samples[-1]`` must be the
        initial value :math:`\theta_i`.

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

        :math:`D` is guaranteed to be continuous; however, its derivatives are
        not.

    See Also
    --------
    o

    Notes
    -----
    An `o` function of :math:`\theta` is constructed by interpolating the input
    data with a PCHIP monotonic cubic spline. The returned `D` uses the spline
    to evaluate the expressions that result from solving the
    Boltzmann-transformed equation for :math:`D`.

    References
    ----------
    [1] GERLERO, G. S.; BERLI, C. L. A.; KLER, P. A. Open-source
    high-performance software packages for direct and inverse solving of
    horizontal capillary flow. Capillarity, 2023, vol. 6, no. 2, pp. 31-40.

    [2] BRUCE, R. R.; KLUTE, A. The measurement of soil moisture diffusivity.
    Soil Science Society of America Journal, 1956, vol. 20, no. 4, pp. 458-462.
    """
    o = np.asarray(o)

    if not np.all(np.diff(o) > 0):
        msg = "o must be strictly increasing"
        raise ValueError(msg)

    samples = np.asarray(samples)

    dsamples = np.diff(samples)
    if not (np.all(dsamples >= -1e-12) or np.all(dsamples <= 1e-12)):
        msg = "samples must be monotonic"
        raise ValueError(msg)

    i = samples[-1]

    samples, indices = np.unique(samples, return_index=True)
    o = o[indices]

    o_func = PchipInterpolator(x=samples, y=o, extrapolate=False)

    o_antiderivative_func = o_func.antiderivative()
    o_antiderivative_i = o_antiderivative_func(i)

    o_funcs = [o_func.derivative(n) for n in range(4)]

    @overload
    def D(theta: float, derivatives: Literal[0] = 0) -> float: ...

    @overload
    def D(
        theta: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        derivatives: Literal[0] = 0,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...

    @overload
    def D(theta: float, derivatives: Literal[1]) -> tuple[float, float]: ...

    @overload
    def D(
        theta: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        derivatives: Literal[1],
    ) -> tuple[
        np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ]: ...

    @overload
    def D(theta: float, derivatives: Literal[2]) -> tuple[float, float, float]: ...

    @overload
    def D(
        theta: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        derivatives: Literal[2],
    ) -> tuple[
        np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ]: ...

    def D(
        theta: float | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        derivatives: Literal[0, 1, 2] = 0,
    ) -> (
        float
        | tuple[float, float]
        | tuple[float, float, float]
        | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
        | tuple[
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        ]
        | tuple[
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        ]
    ):
        Iodtheta = o_antiderivative_func(theta) - o_antiderivative_i

        do_dtheta = o_funcs[1](theta)

        D = -(do_dtheta * Iodtheta) / 2

        if derivatives == 0:
            return D

        o = o_funcs[0](theta)
        d2o_dtheta2 = o_funcs[2](theta)

        dD_dtheta = -(d2o_dtheta2 * Iodtheta + do_dtheta * o) / 2

        if derivatives == 1:
            return D, dD_dtheta

        d3o_dtheta3 = o_funcs[3](theta)

        d2D_dtheta2 = -(d3o_dtheta3 * Iodtheta + 2 * d2o_dtheta2 * o + do_dtheta**2) / 2

        if derivatives == 2:
            return D, dD_dtheta, d2D_dtheta2

        msg = "derivatives must be 0, 1, or 2"
        raise ValueError(msg)

    return D


def sorptivity(
    o: np.ndarray[tuple[int], np.dtype[np.floating]],
    samples: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    i: float | None = None,
    b: float | None = None,
    ob: float = 0,
) -> float:
    r"""
    Extract the sorptivity from samples of a solution.

    Parameters
    ----------
    o : numpy.array_like, shape (n,)
        Points where :math:`\theta` is known, expressed in terms of the
        Boltzmann variable.
    samples : numpy.array_like, shape (n,)
        Values of :math:`\theta` at `o`.
    i : None or float, optional
        Initial value :math:`\theta_i`. If not given, it is taken as
        ``samples[-1]``.
    b : None or float, optional
        Boundary value :math:`\theta_b`. If not given, it is taken as
        ``samples[0]``.

    Returns
    -------
    S : float
        Sorptivity.

    References
    ----------
    [1] PHILIP, J. R. The theory of infiltration: 4. Sorptivity and
    algebraic infiltration equations. Soil Science, 1957, vol. 84, no. 3,
    pp. 257-264.
    """
    o = np.insert(o, 0, ob)
    if b is not None:
        samples = np.insert(samples, 0, b)

    if i is None:
        i = samples[-1]

    return np.trapz(samples - i, o)  # type: ignore[return-value]
