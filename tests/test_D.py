# noqa: N999
import numpy as np
import pytest

import fronts.D


def test_letxs() -> None:
    D = fronts.D.letxs(
        Lw=1.1,
        Ew=1.2,
        Tw=1.3,
        Ls=1.4,
        Es=1.5,
        Ts=1.6,
        alpha=1.7,
        Ks=1.8,
        theta_range=(0.1, 0.9),
    )

    assert D(0.5, 2) == pytest.approx(
        [0.9538846435439435, 1.8163132590925006, -18.814046264749273]
    )
    assert D(0.15, 2) == pytest.approx(
        [0.027610928712352757, 1.0079989974677264, 17.57492212545586]
    )


def test_letd() -> None:
    D = fronts.D.letd(L=1.1, E=1.2, T=1.3, Dwt=1.4, theta_range=(0.1, 0.9))

    assert D(0.5, 2) == pytest.approx(
        [0.6847101900396568, 2.098998093048757, 0.71237654324091653]
    )
    assert D(0.15, 2) == pytest.approx(
        [0.05762335000343946, 1.3113046321882815, 4.376714990066025]
    )


def test_brooks_and_corey() -> None:
    D = fronts.D.brooks_and_corey(
        n=2.0, l=1.0, alpha=1.7, Ks=1.8, theta_range=(0.1, 0.9)
    )

    assert D(0.5, 2) == pytest.approx(
        [0.11698457776983322, 0.73115361106145766, 2.7418260414804663]
    )
    assert D(0.15, 2) == pytest.approx(
        [0.00064625459558823492, 0.032312729779411753, 0.96938189338235281]
    )


def test_van_genuchten() -> None:
    D = fronts.D.van_genuchten(n=1.1, alpha=1.7, Ks=1.8, theta_range=(0.1, 0.9))

    # Reference values computed with mpmath at 500 significant digits
    assert D(0.5, 2) == pytest.approx(
        [3.7784647708968713e-05, 0.0010868161907242002, 0.028557488277842383]
    )
    # theta close to the residual water content: here the terms of the
    # bracketed expression in the Van Genuchten diffusivity cancel almost
    # completely, so a direct evaluation returns garbage (and eventually
    # exactly 0, which breaks solvers that divide by D)
    assert D(0.15, 2) == pytest.approx(
        [1.5544201803211751e-15, 3.5751664147388968e-13, 7.5078494709525571e-11],
        rel=1e-10,
        abs=0.0,
    )


def test_van_genuchten_near_residual() -> None:
    # Reference values computed with mpmath at 500 significant digits
    cases = {
        (1.1, 0.1): (
            2.6134526117355634e-13,
            3.0054705035246448e-11,
            3.1557440287655587e-9,
        ),
        (1.1, 1e-3): (
            2.613452611709472e-36,
            3.0054705034658917e-32,
            3.1557440286391852e-28,
        ),
        (1.5, 1e-4): (
            2.2222222222244518e-15,
            7.7777777777922467e-11,
            1.9444444444523947e-6,
        ),
        (3.0, 1e-6): (
            2.2222222244444491e-13,
            4.4444444522222314e-7,
            0.44444444638888976,
        ),
    }

    for (n, theta), expected in cases.items():
        D = fronts.D.van_genuchten(n=n)
        assert D(theta, 2) == pytest.approx(expected, rel=1e-10, abs=0.0)

    # vectorized evaluation must match the scalar path
    D = fronts.D.van_genuchten(n=1.1)
    theta_v: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = np.array([0.1, 1e-3])
    for k, values in enumerate(D(theta_v, 2)):
        assert values == pytest.approx(
            [cases[1.1, 0.1][k], cases[1.1, 1e-3][k]], rel=1e-14, abs=0.0
        )


def test_van_genuchten_no_underflow_near_residual() -> None:
    theta: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = np.logspace(
        -12, -1, 100
    )

    for n in (1.1, 1.5, 2.0, 3.0, 5.0, 10.0):
        D = fronts.D.van_genuchten(n=n)
        d0, d1, d2 = D(theta, 2)
        assert np.all(d0 > 0), f"n={n}: min D0 = {np.min(d0)}"
        assert np.all(np.isfinite(d1)), f"n={n}"
        assert np.all(np.isfinite(d2)), f"n={n}"
