import fronts.D
import pytest


def test_letxs():
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


def test_letd():
    D = fronts.D.letd(L=1.1, E=1.2, T=1.3, Dwt=1.4, theta_range=(0.1, 0.9))

    assert D(0.5, 2) == pytest.approx(
        [0.6847101900396568, 2.098998093048757, 0.71237654324091653]
    )
    assert D(0.15, 2) == pytest.approx(
        [0.05762335000343946, 1.3113046321882815, 4.376714990066025]
    )
