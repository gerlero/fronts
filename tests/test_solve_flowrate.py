import pytest

import numpy as np
from numpy.testing import assert_allclose

import fronts

def test_Qb():

    Qb = 2
    height = 10

    theta = fronts.solve_flowrate(D="theta", radial='cylindrical', i=0.1, Qb=Qb, height=height)

    t = np.array((1e-6, 1, 1.5, 5, 7.314))
    assert np.all(theta.fluxb(t) == pytest.approx(Qb/(2*np.pi*theta.rb(t)*height)))


def test_noflow():
    theta = fronts.solve_flowrate(D="theta", radial='polar', i=0.1, Qb=0)

    o = np.linspace(1e-6, 20, 100)

    assert_allclose(theta(o=o), theta.i)


def test_badbracket():
    with pytest.raises(ValueError):
        theta = fronts.solve_flowrate(D="theta", i=0, Qb=1, radial='polar', b_bracket=(1, 2))
