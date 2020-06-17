import pytest

import six

import numpy as np
from numpy.testing import assert_allclose

import fronts

def test_nogradient():
    theta = fronts.solve(D="theta", i=1, b=1)

    o = np.linspace(0, 20, 100)

    assert_allclose(theta(o=o), theta.i)


def test_exact():
    # Reference: Philip (1960) Table 1, No. 13
    # https://doi.org/10.1071/PH600001
    theta = fronts.solve(D="0.5*(1 - log(theta))", i=0, b=1)

    o = np.linspace(0, 20, 100)

    assert_allclose(theta(o=o), np.exp(-o), atol=1e-3)


@pytest.mark.skipif(six.PY2, reason="'explicit' method unavailable on Python 2.7")
def test_exact_explicit():
    theta = fronts.solve(D="0.5*(1 - log(theta))", i=0, b=1, method='explicit')

    o = np.linspace(0, 20, 100)

    assert_allclose(theta(o=o), np.exp(-o), atol=1e-3)


def test_exact_bracket():
    theta = fronts.solve(D="0.5*(1 - log(theta))", i=0, b=1, d_dob_bracket=(-1, -2))

    o = np.linspace(0, 20, 100)

    assert len(theta.d_dob_bracket) == 2
    assert_allclose(theta(o=o), np.exp(-o), atol=1e-3)


def test_badbracket():
    with pytest.raises(ValueError):
        theta = fronts.solve(D="theta", i=0, b=1, d_dob_bracket=(-2, -3))
