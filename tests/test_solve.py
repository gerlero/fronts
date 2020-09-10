import pytest

import numpy as np
from numpy.testing import assert_allclose

import fronts
import fronts.D

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


def test_HF135():

    r = np.array([0.    , 0.0025, 0.005 , 0.0075, 0.01  , 0.0125, 0.015 , 0.0175,
           0.02  , 0.0225, 0.025 , 0.0275, 0.03  , 0.0325, 0.035 , 0.0375,
           0.04  , 0.0425, 0.045 , 0.0475, 0.05  ])

    t = 60

    # Data obtained with the porousMultiphaseFoam toolbox, version 1906
    # https://github.com/phorgue/porousMultiphaseFoam
    S_pmf = np.array([0.945   , 0.944845, 0.944188, 0.942814, 0.940517, 0.937055,
           0.93214 , 0.925406, 0.916379, 0.904433, 0.888715, 0.868016,
           0.840562, 0.803597, 0.752494, 0.678493, 0.560999, 0.314848,
           0.102755, 0.102755, 0.102755])
    Sflux_pmf = np.array([ 2.66135e-04,  2.66133e-04,  2.66111e-04,  2.66038e-04,
            2.65869e-04,  2.65542e-04,  2.64975e-04,  2.64060e-04,
            2.62644e-04,  2.60523e-04,  2.57404e-04,  2.52863e-04,
            2.46269e-04,  2.36619e-04,  2.22209e-04,  1.99790e-04,
            1.61709e-04,  7.64565e-05, -2.45199e-21, -7.35598e-21,
            0.00000e+00])

    epsilon = 1e-7

    # Wetting of an HF135 membrane, Van Genuchten model
    # Data from Buser (PhD thesis, 2016)
    # http://hdl.handle.net/1773/38064
    S_range = (0.0473, 0.945)
    k = 5.50e-13  # m**2
    alpha = 0.2555  # 1/m
    n = 2.3521
    Si = 0.102755  # Computed from P0

    Sb = S_range[1] - epsilon

    D = fronts.D.van_genuchten(n=n, alpha=alpha, k=k, theta_range=S_range)

    S = fronts.solve(D=D, i=Si, b=Sb, itol=1e-7)

    assert_allclose(S(r,t), S_pmf, atol=1e-3)
    assert_allclose(S.flux(r,t), Sflux_pmf, atol=1e-6)


def test_exact_explicit():
    pytest.importorskip('scipy', minversion='1.4.0',
                        reason="'explicit' method requires SciPy >= 1.4.0")

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
