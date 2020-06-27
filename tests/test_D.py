import pytest

import functools

import numpy as np
from autograd import deriv

import fronts.D

def van_genuchten_D(theta, m, l, alpha, Ks, theta_range):
    # Reference: Van Genuchten (1980) Equation 11
    # https://doi.org/10.2136/sssaj1980.03615995004400050002x
    
    Se = (theta - theta_range[0])/(theta_range[1] - theta_range[0])

    return (1-m)*Ks/(alpha*m*(theta_range[1] - theta_range[0])) * Se**l*Se**(-1/m) * ((1-Se**(1/m))**(-m) + (1-Se**(1/m))**m - 2)


@pytest.mark.parametrize('theta', (0.5, np.array([0.124, 0.653])))
def test_van_genuchten(theta):

    params = {
        'Ks': 42,
        'alpha': 0.0123,
        'l': 0.27,
        'm': 0.3142,
        'theta_range': (0.123, 0.654)
    }

    D = fronts.D.van_genuchten(**params)

    assert np.all(D(theta, 1)[0] == D(theta))
    assert np.all(D(theta, 2)[0] == D(theta))
    assert np.all(D(theta, 1)[1] == D(theta, 2)[1])

    D_ref = functools.partial(van_genuchten_D, **params)
    
    assert D(theta) == pytest.approx(D_ref(theta))
    assert D(theta, 1)[1] == pytest.approx(deriv(D_ref)(theta))
    assert D(theta, 2)[2] == pytest.approx(deriv(deriv(D_ref))(theta), rel=1e-5)
