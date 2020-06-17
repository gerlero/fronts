import pytest

import numpy as np
from numpy.testing import assert_allclose

import fronts

def test_exact():
    # Reference: Philip (1960) Table 1, No. 13
    # https://doi.org/10.1071/PH600001
    o = np.linspace(0, 20, 100)

    theta = fronts.solve_from_guess(D="0.5*(1 - log(theta))", i=0, b=1,
                         o_guess=o, guess=0.5)

    assert_allclose(theta(o=o), np.exp(-o), atol=1e-3)
   