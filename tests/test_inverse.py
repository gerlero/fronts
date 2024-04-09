import fronts
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_exact():
    # Reference: Philip (1960) Table 1, No. 13
    # https://doi.org/10.1071/PH600001
    o = np.linspace(0, 20, 100)

    D = fronts.inverse(o=o, samples=np.exp(-o))

    theta = np.linspace(1e-6, 1, 100)

    assert_allclose(D(theta), 0.5 * (1 - np.log(theta)), rtol=5e-2)


def test_exact_solve():
    o = np.linspace(0, 20, 100)

    D = fronts.inverse(o=o, samples=np.exp(-o))

    theta = fronts.solve(D=D, b=1, i=0)

    assert_allclose(theta(o=o), np.exp(-o), atol=2e-3)


def test_sorptivity():
    o = np.linspace(0, 20, 100)

    assert fronts.sorptivity(o=o, samples=np.exp(-o), i=0, b=1) == pytest.approx(
        1, abs=5e-3
    )
