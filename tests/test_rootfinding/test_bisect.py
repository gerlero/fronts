import pytest

import sys

import fronts._rootfinding as rootfinding

from checkobj import (check_result,
                      check_notabracketerror,
                      check_iterationlimitreached)

def f(x):
    f.calls += 1
    return x**2 - 1

def test_success():
    ftol = 4 * sys.float_info.epsilon 
    f.calls = 0

    result = rootfinding.bisect(f, (0.1, -1.5), ftol=ftol)

    check_result(result, f, ftol=ftol, f_calls=f.calls,
                 has_bracket=True, has_root=True)
    assert result.root == pytest.approx(-1)


def test_instant():
    bracket =  (-1.00001, 1)
    f_bracket = (f(bracket[0]), None)
    f.calls = 0

    result = rootfinding.bisect(f, bracket, f_bracket=f_bracket, maxiter=0)

    assert f.calls == 1
    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=True)
    assert result.root == 1


def test_instant2():
    bracket =  (1, -1.00001)
    f_bracket = (f(bracket[0]), None)
    f.calls = 0

    result = rootfinding.bisect(f, bracket, f_bracket=f_bracket, maxiter=0)

    assert f.calls == 1
    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=True)
    assert result.root == 1


def test_notabracket():
    interval = (-1.00001, -2)
    f.calls = 0

    with pytest.raises(rootfinding.NotABracketError) as exc_info:
        rootfinding.bisect(f, interval, maxiter=0)

    exc = exc_info.value
    check_notabracketerror(exc, f, interval, f_calls=f.calls)


def test_iterationlimit():
    bracket = (0.1, -1.5)
    f.calls = 0

    with pytest.raises(rootfinding.IterationLimitReached) as exc_info:
        result = rootfinding.bisect(f, bracket, maxiter=1)

    exc = exc_info.value
    assert f.calls == 3
    check_iterationlimitreached(exc, f, f_calls=f.calls)
    assert any(a == b for a,b in zip(bracket, exc.interval))


def test_invalidftol():
    with pytest.raises(ValueError):
        rootfinding.bisect(f, (0, -0.1), ftol=-1e-3)


def test_invalidmaxiter():
    with pytest.raises(ValueError):
        rootfinding.bisect(f, (0, -0.1), maxiter=-0.5)
