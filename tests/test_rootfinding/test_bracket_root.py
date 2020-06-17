import pytest

import sys

import fronts._rootfinding as rootfinding

from checkobj import check_result, check_iterationlimitreached

def f(x):
    f.calls += 1
    return x**2 - 1

def test_success():
    f.calls = 0

    result = rootfinding.bracket_root(f, (0, -0.1))

    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=False)
    assert result.iterations >= 1
    rootfinding.bisect(f, result.bracket, f_bracket=result.f_bracket)


def test_growth_factor2():
    growth_factor = 3.14
    interval = (0, -0.1)
    f.calls = 0

    result = rootfinding.bracket_root(f, interval, growth_factor=growth_factor)

    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=False)

    assert result.bracket[1] - result.bracket[0] == pytest.approx((interval[1] - interval[0])*growth_factor**result.iterations)


def test_instantbracket():
    interval = (0, 2)
    f_interval = (None, f(interval[1]))
    f.calls = 0

    result = rootfinding.bracket_root(f, interval, f_interval=f_interval, maxiter=0)

    assert f.calls == 1
    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=False)
    rootfinding.bisect(f, result.bracket, f_bracket=result.f_bracket)


def test_instantroot():
    interval = (0, -1)
    f_interval = (f(interval[0]), None)
    ftol = 0
    f.calls = 0

    result = rootfinding.bracket_root(f, interval, f_interval=f_interval, ftol=ftol, maxiter=0)

    assert f.calls == 1
    check_result(result, f, ftol=ftol, f_calls=f.calls, has_bracket=False, has_root=True)


def test_instantroot2():
    interval = (-1, 0)
    f_interval = (f(interval[0]), None)
    ftol = 0
    f.calls = 0

    result = rootfinding.bracket_root(f, interval, f_interval=f_interval, ftol=ftol, maxiter=0)

    assert f.calls == 1
    check_result(result, f, ftol=ftol, f_calls=f.calls, has_bracket=False, has_root=True)


def test_instantrootwithbracket():
    interval = (0, -1.00001)
    f_interval = (f(interval[0]), None)
    ftol = 1e-3
    f.calls = 0

    result = rootfinding.bracket_root(f, interval, f_interval=f_interval, ftol=ftol, maxiter=0)

    assert f.calls == 1
    check_result(result, f, ftol=ftol, f_calls=f.calls, has_bracket=True, has_root=True)
    rootfinding.bisect(f, result.bracket)


def test_instantrootwithbracket2():
    interval = (-1.00001, 0)
    f_interval = (f(interval[0]), None)
    ftol = 1e-3
    f.calls = 0

    result = rootfinding.bracket_root(f, interval, f_interval=f_interval, ftol=ftol, maxiter=0)

    assert f.calls == 1
    check_result(result, f, ftol=ftol, f_calls=f.calls, has_bracket=True, has_root=True)
    rootfinding.bisect(f, result.bracket)


def test_iterationlimit():
    f.calls = 0

    with pytest.raises(rootfinding.IterationLimitReached) as exc_info:
        result = rootfinding.bracket_root(f, (-1, -2))

    exc = exc_info.value
    check_iterationlimitreached(exc, f, f_calls=f.calls)

    with pytest.raises(rootfinding.NotABracketError):
        rootfinding.bisect(f, exc.interval, f_bracket=exc.f_interval)


def test_invalidinterval():
    with pytest.raises(ValueError):
        rootfinding.bracket_root(f, (0, 0))


def test_invalidgrowth_factor():
    with pytest.raises(ValueError):
        rootfinding.bracket_root(f, (0, -0.1), growth_factor=0.5)


def test_invalidftol():
    with pytest.raises(ValueError):
        rootfinding.bracket_root(f, (0, -0.1), ftol=-1e-3)


def test_invalidmaxiter():
    with pytest.raises(ValueError):
        rootfinding.bracket_root(f, (0, -0.1), maxiter=-0.5)
