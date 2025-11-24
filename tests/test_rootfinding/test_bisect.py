from __future__ import annotations

import sys

import fronts._rootfinding as rootfinding
import pytest

from .checkobj import check_iterationlimitreached, check_notabracketerror, check_result


def f(x: float) -> float:
    f.calls += 1  # type: ignore [attr-defined]
    return x**2 - 1


def test_success() -> None:
    ftol = 4 * sys.float_info.epsilon
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bisect(f, (0.1, -1.5), ftol=ftol)

    check_result(result, f, ftol=ftol, f_calls=f.calls, has_bracket=True, has_root=True)  # type: ignore [attr-defined]
    assert result.root == pytest.approx(-1)


def test_instant() -> None:
    bracket = (-1.00001, 1)
    f_bracket = (f(bracket[0]), None)
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bisect(f, bracket, f_bracket=f_bracket, maxiter=0)

    assert f.calls == 1  # type: ignore [attr-defined]
    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=True)  # type: ignore [attr-defined]
    assert result.root == 1


def test_instant2() -> None:
    bracket = (1, -1.00001)
    f_bracket = (f(bracket[0]), None)
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bisect(f, bracket, f_bracket=f_bracket, maxiter=0)

    assert f.calls == 1  # type: ignore [attr-defined]
    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=True)  # type: ignore [attr-defined]
    assert result.root == 1


def test_notabracket() -> None:
    interval = (-1.00001, -2)
    f.calls = 0  # type: ignore [attr-defined]

    with pytest.raises(rootfinding.NotABracketError) as exc_info:
        rootfinding.bisect(f, interval, maxiter=0)

    exc = exc_info.value
    check_notabracketerror(exc, f, interval, f_calls=f.calls)  # type: ignore [attr-defined]


def test_iterationlimit() -> None:
    bracket = (0.1, -1.5)
    f.calls = 0  # type: ignore [attr-defined]
    with pytest.raises(rootfinding.IterationLimitReached) as exc_info:
        rootfinding.bisect(f, bracket, maxiter=1)

    exc = exc_info.value
    assert f.calls == 3  # type: ignore [attr-defined]
    check_iterationlimitreached(exc, f, f_calls=f.calls)  # type: ignore [attr-defined]
    assert any(a == b for a, b in zip(bracket, exc.interval, strict=True))


def test_invalidftol() -> None:
    with pytest.raises(ValueError):
        rootfinding.bisect(f, (0, -0.1), ftol=-1e-3)


def test_invalidmaxiter() -> None:
    with pytest.raises(ValueError):
        rootfinding.bisect(f, (0, -0.1), maxiter=-0.5)  # type: ignore [arg-type]
