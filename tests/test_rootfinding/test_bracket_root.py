from __future__ import annotations

import fronts._rootfinding as rootfinding
import pytest

from .checkobj import check_iterationlimitreached, check_result


def f(x: float) -> float:
    f.calls += 1  # type: ignore [attr-defined]
    return x**2 - 1


def test_success() -> None:
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bracket_root(f, (0, -0.1))

    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=False)  # type: ignore [attr-defined]
    assert result.iterations >= 1
    assert result.bracket is not None
    assert result.f_bracket is not None
    rootfinding.bisect(f, result.bracket, f_bracket=result.f_bracket)


def test_growth_factor2() -> None:
    growth_factor = 3.14
    interval = (0, -0.1)
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bracket_root(f, interval, growth_factor=growth_factor)

    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=False)  # type: ignore [attr-defined]
    assert result.bracket is not None
    assert result.bracket[1] - result.bracket[0] == pytest.approx(
        (interval[1] - interval[0]) * growth_factor**result.iterations
    )


def test_instantbracket() -> None:
    interval = (0, 2)
    f_interval = (None, f(interval[1]))
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bracket_root(f, interval, f_interval=f_interval, maxiter=0)

    assert f.calls == 1  # type: ignore [attr-defined]
    check_result(result, f, f_calls=f.calls, has_bracket=True, has_root=False)  # type: ignore [attr-defined]
    assert result.bracket is not None
    assert result.f_bracket is not None
    rootfinding.bisect(f, result.bracket, f_bracket=result.f_bracket)


def test_instantroot() -> None:
    interval = (0, -1)
    f_interval = (f(interval[0]), None)
    ftol = 0
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bracket_root(
        f, interval, f_interval=f_interval, ftol=ftol, maxiter=0
    )

    assert f.calls == 1  # type: ignore [attr-defined]
    check_result(
        result,
        f,
        ftol=ftol,
        f_calls=f.calls,  # type: ignore [attr-defined]
        has_bracket=False,
        has_root=True,
    )


def test_instantroot2() -> None:
    interval = (-1, 0)
    f_interval = (f(interval[0]), None)
    ftol = 0
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bracket_root(
        f, interval, f_interval=f_interval, ftol=ftol, maxiter=0
    )

    assert f.calls == 1  # type: ignore [attr-defined]
    check_result(
        result,
        f,
        ftol=ftol,
        f_calls=f.calls,  # type: ignore [attr-defined]
        has_bracket=False,
        has_root=True,
    )


def test_instantrootwithbracket() -> None:
    interval = (0, -1.00001)
    f_interval = (f(interval[0]), None)
    ftol = 1e-3
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bracket_root(
        f, interval, f_interval=f_interval, ftol=ftol, maxiter=0
    )

    assert f.calls == 1  # type: ignore [attr-defined]
    check_result(result, f, ftol=ftol, f_calls=f.calls, has_bracket=True, has_root=True)  # type: ignore [attr-defined]
    assert result.bracket is not None
    rootfinding.bisect(f, result.bracket)


def test_instantrootwithbracket2() -> None:
    interval = (-1.00001, 0)
    f_interval = (f(interval[0]), None)
    ftol = 1e-3
    f.calls = 0  # type: ignore [attr-defined]

    result = rootfinding.bracket_root(
        f, interval, f_interval=f_interval, ftol=ftol, maxiter=0
    )

    assert f.calls == 1  # type: ignore [attr-defined]
    check_result(result, f, ftol=ftol, f_calls=f.calls, has_bracket=True, has_root=True)  # type: ignore [attr-defined]
    assert result.bracket is not None
    rootfinding.bisect(f, result.bracket)


def test_iterationlimit() -> None:
    f.calls = 0  # type: ignore [attr-defined]

    with pytest.raises(rootfinding.IterationLimitReached) as exc_info:
        rootfinding.bracket_root(f, (-1, -2))

    exc = exc_info.value
    check_iterationlimitreached(exc, f, f_calls=f.calls)  # type: ignore [attr-defined]

    with pytest.raises(rootfinding.NotABracketError):
        rootfinding.bisect(f, exc.interval, f_bracket=exc.f_interval)


def test_invalidinterval() -> None:
    with pytest.raises(ValueError):
        rootfinding.bracket_root(f, (0, 0))


def test_invalidgrowth_factor() -> None:
    with pytest.raises(ValueError):
        rootfinding.bracket_root(f, (0, -0.1), growth_factor=0.5)


def test_invalidftol() -> None:
    with pytest.raises(ValueError):
        rootfinding.bracket_root(f, (0, -0.1), ftol=-1e-3)


def test_invalidmaxiter() -> None:
    with pytest.raises(ValueError):
        rootfinding.bracket_root(f, (0, -0.1), maxiter=-0.5)  # type: ignore [arg-type]
