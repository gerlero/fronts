from __future__ import annotations

from typing import TYPE_CHECKING

import fronts._rootfinding as rootfinding

if TYPE_CHECKING:
    from collections.abc import Callable


def check_result(
    result: rootfinding.Result,
    f: Callable[[float], float],
    ftol: float | None = None,
    f_calls: float | None = None,
    has_root: bool | None = None,
    has_bracket: bool | None = None,
) -> None:
    assert isinstance(result, rootfinding.Result)

    assert 0 <= result.iterations <= result.function_calls
    if f_calls is not None:
        assert result.function_calls == f_calls

    if result.root is not None:
        assert has_root is None or has_root
        assert result.f_root == f(result.root)
        if ftol is not None:
            assert abs(result.f_root) <= ftol
    else:
        assert has_root is None or not has_root
        assert result.root is None
        assert result.f_root is None

    if result.bracket is not None:
        assert has_bracket is None or has_bracket
        assert len(result.bracket) == 2
        assert result.f_bracket is not None
        assert len(result.f_bracket) == 2
        assert all(y == f(x) for x, y in zip(result.bracket, result.f_bracket))
    else:
        assert has_bracket is None or not has_bracket
        assert result.f_bracket is None


def check_iterationlimitreached(
    exc: Exception, f: Callable[[float], float], f_calls: int | None = None
) -> None:
    assert isinstance(exc, rootfinding.IterationLimitReached)

    if f_calls is not None:
        assert exc.function_calls == f_calls

    assert len(exc.interval) == 2
    assert len(exc.f_interval) == 2
    assert all(y == f(x) for x, y in zip(exc.interval, exc.f_interval))


def check_notabracketerror(
    exc: Exception,
    f: Callable[[float], float],
    interval: tuple[float, float],
    f_calls: int | None = None,
) -> None:
    assert isinstance(exc, rootfinding.NotABracketError)

    if f_calls is not None:
        assert f_calls == 2

    assert len(exc.f_interval) == 2
    assert all(y == f(x) for x, y in zip(interval, exc.f_interval))
    assert exc.f_interval[0] * exc.f_interval[1] > 0
