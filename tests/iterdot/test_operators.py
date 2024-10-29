from operator import add, contains, eq, ge, gt, is_, is_not, le, lt, ne, not_

import pytest

from iterdot.operators import (
    Contains,
    GreaterEqual,
    GreaterThan,
    Is,
    IsEqual,
    IsNot,
    LessEqual,
    LessThan,
    Not,
    NotEqual,
    Unpacked,
)


def test_LessThan():
    assert lt(0, 1) == LessThan(1)(0)
    assert lt(2, 1) == LessThan(1)(2)
    assert lt(1, 1) == LessThan(1)(1)

    with pytest.raises(TypeError, match="not supported"):
        _ = LessThan(1)("5")


def test_GreaterThan():
    assert gt(0, 1) == GreaterThan(1)(0)
    assert gt(2, 1) == GreaterThan(1)(2)
    assert gt(1, 1) == GreaterThan(1)(1)

    with pytest.raises(TypeError, match="not supported"):
        _ = GreaterThan(1)("5")


def test_LessEqual():
    assert le(0, 1) == LessEqual(1)(0)
    assert le(2, 1) == LessEqual(1)(2)
    assert le(1, 1) == LessEqual(1)(1)

    with pytest.raises(TypeError, match="not supported"):
        _ = LessEqual(1)("5")


def test_GreaterEqual():
    assert ge(0, 1) == GreaterEqual(1)(0)
    assert ge(2, 1) == GreaterEqual(1)(2)
    assert ge(1, 1) == GreaterEqual(1)(1)

    with pytest.raises(TypeError, match="not supported"):
        _ = GreaterEqual(1)("5")


def test_IsEqual():
    assert eq(0, 1) == IsEqual(1)(0)
    assert eq(2, 1) == IsEqual(1)(2)
    assert eq(1, 1) == IsEqual(1)(1)
    assert eq(1, "5") == IsEqual(1)("5")


def test_NotEqual():
    assert ne(0, 1) == NotEqual(1)(0)
    assert ne(2, 1) == NotEqual(1)(2)
    assert ne(1, 1) == NotEqual(1)(1)
    assert ne(1, "5") == NotEqual(1)("5")


def test_Is():
    assert is_(0, 1) == Is(1)(0)
    assert is_(2, 1) == Is(1)(2)
    assert is_(1, 1) == Is(1)(1)
    assert is_(1, "5") == Is(1)("5")


def test_IsNot():
    assert is_not(0, 1) == IsNot(1)(0)
    assert is_not(2, 1) == IsNot(1)(2)
    assert is_not(1, 1) == IsNot(1)(1)
    assert is_not(1, "5") == IsNot(1)("5")


def test_Not():
    assert not_("A".islower()) == Not(str.islower)("A")


def test_Contains():
    assert contains([1, 2, 3], 1) == Contains(1)([1, 2, 3])
    assert contains([1, 2, 3], 5) == Contains(5)([1, 2, 3])

    with pytest.raises(TypeError, match="not iterable"):
        _ = Contains("5")(1)  # pyright: ignore[reportArgumentType]


def test_unpacked():
    assert add(*(1, 2)) == Unpacked(add)((1, 2))

    with pytest.raises(TypeError, match="must be an iterable"):
        _ = Unpacked(add)(1)  # pyright: ignore[reportAny, reportArgumentType]
