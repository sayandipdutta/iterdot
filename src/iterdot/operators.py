"""
Defines functors for different operators
"""

from collections.abc import Callable, Container
from operator import contains, eq, ge, gt, is_, is_not, le, lt, ne


def binop_factory[T1, T2](
    op: Callable[[T1, T2], bool],
) -> Callable[[T2], Callable[[T1], bool]]:
    def comparator(rhs: T2, /) -> Callable[[T1], bool]:
        """
        Given an operand (rhs), returns a closure that takes another operand (lhs), which returns bool

        Example:
            >>> list(filter(LessThan(5), range(3, 8)))
            [3, 4]
            >>> assert IsEqual(5)(6) == False

        Returns:
            a closure that takes LHS and returns operation result
        """

        def wrapper(lhs: T1, /) -> bool:
            return op(lhs, rhs)

        return wrapper

    return comparator


LessThan = binop_factory(lt)
GreaterThan = binop_factory(gt)
LessEqual = binop_factory(le)
GreaterEqual = binop_factory(ge)

IsEqual = binop_factory(eq)
NotEqual = binop_factory(ne)

Is = binop_factory(is_)
IsNot = binop_factory(is_not)


def Not[T](predicate: Callable[[T], bool]) -> Callable[[T], bool]:
    return lambda x: not predicate(x)


def IsNone(obj: object) -> bool:
    return obj is None


def IsNotNone(obj: object) -> bool:
    return obj is not None


def Contains[T](obj: T) -> Callable[[Container[T]], bool]:
    return lambda container: contains(container, obj)


def Unpacked[*Ts, R](func: Callable[[*Ts], R]) -> Callable[[tuple[*Ts]], R]:
    return lambda tup: func(*tup)
