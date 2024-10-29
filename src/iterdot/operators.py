from collections.abc import Callable, Container
from operator import contains, eq, ge, gt, is_, is_not, le, lt, ne


def comparator_factory[T1, T2](
    op: Callable[[T1, T2], bool],
) -> Callable[[T2], Callable[[T1], bool]]:
    def comparator(rhs: T2) -> Callable[[T1], bool]:
        def wrapper(lhs: T1) -> bool:
            return op(lhs, rhs)

        return wrapper

    return comparator


LessThan = comparator_factory(lt)
GreaterThan = comparator_factory(gt)
LessEqual = comparator_factory(le)
GreaterEqual = comparator_factory(ge)

IsEqual = comparator_factory(eq)
NotEqual = comparator_factory(ne)

Is = comparator_factory(is_)
IsNot = comparator_factory(is_not)


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
