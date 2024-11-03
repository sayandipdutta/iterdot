from collections.abc import Callable, Iterator
from functools import cached_property
from operator import gt, lt
from typing import NamedTuple, Protocol, Self

from iterdot.wtyping import Comparable


class SupportsSub(Protocol):
    def __sub__(self, other: Self) -> Self: ...


class MinMax[T](NamedTuple):
    min: T
    max: T

    @cached_property
    def ptp[TSupportsSub: SupportsSub](self: "MinMax[TSupportsSub]") -> TSupportsSub:
        return self.max - self.min


def lazy_minmax_keyed[T](
    iterator: Iterator[T], key: Callable[[T], Comparable]
) -> MinMax[T]:
    min = max = next(iterator)
    for item in iterator:
        k = key(item)
        if lt(k, key(min)):
            min = item
        if gt(k, key(max)):
            max = item
    return MinMax(min, max)


def lazy_minmax[T: Comparable](iterator: Iterator[T]) -> MinMax[T]:
    min = max = next(iterator)
    for item in iterator:
        if lt(item, min):
            min = item
        if gt(item, max):
            max = item
    return MinMax(min, max)
