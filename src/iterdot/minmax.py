from collections.abc import Callable, Iterator
from operator import lt, gt
from typing import NamedTuple

from iterdot.typing import Comparable


def lazy_minmax_keyed[T, R: Comparable](
    iterator: Iterator[T], key: Callable[[T], R]
) -> tuple[T, T]:
    min = max = next(iterator)
    for item in iterator:
        k = key(item)
        if lt(k, key(min)):
            min = item
        if gt(k, key(max)):
            max = item
    return min, max


def lazy_minmax[T: Comparable](iterator: Iterator[T]) -> tuple[T, T]:
    min = max = next(iterator)
    for item in iterator:
        if lt(item, min):
            min = item
        if gt(item, max):
            max = item
    return min, max


class MinMax[T](NamedTuple):
    min: T
    max: T
