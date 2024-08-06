from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from operator import attrgetter
from typing import override


class Step(int, Enum):
    FORWARD = 1
    BACKWARD = -1


@dataclass(frozen=True, slots=True)
class Indexed[T]:
    index: int
    value: T


@dataclass(eq=False, repr=False, match_args=False, slots=True)
class IndexedIter[T](Iterable[T]):
    iterable: Iterable[T] | IndexedIter[T]
    current_index: int = field(default=-1, init=False)

    @override
    def __iter__(self) -> Iterator[Indexed[T]]:
        if isinstance(self.iterable, IndexedIter):
            for item in self.iterable:
                self.current_index = item.index
                yield item
        else:
            for i, item in enumerate(self.iterable):
                self.current_index = i
                yield Indexed(i, item)

    def into_iter(self) -> IndexedIter[T]:
        if isinstance(self.iterable, IndexedIter):
            return self
        return IndexedIter(self)

    def values(self) -> Iterator[T]:
        return map(attrgetter("value"), self)

    def indices(self) -> Iterator[int]:
        return map(attrgetter("index"), self)
