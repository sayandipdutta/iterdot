from __future__ import annotations

import typing as tp

from iterdot.wtyping import SupportsGT, SupportsLT


class Indexed[T](tp.NamedTuple):
    idx: int
    value: T

    @tp.override
    def __le__[TSupportsLT: SupportsLT](
        self: Indexed[TSupportsLT], value: TSupportsLT, /
    ) -> bool:
        return (self.value < value) or (self.value == value)

    @tp.override
    def __lt__[TSupportsLT: SupportsLT](
        self: Indexed[TSupportsLT], value: TSupportsLT, /
    ) -> bool:
        return self.value < value

    @tp.override
    def __gt__[TSupportsGT: SupportsGT](
        self: Indexed[TSupportsGT], value: TSupportsGT, /
    ) -> bool:
        return self.value > value

    @tp.override
    def __ge__[TSupportsGT: SupportsGT](
        self: Indexed[TSupportsGT], value: TSupportsGT, /
    ) -> bool:
        return (self.value > value) or (self.value == value)

    @tp.override
    def __eq__(self, value: object, /) -> bool:
        return self.value == value
