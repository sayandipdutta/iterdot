import typing as tp


@tp.runtime_checkable
class SupportsLT(tp.Protocol):
    def __lt__(self, other: tp.Any, /) -> bool: ...  # pyright: ignore[reportAny]


@tp.runtime_checkable
class SupportsGT(tp.Protocol):
    def __gt__(self, other: tp.Any, /) -> bool: ...  # pyright: ignore[reportAny]


type Comparable = SupportsLT | SupportsGT
