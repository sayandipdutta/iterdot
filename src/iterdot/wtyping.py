import typing as tp


@tp.runtime_checkable
class SupportsLT(tp.Protocol):
    def __lt__(self, other: tp.Any, /) -> bool: ...  # pyright: ignore[reportAny]  # noqa: ANN401


@tp.runtime_checkable
class SupportsGT(tp.Protocol):
    def __gt__(self, other: tp.Any, /) -> bool: ...  # pyright: ignore[reportAny]  # noqa: ANN401


type Comparable = SupportsLT | SupportsGT
