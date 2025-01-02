import typing as tp
from collections.abc import Callable


@tp.runtime_checkable
class SupportsLT(tp.Protocol):
    def __lt__(self, other: tp.Any, /) -> bool: ...  # pyright: ignore[reportAny]  # noqa: ANN401


@tp.runtime_checkable
class SupportsGT(tp.Protocol):
    def __gt__(self, other: tp.Any, /) -> bool: ...  # pyright: ignore[reportAny]  # noqa: ANN401


type Comparable = SupportsLT | SupportsGT
type Predicate[T] = Callable[[T], bool]


class SupportsAdd[T1, T2](tp.Protocol):
    def __add__(self, other: T1) -> T2: ...


class SupportsRAdd[T1, T2](tp.Protocol):
    def __radd__(self, other: T1) -> T2: ...


class SupportsSumNoDefault(
    SupportsAdd[tp.Any, tp.Any], SupportsRAdd[tp.Any, tp.Any], tp.Protocol
): ...


class SupportsSub(tp.Protocol):
    def __sub__(self, other: tp.Self) -> tp.Self: ...


type Falsy = tp.Literal[False, 0, None, ""] | tuple[()]
