import enum
from dataclasses import dataclass
from typing import Literal


class Default(enum.Enum):
    """Sentinel values used as defaults."""

    Exhausted = enum.auto()
    NoDefault = enum.auto()
    Unavailable = enum.auto()


# TODO: Replace with enum.global_enum if ever supported in pyright
Exhausted: Literal[Default.Exhausted] = Default.Exhausted
NoDefault: Literal[Default.NoDefault] = Default.NoDefault
Unavailable: Literal[Default.Unavailable] = Default.Unavailable


@dataclass(frozen=True, slots=True)
class Ignore: ...


@dataclass(frozen=True, slots=True)
class Raise: ...


@dataclass(frozen=True, slots=True)
class Pad[T]:
    fillvalue: T


IGNORE = Ignore()

type MissingPolicy[T] = Ignore | Raise | Pad[T]
