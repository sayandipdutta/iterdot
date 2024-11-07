import enum
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
