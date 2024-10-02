from __future__ import annotations

from collections.abc import Callable, Iterable
from decimal import Decimal
from fractions import Fraction
from functools import wraps
import statistics as st
import typing as tp

if tp.TYPE_CHECKING:
    from iterdot.chain import Iter


# TODO: a better way to register modules
def register_stats[R, **P, TNumber: (float, Decimal, Fraction) = float](
    func: Callable[tp.Concatenate[Iterable[TNumber], P], R],
) -> Callable[tp.Concatenate[stats[TNumber], P], R]:
    @wraps(func)
    def inner(
        self: stats[TNumber],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(iter(self.iterable), *args, **kwargs)

    return inner


class stats[TNumber: (float, Decimal, Fraction) = float]:
    def __init__(self, iterable: Iter[TNumber]):
        self.iterable = iterable

    fmean = register_stats(st.fmean)
    geometric_mean = register_stats(st.geometric_mean)
    harmonic_mean = register_stats(st.harmonic_mean)
    pstdev = register_stats(st.pstdev)
    pvariance = register_stats(st.pvariance)
    stdev = register_stats(st.stdev)
    variance = register_stats(st.variance)
    median = register_stats(st.median)
    median_low = register_stats(st.median_low)
    median_high = register_stats(st.median_high)
    median_grouped = register_stats(st.median_grouped)
    mode = register_stats(st.mode)
    multimode = register_stats(st.multimode)
    quantiles = register_stats(st.quantiles)
