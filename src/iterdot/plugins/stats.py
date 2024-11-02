from __future__ import annotations

import statistics as st
import typing as tp
from collections.abc import Callable, Iterable
from decimal import Decimal
from fractions import Fraction
from functools import wraps


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


class Stats[TNumber: (float, Decimal, Fraction) = float](tp.NamedTuple):
    mean: TNumber
    stdev: TNumber
    median: TNumber
    minimum: TNumber
    maximum: TNumber
    mode: TNumber
    quantiles: tuple[TNumber, TNumber, TNumber]


@tp.final
class stats[TNumber: (float, Decimal, Fraction) = float]:
    def __init__(self, iterable: Iterable[TNumber]) -> None:
        self.iterable = iterable

    mean = register_stats(st.mean)
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

    def __call__(self) -> Stats[TNumber]:
        lst = list(self.iterable)
        return Stats(
            mean=st.mean(lst),
            stdev=st.stdev(lst),
            median=st.median(lst),
            mode=st.mode(lst),
            minimum=min(lst),
            maximum=max(lst),
            quantiles=tuple[TNumber, TNumber, TNumber](
                st.quantiles(tp.cast(Iterable[TNumber], lst), n=4, method="inclusive")
            ),
        )
