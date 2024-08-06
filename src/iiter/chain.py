from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import enum
from functools import reduce
from itertools import (
    accumulate,
    batched,
    chain,
    compress,
    dropwhile,
    filterfalse,
    pairwise,
    starmap,
    takewhile,
    zip_longest,
)
from typing import (
    Any,
    Concatenate,
    Literal,
    NoReturn,
    Self,
    SupportsIndex,
    cast,
    overload,
)


FillValue = enum.Enum("FillValue", ["MISSING"])


class ChainableIter[T](Iterable[T]):
    def __init__(self, iterable: Iterable[T]) -> None:
        self.iterable = iterable

    def map[R, **P](
        self,
        func: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ChainableIter[R]:
        return ChainableIter(map(func, self.iterable, *args, **kwargs))

    def feed_into[R, **P](
        self,
        func: Callable[Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(self.iterable, *args, **kwargs)

    def accumulate(
        self, func: Callable[[T, T], T], *, initial: T | None = None
    ) -> ChainableIter[T]:
        return ChainableIter(accumulate(self.iterable, func, initial=initial))

    def batched(self, n: int = 2) -> ChainableIter[tuple[T, ...]]:
        return ChainableIter(batched(self.iterable, n))

    def compress(self, selectors: Iterable[SupportsIndex]) -> ChainableIter[T]:
        return ChainableIter(compress(self.iterable, selectors))

    def dropwhile(self, predicate: Callable[[T], bool]) -> ChainableIter[T]:
        return ChainableIter(dropwhile(predicate, self.iterable))

    def takewhile(self, predicate: Callable[[T], bool]) -> ChainableIter[T]:
        return ChainableIter(takewhile(predicate, self.iterable))

    def filter(
        self, predicate: Callable[[T], bool] | None, when: bool = True
    ) -> ChainableIter[T]:
        filter_func = filter[T] if when else filterfalse[T]
        return ChainableIter(filter_func(predicate, self.iterable))

    def pairwise(self) -> ChainableIter[tuple[T, T]]:
        return ChainableIter(pairwise(self.iterable))

    def starmap[*Ts, R](
        self: ChainableIter[tuple[*Ts]], func: Callable[[*Ts], R]
    ) -> ChainableIter[R]:
        return ChainableIter(starmap(func, self.iterable))

    def tee(self) -> ChainableIter[T]: ...

    def first(self) -> T:
        return next(iter(self.iterable))

    def nth_or_last(self, n: int) -> T: ...

    def slice(self, n: int) -> ChainableIter[T]: ...

    def last(self) -> T: ...

    def take(self) -> ChainableIter[T]: ...

    def skip(self) -> ChainableIter[T]: ...

    def skip_take(self, skip: int, take: int, take_first: bool) -> ChainableIter[T]: ...

    @overload
    def chain_from_iter[K](
        self: ChainableIter[Iterable[K]], iterable: None = None
    ) -> ChainableIter[K]: ...

    @overload
    def chain_from_iter[K](
        self, iterable: Iterable[Iterable[K]]
    ) -> ChainableIter[T | K]: ...

    def chain_from_iter[K1, K2](
        self: Self | ChainableIter[Iterable[K1]],
        iterable: Iterable[Iterable[K2]] | None = None,
    ) -> ChainableIter[K1] | ChainableIter[T | K2]:
        if iterable is None:
            return ChainableIter(
                chain.from_iterable(cast(ChainableIter[Iterable[K1]], self))
            )
        return ChainableIter(
            chain(cast(ChainableIter[T], self), chain.from_iterable(iterable))
        )

    def chain[K](self, *iterable: Iterable[K]) -> ChainableIter[T | K]:
        return ChainableIter(chain(self, *iterable))

    def reduce(self, func: Callable[[T, T], T], initial: T | None = None) -> T:
        if initial is None:
            return reduce(func, list(self))
        return reduce(func, list(self), initial)

    def transpose(self) -> ChainableIter[tuple[T]]:
        return ChainableIter(zip(self))

    @overload
    def zip2[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        strict: bool,
    ) -> ChainableIter[tuple[T, T1]]: ...

    @overload
    def zip2[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: Literal[FillValue.MISSING],
        strict: Literal[False],
    ) -> ChainableIter[tuple[T, T1]]: ...

    @overload
    def zip2[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: object,
        strict: Literal[True],
    ) -> NoReturn: ...

    @overload
    def zip2[T1, F](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: F,
        strict: Literal[False],
    ) -> ChainableIter[tuple[T | F, T1 | F]]: ...

    def zip2[T1, F](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: F = FillValue.MISSING,
        strict: bool = False,
    ) -> ChainableIter[tuple[T, T1]] | ChainableIter[tuple[T | F, T1 | F]]:
        it = zip_longest(self, iter1, fillvalue=fill_value)
        if fill_value is FillValue.MISSING:
            it = zip(self, iter1, strict=strict)
        elif strict:
            raise ValueError("Cannot specify both fill_value and strict")
        return ChainableIter(it)

    @overload
    def zip3[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        strict: bool,
    ) -> ChainableIter[tuple[T, T1, T2]]: ...

    @overload
    def zip3[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: Literal[FillValue.MISSING],
        strict: Literal[False],
    ) -> ChainableIter[tuple[T, T1, T2]]: ...

    @overload
    def zip3[T1, T2, F](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: F,
        strict: Literal[False],
    ) -> ChainableIter[tuple[T | F, T1 | F, T2 | F]]: ...

    @overload
    def zip3[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: object,
        strict: Literal[True],
    ) -> NoReturn: ...

    def zip3[T1, T2, F](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: F = FillValue.MISSING,
        strict: bool = False,
    ) -> ChainableIter[tuple[T, T1, T2]] | ChainableIter[tuple[T | F, T1 | F, T2 | F]]:
        it = zip_longest(self, iter1, iter2, fillvalue=fill_value)
        if fill_value is FillValue.MISSING:
            it = zip(self, iter1, iter2, strict=strict)
        elif strict:
            raise ValueError("Cannot specify both fill_value and strict")
        return ChainableIter(it)

    def zip4[T1, T2, T3](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
        *,
        strict: bool = False,
    ) -> ChainableIter[tuple[T, T1, T2, T3]]:
        return ChainableIter(zip(self, iter1, iter2, iter3, strict=strict))

    def zipn(
        self,
        *iters: Iterable[Any],
        strict: bool = False,
    ) -> ChainableIter[tuple[Any, ...]]:
        return ChainableIter(zip(self, *iters, strict=strict))
