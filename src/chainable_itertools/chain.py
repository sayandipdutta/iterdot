from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import itertools as it
from operator import attrgetter
from collections import deque
import typing as tp
from collections.abc import Callable, Iterable, Iterator
import enum
from functools import partial, reduce, wraps
import statistics as st

ValueType = enum.Enum("ValueType", ["NA"])
TNumber = tp.TypeVar("TNumber", float, Decimal, Fraction)


class SupportsLT[T](tp.Protocol):
    def __lt__(self, other: T) -> bool: ...


class SupportsGT[T](tp.Protocol):
    def __gt__(self, other: T) -> bool: ...


type Comparable[T] = SupportsLT[T] | SupportsGT[T]


def _register_stats[TNumber, R, **P](
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


class stats[TNumber]:
    def __init__(self, iterable: ChainableIter[TNumber]):
        self.iterable = iterable

    fmean = _register_stats(st.fmean)
    geometric_mean = _register_stats(st.geometric_mean)
    harmonic_mean = _register_stats(st.harmonic_mean)
    pstdev = _register_stats(st.pstdev)
    pvariance = _register_stats(st.pvariance)
    stdev = _register_stats(st.stdev)
    variance = _register_stats(st.variance)
    median = _register_stats(st.median)
    median_low = _register_stats(st.median_low)
    median_high = _register_stats(st.median_high)
    median_grouped = _register_stats(st.median_grouped)
    mode = _register_stats(st.mode)
    multimode = _register_stats(st.multimode)
    quantiles = _register_stats(st.quantiles)


class MethodType[T]:
    @staticmethod
    def consumer[**P, R](
        func: Callable[tp.Concatenate[Iterable[T], P], R],
    ) -> Callable[tp.Concatenate[ChainableIter[T], P], R]:
        @wraps(func)
        def inner(self: ChainableIter[T], *args: P.args, **kwargs: P.kwargs) -> R:
            return func(self, *args, **kwargs)

        return inner

    @staticmethod
    def augmentor[**P, R](
        func: Callable[tp.Concatenate[Iterable[T], P], Iterable[R]],
    ) -> Callable[tp.Concatenate[ChainableIter[T], P], ChainableIter[R]]:
        @wraps(func)
        def inner(
            self: ChainableIter[T], *args: P.args, **kwargs: P.kwargs
        ) -> ChainableIter[R]:
            return ChainableIter(func(self, *args, **kwargs))

        return inner

    @staticmethod
    def predicated_augmentor[**P, R](
        func: Callable[
            tp.Concatenate[Callable[[T], bool], Iterable[T], P], Iterable[R]
        ],
    ) -> Callable[
        tp.Concatenate[ChainableIter[T], Callable[[T], bool], P], ChainableIter[R]
    ]:
        @wraps(func)
        def inner(
            self: ChainableIter[T],
            predicate: Callable[[T], bool],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> ChainableIter[R]:
            return ChainableIter(func(predicate, self, *args, **kwargs))

        return inner


def prepend[T](iterable: Iterator[T], *val: T) -> Iterator[T]:
    yield from val
    yield from iterable


class ChainableIter[T](Iterable[T]):
    def __init__(self, iterable: Iterable[T], copy_state: bool = False) -> None:
        self.iterable = iterable
        self._iter = iter(iterable)
        self.last_yielded_value: T | tp.Literal[ValueType.NA] = ValueType.NA
        self.last_yielded_index: int = -1
        if copy_state:
            if isinstance(self.iterable, ChainableIter):
                self._counter = self.iterable.indices()
                self.last_yielded_value = self.iterable.last_yielded_value
                self.last_yielded_index = self.iterable.last_yielded_index
            else:
                raise ValueError(
                    "`iterable` is not an instance of ChainableIter, cannot copy indices"
                )
        else:
            self._counter: Iterator[int] = it.count()

    def indices(self) -> Iterator[int]:
        return self._counter

    def peek_next_index(self) -> int:
        if self.last_yielded_index is ValueType.NA:
            return 0
        if self.peek_next_value() is ValueType.NA:
            return -1
        return self.last_yielded_index + 1

    def reset_index(self, index: int = 0):
        self._counter = it.count(index)

    def peek_next_value[TDefault](
        self, default: TDefault = ValueType.NA
    ) -> T | TDefault:
        item = default
        for item in self:
            self._iter = prepend(self, item)
            break
        return item

    def next_value[TDefault](self, default: TDefault = ValueType.NA) -> T | TDefault:
        return next(self) if default is ValueType.NA else next(self, default)

    def get_attribute[K](self, name: str, type: type[K]) -> ChainableIter[K]:
        func = tp.cast(Callable[[T], K], attrgetter(name))  # pyright: ignore[reportInvalidCast]
        return self.partial_map(func)

    compress = MethodType[T].augmentor(it.compress)
    pairwise = MethodType[T].augmentor(it.pairwise)
    batched = MethodType[T].augmentor(it.batched)
    accumulate = MethodType[T].augmentor(it.accumulate)
    chain = MethodType[T].augmentor(it.chain)
    slice = MethodType[T].augmentor(it.islice)
    zipn = MethodType[T].augmentor(zip)

    takewhile = MethodType[T].predicated_augmentor(it.takewhile)
    dropwhile = MethodType[T].predicated_augmentor(it.dropwhile)

    sum = MethodType[T].consumer(sum)
    min = MethodType[Comparable[T]].consumer(min)
    max = MethodType[Comparable[T]].consumer(max)
    to_list = MethodType[T].consumer(list)

    @tp.override
    def __iter__(self) -> Iterator[T]:
        return iter(self.__next__, ValueType.NA)

    def __next__(self) -> T:
        self.last_yielded_value = next(self._iter)
        if self.last_yielded_index is ValueType.NA:
            self.last_yielded_index = 0
        else:
            self.last_yielded_index += 1
        return self.last_yielded_value

    def partial_map[R, **P](
        self,
        func: Callable[tp.Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ChainableIter[R]:
        # breakpoint()
        f = partial(func, *args, **kwargs)
        itr = iter(map(f, self))
        return ChainableIter(itr)

    def map[K, R](
        self,
        func: Callable[tp.Concatenate[T, ...], R],
        *args: Iterable[K],
    ) -> ChainableIter[R]:
        return ChainableIter(map(func, self, *args))

    def feed_into[R, **P](
        self,
        func: Callable[tp.Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(self, *args, **kwargs)

    def filter(
        self, predicate: Callable[[T], bool] | None, when: bool = True
    ) -> ChainableIter[T]:
        filter_func = filter[T] if when else it.filterfalse[T]
        return ChainableIter(filter_func(predicate, self))

    def starmap[*Ts, R](
        self: ChainableIter[tuple[*Ts]], func: Callable[[*Ts], R]
    ) -> ChainableIter[R]:
        return ChainableIter(it.starmap(func, self))

    def first(self) -> T:
        return next(iter(self))

    def nth_or_last(self, n: int) -> T:
        return self.slice(n).last()

    def last(self) -> T:
        return deque(self, maxlen=1).popleft()

    def last_or_default[TDefault](
        self, default: TDefault = ValueType.NA
    ) -> T | TDefault:
        try:
            return deque(self, maxlen=1).popleft()
        except ValueError:
            return default

    def tail(self, n: int) -> ChainableIter[T]:
        return ChainableIter(deque(self, maxlen=n))

    def take(self, n: int) -> ChainableIter[T]:
        return self.slice(n)

    def skip(self, n: int) -> ChainableIter[T]:
        return self.slice(n + 1, None)

    def exhaust(self) -> None:
        out = deque(self, maxlen=0)
        del out

    def foreach[R](self, func: Callable[[T], None]) -> None:
        self.map(func).exhaust()

    @staticmethod
    def _get_skip_take_selectors(
        s1: tuple[bool, int], s2: tuple[bool, int]
    ) -> Iterator[bool]:
        while True:
            yield from it.repeat(*s1)
            yield from it.repeat(*s2)

    def skip_take(
        self, *, skip: int, take: int, take_first: bool = False
    ) -> ChainableIter[T]:
        if take_first:
            selectors = self._get_skip_take_selectors((True, take), (False, skip))
        else:
            selectors = self._get_skip_take_selectors((False, skip), (True, take))
        return self.compress(selectors)

    @tp.overload
    def chain_from_iter[K1](
        self: ChainableIter[Iterable[K1]], iterable: None = None
    ) -> ChainableIter[K1]: ...

    @tp.overload
    def chain_from_iter[K2](
        self, iterable: Iterable[Iterable[K2]]
    ) -> ChainableIter[T | K2]: ...

    def chain_from_iter[K1, K2](
        self: tp.Self | ChainableIter[Iterable[K1]],
        iterable: Iterable[Iterable[K2]] | None = None,
    ) -> ChainableIter[K1] | ChainableIter[T | K2]:
        if iterable is None:
            return ChainableIter(
                it.chain.from_iterable(tp.cast(ChainableIter[Iterable[K1]], self))
            )
        return ChainableIter(
            it.chain(tp.cast(ChainableIter[T], self), it.chain.from_iterable(iterable))
        )

    def reduce(self, func: Callable[[T, T], T], initial: T | None = None) -> T:
        if initial is None:
            return reduce(func, list(self))
        return reduce(func, list(self), initial)

    @tp.overload
    def zip2[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        strict: bool,
    ) -> ChainableIter[tuple[T, T1]]: ...

    @tp.overload
    def zip2[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: tp.Literal[ValueType.NA],
        strict: tp.Literal[False],
    ) -> ChainableIter[tuple[T, T1]]: ...

    @tp.overload
    def zip2[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: object,
        strict: tp.Literal[True],
    ) -> tp.Never: ...

    @tp.overload
    def zip2[T1, F](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: F,
        strict: tp.Literal[False],
    ) -> ChainableIter[tuple[T | F, T1 | F]]: ...

    def zip2[T1, F](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: F = ValueType.NA,
        strict: bool = False,
    ) -> ChainableIter[tuple[T, T1]] | ChainableIter[tuple[T | F, T1 | F]]:
        itbl = it.zip_longest(self, iter1, fillvalue=fill_value)
        if fill_value is ValueType.NA:
            itbl = zip(self, iter1, strict=strict)
        elif strict:
            raise ValueError("Cannot specify both fill_value and strict")
        return ChainableIter(itbl)

    @tp.overload
    def zip3[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        strict: bool,
    ) -> ChainableIter[tuple[T, T1, T2]]: ...

    @tp.overload
    def zip3[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: tp.Literal[ValueType.NA],
        strict: tp.Literal[False],
    ) -> ChainableIter[tuple[T, T1, T2]]: ...

    @tp.overload
    def zip3[T1, T2, F](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: F,
        strict: tp.Literal[False],
    ) -> ChainableIter[tuple[T | F, T1 | F, T2 | F]]: ...

    @tp.overload
    def zip3[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: object,
        strict: tp.Literal[True],
    ) -> tp.Never: ...

    def zip3[T1, T2, F](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: F = ValueType.NA,
        strict: bool = False,
    ) -> ChainableIter[tuple[T, T1, T2]] | ChainableIter[tuple[T | F, T1 | F, T2 | F]]:
        itbl = it.zip_longest(self, iter1, iter2, fillvalue=fill_value)
        if fill_value is ValueType.NA:
            itbl = zip(self, iter1, iter2, strict=strict)
        elif strict:
            raise ValueError("Cannot specify both fill_value and strict")
        return ChainableIter(itbl)

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

    def transpose(
        self: ChainableIter[T],
        strict: bool = False,
    ) -> ChainableIter[T]:
        return ChainableIter(zip(*self))

    @property
    def stats(self: ChainableIter[TNumber]) -> stats[TNumber]:
        return stats[TNumber](self)

    def map_type[R](self, type: type[R]) -> ChainableIter[R]:
        return self.map(type)


if __name__ == "__main__":
    ch = ChainableIter(range(1, 10)).skip_take(skip=2, take=3, take_first=True)
    ch.consume()
    print(ch.last_yielded_value)
    print(ch.last_yielded_index)
    print(ch.peek_next_index())
    print(ch.peek_next_value())
    ch1 = (
        ChainableIter(range(1, 100))
        .skip_take(skip=2, take=3, take_first=True)
        .map_type(float)
        .skip_take(skip=5, take=5)
        .batched(5)
        .transpose()
        .partial_map(sum, start=0.0)
        .to_list()
    )
    print(ch1)
