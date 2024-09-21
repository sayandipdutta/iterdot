from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import itertools as it
from operator import attrgetter
from collections import deque
import typing as tp
from collections.abc import Callable, Iterable, Iterator, Sized
import enum
from functools import partial, reduce, wraps
import statistics as st


class Default(enum.Enum):
    Exhausted = enum.auto()
    NoDefault = enum.auto()
    Unavailable = enum.auto()


# TODO: Replace with enum.global_enum if ever supported in pyright
Exhausted: tp.Literal[Default.Exhausted] = Default.Exhausted
NoDefault: tp.Literal[Default.NoDefault] = Default.NoDefault
Unavailable: tp.Literal[Default.Unavailable] = Default.Unavailable


class Comparable[T](tp.Protocol):
    def __lt__(self: T, other: T, /) -> bool: ...
    def __gt__(self: T, other: T, /) -> bool: ...


# TODO: a better way to register modules
def _register_stats[R, **P, TNumber: (float, Decimal, Fraction) = float](
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


class MethodKind[T]:
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


def prepend[T](*val: T, to: Iterator[T]) -> Iterator[T]:
    return it.chain(val, to)


def flatten(iterable: Iterable[object]) -> Iterable[object]:
    for item in iterable:
        if isinstance(item, Iterable):
            yield from flatten(tp.cast(Iterable[object], item))
        yield item


class ChainableIter[T](Iterable[T]):
    def __init__(self, iterable: Iterable[T]) -> None:
        self.iterable = iterable
        self._iter = (
            iter(iterable)
            if not isinstance(iterable, ChainableIter)
            else iterable._iter
        )
        self._last_yielded_value: T | tp.Literal[Default.Unavailable] = Unavailable
        self._last_yielded_index: int = -1

    def peek_next_index(self) -> int:
        return self._last_yielded_index + 1

    @tp.overload
    def peek_next_value(
        self, default: tp.Literal[Default.Exhausted] = Exhausted
    ) -> T | tp.Literal[Default.Exhausted]: ...
    @tp.overload
    def peek_next_value[TDefault](self, default: TDefault) -> T | TDefault: ...
    def peek_next_value[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        item = default
        for item in self:
            self._iter = prepend(item, to=self._iter)
            self._last_yielded_index -= 1
            break
        return item

    @property
    def last_yielded_value(self) -> T | tp.Literal[Default.Unavailable]:
        return self._last_yielded_value

    @property
    def last_yielded_index(self) -> int:
        return self._last_yielded_index

    # NOTE: consider if it should error
    @tp.overload
    def next_value[TDefault](
        self, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def next_value[TDefault](self, default: TDefault) -> T | TDefault: ...
    def next_value[TDefault](self, default: TDefault = NoDefault) -> T | TDefault:
        return next(self) if default is NoDefault else next(self, default)

    # TODO: add itemgetter
    # TODO: Replace type[K] with TypeForm (when available)
    def getattr[K](self, *names: str, type: type[K]) -> ChainableIter[K]:
        func = tp.cast(Callable[[T], K], attrgetter(*names))  # pyright: ignore[reportInvalidCast]
        return self.map_partial(func)

    compress = MethodKind[T].augmentor(it.compress)
    pairwise = MethodKind[T].augmentor(it.pairwise)
    batched = MethodKind[T].augmentor(it.batched)
    accumulate = MethodKind[T].augmentor(it.accumulate)
    slice = MethodKind[T].augmentor(it.islice)
    zip_with = MethodKind[T].augmentor(zip)

    takewhile = MethodKind[T].predicated_augmentor(it.takewhile)
    dropwhile = MethodKind[T].predicated_augmentor(it.dropwhile)

    sum = MethodKind[T].consumer(sum)
    to_list = MethodKind[T].consumer(list)

    @tp.overload
    def max[TComparable: Comparable[tp.Any], RComparable: Comparable[tp.Any], F](
        self: ChainableIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: tp.Literal[Default.NoDefault] = NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def max[TComparable: Comparable[tp.Any], RComparable: Comparable[tp.Any], F](
        self: ChainableIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F: ...
    def max[TComparable: Comparable[tp.Any], RComparable: Comparable[tp.Any], F](
        self: ChainableIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F:
        match (key, default):
            case (None, Default.NoDefault):
                return max(self)
            case (None, _):
                return max(self, default=default)
            case (_, Default.NoDefault):
                return max(self, key=key)
            case _:
                return max(self, key=key, default=default)

    # TODO: Add minmax
    @tp.overload
    def min[TComparable: Comparable[tp.Any], RComparable: Comparable[tp.Any], F](
        self: ChainableIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def min[TComparable: Comparable[tp.Any], RComparable: Comparable[tp.Any], F](
        self: ChainableIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F: ...
    def min[TComparable: Comparable[tp.Any], RComparable: Comparable[tp.Any], F](
        self: ChainableIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F:
        match (key, default):
            case (None, Default.NoDefault):
                return min(self)
            case (None, _):
                return min(self, default=default)
            case (_, Default.NoDefault):
                return min(self, key=key)
            case _:
                return min(self, key=key, default=default)

    @tp.override
    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        self._last_yielded_value = next(self._iter)
        if self._last_yielded_index is Default.Unavailable:
            self._last_yielded_index = 0
        else:
            self._last_yielded_index += 1
        return self._last_yielded_value

    def map_partial[R, **P](
        self,
        func: Callable[tp.Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ChainableIter[R]:
        f = partial(func, *args, **kwargs)
        itr = iter(map(f, self))
        return ChainableIter(itr)

    @tp.overload
    def map[K, R](
        self,
        func: Callable[[T], R],
    ) -> ChainableIter[R]: ...
    @tp.overload
    def map[K, _, R, **P](
        self,
        func: Callable[tp.Concatenate[T, _, ...], R],
        *args: Iterable[K],
    ) -> ChainableIter[R]: ...
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
        self, predicate: Callable[[T], bool] | None, *, when: bool = True
    ) -> ChainableIter[T]:
        if when:
            return ChainableIter(filter(predicate, self._iter))
        else:
            return ChainableIter(it.filterfalse(predicate, self._iter))

    def starmap[*Ts, R](
        self: ChainableIter[tuple[*Ts]], func: Callable[[*Ts], R]
    ) -> ChainableIter[R]:
        return ChainableIter(it.starmap(func, self._iter))

    @tp.overload
    def first[TDefault](
        self, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def first[TDefault](self, default: TDefault) -> T | TDefault: ...
    def first[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        return next(self) if default is NoDefault else next(self, default)

    # TODO: support getitem
    @tp.overload
    def at[TDefault](
        self, n: int, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def at[TDefault](self, n: int, default: TDefault) -> T | TDefault: ...
    def at[TDefault](self, n: int, default: TDefault = Exhausted) -> T | TDefault:
        return self.skip(n).next_value(default)

    @tp.overload
    def last[TDefault](
        self, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def last[TDefault](self, default: TDefault) -> T | TDefault: ...
    def last[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        try:
            return deque(self, maxlen=1).popleft()
        except IndexError:
            if default is NoDefault:
                raise StopIteration("Underlying iterable is empty")
            return default

    def tail(self, n: int) -> ChainableIter[T]:
        return ChainableIter(deque(self, maxlen=n))

    def skip(self, n: int) -> ChainableIter[T]:
        return self if not n else self.slice(n, None)

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

    # TODO: explore if take_first can be emulated with
    # order of arguments, i.e. by looking at the order
    # of **kwargs, where **kwargs: TypedDict[skip: int, take: int]
    # also check if signature can be overloaded
    # if not, explore decorator approach to patch `take_first`
    # argument based on order of kw args.
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

    # TODO: consider default
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
        fill_value: tp.Literal[Default.NoDefault],
        strict: tp.Literal[False],
    ) -> ChainableIter[tuple[T, T1]]: ...

    # NOTE: redundant??
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
    ) -> ChainableIter[tuple[T | F, T1 | F]]: ...

    def zip2[T1, F](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: F = Default.NoDefault,
        strict: bool = False,
    ) -> ChainableIter[tuple[T, T1]] | ChainableIter[tuple[T | F, T1 | F]]:
        itbl = it.zip_longest(self, iter1, fillvalue=fill_value)
        if fill_value is Default.NoDefault:
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
        fill_value: tp.Literal[Default.NoDefault],
    ) -> ChainableIter[tuple[T, T1, T2]]: ...

    @tp.overload
    def zip3[T1, T2, F](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: F,
    ) -> ChainableIter[tuple[T | F, T1 | F, T2 | F]]: ...

    # NOTE: Redundant??
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
        fill_value: F = Default.NoDefault,
        strict: bool = False,
    ) -> ChainableIter[tuple[T, T1, T2]] | ChainableIter[tuple[T | F, T1 | F, T2 | F]]:
        itbl = it.zip_longest(self, iter1, iter2, fillvalue=fill_value)
        if fill_value is Default.NoDefault:
            itbl = zip(self, iter1, iter2, strict=strict)
        elif strict:
            raise ValueError("Cannot specify both fill_value and strict")
        return ChainableIter(itbl)

    # TODO: overload zip4
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

    def transpose_eager[TSized: Sized](
        self: ChainableIter[TSized],
        strict: bool = False,
    ) -> ChainableIter[TSized]:
        return ChainableIter(zip(*self._iter, strict=strict))

    def interleave[R](
        self,
        other: Iterable[R],
        other_first: bool = False,
    ) -> ChainableIter[T | R]:
        ch = it.chain.from_iterable(
            zip(other, self._iter) if other_first else zip(self._iter, other)
        )
        return ChainableIter(ch)

    @property
    def stats[TNumber: (float, Decimal, Fraction) = float](
        self: ChainableIter[TNumber],
    ) -> stats[TNumber]:
        return stats[TNumber](self)

    def map_type[R](self, type: type[R]) -> ChainableIter[R]:
        return self.map(type)

    def prepend[V](self, *values: V) -> ChainableIter[T | V]:
        return ChainableIter(it.chain(values, self))

    def append[V](self, *values: V) -> ChainableIter[T | V]:
        return ChainableIter(it.chain(self, values))

    def flatten_once(self: ChainableIter[Iterable[T]]) -> ChainableIter[T]:
        return ChainableIter(it.chain.from_iterable(self))

    def flatten(self) -> ChainableIter[object]:
        return ChainableIter(flatten(self))

    def concat[R](
        self, *its: Iterable[R], self_position: tp.Literal["front", "back"] = "front"
    ) -> ChainableIter[T | R]:
        match self_position:
            case "front":
                return ChainableIter(it.chain(self._iter, *its))
            case "back":
                return ChainableIter(it.chain(*its, self._iter))


if __name__ == "__main__":
    ch = ChainableIter(range(1, 10)).skip_take(skip=2, take=3, take_first=True)
    ch.exhaust()
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
        .transpose_eager()
        .map_partial(sum, start=0.0)
        .min()
    )
    print(ch1)
