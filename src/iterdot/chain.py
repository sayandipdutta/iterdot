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

from iterdot.extensions import stats
from iterdot.index import Indexed
from iterdot.minmax import lazy_minmax_keyed, lazy_minmax, MinMax
from iterdot.typing import Comparable


class Default(enum.Enum):
    Exhausted = enum.auto()
    NoDefault = enum.auto()
    Unavailable = enum.auto()


# TODO: Replace with enum.global_enum if ever supported in pyright
Exhausted: tp.Literal[Default.Exhausted] = Default.Exhausted
NoDefault: tp.Literal[Default.NoDefault] = Default.NoDefault
Unavailable: tp.Literal[Default.Unavailable] = Default.Unavailable


class MethodKind[T]:
    @staticmethod
    def consumer[**P, R](
        func: Callable[tp.Concatenate[Iterable[T], P], R],
    ) -> Callable[tp.Concatenate[Iter[T], P], R]:
        @wraps(func)
        def inner(self: Iter[T], *args: P.args, **kwargs: P.kwargs) -> R:
            return func(self, *args, **kwargs)

        return inner

    @staticmethod
    def augmentor[**P, R](
        func: Callable[tp.Concatenate[Iterable[T], P], Iterable[R]],
    ) -> Callable[tp.Concatenate[Iter[T], P], Iter[R]]:
        @wraps(func)
        def inner(self: Iter[T], *args: P.args, **kwargs: P.kwargs) -> Iter[R]:
            return Iter(func(self, *args, **kwargs))

        return inner

    @staticmethod
    def predicated_augmentor[**P, R](
        func: Callable[
            tp.Concatenate[Callable[[T], bool], Iterable[T], P], Iterable[R]
        ],
    ) -> Callable[tp.Concatenate[Iter[T], Callable[[T], bool], P], Iter[R]]:
        @wraps(func)
        def inner(
            self: Iter[T],
            predicate: Callable[[T], bool],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Iter[R]:
            return Iter(func(predicate, self, *args, **kwargs))

        return inner


def prepend[T](*val: T, to: Iterator[T]) -> Iterator[T]:
    return it.chain(val, to)


def flatten(iterable: Iterable[object]) -> Iterable[object]:
    for item in iterable:
        if isinstance(item, Iterable):
            yield from flatten(tp.cast(Iterable[object], item))
        yield item


class Iter[T](Iterable[T]):
    def __init__(self, iterable: Iterable[T]) -> None:
        self.iterable = iterable
        self._iter = (
            iter(iterable) if not isinstance(iterable, Iter) else iterable._iter
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
    def getattr[K](self, *names: str, type: type[K]) -> Iter[K]:
        del type
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
    def max[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: tp.Literal[Default.NoDefault] = NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def max[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F: ...
    def max[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
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

    @tp.overload
    def min[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def min[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F: ...
    def min[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
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

    @tp.overload
    def minmax_eager[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> MinMax[TComparable]: ...
    @tp.overload
    def minmax_eager[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> MinMax[TComparable] | MinMax[F]: ...
    def minmax_eager[TComparable: Comparable, RComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> MinMax[TComparable] | MinMax[F]:
        lst = list(self)
        match (key, default):
            case (None, Default.NoDefault):
                return MinMax(min(lst), max(lst))
            case (None, _):
                m1, m2 = (
                    min(lst, default=default),
                    max(lst, default=default),
                )
                if not lst:
                    return MinMax(tp.cast(F, m1), tp.cast(F, m2))
                else:
                    return MinMax(tp.cast(TComparable, m1), tp.cast(TComparable, m2))
            case (_, Default.NoDefault):
                return MinMax(min(lst, key=key), max(lst, key=key))
            case _:
                m1, m2 = (
                    min(lst, key=key, default=default),
                    max(lst, key=key, default=default),
                )
                if not lst:
                    return MinMax(tp.cast(F, m1), tp.cast(F, m2))
                else:
                    return MinMax(tp.cast(TComparable, m1), tp.cast(TComparable, m2))

    @tp.overload
    def minmax_lazy[RComparable: Comparable](
        self,
        /,
        *,
        key: Callable[[T], RComparable],
    ) -> MinMax[T]: ...
    @tp.overload
    def minmax_lazy[TComparable: Comparable](
        self: Iter[TComparable],
        /,
        *,
        key: None = None,
    ) -> MinMax[TComparable]: ...
    @tp.overload
    def minmax_lazy[RComparable: Comparable, F](
        self,
        /,
        *,
        key: Callable[[T], RComparable],
        default: F,
    ) -> MinMax[T] | MinMax[F]: ...
    @tp.overload
    def minmax_lazy[TComparable: Comparable, F](
        self: Iter[TComparable],
        /,
        *,
        key: None = None,
        default: F,
    ) -> MinMax[TComparable] | MinMax[F]: ...
    @tp.no_type_check
    def minmax_lazy(self, /, *, default=Default.NoDefault, key=None):
        try:
            min, max = (
                lazy_minmax(self) if key is None else lazy_minmax_keyed(self, key=key)
            )
            return MinMax(min, max)
        except StopIteration:
            if default is NoDefault:
                raise
            return MinMax(default, default)

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
    ) -> Iter[R]:
        f = partial(func, *args, **kwargs)
        itr = iter(map(f, self))
        return Iter(itr)

    @tp.overload
    def map[K, R](
        self,
        func: Callable[[T], R],
    ) -> Iter[R]: ...
    @tp.overload
    def map[K, _, R, **P](
        self,
        func: Callable[tp.Concatenate[T, _, ...], R],
        *args: Iterable[K],
    ) -> Iter[R]: ...
    def map[K, R](
        self,
        func: Callable[tp.Concatenate[T, ...], R],
        *args: Iterable[K],
    ) -> Iter[R]:
        return Iter(map(func, self, *args))

    def feed_into[R, **P](
        self,
        func: Callable[tp.Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(self, *args, **kwargs)

    def filter(
        self, predicate: Callable[[T], bool] | None, *, when: bool = True
    ) -> Iter[T]:
        if when:
            return Iter(filter(predicate, self._iter))
        else:
            return Iter(it.filterfalse(predicate, self._iter))

    def starmap[*Ts, R](self: Iter[tuple[*Ts]], func: Callable[[*Ts], R]) -> Iter[R]:
        return Iter(it.starmap(func, self._iter))

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

    def tail(self, n: int) -> Iter[T]:
        return Iter(deque(self, maxlen=n))

    def skip(self, n: int) -> Iter[T]:
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
    def skip_take(self, *, skip: int, take: int, take_first: bool = False) -> Iter[T]:
        if take_first:
            selectors = self._get_skip_take_selectors((True, take), (False, skip))
        else:
            selectors = self._get_skip_take_selectors((False, skip), (True, take))
        return self.compress(selectors)

    @tp.overload
    def chain_from_iter[K1](
        self: Iter[Iterable[K1]], iterable: None = None
    ) -> Iter[K1]: ...

    @tp.overload
    def chain_from_iter[K2](self, iterable: Iterable[Iterable[K2]]) -> Iter[T | K2]: ...

    def chain_from_iter[K1, K2](
        self: tp.Self | Iter[Iterable[K1]],
        iterable: Iterable[Iterable[K2]] | None = None,
    ) -> Iter[K1] | Iter[T | K2]:
        if iterable is None:
            return Iter(it.chain.from_iterable(tp.cast(Iter[Iterable[K1]], self)))
        return Iter(it.chain(tp.cast(Iter[T], self), it.chain.from_iterable(iterable)))

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
    ) -> Iter[tuple[T, T1]]: ...

    @tp.overload
    def zip2[T1](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: tp.Literal[Default.NoDefault],
        strict: tp.Literal[False],
    ) -> Iter[tuple[T, T1]]: ...

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
    ) -> Iter[tuple[T | F, T1 | F]]: ...

    def zip2[T1, F](
        self,
        iter1: Iterable[T1],
        /,
        *,
        fill_value: F = Default.NoDefault,
        strict: bool = False,
    ) -> Iter[tuple[T, T1]] | Iter[tuple[T | F, T1 | F]]:
        itbl = it.zip_longest(self, iter1, fillvalue=fill_value)
        if fill_value is Default.NoDefault:
            itbl = zip(self, iter1, strict=strict)
        elif strict:
            raise ValueError("Cannot specify both fill_value and strict")
        return Iter(itbl)

    @tp.overload
    def zip3[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        strict: bool,
    ) -> Iter[tuple[T, T1, T2]]: ...

    @tp.overload
    def zip3[T1, T2](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: tp.Literal[Default.NoDefault],
    ) -> Iter[tuple[T, T1, T2]]: ...

    @tp.overload
    def zip3[T1, T2, F](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        /,
        *,
        fill_value: F,
    ) -> Iter[tuple[T | F, T1 | F, T2 | F]]: ...

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
    ) -> Iter[tuple[T, T1, T2]] | Iter[tuple[T | F, T1 | F, T2 | F]]:
        itbl = it.zip_longest(self, iter1, iter2, fillvalue=fill_value)
        if fill_value is Default.NoDefault:
            itbl = zip(self, iter1, iter2, strict=strict)
        elif strict:
            raise ValueError("Cannot specify both fill_value and strict")
        return Iter(itbl)

    # TODO: overload zip4
    def zip4[T1, T2, T3](
        self,
        iter1: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
        *,
        strict: bool = False,
    ) -> Iter[tuple[T, T1, T2, T3]]:
        return Iter(zip(self, iter1, iter2, iter3, strict=strict))

    def transpose_eager[TSized: Sized](
        self: Iter[TSized],
        strict: bool = False,
    ) -> Iter[TSized]:
        return Iter(zip(*self._iter, strict=strict))

    def interleave[R](
        self,
        other: Iterable[R],
        other_first: bool = False,
    ) -> Iter[T | R]:
        ch = it.chain.from_iterable(
            zip(other, self._iter) if other_first else zip(self._iter, other)
        )
        return Iter(ch)

    @property
    def stats[TNumber: (float, Decimal, Fraction) = float](
        self: Iter[TNumber],
    ) -> stats[TNumber]:
        return stats[TNumber](self)

    def map_type[R](self, type: type[R]) -> Iter[R]:
        return self.map(type)

    def prepend[V](self, *values: V) -> Iter[T | V]:
        return Iter(it.chain(values, self))

    def append[V](self, *values: V) -> Iter[T | V]:
        return Iter(it.chain(self, values))

    def flatten_once[T1](self: Iter[tuple[T1, ...]]) -> Iter[T1]:
        return Iter(it.chain.from_iterable(self))

    def flatten(self) -> Iter[object]:
        return Iter(flatten(self))

    def concat[R](
        self, *its: Iterable[R], self_position: tp.Literal["front", "back"] = "front"
    ) -> Iter[T | R]:
        match self_position:
            case "front":
                return Iter(it.chain(self._iter, *its))
            case "back":
                return Iter(it.chain(*its, self._iter))

    @tp.overload
    def enumerate(self, indexed: tp.Literal[False]) -> Iter[tuple[int, T]]: ...
    @tp.overload
    def enumerate(self, indexed: tp.Literal[True] = True) -> Iter[Indexed[T]]: ...
    def enumerate(self, indexed: bool = True) -> Iter[tuple[int, T]] | Iter[Indexed[T]]:
        enumerated = Iter(enumerate(self._iter))
        if indexed:
            return enumerated.starmap(Indexed)
        return enumerated


if __name__ == "__main__":
    ch = Iter(range(1, 10)).skip_take(skip=2, take=3, take_first=True)
    ch.exhaust()
    print(ch.last_yielded_value)
    print(ch.last_yielded_index)
    print(ch.peek_next_index())
    print(ch.peek_next_value())
    ch1 = (
        Iter(range(1, 100))
        .skip_take(skip=2, take=3, take_first=True)
        .map(float)
        .skip_take(skip=5, take=5)
        .batched(5)
        .transpose_eager()
        .flatten_once()
        .enumerate()
        .minmax_lazy()
    )
    print(ch1)
    print(ch1.min.idx)
    print(ch1.max.idx)
