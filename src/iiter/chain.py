from __future__ import annotations

import itertools as it
from operator import attrgetter
import typing as tp
from collections import deque
from collections.abc import Callable, Iterable, Iterator
import enum
from functools import reduce, wraps
from typing_extensions import cast, override

ValueType = enum.Enum("ValueType", ["NA"])


class MethodType[T, R]:
    @staticmethod
    def consumer[**P](
        func: Callable[tp.Concatenate[Iterable[T], P], R],
    ) -> Callable[tp.Concatenate[ChainableIter[T], P], R]:
        @wraps(func)
        def inner(self: ChainableIter[T], *args: P.args, **kwargs: P.kwargs) -> R:
            return func(self, *args, **kwargs)

        return inner

    @staticmethod
    def augmentor[**P](
        func: Callable[tp.Concatenate[Iterable[T], P], Iterable[R]],
    ) -> Callable[tp.Concatenate[ChainableIter[T], P], ChainableIter[R]]:
        @wraps(func)
        def inner(
            self: ChainableIter[T], *args: P.args, **kwargs: P.kwargs
        ) -> ChainableIter[R]:
            return ChainableIter(func(self, *args, **kwargs))

        return inner

    @staticmethod
    def predicated_augmentor[**P](
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

    @staticmethod
    def chained[**P](
        func: Callable[P, Iterator[R]],
    ) -> Callable[P, ChainableIter[R]]:
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> ChainableIter[R]:
            return ChainableIter(func(*args, **kwargs))

        return inner


def prepend[T](iterable: Iterator[T], *val: T) -> Iterator[T]:
    yield from val
    yield from iterable


class ChainableIter[T](Iterable[T]):
    def __init__(self, iterable: Iterable[T], copy_indices: bool = False) -> None:
        self.iterable = iterable
        self._iter = iter(iterable)
        if copy_indices:
            if isinstance(self.iterable, ChainableIter):
                self._counter = self.iterable.indices()
            else:
                raise ValueError(
                    "`iterable` is not an instance of ChainableIter, cannot copy indices"
                )
        else:
            self._counter: Iterator[int] = it.count()

    def indices(self) -> Iterator[int]:
        return self._counter

    def next_index(self) -> int:
        val = next(self._counter)
        self._counter = prepend(self._counter, val)
        return val

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
        func = cast(Callable[[T], K], attrgetter(name))  # pyright: ignore[reportInvalidCast]
        return self.map(func)

    compress = MethodType[T, T].augmentor(it.compress)
    pairwise = MethodType[T, tuple[T, T]].augmentor(it.pairwise)
    batched = MethodType[T, tuple[T, ...]].augmentor(it.batched)
    accumulate = MethodType[T, T].augmentor(it.accumulate)
    chain = MethodType[T, T].augmentor(it.chain)
    slice = MethodType[T, T].augmentor(it.islice)
    transpose = MethodType[T, tuple[T]].augmentor(zip)

    takewhile = MethodType[T, T].predicated_augmentor(it.takewhile)
    dropwhile = MethodType[T, T].predicated_augmentor(it.dropwhile)

    sum = MethodType[T, T].consumer(sum)
    to_list = MethodType[T, list[T]].consumer(list)

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self.__next__, None)

    def __next__(self) -> T:
        return next(self._iter)

    def map[R, **P](
        self,
        func: Callable[tp.Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ChainableIter[R]:
        return ChainableIter(map(func, self._iter, *args, **kwargs))

    def feed_into[R, **P](
        self,
        func: Callable[tp.Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(self._iter, *args, **kwargs)

    def filter(
        self, predicate: Callable[[T], bool] | None, when: bool = True
    ) -> ChainableIter[T]:
        filter_func = filter[T] if when else it.filterfalse[T]
        return ChainableIter(filter_func(predicate, self._iter))

    def starmap[*Ts, R](
        self: ChainableIter[tuple[*Ts]], func: Callable[[*Ts], R]
    ) -> ChainableIter[R]:
        return ChainableIter(it.starmap(func, self._iter))

    def tee(self) -> ChainableIter[T]:
        self._iter, copy = it.tee(self._iter)
        return ChainableIter(copy)

    def first(self) -> T:
        return next(iter(self._iter))

    def nth_or_last(self, n: int) -> T:
        return self.slice(n).last()

    def last(self) -> T:
        return deque(self._iter, maxlen=1).popleft()

    def tail(self, n: int) -> ChainableIter[T]:
        return ChainableIter(deque(self, maxlen=n))

    def take(self, n: int) -> ChainableIter[T]:
        return self.slice(n)

    def skip(self, n: int) -> ChainableIter[T]:
        return self.slice(n + 1, None)

    def consume(self) -> None:
        out = deque(self, maxlen=0)
        del out

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

    def zipn(
        self,
        *iters: Iterable[tp.Any],
        strict: bool = False,
    ) -> ChainableIter[tuple[tp.Any, ...]]:
        return ChainableIter(zip(self, *iters, strict=strict))


if __name__ == "__main__":
    ch = ChainableIter(range(10)).skip_take(skip=2, take=3, take_first=True).to_list()
    print(ch)
