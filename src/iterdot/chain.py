# pyright: reportImportCycles=false
from __future__ import annotations

import enum
import itertools as it
import typing as tp
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from decimal import Decimal
from fractions import Fraction
from functools import reduce, wraps
from operator import attrgetter
from pprint import pformat

from iterdot.index import Indexed
from iterdot.minmax import MinMax, lazy_minmax, lazy_minmax_keyed
from iterdot.plugins.stats import stats
from iterdot.wtyping import Comparable


class Default(enum.Enum):
    """Sentinel values used as defaults."""

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
        func: Callable[tp.Concatenate[Callable[[T], bool], Iterable[T], P], Iterable[R]],
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
        if isinstance(item, Iterable) and not isinstance(item, str):
            yield from flatten(tp.cast(Iterable[object], item))
        yield item


class Iter[T](Iterator[T]):
    """
    Iterator over a given iterable, providing method chaining.

    Args:
        iterable: an iterable that is to be turned into an Iter
    """

    def __init__(self, iterable: Iterable[T]) -> None:
        self.iterable = iterable
        self._iter: Iterator[T] = (
            iter(iterable) if not isinstance(iterable, Iter) else iterable._iter
        )
        self._last_yielded_value: T | tp.Literal[Default.Unavailable] = Unavailable
        self._last_yielded_index: int = -1

    def peek_next_index(self) -> int:
        """Peek the next index that would be yielded, if there is element left to yield.

        Returns:
            int: last yielded index

        Example:
            >>> itbl = Iter([1, 2, 3, 4])
            >>> itbl.peek_next_index()
            0
            >>> itbl.peek_next_index()
            0
            >>> _ = itbl.skip(1).next(); itbl.peek_next_index()
            2
            >>> itbl.exhaust(); itbl.peek_next_index()
            4
        """
        return self._last_yielded_index + 1

    @tp.overload
    def peek_next_value(
        self, default: tp.Literal[Default.Exhausted] = Exhausted
    ) -> T | tp.Literal[Default.Exhausted]: ...
    @tp.overload
    def peek_next_value[TDefault](self, default: TDefault) -> T | TDefault: ...
    @tp.no_type_check
    def peek_next_value[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        """Peek the next value that would be yielded, if there is element left to yield.
        Otherwise, return default.

        Returns:
            next value to be yielded or default

        Example:
            >>> itbl = Iter([1, 2, 3, 4])
            >>> itbl.peek_next_value()
            1
            >>> itbl.peek_next_value()
            1
            >>> _ = itbl.skip(1).next(); itbl.peek_next_value()
            3
            >>> itbl.exhaust(); itbl.peek_next_value()
            <Default.Exhausted: 1>
            >>> itbl.exhaust(); itbl.peek_next_value(-1)
            -1
        """
        item = default
        for item in self._iter:
            self._iter = prepend(item, to=self._iter)
            break
        return item

    @property
    def last_yielded_value(self) -> T | tp.Literal[Default.Unavailable]:
        """Return the value that was last yielded, if not yielded at least once,
        return Default.Unavailable.

        Example:
            >>> itbl = Iter([1, 2, 3, 4])
            >>> itbl.last_yielded_value
            <Default.Unavailable: 3>
            >>> itbl.peek_next_value()
            1
            >>> itbl.last_yielded_value
            <Default.Unavailable: 3>
            >>> _ = itbl.skip(1).next(); itbl.last_yielded_value
            2
            >>> itbl.exhaust(); itbl.last_yielded_value
            4
        """
        return self._last_yielded_value

    @property
    def last_yielded_index(self) -> int:
        """Return the index of the value that was last yielded,
        if not yielded at least once, return -1.

        Example:
            >>> itbl = Iter([1, 2, 3, 4])
            >>> itbl.last_yielded_index
            -1
            >>> itbl.peek_next_value()
            1
            >>> itbl.last_yielded_index
            -1
            >>> _ = itbl.skip(1).next(); itbl.last_yielded_index
            1
            >>> itbl.exhaust(); itbl.last_yielded_index
            3
        """
        return self._last_yielded_index

    # NOTE: consider if it should error
    @tp.overload
    def next[TDefault](self, default: tp.Literal[Default.NoDefault] = NoDefault) -> T: ...
    @tp.overload
    def next[TDefault](self, default: TDefault) -> T | TDefault: ...
    def next[TDefault](self, default: TDefault = NoDefault) -> T | TDefault:
        """next value in the iterator.

        Returns:
            next value or default

        Example:
            >>> itbl = Iter([1, 2, 3, 4])
            >>> itbl.next()
            1
            >>> itbl.next()
            2
            >>> itbl.next()
            3
            >>> itbl.next()
            4
            >>> itbl.next(default=-1)
            -1
            >>> itbl.next()
            Traceback (most recent call last):
                ...
            StopIteration
        """
        return next(self) if default is NoDefault else next(self, default)

    # TODO: add itemgetter
    # TODO: Replace type[K] with TypeForm (when available)
    def getattr[K](self, *names: str, type: type[K]) -> Iter[K]:
        del type
        func = tp.cast(Callable[[T], K], attrgetter(*names))  # pyright: ignore[reportInvalidCast]
        return self.map(func)

    compress = MethodKind[T].augmentor(it.compress)
    """see itertools.compress"""
    pairwise = MethodKind[T].augmentor(it.pairwise)
    """see itertools.pairwise"""
    batched = MethodKind[T].augmentor(it.batched)
    """see itertools.batched"""
    accumulate = MethodKind[T].augmentor(it.accumulate)
    """see itertools.accumulate"""
    slice = MethodKind[T].augmentor(it.islice)
    """see itertools.islice"""
    zip_with = MethodKind[T].augmentor(zip)
    """see zip"""

    takewhile = MethodKind[T].predicated_augmentor(it.takewhile)
    """see itertools.takewhile"""
    dropwhile = MethodKind[T].predicated_augmentor(it.dropwhile)
    """see itertools.dropwhile"""

    sum = MethodKind[T].consumer(sum)
    """see sum"""
    to_list = MethodKind[T].consumer(list)
    """convert to list"""

    def load_in_memory(self) -> SeqIter[T]:
        """convert to SeqIter

        Returns:
            SeqIter
        """
        return SeqIter(self)

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
        """calculate the max element in the iterable.

        Args:
            key (optional):
                A function that should be applied on each element
                the result of the function is used for comparison (default: None)

            default (optional):
                default value to return if iterable is empty
                if default is omitted, or default is Default.NoDefault
                a ValueError is raised if the iterable is empty.

        Returns:
            TComparable | F: Either the max element, or default.

        Raises:
            ValueError: If no default is given (similar to default==Default.NoDefault)
                and iterable is empty.

        Example:
            >>> Iter([3, 4, 1, 9]).max()
            9
            >>> Iter([]).max(default=-1)
            -1
            >>> Iter([]).max()
            Traceback (most recent call last):
                ...
            ValueError: max() iterable argument is empty
        """
        try:
            return (
                max(self, key=key)
                if default is Default.NoDefault
                else max(self, key=key, default=default)
            )
        except ValueError:
            raise ValueError("max() iterable argument is empty") from None

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
        """calculate the min element in the iterable.

        Args:
            key (optional):
                A function that should be applied on each element
                the result of the function is used for comparison (default: None)

            default (optional):
                default value to return if iterable is empty
                if default is omitted, or default is Default.NoDefault
                a ValueError is raised if the iterable is empty.

        Returns:
            TComparable | F: Either the min element, or default.

        Raises:
            ValueError: If no default is given (similar to default==Default.NoDefault)
                and iterable is empty.

        Example:
            >>> Iter([3, 4, 1, 9]).min()
            1
            >>> Iter([]).min(default=-1)
            -1
            >>> Iter([]).min()
            Traceback (most recent call last):
                ...
            ValueError: min() iterable argument is empty
        """
        try:
            return (
                min(self, key=key)
                if default is Default.NoDefault
                else min(self, key=key, default=default)
            )
        except ValueError:
            raise ValueError("min() iterable argument is empty") from None

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
        """Eagerly calculate min and max by loading the entire iterator in memory.

        Args:
            key (optional):
                A function that should be applied on each element
                the result of the function is used for comparison (default: None)

            default (optional):
                default value to return if iterable is empty
                if default is omitted, or default is Default.NoDefault
                a ValueError is raised if the iterable is empty.

        Returns:
            MinMax: A NamedTuple containing (min, max)

        Raises:
            ValueError: If no default is given (similar to default==Default.NoDefault)
                and iterable is empty.

        Example:
            >>> Iter([1, 9, 6, 2]).minmax_eager()
            MinMax(min=1, max=9)
            >>> Iter([]).minmax_eager()
            Traceback (most recent call last):
                ...
            ValueError: minmax() iterable argument is empty
        """
        match (tuple(self), default):
            case (), Default.NoDefault:
                raise ValueError("minmax() iterable argument is empty")
            case (), default:
                return MinMax(default, default)
            case tup, _:
                return MinMax(min(tup, key=key), max(tup, key=key))

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
        """Lazily calculate min and max processing one item at a time.

        Args:
            key (optional):
                A function that should be applied on each element
                the result of the function is used for comparison (default: None)

            default (optional):
                default value to return if iterable is empty
                if default is omitted, or default is Default.NoDefault
                a ValueError is raised if the iterable is empty.

        Returns:
            MinMax: A NamedTuple containing (min, max)

        Raises:
            ValueError: If no default is given (similar to default==Default.NoDefault)
                and iterable is empty.

        Example:
            >>> Iter([1, 9, 6, 2]).minmax_lazy()
            MinMax(min=1, max=9)
            >>> Iter([]).minmax_lazy()
            Traceback (most recent call last):
                ...
            ValueError: minmax() iterable argument is empty
        """
        try:
            return (
                lazy_minmax(self)
                if key is None
                else lazy_minmax_keyed(self, key=key)
            )  # fmt: skip
        except StopIteration:
            if default is NoDefault:
                raise ValueError("minmax() iterable argument is empty") from None
            return MinMax(default, default)

    @tp.override
    def __iter__(self) -> Iterator[T]:
        return self

    @tp.override
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
        """
        map a function with partial arguments on each element of self.

        Equivalent to calling `Iter(map(partial(func, *args, **kwargs), self))`

        Args:
            func: callable to be applied whose first argument is items in self
            *args: other positional arguments for func
            **kwargs: keyword arguments for func

        Returns:
            Iter[R]: Iter after mapping

        Example:
            >>> Iter(["0", "1", "10", "11"]).map_partial(int, base=2).to_list()
            [0, 1, 2, 3]
        """
        return Iter(func(item, *args, **kwargs) for item in self)

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
        """
        map a function on each element of self.

        Equivalent to calling `Iter(map(func, self, *args), self))`
        For more info on syntax, see python builtin `map`.

        Args:
            func: callable to be applied whose first argument is items in self
            *args: iterables of other arguments (if any) that the func takes.

        Returns:
            Iter[R]: Iter after mapping

        Example:
            >>> Iter(["0", "1", "10", "11"]).map(int).to_list()
            [0, 1, 10, 11]
        """
        return Iter(map(func, self, *args))

    def feed_into[R, **P](
        self,
        func: Callable[tp.Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """
        Apply a function which takes the whole of self as its first argument.

        Args:
            func: A callable which takes the whole of self as its first argument.
            *args: Other positional arguments to the function if any.
            **kwargs: Keywork arguments to the function if any.

        Returns:
            R: return value of `func`.

        Example:
            >>> Iter([("a", 0), ("b", 1)]).feed_into(dict)
            {'a': 0, 'b': 1}
            >>> Iter([1, 2, 3, 4]).sum(start=10)
            20
        """
        return func(self, *args, **kwargs)

    def filter(
        self, predicate: Callable[[T], bool] | None, *, invert: bool = False
    ) -> Iter[T]:
        """
        Filter self based on predicate.

        Args:
            predicate: A callable that returns bool, or None.
                If predicate is None, it is equivalent to passing bool as predicate.
            invert (optional): if True, return elements for which predicate was True,
                otherwise, return elements for which predicate was False.
                default: False.

        Returns:
            Iter: elements which satisfy the predicate.

        Example:
            >>> Iter([1, 2, 3, 4]).filter(lambda x: x % 2 == 0).to_list()
            [2, 4]
            >>> Iter([1, 2, 3, 4]).filter(lambda x: x % 2 == 0, invert=True).to_list()
            [1, 3]
        """
        return (
            Iter(it.filterfalse(predicate, self._iter))
            if invert
            else Iter(filter(predicate, self._iter))
        )

    def starmap[*Ts, R](self: Iter[tuple[*Ts]], func: Callable[[*Ts], R]) -> Iter[R]:
        """
        see itertools.starmap

        Returns:
            Iter

        Example:
            >>> from operator import add
            >>> Iter([(0, 1), (10, 20)]).starmap(add).to_list()
            [1, 30]
        """
        return Iter(it.starmap(func, self._iter))

    @tp.overload
    def first[TDefault](self, default: tp.Literal[Default.NoDefault]) -> T: ...
    @tp.overload
    def first[TDefault](self, default: TDefault = Exhausted) -> T | TDefault: ...
    def first[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        """
        Return the first item of self, or default if Iterable is empty.

        Args:
            default (optional): Return default if default is not empty, and iterable is empty.
                If default == Default.NoDefault, ValueError is raised if iterable is empty.
                default: Default.Exhausted

        Returns:
            T | TDefault: first item in self, or default.

        Raises:
            StopIteration: if default == Default.NoDefault and iterable is empty

        Example:
            >>> Iter([0, 1, 2, 3]).first()
            0
            >>> Iter([]).first()
            <Default.Exhausted: 1>
            >>> Iter([]).first(default=Default.NoDefault)
            Traceback (most recent call last):
                ...
            StopIteration
        """
        try:
            return next(self) if default is NoDefault else next(self, default)
        except StopIteration:
            raise StopIteration from None

    # TODO: support getitem
    @tp.overload
    def at[TDefault](self, n: int, default: tp.Literal[Default.NoDefault]) -> T: ...
    @tp.overload
    def at[TDefault](self, n: int, default: TDefault = Exhausted) -> T | TDefault: ...
    def at[TDefault](self, n: int, default: TDefault = Exhausted) -> T | TDefault:
        return self.skip(n).next(default)

    @tp.overload
    def last[TDefault](self, default: tp.Literal[Default.NoDefault]) -> T: ...
    @tp.overload
    def last[TDefault](self, default: TDefault = Exhausted) -> T | TDefault: ...
    def last[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        """
        Return the last item of self, or default if Iterable is empty.

        Args:
            default (optional): Return default if default is not empty, and iterable is empty.
                If default == NoDefault, ValueError is raised if iterable is empty.
                default: Default.Exhausted

        Returns:
            T | TDefault: last item in self, or default.

        Raises:
            StopIteration: if default == Default.NoDefault and iterable is empty

        Example:
            >>> Iter([0, 1, 2, 3]).last()
            3
            >>> Iter([]).last()
            <Default.Exhausted: 1>
            >>> Iter([]).last(default=Default.NoDefault)
            Traceback (most recent call last):
                ...
            StopIteration: Underlying iterable is empty
        """
        try:
            return deque(self, maxlen=1).popleft()
        except IndexError:
            if default is NoDefault:
                raise StopIteration("Underlying iterable is empty") from None
            return default

    def tail(self, n: int) -> Iter[T]:
        """
        Return n items from end if available, or return all available items.

        Args:
            n: number of items to return from end.

        Returns:
            Iter: Iter containing at most n elements from last.

        Example:
            >>> Iter([0, 1, 2, 3]).tail(2).to_list()
            [2, 3]
            >>> Iter([0, 1, 2, 3]).tail(5).to_list()
            [0, 1, 2, 3]
        """
        return Iter(deque(self, maxlen=n))

    def head(self, n: int) -> Iter[T]:
        """
        Return n items from beginning if available, or return all available items.

        Args:
            n: number of items to return from beginning.

        Returns:
            Iter: Iter containing at most n elements from beginning.

        Example:
            >>> Iter([0, 1, 2, 3]).head(2).to_list()
            [0, 1]
            >>> Iter([0, 1, 2, 3]).head(5).to_list()
            [0, 1, 2, 3]
        """
        return self.slice(n)

    def skip(self, n: int) -> Iter[T]:
        """Advance the iterator n positions.

        Args:
            n: number of positions to skip

        Returns:
            Iter: Iter advanced n positions.

        Example:
            >>> Iter([1, 2, 3, 4]).skip(2).to_list()
            [3, 4]
        """
        return self if not n else self.slice(n, None)

    def exhaust(self) -> None:
        """Exhaust all items in self. Could be used for side-effects.

        See Also:
            Iter.foreach

        Example:
            >>> it = Iter([1, 2, 3, 4])
            >>> it.exhaust()
            >>> it.next(default=Default.Exhausted)
            <Default.Exhausted: 1>
            >>> Iter([1, 2, 3, 4]).map(print).exhaust()
            1
            2
            3
            4
        """
        deque[T](maxlen=0).extend(self)

    def foreach[R](self, func: Callable[[T], None]) -> None:
        """map func on each item of self, and exhaust.

        See Also:
            Iter.exhaust

        Example:
            >>> Iter([1, 2, 3, 4]).foreach(print)
            1
            2
            3
            4
        """
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
        *,
        strict: bool = False,
    ) -> Iter[TSized]:
        return Iter(zip(*self._iter, strict=strict))

    def interleave[R](
        self,
        other: Iterable[R],
        *,
        other_first: bool = False,
    ) -> Iter[T | R]:
        ch = it.chain.from_iterable(
            zip(other, self._iter, strict=False)
            if other_first
            else zip(self._iter, other, strict=False)
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
    def enumerate(
        self, *, indexed: tp.Literal[False], start: int = 0
    ) -> Iter[tuple[int, T]]: ...
    @tp.overload
    def enumerate(
        self, *, indexed: tp.Literal[True] = True, start: int = 0
    ) -> Iter[Indexed[T]]: ...
    def enumerate(
        self, *, indexed: bool = True, start: int = 0
    ) -> Iter[tuple[int, T]] | Iter[Indexed[T]]:
        enumerated = Iter(enumerate(self._iter, start=start))
        if indexed:
            return enumerated.starmap(Indexed)
        return enumerated


class SeqIter[T](Sequence[T]):
    def __init__(self, iterable: Iterable[T]) -> None:
        if not isinstance(iterable, SeqIter):
            self._iterable = iterable if isinstance(iterable, tuple) else tuple(iterable)
        else:
            self._iterable = iterable.iterable

    @property
    def iterable(self) -> tuple[T, ...]:
        return self._iterable

    @tp.override
    def __iter__(self) -> Iterator[T]:
        return iter(self.iterable)

    @tp.overload
    def __getitem__(self, index: int) -> T: ...
    @tp.overload
    def __getitem__(self, index: slice) -> SeqIter[T]: ...
    @tp.override
    def __getitem__(self, index: int | slice) -> T | SeqIter[T]:
        return (
            self.iterable[index]
            if isinstance(index, int)
            else SeqIter(self.iterable[index])
        )

    @tp.override
    def __len__(self) -> int:
        return len(self.iterable)

    @tp.override
    def __contains__(self, value: object) -> bool:
        return value in self.iterable

    @tp.override
    def __reversed__(self) -> Iter[T]:
        return Iter(reversed(self.iterable))

    def reversed(self) -> Iter[T]:
        return Iter(reversed(self.iterable))

    @tp.overload
    def enumerate(
        self, *, indexed: tp.Literal[False], start: int = 0
    ) -> SeqIter[tuple[int, T]]: ...
    @tp.overload
    def enumerate(
        self, *, indexed: tp.Literal[True] = True, start: int = 0
    ) -> SeqIter[Indexed[T]]: ...
    def enumerate(
        self, *, indexed: bool = True, start: int = 0
    ) -> SeqIter[tuple[int, T]] | SeqIter[Indexed[T]]:
        enumerated = SeqIter(enumerate(self.iterable, start=start))
        if indexed:
            return enumerated.starmap(Indexed)
        return enumerated

    @tp.overload
    def max[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: tp.Literal[Default.NoDefault] = NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def max[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F: ...
    def max[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F:
        return (
            max(self, key=key)
            if default is Default.NoDefault
            else max(self, key=key, default=default)
        )

    @tp.overload
    def min[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def min[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F: ...
    def min[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F:
        return (
            min(self, key=key)
            if default is Default.NoDefault
            else min(self, key=key, default=default)
        )

    @tp.overload
    def minmax[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> MinMax[TComparable]: ...
    @tp.overload
    def minmax[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> MinMax[TComparable] | MinMax[F]: ...
    def minmax[TComparable: Comparable, RComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], RComparable] | None = None,
        default: F = Default.NoDefault,
    ) -> MinMax[TComparable] | MinMax[F]:
        match self.iterable, default:
            case (), Default.NoDefault:
                raise ValueError("minmax() iterable argument is empty")
            case (), default:
                return MinMax(default, default)
            case tup, _:
                return MinMax(min(tup, key=key), max(tup, key=key))

    def iter(self) -> Iter[T]:
        return Iter(self.iterable)

    def map_partial[R, **P](
        self,
        func: Callable[tp.Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> SeqIter[R]:
        return SeqIter(func(item, *args, **kwargs) for item in self)

    @tp.overload
    def map[R](
        self,
        func: Callable[[T], R],
    ) -> SeqIter[R]: ...
    @tp.overload
    def map[K, _, R](
        self,
        func: Callable[tp.Concatenate[T, _, ...], R],
        *args: Iterable[K],
    ) -> SeqIter[R]: ...
    def map[K, R](
        self,
        func: Callable[tp.Concatenate[T, ...], R],
        *args: Iterable[K],
    ) -> SeqIter[R]:
        return SeqIter(map(func, self, *args))

    def feed_into[R, **P](
        self,
        func: Callable[tp.Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(self, *args, **kwargs)

    def filter(
        self, predicate: Callable[[T], bool] | None, *, invert: bool = False
    ) -> SeqIter[T]:
        return (
            SeqIter(it.filterfalse(predicate, self.iterable))
            if invert
            else SeqIter(filter(predicate, self.iterable))
        )

    def starmap[*Ts, R](
        self: SeqIter[tuple[*Ts]], func: Callable[[*Ts], R]
    ) -> SeqIter[R]:
        return SeqIter(it.starmap(func, self))

    @tp.overload
    def first[TDefault](
        self, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def first[TDefault](self, default: TDefault) -> T | TDefault: ...
    def first[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        return self[0] if default is NoDefault else default

    # TODO: support getitem
    @tp.overload
    def at[TDefault](
        self, n: int, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def at[TDefault](self, n: int, default: TDefault) -> T | TDefault: ...
    def at[TDefault](self, n: int, default: TDefault = Exhausted) -> T | TDefault:
        try:
            return self[n]
        except IndexError:
            if default is NoDefault:
                raise
            return default

    @tp.overload
    def last[TDefault](self, default: tp.Literal[Default.NoDefault] = NoDefault) -> T: ...
    @tp.overload
    def last[TDefault](self, default: TDefault) -> T | TDefault: ...
    def last[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        return self[-1] if default is NoDefault else default

    def tail(self, n: int) -> SeqIter[T]:
        return SeqIter(self[-n:])

    def skip(self, n: int) -> SeqIter[T]:
        return SeqIter(self[n:])

    def foreach(self, func: Callable[[T], None]) -> None:
        self.iter().foreach(func)

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
    def skip_take(self, *, skip: int, take: int, take_first: bool = False) -> SeqIter[T]:
        if take_first:
            selectors = self._get_skip_take_selectors((True, take), (False, skip))
        else:
            selectors = self._get_skip_take_selectors((False, skip), (True, take))
        return SeqIter(it.compress(self, selectors))

    # TODO: consider default
    def reduce(self, func: Callable[[T, T], T], initial: T | None = None) -> T:
        if initial is None:
            return reduce(func, self)
        return reduce(func, self, initial)

    def transpose[TSized: Sized](
        self: SeqIter[TSized],
        *,
        strict: bool = False,
    ) -> SeqIter[TSized]:
        return SeqIter(zip(*self, strict=strict))

    @property
    def stats[TNumber: (float, Decimal, Fraction) = float](
        self: SeqIter[TNumber],
    ) -> stats[TNumber]:
        return stats[TNumber](self)

    def prepend[V](self, *values: V) -> SeqIter[T | V]:
        return SeqIter(values + self.iterable)

    def append[V](self, *values: V) -> SeqIter[T | V]:
        return SeqIter(self.iterable + values)

    def concat[R](
        self, *its: Iterable[R], self_position: tp.Literal["front", "back"] = "front"
    ) -> SeqIter[T | R]:
        if self_position not in ["front", "back"]:
            raise ValueError(
                f'`self_position` must be "front" or "back", got {self_position!r}'
            )

        match self_position:
            case "front":
                return SeqIter(it.chain(self.iterable, *its))
            case "back":
                return SeqIter(it.chain(*its, self.iterable))


if __name__ == "__main__":
    from doctest import testmod
    from pathlib import Path

    dummy_file = Path("./dummy.config").resolve()

    # fmt: off
    with dummy_file.open() as file_handle:
        key_val_tuples = (
            Iter(file_handle)
            .map(str.strip)
            .filter(None)
            .map_partial(str.split, sep=" = ", maxsplit=1)
            .load_in_memory()
        )

        config = key_val_tuples.feed_into(dict)
        keys, vals = key_val_tuples.transpose()

    player_scores = [12, 55, 89, 82, 37, 16]
    qualified = (
        Iter(player_scores)
        .filter(lambda x: x > 35)
        .load_in_memory()
    )

    minmax_info = qualified.enumerate().minmax()
    statistics = qualified.stats()
    # fmt: on

    print(f"config: {pformat(config)}")
    print(f"Winner => player_id: {minmax_info.max.idx}, score: {minmax_info.max.value}")
    print(f"Loser  => player_id: {minmax_info.min.idx}, score: {minmax_info.min.value}")
    print(f"Player Stats: {statistics}")

    _ = testmod()
