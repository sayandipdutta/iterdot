# pyright: reportImportCycles=false
from __future__ import annotations

import itertools as it
import typing as tp
from collections import defaultdict, deque
from collections.abc import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
    Sized,
)
from contextlib import suppress
from decimal import Decimal
from fractions import Fraction
from functools import reduce
from operator import add, attrgetter

from iterdot._helpers import (
    flatten,
    prepend,
    skip_take_by_order,
    sliding_window_iter,
    sliding_window_seq,
)
from iterdot.defaults import (
    IGNORE,
    Default,
    Exhausted,
    Ignore,
    MissingPolicy,
    NoDefault,
    Pad,
    Raise,
    Unavailable,
)
from iterdot.index import Indexed
from iterdot.minmax import MinMax, lazy_minmax, lazy_minmax_keyed
from iterdot.operators import IsEqual, Unpacked
from iterdot.plugins.stats import stats
from iterdot.wtyping import Comparable, Predicate, SupportsAdd, SupportsSumNoDefault


@tp.final
class Collector[TIter]:
    def __init__(self, instance: Iterable[TIter]) -> None:
        self.instance = instance

    def __getitem__[T](self, type: type[T]) -> Callable[[], T]:
        inner_type = tp.get_args(type)
        try:
            inner = inner_type[0]  # pyright: ignore[reportAny]
            return lambda: type(map(inner, self.instance))  # pyright: ignore[reportCallIssue, reportAny]
        except (IndexError, TypeError, ValueError):
            return lambda: type(self.instance)  # pyright: ignore[reportCallIssue]

    def __call__(self) -> SeqIter[TIter]:
        return SeqIter(self.instance)


@tp.final
class Iter[T](Iterator[T]):
    """
    Iterator over a given iterable, providing method chaining.

    Args:
        iterable: an iterable that is to be turned into an Iter
    """

    def __init__(self, iterable: Iterable[T] = ()) -> None:
        self.iterable = iterable
        self._iter: Iterator[T] = (
            iter(iterable) if not isinstance(iterable, Iter) else iterable._iter
        )
        self._last_yielded_value: T | tp.Literal[Default.Unavailable] = Unavailable
        self._last_yielded_index: int = -1

    @tp.overload
    @classmethod
    def successor(
        cls, start: T, *, producer: Callable[[T], T], window_size: None = None
    ) -> Iter[T]: ...
    @tp.overload
    @classmethod
    def successor(
        cls, start: tuple[T], *, producer: Callable[[Iterable[T]], T], window_size: int
    ) -> Iter[T]: ...
    @classmethod
    def successor(
        cls,
        start: T | tuple[T],
        *,
        producer: Callable[[T], T] | Callable[[Iterable[T]], T],
        window_size: int | None = None,
    ) -> Iter[T]:
        """
        Create an Iter from a producer function that creates the successive element from last value.

        Args:
            *start (*Ts): starting element(s)
            producer (callable): function that creates new element based on previous value(s)

        Returns:
            Iter

        Example:
            >>> double = lambda x: 2 * x
            >>> Iter.successor(1, producer=double).slice(stop=5).to_list()
            [1, 2, 4, 8, 16]
            >>> Iter.successor((0, 1), producer=sum, window_size=2).slice(stop=6).to_list()
            [0, 1, 1, 2, 3, 5]
        """

        def gen() -> Generator[T]:
            if window_size is None:
                item = tp.cast(T, start)
                func = tp.cast(Callable[[T], T], producer)
                yield item
                while True:
                    yield (item := func(item))
            else:
                items = tp.cast(tuple[T, ...], start)
                assert len(items) == window_size, "window_size != len(start)"
                func = tp.cast(Callable[[Iterable[T]], T], producer)
                window = deque(items, maxlen=len(items))
                yield from window
                while True:
                    yield (item := func(window))
                    window.append(item)

        return Iter(gen())

    @classmethod
    def repeat(cls, value: T, times: int | None = None) -> tp.Self:
        """Create an Iter that repeats a value a specified number of times or infinitely.

        Args:
            value: The value to repeat
            times: Number of times to repeat the value. If None, repeat infinitely.

        Returns:
            Iter: Iterator repeating the value

        Example:
            >>> Iter.repeat(1, times=3).to_list()
            [1, 1, 1]
            >>> Iter.repeat('a', times=2).to_list()
            ['a', 'a']
            >>> Iter.repeat(True).slice(stop=4).to_list()
            [True, True, True, True]
        """
        return cls(it.repeat(value) if times is None else it.repeat(value, times))

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
    def getattr[K](self: Iterable[T], *names: str, type: type[K]) -> Iter[K]:
        del type
        func = tp.cast(Callable[[T], K], attrgetter(*names))  # pyright: ignore[reportInvalidCast]
        return Iter(map(func, self))

    def compress(self: Iterable[T], selectors: Iterable[object]) -> Iter[T]:
        return Iter(it.compress(self, selectors=selectors))

    def pairwise(self: Iterable[T]) -> Iter[tuple[T, T]]:
        return Iter(it.pairwise(self))

    def batched(
        self: Iterable[T], n: int = 2, *, strict: bool = False
    ) -> Iter[tuple[T, ...]]:
        return Iter(it.batched(self, n, strict=strict))

    def takewhile(self: Iterable[T], predicate: Predicate[T]) -> Iter[T]:
        return Iter(it.takewhile(predicate, self))

    def dropwhile(self: Iterable[T], predicate: Predicate[T]) -> Iter[T]:
        return Iter(it.dropwhile(predicate, self))

    def accumulate(
        self: Iterable[T], func: Callable[[T, T], T], *, initial: T | None = None
    ) -> Iter[T]:
        """
        see itertools.accumulate

        Returns:
            Iter
        """
        return Iter(it.accumulate(self, func, initial=initial))

    def slice(
        self: Iterable[T],
        *,
        start: int | None = 0,
        stop: int | None = None,
        step: int | None = 1,
    ) -> Iter[T]:
        """
        see itertools.islice

        Returns:
            Iter
        """
        return Iter(it.islice(self, start, stop, step))

    # fmt: off
    @tp.overload
    def sum(self: Iterable[bool | int], /, start: int = 0) -> int: ...
    @tp.overload
    def sum[S: SupportsSumNoDefault](self: Iterable[S], /) -> S | tp.Literal[0]: ...
    @tp.overload
    def sum[A1: SupportsAdd[object, object], A2: SupportsAdd[object, object]](self: Iterable[A1], /, start: A2) -> A1 | A2: ...
    # fmt: on

    @tp.no_type_check
    def sum(self: Iterable[T], start: T | int = 0) -> T:
        """
        Return the sum of a 'start' value (default: 0) plus an iterable of numbers

        When the iterable is empty, return the start value.
        This function is intended specifically for use with numeric values and may
        reject non-numeric types.

        Returns:
            result of sum
        """
        return sum(self, start=start)

    def to_list(self: Iterable[T]) -> list[T]:
        return list(self)

    @property
    def collect(self: Iter[T]) -> Collector[T]:
        return Collector[T](self)

    @tp.overload
    def find(
        self: Iterable[T],
        finder: Callable[[T], bool] | T,
        default: tp.Literal[Default.NoDefault] = NoDefault,
    ) -> Indexed[T]: ...
    @tp.overload
    def find[F](
        self: Iterable[T],
        finder: Callable[[T], bool] | T,
        default: F,
    ) -> Indexed[T] | F: ...
    def find[F](
        self: Iterable[T],
        finder: Callable[[T], bool] | T,
        default: F = NoDefault,
    ) -> Indexed[T] | F:
        """
        Find a value and its index in the iterable, either by passing the value to find or a callable

        Args:
            finder (T | callable): Either the value to find, or a callable that returns True for the item
            default: Default value to return, if not given or Default.NoDefault it has no effect.
                (default: Default.NoDefault)

        Returns:
            If found returns Indexed object, otherwise return default.

        Example:
            >>> Iter([1, 2, 3, 4]).find(2)
            Indexed(idx=1, value=2)
            >>> Iter([1, 2, 3, 4]).find(lambda x: x**2 == 4)
            Indexed(idx=1, value=2)
            >>> Iter([1, 2, 3, 4]).find(5, default=Default.Unavailable)
            <Default.Unavailable: 3>
        """
        itbl = self if isinstance(self, Iter) else Iter(self)
        if callable(finder):
            finder = tp.cast(Callable[[T], bool], finder)
            return (
                itbl.enumerate().filter(lambda x: finder(x.value)).next(default=default)
            )
        return itbl.enumerate().filter(IsEqual(finder)).next(default=default)

    @tp.overload
    def max[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def max[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F: ...
    def max[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
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
    def min[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def min[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F: ...
    def min[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
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
    def minmax_eager[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> MinMax[TComparable]: ...
    @tp.overload
    def minmax_eager[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = Default.NoDefault,
    ) -> MinMax[TComparable] | MinMax[F]: ...
    def minmax_eager[TComparable: Comparable, F](
        self: Iterable[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
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
    def minmax_lazy(
        self: Iterator[T],
        /,
        *,
        key: Callable[[T], Comparable],
    ) -> MinMax[T]: ...
    @tp.overload
    def minmax_lazy[TComparable: Comparable](
        self: Iterator[TComparable],
        /,
        *,
        key: None = None,
    ) -> MinMax[TComparable]: ...
    @tp.overload
    def minmax_lazy[F](
        self: Iterator[T],
        /,
        *,
        key: Callable[[T], Comparable],
        default: F,
    ) -> MinMax[T] | MinMax[F]: ...
    @tp.overload
    def minmax_lazy[TComparable: Comparable, F](
        self: Iterator[TComparable],
        /,
        *,
        key: None = None,
        default: F,
    ) -> MinMax[TComparable] | MinMax[F]: ...
    @tp.no_type_check
    def minmax_lazy(self: Iterable[T], /, *, default=Default.NoDefault, key=None):
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

    def all(self: Iterable[T], predicate: Callable[[T], bool] | None = None) -> bool:
        """
        Check if all elements are Truthy or if given a prediate, check
        if all element evaluate to True when the predicate is applied.


        If self is empty, return True.

        Args:
            predicate (optional callable): if given, check if predicate is Truthy
                default, None.

        Returns:
            bool: Whether all the items were Truthy.

        Example:
            >>> Iter([True] * 5).all()
            True
            >>> Iter(range(5)).all(lambda x: x < 5)
            True
            >>> Iter(range(5)).all(lambda x: x%2 == 0)
            False
            >>> Iter(()).all()
            True
        """
        return all(self) if predicate is None else all(map(predicate, self))

    def any(self: Iterable[T], predicate: Predicate[T] | None = None) -> bool:
        """
        Check if any elements is Truthy or if given a prediate, check
        if any element evaluates to True when the predicate is applied.

        If self is empty, return False.

        Args:
            predicate (optional callable): if given, check if predicate is Truthy
                default, None.

        Returns:
            bool: Whether any of the items were Truthy.

        Example:
            >>> Iter([True] * 5).any()
            True
            >>> Iter(range(5)).any(lambda x: x < 5)
            True
            >>> Iter(range(5)).any(lambda x: x%2 == 0)
            True
            >>> Iter(()).any()
            False
        """
        return any(self) if predicate is None else any(map(predicate, self))

    def all_equal(self: Iterable[T]) -> bool:
        """
        Check if all elements are equal to each other.

        If self is empty, return True.

        Returns:
            bool: Whether all the items were equal.

        Example:
            >>> Iter([True] * 5).all_equal()
            True
            >>> Iter(range(5)).all_equal()
            False
            >>> Iter(()).all_equal()
            True
        """
        first = next(iter(self), Exhausted)
        if first is Exhausted:
            return True
        return all(first == item for item in self)

    def all_equal_with(self: Iterable[T], value: T | None = None) -> bool:
        """
        Check if all elements are equal to the given `value`.

        If self is empty, return False.

        Returns:
            bool: Whether all the items were equal to `value`.

        Example:
            >>> Iter([2] * 5).all_equal_with(2)
            True
            >>> Iter(range(5)).all_equal_with(2)
            False
            >>> Iter(()).all_equal_with(2)
            False
        """
        # BUG: potential bug, what happens if value is Exhausted, and iterable empty
        if next(iter(self), Exhausted) == Default.Exhausted != value:
            return False
        return all(value == item for item in self)

    def map_partial[R, **P](
        self: Iterable[T],
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
        self: Iterable[T],
        func: Callable[[T], R],
    ) -> Iter[R]: ...
    @tp.overload
    def map[K, _, R, **P](
        self: Iterable[T],
        func: Callable[tp.Concatenate[T, _, ...], R],
        *args: Iterable[K],
    ) -> Iter[R]: ...
    def map[K, R](
        self: Iterable[T],
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
            >>> Iter([1, 2, 3, 4]).feed_into(sum)
            10
        """
        return func(self, *args, **kwargs)

    def filter(
        self: Iterable[T], predicate: Predicate[T] | None, *, invert: bool = False
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
        itbl = self._iter if isinstance(self, Iter) else iter(self)
        return (
            Iter(it.filterfalse(predicate, itbl))
            if invert
            else Iter(filter(predicate, itbl))
        )

    def starmap[*Ts, R](self: Iterable[tuple[*Ts]], func: Callable[[*Ts], R]) -> Iter[R]:
        """
        see itertools.starmap

        Returns:
            Iter

        Example:
            >>> from operator import add
            >>> Iter([(0, 1), (10, 20)]).starmap(add).to_list()
            [1, 30]
        """
        itbl = self._iter if isinstance(self, Iter) else iter(self)
        return Iter(it.starmap(func, itbl))

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

    def tail(self: Iterable[T], n: int) -> Iter[T]:
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

    def head(self: Iterable[T], n: int) -> Iter[T]:
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
        return Iter(it.islice(self, n))

    def skip(self: Iterable[T], n: int) -> Iter[T]:
        """Advance the iterator n positions.

        Args:
            n: number of positions to skip

        Returns:
            Iter: Iter advanced n positions.

        Example:
            >>> Iter([1, 2, 3, 4]).skip(2).to_list()
            [3, 4]
        """
        itbl = self if isinstance(self, Iter) else Iter(self)
        return itbl if not n else itbl.slice(start=n, stop=None)

    def exhaust(self: Iterable[T]) -> None:
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

    def foreach[R](self: Iterable[T], func: Callable[[T], object]) -> None:
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
        return deque[object](maxlen=0).extend(map(func, self))

    @staticmethod
    def _get_skip_take_selectors(
        s1: tuple[bool, int], s2: tuple[bool, int]
    ) -> Iterator[bool]:
        while True:
            yield from it.repeat(*s1)
            yield from it.repeat(*s2)

    @skip_take_by_order
    def skip_take(
        self: Iterable[T], *, skip: int, take: int, take_first: bool = False
    ) -> Iter[T]:
        """
        skip some elements followed by take some elements, or vice versa.

        Whether to take first, or skip first, can be specified in two ways.
        1. Via `take_first` boolean flag.
        2. Via order of the arguments, that is, skip=s, take=t will skip first,
        whereas, take=t, skip=s, will take first.

        Args:
            skip (int): number of elements to skip.
            take (int): number of elements to take.
            take_first (Optional[bool]): If True, take first, else skip first.
                default, False.

        Returns:
            Self

        Examples:
            >>> Iter(range(10)).skip_take(skip=2, take=3).to_list()
            [2, 3, 4, 7, 8, 9]
            >>> Iter(range(10)).skip_take(take=3, skip=2).to_list()
            [0, 1, 2, 5, 6, 7]
        """
        itbl = self if isinstance(self, Iter) else Iter(self)
        if take_first:
            selectors = itbl._get_skip_take_selectors((True, take), (False, skip))
        else:
            selectors = itbl._get_skip_take_selectors((False, skip), (True, take))
        return itbl.compress(selectors)

    @tp.overload
    def chain_from_iter[K1](
        self: Iter[Iterable[K1]], iterable: None = None
    ) -> Iter[K1]: ...

    @tp.overload
    def chain_from_iter[K2](
        self: Iterable[T], iterable: Iterable[Iterable[K2]]
    ) -> Iter[T | K2]: ...

    def chain_from_iter[K1, K2](
        self: Iterable[T] | Iter[Iterable[K1]],
        iterable: Iterable[Iterable[K2]] | None = None,
    ) -> Iter[K1] | Iter[T | K2]:
        """Chain multiple iterables together by flattening them.

        This method has two modes of operation:
        1. When iterable=None: Flattens self when self contains iterables
        2. When iterable is provided: Chains self with the flattened iterable

        Args:
            iterable: Optional iterable of iterables to chain with self.
                     If None, flattens self instead.

        Returns:
            Iter: Iterator over chained/flattened elements

        Example:
            >>> Iter([[1, 2], [3, 4]]).chain_from_iter().to_list()
            [1, 2, 3, 4]
            >>> Iter([1, 2]).chain_from_iter([[3, 4], [5, 6]]).to_list()
            [1, 2, 3, 4, 5, 6]
        """
        if iterable is None:
            return Iter(it.chain.from_iterable(tp.cast(Iter[Iterable[K1]], self)))
        return Iter(it.chain(tp.cast(Iter[T], self), it.chain.from_iterable(iterable)))

    # TODO: consider default
    @tp.overload
    def reduce(
        self: Iterable[T], func: Callable[[T, T], T], initial: T | None = None
    ) -> T: ...
    @tp.overload
    def reduce[I](self: Iterable[T], func: Callable[[I, T], I], initial: I) -> I: ...
    @tp.no_type_check
    def reduce(self: Iterable[T], func, initial=None):
        """Reduce the sequence using a binary function.

        Args:
            func: Binary function to apply cumulatively to the items
            initial: Optional initial value to start the reduction (default: None)

        Returns:
            T: Result of reduction

        Example:
            >>> Iter([1, 2, 3, 4]).reduce(lambda x, y: x + y)
            10
            >>> Iter([1, 2, 3]).reduce(lambda x, y: x * y, initial=10)
            60
            >>> Iter([]).reduce(lambda x, y: x + y, initial=0)
            0
            >>> Iter([]).reduce(lambda x, y: x + y)
            Traceback (most recent call last):
                ...
            TypeError: reduce() of empty iterable with no initial value
        """
        if initial is None:
            return reduce(func, list(self))
        return reduce(func, list(self), initial)

    # fmt: off
    @tp.overload
    def zip[R](self: Iterable[T], other: Iterable[R], *, missing_policy: Ignore = IGNORE) -> Iter[tuple[T, R]]: ...
    @tp.overload
    def zip[R](self: Iterable[T], other: Iterable[R], *, missing_policy: Raise) -> Iter[tuple[T, R]]: ...
    @tp.overload
    def zip[R, F](self: Iterable[T], other: Iterable[R], *, missing_policy: Pad[F]) -> Iter[tuple[T | F, R | F]]: ...
    # fmt: on
    def zip[R, F](
        self: Iterable[T],
        other: Iterable[R],
        *,
        missing_policy: MissingPolicy[F] = IGNORE,
    ) -> Iter[tuple[T, R]] | Iter[tuple[T | F, R | F]]:
        """Zip two iterables together with configurable behavior for unequal lengths.

        This method provides three strategies for handling iterables of unequal length:
        1. Ignore (default): Stop when shortest iterable is exhausted
        2. Raise: Raise an error if iterables have unequal length
        3. Pad: Use a fill value to pad the shorter iterable

        Args:
            other: Second iterable to zip with
            missing_policy: Policy for handling unequal length iterables:
                - Ignore(): Stop when shortest exhausted (default)
                - Raise(): Raise error if lengths unequal
                - Pad(value): Pad shorter with value

        Returns:
            Iter[tuple[T, R]]: Zipped iterator pairs
            Iter[tuple[T | F, R | F]]: Zipped pairs with fill value type when using Pad policy

        Raises:
            ValueError: If iterables have unequal length and if_unequal=Raise()
                or missing_policy receives an unknown MissingPolicy

        Example:
            >>> Iter([1, 2]).zip([3, 4, 5]).to_list()
            [(1, 3), (2, 4)]
            >>> Iter([1]).zip([3, 4], missing_policy=Raise()).to_list()
            Traceback (most recent call last):
                ...
            ValueError: zip() argument 2 is longer than argument 1
            >>> Iter([1, 2]).zip([3], missing_policy=Pad(0)).to_list()
            [(1, 3), (2, 0)]
        """
        match missing_policy:
            case Raise():
                return Iter(zip(self, other, strict=True))
            case Ignore():
                return Iter(zip(self, other, strict=False))
            case Pad(fillvalue=fillvalue):
                return Iter(it.zip_longest(self, other, fillvalue=fillvalue))

        raise ValueError(  # pyright: ignore[reportUnreachable]
            f"{missing_policy=} not recognized. Choices: Raise | Ignore | Pad"
        )

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

    # fmt: off
    @tp.overload
    def zipn[R](self: Iterable[T], *others: Iterable[R], missing_policy: Ignore = IGNORE) -> Iter[tuple[T | R, ...]]: ...
    @tp.overload
    def zipn[R](self: Iterable[T], *others: Iterable[R], missing_policy: Raise) -> Iter[tuple[T | R, ...]]: ...
    @tp.overload
    def zipn[R, F](self: Iterable[T], *others: Iterable[R], missing_policy: Pad[F]) -> Iter[tuple[T | R | F, ...]]: ...
    # fmt: on
    def zipn[R, F](
        self: Iterable[T],
        *others: Iterable[R],
        missing_policy: MissingPolicy[F] = IGNORE,
    ) -> Iter[tuple[T | R, ...]] | Iter[tuple[T | R | F, ...]]:
        """Zip iterables together with configurable behavior for unequal lengths.

        This method provides three strategies for handling iterables of unequal length:
        1. Ignore (default): Stop when shortest iterable is exhausted
        2. Raise: Raise an error if iterables have unequal length
        3. Pad: Use a fill value to pad the shorter iterable

        Args:
            others: Other iterables to zip with
            missing_policy: Policy for handling unequal length iterables:
                - Ignore(): Stop when shortest exhausted (default)
                - Raise(): Raise error if lengths unequal
                - Pad(value): Pad shorter with value

        Returns:
            Iter[tuple[T | R, ...]]: Zipped iterators
            Iter[tuple[T | R | F, ...]]: Zipped iterators with fill value type when using Pad policy

        Raises:
            ValueError: If iterables have unequal length and if_unequal=Raise()
                or missing_policy receives an unknown MissingPolicy

        Example:
            >>> Iter([1, 2]).zipn([3, 4, 5], [10, 20]).to_list()
            [(1, 3, 10), (2, 4, 20)]
            >>> Iter([1]).zipn([3, 4], missing_policy=Raise()).to_list()
            Traceback (most recent call last):
                ...
            ValueError: zip() argument 2 is longer than argument 1
            >>> Iter([1, 2]).zipn([3], missing_policy=Pad(0)).to_list()
            [(1, 3), (2, 0)]
        """
        match missing_policy:
            case Raise():
                return Iter(zip(self, *others, strict=True))
            case Ignore():
                return Iter(zip(self, *others, strict=False))
            case Pad(fillvalue=fillvalue):
                return Iter(it.zip_longest(self, *others, fillvalue=fillvalue))

        raise ValueError(  # pyright: ignore[reportUnreachable]
            f"{missing_policy=} not recognized. Choices: Raise | Ignore | Pad"
        )

    def product3[T2, T3](
        self, it1: Iterable[T2], it2: Iterable[T3]
    ) -> Iter[tuple[T, T2, T3]]:
        """Calculate the cartesian product with two other iterables.

        Args:
            it1: First iterable to calculate product with
            it2: Second iterable to calculate product with

        Returns:
            Iter: Iterator over tuples containing cartesian product

        Example:
            >>> Iter([1]).product3(['a', 'b'], [True, False]).to_list()
            [(1, 'a', True), (1, 'a', False), (1, 'b', True), (1, 'b', False)]
        """
        return Iter(it.product(self, it1, it2))

    def product_n[R](
        self: Iterable[T], *itbl: Iterable[R], repeat: int = 1
    ) -> Iter[tuple[T | R, ...]]:
        """
        see itertools.product

        Returns:
            cartesian product of n iterables
        """
        return Iter(it.product(self, *itbl, repeat=repeat))

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
        self: Iterable[TNumber],
    ) -> stats[TNumber]:
        return stats[TNumber](self)

    def map_type[R](self, type: type[R]) -> Iter[R]:
        """Convert each element to the specified type.

        Args:
            type: The type to convert elements to

        Returns:
            Iter: Iterator with elements converted to the specified type

        Example:
            >>> Iter(['1', '2', '3']).map_type(int).to_list()
            [1, 2, 3]
        """
        return self.map(type)

    def prepend[V](self: Iterable[T], *values: V) -> Iter[T | V]:
        """Prepend values to the beginning of the iterator.

        Args:
            *values: Values to prepend

        Returns:
            Iter: Iterator with values prepended

        Example:
            >>> Iter([3, 4]).prepend(1, 2).to_list()
            [1, 2, 3, 4]
        """
        return Iter(it.chain(values, self))

    def append[V](self: Iterable[T], *values: V) -> Iter[T | V]:
        """Append values to the end of the iterator.

        Args:
            *values: Values to append

        Returns:
            Iter: Iterator with values appended

        Example:
            >>> Iter([1, 2]).append(3, 4).to_list()
            [1, 2, 3, 4]
        """
        return Iter(it.chain(self, values))

    def flatten_once[T1](self: Iterable[Sequence[T1]]) -> Iter[T1]:
        """Flatten one level of nesting in sequences.

        Returns:
            Iter: Iterator with one level of nesting removed

        Example:
            >>> Iter([[1, 2], [3, 4]]).flatten_once().to_list()
            [1, 2, 3, 4]
        """
        return Iter(it.chain.from_iterable(self))

    def flatten(self: Iterable[T]) -> Iter[object]:
        """Recursively flatten nested sequences.

        Returns:
            Iter: Iterator with all nesting removed

        Example:
            >>> Iter([1, [2, [3, 4]], 5]).flatten().to_list()
            [1, 2, 3, 4, 5]
        """
        return Iter(flatten(self))

    def concat[R](
        self, *its: Iterable[R], self_position: tp.Literal["front", "back"] = "front"
    ) -> Iter[T | R]:
        """Concatenate multiple iterables with this iterator.

        Args:
            *its: Iterables to concatenate
            self_position: Where to place this iterator - "front" or "back"

        Returns:
            Iter: Iterator with all iterables concatenated

        Example:
            >>> Iter([1, 2]).concat([3, 4], [5, 6]).to_list()
            [1, 2, 3, 4, 5, 6]
            >>> Iter([1, 2]).concat([3, 4], self_position="back").to_list()
            [3, 4, 1, 2]
        """
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
        """Enumerate the items in the iterator, optionally wrapping them in Indexed objects.

        Args:
            indexed: If True, returns Indexed objects containing index and value.
                    If False, returns (index, value) tuples. (default: True)
            start: The starting value for enumeration (default: 0)

        Returns:
            Iter[tuple[int, T]] if indexed=False: Iterator of (index, value) tuples
            Iter[Indexed[T]] if indexed=True: Iterator of Indexed objects

        Example:
            >>> Iter(['a', 'b', 'c']).enumerate(indexed=True).to_list()
            [Indexed(idx=0, value='a'), Indexed(idx=1, value='b'), Indexed(idx=2, value='c')]
            >>> Iter(['a', 'b', 'c']).enumerate(indexed=False).to_list()
            [(0, 'a'), (1, 'b'), (2, 'c')]
            >>> Iter(['a', 'b', 'c']).enumerate(start=1).to_list()
            [Indexed(idx=1, value='a'), Indexed(idx=2, value='b'), Indexed(idx=3, value='c')]
        """
        enumerated = Iter(enumerate(self._iter, start=start))
        if indexed:
            return enumerated.starmap(Indexed)
        return enumerated

    @tp.overload
    def sliding_window(
        self: Iterator[T],
        n: int,
        *,
        uneven: Raise | Ignore | None = None,
    ) -> Iter[tuple[T, ...]]: ...
    @tp.overload
    def sliding_window[F](
        self: Iterator[T],
        n: int,
        *,
        uneven: Pad[F],
    ) -> Iter[tuple[T | F, ...]]: ...
    def sliding_window[F](
        self: Iterator[T],
        n: int,
        *,
        uneven: MissingPolicy[F] | None = None,
    ) -> Iter[tuple[T, ...]] | Iter[tuple[T | F, ...]]:
        return Iter(sliding_window_iter(self, n, uneven=uneven))

    def product_with[T2](self: Iterable[T], other: Iterable[T2]) -> Iter[tuple[T, T2]]:
        """Calculate the cartesian product with another iterable.

        Args:
            other: Iterable to calculate product with

        Returns:
            Iter: Iterator over tuples containing cartesian product

        Example:
            >>> Iter([1, 2]).product_with(['a', 'b']).to_list()
            [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
        """
        return Iter(it.product(self, other))

    def collect_in[R, **P](
        self,
        container: Callable[tp.Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """
        Similar to feed_into.

        Args:
            container: A callable that takes self as a whole and stores in a container
            *args: args for the callable (if any)
            **kwargs: kwargs for the callable (if any)

        Returns:
            container containing elements of the iterable.
        """
        return container(self, *args, **kwargs)

    def inspect[**P](
        self,
        func: Callable[tp.Concatenate[T, P], object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iter[T]:
        """
        Applies a function `func` over all the elements, and returns the original self.

        Args:
            func (Callable): function to apply on each element.
            *args (object): arguments to the function
            **kwargs (object): keyword arguments to the function

        Returns:
            Iter[T]
        """

        def apply(itbl: Iter[T]) -> Generator[T]:
            for item in itbl:
                _ = func(item, *args, **kwargs)
                yield item

        return Iter(apply(self))

    def groupby_true(
        self: Iterable[T], key: Callable[[T], Hashable]
    ) -> Iter[Iterable[T]]:
        """
        Given a predicate as key, return groups of consecutive items where key evaluates to True

        Args:
            key (callable): a predicate that is to be applied to items.

        Returns:
            Iter: consecutive iterable groups of items where predicate was true.

        Example:
            >>> Iter("...123..4..*&234").groupby_true(str.isdigit).map("".join).to_list()
            ['123', '4', '234']
        """
        return Iter(item[1] for item in it.groupby(self, key) if item[0])

    def groupby_false(
        self: Iterable[T], key: Callable[[T], Hashable]
    ) -> Iter[Iterable[T]]:
        """
        Given a predicate as key, return groups of consecutive items where key evaluates to False

        Args:
            key (callable): a predicate that is to be applied to items.

        Returns:
            Iter: consecutive iterable groups of items where predicate was false.

        Example:
            >>> Iter("...123..4..*&234").groupby_false(str.isdigit).map("".join).to_list()
            ['...', '..', '..*&']
        """
        return Iter(item[1] for item in it.groupby(self, key) if not item[0])

    def groupby(
        self: Iterable[T], key: Callable[[T], Hashable]
    ) -> Iter[tuple[Hashable, Iterable[T]]]:
        """See itertools.groupby

        Returns:
            Iter of tuples, containing what the predicate evaluated to for that group, and the group
        """
        return Iter(it.groupby(self, key))

    def filter_map[R](
        self: Iterable[T], predicate_apply: Callable[[T], tp.Literal[False] | R]
    ) -> Iter[R]:
        """
        Map on filtered elements.

        Args:
            predicate_apply (callable): A callable that either evaluates to False or returns a value

        Returns:
            Iter[R]

        Example:
            >>> Iter([1, 2, 3, 4, 5, 6]).filter_map(lambda x: x % 2 == 0 and x ** 2).to_list()
            [4, 16, 36]
            >>> Iter([1, 2, 3, 4, 5, 6]).filter_map(lambda x: x % 2 == x % 3 and str(x)).to_list()
            ['1', '6']
        """

        def apply() -> Generator[R]:
            for item in self:
                if res := predicate_apply(item):
                    yield res

        return Iter(apply())

    def is_empty(self) -> bool:
        """
        Check if the sequence Iter is empty

        Returns:
            True if empty, else False

        Example:
            >>> it = Iter([1, 2, 3])
            >>> it.is_empty()
            False
            >>> it.to_list()
            [1, 2, 3]
        """
        return self.peek_next_value() is Exhausted

    def flatmap[**P, R](
        self: Iterable[T],
        func: Callable[tp.Concatenate[T, P], Sequence[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iter[R]:
        """Map a function that returns sequences and flatten the results.

        This is equivalent to mapping the function and then calling flatten_once().

        Args:
            func: Function that takes an item and returns a sequence
            *args: Additional positional arguments for func
            **kwargs: Additional keyword arguments for func

        Returns:
            Iter[R]: Iterator over the flattened results

        Example:
            >>> Iter([1, 2]).flatmap(lambda x: [x, x*2]).to_list()
            [1, 2, 2, 4]
            >>> Iter(['ab', 'cd']).flatmap(list).to_list()
            ['a', 'b', 'c', 'd']
        """
        return Iter(it.chain.from_iterable(func(item, *args, **kwargs) for item in self))

    def cycle(self: Iterable[T]) -> Iter[T]:
        """Create an iterator that cycles through the elements infinitely.

        Returns:
            Iter[T]: Iterator that repeats elements indefinitely

        Example:
            >>> Iter([1, 2, 3]).cycle().slice(stop=5).to_list()
            [1, 2, 3, 1, 2]
        """
        return Iter(it.cycle(self))

    def partition(
        self: Iterable[T], predicate: Predicate[T]
    ) -> tuple[SeqIter[T], SeqIter[T]]:
        results = defaultdict[bool, list[T]](list)
        for key, group in it.groupby(self, predicate):
            results[bool(key)].extend(group)
        return SeqIter(results[True]), SeqIter(results[False])


@tp.final
class SeqIter[T](Sequence[T]):
    def __init__(self, iterable: Iterable[T] = ()) -> None:
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
    def __eq__(self, value: object, /) -> bool:
        value = tp.cast(SeqIter[T], value)
        return isinstance(value, SeqIter) and self._iterable == value._iterable  # pyright: ignore[reportUnnecessaryIsInstance]

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
            return enumerated.map(Unpacked(Indexed))
        return enumerated

    @tp.overload
    def max[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def max[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F: ...
    def max[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F:
        return (
            max(self, key=key)
            if default is Default.NoDefault
            else max(self, key=key, default=default)
        )

    @tp.overload
    def min[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def min[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F: ...
    def min[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F:
        return (
            min(self, key=key)
            if default is Default.NoDefault
            else min(self, key=key, default=default)
        )

    @tp.overload
    def minmax[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> MinMax[TComparable]: ...
    @tp.overload
    def minmax[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = Default.NoDefault,
    ) -> MinMax[TComparable] | MinMax[F]: ...
    def minmax[TComparable: Comparable, F](
        self: SeqIter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
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
        """Convert this SeqIter into an Iter.

        Returns:
            Iter[T]: A new Iter containing the same elements

        Example:
            >>> SeqIter([1, 2, 3]).iter().to_list()
            [1, 2, 3]
        """
        return Iter(self.iterable)

    def map_partial[R, **P](
        self,
        func: Callable[tp.Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> SeqIter[R]:
        """Map a function with partial arguments over each element.

        Args:
            func: Callable to apply to each element
            *args: Positional arguments to pass to func after each element
            **kwargs: Keyword arguments to pass to func

        Returns:
            SeqIter[R]: New SeqIter with mapped values

        Example:
            >>> SeqIter(["1", "10", "11"]).map_partial(int, base=2).to_list()
            [1, 2, 3]
        """
        return SeqIter(func(item, *args, **kwargs) for item in self)

    @tp.overload
    def map[R](
        self,
        func: Callable[[T], R],
    ) -> SeqIter[R]:
        """Map a single-argument function over the sequence.

        Args:
            func: Function taking one argument to apply to each element

        Returns:
            SeqIter[R]: New sequence with function applied

        Example:
            >>> SeqIter([1, 2, 3]).map(str).to_list()
            ['1', '2', '3']
        """
        ...

    @tp.overload
    def map[K, _, R](
        self,
        func: Callable[tp.Concatenate[T, _, ...], R],
        *args: Iterable[K],
    ) -> SeqIter[R]:
        """Map a multi-argument function over multiple sequences in parallel.

        Args:
            func: Function taking multiple arguments
            *args: Additional iterables to map over in parallel

        Returns:
            SeqIter[R]: New sequence with function applied

        Example:
            >>> SeqIter([1, 2]).map(pow, [2, 3]).to_list()
            [1, 8]
        """
        ...

    def map[K, R](
        self,
        func: Callable[tp.Concatenate[T, ...], R],
        *args: Iterable[K],
    ) -> SeqIter[R]:
        """Map a function over the sequence.

        This is a unified implementation for both single and multi-argument mapping.
        See the overload signatures for specific documentation.
        """
        return SeqIter(map(func, self, *args))  # noqa: DOC201

    def all(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """
        Check if all elements are Truthy or if given a prediate, check
        if all element evaluate to True when the predicate is applied.


        If self is empty, return True.

        Args:
            predicate (optional callable): if given, check if predicate is Truthy
                default, None.

        Returns:
            bool: Whether all the items were Truthy.

        Example:
            >>> SeqIter([True] * 5).all()
            True
            >>> SeqIter(range(5)).all(lambda x: x < 5)
            True
            >>> SeqIter(range(5)).all(lambda x: x%2 == 0)
            False
            >>> SeqIter(()).all()
            True
        """
        if predicate is None:
            return all(self)
        return self.map(predicate).all()

    def any(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """
        Check if any elements is Truthy or if given a prediate, check
        if any element evaluates to True when the predicate is applied.

        If self is empty, return False.

        Args:
            predicate (optional callable): if given, check if predicate is Truthy
                default, None.

        Returns:
            bool: Whether any of the items were Truthy.

        Example:
            >>> SeqIter([True] * 5).any()
            True
            >>> SeqIter(range(5)).any(lambda x: x < 5)
            True
            >>> SeqIter(range(5)).any(lambda x: x%2 == 0)
            True
            >>> SeqIter(()).any()
            False
        """
        if predicate is None:
            return any(self)
        return self.map(predicate).any()

    def collect_in[R, **P](
        self,
        container: Callable[tp.Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Collect elements into a container using the provided function.

        Args:
            container: Function that takes an iterable and returns a container
            *args: Additional positional arguments for func
            **kwargs: Additional keyword arguments for func

        Returns:
            R: Result of applying func to the sequence

        Example:
            >>> SeqIter([1, 2, 3]).collect_in(list)
            [1, 2, 3]
            >>> SeqIter([1, 2, 3]).collect_in(sum)
            6
        """
        return container(self, *args, **kwargs)

    @tp.overload
    def first[TDefault](
        self, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def first[TDefault](self, default: TDefault) -> T | TDefault: ...
    def first[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        """Return the first element of the sequence.

        Args:
            default: Value to return if sequence is empty.
                    If NoDefault, raises IndexError for empty sequence.
                    Default is Exhausted.

        Returns:
            T | TDefault: First element or default value

        Raises:
            IndexError: If sequence is empty and default is NoDefault

        Example:
            >>> SeqIter([1, 2, 3]).first()
            1
            >>> SeqIter([]).first(default=0)
            0
            >>> SeqIter([]).first(default=NoDefault)
            Traceback (most recent call last):
                ...
            IndexError: ...
        """
        with suppress(IndexError):
            return self[0]
        if default is not NoDefault:
            return default
        raise IndexError("iterable is empty, and no default specified for first.")

    # TODO: support getitem
    @tp.overload
    def at[TDefault](
        self, n: int, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def at[TDefault](self, n: int, default: TDefault) -> T | TDefault: ...
    def at[TDefault](self, n: int, default: TDefault = Exhausted) -> T | TDefault:
        """Return the element at index n.

        Args:
            n: Index of element to return
            default: Value to return if index is out of bounds.
                    If NoDefault, raises IndexError for invalid index.
                    Default is Exhausted.

        Returns:
            T | TDefault: Element at index n or default value

        Raises:
            IndexError: If index is out of bounds and default is NoDefault

        Example:
            >>> SeqIter([1, 2, 3]).at(1)
            2
            >>> SeqIter([1, 2, 3]).at(5, default=0)
            0
            >>> SeqIter([1, 2, 3]).at(5, default=NoDefault)
            Traceback (most recent call last):
                ...
            IndexError: ...
        """
        with suppress(IndexError):
            return self[n]
        if default is not NoDefault:
            return default
        raise IndexError(f"index {n} is out of range for SeqIter of length {len(self)}")

    @tp.overload
    def last[TDefault](self, default: tp.Literal[Default.NoDefault] = NoDefault) -> T: ...
    @tp.overload
    def last[TDefault](self, default: TDefault) -> T | TDefault: ...
    def last[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        """Return the last element of the sequence.

        Args:
            default: Value to return if sequence is empty.
                    If NoDefault, raises IndexError for empty sequence.
                    Default is Exhausted.

        Returns:
            T | TDefault: Last element or default value

        Raises:
            IndexError: If sequence is empty and default is NoDefault

        Example:
            >>> SeqIter([1, 2, 3]).last()
            3
            >>> SeqIter([]).last(default=0)
            0
            >>> SeqIter([]).last(default=NoDefault)
            Traceback (most recent call last):
                ...
            IndexError: ...
        """
        with suppress(IndexError):
            return self[-1]
        if default is not NoDefault:
            return default
        raise IndexError("iterable is empty, and no default specified for last.")

    def tail(self, n: int) -> SeqIter[T]:
        """Return the last n elements.

        Args:
            n: Number of elements to return from the end

        Returns:
            SeqIter[T]: Last n elements

        Example:
            >>> SeqIter([1, 2, 3, 4]).tail(2).to_list()
            [3, 4]
        """
        return SeqIter(self[-n:])

    def skip(self, n: int) -> SeqIter[T]:
        """Skip the first n elements.

        Args:
            n: Number of elements to skip from the start

        Returns:
            SeqIter[T]: Remaining elements after skipping

        Example:
            >>> SeqIter([1, 2, 3, 4]).skip(2).to_list()
            [3, 4]
        """
        return SeqIter(self[n:])

    def slice(
        self, *, start: int | None = 0, stop: int | None = None, step: int | None = 1
    ) -> SeqIter[T]:
        """Slice the sequence with start:stop:step.

        Args:
            start: Starting index (default: 0)
            stop: Ending index (default: None)
            step: Step size (default: 1)

        Returns:
            SeqIter[T]: Sliced sequence

        Example:
            >>> SeqIter(range(10)).slice(start=2, stop=8, step=2).to_list()
            [2, 4, 6]
        """
        return SeqIter(self.iterable[start:stop:step])

    @staticmethod
    def _get_skip_take_selectors(
        s1: tuple[bool, int], s2: tuple[bool, int]
    ) -> Iterator[bool]:
        while True:
            yield from it.repeat(*s1)
            yield from it.repeat(*s2)

    @skip_take_by_order
    def skip_take(self, *, skip: int, take: int, take_first: bool = False) -> SeqIter[T]:
        """
        skip some elements followed by take some elements, or vice versa.

        Whether to take first, or skip first, can be specified in two ways.
        1. Via `take_first` boolean flag.
        2. Via order of the arguments, that is, skip=s, take=t will skip first,
        whereas, take=t, skip=s, will take first.

        Args:
            skip (int): number of elements to skip.
            take (int): number of elements to take.
            take_first (Optional[bool]): If True, take first, else skip first.
                default, False.

        Returns:
            Self

        Examples:
            >>> SeqIter(range(10)).skip_take(skip=2, take=3).to_list()
            [2, 3, 4, 7, 8, 9]
            >>> SeqIter(range(10)).skip_take(take=3, skip=2).to_list()
            [0, 1, 2, 5, 6, 7]
        """
        if take_first:
            selectors = self._get_skip_take_selectors((True, take), (False, skip))
        else:
            selectors = self._get_skip_take_selectors((False, skip), (True, take))
        return SeqIter(it.compress(self, selectors))

    # TODO: consider default
    def reduce(self, func: Callable[[T, T], T], initial: T | None = None) -> T:
        """Reduce the sequence using a binary function.

        Args:
            func: Binary function to apply cumulatively
            initial: Optional initial value (default: None)

        Returns:
            T: Result of reduction

        Example:
            >>> SeqIter([1, 2, 3, 4]).reduce(lambda x, y: x + y)
            10
            >>> SeqIter([1, 2, 3]).reduce(lambda x, y: x * y, initial=10)
            60
        """
        if initial is None:
            return reduce(func, self)
        return reduce(func, self, initial)

    def transpose[TSized: Sized](
        self: Sequence[TSized],
        *,
        strict: bool = False,
    ) -> SeqIter[TSized]:
        """Transpose rows and columns of nested sequences.

        Args:
            strict: If True, raise ValueError if sequences have different lengths (default: False)

        Returns:
            SeqIter[TSized]: Transposed sequences

        Example:
            >>> SeqIter([(1, 2), (3, 4)]).transpose().to_list()
            [(1, 3), (2, 4)]
        """
        return SeqIter(zip(*self, strict=strict))

    @property
    def stats[TNumber: (float, Decimal, Fraction) = float](
        self: SeqIter[TNumber],
    ) -> stats[TNumber]:
        return stats[TNumber](self)

    def prepend[V](self, *values: V) -> SeqIter[T | V]:
        """Add values to the beginning of the sequence.

        Args:
            *values: Values to prepend

        Returns:
            SeqIter[T | V]: New sequence with values prepended

        Example:
            >>> SeqIter([3, 4]).prepend(1, 2).to_list()
            [1, 2, 3, 4]
        """
        return SeqIter(values + self.iterable)

    def append[V](self, *values: V) -> SeqIter[T | V]:
        """Add values to the end of the sequence.

        Args:
            *values: Values to append

        Returns:
            SeqIter[T | V]: New sequence with values appended

        Example:
            >>> SeqIter([1, 2]).append(3, 4).to_list()
            [1, 2, 3, 4]
        """
        return SeqIter(self.iterable + values)

    def concat[R](
        self, *its: Iterable[R], self_position: tp.Literal["front", "back"] = "front"
    ) -> SeqIter[T | R]:
        """Concatenate multiple iterables with this sequence.

        Args:
            *its: Iterables to concatenate
            self_position: Where to place this sequence - "front" or "back" (default: "front")

        Returns:
            SeqIter[T | R]: New sequence with all iterables concatenated

        Raises:
            ValueError: If self_position is not "front" or "back"

        Example:
            >>> SeqIter([1, 2]).concat([3, 4], [5, 6]).to_list()
            [1, 2, 3, 4, 5, 6]
            >>> SeqIter([1, 2]).concat([3, 4], self_position="back").to_list()
            [3, 4, 1, 2]
        """
        if self_position not in ["front", "back"]:
            raise ValueError(
                f'`self_position` must be "front" or "back", got {self_position!r}'
            )

        result: tuple[R] = reduce(add, map(tuple[R], its), _initial := tuple[R]())
        match self_position:
            case "front":
                return SeqIter(self.iterable + result)
            case "back":
                return SeqIter(result + self.iterable)

    @tp.overload
    def sliding_window(
        self,
        n: int,
        *,
        uneven: Raise | Ignore | None = None,
    ) -> SeqIter[tuple[T, ...]]: ...
    @tp.overload
    def sliding_window[F](
        self,
        n: int,
        *,
        uneven: Pad[F],
    ) -> SeqIter[tuple[T | F, ...]]: ...
    def sliding_window[F](
        self,
        n: int,
        *,
        uneven: MissingPolicy[F] | None = None,
    ) -> SeqIter[tuple[T, ...]] | SeqIter[tuple[T | F, ...]]:
        return SeqIter(sliding_window_seq(self, n, uneven=uneven))

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
            >>> SeqIter([("a", 0), ("b", 1)]).feed_into(dict)
            {'a': 0, 'b': 1}
            >>> SeqIter([1, 2, 3, 4]).feed_into(sum, start=10)
            20
        """
        return func(self, *args, **kwargs)

    def to_list(self) -> list[T]:
        """Convert the sequence to a list.

        Returns:
            list[T]: List containing all elements

        Example:
            >>> SeqIter([1, 2, 3]).to_list()
            [1, 2, 3]
        """
        return list(self.iterable)

    def inspect[**P](
        self,
        func: Callable[tp.Concatenate[T, P], object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tp.Self:
        """
        Applies a function `func` over all the elements, and returns the original self.

        Args:
            func (Callable): function to apply on each element.
            *args (object): arguments to the function
            **kwargs (object): keyword arguments to the function

        Returns:
            Self
        """
        _ = self.map_partial(func, *args, **kwargs)
        del _
        return self

    @tp.overload
    def find(
        self,
        finder: Callable[[T], bool] | T,
        default: tp.Literal[Default.NoDefault] = NoDefault,
    ) -> Indexed[T]: ...
    @tp.overload
    def find[F](
        self,
        finder: Callable[[T], bool] | T,
        default: F,
    ) -> Indexed[T] | F: ...
    def find[F](
        self,
        finder: Callable[[T], bool] | T,
        default: F = NoDefault,
    ) -> Indexed[T] | F:
        """
        Find a value and its index in the iterable, either by passing the value to find or a callable

        Args:
            finder (T | callable): Either the value to find, or a callable that returns True for the item
            default: Default to return if not found. If not given or default is Default.NoDefault it
                raises error if not found. (default: Default.NoDefault)

        Returns:
            If found returns Indexed object, otherwise return default if given.

        Example:
            >>> SeqIter([1, 2, 3, 4]).find(2)
            Indexed(idx=1, value=2)
            >>> SeqIter([1, 2, 3, 4]).find(lambda x: x**2 == 4)
            Indexed(idx=1, value=2)
            >>> SeqIter([1, 2, 3, 4]).find(5, default=Default.Unavailable)
            <Default.Unavailable: 3>
        """
        if callable(finder):
            finder = tp.cast(Callable[[T], bool], finder)
            return self.iter().enumerate().filter(lambda x: finder(x.value)).next(default)
        return self.iter().enumerate().filter(IsEqual(finder)).next(default)

    def len(self) -> int:
        """Get the length of the sequence.

        Returns:
            int: Number of elements

        Example:
            >>> SeqIter([1, 2, 3]).len()
            3
        """
        return len(self.iterable)

    def sorted[TComp: Comparable, RComp: Comparable](
        self: SeqIter[TComp],
        *,
        reverse: bool = False,
        key: Callable[[TComp], RComp] | None = None,
    ) -> SeqIter[TComp]:
        """Return a new sorted sequence.

        Args:
            reverse: If True, sort in descending order (default: False)
            key: Function to extract comparison key (default: None)

        Returns:
            SeqIter[TComp]: New sorted sequence

        Example:
            >>> SeqIter([3, 1, 4, 2]).sorted().to_list()
            [1, 2, 3, 4]
            >>> SeqIter([3, 1, 4, 2]).sorted(reverse=True).to_list()
            [4, 3, 2, 1]
        """
        return SeqIter(sorted(self, reverse=reverse, key=key))

    def debug(self) -> SeqIter[T]:
        """Add a breakpoint on each iteration.

        Returns:
            SeqIter[T]: Original sequence unchanged
        """

        def inner() -> Generator[T]:
            for item in self:
                breakpoint()
                yield item

        return SeqIter(inner())

    def is_empty(self) -> bool:
        """
        Check if the sequence Iter is empty

        Returns:
            True if empty, else False

        Example:
            >>> SeqIter([1, 2, 3]).is_empty()
            False
        """
        return not bool(self.iterable)

    @tp.override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(iterable={self.iterable!r})"

    @tp.override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(iterable={self.iterable})"

    repr = __repr__
    str = __str__


if __name__ == "__main__":
    from doctest import testmod
    from pathlib import Path
    from textwrap import dedent

    dummy_file = Path("./dummy.config").resolve()
    content = dedent(
        """
        python-version = 3.13
        configpath = 'path/to/config'

        PORT = 8000
        """
    )
    if not dummy_file.is_file():
        _ = dummy_file.write_text(content)

    # fmt: off
    with dummy_file.open() as file_handle:
        key_val_tuples = (
            Iter(file_handle)
            .map(str.strip)
            .filter(None)
            .map_partial(str.split, sep=" = ", maxsplit=1)
            .collect()
        )

        config = key_val_tuples.collect_in(dict)
        keys, vals = key_val_tuples.transpose()

    player_scores = [12, 55, 89, 82, 37, 16]
    qualified = (
        Iter(player_scores)
        .filter(lambda x: x > 35)
        .collect[SeqIter[int]]()
    )

    minmax_info = qualified.iter().enumerate().collect().minmax()
    statistics = qualified.stats()
    # fmt: on

    print(f"config: {config}")
    print(f"Winner => player_id: {minmax_info.max.idx}, score: {minmax_info.max.value}")
    print(f"Loser  => player_id: {minmax_info.min.idx}, score: {minmax_info.min.value}")
    print(f"Player Stats: {statistics}")

    _ = testmod()
