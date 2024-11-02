# pyright: reportImportCycles=false
from __future__ import annotations

import enum
import itertools as it
import typing as tp
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence, Sized
from decimal import Decimal
from fractions import Fraction
from functools import partial, reduce, wraps
from operator import add, attrgetter

from iterdot._helpers import flatten, prepend, skip_take_by_order, sliding_window
from iterdot.index import Indexed
from iterdot.minmax import MinMax, lazy_minmax, lazy_minmax_keyed
from iterdot.plugins.stats import stats
from iterdot.wtyping import Comparable


class Default(enum.Enum):
    """Sentinel values used as defaults."""

    Exhausted = enum.auto()
    NoDefault = enum.auto()
    Unavailable = enum.auto()


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
    zip_with = MethodKind[T].augmentor(zip)
    """see zip"""
    takewhile = MethodKind[T].predicated_augmentor(it.takewhile)
    """see itertools.takewhile"""
    dropwhile = MethodKind[T].predicated_augmentor(it.dropwhile)
    """see itertools.dropwhile"""

    def accumulate(
        self, func: Callable[[T, T], T], *, initial: T | None = None
    ) -> Iter[T]:
        """
        see itertools.accumulate

        Returns:
            Iter
        """
        return Iter(it.accumulate(self, func, initial=initial))

    def slice(
        self, *, start: int | None = 0, stop: int | None = None, step: int | None = 1
    ) -> Iter[T]:
        """
        see itertools.islice

        Returns:
            Iter
        """
        return Iter(it.islice(self, start, stop, step))

    sum = MethodKind[T].consumer(sum)
    """see sum"""
    to_list = MethodKind[T].consumer(list)
    """convert to list"""

    @property
    def collect(self) -> Collector[T]:
        return Collector[T](self)

    @tp.overload
    def max[TComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def max[TComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = NoDefault,
    ) -> TComparable | F: ...
    def max[TComparable: Comparable, F](
        self: Iter[TComparable],
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
        self: Iter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> TComparable: ...
    @tp.overload
    def min[TComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = Default.NoDefault,
    ) -> TComparable | F: ...
    def min[TComparable: Comparable, F](
        self: Iter[TComparable],
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
        self: Iter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: tp.Literal[Default.NoDefault] = Default.NoDefault,
    ) -> MinMax[TComparable]: ...
    @tp.overload
    def minmax_eager[TComparable: Comparable, F](
        self: Iter[TComparable],
        key: Callable[[TComparable], Comparable] | None = None,
        default: F = Default.NoDefault,
    ) -> MinMax[TComparable] | MinMax[F]: ...
    def minmax_eager[TComparable: Comparable, F](
        self: Iter[TComparable],
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
        self,
        /,
        *,
        key: Callable[[T], Comparable],
    ) -> MinMax[T]: ...
    @tp.overload
    def minmax_lazy[TComparable: Comparable](
        self: Iter[TComparable],
        /,
        *,
        key: None = None,
    ) -> MinMax[TComparable]: ...
    @tp.overload
    def minmax_lazy[F](
        self,
        /,
        *,
        key: Callable[[T], Comparable],
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
            >>> Iter([True] * 5).all()
            True
            >>> Iter(range(5)).all(lambda x: x < 5)
            True
            >>> Iter(range(5)).all(lambda x: x%2 == 0)
            False
            >>> Iter(()).all()
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
            >>> Iter([True] * 5).any()
            True
            >>> Iter(range(5)).any(lambda x: x < 5)
            True
            >>> Iter(range(5)).any(lambda x: x%2 == 0)
            True
            >>> Iter(()).any()
            False
        """
        if predicate is None:
            return any(self)
        return self.map(predicate).any()

    def all_equal(self) -> bool:
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
        first = self.first()
        if first is Exhausted:
            return True
        return all(first == item for item in self)

    def all_equal_with(self, value: T | None = None) -> bool:
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
        if self.first() == Default.Exhausted != value:
            return False
        return all(value == item for item in self)

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
            >>> Iter([1, 2, 3, 4]).feed_into(sum)
            10
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
        return self.slice(stop=n)

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
        return self if not n else self.slice(start=n, stop=None)

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

    @skip_take_by_order
    def skip_take(self, *, skip: int, take: int, take_first: bool = False) -> Iter[T]:
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

    def flatten_once[T1](self: Iter[Sequence[T1]]) -> Iter[T1]:
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

    # TODO: Consider complex padding: +ve int for left, -ve int for right, complex for left right
    @tp.overload
    def sliding_window(
        self, n: int, *, fill_value: tp.Literal[Default.NoDefault] = NoDefault
    ) -> Iter[tuple[T, ...]]: ...
    @tp.overload
    def sliding_window[F](
        self,
        n: int,
        *,
        fill_value: F,
        pad: tp.Literal["left", "right", "both"] = "left",
    ) -> Iter[tuple[T | F, ...]]: ...
    def sliding_window[F](
        self,
        n: int,
        *,
        fill_value: F | tp.Literal[Default.NoDefault] = NoDefault,
        pad: tp.Literal["left", "right", "both"] = "both",
    ) -> Iter[tuple[T, ...]] | Iter[tuple[T | F, ...]]:
        """
        Sliding window over self.

        Args:
            n (int): Size of the window
            fill_value (object, optional): If present, used for padding.
            pad (str): 'left', 'right', or 'both'; indicating which side should be padded.

        Returns:
            Iter: Iter of n-tuples

        Example:
            >>> itbl = [1, 2, 3]
            >>> Iter(itbl).sliding_window(2).to_list()
            [(1, 2), (2, 3)]
            >>> Iter(itbl).sliding_window(2, fill_value=0).to_list()
            [(0, 1), (1, 2), (2, 3), (3, 0)]
            >>> Iter(itbl).sliding_window(2, fill_value=0, pad="left").to_list()
            [(0, 1), (1, 2), (2, 3)]
            >>> Iter(itbl).sliding_window(2, fill_value=0, pad="right").to_list()
            [(1, 2), (2, 3), (3, 0)]
            >>> Iter(itbl).sliding_window(5).to_list()
            [(1, 2, 3)]
            >>> Iter(itbl).sliding_window(5, fill_value=0, pad="left").to_list()
            [(0, 0, 0, 0, 1), (0, 0, 0, 1, 2), (0, 0, 1, 2, 3)]
            >>> Iter([1]).sliding_window(3, fill_value=0).to_list()
            [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        """
        if fill_value is NoDefault:
            return Iter(sliding_window(self, n))
        if self.peek_next_value() is Exhausted:
            return Iter[tuple[T, ...]]()

        fill = partial(it.repeat, fill_value, n - 1)
        padding = ["left", "right"] if pad == "both" else [pad]
        if "left" in padding:
            self = self.concat(fill(), self_position="back")
        if "right" in padding:
            self = self.concat(fill())
        return self.sliding_window(n)

    def product_with[T2](self, other: Iterable[T2]) -> Iter[tuple[T, T2]]:
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

    def product3[T2, T3](
        self, it1: Iterable[T2], it2: Iterable[T3]
    ) -> Iter[tuple[T, T2, T3]]:
        return Iter(it.product(self, it1, it2))


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
        func: Callable[tp.Concatenate[Iterable[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(self, *args, **kwargs)

    @tp.overload
    def first[TDefault](
        self, default: tp.Literal[Default.NoDefault] = NoDefault
    ) -> T: ...
    @tp.overload
    def first[TDefault](self, default: TDefault) -> T | TDefault: ...
    def first[TDefault](self, default: TDefault = Exhausted) -> T | TDefault:
        if self:
            return self[0]
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
        if self:
            return self[-1]
        return self[-1] if default is NoDefault else default

    def tail(self, n: int) -> SeqIter[T]:
        return SeqIter(self[-n:])

    def skip(self, n: int) -> SeqIter[T]:
        return SeqIter(self[n:])

    def slice(
        self, *, start: int | None = 0, stop: int | None = None, step: int | None = 1
    ) -> SeqIter[T]:
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

        result: tuple[R] = reduce(add, map(tuple[R], its), _initial := tuple[R]())
        match self_position:
            case "front":
                return SeqIter(self.iterable + result)
            case "back":
                return SeqIter(result + self.iterable)

    # TODO: Consider complex padding: +ve int for left, -ve int for right, complex for left right
    # TODO: Consider left pad, right pad
    @tp.overload
    def sliding_window(
        self, n: int, *, fill_value: tp.Literal[Default.NoDefault] = NoDefault
    ) -> SeqIter[tuple[T, ...]]: ...
    @tp.overload
    def sliding_window[F](
        self,
        n: int,
        *,
        fill_value: F,
        pad: tp.Literal["left", "right", "both"] = "left",
    ) -> SeqIter[tuple[T | F, ...]]: ...
    def sliding_window[F](
        self,
        n: int,
        *,
        fill_value: F | tp.Literal[Default.NoDefault] = NoDefault,
        pad: tp.Literal["left", "right", "both"] = "both",
    ) -> SeqIter[tuple[T, ...]] | SeqIter[tuple[T | F, ...]]:
        """
        Sliding window over self.

        Args:
            n (int): Size of the window
            fill_value (object, optional): If present, used for padding.
            pad (str): 'left', 'right', or 'both'; indicating which side should be padded.

        Returns:
            Iter: Iter of n-tuples

        Example:
            >>> itbl = [1, 2, 3]
            >>> SeqIter(itbl).sliding_window(2).to_list()
            [(1, 2), (2, 3)]
            >>> SeqIter(itbl).sliding_window(2, fill_value=0).to_list()
            [(0, 1), (1, 2), (2, 3), (3, 0)]
            >>> SeqIter(itbl).sliding_window(2, fill_value=0, pad="left").to_list()
            [(0, 1), (1, 2), (2, 3)]
            >>> SeqIter(itbl).sliding_window(2, fill_value=0, pad="right").to_list()
            [(1, 2), (2, 3), (3, 0)]
            >>> SeqIter(itbl).sliding_window(5).to_list()
            [(1, 2, 3)]
            >>> SeqIter(itbl).sliding_window(5, fill_value=0, pad="left").to_list()
            [(0, 0, 0, 0, 1), (0, 0, 0, 1, 2), (0, 0, 1, 2, 3)]
            >>> SeqIter([1]).sliding_window(3, fill_value=0).to_list()
            [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        """
        if fill_value is NoDefault:
            return SeqIter(sliding_window(self.iterable, n))
        if self.first(default=Exhausted) is Exhausted:
            return SeqIter[tuple[T, ...]]()

        fill = (fill_value,) * (n - 1)
        padding = ["left", "right"] if pad == "both" else [pad]
        if "left" in padding:
            self = SeqIter[T | F](fill + self.iterable)
        if "right" in padding:
            self = SeqIter[T | F](self.iterable + fill)
        return self.sliding_window(n)

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
        return list(self.iterable)

    def len(self) -> int:
        return len(self.iterable)

    def sorted[TComp: Comparable, RComp: Comparable](
        self: SeqIter[TComp],
        *,
        reverse: bool = False,
        key: Callable[[TComp], RComp] | None = None,
    ) -> SeqIter[TComp]:
        return SeqIter(sorted(self, reverse=reverse, key=key))

    def inspect(self, func: Callable[[T], object], *, debug: bool = False) -> SeqIter[T]:
        def inner() -> Generator[T]:
            for item in self:
                if debug:
                    breakpoint()
                _ = func(item)
                del _
                yield item

        return SeqIter(inner())

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
