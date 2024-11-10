from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import wraps
from itertools import chain, islice
from operator import call
from types import NoneType
from typing import cast

from iterdot.defaults import Ignore, MissingPolicy, Pad, Raise


def prepend[T](*val: T, to: Iterator[T]) -> Iterator[T]:
    return chain(val, to)


def flatten(iterable: Iterable[object]) -> Iterable[object]:
    for item in iterable:
        if isinstance(item, Iterable) and not isinstance(item, str):
            yield from flatten(cast(Iterable[object], item))
        else:
            yield item


consume = deque[object](maxlen=0).extend


def skip_take_by_order[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if "take_first" in kwargs:
            return func(*args, **kwargs)
        for key in kwargs:
            match key:
                case "skip" | "take":
                    kwargs["take_first"] = key == "take"
                    break
                case unexpected_kwarg:
                    raise ValueError(f"Unexpected keyword argument: {unexpected_kwarg}")
        return func(*args, **kwargs)

    return wrapper


def sliding_window[T](it: Iterator[T] | Sequence[T], n: int) -> Iterable[tuple[T, ...]]:
    window = deque[T](maxlen=n)
    if isinstance(it, Iterator):
        nitems = islice(it, n)
        window.extend(nitems)
        yield tuple(window)
        for item in it:
            window.append(item)
            yield tuple(window)
    else:
        length = len(it)
        for i in range(length):
            yield tuple(it[i : i + n])
            if length - i - 1 < n:
                break


def sliding_window_iter[T, F](
    it: Iterator[T],
    n: int,
    *,
    stride: int = 1,
    uneven: MissingPolicy[F] | None = None,
) -> Iterator[tuple[T, ...]] | Iterator[tuple[T | F, ...]]:
    match uneven:
        case Pad(fillvalue=pad):
            yield from (
                item if (il := len(item)) == n else (item + (pad,) * (n - il))
                for item in sliding_window_iter(it, n, stride=stride, uneven=None)
            )
        case Ignore():
            window = deque[T | F](last := tuple(islice(it, n)), maxlen=n)
            if not window:
                return

            while last:
                yield tuple(window)
                last = tuple(islice(it, stride))
                if len(last) < stride:
                    return
                window.extend(last)
        case Raise():
            for window in sliding_window_iter(it, n, stride=stride, uneven=None):
                if len(window) == n:
                    yield window
                else:
                    raise ValueError(
                        f"Last window of length {len(window)} is shorter than specified {n=}"
                    )
        case NoneType():
            window = deque[T | F](last := tuple(islice(it, n)), maxlen=n)
            if not window:
                return

            while True:
                yield tuple(window)
                n1 = min(stride, len(window))
                consume(map(call, [window.popleft] * n1))
                stride_remaining = tuple(islice(it, stride - n1))
                if stride - n1 > 0 and not stride_remaining:
                    if not window:
                        return
                else:
                    window.extend(islice(it, stride))
                if not window:
                    return
        case unknown:  # # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(f"Received unknown value for uneven: {unknown!r}")  # pyright: ignore[reportUnreachable]


def sliding_window_seq[T, F](
    it: Sequence[T],
    n: int,
    *,
    stride: int = 1,
    uneven: MissingPolicy[F] | None = None,
) -> Sequence[tuple[T, ...]] | Sequence[tuple[T | F, ...]]:
    if not it:
        return ()
    result = *rest, last = [
        tuple[T | F, ...](it[i : i + n]) for i in range(0, len(it), stride)
    ]
    match uneven:
        case Pad(fillvalue=pad):
            if len(last) < n:
                last += (pad,) * (n - len(last))
            rest.append(last)
            return rest
        case NoneType():
            return result
        case Ignore():
            if len(last) < n:
                return rest
            rest.append(last)
            return rest
        case Raise():
            if len(last) < n:
                raise ValueError(
                    f"Last window of length {len(last)} is shorter than specified {n=}"
                )
            rest.append(last)
            return rest
