from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import wraps
from itertools import chain, groupby, islice, repeat, takewhile
from types import NoneType

from iterdot.defaults import Ignore, MissingPolicy, Pad, Raise


def prepend[T](*val: T, to: Iterator[T]) -> Iterator[T]:
    return chain(val, to)


def flatten(iterable: Iterable[object]) -> Iterable[object]:
    for item in iterable:
        if isinstance(item, Iterable) and not isinstance(item, str):
            yield from flatten(item)
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
            yield from takewhile(
                lambda window: len(window) == n,
                sliding_window_iter(it, n, stride=stride, uneven=None),
            )
        case Raise():
            windows = sliding_window_iter(it, n, stride=stride, uneven=None)
            for winlen, group in groupby(windows, key=len):
                if winlen != n:
                    raise ValueError(
                        f"Last window of length {winlen} is shorter than specified {n=}"
                    )
                yield from group
        case NoneType():
            window = deque[T | F](tuple(islice(it, n)), maxlen=n)
            while window:
                yield tuple(window)
                window.extend(stride_remaining := tuple(islice(it, stride)))
                num_pop = min(len(window), stride - len(stride_remaining))
                consume(window.popleft() for _ in repeat(None, num_pop))

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
        case unknown:  # # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(f"Received unknown value for uneven: {unknown!r}")  # pyright: ignore[reportUnreachable]
