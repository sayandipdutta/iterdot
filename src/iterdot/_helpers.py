from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import wraps
from itertools import chain, islice
from typing import cast


def prepend[T](*val: T, to: Iterator[T]) -> Iterator[T]:
    return chain(val, to)


def flatten(iterable: Iterable[object]) -> Iterable[object]:
    for item in iterable:
        if isinstance(item, Iterable) and not isinstance(item, str):
            yield from flatten(cast(Iterable[object], item))
        else:
            yield item


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


def sliding_window[T](
    it: Iterator[T] | Sequence[T], n: int, stride: int = 1
) -> Iterable[tuple[T, ...]]:
    assert isinstance(n, int)
    assert isinstance(stride, int)
    assert n > 0
    assert stride > 0

    if isinstance(it, Iterator):
        window = deque[T](islice(it, n), maxlen=n)
        while window:
            yield tuple(window)
            window.extend(islice(it, stride))
    else:
        stop = max(0, (len(it) - n) // stride + 1)
        yield from (tuple(it[i : i + n]) for i in range(0, stop, stride))
