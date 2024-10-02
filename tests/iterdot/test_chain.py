from collections import deque
from collections.abc import Iterable
from operator import add
from typing import Any, no_type_check
from chainable_itertools.chain import Iter, Default
import itertools as itl
import pytest


def consume(iterable: Iterable[Any]):
    _ = deque(iterable, maxlen=0)


@pytest.fixture
def integers_from_0_to_1000() -> list[int]:
    return list(range(0, 1_001))


def test_peek_next_index(integers_from_0_to_1000: list[int]):
    it = Iter(integers_from_0_to_1000)
    assert it.peek_next_index() == 0

    _ = next(it)
    assert it.peek_next_index() == 1

    # after the last `next` call, iterator has advanced one position
    # so: it ==> 1 2 3 4 ... 1000
    consume(itl.islice(it, 5, 150, 5))
    # the above call will actually consume upto the stop position
    # the above expression evaluates to [6, 11, 16, ..., 146]
    # but the iterator is consumed upto 150 (1 + 150 -1)

    normal_it = iter(integers_from_0_to_1000)
    _ = next(normal_it)
    consume(itl.islice(normal_it, 5, 150, 5))
    assert it.peek_next_index() == next(normal_it)


def test_peek_next_value(integers_from_0_to_1000: list[int]):
    it = Iter(integers_from_0_to_1000)
    assert it.peek_next_value() == 0

    _ = next(it)
    assert it.peek_next_value() == 1

    # after the last `next` call, iterator has advanced one position
    # so: it ==> 1 2 3 4 ... 1000
    consume(itl.islice(it, 5, 150, 5))
    # the above call will actually consume upto the stop position
    # the above expression evaluates to [6, 11, 16, ..., 146]
    # but the iterator is consumed upto 150 (1 + 150 - 1)

    normal_it = iter(integers_from_0_to_1000)
    _ = next(normal_it)
    consume(itl.islice(normal_it, 5, 150, 5))
    assert it.peek_next_value() == next(normal_it)

    consume(it)

    assert it.peek_next_value() == Default.Exhausted


def test_next_value(integers_from_0_to_1000: list[int]):
    it = Iter(integers_from_0_to_1000)
    assert it.next_value() == 0

    _ = next(it)
    assert it.next_value() == 2

    # after the last 3 calls, iterator has advanced three positions
    # so: it ==> 3 4 5 ... 1000
    consume(itl.islice(it, 5, 150, 5))
    # the above call will actually consume upto the stop position
    # the above expression evaluates to [8, 13, 18, ..., 148]
    # but the iterator is consumed upto 152 (3 + 150 - 1)
    #                                        ^    ^
    #                                        |    |
    #                                        |    stop value
    #                                        iterator first value

    normal_it = iter(integers_from_0_to_1000)
    # consume upto 2
    consume(iter(normal_it.__next__, 2))
    consume(itl.islice(normal_it, 5, 150, 5))
    assert it.next_value() == next(normal_it)

    consume(it)
    with pytest.raises(StopIteration):
        _ = it.next_value()


def test_getattr(integers_from_0_to_1000: list[int]):
    it = Iter(integers_from_0_to_1000)
    real_parts = it.getattr("real", type=int)
    assert list(real_parts) == integers_from_0_to_1000


def test_to_list():
    assert Iter(range(5)).to_list() == list(range(5))
    assert Iter(()).to_list() == []


def test_compress():
    selectors = [0, 0, 1, 0, 1]
    assert Iter(range(5)).compress(selectors).to_list() == list(
        itl.compress(range(5), selectors)
    )


def test_pairwise():
    assert Iter(range(5)).pairwise().to_list() == list(itl.pairwise(range(5)))


def test_batched():
    rng = range(5)
    assert Iter(rng).batched(2).to_list() == list(itl.batched(rng, 2))

    with pytest.raises(ValueError):
        _ = list(itl.batched(rng, 2, strict=True))


def test_accumulate():
    rng = range(5)
    assert Iter(rng).accumulate(add).to_list() == [0, 1, 3, 6, 10]
    assert Iter(()).accumulate(add, initial=0).to_list() == [0]
    assert Iter(()).accumulate(add).to_list() == []


def test_slice(integers_from_0_to_1000: list[int]):
    assert Iter(integers_from_0_to_1000).slice(2).to_list() == [0, 1]
    assert Iter(integers_from_0_to_1000).slice(2, 5).to_list() == [2, 3, 4]
    assert (
        Iter(integers_from_0_to_1000).slice(0, None).to_list()
        == integers_from_0_to_1000
    )
    assert Iter(integers_from_0_to_1000).slice(0, None, 100).to_list() == list(
        range(0, 1001, 100)
    )
    with pytest.raises(ValueError):
        _ = Iter(integers_from_0_to_1000).slice(-1).to_list()
    with pytest.raises(ValueError):
        _ = Iter(integers_from_0_to_1000).slice(0, -1).to_list()
    with pytest.raises(ValueError):
        _ = Iter(integers_from_0_to_1000).slice(0, 5, -1).to_list()


def test_zip_with():
    assert Iter(range(5)).zip_with(range(5, 10)).to_list() == list(
        zip(range(5), range(5, 10))
    )
    assert Iter(range(4)).zip_with(range(5, 10)).to_list() == list(
        zip(range(4), range(5, 10))
    )
    with pytest.raises(ValueError):
        _ = Iter(range(4)).zip_with(range(5, 10), strict=True).to_list()


def test_takewhile():
    assert Iter(range(5)).takewhile(lambda x: x < 3).to_list() == [0, 1, 2]


def test_dropwhile():
    assert Iter(range(5)).dropwhile(lambda x: x < 3).to_list() == [3, 4]


def test_sum():
    assert Iter(range(5)).sum() == 10
    assert Iter(range(5)).sum(start=2) == 12
    assert Iter(()).sum() == 0


def test_max():
    assert Iter(range(-5, -1)).max() == -2
    assert Iter(range(-5, -1)).max(key=abs) == -5
    assert Iter(range(-5, -1)).max(default=10) == -2
    assert Iter(range(-5, -1)).max(key=abs, default=10) == -5
    assert Iter(()).max(key=abs, default=10) == 10
    with pytest.raises(ValueError):
        _ = Iter(()).max()


def test_min():
    assert Iter(range(-5, -1)).min() == -5
    assert Iter(range(-5, -1)).min(key=abs) == -2
    assert Iter(range(-5, -1)).min(default=10) == -5
    assert Iter(range(-5, -1)).min(key=abs, default=10) == -2
    assert Iter(()).min(key=abs, default=10) == 10
    with pytest.raises(ValueError):
        _ = Iter(()).min()


def test__iter__():
    it = Iter(range(5))
    assert iter(it) is it
    assert list(iter(it)) == list(range(5))


def test__next__():
    it = Iter([0])
    assert next(it) == 0
    with pytest.raises(StopIteration):
        _ = next(it)
    assert next(it, None) is None
    assert it.last_yielded_value == 0
    assert it.last_yielded_index == 0
    assert it.peek_next_index() == 1
    assert it.peek_next_value() == Default.Exhausted


def test_map_partial():
    assert Iter(range(5)).map_partial(add, 5).to_list() == [5, 6, 7, 8, 9]
    assert Iter("abcde").map_partial(int, base=16).to_list() == [10, 11, 12, 13, 14]


def test_map():
    assert Iter(range(5)).map(float).to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert Iter(range(5)).map(add, range(4, -1, -1)).to_list() == [4, 4, 4, 4, 4]


def test_feed_into():
    assert Iter(range(5)).feed_into(list) == [0, 1, 2, 3, 4]
    assert Iter(range(5)).feed_into(sum) == 10
    assert Iter(range(10)).feed_into(deque, maxlen=5) == deque([5, 6, 7, 8, 9])


def test_filter():
    assert Iter(range(5)).filter(lambda x: x < 2).to_list() == [0, 1]
    assert Iter(range(5)).filter(lambda x: x >= 2, when=False).to_list() == [0, 1]


def test_starmap():
    itbl = [
        (6, 0, 1),
        (3, 4, 5),
        (9, 7, 0),
    ]
    assert Iter(itbl).starmap(max).to_list() == [6, 5, 9]


def test_first():
    assert Iter(range(5)).first() == 0
    assert Iter(()).first() is Default.Exhausted
    assert Iter(()).first(0) == 0
    with pytest.raises(StopIteration):
        _ = Iter(()).first(default=Default.NoDefault)


def test_at():
    assert Iter([10, 5, 8, 9, 5]).at(3) == 9
    assert Iter(range(5)).at(10) is Default.Exhausted
    assert Iter(range(5)).at(10, default=10) == 10
    with pytest.raises(StopIteration):
        _ = Iter(range(5)).at(10, default=Default.NoDefault)


def test_last():
    assert Iter(range(5)).last() == 4
    assert Iter(()).last() is Default.Exhausted
    assert Iter(()).last(0) == 0
    with pytest.raises(StopIteration):
        _ = Iter(()).last(default=Default.NoDefault)


def test_tail():
    assert Iter(range(100)).tail(5).to_list() == [95, 96, 97, 98, 99]
    assert Iter(()).tail(5).to_list() == []
    assert Iter(range(3)).tail(5).to_list() == [0, 1, 2]


def test_skip():
    assert Iter(range(100)).skip(95).to_list() == [95, 96, 97, 98, 99]
    assert Iter(range(100)).skip(5).skip(90).to_list() == [95, 96, 97, 98, 99]


def test_exhaust():
    itrtr = Iter(range(5, 0, -1))
    itrtr.exhaust()
    assert itrtr.last_yielded_value == 1
    assert itrtr.last_yielded_index == 4
    assert list(itrtr) == []


@no_type_check
def test_foreach(capsys):
    Iter(range(5)).foreach(print)
    assert (
        capsys.readouterr().out
        == """\
0
1
2
3
4
"""
    )
