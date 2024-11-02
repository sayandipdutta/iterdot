# pyright: reportArgumentType=false, reportCallIssue=false
import itertools as itl
from collections import deque
from collections.abc import Iterable
from operator import add
from typing import Any, no_type_check

import pytest

from iterdot.chain import Default, Iter, SeqIter


def consume(iterable: Iterable[Any]):
    _ = deque(iterable, maxlen=0)


@pytest.fixture
def integers_from_0_to_1000() -> list[int]:
    return list(range(0, 1_001))


def Iter_range_10() -> Iter[int]:
    return Iter(range(10))


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


def test_next(integers_from_0_to_1000: list[int]):
    it = Iter(integers_from_0_to_1000)
    assert it.next() == 0

    _ = next(it)
    assert it.next() == 2

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
    assert it.next() == next(normal_it)

    consume(it)
    with pytest.raises(StopIteration):
        _ = it.next()


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

    with pytest.raises(ValueError, match="incomplete batch"):
        _ = list(itl.batched(rng, 2, strict=True))


def test_accumulate():
    rng = range(5)
    assert Iter(rng).accumulate(add).to_list() == [0, 1, 3, 6, 10]
    assert Iter(()).accumulate(add, initial=0).to_list() == [0]
    assert Iter(()).accumulate(add).to_list() == []


def test_slice(integers_from_0_to_1000: list[int]):
    assert Iter(integers_from_0_to_1000).slice(stop=2).to_list() == [0, 1]
    assert Iter(integers_from_0_to_1000).slice(start=2, stop=5).to_list() == [2, 3, 4]
    assert (
        Iter(integers_from_0_to_1000).slice(start=0, stop=None).to_list()
        == integers_from_0_to_1000
    )
    assert Iter(integers_from_0_to_1000).slice(
        start=0, stop=None, step=100
    ).to_list() == list(range(0, 1001, 100))
    with pytest.raises(ValueError, match="None|integer"):
        _ = Iter(integers_from_0_to_1000).slice(stop=-1).to_list()
    with pytest.raises(ValueError, match="None|integer"):
        _ = Iter(integers_from_0_to_1000).slice(start=0, stop=-1).to_list()
    with pytest.raises(ValueError, match="None|integer"):
        _ = Iter(integers_from_0_to_1000).slice(start=0, stop=5, step=-1).to_list()


def test_zip_with():
    assert Iter(range(5)).zip_with(range(5, 10)).to_list() == list(
        zip(range(5), range(5, 10), strict=False)
    )
    assert Iter(range(4)).zip_with(range(5, 10)).to_list() == list(
        zip(range(4), range(5, 10), strict=False)
    )
    with pytest.raises(ValueError, match="shorter|longer"):
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
    with pytest.raises(ValueError, match="empty"):
        _ = Iter(()).max()


def test_min():
    assert Iter(range(-5, -1)).min() == -5
    assert Iter(range(-5, -1)).min(key=abs) == -2
    assert Iter(range(-5, -1)).min(default=10) == -5
    assert Iter(range(-5, -1)).min(key=abs, default=10) == -2
    assert Iter(()).min(key=abs, default=10) == 10
    with pytest.raises(ValueError, match="empty"):
        _ = Iter(()).min()


def test_all():
    assert Iter([True] * 5).all()
    assert not Iter([True, False] * 5).all()
    assert Iter(()).all()


def test_all_predicate():
    assert Iter([0, 2, 4, 6]).all(predicate=lambda x: x % 2 == 0)
    assert Iter(()).all(predicate=lambda x: x % 2 == 0)


def test_any():
    assert Iter([True, False] * 5).any()
    assert not Iter([False] * 5).any()
    assert not Iter(()).any()


def test_any_predicate():
    assert not Iter([0, 2, 4, 6]).any(predicate=lambda x: x % 2 == 1)
    assert not Iter(()).any(predicate=lambda x: x % 2 == 0)


def test_all_equal():
    assert Iter([1] * 5).all_equal()
    assert not Iter([1, 1, 1, 1, 2, 1]).all_equal()
    assert Iter([]).all_equal()


def test_all_equal_with():
    assert Iter([1] * 5).all_equal_with(1)
    assert not Iter[int]([]).all_equal_with(5)


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
    assert Iter(range(5)).filter(lambda x: x >= 2, invert=True).to_list() == [0, 1]


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
        Iter(()).first(default=Default.NoDefault)


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
        Iter(()).last(default=Default.NoDefault)


def test_tail():
    assert Iter(range(100)).tail(5).to_list() == [95, 96, 97, 98, 99]
    assert Iter(()).tail(5).to_list() == []
    assert Iter(range(3)).tail(5).to_list() == [0, 1, 2]


def test_skip():
    assert Iter(range(100)).skip(95).to_list() == [95, 96, 97, 98, 99]
    assert Iter(range(100)).skip(5).skip(90).to_list() == [95, 96, 97, 98, 99]


def test_skip_take():
    r = range(10)
    skip_first = [2, 3, 4, 7, 8, 9]
    take_first = [0, 1, 2, 5, 6, 7]
    assert Iter(r).skip_take(skip=2, take=3).to_list() == skip_first
    assert Iter(r).skip_take(skip=2, take=3, take_first=True).to_list() == take_first
    assert Iter(r).skip_take(take=3, skip=2, take_first=False).to_list() == skip_first
    assert Iter(r).skip_take(take=3, skip=2).to_list() == take_first


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


def test_collect():
    assert isinstance(Iter_range_10().collect(), SeqIter)
    assert isinstance(Iter_range_10().collect[list](), list)
    assert isinstance(Iter_range_10().collect[list[str]](), list)
    assert isinstance(Iter_range_10().collect[list[str]]()[0], str)

    # TRY to support the following
    # it = Iter(zip("abc", "123", strict=True)).collect[tuple[tuple[str, int]]]()
    # print(it)
    # assert isinstance(it[0][1], int)
