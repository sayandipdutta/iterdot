from chainable_itertools.chain import ChainableIter as chit, ValueType
import itertools as itl
import pytest


@pytest.fixture
def integers_from_0_to_1000() -> list[int]:
    return list(range(0, 1_001))


def test_peek_next_index(integers_from_0_to_1000: list[int]):
    it = chit(integers_from_0_to_1000)
    assert it.peek_next_index() == 0

    _ = next(it)
    assert it.peek_next_index() == 1

    # after the last `next` call, iterator has advanced one position
    # so: it ==> 1 2 3 4 ... 1000
    _ = list(itl.islice(it, 5, 150, 5))
    # the above call will actually consume upto the stop position
    # the above expression evaluates to [6, 11, 16, ..., 146]
    # but the iterator is consumed upto 150 (1 + 150 -1)

    normal_it = iter(integers_from_0_to_1000)
    _ = next(normal_it)
    _ = list(itl.islice(normal_it, 5, 150, 5))
    assert it.peek_next_index() == next(normal_it)


def test_peek_next_value(integers_from_0_to_1000: list[int]):
    it = chit(integers_from_0_to_1000)
    assert it.peek_next_value() == 0

    _ = next(it)
    assert it.peek_next_value() == 1

    # after the last `next` call, iterator has advanced one position
    # so: it ==> 1 2 3 4 ... 1000
    _ = list(itl.islice(it, 5, 150, 5))
    # the above call will actually consume upto the stop position
    # the above expression evaluates to [6, 11, 16, ..., 146]
    # but the iterator is consumed upto 150 (1 + 150 - 1)

    normal_it = iter(integers_from_0_to_1000)
    _ = next(normal_it)
    _ = list(itl.islice(normal_it, 5, 150, 5))
    assert it.peek_next_value() == next(normal_it)

    # consume all
    _ = list(it)

    assert it.peek_next_value() == ValueType.NA


def test_next_value(integers_from_0_to_1000: list[int]):
    it = chit(integers_from_0_to_1000)
    assert it.next_value() == 0

    _ = next(it)
    assert it.next_value() == 2

    # after the last 3 calls, iterator has advanced three positions
    # so: it ==> 3 4 5 ... 1000
    _ = list(itl.islice(it, 5, 150, 5))
    # the above call will actually consume upto the stop position
    # the above expression evaluates to [8, 13, 18, ..., 148]
    # but the iterator is consumed upto 152 (3 + 150 - 1)
    #                                        ^    ^
    #                                        |    |
    #                                        |    stop value
    #                                        iterator first value

    normal_it = iter(integers_from_0_to_1000)
    # consume upto 2
    _ = list(iter(normal_it.__next__, 2))
    _ = list(itl.islice(normal_it, 5, 150, 5))
    assert it.next_value() == next(normal_it)

    # consume all
    _ = list(it)

    with pytest.raises(StopIteration):
        _ = it.next_value()


def test_get_attribute(integers_from_0_to_1000: list[int]):
    it = chit(integers_from_0_to_1000)
    real_parts = it.get_attribute("real", type=int)
    assert list(real_parts) == integers_from_0_to_1000


def test_compress(): ...
