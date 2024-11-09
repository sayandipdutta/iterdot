import pytest

from iterdot._helpers import sliding_window_iter, sliding_window_seq
from iterdot.defaults import Ignore, Pad, Raise


def test_sliding_window_iter_none() -> None:
    itbl = [1, 2, 3]
    assert list(sliding_window_iter(iter(itbl), 2, stride=2, uneven=None)) == [
        (1, 2),
        (3,),
    ]
    assert list(sliding_window_iter(iter(itbl), 2)) == [
        (1, 2),
        (2, 3),
        (3,),
    ]
    assert list(sliding_window_iter(iter([1, 2, 3, 4, 5]), 3, stride=2)) == [
        (1, 2, 3),
        (3, 4, 5),
        (5,),
    ]
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=3)
    ) == [
        (1, 2, 3),
        (4, 5),
    ]  # fmt: skip
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=4)
    ) == [
        (1, 2, 3),
        (5,),
    ]  # fmt: skip
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=5)
    ) == [
        (1, 2, 3),
    ]  # fmt: skip


def test_sliding_window_iter_pad() -> None:
    itbl = [1, 2, 3]
    assert list(sliding_window_iter(iter(itbl), 2, stride=2, uneven=Pad(0))) == [
        (1, 2),
        (3, 0),
    ]
    assert list(sliding_window_iter(iter(itbl), 2, uneven=Pad(0))) == [
        (1, 2),
        (2, 3),
        (3, 0),
    ]
    assert list(
        sliding_window_iter(iter([1, 2, 3, 4, 5]), 3, stride=2, uneven=Pad(0))
    ) == [
        (1, 2, 3),
        (3, 4, 5),
        (5, 0, 0),
    ]
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=3, uneven=Pad(0))
    ) == [
        (1, 2, 3),
        (4, 5, 0),
    ]  # fmt: skip
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=4, uneven=Pad(0))
    ) == [
        (1, 2, 3),
        (5, 0, 0),
    ]  # fmt: skip
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=5, uneven=Pad(0))
    ) == [
        (1, 2, 3),
    ]  # fmt: skip


def test_sliding_window_iter_ignore() -> None:
    itbl = [1, 2, 3]
    assert list(sliding_window_iter(iter(itbl), 2, stride=2, uneven=Ignore())) == [
        (1, 2),
    ]
    assert list(sliding_window_iter(iter(itbl), 2, uneven=Ignore())) == [
        (1, 2),
        (2, 3),
    ]
    assert list(
        sliding_window_iter(iter([1, 2, 3, 4, 5]), 3, stride=2, uneven=Ignore())
    ) == [
        (1, 2, 3),
        (3, 4, 5),
    ]
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=3, uneven=Ignore())
    ) == [
        (1, 2, 3),
    ]  # fmt: skip
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=4, uneven=Ignore())
    ) == [
        (1, 2, 3),
    ]  # fmt: skip
    assert list(sliding_window_iter(
        iter([1, 2, 3, 4, 5]), 3, stride=5, uneven=Ignore())
    ) == [
        (1, 2, 3),
    ]  # fmt: skip


def test_sliding_window_iter_raise() -> None:
    itbl = [1, 2, 3]
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_iter(iter(itbl), 2, stride=2, uneven=Raise()))
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_iter(iter(itbl), 2, uneven=Raise()))
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_iter(iter([1, 2, 3, 4, 5]), 3, stride=2, uneven=Raise()))
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_iter(iter([1, 2, 3, 4, 5]), 3, stride=3, uneven=Raise()))
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_iter(iter([1, 2, 3, 4, 5]), 3, stride=4, uneven=Raise()))
    assert list(
        sliding_window_iter(iter([1, 2, 3, 4, 5]), 3, stride=5, uneven=Raise())
    ) == [
        (1, 2, 3),
    ]


def test_sliding_window_seq_pad() -> None:
    itbl = [1, 2, 3]
    assert list(sliding_window_seq(tuple(itbl), 2, stride=2, uneven=Pad(0))) == [
        (1, 2),
        (3, 0),
    ]
    assert list(sliding_window_seq(tuple(itbl), 2, uneven=Pad(0))) == [
        (1, 2),
        (2, 3),
        (3, 0),
    ]
    assert list(sliding_window_seq((1, 2, 3, 4, 5), 3, stride=2, uneven=Pad(0))) == [
        (1, 2, 3),
        (3, 4, 5),
        (5, 0, 0),
    ]
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=3, uneven=Pad(0))
    ) == [
        (1, 2, 3),
        (4, 5, 0),
    ]  # fmt: skip
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=4, uneven=Pad(0))
    ) == [
        (1, 2, 3),
        (5, 0, 0),
    ]  # fmt: skip
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=5, uneven=Pad(0))
    ) == [
        (1, 2, 3),
    ]  # fmt: skip


def test_sliding_window_seq_none() -> None:
    itbl = [1, 2, 3]
    assert list(sliding_window_seq(tuple(itbl), 2, stride=2, uneven=None)) == [
        (1, 2),
        (3,),
    ]
    assert list(sliding_window_seq(tuple(itbl), 2)) == [
        (1, 2),
        (2, 3),
        (3,),
    ]
    assert list(sliding_window_seq((1, 2, 3, 4, 5), 3, stride=2)) == [
        (1, 2, 3),
        (3, 4, 5),
        (5,),
    ]
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=3)
    ) == [
        (1, 2, 3),
        (4, 5),
    ]  # fmt: skip
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=4)
    ) == [
        (1, 2, 3),
        (5,),
    ]  # fmt: skip
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=5)
    ) == [
        (1, 2, 3),
    ]  # fmt: skip


def test_sliding_window_seq_ignore() -> None:
    itbl = [1, 2, 3]
    assert list(sliding_window_seq(tuple(itbl), 2, stride=2, uneven=Ignore())) == [
        (1, 2),
    ]
    assert list(sliding_window_seq(tuple(itbl), 2, uneven=Ignore())) == [
        (1, 2),
        (2, 3),
    ]
    assert list(sliding_window_seq((1, 2, 3, 4, 5), 3, stride=2, uneven=Ignore())) == [
        (1, 2, 3),
        (3, 4, 5),
    ]
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=3, uneven=Ignore())
    ) == [
        (1, 2, 3),
    ]  # fmt: skip
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=4, uneven=Ignore())
    ) == [
        (1, 2, 3),
    ]  # fmt: skip
    assert list(sliding_window_seq(
        (1, 2, 3, 4, 5), 3, stride=5, uneven=Ignore())
    ) == [
        (1, 2, 3),
    ]  # fmt: skip


def test_sliding_window_seq_raise() -> None:
    itbl = [1, 2, 3]
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_seq(tuple(itbl), 2, stride=2, uneven=Raise()))
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_seq(tuple(itbl), 2, uneven=Raise()))
    with pytest.raises(ValueError, match="shorter than specified n"):
        assert list(sliding_window_seq((1, 2, 3, 4, 5), 3, stride=2, uneven=Raise()))
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_seq((1, 2, 3, 4, 5), 3, stride=3, uneven=Raise()))
    with pytest.raises(ValueError, match="shorter than specified n"):
        _ = list(sliding_window_seq((1, 2, 3, 4, 5), 3, stride=4, uneven=Raise()))
    assert list(sliding_window_seq((1, 2, 3, 4, 5), 3, stride=5, uneven=Raise())) == [
        (1, 2, 3),
    ]
    assert list(sliding_window_seq((1, 2, 3, 4, 5, 6), 3, stride=3, uneven=Raise())) == [
        (1, 2, 3),
        (4, 5, 6),
    ]
