import functools

from dqx import functions


def test_is_gt() -> None:
    assert functions.is_gt(1, 1) is False
    assert functools.partial(functions.is_gt, b=1.1)(1) is False
    