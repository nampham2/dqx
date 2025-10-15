import functools

import pytest

from dqx import functions
from dqx.common import DQXError


def test_is_gt() -> None:
    assert functions.is_gt(1, 1) is False
    assert functools.partial(functions.is_gt, b=1.1)(1) is False
    assert functions.is_gt(2.0, 1.0) is True
    assert functions.is_gt(1.0, 2.0) is False
    assert functions.is_gt(1.0, 1.0 - functions.EPSILON * 2) is True
    assert functions.is_gt(1.0, 1.0, tol=0.1) is False


def test_is_geq() -> None:
    assert functions.is_geq(1.0, 1.0) is True
    assert functions.is_geq(2.0, 1.0) is True
    assert functions.is_geq(1.0, 2.0) is False
    assert functions.is_geq(1.0, 1.0 + functions.EPSILON * 0.5) is True


def test_is_leq() -> None:
    assert functions.is_leq(1.0, 1.0) is True
    assert functions.is_leq(1.0, 2.0) is True
    assert functions.is_leq(2.0, 1.0) is False
    assert functions.is_leq(1.0, 1.0 - functions.EPSILON * 0.5) is True


def test_is_lt() -> None:
    assert functions.is_lt(1.0, 2.0) is True
    assert functions.is_lt(2.0, 1.0) is False
    assert functions.is_lt(1.0, 1.0) is False
    assert functions.is_lt(1.0, 1.0 + functions.EPSILON * 2) is True


def test_is_eq() -> None:
    assert functions.is_eq(1.0, 1.0) is True
    assert functions.is_eq(2.0, 1.0) is False
    assert functions.is_eq(1.0, 1.0 + functions.EPSILON * 0.5) is True
    assert functions.is_eq(1.0, 1.0 + functions.EPSILON * 2) is False


def test_within_tol() -> None:
    with pytest.raises(DQXError, match="Either relative tolerance or absolute tolerance must be provided!"):
        functions.within_tol(1.0, 1.0)

    with pytest.raises(
        DQXError, match="Both relative tolerance and absolute tolerance cannot be provided simultaneously!"
    ):
        functions.within_tol(1.0, 1.0, rel_tol=0.1, abs_tol=0.1)

    assert functions.within_tol(1.0, 1.05, abs_tol=0.1) is True
    assert functions.within_tol(1.0, 1.15, abs_tol=0.1) is False

    assert functions.within_tol(1.0, 1.05, rel_tol=0.1) is True
    assert functions.within_tol(1.0, 1.15, rel_tol=0.1) is False


def test_is_zero() -> None:
    assert functions.is_zero(0.0) is True
    assert functions.is_zero(functions.EPSILON * 0.5) is True
    assert functions.is_zero(functions.EPSILON * 2) is False
    assert functions.is_zero(1.0) is False


def test_is_positive() -> None:
    assert functions.is_positive(1.0) is True
    assert functions.is_positive(functions.EPSILON * 2) is True
    assert functions.is_positive(functions.EPSILON * 0.5) is False
    assert functions.is_positive(0.0) is False
    assert functions.is_positive(-1.0) is False


def test_is_negative() -> None:
    assert functions.is_negative(-1.0) is True
    assert functions.is_negative(-functions.EPSILON * 2) is True
    assert functions.is_negative(-functions.EPSILON * 0.5) is False
    assert functions.is_negative(0.0) is False
    assert functions.is_negative(1.0) is False


def test_is_between() -> None:
    """Test is_between function with various inputs."""
    # Basic integer tests
    assert functions.is_between(5, 1, 10) is True
    assert functions.is_between(0, 1, 10) is False
    assert functions.is_between(11, 1, 10) is False

    # Boundary tests (inclusive)
    assert functions.is_between(1, 1, 10) is True
    assert functions.is_between(10, 1, 10) is True

    # Floating point tests
    assert functions.is_between(5.5, 5.0, 6.0) is True
    assert functions.is_between(4.9, 5.0, 6.0) is False

    # Tolerance tests
    epsilon = 1e-9
    assert functions.is_between(5.0 - epsilon / 2, 5.0, 10.0, tol=epsilon) is True
    assert functions.is_between(5.0 - epsilon * 2, 5.0, 10.0, tol=epsilon) is False
    assert functions.is_between(10.0 + epsilon / 2, 5.0, 10.0, tol=epsilon) is True
    assert functions.is_between(10.0 + epsilon * 2, 5.0, 10.0, tol=epsilon) is False

    # Equal bounds
    assert functions.is_between(5, 5, 5) is True
    assert functions.is_between(4, 5, 5) is False
    assert functions.is_between(6, 5, 5) is False

    # Negative numbers
    assert functions.is_between(-5, -10, -1) is True
    assert functions.is_between(-11, -10, -1) is False
    assert functions.is_between(0, -10, -1) is False

    # Mixed positive/negative
    assert functions.is_between(0, -5, 5) is True
    assert functions.is_between(-3, -5, 5) is True
    assert functions.is_between(3, -5, 5) is True
    assert functions.is_between(-6, -5, 5) is False
    assert functions.is_between(6, -5, 5) is False
