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


def test_is_neq() -> None:
    """Test is_neq function - not equal comparison."""
    # Basic inequality
    assert functions.is_neq(1.0, 2.0) is True
    assert functions.is_neq(2.0, 1.0) is True

    # Equal values should return False
    assert functions.is_neq(1.0, 1.0) is False

    # Within tolerance should be considered equal (return False)
    assert functions.is_neq(1.0, 1.0 + functions.EPSILON * 0.5) is False

    # Outside tolerance should be considered not equal (return True)
    assert functions.is_neq(1.0, 1.0 + functions.EPSILON * 2) is True

    # Custom tolerance
    assert functions.is_neq(1.0, 1.05, tol=0.1) is False  # Within 0.1 tolerance
    assert functions.is_neq(1.0, 1.15, tol=0.1) is True  # Outside 0.1 tolerance


def test_coalesce() -> None:
    """Test coalesce function - returns first non-None value."""
    import sympy as sp

    # Basic usage with concrete values
    result = functions.coalesce(5, 0)
    assert float(result) == 5

    # First value is concrete
    result = functions.coalesce(10, 20, 30)
    assert float(result) == 10

    # With sympy numbers
    result = functions.coalesce(sp.Integer(42), sp.Integer(0))
    assert float(result) == 42

    # Zero is a valid non-None value
    result = functions.coalesce(0, 10)
    assert float(result) == 0

    # With symbols - should remain unevaluated
    x = sp.Symbol("x")
    result = functions.coalesce(x, 0)
    assert result.has(x)  # Contains the symbol

    # None is skipped to return first valid value
    result = functions.coalesce(None, 5)
    assert float(result) == 5

    result = functions.coalesce(None, None, 10)
    assert float(result) == 10

    # All None/NaN values should return NaN
    result = functions.coalesce(None, None)
    assert result == sp.S.NaN

    result = functions.coalesce(sp.S.NaN, None)
    assert result == sp.S.NaN

    result = functions.coalesce(None, sp.S.NaN, None)
    assert result == sp.S.NaN

    # Error on empty args
    with pytest.raises(DQXError, match="coalesce requires at least one argument"):
        functions.coalesce()
