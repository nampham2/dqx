"""Test parameter support in operations."""

from typing import Any, Type

import pytest

from dqx.ops import (
    Average,
    CountValues,
    DuplicateCount,
    First,
    Maximum,
    Minimum,
    NegativeCount,
    NullCount,
    NumRows,
    OpValueMixin,
    Sum,
    UniqueCount,
    Variance,
)


def test_opvaluemixin_accepts_parameters() -> None:
    """OpValueMixin should accept and store parameters."""
    params = {"region": "US", "threshold": 100}

    class TestOp(OpValueMixin[float]):
        pass

    op = TestOp(parameters=params)
    assert op.parameters == params


def test_opvaluemixin_defaults_empty_parameters() -> None:
    """OpValueMixin should default to empty dict when no parameters."""

    class TestOp(OpValueMixin[float]):
        pass

    op = TestOp()
    assert op.parameters == {}


def test_opvaluemixin_converts_none_to_empty_dict() -> None:
    """OpValueMixin should convert None parameters to empty dict."""

    class TestOp(OpValueMixin[float]):
        pass

    op = TestOp(parameters=None)
    assert op.parameters == {}


def test_sqlop_protocol_has_parameters() -> None:
    """SqlOp protocol should include parameters property."""
    from dqx.ops import NumRows, SqlOp

    # Existing ops should satisfy protocol
    op = NumRows()
    assert isinstance(op, SqlOp)
    assert hasattr(op, "parameters")
    assert op.parameters == {}


def test_numrows_accepts_parameters() -> None:
    """NumRows should accept parameters."""
    params = {"region": "US"}
    op = NumRows(parameters=params)
    assert op.parameters == params

    # Backward compatibility - no parameters
    op2 = NumRows()
    assert op2.parameters == {}


@pytest.mark.parametrize(
    "op_class,column",
    [
        (Average, "price"),
        (Sum, "amount"),
        (Minimum, "value"),
        (Maximum, "value"),
        (Variance, "score"),
        (First, "id"),
        (NullCount, "field"),
        (NegativeCount, "balance"),
        (UniqueCount, "user_id"),
    ],
)
def test_single_column_ops_accept_parameters(op_class: Type[Any], column: str) -> None:
    """All single-column operations should accept parameters."""
    params = {"region": "EU", "min_value": 50}

    # With parameters
    op1 = op_class(column, parameters=params)
    assert op1.column == column
    assert op1.parameters == params

    # Without parameters (backward compatibility)
    op2 = op_class(column)
    assert op2.column == column
    assert op2.parameters == {}


def test_duplicate_count_accepts_parameters() -> None:
    """DuplicateCount should accept parameters."""
    cols = ["user_id", "date"]
    params = {"min_count": 2}

    op1 = DuplicateCount(cols, parameters=params)
    assert op1.columns == sorted(cols)
    assert op1.parameters == params

    # Backward compatibility
    op2 = DuplicateCount(cols)
    assert op2.parameters == {}


def test_count_values_accepts_parameters() -> None:
    """CountValues should accept parameters."""
    params = {"category": "electronics"}

    # Single value
    op1 = CountValues("status", "active", parameters=params)
    assert op1.parameters == params

    # Multiple values
    op2 = CountValues("status", ["active", "pending"], parameters=params)
    assert op2.parameters == params

    # Backward compatibility
    op3 = CountValues("status", "active")
    assert op3.parameters == {}
