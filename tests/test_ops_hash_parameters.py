"""Test that all operation __hash__ methods include parameters."""

import pytest

from dqx.ops import (
    Average,
    CountValues,
    CustomSQL,
    DuplicateCount,
    First,
    Maximum,
    Minimum,
    NegativeCount,
    NullCount,
    NumRows,
    Sum,
    UniqueCount,
    Variance,
)


def test_numrows_hash_includes_parameters() -> None:
    """Test that NumRows hash includes parameters."""
    op1 = NumRows()
    op2 = NumRows()
    op3 = NumRows(parameters={"region": "US"})
    op4 = NumRows(parameters={"region": "US"})
    op5 = NumRows(parameters={"region": "EU"})

    # Same parameters should have same hash
    assert hash(op1) == hash(op2)
    assert hash(op3) == hash(op4)

    # Different parameters should have different hash
    assert hash(op1) != hash(op3)
    assert hash(op3) != hash(op5)


@pytest.mark.parametrize(
    "OpClass,column",
    [
        (Average, "price"),
        (Minimum, "value"),
        (Maximum, "value"),
        (Sum, "amount"),
        (Variance, "score"),
        (First, "id"),
        (NullCount, "email"),
        (NegativeCount, "balance"),
        (UniqueCount, "user_id"),
    ],
)
def test_single_column_ops_hash_includes_parameters(OpClass: type, column: str) -> None:
    """Test that single column operations include parameters in hash."""
    op1 = OpClass(column)
    op2 = OpClass(column)
    op3 = OpClass(column, parameters={"region": "US"})
    op4 = OpClass(column, parameters={"region": "US"})
    op5 = OpClass(column, parameters={"region": "EU"})

    # Same parameters should have same hash
    assert hash(op1) == hash(op2)
    assert hash(op3) == hash(op4)

    # Different parameters should have different hash
    assert hash(op1) != hash(op3)
    assert hash(op3) != hash(op5)


def test_customsql_hash_includes_parameters() -> None:
    """Test that CustomSQL hash includes parameters."""
    op1 = CustomSQL("COUNT(*)")
    op2 = CustomSQL("COUNT(*)")
    op3 = CustomSQL("COUNT(*)", parameters={"status": "active"})
    op4 = CustomSQL("COUNT(*)", parameters={"status": "active"})
    op5 = CustomSQL("COUNT(*)", parameters={"status": "inactive"})

    # Same parameters should have same hash
    assert hash(op1) == hash(op2)
    assert hash(op3) == hash(op4)

    # Different parameters should have different hash
    assert hash(op1) != hash(op3)
    assert hash(op3) != hash(op5)


def test_duplicatecount_hash_includes_parameters() -> None:
    """Test that DuplicateCount hash includes parameters."""
    op1 = DuplicateCount(["email"])
    op2 = DuplicateCount(["email"])
    op3 = DuplicateCount(["email"], parameters={"domain": "gmail.com"})
    op4 = DuplicateCount(["email"], parameters={"domain": "gmail.com"})
    op5 = DuplicateCount(["email"], parameters={"domain": "yahoo.com"})

    # Same parameters should have same hash
    assert hash(op1) == hash(op2)
    assert hash(op3) == hash(op4)

    # Different parameters should have different hash
    assert hash(op1) != hash(op3)
    assert hash(op3) != hash(op5)


def test_countvalues_hash_includes_parameters() -> None:
    """Test that CountValues hash includes parameters."""
    op1 = CountValues("status", "active")
    op2 = CountValues("status", "active")
    op3 = CountValues("status", "active", parameters={"priority": "high"})
    op4 = CountValues("status", "active", parameters={"priority": "high"})
    op5 = CountValues("status", "active", parameters={"priority": "low"})

    # Same parameters should have same hash
    assert hash(op1) == hash(op2)
    assert hash(op3) == hash(op4)

    # Different parameters should have different hash
    assert hash(op1) != hash(op3)
    assert hash(op3) != hash(op5)
