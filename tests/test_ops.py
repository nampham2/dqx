"""Test cases for op-related functionality.

This module tests the operations (ops) used in DQX, including
both the protocol validation and concrete implementations.
"""

import pytest

from dqx import ops
from dqx.common import DQXError


def test_op_protocol() -> None:
    """Test that concrete ops conform to the Op protocol."""
    assert isinstance(ops.NumRows(), ops.Op)
    assert isinstance(ops.Average("col"), ops.Op)
    assert isinstance(ops.Minimum("col"), ops.Op)
    assert isinstance(ops.Maximum("col"), ops.Op)
    assert isinstance(ops.Sum("col"), ops.Op)
    assert isinstance(ops.Variance("col"), ops.Op)
    assert isinstance(ops.First("col"), ops.Op)
    assert isinstance(ops.NullCount("col"), ops.Op)
    assert isinstance(ops.NegativeCount("col"), ops.Op)
    assert isinstance(ops.DuplicateCount(["col"]), ops.Op)
    assert isinstance(ops.CountValues("col", 1), ops.Op)


def test_sql_op_protocol() -> None:
    """Test that concrete ops conform to the SqlOp protocol."""
    assert isinstance(ops.NumRows(), ops.SqlOp)
    assert isinstance(ops.Average("col"), ops.SqlOp)
    assert isinstance(ops.Minimum("col"), ops.SqlOp)
    assert isinstance(ops.Maximum("col"), ops.SqlOp)
    assert isinstance(ops.Sum("col"), ops.SqlOp)
    assert isinstance(ops.Variance("col"), ops.SqlOp)
    assert isinstance(ops.First("col"), ops.SqlOp)
    assert isinstance(ops.NullCount("col"), ops.SqlOp)
    assert isinstance(ops.NegativeCount("col"), ops.SqlOp)
    assert isinstance(ops.DuplicateCount(["col"]), ops.SqlOp)
    assert isinstance(ops.CountValues("col", "test"), ops.SqlOp)


def test_num_rows() -> None:
    op = ops.NumRows()
    assert op.name == "num_rows()"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_num_rows()"

    # Test value assignment
    with pytest.raises(DQXError, match="NumRows op has not been collected yet!"):
        op.value()

    op.assign(42.0)
    assert op.value() == pytest.approx(42.0)

    # Test clear
    op.clear()
    with pytest.raises(DQXError):
        op.value()

    # Test equality
    op2 = ops.NumRows()
    assert op == op2
    assert op != ops.Average("col")

    # Test hash
    assert hash(op) == hash(op2)

    # Test string representation
    assert str(op) == "num_rows()"
    assert repr(op) == "num_rows()"


def test_average() -> None:
    op = ops.Average("price")
    assert op.name == "average(price)"
    assert op.column == "price"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_average(price)"

    # Test value assignment
    with pytest.raises(DQXError, match="Average op has not been collected yet!"):
        op.value()

    op.assign(99.5)
    assert op.value() == pytest.approx(99.5)

    # Test equality
    op2 = ops.Average("price")
    op3 = ops.Average("quantity")
    assert op == op2
    assert op != op3
    assert op != ops.NumRows()

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "average(price)"
    assert repr(op) == "average(price)"


def test_minimum() -> None:
    op = ops.Minimum("age")
    assert op.name == "minimum(age)"
    assert op.column == "age"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_minimum(age)"

    # Test value assignment
    op.assign(18.0)
    assert op.value() == pytest.approx(18.0)

    # Test equality
    op2 = ops.Minimum("age")
    op3 = ops.Minimum("height")
    assert op == op2
    assert op != op3

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "minimum(age)"
    assert repr(op) == "minimum(age)"


def test_maximum() -> None:
    op = ops.Maximum("score")
    assert op.name == "maximum(score)"
    assert op.column == "score"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_maximum(score)"

    # Test value assignment
    op.assign(100.0)
    assert op.value() == pytest.approx(100.0)

    # Test equality
    op2 = ops.Maximum("score")
    op3 = ops.Maximum("points")
    assert op == op2
    assert op != op3

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "maximum(score)"
    assert repr(op) == "maximum(score)"


def test_sum() -> None:
    op = ops.Sum("revenue")
    assert op.name == "sum(revenue)"
    assert op.column == "revenue"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_sum(revenue)"

    # Test value assignment
    op.assign(12345.67)
    assert op.value() == pytest.approx(12345.67)

    # Test equality
    op2 = ops.Sum("revenue")
    op3 = ops.Sum("cost")
    assert op == op2
    assert op != op3

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "sum(revenue)"
    assert repr(op) == "sum(revenue)"


def test_variance() -> None:
    op = ops.Variance("temperature")
    assert op.name == "variance(temperature)"
    assert op.column == "temperature"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_variance(temperature)"

    # Test value assignment
    op.assign(5.25)
    assert op.value() == pytest.approx(5.25)

    # Test equality
    op2 = ops.Variance("temperature")
    op3 = ops.Variance("humidity")
    assert op == op2
    assert op != op3

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "variance(temperature)"
    assert repr(op) == "variance(temperature)"


def test_first() -> None:
    op = ops.First("timestamp")
    assert op.name == "first(timestamp)"
    assert op.column == "timestamp"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_first(timestamp)"

    # Test value assignment
    op.assign(1234567890.0)
    assert op.value() == pytest.approx(1234567890.0)

    # Test equality
    op2 = ops.First("timestamp")
    op3 = ops.First("date")
    assert op == op2
    assert op != op3

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "first(timestamp)"
    assert repr(op) == "first(timestamp)"


def test_null_count() -> None:
    op = ops.NullCount("email")
    assert op.name == "null_count(email)"
    assert op.column == "email"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_null_count(email)"

    # Test value assignment
    op.assign(15.0)
    assert op.value() == pytest.approx(15.0)

    # Test equality
    op2 = ops.NullCount("email")
    op3 = ops.NullCount("phone")
    assert op == op2
    assert op != op3

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "null_count(email)"
    assert repr(op) == "null_count(email)"


def test_negative_count() -> None:
    op = ops.NegativeCount("balance")
    assert op.name == "negative_count(balance)"
    assert op.column == "balance"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_negative_count(balance)"

    # Test value assignment
    op.assign(3.0)
    assert op.value() == pytest.approx(3.0)

    # Test equality
    op2 = ops.NegativeCount("balance")
    op3 = ops.NegativeCount("profit")
    assert op == op2
    assert op != op3

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "negative_count(balance)"
    assert repr(op) == "negative_count(balance)"


def test_duplicate_count() -> None:
    # Test single column
    op = ops.DuplicateCount(["email"])
    assert op.name == "duplicate_count(email)"
    assert op.columns == ["email"]
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_duplicate_count(email)"

    # Test multiple columns (should be sorted)
    op2 = ops.DuplicateCount(["user_id", "email"])
    assert op2.name == "duplicate_count(email,user_id)"
    assert op2.columns == ["email", "user_id"]  # Sorted

    # Test that order doesn't matter for equality (due to sorting)
    op3 = ops.DuplicateCount(["email", "user_id"])
    assert op2 == op3
    assert hash(op2) == hash(op3)

    # Test empty columns error
    with pytest.raises(ValueError, match="DuplicateCount requires at least one column"):
        ops.DuplicateCount([])

    # Test value assignment
    op.assign(10.0)
    assert op.value() == pytest.approx(10.0)

    # Test string representation
    assert str(op) == "duplicate_count(email)"
    assert repr(op) == "duplicate_count(email)"


def test_count_values_single() -> None:
    # Test with single integer value
    op_int = ops.CountValues("status", 1)
    assert isinstance(op_int, ops.Op)
    assert isinstance(op_int, ops.SqlOp)
    assert op_int.name == "count_values(status,1)"

    # Test with single string value
    op_str = ops.CountValues("category", "active")
    assert op_str.name == "count_values(category,active)"

    # Test value assignment
    with pytest.raises(DQXError, match="CountValues op has not been collected yet!"):
        op_int.value()

    op_int.assign(42.0)
    assert op_int.value() == pytest.approx(42.0)

    # Test clear functionality
    op_int.clear()
    with pytest.raises(DQXError):
        op_int.value()


def test_count_values_multiple() -> None:
    # Test with multiple integer values
    op_ints = ops.CountValues("type_id", [1, 2, 3])
    assert op_ints.name == "count_values(type_id,[1,2,3])"

    # Test with multiple string values
    op_strs = ops.CountValues("status", ["active", "pending"])
    assert op_strs.name == "count_values(status,[active,pending])"

    # Test single-item list
    op_single_list = ops.CountValues("category", ["electronics"])
    assert op_single_list.name == "count_values(category,[electronics])"


def test_count_values_invalid_types() -> None:
    # Test invalid single value type
    with pytest.raises(ValueError, match="CountValues accepts int, str, list\\[int\\], or list\\[str\\]"):
        ops.CountValues("column", 3.14)  # type: ignore

    with pytest.raises(ValueError, match="CountValues accepts int, str, list\\[int\\], or list\\[str\\]"):
        ops.CountValues("column", True)  # type: ignore

    # Test empty list
    with pytest.raises(ValueError, match="CountValues requires at least one value"):
        ops.CountValues("column", [])

    # Test mixed type list
    with pytest.raises(ValueError, match="CountValues list must contain all integers or all strings"):
        ops.CountValues("column", [1, "two", 3])  # type: ignore


def test_count_values_equality() -> None:
    # Single values
    op1 = ops.CountValues("col", 1)
    op2 = ops.CountValues("col", 1)
    op3 = ops.CountValues("col", 2)
    op4 = ops.CountValues("col", "1")  # String "1" vs int 1

    assert op1 == op2
    assert op1 != op3
    assert op1 != op4  # Different types

    # List values
    op5 = ops.CountValues("col", [1, 2])
    op6 = ops.CountValues("col", [1, 2])
    op7 = ops.CountValues("col", [2, 1])  # Different order

    assert op5 == op6
    assert op5 != op7  # Order matters

    # Single vs list
    op8 = ops.CountValues("col", 1)
    op9 = ops.CountValues("col", [1])
    assert op8 != op9  # Different formats


def test_count_values_hashing() -> None:
    op1 = ops.CountValues("col", ["test", "values"])
    op2 = ops.CountValues("col", ["test", "values"])
    op3 = ops.CountValues("col", ["different", "values"])

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)

    # Test deduplication in sets
    assert {op1, op2} == {op1}


def test_count_values_string_repr() -> None:
    op_single = ops.CountValues("user_id", 123)
    op_list = ops.CountValues("status", ["active", "pending", "completed"])

    assert str(op_single) == "count_values(user_id,123)"
    assert repr(op_single) == "count_values(user_id,123)"
    assert str(op_list) == "count_values(status,[active,pending,completed])"
    assert repr(op_list) == "count_values(status,[active,pending,completed])"


def test_count_values_special_characters() -> None:
    # Test backslashes (Windows paths)
    op_path = ops.CountValues("path", "C:\\Users\\test")
    assert op_path.name == "count_values(path,C:\\Users\\test)"

    # Test quotes in strings
    op_quote = ops.CountValues("name", "O'Brien")
    assert op_quote.name == "count_values(name,O'Brien)"

    # Test Unicode
    op_unicode = ops.CountValues("name", "José")
    assert op_unicode.name == "count_values(name,José)"

    # Test mixed quotes
    op_mixed = ops.CountValues("text", 'He said "Hello"')
    assert op_mixed.name == 'count_values(text,He said "Hello")'


def test_sql_op_properties() -> None:
    """Test that all SqlOp implementations have required properties."""
    sql_ops: list[ops.SqlOp] = [
        ops.NumRows(),
        ops.Average("col"),
        ops.Minimum("col"),
        ops.Maximum("col"),
        ops.Sum("col"),
        ops.Variance("col"),
        ops.First("col"),
        ops.NullCount("col"),
        ops.NegativeCount("col"),
        ops.DuplicateCount(["col"]),
        ops.CountValues("col", 1),
    ]

    for op in sql_ops:
        # Each should have a unique prefix
        assert op.prefix is not None
        assert len(op.prefix) > 0

        # Each should have a sql_col property
        assert op.sql_col is not None
        assert op.prefix in op.sql_col

        # For CountValues, sql_col uses a hash instead of the full name
        # to avoid special characters in column aliases
        if not isinstance(op, ops.CountValues):
            assert op.name in op.sql_col

        # Each should have a name
        assert op.name is not None
        assert len(op.name) > 0


def test_op_value_assignment_and_clearing() -> None:
    """Test value assignment and clearing for all ops."""
    ops_to_test = [
        ops.NumRows(),
        ops.Average("col"),
        ops.Minimum("col"),
        ops.Maximum("col"),
        ops.Sum("col"),
        ops.Variance("col"),
        ops.First("col"),
        ops.NullCount("col"),
        ops.NegativeCount("col"),
        ops.DuplicateCount(["col"]),
        ops.CountValues("col", "test"),
    ]

    for op in ops_to_test:
        # Initially, should raise error when accessing value
        with pytest.raises(DQXError):
            op.value()

        # Assign a value
        op.assign(123.45)
        assert op.value() == pytest.approx(123.45)

        # Clear the value
        op.clear()
        with pytest.raises(DQXError):
            op.value()
