import pytest

from dqx import ops
from dqx.common import DQXError
from dqx.ops import Op, SqlOp


def test_num_rows() -> None:
    op = ops.NumRows()
    assert isinstance(op, Op)
    assert isinstance(op, SqlOp)
    assert op.name == "num_rows()"

    op1 = ops.NumRows()
    assert op == op1

    with pytest.raises(DQXError, match="NumRows op has not been collected yet!"):
        op.value()

    op.assign(0.2)
    assert op.value() == pytest.approx(0.2)


def test_average() -> None:
    op = ops.Average("column")
    assert isinstance(op, Op)
    assert op.name == "average(column)"
    assert set([op, op]) == {op}


def test_minimum() -> None:
    op = ops.Minimum("column")
    assert isinstance(op, Op)
    assert op.name == "minimum(column)"


def test_maximum() -> None:
    op = ops.Maximum("column")
    assert isinstance(op, Op)
    assert isinstance(op, SqlOp)
    assert op.name == "maximum(column)"


def test_sum() -> None:
    op = ops.Sum("column")
    assert isinstance(op, Op)
    assert op.name == "sum(column)"


def test_first() -> None:
    op = ops.First("column")
    assert isinstance(op, Op)
    assert op.name == "first(column)"


def test_null_count() -> None:
    op = ops.NullCount("column")
    assert isinstance(op, Op)
    assert op.name == "null_count(column)"


def test_equality() -> None:
    num_rows_op1 = ops.NumRows()
    num_rows_op2 = ops.NumRows()
    average_op1 = ops.Average("column")
    average_op2 = ops.Average("column")
    minimum_op1 = ops.Minimum("column")
    minimum_op2 = ops.Minimum("column")
    maximum_op1 = ops.Maximum("column")
    maximum_op2 = ops.Maximum("column")
    sum_op1 = ops.Sum("column")
    sum_op2 = ops.Sum("column")
    first_op1 = ops.First("column")
    first_op2 = ops.First("column")
    null_count_op1 = ops.NullCount("column")
    null_count_op2 = ops.NullCount("column")
    negative_count_op1 = ops.NegativeCount("column")
    negative_count_op2 = ops.NegativeCount("column")

    assert num_rows_op1 == num_rows_op2
    assert average_op1 == average_op2
    assert minimum_op1 == minimum_op2
    assert maximum_op1 == maximum_op2
    assert sum_op1 == sum_op2
    assert first_op1 == first_op2
    assert null_count_op1 == null_count_op2
    assert negative_count_op1 == negative_count_op2


def test_dedup_ops() -> None:
    num_rows_op = ops.NumRows()
    average_op = ops.Average("column")

    assert {num_rows_op, average_op} == {average_op, num_rows_op}
    assert {ops.Average("col"), num_rows_op, average_op} == {average_op, num_rows_op, ops.Average("col")}
    assert {ops.Average("column"), num_rows_op, average_op} == {average_op, num_rows_op}


def test_hashing() -> None:
    num_rows_op = ops.NumRows()
    average_op = ops.Average("column")
    minimum_op = ops.Minimum("column")
    maximum_op = ops.Maximum("column")
    sum_op = ops.Sum("column")
    first_op = ops.First("column")
    null_count_op = ops.NullCount("column")
    negative_count_op = ops.NegativeCount("column")

    assert hash(num_rows_op) == hash(ops.NumRows())
    assert hash(average_op) == hash(ops.Average("column"))
    assert hash(minimum_op) == hash(ops.Minimum("column"))
    assert hash(maximum_op) == hash(ops.Maximum("column"))
    assert hash(sum_op) == hash(ops.Sum("column"))
    assert hash(first_op) == hash(ops.First("column"))
    assert hash(null_count_op) == hash(ops.NullCount("column"))
    assert hash(negative_count_op) == hash(ops.NegativeCount("column"))


def test_average_not_equal() -> None:
    average_op1 = ops.Average("column1")
    average_op2 = ops.Average("column2")
    assert average_op1 != average_op2
    assert average_op1 != 1


# Test missing classes: Variance and NegativeCount
def test_variance() -> None:
    op = ops.Variance("column")
    assert isinstance(op, Op)
    assert isinstance(op, SqlOp)
    assert op.name == "variance(column)"
    # Test correct equality behavior
    op1 = ops.Variance("column")
    op2 = ops.Variance("column")
    assert op1 == op2  # Same column, should be equal


def test_variance_not_equal_to_sum() -> None:
    # Test that Variance is correctly not equal to Sum (bug is fixed)
    variance_op = ops.Variance("test_col")
    sum_op = ops.Sum("test_col")
    sum_op_different = ops.Sum("different_col")

    # Variance should not be equal to Sum, even with same column
    assert variance_op != sum_op  # Different op types
    assert variance_op != sum_op_different  # Different op types and columns


def test_negative_count() -> None:
    op = ops.NegativeCount("column")
    assert isinstance(op, Op)
    assert isinstance(op, SqlOp)
    assert op.name == "negative_count(column)"


# Test SQL properties for all ops
def test_num_rows_sql_properties() -> None:
    op = ops.NumRows()
    assert op.prefix is not None
    assert len(op.prefix) > 0
    assert op.sql_col == f"{op.prefix}_{op.name}"


def test_average_sql_properties() -> None:
    op = ops.Average("test_col")
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_{op.name}"


def test_minimum_sql_properties() -> None:
    op = ops.Minimum("test_col")
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_{op.name}"


def test_maximum_sql_properties() -> None:
    op = ops.Maximum("test_col")
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_{op.name}"


def test_sum_sql_properties() -> None:
    op = ops.Sum("test_col")
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_{op.name}"


def test_variance_sql_properties() -> None:
    op = ops.Variance("test_col")
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_{op.name}"


def test_first_sql_properties() -> None:
    op = ops.First("test_col")
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_{op.name}"


def test_null_count_sql_properties() -> None:
    op = ops.NullCount("test_col")
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_{op.name}"


def test_negative_count_sql_properties() -> None:
    op = ops.NegativeCount("test_col")
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_{op.name}"


# Test string representations
def test_num_rows_string_repr() -> None:
    op = ops.NumRows()
    assert str(op) == "num_rows()"
    assert repr(op) == "num_rows()"


def test_average_string_repr() -> None:
    op = ops.Average("test_col")
    assert str(op) == "average(test_col)"
    assert repr(op) == "average(test_col)"


def test_minimum_string_repr() -> None:
    op = ops.Minimum("test_col")
    assert str(op) == "minimum(test_col)"
    assert repr(op) == "minimum(test_col)"


def test_maximum_string_repr() -> None:
    op = ops.Maximum("test_col")
    assert str(op) == "maximum(test_col)"
    assert repr(op) == "maximum(test_col)"


def test_sum_string_repr() -> None:
    op = ops.Sum("test_col")
    assert str(op) == "sum(test_col)"
    assert repr(op) == "sum(test_col)"


def test_variance_string_repr() -> None:
    op = ops.Variance("test_col")
    assert str(op) == "variance(test_col)"
    assert repr(op) == "variance(test_col)"


def test_first_string_repr() -> None:
    op = ops.First("test_col")
    assert str(op) == "first(test_col)"
    assert repr(op) == "first(test_col)"


def test_null_count_string_repr() -> None:
    op = ops.NullCount("test_col")
    assert str(op) == "null_count(test_col)"
    assert repr(op) == "null_count(test_col)"


def test_negative_count_string_repr() -> None:
    op = ops.NegativeCount("test_col")
    assert str(op) == "negative_count(test_col)"
    assert repr(op) == "negative_count(test_col)"


# Test clear functionality
def test_clear_functionality() -> None:
    op = ops.NumRows()

    # Initially should raise error
    with pytest.raises(DQXError):
        op.value()

    # Assign a value
    op.assign(5.0)
    assert op.value() == 5.0

    # Clear and test again
    op.clear()
    with pytest.raises(DQXError):
        op.value()


def test_clear_functionality_all_ops() -> None:
    ops_list = [
        ops.Average("col"),
        ops.Minimum("col"),
        ops.Maximum("col"),
        ops.Sum("col"),
        ops.Variance("col"),
        ops.First("col"),
        ops.NullCount("col"),
        ops.NegativeCount("col"),
    ]

    for op in ops_list:
        # Initially should raise error
        with pytest.raises(DQXError):
            op.value()

        # Assign a value
        op.assign(10.0)
        assert op.value() == 10.0

        # Clear and test again
        op.clear()
        with pytest.raises(DQXError):
            op.value()


# Test edge cases for equality comparisons
def test_equality_edge_cases() -> None:
    # Test NumRows equality with non-NumRows
    num_rows = ops.NumRows()
    assert num_rows != "not an op"
    assert num_rows != 42
    assert num_rows != ops.Average("col")

    # Test Average equality with non-Average
    average = ops.Average("col")
    assert average != "not an op"
    assert average != 42
    assert average != ops.NumRows()

    # Test Minimum equality with non-Minimum
    minimum = ops.Minimum("col")
    assert minimum != "not an op"
    assert minimum != 42

    # Test Maximum equality with non-Maximum
    maximum = ops.Maximum("col")
    assert maximum != "not an op"
    assert maximum != 42

    # Test Sum equality with non-Sum
    sum_op = ops.Sum("col")
    assert sum_op != "not an op"
    assert sum_op != 42

    # Test Variance equality with non-Variance
    variance = ops.Variance("col")
    assert variance != "not an op"
    assert variance != 42
    assert variance != ops.Sum("col")  # Different op type

    # Test First equality with non-First
    first = ops.First("col")
    assert first != "not an op"
    assert first != 42

    # Test NullCount equality with non-NullCount
    null_count = ops.NullCount("col")
    assert null_count != "not an op"
    assert null_count != 42

    # Test NegativeCount equality with non-NegativeCount
    negative_count = ops.NegativeCount("col")
    assert negative_count != "not an op"
    assert negative_count != 42


# Test different column names for inequality
def test_inequality_different_columns() -> None:
    # Test all ops with different columns
    assert ops.Average("col1") != ops.Average("col2")
    assert ops.Minimum("col1") != ops.Minimum("col2")
    assert ops.Maximum("col1") != ops.Maximum("col2")
    assert ops.Sum("col1") != ops.Sum("col2")
    assert ops.Variance("col1") != ops.Variance("col2")  # Works correctly now
    assert ops.First("col1") != ops.First("col2")
    assert ops.NullCount("col1") != ops.NullCount("col2")
    assert ops.NegativeCount("col1") != ops.NegativeCount("col2")


# Test hash consistency for ops with different columns
def test_hash_different_columns() -> None:
    # Hashes should be different for different columns
    assert hash(ops.Average("col1")) != hash(ops.Average("col2"))
    assert hash(ops.Minimum("col1")) != hash(ops.Minimum("col2"))
    assert hash(ops.Maximum("col1")) != hash(ops.Maximum("col2"))
    assert hash(ops.Sum("col1")) != hash(ops.Sum("col2"))
    assert hash(ops.Variance("col1")) != hash(ops.Variance("col2"))
    assert hash(ops.First("col1")) != hash(ops.First("col2"))
    assert hash(ops.NullCount("col1")) != hash(ops.NullCount("col2"))
    assert hash(ops.NegativeCount("col1")) != hash(ops.NegativeCount("col2"))


def test_duplicate_count() -> None:
    op = ops.DuplicateCount(["column1", "column2"])
    assert isinstance(op, Op)
    assert isinstance(op, SqlOp)
    # Columns should be sorted
    assert op.name == "duplicate_count(column1,column2)"

    # Test empty columns raises error
    with pytest.raises(ValueError, match="DuplicateCount requires at least one column"):
        ops.DuplicateCount([])

    # Test value assignment
    with pytest.raises(DQXError, match="DuplicateCount op has not been collected yet!"):
        op.value()

    op.assign(42.0)
    assert op.value() == pytest.approx(42.0)

    # Test clear functionality
    op.clear()
    with pytest.raises(DQXError):
        op.value()


def test_duplicate_count_equality() -> None:
    op1 = ops.DuplicateCount(["col1", "col2"])
    op2 = ops.DuplicateCount(["col1", "col2"])
    op3 = ops.DuplicateCount(["col1"])
    op4 = ops.DuplicateCount(["col2", "col1"])  # Different order

    assert op1 == op2
    assert op1 != op3
    assert op1 == op4  # Should be equal after sorting
    assert op1 != "not an op"
    assert op1 != 42


def test_duplicate_count_hashing() -> None:
    op1 = ops.DuplicateCount(["col1", "col2"])
    op2 = ops.DuplicateCount(["col1", "col2"])
    op3 = ops.DuplicateCount(["col1"])
    op4 = ops.DuplicateCount(["col2", "col1"])  # Different order

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)
    assert hash(op1) == hash(op4)  # Should have same hash after sorting

    # Test deduplication in sets
    assert {op1, op2, op4} == {op1}


def test_duplicate_count_string_repr() -> None:
    op = ops.DuplicateCount(["col3", "col1", "col2"])
    # Should be sorted alphabetically
    assert str(op) == "duplicate_count(col1,col2,col3)"
    assert repr(op) == "duplicate_count(col1,col2,col3)"


def test_duplicate_count_sql_properties() -> None:
    op = ops.DuplicateCount(["test_col"])
    assert op.prefix is not None
    assert len(op.prefix) > 0
    assert op.sql_col == f"{op.prefix}_{op.name}"
