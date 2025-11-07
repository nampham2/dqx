"""Test cases for operations module."""

import pytest

from dqx import ops
from dqx.common import DQXError


def test_num_rows() -> None:
    """Test NumRows operation basic functionality."""
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

    # Test hash
    assert hash(op) == hash(op2)

    # Test string representation
    assert str(op) == "num_rows()"
    assert repr(op) == "num_rows()"


def test_average() -> None:
    """Test Average operation basic functionality."""
    op = ops.Average("price")
    assert op.name == "average(price)"
    assert op.column == "price"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_average(price)"

    # Test value assignment
    with pytest.raises(DQXError, match="Average op has not been collected yet!"):
        op.value()

    op.assign(25.5)
    assert op.value() == pytest.approx(25.5)

    # Test clear
    op.clear()
    with pytest.raises(DQXError):
        op.value()

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
    """Test Minimum operation basic functionality."""
    op = ops.Minimum("score")
    assert op.name == "minimum(score)"
    assert op.column == "score"

    # Test value handling
    with pytest.raises(DQXError):
        op.value()

    op.assign(10.0)
    assert op.value() == pytest.approx(10.0)

    op.clear()
    with pytest.raises(DQXError):
        op.value()


def test_maximum() -> None:
    """Test Maximum operation basic functionality."""
    op = ops.Maximum("score")
    assert op.name == "maximum(score)"
    assert op.column == "score"

    # Test value handling
    with pytest.raises(DQXError):
        op.value()

    op.assign(100.0)
    assert op.value() == pytest.approx(100.0)

    op.clear()
    with pytest.raises(DQXError):
        op.value()


def test_sum() -> None:
    """Test Sum operation basic functionality."""
    op = ops.Sum("amount")
    assert op.name == "sum(amount)"
    assert op.column == "amount"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_sum(amount)"

    # Test value assignment
    with pytest.raises(DQXError, match="Sum op has not been collected yet!"):
        op.value()

    op.assign(1000.0)
    assert op.value() == pytest.approx(1000.0)

    # Test clear
    op.clear()
    with pytest.raises(DQXError):
        op.value()

    # Test equality
    op2 = ops.Sum("amount")
    op3 = ops.Sum("quantity")
    assert op == op2
    assert op != op3

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "sum(amount)"
    assert repr(op) == "sum(amount)"


def test_variance() -> None:
    """Test Variance operation basic functionality."""
    op = ops.Variance("values")
    assert op.name == "variance(values)"
    assert op.column == "values"

    # Test value handling
    with pytest.raises(DQXError):
        op.value()

    op.assign(25.5)
    assert op.value() == pytest.approx(25.5)


def test_first() -> None:
    """Test First operation basic functionality."""
    op = ops.First("timestamp")
    assert op.name == "first(timestamp)"
    assert op.column == "timestamp"

    # Test value handling
    with pytest.raises(DQXError):
        op.value()

    op.assign(123456789.0)
    assert op.value() == pytest.approx(123456789.0)


def test_null_count() -> None:
    """Test NullCount operation basic functionality."""
    op = ops.NullCount("email")
    assert op.name == "null_count(email)"
    assert op.column == "email"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_null_count(email)"

    # Test value assignment
    with pytest.raises(DQXError, match="NullCount op has not been collected yet!"):
        op.value()

    op.assign(5.0)
    assert op.value() == pytest.approx(5.0)

    # Test clear
    op.clear()
    with pytest.raises(DQXError):
        op.value()

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
    """Test NegativeCount operation basic functionality."""
    op = ops.NegativeCount("balance")
    assert op.name == "negative_count(balance)"
    assert op.column == "balance"

    # Test value handling
    with pytest.raises(DQXError):
        op.value()

    op.assign(3.0)
    assert op.value() == pytest.approx(3.0)


def test_duplicate_count() -> None:
    """Test DuplicateCount operation basic functionality."""
    # Test single column
    op = ops.DuplicateCount(["email"])
    assert op.name == "duplicate_count(email)"
    assert op.columns == ["email"]  # Should be sorted
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

    # Test value handling
    with pytest.raises(DQXError, match="DuplicateCount op has not been collected yet!"):
        op.value()

    op.assign(10.0)
    assert op.value() == pytest.approx(10.0)

    op.clear()
    with pytest.raises(DQXError):
        op.value()

    # Test empty columns raises error
    with pytest.raises(ValueError, match="DuplicateCount requires at least one column"):
        ops.DuplicateCount([])


def test_count_values() -> None:
    """Test CountValues operation basic functionality."""
    # Test single integer value
    op = ops.CountValues("status", 1)
    assert op.name == "count_values(status,1)"
    assert op.column == "status"
    assert op.values == 1
    assert op._values == [1]
    assert op._is_single is True

    # Test single string value
    op2 = ops.CountValues("status", "active")
    assert op2.name == "count_values(status,active)"
    assert op2.values == "active"
    assert op2._values == ["active"]
    assert op2._is_single is True

    # Test single boolean value
    op3 = ops.CountValues("is_valid", True)
    assert op3.name == "count_values(is_valid,True)"
    assert op3.values is True  # Use 'is' for boolean comparison
    assert op3._values == [True]
    assert op3._is_single is True

    # Test list of integers
    op4 = ops.CountValues("type_id", [1, 2, 3])
    assert op4.name == "count_values(type_id,[1,2,3])"
    assert op4.values == [1, 2, 3]
    assert op4._values == [1, 2, 3]
    assert op4._is_single is False

    # Test list of strings
    op5 = ops.CountValues("status", ["pending", "active"])
    assert op5.name == "count_values(status,[pending,active])"
    assert op5.values == ["pending", "active"]
    assert op5._values == ["pending", "active"]
    assert op5._is_single is False

    # Test sql_col (uses hash for uniqueness)
    assert op.sql_col.startswith(f"{op.prefix}_count_values_status_")

    # Test value assignment
    with pytest.raises(DQXError, match="CountValues op has not been collected yet!"):
        op.value()

    op.assign(42.0)
    assert op.value() == pytest.approx(42.0)

    # Test clear
    op.clear()
    with pytest.raises(DQXError):
        op.value()

    # Test equality (distinguishes True from 1, False from 0)
    bool_op1 = ops.CountValues("col", True)
    bool_op2 = ops.CountValues("col", True)
    int_op = ops.CountValues("col", 1)
    assert bool_op1 == bool_op2
    assert bool_op1 != int_op  # True != 1 for our purposes

    # Test hash
    assert hash(bool_op1) == hash(bool_op2)
    assert hash(bool_op1) != hash(int_op)

    # Test string representation
    assert str(op) == "count_values(status,1)"
    assert repr(op) == "count_values(status,1)"

    # Test empty list raises error
    with pytest.raises(ValueError, match="CountValues requires at least one value"):
        ops.CountValues("col", [])

    # Test mixed types raise error
    with pytest.raises(ValueError, match="CountValues list must contain all integers or all strings"):
        ops.CountValues("col", [1, "two"])  # type: ignore

    # Test booleans in lists raise error
    with pytest.raises(ValueError, match="CountValues list must contain all integers or all strings"):
        ops.CountValues("col", [True, False])  # type: ignore

    # Test invalid type raises error
    with pytest.raises(ValueError, match="CountValues accepts"):
        ops.CountValues("col", 3.14)  # type: ignore


def test_unique_count() -> None:
    """Test UniqueCount operation basic functionality."""
    op = ops.UniqueCount("product_id")
    assert op.name == "unique_count(product_id)"
    assert op.column == "product_id"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_unique_count(product_id)"

    # Test value assignment
    with pytest.raises(DQXError, match="UniqueCount op has not been collected yet!"):
        op.value()

    op.assign(42.0)
    assert op.value() == pytest.approx(42.0)

    # Test clear
    op.clear()
    with pytest.raises(DQXError):
        op.value()

    # Test equality
    op2 = ops.UniqueCount("product_id")
    op3 = ops.UniqueCount("user_id")
    assert op == op2
    assert op != op3
    assert op != ops.Average("product_id")

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "unique_count(product_id)"
    assert repr(op) == "unique_count(product_id)"


def test_op_protocol() -> None:
    """Test that all ops implement the Op protocol."""
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
    assert isinstance(ops.UniqueCount("col"), ops.Op)


def test_sql_op_protocol() -> None:
    """Test that all ops implement the SqlOp protocol."""
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
    assert isinstance(ops.UniqueCount("col"), ops.SqlOp)


def test_sql_op_properties() -> None:
    """Test that all SqlOp implementations have required properties."""
    sql_ops = [
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
        ops.UniqueCount("col"),
    ]

    for op in sql_ops:
        # All SqlOps should have these properties
        assert hasattr(op, "name")
        assert hasattr(op, "prefix")
        assert hasattr(op, "sql_col")
        assert hasattr(op, "value")
        assert hasattr(op, "assign")
        assert hasattr(op, "clear")

        # Check that sql_col follows expected pattern
        assert op.prefix in op.sql_col
        # CountValues uses a hash in sql_col for uniqueness
        if not isinstance(op, ops.CountValues):
            assert op.name in op.sql_col


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
        ops.CountValues("col", 1),
        ops.UniqueCount("col"),
    ]

    for op in ops_to_test:
        # Should raise when no value assigned
        with pytest.raises(DQXError):
            op.value()

        # Assign a value
        op.assign(123.45)
        assert op.value() == pytest.approx(123.45)

        # Clear the value
        op.clear()

        # Should raise again after clearing
        with pytest.raises(DQXError):
            op.value()


def test_op_match_args() -> None:
    """Test that ops with columns have proper __match_args__ for pattern matching."""
    # These ops should have match_args for their column and parameters
    assert ops.Average.__match_args__ == ("column", "parameters")
    assert ops.Minimum.__match_args__ == ("column", "parameters")
    assert ops.Maximum.__match_args__ == ("column", "parameters")
    assert ops.Sum.__match_args__ == ("column", "parameters")
    assert ops.Variance.__match_args__ == ("column", "parameters")
    assert ops.First.__match_args__ == ("column", "parameters")
    assert ops.NullCount.__match_args__ == ("column", "parameters")
    assert ops.NegativeCount.__match_args__ == ("column", "parameters")
    assert ops.UniqueCount.__match_args__ == ("column", "parameters")

    # DuplicateCount has columns (plural) and parameters
    assert ops.DuplicateCount.__match_args__ == ("columns", "parameters")

    # CountValues has column, values, and parameters
    assert ops.CountValues.__match_args__ == ("column", "values", "parameters")

    # NumRows has only parameters
    assert ops.NumRows.__match_args__ == ("parameters",)

    # CustomSQL has sql_expression and parameters
    assert ops.CustomSQL.__match_args__ == ("sql_expression", "parameters")


def test_ops_hash_includes_parameters() -> None:
    """Test that hash function includes parameters for all operations."""
    # Test NumRows
    op1 = ops.NumRows()
    op2 = ops.NumRows(parameters={"region": "US"})
    op3 = ops.NumRows(parameters={"region": "EU"})
    op4 = ops.NumRows(parameters={"region": "US"})

    assert hash(op1) != hash(op2)  # Different parameters
    assert hash(op2) != hash(op3)  # Different parameter values
    assert hash(op2) == hash(op4)  # Same parameters

    # Test Average
    avg1 = ops.Average("price")
    avg2 = ops.Average("price", parameters={"category": "electronics"})
    avg3 = ops.Average("price", parameters={"category": "books"})
    avg4 = ops.Average("price", parameters={"category": "electronics"})

    assert hash(avg1) != hash(avg2)  # Different parameters
    assert hash(avg2) != hash(avg3)  # Different parameter values
    assert hash(avg2) == hash(avg4)  # Same parameters

    # Test CustomSQL
    sql1 = ops.CustomSQL("COUNT(*)")
    sql2 = ops.CustomSQL("COUNT(*)", parameters={"status": "active"})
    sql3 = ops.CustomSQL("COUNT(*)", parameters={"status": "inactive"})
    sql4 = ops.CustomSQL("COUNT(*)", parameters={"status": "active"})

    assert hash(sql1) != hash(sql2)  # Different parameters
    assert hash(sql2) != hash(sql3)  # Different parameter values
    assert hash(sql2) == hash(sql4)  # Same parameters

    # Test Minimum
    min1 = ops.Minimum("score")
    min2 = ops.Minimum("score", parameters={"year": 2023})
    min3 = ops.Minimum("score", parameters={"year": 2024})
    min4 = ops.Minimum("score", parameters={"year": 2023})

    assert hash(min1) != hash(min2)  # Different parameters
    assert hash(min2) != hash(min3)  # Different parameter values
    assert hash(min2) == hash(min4)  # Same parameters

    # Test Maximum
    max1 = ops.Maximum("score")
    max2 = ops.Maximum("score", parameters={"level": "beginner"})
    max3 = ops.Maximum("score", parameters={"level": "advanced"})
    max4 = ops.Maximum("score", parameters={"level": "beginner"})

    assert hash(max1) != hash(max2)  # Different parameters
    assert hash(max2) != hash(max3)  # Different parameter values
    assert hash(max2) == hash(max4)  # Same parameters

    # Test Sum
    sum1 = ops.Sum("amount")
    sum2 = ops.Sum("amount", parameters={"currency": "USD"})
    sum3 = ops.Sum("amount", parameters={"currency": "EUR"})
    sum4 = ops.Sum("amount", parameters={"currency": "USD"})

    assert hash(sum1) != hash(sum2)  # Different parameters
    assert hash(sum2) != hash(sum3)  # Different parameter values
    assert hash(sum2) == hash(sum4)  # Same parameters

    # Test Variance
    var1 = ops.Variance("values")
    var2 = ops.Variance("values", parameters={"experiment": "A"})
    var3 = ops.Variance("values", parameters={"experiment": "B"})
    var4 = ops.Variance("values", parameters={"experiment": "A"})

    assert hash(var1) != hash(var2)  # Different parameters
    assert hash(var2) != hash(var3)  # Different parameter values
    assert hash(var2) == hash(var4)  # Same parameters

    # Test First
    first1 = ops.First("timestamp")
    first2 = ops.First("timestamp", parameters={"source": "web"})
    first3 = ops.First("timestamp", parameters={"source": "mobile"})
    first4 = ops.First("timestamp", parameters={"source": "web"})

    assert hash(first1) != hash(first2)  # Different parameters
    assert hash(first2) != hash(first3)  # Different parameter values
    assert hash(first2) == hash(first4)  # Same parameters

    # Test NullCount
    null1 = ops.NullCount("email")
    null2 = ops.NullCount("email", parameters={"user_type": "free"})
    null3 = ops.NullCount("email", parameters={"user_type": "premium"})
    null4 = ops.NullCount("email", parameters={"user_type": "free"})

    assert hash(null1) != hash(null2)  # Different parameters
    assert hash(null2) != hash(null3)  # Different parameter values
    assert hash(null2) == hash(null4)  # Same parameters

    # Test NegativeCount
    neg1 = ops.NegativeCount("balance")
    neg2 = ops.NegativeCount("balance", parameters={"account_type": "checking"})
    neg3 = ops.NegativeCount("balance", parameters={"account_type": "savings"})
    neg4 = ops.NegativeCount("balance", parameters={"account_type": "checking"})

    assert hash(neg1) != hash(neg2)  # Different parameters
    assert hash(neg2) != hash(neg3)  # Different parameter values
    assert hash(neg2) == hash(neg4)  # Same parameters

    # Test UniqueCount
    unique1 = ops.UniqueCount("product_id")
    unique2 = ops.UniqueCount("product_id", parameters={"store": "online"})
    unique3 = ops.UniqueCount("product_id", parameters={"store": "retail"})
    unique4 = ops.UniqueCount("product_id", parameters={"store": "online"})

    assert hash(unique1) != hash(unique2)  # Different parameters
    assert hash(unique2) != hash(unique3)  # Different parameter values
    assert hash(unique2) == hash(unique4)  # Same parameters

    # Test DuplicateCount
    dup1 = ops.DuplicateCount(["email"])
    dup2 = ops.DuplicateCount(["email"], parameters={"domain": "gmail.com"})
    dup3 = ops.DuplicateCount(["email"], parameters={"domain": "yahoo.com"})
    dup4 = ops.DuplicateCount(["email"], parameters={"domain": "gmail.com"})

    assert hash(dup1) != hash(dup2)  # Different parameters
    assert hash(dup2) != hash(dup3)  # Different parameter values
    assert hash(dup2) == hash(dup4)  # Same parameters

    # Test CountValues
    count1 = ops.CountValues("status", "active")
    count2 = ops.CountValues("status", "active", parameters={"priority": "high"})
    count3 = ops.CountValues("status", "active", parameters={"priority": "low"})
    count4 = ops.CountValues("status", "active", parameters={"priority": "high"})

    assert hash(count1) != hash(count2)  # Different parameters
    assert hash(count2) != hash(count3)  # Different parameter values
    assert hash(count2) == hash(count4)  # Same parameters

    # Test with multiple parameters
    multi1 = ops.Average("price", parameters={"category": "electronics", "year": 2023})
    multi2 = ops.Average("price", parameters={"year": 2023, "category": "electronics"})
    multi3 = ops.Average("price", parameters={"category": "electronics", "year": 2024})

    # Order shouldn't matter due to sorted()
    assert hash(multi1) == hash(multi2)
    assert hash(multi1) != hash(multi3)  # Different parameter values

    # Test that operations can be used in sets/dicts correctly
    op_set = {
        ops.NumRows(),
        ops.NumRows(parameters={"region": "US"}),
        ops.NumRows(parameters={"region": "EU"}),
        ops.NumRows(parameters={"region": "US"}),  # Duplicate
    }
    assert len(op_set) == 3  # Only 3 unique operations

    # Test in dictionary
    op_dict = {
        ops.Average("price"): "no params",
        ops.Average("price", parameters={"cat": "A"}): "param A",
        ops.Average("price", parameters={"cat": "B"}): "param B",
    }
    assert len(op_dict) == 3
    assert op_dict[ops.Average("price", parameters={"cat": "A"})] == "param A"
