"""Test cases for SQL dialect functionality.

This module tests the SQL dialect abstraction layer and
concrete dialect implementations.
"""

from datetime import date

import pytest

from dqx import ops
from dqx.common import DQXError
from dqx.dialect import (
    BigQueryDialect,
    DuckDBDialect,
    auto_register,
    build_cte_query,
    get_dialect,
    register_dialect,
)
from dqx.models import BatchCTEData, ResultKey


def test_build_cte_query() -> None:
    """Test the standalone CTE query builder."""
    cte_sql = "SELECT * FROM sales WHERE date = '2024-01-01'"
    expressions = [
        "COUNT(*) AS total_count",
        "AVG(amount) AS avg_amount",
        "SUM(quantity) AS total_quantity",
    ]

    query = build_cte_query(cte_sql, expressions)

    expected = (
        "WITH source AS (SELECT * FROM sales WHERE date = '2024-01-01') "
        "SELECT COUNT(*) AS total_count, AVG(amount) AS avg_amount, "
        "SUM(quantity) AS total_quantity FROM source"
    )
    assert query == expected


def test_build_cte_query_empty_expressions() -> None:
    """Test CTE query builder with empty expressions."""
    with pytest.raises(ValueError, match="No SELECT expressions provided"):
        build_cte_query("SELECT * FROM table", [])


def test_duckdb_dialect_translate_sql_op() -> None:
    """Test DuckDB dialect SQL translation for various ops."""
    dialect = DuckDBDialect()

    # Test NumRows
    op_num = ops.NumRows()
    sql = dialect.translate_sql_op(op_num)
    assert sql == f"CAST(COUNT(*) AS DOUBLE) AS '{op_num.sql_col}'"

    # Test Average
    op_avg = ops.Average("price")
    sql = dialect.translate_sql_op(op_avg)
    assert sql == f"CAST(AVG(price) AS DOUBLE) AS '{op_avg.sql_col}'"

    # Test NullCount
    op_null = ops.NullCount("email")
    sql = dialect.translate_sql_op(op_null)
    assert sql == f"CAST(COUNT_IF(email IS NULL) AS DOUBLE) AS '{op_null.sql_col}'"

    # Test DuplicateCount with single column
    op_dup1 = ops.DuplicateCount(["user_id"])
    sql = dialect.translate_sql_op(op_dup1)
    assert sql == f"CAST(COUNT(*) - COUNT(DISTINCT user_id) AS DOUBLE) AS '{op_dup1.sql_col}'"

    # Test DuplicateCount with multiple columns
    op_dup2 = ops.DuplicateCount(["user_id", "email"])
    sql = dialect.translate_sql_op(op_dup2)
    # Note: columns are sorted in the op
    assert sql == f"CAST(COUNT(*) - COUNT(DISTINCT (email, user_id)) AS DOUBLE) AS '{op_dup2.sql_col}'"


def test_bigquery_dialect_translate_sql_op() -> None:
    """Test BigQuery dialect SQL translation."""
    dialect = BigQueryDialect()

    # Test NumRows - uses FLOAT64 instead of DOUBLE
    op_num = ops.NumRows()
    sql = dialect.translate_sql_op(op_num)
    assert sql == f"CAST(COUNT(*) AS FLOAT64) AS `{op_num.sql_col}`"

    # Test Variance - uses VAR_SAMP
    op_var = ops.Variance("score")
    sql = dialect.translate_sql_op(op_var)
    assert sql == f"CAST(VAR_SAMP(score) AS FLOAT64) AS `{op_var.sql_col}`"

    # Test First - uses MIN for deterministic result
    op_first = ops.First("timestamp")
    sql = dialect.translate_sql_op(op_first)
    assert sql == f"CAST(MIN(timestamp) AS FLOAT64) AS `{op_first.sql_col}`"

    # Test NullCount - uses COUNTIF
    op_null = ops.NullCount("phone")
    sql = dialect.translate_sql_op(op_null)
    assert sql == f"CAST(COUNTIF(phone IS NULL) AS FLOAT64) AS `{op_null.sql_col}`"


def test_translate_count_values() -> None:
    """Test CountValues translation for both dialects."""
    from dqx import ops
    from dqx.dialect import BigQueryDialect, DuckDBDialect

    # Test DuckDB with single integer value
    dialect_duck = DuckDBDialect()
    op_int = ops.CountValues("status", 1)
    sql_int = dialect_duck.translate_sql_op(op_int)
    assert sql_int == f"CAST(COUNT_IF(status = 1) AS DOUBLE) AS '{op_int.sql_col}'"

    # Test DuckDB with single string value
    op_str = ops.CountValues("category", "active")
    sql_str = dialect_duck.translate_sql_op(op_str)
    assert sql_str == f"CAST(COUNT_IF(category = 'active') AS DOUBLE) AS '{op_str.sql_col}'"

    # Test DuckDB with single boolean value
    op_bool_true = ops.CountValues("is_active", True)
    sql_bool_true = dialect_duck.translate_sql_op(op_bool_true)
    assert sql_bool_true == f"CAST(COUNT_IF(is_active = TRUE) AS DOUBLE) AS '{op_bool_true.sql_col}'"

    op_bool_false = ops.CountValues("is_verified", False)
    sql_bool_false = dialect_duck.translate_sql_op(op_bool_false)
    assert sql_bool_false == f"CAST(COUNT_IF(is_verified = FALSE) AS DOUBLE) AS '{op_bool_false.sql_col}'"

    # Test DuckDB with string containing quotes
    op_quote = ops.CountValues("name", "O'Brien")
    sql_quote = dialect_duck.translate_sql_op(op_quote)
    assert sql_quote == f"CAST(COUNT_IF(name = 'O''Brien') AS DOUBLE) AS '{op_quote.sql_col}'"

    # Test DuckDB with string containing backslashes
    op_backslash = ops.CountValues("path", "C:\\Users\\test")
    sql_backslash = dialect_duck.translate_sql_op(op_backslash)
    assert sql_backslash == f"CAST(COUNT_IF(path = 'C:\\\\Users\\\\test') AS DOUBLE) AS '{op_backslash.sql_col}'"

    # Test DuckDB with multiple integer values
    op_ints = ops.CountValues("type_id", [1, 2, 3])
    sql_ints = dialect_duck.translate_sql_op(op_ints)
    assert sql_ints == f"CAST(COUNT_IF(type_id IN (1, 2, 3)) AS DOUBLE) AS '{op_ints.sql_col}'"

    # Test DuckDB with multiple string values
    op_strs = ops.CountValues("status", ["active", "pending", "completed"])
    sql_strs = dialect_duck.translate_sql_op(op_strs)
    assert sql_strs == f"CAST(COUNT_IF(status IN ('active', 'pending', 'completed')) AS DOUBLE) AS '{op_strs.sql_col}'"

    # Test BigQuery
    dialect_bq = BigQueryDialect()
    sql_bq = dialect_bq.translate_sql_op(op_int)
    assert sql_bq == f"CAST(COUNTIF(status = 1) AS FLOAT64) AS `{op_int.sql_col}`"

    # Test BigQuery with IN clause
    sql_bq_in = dialect_bq.translate_sql_op(op_ints)
    assert sql_bq_in == f"CAST(COUNTIF(type_id IN (1, 2, 3)) AS FLOAT64) AS `{op_ints.sql_col}`"

    # Test BigQuery with boolean values
    sql_bq_bool_true = dialect_bq.translate_sql_op(op_bool_true)
    assert sql_bq_bool_true == f"CAST(COUNTIF(is_active = TRUE) AS FLOAT64) AS `{op_bool_true.sql_col}`"

    sql_bq_bool_false = dialect_bq.translate_sql_op(op_bool_false)
    assert sql_bq_bool_false == f"CAST(COUNTIF(is_verified = FALSE) AS FLOAT64) AS `{op_bool_false.sql_col}`"


def test_dialect_unsupported_op() -> None:
    """Test error handling for unsupported ops."""

    class UnsupportedOp:
        """Mock unsupported op."""

        pass

    dialect = DuckDBDialect()
    with pytest.raises(ValueError, match="Unsupported SqlOp type: UnsupportedOp"):
        dialect.translate_sql_op(UnsupportedOp())  # type: ignore


def test_dialect_registry() -> None:
    """Test dialect registration and retrieval."""
    # Test retrieving built-in dialects
    duckdb = get_dialect("duckdb")
    assert isinstance(duckdb, DuckDBDialect)
    assert duckdb.name == "duckdb"

    bigquery = get_dialect("bigquery")
    assert isinstance(bigquery, BigQueryDialect)
    assert bigquery.name == "bigquery"

    # Test error for non-existent dialect
    with pytest.raises(DQXError, match="Dialect 'postgresql' not found"):
        get_dialect("postgresql")


def test_register_dialect(isolated_dialect_registry: dict[str, type]) -> None:
    """Test manual dialect registration."""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        pass

    class TestDialect:
        name = "test_dialect"

        def translate_sql_op(self, op: ops.SqlOp) -> str:
            return f"TEST SQL FOR {op.name}"

        def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
            return build_cte_query(cte_sql, select_expressions)

        def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
            # Simple implementation for testing
            return "TEST BATCH CTE QUERY"

    # Register the dialect
    register_dialect("test_dialect", TestDialect)  # type: ignore

    # Retrieve and test
    dialect = get_dialect("test_dialect")
    assert isinstance(dialect, TestDialect)
    assert dialect.name == "test_dialect"

    # Test duplicate registration error
    with pytest.raises(ValueError, match="Dialect 'test_dialect' is already registered"):
        register_dialect("test_dialect", TestDialect)  # type: ignore


def test_auto_register_decorator(isolated_dialect_registry: dict[str, type]) -> None:
    """Test the auto_register decorator."""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        pass

    @auto_register  # type: ignore
    class AutoTestDialect:
        name = "auto_test"

        def translate_sql_op(self, op: ops.SqlOp) -> str:
            return "AUTO TEST SQL"

        def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
            return build_cte_query(cte_sql, select_expressions)

        def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
            # Simple implementation for testing
            return "AUTO TEST BATCH CTE QUERY"

    # Should be automatically registered
    dialect = get_dialect("auto_test")
    assert isinstance(dialect, AutoTestDialect)
    assert dialect.name == "auto_test"


def test_batch_cte_query_duckdb() -> None:
    """Test DuckDB batch CTE query generation with MAP."""
    dialect = DuckDBDialect()

    # Create test data
    key1 = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})
    key2 = ResultKey(yyyy_mm_dd=date(2024, 1, 2), tags={})

    cte_data = [
        BatchCTEData(
            key=key1,
            cte_sql="SELECT * FROM sales WHERE date = '2024-01-01'",
            ops=[ops.NumRows(), ops.Average("amount")],
        ),
        BatchCTEData(
            key=key2,
            cte_sql="SELECT * FROM sales WHERE date = '2024-01-02'",
            ops=[ops.NumRows(), ops.Sum("amount")],
        ),
    ]

    query = dialect.build_batch_cte_query(cte_data)

    # Check structure
    assert "WITH" in query
    assert "source_2024_01_01_0" in query
    assert "metrics_2024_01_01_0" in query
    assert "source_2024_01_02_1" in query
    assert "metrics_2024_01_02_1" in query
    assert "UNION ALL" in query
    assert "MAP {" in query
    assert "'2024-01-01' as date" in query
    assert "'2024-01-02' as date" in query


def test_batch_cte_query_bigquery() -> None:
    """Test BigQuery batch CTE query generation with STRUCT."""
    dialect = BigQueryDialect()

    key = ResultKey(yyyy_mm_dd=date(2024, 3, 15), tags={})
    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM orders WHERE date = '2024-03-15'",
            ops=[ops.Maximum("price"), ops.Minimum("price")],
        )
    ]

    query = dialect.build_batch_cte_query(cte_data)

    # Check structure
    assert "WITH" in query
    assert "source_2024_03_15_0" in query
    assert "metrics_2024_03_15_0" in query
    assert "STRUCT(" in query
    assert "'2024-03-15' as date" in query
    # BigQuery uses backticks for column aliases
    assert "`" in query


def test_batch_cte_query_empty() -> None:
    """Test batch CTE query with empty data."""
    dialect = DuckDBDialect()

    with pytest.raises(ValueError, match="No CTE data provided"):
        dialect.build_batch_cte_query([])


def test_batch_cte_query_no_ops() -> None:
    """Test batch CTE query with no ops."""
    dialect = DuckDBDialect()

    key = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})
    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM sales",
            ops=[],  # No ops
        )
    ]

    with pytest.raises(ValueError, match="No metrics to compute"):
        dialect.build_batch_cte_query(cte_data)


def test_dialect_build_cte_query_method() -> None:
    """Test that dialect's build_cte_query delegates correctly."""
    dialect = DuckDBDialect()
    cte_sql = "SELECT * FROM users"
    expressions = ["COUNT(*) AS count", "AVG(age) AS avg_age"]

    result = dialect.build_cte_query(cte_sql, expressions)

    # Should match the standalone function
    expected = build_cte_query(cte_sql, expressions)
    assert result == expected


def test_all_ops_covered() -> None:
    """Ensure all SqlOp types are handled by dialects."""
    dialect = DuckDBDialect()

    # List of all ops to test
    test_ops: list[ops.SqlOp] = [
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
        ops.CountValues("col", "test"),
        ops.CountValues("col", True),
        ops.CountValues("col", False),
        ops.CountValues("col", [1, 2, 3]),
        ops.CountValues("col", ["a", "b", "c"]),
    ]

    # All should translate without error
    for op in test_ops:
        sql = dialect.translate_sql_op(op)
        assert sql is not None
        assert op.sql_col in sql
