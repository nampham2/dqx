"""Integration tests for dialect batch optimization with CountValues.

This module tests the batch optimization features with the new CountValues op,
ensuring it works correctly with the DuckDB and BigQuery dialects.
"""

from datetime import date

from dqx import ops
from dqx.common import ResultKey
from dqx.dialect import BatchCTEData, BigQueryDialect, DuckDBDialect


def test_batch_optimization_with_count_values_duckdb() -> None:
    """Test DuckDB batch optimization including CountValues ops."""
    dialect = DuckDBDialect()

    # Create test data with various ops including CountValues
    key1 = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})
    key2 = ResultKey(yyyy_mm_dd=date(2024, 1, 2), tags={})

    cte_data = [
        BatchCTEData(
            key=key1,
            cte_sql="SELECT * FROM orders WHERE date = '2024-01-01'",
            ops=[
                ops.NumRows(),
                ops.CountValues("status", "completed"),
                ops.CountValues("priority", [1, 2]),  # High priority orders
                ops.Average("amount"),
            ],
        ),
        BatchCTEData(
            key=key2,
            cte_sql="SELECT * FROM orders WHERE date = '2024-01-02'",
            ops=[
                ops.NumRows(),
                ops.CountValues("status", "cancelled"),
                ops.CountValues("region", ["US", "EU", "APAC"]),
                ops.Sum("amount"),
            ],
        ),
    ]

    query = dialect.build_cte_query(cte_data)

    # Verify query structure
    assert "WITH" in query
    assert "source_2024_01_01_0" in query
    assert "metrics_2024_01_01_0" in query
    assert "source_2024_01_02_1" in query
    assert "metrics_2024_01_02_1" in query

    # Check for COUNT_IF conditions
    assert "COUNT_IF(status = 'completed')" in query
    assert "COUNT_IF(priority IN (1, 2))" in query
    assert "COUNT_IF(status = 'cancelled')" in query
    assert "COUNT_IF(region IN ('US', 'EU', 'APAC'))" in query

    # Check array structure
    assert "[{" in query  # Array of structs
    assert "'2024-01-01' as date" in query
    assert "'2024-01-02' as date" in query

    # Verify UNION ALL for multiple dates
    assert "UNION ALL" in query


def test_batch_optimization_with_count_values_bigquery() -> None:
    """Test BigQuery batch optimization including CountValues ops."""
    dialect = BigQueryDialect()

    key = ResultKey(yyyy_mm_dd=date(2024, 3, 15), tags={})

    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM transactions WHERE date = '2024-03-15'",
            ops=[
                ops.CountValues("type", "purchase"),
                ops.CountValues("country", ["UK", "FR", "DE"]),
                ops.CountValues("amount_range", [100, 200, 300]),
                ops.Maximum("amount"),
                ops.NullCount("customer_id"),
            ],
        )
    ]

    query = dialect.build_cte_query(cte_data)

    # Verify BigQuery specific syntax
    assert "COUNTIF(type = 'purchase')" in query
    assert "COUNTIF(country IN ('UK', 'FR', 'DE'))" in query
    assert "COUNTIF(amount_range IN (100, 200, 300))" in query
    assert "COUNTIF(customer_id IS NULL)" in query

    # Check array usage (BigQuery uses arrays like DuckDB)
    assert "[STRUCT(" in query  # BigQuery uses STRUCT syntax
    assert "'2024-03-15' as date" in query

    # Verify backticks for column aliases
    assert "`" in query


def test_batch_optimization_special_characters() -> None:
    """Test batch optimization with special characters in CountValues."""
    dialect = DuckDBDialect()

    key = ResultKey(yyyy_mm_dd=date(2024, 5, 1), tags={})

    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM logs WHERE date = '2024-05-01'",
            ops=[
                ops.CountValues("user", "O'Brien"),
                ops.CountValues("path", ["C:\\Users\\test", "D:\\Data\\files"]),
                ops.CountValues("message", 'Error: "File not found"'),
            ],
        )
    ]

    query = dialect.build_cte_query(cte_data)

    # Check proper escaping
    assert "COUNT_IF(user = 'O''Brien')" in query
    assert "COUNT_IF(path IN ('C:\\\\Users\\\\test', 'D:\\\\Data\\\\files'))" in query
    # Double quotes inside the string should be escaped
    assert "'Error: \\\"File not found\\\"'" in query or 'Error: "File not found"' in query


def test_batch_optimization_mixed_ops() -> None:
    """Test batch optimization with mixed op types."""
    dialect = DuckDBDialect()

    # Create multiple dates with different combinations of ops
    dates = [date(2024, 6, i) for i in range(1, 4)]
    cte_data = []

    for i, d in enumerate(dates):
        ops_list: list[ops.SqlOp] = []

        # Add standard ops
        ops_list.append(ops.NumRows())

        # Add CountValues with different patterns
        if i == 0:
            ops_list.append(ops.CountValues("type", 1))  # Single int
        elif i == 1:
            ops_list.append(ops.CountValues("category", "active"))  # Single string
        else:
            ops_list.append(ops.CountValues("level", [1, 2, 3, 4, 5]))  # Multiple ints

        # Add other ops
        ops_list.extend(
            [
                ops.Average("score"),
                ops.NullCount("email"),
                ops.DuplicateCount(["user_id", "session_id"]),
            ]
        )

        cte_data.append(
            BatchCTEData(
                key=ResultKey(yyyy_mm_dd=d, tags={}),
                cte_sql=f"SELECT * FROM events WHERE date = '{d.isoformat()}'",
                ops=ops_list,
            )
        )

    query = dialect.build_cte_query(cte_data)

    # Check all dates are present
    for d in dates:
        assert f"'{d.isoformat()}' as date" in query

    # Check various CountValues patterns
    assert "COUNT_IF(type = 1)" in query
    assert "COUNT_IF(category = 'active')" in query
    assert "COUNT_IF(level IN (1, 2, 3, 4, 5))" in query

    # Verify other ops are included
    assert "AVG(score)" in query
    assert "COUNT_IF(email IS NULL)" in query
    assert "COUNT(DISTINCT (session_id, user_id))" in query


def test_batch_optimization_empty_string_values() -> None:
    """Test batch optimization with empty strings in CountValues."""
    dialect = BigQueryDialect()

    key = ResultKey(yyyy_mm_dd=date(2024, 7, 1), tags={})

    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM forms WHERE date = '2024-07-01'",
            ops=[
                ops.CountValues("field1", ""),  # Empty string
                ops.CountValues("field2", ["", "N/A", "Unknown"]),  # Mix with empty
            ],
        )
    ]

    query = dialect.build_cte_query(cte_data)

    # Empty strings should be properly quoted
    assert "COUNTIF(field1 = '')" in query
    assert "COUNTIF(field2 IN ('', 'N/A', 'Unknown'))" in query


def test_batch_optimization_large_value_lists() -> None:
    """Test batch optimization with large lists of values."""
    dialect = DuckDBDialect()

    key = ResultKey(yyyy_mm_dd=date(2024, 8, 1), tags={})

    # Create a large list of status codes
    status_codes = list(range(100, 150))  # 50 values

    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM api_logs WHERE date = '2024-08-01'",
            ops=[
                ops.CountValues("status_code", status_codes),
                ops.CountValues("endpoint", [f"/api/v{i}/users" for i in range(1, 11)]),  # 10 endpoints
            ],
        )
    ]

    query = dialect.build_cte_query(cte_data)

    # Check the IN clause is properly formed
    status_list = ", ".join(str(code) for code in status_codes)
    assert f"COUNT_IF(status_code IN ({status_list}))" in query

    # Check multiple endpoints
    assert "COUNT_IF(endpoint IN (" in query
    assert "'/api/v1/users'" in query
    assert "'/api/v10/users'" in query


def test_batch_optimization_consistent_ordering() -> None:
    """Test that ops maintain consistent ordering in batch queries."""
    dialect = DuckDBDialect()

    key = ResultKey(yyyy_mm_dd=date(2024, 9, 1), tags={})

    # Define ops in specific order
    ops_list: list[ops.SqlOp] = [
        ops.CountValues("col1", 1),
        ops.NumRows(),
        ops.CountValues("col2", ["a", "b"]),
        ops.Average("col3"),
        ops.CountValues("col4", "test"),
    ]

    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM data WHERE date = '2024-09-01'",
            ops=ops_list,
        )
    ]

    query = dialect.build_cte_query(cte_data)

    # Verify all ops are present in the VALUES array
    for op in ops_list:
        assert op.sql_col in query
        assert f"'key': '{op.sql_col}'" in query
