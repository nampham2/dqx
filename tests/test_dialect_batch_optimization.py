"""Test batch SQL optimization using MAP feature."""

import datetime

import pytest

from dqx.common import ResultKey
from dqx.dialect import DuckDBDialect
from dqx.models import BatchCTEData
from dqx.ops import Average, Maximum, Minimum, NullCount, Sum


class TestBatchCTEQueryMap:
    """Test MAP-based batch CTE queries."""

    def test_build_batch_cte_query_single_date(self) -> None:
        """Test MAP query with single date."""
        dialect = DuckDBDialect()

        # Create test data
        key = ResultKey(datetime.date(2024, 1, 1), {})
        ops = [
            Sum("revenue"),
            Average("price"),
        ]

        cte_data = [BatchCTEData(key=key, cte_sql="SELECT * FROM sales WHERE yyyy_mm_dd = '2024-01-01'", ops=ops)]  # type: ignore[arg-type]

        # This method will be updated to use MAP
        sql = dialect.build_batch_cte_query(cte_data)

        # Verify MAP structure
        assert "MAP {" in sql
        # Check that the sql_col values are used as MAP keys
        assert f"'{ops[0].sql_col}': \"{ops[0].sql_col}\"" in sql  # type: ignore[attr-defined] # Sum(revenue)
        assert f"'{ops[1].sql_col}': \"{ops[1].sql_col}\"" in sql  # type: ignore[attr-defined] # Average(price)
        assert "as values FROM" in sql
        assert sql.count("SELECT '2024-01-01' as date") == 1  # Only one row

    def test_build_batch_cte_query_multiple_dates(self) -> None:
        """Test MAP query with multiple dates."""
        dialect = DuckDBDialect()

        # Create test data for 3 dates
        cte_data = []
        for day in [1, 2, 3]:
            key = ResultKey(datetime.date(2024, 1, day), {})
            ops = [
                Sum("revenue"),
                Average("price"),
                NullCount("customer_id"),
            ]

            cte_data.append(
                BatchCTEData(key=key, cte_sql=f"SELECT * FROM sales WHERE yyyy_mm_dd = '2024-01-0{day}'", ops=ops)  # type: ignore[arg-type]
            )

        sql = dialect.build_batch_cte_query(cte_data)

        # Should have 9 SELECT statements: 3 source CTEs + 3 metrics CTEs + 3 final SELECTs
        assert sql.count("SELECT") == 9
        assert sql.count("UNION ALL") == 2  # 3 results joined by 2 UNION ALLs
        assert sql.count("MAP {") == 3  # One MAP per date

    def test_build_batch_cte_query_preserves_column_aliases(self) -> None:
        """Test that MAP keys use the correct sql_col aliases."""
        dialect = DuckDBDialect()

        key = ResultKey(datetime.date(2024, 1, 1), {})
        ops = [
            Sum("total_sales"),
            Minimum("min_price"),
            Maximum("max_price"),
        ]

        # Get the actual sql_col values
        expected_keys = [op.sql_col for op in ops]  # type: ignore[attr-defined]

        cte_data = [BatchCTEData(key=key, cte_sql="SELECT * FROM orders", ops=ops)]  # type: ignore[arg-type]

        sql = dialect.build_batch_cte_query(cte_data)

        # Verify each sql_col appears as a MAP key
        for sql_col in expected_keys:
            assert f"'{sql_col}': \"{sql_col}\"" in sql

    def test_build_batch_cte_query_empty_data(self) -> None:
        """Test error handling for empty CTE data."""
        dialect = DuckDBDialect()

        with pytest.raises(ValueError, match="No CTE data provided"):
            dialect.build_batch_cte_query([])

    def test_build_batch_cte_query_no_ops(self) -> None:
        """Test error handling when no ops provided."""
        dialect = DuckDBDialect()

        cte_data = [
            BatchCTEData(
                key=ResultKey(datetime.date(2024, 1, 1), {}),
                cte_sql="SELECT * FROM sales",
                ops=[],  # No ops
            )
        ]

        with pytest.raises(ValueError, match="No metrics to compute"):
            dialect.build_batch_cte_query(cte_data)


def test_map_query_with_duckdb_execution() -> None:
    """Test MAP query with actual DuckDB execution."""
    import duckdb

    from dqx import ops
    from dqx.common import ResultKey
    from dqx.dialect import DuckDBDialect
    from dqx.models import BatchCTEData

    # Create test database
    conn = duckdb.connect(":memory:")

    # Create test data for multiple dates
    conn.execute("""
        CREATE TABLE sales AS
        SELECT
            '2024-01-01'::DATE as yyyy_mm_dd,
            100.0 as revenue,
            25.0 as price,
            1 as customer_id
        UNION ALL
        SELECT '2024-01-01'::DATE, 200.0, 30.0, 2
        UNION ALL
        SELECT '2024-01-01'::DATE, 150.0, NULL, NULL
        UNION ALL
        SELECT '2024-01-02'::DATE, 300.0, 40.0, 3
        UNION ALL
        SELECT '2024-01-02'::DATE, 250.0, 35.0, NULL
        UNION ALL
        SELECT '2024-01-03'::DATE, 400.0, 50.0, 4
    """)

    dialect = DuckDBDialect()

    # Create batch data
    cte_data = []
    for day in [1, 2, 3]:
        date = datetime.date(2024, 1, day)
        key = ResultKey(date, {})

        ops_list = [
            ops.Sum("revenue"),
            ops.Average("price"),
            ops.NullCount("customer_id"),
        ]

        cte_data.append(
            BatchCTEData(key=key, cte_sql=f"SELECT * FROM sales WHERE yyyy_mm_dd = '{date.isoformat()}'", ops=ops_list)  # type: ignore[arg-type]
        )

    # Generate MAP query
    sql = dialect.build_batch_cte_query(cte_data)

    # Execute query
    result = conn.execute(sql).fetchall()

    # Verify results
    assert len(result) == 3  # One row per date

    # Check first date results
    date1, values1 = result[0]
    assert date1 == "2024-01-01"
    assert isinstance(values1, dict)  # DuckDB returns MAP as dict
    # Sum of revenue for 2024-01-01: 100 + 200 + 150 = 450
    assert values1[cte_data[0].ops[0].sql_col] == 450.0
    # Average of price for 2024-01-01: (25 + 30) / 2 = 27.5 (NULL excluded)
    assert values1[cte_data[0].ops[1].sql_col] == 27.5
    # Null count for customer_id: 1
    assert values1[cte_data[0].ops[2].sql_col] == 1.0

    # Check second date results
    date2, values2 = result[1]
    assert date2 == "2024-01-02"
    # Sum of revenue for 2024-01-02: 300 + 250 = 550
    assert values2[cte_data[1].ops[0].sql_col] == 550.0

    conn.close()
