"""Test analyzer with MAP-based batch optimization."""

import datetime
from typing import Any

import duckdb

from dqx import ops
from dqx.analyzer import Analyzer, analyze_batch_sql_ops
from dqx.common import ResultKey
from dqx.dialect import BatchCTEData, DuckDBDialect


class TestAnalyzerBatchOptimization:
    """Test analyzer with MAP-based batch queries."""

    def test_analyze_batch_sql_ops_with_map(self) -> None:
        """Test analyze_batch_sql_ops with MAP results."""
        # Create test database
        conn = duckdb.connect(":memory:")

        # For simplicity, we'll mock a data source that returns predictable CTEs
        from unittest.mock import Mock

        ds = Mock()
        ds.dialect = "duckdb"

        # Mock the cte method to return SQL that selects from a test table
        def mock_cte(date: datetime.date) -> str:
            date_str = date.isoformat()
            return f"SELECT * FROM (SELECT '{date_str}'::DATE as yyyy_mm_dd, revenue, price, customer_id FROM sales WHERE yyyy_mm_dd = '{date_str}')"

        ds.cte = mock_cte

        # Mock the query method to execute on our connection
        def mock_query(sql: str) -> Any:
            return conn.execute(sql)

        ds.query = mock_query

        # Create test data
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

        # Create ops for multiple dates
        ops_by_key = {}
        for day in [1, 2, 3]:
            date = datetime.date(2024, 1, day)
            key = ResultKey(date, {})

            ops_list = [
                ops.Sum("revenue"),
                ops.Average("price"),
                ops.NullCount("customer_id"),
            ]

            ops_by_key[key] = ops_list

        # Execute batch analysis
        analyze_batch_sql_ops(ds, ops_by_key)  # type: ignore[arg-type]

        # Verify results for each date
        # Date 2024-01-01: Sum=450, Avg=27.5, NullCount=1
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 1), {})][0].value() == 450.0
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 1), {})][1].value() == 27.5
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 1), {})][2].value() == 1.0

        # Date 2024-01-02: Sum=550, Avg=37.5, NullCount=1
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 2), {})][0].value() == 550.0
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 2), {})][1].value() == 37.5
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 2), {})][2].value() == 1.0

        # Date 2024-01-03: Sum=400, Avg=50, NullCount=0
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 3), {})][0].value() == 400.0
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 3), {})][1].value() == 50.0
        assert ops_by_key[ResultKey(datetime.date(2024, 1, 3), {})][2].value() == 0.0

        conn.close()

    def test_analyzer_with_map_batch_analysis(self) -> None:
        """Test full analyzer workflow with MAP-based batching."""
        # Create test database
        conn = duckdb.connect(":memory:")

        # For this test, we'll use a mock data source
        from unittest.mock import Mock

        ds = Mock()
        ds.dialect = "duckdb"
        ds.name = "test_dataset"  # Add dataset name

        # Mock the cte method to return SQL that selects from a test table
        def mock_cte(date: datetime.date) -> str:
            date_str = date.isoformat()
            return f"SELECT * FROM (SELECT '{date_str}'::DATE as yyyy_mm_dd, amount, status FROM orders WHERE yyyy_mm_dd = '{date_str}')"

        ds.cte = mock_cte

        # Mock the query method to execute on our connection
        def mock_query(sql: str) -> Any:
            return conn.execute(sql)

        ds.query = mock_query

        # Create test data
        conn.execute("""
            CREATE TABLE orders AS
            SELECT
                '2024-01-01'::DATE as yyyy_mm_dd,
                100.0 as amount,
                'completed' as status
            UNION ALL
            SELECT '2024-01-01'::DATE, 200.0, 'pending'
            UNION ALL
            SELECT '2024-01-01'::DATE, 150.0, 'completed'
            UNION ALL
            SELECT '2024-01-02'::DATE, 300.0, 'completed'
            UNION ALL
            SELECT '2024-01-02'::DATE, 250.0, 'pending'
            UNION ALL
            SELECT '2024-01-03'::DATE, 400.0, 'completed'
        """)

        # Import necessary components
        # Create a simple metric provider and analyzer
        # For testing, we don't need a real DB, so we'll mock it
        from unittest.mock import MagicMock

        from dqx import specs
        from dqx.common import ExecutionId
        from dqx.orm.repositories import MetricDB
        from dqx.provider import MetricProvider
        from dqx.specs import MetricSpec

        mock_db = MagicMock(spec=MetricDB)
        execution_id = ExecutionId("test-exec")
        provider = MetricProvider(mock_db, execution_id)

        # Define metrics for multiple dates
        metrics_by_key: dict[ResultKey, list[MetricSpec]] = {}
        for day in [1, 2, 3]:
            date = datetime.date(2024, 1, day)
            key = ResultKey(date, {})

            # Use concrete metric specs
            metrics_by_key[key] = [
                specs.Sum("amount"),
                specs.Average("amount"),
                specs.NumRows(),
            ]

        # Create analyzer with proper arguments
        analyzer = Analyzer(
            datasources=[ds],  # type: ignore[list-item]
            provider=provider,
            key=ResultKey(datetime.date(2024, 1, 1), {}),
            execution_id=execution_id,
        )

        # Run batch analysis
        report = analyzer.analyze_simple_metrics(ds, metrics_by_key)  # type: ignore[arg-type]

        # Verify report contains all metrics
        assert len(report) == 9  # 3 metrics × 3 dates

        # Check specific values using 3-tuple keys
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        # Sum of all amounts for 2024-01-01: 100 + 200 + 150 = 450
        assert report[(metrics_by_key[key1][0], key1, "test_dataset")].value == 450.0  # type: ignore[index]
        # Average of all amounts for 2024-01-01: (100 + 200 + 150) / 3 = 150
        assert report[(metrics_by_key[key1][1], key1, "test_dataset")].value == 150.0  # type: ignore[index]
        # NumRows for 2024-01-01: 3 rows
        assert report[(metrics_by_key[key1][2], key1, "test_dataset")].value == 3.0  # type: ignore[index]

        conn.close()

    def test_verify_map_reduces_query_size(self) -> None:
        """Verify that MAP approach returns N rows instead of N*M rows."""
        # Create test database
        conn = duckdb.connect(":memory:")

        # Create test data
        conn.execute("""
            CREATE TABLE test_data AS
            SELECT
                '2024-01-01'::DATE as yyyy_mm_dd,
                i as value1,
                i * 2 as value2,
                i * 3 as value3,
                i * 4 as value4,
                i * 5 as value5
            FROM generate_series(1, 100) as t(i)
        """)

        dialect = DuckDBDialect()

        # Create batch data with many metrics
        key = ResultKey(datetime.date(2024, 1, 1), {})
        many_ops = [
            ops.Sum("value1"),
            ops.Sum("value2"),
            ops.Sum("value3"),
            ops.Sum("value4"),
            ops.Sum("value5"),
            ops.Average("value1"),
            ops.Average("value2"),
            ops.Average("value3"),
            ops.Average("value4"),
            ops.Average("value5"),
        ]

        cte_data = [
            BatchCTEData(key=key, cte_sql="SELECT * FROM test_data WHERE yyyy_mm_dd = '2024-01-01'", ops=many_ops)  # type: ignore[arg-type]
        ]

        # Generate MAP query
        sql = dialect.build_batch_cte_query(cte_data)

        # Execute and verify result size
        result = conn.execute(sql).fetchall()

        # Should return exactly 1 row (not 10 rows as unpivot would)
        assert len(result) == 1

        # Verify the row contains a MAP with all metrics
        date_str, values_map = result[0]
        assert date_str == "2024-01-01"
        assert isinstance(values_map, dict)
        assert len(values_map) == 10  # All 10 metrics in the MAP

        conn.close()


def test_performance_comparison() -> None:
    """Compare performance of MAP vs unpivot approaches (for documentation)."""
    import time

    conn = duckdb.connect(":memory:")

    # Create larger test dataset
    conn.execute("""
        CREATE TABLE large_sales AS
        SELECT
            ('2024-01-' || LPAD(CAST((i % 31) + 1 AS VARCHAR), 2, '0'))::DATE as yyyy_mm_dd,
            RANDOM() * 1000 as revenue,
            RANDOM() * 100 as price,
            CASE WHEN RANDOM() < 0.3 THEN NULL ELSE i END as customer_id
        FROM generate_series(1, 10000) as t(i)
    """)

    # Test data: 31 dates with 10 metrics each
    dialect = DuckDBDialect()
    cte_data = []

    for day in range(1, 32):  # 31 days
        date = datetime.date(2024, 1, day)
        key = ResultKey(date, {})

        # Create 10 different metrics
        ops_list = [
            ops.Sum("revenue"),
            ops.Average("revenue"),
            ops.Minimum("revenue"),
            ops.Maximum("revenue"),
            ops.Variance("revenue"),
            ops.Sum("price"),
            ops.Average("price"),
            ops.Minimum("price"),
            ops.Maximum("price"),
            ops.NullCount("customer_id"),
        ]

        cte_data.append(
            BatchCTEData(
                key=key,
                cte_sql=f"SELECT * FROM large_sales WHERE yyyy_mm_dd = '2024-01-{day:02d}'",
                ops=ops_list,  # type: ignore[arg-type]
            )
        )

    # Generate MAP query
    map_sql = dialect.build_batch_cte_query(cte_data)

    # Time MAP query execution
    start_time = time.time()
    map_result = conn.execute(map_sql).fetchall()
    map_time = time.time() - start_time

    # Verify result structure
    assert len(map_result) == 31  # One row per date
    assert all(isinstance(row[1], dict) and len(row[1]) == 10 for row in map_result)

    # Log performance info (not an assertion, just for documentation)
    print("\nPerformance Results:")
    print(f"MAP approach: {len(map_result)} rows returned in {map_time:.3f} seconds")
    print(f"Unpivot approach would return: {31 * 10} rows")
    print(f"Row reduction: {(1 - 31 / (31 * 10)) * 100:.1f}%")

    conn.close()
