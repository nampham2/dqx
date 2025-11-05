"""Test analyzer with MAP-based batch optimization."""

import datetime
from collections.abc import Sequence

import duckdb

from dqx import ops
from dqx.analyzer import Analyzer, analyze_sql_ops
from dqx.common import ResultKey
from dqx.dialect import BatchCTEData, DuckDBDialect
from tests.fixtures.data_fixtures import CommercialDataSource


class TestAnalyzerBatchOptimization:
    """Test analyzer with MAP-based batch queries."""

    def test_analyze_batch_sql_ops_with_array(self) -> None:
        """Test analyze_batch_sql_ops with array results."""
        # Create datasource with predictable data
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 1, 3)

        # Use CommercialDataSource from fixtures
        ds = CommercialDataSource(
            start_date=start_date, end_date=end_date, name="test_sales", records_per_day=10, seed=42
        )

        # Create ops for multiple dates
        ops_by_key: dict[ResultKey, list[ops.SqlOp]] = {}
        for day in [1, 2, 3]:
            date = datetime.date(2024, 1, day)
            key = ResultKey(date, {})

            ops_list: list[ops.SqlOp] = [
                ops.Sum("price"),
                ops.Average("price"),
                ops.NullCount("delivered"),
                ops.NumRows(),
            ]

            ops_by_key[key] = ops_list

        # Execute batch analysis
        analyze_sql_ops(ds, ops_by_key)

        # Verify results for each date
        # With seed=42 and the known data generation pattern, we can verify:
        # - Each day should have roughly 10 records (with +/- 20% variation)
        # - Prices and taxes have daily variations
        # - Some delivered values are NULL

        # Date 2024-01-01
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        assert ops_by_key[key1][0].value() > 0  # Sum of prices
        assert ops_by_key[key1][1].value() > 0  # Average price
        assert ops_by_key[key1][2].value() >= 0  # Null count for delivered
        assert 8 <= ops_by_key[key1][3].value() <= 12  # NumRows should be around 10 +/- 20%

        # Date 2024-01-02
        key2 = ResultKey(datetime.date(2024, 1, 2), {})
        assert ops_by_key[key2][0].value() > 0  # Sum of prices
        assert ops_by_key[key2][1].value() > 0  # Average price
        assert ops_by_key[key2][2].value() >= 0  # Null count for delivered
        assert 8 <= ops_by_key[key2][3].value() <= 12  # NumRows should be around 10 +/- 20%

        # Date 2024-01-03
        key3 = ResultKey(datetime.date(2024, 1, 3), {})
        assert ops_by_key[key3][0].value() > 0  # Sum of prices
        assert ops_by_key[key3][1].value() > 0  # Average price
        assert ops_by_key[key3][2].value() >= 0  # Null count for delivered
        assert 8 <= ops_by_key[key3][3].value() <= 12  # NumRows should be around 10 +/- 20%

    def test_analyzer_with_array_batch_analysis(self) -> None:
        """Test full analyzer workflow with array-based batching."""
        # Use CommercialDataSource for real data
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 1, 3)

        ds = CommercialDataSource(
            start_date=start_date, end_date=end_date, name="test_orders", records_per_day=5, seed=100
        )

        # Import necessary components
        from dqx import specs
        from dqx.common import ExecutionId
        from dqx.orm.repositories import InMemoryMetricDB
        from dqx.provider import MetricProvider
        from dqx.specs import MetricSpec

        # Use InMemoryMetricDB for testing
        metric_db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec")
        provider = MetricProvider(metric_db, execution_id)

        # Define metrics for multiple dates
        metrics_by_key: dict[ResultKey, list[MetricSpec]] = {}
        for day in [1, 2, 3]:
            date = datetime.date(2024, 1, day)
            key = ResultKey(date, {})

            # Use concrete metric specs
            metrics_by_key[key] = [
                specs.Sum("price"),
                specs.Average("price"),
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
        report = analyzer.analyze_simple_metrics(ds, metrics_by_key)

        # Verify report contains all metrics
        assert len(report) == 9  # 3 metrics × 3 dates

        # Check that all metrics have been computed
        for day in [1, 2, 3]:
            date = datetime.date(2024, 1, day)
            key = ResultKey(date, {})

            # Verify all metrics exist and have reasonable values
            sum_metric = report[(metrics_by_key[key][0], key, "test_orders")]  # type: ignore[index]
            avg_metric = report[(metrics_by_key[key][1], key, "test_orders")]  # type: ignore[index]
            num_rows_metric = report[(metrics_by_key[key][2], key, "test_orders")]  # type: ignore[index]

            assert sum_metric.value > 0  # Sum should be positive
            assert avg_metric.value > 0  # Average should be positive
            assert 4 <= num_rows_metric.value <= 6  # Should be around 5 +/- 20%

            # Sanity check: average * count ≈ sum (with some tolerance for rounding)
            expected_sum = avg_metric.value * num_rows_metric.value
            assert abs(sum_metric.value - expected_sum) < 0.01

        # No cleanup needed for InMemoryMetricDB

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
        many_ops: Sequence[ops.SqlOp] = [
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
            BatchCTEData(key=key, cte_sql="SELECT * FROM test_data WHERE yyyy_mm_dd = '2024-01-01'", ops=many_ops)
        ]

        # Generate MAP query
        sql = dialect.build_batch_cte_query(cte_data)

        # Execute and verify result size
        result = conn.execute(sql).fetchall()

        # Should return exactly 1 row (not 10 rows as unpivot would)
        assert len(result) == 1

        # Verify the row contains an array with all metrics
        date_str, values_array = result[0]
        assert date_str == "2024-01-01"
        assert isinstance(values_array, list)
        assert len(values_array) == 10  # All 10 metrics in the array

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
        ops_list: Sequence[ops.SqlOp] = [
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
                ops=ops_list,
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
    # Now we're returning arrays instead of dicts
    assert all(isinstance(row[1], list) and len(row[1]) == 10 for row in map_result)

    # Log performance info (not an assertion, just for documentation)
    print("\nPerformance Results:")
    print(f"Array approach: {len(map_result)} rows returned in {map_time:.3f} seconds")
    print(f"Unpivot approach would return: {31 * 10} rows")
    print(f"Row reduction: {(1 - 31 / (31 * 10)) * 100:.1f}%")

    conn.close()
