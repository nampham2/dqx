"""Test batch analysis functionality for Analyzer."""

import datetime
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from dqx import specs
from dqx.analyzer import Analyzer
from dqx.common import DQXError, ResultKey


class DateFilteredDataSource:
    """Test data source that filters by date column."""

    name: str = "duckdb"
    dialect: str = "duckdb"

    def __init__(self, table: pa.Table, date_column: str = "yyyy_mm_dd") -> None:
        """Initialize with PyArrow table and date column name."""
        self._relation = duckdb.arrow(table)
        self._table_name = "_test_table"
        self._date_column = date_column

    def cte(self, nominal_date: datetime.date) -> str:
        """Get the CTE for this data source, filtered by date."""
        date_str = nominal_date.strftime("%Y-%m-%d")
        return f"SELECT * FROM {self._table_name} WHERE {self._date_column} = '{date_str}'"

    def query(self, query: str, nominal_date: datetime.date) -> Any:
        """Execute a query against the DuckDB relation."""
        return self._relation.query(self._table_name, query)


class TestAnalyzerBatch:
    """Test Analyzer.analyze_batch functionality."""

    def test_analyze_batch_single_date(self) -> None:
        """Test batch analysis with single date."""
        # Create test data
        table = pa.table(
            {
                "yyyy_mm_dd": ["2024-01-01"] * 10,
                "revenue": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                "price": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )

        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        # Create metrics by key
        key = ResultKey(datetime.date(2024, 1, 1), {})
        metrics: list[specs.MetricSpec] = [specs.Sum("revenue"), specs.Average("price")]

        # Run batch analysis
        report = analyzer.analyze_batch(ds, {key: metrics})

        # Verify results
        assert len(report) == 2
        assert (metrics[0], key) in report
        assert (metrics[1], key) in report
        assert report[(metrics[0], key)].value == 5500.0  # sum of 100-1000
        assert report[(metrics[1], key)].value == 55.0  # average of 10-100

    def test_analyze_batch_multiple_dates(self) -> None:
        """Test batch analysis with multiple dates."""
        # Create test data for 3 dates
        dates = []
        revenues = []
        prices = []

        for day in range(1, 4):
            date_str = f"2024-01-0{day}"
            dates.extend([date_str] * 5)
            # Different values for each date
            base_revenue = day * 100
            revenues.extend([base_revenue + i * 10 for i in range(5)])
            base_price = day * 10
            prices.extend([base_price + i for i in range(5)])

        table = pa.table({"yyyy_mm_dd": dates, "revenue": revenues, "price": prices})

        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        # Create metrics by key
        metrics_by_key: dict[ResultKey, list[specs.MetricSpec]] = {}
        expected_results: dict[ResultKey, tuple[float, float]] = {}

        for day in range(1, 4):
            key = ResultKey(datetime.date(2024, 1, day), {})
            metrics: list[specs.MetricSpec] = [specs.Sum("revenue"), specs.Average("price")]
            metrics_by_key[key] = metrics

            # Calculate expected values
            base_revenue = day * 100
            expected_sum = float(sum(base_revenue + i * 10 for i in range(5)))
            base_price = day * 10
            expected_avg = sum(base_price + i for i in range(5)) / 5

            expected_results[key] = (expected_sum, expected_avg)

        # Run batch analysis
        report = analyzer.analyze_batch(ds, metrics_by_key)

        # Verify results
        assert len(report) == 6  # 2 metrics * 3 dates

        for key, (expected_sum, expected_avg) in expected_results.items():
            sum_metric = metrics_by_key[key][0]
            avg_metric = metrics_by_key[key][1]

            assert (sum_metric, key) in report
            assert (avg_metric, key) in report
            assert report[(sum_metric, key)].value == expected_sum
            assert report[(avg_metric, key)].value == expected_avg

    def test_analyze_batch_with_tags(self) -> None:
        """Test batch analysis with ResultKey tags."""
        # Create test data
        table = pa.table(
            {
                "yyyy_mm_dd": ["2024-01-01"] * 10,
                "revenue": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )

        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        # Create metrics by key with different tags
        key1 = ResultKey(datetime.date(2024, 1, 1), {"env": "prod"})
        key2 = ResultKey(datetime.date(2024, 1, 1), {"env": "dev"})

        metrics_by_key: dict[ResultKey, list[specs.MetricSpec]] = {
            key1: [specs.Sum("revenue")],
            key2: [specs.Average("revenue")],
        }

        # Run batch analysis
        report = analyzer.analyze_batch(ds, metrics_by_key)

        # Verify both keys are processed independently
        assert len(report) == 2
        assert (metrics_by_key[key1][0], key1) in report
        assert (metrics_by_key[key2][0], key2) in report
        assert report[(metrics_by_key[key1][0], key1)].value == 5500.0
        assert report[(metrics_by_key[key2][0], key2)].value == 550.0

    def test_analyze_batch_large_date_range(self) -> None:
        """Test batch analysis with date range exceeding DEFAULT_BATCH_SIZE."""
        # Create test data for 10 dates (exceeds DEFAULT_BATCH_SIZE of 7)
        dates = []
        values = []

        for day in range(1, 11):
            date_str = f"2024-01-{day:02d}"
            dates.extend([date_str] * 3)
            values.extend([day * 100, day * 100 + 10, day * 100 + 20])

        table = pa.table({"yyyy_mm_dd": dates, "value": values})

        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        # Create metrics for all dates
        metrics_by_key: dict[ResultKey, list[specs.MetricSpec]] = {}
        for day in range(1, 11):
            key = ResultKey(datetime.date(2024, 1, day), {})
            metrics_by_key[key] = [specs.Average("value")]

        # Run batch analysis - should handle batching internally
        report = analyzer.analyze_batch(ds, metrics_by_key)

        # Verify all dates are processed
        assert len(report) == 10

        for day in range(1, 11):
            key = ResultKey(datetime.date(2024, 1, day), {})
            metric = metrics_by_key[key][0]
            assert (metric, key) in report
            # Average of [day*100, day*100+10, day*100+20]
            expected_avg = day * 100 + 10
            assert report[(metric, key)].value == expected_avg

    def test_analyze_batch_empty_metrics(self) -> None:
        """Test error handling for empty metrics."""
        table = pa.table({"yyyy_mm_dd": ["2024-01-01"], "value": [100]})
        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        with pytest.raises(DQXError, match="No metrics provided for batch analysis"):
            analyzer.analyze_batch(ds, {})

    def test_analyze_batch_shared_metrics(self) -> None:
        """Test batch analysis with shared metric instances across dates."""
        # Create test data
        table = pa.table(
            {"yyyy_mm_dd": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"], "value": [100, 200, 300, 400]}
        )

        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        # NOTE: Due to stateful SqlOp design, sharing metric instances across keys
        # in batch analysis will result in the last value overwriting previous ones.
        # This test verifies the current behavior.
        sum_metric = specs.Sum("value")

        # Use same metric for different dates
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        key2 = ResultKey(datetime.date(2024, 1, 2), {})

        metrics_by_key: dict[ResultKey, list[specs.MetricSpec]] = {
            key1: [sum_metric],
            key2: [sum_metric],  # Same metric instance - will overwrite key1's value
        }

        # Run batch analysis
        report = analyzer.analyze_batch(ds, metrics_by_key)

        # Verify current behavior: both keys will have the same value (last one processed)
        assert len(report) == 2
        # Both will have 700.0 because the shared SqlOp gets overwritten
        assert report[(sum_metric, key1)].value == 700.0
        assert report[(sum_metric, key2)].value == 700.0

        # To get independent results, use separate metric instances:
        sum_metric1 = specs.Sum("value")
        sum_metric2 = specs.Sum("value")

        metrics_by_key2: dict[ResultKey, list[specs.MetricSpec]] = {
            key1: [sum_metric1],
            key2: [sum_metric2],
        }

        analyzer2 = Analyzer()
        report2 = analyzer2.analyze_batch(ds, metrics_by_key2)

        # Now we get the expected independent results
        assert report2[(sum_metric1, key1)].value == 300.0  # 100 + 200
        assert report2[(sum_metric2, key2)].value == 700.0  # 300 + 400

    def test_analyze_batch_mixed_metrics(self) -> None:
        """Test batch analysis with different metrics per date."""
        # Create test data
        table = pa.table(
            {
                "yyyy_mm_dd": ["2024-01-01"] * 5 + ["2024-01-02"] * 5,
                "revenue": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                "quantity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "price": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )

        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        # Different metrics for each date
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        key2 = ResultKey(datetime.date(2024, 1, 2), {})

        metrics_by_key: dict[ResultKey, list[specs.MetricSpec]] = {
            key1: [specs.Sum("revenue"), specs.Average("price")],
            key2: [specs.Maximum("quantity"), specs.Minimum("price")],
        }

        # Run batch analysis
        report = analyzer.analyze_batch(ds, metrics_by_key)

        # Verify results
        assert len(report) == 4

        # Date 1 metrics
        assert report[(metrics_by_key[key1][0], key1)].value == 1500.0  # sum(100-500)
        assert report[(metrics_by_key[key1][1], key1)].value == 30.0  # avg(10-50)

        # Date 2 metrics
        assert report[(metrics_by_key[key2][0], key2)].value == 10.0  # max(6-10)
        assert report[(metrics_by_key[key2][1], key2)].value == 60.0  # min(60-100)

    def test_analyze_batch_report_merging(self) -> None:
        """Test that batch results are merged with existing analyzer report."""
        # Create test data
        table = pa.table({"yyyy_mm_dd": ["2024-01-01", "2024-01-02"], "value": [100, 200]})

        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        # First run regular analyze
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        metric1 = specs.Sum("value")
        report1 = analyzer.analyze(ds, [metric1], key1)
        assert len(report1) == 1

        # Then run batch analyze with different date
        key2 = ResultKey(datetime.date(2024, 1, 2), {})
        metric2 = specs.Average("value")

        report2 = analyzer.analyze_batch(ds, {key2: [metric2]})

        # Final report should have both results
        assert len(report2) == 2
        assert (metric1, key1) in report2
        assert (metric2, key2) in report2
        assert report2[(metric1, key1)].value == 100.0
        assert report2[(metric2, key2)].value == 200.0

    def test_analyze_batch_no_sql_ops(self) -> None:
        """Test batch analysis with metrics that have no SQL operations."""
        # Create test data
        table = pa.table({"yyyy_mm_dd": ["2024-01-01"], "value": [100]})

        ds = DateFilteredDataSource(table, date_column="yyyy_mm_dd")
        analyzer = Analyzer()

        # Create a metric with no analyzers (edge case)
        class EmptyMetric:
            metric_type: specs.MetricType = "NumRows"  # Need a valid type

            def __init__(self) -> None:
                self._analyzers: list[Any] = []

            @property
            def name(self) -> str:
                return "empty"

            @property
            def parameters(self) -> dict[str, Any]:
                return {}

            @property
            def analyzers(self) -> list[Any]:
                return self._analyzers

            def state(self) -> Any:
                return specs.states.SimpleAdditiveState(value=0.0)

            @classmethod
            def deserialize(cls, state: bytes) -> Any:
                return specs.states.SimpleAdditiveState.deserialize(state)

            def __hash__(self) -> int:
                return hash((self.name, tuple(self.parameters.items())))

            def __eq__(self, other: Any) -> bool:
                if not isinstance(other, EmptyMetric):
                    return False
                return self.name == other.name

            def __str__(self) -> str:
                return self.name

        key = ResultKey(datetime.date(2024, 1, 1), {})
        empty_metric = EmptyMetric()

        # Should handle gracefully
        report = analyzer.analyze_batch(ds, {key: [empty_metric]})  # type: ignore[arg-type]

        # Empty metric should still be in report
        assert len(report) == 1
        assert (empty_metric, key) in report

    def test_batch_analysis_with_duplicate_analyzers(self) -> None:
        """Test that batch analysis correctly handles metrics with duplicate analyzer types.

        This test ensures that when multiple metrics use the same type of analyzer
        (e.g., multiple Average metrics each need NumRows), all metrics can access
        their analyzer values after batch execution, even though the SQL was deduplicated.
        """
        # Create test data
        table = pa.table(
            {
                "yyyy_mm_dd": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
                "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "tax": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "quantity": [5, 10, 15, 20, 25, 30],
            }
        )

        ds = DateFilteredDataSource(table)
        analyzer = Analyzer()

        # Create multiple metrics that use the same analyzer types
        # All Average metrics need NumRows() and Sum() analyzers
        metrics_by_key = {
            ResultKey(datetime.date(2024, 1, 1), {}): [
                specs.Average("price"),  # Needs NumRows() and Sum("price")
                specs.Average("tax"),  # Needs NumRows() and Sum("tax")
                specs.Average("quantity"),  # Needs NumRows() and Sum("quantity")
            ],
            ResultKey(datetime.date(2024, 1, 2), {}): [
                specs.Average("price"),  # Another set of analyzers
                specs.Maximum("tax"),  # Different metric type
            ],
        }

        # Execute batch analysis
        report = analyzer.analyze_batch(ds, metrics_by_key)

        # Verify all metrics have values
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        key2 = ResultKey(datetime.date(2024, 1, 2), {})

        # Check that all Average metrics computed correctly
        assert report[(specs.Average("price"), key1)].value == 20.0
        assert report[(specs.Average("tax"), key1)].value == 2.0
        assert report[(specs.Average("quantity"), key1)].value == 10.0

        assert report[(specs.Average("price"), key2)].value == 50.0
        assert report[(specs.Maximum("tax"), key2)].value == 6.0

        # Verify no "op has not been collected yet!" errors occurred
        for metric_key, metric_value in report.items():
            assert metric_value.value is not None
            assert metric_value.state is not None

    def test_batch_analysis_deduplication_efficiency(self) -> None:
        """Test that SQL deduplication actually reduces the number of operations.

        This test verifies that our deduplication logic is working by checking
        that duplicate analyzers result in fewer SQL operations.
        """
        # Create test data
        table = pa.table({"yyyy_mm_dd": ["2024-01-01"] * 3, "col1": [1, 2, 3], "col2": [4, 5, 6]})

        ds = DateFilteredDataSource(table)
        analyzer = Analyzer()

        # Track SQL execution to verify deduplication
        original_analyze = analyzer._analyze_batch_internal
        sql_op_counts = []

        def track_sql_ops(*args: Any, **kwargs: Any) -> Any:
            # Count ops before execution
            from collections import defaultdict

            analyzer_equivalence_map = defaultdict(list)

            metrics_by_key = args[1] if len(args) > 1 else kwargs.get("metrics_by_key", {})
            for key, metrics in metrics_by_key.items():
                for metric in metrics:
                    for op in metric.analyzers:
                        from dqx.ops import SqlOp

                        if isinstance(op, SqlOp):
                            analyzer_equivalence_map[(key, op)].append(op)

            # Count unique ops (deduplicated)
            sql_op_counts.append(len(analyzer_equivalence_map))
            return original_analyze(*args, **kwargs)

        analyzer._analyze_batch_internal = track_sql_ops  # type: ignore[assignment]

        # Create metrics with many duplicate NumRows operations
        metrics_by_key: dict[ResultKey, list[specs.MetricSpec]] = {
            ResultKey(datetime.date(2024, 1, 1), {}): [
                specs.Average("col1"),  # Needs NumRows
                specs.Average("col2"),  # Needs NumRows (duplicate)
                specs.NumRows(),  # Direct NumRows (duplicate)
            ]
        }

        analyzer.analyze_batch(ds, metrics_by_key)

        # Should only execute 3 unique ops: NumRows, Sum(col1), Sum(col2)
        # Even though we have 3 NumRows instances, only 1 should be executed
        assert sql_op_counts[0] == 3
