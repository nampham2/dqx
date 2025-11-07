"""Comprehensive test coverage for analyzer.py."""

import datetime
import logging
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx import models, ops, specs, states
from dqx.analyzer import AnalysisReport, Analyzer, analyze_sql_ops
from dqx.cache import MetricCache
from dqx.common import (
    DQXError,
    ExecutionId,
    ResultKey,
    SqlDataSource,
)
from dqx.orm.repositories import InMemoryMetricDB, MetricDB
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.specs import MetricSpec
from tests.fixtures.data_fixtures import CommercialDataSource


class TestAnalysisReport:
    """Test AnalysisReport class functionality."""

    def test_analysis_report_init_empty(self) -> None:
        """Test AnalysisReport initialization with no data."""
        report = AnalysisReport()
        assert len(report) == 0
        assert report.data == {}

    def test_analysis_report_init_with_data(self) -> None:
        """Test AnalysisReport initialization with data."""
        # Create test data
        spec = specs.Sum("revenue")
        key = ResultKey(datetime.date(2024, 1, 1), {})
        state = states.SimpleAdditiveState(value=1000.0)
        metric = models.Metric(spec=spec, key=key, state=state, dataset="sales")

        # Initialize with data - cast to proper type
        data: dict[tuple[MetricSpec, ResultKey, str], models.Metric] = {(spec, key, "sales"): metric}
        report = AnalysisReport(data)

        assert len(report) == 1
        assert report[(spec, key, "sales")] == metric

    def test_analysis_report_merge(self) -> None:
        """Test merging two AnalysisReports with overlapping and non-overlapping data."""
        # Create first report
        spec1 = specs.Sum("revenue")
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        state1 = states.SimpleAdditiveState(value=1000.0)
        metric1 = models.Metric(spec=spec1, key=key1, state=state1, dataset="sales")
        report1 = AnalysisReport({(spec1, key1, "sales"): metric1})

        # Create second report with same key (for merge conflict)
        state2 = states.SimpleAdditiveState(value=500.0)
        metric2 = models.Metric(spec=spec1, key=key1, state=state2, dataset="sales")

        # Create third metric with different key
        spec3 = specs.Average("price")
        key3 = ResultKey(datetime.date(2024, 1, 2), {})
        state3 = states.Average(avg=50.0, n=20)
        metric3 = models.Metric(spec=spec3, key=key3, state=state3, dataset="sales")

        report2 = AnalysisReport({(spec1, key1, "sales"): metric2, (spec3, key3, "sales"): metric3})

        # Merge reports
        merged = report1.merge(report2)

        # Check merged result
        assert len(merged) == 2

        # The conflicting key should have merged values (1000 + 500 = 1500)
        merged_metric1 = merged[(spec1, key1, "sales")]
        assert merged_metric1.value == 1500.0

        # The non-conflicting metric should be present
        assert merged[(spec3, key3, "sales")] == metric3

    def test_analysis_report_show(self) -> None:
        """Test the show method calls print_analysis_report correctly."""
        # Create test report
        spec = specs.Sum("revenue")
        key = ResultKey(datetime.date(2024, 1, 1), {})
        state = states.SimpleAdditiveState(value=1000.0)
        metric = models.Metric(spec=spec, key=key, state=state, dataset="sales")
        report = AnalysisReport({(spec, key, "sales"): metric})

        # Mock the display function
        with patch("dqx.display.print_analysis_report") as mock_print:
            symbol_lookup: dict[tuple[MetricSpec, ResultKey, str], str] = {(spec, key, "sales"): "x_1"}
            report.show(symbol_lookup)

            # Verify it was called correctly
            mock_print.assert_called_once_with(report, symbol_lookup)

    def test_analysis_report_persist_empty(self) -> None:
        """Test persisting an empty report logs a warning."""
        # Create empty report
        report = AnalysisReport()

        # Create mocks
        mock_db = Mock(spec=MetricDB)
        mock_cache = Mock(spec=MetricCache)

        # Persist should log warning and return early
        with patch.object(logging.getLogger("dqx.analyzer"), "warning") as mock_warning:
            report.persist(mock_db, mock_cache)
            mock_warning.assert_called_once_with("Try to save an EMPTY analysis report!")

        # Verify cache methods were not called
        mock_cache.put.assert_not_called()
        mock_cache.write_back.assert_not_called()

    def test_analysis_report_persist_with_data(self) -> None:
        """Test persisting a report with data."""
        # Create test report
        spec = specs.Sum("revenue")
        key = ResultKey(datetime.date(2024, 1, 1), {})
        state = states.SimpleAdditiveState(value=1000.0)
        metric = models.Metric(spec=spec, key=key, state=state, dataset="sales")
        report = AnalysisReport({(spec, key, "sales"): metric})

        # Create mocks
        mock_db = Mock(spec=MetricDB)
        mock_cache = Mock(spec=MetricCache)

        # Persist
        with patch.object(logging.getLogger("dqx.analyzer"), "info") as mock_info:
            report.persist(mock_db, mock_cache)
            mock_info.assert_called_once_with("Overwriting analysis report ...")

        # Verify cache methods were called correctly
        mock_cache.put.assert_called_once()
        called_metrics = mock_cache.put.call_args[0][0]
        assert len(called_metrics) == 1
        assert called_metrics[0] == metric
        assert mock_cache.put.call_args[1]["mark_dirty"] is True
        mock_cache.write_back.assert_called_once()


class TestAnalyzeSqlOps:
    """Test analyze_sql_ops function error scenarios."""

    def test_analyze_sql_ops_empty(self) -> None:
        """Test analyze_sql_ops with empty ops_by_key."""
        mock_ds = Mock(spec=SqlDataSource)
        analyze_sql_ops(mock_ds, {})
        # Should return immediately without any calls
        mock_ds.cte.assert_not_called()
        mock_ds.query.assert_not_called()

    def test_analyze_sql_ops_unexpected_date(self) -> None:
        """Test analyze_sql_ops raises DQXError for unexpected dates in results."""
        # Create mock datasource
        mock_ds = Mock(spec=SqlDataSource)
        mock_ds.name = "test_ds"
        mock_ds.cte.return_value = "SELECT * FROM test"
        mock_ds.dialect = "duckdb"

        # Mock query result with unexpected date
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("2024-01-01", [{"key": "sum_revenue", "value": 100.0}]),
            ("2024-01-02", [{"key": "sum_revenue", "value": 200.0}]),  # Unexpected!
        ]
        mock_ds.query.return_value = mock_result

        # Create ops for only one date
        key = ResultKey(datetime.date(2024, 1, 1), {})
        ops_list: list[ops.SqlOp[float]] = [ops.Sum("revenue")]
        ops_by_key: dict[ResultKey, list[ops.SqlOp[float]]] = {key: ops_list}

        # Should raise DQXError for unexpected date
        with pytest.raises(DQXError) as exc_info:
            analyze_sql_ops(mock_ds, ops_by_key)

        assert "Unexpected date '2024-01-02' in SQL results" in str(exc_info.value)
        assert "Expected dates: ['2024-01-01']" in str(exc_info.value)

    def test_analyze_sql_ops_missing_dates(self) -> None:
        """Test analyze_sql_ops raises DQXError for missing dates in results."""
        # Create mock datasource
        mock_ds = Mock(spec=SqlDataSource)
        mock_ds.name = "test_ds"
        mock_ds.cte.return_value = "SELECT * FROM test"
        mock_ds.dialect = "duckdb"

        # Mock query result missing a date
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("2024-01-01", [{"key": "sum_revenue", "value": 100.0}]),
            # Missing 2024-01-02
        ]
        mock_ds.query.return_value = mock_result

        # Create ops for two dates
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        key2 = ResultKey(datetime.date(2024, 1, 2), {})
        ops_list: list[ops.SqlOp[float]] = [ops.Sum("revenue")]
        ops_by_key: dict[ResultKey, list[ops.SqlOp[float]]] = {key1: ops_list.copy(), key2: ops_list.copy()}

        # Should raise DQXError for missing date
        with pytest.raises(DQXError) as exc_info:
            analyze_sql_ops(mock_ds, ops_by_key)

        assert "Missing dates in SQL results: ['2024-01-02']" in str(exc_info.value)

    def test_analyze_sql_ops_value_validation_error(self) -> None:
        """Test analyze_sql_ops raises DQXError for invalid values."""
        # Create mock datasource
        mock_ds = Mock(spec=SqlDataSource)
        mock_ds.name = "test_ds"
        mock_ds.cte.return_value = "SELECT * FROM test"
        mock_ds.dialect = "duckdb"

        # Create ops
        key = ResultKey(datetime.date(2024, 1, 1), {})
        sum_op = ops.Sum("revenue")
        ops_by_key: dict[ResultKey, list[ops.SqlOp[float]]] = {key: [sum_op]}

        # Mock query result with null value - the key must match the op's sql_col
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("2024-01-01", [{"key": sum_op.sql_col, "value": None}]),  # Null value!
        ]
        mock_ds.query.return_value = mock_result

        # Should raise DQXError for null value
        with pytest.raises(DQXError) as exc_info:
            analyze_sql_ops(mock_ds, ops_by_key)

        # The symbol in the error message has a dynamically generated prefix
        assert "Null value encountered for symbol" in str(exc_info.value)
        assert "sum(revenue)" in str(exc_info.value)
        assert "on date 2024-01-01" in str(exc_info.value)


class TestAnalyzer:
    """Test Analyzer class functionality."""

    def test_analyzer_init(self) -> None:
        """Test Analyzer initialization and properties."""
        # Create test components
        mock_ds = Mock(spec=SqlDataSource)
        mock_ds.name = "test_ds"

        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.8)
        key = ResultKey(datetime.date(2024, 1, 1), {})

        # Create analyzer
        analyzer = Analyzer(
            datasources=[mock_ds], provider=provider, key=key, execution_id=execution_id, data_av_threshold=0.9
        )

        # Test properties
        assert analyzer.datasources == [mock_ds]
        assert analyzer.provider == provider
        assert analyzer.key == key
        assert analyzer.execution_id == execution_id
        assert analyzer.data_av_threshold == 0.9
        assert analyzer.metrics == provider.registry.metrics
        assert analyzer.db == provider._db
        assert analyzer.cache == provider._cache

    def test_analyzer_analyze_simple_metrics(self) -> None:
        """Test analyze_simple_metrics with batching."""
        # Use real data source for integration
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 1, 3)
        ds = CommercialDataSource(
            start_date=start_date, end_date=end_date, name="sales_data", records_per_day=5, seed=42
        )

        # Create provider and analyzer
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.8)
        analyzer = Analyzer(
            datasources=[cast(SqlDataSource, ds)],
            provider=provider,
            key=ResultKey(datetime.date(2024, 1, 1), {}),
            execution_id=execution_id,
            data_av_threshold=0.9,
        )

        # Create metrics for multiple dates
        metrics_by_key: dict[ResultKey, list[MetricSpec]] = {}
        for day in [1, 2, 3]:
            date = datetime.date(2024, 1, day)
            key = ResultKey(date, {})
            metrics_by_key[key] = [
                specs.Sum("price"),
                specs.Average("price"),
                specs.NumRows(),
            ]

        # Analyze
        report = analyzer.analyze_simple_metrics(ds, metrics_by_key)

        # Verify report
        assert len(report) == 9  # 3 metrics x 3 dates

        # Check all metrics have values
        for metric in report.values():
            assert metric.value is not None
            assert metric.dataset == "sales_data"
            assert metric.metadata is not None
            assert metric.metadata.execution_id == execution_id

    def test_analyzer_analyze_extended_metrics_success(self) -> None:
        """Test analyze_extended_metrics with Success results."""
        # Create mock components
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.8)

        # Create a simple metric first
        provider.average("revenue", dataset="sales")

        # Mock the function to return Success
        mock_fn = Mock(return_value=Success(0.5))  # 50% increase

        # Create symbolic metric for testing
        sym_metric = SymbolicMetric(
            name="day_over_day(average(revenue))",
            symbol=sp.Symbol("x_1"),
            metric_spec=specs.DayOverDay.from_base_spec(specs.Average("revenue")),
            fn=mock_fn,
            lag=0,
            data_av_ratio=1.0,
            required_metrics=[],
            dataset="sales",
        )

        # Create analyzer
        analyzer = Analyzer(
            datasources=[],
            provider=provider,
            key=ResultKey(datetime.date(2024, 1, 2), {}),
            execution_id=execution_id,
            data_av_threshold=0.9,
        )

        # Mock cache on the provider
        mock_cache = Mock(spec=MetricCache)
        with patch.object(analyzer.provider, "_cache", mock_cache):
            # Analyze extended metrics
            report = analyzer.analyze_extended_metrics([sym_metric])

            # Verify report
            assert len(report) == 1

            # Get the metric from report
            metric_key = (sym_metric.metric_spec, ResultKey(datetime.date(2024, 1, 2), {}), "sales")
            metric = report[metric_key]

            # Verify the metric
            assert metric.value == 0.5
            assert metric.dataset == "sales"
            assert metric.metadata is not None
            assert metric.metadata.execution_id == execution_id

            # Verify cache was called
            mock_cache.put.assert_called_once()
            mock_cache.write_back.assert_called_once()

    def test_analyzer_analyze_extended_metrics_failure(self) -> None:
        """Test analyze_extended_metrics with Failure results."""
        # Create mock components
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.8)

        # Create a simple metric first
        provider.average("revenue", dataset="sales")

        # Mock the function to return Failure
        mock_fn = Mock(return_value=Failure("Previous day value not found"))

        # Create symbolic metric for testing
        sym_metric = SymbolicMetric(
            name="day_over_day(average(revenue))",
            symbol=sp.Symbol("x_1"),
            metric_spec=specs.DayOverDay.from_base_spec(specs.Average("revenue")),
            fn=mock_fn,
            lag=0,
            data_av_ratio=1.0,
            required_metrics=[],
            dataset="sales",
        )

        # Create analyzer
        analyzer = Analyzer(
            datasources=[],
            provider=provider,
            key=ResultKey(datetime.date(2024, 1, 2), {}),
            execution_id=execution_id,
            data_av_threshold=0.9,
        )

        # Mock cache on the provider
        mock_cache = Mock(spec=MetricCache)
        with patch.object(analyzer.provider, "_cache", mock_cache):
            # Mock logger to verify warning
            with patch.object(logging.getLogger("dqx.analyzer"), "warning") as mock_warning:
                # Analyze extended metrics
                report = analyzer.analyze_extended_metrics([sym_metric])

                # Verify report is empty (metric was not added due to failure)
                assert len(report) == 0

                # Verify warning was logged
                mock_warning.assert_called_once()
                warning_msg = mock_warning.call_args[0][0]
                assert "Failed to evaluate" in warning_msg
                assert "day_over_day(average(revenue))" in warning_msg

            # Verify cache was still flushed
            mock_cache.write_back.assert_called_once()

    def test_analyzer_analyze_main_workflow(self) -> None:
        """Test the main analyze() workflow end-to-end."""
        # Create real data source
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 1, 2)
        ds = CommercialDataSource(
            start_date=start_date, end_date=end_date, name="test_data", records_per_day=10, seed=100
        )

        # Create provider and register metrics
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-main")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.8)

        # Register simple metrics
        provider.sum("price", dataset="test_data")
        provider.average("price", dataset="test_data")
        provider.num_rows(dataset="test_data")

        # Create analyzer
        analyzer = Analyzer(
            datasources=[cast(SqlDataSource, ds)],
            provider=provider,
            key=ResultKey(datetime.date(2024, 1, 1), {}),
            execution_id=execution_id,
            data_av_threshold=0.0,  # Accept all metrics
        )

        # Analyze
        report = analyzer.analyze()

        # Verify report contains all metrics
        assert len(report) == 3  # 3 metrics for 1 date

        # Verify all metrics are in DB
        all_metrics = db.get_by_execution_id(execution_id)
        assert len(all_metrics) >= 3

    def test_analyzer_data_availability_filtering(self) -> None:
        """Test that metrics are filtered by data availability threshold."""
        # Create mock components
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.8)

        # Create metrics with different data availability
        # Mock functions for metrics
        mock_fn_high = Mock(return_value=Success(100.0))
        mock_fn_low = Mock(return_value=Success(50.0))

        high_av_metric = SymbolicMetric(
            name="sum(revenue)",
            symbol=sp.Symbol("x_1"),
            metric_spec=specs.Sum("revenue"),
            fn=mock_fn_high,
            lag=0,
            data_av_ratio=0.95,  # Above threshold
            required_metrics=[],
            dataset="sales",
        )

        low_av_metric = SymbolicMetric(
            name="average(cost)",
            symbol=sp.Symbol("x_2"),
            metric_spec=specs.Average("cost"),
            fn=mock_fn_low,
            lag=0,
            data_av_ratio=0.5,  # Below threshold
            required_metrics=[],
            dataset="sales",
        )

        # Mock provider registry
        mock_registry = Mock()
        mock_registry.metrics = [high_av_metric, low_av_metric]

        # Create analyzer with high threshold
        analyzer = Analyzer(
            datasources=[],
            provider=provider,
            key=ResultKey(datetime.date(2024, 1, 1), {}),
            execution_id=execution_id,
            data_av_threshold=0.9,
        )

        # Mock the registry by accessing the private attribute
        with patch.object(analyzer.provider, "_registry", mock_registry):
            # Mock analyze_simple_metrics and analyze_extended_metrics
            with patch.object(analyzer, "analyze_simple_metrics", return_value=AnalysisReport()):
                with patch.object(analyzer, "analyze_extended_metrics", return_value=AnalysisReport()) as mock_extended:
                    # Run analyze
                    analyzer.analyze()

                    # Verify only high availability metric was processed
                    called_metrics = mock_extended.call_args[0][0]
                    assert len(called_metrics) == 1
                    assert called_metrics[0] == high_av_metric

    def test_analyzer_analyze_empty_metrics(self) -> None:
        """Test analyze with no metrics raises error."""
        # Create minimal components
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.8)

        mock_ds = Mock(spec=SqlDataSource)
        mock_ds.name = "test_ds"

        # Create analyzer
        analyzer = Analyzer(
            datasources=[mock_ds],
            provider=provider,
            key=ResultKey(datetime.date(2024, 1, 1), {}),
            execution_id=execution_id,
            data_av_threshold=0.9,
        )

        # Try to analyze simple metrics with empty dict
        with pytest.raises(DQXError) as exc_info:
            analyzer.analyze_simple_metrics(mock_ds, {})

        assert "No metrics provided for batch analysis!" in str(exc_info.value)

    def test_analyzer_batch_size_splitting(self) -> None:
        """Test that large date ranges are split into batches."""
        # Create mock datasource
        mock_ds = Mock(spec=SqlDataSource)
        mock_ds.name = "test_ds"
        mock_ds.cte.return_value = "SELECT * FROM test"
        mock_ds.dialect = "duckdb"

        # Mock query to return proper results
        def mock_query_result(sql: str) -> MagicMock:
            result = MagicMock()
            # Return results for whatever dates are in the query
            # This is a simplified mock - in reality we'd parse the SQL
            result.fetchall.return_value = []
            return result

        mock_ds.query.side_effect = mock_query_result

        # Create analyzer
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.8)
        analyzer = Analyzer(
            datasources=[mock_ds],
            provider=provider,
            key=ResultKey(datetime.date(2024, 1, 1), {}),
            execution_id=execution_id,
            data_av_threshold=0.9,
        )

        # Create metrics for 20 dates (more than DEFAULT_BATCH_SIZE=14)
        metrics_by_key: dict[ResultKey, list[MetricSpec]] = {}
        for day in range(1, 21):
            date = datetime.date(2024, 1, day)
            key = ResultKey(date, {})
            metrics_by_key[key] = [specs.Sum("revenue")]

        # Mock _analyze_internal to track batch sizes
        batch_sizes: list[int] = []

        def mock_analyze_internal(ds: SqlDataSource, batch: dict[ResultKey, list[MetricSpec]]) -> AnalysisReport:
            batch_sizes.append(len(batch))
            # Return empty report
            return AnalysisReport()

        with patch.object(analyzer, "_analyze_internal", side_effect=mock_analyze_internal):
            # Analyze
            analyzer.analyze_simple_metrics(mock_ds, metrics_by_key)

        # Verify batches were split correctly
        assert len(batch_sizes) == 2  # Should have 2 batches
        assert batch_sizes[0] == 14  # First batch should be DEFAULT_BATCH_SIZE
        assert batch_sizes[1] == 6  # Second batch should have remaining 6 dates
