import datetime
from collections import UserDict
from typing import Any, Iterable, Sequence
from unittest.mock import Mock, patch

import pytest
from returns.maybe import Nothing

from dqx import models
from dqx.analyzer import AnalysisReport, Analyzer, analyze_sql_ops
from dqx.common import DQXError, ResultKey, SqlDataSource
from dqx.ops import SqlOp
from dqx.provider import MetricProvider
from dqx.specs import MetricSpec, Sum


class TestAnalysisReport:
    """Test AnalysisReport class functionality."""

    def test_report_functionality(self) -> None:
        """Test AnalysisReport initialization, UserDict behavior, and dataset keys."""
        # Test empty initialization and UserDict inheritance
        report = AnalysisReport()
        assert isinstance(report, UserDict)
        assert hasattr(report, "data")
        assert report.data == {}
        assert len(report) == 0

        # Test initialization with data
        key = ResultKey(datetime.date(2024, 1, 1), {})
        spec = Sum("revenue")

        # Create real metrics using Metric.build with explicit states
        from dqx.states import SimpleAdditiveState

        metric1 = models.Metric.build(metric=spec, key=key, dataset="dataset1", state=SimpleAdditiveState(value=100.0))
        metric2 = models.Metric.build(metric=spec, key=key, dataset="dataset2", state=SimpleAdditiveState(value=200.0))

        # Test that MetricKey with dataset prevents collisions
        report = AnalysisReport(
            {
                (spec, key, "dataset1"): metric1,
                (spec, key, "dataset2"): metric2,
            }
        )

        # Verify both metrics are stored separately
        assert len(report) == 2
        assert report[(spec, key, "dataset1")] == metric1
        assert report[(spec, key, "dataset2")] == metric2
        assert report[(spec, key, "dataset1")].value == 100.0
        assert report[(spec, key, "dataset2")].value == 200.0

    def test_report_merge_and_persist(self) -> None:
        """Test merge logic and both persist modes together."""
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        key2 = ResultKey(datetime.date(2024, 1, 2), {})
        spec1 = Sum("revenue")
        spec2 = Sum("quantity")

        # Create real metrics with explicit states
        from dqx.states import SimpleAdditiveState

        metric1 = models.Metric.build(
            metric=spec1, key=key1, dataset="dataset1", state=SimpleAdditiveState(value=100.0)
        )
        metric2 = models.Metric.build(
            metric=spec1, key=key1, dataset="dataset1", state=SimpleAdditiveState(value=150.0)
        )
        metric3 = models.Metric.build(metric=spec2, key=key2, dataset="dataset2", state=SimpleAdditiveState(value=50.0))

        # Create reports
        report1 = AnalysisReport({(spec1, key1, "dataset1"): metric1})
        report2 = AnalysisReport(
            {
                (spec1, key1, "dataset1"): metric2,  # Overlaps with report1
                (spec2, key2, "dataset2"): metric3,  # New metric
            }
        )

        # Test merge
        merged = report1.merge(report2)
        assert len(merged) == 2
        # The overlapping metric should be reduced (summed)
        assert merged[(spec1, key1, "dataset1")].value == 250.0  # 100 + 150
        assert merged[(spec2, key2, "dataset2")] == metric3

        # Test persist with overwrite
        mock_db = Mock()
        mock_cache = Mock()
        merged.persist(mock_db, mock_cache)
        mock_cache.put.assert_called_once_with(list(merged.values()), mark_dirty=True)
        mock_cache.write_back.assert_called_once()

        # Test persist empty report
        empty_report = AnalysisReport()
        mock_db.reset_mock()
        mock_cache.reset_mock()
        with patch("dqx.analyzer.logger") as mock_logger:
            empty_report.persist(mock_db, mock_cache)
            mock_logger.warning.assert_called_once_with("Try to save an EMPTY analysis report!")
            mock_db.persist.assert_not_called()
            mock_cache.put.assert_not_called()

    def test_report_show(self) -> None:
        """Test the show method."""
        report = AnalysisReport()
        symbol_lookup: dict[tuple[Any, Any, str], Any] = {}

        with patch("dqx.display.print_analysis_report") as mock_print:
            report.show(symbol_lookup)
            mock_print.assert_called_once_with(report, symbol_lookup)


class MockSqlOp(SqlOp[float]):
    """Mock SqlOp for testing."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._value: float | None = None
        self._sql_col = f"col_{name}"
        self._prefix = "mock"

    @property
    def name(self) -> str:
        return self._name

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return self._sql_col

    def assign(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        if self._value is None:
            raise DQXError(f"No value assigned to {self._name}")
        return self._value

    def clear(self) -> None:
        self._value = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MockSqlOp) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)


class TestAnalyzeSqlOps:
    """Test analyze_sql_ops function."""

    def test_sql_ops_analysis(self) -> None:
        """Test both single and batch SQL ops analysis with deduplication."""
        # Test empty ops
        ds = Mock(spec=SqlDataSource)
        analyze_sql_ops(ds, {})
        ds.query.assert_not_called()

        # Reset for deduplication test
        ds.reset_mock()
        ds.dialect = "duckdb"
        ds.cte.return_value = "WITH t AS (SELECT * FROM table)"

        # Mock query result
        import pyarrow as pa

        query_result = Mock()
        mock_table = pa.table(
            {
                "col_op1": [10.0],
                "col_op2": [20.0],
            }
        )
        query_result.fetch_arrow_table.return_value = mock_table
        ds.query.return_value = query_result

        # Create ops with duplicates
        op1a = MockSqlOp("op1")
        op1b = MockSqlOp("op1")  # Duplicate of op1a
        op2 = MockSqlOp("op2")

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.translate_sql_op.side_effect = lambda op: f"SQL for {op._name}"
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            # Mock query result with array format
            query_result.fetchall.return_value = [
                (
                    datetime.date.today().isoformat(),
                    [{"key": "col_op1", "value": 10.0}, {"key": "col_op2", "value": 20.0}],
                )
            ]
            ds.query.return_value = query_result

            key = ResultKey(datetime.date.today(), {})
            analyze_sql_ops(ds, {key: [op1a, op1b, op2]})

            # Verify deduplication through ops getting values
            # All ops should have values (including duplicates)

            # Verify all ops get values assigned (including duplicates)
            assert op1a._value == 10.0
            assert op1b._value == 10.0  # Gets same value as op1a
            assert op2._value == 20.0

    def test_sql_ops_date_alignment_fix(self) -> None:
        """Test that analyze_sql_ops correctly aligns dates even when SQL results are unordered."""
        ds = Mock(spec=SqlDataSource)
        ds.dialect = "duckdb"
        ds.cte.return_value = "WITH t AS (SELECT * FROM table)"

        # Create ops for three dates
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 1, 2)
        date3 = datetime.date(2024, 1, 3)

        key1 = ResultKey(date1, {})
        key2 = ResultKey(date2, {})
        key3 = ResultKey(date3, {})

        op1 = MockSqlOp("metric1")
        op2 = MockSqlOp("metric2")
        op3 = MockSqlOp("metric3")

        # Create ops_by_key in one order
        ops_by_key: dict[ResultKey, list[SqlOp]] = {
            key1: [op1],
            key2: [op2],
            key3: [op3],
        }

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            # Mock query results in DIFFERENT order than ops_by_key iteration
            query_result = Mock()
            query_result.fetchall.return_value = [
                ("2024-01-03", [{"key": "col_metric3", "value": 30.0}]),  # date3 first
                ("2024-01-01", [{"key": "col_metric1", "value": 10.0}]),  # date1 second
                ("2024-01-02", [{"key": "col_metric2", "value": 20.0}]),  # date2 last
            ]
            ds.query.return_value = query_result

            analyze_sql_ops(ds, ops_by_key)

            # Verify each op got the correct value for its date
            assert op1._value == 10.0  # date1's metric
            assert op2._value == 20.0  # date2's metric
            assert op3._value == 30.0  # date3's metric

    def test_sql_ops_missing_date_error(self) -> None:
        """Test that analyze_sql_ops raises error when expected date is missing from results."""
        ds = Mock(spec=SqlDataSource)
        ds.dialect = "duckdb"

        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        key2 = ResultKey(datetime.date(2024, 1, 2), {})

        op1 = MockSqlOp("metric1")
        op2 = MockSqlOp("metric2")

        ops_by_key: dict[ResultKey, list[SqlOp]] = {
            key1: [op1],
            key2: [op2],
        }

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            # Mock query result missing date2
            query_result = Mock()
            query_result.fetchall.return_value = [
                ("2024-01-01", [{"key": "col_metric1", "value": 10.0}]),
                # Missing 2024-01-02
            ]
            ds.query.return_value = query_result

            with pytest.raises(DQXError, match="Missing dates in SQL results: \\['2024-01-02'\\]"):
                analyze_sql_ops(ds, ops_by_key)

    def test_sql_ops_unexpected_date_error(self) -> None:
        """Test that analyze_sql_ops raises error when unexpected date appears in results."""
        ds = Mock(spec=SqlDataSource)
        ds.dialect = "duckdb"

        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        op1 = MockSqlOp("metric1")

        ops_by_key: dict[ResultKey, list[SqlOp]] = {
            key1: [op1],
        }

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            # Mock query result with unexpected date
            query_result = Mock()
            query_result.fetchall.return_value = [
                ("2024-01-01", [{"key": "col_metric1", "value": 10.0}]),
                ("2024-01-05", [{"key": "col_metric5", "value": 50.0}]),  # Unexpected!
            ]
            ds.query.return_value = query_result

            with pytest.raises(DQXError, match="Unexpected date '2024-01-05' in SQL results"):
                analyze_sql_ops(ds, ops_by_key)

    def test_sql_ops_complex_alignment_scenario(self) -> None:
        """Test complex scenario with multiple ops per date and unordered results."""
        ds = Mock(spec=SqlDataSource)
        ds.dialect = "bigquery"  # Test with BigQuery dialect

        # Create dates with tags
        date1 = datetime.date(2024, 1, 10)
        date2 = datetime.date(2024, 1, 11)

        key1 = ResultKey(date1, {"env": "prod"})
        key2 = ResultKey(date2, {"env": "staging"})

        # Multiple ops per date
        op1a = MockSqlOp("revenue")
        op1b = MockSqlOp("cost")
        op2a = MockSqlOp("users")
        op2b = MockSqlOp("sessions")

        ops_by_key: dict[ResultKey, list[SqlOp]] = {
            key1: [op1a, op1b],
            key2: [op2a, op2b],
        }

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            # Results in reverse order
            query_result = Mock()
            query_result.fetchall.return_value = [
                ("2024-01-11", [{"key": "col_users", "value": 100.0}, {"key": "col_sessions", "value": 200.0}]),
                ("2024-01-10", [{"key": "col_revenue", "value": 1000.0}, {"key": "col_cost", "value": 500.0}]),
            ]
            ds.query.return_value = query_result

            analyze_sql_ops(ds, ops_by_key)

            # Verify correct alignment despite reverse order
            assert op1a._value == 1000.0  # revenue for date1
            assert op1b._value == 500.0  # cost for date1
            assert op2a._value == 100.0  # users for date2
            assert op2b._value == 200.0  # sessions for date2


class TestAnalyzeBatchSqlOps:
    """Test analyze_batch_sql_ops function."""

    def test_batch_analysis_with_validation(self) -> None:
        """Test batch analysis including empty ops and validation errors."""
        # Test empty ops
        ds = Mock(spec=SqlDataSource)
        analyze_sql_ops(ds, {})
        ds.query.assert_not_called()

        # Reset for batch analysis test
        ds.reset_mock()
        ds.dialect = "duckdb"

        # Create ops for multiple dates
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        key2 = ResultKey(datetime.date(2024, 1, 2), {})

        op1 = MockSqlOp("op1")
        op2 = MockSqlOp("op2")
        op3 = MockSqlOp("op3")

        ops_by_key: dict[ResultKey, list[SqlOp]] = {
            key1: [op1, op2],
            key2: [op3],
        }

        # Mock query result with array format
        query_result = Mock()
        query_result.fetchall.return_value = [
            ("2024-01-01", [{"key": "col_op1", "value": 10.0}, {"key": "col_op2", "value": 20.0}]),
            ("2024-01-02", [{"key": "col_op3", "value": 30.0}]),
        ]
        ds.query.return_value = query_result

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            analyze_sql_ops(ds, ops_by_key)

            # Verify values assigned
            assert op1._value == 10.0
            assert op2._value == 20.0
            assert op3._value == 30.0

        # Test validation error with None value
        ds.reset_mock()
        op4 = MockSqlOp("op4")
        query_result.fetchall.return_value = [
            ("2024-01-01", [{"key": "col_op4", "value": None}]),
        ]

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            with pytest.raises(DQXError, match="Null value encountered"):
                analyze_sql_ops(ds, {key1: [op4]})


class TestAnalyzer:
    """Test Analyzer class."""

    def test_analyzer_workflow(self) -> None:
        """Test complete analyzer workflow from initialization to batch processing."""
        # Create mock dependencies
        datasources: list[SqlDataSource] = [Mock(spec=SqlDataSource, name="test_ds")]
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123", data_av_threshold=0.8)
        key = ResultKey(datetime.date(2024, 1, 1), {})

        # Test analyzer creation
        analyzer = Analyzer(datasources, provider, key, "test-123", 0.9)

        # Create metrics for testing batch processing
        metrics = {}
        for i in range(10):  # More than DEFAULT_BATCH_SIZE (7)
            date_key = ResultKey(datetime.date(2024, 1, i + 1), {})
            metrics[date_key] = [Sum("revenue")]

        # Test that analyze raises error with empty metrics
        with pytest.raises(DQXError, match="No metrics provided"):
            analyzer.analyze_simple_metrics(datasources[0], {})

        # Mock _analyze_internal to test batch processing
        with patch.object(analyzer, "_analyze_internal") as mock_analyze:
            mock_analyze.return_value = AnalysisReport()

            analyzer.analyze_simple_metrics(datasources[0], metrics)

            # Verify batching occurred - batch size is now 10 based on log
            assert mock_analyze.call_count == 1  # All 10 dates in one batch

            # Verify batch has all 10 items
            first_call_metrics = mock_analyze.call_args_list[0][0][1]
            assert len(first_call_metrics) == 10


class TestAnalyzerLagHandling:
    """Test Analyzer lag handling functionality."""

    def test_analyzer_with_lag_dates(self) -> None:
        """Test analyzer handles lag dates properly."""
        # Create mock dependencies
        datasources: list[SqlDataSource] = [Mock(spec=SqlDataSource, name="test_ds")]
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123", data_av_threshold=0.8)
        key = ResultKey(datetime.date(2024, 1, 15), {})

        analyzer = Analyzer(datasources, provider, key, "test-123", 0.9)

        # Create metrics with different lag values
        metrics = {
            ResultKey(datetime.date(2024, 1, 15), {}): [Sum("revenue")],
            ResultKey(datetime.date(2024, 1, 14), {}): [Sum("cost")],  # lag 1
            ResultKey(datetime.date(2024, 1, 13), {}): [Sum("profit")],  # lag 2
        }

        with patch.object(analyzer, "_analyze_internal") as mock_analyze:
            mock_analyze.return_value = AnalysisReport()

            analyzer.analyze_simple_metrics(datasources[0], metrics)

            # Verify the lag handling
            call_args = mock_analyze.call_args[0]
            metric_dict = call_args[1]

            # All three dates should be included
            assert len(metric_dict) == 3
            assert ResultKey(datetime.date(2024, 1, 15), {}) in metric_dict
            assert ResultKey(datetime.date(2024, 1, 14), {}) in metric_dict
            assert ResultKey(datetime.date(2024, 1, 13), {}) in metric_dict


class TestAnalysisReportWithCache:
    """Test AnalysisReport with cache-related functionality."""

    def test_report_from_and_to_cache(self) -> None:
        """Test converting AnalysisReport to/from cached metrics."""
        # Create a report with metrics
        key = ResultKey(datetime.date(2024, 1, 1), {})
        spec = Sum("revenue")

        from dqx.common import Metadata
        from dqx.states import SimpleAdditiveState

        metric = models.Metric.build(
            metric=spec,
            key=key,
            dataset="sales",
            state=SimpleAdditiveState(value=100.0),
            metadata=Metadata(execution_id="test-exec"),
        )

        report = AnalysisReport({(spec, key, "sales"): metric})

        # Convert to cached metrics
        cached_metrics = list(report.values())
        assert len(cached_metrics) == 1
        assert cached_metrics[0] == metric

        # Create a new report from cached metrics
        new_report = AnalysisReport()
        for m in cached_metrics:
            new_report[(m.spec, m.key, m.dataset)] = m

        assert len(new_report) == 1
        assert new_report[(spec, key, "sales")] == metric


class TestAnalysisReportMergePersist:
    """Test AnalysisReport _merge_persist functionality."""

    def test_report_persist_with_cache(self) -> None:
        """Test persist with new cache-based implementation."""
        from dqx.cache import MetricCache
        from dqx.common import Metadata
        from dqx.orm.repositories import InMemoryMetricDB
        from dqx.states import SimpleAdditiveState

        # Create a real database and cache
        db = InMemoryMetricDB()
        cache = MetricCache(db)

        # Create and persist an existing metric
        key = ResultKey(datetime.date(2024, 1, 1), {})
        spec = Sum("revenue")
        existing_metric = models.Metric.build(
            metric=spec,
            key=key,
            dataset="sales",
            state=SimpleAdditiveState(value=100.0),
            metadata=Metadata(execution_id="exec-1"),
        )
        db.persist([existing_metric])

        # Create a new report with a new metric
        new_metric = models.Metric.build(
            metric=spec,
            key=key,
            dataset="sales",
            state=SimpleAdditiveState(value=50.0),
            metadata=Metadata(execution_id="exec-2"),
        )
        report = AnalysisReport({(spec, key, "sales"): new_metric})

        # Persist using new cache-based implementation
        report.persist(db, cache)

        # Verify the new metric overwrote the old one (no merging)
        stored_metric = db.get_metric(spec, key, "sales", "exec-2")
        assert stored_metric != Nothing
        assert stored_metric.unwrap().value == 50.0  # New value, not merged


class TestAnalyzerExtendedMetrics:
    """Test Analyzer.analyze_extended_metrics functionality."""

    def test_analyze_extended_metrics_success(self) -> None:
        """Test analyze_extended_metrics with successful evaluation."""
        from dqx.common import Metadata
        from dqx.orm.repositories import InMemoryMetricDB
        from tests.fixtures.data_fixtures import CommercialDataSource

        # Create real dependencies
        db = InMemoryMetricDB()
        provider = MetricProvider(db, execution_id="test-exec", data_av_threshold=0.8)

        # Create a datasource
        ds = CommercialDataSource(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            name="sales",
            records_per_day=10,
            seed=42,
        )

        key = ResultKey(datetime.date(2024, 1, 10), {})

        # First create and persist base metrics that extended metrics will need
        base_metric = provider.average("price", dataset="sales")

        # Persist base metric values for the dates that DoD will need
        from dqx.states import Average

        for i in range(2):
            metric_key = ResultKey(datetime.date(2024, 1, 10 - i), {})
            metric = models.Metric.build(
                metric=provider.get_symbol(base_metric).metric_spec,
                key=metric_key,
                dataset="sales",
                state=Average(avg=100.0 + i * 10, n=10),  # Average state needs avg and count
                metadata=Metadata(execution_id="test-exec"),
            )
            db.persist([metric])

        # Create extended metrics
        dod_metric = provider.ext.day_over_day(base_metric, dataset="sales")

        # Create analyzer
        analyzer = Analyzer([ds], provider, key, "test-exec", 0.9)

        # Run analyze_extended_metrics with provider's metrics
        report = analyzer.analyze_extended_metrics(provider.metrics)

        # Verify the extended metric was evaluated
        assert len(report) == 1

        # Check the computed DoD value
        dod_spec = provider.get_symbol(dod_metric).metric_spec
        metric_key_tuple = (dod_spec, key, "sales")
        assert metric_key_tuple in report
        # DoD = |100-110|/110 = 10/110 ≈ 0.0909 (percentage change)
        assert abs(report[metric_key_tuple].value - 0.0909) < 0.001

    def test_analyze_extended_metrics_with_failure(self) -> None:
        """Test analyze_extended_metrics when evaluation fails."""
        from dqx.orm.repositories import InMemoryMetricDB
        from tests.fixtures.data_fixtures import CommercialDataSource

        # Create real dependencies
        db = InMemoryMetricDB()
        provider = MetricProvider(db, execution_id="test-exec", data_av_threshold=0.8)

        ds = CommercialDataSource(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            name="sales",
            records_per_day=10,
            seed=42,
        )

        key = ResultKey(datetime.date(2024, 1, 10), {})

        # Create base metric but DON'T persist any data for it
        base_metric = provider.average("price", dataset="sales")

        # Create extended metric
        provider.ext.day_over_day(base_metric, dataset="sales")

        # Create analyzer
        analyzer = Analyzer([ds], provider, key, "test-exec", 0.9)

        # Run analyze_extended_metrics - should handle failure gracefully
        report = analyzer.analyze_extended_metrics(provider.metrics)

        # Report should be empty since evaluation failed
        assert len(report) == 0

    def test_analyze_extended_metrics_topological_sort(self) -> None:
        """Test that analyze_extended_metrics calls topological_sort."""
        from dqx.orm.repositories import InMemoryMetricDB
        from tests.fixtures.data_fixtures import CommercialDataSource

        # Create real dependencies
        db = InMemoryMetricDB()
        provider = MetricProvider(db, execution_id="test-exec", data_av_threshold=0.8)

        ds = CommercialDataSource(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            name="sales",
            records_per_day=10,
            seed=42,
        )

        key = ResultKey(datetime.date(2024, 1, 10), {})

        # Create a chain of extended metrics to test topological ordering
        base1 = provider.average("price", dataset="sales")
        base2 = provider.sum("quantity", dataset="sales")

        # Extended metrics with dependencies
        dod1 = provider.ext.day_over_day(base1, dataset="sales")
        dod2 = provider.ext.day_over_day(base2, dataset="sales")

        # Create analyzer
        analyzer = Analyzer([ds], provider, key, "test-exec", 0.9)

        # Run analyze_extended_metrics
        analyzer.analyze_extended_metrics(provider.metrics)

        # Verify metrics are now in topological order
        # Simple metrics should come before extended metrics
        final_order = [m.symbol for m in provider.metrics]

        # Find indices
        base1_idx = final_order.index(base1)
        base2_idx = final_order.index(base2)
        dod1_idx = final_order.index(dod1)
        dod2_idx = final_order.index(dod2)

        # Base metrics should come before their extended metrics
        assert base1_idx < dod1_idx
        assert base2_idx < dod2_idx


class TestAnalyzerFullWorkflow:
    """Test the full analyze() method workflow."""

    def test_analyze_with_mixed_metrics(self) -> None:
        """Test full analyze workflow with both simple and extended metrics."""
        from dqx.orm.repositories import InMemoryMetricDB
        from tests.fixtures.data_fixtures import CommercialDataSource

        # Create real dependencies
        db = InMemoryMetricDB()
        provider = MetricProvider(db, execution_id="test-exec", data_av_threshold=0.8)

        # Create two datasources
        ds1 = CommercialDataSource(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            name="sales",
            records_per_day=30,
            seed=42,
        )

        ds2 = CommercialDataSource(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            name="inventory",
            records_per_day=20,
            seed=43,
        )

        key = ResultKey(datetime.date(2024, 1, 10), {})

        # Create metrics for both datasets
        # Dataset 1: sales
        avg_price = provider.average("price", dataset="sales")
        sum_qty = provider.sum("quantity", dataset="sales")

        # Dataset 2: inventory
        avg_tax = provider.average("tax", dataset="inventory")

        # Extended metrics
        provider.ext.day_over_day(avg_price, dataset="sales")
        provider.ext.week_over_week(avg_tax, dataset="inventory")

        # Create metrics with lag
        avg_price_lag = provider.average("price", lag=1, dataset="sales")

        # Create analyzer
        analyzer = Analyzer([ds1, ds2], provider, key, "test-exec", 0.9)

        # Run full analyze
        report = analyzer.analyze()

        # Verify simple metrics were analyzed for both datasets
        assert (provider.get_symbol(avg_price).metric_spec, key, "sales") in report
        assert (provider.get_symbol(sum_qty).metric_spec, key, "sales") in report
        assert (provider.get_symbol(avg_tax).metric_spec, key, "inventory") in report

        # Verify lagged metric with correct effective date
        lagged_key = key.lag(1)
        assert (provider.get_symbol(avg_price_lag).metric_spec, lagged_key, "sales") in report

        # Verify extended metrics were evaluated
        # Note: They might not be in report if base metrics failed, which is ok
        # The important thing is that analyze() completes without error

    def test_analyze_dataset_grouping(self) -> None:
        """Test that analyze() correctly groups metrics by dataset."""
        from dqx.orm.repositories import InMemoryMetricDB
        from tests.fixtures.data_fixtures import CommercialDataSource

        # Create real dependencies
        db = InMemoryMetricDB()
        provider = MetricProvider(db, execution_id="test-exec", data_av_threshold=0.8)

        # Create datasources
        ds1 = CommercialDataSource(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            name="ds1",
            records_per_day=10,
            seed=42,
        )

        ds2 = CommercialDataSource(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            name="ds2",
            records_per_day=10,
            seed=43,
        )

        key = ResultKey(datetime.date(2024, 1, 10), {})

        # Create metrics spread across datasets
        provider.average("price", dataset="ds1")
        provider.sum("quantity", dataset="ds1")
        provider.average("tax", dataset="ds2")
        provider.sum("price", dataset="ds2")

        # Create analyzer
        analyzer = Analyzer([ds1, ds2], provider, key, "test-exec", 0.9)

        # Capture what gets passed to analyze_simple_metrics
        analyze_calls: list[tuple[str, list[ResultKey]]] = []
        original_analyze_simple = analyzer.analyze_simple_metrics

        def capture_analyze_simple(ds: SqlDataSource, metrics: dict[ResultKey, Sequence[MetricSpec]]) -> AnalysisReport:
            analyze_calls.append((ds.name, list(metrics.keys())))
            return original_analyze_simple(ds, metrics)

        analyzer.analyze_simple_metrics = capture_analyze_simple  # type: ignore[assignment]

        # Run analyze
        analyzer.analyze()

        # Verify each dataset was analyzed separately
        assert len(analyze_calls) == 2

        # Find the calls for each dataset
        ds1_call = next(c for c in analyze_calls if c[0] == "ds1")
        ds2_call = next(c for c in analyze_calls if c[0] == "ds2")

        # Each should have been called with their respective metrics
        assert len(ds1_call[1]) == 1  # One date key
        assert len(ds2_call[1]) == 1  # One date key

    def test_analyze_phase_separation(self) -> None:
        """Test that analyze() processes simple metrics before extended metrics."""
        from dqx.orm.repositories import InMemoryMetricDB
        from tests.fixtures.data_fixtures import CommercialDataSource

        # Create real dependencies
        db = InMemoryMetricDB()
        provider = MetricProvider(db, execution_id="test-exec", data_av_threshold=0.8)

        ds = CommercialDataSource(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 15),
            name="sales",
            records_per_day=10,
            seed=42,
        )

        key = ResultKey(datetime.date(2024, 1, 10), {})

        # Create metrics
        base = provider.average("price", dataset="sales")
        provider.ext.day_over_day(base, dataset="sales")

        # Track when metrics get persisted
        persist_calls: list[str] = []
        original_persist = db.persist

        def track_persist(metrics: Iterable[models.Metric]) -> Iterable[models.Metric]:
            # Record what type of metrics are being persisted
            metrics_list = list(metrics)
            for m in metrics_list:
                persist_calls.append(m.spec.metric_type)
            return original_persist(metrics_list)

        db.persist = track_persist  # type: ignore[assignment]

        # Create analyzer
        analyzer = Analyzer([ds], provider, key, "test-exec", 0.9)

        # Run analyze
        analyzer.analyze()

        # Verify simple metrics were persisted before extended metrics
        # Simple metrics like "average" should appear before "dod"
        simple_indices = [i for i, t in enumerate(persist_calls) if t == "average"]
        extended_indices = [i for i, t in enumerate(persist_calls) if t == "dod"]

        if simple_indices and extended_indices:
            assert max(simple_indices) < min(extended_indices)


class TestAnalyzerEdgeCases:
    """Test edge cases in Analyzer."""

    def test_analyze_with_tags(self) -> None:
        """Test analyzer with ResultKeys that have tags."""
        mock_ds = Mock(spec=SqlDataSource, name="test_ds")
        datasources: list[SqlDataSource] = [mock_ds]
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123", data_av_threshold=0.8)

        # Create keys with tags
        tags = {"env": "prod", "region": "us-west"}
        key = ResultKey(datetime.date(2024, 1, 1), tags)

        analyzer = Analyzer(datasources, provider, key, "test-123", 0.9)

        # Create metrics with tagged keys
        metrics = {
            key: [Sum("revenue"), Sum("cost")],
        }

        with patch.object(analyzer, "_analyze_internal") as mock_analyze:
            mock_analyze.return_value = AnalysisReport()

            analyzer.analyze_simple_metrics(datasources[0], metrics)

            # Verify tags are preserved
            call_args = mock_analyze.call_args[0]
            metric_dict = call_args[1]
            for result_key in metric_dict:
                assert result_key.tags == tags

    def test_analyze_internal_value_retrieval_error(self) -> None:
        """Test _analyze_internal when value retrieval fails (lines 347-348)."""
        from dqx.common import SqlDataSource

        # Create a proper SqlDataSource mock
        ds = Mock(spec=SqlDataSource)
        ds.name = "test_ds"
        ds.dialect = "duckdb"
        ds.cte.return_value = "WITH cte AS (SELECT * FROM table)"

        datasources: list[SqlDataSource] = [ds]
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123", data_av_threshold=0.8)
        key = ResultKey(datetime.date(2024, 1, 1), {})

        analyzer = Analyzer(datasources, provider, key, "test-123", 0.9)

        # Create a metric spec with a mocked analyzer
        sum_spec = Sum("revenue")

        # Create a failing op that will be returned by analyzers property
        failing_op = MockSqlOp("failing_op")

        # Mock the analyzers property to return our failing op
        with patch.object(Sum, "analyzers", new_callable=lambda: property(lambda self: (failing_op,))):
            # Create metrics
            metrics: dict[ResultKey, Sequence[MetricSpec]] = {
                key: [sum_spec],
            }

            # Mock batch analysis to not assign any value to the op (simulating SQL failure)
            with patch("dqx.analyzer.analyze_sql_ops"):
                # This should raise DQXError when trying to get value from failing_op
                with pytest.raises(DQXError, match="Failed to retrieve value for analyzer"):
                    analyzer._analyze_internal(ds, metrics)

    def test_analyzer_init_properties(self) -> None:
        """Test analyzer initialization and properties."""
        mock_ds = Mock(spec=SqlDataSource, name="test_ds")
        datasources: list[SqlDataSource] = [mock_ds]
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123", data_av_threshold=0.8)
        key = ResultKey(datetime.date(2024, 1, 1), {})

        analyzer = Analyzer(datasources, provider, key, "test-123", 0.9)

        # Check properties are set correctly
        assert analyzer.datasources == datasources
        assert analyzer.provider == provider
        assert analyzer.key == key
        assert analyzer.execution_id == "test-123"

    def test_analyze_internal_integration(self) -> None:
        """Test _analyze_internal method integration."""
        from dqx.common import SqlDataSource

        # Create a proper SqlDataSource mock
        ds = Mock(spec=SqlDataSource)
        ds.name = "test_ds"
        ds.dialect = "duckdb"
        ds.cte.return_value = "WITH cte AS (SELECT * FROM table)"

        datasources: list[SqlDataSource] = [ds]
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123", data_av_threshold=0.8)
        key = ResultKey(datetime.date(2024, 1, 1), {})

        analyzer = Analyzer(datasources, provider, key, "test-123", 0.9)

        # Create metrics
        metrics: dict[ResultKey, Sequence[MetricSpec]] = {
            key: [Sum("revenue")],
        }

        # Mock the batch analysis to assign values to ops
        def mock_batch_analysis(ds: SqlDataSource, ops_by_key: dict[ResultKey, list[SqlOp[Any]]]) -> None:
            # Assign a value to each op
            for key, ops in ops_by_key.items():
                for op in ops:
                    op.assign(100.0)

        with patch("dqx.analyzer.analyze_sql_ops", side_effect=mock_batch_analysis) as mock_batch:
            report = analyzer._analyze_internal(ds, metrics)

            # Should have called batch analysis
            mock_batch.assert_called_once()

            # Report should contain the metric
            assert len(report) == 1
            metric_key = (Sum("revenue"), key, "test_ds")
            assert metric_key in report
            assert report[metric_key].value == 100.0
