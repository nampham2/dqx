import datetime
from collections import UserDict
from typing import Any
from unittest.mock import Mock, patch

import pytest

from dqx import models
from dqx.analyzer import AnalysisReport, Analyzer, analyze_batch_sql_ops, analyze_sql_ops
from dqx.common import DQXError, ResultKey, SqlDataSource
from dqx.ops import SqlOp
from dqx.provider import MetricProvider
from dqx.specs import Sum


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
        merged.persist(mock_db, mock_cache, overwrite=True)
        mock_db.persist.assert_called_once()
        persisted_metrics = list(mock_db.persist.call_args[0][0])
        assert len(persisted_metrics) == 2
        mock_cache.put.assert_called_once()

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
        analyze_sql_ops(ds, [], datetime.date.today())
        ds.query.assert_not_called()

        # Reset for deduplication test
        ds.reset_mock()
        ds.dialect = "duckdb"
        ds.cte.return_value = "WITH t AS (SELECT * FROM table)"

        # Mock query result
        query_result = Mock()
        query_result.fetchnumpy.return_value = {
            "col_op1": [10.0],
            "col_op2": [20.0],
        }
        ds.query.return_value = query_result

        # Create ops with duplicates
        op1a = MockSqlOp("op1")
        op1b = MockSqlOp("op1")  # Duplicate of op1a
        op2 = MockSqlOp("op2")

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.translate_sql_op.side_effect = lambda op: f"SQL for {op._name}"
            mock_dialect.build_cte_query.return_value = "SELECT ..."
            mock_get_dialect.return_value = mock_dialect

            analyze_sql_ops(ds, [op1a, op1b, op2], datetime.date.today())

            # Verify deduplication
            assert mock_dialect.translate_sql_op.call_count == 2  # Only unique ops

            # Verify all ops get values assigned (including duplicates)
            assert op1a._value == 10.0
            assert op1b._value == 10.0  # Gets same value as op1a
            assert op2._value == 20.0


class TestAnalyzeBatchSqlOps:
    """Test analyze_batch_sql_ops function."""

    def test_batch_analysis_with_validation(self) -> None:
        """Test batch analysis including empty ops and validation errors."""
        # Test empty ops
        ds = Mock(spec=SqlDataSource)
        analyze_batch_sql_ops(ds, {})
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

        # Mock query result with MAP values
        query_result = Mock()
        query_result.fetchall.return_value = [
            ("2024-01-01", {"col_op1": 10.0, "col_op2": 20.0}),
            ("2024-01-02", {"col_op3": 30.0}),
        ]
        ds.query.return_value = query_result

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            analyze_batch_sql_ops(ds, ops_by_key)

            # Verify values assigned
            assert op1._value == 10.0
            assert op2._value == 20.0
            assert op3._value == 30.0

        # Test validation error with None value
        ds.reset_mock()
        op4 = MockSqlOp("op4")
        query_result.fetchall.return_value = [
            ("2024-01-01", {"col_op4": None}),
        ]

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            with pytest.raises(DQXError, match="Null value encountered"):
                analyze_batch_sql_ops(ds, {key1: [op4]})


class TestAnalyzer:
    """Test Analyzer class."""

    def test_analyzer_workflow(self) -> None:
        """Test complete analyzer workflow from initialization to batch processing."""
        # Create mock dependencies
        datasources: list[SqlDataSource] = [Mock(spec=SqlDataSource, name="test_ds")]
        mock_db = Mock()
        provider = MetricProvider(mock_db, execution_id="test-123")
        key = ResultKey(datetime.date(2024, 1, 1), {})

        # Test analyzer creation
        analyzer = Analyzer(datasources, provider, key, "test-123")

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
