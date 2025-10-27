import datetime
from collections import UserDict
from unittest.mock import Mock, patch

import pytest

from dqx import models
from dqx.analyzer import AnalysisReport, Analyzer, MetricKey, analyze_batch_sql_ops, analyze_sql_ops
from dqx.common import DQXError, Metadata, ResultKey, SqlDataSource
from dqx.ops import SqlOp
from dqx.specs import MetricSpec, Sum


class TestAnalysisReport:
    """Test AnalysisReport class functionality."""

    def test_report_inherits_userdict(self) -> None:
        """Test that AnalysisReport properly inherits from UserDict."""
        report = AnalysisReport()
        assert isinstance(report, UserDict)
        assert hasattr(report, "data")
        assert report.data == {}

    def test_report_initialization(self) -> None:
        """Test AnalysisReport initialization."""
        # Empty initialization
        report1 = AnalysisReport()
        assert len(report1) == 0
        assert report1.symbol_mapping == {}

        # Initialize with data
        key = ResultKey(datetime.date(2024, 1, 1), {})
        metric = Mock(spec=models.Metric)
        spec: MetricSpec = Sum("revenue")  # Use concrete MetricSpec with type annotation
        data: dict[MetricKey, models.Metric] = {(spec, key, "dataset1"): metric}
        report2 = AnalysisReport(data)
        assert len(report2) == 1
        assert report2.symbol_mapping == {}

    def test_report_merge(self) -> None:
        """Test merging two AnalysisReports."""
        # Create mock metrics with reduce capability
        metric1 = Mock(spec=models.Metric)
        metric2 = Mock(spec=models.Metric)
        metric3 = Mock(spec=models.Metric)
        merged_metric = Mock(spec=models.Metric)

        # Configure reduce to return a merged metric
        with patch.object(models.Metric, "reduce", return_value=merged_metric) as mock_reduce:
            key1 = ResultKey(datetime.date(2024, 1, 1), {})
            key2 = ResultKey(datetime.date(2024, 1, 2), {})
            spec1: MetricSpec = Sum("revenue")  # Use concrete MetricSpec with type annotation
            spec2: MetricSpec = Sum("quantity")  # Use concrete MetricSpec with type annotation

            # Report 1
            report1 = AnalysisReport({(spec1, key1, "dataset1"): metric1})
            report1.symbol_mapping = {(spec1, key1, "dataset1"): "x_1"}

            # Report 2 with overlapping and new metrics
            report2 = AnalysisReport(
                {
                    (spec1, key1, "dataset1"): metric2,  # Overlaps with report1
                    (spec2, key2, "dataset2"): metric3,  # New metric
                }
            )
            report2.symbol_mapping = {
                (spec1, key1, "dataset1"): "x_1",
                (spec2, key2, "dataset2"): "x_2",
            }

            # Merge
            merged = report1.merge(report2)

            # Verify merge
            assert len(merged) == 2
            assert merged[(spec1, key1, "dataset1")] == merged_metric  # Used reduce
            assert merged[(spec2, key2, "dataset2")] == metric3  # Just added
            assert merged.symbol_mapping == {
                (spec1, key1, "dataset1"): "x_1",
                (spec2, key2, "dataset2"): "x_2",
            }

            # Verify reduce was called for the overlapping metric
            mock_reduce.assert_called_once_with([metric1, metric2])

    def test_report_show(self) -> None:
        """Test the show method."""
        report = AnalysisReport()
        with patch("dqx.display.print_analysis_report") as mock_print:
            report.show("test_ds")
            mock_print.assert_called_once_with({"test_ds": report})

    def test_persist_empty_report(self) -> None:
        """Test persisting an empty report logs warning."""
        report = AnalysisReport()
        mock_db = Mock()

        with patch("dqx.analyzer.logger") as mock_logger:
            report.persist(mock_db)
            mock_logger.warning.assert_called_once_with("Try to save an EMPTY analysis report!")
            mock_db.persist.assert_not_called()

    def test_persist_overwrite(self) -> None:
        """Test persisting with overwrite=True."""
        metric = Mock(spec=models.Metric)
        key = ResultKey(datetime.date(2024, 1, 1), {})
        report = AnalysisReport({(Sum("revenue"), key, "dataset1"): metric})
        mock_db = Mock()

        report.persist(mock_db, overwrite=True)
        # The persist method calls db.persist with report.values() which is a ValuesView
        mock_db.persist.assert_called_once()
        # Check that the values passed contain our metric
        call_args = mock_db.persist.call_args[0][0]
        assert list(call_args) == [metric]

    def test_persist_merge(self) -> None:
        """Test persisting with overwrite=False (merge mode)."""
        # Setup
        spec: MetricSpec = Sum("revenue")  # Use concrete MetricSpec with type annotation
        key = ResultKey(datetime.date(2024, 1, 1), {})
        metric = Mock(spec=models.Metric)
        metric.key = key
        metric.spec = spec

        db_metric = Mock(spec=models.Metric)
        merged_metric = Mock(spec=models.Metric)

        report = AnalysisReport({(spec, key, "dataset1"): metric})
        mock_db = Mock()
        mock_db.get.return_value = Mock(unwrap=Mock(return_value=db_metric))

        # Configure merge
        with patch.object(AnalysisReport, "merge") as mock_merge:
            mock_merge.return_value = AnalysisReport({(spec, key, "dataset1"): merged_metric})

            report.persist(mock_db, overwrite=False)

            # Verify
            mock_db.get.assert_called_once_with(key, spec)
            # The persist method calls db.persist with merged_report.values()
            mock_db.persist.assert_called_once()
            # Check that the values passed contain our merged metric
            call_args = mock_db.persist.call_args[0][0]
            assert list(call_args) == [merged_metric]

    def test_report_with_dataset_in_key(self) -> None:
        """Test that MetricKey with dataset prevents collisions."""
        # Create two metrics with same spec and key but different datasets
        spec = Sum("revenue")
        key = ResultKey(datetime.date(2024, 1, 1), {})

        metric1 = Mock(spec=models.Metric)
        metric1.value = 100.0
        metric2 = Mock(spec=models.Metric)
        metric2.value = 200.0

        # Create report with both metrics
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

    def test_empty_ops(self) -> None:
        """Test with empty ops list."""
        ds = Mock(spec=SqlDataSource)
        analyze_sql_ops(ds, [], datetime.date.today())
        # Should not call any methods on ds
        ds.query.assert_not_called()

    def test_deduplication(self) -> None:
        """Test that duplicate ops are deduplicated."""
        ds = Mock(spec=SqlDataSource)
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

    def test_empty_ops(self) -> None:
        """Test with empty ops dictionary."""
        ds = Mock(spec=SqlDataSource)
        analyze_batch_sql_ops(ds, {})
        ds.query.assert_not_called()

    def test_batch_analysis(self) -> None:
        """Test batch analysis with multiple dates."""
        ds = Mock(spec=SqlDataSource)
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

    def test_validation_errors(self) -> None:
        """Test value validation in batch analysis."""
        ds = Mock(spec=SqlDataSource)
        ds.dialect = "duckdb"

        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        op1 = MockSqlOp("op1")

        # Test with None value
        query_result = Mock()
        query_result.fetchall.return_value = [
            ("2024-01-01", {"col_op1": None}),
        ]
        ds.query.return_value = query_result

        with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
            mock_get_dialect.return_value = mock_dialect

            with pytest.raises(DQXError, match="Null value encountered"):
                analyze_batch_sql_ops(ds, {key1: [op1]})


class TestAnalyzer:
    """Test Analyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test Analyzer initialization."""
        # Default initialization
        analyzer1 = Analyzer()
        assert isinstance(analyzer1._report, AnalysisReport)
        assert isinstance(analyzer1._metadata, Metadata)
        assert analyzer1._symbol_lookup == {}

        # With metadata and symbol lookup
        metadata = Metadata(execution_id="test-123")
        spec: MetricSpec = Sum("revenue")  # Use concrete MetricSpec instead of Mock
        key = ResultKey(datetime.date(2024, 1, 1), {})  # Use concrete ResultKey
        dataset = "test_dataset"
        symbol_lookup: dict[MetricKey, str] = {(spec, key, dataset): "x_1"}
        analyzer2 = Analyzer(metadata=metadata, symbol_lookup=symbol_lookup)
        assert analyzer2._metadata == metadata
        assert analyzer2._symbol_lookup == symbol_lookup

    def test_analyze_empty_metrics(self) -> None:
        """Test analyze with empty metrics raises error."""
        analyzer = Analyzer()
        ds = Mock(spec=SqlDataSource)

        with pytest.raises(DQXError, match="No metrics provided"):
            analyzer.analyze(ds, {})

    def test_analyze_batch_processing(self) -> None:
        """Test that large batches are split according to DEFAULT_BATCH_SIZE."""
        analyzer = Analyzer()
        ds = Mock(spec=SqlDataSource)
        ds.name = "test_ds"

        # Create metrics for more dates than DEFAULT_BATCH_SIZE
        metrics = {}
        for i in range(10):  # More than DEFAULT_BATCH_SIZE (7)
            key = ResultKey(datetime.date(2024, 1, i + 1), {})
            metrics[key] = [Sum("revenue")]

        # Mock _analyze_internal to track calls
        with patch.object(analyzer, "_analyze_internal") as mock_analyze:
            mock_analyze.return_value = AnalysisReport()

            analyzer.analyze(ds, metrics)

            # Verify batching occurred
            assert mock_analyze.call_count == 2  # 10 dates / 7 batch size = 2 batches

            # Verify first batch has 7 items
            first_call_metrics = mock_analyze.call_args_list[0][0][1]
            assert len(first_call_metrics) == 7

            # Verify second batch has 3 items
            second_call_metrics = mock_analyze.call_args_list[1][0][1]
            assert len(second_call_metrics) == 3
