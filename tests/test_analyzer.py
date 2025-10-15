"""Tests for the analyzer module with proper isolation and minimal mocking."""

import datetime as dt
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import Mock, patch

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from dqx import models, specs
from dqx.analyzer import AnalysisReport, Analyzer, analyze_sketch_ops, analyze_sql_ops
from dqx.common import DQXError, ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.ops import SketchOp, SqlOp
from dqx.orm.repositories import InMemoryMetricDB
from dqx.orm.repositories import Metric as MetricTable
from dqx.states import Average, SimpleAdditiveState, SketchState


class FakeRelation:
    """A test implementation of a query result relation."""

    def __init__(self, data: dict[str, np.ndarray]):
        self._data = data

    def fetchnumpy(self) -> dict[str, np.ndarray]:
        """Return the data as numpy arrays."""
        return self._data

    def fetch_arrow_reader(self, batch_size: int) -> Iterator[pa.RecordBatch]:
        """Return an iterator of Arrow record batches for sketch ops."""
        if not self._data:
            return

        # Convert numpy arrays to Arrow arrays and yield as record batches
        total_size = len(next(iter(self._data.values()))) if self._data else 0
        for i in range(0, total_size, batch_size):
            batch_data = {}
            for col, arr in self._data.items():
                batch_data[col] = pa.array(arr[i : i + batch_size])
            yield pa.RecordBatch.from_pydict(batch_data)


class FakeSqlDataSource:
    """A test implementation of SqlDataSource protocol."""

    def __init__(
        self,
        name: str = "test_ds",
        cte: str = "SELECT * FROM test",
        dialect: str = "duckdb",
        data: dict[str, np.ndarray] | None = None,
    ):
        self.name = name
        self.cte = cte
        self.dialect = dialect
        self._data = data or {}

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """Return a mock DuckDB relation."""
        # Create a mock that returns our FakeRelation
        mock_relation = Mock(spec=duckdb.DuckDBPyRelation)
        mock_relation.fetchnumpy.return_value = self._data
        mock_relation.fetch_arrow_reader.return_value = FakeRelation(self._data).fetch_arrow_reader(100000)
        return mock_relation


class FakeSketchState(SketchState):
    """A simple sketch state implementation for testing."""

    def __init__(self) -> None:
        self._values: list[Any] = []

    @property
    def value(self) -> float:
        """Get the sketch value (e.g., count of unique values)."""
        return float(len(set(self._values)))

    @classmethod
    def identity(cls) -> "FakeSketchState":
        """Return identity state."""
        return cls()

    def serialize(self) -> bytes:
        """Serialize the state."""
        return b""

    @classmethod
    def deserialize(cls, state: bytes) -> "FakeSketchState":
        """Deserialize the state."""
        return cls()

    def merge(self, other: "FakeSketchState") -> "FakeSketchState":
        """Merge with another state."""
        new_state = FakeSketchState()
        new_state._values = self._values + other._values
        return new_state

    def __copy__(self) -> "FakeSketchState":
        """Copy the state."""
        new_state = FakeSketchState()
        new_state._values = self._values.copy()
        return new_state

    def __eq__(self, other: Any) -> bool:
        """Check equality."""
        return isinstance(other, FakeSketchState) and self._values == other._values

    def fit(self, batch: pa.Array) -> None:
        """Fit the sketch with data."""
        self._values.extend(batch.to_pylist())


@pytest.fixture
def sample_data() -> pa.Table:
    """Create a sample PyArrow table for testing."""
    int_col = pa.array(range(10))
    null_col = pa.array([0, None, 2, 3, 4, 5, 6, 7, 8, 9])
    neg_col = pa.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    return pa.Table.from_arrays([int_col, null_col, neg_col], names=["int_col", "null_col", "neg_col"])


@pytest.fixture
def arrow_data_source(sample_data: pa.Table) -> ArrowDataSource:
    """Create an ArrowDataSource with sample data."""
    return ArrowDataSource(sample_data)


@pytest.fixture
def result_key() -> ResultKey:
    """Create a result key for testing."""
    return ResultKey(yyyy_mm_dd=dt.date(2025, 2, 4), tags={})


@pytest.fixture
def test_metrics() -> list[specs.MetricSpec]:
    """Create a list of test metrics."""
    return [
        specs.NumRows(),
        specs.Average("int_col"),
        specs.Minimum("int_col"),
        specs.Maximum("int_col"),
        specs.Sum("int_col"),
        specs.NullCount("null_col"),
        specs.NegativeCount("neg_col"),
        specs.ApproxCardinality("int_col"),
    ]


class TestAnalyzer:
    """Test the main Analyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test that analyzer initializes with empty report."""
        analyzer = Analyzer()
        assert len(analyzer.report) == 0
        assert isinstance(analyzer.report, AnalysisReport)

    def test_analyzer_single_analysis(
        self, arrow_data_source: ArrowDataSource, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test analyzing a single data source."""
        analyzer = Analyzer()

        # Analyze the data
        report = analyzer.analyze(arrow_data_source, test_metrics, result_key)

        # Check results
        assert len(report) == len(test_metrics)
        assert report[(test_metrics[0], result_key)].value == pytest.approx(10)  # NumRows
        assert report[(test_metrics[1], result_key)].value == pytest.approx(4.5)  # Average
        assert report[(test_metrics[2], result_key)].value == pytest.approx(0)  # Minimum
        assert report[(test_metrics[3], result_key)].value == pytest.approx(9)  # Maximum

    def test_analyzer_accumulation(
        self, arrow_data_source: ArrowDataSource, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test that multiple analyses accumulate properly."""
        analyzer = Analyzer()

        # First analysis
        report1 = analyzer.analyze(arrow_data_source, test_metrics, result_key)
        assert report1[(test_metrics[0], result_key)].value == pytest.approx(10)

        # Second analysis should accumulate
        report2 = analyzer.analyze(arrow_data_source, test_metrics, result_key)
        assert report2[(test_metrics[0], result_key)].value == pytest.approx(20)

    def test_analyzer_multiple_keys(
        self, arrow_data_source: ArrowDataSource, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test analyzing with different result keys."""
        analyzer = Analyzer()

        # Analyze with first key
        report1 = analyzer.analyze(arrow_data_source, test_metrics, result_key)
        assert len(report1) == len(test_metrics)

        # Analyze with different key (previous day)
        prev_key = result_key.lag(1)
        report2 = analyzer.analyze(arrow_data_source, test_metrics, prev_key)
        assert len(report2) == 2 * len(test_metrics)  # Both days

        # Check that both keys have their data
        assert (test_metrics[0], result_key) in report2
        assert (test_metrics[0], prev_key) in report2

    def test_analyzer_empty_metrics_error(self, arrow_data_source: ArrowDataSource, result_key: ResultKey) -> None:
        """Test that analyzing with empty metrics raises error."""
        analyzer = Analyzer()

        with pytest.raises(DQXError, match="No metrics provided"):
            analyzer.analyze(arrow_data_source, [], result_key)

    def test_analyzer_protocol_implementation(self) -> None:
        """Test that Analyzer implements the protocol."""
        from dqx.common import Analyzer as AnalyzerProtocol

        analyzer = Analyzer()
        assert isinstance(analyzer, AnalyzerProtocol)


class TestAnalysisReport:
    """Test the AnalysisReport class."""

    def test_empty_report(self) -> None:
        """Test creating an empty report."""
        # Explicitly pass an empty dict to avoid mutable default argument issues
        report = AnalysisReport({})
        assert len(report) == 0
        assert hasattr(report, "__getitem__")  # dict-like behavior
        assert hasattr(report, "__setitem__")  # dict-like behavior

    def test_report_with_data(self, result_key: ResultKey) -> None:
        """Test creating a report with initial data."""
        metric_spec = specs.NumRows()
        state = SimpleAdditiveState(10.0)
        metric = models.Metric.build(metric_spec, result_key, state=state)

        report = AnalysisReport({(metric_spec, result_key): metric})
        assert len(report) == 1
        assert report[(metric_spec, result_key)].value == 10.0

    def test_report_merge_basic(self, result_key: ResultKey) -> None:
        """Test basic merging of two reports."""
        # Create first report
        metric1_spec = specs.NumRows()
        state1 = SimpleAdditiveState(10.0)
        metric1 = models.Metric.build(metric1_spec, result_key, state=state1)
        report1 = AnalysisReport({(metric1_spec, result_key): metric1})

        # Create second report with overlapping and new metrics
        metric2_spec = specs.Average("col")
        state2 = Average(avg=20.0, n=5.0)
        metric2 = models.Metric.build(metric2_spec, result_key, state=state2)

        state1_new = SimpleAdditiveState(5.0)
        metric1_new = models.Metric.build(metric1_spec, result_key, state=state1_new)

        report2 = AnalysisReport({(metric1_spec, result_key): metric1_new, (metric2_spec, result_key): metric2})

        # Merge reports
        merged = report1.merge(report2)

        # Verify results
        assert len(merged) == 2
        assert merged[(metric1_spec, result_key)].value == 15.0  # 10 + 5
        assert merged[(metric2_spec, result_key)].value == 20.0

    def test_merge_empty_reports(self) -> None:
        """Test merging two empty AnalysisReports."""
        report1 = AnalysisReport()
        report2 = AnalysisReport()
        merged = report1.merge(report2)
        assert len(merged) == 0

    def test_merge_non_overlapping_reports(self, result_key: ResultKey) -> None:
        """Test merging reports with different metrics."""
        metric1 = models.Metric.build(specs.Average("price"), result_key, SimpleAdditiveState(10.0))
        metric2 = models.Metric.build(specs.Average("quantity"), result_key, SimpleAdditiveState(5.0))

        report1 = AnalysisReport({(metric1.spec, result_key): metric1})
        report2 = AnalysisReport({(metric2.spec, result_key): metric2})

        merged = report1.merge(report2)
        assert len(merged) == 2
        assert (metric1.spec, result_key) in merged
        assert (metric2.spec, result_key) in merged

    def test_merge_overlapping_reports(self, result_key: ResultKey) -> None:
        """Test merging reports with same metric - should use Metric.reduce."""
        spec = specs.Average("price")

        # Create two metrics with same spec but different values
        metric1 = models.Metric.build(spec, result_key, SimpleAdditiveState(10.0))
        metric2 = models.Metric.build(spec, result_key, SimpleAdditiveState(20.0))

        report1 = AnalysisReport({(spec, result_key): metric1})
        report2 = AnalysisReport({(spec, result_key): metric2})

        merged = report1.merge(report2)
        assert len(merged) == 1

        # The merge should have used Metric.reduce
        # which calls state.merge, which for SimpleAdditiveState adds values
        merged_metric = merged[(spec, result_key)]
        assert merged_metric.value == 30.0  # 10.0 + 20.0

    def test_merge_with_different_result_keys(self) -> None:
        """Test merging reports with different ResultKeys."""
        key1 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
        key2 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 2), tags={})
        spec = specs.Average("price")

        metric1 = models.Metric.build(spec, key1, SimpleAdditiveState(10.0))
        metric2 = models.Metric.build(spec, key2, SimpleAdditiveState(20.0))

        report1 = AnalysisReport({(spec, key1): metric1})
        report2 = AnalysisReport({(spec, key2): metric2})

        merged = report1.merge(report2)
        assert len(merged) == 2
        assert merged[(spec, key1)].value == 10.0
        assert merged[(spec, key2)].value == 20.0

    def test_merge_multiple_metrics_simultaneously(self, result_key: ResultKey) -> None:
        """Test merging multiple different metrics at once."""
        # Report 1: price average and sum
        price_avg1 = models.Metric.build(specs.Average("price"), result_key, SimpleAdditiveState(10.0))
        price_sum1 = models.Metric.build(specs.Sum("price"), result_key, SimpleAdditiveState(100.0))
        report1 = AnalysisReport({(price_avg1.spec, result_key): price_avg1, (price_sum1.spec, result_key): price_sum1})

        # Report 2: price average, quantity average, and quantity sum
        price_avg2 = models.Metric.build(specs.Average("price"), result_key, SimpleAdditiveState(5.0))
        qty_avg = models.Metric.build(specs.Average("quantity"), result_key, SimpleAdditiveState(3.0))
        qty_sum = models.Metric.build(specs.Sum("quantity"), result_key, SimpleAdditiveState(30.0))
        report2 = AnalysisReport(
            {
                (price_avg2.spec, result_key): price_avg2,
                (qty_avg.spec, result_key): qty_avg,
                (qty_sum.spec, result_key): qty_sum,
            }
        )

        merged = report1.merge(report2)
        assert len(merged) == 4
        # Price average should be merged: 10.0 + 5.0 = 15.0
        assert merged[(specs.Average("price"), result_key)].value == 15.0
        # Price sum should remain from report1
        assert merged[(specs.Sum("price"), result_key)].value == 100.0
        # Quantity metrics should be from report2
        assert merged[(specs.Average("quantity"), result_key)].value == 3.0
        assert merged[(specs.Sum("quantity"), result_key)].value == 30.0

    def test_merge_empty_with_non_empty(self, result_key: ResultKey) -> None:
        """Test merging empty report with non-empty report."""
        metric = models.Metric.build(specs.Average("price"), result_key, SimpleAdditiveState(10.0))

        empty_report = AnalysisReport()
        non_empty_report = AnalysisReport({(metric.spec, result_key): metric})

        # Test both directions
        merged1 = empty_report.merge(non_empty_report)
        merged2 = non_empty_report.merge(empty_report)

        assert len(merged1) == 1
        assert len(merged2) == 1
        assert merged1[(metric.spec, result_key)].value == 10.0
        assert merged2[(metric.spec, result_key)].value == 10.0

    def test_merge_preserve_identity_behavior(self, result_key: ResultKey) -> None:
        """Test that Metric.reduce behavior with identity matches toolz.merge_with."""
        spec = specs.Average("price")

        # Create metrics
        metric1 = models.Metric.build(spec, result_key, SimpleAdditiveState(10.0))
        metric2 = models.Metric.build(spec, result_key, SimpleAdditiveState(20.0))

        # Test reduce directly
        reduced = models.Metric.reduce([metric1, metric2])
        assert reduced.value == 30.0

        # Verify the identity behavior
        identity = metric1.identity()
        assert identity.value == 0.0  # SimpleAdditiveState identity is 0.0

    def test_report_show(self, result_key: ResultKey, capsys: pytest.CaptureFixture[str]) -> None:
        """Test the show method of AnalysisReport."""
        metric_spec = specs.NumRows()
        state = SimpleAdditiveState(42.0)
        metric = models.Metric.build(metric_spec, result_key, state=state)

        report = AnalysisReport({(metric_spec, result_key): metric})
        report.show()

        # Verify output was produced (Rich console output)
        captured = capsys.readouterr()
        assert "NumRows" in captured.out or len(captured.out) > 0


class TestBatchAnalysis:
    """Test batch data source analysis."""

    def test_analyze_sql_data_source_directly(
        self, arrow_data_source: ArrowDataSource, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test that analyze method works with SqlDataSource."""
        analyzer = Analyzer()

        # Should delegate to analyze_single
        result = analyzer.analyze(arrow_data_source, test_metrics, result_key)
        assert len(result) == len(test_metrics)


class TestPersistence:
    """Test analyzer persistence functionality."""

    def test_persist_empty_report(self) -> None:
        """Test persisting an empty report logs warning."""
        analyzer = Analyzer()
        db = InMemoryMetricDB()

        # Use caplog fixture would be better, but for now use patch
        with patch("dqx.analyzer.logger.warning") as mock_warning:
            analyzer.report.persist(db)
            mock_warning.assert_called_once_with("Try to save an EMPTY analysis report!")

    def test_persist_overwrite(self, result_key: ResultKey) -> None:
        """Test persisting with overwrite mode."""
        analyzer = Analyzer()

        # Add a metric to the report
        metric_spec = specs.NumRows()
        state = SimpleAdditiveState(10.0)
        metric = models.Metric.build(metric_spec, result_key, state=state)
        analyzer._report[(metric_spec, result_key)] = metric

        # Persist to database
        db = InMemoryMetricDB()
        analyzer.report.persist(db, overwrite=True)

        # Verify metric was saved
        saved_metrics = db.search(MetricTable.metric_id != None)  # noqa: E711
        assert len(saved_metrics) == 1

    def test_persist_merge(self, result_key: ResultKey) -> None:
        """Test persisting with merge mode."""
        analyzer = Analyzer()

        # Add a metric to the analyzer report
        metric_spec = specs.NumRows()
        state1 = SimpleAdditiveState(10.0)
        metric1 = models.Metric.build(metric_spec, result_key, state=state1)
        analyzer._report[(metric_spec, result_key)] = metric1

        # Create database with existing metric
        db = InMemoryMetricDB()
        state2 = SimpleAdditiveState(5.0)
        metric2 = models.Metric.build(metric_spec, result_key, state=state2)
        db.persist([metric2])

        # Persist with merge
        analyzer.report.persist(db, overwrite=False)

        # Verify metric exists and has been properly handled
        # Search for metrics with the same spec and key
        saved_metrics = db.search(
            MetricTable.metric_type == metric_spec.metric_type,
            MetricTable.yyyy_mm_dd == result_key.yyyy_mm_dd,
            MetricTable.tags == result_key.tags,
        )
        # The merge mode merges values but both records remain in DB
        assert len(saved_metrics) == 2
        values = sorted([m.value for m in saved_metrics])
        # Original value (5.0) and merged value (5.0 + 10.0 = 15.0)
        assert values == [5.0, 15.0]


class TestAnalyzeFunctions:
    """Test standalone analyze functions."""

    def test_analyze_sketch_ops_empty(self) -> None:
        """Test analyze_sketch_ops with empty ops list."""
        # Use a real ArrowDataSource
        data = pa.Table.from_pydict({"col": [1, 2, 3]})
        ds = ArrowDataSource(data)
        analyze_sketch_ops(ds, [])
        # Should return without errors

    def test_analyze_sketch_ops(self) -> None:
        """Test analyze_sketch_ops with actual ops."""
        # Use ArrowDataSource with test data
        data = pa.Table.from_pydict({"test_col": [1, 2, 3, 2, 1]})
        ds = ArrowDataSource(data)

        # Use real sketch op
        op1 = cast(SketchOp, specs.ApproxCardinality("test_col").analyzers[0])
        op2 = cast(SketchOp, specs.ApproxCardinality("test_col").analyzers[0])

        # Analyze
        analyze_sketch_ops(ds, [op1, op2])

        # Verify both ops got values assigned
        assert op1.value().value > 0  # Should have detected unique values
        assert op2.value().value > 0

    def test_analyze_sql_ops_empty(self) -> None:
        """Test analyze_sql_ops with empty ops list."""
        data = pa.Table.from_pydict({"col": [1, 2, 3]})
        ds = ArrowDataSource(data)
        analyze_sql_ops(ds, [])
        # Should return without errors

    def test_analyze_sql_ops_missing_dialect(self) -> None:
        """Test analyze_sql_ops with missing dialect."""
        # Create data source without dialect
        mock_ds = Mock()
        mock_ds.name = "no_dialect"
        mock_ds.cte = "SELECT 1"

        # Use real SQL op
        op = cast(SqlOp, specs.NumRows().analyzers[0])

        # The error is about dialect not found in registry, not about missing dialect
        with pytest.raises(DQXError, match="not found in registry"):
            analyze_sql_ops(mock_ds, [op])

    def test_analyze_sql_ops(self) -> None:
        """Test analyze_sql_ops with actual ops."""
        # Use ArrowDataSource
        data = pa.Table.from_pydict({"test_col": [42.0]})
        ds = ArrowDataSource(data)

        # Use real SQL ops
        op1 = cast(SqlOp, specs.NumRows().analyzers[0])
        op2 = cast(SqlOp, specs.Average("test_col").analyzers[1])  # Note: Average has NumRows at [0], Average at [1]

        # Analyze
        analyze_sql_ops(ds, [op1, op2])

        # Verify ops got values
        assert op1.value() == 1.0  # 1 row
        assert op2.value() == 42.0  # Average of [42.0]

    def test_sql_ops_deduplication(self) -> None:
        """Test that duplicate SQL ops are only executed once."""
        # Create test data
        data = pa.Table.from_pydict({"value": [1, 2, 3, 4, 5]})
        ds = ArrowDataSource(data)

        # Create duplicate ops
        sum1 = cast(SqlOp, specs.Sum("value").analyzers[0])
        sum2 = cast(SqlOp, specs.Sum("value").analyzers[0])
        avg1 = cast(SqlOp, specs.Average("value").analyzers[1])  # Average has NumRows at [0], Average at [1]
        avg2 = cast(SqlOp, specs.Average("value").analyzers[1])

        ops = [sum1, sum2, avg1, avg2, sum1]  # sum1 appears 3 times total

        # Analyze
        analyze_sql_ops(ds, ops)

        # All duplicate ops should have the same value assigned
        assert sum1.value() == 15.0  # 1+2+3+4+5
        assert sum2.value() == 15.0  # Same instance should have same value
        assert avg1.value() == 3.0  # (1+2+3+4+5)/5
        assert avg2.value() == 3.0  # Same instance should have same value

    def test_sql_ops_order_preservation(self) -> None:
        """Test that deduplication preserves order of first occurrence."""
        data = pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds = ArrowDataSource(data)

        # Create ops in specific order
        max_a = cast(SqlOp, specs.Maximum("a").analyzers[0])
        sum_b = cast(SqlOp, specs.Sum("b").analyzers[0])
        avg_a = cast(SqlOp, specs.Average("a").analyzers[1])  # Average has NumRows at [0], Average at [1]
        min_b = cast(SqlOp, specs.Minimum("b").analyzers[0])

        # Add with duplicates in different positions
        ops = [max_a, sum_b, avg_a, max_a, min_b, sum_b, avg_a]

        # Analyze the ops
        analyze_sql_ops(ds, ops)

        # Verify all ops got values assigned
        assert max_a.value() == 3.0
        assert sum_b.value() == 15.0
        assert avg_a.value() == 2.0
        assert min_b.value() == 4.0

    def test_mixed_column_deduplication(self) -> None:
        """Test deduplication with ops on different columns."""
        data = pa.Table.from_pydict({"price": [10.0, 20.0, 30.0], "quantity": [1, 2, 3], "tax": [1.0, 2.0, 3.0]})
        ds = ArrowDataSource(data)

        # Create ops on different columns with some duplicates
        sum_price1 = cast(SqlOp, specs.Sum("price").analyzers[0])
        avg_qty1 = cast(SqlOp, specs.Average("quantity").analyzers[1])  # Average has NumRows at [0], Average at [1]
        sum_price2 = cast(SqlOp, specs.Sum("price").analyzers[0])  # Duplicate
        max_tax = cast(SqlOp, specs.Maximum("tax").analyzers[0])
        avg_qty2 = cast(SqlOp, specs.Average("quantity").analyzers[1])  # Duplicate
        min_price = cast(SqlOp, specs.Minimum("price").analyzers[0])

        ops = [sum_price1, avg_qty1, sum_price2, max_tax, avg_qty2, min_price]

        analyze_sql_ops(ds, ops)

        # Verify all ops got values
        assert sum_price1.value() == 60.0  # Sum of price
        assert sum_price2.value() == 60.0  # Duplicate should have same value
        assert avg_qty1.value() == 2.0  # Average of quantity
        assert avg_qty2.value() == 2.0  # Duplicate should have same value
        assert max_tax.value() == 3.0  # Max tax
        assert min_price.value() == 10.0  # Min price


class TestDuckDBSetup:
    """Test DuckDB setup functionality."""

    def test_setup_duckdb(self) -> None:
        """Test that _setup_duckdb calls duckdb.execute correctly."""
        analyzer = Analyzer()

        with patch("duckdb.execute") as mock_execute:
            analyzer._setup_duckdb()
            mock_execute.assert_called_once_with("SET enable_progress_bar = false")

    def test_setup_called_during_analyze(
        self, arrow_data_source: ArrowDataSource, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test that _setup_duckdb is called during analysis."""
        analyzer = Analyzer()

        with patch.object(analyzer, "_setup_duckdb") as mock_setup:
            analyzer.analyze(arrow_data_source, test_metrics, result_key)
            mock_setup.assert_called_once()
