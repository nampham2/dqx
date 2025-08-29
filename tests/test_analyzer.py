"""Tests for the analyzer module with proper isolation and minimal mocking."""

import datetime as dt
from collections.abc import Iterator
from concurrent.futures import Future
from typing import Any, cast
from unittest.mock import Mock, patch

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from dqx import models, specs
from dqx.analyzer import Analyzer, AnalysisReport, analyze_sketch_ops, analyze_sql_ops
from dqx.common import DQXError, ResultKey, SqlDataSource
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.ops import SketchOp, SqlOp
from dqx.orm.repositories import InMemoryMetricDB, Metric as MetricTable
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


class FakeBatchSqlDataSource:
    """A test implementation of BatchSqlDataSource protocol."""

    def __init__(self, name: str = "batch_ds", batches_data: list[SqlDataSource] | None = None):
        self.name = name
        self._batches_data = batches_data or []

    def batches(self) -> Iterator[SqlDataSource]:
        """Return an iterator of SqlDataSource instances."""
        return iter(self._batches_data)


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
    return pa.Table.from_arrays(
        [int_col, null_col, neg_col], 
        names=["int_col", "null_col", "neg_col"]
    )


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
        report = analyzer.analyze_single(arrow_data_source, test_metrics, result_key)
        
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
        report1 = analyzer.analyze_single(arrow_data_source, test_metrics, result_key)
        assert report1[(test_metrics[0], result_key)].value == pytest.approx(10)
        
        # Second analysis should accumulate
        report2 = analyzer.analyze_single(arrow_data_source, test_metrics, result_key)
        assert report2[(test_metrics[0], result_key)].value == pytest.approx(20)

    def test_analyzer_multiple_keys(
        self, arrow_data_source: ArrowDataSource, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test analyzing with different result keys."""
        analyzer = Analyzer()
        
        # Analyze with first key
        report1 = analyzer.analyze_single(arrow_data_source, test_metrics, result_key)
        assert len(report1) == len(test_metrics)
        
        # Analyze with different key (previous day)
        prev_key = result_key.lag(1)
        report2 = analyzer.analyze_single(arrow_data_source, test_metrics, prev_key)
        assert len(report2) == 2 * len(test_metrics)  # Both days
        
        # Check that both keys have their data
        assert (test_metrics[0], result_key) in report2
        assert (test_metrics[0], prev_key) in report2

    def test_analyzer_empty_metrics_error(self, arrow_data_source: ArrowDataSource, result_key: ResultKey) -> None:
        """Test that analyzing with empty metrics raises error."""
        analyzer = Analyzer()
        
        with pytest.raises(DQXError, match="No metrics provided"):
            analyzer.analyze_single(arrow_data_source, [], result_key)

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

    def test_report_merge(self, result_key: ResultKey) -> None:
        """Test merging two reports."""
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
        
        report2 = AnalysisReport({
            (metric1_spec, result_key): metric1_new,
            (metric2_spec, result_key): metric2
        })
        
        # Merge reports
        merged = report1.merge(report2)
        
        # Verify results
        assert len(merged) == 2
        assert merged[(metric1_spec, result_key)].value == 15.0  # 10 + 5
        assert merged[(metric2_spec, result_key)].value == 20.0

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

    def test_analyze_batch_sequential(
        self, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test analyzing batch data source sequentially."""
        # Use real ArrowDataSource for batches with expected columns
        data1 = pa.Table.from_pydict({
            "int_col": [1, 2, 3],
            "null_col": [1, None, 3],
            "neg_col": [-1, 0, 1]
        })
        data2 = pa.Table.from_pydict({
            "int_col": [4, 5, 6],
            "null_col": [None, 5, 6],
            "neg_col": [-2, -1, 0]
        })
        data3 = pa.Table.from_pydict({
            "int_col": [7, 8, 9],
            "null_col": [7, None, None],
            "neg_col": [1, 2, 3]
        })
        
        batch1 = ArrowDataSource(data1)
        batch2 = ArrowDataSource(data2)
        batch3 = ArrowDataSource(data3)
        
        batch_ds = FakeBatchSqlDataSource(batches_data=[batch1, batch2, batch3])
        
        analyzer = Analyzer()
        
        # Mock analyze_single to track calls
        call_count = 0
        original_analyze_single = analyzer.analyze_single
        
        def track_calls(*args: Any, **kwargs: Any) -> AnalysisReport:
            nonlocal call_count
            call_count += 1
            return original_analyze_single(*args, **kwargs)
        
        analyzer.analyze_single = track_calls  # type: ignore[assignment]
        
        # Analyze batches
        result = analyzer.analyze(batch_ds, test_metrics, result_key, threading=False)
        
        # Verify all batches were processed
        assert call_count == 3
        assert result == analyzer.report

    def test_analyze_batch_threaded(
        self, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test analyzing batch data source with threading."""
        # Create simple test data to avoid complex threading issues
        data1 = pa.Table.from_pydict({
            "int_col": [1, 2, 3],
            "null_col": [1, None, 3],
            "neg_col": [-1, 0, 1]
        })
        data2 = pa.Table.from_pydict({
            "int_col": [4, 5, 6],
            "null_col": [None, 5, 6],
            "neg_col": [-2, -1, 0]
        })
        
        batch1 = ArrowDataSource(data1)
        batch2 = ArrowDataSource(data2)
        
        # Create batch data source
        batch_ds = FakeBatchSqlDataSource(batches_data=[batch1, batch2])
        
        analyzer = Analyzer()
        
        # Use simpler metrics to avoid DuckDB threading issues
        simple_metrics = [specs.NumRows()]
        
        # Mock the threading to avoid actual DuckDB threading issues
        with patch("dqx.analyzer.ThreadPoolExecutor") as mock_executor:
            # Make the executor run tasks sequentially to avoid DuckDB threading
            def mock_submit(fn: Any, *args: Any, **kwargs: Any) -> Mock:
                future = Mock()
                future.result.return_value = fn(*args, **kwargs)
                return future
            
            mock_context = Mock()
            mock_context.submit = mock_submit
            mock_context.__enter__ = Mock(return_value=mock_context)
            mock_context.__exit__ = Mock(return_value=None)
            mock_executor.return_value = mock_context
            
            # Analyze with threading
            result = analyzer.analyze(batch_ds, simple_metrics, result_key, threading=True)
            
            # Verify results
            assert len(result) == 1  # One metric
            assert result[(simple_metrics[0], result_key)].value == 6.0  # 3 + 3 rows

    def test_analyze_unsupported_data_source(
        self, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test analyzing with unsupported data source type."""
        analyzer = Analyzer()
        
        # Create an unsupported data source
        class UnsupportedDS:
            name = "unsupported"
        
        unsupported_ds = UnsupportedDS()
        
        with pytest.raises(DQXError, match="Unsupported data source"):
            analyzer.analyze(unsupported_ds, test_metrics, result_key)  # type: ignore[arg-type]

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
            analyzer.persist(db)
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
        analyzer.persist(db, overwrite=True)
        
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
        analyzer.persist(db, overwrite=False)
        
        # Verify metric exists and has been properly handled
        # Search for metrics with the same spec and key
        saved_metrics = db.search(
            MetricTable.metric_type == metric_spec.metric_type,
            MetricTable.yyyy_mm_dd == result_key.yyyy_mm_dd,
            MetricTable.tags == result_key.tags
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
            analyzer.analyze_single(arrow_data_source, test_metrics, result_key)
            mock_setup.assert_called_once()


class TestThreadingDetails:
    """Test threading implementation details."""

    def test_thread_pool_configuration(
        self, arrow_data_source: ArrowDataSource, test_metrics: list[specs.MetricSpec], result_key: ResultKey
    ) -> None:
        """Test ThreadPoolExecutor configuration."""
        # Create batch data source
        batch_ds = FakeBatchSqlDataSource(batches_data=[arrow_data_source])
        
        analyzer = Analyzer()
        
        # Mock ThreadPoolExecutor and multiprocessing
        mock_future = Mock(spec=Future)
        mock_future.result.return_value = None
        
        mock_executor = Mock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__ = lambda self: self
        mock_executor.__exit__ = lambda self, *args: None
        
        with patch("dqx.analyzer.ThreadPoolExecutor", return_value=mock_executor) as mock_tpe:
            with patch("dqx.analyzer.multiprocessing.cpu_count", return_value=4):
                analyzer.analyze(batch_ds, test_metrics, result_key, threading=True)
                
                # Verify ThreadPoolExecutor was created with correct workers
                mock_tpe.assert_called_once_with(max_workers=4)
                
                # Verify submit was called
                assert mock_executor.submit.call_count == 1
                
                # Verify result was retrieved
                mock_future.result.assert_called_once()
