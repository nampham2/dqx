"""Test coverage for missing lines in analyzer.py."""

import datetime
from typing import Any, Sequence

import duckdb
import pyarrow as pa

from dqx import specs
from dqx.analyzer import Analyzer, analyze_batch_sql_ops
from dqx.common import DQXError, ResultKey
from dqx.ops import SqlOp


class MockDataSource:
    """Mock data source for testing."""

    name: str = "duckdb"
    dialect: str = "duckdb"

    def __init__(self, table: pa.Table) -> None:
        self._relation = duckdb.arrow(table)
        self._table_name = "_test_table"

    def cte(self, nominal_date: datetime.date) -> str:
        """Get CTE for this data source."""
        date_str = nominal_date.strftime("%Y-%m-%d")
        return f"SELECT * FROM {self._table_name} WHERE date_col = '{date_str}'"

    def query(self, query: str) -> Any:
        """Execute a query."""
        return self._relation.query(self._table_name, query)


class TestAnalyzerCoverageBatch:
    """Test missing coverage lines in analyzer.py."""

    def test_analyze_batch_sql_ops_empty_ops(self) -> None:
        """Test analyze_batch_sql_ops with empty ops_by_key (covers line 179)."""
        # Create a dummy data source
        table = pa.table({"date_col": ["2024-01-01"], "value": [100]})
        ds = MockDataSource(table)

        # Call analyze_batch_sql_ops with empty dict - should return immediately
        analyze_batch_sql_ops(ds, {})  # This covers line 179: return

        # No exception should be raised

    def test_analyze_batch_internal_empty_metrics_for_key(self) -> None:
        """Test _analyze_batch_internal with empty metrics for a key (covers line 391)."""
        # Create test data for both dates to avoid masked value errors
        table = pa.table({"date_col": ["2024-01-01", "2024-01-02"], "value": [100, 200]})
        ds = MockDataSource(table)
        analyzer = Analyzer()

        # Create metrics_by_key with one key having empty metrics
        key1 = ResultKey(datetime.date(2024, 1, 1), {})
        key2 = ResultKey(datetime.date(2024, 1, 2), {})

        metrics_by_key: dict[ResultKey, Sequence[specs.MetricSpec]] = {
            key1: [],  # Empty metrics list - will trigger warning
            key2: [specs.Sum("value")],
        }

        # Call _analyze_batch_internal - this will log a warning for key1
        # We can see from the test output that the warning is logged, which means line 391 is covered
        report = analyzer._analyze_batch_internal(ds, metrics_by_key)

        # Report should still be created but only for key2
        assert len(report) == 1
        assert (specs.Sum("value"), key2) in report

    def test_analyze_batch_internal_sql_op_without_value(self) -> None:
        """Test SqlOp propagation when representative has no value (covers lines 424-426)."""
        # Create test data
        table = pa.table({"date_col": ["2024-01-01"], "value": [100]})
        ds = MockDataSource(table)

        # Create a custom SqlOp that doesn't get a value assigned
        class NoValueSqlOp(SqlOp):
            """SqlOp that simulates not having a value assigned."""

            def __init__(self) -> None:
                super().__init__()
                self._has_value = False

            def value(self) -> float:
                """Override to always raise DQXError."""
                raise DQXError("Op has not been collected yet!")

            def assign(self, value: float) -> None:
                """Override to not actually assign value."""
                # Don't assign, so value() will continue to raise
                pass

            def clear(self) -> None:
                """Clear the value."""
                pass

            @property
            def name(self) -> str:
                """Name of the op."""
                return "no_value_op"

            @property
            def prefix(self) -> str:
                """Prefix for the op."""
                return "_test"

            @property
            def sql_col(self) -> str:
                return "test_col"

            def __eq__(self, other: Any) -> bool:
                return isinstance(other, NoValueSqlOp)

            def __hash__(self) -> int:
                return hash("NoValueSqlOp")

        # Create a metric that uses this op
        class TestMetric(specs.MetricSpec):
            metric_type: specs.MetricType = "NumRows"  # Use valid type

            def __init__(self) -> None:
                self._no_value_op = NoValueSqlOp()
                self._analyzers = [self._no_value_op]

            @property
            def name(self) -> str:
                return "test_metric"

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
                return hash(self.name)

            def __eq__(self, other: Any) -> bool:
                return isinstance(other, TestMetric)

            def __str__(self) -> str:
                return self.name

        # Mock analyze_batch_sql_ops to not actually execute SQL
        def mock_analyze_batch_sql_ops(ds: Any, ops_by_key: Any) -> None:
            # Don't assign any values - this simulates SQL execution that doesn't
            # return values for some ops
            pass

        # Patch the function temporarily
        import dqx.analyzer

        original_func = dqx.analyzer.analyze_batch_sql_ops
        dqx.analyzer.analyze_batch_sql_ops = mock_analyze_batch_sql_ops

        try:
            analyzer = Analyzer()
            key = ResultKey(datetime.date(2024, 1, 1), {})
            test_metric: Any = TestMetric()

            # This should handle the DQXError gracefully (covers lines 424-426)
            report = analyzer._analyze_batch_internal(ds, {key: [test_metric]})

            # Report should still be created even though op has no value
            assert len(report) == 1
            assert (test_metric, key) in report

        finally:
            # Restore original function
            dqx.analyzer.analyze_batch_sql_ops = original_func
