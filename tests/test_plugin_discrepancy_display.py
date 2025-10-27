"""Test the data discrepancy functionality in plugins."""

from datetime import date

import pyarrow as pa

from dqx.common import ResultKey
from dqx.data import MetricTraceStats
from dqx.plugins import PluginExecutionContext


def create_trace_table_with_discrepancies() -> pa.Table:
    """Create a trace table with some discrepancies for testing."""
    return pa.table(
        {
            "date": [date(2025, 1, 1), date(2025, 1, 1), date(2025, 1, 1)],
            "metric": ["metric1", "metric2", "metric3"],
            "symbol": ["x_1", "x_2", "x_3"],
            "type": ["sum", "avg", "count"],
            "dataset": ["ds1", "ds1", "ds1"],
            "value_db": [100.0, 200.0, 300.0],
            "value_analysis": [100.0, 210.0, 300.0],  # metric2 has discrepancy
            "value_final": [100.0, 210.0, 310.0],  # metric3 also has discrepancy
            "error": [None, None, None],
            "tags": ["-", "-", "-"],
            "is_extended": [False, False, False],
        }
    )


def create_trace_table_no_discrepancies() -> pa.Table:
    """Create a trace table with no discrepancies for testing."""
    return pa.table(
        {
            "date": [date(2025, 1, 1), date(2025, 1, 1)],
            "metric": ["metric1", "metric2"],
            "symbol": ["x_1", "x_2"],
            "type": ["sum", "avg"],
            "dataset": ["ds1", "ds1"],
            "value_db": [100.0, 200.0],
            "value_analysis": [100.0, 200.0],
            "value_final": [100.0, 200.0],
            "error": [None, None],
            "tags": ["-", "-"],
            "is_extended": [False, False],
        }
    )


def create_trace_table_with_extended_metrics() -> pa.Table:
    """Create a trace table with extended metrics (should not count as discrepancies)."""
    return pa.table(
        {
            "date": [date(2025, 1, 1), date(2025, 1, 1)],
            "metric": ["metric1", "metric1_dod"],
            "symbol": ["x_1", "x_2"],
            "type": ["sum", "dod"],
            "dataset": ["ds1", "ds1"],
            "value_db": [100.0, None],
            "value_analysis": [100.0, None],
            "value_final": [100.0, 50.0],
            "error": [None, None],
            "tags": ["-", "-"],
            "is_extended": [False, True],  # Second metric is extended
        }
    )


class TestDataDiscrepancyStats:
    """Test the data_discrepancy_stats method of PluginExecutionContext."""

    def test_no_discrepancies(self) -> None:
        """Test stats when there are no discrepancies."""
        context = PluginExecutionContext(
            suite_name="Test Suite",
            datasources=["ds1"],
            key=ResultKey(yyyy_mm_dd=date(2025, 1, 1), tags={}),
            timestamp=1234567890.0,
            duration_ms=100.0,
            results=[],
            symbols=[],
            trace=create_trace_table_no_discrepancies(),
        )

        stats = context.data_discrepancy_stats()
        assert stats is not None
        assert isinstance(stats, MetricTraceStats)
        assert stats.total_rows == 2
        assert stats.discrepancy_count == 0
        assert stats.discrepancy_rows == []
        assert stats.discrepancy_details == []

    def test_with_discrepancies(self) -> None:
        """Test stats when there are discrepancies."""
        context = PluginExecutionContext(
            suite_name="Test Suite",
            datasources=["ds1"],
            key=ResultKey(yyyy_mm_dd=date(2025, 1, 1), tags={}),
            timestamp=1234567890.0,
            duration_ms=100.0,
            results=[],
            symbols=[],
            trace=create_trace_table_with_discrepancies(),
        )

        stats = context.data_discrepancy_stats()
        assert stats is not None
        assert isinstance(stats, MetricTraceStats)
        assert stats.total_rows == 3
        assert stats.discrepancy_count == 2
        assert stats.discrepancy_rows == [1, 2]

        # Check discrepancy details
        assert len(stats.discrepancy_details) == 2

        # First discrepancy (metric2)
        detail1 = stats.discrepancy_details[0]
        assert detail1["row_index"] == 1
        assert detail1["metric"] == "metric2"
        assert detail1["value_db"] == 200.0
        assert detail1["value_analysis"] == 210.0
        assert "value_db != value_analysis" in detail1["discrepancies"]

        # Second discrepancy (metric3)
        detail2 = stats.discrepancy_details[1]
        assert detail2["row_index"] == 2
        assert detail2["metric"] == "metric3"
        assert detail2["value_analysis"] == 300.0
        assert detail2["value_final"] == 310.0
        assert "value_analysis != value_final" in detail2["discrepancies"]

    def test_extended_metrics_not_counted(self) -> None:
        """Test that extended metrics are not counted as discrepancies."""
        context = PluginExecutionContext(
            suite_name="Test Suite",
            datasources=["ds1"],
            key=ResultKey(yyyy_mm_dd=date(2025, 1, 1), tags={}),
            timestamp=1234567890.0,
            duration_ms=100.0,
            results=[],
            symbols=[],
            trace=create_trace_table_with_extended_metrics(),
        )

        stats = context.data_discrepancy_stats()
        assert stats is not None
        assert stats.total_rows == 2
        assert stats.discrepancy_count == 0  # Extended metric differences don't count
        assert stats.discrepancy_rows == []
        assert stats.discrepancy_details == []

    def test_empty_trace_table(self) -> None:
        """Test stats with empty trace table."""
        empty_trace = pa.table(
            {
                "date": pa.array([], type=pa.date32()),
                "metric": pa.array([], type=pa.string()),
                "symbol": pa.array([], type=pa.string()),
                "type": pa.array([], type=pa.string()),
                "dataset": pa.array([], type=pa.string()),
                "value_db": pa.array([], type=pa.float64()),
                "value_analysis": pa.array([], type=pa.float64()),
                "value_final": pa.array([], type=pa.float64()),
                "error": pa.array([], type=pa.string()),
                "tags": pa.array([], type=pa.string()),
                "is_extended": pa.array([], type=pa.bool_()),
            }
        )

        context = PluginExecutionContext(
            suite_name="Test Suite",
            datasources=["ds1"],
            key=ResultKey(yyyy_mm_dd=date(2025, 1, 1), tags={}),
            timestamp=1234567890.0,
            duration_ms=100.0,
            results=[],
            symbols=[],
            trace=empty_trace,
        )

        stats = context.data_discrepancy_stats()
        assert stats is None  # Should return None for empty trace

    def test_none_trace_table(self) -> None:
        """Test stats with None trace table."""
        context = PluginExecutionContext(
            suite_name="Test Suite",
            datasources=["ds1"],
            key=ResultKey(yyyy_mm_dd=date(2025, 1, 1), tags={}),
            timestamp=1234567890.0,
            duration_ms=100.0,
            results=[],
            symbols=[],
            trace=None,  # type: ignore[arg-type]
        )

        stats = context.data_discrepancy_stats()
        assert stats is None  # Should return None for None trace
