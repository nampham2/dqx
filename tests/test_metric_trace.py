"""Tests for metric_trace functionality."""

from datetime import date
from typing import Any

import pyarrow as pa
from returns.result import Failure, Success

from dqx import specs
from dqx.analyzer import AnalysisReport
from dqx.common import Metadata, ResultKey
from dqx.data import metric_trace
from dqx.display import print_metric_trace
from dqx.models import Metric
from dqx.provider import SymbolInfo
from dqx.states import SimpleAdditiveState


class TestMetricTrace:
    """Test cases for metric_trace function."""

    def test_empty_inputs(self) -> None:
        """Test metric_trace with all empty inputs."""
        result = metric_trace([], "exec-123", {}, [])

        # Should return empty table with correct schema
        assert result.num_rows == 0
        assert set(result.column_names) == {
            "date",
            "metric",
            "symbol",
            "type",
            "dataset",
            "value_db",
            "value_analysis",
            "value_final",
            "error",
            "tags",
            "is_extended",
        }

    def test_metrics_only(self) -> None:
        """Test metric_trace with only metrics from DB."""
        test_date = date(2024, 1, 1)
        metrics = [
            Metric.build(
                specs.NumRows(),
                ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod"}),
                dataset="sales_table",
                state=SimpleAdditiveState(100.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
            Metric.build(
                specs.NullCount("some_column"),
                ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod"}),
                dataset="sales_table",
                state=SimpleAdditiveState(5.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
        ]

        result = metric_trace(metrics, "exec-123", {}, [])

        assert result.num_rows == 2
        data = result.to_pydict()

        # Check rows (they're sorted by metric name)
        # Find the num_rows metric
        num_rows_idx = data["metric"].index("num_rows()")
        assert data["date"][num_rows_idx] == test_date
        assert data["symbol"][num_rows_idx] is None
        assert data["type"][num_rows_idx] == "NumRows"
        assert data["dataset"][num_rows_idx] == "sales_table"
        assert data["value_db"][num_rows_idx] == 100.0
        assert data["value_analysis"][num_rows_idx] is None
        assert data["value_final"][num_rows_idx] is None
        assert data["error"][num_rows_idx] is None
        assert data["tags"][num_rows_idx] == "env=prod"

        # Check null_count metric
        null_count_idx = data["metric"].index("null_count(some_column)")
        assert data["metric"][null_count_idx] == "null_count(some_column)"

    def test_analysis_reports_only(self) -> None:
        """Test metric_trace with only analysis reports."""
        test_date = date(2024, 1, 1)
        spec = specs.NumRows()
        key = ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod"})

        report = AnalysisReport()
        report[(spec, key)] = Metric.build(
            spec,
            key,
            dataset="sales_table",
            state=SimpleAdditiveState(100.0),
        )
        report.symbol_mapping[(spec, key)] = "x_1"

        reports = {"sales_table": report}
        result = metric_trace([], "exec-123", reports, [])

        assert result.num_rows == 1
        data = result.to_pydict()

        assert data["metric"][0] == "num_rows()"
        assert data["symbol"][0] == "x_1"
        assert data["value_db"][0] is None
        assert data["value_analysis"][0] == 100.0
        assert data["value_final"][0] is None

    def test_symbols_only(self) -> None:
        """Test metric_trace with only symbols."""
        test_date = date(2024, 1, 1)
        symbols = [
            SymbolInfo(
                name="x_1",
                metric="num_rows()",
                dataset="sales_table",
                value=Success(100.0),
                yyyy_mm_dd=test_date,
                tags={"env": "prod"},
            ),
            SymbolInfo(
                name="x_2",
                metric="null_count(some_column)",
                dataset="sales_table",
                value=Failure("Division by zero"),
                yyyy_mm_dd=test_date,
                tags={"env": "prod"},
            ),
        ]

        result = metric_trace([], "exec-123", {}, symbols)

        assert result.num_rows == 2
        data = result.to_pydict()

        # Check successful symbol
        assert data["symbol"][0] == "x_1"
        assert data["metric"][0] == "num_rows()"
        assert data["value_final"][0] == 100.0
        assert data["error"][0] is None

        # Check failed symbol
        assert data["symbol"][1] == "x_2"
        assert data["value_final"][1] is None
        assert data["error"][1] == "Division by zero"

    def test_full_join_metrics_and_reports(self) -> None:
        """Test FULL OUTER JOIN between metrics and analysis reports."""
        test_date = date(2024, 1, 1)

        # Metrics from DB
        metrics = [
            Metric.build(
                specs.NumRows(),
                ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod"}),
                dataset="sales_table",
                state=SimpleAdditiveState(100.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
            Metric.build(
                specs.Average("user_id"),
                ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod"}),
                dataset="sales_table",
                state=SimpleAdditiveState(95.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
        ]

        # Analysis reports (with one overlapping metric)
        spec1 = specs.NumRows()
        spec2 = specs.NullCount("some_column")
        key = ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod"})

        report = AnalysisReport()
        report[(spec1, key)] = Metric.build(spec1, key, dataset="sales_table", state=SimpleAdditiveState(100.0))
        report[(spec2, key)] = Metric.build(spec2, key, dataset="sales_table", state=SimpleAdditiveState(5.0))
        report.symbol_mapping[(spec1, key)] = "x_1"
        report.symbol_mapping[(spec2, key)] = "x_2"

        reports = {"sales_table": report}
        result = metric_trace(metrics, "exec-123", reports, [])

        assert result.num_rows == 3
        data = result.to_pydict()

        # Sort by metric name for consistent testing
        # Find each metric by name
        metrics_list = data["metric"]

        # Check Average (only in DB)
        idx = metrics_list.index("average(user_id)")
        assert data["symbol"][idx] is None
        assert data["value_db"][idx] == 95.0
        assert data["value_analysis"][idx] is None

        # Check NullCount (only in analysis)
        idx = metrics_list.index("null_count(some_column)")
        assert data["symbol"][idx] == "x_2"
        assert data["value_db"][idx] is None
        assert data["value_analysis"][idx] == 5.0

        # Check NumRows (in both)
        idx = metrics_list.index("num_rows()")
        assert data["symbol"][idx] == "x_1"
        assert data["value_db"][idx] == 100.0
        assert data["value_analysis"][idx] == 100.0

    def test_left_join_with_symbols(self) -> None:
        """Test LEFT JOIN from symbols perspective."""
        test_date = date(2024, 1, 1)

        # Metrics from DB
        metrics = [
            Metric.build(
                specs.NumRows(),
                ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod"}),
                dataset="sales_table",
                state=SimpleAdditiveState(100.0),
                metadata=Metadata(execution_id="exec-123"),
            )
        ]

        # Analysis reports
        spec = specs.NumRows()
        key = ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod"})
        report = AnalysisReport()
        report[(spec, key)] = Metric.build(spec, key, dataset="sales_table", state=SimpleAdditiveState(101.0))
        report.symbol_mapping[(spec, key)] = "x_1"
        reports = {"sales_table": report}

        # Symbols (including one that doesn't match)
        symbols = [
            SymbolInfo(
                name="x_1",
                metric="num_rows()",
                dataset="sales_table",
                value=Success(102.0),
                yyyy_mm_dd=test_date,
                tags={"env": "prod"},
            ),
            SymbolInfo(
                name="x_2",
                metric="average(value)",
                dataset="sales_table",
                value=Success(50.0),
                yyyy_mm_dd=test_date,
                tags={"env": "prod"},
            ),
        ]

        result = metric_trace(metrics, "exec-123", reports, symbols)

        assert result.num_rows == 2
        data = result.to_pydict()

        # Find rows by symbol
        x1_idx = data["symbol"].index("x_1")
        x2_idx = data["symbol"].index("x_2")

        # Check x_1 (has all three values - discrepancy!)
        assert data["metric"][x1_idx] == "num_rows()"
        assert data["value_db"][x1_idx] == 100.0
        assert data["value_analysis"][x1_idx] == 101.0
        assert data["value_final"][x1_idx] == 102.0

        # Check x_2 (only has final value)
        assert data["metric"][x2_idx] == "average(value)"
        assert data["value_db"][x2_idx] is None
        assert data["value_analysis"][x2_idx] is None
        assert data["value_final"][x2_idx] == 50.0

    def test_value_discrepancy_detection(self) -> None:
        """Test that value discrepancies are properly shown in the trace."""
        test_date = date(2024, 1, 1)

        # Create data with discrepancies
        metrics = [
            Metric.build(
                specs.NumRows(),
                ResultKey(yyyy_mm_dd=test_date, tags={}),
                dataset="sales_table",
                state=SimpleAdditiveState(100.0),
                metadata=Metadata(execution_id="exec-123"),
            )
        ]

        spec = specs.NumRows()
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        report = AnalysisReport()
        report[(spec, key)] = Metric.build(spec, key, dataset="sales_table", state=SimpleAdditiveState(101.0))
        report.symbol_mapping[(spec, key)] = "x_1"
        reports = {"sales_table": report}

        symbols = [
            SymbolInfo(
                name="x_1",
                metric="num_rows()",
                dataset="sales_table",
                value=Success(102.0),
                yyyy_mm_dd=test_date,
                tags={},
            )
        ]

        result = metric_trace(metrics, "exec-123", reports, symbols)

        data = result.to_pydict()
        assert data["value_db"][0] == 100.0
        assert data["value_analysis"][0] == 101.0
        assert data["value_final"][0] == 102.0

    def test_different_dates_and_datasets(self) -> None:
        """Test that joins work correctly with different dates and datasets."""
        date1 = date(2024, 1, 1)
        date2 = date(2024, 1, 2)

        # Metrics with different dates and datasets
        metrics = [
            Metric.build(
                specs.NumRows(),
                ResultKey(yyyy_mm_dd=date1, tags={}),
                dataset="sales_table",
                state=SimpleAdditiveState(100.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
            Metric.build(
                specs.NumRows(),
                ResultKey(yyyy_mm_dd=date2, tags={}),
                dataset="sales_table",
                state=SimpleAdditiveState(110.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
            Metric.build(
                specs.NumRows(),
                ResultKey(yyyy_mm_dd=date1, tags={}),
                dataset="orders_table",
                state=SimpleAdditiveState(200.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
        ]

        # Analysis reports for same metrics
        reports = {}
        for m in metrics:
            ds = m.dataset
            if ds not in reports:
                reports[ds] = AnalysisReport()
            reports[ds][(m.spec, m.key)] = m

        result = metric_trace(metrics, "exec-123", reports, [])

        assert result.num_rows == 3
        data = result.to_pydict()

        # All should have matching values
        for i in range(3):
            assert data["value_db"][i] == data["value_analysis"][i]

    def test_print_metric_trace(self, capsys: Any) -> None:
        """Test the print_metric_trace display function."""
        test_date = date(2024, 1, 1)

        # Create a simple trace table
        trace = pa.table(
            {
                "date": [test_date, test_date],
                "metric": ["num_rows()", "null_count(some_column)"],
                "symbol": ["x_1", "x_2"],
                "type": ["base", "base"],
                "dataset": ["sales_table", "sales_table"],
                "value_db": [100.0, 5.0],
                "value_analysis": [101.0, 5.0],
                "value_final": [102.0, None],
                "error": [None, "Failed check"],
                "tags": ["env=prod", "-"],
                "is_extended": [False, False],
            }
        )

        # Print the trace
        print_metric_trace(trace, "exec-123")

        # Check output contains expected elements
        captured = capsys.readouterr()
        assert "Metric Trace for Execution: exec-123" in captured.out
        assert "num_rows" in captured.out.lower()
        assert "x_1" in captured.out
        assert "Found 1 row(s) with value discrepancies" in captured.out
        # Type column is removed, so it shouldn't appear in output
        assert "Type" not in captured.out

    def test_print_metric_trace_symbol_sorting(self, capsys: Any) -> None:
        """Test that print_metric_trace sorts by symbol indices correctly."""
        test_date = date(2024, 1, 1)

        # Create a trace table with symbols out of order
        trace = pa.table(
            {
                "date": [test_date] * 6,
                "metric": ["metric_a", "metric_b", "metric_c", "metric_d", "metric_e", "metric_f"],
                "symbol": ["x_10", "x_2", "x_1", None, "x_20", "-"],
                "type": ["base"] * 6,
                "dataset": ["test"] * 6,
                "value_db": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "value_analysis": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "value_final": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "error": [None] * 6,
                "tags": ["-"] * 6,
                "is_extended": [False] * 6,
            }
        )

        # Print the trace
        print_metric_trace(trace, "exec-123")

        # Check that symbols appear in correct order
        captured = capsys.readouterr()
        lines = captured.out.split("\n")

        # Find lines with x_ symbols
        symbol_lines = []
        for line in lines:
            if "x_" in line and "metric_" in line:
                # Extract the symbol from the line
                parts = line.split("│")
                if len(parts) > 3:  # Make sure we have enough columns
                    symbol = parts[3].strip()  # Symbol is in 3rd column (0-indexed)
                    if symbol.startswith("x_"):
                        symbol_lines.append(symbol)

        # Verify order: x_1, x_2, x_10, x_20
        assert symbol_lines == ["x_1", "x_2", "x_10", "x_20"]

    def test_column_name_consistency(self) -> None:
        """Test that all PyArrow functions use lowercase column names."""
        test_date = date(2024, 1, 1)

        # Create test data
        metrics = [
            Metric.build(
                specs.Average("test_column"),
                ResultKey(yyyy_mm_dd=test_date, tags={}),
                dataset="test_dataset",
                state=SimpleAdditiveState(100.0),
                metadata=Metadata(execution_id="exec-123"),
            )
        ]

        spec = specs.Average("test_column")
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        report = AnalysisReport()
        report[(spec, key)] = Metric.build(spec, key, dataset="test_dataset", state=SimpleAdditiveState(100.0))
        reports = {"test_dataset": report}

        symbols = [
            SymbolInfo(
                name="x_1",
                metric="average(test_column)",
                dataset="test_dataset",
                value=Success(100.0),
                yyyy_mm_dd=test_date,
                tags={},
            )
        ]

        # Test individual table functions
        from dqx.data import (
            analysis_reports_to_pyarrow_table,
            metrics_to_pyarrow_table,
            symbols_to_pyarrow_table,
        )

        metrics_table = metrics_to_pyarrow_table(metrics, "exec-123")
        assert all(col.islower() for col in metrics_table.column_names)

        reports_table = analysis_reports_to_pyarrow_table(reports)
        assert all(col.islower() for col in reports_table.column_names)

        symbols_table = symbols_to_pyarrow_table(symbols)
        assert all(col.islower() for col in symbols_table.column_names)

        # Test trace table
        trace = metric_trace(metrics, "exec-123", reports, symbols)
        assert all(col.islower() for col in trace.column_names)
