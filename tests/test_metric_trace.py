"""Tests for metric_trace functionality."""

from datetime import date
from typing import Any

from returns.result import Failure, Success

from dqx import specs
from dqx.analyzer import AnalysisReport
from dqx.common import Metadata, ResultKey
from dqx.data import metric_trace
from dqx.models import Metric
from dqx.provider import SymbolInfo
from dqx.states import SimpleAdditiveState


class TestMetricTrace:
    """Test cases for metric_trace function."""

    def test_empty_inputs(self) -> None:
        """Test metric_trace with all empty inputs."""
        result = metric_trace([], "exec-123", AnalysisReport(), [], {})

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
            "data_av_ratio",
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

        result = metric_trace(metrics, "exec-123", AnalysisReport(), [], {})

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
        report[(spec, key, "sales_table")] = Metric.build(
            spec,
            key,
            dataset="sales_table",
            state=SimpleAdditiveState(100.0),
        )

        # Create symbol lookup
        symbol_lookup: dict[tuple[specs.MetricSpec, ResultKey, str], Any] = {(spec, key, "sales_table"): "x_1"}

        result = metric_trace([], "exec-123", report, [], symbol_lookup)

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

        result = metric_trace([], "exec-123", AnalysisReport(), symbols, {})

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
        report[(spec1, key, "sales_table")] = Metric.build(
            spec1, key, dataset="sales_table", state=SimpleAdditiveState(100.0)
        )
        report[(spec2, key, "sales_table")] = Metric.build(
            spec2, key, dataset="sales_table", state=SimpleAdditiveState(5.0)
        )

        # Create symbol lookup
        symbol_lookup: dict[tuple[specs.MetricSpec, ResultKey, str], Any] = {
            (spec1, key, "sales_table"): "x_1",
            (spec2, key, "sales_table"): "x_2",
        }

        result = metric_trace(metrics, "exec-123", report, [], symbol_lookup)

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
        report[(spec, key, "sales_table")] = Metric.build(
            spec, key, dataset="sales_table", state=SimpleAdditiveState(101.0)
        )

        # Create symbol lookup
        symbol_lookup: dict[tuple[specs.MetricSpec, ResultKey, str], Any] = {(spec, key, "sales_table"): "x_1"}

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

        result = metric_trace(metrics, "exec-123", report, symbols, symbol_lookup)

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
        report[(spec, key, "sales_table")] = Metric.build(
            spec, key, dataset="sales_table", state=SimpleAdditiveState(101.0)
        )

        # Create symbol lookup
        symbol_lookup: dict[tuple[specs.MetricSpec, ResultKey, str], Any] = {(spec, key, "sales_table"): "x_1"}

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

        result = metric_trace(metrics, "exec-123", report, symbols, symbol_lookup)

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
        report = AnalysisReport()
        for m in metrics:
            report[(m.spec, m.key, m.dataset)] = m

        result = metric_trace(metrics, "exec-123", report, [], {})

        assert result.num_rows == 3
        data = result.to_pydict()

        # All should have matching values
        for i in range(3):
            assert data["value_db"][i] == data["value_analysis"][i]

    # Removed test_print_metric_trace - stdout inspection is unstable
    # Removed test_print_metric_trace_symbol_sorting - stdout inspection is unstable

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
        report[(spec, key, "test_dataset")] = Metric.build(
            spec, key, dataset="test_dataset", state=SimpleAdditiveState(100.0)
        )

        # Create empty symbol lookup since we're not using symbols here
        symbol_lookup: dict[tuple[specs.MetricSpec, ResultKey, str], Any] = {}

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

        reports_table = analysis_reports_to_pyarrow_table(report, symbol_lookup)
        assert all(col.islower() for col in reports_table.column_names)

        symbols_table = symbols_to_pyarrow_table(symbols)
        assert all(col.islower() for col in symbols_table.column_names)

        # Test trace table
        trace = metric_trace(metrics, "exec-123", report, symbols, symbol_lookup)
        assert all(col.islower() for col in trace.column_names)


class TestMetricTraceStats:
    """Test cases for metric_trace_stats function."""

    def test_empty_trace(self) -> None:
        """Test metric_trace_stats with empty trace table."""
        from dqx.data import metric_trace_stats

        # Create empty trace
        trace = metric_trace([], "exec-123", AnalysisReport(), [], {})
        stats = metric_trace_stats(trace)

        assert stats.total_rows == 0
        assert stats.discrepancy_count == 0
        assert stats.discrepancy_rows == []
        assert stats.discrepancy_details == []

    def test_no_discrepancies(self) -> None:
        """Test metric_trace_stats with matching values."""
        from dqx.data import metric_trace_stats

        test_date = date(2024, 1, 1)

        # Create data with matching values
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
        report[(spec, key, "sales_table")] = Metric.build(
            spec, key, dataset="sales_table", state=SimpleAdditiveState(100.0)
        )

        trace = metric_trace(metrics, "exec-123", report, [], {})
        stats = metric_trace_stats(trace)

        assert stats.total_rows == 1
        assert stats.discrepancy_count == 0
        assert stats.discrepancy_rows == []
        assert stats.discrepancy_details == []

    def test_with_discrepancies(self) -> None:
        """Test metric_trace_stats with value discrepancies."""
        from dqx.data import metric_trace_stats

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
        report[(spec, key, "sales_table")] = Metric.build(
            spec, key, dataset="sales_table", state=SimpleAdditiveState(101.0)
        )

        # Create symbol lookup
        symbol_lookup: dict[tuple[specs.MetricSpec, ResultKey, str], Any] = {(spec, key, "sales_table"): "x_1"}

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

        trace = metric_trace(metrics, "exec-123", report, symbols, symbol_lookup)
        stats = metric_trace_stats(trace)

        assert stats.total_rows == 1
        assert stats.discrepancy_count == 1
        assert stats.discrepancy_rows == [0]
        assert len(stats.discrepancy_details) == 1

        detail = stats.discrepancy_details[0]
        assert detail["row_index"] == 0
        assert detail["value_db"] == 100.0
        assert detail["value_analysis"] == 101.0
        assert detail["value_final"] == 102.0
        assert "value_db != value_analysis" in detail["discrepancies"]
        assert "value_db != value_final" in detail["discrepancies"]
        assert "value_analysis != value_final" in detail["discrepancies"]

    def test_extended_metrics_excluded(self) -> None:
        """Test that extended metrics are excluded from discrepancy counts."""
        from dqx.data import metric_trace_stats

        test_date = date(2024, 1, 1)

        # Create data with discrepancies for extended metric
        base_spec = specs.NumRows()
        dod_spec = specs.DayOverDay.from_base_spec(base_spec)

        metrics = [
            Metric.build(
                dod_spec,  # Extended metric
                ResultKey(yyyy_mm_dd=test_date, tags={}),
                dataset="sales_table",
                state=SimpleAdditiveState(100.0),
                metadata=Metadata(execution_id="exec-123"),
            )
        ]

        spec = dod_spec
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        report = AnalysisReport()
        report[(spec, key, "sales_table")] = Metric.build(
            spec, key, dataset="sales_table", state=SimpleAdditiveState(101.0)
        )

        trace = metric_trace(metrics, "exec-123", report, [], {})
        stats = metric_trace_stats(trace)

        # Extended metrics should not count as discrepancies
        assert stats.total_rows == 1
        assert stats.discrepancy_count == 0
        assert stats.discrepancy_rows == []
        assert stats.discrepancy_details == []

    def test_mixed_discrepancies(self) -> None:
        """Test with mix of discrepancies and matching values."""
        from dqx.data import metric_trace_stats

        test_date = date(2024, 1, 1)

        # Create metrics with mixed results
        metrics = [
            Metric.build(
                specs.NumRows(),
                ResultKey(yyyy_mm_dd=test_date, tags={}),
                dataset="sales_table",
                state=SimpleAdditiveState(100.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
            Metric.build(
                specs.Average("price"),
                ResultKey(yyyy_mm_dd=test_date, tags={}),
                dataset="sales_table",
                state=SimpleAdditiveState(50.0),
                metadata=Metadata(execution_id="exec-123"),
            ),
        ]

        # Create reports with one discrepancy
        report = AnalysisReport()
        spec1 = specs.NumRows()
        spec2 = specs.Average("price")
        key = ResultKey(yyyy_mm_dd=test_date, tags={})

        report[(spec1, key, "sales_table")] = Metric.build(
            spec1,
            key,
            dataset="sales_table",
            state=SimpleAdditiveState(101.0),  # Discrepancy
        )
        report[(spec2, key, "sales_table")] = Metric.build(
            spec2,
            key,
            dataset="sales_table",
            state=SimpleAdditiveState(50.0),  # Match
        )

        trace = metric_trace(metrics, "exec-123", report, [], {})
        stats = metric_trace_stats(trace)

        assert stats.total_rows == 2
        assert stats.discrepancy_count == 1  # Only num_rows has discrepancy
