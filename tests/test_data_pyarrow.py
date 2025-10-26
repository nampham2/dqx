"""Tests for PyArrow transformation functions in data module."""

import datetime as dt
from typing import Sequence

import pyarrow as pa
import pytest

from dqx import specs, states
from dqx.analyzer import AnalysisReport
from dqx.common import ResultKey
from dqx.data import analysis_reports_to_pyarrow_table, metrics_to_pyarrow_table
from dqx.models import Metric


@pytest.fixture
def sample_metrics() -> Sequence[Metric]:
    """Create sample metrics for testing."""
    key1 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 26), tags={"env": "prod"})
    key2 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 25), tags={"env": "staging", "region": "us-west"})
    key3 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 26), tags={})

    return [
        Metric.build(
            specs.Average("price"),
            key1,
            dataset="sales",
            state=states.Average(25.5, 100),
        ),
        Metric.build(
            specs.Sum("quantity"),
            key2,
            dataset="inventory",
            state=states.SimpleAdditiveState(1000.0),
        ),
        Metric.build(
            specs.NumRows(),
            key3,
            dataset="users",
            state=states.SimpleAdditiveState(42.0),
        ),
    ]


@pytest.fixture
def sample_analysis_reports() -> dict[str, AnalysisReport]:
    """Create sample analysis reports for testing."""
    # Create metrics with different dates for sorting tests
    key1 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 26), tags={"env": "prod"})
    key2 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 25), tags={})

    metric1 = Metric.build(
        specs.Average("price"),
        key1,
        dataset="sales",
        state=states.Average(25.5, 100),
    )
    metric2 = Metric.build(
        specs.Sum("quantity"),
        key2,
        dataset="inventory",
        state=states.SimpleAdditiveState(1000.0),
    )

    # Create report with symbol mappings
    report1 = AnalysisReport()
    metric_key1 = (specs.Average("price"), key1)
    metric_key2 = (specs.Sum("quantity"), key2)

    report1[metric_key1] = metric1
    report1[metric_key2] = metric2

    # Add symbol mappings
    report1.symbol_mapping[metric_key1] = "avg_price"
    # Intentionally don't add mapping for metric2 to test missing symbol

    return {"datasource1": report1}


def test_metrics_to_pyarrow_basic(sample_metrics: Sequence[Metric]) -> None:
    """Test basic conversion of metrics to PyArrow table."""
    execution_id = "test-exec-123"
    table = metrics_to_pyarrow_table(sample_metrics, execution_id)

    # Verify schema
    assert table.schema.names == ["Date", "Metric Name", "Type", "Dataset", "Value", "Tags"]
    assert table.schema.field("Date").type == pa.date32()
    assert table.schema.field("Value").type == pa.float64()

    # Verify data
    assert table.num_rows == 3

    # Convert to dict for easier assertions
    result = table.to_pydict()

    # Check first row (should be 2024-01-26 due to sorting)
    assert result["Date"][0] == dt.date(2024, 1, 26)
    assert result["Metric Name"][0] == "average(price)"
    assert result["Type"][0] == "Average"
    assert result["Dataset"][0] == "sales"
    assert result["Value"][0] == pytest.approx(25.5)
    assert result["Tags"][0] == "env=prod"


def test_metrics_to_pyarrow_sorting(sample_metrics: Sequence[Metric]) -> None:
    """Test that metrics are sorted by date (newest first) then by name."""
    table = metrics_to_pyarrow_table(sample_metrics, "exec-id")
    result = table.to_pydict()

    # Verify sorting: 2024-01-26 entries come before 2024-01-25
    dates = result["Date"]
    names = result["Metric Name"]

    # Find indices
    idx_26 = [i for i, d in enumerate(dates) if d == dt.date(2024, 1, 26)]
    idx_25 = [i for i, d in enumerate(dates) if d == dt.date(2024, 1, 25)]

    # All 2024-01-26 entries should come before 2024-01-25
    assert max(idx_26) < min(idx_25)

    # Within same date, check alphabetical order
    names_26 = [names[i] for i in idx_26]
    assert names_26 == sorted(names_26)


def test_metrics_to_pyarrow_tag_formatting(sample_metrics: Sequence[Metric]) -> None:
    """Test tag formatting in PyArrow table."""
    table = metrics_to_pyarrow_table(sample_metrics, "exec-id")
    result = table.to_pydict()

    tags = result["Tags"]

    # Empty tags should be "-"
    assert "-" in tags

    # Single tag
    assert "env=prod" in tags

    # Multiple tags should be comma-separated (order may vary)
    multi_tag_entries = [t for t in tags if "," in t]
    assert len(multi_tag_entries) == 1
    assert "env=staging" in multi_tag_entries[0]
    assert "region=us-west" in multi_tag_entries[0]


def test_metrics_to_pyarrow_empty_input() -> None:
    """Test with empty metrics list."""
    table = metrics_to_pyarrow_table([], "empty-exec")

    # Should have schema but no rows
    assert table.schema.names == ["Date", "Metric Name", "Type", "Dataset", "Value", "Tags"]
    assert table.num_rows == 0


def test_analysis_reports_to_pyarrow_basic(sample_analysis_reports: dict[str, AnalysisReport]) -> None:
    """Test basic conversion of analysis reports to PyArrow table."""
    table = analysis_reports_to_pyarrow_table(sample_analysis_reports)

    # Verify schema
    expected_columns = ["Date", "Metric Name", "Symbol", "Type", "Dataset", "Value", "Tags"]
    assert table.schema.names == expected_columns
    assert table.schema.field("Date").type == pa.date32()
    assert table.schema.field("Value").type == pa.float64()

    # Verify data
    assert table.num_rows == 2

    result = table.to_pydict()

    # Check data integrity
    assert dt.date(2024, 1, 26) in result["Date"]
    assert dt.date(2024, 1, 25) in result["Date"]
    assert "average(price)" in result["Metric Name"]
    assert "sum(quantity)" in result["Metric Name"]


def test_analysis_reports_to_pyarrow_symbol_mapping(sample_analysis_reports: dict[str, AnalysisReport]) -> None:
    """Test symbol mapping in analysis reports."""
    table = analysis_reports_to_pyarrow_table(sample_analysis_reports)
    result = table.to_pydict()

    names = result["Metric Name"]

    # Find symbol for average(price) - should have mapping
    avg_idx = names.index("average(price)")
    assert result["Symbol"][avg_idx] == "avg_price"

    # Find symbol for sum(quantity) - should be "-" (no mapping)
    sum_idx = names.index("sum(quantity)")
    assert result["Symbol"][sum_idx] == "-"


def test_analysis_reports_to_pyarrow_empty_reports() -> None:
    """Test with empty reports."""
    # Empty dict
    table = analysis_reports_to_pyarrow_table({})
    assert table.num_rows == 0
    assert table.schema.names == ["Date", "Metric Name", "Symbol", "Type", "Dataset", "Value", "Tags"]

    # Dict with empty report
    empty_report = AnalysisReport()
    table = analysis_reports_to_pyarrow_table({"empty": empty_report})
    assert table.num_rows == 0


def test_analysis_reports_to_pyarrow_multiple_datasources() -> None:
    """Test with multiple datasources."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 26), tags={})
    metric1 = Metric.build(
        specs.NullCount("orders"),
        key,
        dataset="sales",
        state=states.SimpleAdditiveState(100.0),
    )
    metric2 = Metric.build(
        specs.NullCount("users"),
        key,
        dataset="users",
        state=states.SimpleAdditiveState(50.0),
    )

    report1 = AnalysisReport()
    report1[(specs.NullCount("orders"), key)] = metric1

    report2 = AnalysisReport()
    report2[(specs.NullCount("users"), key)] = metric2

    reports = {"datasource1": report1, "datasource2": report2}
    table = analysis_reports_to_pyarrow_table(reports)

    assert table.num_rows == 2
    result = table.to_pydict()
    assert "null_count(orders)" in result["Metric Name"]
    assert "null_count(users)" in result["Metric Name"]


def test_analysis_reports_to_pyarrow_missing_dataset() -> None:
    """Test handling of missing dataset in metrics."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 26), tags={})
    metric = Metric.build(
        specs.NullCount("test"),
        key,
        dataset="",  # Empty dataset
        state=states.SimpleAdditiveState(10.0),
    )
    # Manually set dataset to None to test handling
    metric.dataset = None  # type: ignore[assignment]

    report = AnalysisReport()
    report[(specs.NullCount("test"), key)] = metric

    table = analysis_reports_to_pyarrow_table({"test": report})
    result = table.to_pydict()

    # Should show "-" for missing dataset
    assert result["Dataset"][0] == "-"
