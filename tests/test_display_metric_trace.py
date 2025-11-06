"""Test for print_metric_trace display functionality."""

from datetime import date

import pyarrow as pa

from dqx.display import print_metric_trace


def test_print_metric_trace_with_data_availability() -> None:
    """Test that print_metric_trace displays data availability column correctly."""
    # Create a test trace table with data_av_ratio
    data = {
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        "metric": ["num_rows()", "average(price)", "null_count(id)"],
        "symbol": ["x_1", "x_2", "x_3"],
        "type": ["NumRows", "Average", "NullCount"],
        "dataset": ["orders", "orders", "orders"],
        "value_db": [100.0, 50.0, None],
        "value_analysis": [100.0, 51.0, 5.0],
        "value_final": [100.0, 51.0, 5.0],
        "error": [None, None, None],
        "tags": ["env=prod", "env=prod", "env=prod"],
        "is_extended": [False, False, False],
        "data_av_ratio": [0.95, 0.75, 0.40],  # High, medium, low availability
    }

    trace_table = pa.table(data)

    # Should not raise any exceptions
    print_metric_trace(trace_table)

    # Test with custom threshold
    print_metric_trace(trace_table, data_av_threshold=0.8)


def test_print_metric_trace_without_data_availability() -> None:
    """Test that print_metric_trace handles missing data_av_ratio gracefully."""
    # Create a test trace table without data_av_ratio
    data = {
        "date": [date(2024, 1, 1)],
        "metric": ["num_rows()"],
        "symbol": ["x_1"],
        "type": ["NumRows"],
        "dataset": ["orders"],
        "value_db": [100.0],
        "value_analysis": [100.0],
        "value_final": [100.0],
        "error": [None],
        "tags": ["env=prod"],
        "is_extended": [False],
    }

    trace_table = pa.table(data)

    # Should not raise any exceptions and show N/A for data availability
    print_metric_trace(trace_table)


def test_print_metric_trace_with_errors() -> None:
    """Test that print_metric_trace displays errors in combined column."""
    data = {
        "date": [date(2024, 1, 1), date(2024, 1, 2)],
        "metric": ["num_rows()", "average(price)"],
        "symbol": ["x_1", "x_2"],
        "type": ["NumRows", "Average"],
        "dataset": ["orders", "orders"],
        "value_db": [100.0, None],
        "value_analysis": [100.0, None],
        "value_final": [100.0, None],
        "error": [None, "Division by zero"],
        "tags": ["env=prod", "env=prod"],
        "is_extended": [False, False],
        "data_av_ratio": [0.95, 0.0],
    }

    trace_table = pa.table(data)

    # Should not raise any exceptions
    print_metric_trace(trace_table)


def test_print_metric_trace_with_discrepancies() -> None:
    """Test that print_metric_trace highlights discrepancies."""
    data = {
        "date": [date(2024, 1, 1)],
        "metric": ["num_rows()"],
        "symbol": ["x_1"],
        "type": ["NumRows"],
        "dataset": ["orders"],
        "value_db": [100.0],
        "value_analysis": [101.0],  # Discrepancy
        "value_final": [102.0],  # Another discrepancy
        "error": [None],
        "tags": ["env=prod"],
        "is_extended": [False],
        "data_av_ratio": [0.95],
    }

    trace_table = pa.table(data)

    # Should not raise any exceptions and highlight discrepancies
    print_metric_trace(trace_table)


def test_print_metric_trace_empty_table() -> None:
    """Test print_metric_trace with empty table."""
    # Create empty table with correct schema
    schema = pa.schema(
        [
            pa.field("date", pa.date32()),
            pa.field("metric", pa.string()),
            pa.field("symbol", pa.string()),
            pa.field("type", pa.string()),
            pa.field("dataset", pa.string()),
            pa.field("value_db", pa.float64()),
            pa.field("value_analysis", pa.float64()),
            pa.field("value_final", pa.float64()),
            pa.field("error", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("is_extended", pa.bool_()),
            pa.field("data_av_ratio", pa.float64()),
        ]
    )

    trace_table = pa.table({col: [] for col in schema.names}, schema=schema)

    # Should not raise any exceptions
    print_metric_trace(trace_table)


if __name__ == "__main__":
    # Run tests manually to see output
    test_print_metric_trace_with_data_availability()
    print("\n" + "=" * 50 + "\n")
    test_print_metric_trace_without_data_availability()
    print("\n" + "=" * 50 + "\n")
    test_print_metric_trace_with_errors()
    print("\n" + "=" * 50 + "\n")
    test_print_metric_trace_with_discrepancies()
