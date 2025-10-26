"""
Demonstration of metric_trace functionality for debugging metric values.

This example shows how to trace metric values across:
1. Metrics stored in the database
2. Analysis reports from VerificationSuite
3. Symbol values from Provider

The metric_trace function helps identify discrepancies between these sources.
"""

from datetime import date

from returns.result import Failure, Success

from dqx import data, display, specs
from dqx.analyzer import AnalysisReport
from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.provider import SymbolInfo
from dqx.states import SimpleAdditiveState


def create_sample_metrics() -> list[Metric]:
    """Create sample metrics as if they were stored in DB."""
    test_date = date(2024, 1, 15)

    metrics = [
        # NumRows metric
        Metric.build(
            specs.NumRows(),
            ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod", "region": "US"}),
            dataset="sales_table",
            state=SimpleAdditiveState(1000000.0),
            metadata=Metadata(execution_id="exec-123"),
        ),
        # NullCount metric
        Metric.build(
            specs.NullCount("order_id"),
            ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod", "region": "US"}),
            dataset="sales_table",
            state=SimpleAdditiveState(0.0),
            metadata=Metadata(execution_id="exec-123"),
        ),
        # Average metric
        Metric.build(
            specs.Average("amount"),
            ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod", "region": "US"}),
            dataset="sales_table",
            state=SimpleAdditiveState(125.50),
            metadata=Metadata(execution_id="exec-123"),
        ),
        # Another dataset
        Metric.build(
            specs.NumRows(),
            ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod", "region": "US"}),
            dataset="users_table",
            state=SimpleAdditiveState(50000.0),
            metadata=Metadata(execution_id="exec-123"),
        ),
    ]

    return metrics


def create_sample_reports() -> dict[str, AnalysisReport]:
    """Create sample analysis reports with some differences from DB metrics."""
    test_date = date(2024, 1, 15)

    # Create sales_table report
    sales_report = AnalysisReport()

    # NumRows - slightly different value (simulating recalculation)
    spec1 = specs.NumRows()
    key1 = ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod", "region": "US"})
    sales_report[(spec1, key1)] = Metric.build(spec1, key1, dataset="sales_table", state=SimpleAdditiveState(1000001.0))
    sales_report.symbol_mapping[(spec1, key1)] = "x_1"

    # NullCount - same value
    spec2 = specs.NullCount("order_id")
    key2 = ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod", "region": "US"})
    sales_report[(spec2, key2)] = Metric.build(spec2, key2, dataset="sales_table", state=SimpleAdditiveState(0.0))
    sales_report.symbol_mapping[(spec2, key2)] = "x_2"

    # Add a metric not in DB
    spec3 = specs.Maximum("amount")
    key3 = ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod", "region": "US"})
    sales_report[(spec3, key3)] = Metric.build(spec3, key3, dataset="sales_table", state=SimpleAdditiveState(9999.99))
    sales_report.symbol_mapping[(spec3, key3)] = "x_3"

    # Create users_table report
    users_report = AnalysisReport()

    spec4 = specs.NumRows()
    key4 = ResultKey(yyyy_mm_dd=test_date, tags={"env": "prod", "region": "US"})
    users_report[(spec4, key4)] = Metric.build(spec4, key4, dataset="users_table", state=SimpleAdditiveState(50000.0))
    users_report.symbol_mapping[(spec4, key4)] = "x_4"

    return {"sales_table": sales_report, "users_table": users_report}


def create_sample_symbols() -> list[SymbolInfo]:
    """Create sample symbols with some successes and failures."""
    test_date = date(2024, 1, 15)

    symbols = [
        # x_1 - NumRows with yet another value
        SymbolInfo(
            name="x_1",
            metric="num_rows()",
            dataset="sales_table",
            value=Success(1000002.0),
            yyyy_mm_dd=test_date,
            tags={"env": "prod", "region": "US"},
        ),
        # x_2 - NullCount success
        SymbolInfo(
            name="x_2",
            metric="null_count(order_id)",
            dataset="sales_table",
            value=Success(0.0),
            yyyy_mm_dd=test_date,
            tags={"env": "prod", "region": "US"},
        ),
        # x_3 - Maximum with failure
        SymbolInfo(
            name="x_3",
            metric="maximum(amount)",
            dataset="sales_table",
            value=Failure("Column 'amount' not found"),
            yyyy_mm_dd=test_date,
            tags={"env": "prod", "region": "US"},
        ),
        # x_4 - users NumRows
        SymbolInfo(
            name="x_4",
            metric="num_rows()",
            dataset="users_table",
            value=Success(50000.0),
            yyyy_mm_dd=test_date,
            tags={"env": "prod", "region": "US"},
        ),
        # x_5 - A symbol not in reports or metrics
        SymbolInfo(
            name="x_5",
            metric="minimum(price)",
            dataset="products_table",
            value=Success(0.99),
            yyyy_mm_dd=test_date,
            tags={"env": "prod", "region": "US"},
        ),
    ]

    return symbols


def main() -> None:
    """Demonstrate metric_trace functionality."""
    print("=== Metric Trace Demo ===\n")

    # Create sample data
    metrics = create_sample_metrics()
    reports = create_sample_reports()
    symbols = create_sample_symbols()

    print(f"Created {len(metrics)} metrics from DB")
    print(f"Created {len(reports)} analysis reports")
    print(f"Created {len(symbols)} symbols\n")

    # Create the trace
    trace_table = data.metric_trace(metrics, "exec-123", reports, symbols)

    # Display the trace
    display.print_metric_trace(trace_table, "exec-123")

    print("\n=== Analysis ===")
    print("Notice the following in the trace:")
    print("1. NumRows for sales_table has different values:")
    print("   - DB: 1000000.0")
    print("   - Analysis: 1000001.0")
    print("   - Final: 1000002.0")
    print("   This is flagged as a discrepancy!\n")

    print("2. NullCount has consistent values across all sources")
    print("3. Average(amount) only exists in DB (not in analysis/symbols)")
    print("4. Maximum(amount) exists in analysis but failed in symbol evaluation")
    print("5. Minimum(price) only exists as a symbol (not in DB/analysis)")

    print("\n=== PyArrow Table Access ===")
    print("You can also work with the raw PyArrow table:")

    # Work with PyArrow table directly
    table_data = trace_table.to_pydict()

    # Find all rows with discrepancies
    print("\nChecking for discrepancies...")
    discrepancy_count = 0
    for i in range(trace_table.num_rows):
        value_db = table_data["value_db"][i]
        value_analysis = table_data["value_analysis"][i]

        if value_db is not None and value_analysis is not None and value_db != value_analysis:
            print(
                f"  - {table_data['metric'][i]} in {table_data['dataset'][i]}: DB={value_db}, Analysis={value_analysis}"
            )
            discrepancy_count += 1

    if discrepancy_count > 0:
        print(f"\nTotal: {discrepancy_count} metrics with DB/Analysis discrepancies")
    else:
        print("No discrepancies found between DB and Analysis values")

    # Find all failed symbols
    print("\nChecking for failed symbol evaluations...")
    failure_count = 0
    for i in range(trace_table.num_rows):
        error = table_data["error"][i]
        if error is not None:
            print(f"  - {table_data['symbol'][i]} ({table_data['metric'][i]}): {error}")
            failure_count += 1

    if failure_count > 0:
        print(f"\nTotal: {failure_count} failed symbol evaluations")
    else:
        print("All symbols evaluated successfully")

    print("\n=== Additional Table Operations ===")
    print(f"Total rows in trace: {trace_table.num_rows}")
    print(f"Columns: {trace_table.column_names}")
    print("\nYou can filter, sort, or export this table using PyArrow operations.")


if __name__ == "__main__":
    main()
