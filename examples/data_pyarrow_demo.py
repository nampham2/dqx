"""Demo of PyArrow transformation functions in data module."""

import datetime as dt

from dqx import data, specs, states
from dqx.analyzer import AnalysisReport
from dqx.common import ResultKey
from dqx.models import Metric


def demo_metrics_to_pyarrow() -> None:
    """Demonstrate metrics_to_pyarrow_table function."""
    print("=== Demo: metrics_to_pyarrow_table ===\n")

    # Create sample metrics
    metrics = [
        Metric.build(
            specs.Average("price"),
            ResultKey(yyyy_mm_dd=dt.date(2024, 1, 26), tags={"env": "prod", "region": "us-east"}),
            dataset="sales",
            state=states.Average(25.5, 100),
        ),
        Metric.build(
            specs.Sum("quantity"),
            ResultKey(yyyy_mm_dd=dt.date(2024, 1, 25), tags={"env": "staging"}),
            dataset="inventory",
            state=states.SimpleAdditiveState(1000.0),
        ),
        Metric.build(
            specs.NumRows(),
            ResultKey(yyyy_mm_dd=dt.date(2024, 1, 26), tags={}),
            dataset="users",
            state=states.SimpleAdditiveState(42.0),
        ),
    ]

    # Convert to PyArrow table
    execution_id = "demo-exec-123"
    table = data.metrics_to_pyarrow_table(metrics, execution_id)

    print(f"Table schema: {table.schema}")
    print(f"Number of rows: {table.num_rows}\n")

    # Show table contents
    print("Table contents:")
    for col_name in table.schema.names:
        print(f"{col_name}: {table[col_name].to_pylist()}")
    print()


def demo_analysis_reports_to_pyarrow() -> None:
    """Demonstrate analysis_reports_to_pyarrow_table function."""
    print("=== Demo: analysis_reports_to_pyarrow_table ===\n")

    # Create sample analysis reports
    key1 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 26), tags={"env": "prod"})
    key2 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 25), tags={"env": "dev", "version": "1.2.3"})

    metric1 = Metric.build(
        specs.Average("response_time"),
        key1,
        dataset="api_metrics",
        state=states.Average(125.5, 1000),
    )
    metric2 = Metric.build(
        specs.NullCount("user_id"),
        key2,
        dataset="user_data",
        state=states.SimpleAdditiveState(15.0),
    )

    # Create reports with symbol mappings
    report1 = AnalysisReport()
    metric_key1 = (specs.Average("response_time"), key1)
    report1[metric_key1] = metric1
    report1.symbol_mapping[metric_key1] = "avg_api_response"

    report2 = AnalysisReport()
    metric_key2 = (specs.NullCount("user_id"), key2)
    report2[metric_key2] = metric2
    # No symbol mapping for this one to show "-" default

    reports = {"api_datasource": report1, "user_datasource": report2}

    # Convert to PyArrow table
    table = data.analysis_reports_to_pyarrow_table(reports)

    print(f"Table schema: {table.schema}")
    print(f"Number of rows: {table.num_rows}\n")

    # Show table contents
    print("Table contents:")
    for col_name in table.schema.names:
        print(f"{col_name}: {table[col_name].to_pylist()}")
    print()


def main() -> None:
    """Run the demos."""
    demo_metrics_to_pyarrow()
    print("\n" + "=" * 60 + "\n")
    demo_analysis_reports_to_pyarrow()


if __name__ == "__main__":
    main()
