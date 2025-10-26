"""Demo showing how display functions now use PyArrow transformation internally."""

import datetime as dt

from dqx import specs, states
from dqx.analyzer import AnalysisReport
from dqx.common import ResultKey
from dqx.display import print_analysis_report, print_metrics_by_execution_id
from dqx.models import Metric


def demo_display_with_pyarrow() -> None:
    """Demonstrate how display functions now use PyArrow internally."""
    print("=== Display Functions with PyArrow Integration ===\n")

    # Create sample metrics
    metrics = [
        Metric.build(
            specs.Average("response_time"),
            ResultKey(yyyy_mm_dd=dt.date(2024, 3, 15), tags={"env": "prod", "region": "us-east"}),
            dataset="api_metrics",
            state=states.Average(125.5, 1000),
        ),
        Metric.build(
            specs.Sum("total_orders"),
            ResultKey(yyyy_mm_dd=dt.date(2024, 3, 14), tags={"env": "staging"}),
            dataset="sales",
            state=states.SimpleAdditiveState(5000.0),
        ),
        Metric.build(
            specs.NullCount("customer_id"),
            ResultKey(yyyy_mm_dd=dt.date(2024, 3, 15), tags={}),
            dataset="customers",
            state=states.SimpleAdditiveState(12.0),
        ),
    ]

    print("1. Display metrics using print_metrics_by_execution_id:")
    print("   (Internally uses metrics_to_pyarrow_table for sorting and formatting)\n")
    print_metrics_by_execution_id(metrics, "demo-exec-123")

    print("\n" + "=" * 60 + "\n")

    # Create analysis reports with symbol mappings
    print("2. Display analysis reports using print_analysis_report:")
    print("   (Internally uses analysis_reports_to_pyarrow_table for processing)\n")

    key1 = ResultKey(yyyy_mm_dd=dt.date(2024, 3, 15), tags={"env": "prod"})
    key2 = ResultKey(yyyy_mm_dd=dt.date(2024, 3, 14), tags={"env": "dev", "version": "2.1.0"})

    metric1 = Metric.build(
        specs.Average("latency"),
        key1,
        dataset="performance",
        state=states.Average(45.2, 500),
    )
    metric2 = Metric.build(
        specs.Maximum("memory_usage"),
        key2,
        dataset="system",
        state=states.SimpleAdditiveState(1024.0),
    )

    # Create reports with symbol mappings
    report1 = AnalysisReport()
    metric_key1 = (specs.Average("latency"), key1)
    report1[metric_key1] = metric1
    report1.symbol_mapping[metric_key1] = "avg_latency"

    report2 = AnalysisReport()
    metric_key2 = (specs.Maximum("memory_usage"), key2)
    report2[metric_key2] = metric2
    report2.symbol_mapping[metric_key2] = "max_memory"

    reports = {"backend": report1, "system": report2}
    print_analysis_report(reports)

    print("\n" + "=" * 60 + "\n")
    print("Benefits of PyArrow integration:")
    print("- Consistent data transformation logic")
    print("- Sorting and formatting handled in one place")
    print("- Easy to export data to other formats (Parquet, CSV, etc.)")
    print("- Better performance for large datasets")


if __name__ == "__main__":
    demo_display_with_pyarrow()
