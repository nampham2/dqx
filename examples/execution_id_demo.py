"""
Demonstration of the execution ID feature in DQX.

This example shows how to:
1. Access the unique execution ID for each suite run
2. Query metrics by execution ID to retrieve all metrics from a specific run
3. Use execution IDs to isolate metrics between different suite executions
"""

import datetime

import pyarrow as pa

from dqx import data
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def main() -> None:
    """Demonstrate execution ID functionality."""

    # Define data quality checks
    @check(name="Sales Data Quality", datasets=["sales"])
    def sales_quality_check(mp: MetricProvider, ctx: Context) -> None:
        # Basic metrics
        avg_amount = mp.average("amount")
        min_amount = mp.minimum("amount")
        max_amount = mp.maximum("amount")
        row_count = mp.num_rows()

        # Assertions
        ctx.assert_that(avg_amount).where(name="Average sale amount").is_between(10, 1000)
        ctx.assert_that(min_amount).where(name="Minimum sale amount").is_positive()
        ctx.assert_that(max_amount).where(name="Maximum sale amount").is_lt(10000)
        ctx.assert_that(row_count).where(name="Row count").is_gt(0)

    @check(name="Time Series Analysis", datasets=["sales"])
    def time_series_check(mp: MetricProvider, ctx: Context) -> None:
        # Extended metrics with lag
        avg_amount = mp.average("amount")
        dod_amount = mp.ext.day_over_day(avg_amount)

        ctx.assert_that(dod_amount).where(name="Day-over-day change").is_between(-50, 50)

    # Create sample data for multiple days
    day1_data = pa.table({"amount": [100.0, 200.0, 150.0, 300.0, 250.0], "product": ["A", "B", "A", "C", "B"]})

    day2_data = pa.table({"amount": [120.0, 180.0, 160.0, 320.0, 240.0], "product": ["A", "B", "A", "C", "B"]})

    # Initialize database
    db = InMemoryMetricDB()

    print("=== DQX Execution ID Demo ===\n")

    # Run 1: Process day 1 data
    print("Running suite for Day 1...")
    suite1 = VerificationSuite([sales_quality_check, time_series_check], db, "Sales DQ Suite")
    key1 = ResultKey(datetime.date(2024, 1, 1), {"env": "prod", "region": "us-east"})
    datasource1 = DuckRelationDataSource.from_arrow(day1_data, "sales")

    suite1.run([datasource1], key1)
    exec_id1 = suite1.execution_id
    print(f"Suite 1 Execution ID: {exec_id1}")

    # Run 2: Process day 2 data
    print("\nRunning suite for Day 2...")
    suite2 = VerificationSuite([sales_quality_check, time_series_check], db, "Sales DQ Suite")
    key2 = ResultKey(datetime.date(2024, 1, 2), {"env": "prod", "region": "us-east"})
    datasource2 = DuckRelationDataSource.from_arrow(day2_data, "sales")

    suite2.run([datasource2], key2)
    exec_id2 = suite2.execution_id
    print(f"Suite 2 Execution ID: {exec_id2}")

    # Demonstrate retrieving metrics by execution ID
    print("\n=== Retrieving Metrics by Execution ID ===")

    # Get metrics from first execution
    print(f"\nMetrics from Execution 1 ({exec_id1[:8]}...):")
    metrics1 = data.metrics_by_execution_id(db, exec_id1)
    for metric in metrics1:
        print(f"  - {metric.spec.name} on {metric.key.yyyy_mm_dd}: value={metric.value}, tags={metric.key.tags}")

    # Get metrics from second execution
    print(f"\nMetrics from Execution 2 ({exec_id2[:8]}...):")
    metrics2 = data.metrics_by_execution_id(db, exec_id2)
    for metric in metrics2:
        print(f"  - {metric.spec.name} on {metric.key.yyyy_mm_dd}: value={metric.value}, tags={metric.key.tags}")

    # Show that executions are isolated
    print("\n=== Execution Isolation ===")
    print(f"Execution 1 has {len(metrics1)} metrics")
    print(f"Execution 2 has {len(metrics2)} metrics (includes lag metrics)")

    # Verify each metric has the correct execution_id
    exec1_ids = {m.key.tags["__execution_id"] for m in metrics1}
    exec2_ids = {m.key.tags["__execution_id"] for m in metrics2}

    print(f"\nExecution 1 metrics all have ID: {list(exec1_ids)[0][:8]}...")
    print(f"Execution 2 metrics all have ID: {list(exec2_ids)[0][:8]}...")
    print("\nExecutions are properly isolated!")

    # Show how to query for metrics from a specific execution and date
    print("\n=== Advanced Queries ===")

    # Filter execution 2 metrics by date
    current_date_metrics = [m for m in metrics2 if m.key.yyyy_mm_dd == datetime.date(2024, 1, 2)]
    lag_date_metrics = [m for m in metrics2 if m.key.yyyy_mm_dd == datetime.date(2024, 1, 1)]

    print(f"Execution 2 has {len(current_date_metrics)} metrics for current date (2024-01-02)")
    print(f"Execution 2 has {len(lag_date_metrics)} metrics for lag date (2024-01-01)")

    # Demonstrate use case: Finding all metrics from a failed suite run
    print("\n=== Use Case: Debugging Failed Suite Runs ===")
    print("With execution IDs, you can:")
    print("1. Log the execution ID when a suite fails")
    print("2. Later retrieve all metrics from that specific run")
    print("3. Analyze exactly what values were computed during the failure")
    print(f"\nExample: data.metrics_by_execution_id(db, '{exec_id2[:8]}...')")
    print("Returns all metrics computed during that specific suite execution")


if __name__ == "__main__":
    main()
