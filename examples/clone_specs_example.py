#!/usr/bin/env python3
"""Example demonstrating the use of clone() method for MetricSpec objects."""

import datetime

import pyarrow as pa

from dqx import specs
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def main() -> None:
    """Demonstrate cloning specs for reuse across different datasets."""
    # Create two datasets with different data ranges
    sales_q1 = pa.table(
        {
            "amount": [100, 200, 150, 300, 250],
            "region": ["North", "South", "North", "East", "West"],
        }
    )

    sales_q2 = pa.table(
        {
            "amount": [150, 250, 200, 350, 300],
            "region": ["North", "South", "North", "East", "West"],
        }
    )

    # Create datasources
    ds_q1 = DuckRelationDataSource.from_arrow(sales_q1, "sales_q1")
    ds_q2 = DuckRelationDataSource.from_arrow(sales_q2, "sales_q2")

    # Define reusable specs
    common_specs = [
        specs.NumRows(),
        specs.Average("amount"),
        specs.Sum("amount"),
        specs.Minimum("amount"),
        specs.Maximum("amount"),
    ]

    # Clone specs for Q2 analysis (to ensure independence)
    q2_specs = [spec.clone() for spec in common_specs]

    # Create checks for each quarter
    @check(name="Q1 Sales Metrics", datasets=["sales_q1"])
    def check_q1(mp: MetricProvider, ctx: Context) -> None:
        """Analyze Q1 sales using original specs."""
        for spec in common_specs:
            mp.metric(spec)  # Compute the metric

        # Add specific assertions for Q1
        avg_amount = mp.average("amount")
        ctx.assert_that(avg_amount).where(name="Q1 average sales", severity="P2").is_between(180, 220)

    @check(name="Q2 Sales Metrics", datasets=["sales_q2"])
    def check_q2(mp: MetricProvider, ctx: Context) -> None:
        """Analyze Q2 sales using cloned specs."""
        for spec in q2_specs:
            mp.metric(spec)  # Compute the metric

        # Add specific assertions for Q2
        avg_amount = mp.average("amount")
        ctx.assert_that(avg_amount).where(name="Q2 average sales", severity="P2").is_between(230, 270)

    # Run verification suites
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run Q1 analysis
    print("Running Q1 Sales Analysis...")
    suite_q1 = VerificationSuite([check_q1], db, "Q1 Sales Suite")
    suite_q1.run([ds_q1], key)

    # Run Q2 analysis
    print("\nRunning Q2 Sales Analysis...")
    suite_q2 = VerificationSuite([check_q2], db, "Q2 Sales Suite")
    suite_q2.run([ds_q2], key)

    # Show that both sets of specs computed independently
    print("\nMetric Trace Summary:")
    # Get traces from both suites
    trace1 = suite_q1.metric_trace(db)
    trace2 = suite_q2.metric_trace(db)

    print("Q1 Metrics:")
    for row in trace1.to_pylist():
        print(f"  - {row['metric']}: {row['value_final']}")

    print("\nQ2 Metrics:")
    for row in trace2.to_pylist():
        print(f"  - {row['metric']}: {row['value_final']}")

    print("\nClone() method benefits:")
    print("1. Specs can be reused across different datasets")
    print("2. Each clone has independent analyzer instances")
    print("3. No interference between metric computations")
    print("4. Cleaner code by defining common specs once")


if __name__ == "__main__":
    main()
