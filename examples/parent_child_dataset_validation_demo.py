#!/usr/bin/env python3
"""Demo of parent-child dataset validation in DQX.

This example shows how DQX validates dataset consistency between
parent metrics (derived metrics like day_over_day) and their child metrics
(base metrics that the derived metrics depend on).

In DQX:
- Parent: Derived metrics (e.g., day_over_day, week_over_week)
- Child: Base metrics that the derived metrics depend on
"""

import logging
from datetime import date

import pyarrow as pa

from dqx.api import Context, VerificationSuite, check
from dqx.common import DQXError, ResultKey, SqlDataSource
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider

# Configure logging to see validation details
logging.basicConfig(level=logging.INFO, format="%(message)s")


def demo_parent_child_dataset_mismatch() -> None:
    """Demonstrate detection of parent-child dataset mismatch."""
    print("\n=== Demo 1: Parent-Child Dataset Mismatch ===\n")

    # Create check function
    @check(name="Daily Revenue Check", datasets=["staging"])
    def revenue_check(mp: MetricProvider, ctx: Context) -> None:
        # Create a base metric (child) using production dataset
        revenue_sum = mp.sum("revenue", dataset="production")

        # Create a derived metric (parent) with DIFFERENT dataset - this is an error!
        revenue_dod = mp.ext.day_over_day(revenue_sum, dataset="staging")

        # Add assertion using the parent (derived) metric
        # The child (base) metric is a dependency and will be validated
        ctx.assert_that(revenue_dod).where(name="Revenue DoD must be positive").is_positive()

    try:
        # Create a verification suite
        db = InMemoryMetricDB()
        suite = VerificationSuite([revenue_check], db, "Revenue Monitoring")

        # Create dummy data sources using PyArrow tables
        datasources: dict[str, SqlDataSource] = {
            "production": DuckRelationDataSource.from_arrow(pa.table({"revenue": [1500]})),
            "staging": DuckRelationDataSource.from_arrow(pa.table({"revenue": [1200]})),
        }

        # Run the suite - this will trigger validation
        key = ResultKey(date.today(), tags={})
        suite.run(datasources, key)

    except DQXError as e:
        print(f"❌ Validation Error (as expected):\n{e}")
        print("\nThis error prevents subtle bugs where parent and child metrics")
        print("accidentally use different datasets, leading to incorrect results.")


def demo_valid_parent_child_datasets() -> None:
    """Demonstrate valid parent-child dataset configuration."""
    print("\n\n=== Demo 2: Valid Parent-Child Configuration ===\n")

    @check(name="Daily Revenue Check", datasets=["production"])
    def revenue_check(mp: MetricProvider, ctx: Context) -> None:
        # Create base metric (child) using production dataset
        revenue_sum = mp.sum("revenue", dataset="production")

        # Create derived metrics (parents) with SAME dataset - this is correct
        revenue_dod = mp.ext.day_over_day(revenue_sum, dataset="production")
        revenue_wow = mp.ext.week_over_week(revenue_sum, dataset="production")

        # Add assertions
        ctx.assert_that(revenue_sum).where(name="Revenue must exceed $1000").is_gt(1000)

        ctx.assert_that(revenue_dod).where(name="Daily revenue change within limits").is_gt(
            -0.1
        )  # No more than 10% daily decrease

        # Use the unused variable to avoid linting warnings
        _ = revenue_wow

    try:
        # Create a verification suite
        db = InMemoryMetricDB()
        suite = VerificationSuite([revenue_check], db, "Revenue Monitoring")

        # Create dummy data sources with multiple revenue values for time-based metrics
        datasources: dict[str, SqlDataSource] = {
            "production": DuckRelationDataSource.from_arrow(pa.table({"revenue": [1500, 1400]})),
            "staging": DuckRelationDataSource.from_arrow(pa.table({"revenue": [1200, 1100]})),
        }

        # Run the suite - this will pass validation
        key = ResultKey(date.today(), tags={})
        suite.run(datasources, key)

        print("✅ Validation passed! Parent and child datasets are consistent.")
        print("   Child metric: sum(revenue) with dataset='production'")
        print("   Parent metric: day_over_day(sum(revenue)) with dataset='production'")
        print("   Parent metric: week_over_week(sum(revenue)) with dataset='production'")
    except DQXError as e:
        print(f"Unexpected error: {e}")


def demo_dataset_propagation() -> None:
    """Demonstrate automatic dataset propagation from parent to child."""
    print("\n\n=== Demo 3: Automatic Dataset Propagation ===\n")

    @check(name="Revenue Analysis", datasets=["production"])
    def revenue_analysis(mp: MetricProvider, ctx: Context) -> None:
        # Create base metric (child) WITHOUT dataset
        revenue_sum = mp.sum("revenue")  # No dataset specified

        # Create derived metrics (parents) WITH dataset
        # The child will inherit dataset from parent automatically
        revenue_dod = mp.ext.day_over_day(revenue_sum, dataset="production")
        revenue_wow = mp.ext.week_over_week(revenue_sum, dataset="production")

        # Add assertions
        ctx.assert_that(revenue_sum).where(name="Positive revenue").is_positive()
        ctx.assert_that(revenue_dod).where(name="Daily change check").is_gt(-0.5)

        # Use the unused variable to avoid linting warnings
        _ = revenue_wow

    try:
        # Create a verification suite
        db = InMemoryMetricDB()
        suite = VerificationSuite([revenue_analysis], db, "Revenue Monitoring")

        # Create dummy data source with revenue values
        datasources: dict[str, SqlDataSource] = {
            "production": DuckRelationDataSource.from_arrow(pa.table({"revenue": [1000, 1100]})),
        }

        # Run the suite
        key = ResultKey(date.today(), tags={})
        suite.run(datasources, key)

        print("✅ Validation passed! Child dataset was automatically propagated from parent.")
        print("   Child: sum(revenue) - originally no dataset")
        print("   Parents with dataset='production':")
        print("   - day_over_day(sum(revenue))")
        print("   - week_over_week(sum(revenue))")
        print("   Child inherits dataset='production' from parents")
    except DQXError as e:
        print(f"Unexpected error: {e}")


def demo_multiple_children_validation() -> None:
    """Demonstrate validation with multiple parent-child relationships."""
    print("\n\n=== Demo 4: Multiple Parents with Dataset Issues ===\n")

    @check(name="Metrics Check", datasets=["prod", "staging", "dev"])
    def metrics_check(mp: MetricProvider, ctx: Context) -> None:
        # Base metric (child) with different dataset
        total_sales = mp.sum("sales", dataset="testing")

        # Multiple parents with different datasets - errors expected!
        sales_dod = mp.ext.day_over_day(total_sales, dataset="staging")
        sales_wow = mp.ext.week_over_week(total_sales, dataset="dev")
        sales_stddev = mp.ext.stddev(total_sales, lag=0, n=7, dataset="prod")

        # Add assertions on the parent metrics
        ctx.assert_that(sales_dod).where(name="Daily change positive").is_positive()
        ctx.assert_that(sales_wow).where(name="Weekly change positive").is_positive()
        ctx.assert_that(sales_stddev).where(name="StdDev positive").is_positive()

    try:
        db = InMemoryMetricDB()
        suite = VerificationSuite([metrics_check], db, "Complex Metrics")

        datasources: dict[str, SqlDataSource] = {
            "prod": DuckRelationDataSource.from_arrow(pa.table({"sales": [1000]})),
            "staging": DuckRelationDataSource.from_arrow(pa.table({"sales": [900]})),
            "dev": DuckRelationDataSource.from_arrow(pa.table({"sales": [800]})),
        }

        key = ResultKey(date.today(), tags={})
        suite.run(datasources, key)

    except DQXError as e:
        print(f"❌ Validation Error (as expected):\n{e}")
        print("\nNote: The error reports that the child (base metric) has a different")
        print("dataset than its parents (derived metrics).")


if __name__ == "__main__":
    print("Parent-Child Dataset Validation Demo")
    print("====================================")
    print("This demo shows how DQX validates dataset consistency")
    print("between parent metrics (derived) and their children (base metrics).")

    demo_parent_child_dataset_mismatch()
    demo_valid_parent_child_datasets()
    demo_dataset_propagation()
    demo_multiple_children_validation()

    print("\n\n✨ Key Takeaways:")
    print("1. In DQX, derived metrics (day_over_day, etc.) are parents")
    print("2. Base metrics that they depend on are children")
    print("3. Children must use the same dataset as their parents")
    print("4. Children without datasets inherit from their parents")
    print("5. Dataset mismatches are caught during graph validation")
    print("6. This prevents subtle bugs from mixed dataset usage")
