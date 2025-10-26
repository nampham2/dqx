"""
Demo showing how to access and display analysis reports with symbol mappings.

This example demonstrates:
1. Running a VerificationSuite with multiple datasources
2. Accessing the suite._analysis_reports after execution
3. Displaying reports with symbol names using print_analysis_report()
"""

import datetime
from typing import Any

import pyarrow as pa

from dqx.api import VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.display import print_analysis_report
from dqx.orm.repositories import InMemoryMetricDB


@check(name="Sales Metrics Check")
def check_sales_metrics(mp: Any, ctx: Any) -> None:
    """Define sales-related metrics and assertions."""
    # Define symbols for readability
    total_sales = mp.sum("sales_amount")
    avg_sale = mp.average("sales_amount")
    sales_count = mp.null_count("sales_amount")  # Using null_count as example

    # Create assertions with the symbols
    ctx.assert_that(total_sales).where(name="Total sales positive").is_positive()
    ctx.assert_that(avg_sale).where(name="Average sale reasonable").is_between(10, 1000)
    ctx.assert_that(sales_count).where(name="Has sales records").is_geq(0)


@check(name="Product Metrics Check")
def check_product_metrics(mp: Any, ctx: Any) -> None:
    """Define product-related metrics and assertions."""
    # Define symbols for product metrics
    unique_products = mp.null_count("product_id")  # Using null_count as example
    max_price = mp.maximum("price")
    min_price = mp.minimum("price")

    # Create assertions
    ctx.assert_that(unique_products).where(name="Has multiple products").is_geq(0)
    ctx.assert_that(max_price).where(name="Max price reasonable").is_leq(10000)
    ctx.assert_that(min_price).where(name="Min price positive").is_positive()


def create_sample_data() -> tuple[pa.Table, pa.Table]:
    """Create sample sales data for two different dates."""
    # Day 1 data
    sales_data_day1 = pa.table(
        {
            "sales_amount": [100.5, 200.0, 150.75, 300.25, 175.0],
            "product_id": [1, 2, 1, 3, 2],
            "price": [50.0, 75.0, 50.0, 100.0, 75.0],
        }
    )

    # Day 2 data with different values
    sales_data_day2 = pa.table(
        {
            "sales_amount": [250.0, 180.5, 320.0, 145.75, 290.0],
            "product_id": [1, 3, 2, 4, 1],
            "price": [50.0, 100.0, 75.0, 125.0, 50.0],
        }
    )

    return sales_data_day1, sales_data_day2


def main() -> None:
    print("=== Analysis Report with Symbols Demo ===\n")

    # Create sample data
    sales_day1, sales_day2 = create_sample_data()

    # Create in-memory database and datasources
    db = InMemoryMetricDB()

    # Create datasources for two different days
    ds_day1 = DuckRelationDataSource.from_arrow(sales_day1, "sales")
    ds_day2 = DuckRelationDataSource.from_arrow(sales_day2, "sales")

    # Define checks and create suite
    checks = [check_sales_metrics, check_product_metrics]
    suite = VerificationSuite(checks, db, "Sales Analysis Suite")

    # Run suite for day 1
    print("Running suite for Day 1...")
    key_day1 = ResultKey(datetime.date(2024, 1, 1), {"env": "prod"})
    suite.run([ds_day1], key_day1)

    # Access and display the analysis report
    print("\n1. Analysis Report for Day 1:")
    print("-" * 50)

    # The suite stores analysis reports by datasource name
    if suite._analysis_reports:
        print_analysis_report(suite._analysis_reports)
    else:
        print("No analysis reports available")

    # Show how to access individual reports
    print("\n2. Accessing individual datasource reports:")
    print("-" * 50)
    for ds_name, report in suite._analysis_reports.items():
        print(f"\nDatasource: {ds_name}")
        print(f"Number of metrics: {len(report)}")
        print(f"Symbol mappings: {len(report.symbol_mapping)}")

        # Show a few symbol mappings
        print("\nSample symbol mappings:")
        for i, ((metric_spec, result_key), symbol) in enumerate(report.symbol_mapping.items()):
            if i >= 3:  # Just show first 3
                break
            print(f"  {metric_spec.name} -> {symbol}")

    # Run another suite for day 2 to show multiple executions
    print("\n\n3. Running another suite for Day 2:")
    print("-" * 50)

    suite_day2 = VerificationSuite(checks, db, "Sales Analysis Suite")
    key_day2 = ResultKey(datetime.date(2024, 1, 2), {"env": "prod"})
    suite_day2.run([ds_day2], key_day2)

    print("\nAnalysis Report for Day 2:")
    print_analysis_report(suite_day2._analysis_reports)

    # Demonstrate showing a single report directly
    print("\n\n4. Using AnalysisReport.show() method:")
    print("-" * 50)
    if "sales" in suite._analysis_reports:
        print("\nShowing report for 'sales' datasource:")
        suite._analysis_reports["sales"].show("sales")


if __name__ == "__main__":
    main()
