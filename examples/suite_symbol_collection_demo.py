"""Demonstration of VerificationSuite.collect_symbols() feature.

This example shows practical use cases for symbol collection:
1. Basic symbol collection from a verification suite
2. Debugging failed assertions using symbol values
3. Cross-dataset metric analysis
4. Symbol value statistics and trends
5. Integration with result collection for comprehensive reports
"""

from datetime import date
from typing import Any

import pyarrow as pa
from returns.result import Failure, Success

from dqx.api import VerificationSuite, check
from dqx.common import ResultKey, SqlDataSource
from dqx.extensions.duckds import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


# Define checks for our data quality suite
@check(name="Price Validation", datasets=["orders"])
def validate_prices(mp: MetricProvider, ctx: Any) -> None:
    """Validate that prices are within expected ranges."""
    avg_price = mp.average("price")
    min_price = mp.minimum("price")
    max_price = mp.maximum("price")

    ctx.assert_that(avg_price).where(name="Average price above minimum", severity="P1").is_gt(10)
    ctx.assert_that(avg_price).where(name="Average price below maximum", severity="P1").is_lt(1000)

    ctx.assert_that(min_price).where(name="No negative prices", severity="P0").is_geq(0)

    ctx.assert_that(max_price).where(name="Maximum price is not extreme", severity="P2").is_lt(10000)


@check(name="Quantity Validation", datasets=["orders"])
def validate_quantities(mp: MetricProvider, ctx: Any) -> None:
    """Validate order quantities."""
    total_quantity = mp.sum("quantity")
    avg_quantity = mp.average("quantity")

    ctx.assert_that(total_quantity).where(name="Total quantity is positive", severity="P0").is_positive()

    ctx.assert_that(avg_quantity).where(name="Average quantity is positive", severity="P1").is_gt(0)
    ctx.assert_that(avg_quantity).where(name="Average quantity is not excessive", severity="P1").is_lt(100)


@check(name="Customer Metrics", datasets=["customers"])
def validate_customers(mp: MetricProvider, ctx: Any) -> None:
    """Validate customer data quality."""
    customer_count = mp.num_rows()
    avg_age = mp.average("age")

    ctx.assert_that(customer_count).where(name="Sufficient customer base", severity="P1").is_gt(100)

    ctx.assert_that(avg_age).where(name="Average age above minimum", severity="P2").is_gt(18)
    ctx.assert_that(avg_age).where(name="Average age below maximum", severity="P2").is_lt(80)


@check(name="Cross-Dataset Validation")
def cross_dataset_checks(mp: MetricProvider, ctx: Any) -> None:
    """Validate relationships across datasets."""
    order_count = mp.num_rows(dataset="orders")
    customer_count = mp.num_rows(dataset="customers")
    orders_per_customer = order_count / customer_count

    ctx.assert_that(orders_per_customer).where(name="Orders per customer above minimum", severity="P1").is_gt(0.5)
    ctx.assert_that(orders_per_customer).where(name="Orders per customer below maximum", severity="P1").is_lt(10)


def create_sample_data() -> dict[str, pa.Table]:
    """Create sample data for demonstration."""
    # Orders dataset with some edge cases
    orders_data = pa.table(
        {
            "order_id": list(range(1, 501)),
            "customer_id": [i % 120 + 1 for i in range(500)],
            "price": [
                150.0 if i < 400 else 9500.0  # Some extreme prices
                for i in range(500)
            ],
            "quantity": [
                3 if i < 450 else 150  # Some extreme quantities
                for i in range(500)
            ],
        }
    )

    # Customers dataset
    customers_data = pa.table(
        {
            "customer_id": list(range(1, 121)),
            "age": [25 + (i % 40) for i in range(120)],
            "signup_date": ["2024-01-01"] * 120,
        }
    )

    return {"orders": orders_data, "customers": customers_data}


def main() -> None:
    """Demonstrate various use cases for symbol collection."""
    # Set up
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=date(2025, 1, 13), tags={"env": "prod", "region": "us-east"})

    # Create verification suite
    suite = VerificationSuite(
        checks=[validate_prices, validate_quantities, validate_customers, cross_dataset_checks],
        db=db,
        name="E-commerce Data Quality Suite",
    )

    # Create data sources
    tables = create_sample_data()
    datasources: dict[str, SqlDataSource] = {
        name: DuckRelationDataSource.from_arrow(table) for name, table in tables.items()
    }

    # Run the suite
    suite.run(datasources, key)

    print("=== VerificationSuite Symbol Collection Demo ===\n")

    # 1. Basic Symbol Collection
    print("1. Basic Symbol Collection")
    print("-" * 50)
    symbols = suite.collect_symbols()
    print(f"Total symbols registered: {len(symbols)}")
    print("\nAll symbols:")
    for symbol in symbols:
        value_str = (
            f"{symbol.value.unwrap():.2f}" if isinstance(symbol.value, Success) else f"FAILED: {symbol.value.failure()}"
        )
        print(f"  {symbol.name}: {symbol.metric} = {value_str}")
    print()

    # 2. Debugging Failed Assertions
    print("2. Debugging Failed Assertions")
    print("-" * 50)
    results = suite.collect_results()
    failed_results = [r for r in results if r.status == "FAILURE"]

    if failed_results:
        print(f"Found {len(failed_results)} failed assertions:\n")

        for result in failed_results:
            print(f"❌ {result.check} / {result.assertion}")
            print(f"   Expression: {result.expression}")

            # Get symbols used in this assertion's expression
            if isinstance(result.metric, Failure):
                failures = result.metric.failure()
                for failure in failures:
                    print(f"   Error: {failure.error_message}")
                    print("   Symbol values:")
                    for sym in failure.symbols:
                        value_str = f"{sym.value.unwrap():.2f}" if isinstance(sym.value, Success) else "FAILED"
                        print(f"     - {sym.name} ({sym.metric}): {value_str}")
            print()

    # 3. Cross-Dataset Analysis
    print("3. Cross-Dataset Analysis")
    print("-" * 50)

    # Group symbols by dataset
    by_dataset: dict[str | None, list] = {}
    for symbol in symbols:
        dataset = symbol.dataset
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(symbol)

    for dataset, dataset_symbols in sorted(by_dataset.items()):
        dataset_name = dataset or "cross-dataset"
        print(f"\nDataset: {dataset_name}")
        print(f"  Metrics computed: {len(dataset_symbols)}")

        # Show value statistics for successful metrics
        successful_values = [s.value.unwrap() for s in dataset_symbols if isinstance(s.value, Success)]

        if successful_values:
            print(f"  Value range: {min(successful_values):.2f} - {max(successful_values):.2f}")

    # 4. Symbol Value Analysis
    print("\n4. Symbol Value Analysis")
    print("-" * 50)

    # Categorize symbols by metric type
    metric_types: dict[str, list] = {}
    for symbol in symbols:
        # Extract metric type from the metric string
        metric_str = str(symbol.metric)
        metric_type = metric_str.split("(")[0] if "(" in metric_str else "unknown"

        if metric_type not in metric_types:
            metric_types[metric_type] = []
        metric_types[metric_type].append(symbol)

    print("Metrics by type:")
    for metric_type, type_symbols in sorted(metric_types.items()):
        success_count = sum(1 for s in type_symbols if isinstance(s.value, Success))
        print(f"  {metric_type}: {len(type_symbols)} total, {success_count} successful")

    # 5. Integration with Result Collection
    print("\n5. Comprehensive Analysis Report")
    print("-" * 50)

    # Create a summary report
    total_assertions = len(results)
    passed_assertions = sum(1 for r in results if r.status == "OK")
    total_symbols = len(symbols)
    successful_symbols = sum(1 for s in symbols if isinstance(s.value, Success))

    print(f"Suite: {suite._name}")
    print(f"Date: {key.yyyy_mm_dd}")
    print(f"Tags: {key.tags}")
    print()
    print(
        f"Assertions: {passed_assertions}/{total_assertions} passed ({passed_assertions / total_assertions * 100:.1f}%)"
    )
    print(
        f"Symbols: {successful_symbols}/{total_symbols} computed successfully ({successful_symbols / total_symbols * 100:.1f}%)"
    )

    # Show critical failures (P0)
    p0_failures = [r for r in failed_results if r.severity == "P0"]
    if p0_failures:
        print(f"\n⚠️  Critical (P0) Failures: {len(p0_failures)}")
        for failure in p0_failures:
            print(f"   - {failure.check}: {failure.assertion}")

    # Symbol coverage by check
    print("\nSymbol Usage by Check:")
    check_names = {r.check for r in results}
    for check_name in sorted(check_names):
        check_results = [r for r in results if r.check == check_name]
        # Count unique symbols referenced in this check's assertions
        print(f"  {check_name}: {len(check_results)} assertions")


if __name__ == "__main__":
    main()
