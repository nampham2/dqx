"""Demo of print_metrics_by_execution_id display function."""

from datetime import date

import pyarrow as pa

from dqx import data, display
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def main() -> None:
    """Run demo of metrics_by_execution_id display."""

    # Create a check that uses multiple metrics
    @check(name="Product Metrics Check")
    def product_metrics(mp: MetricProvider, ctx: Context) -> None:
        avg_price = mp.average("price")
        min_price = mp.minimum("price")
        max_price = mp.maximum("price")
        row_count = mp.num_rows()

        ctx.assert_that(avg_price).where(name="Avg price check").is_gt(15)
        ctx.assert_that(min_price).where(name="Min price check").is_positive()
        ctx.assert_that(max_price).where(name="Max price check").is_lt(100)
        ctx.assert_that(row_count).where(name="Row count check").is_eq(3)

    # Create test data
    table = pa.table(
        {
            "product_id": [1, 2, 3],
            "product_name": ["Widget A", "Widget B", "Widget C"],
            "price": [19.99, 29.99, 39.99],
            "quantity": [100, 150, 200],
        }
    )

    # Create datasource
    datasource = DuckRelationDataSource.from_arrow(table, "products")

    # Set up database and suite
    db = InMemoryMetricDB()
    suite = VerificationSuite([product_metrics], db, "Product Quality Suite")

    # Run with specific date and tags
    key = ResultKey(date(2024, 1, 26), {"env": "prod", "region": "us-east-1"})
    suite.run([datasource], key)

    # Get the execution ID
    execution_id = suite.execution_id
    print(f"Suite execution ID: {execution_id}")
    print()

    # Retrieve metrics by execution ID
    metrics = data.metrics_by_execution_id(db, execution_id)

    # Display metrics using the new function
    display.print_metrics_by_execution_id(metrics, execution_id)

    print("\n" + "=" * 80 + "\n")

    # Also demonstrate with more metrics and different tags
    @check(name="Inventory Analysis Check")
    def inventory_check(mp: MetricProvider, ctx: Context) -> None:
        avg_quantity = mp.average("quantity")
        total_quantity = mp.sum("quantity")

        # Create additional metrics to show variety in the output
        _ = mp.minimum("quantity")  # Will show in metrics table
        _ = mp.maximum("quantity")  # Will show in metrics table
        _ = mp.variance("price")  # Will show in metrics table

        ctx.assert_that(avg_quantity).where(name="Avg quantity check").is_gt(100)
        ctx.assert_that(total_quantity).where(name="Total quantity check").is_eq(450)

    # Create another suite with different tags
    suite2 = VerificationSuite([inventory_check], db, "Inventory Analysis Suite")

    # Run with different tags to show variety
    key2 = ResultKey(date(2024, 1, 26), {"env": "staging", "region": "eu-west-1", "version": "v1.2.3"})
    suite2.run([datasource], key2)

    # Get metrics for the second execution
    execution_id2 = suite2.execution_id
    print(f"Inventory suite execution ID: {execution_id2}")
    print()

    metrics2 = data.metrics_by_execution_id(db, execution_id2)

    # Display metrics with different tags
    display.print_metrics_by_execution_id(metrics2, execution_id2)


if __name__ == "__main__":
    main()
