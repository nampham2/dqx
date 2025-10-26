"""Demo script showcasing metadata functionality in DQX."""

from datetime import date

import pyarrow as pa

from dqx import data
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def main() -> None:
    """Demonstrate metadata functionality."""
    print("=== DQX Metadata Demo ===\n")

    # Create sample data
    sales_data = pa.table(
        {
            "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "product_name": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "quantity": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "category": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
        }
    )

    # Create datasource
    datasource = DuckRelationDataSource.from_arrow(sales_data, "sales")

    # Define quality checks
    @check(name="Basic Sales Checks")
    def basic_checks(mp: MetricProvider, ctx: Context) -> None:
        avg_price = mp.average("price")
        min_price = mp.minimum("price")
        max_price = mp.maximum("price")
        total_quantity = mp.sum("quantity")

        ctx.assert_that(avg_price).where(name="Average price check").is_between(40, 60)
        ctx.assert_that(min_price).where(name="Min price check").is_positive()
        ctx.assert_that(max_price).where(name="Max price check").is_lt(150)
        ctx.assert_that(total_quantity).where(name="Total quantity check").is_gt(200)

    @check(name="Category Analysis")
    def category_checks(mp: MetricProvider, ctx: Context) -> None:
        category_x_count = mp.count_values("category", "X")
        category_y_count = mp.count_values("category", "Y")

        ctx.assert_that(category_x_count).where(name="Category X count").is_eq(5)
        ctx.assert_that(category_y_count).where(name="Category Y count").is_eq(5)

    # Initialize database
    db = InMemoryMetricDB()

    # Run first verification suite
    print("1. Running Sales Quality Suite...")
    suite1 = VerificationSuite([basic_checks, category_checks], db, "Sales Quality Suite")
    key1 = ResultKey(date.today(), {"env": "prod", "region": "us-west"})
    suite1.run([datasource], key1)
    exec_id1 = suite1.execution_id
    print(f"   Execution ID: {exec_id1}")

    # Run second verification suite with different tags
    print("\n2. Running Sales Quality Suite with different tags...")
    suite2 = VerificationSuite([basic_checks], db, "Sales Quality Suite")
    key2 = ResultKey(date.today(), {"env": "staging", "region": "eu-central"})
    suite2.run([datasource], key2)
    exec_id2 = suite2.execution_id
    print(f"   Execution ID: {exec_id2}")

    # Demonstrate metadata retrieval
    print("\n3. Retrieving metrics by execution ID...")

    # Get metrics from first execution
    metrics1 = data.metrics_by_execution_id(db, exec_id1)
    print(f"\n   Metrics from execution {exec_id1[:8]}... (prod/us-west):")
    print(f"   Found {len(metrics1)} metrics")
    for metric in metrics1[:3]:  # Show first 3
        print(f"   - {metric.spec.name}: {metric.value:.2f}")
        print(f"     Dataset: {metric.dataset}")
        print(f"     Tags: {metric.key.tags}")
        if metric.metadata and metric.metadata.execution_id:
            print(
                f"     Metadata: execution_id={metric.metadata.execution_id[:8]}..., ttl_hours={metric.metadata.ttl_hours}"
            )

    # Get metrics from second execution
    metrics2 = data.metrics_by_execution_id(db, exec_id2)
    print(f"\n   Metrics from execution {exec_id2[:8]}... (staging/eu-central):")
    print(f"   Found {len(metrics2)} metrics")

    # Demonstrate metadata isolation
    print("\n4. Verifying metadata isolation...")

    # Check that execution IDs are properly isolated
    exec_ids_1 = {m.metadata.execution_id for m in metrics1 if m.metadata}
    exec_ids_2 = {m.metadata.execution_id for m in metrics2 if m.metadata}

    print(f"   Execution IDs in first suite: {len(exec_ids_1)} unique")
    print(f"   Execution IDs in second suite: {len(exec_ids_2)} unique")
    print(f"   No overlap: {len(exec_ids_1 & exec_ids_2) == 0}")

    # Verify no __execution_id in tags
    print("\n5. Verifying clean tags (no __execution_id injection)...")
    from dqx.orm.repositories import Metric as DBMetric

    all_metrics = db.search(DBMetric.yyyy_mm_dd == date.today())
    has_exec_id_in_tags = any("__execution_id" in m.key.tags for m in all_metrics)
    print(f"   Found __execution_id in tags: {has_exec_id_in_tags}")
    print(
        f"   All metrics have metadata.execution_id: {all(m.metadata and m.metadata.execution_id for m in all_metrics)}"
    )

    # Show how to query metrics by different criteria
    print("\n6. Alternative query methods...")

    # Query by date and tags
    prod_metrics = [m for m in all_metrics if m.key.tags.get("env") == "prod"]
    staging_metrics = [m for m in all_metrics if m.key.tags.get("env") == "staging"]

    print(f"   Production metrics: {len(prod_metrics)}")
    print(f"   Staging metrics: {len(staging_metrics)}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
