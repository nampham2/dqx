"""Integration tests for execution ID functionality."""

from datetime import date

import pyarrow as pa

from dqx import data
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_execution_id_full_flow() -> None:
    """Test the complete execution ID flow from suite creation to retrieval."""

    # Create checks with various metrics and extended operations
    @check(name="Basic Metrics Check")
    def basic_check(mp: MetricProvider, ctx: Context) -> None:
        # Basic metrics
        avg_price = mp.average("price")
        min_price = mp.minimum("price")
        max_price = mp.maximum("price")
        num_rows = mp.num_rows()

        # Assertions
        ctx.assert_that(avg_price).where(name="Average price check").is_gt(10)
        ctx.assert_that(min_price).where(name="Min price check").is_positive()
        ctx.assert_that(max_price).where(name="Max price check").is_lt(100)
        ctx.assert_that(num_rows).where(name="Row count check").is_eq(5)

    @check(name="Extended Metrics Check")
    def extended_check(mp: MetricProvider, ctx: Context) -> None:
        # Extended metrics
        avg_tax = mp.average("tax")
        dod_tax = mp.ext.day_over_day(avg_tax)

        ctx.assert_that(dod_tax).where(name="Tax day-over-day check").noop()

    # Create test data for multiple days
    day1_data = pa.table({"price": [10.0, 20.0, 30.0, 40.0, 50.0], "tax": [1.0, 2.0, 3.0, 4.0, 5.0]})

    day2_data = pa.table({"price": [15.0, 25.0, 35.0, 45.0, 55.0], "tax": [1.5, 2.5, 3.5, 4.5, 5.5]})

    # Create datasources
    ds1 = DuckRelationDataSource.from_arrow(day1_data, "sales_data")
    ds2 = DuckRelationDataSource.from_arrow(day2_data, "sales_data")

    # Initialize database
    db = InMemoryMetricDB()

    # Run suite for day 1
    suite1 = VerificationSuite([basic_check, extended_check], db, "Sales DQ Suite")
    key1 = ResultKey(date(2024, 1, 1), {"env": "prod", "region": "us-east"})
    suite1.run([ds1], key1)
    exec_id1 = suite1.execution_id

    # Run suite for day 2
    suite2 = VerificationSuite([basic_check, extended_check], db, "Sales DQ Suite")
    key2 = ResultKey(date(2024, 1, 2), {"env": "prod", "region": "us-east"})
    suite2.run([ds2], key2)
    exec_id2 = suite2.execution_id

    # Verify execution IDs are unique
    assert exec_id1 != exec_id2

    # Retrieve metrics for first execution
    metrics1 = data.metrics_by_execution_id(db, exec_id1)

    # Verify we got the expected number of metrics
    # Basic: 4 metrics (avg, min, max, num_rows) + Extended: 2 metrics (avg_tax for both days)
    assert len(metrics1) >= 5

    # Verify all metrics have the correct execution_id
    for metric in metrics1:
        assert metric.key.tags["__execution_id"] == exec_id1
        assert metric.key.tags["env"] == "prod"
        assert metric.key.tags["region"] == "us-east"

    # Verify metric types
    metric_types = {m.spec.metric_type for m in metrics1}
    assert "Average" in metric_types
    assert "Minimum" in metric_types
    assert "Maximum" in metric_types
    assert "NumRows" in metric_types

    # Retrieve metrics for second execution
    metrics2 = data.metrics_by_execution_id(db, exec_id2)

    # Should have metrics including the lagged ones
    assert len(metrics2) >= 5

    # Verify dates in second execution
    dates2 = {m.key.yyyy_mm_dd for m in metrics2}
    assert date(2024, 1, 2) in dates2  # Current date
    assert date(2024, 1, 1) in dates2  # Lag date for day_over_day

    # Verify no overlap between executions
    exec_ids_1 = {m.key.tags["__execution_id"] for m in metrics1}
    exec_ids_2 = {m.key.tags["__execution_id"] for m in metrics2}
    assert exec_ids_1 == {exec_id1}
    assert exec_ids_2 == {exec_id2}


def test_multiple_datasets_single_execution() -> None:
    """Test execution ID with multiple datasets in a single run."""

    @check(name="Multi-Dataset Check", datasets=["orders", "products"])
    def multi_check(mp: MetricProvider, ctx: Context) -> None:
        # Metrics from orders dataset
        avg_amount = mp.average("amount", dataset="orders")

        # Metrics from products dataset
        avg_price = mp.average("price", dataset="products")

        ctx.assert_that(avg_amount).where(name="Order amount check").is_positive()
        ctx.assert_that(avg_price).where(name="Product price check").is_positive()

    # Create test data
    orders_data = pa.table({"amount": [100.0, 200.0, 300.0]})
    products_data = pa.table({"price": [10.0, 20.0, 30.0]})

    orders_ds = DuckRelationDataSource.from_arrow(orders_data, "orders")
    products_ds = DuckRelationDataSource.from_arrow(products_data, "products")

    # Run suite
    db = InMemoryMetricDB()
    suite = VerificationSuite([multi_check], db, "Multi-Dataset Suite")
    key = ResultKey(date.today(), {"env": "test"})
    suite.run([orders_ds, products_ds], key)

    # Retrieve metrics
    metrics = data.metrics_by_execution_id(db, suite.execution_id)

    # Should have metrics from both datasets
    datasets = {m.dataset for m in metrics}
    assert "orders" in datasets
    assert "products" in datasets

    # All should have the same execution_id
    for metric in metrics:
        assert metric.key.tags["__execution_id"] == suite.execution_id


def test_execution_id_persistence_across_queries() -> None:
    """Test that execution ID remains consistent across different query methods."""

    @check(name="Persistence Check")
    def simple_check(mp: MetricProvider, ctx: Context) -> None:
        avg = mp.average("value")
        ctx.assert_that(avg).where(name="Value check").is_positive()

    # Create test data
    table = pa.table({"value": [1.0, 2.0, 3.0]})
    datasource = DuckRelationDataSource.from_arrow(table, "test_data")

    # Run suite
    db = InMemoryMetricDB()
    suite = VerificationSuite([simple_check], db, "Test Suite")
    key = ResultKey(date.today(), {"env": "test"})
    suite.run([datasource], key)
    exec_id = suite.execution_id

    # Query by execution ID
    metrics_by_exec = data.metrics_by_execution_id(db, exec_id)

    # Query by date and dataset
    from dqx.orm.repositories import Metric as DBMetric

    metrics_by_date = db.search(DBMetric.yyyy_mm_dd == date.today(), DBMetric.dataset == "test_data")

    # Filter to only metrics with our execution_id
    metrics_by_date_filtered = [
        m for m in metrics_by_date if "__execution_id" in m.key.tags and m.key.tags["__execution_id"] == exec_id
    ]

    # Should get the same metrics
    assert len(metrics_by_exec) == len(metrics_by_date_filtered)

    # Verify they're the same metrics
    exec_metric_ids = {m.metric_id for m in metrics_by_exec}
    date_metric_ids = {m.metric_id for m in metrics_by_date_filtered}
    assert exec_metric_ids == date_metric_ids
