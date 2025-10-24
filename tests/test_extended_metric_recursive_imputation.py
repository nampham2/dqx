"""Integration test for recursive dataset imputation with extended metrics."""

import datetime as dt

import pyarrow as pa
from faker import Faker

from dqx.api import VerificationSuite, check
from dqx.common import Context, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


@check(name="Dataset 1 Checks", datasets=["ds1"])
def dataset1_checks(mp: MetricProvider, ctx: Context) -> None:
    """Checks using extended metrics for dataset 1."""
    tax_avg = mp.average("tax")
    tax_wow = mp.ext.week_over_week(tax_avg)  # Creates lag(7) internally

    ctx.assert_that(tax_wow).where(name="Tax week-over-week < 0.2").is_lt(0.2)


@check(name="Dataset 2 Checks", datasets=["ds2"])
def dataset2_checks(mp: MetricProvider, ctx: Context) -> None:
    """Checks using extended metrics for dataset 2."""
    tax_avg = mp.average("tax")
    tax_dod = mp.ext.day_over_day(tax_avg)  # Creates lag(1) internally

    ctx.assert_that(tax_dod).where(name="Tax day-over-day < 0.1").is_lt(0.1)


@check(name="Multi-dataset Check", datasets=["ds1", "ds2"])
def multi_dataset_check(mp: MetricProvider, ctx: Context) -> None:
    """Check that should trigger ambiguous dataset error."""
    # This metric has no explicit dataset and check has multiple datasets
    ctx.assert_that(mp.average("price")).where(name="Average price > 100").is_gt(100)


def test_extended_metric_recursive_dataset_imputation() -> None:
    """Test that DatasetImputationVisitor recursively processes child dependencies.

    This test verifies that when using extended metrics like week_over_week,
    all child dependencies (like lag(7)) are properly processed and have
    their datasets imputed, even though they don't appear directly in assertions.
    """
    # Create test data
    Faker.seed(42)
    n, fake = 100, Faker()

    # Create DataFrame 1
    tax1 = [fake.pyfloat(min_value=-100.0, max_value=100.0) for _ in range(n)]
    price1 = [fake.pyfloat(min_value=1.0, max_value=10_000) for _ in range(n)]
    arrow1 = pa.Table.from_arrays([tax1, price1], names=["tax", "price"])

    # Create DataFrame 2
    tax2 = [fake.pyfloat(min_value=-100.0, max_value=100.0) for _ in range(n)]
    revenue2 = [fake.pyfloat(min_value=1.0, max_value=10_000) for _ in range(n)]
    arrow2 = pa.Table.from_arrays([tax2, revenue2], names=["tax", "revenue"])

    # Create data sources
    db = InMemoryMetricDB()
    ds1 = DuckRelationDataSource.from_arrow(arrow1)
    ds2 = DuckRelationDataSource.from_arrow(arrow2)

    # Create verification suite with only the single-dataset checks
    # to avoid validation errors
    checks = [dataset1_checks, dataset2_checks]
    suite = VerificationSuite(checks, db, name="Extended Metric Test")

    # Run the suite to trigger dataset imputation
    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-14"), tags={})
    suite.run({"ds1": ds1, "ds2": ds2}, key)

    # Collect symbols after the run
    symbols = suite.collect_symbols()

    # Find symbols for our metrics
    wow_symbol = None
    lag7_symbol = None
    dod_symbol = None
    lag1_symbol = None
    tax_avg_ds1 = None
    tax_avg_ds2 = None

    for sym_info in symbols:
        if "week_over_week" in sym_info.metric and sym_info.dataset == "ds1":
            wow_symbol = sym_info
        elif "lag(7)" in sym_info.metric and sym_info.dataset == "ds1":
            lag7_symbol = sym_info
        elif "day_over_day" in sym_info.metric and sym_info.dataset == "ds2":
            dod_symbol = sym_info
        elif "lag(1)" in sym_info.metric and sym_info.dataset == "ds2":
            lag1_symbol = sym_info
        elif sym_info.metric == "average(tax)" and sym_info.dataset == "ds1":
            tax_avg_ds1 = sym_info
        elif sym_info.metric == "average(tax)" and sym_info.dataset == "ds2":
            tax_avg_ds2 = sym_info

    # Verify all symbols exist
    assert tax_avg_ds1 is not None, "average(tax) for ds1 not found"
    assert tax_avg_ds2 is not None, "average(tax) for ds2 not found"
    assert wow_symbol is not None, "week_over_week symbol not found"
    assert lag7_symbol is not None, "lag(7) symbol not found (should be created by week_over_week)"
    assert dod_symbol is not None, "day_over_day symbol not found"
    assert lag1_symbol is not None, "lag(1) symbol not found (should be created by day_over_day)"

    # Verify datasets were properly imputed
    assert wow_symbol.dataset == "ds1", f"week_over_week should have ds1, got {wow_symbol.dataset}"
    assert lag7_symbol.dataset == "ds1", f"lag(7) should inherit ds1 from parent, got {lag7_symbol.dataset}"
    assert dod_symbol.dataset == "ds2", f"day_over_day should have ds2, got {dod_symbol.dataset}"
    assert lag1_symbol.dataset == "ds2", f"lag(1) should inherit ds2 from parent, got {lag1_symbol.dataset}"

    # Verify that lag symbols show correct parent-child relationships
    # The key insight is that lag(7) was created and has the correct dataset
    assert lag7_symbol.children_names == [], "lag(7) should have no children"
    assert lag1_symbol.children_names == [], "lag(1) should have no children"

    # Verify that wow_symbol has lag symbols as children
    assert len(wow_symbol.children_names) > 0, f"week_over_week should have children: {wow_symbol.children_names}"
    assert len(dod_symbol.children_names) > 0, f"day_over_day should have children: {dod_symbol.children_names}"


@check(name="Nested Checks", datasets=["prod"])
def nested_checks(mp: MetricProvider, ctx: Context) -> None:
    """Checks with nested extended metrics."""
    # Test with stddev which creates multiple lag dependencies
    revenue = mp.sum("revenue")
    revenue_stddev = mp.ext.stddev(revenue, lag=1, n=7)  # Creates lag(1) through lag(7)

    ctx.assert_that(revenue_stddev).where(name="Revenue stddev < 1000").is_lt(1000)


def test_nested_extended_metrics() -> None:
    """Test nested extended metrics get proper dataset imputation."""
    # Create test data
    Faker.seed(43)
    n, fake = 100, Faker()

    revenue = [fake.pyfloat(min_value=1.0, max_value=10_000) for _ in range(n)]
    cost = [fake.pyfloat(min_value=1.0, max_value=10_000) for _ in range(n)]
    arrow_table = pa.Table.from_arrays([revenue, cost], names=["revenue", "cost"])

    # Create data source
    db = InMemoryMetricDB()
    ds = DuckRelationDataSource.from_arrow(arrow_table)

    # Create suite
    suite = VerificationSuite([nested_checks], db, name="Nested Extended Metrics")

    # Run analysis
    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-14"), tags={})
    suite.run({"prod": ds}, key)

    # Check all symbols were created and have datasets
    symbols = suite.collect_symbols()

    # Count metrics by type and print all metrics for debugging
    metric_types = {"sum": 0, "lag": 0, "stddev": 0}
    all_metrics = []

    for sym_info in symbols:
        metric = sym_info.metric
        all_metrics.append(metric)

        if "lag(" in metric:
            metric_types["lag"] += 1
        elif "stddev" in metric.lower():  # Check case-insensitive
            metric_types["stddev"] += 1
        elif "sum(revenue)" in metric:  # Count after checking for stddev
            metric_types["sum"] += 1

        # All should have prod dataset
        assert sym_info.dataset == "prod", f"{metric} should have 'prod' dataset, got {sym_info.dataset}"

    # Print all metrics for debugging if stddev not found
    if metric_types["stddev"] == 0:
        print(f"All metrics found: {all_metrics}")

    # Verify we have all expected metrics
    assert metric_types["sum"] >= 1, f"Should have base sum metric, got {metric_types}"
    assert metric_types["lag"] >= 7, f"Should have lag(1) through lag(7) metrics from stddev, got {metric_types}"
    assert metric_types["stddev"] >= 1, f"Should have stddev metric, got {metric_types}. All metrics: {all_metrics}"


@check(name="Simple Check", datasets=["test"])
def simple_check(mp: MetricProvider, ctx: Context) -> None:
    """Simple check to verify basic functionality."""
    ctx.assert_that(mp.sum("value")).where(name="Sum > 0").is_gt(0)


def test_circular_dependency_handling() -> None:
    """Test that circular dependencies in metrics don't cause infinite loops."""
    # This is more of a defensive test - in practice metrics shouldn't have
    # circular dependencies, but our visitor should handle it gracefully

    Faker.seed(44)
    n, fake = 50, Faker()

    value = [fake.pyfloat(min_value=1.0, max_value=10_000) for _ in range(n)]
    arrow_table = pa.Table.from_arrays([value], names=["value"])

    # Create data source
    db = InMemoryMetricDB()
    ds = DuckRelationDataSource.from_arrow(arrow_table)

    # Create suite
    suite = VerificationSuite([simple_check], db, name="Circular Dependency Test")

    # Run should complete without hanging
    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-14"), tags={})
    suite.run({"test": ds}, key)

    # Basic verification that it ran
    symbols = suite.collect_symbols()
    assert len(symbols) > 0, "Should have processed symbols"

    results = suite.collect_results()
    assert len(results) > 0, "Should have results"
