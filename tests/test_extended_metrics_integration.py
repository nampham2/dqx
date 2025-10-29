"""Integration tests for complex extended metric dependencies."""

from datetime import date

import pyarrow as pa
from returns.maybe import Nothing

from dqx import specs
from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_stddev_of_dod_creates_dependencies() -> None:
    """Test that stddev(dod(average(tax))) creates all necessary dependencies."""
    # Setup
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec-123")

    # Create the complex metric
    avg_tax = provider.average("tax")
    dod_avg_tax = provider.ext.day_over_day(avg_tax)
    stddev_dod_avg_tax = provider.ext.stddev(dod_avg_tax, offset=0, n=7)

    # Verify the stddev metric has correct dependencies
    stddev_metric = provider.get_symbol(stddev_dod_avg_tax)
    assert len(stddev_metric.required_metrics) == 7  # 7 days of DoD values

    # Create test data for 10 days
    tables = []
    for day in range(1, 11):
        table = pa.table({"tax": [100.0 + day, 200.0 + day, 300.0 + day], "date": [f"2024-01-{day:02d}"] * 3})
        tables.append((date(2024, 1, day), table))

    # Create check
    @check(name="Tax DoD Stddev Check")
    def tax_dod_stddev_check(mp: MetricProvider, ctx: Context) -> None:
        avg = mp.average("tax")
        dod = mp.ext.day_over_day(avg)
        std = mp.ext.stddev(dod, offset=0, n=7)
        std = mp.ext.stddev(dod, offset=0, n=7)
        ctx.assert_that(std).where(name="Collect stddev").noop()

    # Run suite for each day
    for key_date, table_data in tables:
        suite = VerificationSuite([tax_dod_stddev_check], db, "Test Suite")
        datasource = DuckRelationDataSource.from_arrow(table_data, "tax_data")
        key = ResultKey(yyyy_mm_dd=key_date, tags={"test": "stddev_dod"})
        suite.run([datasource], key)

    # Verify all required metrics exist in DB
    # Should have average(tax) for days 1-9 (to compute DoD for days 2-9)
    for day in range(1, 10):
        key_day = ResultKey(yyyy_mm_dd=date(2024, 1, day), tags={"test": "stddev_dod"})
        avg_metric = db.get(key_day, specs.Average("tax"))
        assert avg_metric != Nothing, f"Missing average(tax) for day {day}"

    # Verify DoD metrics exist for days 2-9
    for day in range(2, 10):
        key_day = ResultKey(yyyy_mm_dd=date(2024, 1, day), tags={"test": "stddev_dod"})
        dod_spec = specs.DayOverDay.from_base_spec(specs.Average("tax"))
        dod_metric = db.get(key_day, dod_spec)
        assert dod_metric != Nothing, f"Missing dod(average(tax)) for day {day}"

    # Verify the final stddev metric exists
    final_key = ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={"test": "stddev_dod"})
    stddev_spec = specs.Stddev.from_base_spec(specs.DayOverDay.from_base_spec(specs.Average("tax")), offset=0, n=7)
    final_metric = db.get(final_key, stddev_spec)
    assert final_metric != Nothing


def test_nested_extended_metrics_combinations() -> None:
    """Test various combinations of nested extended metrics."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec-124")

    # Test 1: WoW of DoD
    avg = provider.average("revenue")
    dod = provider.ext.day_over_day(avg)
    wow_dod = provider.ext.week_over_week(dod)

    wow_metric = provider.get_symbol(wow_dod)
    assert len(wow_metric.required_metrics) == 2  # DoD at lag 0 and lag 7

    # Test 2: DoD of WoW
    sum_metric = provider.sum("cost")
    wow = provider.ext.week_over_week(sum_metric)
    dod_wow = provider.ext.day_over_day(wow)

    dod_metric = provider.get_symbol(dod_wow)
    assert len(dod_metric.required_metrics) == 2  # WoW at lag 0 and lag 1

    # Test 3: Stddev of WoW
    min_metric = provider.minimum("price")
    wow_min = provider.ext.week_over_week(min_metric)
    stddev_wow = provider.ext.stddev(wow_min, offset=0, n=5)

    stddev_metric = provider.get_symbol(stddev_wow)
    assert len(stddev_metric.required_metrics) == 5  # 5 days of WoW values


def test_extended_metric_dependency_creation_bug() -> None:
    """Test that shows the bug where extended metrics can't create extended dependencies."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec-125")

    # Create the complex metric chain
    avg = provider.average("tax")
    dod = provider.ext.day_over_day(avg)
    stddev = provider.ext.stddev(dod, offset=0, n=7)

    # Let's actually evaluate the metric to trigger the bug
    test_date = date(2024, 1, 10)
    table = pa.table({"tax": [100.0, 200.0, 300.0], "date": ["2024-01-10"] * 3})

    # Create a check that uses the complex metric
    @check(name="Test Complex Metric")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        avg = mp.average("tax")
        dod = mp.ext.day_over_day(avg)
        std = mp.ext.stddev(dod, offset=0, n=7)
        ctx.assert_that(std).where(name="Collect stddev").noop()

    # Try to run the suite - this should fail if the bug exists
    suite = VerificationSuite([test_check], db, "Test Suite")
    datasource = DuckRelationDataSource.from_arrow(table, "tax_data")
    key = ResultKey(yyyy_mm_dd=test_date, tags={"test": "bug"})

    # The bug would manifest here when the system tries to evaluate
    # the stddev's dependencies (which are DayOverDay metrics)
    # but they were created incorrectly by provider.metric()
    suite.run([datasource], key)

    # If the bug was present, the execution would have failed
    # Let's verify the metrics were created correctly by checking the database

    # Check that the stddev metric was computed
    stddev_spec = specs.Stddev.from_base_spec(specs.DayOverDay.from_base_spec(specs.Average("tax")), offset=0, n=7)
    stddev_result = db.get(key, stddev_spec)

    # The bug is fixed if we can successfully get a result
    assert stddev_result != Nothing, "Stddev metric was not computed - bug still present"

    # Additionally verify the dependencies are correct
    stddev_metric = provider.get_symbol(stddev)
    assert len(stddev_metric.required_metrics) == 7, (
        f"Expected 7 dependencies, got {len(stddev_metric.required_metrics)}"
    )

    for req_metric in stddev_metric.required_metrics:
        req_symbolic = provider.get_symbol(req_metric)
        # The metric spec should be DayOverDay
        assert isinstance(req_symbolic.metric_spec, specs.DayOverDay), (
            f"Expected DayOverDay spec, got {type(req_symbolic.metric_spec)}"
        )
        # The retrieval function should exist
        assert req_symbolic.fn is not None, f"Missing retrieval function for {req_metric}"
