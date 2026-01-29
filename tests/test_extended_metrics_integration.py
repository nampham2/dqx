"""Integration tests for complex extended metric dependencies."""

from datetime import date

import pyarrow as pa

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
    provider = MetricProvider(db, "test-exec-123", data_av_threshold=0.8)

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
        ctx.assert_that(std).config(name="Collect stddev").noop()

    # Run suite for each day
    for key_date, table_data in tables:
        suite = VerificationSuite([tax_dod_stddev_check], db, "Test Suite")
        datasource = DuckRelationDataSource.from_arrow(table_data, "tax_data")
        key = ResultKey(yyyy_mm_dd=key_date, tags={"test": "stddev_dod"})
        suite.run([datasource], key)

    # Since each suite run has a different execution ID, we need to check metrics differently
    # For each day, verify that the required metrics were computed during that day's suite run

    # Verify all required metrics exist in DB by checking if we can retrieve them
    # Should have average(tax) for all days where suites were run

    # We can't use get_metric without execution_id, so let's just verify the last execution has all the metrics it needs

    # For the last execution (day 10), it should have computed:
    # - Average(tax) for days 3-10 (8 days)
    # - DayOverDay for days 4-10 (7 days)
    # - Stddev for day 10 (1 metric)
    last_suite = suite  # The last suite from the loop above
    all_metrics = db.get_by_execution_id(last_suite.execution_id)

    # Group metrics by type
    metrics_by_type: dict[str, list] = {"Average": [], "DayOverDay": [], "Stddev": [], "NumRows": []}
    for metric in all_metrics:
        metric_type = metric.spec.metric_type
        if metric_type in metrics_by_type:
            metrics_by_type[metric_type].append(metric)

    # Verify we have the right number of each metric type
    # The last suite (day 10) computes stddev which needs 7 days of DoD values
    # Each DoD needs the current and previous day's average
    # So we need averages for days 3-10 (8 days) and DoD for days 4-10 (7 days)
    assert len(metrics_by_type["Average"]) >= 8, (
        f"Expected at least 8 Average metrics, got {len(metrics_by_type['Average'])}"
    )
    assert len(metrics_by_type["DayOverDay"]) >= 7, (
        f"Expected at least 7 DayOverDay metrics, got {len(metrics_by_type['DayOverDay'])}"
    )
    assert len(metrics_by_type["Stddev"]) >= 1, (
        f"Expected at least 1 Stddev metric, got {len(metrics_by_type['Stddev'])}"
    )

    # Verify the stddev metric has the expected parameters
    stddev_metrics = [m for m in all_metrics if m.spec.metric_type == "Stddev"]
    assert len(stddev_metrics) > 0, "No Stddev metric found"
    stddev_metric_obj = stddev_metrics[0]
    assert stddev_metric_obj.spec.parameters.get("offset") == 0
    assert stddev_metric_obj.spec.parameters.get("n") == 7


def test_nested_extended_metrics_combinations() -> None:
    """Test various combinations of nested extended metrics."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec-124", data_av_threshold=0.8)

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
    provider = MetricProvider(db, "test-exec-125", data_av_threshold=0.8)

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
        ctx.assert_that(std).config(name="Collect stddev").noop()

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

    # Get all metrics from the execution
    all_metrics = db.get_by_execution_id(suite.execution_id)

    # Check that the stddev metric was computed
    found_stddev = False
    for metric in all_metrics:
        if (
            metric.spec.metric_type == "Stddev"
            and metric.spec.parameters.get("offset") == 0
            and metric.spec.parameters.get("n") == 7
            and metric.key == key
        ):
            found_stddev = True
            break

    # The bug is fixed if we can successfully get a result
    assert found_stddev, "Stddev metric was not computed - bug still present"

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
