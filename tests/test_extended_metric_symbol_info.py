"""Test that extended metrics (day_over_day, stddev) have correct SymbolInfo names."""

import datetime as dt

import pyarrow as pa

from dqx.api import VerificationSuite, check
from dqx.common import Context, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


@check(name="Extended Metric Test", datasets=["ds1"])
def extended_metric_check(mp: MetricProvider, ctx: Context) -> None:
    """Test check using day_over_day, week_over_week, and stddev metrics."""
    # Create day_over_day metric
    dod_metric = mp.ext.day_over_day(mp.maximum("tax"))
    ctx.assert_that(dod_metric).where(name="Day over day check").is_geq(0.5)

    # Create week_over_week metric
    wow_metric = mp.ext.week_over_week(mp.sum("price"))
    ctx.assert_that(wow_metric).where(name="Week over week check").is_geq(0.8)

    # Create stddev metric
    stddev_metric = mp.ext.stddev(mp.average("price"), lag=1, n=7)
    ctx.assert_that(stddev_metric).where(name="Standard deviation check").is_leq(10000.0)


def test_extended_metrics_symbol_info_names(commerce_data_c1: pa.Table) -> None:
    """Test that extended metrics have correct names in SymbolInfo."""
    db = InMemoryMetricDB()
    ds = DuckRelationDataSource.from_arrow(commerce_data_c1, "ds1")

    # Create and run suite
    suite = VerificationSuite([extended_metric_check], db, "Extended Metric Test Suite")
    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    suite.run([ds], key)

    # Collect symbols
    symbols = suite.provider.collect_symbols(key)

    # Find our extended metric symbols
    symbol_metrics = {s.metric for s in symbols}

    # Verify day_over_day metric name is correct
    assert any("day_over_day(maximum(tax))" in metric for metric in symbol_metrics), (
        f"Expected 'day_over_day(maximum(tax))' in symbol metrics, but got: {symbol_metrics}"
    )

    # Verify week_over_week metric name is correct
    assert any("week_over_week(sum(price))" in metric for metric in symbol_metrics), (
        f"Expected 'week_over_week(sum(price))' in symbol metrics, but got: {symbol_metrics}"
    )

    # Verify stddev metric name is correct (includes lag and n parameters)
    assert any("stddev(average(price), lag=1, n=7)" in metric for metric in symbol_metrics), (
        f"Expected 'stddev(average(price), lag=1, n=7)' in symbol metrics, but got: {symbol_metrics}"
    )

    # Also verify the bug scenario from test_api_e2e
    # Find the specific metrics used in the e2e test
    for symbol in symbols:
        # Check that day_over_day metrics are named correctly
        if "day_over_day" in symbol.metric and "tax" in symbol.metric:
            assert symbol.metric in ["day_over_day(average(tax))", "day_over_day(maximum(tax))"], (
                f"Unexpected day_over_day metric name: {symbol.metric}"
            )

            # Specifically check for the bug case
            assert symbol.metric != "maximum(tax)", (
                "Bug reproduced: day_over_day(maximum(tax)) is showing as just 'maximum(tax)'"
            )

        # Check that week_over_week metrics are named correctly
        if "week_over_week" in symbol.metric and "price" in symbol.metric:
            assert symbol.metric == "week_over_week(sum(price))", (
                f"Unexpected week_over_week metric name: {symbol.metric}"
            )

            # Ensure it doesn't show as just the base metric
            assert symbol.metric != "sum(price)", (
                "Bug reproduced: week_over_week(sum(price)) is showing as just 'sum(price)'"
            )
