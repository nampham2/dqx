"""Test that extended metrics (lag) show correct dates in symbol table."""

import datetime as dt

import pyarrow as pa
from faker import Faker

from dqx.api import VerificationSuite, check
from dqx.common import Context, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


@check(name="Test Check", datasets=["ds1"])
def _test_check(mp: MetricProvider, ctx: Context) -> None:
    """Test check with extended metrics. This is a check function, not a test."""
    tax_avg = mp.average("tax")
    tax_wow = mp.ext.week_over_week(tax_avg)
    tax_dod = mp.ext.day_over_day(tax_avg)

    # Also test stddev which creates multiple lag dependencies
    revenue_sum = mp.sum("revenue")
    revenue_stddev = mp.ext.stddev(revenue_sum, lag=1, n=3)  # Creates lag(1), lag(2), lag(3)

    ctx.assert_that(tax_wow).where(name="Tax WoW < 2").is_lt(2)
    ctx.assert_that(tax_dod).where(name="Tax DoD < 1").is_lt(1)
    ctx.assert_that(revenue_stddev).where(name="Revenue stddev < 1000").is_lt(1000)


def test_lag_metrics_have_correct_dates() -> None:
    """Test that lag metrics show correct lagged dates in symbol table."""
    # Create test data
    Faker.seed(42)
    n, fake = 100, Faker()

    tax = [fake.pyfloat(min_value=-100.0, max_value=100.0) for _ in range(n)]
    revenue = [fake.pyfloat(min_value=1000.0, max_value=10000.0) for _ in range(n)]
    arrow_table = pa.Table.from_arrays([tax, revenue], names=["tax", "revenue"])

    # Create data source
    db = InMemoryMetricDB()
    ds = DuckRelationDataSource.from_arrow(arrow_table, "ds1")

    # Create suite
    suite = VerificationSuite([_test_check], db, name="Lag Date Test")

    # Run with nominal date 2025-01-15
    nominal_date = dt.date(2025, 1, 15)
    key = ResultKey(yyyy_mm_dd=nominal_date, tags={})
    suite.run([ds], key)

    # Collect symbols and check dates
    symbols = suite.provider.collect_symbols(key)

    # Create a map for easier lookup
    symbol_map = {sym.metric: sym for sym in symbols}

    # Verify base metrics have nominal date
    assert "average(tax)" in symbol_map
    assert symbol_map["average(tax)"].yyyy_mm_dd == nominal_date, (
        f"average(tax) should have nominal date {nominal_date}"
    )

    assert "sum(revenue)" in symbol_map
    assert symbol_map["sum(revenue)"].yyyy_mm_dd == nominal_date, (
        f"sum(revenue) should have nominal date {nominal_date}"
    )

    # Find and verify lag metrics have correct dates
    lag_metrics = {name: info for name, info in symbol_map.items() if "lag(" in name}

    # Should have lag(7) from week_over_week, lag(1) from day_over_day, and lag(1,2,3) from stddev
    assert len(lag_metrics) >= 4, (
        f"Should have at least 4 lag metrics, got {len(lag_metrics)}: {list(lag_metrics.keys())}"
    )

    # Check specific lag metrics
    for metric_name, symbol_info in lag_metrics.items():
        if "lag(7)" in metric_name:
            expected_date = nominal_date - dt.timedelta(days=7)
            assert symbol_info.yyyy_mm_dd == expected_date, (
                f"{metric_name} should have date {expected_date}, got {symbol_info.yyyy_mm_dd}"
            )
        elif "lag(1)" in metric_name:
            expected_date = nominal_date - dt.timedelta(days=1)
            assert symbol_info.yyyy_mm_dd == expected_date, (
                f"{metric_name} should have date {expected_date}, got {symbol_info.yyyy_mm_dd}"
            )
        elif "lag(2)" in metric_name:
            expected_date = nominal_date - dt.timedelta(days=2)
            assert symbol_info.yyyy_mm_dd == expected_date, (
                f"{metric_name} should have date {expected_date}, got {symbol_info.yyyy_mm_dd}"
            )
        elif "lag(3)" in metric_name:
            expected_date = nominal_date - dt.timedelta(days=3)
            assert symbol_info.yyyy_mm_dd == expected_date, (
                f"{metric_name} should have date {expected_date}, got {symbol_info.yyyy_mm_dd}"
            )


def test_nested_lag_metrics() -> None:
    """Test that nested extended metrics handle lag dates correctly."""
    Faker.seed(43)
    n, fake = 50, Faker()

    value = [fake.pyfloat(min_value=100.0, max_value=1000.0) for _ in range(n)]
    arrow_table = pa.Table.from_arrays([value], names=["value"])

    db = InMemoryMetricDB()
    ds = DuckRelationDataSource.from_arrow(arrow_table, "ds1")

    @check(name="Nested Check", datasets=["ds1"])
    def nested_check(mp: MetricProvider, ctx: Context) -> None:
        # Create nested extended metrics
        base = mp.average("value")
        wow = mp.ext.week_over_week(base)  # Creates lag(7) of base
        wow_of_wow = mp.ext.week_over_week(wow)  # Should create lag(7) of wow

        ctx.assert_that(wow_of_wow).where(name="WoW of WoW < 1.5").is_lt(1.5)

    suite = VerificationSuite([nested_check], db, name="Nested Test")

    nominal_date = dt.date(2025, 1, 20)
    key = ResultKey(yyyy_mm_dd=nominal_date, tags={})
    suite.run([ds], key)

    symbols = suite.provider.collect_symbols(key)
    symbol_map = {sym.metric: sym for sym in symbols}

    # Check all lag metrics have correct dates
    for metric_name, symbol_info in symbol_map.items():
        if metric_name == "average(value)":
            assert symbol_info.yyyy_mm_dd == nominal_date
        elif "lag(7)" in metric_name and "average(value)" in metric_name:
            # Direct lag of base metric
            assert symbol_info.yyyy_mm_dd == nominal_date - dt.timedelta(days=7)
        elif "lag(7)" in metric_name and "week_over_week" in metric_name:
            # Lag of week_over_week (which itself might be lagged)
            assert symbol_info.yyyy_mm_dd == nominal_date - dt.timedelta(days=7)


if __name__ == "__main__":
    test_lag_metrics_have_correct_dates()
    test_nested_lag_metrics()
    print("All tests passed!")
