"""Tests for lag date handling functionality."""

import datetime
from typing import Any

import pyarrow as pa

from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider, SymbolicMetric


def test_pending_metrics_returns_symbolic_metrics() -> None:
    """Test that pending_metrics returns SymbolicMetric objects with key providers."""
    db = InMemoryMetricDB()
    ctx = Context("Test Suite", db)

    # Create metrics with different lags
    mp = ctx.provider
    mp.average("price")
    mp.average("price", key=ctx.key.lag(1))

    pending = ctx.pending_metrics()

    assert len(pending) == 2
    assert all(isinstance(m, SymbolicMetric) for m in pending)
    assert pending[0].key_provider._lag == 0
    assert pending[1].key_provider._lag == 1


def test_suite_analyzes_metrics_with_correct_dates(monkeypatch: Any) -> None:
    """Test that metrics with different lags are analyzed for correct dates."""
    db = InMemoryMetricDB()

    @check(name="Test Check", datasets=["ds1"])
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        current = mp.average("value")
        lagged = mp.average("value", key=ctx.key.lag(1))
        ctx.assert_that(current / lagged).where(name="Ratio check").is_eq(1.0)

    # Track analyzer calls
    from dqx.analyzer import Analyzer

    analyze_calls = []
    original_analyze = Analyzer.analyze

    def track_analyze(self: Any, ds: Any, metrics_by_key: Any) -> Any:
        # Track all the keys used in the analyze call
        for key, metrics in metrics_by_key.items():
            analyze_calls.append((key, key.yyyy_mm_dd, len(metrics)))
        return original_analyze(self, ds, metrics_by_key)

    monkeypatch.setattr(Analyzer, "analyze", track_analyze)

    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(datetime.date(2025, 1, 15), {})

    # Create test data
    data = pa.table({"value": [100.0] * 10})
    ds = DuckRelationDataSource.from_arrow(data)

    suite.run({"ds1": ds}, key)

    # Verify analyze was called with metrics for different dates
    assert len(analyze_calls) == 2

    # Extract the dates from the calls
    dates_used = {call[1] for call in analyze_calls}

    assert datetime.date(2025, 1, 15) in dates_used
    assert datetime.date(2025, 1, 14) in dates_used

    # Verify each date has metrics
    for call in analyze_calls:
        assert call[2] > 0  # Each date should have at least one metric


def test_collect_symbols_with_lagged_dates() -> None:
    """Test that collected symbols show correct effective dates for lagged metrics."""
    db = InMemoryMetricDB()

    @check(name="Time Series Check", datasets=["ds1"])
    def time_series_check(mp: MetricProvider, ctx: Context) -> None:
        current = mp.average("value")  # Should be 2025-01-15
        lag1 = mp.average("value", key=ctx.key.lag(1))  # Should be 2025-01-14
        lag2 = mp.average("value", key=ctx.key.lag(2))  # Should be 2025-01-13

        # Create assertions to ensure symbols are registered
        ctx.assert_that(current).where(name="Current check").is_gt(0)
        ctx.assert_that(lag1).where(name="Lag 1 check").is_gt(0)
        ctx.assert_that(lag2).where(name="Lag 2 check").is_gt(0)

    suite = VerificationSuite([time_series_check], db, "Test Suite")
    key = ResultKey(datetime.date(2025, 1, 15), {"env": "test"})

    # Create test data
    data = pa.table({"value": [100.0] * 10})
    ds = DuckRelationDataSource.from_arrow(data)

    # Run suite
    suite.run({"ds1": ds}, key)

    # Collect symbols
    symbols = suite.provider.collect_symbols(key)

    # Find the three symbols we created by checking their dates
    # Since all metrics are average(value), we differentiate by date
    date_to_sym = {s.yyyy_mm_dd: s for s in symbols}

    current_sym = date_to_sym[datetime.date(2025, 1, 15)]
    lag1_sym = date_to_sym[datetime.date(2025, 1, 14)]
    lag2_sym = date_to_sym[datetime.date(2025, 1, 13)]

    # Verify dates
    assert current_sym.yyyy_mm_dd == datetime.date(2025, 1, 15)
    assert lag1_sym.yyyy_mm_dd == datetime.date(2025, 1, 14)
    assert lag2_sym.yyyy_mm_dd == datetime.date(2025, 1, 13)

    # Verify tags are preserved
    assert all(s.tags == {"env": "test"} for s in [current_sym, lag1_sym, lag2_sym])


def test_mixed_lag_and_no_lag_metrics() -> None:
    """Test that metrics with and without lag work together correctly."""
    db = InMemoryMetricDB()

    @check(name="Mixed Lag Check", datasets=["ds1"])
    def mixed_check(mp: MetricProvider, ctx: Context) -> None:
        # Mix of lagged and non-lagged metrics
        current_avg = mp.average("value")
        current_sum = mp.sum("value")
        yesterday_avg = mp.average("value", key=ctx.key.lag(1))
        yesterday_sum = mp.sum("value", key=ctx.key.lag(1))

        # Use them in assertions
        ctx.assert_that(current_avg).where(name="Current avg").is_gt(0)
        ctx.assert_that(current_sum).where(name="Current sum").is_gt(0)
        ctx.assert_that(yesterday_avg).where(name="Yesterday avg").is_gt(0)
        ctx.assert_that(yesterday_sum).where(name="Yesterday sum").is_gt(0)

    suite = VerificationSuite([mixed_check], db, "Test Suite")
    key = ResultKey(datetime.date(2025, 1, 15), {})

    data = pa.table({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})
    ds = DuckRelationDataSource.from_arrow(data)

    suite.run({"ds1": ds}, key)
    symbols = suite.provider.collect_symbols(key)

    # Verify we have 4 symbols
    assert len(symbols) == 4

    # Separate by date
    current_symbols = [s for s in symbols if s.yyyy_mm_dd == datetime.date(2025, 1, 15)]
    yesterday_symbols = [s for s in symbols if s.yyyy_mm_dd == datetime.date(2025, 1, 14)]

    assert len(current_symbols) == 2  # current_avg and current_sum
    assert len(yesterday_symbols) == 2  # yesterday_avg and yesterday_sum


def test_missing_historical_data_graceful_handling() -> None:
    """Test graceful handling when lagged date has no data."""
    db = InMemoryMetricDB()

    @check(name="Missing History Check", datasets=["ds1"])
    def missing_check(mp: MetricProvider, ctx: Context) -> None:
        current = mp.average("value")
        # This will compute for 30 days ago
        historical = mp.average("value", key=ctx.key.lag(30))

        # Should handle missing data gracefully
        ctx.assert_that(current).where(name="Current exists").is_gt(0)
        # Historical data might be None or 0 depending on implementation
        ctx.assert_that(historical).where(name="Historical check").is_between(0, float("inf"))

    suite = VerificationSuite([missing_check], db, "Test Suite")
    key = ResultKey(datetime.date(2025, 1, 15), {})

    # Data source only has current data
    data = pa.table({"value": [100.0] * 10})
    ds = DuckRelationDataSource.from_arrow(data)

    # Should not fail even if historical data is missing
    suite.run({"ds1": ds}, key)

    # Verify both symbols are collected
    symbols = suite.provider.collect_symbols(key)
    assert len(symbols) == 2

    # Verify dates
    current_sym = next(s for s in symbols if s.yyyy_mm_dd == datetime.date(2025, 1, 15))
    historical_sym = next(s for s in symbols if s.yyyy_mm_dd == datetime.date(2024, 12, 16))

    # Success object contains the value directly
    from returns.result import Failure, Success

    # Current data should be successful
    assert isinstance(current_sym.value, Success)
    assert current_sym.value.unwrap() > 0

    # Historical data might fail if not found (which is expected for 30 days ago)
    # This is actually good - it means the system handles missing data gracefully
    assert isinstance(historical_sym.value, (Success, Failure))
    if isinstance(historical_sym.value, Success):
        assert historical_sym.value.unwrap() >= 0
    else:
        # It's a Failure, which is fine for missing historical data
        assert "not found" in str(historical_sym.value)


def test_large_lag_values() -> None:
    """Test lag values for monthly and yearly comparisons."""
    db = InMemoryMetricDB()

    @check(name="Large Lag Check", datasets=["ds1"])
    def large_lag_check(mp: MetricProvider, ctx: Context) -> None:
        current = mp.average("revenue")
        last_month = mp.average("revenue", key=ctx.key.lag(30))
        last_year = mp.average("revenue", key=ctx.key.lag(365))

        # Month-over-month growth
        mom_growth = (current - last_month) / last_month * 100
        # Year-over-year growth
        yoy_growth = (current - last_year) / last_year * 100

        ctx.assert_that(mom_growth).where(name="MoM Growth %").is_between(-100, 1000)
        ctx.assert_that(yoy_growth).where(name="YoY Growth %").is_between(-100, 1000)

    suite = VerificationSuite([large_lag_check], db, "Test Suite")
    key = ResultKey(datetime.date(2025, 1, 15), {})

    data = pa.table({"revenue": [1000.0] * 50})
    ds = DuckRelationDataSource.from_arrow(data)

    suite.run({"ds1": ds}, key)
    symbols = suite.provider.collect_symbols(key)

    # Find the metrics
    dates_found = {s.yyyy_mm_dd for s in symbols}

    # Verify correct date calculations
    assert datetime.date(2025, 1, 15) in dates_found  # current
    assert datetime.date(2024, 12, 16) in dates_found  # 30 days ago
    assert datetime.date(2024, 1, 16) in dates_found  # 365 days ago (2025 is not a leap year)


def test_date_boundary_conditions() -> None:
    """Test lag calculations across year and month boundaries."""
    db = InMemoryMetricDB()

    @check(name="Boundary Check", datasets=["ds1"])
    def boundary_check(mp: MetricProvider, ctx: Context) -> None:
        current = mp.num_rows()
        yesterday = mp.num_rows(key=ctx.key.lag(1))
        last_week = mp.num_rows(key=ctx.key.lag(7))

        ctx.assert_that(current).where(name="Current count").is_gt(0)
        ctx.assert_that(yesterday).where(name="Yesterday count").is_gt(0)
        ctx.assert_that(last_week).where(name="Last week count").is_gt(0)

    # Test case 1: New Year boundary
    suite = VerificationSuite([boundary_check], db, "Test Suite")
    key = ResultKey(datetime.date(2025, 1, 2), {})

    data = pa.table({"id": list(range(100))})
    ds = DuckRelationDataSource.from_arrow(data)

    suite.run({"ds1": ds}, key)
    symbols = suite.provider.collect_symbols(key)

    # Verify dates cross year boundary correctly
    dates = {s.yyyy_mm_dd for s in symbols}
    assert datetime.date(2025, 1, 2) in dates  # Jan 2, 2025
    assert datetime.date(2025, 1, 1) in dates  # Jan 1, 2025
    assert datetime.date(2024, 12, 26) in dates  # Dec 26, 2024

    # Test case 2: Month boundary
    key2 = ResultKey(datetime.date(2025, 3, 1), {})
    suite2 = VerificationSuite([boundary_check], db, "Test Suite 2")

    suite2.run({"ds1": ds}, key2)
    symbols2 = suite2.provider.collect_symbols(key2)

    dates2 = {s.yyyy_mm_dd for s in symbols2}
    assert datetime.date(2025, 3, 1) in dates2  # Mar 1, 2025
    assert datetime.date(2025, 2, 28) in dates2  # Feb 28, 2025 (non-leap year)
    assert datetime.date(2025, 2, 22) in dates2  # Feb 22, 2025
