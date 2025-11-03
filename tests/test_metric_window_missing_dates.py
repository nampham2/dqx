"""Test to demonstrate get_metric_window behavior with missing dates.

This test confirms that get_metric_window returns only the dates that have
actual metric values, without filling missing dates with 0. This behavior
causes downstream compute functions to fail with helpful error messages.

Tests use pattern matching and functional style from the Returns library.
"""

import datetime as dt

import pytest
from returns.maybe import Maybe, Some
from returns.pipeline import is_successful
from returns.result import Failure, Result, Success

from dqx import compute, specs, states
from dqx.cache import MetricCache
from dqx.common import Metadata, ResultKey, TimeSeries
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB


def test_get_metric_window_with_no_metrics() -> None:
    """Test that get_metric_window returns Some({}) when no metrics exist."""
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=dt.date(2025, 2, 10), tags={})
    execution_id = "test-exec-123"

    # Request a 5-day window where no metrics exist
    result: Maybe[TimeSeries] = db.get_metric_window(
        specs.Average("revenue"), key, lag=0, window=5, dataset="test_dataset", execution_id=execution_id
    )

    # Verify it returns Some with empty dict using pattern matching
    match result:
        case Some(ts):
            assert ts == {}
        case _:  # Nothing case
            pytest.fail("Expected Some({}), got Nothing")


def test_get_metric_window_with_partial_metrics() -> None:
    """Test that get_metric_window returns only dates with actual metrics."""
    db = InMemoryMetricDB()
    base_date = dt.date(2025, 2, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})
    execution_id = "test-exec-456"
    spec = specs.Sum("revenue")

    # Create metrics for only days 1, 3, and 5 (skip days 2 and 4)
    # Window will span from 2025-02-06 to 2025-02-10
    metrics_to_persist = [
        Metric.build(
            spec,
            ResultKey(yyyy_mm_dd=dt.date(2025, 2, 6), tags={}),
            dataset="test_dataset",
            state=states.SimpleAdditiveState(100.0),
            metadata=Metadata(execution_id=execution_id, ttl_hours=168),
        ),
        Metric.build(
            spec,
            ResultKey(yyyy_mm_dd=dt.date(2025, 2, 8), tags={}),
            dataset="test_dataset",
            state=states.SimpleAdditiveState(300.0),
            metadata=Metadata(execution_id=execution_id, ttl_hours=168),
        ),
        Metric.build(
            spec,
            ResultKey(yyyy_mm_dd=dt.date(2025, 2, 10), tags={}),
            dataset="test_dataset",
            state=states.SimpleAdditiveState(500.0),
            metadata=Metadata(execution_id=execution_id, ttl_hours=168),
        ),
    ]

    # Persist metrics
    for metric in metrics_to_persist:
        db.persist([metric])

    # Request the full 5-day window
    result: Maybe[TimeSeries] = db.get_metric_window(
        spec, key, lag=0, window=5, dataset="test_dataset", execution_id=execution_id
    )

    # Verify using pattern matching
    match result:
        case Some(ts):
            # Extract values from Metric objects for comparison
            actual_values = {date: metric.value for date, metric in ts.items()}
            expected_values = {
                dt.date(2025, 2, 6): 100.0,
                dt.date(2025, 2, 8): 300.0,
                dt.date(2025, 2, 10): 500.0,
            }
            assert actual_values == expected_values
            # Verify missing dates are not filled with 0
            assert dt.date(2025, 2, 7) not in ts
            assert dt.date(2025, 2, 9) not in ts
        case _:  # Nothing case
            pytest.fail("Expected Some(timeseries), got Nothing")


def test_day_over_day_fails_with_missing_yesterday() -> None:
    """Test that day_over_day fails when yesterday's metric is missing."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    base_date = dt.date(2025, 2, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})
    execution_id = "test-exec-789"
    spec = specs.Average("revenue")

    # Create metric for today only (2025-02-10), skip yesterday (2025-02-09)
    today_metric = Metric.build(
        spec,
        key,
        dataset="test_dataset",
        state=states.Average(500.0, 10),
        metadata=Metadata(execution_id=execution_id, ttl_hours=168),
    )
    db.persist([today_metric])

    # Try to calculate day-over-day
    result: Result[float, str] = compute.day_over_day(
        metric=spec, dataset="test_dataset", nominal_key=key, execution_id=execution_id, cache=cache
    )

    # Verify using pattern matching
    match result:
        case Failure(error):
            assert "missing metrics" in error
            assert "2025-02-09" in error  # Yesterday's date should be mentioned
        case Success(_):
            pytest.fail("Expected Failure due to missing yesterday metric")


def test_stddev_fails_with_missing_dates_in_window() -> None:
    """Test that stddev fails when any date in the window is missing."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    base_date = dt.date(2025, 2, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})
    execution_id = "test-exec-999"
    spec = specs.Sum("sales")

    # Create metrics for days 1, 3, 5, 7 (missing 2, 4, 6) over 7 days
    dates_with_metrics = [
        dt.date(2025, 2, 4),  # Day 1
        dt.date(2025, 2, 6),  # Day 3 (skip day 2: 2025-02-05)
        dt.date(2025, 2, 8),  # Day 5 (skip day 4: 2025-02-07)
        dt.date(2025, 2, 10),  # Day 7 (skip day 6: 2025-02-09)
    ]

    for date in dates_with_metrics:
        metric = Metric.build(
            spec,
            ResultKey(yyyy_mm_dd=date, tags={}),
            dataset="test_dataset",
            state=states.SimpleAdditiveState(100.0),
            metadata=Metadata(execution_id=execution_id, ttl_hours=168),
        )
        db.persist([metric])

    # Try to calculate stddev over 7-day window
    result: Result[float, str] = compute.stddev(
        metric=spec, size=7, dataset="test_dataset", nominal_key=key, execution_id=execution_id, cache=cache
    )

    # Verify using pattern matching
    match result:
        case Failure(error):
            assert "missing metrics" in error
            assert "3 dates" in error  # Should mention 3 missing dates
        case Success(_):
            pytest.fail("Expected Failure due to missing dates in window")


def test_week_over_week_succeeds_with_sparse_data() -> None:
    """Test week_over_week behavior with sparse data - only needs specific lag points."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    base_date = dt.date(2025, 2, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})
    execution_id = "test-exec-wow"
    spec = specs.Average("revenue")

    # For week-over-week, we only need lag points 0 and 7
    # Create metric for today
    today_metric = Metric.build(
        spec,
        ResultKey(yyyy_mm_dd=dt.date(2025, 2, 10), tags={}),
        dataset="test_dataset",
        state=states.Average(700.0, 10),
        metadata=Metadata(execution_id=execution_id, ttl_hours=168),
    )

    # Create metric for week ago
    week_ago_metric = Metric.build(
        spec,
        ResultKey(yyyy_mm_dd=dt.date(2025, 2, 3), tags={}),
        dataset="test_dataset",
        state=states.Average(350.0, 10),
        metadata=Metadata(execution_id=execution_id, ttl_hours=168),
    )

    db.persist([today_metric, week_ago_metric])

    # Calculate week-over-week (should succeed)
    result: Result[float, str] = compute.week_over_week(
        metric=spec, dataset="test_dataset", nominal_key=key, execution_id=execution_id, cache=cache
    )

    # Verify using pattern matching and pipeline functions
    assert is_successful(result)

    match result:
        case Success(value):
            # WoW = |700-350|/350 = 350/350 = 1.0 (percentage change)
            assert value == pytest.approx(1.0)
        case Failure(_):
            pytest.fail("Expected Success for week-over-week with required lag points")


def test_week_over_week_fails_with_missing_week_ago() -> None:
    """Test that week_over_week fails when the week-ago metric is missing."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    base_date = dt.date(2025, 2, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})
    execution_id = "test-exec-wow-fail"
    spec = specs.Average("revenue")

    # Create metric for today only, skip week ago
    today_metric = Metric.build(
        spec,
        ResultKey(yyyy_mm_dd=dt.date(2025, 2, 10), tags={}),
        dataset="test_dataset",
        state=states.Average(700.0, 10),
        metadata=Metadata(execution_id=execution_id, ttl_hours=168),
    )

    db.persist([today_metric])

    # Try to calculate week-over-week
    result: Result[float, str] = compute.week_over_week(
        metric=spec, dataset="test_dataset", nominal_key=key, execution_id=execution_id, cache=cache
    )

    # Verify it fails
    assert not is_successful(result)

    match result:
        case Failure(error):
            assert "missing metrics" in error
            assert "2025-02-03" in error  # Week ago date should be mentioned
        case Success(_):
            pytest.fail("Expected Failure due to missing week-ago metric")


def test_compute_function_with_completely_missing_metric() -> None:
    """Test compute function behavior when metric is completely missing (Nothing case)."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    key = ResultKey(yyyy_mm_dd=dt.date(2025, 2, 10), tags={})
    execution_id = "test-exec-nothing"

    # Don't persist any metrics - database is empty

    # Test simple_metric
    result: Result[float, str] = compute.simple_metric(
        metric=specs.Average("nonexistent"),
        dataset="test_dataset",
        nominal_key=key,
        execution_id=execution_id,
        cache=cache,
    )

    match result:
        case Failure(error):
            assert "not found" in error
            assert "2025-02-10" in error
        case Success(_):
            pytest.fail("Expected Failure for nonexistent metric")


def test_division_by_zero_handling() -> None:
    """Test that day_over_day and week_over_week handle division by zero gracefully."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    base_date = dt.date(2025, 2, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})
    execution_id = "test-exec-zero"
    spec = specs.Sum("revenue")

    # Create metrics where previous value is 0
    metrics = [
        Metric.build(
            spec,
            ResultKey(yyyy_mm_dd=dt.date(2025, 2, 9), tags={}),  # yesterday
            dataset="test_dataset",
            state=states.SimpleAdditiveState(0.0),  # Zero value
            metadata=Metadata(execution_id=execution_id, ttl_hours=168),
        ),
        Metric.build(
            spec,
            ResultKey(yyyy_mm_dd=dt.date(2025, 2, 10), tags={}),  # today
            dataset="test_dataset",
            state=states.SimpleAdditiveState(100.0),
            metadata=Metadata(execution_id=execution_id, ttl_hours=168),
        ),
    ]

    for metric in metrics:
        db.persist([metric])

    # Test day_over_day with zero yesterday
    result: Result[float, str] = compute.day_over_day(
        metric=spec, dataset="test_dataset", nominal_key=key, execution_id=execution_id, cache=cache
    )

    match result:
        case Failure(error):
            assert "zero" in error.lower()
            assert "2025-02-09" in error
        case Success(_):
            pytest.fail("Expected Failure due to division by zero")
