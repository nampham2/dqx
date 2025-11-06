"""Additional tests for compute.py to achieve 100% coverage."""

import statistics
from datetime import date, timedelta
from unittest.mock import patch

import pytest
from returns.result import Failure, Success

from dqx.cache import MetricCache
from dqx.common import ExecutionId, Metadata, ResultKey
from dqx.compute import day_over_day, stddev, week_over_week
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB, MetricDB
from dqx.specs import MetricSpec, Sum
from dqx.states import SimpleAdditiveState


def populate_metric(
    db: MetricDB,
    metric_spec: MetricSpec,
    key: ResultKey,
    value: float,
    dataset: str = "test_dataset",
    execution_id: ExecutionId = "test-exec-123",
) -> None:
    """Helper to add a metric to the database."""
    metadata = Metadata(execution_id=execution_id)
    metric = Metric.build(
        metric=metric_spec,
        key=key,
        dataset=dataset,
        state=SimpleAdditiveState(value=value),
        metadata=metadata,
    )
    db.persist([metric])


def test_day_over_day_with_failing_get_metric_window() -> None:
    """Test day_over_day when get_metric_window returns Nothing."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    metric_spec = Sum("revenue")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={})

    # Don't add any metrics to DB, so cache will return empty window
    result = day_over_day(metric_spec, "test_dataset", key, "exec-123", cache)

    # Should return Failure with missing dates message
    assert isinstance(result, Failure)
    assert "There are 2 dates with missing metrics" in result.failure()


def test_week_over_week_with_failing_get_metric_window() -> None:
    """Test week_over_week when get_metric_window returns Nothing."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    metric_spec = Sum("revenue")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={})

    # Don't add any metrics to DB, so cache will return empty window
    result = week_over_week(metric_spec, "test_dataset", key, "exec-123", cache)

    # Should return Failure with missing dates message
    assert isinstance(result, Failure)
    assert "There are 2 dates with missing metrics" in result.failure()


def test_stddev_with_failing_get_metric_window() -> None:
    """Test stddev when get_metric_window returns Nothing."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    metric_spec = Sum("revenue")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={})

    # Don't add any metrics to DB, so cache will return empty window
    result = stddev(metric_spec, "test_dataset", key, "exec-123", 5, cache)

    # Should return Failure with no data message
    assert isinstance(result, Failure)
    assert result.failure() == "No data to calculate standard deviation"


def test_stddev_with_statistics_error() -> None:
    """Test stddev when statistics.stdev raises StatisticsError."""
    db = InMemoryMetricDB()
    cache = MetricCache(db)
    metric_spec = Sum("revenue")
    base_date = date(2024, 1, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})

    # Populate data for 5 days
    for i in range(5):
        date_key = ResultKey(yyyy_mm_dd=base_date - timedelta(days=i), tags={})
        populate_metric(db, metric_spec, date_key, float(i * 10), "test_dataset", "exec-123")

    # Mock statistics.stdev to raise StatisticsError
    with patch("dqx.compute.statistics.stdev", side_effect=statistics.StatisticsError("Test error")):
        result = stddev(metric_spec, "test_dataset", key, "exec-123", 5, cache)

        # Should return Failure with the error message
        assert isinstance(result, Failure)
        assert result.failure() == "Failed to calculate standard deviation: Test error"


def test_pass_statements_coverage() -> None:
    """Test to ensure pass statements in match cases are covered."""
    # This test verifies that the pass statements in the match cases are executed
    # These pass statements are necessary for the match syntax but don't do anything

    db = InMemoryMetricDB()
    cache = MetricCache(db)
    metric_spec = Sum("revenue")
    base_date = date(2024, 1, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})

    # For day_over_day - populate valid data that will pass both checks
    populate_metric(db, metric_spec, key, 150.0, "test_dataset", "exec-123")
    yesterday_key = ResultKey(yyyy_mm_dd=base_date - timedelta(days=1), tags={})
    populate_metric(db, metric_spec, yesterday_key, 100.0, "test_dataset", "exec-123")

    result = day_over_day(metric_spec, "test_dataset", key, "exec-123", cache)
    assert isinstance(result, Success)
    # DoD = |150-100|/100 = 50/100 = 0.5 (percentage change)
    assert result.unwrap() == pytest.approx(0.5)

    # For week_over_week - populate valid data that will pass both checks
    week_ago_key = ResultKey(yyyy_mm_dd=base_date - timedelta(days=7), tags={})
    populate_metric(db, metric_spec, week_ago_key, 50.0, "test_dataset", "exec-123")

    result = week_over_week(metric_spec, "test_dataset", key, "exec-123", cache)
    assert isinstance(result, Success)
    # WoW = |150-50|/50 = 100/50 = 2.0 (percentage change)
    assert result.unwrap() == pytest.approx(2.0)

    # For stddev - populate valid data for all 5 days with a different execution_id
    # to avoid interfering with the previous tests
    # Note: We're adding metrics in reverse chronological order (newest first)
    # The values will be:
    # 2024-01-10: 10.0
    # 2024-01-09: 15.0
    # 2024-01-08: 20.0
    # 2024-01-07: 25.0
    # 2024-01-06: 30.0
    for i in range(5):
        date_key = ResultKey(yyyy_mm_dd=base_date - timedelta(days=i), tags={})
        populate_metric(db, metric_spec, date_key, float(10 + i * 5), "test_dataset", "stddev-exec-123")

    result = stddev(metric_spec, "test_dataset", key, "stddev-exec-123", 5, cache)
    assert isinstance(result, Success)

    # The stddev function extracts values in chronological order (oldest to newest)
    # So it will compute stddev of [30, 25, 20, 15, 10]
    expected = statistics.stdev([30, 25, 20, 15, 10])
    assert result.unwrap() == pytest.approx(expected)
