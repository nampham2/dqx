"""Additional tests for compute.py to achieve 100% coverage."""

import statistics
from datetime import date, timedelta
from unittest.mock import patch

import pytest
from returns.maybe import Nothing
from returns.result import Failure, Success

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
    metric_spec = Sum("revenue")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={})

    # Mock get_metric_window to return Nothing
    with patch.object(db, "get_metric_window", return_value=Nothing):
        result = day_over_day(db, metric_spec, "test_dataset", key, "exec-123")

        # Should return Failure with "Metric not found" message
        assert isinstance(result, Failure)
        assert result.failure() == "Metric not found in the metric database"


def test_week_over_week_with_failing_get_metric_window() -> None:
    """Test week_over_week when get_metric_window returns Nothing."""
    db = InMemoryMetricDB()
    metric_spec = Sum("revenue")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={})

    # Mock get_metric_window to return Nothing
    with patch.object(db, "get_metric_window", return_value=Nothing):
        result = week_over_week(db, metric_spec, "test_dataset", key, "exec-123")

        # Should return Failure with "Metric not found" message
        assert isinstance(result, Failure)
        assert result.failure() == "Metric not found in the metric database"


def test_stddev_with_failing_get_metric_window() -> None:
    """Test stddev when get_metric_window returns Nothing."""
    db = InMemoryMetricDB()
    metric_spec = Sum("revenue")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={})

    # Mock get_metric_window to return Nothing
    with patch.object(db, "get_metric_window", return_value=Nothing):
        result = stddev(db, metric_spec, 5, "test_dataset", key, "exec-123")

        # Should return Failure with "Metric not found" message
        assert isinstance(result, Failure)
        assert result.failure() == "Metric not found in the metric database"


def test_stddev_with_statistics_error() -> None:
    """Test stddev when statistics.stdev raises StatisticsError."""
    db = InMemoryMetricDB()
    metric_spec = Sum("revenue")
    base_date = date(2024, 1, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})

    # Populate data for 5 days
    for i in range(5):
        date_key = ResultKey(yyyy_mm_dd=base_date - timedelta(days=i), tags={})
        populate_metric(db, metric_spec, date_key, float(i * 10), "test_dataset", "exec-123")

    # Mock statistics.stdev to raise StatisticsError
    with patch("dqx.compute.statistics.stdev", side_effect=statistics.StatisticsError("Test error")):
        result = stddev(db, metric_spec, 5, "test_dataset", key, "exec-123")

        # Should return Failure with the error message
        assert isinstance(result, Failure)
        assert result.failure() == "Failed to calculate standard deviation: Test error"


def test_pass_statements_coverage() -> None:
    """Test to ensure pass statements in match cases are covered."""
    # This test verifies that the pass statements in the match cases are executed
    # These pass statements are necessary for the match syntax but don't do anything

    db = InMemoryMetricDB()
    metric_spec = Sum("revenue")
    base_date = date(2024, 1, 10)
    key = ResultKey(yyyy_mm_dd=base_date, tags={})

    # For day_over_day - populate valid data that will pass both checks
    populate_metric(db, metric_spec, key, 150.0, "test_dataset", "exec-123")
    yesterday_key = ResultKey(yyyy_mm_dd=base_date - timedelta(days=1), tags={})
    populate_metric(db, metric_spec, yesterday_key, 100.0, "test_dataset", "exec-123")

    result = day_over_day(db, metric_spec, "test_dataset", key, "exec-123")
    assert isinstance(result, Success)
    assert result.unwrap() == pytest.approx(1.5)

    # For week_over_week - populate valid data that will pass both checks
    week_ago_key = ResultKey(yyyy_mm_dd=base_date - timedelta(days=7), tags={})
    populate_metric(db, metric_spec, week_ago_key, 50.0, "test_dataset", "exec-123")

    result = week_over_week(db, metric_spec, "test_dataset", key, "exec-123")
    assert isinstance(result, Success)
    assert result.unwrap() == pytest.approx(3.0)  # 150/50

    # For stddev - populate valid data for all 5 days
    for i in range(5):
        date_key = ResultKey(yyyy_mm_dd=base_date - timedelta(days=i), tags={})
        populate_metric(db, metric_spec, date_key, float(10 + i * 5), "test_dataset", "exec-123")

    result = stddev(db, metric_spec, 5, "test_dataset", key, "exec-123")
    assert isinstance(result, Success)
    # Values are [10, 15, 20, 25, 30] in reverse-chronological order (newest→oldest)
    # Note: The loop intentionally overwrites any earlier metrics so get_metric_window
    # (which selects the latest metric per day ordered by created desc) will use
    # these fresh values [10, 15, 20, 25, 30] for the stddev calculation
    expected = statistics.stdev([10, 15, 20, 25, 30])
    assert result.unwrap() == pytest.approx(expected)
