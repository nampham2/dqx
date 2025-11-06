"""Computation functions for metrics.

This module provides functions to compute various metrics from the database.
Each function retrieves data and performs calculations as needed.
"""

import datetime as dt
import statistics
from typing import TYPE_CHECKING

from returns.maybe import Some
from returns.result import Failure, Result, Success

from dqx.common import ExecutionId, ResultKey, TimeSeries
from dqx.specs import MetricSpec

if TYPE_CHECKING:
    from dqx.cache import MetricCache

# Error messages
METRIC_NOT_FOUND = "Metric not found in the metric database"


def _timeseries_check(ts: TimeSeries, from_date: dt.date, window: int, limit: int = 5) -> Result[TimeSeries, str]:
    """Validate that a timeseries has all expected dates.

    Args:
        ts: The timeseries to check
        from_date: Starting date
        window: Number of days expected
        limit: Maximum number of missing dates to report

    Returns:
        Success with the timeseries if valid, Failure with error message otherwise
    """
    expected_dates = {from_date + dt.timedelta(days=i) for i in range(window)}
    actual_dates = set(ts.keys())
    missing_dates = expected_dates - actual_dates

    if missing_dates:
        sorted_missing = sorted(missing_dates)[:limit]
        dates_str = ", ".join(d.isoformat() for d in sorted_missing)
        return Failure(f"There are {len(missing_dates)} dates with missing metrics: {dates_str}.")

    return Success(ts)


def _sparse_timeseries_check(
    ts: TimeSeries, base_date: dt.date, lag_points: list[int], limit: int = 5
) -> Result[TimeSeries, str]:
    """Validate that a timeseries has values for specific lag points.

    Unlike _timeseries_check which expects contiguous dates, this function
    checks for specific dates calculated from lag points.

    Args:
        ts: The timeseries to check
        base_date: The reference date
        lag_points: List of lag values (days to subtract from base_date)
        limit: Maximum number of missing dates to report

    Returns:
        Success with the timeseries if valid, Failure with error message otherwise
    """
    expected_dates = {base_date - dt.timedelta(days=lag) for lag in lag_points}
    actual_dates = set(ts.keys())
    missing_dates = expected_dates - actual_dates

    if missing_dates:
        sorted_missing = sorted(missing_dates)[:limit]
        dates_str = ", ".join(d.isoformat() for d in sorted_missing)
        return Failure(f"There are {len(missing_dates)} dates with missing metrics: {dates_str}.")

    return Success(ts)


def simple_metric(
    metric: MetricSpec, dataset: str, nominal_key: ResultKey, execution_id: ExecutionId, cache: "MetricCache"
) -> Result[float, str]:
    """Retrieve a simple metric value using cache then database fallback.

    Args:
        metric: The metric specification to retrieve.
        dataset: The dataset name where the metric was computed.
        nominal_key: The result key containing date and tags.
        execution_id: The execution ID to filter by.
        cache: The metric cache instance.

    Returns:
        Success with the metric value if found, Failure with error message otherwise.
    """
    # Try cache first
    cache_key = (metric, nominal_key, dataset, execution_id)
    maybe_metric = cache.get(cache_key)

    match maybe_metric:
        case Some(metric_value):
            # Cache always returns Metric objects
            return Success(metric_value.value)
        case _:
            # If not found, return failure
            error_msg = (
                f"Metric {metric.name} for {nominal_key.yyyy_mm_dd.isoformat()} on dataset '{dataset}' not found!"
            )
            return Failure(error_msg)


def day_over_day(
    metric: MetricSpec, dataset: str, nominal_key: ResultKey, execution_id: ExecutionId, cache: "MetricCache"
) -> Result[float, str]:
    """Calculate day-over-day absolute percentage change for a metric.

    Computes the absolute percentage change between today's value and yesterday's value
    using the formula: abs((today - yesterday) / yesterday).

    Args:
        metric: The metric specification to calculate percentage change for.
        dataset: The dataset name where metrics were computed.
        nominal_key: The result key for the nominal date.
        execution_id: The execution ID to filter by.
        cache: The metric cache instance.

    Returns:
        Success with the absolute percentage change (e.g., 0.5 for 50% change)
        if calculation succeeds, Failure otherwise.
    """
    # Get two days of data using cache
    base_key = nominal_key.lag(0)  # Use lag=0 as the base
    ts = cache.get_sparse_timeseries(metric, base_key, dataset, execution_id, lags=[0, 1])

    # Validate we have all required dates
    check_result = _sparse_timeseries_check(ts, base_key.yyyy_mm_dd, [0, 1])
    match check_result:
        case Failure() as failure:
            return failure
        case Success():
            pass

    # Calculate the ratio
    today = base_key.yyyy_mm_dd
    yesterday = today - dt.timedelta(days=1)

    if ts[yesterday].value == 0:
        return Failure(f"Cannot calculate day over day: previous day value ({yesterday}) is zero.")

    return Success(abs((ts[today].value - ts[yesterday].value) / ts[yesterday].value))


def week_over_week(
    metric: MetricSpec, dataset: str, nominal_key: ResultKey, execution_id: ExecutionId, cache: "MetricCache"
) -> Result[float, str]:
    """Calculate week-over-week absolute percentage change for a metric.

    Computes the absolute percentage change between today's value and the value
    from 7 days ago using the formula: abs((today - week_ago) / week_ago).

    Args:
        metric: The metric specification to calculate percentage change for.
        dataset: The dataset name where metrics were computed.
        nominal_key: The result key for the nominal date.
        execution_id: The execution ID to filter by.
        cache: The metric cache instance.

    Returns:
        Success with the absolute percentage change (e.g., 1.1 for 110% change)
        if calculation succeeds, Failure otherwise.
    """
    # Get eight days of data using cache
    base_key = nominal_key.lag(0)  # Use lag=0 as the base
    lags = [0, 7]

    ts = cache.get_sparse_timeseries(metric, base_key, dataset, execution_id, lags=lags)

    # We only need values at specific lag points: 0 and 7
    check_result = _sparse_timeseries_check(ts, base_key.yyyy_mm_dd, lags)
    match check_result:
        case Failure() as failure:
            return failure
        case Success():
            pass

    # Calculate the ratio
    today = base_key.yyyy_mm_dd
    week_ago = today - dt.timedelta(days=7)

    if ts[week_ago].value == 0:
        return Failure(f"Cannot calculate week over week: week ago value ({week_ago}) is zero.")

    return Success(abs((ts[today].value - ts[week_ago].value) / ts[week_ago].value))


def stddev(
    metric: MetricSpec, dataset: str, nominal_key: ResultKey, execution_id: ExecutionId, size: int, cache: "MetricCache"
) -> Result[float, str]:
    """Calculate standard deviation for a metric over a window.

    Args:
        metric: The metric specification to calculate stddev for.
        dataset: The dataset name where metrics were computed.
        nominal_key: The result key for the end date of the window.
        execution_id: The execution ID to filter by.
        size: Number of days to include in the calculation.
        cache: The metric cache instance.

    Returns:
        Success with the standard deviation if calculation succeeds, Failure otherwise.
    """
    # Get the time window of data using cache
    base_key = nominal_key.lag(0)  # Use lag=0 as the base
    ts = cache.get_timeseries(metric, base_key, dataset, execution_id, window=size)

    # Validate we have all required dates
    from_date = base_key.yyyy_mm_dd - dt.timedelta(days=size - 1)

    # Extract values in chronological order, ignore missing values
    values: list[float] = [
        ts[from_date + dt.timedelta(days=i)].value for i in range(size) if from_date + dt.timedelta(days=i) in ts
    ]

    # Calculate standard deviation
    if len(values) < 1:
        return Failure("No data to calculate standard deviation")

    if len(values) < 2:
        # Standard deviation of a single value is 0
        return Success(0.0)

    try:
        std_dev = statistics.stdev(values)
        return Success(std_dev)
    except statistics.StatisticsError as e:
        # This shouldn't happen given our checks, but handle it gracefully
        return Failure(f"Failed to calculate standard deviation: {e}")
