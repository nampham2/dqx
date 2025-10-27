"""Computation functions for metrics.

This module provides functions to compute various metrics from the database.
Each function retrieves data and performs calculations as needed.
"""

import datetime as dt
import statistics

from returns.converters import maybe_to_result as convert_maybe_to_result
from returns.result import Failure, Result, Success

from dqx.common import ResultKey, TimeSeries
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

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


def simple_metric(db: MetricDB, metric: MetricSpec, dataset: str, nominal_key: ResultKey) -> Result[float, str]:
    """Retrieve a simple metric value from the database for a specific dataset.

    Fetches the value of a metric for the given specification, date, and dataset.
    The dataset parameter is mandatory to ensure correct metric retrieval when
    the same metric exists across multiple datasets.

    Args:
        db: The metric database instance.
        metric: The metric specification to retrieve.
        dataset: The dataset name where the metric was computed.
        nominal_key: The result key containing date and tags.

    Returns:
        Success with the metric value if found, Failure with error message otherwise.
    """
    value = db.get_metric_value(metric, nominal_key, dataset)
    error_msg = f"Metric {metric.name} for {nominal_key.yyyy_mm_dd.isoformat()} on dataset '{dataset}' not found!"
    return convert_maybe_to_result(value, error_msg)


def day_over_day(db: MetricDB, metric: MetricSpec, dataset: str, nominal_key: ResultKey) -> Result[float, str]:
    """Calculate day-over-day ratio for a metric.

    Retrieves the metric value for the nominal date and the previous day,
    then calculates the ratio. The dataset parameter ensures metrics are
    retrieved from the correct dataset.

    Args:
        db: The metric database instance.
        metric: The metric specification to calculate ratio for.
        dataset: The dataset name where metrics were computed.
        nominal_key: The result key for the nominal date.

    Returns:
        Success with the ratio if calculation succeeds, Failure otherwise.
    """
    # Get two days of data starting from the base date
    base_key = nominal_key.lag(0)  # Use lag=0 as the base
    maybe_ts = db.get_metric_window(metric, base_key, lag=0, window=2, dataset=dataset)

    # Convert Maybe to Result
    ts_result = convert_maybe_to_result(maybe_ts, METRIC_NOT_FOUND)
    if isinstance(ts_result, Failure):
        return ts_result
    ts = ts_result.unwrap()

    # Validate we have all required dates
    check_result = _timeseries_check(ts, base_key.yyyy_mm_dd - dt.timedelta(days=1), 2)
    if isinstance(check_result, Failure):
        return check_result

    # Calculate the ratio
    today = base_key.yyyy_mm_dd
    yesterday = today - dt.timedelta(days=1)

    if ts[yesterday] == 0:
        return Failure(f"Cannot calculate day over day: previous day value ({yesterday}) is zero.")

    return Success(ts[today] / ts[yesterday])


def week_over_week(db: MetricDB, metric: MetricSpec, dataset: str, nominal_key: ResultKey) -> Result[float, str]:
    """Calculate week-over-week ratio for a metric.

    Retrieves the metric value for the nominal date and a week prior,
    then calculates the ratio. The dataset parameter ensures metrics are
    retrieved from the correct dataset.

    Args:
        db: The metric database instance.
        metric: The metric specification to calculate ratio for.
        dataset: The dataset name where metrics were computed.
        nominal_key: The result key for the nominal date.

    Returns:
        Success with the ratio if calculation succeeds, Failure otherwise.
    """
    # Get eight days of data starting from the base date
    base_key = nominal_key.lag(0)  # Use lag=0 as the base
    maybe_ts = db.get_metric_window(metric, base_key, lag=0, window=8, dataset=dataset)

    # Convert Maybe to Result
    ts_result = convert_maybe_to_result(maybe_ts, METRIC_NOT_FOUND)
    if isinstance(ts_result, Failure):
        return ts_result
    ts = ts_result.unwrap()

    # We only need values at specific lag points: 0 and 7
    check_result = _sparse_timeseries_check(ts, base_key.yyyy_mm_dd, [0, 7])
    if isinstance(check_result, Failure):
        return check_result

    # Calculate the ratio
    today = base_key.yyyy_mm_dd
    week_ago = today - dt.timedelta(days=7)

    if ts[week_ago] == 0:
        return Failure(f"Cannot calculate week over week: week ago value ({week_ago}) is zero.")

    return Success(ts[today] / ts[week_ago])


def stddev(db: MetricDB, metric: MetricSpec, size: int, dataset: str, nominal_key: ResultKey) -> Result[float, str]:
    """Calculate standard deviation for a metric over a window.

    Retrieves metric values for a specified window size and calculates
    the standard deviation. The dataset parameter ensures metrics are
    retrieved from the correct dataset.

    Args:
        db: The metric database instance.
        metric: The metric specification to calculate stddev for.
        size: Number of days to include in the calculation.
        dataset: The dataset name where metrics were computed.
        nominal_key: The result key for the end date of the window.

    Returns:
        Success with the standard deviation if calculation succeeds, Failure otherwise.
    """
    # Get the time window of data
    base_key = nominal_key.lag(0)  # Use lag=0 as the base
    maybe_ts = db.get_metric_window(metric, base_key, lag=0, window=size, dataset=dataset)

    # Convert Maybe to Result
    ts_result = convert_maybe_to_result(maybe_ts, METRIC_NOT_FOUND)
    if isinstance(ts_result, Failure):
        return ts_result
    ts = ts_result.unwrap()

    # Validate we have all required dates
    from_date = base_key.yyyy_mm_dd - dt.timedelta(days=size - 1)
    check_result = _timeseries_check(ts, from_date, size)
    if isinstance(check_result, Failure):
        return check_result

    # Extract values in chronological order
    values = [ts[from_date + dt.timedelta(days=i)] for i in range(size)]

    # Calculate standard deviation
    if len(values) < 2:
        # Standard deviation of a single value is 0
        return Success(0.0)

    try:
        std_dev = statistics.stdev(values)
        return Success(std_dev)
    except statistics.StatisticsError as e:
        # This shouldn't happen given our checks, but handle it gracefully
        return Failure(f"Failed to calculate standard deviation: {e}")
