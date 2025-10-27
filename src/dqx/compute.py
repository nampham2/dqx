import datetime as dt

import numpy as np
from returns.converters import maybe_to_result
from returns.pipeline import flow
from returns.pointfree import bind
from returns.result import Failure, Result, Success

from dqx.common import ResultKey, TimeSeries
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

METRIC_NOT_FOUND = "Metric not found in the metric database"


def _timeseries_check(ts: TimeSeries, from_date: dt.date, window: int, limit: int = 5) -> Result[TimeSeries, str]:
    missings: list[dt.date] = []
    for lag in range(window):
        current = from_date + dt.timedelta(days=lag)
        if current not in ts:
            missings.append(current)

    if not missings:
        return Success(ts)

    # Some dates are missing, constructing the message
    missing_dates_str = ", ".join(m.isoformat() for m in missings[:limit])
    return Failure(f"There are {len(missings)} dates with missing metrics: {missing_dates_str}.")


def _sparse_timeseries_check(
    ts: TimeSeries, base_date: dt.date, lag_points: list[int], limit: int = 5
) -> Result[TimeSeries, str]:
    """Check if specific lag points exist in the timeseries.

    Args:
        ts: The timeseries to check
        base_date: The base date to calculate lags from
        lag_points: List of lag values to check (e.g., [0, 7] for lag_0 and lag_7)
        limit: Maximum number of missing dates to show in error message

    Returns:
        Success with the timeseries if all required dates exist,
        Failure with error message listing missing dates (up to limit)
    """
    missings: list[dt.date] = []
    for lag in lag_points:
        # lag_0 is base_date, lag_1 is base_date - 1 day, etc.
        current = base_date - dt.timedelta(days=lag)
        if current not in ts:
            missings.append(current)

    if not missings:
        return Success(ts)

    # Apply limit to the error message
    missing_dates_str = ", ".join(m.isoformat() for m in missings[:limit])
    return Failure(f"There are {len(missings)} dates with missing metrics: {missing_dates_str}.")


def simple_metric(db: MetricDB, metric: MetricSpec, lag: int, nominal_key: ResultKey) -> Result[float, str]:
    key = nominal_key.lag(lag)
    value = db.get_metric_value(metric, key)
    return maybe_to_result(value, f"Metric {metric.name} not found!")


def day_over_day(db: MetricDB, metric: MetricSpec, nominal_key: ResultKey) -> Result[float, str]:
    def _dod(ts: TimeSeries) -> Result[float, str]:
        """Calculate the day over day metric."""
        lag_0 = ts[nominal_key.yyyy_mm_dd]
        lag_1 = ts[nominal_key.lag(1).yyyy_mm_dd]

        # Checking for divide by zero
        if lag_1 == 0:
            return Failure(f"Metric for {nominal_key.lag(1).yyyy_mm_dd.isoformat()} is zero.")
        return Success(lag_0 / lag_1)

    return flow(
        db.get_metric_window(metric, nominal_key, lag=0, window=2),
        lambda ts: maybe_to_result(ts, METRIC_NOT_FOUND),
        bind(lambda ts: _sparse_timeseries_check(ts, nominal_key.yyyy_mm_dd, [0, 1])),
        bind(_dod),
    )


def week_over_week(db: MetricDB, metric: MetricSpec, nominal_key: ResultKey) -> Result[float, str]:
    def _wow(ts: TimeSeries) -> Result[float, str]:
        """Calculate the week over week metric."""
        lag_0 = ts[nominal_key.yyyy_mm_dd]
        lag_7 = ts[nominal_key.lag(7).yyyy_mm_dd]

        # Checking for divide by zero
        if lag_7 == 0:
            return Failure(f"Metric for {nominal_key.lag(7).yyyy_mm_dd.isoformat()} is zero.")
        return Success(lag_0 / lag_7)

    return flow(
        db.get_metric_window(metric, nominal_key, lag=0, window=8),
        lambda ts: maybe_to_result(ts, METRIC_NOT_FOUND),
        bind(lambda ts: _sparse_timeseries_check(ts, nominal_key.yyyy_mm_dd, [0, 7])),
        bind(_wow),
    )


def stddev(db: MetricDB, metric: MetricSpec, lag: int, size: int, nominal_key: ResultKey) -> Result[float, str]:
    # Apply lag to get the effective base date
    base_key = nominal_key.lag(lag)

    def _stddev(ts: TimeSeries) -> Result[float, str]:
        return Success(np.std(list(ts.values())).item())

    return flow(
        db.get_metric_window(metric, base_key, lag=0, window=size),
        lambda ts: maybe_to_result(ts, METRIC_NOT_FOUND),
        bind(lambda ts: _timeseries_check(ts, base_key.yyyy_mm_dd, size)),
        bind(_stddev),
    )
