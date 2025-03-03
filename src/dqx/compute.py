import datetime as dt

import numpy as np
from returns.converters import maybe_to_result
from returns.pipeline import flow
from returns.pointfree import bind
from returns.result import Failure, Result, Success

from dqx.common import ResultKey, ResultKeyProvider, TimeSeries
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


def simple_metric(
    db: MetricDB, metric: MetricSpec, key_provider: ResultKeyProvider, nominal_key: ResultKey
) -> Result[float, str]:
    key = key_provider.create(nominal_key)
    value = db.get_metric_value(metric, key)
    return maybe_to_result(value, f"Metric {metric.name} not found")


def day_over_day(
    db: MetricDB, metric: MetricSpec, key_provider: ResultKeyProvider, nominal_key: ResultKey
) -> Result[float, str]:
    key = key_provider.create(nominal_key)

    def _dod(ts: TimeSeries) -> Result[float, str]:
        """Calculate the day over day metric."""
        lag_0 = ts[key.yyyy_mm_dd]
        lag_1 = ts[key.lag(1).yyyy_mm_dd]

        # Checking for divide by zero
        if lag_1 == 0:
            return Failure(f"Metric for {key.lag(1).yyyy_mm_dd.isoformat()} is zero.")
        return Success(lag_0 / lag_1)

    return flow(
        db.get_metric_window(metric, key, lag=0, window=2),
        lambda ts: maybe_to_result(ts, METRIC_NOT_FOUND),
        bind(lambda ts: _timeseries_check(ts, key.lag(1).yyyy_mm_dd, 2)),
        bind(_dod),
    )


def stddev(
    db: MetricDB, metric: MetricSpec, lag: int, size: int, key_provider: ResultKeyProvider, nominal_key: ResultKey
) -> Result[float, str]:
    key = key_provider.create(nominal_key)

    def _stddev(ts: TimeSeries) -> Result[float, str]:
        return Success(np.std(list(ts.values())).item())

    return flow(
        db.get_metric_window(metric, key, lag, size),
        lambda ts: maybe_to_result(ts, METRIC_NOT_FOUND),
        bind(lambda ts: _timeseries_check(ts, key.lag(lag).yyyy_mm_dd, size)),
        bind(_stddev),
    )
