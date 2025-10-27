import datetime as dt

import pyarrow as pa
import sympy as sp

from dqx import data
from dqx.api import VerificationSuite, check
from dqx.common import Context, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.display import (
    print_assertion_results,
    print_metric_trace,
)
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


@check(name="Simple Checks", datasets=["ds1"])
def simple_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.null_count("delivered")).where(name="Delivered null count is less than 100").is_leq(100)
    ctx.assert_that(mp.minimum("quantity")).where(name="Minimum quantity check").is_leq(2.5)
    ctx.assert_that(mp.average("price")).where(name="Average price check").is_geq(10.0)
    ctx.assert_that(mp.ext.day_over_day(mp.average("tax"))).where(name="Tax day-over-day check").is_geq(0.5)
    ctx.assert_that(mp.duplicate_count(["name"], dataset="ds1")).where(name="No duplicates on name").is_eq(0)
    ctx.assert_that(mp.minimum("quantity", dataset="ds1")).where(
        name="Quantity minimum is between 1 and 5",
    ).is_between(1, 5.0)
    ctx.assert_that(mp.count_values("name", "np", dataset="ds1")).where(name="NP never buys here").is_eq(0)


@check(name="complex metrics", datasets=["ds1"])
def complex_metrics(mp: MetricProvider, ctx: Context) -> None:
    tax_stddev = mp.ext.stddev(mp.average("tax"), lag=1, n=7)
    ctx.assert_that(tax_stddev).where(name="Tax stddev is small").is_leq(10.0)


@check(name="Delivered null percentage", datasets=["ds1"])
def null_percentage(mp: MetricProvider, ctx: Context) -> None:
    null_count = mp.null_count("delivered", dataset="ds1")
    nr = mp.num_rows()
    ctx.assert_that(null_count / nr).where(name="null percentage is less than 40%").is_leq(0.4)


@check(name="Manual Day Over Day", datasets=["ds1"])
def manual_day_over_day(mp: MetricProvider, ctx: Context) -> None:
    tax_avg = mp.average("tax")
    tax_avg_lag = mp.average("tax", lag=1)
    ctx.assert_that(tax_avg / tax_avg_lag).where(name="Tax average day-over-day equals 1.0").is_eq(1.0, tol=0.01)


@check(name="Rate of change", datasets=["ds2"])
def rate_of_change(mp: MetricProvider, ctx: Context) -> None:
    tax_dod = mp.ext.day_over_day(mp.maximum("tax"))
    tax_wow = mp.ext.week_over_week(mp.average("tax"))
    rate = sp.Abs(tax_dod - 1.0)
    ctx.assert_that(rate).where(name="Maximum tax rate change is less than 20%").is_leq(0.2)
    ctx.assert_that(tax_wow).where(name="Average tax week-over-week change is less than 30%").is_leq(0.3)


@check(name="Cross Dataset Check", datasets=["ds1", "ds2"])
def cross_dataset_check(mp: MetricProvider, ctx: Context) -> None:
    tax_avg_1 = mp.average("tax", dataset="ds1")
    tax_avg_2 = mp.average("tax", dataset="ds2")

    ctx.assert_that(sp.Abs(tax_avg_1 / tax_avg_2 - 1)).where(name="Tax average ratio between datasets").is_lt(
        0.2, tol=0.01
    )
    ctx.assert_that(mp.first("tax", dataset="ds1")).where(name="random tax value").noop()


def test_e2e_suite(commerce_data_c1: pa.Table, commerce_data_c2: pa.Table) -> None:
    db = InMemoryMetricDB()
    ds1 = DuckRelationDataSource.from_arrow(commerce_data_c1, "ds1")
    ds2 = DuckRelationDataSource.from_arrow(commerce_data_c2, "ds2")

    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    checks = [simple_checks, manual_day_over_day, rate_of_change, null_percentage, cross_dataset_check, complex_metrics]

    # Run for today
    suite = VerificationSuite(checks, db, name="Simple test suite")

    suite.run([ds1, ds2], key)
    suite.graph.print_tree()

    print_assertion_results(suite.collect_results())
    print_metric_trace(trace := suite.metric_trace(db), suite.execution_id)

    assert data.metric_trace_stats(trace).discrepancy_count == 0, "There should be no discrepancies in the metrics"
