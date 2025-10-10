import datetime as dt

import pyarrow as pa
import sympy as sp

from dqx import specs
from dqx.api import VerificationSuite, check
from dqx.common import Context, ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


@check(datasets=["ds1"])
def simple_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.null_count("delivered")).where(name="Delivered null count is less than 100").is_leq(100)
    ctx.assert_that(mp.minimum("quantity")).is_leq(2.5)
    ctx.assert_that(mp.average("price")).is_geq(10.0)
    ctx.assert_that(mp.ext.day_over_day(specs.Average("tax"))).is_geq(0.5)


@check(name="Delivered null percentage", datasets=["ds1"])
def null_percentage(mp: MetricProvider, ctx: Context) -> None:
    null_count = mp.null_count("delivered", dataset="ds1")
    nr = mp.num_rows()
    ctx.assert_that(null_count / nr).where(name="null percentage is less than 40%").is_leq(0.4)


@check
def manual_day_over_day(mp: MetricProvider, ctx: Context) -> None:
    tax_avg = mp.average("tax")
    tax_avg_lag = mp.average("tax", key=ctx.key.lag(1))
    ctx.assert_that(tax_avg / tax_avg_lag).where().is_eq(1.0, tol=0.01)


@check(name="Rate of change", datasets=["ds2"])
def rate_of_change(mp: MetricProvider, ctx: Context) -> None:
    tax_avg = mp.ext.day_over_day(specs.Maximum("tax"))
    rate = sp.Abs(tax_avg - 1.0)
    ctx.assert_that(rate).where(name="Maximum tax rate change is less than 20%").is_leq(0.2)


@check(datasets=["ds1"])
def sketch_check(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.approx_cardinality("address")).is_geq(100)


@check(datasets=["ds1", "ds2"])
def cross_dataset_check(mp: MetricProvider, ctx: Context) -> None:
    tax_avg_1 = mp.average("tax", dataset="ds1")
    tax_avg_2 = mp.average("tax", dataset="ds2")
    # Allow for identical datasets (difference can be 0)
    ctx.assert_that(sp.Abs(tax_avg_1 / tax_avg_2 - 1)).is_lt(0.2, tol=0.01)


def test_verification_suite(commerce_data_c1: pa.Table, commerce_data_c2: pa.Table) -> None:
    db = InMemoryMetricDB()
    ds1 = ArrowDataSource(commerce_data_c1)
    ds2 = ArrowDataSource(commerce_data_c2)

    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    checks = [simple_checks, manual_day_over_day, rate_of_change, null_percentage, sketch_check, cross_dataset_check]

    # Run once for yesterday
    suite = VerificationSuite(checks, db, name="Simple test suite")
    suite.run({"ds1": ds1, "ds2": ds2}, key.lag(1))

    # Run for today
    suite = VerificationSuite(checks, db, name="Simple test suite")
    ctx = suite._context

    suite.run({"ds1": ds1, "ds2": ds2}, key)
    ctx._graph.print_tree()
