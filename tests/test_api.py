import datetime as dt

import pyarrow as pa
import sympy as sp
from rich.console import Console

from dqx import specs
from dqx.api import VerificationSuite, check
from dqx.common import Context, MetricProvider, ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB


@check
def simple_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.null_count("delivered")).is_leq(100)
    ctx.assert_that(mp.minimum("quantity")).is_leq(2.5)
    ctx.assert_that(mp.average("price")).is_geq(10.0)


@check(description="Delivered null percentage")
def null_percentage(mp: MetricProvider, ctx: Context) -> None:
    null_count = mp.null_count("delivered")
    nr = mp.num_rows()
    ctx.assert_that(null_count / nr).label("null percentage is less than 40%").is_leq(0.4)


@check(description="Manual day-over-day check")
def manual_day_over_day(mp: MetricProvider, ctx: Context) -> None:
    tax_avg = mp.average("tax")
    tax_avg_lag = mp.average("tax", key=ctx.key.lag(1))
    ctx.assert_that(tax_avg / tax_avg_lag).is_eq(1.0, tol=0.01)


@check(description="Rate of change")
def rate_of_change(mp: MetricProvider, ctx: Context) -> None:
    tax_avg = mp.ext.day_over_day(specs.Maximum("tax"))
    rate = sp.Abs(tax_avg - 1.0)
    ctx.assert_that(rate).label("Maximum tax rate change is less than 20%").is_leq(0.2)


@check
def sketch_check(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.approx_cardinality("address")).is_geq(100)


def test_inspect_no_run() -> None:
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    checks = [simple_checks, manual_day_over_day, rate_of_change, null_percentage, sketch_check]

    # Run once for yesterday
    suite = VerificationSuite(checks, db, name="Simple test suite")
    ctx = suite.collect(key)
    tree = ctx._graph.inspect()
    Console().print(tree)

    # assert set(suite.collect(key).pending_metrics()) == {
    #     specs.NumRows(),
    #     specs.NullCount("quantity"),
    #     specs.Minimum("quantity"),
    #     specs.Average("price"),
    #     specs.Minimum("price"),
    # }


def test_verification_suite(commerce_data: pa.Table) -> None:
    db = InMemoryMetricDB()
    ds = ArrowDataSource(commerce_data)
    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})

    checks = [simple_checks, manual_day_over_day, rate_of_change, null_percentage, sketch_check]

    # Run once for yesterday
    suite = VerificationSuite(checks, db, name="Simple test suite")
    suite.run(ds, key.lag(1))

    # Run for today
    suite = VerificationSuite(checks, db, name="Simple test suite")

    ctx = suite.run(ds, key)

    tree = ctx._graph.inspect()
    Console().print(tree)
