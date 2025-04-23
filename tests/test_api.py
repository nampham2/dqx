import datetime as dt

import pyarrow as pa
import sympy as sp
from rich.console import Console

from dqx import specs
from dqx.api import VerificationSuite, check
from dqx.common import Context, MetricProvider, ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB


@check(datasets=["abc"])
def simple_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.null_count("delivered")).is_leq(100)
    ctx.assert_that(mp.minimum("quantity")).is_leq(2.5)
    ctx.assert_that(mp.average("price")).is_geq(10.0)
    ctx.assert_that(mp.ext.day_over_day(specs.Average("tax"))).is_geq(0.5)


@check(label="Delivered null percentage", datasets=["ds1"])
def null_percentage(mp: MetricProvider, ctx: Context) -> None:
    null_count = mp.null_count("delivered", datasets=["ds1"])
    nr = mp.num_rows()
    ctx.assert_that(null_count / nr).on(label="null percentage is less than 40%").is_leq(0.4)


@check(label="Manual day-over-day check", datasets=["ds1"])
def manual_day_over_day(mp: MetricProvider, ctx: Context) -> None:
    tax_avg = mp.average("tax")
    tax_avg_lag = mp.average("tax", key=ctx.key.lag(1))
    ctx.assert_that(tax_avg / tax_avg_lag).on().is_eq(1.0, tol=0.01)


@check(label="Rate of change", datasets=["ds2"])
def rate_of_change(mp: MetricProvider, ctx: Context) -> None:
    tax_avg = mp.ext.day_over_day(specs.Maximum("tax"))
    rate = sp.Abs(tax_avg - 1.0)
    ctx.assert_that(rate).on(label="Maximum tax rate change is less than 20%").is_leq(0.2)


@check(datasets=["ds1"])
def sketch_check(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.approx_cardinality("address", datasets=["ds2"])).is_geq(100)

@check()
def cross_dataset_check(mp: MetricProvider, ctx: Context) -> None:
    tax_avg_1= mp.average("tax", datasets=["ds1"])
    tax_avg_2 = mp.average("tax", datasets=["ds2"])
    ctx.assert_that(sp.Abs(tax_avg_1 / tax_avg_2 - 1)).is_lt(0.2, tol=0.01)


def test_inspect_no_run() -> None:
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    checks = [simple_checks, manual_day_over_day, rate_of_change, null_percentage, sketch_check]

    # Run once for yesterday
    suite = VerificationSuite(checks, db, name="Simple test suite")
    ctx = suite.collect(key)
    ctx._graph.propagate(["ds1", "ds2"])
    tree = ctx._graph.inspect()
    Console().print(tree)

    # assert set(suite.collect(key).pending_metrics()) == {
    #     specs.NumRows(),
    #     specs.NullCount("quantity"),
    #     specs.Minimum("quantity"),
    #     specs.Average("price"),
    #     specs.Minimum("price"),
    # }


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

    ctx = suite.run({"ds1": ds1, "ds2": ds2}, key)

    tree = ctx._graph.inspect()
    Console().print(tree)
