import datetime as dt

import pyarrow as pa
import pytest

from dqx.api import VerificationSuite, check
from dqx.common import Context, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.display import print_assertion_results, print_symbols
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


@check(name="overall")
def simple_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.num_rows()).where(
        name="row count is more between 1e3 and 15e3",
        severity="P0",
    ).is_between(1e3, 15e3)

    ctx.assert_that(mp.num_rows()).where(name="row count is less than 1000").is_leq(1e3)


@check(name="booking basic")
def booking_basic(mp: MetricProvider, ctx: Context) -> None:
    nr = mp.num_rows()
    bbasic_count = mp.count_values("id_source", "bbasic")

    ctx.assert_that(nr).where(name="booking basic percentage is less than 5%").is_leq(0.05)
    ctx.assert_that(bbasic_count / nr).where(name="booking basic percentage is less than 5%").is_leq(0.05)


@check(name="nits")
def nits(mp: MetricProvider, ctx: Context) -> None:
    nits_null_count = mp.null_count("nits_score")
    nits_corrected_null_count = mp.null_count("nits_score_corrected")
    nr = mp.num_rows()

    ctx.assert_that(nits_null_count / nr).where(
        name="nits null percentage is less than 5%",
        severity="P0",
    ).is_leq(0.05)

    ctx.assert_that(nits_corrected_null_count / nr).where(
        name="corrected nits null percentage is less than 5%",
        severity="P0",
    ).is_leq(0.05)

    ctx.assert_that(mp.average("nits_score")).where(
        name="nits average is between 0.2 and 0.7",
        severity="P0",
    ).is_between(0.2, 0.7)


@check(name="commission")
def commission(mp: MetricProvider, ctx: Context) -> None:
    commission_null_count = mp.null_count("commission_amount_euro")
    nr = mp.num_rows()

    ctx.assert_that(commission_null_count / nr).where(
        name="nits null percentage is less than 5%",
        severity="P0",
    ).is_leq(0.05)


@check(name="bookings")
def bookings(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.ext.day_over_day(mp.sum("bookings"))).where(
        name="bookings change wow is less than 20%",
        severity="P0",
    ).is_leq(0.2)


@check(name="metric collection")
def metric_collection(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.sum("roomnights")).where(name="total roomnights").noop()
    ctx.assert_that(mp.sum("commission_amount_euro")).where(name="total commission").noop()
    ctx.assert_that(mp.sum("commission_amount_euro_bsb_corrected")).where(name="total commission bsb corrected").noop()
    ctx.assert_that(mp.sum("bookings")).where(name="total bookings").noop()


@pytest.mark.skip(reason="BigQuery tests are disabled temporarily")
def test_bq_e2e_suite(commerce_data_c1: pa.Table, commerce_data_c2: pa.Table) -> None:
    db = InMemoryMetricDB()
    ds1 = DuckRelationDataSource.from_arrow(commerce_data_c1)
    ds2 = DuckRelationDataSource.from_arrow(commerce_data_c2)

    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    checks = [simple_checks, booking_basic]

    # Run once for yesterday
    suite = VerificationSuite(checks, db, name="Simple test suite")
    suite.run({"ds1": ds1, "ds2": ds2}, key.lag(1))

    # Run for today
    suite = VerificationSuite(checks, db, name="Simple test suite")

    suite.run({"ds1": ds1, "ds2": ds2}, key)
    suite.graph.print_tree()

    print_assertion_results(suite.collect_results())
    print_symbols(suite.collect_symbols())
