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
    nr = mp.num_rows()

    ctx.assert_that(mp.duplicate_count(["hotelreservation_id"])).where(
        name="no duplicate hotelreservation_id",
        severity="P0",
    ).is_eq(0)

    ctx.assert_that(mp.num_rows()).where(
        name="row count is between 1e3 and 15e3",
        severity="P0",
    ).is_between(1e3, 15e3)

    null_hotel_id = mp.null_count("hotel_id")
    ctx.assert_that(null_hotel_id / nr).where(
        name="hotel_id null percentage is less than 5%",
        severity="P1",
    ).is_leq(0.05)

    null_partner_silo = mp.null_count("partner_silo")
    ctx.assert_that(null_partner_silo / nr).where(
        name="partner_silo null percentage is less than 5%", severity="P1"
    ).is_leq(0.05)

    null_partner_market = mp.null_count("partner_market")
    ctx.assert_that(null_partner_market / nr).where(
        name="partner_market null percentage is less than 5%", severity="P1"
    ).is_leq(0.05)

    null_dow = mp.null_count("day_of_week")
    ctx.assert_that(null_dow / nr).where(name="day_of_week null percentage is less than 5%", severity="P1").is_leq(0.05)

    null_bw = mp.null_count("booking_window")
    ctx.assert_that(null_bw / nr).where(
        name="booking_window null percentage is less than 5%",
        severity="P1",
    ).is_leq(0.05)

    null_los = mp.null_count("length_of_stay")
    ctx.assert_that(null_los / nr).where(
        name="length_of_stay null percentage is less than 5%",
        severity="P1",
    ).is_leq(0.05)

    null_device = mp.null_count("device")
    ctx.assert_that(null_device / nr).where(
        name="device null percentage is less than 5%",
        severity="P1",
    ).is_leq(0.05)

    null_dd = mp.null_count("default_date_flag")
    ctx.assert_that(null_dd / nr).where(
        name="default_date_flag null percentage is less than 5%",
        severity="P1",
    ).is_leq(0.05)

    null_ad_type = mp.null_count("ad_type")
    ctx.assert_that(null_ad_type / nr).where(
        name="ad_type null percentage is less than 5%",
        severity="P1",
    ).is_leq(0.05)

    null_int_dom = mp.null_count("int_dom")
    ctx.assert_that(null_int_dom / nr).where(
        name="int_dom null percentage is less than 5%",
        severity="P1",
    ).is_leq(0.05)

    null_yyyy_mm_dd = mp.null_count("yyyy_mm_dd")
    ctx.assert_that(null_yyyy_mm_dd / nr).where(
        name="yyyy_mm_dd null percentage is less than 5%",
        severity="P1",
    ).is_leq(0.05)

    null_hotel_reservation_id = mp.null_count("hotelreservation_id")
    ctx.assert_that(null_hotel_reservation_id).where(
        name="no null hotelreservation_id",
        severity="P0",
    ).is_eq(0)


@check(name="booking basic")
def booking_basic(mp: MetricProvider, ctx: Context) -> None:
    nr = mp.num_rows()
    bbasic_count = mp.count_values("id_source", "bbasic")
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
    ctx.assert_that(mp.ext.week_over_week(mp.sum("bookings"))).where(
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
def test_bq_e2e_suite(commerce_data_c1: pa.Table) -> None:
    db = InMemoryMetricDB()
    ds1 = DuckRelationDataSource.from_arrow(commerce_data_c1)

    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    checks = [simple_checks, booking_basic, nits, commission, bookings, metric_collection]

    suite = VerificationSuite(checks, db, name="Simple test suite")

    suite.run({"ds1": ds1}, key)
    suite.graph.print_tree()

    print_assertion_results(suite.collect_results())
    print_symbols(suite.collect_symbols())
