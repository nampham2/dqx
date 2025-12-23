import datetime as dt

import sympy as sp

from dqx.api import VerificationSuite, check
from dqx.common import Context, ResultKey
from dqx.display import print_assertion_results, print_metric_trace
from dqx.orm.repositories import InMemoryMetricDB
from dqx.profiles import HolidayProfile, check as profile_check, tag
from dqx.provider import MetricProvider
from tests.fixtures.data_fixtures import CommercialDataSource


@check(name="Simple Checks", datasets=["ds1"])
def simple_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.null_count("delivered")).where(name="Delivered null count is less than 100").is_leq(100)
    ctx.assert_that(mp.minimum("quantity")).where(name="Minimum quantity check").is_leq(2.5)
    ctx.assert_that(mp.average("price")).where(name="Average price check").is_geq(10.0)
    ctx.assert_that(mp.ext.day_over_day(mp.average("tax"))).where(name="Tax day-over-day check").is_geq(0.5)
    ctx.assert_that(mp.duplicate_count(["name"], dataset="ds1")).where(name="No duplicates on name").is_eq(0)
    ctx.assert_that(
        mp.minimum(
            "quantity",
            dataset="ds1",
            parameters={"min_quantity": 10},
        )
    ).where(name="Quantity minimum is between 1 and 5").is_between(1, 5.0)
    ctx.assert_that(mp.count_values("name", "np", dataset="ds1")).where(name="NP never buys here").is_eq(0)
    ctx.assert_that(mp.unique_count("name")).where(name="At least 5 unique customers").is_geq(5)


@check(name="Custom checks", datasets=["ds1"])
def custom_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.custom_sql("count(*)", parameters={"min_quantity": 20})).where(name="Count orders").is_gt(100)


@check(name="complex metrics", datasets=["ds1"])
def complex_metrics(mp: MetricProvider, ctx: Context) -> None:
    tax = mp.average("tax")
    tax_stddev = mp.ext.stddev(mp.ext.day_over_day(tax), offset=1, n=7)
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
    ctx.assert_that(tax_avg / tax_avg_lag).where(name="Tax average day-over-day equals 1.0", tags={"xmas"}).is_eq(
        1.0, tol=0.01
    )


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


def test_e2e_suite() -> None:
    db = InMemoryMetricDB()

    # Define date ranges for the two datasources
    # ds1: Full month of January 2025
    ds1_start_date = dt.date(2025, 1, 1)
    ds1_end_date = dt.date(2025, 1, 31)

    # ds2: Slightly different range - starts earlier, ends on same day
    # This allows testing scenarios where historical data availability differs
    ds2_start_date = dt.date(2024, 12, 15)  # Starts mid-December 2024
    ds2_end_date = dt.date(2025, 1, 31)

    # Create the datasources with their respective date ranges
    ds1 = CommercialDataSource(
        start_date=ds1_start_date,
        end_date=ds1_end_date,
        name="ds1",
        records_per_day=30,
        seed=1050,  # Same seed as original commerce_data_c1
        skip_dates={dt.date.fromisoformat("2025-01-13")},
    )

    ds2 = CommercialDataSource(
        start_date=ds2_start_date,
        end_date=ds2_end_date,
        name="ds2",
        records_per_day=35,
        seed=2100,  # Same seed as original commerce_data_c2
        skip_dates={dt.date.fromisoformat("2025-01-14")},
    )

    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={"env": "prod", "partner": "gha"})
    checks = [
        simple_checks,
        custom_checks,
        manual_day_over_day,
        rate_of_change,
        null_percentage,
        cross_dataset_check,
        complex_metrics,
    ]

    # Run for today
    suite = VerificationSuite(
        checks,
        db,
        name="Simple test suite",
        data_av_threshold=0.8,
        log_level="DEBUG",
    )

    suite.run([ds1, ds2], key)
    print_assertion_results(suite.collect_results())
    print_metric_trace(suite.metric_trace(db), suite.data_av_threshold)


def test_e2e_suite_with_profiles() -> None:
    """Test e2e suite with profiles that modify assertion behavior."""
    db = InMemoryMetricDB()

    ds1_start_date = dt.date(2025, 1, 1)
    ds1_end_date = dt.date(2025, 1, 31)

    ds1 = CommercialDataSource(
        start_date=ds1_start_date,
        end_date=ds1_end_date,
        name="ds1",
        records_per_day=30,
        seed=1050,
        skip_dates={dt.date.fromisoformat("2025-01-13")},
    )

    # Create a profile that:
    # 1. Disables the "Delivered null percentage" check entirely
    # 2. Applies a metric_multiplier to assertions tagged with "xmas"
    holiday_profile = HolidayProfile(
        name="January Holiday",
        start_date=dt.date(2025, 1, 1),
        end_date=dt.date(2025, 1, 31),
        rules=[
            profile_check("Delivered null percentage").disable(),
            tag("xmas").set(metric_multiplier=1.0),  # No change, just verify it's applied
        ],
    )

    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={"env": "prod"})

    # Use only checks that work with ds1
    checks = [
        simple_checks,
        manual_day_over_day,
        null_percentage,  # This will be disabled by profile
    ]

    suite = VerificationSuite(
        checks,
        db,
        name="Profile test suite",
        data_av_threshold=0.8,
        log_level="DEBUG",
        profiles=[holiday_profile],
    )

    suite.run([ds1], key)
    results = suite.collect_results()

    # Verify profile effects
    # 1. "Delivered null percentage" assertions should be SKIPPED
    disabled_results = [r for r in results if r.check == "Delivered null percentage"]
    assert len(disabled_results) > 0, "Expected Delivered null percentage assertions"
    for r in disabled_results:
        assert r.status == "SKIPPED", f"Expected SKIPPED for {r.assertion}, got {r.status}"

    # 2. "Manual Day Over Day" assertion with xmas tag should be evaluated (not skipped)
    xmas_results = [r for r in results if "xmas" in r.assertion_tags]
    assert len(xmas_results) > 0, "Expected assertions with xmas tag"
    for r in xmas_results:
        # Should be evaluated (PASSED or FAILED), not SKIPPED
        assert r.status in ("PASSED", "FAILED"), f"Expected evaluation for {r.assertion}, got {r.status}"

    # 3. Other assertions should be evaluated normally
    simple_results = [r for r in results if r.check == "Simple Checks"]
    assert len(simple_results) > 0, "Expected Simple Checks assertions"

    print_assertion_results(results)
    print(f"\nProfile test completed: {len(results)} assertions evaluated")
    print(f"  - SKIPPED by profile: {len([r for r in results if r.status == 'SKIPPED'])}")
    print(f"  - PASSED: {len([r for r in results if r.status == 'PASSED'])}")
    print(f"  - FAILED: {len([r for r in results if r.status == 'FAILED'])}")


@check(name="Multiplier Test Check", datasets=["ds1"])
def multiplier_test_check(mp: MetricProvider, ctx: Context) -> None:
    """Check designed to fail without multiplier, pass with multiplier=2.0.

    Uses a threshold between the raw metric and the scaled metric.
    The threshold is chosen so raw_value < threshold < raw_value * 2.0
    """
    tax_avg = mp.average("tax")
    # Tax average with this seed is around 50-80
    # Threshold 80: raw ~70 < 80 = FAILED, scaled ~140 > 80 = PASSED
    ctx.assert_that(tax_avg).where(name="Tax average exceeds threshold", tags={"multiplier-test"}).is_gt(80)


def test_e2e_profile_metric_multiplier_effect() -> None:
    """Test that metric_multiplier actually changes assertion outcomes.

    This test demonstrates that a profile with metric_multiplier can change
    a failing assertion to passing by scaling the metric value before validation.
    """
    db = InMemoryMetricDB()

    ds1 = CommercialDataSource(
        start_date=dt.date(2025, 1, 1),
        end_date=dt.date(2025, 1, 31),
        name="ds1",
        records_per_day=30,
        seed=1050,
        skip_dates={dt.date.fromisoformat("2025-01-13")},
    )

    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})

    # First run WITHOUT profile - assertion should FAIL
    # (tax average ~100 is not > 150)
    suite_no_profile = VerificationSuite(
        [multiplier_test_check],
        db,
        name="No profile suite",
        data_av_threshold=0.8,
        profiles=[],
    )
    suite_no_profile.run([ds1], key)
    results_no_profile = suite_no_profile.collect_results()

    # Second run WITH profile - multiplier=2.0 should make assertion PASS
    # (tax average ~100 * 2.0 = ~200 > 150)
    db2 = InMemoryMetricDB()
    profile_with_multiplier = HolidayProfile(
        name="Multiplier Test",
        start_date=dt.date(2025, 1, 1),
        end_date=dt.date(2025, 1, 31),
        rules=[
            tag("multiplier-test").set(metric_multiplier=2.0),
        ],
    )
    suite_with_profile = VerificationSuite(
        [multiplier_test_check],
        db2,
        name="With profile suite",
        data_av_threshold=0.8,
        profiles=[profile_with_multiplier],
    )
    suite_with_profile.run([ds1], key)
    results_with_profile = suite_with_profile.collect_results()

    # Verify the multiplier actually changed the outcome
    assert len(results_no_profile) == 1
    assert len(results_with_profile) == 1
    assert results_no_profile[0].status == "FAILED", "Without multiplier, assertion should fail"
    assert results_with_profile[0].status == "PASSED", "With multiplier=2.0, assertion should pass"

    print("\nMetric multiplier effect test:")
    print(f"  Without profile: {results_no_profile[0].status} (metric not scaled)")
    print(f"  With profile (multiplier=2.0): {results_with_profile[0].status} (metric doubled)")
