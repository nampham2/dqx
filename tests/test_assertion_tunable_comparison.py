"""Tests for comparing metrics to tunables in assertions."""

import datetime

import pyarrow as pa
import pytest

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.tunables import TunableFloat, TunableInt, TunablePercent


@pytest.fixture
def db() -> InMemoryMetricDB:
    """Create an in-memory metric database."""
    return InMemoryMetricDB()


@pytest.fixture
def datasource() -> DuckRelationDataSource:
    """Create a test datasource with sample data."""
    data = pa.table(
        {
            "quantity": [5, 10, 15, 20, 25],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "score": [75.0, 80.0, 85.0, 90.0, 95.0],
            "count": [100, 150, 200, 250, 300],
        }
    )
    return DuckRelationDataSource.from_arrow(data, "test_data")


@pytest.fixture
def key() -> ResultKey:
    """Create a test result key."""
    return ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})


class TestIsLeqWithTunable:
    """Tests for is_leq with tunable thresholds."""

    def test_is_leq_with_tunable_float_pass(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_leq passes when metric value is less than or equal to tunable threshold."""
        MAX_QTY = TunableFloat("MAX_QTY", value=30.0, bounds=(0.0, 100.0))

        @check(name="Quantity Check")
        def quantity_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.maximum("quantity")).where(name="Max quantity").is_leq(MAX_QTY)

        suite = VerificationSuite([quantity_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert len(results) == 1
        assert results[0].status == "PASSED"
        assert results[0].metric.unwrap() == 25.0  # Original metric value preserved

    def test_is_leq_with_tunable_float_fail(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_leq fails when metric value exceeds tunable threshold."""
        MAX_QTY = TunableFloat("MAX_QTY", value=20.0, bounds=(0.0, 100.0))

        @check(name="Quantity Check")
        def quantity_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.maximum("quantity")).where(name="Max quantity").is_leq(MAX_QTY)

        suite = VerificationSuite([quantity_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert len(results) == 1
        assert results[0].status == "FAILED"
        assert results[0].metric.unwrap() == 25.0  # Original metric value preserved

    def test_is_leq_lazy_evaluation(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_leq evaluates tunable value lazily at validation time."""
        MAX_QTY = TunableFloat("MAX_QTY", value=30.0, bounds=(0.0, 100.0))

        @check(name="Quantity Check")
        def quantity_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.maximum("quantity")).where(name="Max quantity").is_leq(MAX_QTY)

        # Run 1: MAX_QTY = 30, maximum = 25 -> PASS
        suite = VerificationSuite([quantity_check], db, "Test Suite")
        suite.run([datasource], key)
        results1 = suite.collect_results()
        assert results1[0].status == "PASSED"

        # Tune threshold and reset
        MAX_QTY.set(20.0)
        suite2 = VerificationSuite([quantity_check], db, "Test Suite 2")

        # Run 2: MAX_QTY = 20, maximum = 25 -> FAIL (lazy evaluation working!)
        suite2.run([datasource], key)
        results2 = suite2.collect_results()
        assert results2[0].status == "FAILED"
        assert results2[0].metric.unwrap() == 25.0  # Same metric value

    def test_is_leq_with_static_float_unchanged(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_leq with static float still works as before."""

        @check(name="Quantity Check")
        def quantity_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.maximum("quantity")).where(name="Max quantity").is_leq(30.0)

        suite = VerificationSuite([quantity_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert len(results) == 1
        assert results[0].status == "PASSED"
        assert results[0].metric.unwrap() == 25.0


class TestIsGeqWithTunable:
    """Tests for is_geq with tunable thresholds."""

    def test_is_geq_with_tunable_percent_pass(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_geq passes when metric value meets or exceeds tunable threshold."""
        MIN_SCORE = TunablePercent("MIN_SCORE", value=0.70, bounds=(0.0, 1.0))

        @check(name="Score Check")
        def score_check(mp: MetricProvider, ctx: Context) -> None:
            # Average score is 85, which as decimal is 85.0
            # We compare: 85.0 >= 0.70 (tunable stores fraction, not percentage)
            ctx.assert_that(mp.average("score")).where(name="Min score").is_geq(MIN_SCORE)

        suite = VerificationSuite([score_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert len(results) == 1
        assert results[0].status == "PASSED"

    def test_is_geq_with_tunable_percent_fail(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_geq fails when metric value is below tunable threshold."""
        MIN_PRICE = TunableFloat("MIN_PRICE", value=35.0, bounds=(0.0, 100.0))

        @check(name="Price Check")
        def price_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("price")).where(name="Min price").is_geq(MIN_PRICE)

        suite = VerificationSuite([price_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert len(results) == 1
        assert results[0].status == "FAILED"  # average price is 30.0, MIN_PRICE is 35.0


class TestIsGtWithTunable:
    """Tests for is_gt with tunable thresholds."""

    def test_is_gt_with_tunable_pass(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_gt passes when metric value is strictly greater than tunable threshold."""
        THRESHOLD = TunableFloat("THRESHOLD", value=20.0, bounds=(0.0, 100.0))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("price")).where(name="Price check").is_gt(THRESHOLD)

        suite = VerificationSuite([test_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"  # average price is 30.0 > 20.0


class TestIsLtWithTunable:
    """Tests for is_lt with tunable thresholds."""

    def test_is_lt_with_tunable_pass(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_lt passes when metric value is strictly less than tunable threshold."""
        THRESHOLD = TunableFloat("THRESHOLD", value=40.0, bounds=(0.0, 100.0))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("price")).where(name="Price check").is_lt(THRESHOLD)

        suite = VerificationSuite([test_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"  # average price is 30.0 < 40.0


class TestIsEqWithTunable:
    """Tests for is_eq with tunable thresholds."""

    def test_is_eq_with_tunable_int_pass(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_eq passes when metric value equals tunable threshold within tolerance."""
        TARGET_COUNT = TunableInt("TARGET_COUNT", value=200, bounds=(0, 1000))

        @check(name="Count Check")
        def count_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("count")).where(name="Avg count").is_eq(TARGET_COUNT)

        suite = VerificationSuite([count_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"  # average count is 200.0

    def test_is_eq_with_tunable_int_fail(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_eq fails when metric value does not equal tunable threshold."""
        TARGET_COUNT = TunableInt("TARGET_COUNT", value=250, bounds=(0, 1000))

        @check(name="Count Check")
        def count_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("count")).where(name="Avg count").is_eq(TARGET_COUNT)

        suite = VerificationSuite([count_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "FAILED"  # average count is 200.0, not 250


class TestIsNeqWithTunable:
    """Tests for is_neq with tunable thresholds."""

    def test_is_neq_with_tunable_pass(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_neq passes when metric value differs from tunable threshold."""
        INVALID_VALUE = TunableFloat("INVALID_VALUE", value=100.0, bounds=(0.0, 200.0))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("price")).where(name="Price check").is_neq(INVALID_VALUE)

        suite = VerificationSuite([test_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"  # average price is 30.0 ≠ 100.0


class TestIsBetweenWithTunable:
    """Tests for is_between with tunable bounds."""

    def test_is_between_both_static_pass(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_between with static bounds works as before."""

        @check(name="Score Check")
        def score_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("score")).where(name="Score range").is_between(70.0, 100.0)

        suite = VerificationSuite([score_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"
        assert results[0].metric.unwrap() == 85.0

    def test_is_between_both_tunables_pass(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_between passes when metric is within tunable bounds."""
        LOWER = TunableFloat("LOWER", value=70.0, bounds=(0.0, 100.0))
        UPPER = TunableFloat("UPPER", value=100.0, bounds=(0.0, 100.0))

        @check(name="Score Check")
        def score_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("score")).where(name="Score range").is_between(LOWER, UPPER)

        suite = VerificationSuite([score_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"
        assert results[0].metric.unwrap() == 85.0  # Original metric preserved

    def test_is_between_both_tunables_fail(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_between fails when metric is outside tunable bounds."""
        LOWER = TunableFloat("LOWER", value=90.0, bounds=(0.0, 100.0))
        UPPER = TunableFloat("UPPER", value=100.0, bounds=(0.0, 100.0))

        @check(name="Score Check")
        def score_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("score")).where(name="Score range").is_between(LOWER, UPPER)

        suite = VerificationSuite([score_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "FAILED"  # average score is 85.0, not in [90, 100]

    def test_is_between_mixed_static_tunable_lower(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_between works with static lower bound and tunable upper bound."""
        UPPER = TunableFloat("UPPER", value=100.0, bounds=(0.0, 100.0))

        @check(name="Score Check")
        def score_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("score")).where(name="Score range").is_between(70.0, UPPER)

        suite = VerificationSuite([score_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"

    def test_is_between_mixed_tunable_lower_static(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_between works with tunable lower bound and static upper bound."""
        LOWER = TunableFloat("LOWER", value=70.0, bounds=(0.0, 100.0))

        @check(name="Score Check")
        def score_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("score")).where(name="Score range").is_between(LOWER, 100.0)

        suite = VerificationSuite([score_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"

    def test_is_between_lazy_evaluation_upper_bound(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """is_between evaluates tunable bounds lazily."""
        LOWER = TunableFloat("LOWER", value=70.0, bounds=(0.0, 100.0))
        UPPER = TunableFloat("UPPER", value=100.0, bounds=(0.0, 100.0))

        @check(name="Score Check")
        def score_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("score")).where(name="Score range").is_between(LOWER, UPPER)

        # Run 1: average score = 85, bounds = [70, 100] -> PASS
        suite = VerificationSuite([score_check], db, "Test Suite")
        suite.run([datasource], key)
        results1 = suite.collect_results()
        assert results1[0].status == "PASSED"

        # Tune upper bound
        UPPER.set(80.0)
        suite2 = VerificationSuite([score_check], db, "Test Suite 2")

        # Run 2: average score = 85, bounds = [70, 80] -> FAIL (lazy evaluation!)
        suite2.run([datasource], key)
        results2 = suite2.collect_results()
        assert results2[0].status == "FAILED"

    def test_is_between_static_invalid_bounds_raises(self, db: InMemoryMetricDB) -> None:
        """is_between with static invalid bounds (lower > upper) raises ValueError immediately."""

        @check(name="Bad Check")
        def bad_check(mp: MetricProvider, ctx: Context) -> None:
            with pytest.raises(ValueError, match="Invalid range"):
                ctx.assert_that(mp.average("score")).where(name="Bad range").is_between(100.0, 70.0)

        # The suite can be created, but the check will raise when executed
        _suite = VerificationSuite([bad_check], db, "Test Suite")


class TestMultipleTunablesInSuite:
    """Tests for using multiple tunables in a single suite."""

    def test_multiple_tunables_different_assertions(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """Multiple tunables can be used in different assertions."""
        MIN_QTY = TunableFloat("MIN_QTY", value=5.0, bounds=(0.0, 50.0))
        MAX_PRICE = TunableFloat("MAX_PRICE", value=60.0, bounds=(0.0, 100.0))

        @check(name="Multi Check")
        def multi_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.minimum("quantity")).where(name="Min qty").is_geq(MIN_QTY)
            ctx.assert_that(mp.maximum("price")).where(name="Max price").is_leq(MAX_PRICE)

        suite = VerificationSuite([multi_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert len(results) == 2
        assert all(r.status == "PASSED" for r in results)

        # Note: Tunables in validator closures are not automatically collected by the graph
        # This is expected behavior - they work via closure capture for lazy evaluation


class TestTunableComparisonEdgeCases:
    """Tests for edge cases in tunable comparisons."""

    def test_tunable_at_exact_boundary(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """Comparison works correctly when metric equals tunable value exactly."""
        EXACT_VALUE = TunableFloat("EXACT_VALUE", value=25.0, bounds=(0.0, 100.0))

        @check(name="Exact Check")
        def exact_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.maximum("quantity")).where(name="Exact").is_leq(EXACT_VALUE)

        suite = VerificationSuite([exact_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert results[0].status == "PASSED"  # 25.0 <= 25.0 (within tolerance)

    def test_tunable_with_tolerance(
        self, db: InMemoryMetricDB, datasource: DuckRelationDataSource, key: ResultKey
    ) -> None:
        """Tolerance parameter works correctly with tunables."""
        TARGET = TunableFloat("TARGET", value=25.001, bounds=(0.0, 100.0))

        @check(name="Tolerance Check")
        def tolerance_check(mp: MetricProvider, ctx: Context) -> None:
            # maximum(quantity) = 25.0, TARGET = 25.001
            # With default tolerance (1e-9), should fail
            ctx.assert_that(mp.maximum("quantity")).where(name="Default tol").is_eq(TARGET)
            # With larger tolerance (0.01), should pass
            ctx.assert_that(mp.maximum("quantity")).where(name="Large tol").is_eq(TARGET, tol=0.01)

        suite = VerificationSuite([tolerance_check], db, "Test Suite")
        suite.run([datasource], key)
        results = suite.collect_results()

        assert len(results) == 2
        assert results[0].status == "FAILED"  # default tolerance too tight
        assert results[1].status == "PASSED"  # larger tolerance allows match
