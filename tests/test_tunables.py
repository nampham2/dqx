"""Tests for tunable constants."""

import datetime
from typing import Any

import pytest

from dqx.tunables import (
    TunableChoice,
    TunableInt,
    TunableFloat,
)


class TestTunableFloat:
    """Tests for TunableFloat."""

    def test_create_valid(self) -> None:
        """Can create TunableFloat with valid value within bounds."""
        t = TunableFloat("tolerance", value=0.5, bounds=(0.0, 1.0))
        assert t.name == "tolerance"
        assert t.value == 0.5
        assert t.bounds == (0.0, 1.0)

    def test_invalid_bounds_order(self) -> None:
        """Raises error if min > max."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            TunableFloat("x", value=0.5, bounds=(1.0, 0.0))

    def test_value_outside_bounds(self) -> None:
        """Raises error if initial value outside bounds."""
        with pytest.raises(ValueError, match="outside bounds"):
            TunableFloat("x", value=2.0, bounds=(0.0, 1.0))

    def test_set_valid_value(self) -> None:
        """Can set value within bounds."""
        t = TunableFloat("x", value=0.5, bounds=(0.0, 1.0))
        t.set(0.8)
        assert t.value == 0.8

    def test_set_invalid_value(self) -> None:
        """Raises error when setting value outside bounds."""
        t = TunableFloat("x", value=0.5, bounds=(0.0, 1.0))
        with pytest.raises(ValueError, match="outside bounds"):
            t.set(1.5)

    def test_to_dict(self) -> None:
        """to_dict returns correct structure."""
        t = TunableFloat("x", value=0.5, bounds=(0.0, 1.0))
        d = t.to_dict()
        assert d == {
            "name": "x",
            "type": "float",
            "value": 0.5,
            "bounds": (0.0, 1.0),
        }


class TestTunableInt:
    """Tests for TunableInt."""

    def test_create_valid(self) -> None:
        """Can create TunableInt with valid value."""
        t = TunableInt("min_rows", value=1000, bounds=(100, 10000))
        assert t.value == 1000
        assert t.bounds == (100, 10000)

    def test_rejects_float(self) -> None:
        """Raises TypeError for float value."""
        t = TunableInt("x", value=100, bounds=(0, 200))
        with pytest.raises(TypeError, match="must be int"):
            t.set(100.5)  # type: ignore[arg-type]

    def test_rejects_bool(self) -> None:
        """Raises TypeError for bool value (even though bool is subclass of int)."""
        t = TunableInt("x", value=1, bounds=(0, 10))
        with pytest.raises(TypeError, match="must be int"):
            t.set(True)  # type: ignore[arg-type]

    def test_to_dict(self) -> None:
        """to_dict returns correct structure."""
        t = TunableInt("x", value=100, bounds=(0, 200))
        d = t.to_dict()
        assert d == {
            "name": "x",
            "type": "int",
            "value": 100,
            "bounds": (0, 200),
        }


class TestTunableChoice:
    """Tests for TunableChoice."""

    def test_create_valid(self) -> None:
        """Can create TunableChoice with valid value."""
        t = TunableChoice("method", value="mean", choices=("mean", "median", "max"))
        assert t.value == "mean"
        assert t.choices == ("mean", "median", "max")

    def test_empty_choices_raises(self) -> None:
        """Raises error if choices is empty."""
        with pytest.raises(ValueError, match="choices cannot be empty"):
            TunableChoice("x", value="a", choices=())

    def test_invalid_choice_raises(self) -> None:
        """Raises error if value not in choices."""
        with pytest.raises(ValueError, match="not in choices"):
            TunableChoice("x", value="invalid", choices=("a", "b", "c"))

    def test_set_valid_choice(self) -> None:
        """Can set value to another valid choice."""
        t = TunableChoice("x", value="a", choices=("a", "b", "c"))
        t.set("b")
        assert t.value == "b"

    def test_set_invalid_choice(self) -> None:
        """Raises error when setting invalid choice."""
        t = TunableChoice("x", value="a", choices=("a", "b", "c"))
        with pytest.raises(ValueError, match="not in choices"):
            t.set("invalid")

    def test_to_dict(self) -> None:
        """to_dict returns correct structure."""
        t = TunableChoice("x", value="a", choices=("a", "b", "c"))
        d = t.to_dict()
        assert d == {
            "name": "x",
            "type": "choice",
            "value": "a",
            "choices": ("a", "b", "c"),
        }


class TestTunableHistory:
    """Tests for tunable change history tracking."""

    def test_history_starts_empty(self) -> None:
        """New tunable has no history."""
        t = TunableFloat("x", value=0.5, bounds=(0.0, 1.0))
        assert t.history == []

    def test_set_records_history(self) -> None:
        """Setting value records change in history."""
        t = TunableFloat("x", value=0.5, bounds=(0.0, 1.0))
        t.set(0.8, agent="rl_optimizer", reason="Episode 42")

        assert len(t.history) == 1
        change = t.history[0]
        assert change.old_value == 0.5
        assert change.new_value == 0.8
        assert change.agent == "rl_optimizer"
        assert change.reason == "Episode 42"
        assert isinstance(change.timestamp, datetime.datetime)

    def test_multiple_changes_recorded(self) -> None:
        """Multiple changes are all recorded."""
        t = TunableFloat("x", value=0.5, bounds=(0.0, 1.0))
        t.set(0.6, agent="human")
        t.set(0.7, agent="rl_optimizer")
        t.set(0.8, agent="autotuner")

        assert len(t.history) == 3
        assert [c.new_value for c in t.history] == [0.6, 0.7, 0.8]
        assert [c.agent for c in t.history] == ["human", "rl_optimizer", "autotuner"]

    def test_history_is_direct_access(self) -> None:
        """History field is directly accessible."""
        t = TunableFloat("x", value=0.5, bounds=(0.0, 1.0))
        t.set(0.6)
        assert len(t.history) == 1
        assert t.history[0].new_value == 0.6


class TestVerificationSuiteTunables:
    """Tests for VerificationSuite tunable integration."""

    def test_suite_with_tunables(self) -> None:
        """Suite can be created with tunables."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=1000, bounds=(100, 10000))

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="threshold_test").is_gt(0)
            ctx.assert_that(x - min_rows).where(name="min_rows_test").is_gt(0)

        db = InMemoryMetricDB()

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        params = suite.get_tunable_params()
        assert len(params) == 2
        param_names = {p["name"] for p in params}
        assert param_names == {"THRESHOLD", "MIN_ROWS"}

    def test_get_param(self) -> None:
        """Can get individual param value."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        db = InMemoryMetricDB()

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        assert suite.get_param("THRESHOLD") == 0.05

    def test_get_param_not_found(self) -> None:
        """KeyError raised for unknown param."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            ctx.assert_that(mp.num_rows()).where(name="test").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        with pytest.raises(KeyError, match="not found"):
            suite.get_param("UNKNOWN")

    def test_set_param(self) -> None:
        """
        Verifies that a tunable parameter can be updated via a VerificationSuite and its new value retrieved.

        Creates a VerificationSuite with a TunableFloat, calls set_param to change the parameter (including agent and reason), and asserts get_param returns the updated value.
        """
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        db = InMemoryMetricDB()

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        suite.set_param("THRESHOLD", 0.10, agent="rl_optimizer", reason="Test")
        assert suite.get_param("THRESHOLD") == 0.10

    def test_set_param_validates_bounds(self) -> None:
        """set_param validates value is within bounds."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        db = InMemoryMetricDB()

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        with pytest.raises(ValueError, match="outside bounds"):
            suite.set_param("THRESHOLD", 0.50)

    def test_get_param_history(self) -> None:
        """Can get param change history."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        db = InMemoryMetricDB()

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        suite.set_param("THRESHOLD", 0.10, agent="rl_optimizer", reason="Episode 1")
        suite.set_param("THRESHOLD", 0.15, agent="rl_optimizer", reason="Episode 2")

        history = suite.get_param_history("THRESHOLD")
        assert len(history) == 2
        assert history[0].new_value == 0.10
        assert history[1].new_value == 0.15

    def test_duplicate_tunable_name_deduplicates(self) -> None:
        """
        Verifies that SymPy deduplicates tunables by name, so only 1 tunable is discovered when two instances have the same name.
        """
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        t1 = TunableFloat("X", value=0.5, bounds=(0.0, 1.0))
        t2 = TunableFloat("X", value=0.3, bounds=(0.0, 1.0))  # Duplicate name

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            x = mp.num_rows()
            # Use both tunables in the check - SymPy should deduplicate by name
            ctx.assert_that(x - t1).where(name="test1").is_gt(0)
            ctx.assert_that(x - t2).where(name="test2").is_gt(0)

        db = InMemoryMetricDB()

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        # SymPy deduplicates by name, so only 1 tunable should be discovered
        params = suite.get_tunable_params()
        assert len(params) == 1
        assert params[0]["name"] == "X"

    def test_set_param_not_found(self) -> None:
        """KeyError raised when setting unknown param."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            ctx.assert_that(mp.num_rows()).where(name="test").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        with pytest.raises(KeyError, match="not found"):
            suite.set_param("UNKNOWN", 0.5)

    def test_get_param_history_not_found(self) -> None:
        """KeyError raised when getting history for unknown param."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            ctx.assert_that(mp.num_rows()).where(name="test").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        with pytest.raises(KeyError, match="not found"):
            suite.get_param_history("UNKNOWN")


class TestTunableFloatValidation:
    """Additional tests for TunableFloat validation."""

    def test_invalid_bounds_order(self) -> None:
        """Raises error if min > max."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            TunableFloat("x", value=0.05, bounds=(0.20, 0.0))


class TestTunableIntValidation:
    """Additional tests for TunableInt validation."""

    def test_invalid_bounds_order(self) -> None:
        """Raises error if min > max."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            TunableInt("x", value=50, bounds=(100, 0))

    def test_value_outside_bounds_on_set(self) -> None:
        """Raises error when setting value outside bounds."""
        t = TunableInt("x", value=50, bounds=(0, 100))
        with pytest.raises(ValueError, match="outside bounds"):
            t.set(150)


class TestTunableRuntimeBehavior:
    """Tests that verify tunables actually affect assertion behavior at runtime."""

    def test_set_param_changes_assertion_threshold_at_runtime(self) -> None:
        """Verify that set_param actually changes the threshold used in running checks."""
        import datetime as dt

        from dqx.api import VerificationSuite, check
        from dqx.common import Context, ResultKey
        from dqx.orm.repositories import InMemoryMetricDB
        from dqx.provider import MetricProvider
        from tests.fixtures.data_fixtures import CommercialDataSource

        # Set up test data (seed=1050 produces ~25.8% null rate with 31 rows)
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1050,
        )

        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Create tunable with initial threshold (20%) - should FAIL (null rate ~25.8% > 20%)
        null_threshold = TunableFloat("NULL_THRESHOLD", value=0.20, bounds=(0.0, 0.50))

        @check(name="Null Rate Check", datasets=["orders"])
        def null_rate_check(mp: MetricProvider, ctx: Context) -> None:
            null_rate = mp.null_count("delivered") / mp.num_rows()
            # Use tunable directly in expression (not .value)
            ctx.assert_that(null_rate - null_threshold).where(name="null_rate_assertion").is_lt(0)

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[null_rate_check],
            db=db,
            name="Test Suite",
        )

        # Run with initial threshold (20%) - should FAIL
        suite.run([ds], key)
        result1 = suite.collect_results()
        initial_status = result1[0].status

        # Use set_param to change threshold to 30% (more lenient) - should PASS (null rate ~25.8% < 30%)
        suite.set_param("NULL_THRESHOLD", 0.30, agent="test", reason="Relax threshold")
        suite.reset()
        suite.run([ds], key)
        result2 = suite.collect_results()
        updated_status = result2[0].status

        # Verify the threshold change affected the assertion result
        assert initial_status == "FAILED", "Initial check should fail with 20% threshold"
        assert updated_status == "PASSED", "Updated check should pass with 30% threshold"
