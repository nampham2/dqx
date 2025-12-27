"""Tests for tunable constants."""

import datetime
from typing import Any

import pytest

from dqx.tunables import (
    TunableChoice,
    TunableFloat,
    TunableInt,
    TunablePercent,
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


class TestTunablePercent:
    """Tests for TunablePercent."""

    def test_create_valid(self) -> None:
        """Can create TunablePercent with valid value."""
        t = TunablePercent("threshold", value=0.05, bounds=(0.0, 0.20))
        assert t.value == 0.05
        assert t.bounds == (0.0, 0.20)

    def test_validation_error_message_shows_percentage(self) -> None:
        """Error message shows percentages for readability."""
        t = TunablePercent("x", value=0.05, bounds=(0.0, 0.10))
        with pytest.raises(ValueError, match=r"15\.0%.*outside bounds.*0\.0%.*10\.0%"):
            t.set(0.15)

    def test_to_dict(self) -> None:
        """to_dict returns correct structure with percent type."""
        t = TunablePercent("x", value=0.05, bounds=(0.0, 0.20))
        d = t.to_dict()
        assert d["type"] == "percent"


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

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=1000, bounds=(100, 10000))

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[threshold, min_rows],
        )

        params = suite.get_tunable_params()
        assert len(params) == 2
        assert params[0]["name"] == "THRESHOLD"
        assert params[1]["name"] == "MIN_ROWS"

    def test_get_param(self) -> None:
        """Can get individual param value."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[threshold],
        )

        assert suite.get_param("THRESHOLD") == 0.05

    def test_get_param_not_found(self) -> None:
        """KeyError raised for unknown param."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[],
        )

        with pytest.raises(KeyError, match="not found"):
            suite.get_param("UNKNOWN")

    def test_set_param(self) -> None:
        """
        Verifies that a tunable parameter can be updated via a VerificationSuite and its new value retrieved.

        Creates a VerificationSuite with a TunablePercent, calls set_param to change the parameter (including agent and reason), and asserts get_param returns the updated value.
        """
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[threshold],
        )

        suite.set_param("THRESHOLD", 0.10, agent="rl_optimizer", reason="Test")
        assert suite.get_param("THRESHOLD") == 0.10

    def test_set_param_validates_bounds(self) -> None:
        """set_param validates value is within bounds."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[threshold],
        )

        with pytest.raises(ValueError, match="outside bounds"):
            suite.set_param("THRESHOLD", 0.50)

    def test_get_param_history(self) -> None:
        """Can get param change history."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[threshold],
        )

        suite.set_param("THRESHOLD", 0.10, agent="rl_optimizer", reason="Episode 1")
        suite.set_param("THRESHOLD", 0.15, agent="rl_optimizer", reason="Episode 2")

        history = suite.get_param_history("THRESHOLD")
        assert len(history) == 2
        assert history[0].new_value == 0.10
        assert history[1].new_value == 0.15

    def test_duplicate_tunable_name_raises(self) -> None:
        """
        Verifies that constructing a VerificationSuite with two tunables that share the same name raises an error.

        Raises:
            DQXError: with message "Duplicate tunable name" when duplicate tunable names are provided to VerificationSuite.
        """
        from dqx.api import VerificationSuite, check
        from dqx.common import DQXError
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        t1 = TunableFloat("X", value=0.5, bounds=(0.0, 1.0))
        t2 = TunableFloat("X", value=0.3, bounds=(0.0, 1.0))  # Duplicate name

        with pytest.raises(DQXError, match="Duplicate tunable name"):
            VerificationSuite(
                checks=[test_check],
                db=db,
                name="Test Suite",
                tunables=[t1, t2],
            )

    def test_set_param_not_found(self) -> None:
        """KeyError raised when setting unknown param."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[],
        )

        with pytest.raises(KeyError, match="not found"):
            suite.set_param("UNKNOWN", 0.5)

    def test_get_param_history_not_found(self) -> None:
        """KeyError raised when getting history for unknown param."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        @check(name="Test Check")
        def test_check(mp: Any, ctx: Any) -> None:
            pass

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[],
        )

        with pytest.raises(KeyError, match="not found"):
            suite.get_param_history("UNKNOWN")


class TestTunablePercentValidation:
    """Additional tests for TunablePercent validation."""

    def test_invalid_bounds_order(self) -> None:
        """Raises error if min > max."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            TunablePercent("x", value=0.05, bounds=(0.20, 0.0))


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
