"""Tests for configuration file loading."""

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from dqx.api import VerificationSuite, check
from dqx.common import Context, DQXError, ResultKey
from dqx.config import apply_tunables_from_config, load_config, validate_config_structure
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.tunables import Tunable, TunableChoice, TunableFloat, TunableInt, TunablePercent
from tests.fixtures.data_fixtures import CommercialDataSource


class TestConfigLoading:
    """Tests for basic config file loading and parsing."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Load and parse valid YAML configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables:
  THRESHOLD: 0.05
  MIN_ROWS: 1000
"""
        )

        config = load_config(config_file)
        assert config["tunables"]["THRESHOLD"] == 0.05
        assert config["tunables"]["MIN_ROWS"] == 1000

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        """Raise DQXError if config file does not exist."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(DQXError, match="Configuration file not found"):
            load_config(config_file)

    def test_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Raise DQXError for malformed YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("tunables:\n  THRESHOLD: [invalid yaml")

        with pytest.raises(DQXError, match="Invalid YAML syntax"):
            load_config(config_file)

    def test_config_empty_file(self, tmp_path: Path) -> None:
        """Raise DQXError for empty file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(DQXError, match="Configuration file is empty"):
            load_config(config_file)

    def test_config_not_dict(self, tmp_path: Path) -> None:
        """Raise DQXError if config is not a dictionary."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(DQXError, match="must contain a YAML dictionary"):
            load_config(config_file)

    def test_config_missing_tunables_section(self, tmp_path: Path) -> None:
        """Raise DQXError if 'tunables' key is missing."""
        config_file = tmp_path / "no_tunables.yaml"
        config_file.write_text("other_section:\n  key: value")

        config = load_config(config_file)
        with pytest.raises(DQXError, match="must contain 'tunables' key"):
            validate_config_structure(config)

    def test_config_tunables_not_dict(self, tmp_path: Path) -> None:
        """Raise DQXError if tunables section is not a dictionary."""
        config_file = tmp_path / "bad_tunables.yaml"
        config_file.write_text("tunables:\n  - item")

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'tunables' section must be a dictionary"):
            validate_config_structure(config)

    def test_config_empty_tunables_section(self, tmp_path: Path) -> None:
        """Allow empty tunables section (no-op)."""
        config_file = tmp_path / "empty_tunables.yaml"
        config_file.write_text("tunables: {}")

        config = load_config(config_file)
        validate_config_structure(config)  # Should not raise
        assert config["tunables"] == {}

    def test_config_null_tunables_section(self, tmp_path: Path) -> None:
        """Allow null tunables section (no-op)."""
        config_file = tmp_path / "null_tunables.yaml"
        config_file.write_text("tunables:")

        config = load_config(config_file)
        validate_config_structure(config)  # Should not raise
        assert config["tunables"] is None


class TestApplyTunables:
    """Tests for applying tunables from config to suite tunables."""

    def test_apply_single_tunable(self) -> None:
        """Apply a single tunable value from config."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config = {"tunables": {"THRESHOLD": 0.05}}

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.05
        assert len(threshold.history) == 1
        assert threshold.history[0].agent == "config"

    def test_apply_multiple_tunables(self) -> None:
        """Apply multiple tunable values from config."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=100, bounds=(0, 10000))
        tunables: dict[str, Tunable[Any]] = {"THRESHOLD": threshold, "MIN_ROWS": min_rows}
        config = {"tunables": {"THRESHOLD": 0.05, "MIN_ROWS": 1000}}

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.05
        assert min_rows.value == 1000

    def test_apply_partial_tunables(self) -> None:
        """Config with subset of tunables should work (partial override)."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=100, bounds=(0, 10000))
        tunables: dict[str, Tunable[Any]] = {"THRESHOLD": threshold, "MIN_ROWS": min_rows}
        config = {"tunables": {"THRESHOLD": 0.05}}  # Only set THRESHOLD

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.05
        assert min_rows.value == 100  # Unchanged

    def test_apply_extra_tunable_raises(self) -> None:
        """Config with tunables not in suite should raise DQXError."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config = {"tunables": {"THRESHOLD": 0.05, "UNKNOWN": 999}}

        # Should raise DQXError for unknown tunable
        with pytest.raises(DQXError, match="Configuration contains tunable 'UNKNOWN' not found in suite"):
            apply_tunables_from_config(config, tunables)

    def test_apply_invalid_tunable_value(self) -> None:
        """Invalid tunable value should raise ValueError."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config = {"tunables": {"THRESHOLD": 0.50}}  # Outside bounds

        with pytest.raises(ValueError, match="Invalid value for tunable 'THRESHOLD'"):
            apply_tunables_from_config(config, tunables)

    def test_apply_all_tunable_types(self) -> None:
        """Test all tunable types: Float, Percent, Int, Choice."""
        t_float = TunableFloat("FLOAT", value=0.5, bounds=(0.0, 1.0))
        t_percent = TunablePercent("PERCENT", value=0.05, bounds=(0.0, 0.20))
        t_int = TunableInt("INT", value=100, bounds=(0, 1000))
        t_choice = TunableChoice("CHOICE", value="mean", choices=("mean", "median", "max"))

        tunables: dict[str, Tunable[Any]] = {"FLOAT": t_float, "PERCENT": t_percent, "INT": t_int, "CHOICE": t_choice}
        config = {"tunables": {"FLOAT": 0.8, "PERCENT": 0.10, "INT": 500, "CHOICE": "median"}}

        apply_tunables_from_config(config, tunables)

        assert t_float.value == 0.8
        assert t_percent.value == 0.10
        assert t_int.value == 500
        assert t_choice.value == "median"

    def test_apply_empty_config_tunables(self) -> None:
        """Empty tunables config should be a no-op."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config: dict[str, Any] = {"tunables": {}}

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.01  # Unchanged

    def test_apply_null_config_tunables(self) -> None:
        """Null tunables config should be a no-op."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config = {"tunables": None}

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.01  # Unchanged


class TestSuiteWithConfig:
    """Tests for VerificationSuite with config file parameter."""

    def test_suite_with_config_file(self, tmp_path: Path) -> None:
        """Create suite with config file, verify tunables are set."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=100, bounds=(0, 10000))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="threshold_test").is_gt(0)
            ctx.assert_that(x - min_rows).where(name="min_rows_test").is_gt(0)

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables:
  THRESHOLD: 0.05
  MIN_ROWS: 1000
"""
        )

        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=[test_check], db=db, name="Test Suite", config=config_file)

        # Verify tunables were set from config
        assert suite.get_param("THRESHOLD") == 0.05
        assert suite.get_param("MIN_ROWS") == 1000

    def test_suite_without_config_file(self) -> None:
        """Create suite without config file, use default tunable values."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=[test_check], db=db, name="Test Suite", config=None)

        # Verify default value
        assert suite.get_param("THRESHOLD") == 0.01

    def test_suite_config_partial_tunables(self, tmp_path: Path) -> None:
        """Config with subset of tunables should work."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=100, bounds=(0, 10000))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="t1").is_gt(0)
            ctx.assert_that(x - min_rows).where(name="t2").is_gt(0)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("tunables:\n  THRESHOLD: 0.05")

        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=[test_check], db=db, name="Test Suite", config=config_file)

        assert suite.get_param("THRESHOLD") == 0.05
        assert suite.get_param("MIN_ROWS") == 100  # Default unchanged

    def test_suite_config_invalid_file(self, tmp_path: Path) -> None:
        """Invalid config file should raise DQXError during suite construction."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("tunables:\n  THRESHOLD: [bad")

        db = InMemoryMetricDB()
        with pytest.raises(DQXError, match="Invalid YAML syntax"):
            VerificationSuite(checks=[test_check], db=db, name="Test Suite", config=config_file)


class TestConfigResetCompatibility:
    """Tests for config compatibility with reset() operation."""

    def test_config_preserved_after_reset_unchanged(self, tmp_path: Path) -> None:
        """After reset(), tunables retain their current values (config not reloaded)."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("tunables:\n  THRESHOLD: 0.05")

        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=[test_check], db=db, name="Test Suite", config=config_file)

        assert suite.get_param("THRESHOLD") == 0.05

        # Reset without changing tunable
        suite.reset()

        # Tunable should still be 0.05 (not reloaded from config)
        assert suite.get_param("THRESHOLD") == 0.05

    def test_set_param_survives_reset(self, tmp_path: Path) -> None:
        """Tunable modified via set_param() should retain value after reset()."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("tunables:\n  THRESHOLD: 0.05")

        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=[test_check], db=db, name="Test Suite", config=config_file)

        # Initial value from config
        assert suite.get_param("THRESHOLD") == 0.05

        # Modify via set_param
        suite.set_param("THRESHOLD", 0.15, agent="test", reason="Testing")

        # Value should be updated
        assert suite.get_param("THRESHOLD") == 0.15

        # Reset
        suite.reset()

        # Modified value should be preserved (config NOT reloaded)
        assert suite.get_param("THRESHOLD") == 0.15

    def test_reset_does_not_reload_config(self, tmp_path: Path) -> None:
        """reset() should not reload config file from disk."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="test").is_gt(0)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("tunables:\n  THRESHOLD: 0.05")

        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=[test_check], db=db, name="Test Suite", config=config_file)

        assert suite.get_param("THRESHOLD") == 0.05

        # Modify the config file on disk
        config_file.write_text("tunables:\n  THRESHOLD: 0.10")

        # Reset suite
        suite.reset()

        # Value should still be 0.05 (not reloaded from disk)
        assert suite.get_param("THRESHOLD") == 0.05


class TestConfigIntegration:
    """Integration tests with full suite execution."""

    def test_end_to_end_with_config(self, tmp_path: Path) -> None:
        """Full workflow: config → run → set_param → reset → run."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.50))

        @check(name="Null Rate Check", datasets=["orders"])
        def null_rate_check(mp: MetricProvider, ctx: Context) -> None:
            null_rate = mp.null_count("delivered") / mp.num_rows()
            ctx.assert_that(null_rate - threshold).where(name="null_rate_assertion").is_lt(0)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("tunables:\n  THRESHOLD: 0.20")

        # Set up test data (seed=1050 produces ~25.8% null rate)
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1050,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=[null_rate_check], db=db, name="Test Suite", config=config_file)

        # Run 1: Should FAIL (null rate ~25.8% > threshold 20%)
        suite.run([ds], key)
        result1 = suite.collect_results()[0]
        assert result1.status == "FAILED"

        # Modify threshold
        suite.set_param("THRESHOLD", 0.30, agent="test")
        suite.reset()

        # Run 2: Should PASS (null rate ~25.8% < threshold 30%)
        suite.run([ds], key)
        result2 = suite.collect_results()[0]
        assert result2.status == "PASSED"

    def test_config_changes_assertion_behavior(self, tmp_path: Path) -> None:
        """Verify config actually affects check results."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.50))

        @check(name="Null Rate Check", datasets=["orders"])
        def null_rate_check(mp: MetricProvider, ctx: Context) -> None:
            null_rate = mp.null_count("delivered") / mp.num_rows()
            ctx.assert_that(null_rate - threshold).where(name="null_rate_assertion").is_lt(0)

        # Config with strict threshold (should fail)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("tunables:\n  THRESHOLD: 0.20")

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1050,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        db = InMemoryMetricDB()
        suite = VerificationSuite(checks=[null_rate_check], db=db, name="Test Suite", config=config_file)

        suite.run([ds], key)
        result = suite.collect_results()[0]
        assert result.status == "FAILED"


class TestEdgeCases:
    """Edge case tests for config loading."""

    def test_config_with_no_tunables_in_suite(self, tmp_path: Path) -> None:
        """Suite with no tunables should raise DQXError for unknown tunables in config."""

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            # No tunables used
            ctx.assert_that(mp.num_rows()).where(name="test").is_positive()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("tunables:\n  UNKNOWN: 0.05")

        db = InMemoryMetricDB()
        # Should raise DQXError for unknown tunable
        with pytest.raises(DQXError, match="Configuration contains tunable 'UNKNOWN' not found in suite"):
            VerificationSuite(checks=[test_check], db=db, name="Test Suite", config=config_file)
