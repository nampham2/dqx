"""Tests for configuration file loading."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from dqx.api import VerificationSuite, check
from dqx.common import Context, DQXError, ResultKey
from dqx.config import apply_tunables_from_config, load_config, validate_config_structure
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.tunables import Tunable, TunableChoice, TunableInt, TunableFloat
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

    def test_config_profiles_not_list(self, tmp_path: Path) -> None:
        """Raise DQXError if profiles section is not a list."""
        config_file = tmp_path / "bad_profiles.yaml"
        config_file.write_text("tunables: {}\nprofiles: not_a_list")

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'profiles' section must be a list"):
            validate_config_structure(config)


class TestApplyTunables:
    """Tests for applying tunables from config to suite tunables."""

    def test_apply_single_tunable(self) -> None:
        """Apply a single tunable value from config."""
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config = {"tunables": {"THRESHOLD": 0.05}}

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.05
        assert len(threshold.history) == 1
        assert threshold.history[0].agent == "config"

    def test_apply_multiple_tunables(self) -> None:
        """Apply multiple tunable values from config."""
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=100, bounds=(0, 10000))
        tunables: dict[str, Tunable[Any]] = {"THRESHOLD": threshold, "MIN_ROWS": min_rows}
        config = {"tunables": {"THRESHOLD": 0.05, "MIN_ROWS": 1000}}

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.05
        assert min_rows.value == 1000

    def test_apply_partial_tunables(self) -> None:
        """Config with subset of tunables should work (partial override)."""
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=100, bounds=(0, 10000))
        tunables: dict[str, Tunable[Any]] = {"THRESHOLD": threshold, "MIN_ROWS": min_rows}
        config = {"tunables": {"THRESHOLD": 0.05}}  # Only set THRESHOLD

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.05
        assert min_rows.value == 100  # Unchanged

    def test_apply_extra_tunable_raises(self) -> None:
        """Config with tunables not in suite should raise DQXError."""
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config = {"tunables": {"THRESHOLD": 0.05, "UNKNOWN": 999}}

        # Should raise DQXError for unknown tunable
        with pytest.raises(DQXError, match="Configuration contains tunable 'UNKNOWN' not found in suite"):
            apply_tunables_from_config(config, tunables)

    def test_apply_invalid_tunable_value(self) -> None:
        """Invalid tunable value should raise DQXError."""
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config = {"tunables": {"THRESHOLD": 0.50}}  # Outside bounds

        with pytest.raises(DQXError, match="Invalid value for tunable 'THRESHOLD'"):
            apply_tunables_from_config(config, tunables)

    def test_apply_all_tunable_types(self) -> None:
        """Test all tunable types: Float, Percent, Int, Choice."""
        t_float = TunableFloat("FLOAT", value=0.5, bounds=(0.0, 1.0))
        t_percent = TunableFloat("PERCENT", value=0.05, bounds=(0.0, 0.20))
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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config: dict[str, Any] = {"tunables": {}}

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.01  # Unchanged

    def test_apply_null_config_tunables(self) -> None:
        """Null tunables config should be a no-op."""
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
        tunables = {"THRESHOLD": threshold}
        config = {"tunables": None}

        apply_tunables_from_config(config, tunables)

        assert threshold.value == 0.01  # Unchanged


class TestSuiteWithConfig:
    """Tests for VerificationSuite with config file parameter."""

    def test_suite_with_config_file(self, tmp_path: Path) -> None:
        """Create suite with config file, verify tunables are set."""
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))
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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.20))

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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.50))

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
        threshold = TunableFloat("THRESHOLD", value=0.01, bounds=(0.0, 0.50))

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


class TestProfileLoading:
    """Tests for loading profiles from YAML configuration."""

    def test_load_single_profile_with_disable_rule(self, tmp_path: Path) -> None:
        """Load a profile with a single disable rule."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Holiday Season"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: "Volume"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert profiles[0].name == "Holiday Season"
        assert profiles[0].start_date == dt.date(2024, 12, 20)
        assert profiles[0].end_date == dt.date(2025, 1, 5)
        assert len(profiles[0].rules) == 1
        assert profiles[0].rules[0].disabled is True

    def test_load_profile_with_scale_rule(self, tmp_path: Path) -> None:
        """Load a profile with a scale rule."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Black Friday"
    type: "seasonal"
    start_date: "2024-11-29"
    end_date: "2024-11-29"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 3.5
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert profiles[0].rules[0].metric_multiplier == 3.5

    def test_load_profile_with_set_severity_rule(self, tmp_path: Path) -> None:
        """Load a profile with a set_severity rule."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Critical Period"
    type: "seasonal"
    start_date: "2024-12-31"
    end_date: "2025-01-01"
    rules:
      - action: "set_severity"
        target: "tag"
        name: "fraud"
        severity: "P0"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert profiles[0].rules[0].severity == "P0"

    def test_load_multiple_profiles(self, tmp_path: Path) -> None:
        """Load multiple profiles from config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Profile 1"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2024-12-25"
    rules:
      - action: "disable"
        target: "check"
        name: "Check1"

  - name: "Profile 2"
    type: "seasonal"
    start_date: "2025-01-01"
    end_date: "2025-01-05"
    rules:
      - action: "scale"
        target: "tag"
        name: "tag1"
        multiplier: 2.0
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 2
        assert profiles[0].name == "Profile 1"
        assert profiles[1].name == "Profile 2"

    def test_load_profile_with_multiple_rules(self, tmp_path: Path) -> None:
        """Load a profile with multiple rules."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Complex Profile"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: "Volume"
      - action: "scale"
        target: "tag"
        name: "reconciliation"
        multiplier: 1.5
      - action: "set_severity"
        target: "tag"
        name: "critical"
        severity: "P0"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert len(profiles[0].rules) == 3

    def test_load_empty_profiles_section(self, tmp_path: Path) -> None:
        """Empty profiles section should return empty list."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert profiles == []

    def test_load_no_profiles_section(self, tmp_path: Path) -> None:
        """Missing profiles section should return empty list."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables:
  THRESHOLD: 0.05
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert profiles == []

    def test_profile_missing_name(self, tmp_path: Path) -> None:
        """Profile without name should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'name' is required"):
            load_profiles_from_config(config)

    def test_profile_invalid_type(self, tmp_path: Path) -> None:
        """Profile with invalid type should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "invalid_type"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="unknown type 'invalid_type'"):
            load_profiles_from_config(config)

    def test_profile_type_defaults_to_seasonal(self, tmp_path: Path) -> None:
        """Profile without type should default to 'seasonal'."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config
        from dqx.profiles import SeasonalProfile

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert profiles[0].name == "Test"
        assert isinstance(profiles[0], SeasonalProfile)

    def test_profile_invalid_date_format(self, tmp_path: Path) -> None:
        """Profile with invalid date format should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "12/20/2024"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="invalid 'start_date' format"):
            load_profiles_from_config(config)

    def test_profile_end_before_start(self, tmp_path: Path) -> None:
        """Profile with end_date before start_date should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2025-01-05"
    end_date: "2024-12-20"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match=r"'end_date'.*must be.*'start_date'"):
            load_profiles_from_config(config)

    def test_duplicate_profile_names_in_config(self, tmp_path: Path) -> None:
        """Duplicate profile names within config should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Holiday"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2024-12-25"
    rules: []
  - name: "Holiday"
    type: "seasonal"
    start_date: "2025-01-01"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="Duplicate profile name 'Holiday' at index 1"):
            load_profiles_from_config(config)

    def test_rule_missing_action(self, tmp_path: Path) -> None:
        """Rule without action should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - target: "check"
        name: "Volume"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'action' is required"):
            load_profiles_from_config(config)

    def test_rule_invalid_action(self, tmp_path: Path) -> None:
        """Rule with invalid action should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "invalid_action"
        target: "check"
        name: "Volume"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="invalid action 'invalid_action'"):
            load_profiles_from_config(config)

    def test_rule_invalid_target(self, tmp_path: Path) -> None:
        """Rule with invalid target should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "assertion"
        name: "Volume"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="invalid target 'assertion'"):
            load_profiles_from_config(config)

    def test_scale_rule_missing_multiplier(self, tmp_path: Path) -> None:
        """Scale rule without multiplier should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'multiplier' is required"):
            load_profiles_from_config(config)

    def test_scale_rule_negative_multiplier(self, tmp_path: Path) -> None:
        """Scale rule with negative multiplier should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: -1.5
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'multiplier' must be positive"):
            load_profiles_from_config(config)

    def test_set_severity_rule_missing_severity(self, tmp_path: Path) -> None:
        """Set_severity rule without severity should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "set_severity"
        target: "tag"
        name: "fraud"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'severity' is required"):
            load_profiles_from_config(config)

    def test_set_severity_rule_invalid_severity(self, tmp_path: Path) -> None:
        """Set_severity rule with invalid severity should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "set_severity"
        target: "tag"
        name: "fraud"
        severity: "P4"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="invalid severity 'P4'"):
            load_profiles_from_config(config)

    def test_unknown_field_at_profile_level(self, tmp_path: Path) -> None:
        """Unknown field at profile level should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    unknown_field: "value"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match=r"unknown field.*unknown_field"):
            load_profiles_from_config(config)

    def test_unknown_field_at_rule_level(self, tmp_path: Path) -> None:
        """Unknown field at rule level should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: "Volume"
        unknown_field: "value"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match=r"unknown field.*unknown_field"):
            load_profiles_from_config(config)

    def test_rule_not_dict(self, tmp_path: Path) -> None:
        """Rule that's not a dict should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - "not a dict"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="rule at index 0 must be a dictionary"):
            load_profiles_from_config(config)

    def test_rule_missing_target(self, tmp_path: Path) -> None:
        """Rule without target should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        name: "Volume"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'target' is required"):
            load_profiles_from_config(config)

    def test_rule_missing_name(self, tmp_path: Path) -> None:
        """Rule without name should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'name' is required"):
            load_profiles_from_config(config)

    def test_rule_name_not_string(self, tmp_path: Path) -> None:
        """Rule with name that's not a string should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: 123
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'name' must be a string"):
            load_profiles_from_config(config)

    def test_scale_rule_multiplier_not_number(self, tmp_path: Path) -> None:
        """Scale rule with multiplier that's not a number should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: "not a number"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'multiplier' must be a number"):
            load_profiles_from_config(config)

    def test_profile_dict_not_dict(self, tmp_path: Path) -> None:
        """Profile that's not a dict should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - "not a dict"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="Profile at index 0 must be a dictionary"):
            load_profiles_from_config(config)

    def test_profile_name_not_string(self, tmp_path: Path) -> None:
        """Profile with name that's not a string should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: 123
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'name' must be a string"):
            load_profiles_from_config(config)

    def test_profile_missing_start_date(self, tmp_path: Path) -> None:
        """Profile without start_date should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'start_date' is required"):
            load_profiles_from_config(config)

    def test_profile_start_date_not_string(self, tmp_path: Path) -> None:
        """Profile with start_date that's not a string should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: 20241220
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'start_date' must be a string"):
            load_profiles_from_config(config)

    def test_profile_missing_end_date(self, tmp_path: Path) -> None:
        """Profile without end_date should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'end_date' is required"):
            load_profiles_from_config(config)

    def test_profile_end_date_not_string(self, tmp_path: Path) -> None:
        """Profile with end_date that's not a string should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: 20250105
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'end_date' must be a string"):
            load_profiles_from_config(config)

    def test_profile_invalid_end_date_format(self, tmp_path: Path) -> None:
        """Profile with invalid end_date format should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "01/05/2025"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="invalid 'end_date' format"):
            load_profiles_from_config(config)

    def test_profile_rules_not_list(self, tmp_path: Path) -> None:
        """Profile with rules that's not a list should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules: "not a list"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'rules' must be a list"):
            load_profiles_from_config(config)

    def test_load_permanent_profile_basic(self, tmp_path: Path) -> None:
        """Load a basic permanent profile."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Production Baseline"
    type: "permanent"
    rules:
      - action: "disable"
        target: "check"
        name: "Dev Check"
"""
        )

        from dqx.config import load_config, load_profiles_from_config
        from dqx.profiles import PermanentProfile

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert isinstance(profiles[0], PermanentProfile)
        assert profiles[0].name == "Production Baseline"
        assert len(profiles[0].rules) == 1
        assert profiles[0].rules[0].disabled is True

    def test_load_permanent_profile_with_multiple_rules(self, tmp_path: Path) -> None:
        """Load permanent profile with multiple rules."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Baseline"
    type: "permanent"
    rules:
      - action: "disable"
        target: "check"
        name: "Check1"
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 0.9
      - action: "set_severity"
        target: "tag"
        name: "critical"
        severity: "P0"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert len(profiles[0].rules) == 3

    def test_load_permanent_profile_with_tag_rules(self, tmp_path: Path) -> None:
        """Load permanent profile with tag selector rules."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Tag Profile"
    type: "permanent"
    rules:
      - action: "scale"
        target: "tag"
        name: "test_tag"
        multiplier: 1.5
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert profiles[0].rules[0].metric_multiplier == 1.5

    def test_load_permanent_profile_with_check_rules(self, tmp_path: Path) -> None:
        """Load permanent profile with check selector rules."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Check Profile"
    type: "permanent"
    rules:
      - action: "disable"
        target: "check"
        name: "TestCheck"
"""
        )

        from dqx.config import load_config, load_profiles_from_config
        from dqx.profiles import AssertionSelector

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert isinstance(profiles[0].rules[0].selector, AssertionSelector)
        assert profiles[0].rules[0].selector.check == "TestCheck"

    def test_load_permanent_profile_with_assertion_rules(self, tmp_path: Path) -> None:
        """Load permanent profile with assertion selector rules."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Assertion Profile"
    type: "permanent"
    rules:
      - action: "disable"
        target: "check"
        name: "MyCheck"
"""
        )

        from dqx.config import load_config, load_profiles_from_config
        from dqx.profiles import AssertionSelector

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert isinstance(profiles[0].rules[0].selector, AssertionSelector)
        assert profiles[0].rules[0].selector.check == "MyCheck"
        assert profiles[0].rules[0].selector.assertion is None

    def test_load_mixed_seasonal_and_permanent_profiles(self, tmp_path: Path) -> None:
        """Load config with both seasonal and permanent profiles."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Permanent Baseline"
    type: "permanent"
    rules:
      - action: "disable"
        target: "check"
        name: "DevCheck"

  - name: "Holiday Season"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 2.0
"""
        )

        from dqx.config import load_config, load_profiles_from_config
        from dqx.profiles import PermanentProfile, SeasonalProfile

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 2
        assert isinstance(profiles[0], PermanentProfile)
        assert isinstance(profiles[1], SeasonalProfile)

    def test_permanent_profile_missing_name(self, tmp_path: Path) -> None:
        """Permanent profile without name should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - type: "permanent"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'name' is required"):
            load_profiles_from_config(config)

    def test_permanent_profile_missing_rules(self, tmp_path: Path) -> None:
        """Permanent profile without rules should default to empty list."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "permanent"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert profiles[0].rules == []

    def test_permanent_profile_with_start_date_error(self, tmp_path: Path) -> None:
        """Permanent profile with start_date should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "permanent"
    start_date: "2024-12-20"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'start_date' not allowed for permanent profiles"):
            load_profiles_from_config(config)

    def test_permanent_profile_with_end_date_error(self, tmp_path: Path) -> None:
        """Permanent profile with end_date should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "permanent"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'end_date' not allowed for permanent profiles"):
            load_profiles_from_config(config)

    def test_permanent_profile_invalid_rules(self, tmp_path: Path) -> None:
        """Permanent profile with invalid rules should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "permanent"
    rules:
      - action: "invalid_action"
        target: "check"
        name: "Test"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="invalid action 'invalid_action'"):
            load_profiles_from_config(config)

    def test_permanent_profile_rules_not_list(self, tmp_path: Path) -> None:
        """Permanent profile with rules that's not a list should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test Permanent"
    type: "permanent"
    rules: "not a list"
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="'rules' must be a list"):
            load_profiles_from_config(config)

    def test_default_type_is_seasonal(self, tmp_path: Path) -> None:
        """Profile without type should default to seasonal."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config
        from dqx.profiles import SeasonalProfile

        config = load_config(config_file)
        profiles = load_profiles_from_config(config)

        assert len(profiles) == 1
        assert isinstance(profiles[0], SeasonalProfile)

    def test_unknown_profile_type_error(self, tmp_path: Path) -> None:
        """Profile with unknown type should raise DQXError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test"
    type: "temporary"
    rules: []
"""
        )

        from dqx.config import load_config, load_profiles_from_config

        config = load_config(config_file)
        with pytest.raises(DQXError, match="unknown type 'temporary'"):
            load_profiles_from_config(config)


class TestSuiteWithProfilesFromConfig:
    """Tests for VerificationSuite with profiles loaded from config."""

    def test_suite_loads_profiles_from_config(self, tmp_path: Path) -> None:
        """Verify suite loads profiles from config file."""

        @check(name="Test Check", datasets=["ds"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="test", tags={"volume"}).is_positive()

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Test Profile"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 2.0
"""
        )

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            config=config_file,
        )

        # Verify profile was loaded
        assert len(suite._profiles) == 1
        assert suite._profiles[0].name == "Test Profile"

    def test_suite_merges_config_and_api_profiles(self, tmp_path: Path) -> None:
        """Verify suite merges profiles from config and API."""
        from dqx.profiles import SeasonalProfile, tag

        @check(name="Test Check", datasets=["ds"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="test").is_positive()

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Config Profile"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: "Test Check"
"""
        )

        api_profile = SeasonalProfile(
            name="API Profile",
            start_date=dt.date(2025, 1, 10),
            end_date=dt.date(2025, 1, 15),
            rules=[tag("test").set(metric_multiplier=2.0)],
        )

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            config=config_file,
            profiles=[api_profile],
        )

        # Verify both profiles are loaded
        assert len(suite._profiles) == 2
        assert suite._profiles[0].name == "Config Profile"
        assert suite._profiles[1].name == "API Profile"

    def test_suite_duplicate_profile_names_config_vs_api(self, tmp_path: Path) -> None:
        """Duplicate profile names between config and API should raise DQXError."""
        from dqx.profiles import SeasonalProfile, tag

        @check(name="Test Check", datasets=["ds"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="test").is_positive()

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
tunables: {}
profiles:
  - name: "Holiday"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: "Test Check"
"""
        )

        api_profile = SeasonalProfile(
            name="Holiday",
            start_date=dt.date(2025, 1, 10),
            end_date=dt.date(2025, 1, 15),
            rules=[tag("test").set(metric_multiplier=2.0)],
        )

        db = InMemoryMetricDB()
        with pytest.raises(DQXError, match=r"Duplicate profile name.*'Holiday'"):
            VerificationSuite(
                checks=[test_check],
                db=db,
                name="Test Suite",
                config=config_file,
                profiles=[api_profile],
            )
