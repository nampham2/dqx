"""Configuration file loading for tunable parameters and profiles.

This module provides utilities for loading tunable values and profiles from YAML configuration files.
Configuration files allow users to set tunable threshold values and define profiles without modifying code.

Example YAML structure:
    tunables:
      THRESHOLD: 0.05
      MIN_ROWS: 1000
      MAX_NULL_RATE: 0.10

    profiles:
      - name: "Holiday Season"
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
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from dqx.common import DQXError, SeverityLevel
from dqx.tunables import Tunable

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and parse YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.

    Raises:
        DQXError: If file not found, YAML syntax is invalid, or required structure is missing.

    Example:
        >>> config = load_config(Path("config.yaml"))
        >>> print(config["tunables"]["THRESHOLD"])
        0.05
    """
    if not config_path.exists():
        raise DQXError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise DQXError(f"Invalid YAML syntax in {config_path}: {e}") from e

    if config is None:
        raise DQXError(f"Configuration file is empty: {config_path}")

    if not isinstance(config, dict):
        raise DQXError(f"Configuration file must contain a YAML dictionary, got {type(config).__name__}")

    return config


def validate_config_structure(config: dict[str, Any]) -> None:
    """Validate that the configuration has the expected top-level structure.

    Args:
        config: Parsed configuration dictionary.

    Raises:
        DQXError: If the configuration is missing required keys or has invalid structure.
    """
    if "tunables" not in config:
        raise DQXError("Configuration file must contain 'tunables' key at top level")

    tunables_section = config["tunables"]
    if tunables_section is None:
        # Allow empty tunables section (tunables: null or tunables:)
        pass
    elif not isinstance(tunables_section, dict):
        raise DQXError(f"'tunables' section must be a dictionary, got {type(tunables_section).__name__}")

    # Validate profiles section if present (optional)
    if "profiles" in config:
        profiles_section = config["profiles"]
        if profiles_section is not None and not isinstance(profiles_section, list):
            raise DQXError(f"'profiles' section must be a list, got {type(profiles_section).__name__}")


def apply_tunables_from_config(config: dict[str, Any], tunables: Mapping[str, Tunable[Any]]) -> None:
    """Apply tunable values from configuration to suite tunables.

    Extracts the 'tunables' section from the config and applies each value to the
    corresponding tunable in the suite. Tunables in the suite that are not in the
    config are left unchanged.

    Args:
        config: Parsed configuration dictionary containing 'tunables' section.
        tunables: Mapping of tunable name to Tunable object from the suite.

    Raises:
        DQXError: If configuration structure is invalid, an unknown tunable is
            specified, or a tunable value fails validation.

    Example:
        >>> config = {"tunables": {"THRESHOLD": 0.05, "MIN_ROWS": 1000}}
        >>> tunables = {"THRESHOLD": threshold_tunable, "MIN_ROWS": min_rows_tunable}
        >>> apply_tunables_from_config(config, tunables)
    """
    validate_config_structure(config)

    tunables_config = config["tunables"]

    # Handle None or empty dict
    if not tunables_config:
        return

    # Apply each tunable from config
    for name, value in tunables_config.items():
        if name not in tunables:
            raise DQXError(
                f"Configuration contains tunable '{name}' not found in suite. "
                f"Available tunables: {list(tunables.keys())}"
            )

        # Let Tunable.set() handle type validation
        try:
            tunables[name].set(value, agent="config", reason="Loaded from config file")
            logger.debug(f"Set tunable '{name}' to {value} from config")
        except (ValueError, TypeError) as e:
            # Re-raise with more context about which config value failed
            raise DQXError(f"Invalid value for tunable '{name}' in config: {e}") from e


def _validate_no_unknown_fields(data: dict[str, Any], valid_fields: set[str], context: str) -> None:
    """Validate that dictionary contains no unknown fields.

    Args:
        data: Dictionary to validate.
        valid_fields: Set of allowed field names.
        context: Context string for error messages (e.g., "Profile 'Holiday', rule 0").

    Raises:
        DQXError: If unknown fields are present.
    """
    unknown = set(data.keys()) - valid_fields
    if unknown:
        raise DQXError(f"{context}: unknown field(s) {sorted(unknown)}. Valid fields: {sorted(valid_fields)}")


def _parse_rules(profile_name: str, rules_data: list[dict[str, Any]]) -> list[Any]:
    """Parse rules array from profile config.

    Args:
        profile_name: Name of profile (for error messages).
        rules_data: List of rule dictionaries.

    Returns:
        List of Rule objects.

    Raises:
        DQXError: If rule configuration is invalid.
    """
    from dqx.profiles import AssertionSelector, Rule, TagSelector

    rules: list[Any] = []

    for rule_idx, rule_dict in enumerate(rules_data):
        if not isinstance(rule_dict, dict):
            raise DQXError(f"Profile '{profile_name}': rule at index {rule_idx} must be a dictionary")

        try:
            # Validate action
            action = rule_dict.get("action")
            if not action:
                raise DQXError(f"Profile '{profile_name}', rule {rule_idx}: 'action' is required")
            if action not in ("disable", "scale", "set_severity"):
                raise DQXError(
                    f"Profile '{profile_name}', rule {rule_idx}: "
                    f"invalid action '{action}' (must be 'disable', 'scale', or 'set_severity')"
                )

            # Validate target
            target = rule_dict.get("target")
            if not target:
                raise DQXError(f"Profile '{profile_name}', rule {rule_idx}: 'target' is required")
            if target not in ("check", "tag"):
                raise DQXError(
                    f"Profile '{profile_name}', rule {rule_idx}: invalid target '{target}' (must be 'check' or 'tag')"
                )

            # Validate name
            name = rule_dict.get("name")
            if not name:
                raise DQXError(f"Profile '{profile_name}', rule {rule_idx}: 'name' is required")
            if not isinstance(name, str):
                raise DQXError(f"Profile '{profile_name}', rule {rule_idx}: 'name' must be a string")

            # Build selector
            selector: AssertionSelector | TagSelector
            if target == "check":
                selector = AssertionSelector(check=name, assertion=None)
            else:  # target == "tag"
                selector = TagSelector(tag=name)

            # Build rule based on action
            rule: Any  # Initialize to satisfy type checker
            if action == "disable":
                # Validate no unknown fields
                valid_fields = {"action", "target", "name"}
                _validate_no_unknown_fields(rule_dict, valid_fields, f"Profile '{profile_name}', rule {rule_idx}")
                rule = Rule(selector=selector, disabled=True)

            elif action == "scale":
                # Validate multiplier
                multiplier = rule_dict.get("multiplier")
                if multiplier is None:
                    raise DQXError(
                        f"Profile '{profile_name}', rule {rule_idx}: 'multiplier' is required for 'scale' action"
                    )
                if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
                    raise DQXError(f"Profile '{profile_name}', rule {rule_idx}: 'multiplier' must be a number")
                if multiplier <= 0:
                    raise DQXError(
                        f"Profile '{profile_name}', rule {rule_idx}: 'multiplier' must be positive, got {multiplier}"
                    )

                # Validate no unknown fields
                valid_fields = {"action", "target", "name", "multiplier"}
                _validate_no_unknown_fields(rule_dict, valid_fields, f"Profile '{profile_name}', rule {rule_idx}")

                rule = Rule(selector=selector, metric_multiplier=float(multiplier))

            elif action == "set_severity":
                # Validate severity
                severity = rule_dict.get("severity")
                if not severity:
                    raise DQXError(
                        f"Profile '{profile_name}', rule {rule_idx}: 'severity' is required for 'set_severity' action"
                    )
                if severity not in ("P0", "P1", "P2", "P3"):
                    raise DQXError(
                        f"Profile '{profile_name}', rule {rule_idx}: "
                        f"invalid severity '{severity}' (must be 'P0', 'P1', 'P2', or 'P3')"
                    )

                # Validate no unknown fields
                valid_fields = {"action", "target", "name", "severity"}
                _validate_no_unknown_fields(rule_dict, valid_fields, f"Profile '{profile_name}', rule {rule_idx}")

                rule = Rule(selector=selector, severity=cast(SeverityLevel, severity))

            else:  # pragma: no cover
                # This should never happen as we validate action above
                raise DQXError(f"Profile '{profile_name}', rule {rule_idx}: invalid action '{action}'")

            rules.append(rule)

        except DQXError:
            raise  # Re-raise DQXError as-is
        except Exception as e:  # pragma: no cover
            raise DQXError(
                f"Profile '{profile_name}', rule {rule_idx}: failed to parse: {e}"
            ) from e  # pragma: no cover

    return rules


def load_profiles_from_config(config: dict[str, Any]) -> list[Any]:
    """Load profiles from YAML config dictionary.

    Parses the 'profiles' section of the config and creates SeasonalProfile
    or PermanentProfile objects with Rule instances. Validates all required
    fields and data types.

    Args:
        config: Parsed YAML config dictionary (from load_config).

    Returns:
        List of Profile objects (SeasonalProfile or PermanentProfile).
        Returns empty list if no profiles defined.

    Raises:
        DQXError: If profile configuration is invalid (missing fields,
                 bad dates, invalid rule actions, duplicate names, etc.).

    Example:
        >>> config = load_config(Path("config.yaml"))
        >>> profiles = load_profiles_from_config(config)
        >>> print(f"Loaded {len(profiles)} profile(s)")
    """
    from datetime import date as dt_date

    from dqx.profiles import PermanentProfile, SeasonalProfile

    # Get profiles section (may be None or missing)
    profiles_data = config.get("profiles")

    if profiles_data is None:
        return []

    if not isinstance(profiles_data, list):  # pragma: no cover
        raise DQXError(f"'profiles' section must be a list, got {type(profiles_data).__name__}")  # pragma: no cover

    profiles: list[Any] = []
    seen_names: set[str] = set()

    for idx, profile_dict in enumerate(profiles_data):
        if not isinstance(profile_dict, dict):
            raise DQXError(f"Profile at index {idx} must be a dictionary, got {type(profile_dict).__name__}")

        try:
            # Validate required fields
            name = profile_dict.get("name")
            if not name:
                raise DQXError(f"Profile at index {idx}: 'name' is required")
            if not isinstance(name, str):
                raise DQXError(f"Profile at index {idx}: 'name' must be a string")

            # Check for duplicate names within config
            if name in seen_names:
                raise DQXError(
                    f"Duplicate profile name '{name}' at index {idx} in config file. Profile names must be unique."
                )
            seen_names.add(name)

            # Validate type (optional, defaults to "seasonal")
            profile_type = profile_dict.get("type", "seasonal")

            profile: Any  # Union of SeasonalProfile | PermanentProfile

            if profile_type == "seasonal":
                # Parse dates
                start_date_str = profile_dict.get("start_date")
                if not start_date_str:
                    raise DQXError(f"Profile '{name}': 'start_date' is required")
                if not isinstance(start_date_str, str):
                    raise DQXError(f"Profile '{name}': 'start_date' must be a string in ISO 8601 format (YYYY-MM-DD)")

                end_date_str = profile_dict.get("end_date")
                if not end_date_str:
                    raise DQXError(f"Profile '{name}': 'end_date' is required")
                if not isinstance(end_date_str, str):
                    raise DQXError(f"Profile '{name}': 'end_date' must be a string in ISO 8601 format (YYYY-MM-DD)")

                try:
                    start_date = dt_date.fromisoformat(start_date_str)
                except ValueError as e:
                    raise DQXError(
                        f"Profile '{name}': invalid 'start_date' format '{start_date_str}'. "
                        f"Use ISO 8601 format (YYYY-MM-DD). Error: {e}"
                    ) from e

                try:
                    end_date = dt_date.fromisoformat(end_date_str)
                except ValueError as e:
                    raise DQXError(
                        f"Profile '{name}': invalid 'end_date' format '{end_date_str}'. "
                        f"Use ISO 8601 format (YYYY-MM-DD). Error: {e}"
                    ) from e

                # Validate date range
                if end_date < start_date:
                    raise DQXError(f"Profile '{name}': 'end_date' ({end_date}) must be >= 'start_date' ({start_date})")

                # Parse rules
                rules_data = profile_dict.get("rules", [])
                if not isinstance(rules_data, list):
                    raise DQXError(f"Profile '{name}': 'rules' must be a list")

                rules = _parse_rules(name, rules_data)

                # Validate no unknown fields at profile level
                valid_fields = {"name", "type", "start_date", "end_date", "rules"}
                _validate_no_unknown_fields(profile_dict, valid_fields, f"Profile '{name}'")

                # Create SeasonalProfile
                profile = SeasonalProfile(
                    name=name,
                    start_date=start_date,
                    end_date=end_date,
                    rules=rules,
                )
                profiles.append(profile)

            elif profile_type == "permanent":
                # Validate no date fields present
                if "start_date" in profile_dict:
                    raise DQXError(f"Profile '{name}': 'start_date' not allowed for permanent profiles")
                if "end_date" in profile_dict:
                    raise DQXError(f"Profile '{name}': 'end_date' not allowed for permanent profiles")

                # Parse rules
                rules_data = profile_dict.get("rules", [])
                if not isinstance(rules_data, list):
                    raise DQXError(f"Profile '{name}': 'rules' must be a list")

                rules = _parse_rules(name, rules_data)

                # Validate no unknown fields at profile level
                valid_fields = {"name", "type", "rules"}
                _validate_no_unknown_fields(profile_dict, valid_fields, f"Profile '{name}'")

                # Create PermanentProfile
                profile = PermanentProfile(
                    name=name,
                    rules=rules,
                )
                profiles.append(profile)

            else:
                raise DQXError(f"Profile '{name}': unknown type '{profile_type}' (must be 'seasonal' or 'permanent')")

        except DQXError:
            raise  # Re-raise DQXError as-is
        except Exception as e:  # pragma: no cover
            # Catch any unexpected errors during parsing
            raise DQXError(f"Failed to parse profile at index {idx}: {e}") from e  # pragma: no cover

    return profiles
