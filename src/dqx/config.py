"""Configuration file loading for tunable parameters.

This module provides utilities for loading tunable values from YAML configuration files.
Configuration files allow users to set tunable threshold values without modifying code.

Example YAML structure:
    tunables:
      THRESHOLD: 0.05
      MIN_ROWS: 1000
      MAX_NULL_RATE: 0.10
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dqx.common import DQXError
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
        return

    if not isinstance(tunables_section, dict):
        raise DQXError(f"'tunables' section must be a dictionary, got {type(tunables_section).__name__}")


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
