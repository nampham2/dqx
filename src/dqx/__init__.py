"""DQX - Data Quality eXplorer."""

import logging
from typing import TYPE_CHECKING, Any

from rich.logging import RichHandler

from dqx.profiles import (
    Profile,
    Rule,
    SeasonalProfile,
    assertion,
    check,
    tag,
)
from dqx.tunables import (
    Tunable,
    TunableChange,
    TunableChoice,
    TunableFloat,
    TunableInt,
)

if TYPE_CHECKING:
    from dqx.contract import Contract, ContractValidationError, ContractWarning, SchemaValidationError

__all__ = [
    "Contract",
    "ContractValidationError",
    "ContractWarning",
    "Profile",
    "Rule",
    "SchemaValidationError",
    "SeasonalProfile",
    "Tunable",
    "TunableChange",
    "TunableChoice",
    "TunableFloat",
    "TunableInt",
    "assertion",
    "check",
    "tag",
    "setup_logger",
]


def __getattr__(name: str) -> Any:
    if name in ("Contract", "ContractValidationError", "ContractWarning", "SchemaValidationError"):
        import importlib

        module = importlib.import_module("dqx.contract")
        return getattr(module, name)
    raise AttributeError(f"module 'dqx' has no attribute {name!r}")


# Version information
try:
    from importlib.metadata import version

    __version__ = version("dqlib")
except Exception:  # pragma: no cover
    # Fallback for development or when package isn't installed
    __version__ = "0.0.0.dev"

DEFAULT_LOGGER_NAME = "dqx"


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: int | str = logging.INFO,
    force_reconfigure: bool = False,
) -> None:
    # Get or create logger
    logger = logging.getLogger(name)

    # Configure logger if it has no handlers or force_reconfigure is True
    if not logger.handlers or force_reconfigure:
        # Clear existing handlers if force_reconfigure
        if force_reconfigure and logger.handlers:
            logger.handlers.clear()

        # Create Rich console handler
        handler = RichHandler(
            show_time=True,
            show_level=True,
            show_path=True,
            rich_tracebacks=True,
            markup=True,
            log_time_format="[%X]",
            omit_repeated_times=False,
        )

        # Rich handles most formatting internally, but we keep message-only formatter
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    # Always set the level (even if logger already has handlers)
    logger.setLevel(level)
