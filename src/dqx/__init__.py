"""DQX - Data Quality eXplorer."""

import logging

from rich.logging import RichHandler

# Version information
try:
    from importlib.metadata import version

    __version__ = version("dqlib")
except Exception:
    # Fallback for development or when package isn't installed
    __version__ = "0.0.0.dev"

DEFAULT_FORMAT = "%(asctime)s [%(levelname).1s] %(message)s"
DEFAULT_LOGGER_NAME = "dqx"


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: int = logging.INFO,
    format_string: str = DEFAULT_FORMAT,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """
    Get a configured logger instance for DQX with Rich formatting.

    This function provides a centralized way to create and configure loggers
    for the DQX library with colorized output and enhanced formatting using
    Rich. The logger automatically provides:

    - Color-coded log levels (DEBUG=blue, INFO=green, WARNING=yellow, ERROR=red)
    - Clean timestamp formatting (HH:MM:SS)
    - Beautiful exception tracebacks with syntax highlighting
    - Support for Rich markup in log messages

    The function is thread-safe as it uses Python's built-in logging module.

    Args:
        name: Logger name. Defaults to "dqx". Can be used to create child loggers
            like "dqx.analyzer" for specific modules.
        level: Logging level as an integer (logging.INFO, logging.DEBUG, etc.).
            Defaults to logging.INFO.
        format_string: This parameter is kept for API compatibility but is ignored
            as Rich handles formatting internally.
        force_reconfigure: If True, reconfigure the logger even if handlers already
            exist. Useful for changing configuration at runtime. Defaults to False.

    Returns:
        A configured logger instance with Rich formatting.

    Example:
        Basic usage:
        >>> logger = get_logger()
        >>> logger.info("Starting DQX processing")

        With markup:
        >>> logger.info("[bold green]Success![/bold green] All checks passed")

        Debug logging:
        >>> debug_logger = get_logger("dqx.debug", level=logging.DEBUG)
        >>> debug_logger.debug("Detailed debug information")

    Note:
        When output is redirected (not a TTY), Rich automatically disables
        colors and formatting to ensure clean log files.
    """
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
            show_path=False,
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

    return logger
