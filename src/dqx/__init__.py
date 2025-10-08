"""DQX - Data Quality eXplorer."""

import logging

DEFAULT_FORMAT = "%(asctime)s [%(levelname).1s] %(message)s"
DEFAULT_LOGGER_NAME = "dqx"


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: int = logging.INFO,
    format_string: str = DEFAULT_FORMAT,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """
    Get a configured logger instance for DQX.

    This function provides a centralized way to create and configure loggers
    for the DQX library. It ensures consistent logging setup across the project.
    The function is thread-safe as it uses Python's built-in logging module.

    Args:
        name: Logger name. Defaults to "dqx". Can be used to create child loggers
            like "dqx.analyzer" for specific modules.
        level: Logging level as an integer (logging.INFO, logging.DEBUG, etc.).
            Defaults to logging.INFO.
        format_string: Custom format string for log messages. If None, uses
            DEFAULT_FORMAT: '%(asctime)s [%(levelname).1s] %(message)s'.
        force_reconfigure: If True, reconfigure the logger even if handlers already
            exist. Useful for changing configuration at runtime. Defaults to False.

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger()
        >>> logger.info("Starting DQX processing")

        >>> debug_logger = get_logger("dqx.debug", level=logging.DEBUG)
        >>> debug_logger.debug("Detailed debug information")
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Configure logger if it has no handlers or force_reconfigure is True
    if not logger.handlers or force_reconfigure:
        # Clear existing handlers if force_reconfigure
        if force_reconfigure and logger.handlers:
            logger.handlers.clear()

        # Create console handler
        handler = logging.StreamHandler()

        # Set format
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    # Always set the level (even if logger already has handlers)
    logger.setLevel(level)

    return logger
