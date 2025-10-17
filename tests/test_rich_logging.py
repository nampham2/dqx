"""Tests for Rich logging integration."""

import logging

from rich.logging import RichHandler

from dqx import get_logger


def test_get_logger_returns_rich_handler() -> None:
    """Test that get_logger returns a logger with RichHandler."""
    logger = get_logger("test.rich")

    # Check that the logger has exactly one handler
    assert len(logger.handlers) == 1

    # Check that the handler is a RichHandler
    assert isinstance(logger.handlers[0], RichHandler)


def test_rich_handler_configuration() -> None:
    """Test that RichHandler is configured correctly."""
    logger = get_logger("test.config")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    # Rich stores these in the console object
    assert handler.console is not None
    assert handler._log_render.show_time is True
    assert handler._log_render.show_level is True
    assert handler._log_render.show_path is False


def test_rich_tracebacks_enabled() -> None:
    """Test that rich tracebacks are enabled."""
    logger = get_logger("test.tracebacks")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    assert handler.rich_tracebacks is True


def test_markup_enabled() -> None:
    """Test that markup is enabled by default."""
    logger = get_logger("test.markup")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    assert handler.markup is True


def test_time_format() -> None:
    """Test that the time format is set correctly."""
    logger = get_logger("test.time")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    assert handler._log_render.time_format == "[%X]"


def test_force_reconfigure_with_rich() -> None:
    """Test that force_reconfigure works with RichHandler."""
    logger = get_logger("test.reconfigure")

    # First handler
    assert len(logger.handlers) == 1
    first_handler = logger.handlers[0]

    # Force reconfigure
    logger2 = get_logger("test.reconfigure", force_reconfigure=True)

    # Should be the same logger instance
    assert logger2 is logger

    # But with a new handler
    assert len(logger.handlers) == 1
    assert logger.handlers[0] is not first_handler
    assert isinstance(logger.handlers[0], RichHandler)


def test_logger_level_setting() -> None:
    """Test that logger level is set correctly."""
    # Test default level
    logger = get_logger("test.level.default")
    assert logger.level == logging.INFO

    # Test custom level
    debug_logger = get_logger("test.level.debug", level=logging.DEBUG)
    assert debug_logger.level == logging.DEBUG


def test_no_propagation() -> None:
    """Test that logger propagation is disabled."""
    logger = get_logger("test.propagate")
    assert logger.propagate is False


def test_omit_repeated_times_disabled() -> None:
    """Test that omit_repeated_times is set to False to show all timestamps."""
    logger = get_logger("test.timestamps")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    # RichHandler stores this in the console's log_render object
    assert handler._log_render.omit_repeated_times is False
