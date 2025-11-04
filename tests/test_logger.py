"""Tests for DQX logging functionality including Rich integration."""

import logging
import uuid
from typing import Generator

import pytest
from rich.logging import RichHandler

from dqx import DEFAULT_LOGGER_NAME, setup_logger


@pytest.fixture(autouse=True)
def reset_logger_state() -> Generator[None, None, None]:
    """Reset logger state before and after each test to ensure proper isolation."""
    # Store original state
    original_loggers = dict(logging.Logger.manager.loggerDict)

    # Clear all existing loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("dqx"):  # Only clear dqx loggers
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True

    # Run the test
    yield

    # Clean up loggers created during the test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("dqx"):
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True

            # Remove logger from manager if it didn't exist before
            if logger_name not in original_loggers:
                try:
                    del logging.Logger.manager.loggerDict[logger_name]
                except KeyError:
                    pass


class TestSetupLogger:
    """Test cases for setup_logger function."""

    def test_setup_logger_default_name(self) -> None:
        """Test that default logger name is 'dqx'."""
        setup_logger(force_reconfigure=True)
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        assert logger.name == DEFAULT_LOGGER_NAME

    def test_setup_logger_custom_name(self) -> None:
        """Test creating logger with custom name."""
        setup_logger("dqx.custom", force_reconfigure=True)
        logger = logging.getLogger("dqx.custom")
        assert logger.name == "dqx.custom"

    def test_setup_logger_default_level(self) -> None:
        """Test that default log level is INFO."""
        setup_logger("dqx.test.default_level", force_reconfigure=True)
        logger = logging.getLogger("dqx.test.default_level")
        assert logger.level == logging.INFO

    def test_setup_logger_custom_level_int(self) -> None:
        """Test setting custom log level using integer."""
        setup_logger("dqx.test.level_int", level=logging.DEBUG, force_reconfigure=True)
        logger = logging.getLogger("dqx.test.level_int")
        assert logger.level == logging.DEBUG

    def test_setup_logger_has_handler(self) -> None:
        """Test that logger has a RichHandler."""
        # Use a unique logger to ensure clean state
        logger_name = f"dqx.test.handler_{uuid.uuid4().hex[:8]}"
        setup_logger(logger_name)
        logger = logging.getLogger(logger_name)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], RichHandler)

    def test_setup_logger_default_format(self) -> None:
        """Test that logger uses default format string."""
        setup_logger(force_reconfigure=True)
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        handler = logger.handlers[0]
        formatter = handler.formatter
        assert formatter is not None
        # Rich handles formatting internally, so formatter just returns message
        record = logging.LogRecord(
            name=DEFAULT_LOGGER_NAME,
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        # With Rich, the formatter only returns the message
        assert formatted == "Test message"

    def test_setup_logger_custom_format(self) -> None:
        """Test that Rich always uses message-only formatter."""
        setup_logger(force_reconfigure=True)
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        handler = logger.handlers[0]
        formatter = handler.formatter
        assert formatter is not None
        # Rich uses message-only formatter
        record = logging.LogRecord(
            name="dqx",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        # With Rich, the formatter only returns the message
        assert formatted == "Test message"

    def test_setup_logger_no_duplicate_handlers(self) -> None:
        """Test that calling setup_logger multiple times doesn't add duplicate handlers."""
        # Use a unique logger name for this test
        logger_name = f"dqx.test.no_dup_{uuid.uuid4().hex[:8]}"

        # First call
        setup_logger(logger_name)
        logger = logging.getLogger(logger_name)
        assert len(logger.handlers) == 1

        # Second call
        setup_logger(logger_name)
        logger2 = logging.getLogger(logger_name)
        assert logger is logger2  # Same logger instance
        assert len(logger2.handlers) == 1  # Still only one handler

    def test_setup_logger_force_reconfigure(self) -> None:
        """Test force_reconfigure parameter."""
        # Use a unique logger name for this test
        logger_name = f"dqx.test.reconfigure_{uuid.uuid4().hex[:8]}"

        # Create logger with INFO level
        setup_logger(logger_name, level=logging.INFO)
        logger = logging.getLogger(logger_name)
        assert logger.level == logging.INFO

        # Reconfigure with DEBUG level
        setup_logger(logger_name, level=logging.DEBUG, force_reconfigure=True)
        logger = logging.getLogger(logger_name)
        assert logger.level == logging.DEBUG

    def test_setup_logger_force_reconfigure_clears_handlers(self) -> None:
        """Test that force_reconfigure clears existing handlers."""
        # Use a unique logger name for this test
        logger_name = f"dqx.test.clear_{uuid.uuid4().hex[:8]}"

        setup_logger(logger_name)
        logger = logging.getLogger(logger_name)
        original_handler = logger.handlers[0]

        # Force reconfigure
        setup_logger(logger_name, force_reconfigure=True)
        logger = logging.getLogger(logger_name)

        assert len(logger.handlers) == 1
        assert logger.handlers[0] is not original_handler

    def test_setup_logger_no_propagation(self) -> None:
        """Test that logger propagation is disabled."""
        setup_logger(force_reconfigure=True)
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        assert logger.propagate is False

    def test_setup_logger_default_format_constant(self) -> None:
        """Test that DEFAULT_FORMAT constant is used when format_string is None."""
        setup_logger("dqx.test.default_format", force_reconfigure=True)
        logger = logging.getLogger("dqx.test.default_format")
        handler = logger.handlers[0]
        formatter = handler.formatter
        assert formatter is not None
        # Rich handles formatting internally, formatter just returns message
        record = logging.LogRecord(
            name="dqx.test.default_format",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        # With Rich, the formatter only returns the message
        assert formatted == "Test"

    def test_all_valid_log_levels(self) -> None:
        """Test all valid log levels work correctly."""
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        for int_level, str_level in levels:
            # Test with integer only (string levels no longer supported)
            logger_name = f"dqx.test.level.{str_level}"
            setup_logger(logger_name, level=int_level)
            logger = logging.getLogger(logger_name)
            assert logger.level == int_level


class TestRichIntegration:
    """Test cases for Rich handler integration."""

    def test_rich_handler_configuration(self) -> None:
        """Test that RichHandler is configured correctly."""
        setup_logger("test.config", force_reconfigure=True)
        logger = logging.getLogger("test.config")
        handler = logger.handlers[0]

        assert isinstance(handler, RichHandler)
        # Rich stores these in the console object
        assert handler.console is not None
        assert handler._log_render.show_time is True
        assert handler._log_render.show_level is True
        assert handler._log_render.show_path is True  # Updated to match current config

    def test_rich_tracebacks_enabled(self) -> None:
        """Test that rich tracebacks are enabled."""
        setup_logger("test.tracebacks", force_reconfigure=True)
        logger = logging.getLogger("test.tracebacks")
        handler = logger.handlers[0]

        assert isinstance(handler, RichHandler)
        assert handler.rich_tracebacks is True

    def test_markup_enabled(self) -> None:
        """Test that markup is enabled by default."""
        setup_logger("test.markup", force_reconfigure=True)
        logger = logging.getLogger("test.markup")
        handler = logger.handlers[0]

        assert isinstance(handler, RichHandler)
        assert handler.markup is True

    def test_time_format(self) -> None:
        """Test that the time format is set correctly."""
        setup_logger("test.time", force_reconfigure=True)
        logger = logging.getLogger("test.time")
        handler = logger.handlers[0]

        assert isinstance(handler, RichHandler)
        assert handler._log_render.time_format == "[%X]"

    def test_force_reconfigure_with_rich(self) -> None:
        """Test that force_reconfigure works with RichHandler."""
        logger_name = "test.reconfigure.rich"
        setup_logger(logger_name, force_reconfigure=True)
        logger = logging.getLogger(logger_name)

        # First handler
        assert len(logger.handlers) == 1
        first_handler = logger.handlers[0]

        # Force reconfigure
        setup_logger(logger_name, force_reconfigure=True)
        logger2 = logging.getLogger(logger_name)

        # Should be the same logger instance
        assert logger2 is logger

        # But with a new handler
        assert len(logger.handlers) == 1
        assert logger.handlers[0] is not first_handler
        assert isinstance(logger.handlers[0], RichHandler)

    def test_logger_level_setting(self) -> None:
        """Test that logger level is set correctly."""
        # Test default level
        setup_logger("test.level.default", force_reconfigure=True)
        logger = logging.getLogger("test.level.default")
        assert logger.level == logging.INFO

        # Test custom level
        setup_logger("test.level.debug", level=logging.DEBUG, force_reconfigure=True)
        debug_logger = logging.getLogger("test.level.debug")
        assert debug_logger.level == logging.DEBUG

    def test_omit_repeated_times_disabled(self) -> None:
        """Test that omit_repeated_times is set to False to show all timestamps."""
        setup_logger("test.timestamps", force_reconfigure=True)
        logger = logging.getLogger("test.timestamps")
        handler = logger.handlers[0]

        assert isinstance(handler, RichHandler)
        # RichHandler stores this in the console's log_render object
        assert handler._log_render.omit_repeated_times is False
