import logging
import uuid
from io import StringIO
from typing import Generator
from unittest.mock import patch

import pytest
from rich.logging import RichHandler

from dqx import DEFAULT_LOGGER_NAME, get_logger


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


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_default_name(self) -> None:
        """Test that default logger name is 'dqx'."""
        logger = get_logger(force_reconfigure=True)
        assert logger.name == DEFAULT_LOGGER_NAME

    def test_get_logger_custom_name(self) -> None:
        """Test creating logger with custom name."""
        logger = get_logger("dqx.custom", force_reconfigure=True)
        assert logger.name == "dqx.custom"

    def test_get_logger_default_level(self) -> None:
        """Test that default log level is INFO."""
        logger = get_logger("dqx.test.default_level", force_reconfigure=True)
        assert logger.level == logging.INFO

    def test_get_logger_custom_level_int(self) -> None:
        """Test setting custom log level using integer."""
        logger = get_logger("dqx.test.level_int", level=logging.DEBUG, force_reconfigure=True)
        assert logger.level == logging.DEBUG

    def test_get_logger_has_handler(self) -> None:
        """Test that logger has a RichHandler."""
        # Use a unique logger to ensure clean state
        logger_name = f"dqx.test.handler_{uuid.uuid4().hex[:8]}"
        logger = get_logger(logger_name)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], RichHandler)

    def test_get_logger_default_format(self) -> None:
        """Test that logger uses default format string."""
        logger = get_logger(force_reconfigure=True)
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

    def test_get_logger_custom_format(self) -> None:
        """Test setting custom format string."""
        custom_format = "%(name)s - %(message)s"
        logger = get_logger(format_string=custom_format, force_reconfigure=True)
        handler = logger.handlers[0]
        formatter = handler.formatter
        assert formatter is not None
        # Rich ignores custom format and uses message-only formatter
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

    def test_get_logger_no_duplicate_handlers(self) -> None:
        """Test that calling get_logger multiple times doesn't add duplicate handlers."""
        # Use a unique logger name for this test
        logger_name = f"dqx.test.no_dup_{uuid.uuid4().hex[:8]}"

        # First call
        logger1 = get_logger(logger_name)
        assert len(logger1.handlers) == 1

        # Second call
        logger2 = get_logger(logger_name)
        assert logger1 is logger2  # Same logger instance
        assert len(logger2.handlers) == 1  # Still only one handler

    def test_get_logger_force_reconfigure(self) -> None:
        """Test force_reconfigure parameter."""
        # Use a unique logger name for this test
        logger_name = f"dqx.test.reconfigure_{uuid.uuid4().hex[:8]}"

        # Create logger with INFO level
        logger = get_logger(logger_name, level=logging.INFO)
        assert logger.level == logging.INFO

        # Reconfigure with DEBUG level
        logger = get_logger(logger_name, level=logging.DEBUG, force_reconfigure=True)
        assert logger.level == logging.DEBUG

    def test_get_logger_force_reconfigure_clears_handlers(self) -> None:
        """Test that force_reconfigure clears existing handlers."""
        # Use a unique logger name for this test
        logger_name = f"dqx.test.clear_{uuid.uuid4().hex[:8]}"

        logger = get_logger(logger_name)
        original_handler = logger.handlers[0]

        # Force reconfigure with different format
        logger = get_logger(logger_name, format_string="%(message)s", force_reconfigure=True)

        assert len(logger.handlers) == 1
        assert logger.handlers[0] is not original_handler

    def test_get_logger_no_propagation(self) -> None:
        """Test that logger propagation is disabled."""
        logger = get_logger(force_reconfigure=True)
        assert logger.propagate is False

    def test_get_logger_default_format_constant(self) -> None:
        """Test that DEFAULT_FORMAT constant is used when format_string is None."""
        logger = get_logger("dqx.test.default_format", force_reconfigure=True)
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

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_logger_output(self, mock_stdout: StringIO) -> None:
        """Test actual logging output."""
        logger = get_logger("dqx.test.output", level=logging.INFO)
        logger.info("Test info message")

        output = mock_stdout.getvalue()
        assert "Test info message" in output
        assert "INFO" in output  # Rich shows full level name

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_logger_level_filtering(self, mock_stdout: StringIO) -> None:
        """Test that log level filtering works correctly."""
        logger = get_logger("dqx.test.filter", level=logging.WARNING)

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        output = mock_stdout.getvalue()
        assert "Debug message" not in output
        assert "Info message" not in output
        assert "Warning message" in output
        assert "Error message" in output

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
            logger = get_logger(f"dqx.test.level.{str_level}", level=int_level)
            assert logger.level == int_level

    @pytest.mark.demo
    def test_get_logger_stdout_demo(self) -> None:
        """
        Demonstration test that actually outputs to stdout.

        Run with: pytest tests/test___init__.py::TestGetLogger::test_get_logger_stdout_demo -s
        or: pytest -m demo -s

        This test shows actual logger output with different log levels and formats.
        """
        print("\n" + "=" * 60)
        print("DQX Logger Demonstration")
        print("=" * 60 + "\n")

        # Default logger with INFO level
        print("1. Default logger (INFO level):")
        print("-" * 40)
        logger = get_logger(force_reconfigure=True)
        logger.debug("This DEBUG message won't show (below INFO level)")
        logger.info("This is an INFO message")
        logger.warning("This is a WARNING message")
        logger.error("This is an ERROR message")
        logger.critical("This is a CRITICAL message")

        # Custom logger with DEBUG level
        print("\n2. Debug logger (DEBUG level):")
        print("-" * 40)
        debug_logger = get_logger("dqx.debug", level=logging.DEBUG, force_reconfigure=True)
        debug_logger.debug("Now DEBUG messages are visible!")
        debug_logger.info("INFO message from debug logger")

        # Logger with custom format
        print("\n3. Custom format logger:")
        print("-" * 40)
        custom_logger = get_logger(
            "dqx.custom", format_string="[%(levelname)s] %(name)s: %(message)s", force_reconfigure=True
        )
        custom_logger.info("Notice the different format!")
        custom_logger.warning("No timestamp in this format")

        # Hierarchical loggers
        print("\n4. Hierarchical loggers:")
        print("-" * 40)
        parent_logger = get_logger("dqx.parent", force_reconfigure=True)
        child_logger = get_logger("dqx.parent.child", force_reconfigure=True)
        parent_logger.info("Message from parent logger")
        child_logger.info("Message from child logger")

        # Logger with different levels filtering
        print("\n5. Level filtering demonstration:")
        print("-" * 40)
        filter_logger = get_logger("dqx.filter", level=logging.WARNING, force_reconfigure=True)
        filter_logger.debug("DEBUG: This won't show")
        filter_logger.info("INFO: This won't show")
        filter_logger.warning("WARNING: This will show")
        filter_logger.error("ERROR: This will show")

        print("\n" + "=" * 60)
        print("End of demonstration")
        print("=" * 60 + "\n")

        # Simple assertion to make it a valid test
        assert True
