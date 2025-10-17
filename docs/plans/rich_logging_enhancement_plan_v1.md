# Rich Logging Enhancement Implementation Plan v1

## Overview

This plan outlines the implementation of colorized console logging for DQX using Rich's `RichHandler`. The enhancement will replace the standard `StreamHandler` with Rich's colorful, formatted logging while maintaining the existing API.

## Objectives

1. Replace standard logging with Rich's colorized output
2. Enable rich tracebacks for better exception formatting
3. Support markup in log messages for enhanced formatting
4. Maintain backward compatibility in the API
5. Ensure 100% test coverage

## Configuration Decisions

Based on requirements:
- **Rich Tracebacks**: `rich_tracebacks=True` - Enable beautiful exception formatting
- **Show Path**: `show_path=False` - Keep output clean without file paths
- **Markup Support**: `markup=True` - Allow rich text markup in log messages
- **Time Format**: `[%X]` - Use Rich's clean HH:MM:SS format

## Implementation Tasks

### Phase 1: Core Implementation (Tasks 1-3)

#### Task 1: Update get_logger() Function
**File**: `src/dqx/__init__.py`

**Current Implementation**:
```python
from logging import StreamHandler

# Inside get_logger()
handler = logging.StreamHandler()
formatter = logging.Formatter(format_string)
handler.setFormatter(formatter)
```

**New Implementation**:
```python
from rich.logging import RichHandler

# Inside get_logger()
handler = RichHandler(
    show_time=True,
    show_level=True,
    show_path=False,
    rich_tracebacks=True,
    markup=True,
    log_time_format="[%X]"
)

# Rich handles most formatting internally, but we keep message-only formatter
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
```

**Key Changes**:
1. Import `RichHandler` from `rich.logging`
2. Replace `StreamHandler` instantiation with `RichHandler`
3. Configure Rich-specific options
4. Simplify formatter to just `%(message)s` as Rich handles the rest

#### Task 2: Write Unit Tests for RichHandler
**File**: `tests/test_rich_logging.py` (new file)

```python
"""Tests for Rich logging integration."""

import logging
from unittest.mock import patch

import pytest
from rich.logging import RichHandler

from dqx import get_logger


def test_get_logger_returns_rich_handler():
    """Test that get_logger returns a logger with RichHandler."""
    logger = get_logger("test.rich")

    # Check that the logger has exactly one handler
    assert len(logger.handlers) == 1

    # Check that the handler is a RichHandler
    assert isinstance(logger.handlers[0], RichHandler)


def test_rich_handler_configuration():
    """Test that RichHandler is configured correctly."""
    logger = get_logger("test.config")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    # Rich stores these in the console object
    assert handler.console is not None
    assert handler._log_render.show_time is True
    assert handler._log_render.show_level is True
    assert handler._log_render.show_path is False


def test_rich_tracebacks_enabled():
    """Test that rich tracebacks are enabled."""
    logger = get_logger("test.tracebacks")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    assert handler.rich_tracebacks is True


def test_markup_enabled():
    """Test that markup is enabled by default."""
    logger = get_logger("test.markup")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    assert handler.markup is True


def test_time_format():
    """Test that the time format is set correctly."""
    logger = get_logger("test.time")
    handler = logger.handlers[0]

    assert isinstance(handler, RichHandler)
    assert handler._log_render.time_format == "[%X]"


def test_force_reconfigure_with_rich():
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


def test_logger_level_setting():
    """Test that logger level is set correctly."""
    # Test default level
    logger = get_logger("test.level.default")
    assert logger.level == logging.INFO

    # Test custom level
    debug_logger = get_logger("test.level.debug", level=logging.DEBUG)
    assert debug_logger.level == logging.DEBUG


def test_no_propagation():
    """Test that logger propagation is disabled."""
    logger = get_logger("test.propagate")
    assert logger.propagate is False
```

#### Task 3: Update Existing Tests
**Files to check and update**:
- `tests/test_logger.py` - Update if it checks handler types
- Any other tests that might check logging handler types

**Common updates needed**:
```python
# OLD
assert isinstance(logger.handlers[0], logging.StreamHandler)

# NEW
from rich.logging import RichHandler
assert isinstance(logger.handlers[0], RichHandler)
```

### Phase 2: Visual Demo and Testing (Tasks 4-5)

#### Task 4: Create Visual Logging Demo
**File**: `examples/rich_logging_demo.py` (new file)

```python
"""Demo of Rich logging capabilities in DQX."""

import logging
import time

from dqx import get_logger


def main() -> None:
    """Demonstrate Rich logging features."""
    # Get logger
    logger = get_logger("dqx.demo", level=logging.DEBUG)

    print("=== DQX Rich Logging Demo ===\n")

    # 1. Basic log levels
    print("1. Basic Log Levels:")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    time.sleep(0.5)
    print("\n2. Rich Markup Support:")

    # 2. Rich markup examples
    logger.info("[bold cyan]Bold cyan text[/bold cyan] in log message")
    logger.warning("Contains [yellow]highlighted[/yellow] warning text")
    logger.error("[red]Error[/red] with [bold]emphasis[/bold]")

    time.sleep(0.5)
    print("\n3. Structured Information:")

    # 3. Structured logging
    logger.info("Processing data: [green]✓[/green] Stage 1 complete")
    logger.info("Processing data: [green]✓[/green] Stage 2 complete")
    logger.info("Processing data: [red]✗[/red] Stage 3 failed")

    time.sleep(0.5)
    print("\n4. Exception with Rich Traceback:")

    # 4. Exception logging with rich traceback
    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.exception("Failed to perform calculation")

    time.sleep(0.5)
    print("\n5. Long Messages:")

    # 5. Long message handling
    long_message = (
        "This is a very long log message that demonstrates how Rich handles "
        "text wrapping in the console. It should wrap nicely at the terminal "
        "width while maintaining the log level indicator and timestamp alignment."
    )
    logger.info(long_message)

    time.sleep(0.5)
    print("\n6. Data Validation Example:")

    # 6. DQX-specific example
    logger.info("[bold]Starting data quality check[/bold]")
    logger.info("Analyzing dataset: [cyan]sales_data[/cyan]")
    logger.info("  • Checking metric: [green]total_revenue[/green]")
    logger.info("  • Assertion: total_revenue > 0")
    logger.warning("  • Result: [yellow]FAILURE[/yellow] - total_revenue = -100")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
```

#### Task 5: Manual Testing Script
**File**: `.tmp/test_rich_logging.py` (temporary test file)

```python
"""Manual testing script for Rich logging."""

import logging
import sys

from dqx import get_logger


def test_different_levels():
    """Test different logging levels."""
    for level_name, level in [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ]:
        logger = get_logger(f"test.{level_name.lower()}", level=level)
        print(f"\n--- Testing {level_name} level ---")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")


def test_exception_handling():
    """Test exception handling with rich tracebacks."""
    logger = get_logger("test.exception")

    print("\n--- Testing Exception Handling ---")
    try:
        # Nested function calls for better traceback
        def inner():
            return 1 / 0

        def middle():
            return inner()

        def outer():
            return middle()

        outer()
    except Exception:
        logger.exception("Caught an exception")


def test_markup():
    """Test markup support."""
    logger = get_logger("test.markup")

    print("\n--- Testing Markup ---")
    logger.info("Plain text")
    logger.info("[bold]Bold text[/bold]")
    logger.info("[red]Red[/red] [green]Green[/green] [blue]Blue[/blue]")
    logger.info("[bold red on yellow]Alert![/bold red on yellow]")


def test_non_tty():
    """Test behavior when output is not a TTY."""
    print("\n--- Testing Non-TTY Output ---")
    print("Redirect output to see plain text: python test_rich_logging.py > output.txt")

    logger = get_logger("test.nontty")
    logger.info("This should be plain text when redirected")


if __name__ == "__main__":
    test_different_levels()
    test_exception_handling()
    test_markup()
    test_non_tty()
```

### Phase 3: Documentation and Final Steps (Tasks 6-7)

#### Task 6: Update Documentation
**File**: `src/dqx/__init__.py` - Update docstring

```python
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
```

**File**: Update README.md or relevant docs to mention Rich logging

```markdown
## Logging

DQX uses [Rich](https://rich.readthedocs.io/) for beautiful, colorized console logging:

```python
from dqx import get_logger

logger = get_logger()
logger.info("Starting validation")
logger.warning("[yellow]Warning:[/yellow] Missing data detected")
logger.error("Validation failed")
```

Features:
- Color-coded log levels
- Clean timestamps (HH:MM:SS format)
- Beautiful exception tracebacks
- Optional Rich markup support for enhanced formatting
- Automatic plain text when output is redirected
```

#### Task 7: Run Pre-commit and Tests
**Commands to run**:

```bash
# Run mypy first
uv run mypy src/dqx tests/

# Run ruff
uv run ruff check --fix src/dqx tests/

# Run specific test file
uv run pytest tests/test_rich_logging.py -v

# Run all tests with coverage
uv run pytest tests/ -v --cov=dqx

# Run pre-commit hooks
bin/run-hooks.sh
```

## Success Criteria

1. ✅ `get_logger()` returns a logger with `RichHandler`
2. ✅ Log levels are color-coded in terminal output
3. ✅ Rich markup is supported in log messages
4. ✅ Exceptions show rich tracebacks with syntax highlighting
5. ✅ All existing tests pass
6. ✅ New tests achieve 100% coverage for changes
7. ✅ Demo script shows visual improvements
8. ✅ Documentation is updated
9. ✅ Pre-commit hooks pass

## Rollback Plan

If issues arise:
1. The change is isolated to `get_logger()` function
2. Can revert by changing `RichHandler` back to `StreamHandler`
3. No data or state changes involved

## Notes

- Rich automatically detects TTY and disables colors for file output
- The `format_string` parameter is kept for API compatibility but ignored
- Rich's default colors work well in both light and dark terminals
- Performance impact is minimal as Rich only adds formatting for TTY output
