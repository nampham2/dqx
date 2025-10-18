"""Additional tests to improve coverage for timer.py."""

from time import sleep

import pytest

from dqx.timer import Metric, Registry, TimeLimiting, Timer


def test_timer_context_manager() -> None:
    """Test Timer as a context manager."""
    registry = Registry()

    # Test Timer context manager
    with registry.timer("test_operation") as timer:
        sleep(0.01)  # Small delay to ensure measurable time

    # Verify elapsed_ms works after exit
    elapsed = timer.elapsed_ms()
    assert elapsed > 0

    # Verify the metric was collected in registry
    assert "test_operation" in registry
    assert registry["test_operation"] > 0


def test_timer_elapsed_ms_before_stop() -> None:
    """Test Timer.elapsed_ms() raises error when called before stop."""
    metric = Metric("test", Registry())
    timer = Timer(metric)

    # Enter the context but don't exit yet
    timer.__enter__()

    # Should raise RuntimeError since tock is None
    with pytest.raises(RuntimeError, match="Timer has not been stopped yet"):
        timer.elapsed_ms()

    # Clean up
    timer.__exit__(None, None, None)


def test_registry_timer_method() -> None:
    """Test Registry.timer() method creates Timer with correct metric."""
    registry = Registry()

    # Get a timer from registry
    timer = registry.timer("my_operation")

    # Verify it's a Timer instance
    assert isinstance(timer, Timer)

    # Use the timer
    with timer:
        sleep(0.01)

    # Verify the metric was recorded
    assert "my_operation" in registry
    assert registry["my_operation"] > 0


def test_time_limiting_without_limit() -> None:
    """Test TimeLimiting with None time limit (no limit)."""
    # Test with None - should not set alarm
    with TimeLimiting(None) as timer:
        sleep(0.01)  # Should complete without timeout
        result = 42

    # Verify it worked
    assert result == 42
    assert timer.elapsed_ms() > 0

    # Test with 0 (falsy) - should also not set alarm
    with TimeLimiting(0) as timer:
        sleep(0.01)
        result = 100

    assert result == 100
    assert timer.elapsed_ms() > 0


def test_metric_collect() -> None:
    """Test Metric.collect() method."""
    registry = Registry()
    metric = Metric("test_metric", registry)

    # Initially should be None
    assert metric.value is None

    # Collect a value
    metric.collect(123.45)

    # Verify it was stored
    assert metric.value == 123.45
    assert registry["test_metric"] == 123.45

    # Collect another value (should overwrite)
    metric.collect(678.90)
    assert metric.value == 678.90
    assert registry["test_metric"] == 678.90


def test_timer_with_exception() -> None:
    """Test Timer properly handles exceptions during execution."""
    registry = Registry()

    # Timer should still record time even if exception occurs
    try:
        with registry.timer("failed_operation"):
            sleep(0.01)
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Verify the time was still recorded
    assert "failed_operation" in registry
    assert registry["failed_operation"] > 0


def test_timed_decorator_with_exception() -> None:
    """Test Timer.timed decorator handles exceptions properly."""
    metric = Metric("error_func", Registry())

    @Timer.timed(collector=metric)
    def failing_function() -> None:
        sleep(0.01)
        raise RuntimeError("Test error")

    # Call the function and catch exception
    with pytest.raises(RuntimeError, match="Test error"):
        failing_function()

    # Verify time was still collected
    assert metric.value is not None
    assert metric.value > 0
