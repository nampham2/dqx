"""Tests for plugin data structures."""

from dataclasses import FrozenInstanceError
from datetime import date

import pyarrow as pa
import pytest
from returns.result import Failure, Success

from dqx.common import (
    AssertionResult,
    PluginMetadata,
    ResultKey,
)
from dqx.orm.repositories import MetricStats
from dqx.plugins import PluginExecutionContext
from dqx.provider import SymbolInfo


def _create_empty_trace() -> pa.Table:
    """Create an empty PyArrow table for trace parameter."""
    return pa.table({})


def test_plugin_metadata_frozen() -> None:
    """Test that PluginMetadata is immutable."""
    metadata = PluginMetadata(
        name="test", version="1.0.0", author="Test Author", description="Test plugin", capabilities={"test"}
    )

    # Should not be able to modify
    with pytest.raises(FrozenInstanceError):
        metadata.name = "changed"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        metadata.capabilities = {"changed"}  # type: ignore[misc]


def test_plugin_metadata_creation() -> None:
    """Test PluginMetadata creation with all fields."""
    metadata = PluginMetadata(
        name="my_plugin",
        version="2.1.0",
        author="John Doe",
        description="A test plugin for testing",
        capabilities={"email", "slack", "webhook"},
    )

    assert metadata.name == "my_plugin"
    assert metadata.version == "2.1.0"
    assert metadata.author == "John Doe"
    assert metadata.description == "A test plugin for testing"
    assert metadata.capabilities == {"email", "slack", "webhook"}


def test_plugin_metadata_default_capabilities() -> None:
    """Test PluginMetadata with default empty capabilities."""
    metadata = PluginMetadata(name="minimal", version="1.0.0", author="Author", description="Minimal plugin")

    assert metadata.capabilities == set()


def test_plugin_execution_context_creation() -> None:
    """Test PluginExecutionContext creation."""
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 1),
            suite="test",
            check="check1",
            assertion="assert1",
            severity="P1",
            status="OK",
            metric=Success(100.0),
            expression="x > 0",
            tags={},
        )
    ]

    symbols = [
        SymbolInfo(
            name="x",
            metric="count",
            dataset="ds1",
            value=Success(10.0),
            yyyy_mm_dd=date(2024, 1, 1),
            tags={},
        )
    ]

    context = PluginExecutionContext(
        suite_name="test",
        execution_id="test_creation",
        datasources=["ds1", "ds2"],
        key=ResultKey(date(2024, 1, 1), {"env": "prod"}),
        timestamp=1704067200.0,
        duration_ms=2500.0,
        results=results,
        symbols=symbols,
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
    )

    assert context.suite_name == "test"
    assert context.datasources == ["ds1", "ds2"]
    assert context.key.yyyy_mm_dd == date(2024, 1, 1)
    assert context.key.tags == {"env": "prod"}
    assert context.timestamp == 1704067200.0
    assert context.duration_ms == 2500.0
    assert len(context.results) == 1
    assert len(context.symbols) == 1


def test_context_total_assertions() -> None:
    """Test total_assertions method."""
    context = PluginExecutionContext(
        suite_name="test",
        execution_id="test_total_assertions",
        datasources=[],
        key=ResultKey(date(2024, 1, 1), {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=[
            AssertionResult(
                yyyy_mm_dd=date(2024, 1, 1),
                suite="test",
                check="check",
                assertion="assert",
                severity="P1",
                status="OK",
                metric=Success(100.0),
                expression="x > 0",
                tags={},
            )
            for _ in range(5)
        ],
        symbols=[],
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
    )

    assert context.total_assertions() == 5


def test_context_failed_assertions() -> None:
    """Test failed_assertions method."""
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 1),
            suite="test",
            check="check",
            assertion=f"assert{i}",
            severity="P1",
            status="FAILURE" if i < 3 else "OK",
            metric=Success(100.0),
            expression="x > 0",
            tags={},
        )
        for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        execution_id="test_failed_assertions",
        datasources=[],
        key=ResultKey(date(2024, 1, 1), {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=results,
        symbols=[],
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
    )

    assert context.failed_assertions() == 3
    assert context.passed_assertions() == 2


def test_context_assertion_pass_rate() -> None:
    """Test assertion_pass_rate calculation."""
    # 3 passed, 2 failed = 60% pass rate
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 1),
            suite="test",
            check="check",
            assertion=f"assert{i}",
            severity="P1",
            status="OK" if i < 3 else "FAILURE",
            metric=Success(100.0),
            expression="x > 0",
            tags={},
        )
        for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        execution_id="test_pass_rate",
        datasources=[],
        key=ResultKey(date(2024, 1, 1), {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=results,
        symbols=[],
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
    )

    assert context.assertion_pass_rate() == 60.0

    # Empty results = 100% pass rate
    empty_context = PluginExecutionContext(
        suite_name="test",
        execution_id="test_empty_pass_rate",
        datasources=[],
        key=ResultKey(date(2024, 1, 1), {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=[],
        symbols=[],
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
    )

    assert empty_context.assertion_pass_rate() == 100.0


def test_context_symbol_methods() -> None:
    """Test symbol-related methods."""
    symbols = [
        SymbolInfo(
            name=f"sym{i}",
            metric="count",
            dataset="ds1",
            value=Success(float(i)) if i < 3 else Failure("Error"),
            yyyy_mm_dd=date(2024, 1, 1),
            tags={},
        )
        for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        execution_id="test_symbol_methods",
        datasources=["ds1"],
        key=ResultKey(date(2024, 1, 1), {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=[],
        symbols=symbols,
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
    )

    assert context.total_symbols() == 5
    assert context.failed_symbols() == 2


def test_context_assertions_by_severity() -> None:
    """Test assertions_by_severity grouping."""
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 1),
            suite="test",
            check="check",
            assertion=f"assert{i}",
            severity=f"P{i % 3}" if i % 3 < 4 else "P3",  # type: ignore[arg-type]
            status="OK",
            metric=Success(100.0),
            expression="x > 0",
            tags={},
        )
        for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        execution_id="test_by_severity",
        datasources=[],
        key=ResultKey(date(2024, 1, 1), {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=results,
        symbols=[],
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
    )

    by_severity = context.assertions_by_severity()
    assert by_severity == {"P0": 2, "P1": 2, "P2": 1}


def test_context_failures_by_severity() -> None:
    """Test failures_by_severity grouping."""
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 1),
            suite="test",
            check="check",
            assertion=f"assert{i}",
            severity=f"P{i}" if i < 4 else "P3",  # type: ignore[arg-type]
            status="FAILURE" if i < 3 else "OK",  # P0, P1, P2 fail
            metric=Success(100.0),
            expression="x > 0",
            tags={},
        )
        for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        execution_id="test_failures_by_severity",
        datasources=[],
        key=ResultKey(date(2024, 1, 1), {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=results,
        symbols=[],
        trace=_create_empty_trace(),
        metrics_stats=MetricStats(total_metrics=0, expired_metrics=0),
    )

    failures = context.failures_by_severity()
    assert failures == {"P0": 1, "P1": 1, "P2": 1}
    # P3 and P4 passed, so not in failures
