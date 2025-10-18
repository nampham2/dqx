# Plugin Architecture Implementation Plan

## Overview

This plan implements a plugin system for DQX that allows external integrations to process validation results. The plugin system will enable third-party packages to register processors via Python entry points in `pyproject.toml` and automatically execute them after validation suite runs.

## Background

Currently, DQX executes validation checks and collects results, but has no mechanism for external systems to react to these results. This plugin architecture will allow:

- Sending alerts/emails when assertions fail
- Writing results to custom databases
- Workflow & Orchestration Integration: Trigger downstream workflows in Airflow/Prefect/Dagster based on validation results
- Data Catalog Integration: Sync quality metrics to DataHub, Amundsen, or Collibra for unified data discovery
- Integration with monitoring systems: Send metrics to Prometheus, DataDog, or New Relic for operational visibility
- Dashboarding & debugging: Export results to Tableau/PowerBI or create custom dashboards for troubleshooting data issues
- Custom reporting and analytics

## Design Decisions

1. **Protocol-based Interface**: Following DQX's existing patterns (SqlDataSource, Analyzer protocols)
2. **Entry Point Discovery**: Using Python's standard `importlib.metadata` for plugin discovery
3. **Single File Implementation**: All plugin code in `src/dqx/plugins.py` for simplicity
4. **Full Type Annotations**: Ensuring mypy compliance with `disallow_untyped_defs = true`

## Implementation Guide

Each task group below is self-contained and ends with validation and a git commit. Complete each group before moving to the next.

---

## Task Group 1: Core Plugin Infrastructure

### Step 1.1: Create the Plugin Module

Create `src/dqx/plugins.py` with the complete implementation:

```python
"""Plugin system for DQX result processing."""

import importlib.metadata
import logging
import time
from typing import Any, Protocol

from dqx.common import AssertionResult, SymbolInfo

logger = logging.getLogger(__name__)


class ResultProcessor(Protocol):
    """Protocol that all DQX result processor plugins must implement."""

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """
        Process validation results after suite execution.

        Args:
            results: List of assertion results from the suite
            symbols: List of symbol values from the suite
            suite_name: Name of the verification suite
            context: Additional context (datasources, key, timestamp)
        """
        ...

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.

        Args:
            config: Plugin-specific configuration dictionary
        """
        ...


class PluginManager:
    """Manages DQX result processor plugins."""

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self._plugins: dict[str, ResultProcessor] = {}
        self._load_plugins()

    def _load_plugins(self) -> None:
        """Discover and load plugins from entry points."""
        try:
            # Discover all plugins in the "dqx.plugins" group
            entry_points = importlib.metadata.entry_points(group="dqx.plugins")

            for ep in entry_points:
                try:
                    logger.info(f"Loading plugin: {ep.name}")

                    # Load the plugin class
                    plugin_class = ep.load()

                    # Instantiate the plugin
                    plugin = plugin_class()

                    # Verify it implements the protocol
                    if not isinstance(plugin, ResultProcessor):
                        logger.error(
                            f"Plugin {ep.name} does not implement ResultProcessor protocol"
                        )
                        continue

                    # Store the plugin
                    self._plugins[ep.name] = plugin
                    logger.info(f"Successfully loaded plugin: {ep.name}")

                except Exception as e:
                    logger.error(f"Failed to load plugin {ep.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")

    def get_plugins(self) -> dict[str, ResultProcessor]:
        """
        Get all loaded plugins.

        Returns:
            Dictionary mapping plugin names to plugin instances
        """
        return self._plugins

    def process_all(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any],
        config: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """
        Process results through all loaded plugins.

        Args:
            results: List of assertion results
            symbols: List of symbol values
            suite_name: Name of the verification suite
            context: Execution context
            config: Optional per-plugin configuration
        """
        if not self._plugins:
            logger.debug("No plugins loaded, skipping plugin processing")
            return

        logger.info(f"Processing results through {len(self._plugins)} plugin(s)")

        for name, plugin in self._plugins.items():
            try:
                # Initialize plugin with config if provided
                if config and name in config:
                    plugin.initialize(config[name])
                else:
                    plugin.initialize({})

                # Process results
                start_time = time.time()
                plugin.process(results, symbols, suite_name, context)
                elapsed = time.time() - start_time

                logger.info(f"Plugin {name} processed results in {elapsed:.2f}s")

            except Exception as e:
                # Log error but don't fail the entire suite
                logger.error(f"Plugin {name} failed during processing: {e}")
```

### Step 1.2: Validation

Run the following commands to ensure the code is correct:

```bash
# Type check
uv run mypy src/dqx/plugins.py

# Lint check
uv run ruff check src/dqx/plugins.py

# Fix any linting issues
uv run ruff check --fix src/dqx/plugins.py
```

### Step 1.3: Git Commit

Once validation passes:

```bash
git add src/dqx/plugins.py
git commit -m "feat(plugins): add ResultProcessor protocol and PluginManager

- Define ResultProcessor protocol for plugin interface
- Implement PluginManager with entry point discovery
- Add error handling and logging for plugin lifecycle
- Include full type annotations for mypy compliance"
```

---

## Task Group 2: API Integration

### Step 2.1: Add Import to api.py

Add this import at the top of `src/dqx/api.py`:

```python
from dqx.plugins import PluginManager
```

Also add `time` import if not already present:

```python
import time
```

### Step 2.2: Modify VerificationSuite Constructor

Update the `__init__` method signature and add plugin support attributes:

```python
def __init__(
    self,
    checks: Sequence[CheckProducer | DecoratedCheck],
    db: MetricDB,
    name: str,
    plugin_config: dict[str, dict[str, Any]] | None = None,  # NEW
) -> None:
    """
    Initialize the verification suite.

    Args:
        checks: Sequence of check functions to execute
        db: Database for storing and retrieving metrics
        name: Human-readable name for the suite
        plugin_config: Optional configuration for plugins  # NEW

    Raises:
        DQXError: If no checks provided or name is empty
    """
    if not checks:
        raise DQXError("At least one check must be provided")
    if not name.strip():
        raise DQXError("Suite name cannot be empty")

    self._checks: Sequence[CheckProducer | DecoratedCheck] = checks
    self._name = name.strip()

    # Create a context
    self._context = Context(suite=self._name, db=db)

    # State tracking for result collection
    self.is_evaluated = False
    self._key: ResultKey | None = None

    # Graph state tracking
    self._graph_built = False

    # Plugin support
    self._plugin_manager: PluginManager | None = None  # NEW
    self._plugin_config = plugin_config  # NEW
```

### Step 2.3: Add Plugin Manager Lazy Loading

Add this method after the constructor:

```python
def _get_plugin_manager(self) -> PluginManager:
    """Get or create plugin manager instance (lazy loading)."""
    if self._plugin_manager is None:
        self._plugin_manager = PluginManager()
    return self._plugin_manager
```

### Step 2.4: Integrate Plugin Execution in run()

In the `run()` method, after the line `self.is_evaluated = True`, add:

```python
# Execute plugins if any are loaded
plugin_manager = self._get_plugin_manager()
if plugin_manager.get_plugins():
    logger.info("Executing result processor plugins...")

    # Create execution context
    context = {
        "datasources": list(datasources.keys()),
        "key": key,
        "timestamp": time.time(),
    }

    # Collect results and symbols
    results = self.collect_results()
    symbols = self.collect_symbols()

    # Process through plugins
    plugin_manager.process_all(
        results=results,
        symbols=symbols,
        suite_name=self._name,
        context=context,
        config=self._plugin_config
    )
```

### Step 2.5: Validation

```bash
# Type check
uv run mypy src/dqx/api.py

# Lint check
uv run ruff check src/dqx/api.py

# Run existing API tests to ensure backward compatibility
uv run pytest tests/test_api.py -v
```

### Step 2.6: Git Commit

```bash
git add src/dqx/api.py
git commit -m "feat(api): integrate plugin execution in VerificationSuite

- Add plugin_config parameter to VerificationSuite
- Implement lazy loading of PluginManager
- Execute plugins after suite evaluation
- Pass results, symbols, and context to plugins"
```

---

## Task Group 3: Test Infrastructure

### Step 3.1: Create Plugin Tests

Create `tests/test_plugins.py` with the complete test suite:

```python
"""Tests for the plugin system."""

import importlib.metadata
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dqx.common import AssertionResult, ResultValue, SymbolInfo
from dqx.plugins import PluginManager, ResultProcessor


class MockSuccessPlugin:
    """Mock plugin that tracks all calls."""

    def __init__(self) -> None:
        self.initialized = False
        self.processed = False
        self.config: dict[str, Any] = {}
        self.received_results: list[AssertionResult] = []
        self.received_symbols: list[SymbolInfo] = []
        self.received_suite_name = ""
        self.received_context: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any]) -> None:
        """Track initialization."""
        self.initialized = True
        self.config = config

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """Track processing."""
        self.processed = True
        self.received_results = results
        self.received_symbols = symbols
        self.received_suite_name = suite_name
        self.received_context = context


class MockFailurePlugin:
    """Mock plugin that raises exceptions."""

    def initialize(self, config: dict[str, Any]) -> None:
        """Raise during initialization."""
        raise RuntimeError("Plugin initialization failed")

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """Raise during processing."""
        raise RuntimeError("Plugin processing failed")


class NotAPlugin:
    """Class that doesn't implement ResultProcessor."""

    def do_something(self) -> None:
        """Some unrelated method."""
        pass


def test_plugin_manager_initialization() -> None:
    """Test that PluginManager initializes correctly."""
    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        assert isinstance(manager._plugins, dict)
        assert len(manager._plugins) == 0


def test_plugin_discovery_success() -> None:
    """Test successful plugin discovery and loading."""
    # Create mock entry point
    mock_ep = MagicMock()
    mock_ep.name = "test_plugin"
    mock_ep.load.return_value = MockSuccessPlugin

    with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
        manager = PluginManager()

        assert len(manager._plugins) == 1
        assert "test_plugin" in manager._plugins
        assert isinstance(manager._plugins["test_plugin"], MockSuccessPlugin)


def test_plugin_discovery_invalid_plugin() -> None:
    """Test handling of plugins that don't implement ResultProcessor."""
    # Create mock entry point for invalid plugin
    mock_ep = MagicMock()
    mock_ep.name = "invalid_plugin"
    mock_ep.load.return_value = NotAPlugin

    with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
        manager = PluginManager()

        # Invalid plugin should not be loaded
        assert len(manager._plugins) == 0


def test_plugin_discovery_load_failure() -> None:
    """Test handling of plugins that fail to load."""
    # Create mock entry point that raises on load
    mock_ep = MagicMock()
    mock_ep.name = "broken_plugin"
    mock_ep.load.side_effect = ImportError("Cannot import plugin")

    with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
        manager = PluginManager()

        # Broken plugin should not crash the manager
        assert len(manager._plugins) == 0


def test_get_plugins() -> None:
    """Test getting loaded plugins."""
    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        plugins = manager.get_plugins()

        assert isinstance(plugins, dict)
        assert len(plugins) == 0


def test_process_all_no_plugins() -> None:
    """Test process_all when no plugins are loaded."""
    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()

        # Should not raise any errors
        manager.process_all([], [], "test_suite", {})


def test_process_all_with_plugin() -> None:
    """Test processing results through a plugin."""
    # Create mock plugin
    mock_plugin = MockSuccessPlugin()

    # Create test data
    result = AssertionResult(
        yyyy_mm_dd="2024-01-01",
        suite="test_suite",
        check="test_check",
        assertion="test_assertion",
        severity="P1",
        status="SUCCESS",
        metric="test_metric",
        expression="x > 0",
        tags={"env": "test"}
    )

    symbol = SymbolInfo(
        name="x_1",
        metric="average",
        dataset="test_dataset",
        value=ResultValue.success(42.0),
        yyyy_mm_dd="2024-01-01",
        suite="test_suite",
        tags={"env": "test"}
    )

    context = {"key": "value"}
    config = {"test_plugin": {"setting": "value"}}

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["test_plugin"] = mock_plugin

        manager.process_all(
            results=[result],
            symbols=[symbol],
            suite_name="test_suite",
            context=context,
            config=config
        )

        # Verify plugin was initialized and processed
        assert mock_plugin.initialized
        assert mock_plugin.processed
        assert mock_plugin.config == {"setting": "value"}
        assert len(mock_plugin.received_results) == 1
        assert len(mock_plugin.received_symbols) == 1
        assert mock_plugin.received_suite_name == "test_suite"
        assert mock_plugin.received_context == context


def test_process_all_plugin_failure() -> None:
    """Test that plugin failures don't crash the suite."""
    # Create failing plugin
    mock_plugin = MockFailurePlugin()

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["failing_plugin"] = mock_plugin

        # Should not raise - failures are logged
        manager.process_all([], [], "test_suite", {})


def test_process_all_without_config() -> None:
    """Test processing without plugin configuration."""
    mock_plugin = MockSuccessPlugin()

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["test_plugin"] = mock_plugin

        manager.process_all(
            results=[],
            symbols=[],
            suite_name="test_suite",
            context={},
            config=None  # No config provided
        )

        assert mock_plugin.initialized
        assert mock_plugin.config == {}  # Should get empty config


def test_multiple_plugins() -> None:
    """Test processing through multiple plugins."""
    plugin1 = MockSuccessPlugin()
    plugin2 = MockSuccessPlugin()

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["plugin1"] = plugin1
        manager._plugins["plugin2"] = plugin2

        manager.process_all([], [], "test_suite", {})

        assert plugin1.processed
        assert plugin2.processed


def test_plugin_execution_time_logging() -> None:
    """Test that plugin execution time is logged."""
    mock_plugin = MockSuccessPlugin()

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["test_plugin"] = mock_plugin

        with patch("dqx.plugins.logger") as mock_logger:
            manager.process_all([], [], "test_suite", {})

            # Check that execution time was logged
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("processed results in" in call for call in info_calls)
```

### Step 3.2: Add Integration Test

Add this to `tests/test_api.py` (or create a new file `tests/test_api_plugins.py`):

```python
def test_verification_suite_plugin_config() -> None:
    """Test that VerificationSuite accepts plugin configuration."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB

    @check(name="test_check")
    def my_check(mp, ctx):
        pass

    # Should accept plugin_config parameter
    suite = VerificationSuite(
        checks=[my_check],
        db=MetricDB(),
        name="Test Suite",
        plugin_config={"test": {"key": "value"}}
    )

    assert suite._plugin_config == {"test": {"key": "value"}}
```

### Step 3.3: Validation

```bash
# Type check
uv run mypy tests/test_plugins.py

# Run the new tests
uv run pytest tests/test_plugins.py -v

# Check coverage
uv run pytest tests/test_plugins.py -v --cov=dqx.plugins --cov-report=term-missing
```

### Step 3.4: Git Commit

```bash
git add tests/test_plugins.py tests/test_api.py
git commit -m "test(plugins): add comprehensive plugin system tests

- Test plugin discovery and loading
- Test error handling for invalid/failing plugins
- Test configuration passing and processing
- Add mock plugins for testing
- Ensure 100% code coverage"
```

---

## Task Group 4: Built-in Audit Plugin

### Step 4.1: Create Plugin Directory Structure

Create the plugins submodule structure:

```bash
mkdir -p src/dqx/plugins
touch src/dqx/plugins/__init__.py
```

### Step 4.2: Create Audit Plugin

Create `src/dqx/plugins/audit.py`:

```python
"""Built-in audit plugin for DQX execution tracking."""

import logging
import time
from pathlib import Path
from typing import Any

from dqx.common import AssertionResult, SymbolInfo

logger = logging.getLogger(__name__)


class AuditPlugin:
    """
    DQX built-in audit plugin for tracking suite execution.

    This plugin provides basic auditing functionality including:
    - Execution timing
    - Result statistics
    - Performance metrics
    """

    def __init__(self) -> None:
        """Initialize the audit plugin."""
        self.start_time: float | None = None
        self.audit_file: Path | None = None
        self.log_to_file: bool = False

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize with audit configuration.

        Args:
            config: Configuration with optional keys:
                - audit_file: Path to write audit logs
                - log_to_file: Whether to write to file (default: False)
        """
        self.log_to_file = config.get("log_to_file", False)

        if self.log_to_file:
            audit_file_path = config.get("audit_file", "dqx_audit.log")
            self.audit_file = Path(audit_file_path)
            logger.info(f"Audit logging to file: {self.audit_file}")

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """
        Process and audit the validation results.

        Args:
            results: List of assertion results
            symbols: List of symbol values
            suite_name: Name of the verification suite
            context: Execution context
        """
        # Calculate statistics
        total_assertions = len(results)
        passed = sum(1 for r in results if r.status == "SUCCESS")
        failed = sum(1 for r in results if r.status == "FAILURE")

        # Group failures by severity
        failures_by_severity: dict[str, int] = {}
        for r in results:
            if r.status == "FAILURE":
                failures_by_severity[r.severity] = failures_by_severity.get(r.severity, 0) + 1

        # Calculate unique checks and datasets
        unique_checks = len(set(r.check for r in results))
        unique_datasets = context.get("datasources", [])

        # Build audit message
        timestamp = context.get("timestamp", time.time())
        audit_lines = [
            f"=== DQX Audit Log ===",
            f"Suite: {suite_name}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}",
            f"Date: {context.get('key', {}).get('yyyy_mm_dd', 'N/A')}",
            f"Datasets: {', '.join(unique_datasets) if unique_datasets else 'None'}",
            f"",
            f"Execution Summary:",
            f"  Total Checks: {unique_checks}",
            f"  Total Assertions: {total_assertions}",
            f"  Passed: {passed} ({passed/total_assertions*100:.1f}%)" if total_assertions else "  Passed: 0",
            f"  Failed: {failed} ({failed/total_assertions*100:.1f}%)" if total_assertions else "  Failed: 0",
        ]

        if failures_by_severity:
            audit_lines.extend([
                f"",
                f"Failures by Severity:",
            ])
            for severity, count in sorted(failures_by_severity.items()):
                audit_lines.append(f"  {severity}: {count}")

        # Add symbol statistics
        if symbols:
            successful_symbols = sum(1 for s in symbols if s.value.is_success())
            audit_lines.extend([
                f"",
                f"Symbol Statistics:",
                f"  Total Symbols: {len(symbols)}",
                f"  Successful: {successful_symbols}",
                f"  Failed: {len(symbols) - successful_symbols}",
            ])

        audit_lines.append("=" * 20)

        # Log the audit
        audit_message = "\n".join(audit_lines)
        logger.info(f"Audit Report:\n{audit_message}")

        # Write to file if configured
        if self.log_to_file and self.audit_file:
            try:
                with open(self.audit_file, "a") as f:
                    f.write(f"{audit_message}\n\n")
                logger.debug(f"Audit written to {self.audit_file}")
            except Exception as e:
                logger.error(f"Failed to write audit to file: {e}")
```

### Step 4.3: Update Package __init__.py

Update `src/dqx/plugins/__init__.py`:

```python
"""DQX built-in plugins."""

from dqx.plugins.audit import AuditPlugin

__all__ = ["AuditPlugin"]
```

### Step 4.4: Register Built-in Plugin

Add to `pyproject.toml` in the appropriate section:

```toml
[project.entry-points."dqx.plugins"]
dqx_audit = "dqx.plugins.audit:AuditPlugin"
```

### Step 4.5: Create Tests for Audit Plugin

Create `tests/test_audit_plugin.py`:

```python
"""Tests for the built-in audit plugin."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dqx.common import AssertionResult, ResultValue, SymbolInfo
from dqx.plugins.audit import AuditPlugin


def create_test_results() -> tuple[list[AssertionResult], list[SymbolInfo]]:
    """Create test data for audit plugin testing."""
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test_suite",
            check="check1",
            assertion="assertion1",
            severity="P1",
            status="SUCCESS",
            metric="metric1",
            expression="x > 0",
            tags={}
        ),
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test_suite",
            check="check1",
            assertion="assertion2",
            severity="P0",
            status="FAILURE",
            metric="metric2",
            expression="y < 10",
            tags={}
        ),
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test_suite",
            check="check2",
            assertion="assertion3",
            severity="P1",
            status="FAILURE",
            metric="metric3",
            expression="z == 0",
            tags={}
        ),
    ]

    symbols = [
        SymbolInfo(
            name="x",
            metric="average",
            dataset="dataset1",
            value=ResultValue.success(5.0),
            yyyy_mm_dd="2024-01-01",
            suite="test_suite",
            tags={}
        ),
        SymbolInfo(
            name="y",
            metric="count",
            dataset="dataset2",
            value=ResultValue.failure(Exception("Failed to compute")),
            yyyy_mm_dd="2024-01-01",
            suite="test_suite",
            tags={}
        ),
    ]

    return results, symbols


def test_audit_plugin_initialization() -> None:
    """Test audit plugin initializes correctly."""
    plugin = AuditPlugin()

    # Default initialization
    plugin.initialize({})
    assert not plugin.log_to_file
    assert plugin.audit_file is None

    # With file logging
    plugin.initialize({"log_to_file": True, "audit_file": "test.log"})
    assert plugin.log_to_file
    assert plugin.audit_file == Path("test.log")


def test_audit_plugin_process_logs_statistics() -> None:
    """Test audit plugin logs execution statistics."""
    plugin = AuditPlugin()
    plugin.initialize({})

    results, symbols = create_test_results()
    context = {
        "datasources": ["dataset1", "dataset2"],
        "key": {"yyyy_mm_dd": "2024-01-01"},
        "timestamp": 1704067200.0,  # 2024-01-01 00:00:00 UTC
    }

    with patch("dqx.plugins.audit.logger") as mock_logger:
        plugin.process(results, symbols, "Test Suite", context)

        # Verify logger was called
        assert mock_logger.info.called

        # Get the logged message
        call_args = mock_logger.info.call_args[0][0]

        # Verify content
        assert "Suite: Test Suite" in call_args
        assert "Total Checks: 2" in call_args
        assert "Total Assertions: 3" in call_args
        assert "Passed: 1 (33.3%)" in call_args
        assert "Failed: 2 (66.7%)" in call_args
        assert "P0: 1" in call_args
        assert "P1: 1" in call_args
        assert "Total Symbols: 2" in call_args


def test_audit_plugin_file_writing() -> None:
    """Test audit plugin writes to file when configured."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
        audit_file = Path(tf.name)

    try:
        plugin = AuditPlugin()
        plugin.initialize({"log_to_file": True, "audit_file": str(audit_file)})

        results, symbols = create_test_results()
        context = {
            "datasources": ["dataset1"],
            "key": {"yyyy_mm_dd": "2024-01-01"},
        }

        plugin.process(results, symbols, "File Test Suite", context)

        # Verify file was written
        assert audit_file.exists()
        content = audit_file.read_text()

        assert "=== DQX Audit Log ===" in content
        assert "Suite: File Test Suite" in content
        assert "Total Assertions: 3" in content

    finally:
        # Clean up
        if audit_file.exists():
            audit_file.unlink()


def test_audit_plugin_handles_empty_results() -> None:
    """Test audit plugin handles empty results gracefully."""
    plugin = AuditPlugin()
    plugin.initialize({})

    # Should not crash with empty data
    plugin.process([], [], "Empty Suite", {})


def test_audit_plugin_file_write_error_handling() -> None:
    """Test audit plugin handles file write errors gracefully."""
    plugin = AuditPlugin()
    plugin.initialize({
        "log_to_file": True,
        "audit_file": "/invalid/path/that/does/not/exist/audit.log"
    })

    results, symbols = create_test_results()

    # Should log error but not crash
    with patch("dqx.plugins.audit.logger") as mock_logger:
        plugin.process(results, symbols, "Test Suite", {})

        # Verify error was logged
        error_calls = [call for call in mock_logger.error.call_args_list]
        assert len(error_calls) > 0
        assert "Failed to write audit to file" in str(error_calls[0])
```

### Step 4.6: Validation

```bash
# Type check
uv run mypy src/dqx/plugins/audit.py

# Lint check
uv run ruff check src/dqx/plugins/

# Run tests
uv run pytest tests/test_audit_plugin.py -v

# Check coverage
uv run pytest tests/test_audit_plugin.py -v --cov=dqx.plugins.audit --cov-report=term-missing
```

### Step 4.7: Git Commit

```bash
git add src/dqx/plugins/ tests/test_audit_plugin.py pyproject.toml
git commit -m "feat(plugins): add built-in audit plugin

- Create AuditPlugin for execution tracking and statistics
- Register plugin via entry point in pyproject.toml
- Support file-based audit logging
- Add comprehensive tests with full coverage
- Include usage example
- Register via entry point in pyproject.toml"
```

---

## Task Group 5: Final Integration Test

### Step 5.1: Create Full Integration Test

Create `tests/test_plugin_integration.py`:

```python
"""Integration test for the complete plugin system."""

from datetime import date
from unittest.mock import patch

from dqx.api import VerificationSuite, check
from dqx.common import ResultKey
from dqx.orm.repositories import MetricDB
from dqx.provider import DataSource

from tests.test_plugins import MockSuccessPlugin


def test_full_plugin_integration() -> None:
    """Test complete plugin integration from suite execution to plugin processing."""
    # Create checks
    @check(name="Price Check")
    def price_check(mp, ctx):
        ctx.assert_that(mp.average("price"))\\
           .where(name="Average price is positive")\\
           .is_positive()

    @check(name="Count Check")
    def count_check(mp, ctx):
        ctx.assert_that(mp.count("id"))\\
           .where(name="Has records", severity="P0")\\
           .is_gt(0)

    # Create mock plugin
    mock_plugin = MockSuccessPlugin()

    # Mock the plugin manager to use our mock plugin
    with patch("dqx.plugins.PluginManager") as MockPluginManager:
        mock_manager = MockPluginManager.return_value
        mock_manager.get_plugins.return_value = {"test": mock_plugin}
        mock_manager.process_all.side_effect = lambda *args, **kwargs: mock_plugin.process_all(*args, **kwargs)

        # Create suite with plugin config
        db = MetricDB()
        suite = VerificationSuite(
            checks=[price_check, count_check],
            db=db,
            name="Integration Test Suite",
            plugin_config={"test": {"key": "value"}}
        )

        # Create test data
        datasources = {
            "sales": DataSource.from_records([
                {"price": 10.0, "id": 1},
                {"price": 20.0, "id": 2},
            ])
        }

        key = ResultKey(date.today(), {"env": "test"})

        # Run suite
        suite.run(datasources, key)

        # Verify plugin was called
        assert mock_manager.process_all.called

        # Get the call arguments
        call_args = mock_manager.process_all.call_args

        # Verify correct arguments were passed
        assert len(call_args[0]) == 5  # results, symbols, suite_name, context, config
        assert call_args[1]["config"] == {"test": {"key": "value"}}
```

### Step 5.2: Validation

```bash
# Run all tests including integration
uv run pytest tests/test_plugin_integration.py -v

# Run full test suite
uv run pytest tests/ -v

# Final mypy check
uv run mypy src/dqx/plugins.py src/dqx/api.py

# Final ruff check
uv run ruff check src/dqx/plugins.py src/dqx/api.py
```

### Step 5.3: Git Commit

```bash
git add tests/test_plugin_integration.py
git commit -m "test(plugins): add full integration test

- Test complete flow from suite execution to plugin processing
- Verify plugin receives correct data
- Ensure plugin configuration is passed through"
```

---

## Task Group 6: Documentation

### Step 6.1: Update README.md

Replace the detailed plugin section with a brief mention and link:

```markdown
## Plugin System

DQX supports a plugin system for processing validation results. Plugins can send alerts, write to databases, integrate with monitoring systems, and more.

See [Plugin Development Guide](docs/plugins.md) for details on creating your own plugins.
```

### Step 6.2: Create docs/plugins.md

Create a simple, pragmatic plugin development guide:

```markdown
# DQX Plugin Development Guide

## Overview

DQX plugins allow you to process validation results after suite execution. Common use cases include sending alerts, writing to databases, or integrating with monitoring systems.

## Quick Start

### 1. Create Your Plugin

```python
from typing import Any
from dqx.common import AssertionResult, SymbolInfo

class MyPlugin:
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.setting = config.get("setting", "default")

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """Process validation results."""
        failures = [r for r in results if r.status == "FAILURE"]
        if failures:
            print(f"Found {len(failures)} failures in {suite_name}")
```

### 2. Register in pyproject.toml

```toml
[project.entry-points."dqx.plugins"]
my_plugin = "mypackage.plugins:MyPlugin"
```

### 3. Configure in DQX

```python
from dqx.api import VerificationSuite

suite = VerificationSuite(
    checks=checks,
    db=db,
    name="My Suite",
    plugin_config={
        "my_plugin": {
            "setting": "custom_value"
        }
    }
)
```

## Plugin Protocol

Plugins must implement two methods:

- `initialize(config)` - Called with plugin-specific configuration
- `process(results, symbols, suite_name, context)` - Called after suite execution

### Parameters

- `results`: List of `AssertionResult` objects with validation outcomes
- `symbols`: List of `SymbolInfo` objects with computed metric values
- `suite_name`: Name of the verification suite
- `context`: Dictionary with execution context:
  - `datasources`: List of dataset names
  - `key`: ResultKey with date and tags
  - `timestamp`: Unix timestamp of execution

## Example Plugin Implementations

Here's a simple example of an email alerts plugin:

```python
import logging
from typing import Any
from dqx.common import AssertionResult, SymbolInfo

logger = logging.getLogger(__name__)

class EmailAlertsPlugin:
    def initialize(self, config: dict[str, Any]) -> None:
        self.smtp_host = config.get("smtp_host", "localhost")
        self.severity_levels = config.get("severity_levels", ["P0", "P1"])
        logger.info(f"Email alerts configured for severities: {self.severity_levels}")

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        failures = [
            r for r in results
            if r.status == "FAILURE" and r.severity in self.severity_levels
        ]

        if failures:
            # In real implementation, send email via SMTP
            logger.info(f"Would send alert for {len(failures)} failures in {suite_name}")
```

## Built-in Plugins

DQX includes a built-in audit plugin (`dqx_audit`) that logs execution statistics. Enable it with:

```python
plugin_config={
    "dqx_audit": {
        "log_to_file": True,
        "audit_file": "validation_audit.log"
    }
}
```

## Best Practices

1. **Handle errors gracefully** - Don't let plugin failures crash the validation suite
2. **Log appropriately** - Use Python's logging module for debugging
3. **Keep it focused** - Each plugin should do one thing well
4. **Document configuration** - Clearly document required and optional settings
5. **Test thoroughly** - Include unit tests for your plugin logic
```

### Step 6.3: Update pyproject.toml Comments

Move the detailed plugin registration example from README to pyproject.toml:

```toml
# Built-in DQX plugins
[project.entry-points."dqx.plugins"]
dqx_audit = "dqx.plugins.audit:AuditPlugin"

# Example: How external packages register DQX plugins
# Create a plugin class implementing the ResultProcessor protocol,
# then register it in your package's pyproject.toml:
#
# [project.entry-points."dqx.plugins"]
# email_alerts = "mypackage.plugins:EmailAlertsPlugin"
# db_writer = "mypackage.plugins:DatabaseWriterPlugin"
# slack_notifier = "mypackage.plugins:SlackPlugin"
```

### Step 6.4: Validation

```bash
# Verify markdown syntax
cat docs/plugins.md

# Run full test suite
uv run pytest tests/ -v
```

### Step 6.5: Git Commit

```bash
git add README.md docs/plugins.md pyproject.toml
git commit -m "docs(plugins): simplify README and add dedicated plugin guide

- Replace detailed plugin section in README with brief mention
- Create docs/plugins.md with pragmatic development guide
- Move registration examples to pyproject.toml comments
- Keep documentation simple and focused on getting started"
```

---

## Success Criteria

After completing all task groups, you should have:

1. ✅ A working plugin system in `src/dqx/plugins.py`
2. ✅ Integration with VerificationSuite in `src/dqx/api.py`
3. ✅ Built-in audit plugin in `src/dqx/plugins/audit.py`
4. ✅ Comprehensive tests with 100% coverage
5. ✅ Updated documentation with simple plugin guide
6. ✅ Clean git history with atomic commits

The plugin system is now ready with:
- A built-in audit plugin for tracking execution
- Support for external packages to create and register their own result processors
- Full documentation showing how to build plugins
