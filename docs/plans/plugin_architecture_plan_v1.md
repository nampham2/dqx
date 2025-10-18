# Plugin Architecture Implementation Plan

## Overview

This plan implements a plugin system for DQX that allows external integrations to process validation results. The plugin system will enable third-party packages to register processors via Python entry points in `pyproject.toml` and automatically execute them after validation suite runs.

## Background

Currently, DQX executes validation checks and collects results, but has no mechanism for external systems to react to these results. This plugin architecture will allow:

- Sending alerts/emails when assertions fail
- Writing results to custom databases
- Integration with monitoring systems
- Custom reporting and analytics

## Design Decisions

1. **Protocol-based Interface**: Following DQX's existing patterns (SqlDataSource, Analyzer protocols)
2. **Entry Point Discovery**: Using Python's standard `importlib.metadata` for plugin discovery
3. **Single File Implementation**: All plugin code in `src/dqx/plugins.py` for simplicity
4. **Full Type Annotations**: Ensuring mypy compliance with `disallow_untyped_defs = true`

## Architecture

### Plugin Protocol

```python
class ResultProcessor(Protocol):
    """Protocol that all DQX result processor plugins must implement."""

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """Process validation results after suite execution."""
        ...

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        ...
```

### Entry Point Registration (Future Usage)

**Note**: This example shows how external packages will register plugins in the future. No external plugins exist yet.

External plugins will register via `pyproject.toml`:

```toml
# Example from future external package (e.g., bkng-dqx)
[project.optional-dependencies]
bkng = ["bkng-dqx>=1.1"]

[project.entry-points."dqx.plugins"]
bkng_dqx = "external.bkng.dqx:Integration"
```

## Complete Implementation Code

### src/dqx/plugins.py

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

### VerificationSuite Integration

Add the following changes to `src/dqx/api.py`:

```python
# Add import at the top
from dqx.plugins import PluginManager

# In VerificationSuite class, add after __init__:
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

    def _get_plugin_manager(self) -> PluginManager:  # NEW
        """Get or create plugin manager instance (lazy loading)."""
        if self._plugin_manager is None:
            self._plugin_manager = PluginManager()
        return self._plugin_manager

    # In the run() method, add after self.is_evaluated = True:
        # Mark suite as evaluated only after successful completion
        self.is_evaluated = True

        # Execute plugins if any are loaded  # NEW
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

## Complete Test Implementation

### tests/test_plugins.py

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

### Integration Test Example

```python
# In tests/test_api.py or separate integration test file

def test_verification_suite_plugin_integration() -> None:
    """Test that VerificationSuite integrates with plugins correctly."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB

    # Create a test check
    @check(name="test_check")
    def my_check(mp, ctx):
        ctx.assert_that(mp.average("col1"))\\
           .where(name="Test assertion")\\
           .is_positive()

    # Create mock plugin
    mock_plugin = MockSuccessPlugin()

    # Mock the plugin manager
    with patch("dqx.plugins.PluginManager") as MockPluginManager:
        mock_manager = MockPluginManager.return_value
        mock_manager.get_plugins.return_value = {"test": mock_plugin}
        mock_manager.process_all = MagicMock()

        # Create and run suite
        db = MetricDB()
        suite = VerificationSuite(
            checks=[my_check],
            db=db,
            name="Test Suite",
            plugin_config={"test": {"key": "value"}}
        )

        # Run would need mock datasources and key
        # This example shows the integration pattern
```

## Git Commit Strategy

Each task group should have its own commit following conventional commit format:

### Commit 1: Core Plugin Infrastructure
```bash
git add src/dqx/plugins.py
git commit -m "feat(plugins): add ResultProcessor protocol and PluginManager

- Define ResultProcessor protocol for plugin interface
- Implement PluginManager with entry point discovery
- Add error handling and logging for plugin lifecycle
- Include full type annotations for mypy compliance"
```

### Commit 2: VerificationSuite Integration
```bash
git add src/dqx/api.py
git commit -m "feat(api): integrate plugin execution in VerificationSuite

- Add plugin_config parameter to VerificationSuite
- Implement lazy loading of PluginManager
- Execute plugins after suite evaluation
- Pass results, symbols, and context to plugins"
```

### Commit 3: Test Infrastructure
```bash
git add tests/test_plugins.py
git commit -m "test(plugins): add comprehensive plugin system tests

- Test plugin discovery and loading
- Test error handling for invalid/failing plugins
- Test configuration passing and processing
- Add mock plugins for testing
- Ensure 100% code coverage"
```

### Commit 4: Documentation
```bash
git add examples/plugin_*.py README.md
git commit -m "docs(plugins): add plugin examples and documentation

- Add email alerts plugin example
- Add database writer plugin example
- Update README with plugin development guide
- Include configuration examples"
```

### Commit 5: Final Validation
```bash
git add -u
git commit -m "chore(plugins): final validation and cleanup

- Fix any remaining mypy/ruff issues
- Update pyproject.toml with entry point example
- Ensure all tests pass
- Update documentation"
```

## Implementation Tasks

### Task Group 1: Core Plugin Infrastructure

**Task 1.1: Create Plugin Protocol and Manager**
- Create `src/dqx/plugins.py` file
- Define `ResultProcessor` protocol with full type annotations
- Implement `PluginManager` class with:
  - `__init__` method to initialize plugin storage
  - `_load_plugins` method to discover and instantiate plugins
  - `get_plugins` method to return loaded plugins
  - `process_all` method to execute all plugins with error handling
- Add appropriate logging for plugin loading and execution

**Task 1.2: Implement Plugin Discovery**
- Use `importlib.metadata.entry_points(group="dqx.plugins")` for discovery
- Handle missing/invalid plugins gracefully
- Log plugin loading success/failure
- Ensure plugins are instantiated correctly

**Task 1.3: Add Type Stubs and Imports**
- Import necessary types from `dqx.common`
- Add logger configuration
- Ensure all methods have return type annotations
- Add docstrings with parameter descriptions

### Task Group 2: Integration with VerificationSuite

**Task 2.1: Modify VerificationSuite Class**
- Add `_plugin_manager: PluginManager | None = None` attribute
- Create `_get_plugin_manager()` method for lazy initialization
- Import `PluginManager` from `dqx.plugins`
- Maintain backward compatibility

**Task 2.2: Integrate Plugin Execution**
- In `VerificationSuite.run()`, after `self.is_evaluated = True`
- Create execution context with datasources, key, and timestamp
- Call `self.collect_results()` and `self.collect_symbols()`
- Execute plugins via `plugin_manager.process_all()`
- Ensure plugin failures don't break main execution

**Task 2.3: Add Configuration Support**
- Add optional `plugin_config` parameter to `VerificationSuite.__init__`
- Pass configuration to plugins during initialization
- Document configuration format

### Task Group 3: Testing Infrastructure

**Task 3.1: Create Core Plugin Tests**
- Create `tests/test_plugins.py` file
- Test `PluginManager` initialization
- Test plugin discovery with mock entry points
- Test error handling for failed plugins
- Ensure all tests have type annotations

**Task 3.2: Create Integration Tests**
- Test `VerificationSuite` with mock plugins
- Verify plugins receive correct data
- Test plugin execution order
- Test configuration passing

**Task 3.3: Create Example Test Plugin**
- Implement a simple test plugin that logs results
- Use in integration tests
- Demonstrate protocol implementation

### Task Group 4: Documentation and Examples

**Task 4.1: Update API Documentation**
- Document `ResultProcessor` protocol
- Document `PluginManager` class
- Add plugin development guide to README
- Include configuration examples

**Task 4.2: Create Example Plugins**
- Create `examples/plugin_email_alerts.py` showing email alerts
- Create `examples/plugin_custom_db.py` showing database writing
- Include inline documentation
- Show proper type annotations

**Task 4.3: Update pyproject.toml Example**
- Add commented example of plugin registration
- Document entry point format
- Show optional dependency configuration

### Task Group 5: Final Validation

**Task 5.1: Run Type Checking**
- Run `uv run mypy src/dqx/plugins.py`
- Run `uv run mypy tests/test_plugins.py`
- Fix any type annotation issues

**Task 5.2: Run Linting and Tests**
- Run `uv run ruff check --fix`
- Run `uv run pytest tests/test_plugins.py -v`
- Ensure 100% test coverage for new code

**Task 5.3: Run Pre-commit and Full Test Suite**
- Run `bin/run-hooks.sh`
- Run `uv run pytest tests/ -v`
- Commit only when all tests pass

## Technical Considerations

### Error Handling
- Plugin failures must not crash the validation suite
- All plugin exceptions should be caught and logged
- Consider adding a `fail_on_plugin_error` configuration option

### Performance
- Plugins execute synchronously after validation
- Consider future async plugin support if needed
- Plugin execution time should be logged

### Security
- Plugins run with same permissions as DQX
- Document security implications in plugin guide
- Consider plugin sandboxing in future versions

### Backward Compatibility
- Plugin system is entirely optional
- Existing code continues to work unchanged
- No breaking changes to public APIs

## Example Plugin Implementations

### Email Alerts Plugin (examples/plugin_email_alerts.py)

```python
"""Example email alerts plugin for DQX."""

import logging
from typing import Any

from dqx.common import AssertionResult, SymbolInfo

logger = logging.getLogger(__name__)


class EmailAlertsPlugin:
    """Send email alerts for assertion failures."""

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with email configuration."""
        self.smtp_host = config.get("smtp_host", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.from_email = config.get("from_email", "dqx@example.com")
        self.to_emails = config.get("to_emails", [])
        self.severity_levels = config.get("severity_levels", ["P0", "P1"])

        # In real implementation, validate SMTP connection
        logger.info(f"Email alerts configured for severities: {self.severity_levels}")

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """Send email if critical failures found."""
        # Filter for failures matching our severity levels
        failures = [
            r for r in results
            if r.status == "FAILURE" and r.severity in self.severity_levels
        ]

        if not failures:
            logger.info("No critical failures found, no email sent")
            return

        # Build email content
        subject = f"DQX Alert: {len(failures)} failures in {suite_name}"
        body = self._build_email_body(failures, suite_name, context)

        # Send email (mock implementation)
        self._send_email(subject, body)
        logger.info(f"Alert email sent for {len(failures)} failures")

    def _build_email_body(
        self,
        failures: list[AssertionResult],
        suite_name: str,
        context: dict[str, Any]
    ) -> str:
        """Build email body with failure details."""
        lines = [
            f"DQX Validation Suite: {suite_name}",
            f"Date: {context.get('key', {}).get('yyyy_mm_dd', 'Unknown')}",
            f"Datasets: {', '.join(context.get('datasources', []))}",
            "",
            "Failures:",
        ]

        for f in failures:
            lines.extend([
                f"- [{f.severity}] {f.check}/{f.assertion}",
                f"  Expression: {f.expression}",
                f"  Metric: {f.metric}",
                ""
            ])

        return "\n".join(lines)

    def _send_email(self, subject: str, body: str) -> None:
        """Send email via SMTP (mock implementation)."""
        # In real implementation, use smtplib
        logger.info(f"[MOCK] Sending email: {subject}")
        logger.debug(f"[MOCK] Email body:\n{body}")


# Usage in external package pyproject.toml:
# [project.entry-points."dqx.plugins"]
# email_alerts = "your_package.plugins:EmailAlertsPlugin"
```

### Database Writer Plugin (examples/plugin_database_writer.py)

```python
"""Example database writer plugin for DQX."""

import json
import logging
from datetime import datetime
from typing import Any

from dqx.common import AssertionResult, SymbolInfo

logger = logging.getLogger(__name__)


class DatabaseWriterPlugin:
    """Write validation results to external database."""

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize database connection."""
        self.db_url = config.get("database_url", "postgresql://localhost/dqx")
        self.table_prefix = config.get("table_prefix", "dqx_")
        self.batch_size = config.get("batch_size", 1000)

        # In real implementation, create connection pool
        logger.info(f"Database writer configured for: {self.db_url}")

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """Write results to database."""
        timestamp = datetime.fromtimestamp(context.get("timestamp", 0))

        # Write assertion results
        self._write_results(results, suite_name, timestamp)

        # Write symbol values
        self._write_symbols(symbols, suite_name, timestamp)

        logger.info(
            f"Wrote {len(results)} results and {len(symbols)} symbols to database"
        )

    def _write_results(
        self,
        results: list[AssertionResult],
        suite_name: str,
        timestamp: datetime
    ) -> None:
        """Write assertion results to database."""
        # Mock implementation - in reality, use SQLAlchemy or similar
        rows = []
        for r in results:
            rows.append({
                "suite_name": suite_name,
                "check_name": r.check,
                "assertion_name": r.assertion,
                "severity": r.severity,
                "status": r.status,
                "expression": r.expression,
                "metric": r.metric,
                "date": r.yyyy_mm_dd,
                "tags": json.dumps(r.tags),
                "timestamp": timestamp,
            })

        # Batch insert
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            logger.debug(f"[MOCK] Inserting {len(batch)} results")

    def _write_symbols(
        self,
        symbols: list[SymbolInfo],
        suite_name: str,
        timestamp: datetime
    ) -> None:
        """Write symbol values to database."""
        rows = []
        for s in symbols:
            if s.value.is_success():
                rows.append({
                    "suite_name": suite_name,
                    "symbol_name": s.name,
                    "metric": s.metric,
                    "dataset": s.dataset,
                    "value": s.value.unwrap(),
                    "date": s.yyyy_mm_dd,
                    "tags": json.dumps(s.tags),
                    "timestamp": timestamp,
                })

        # Batch insert
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            logger.debug(f"[MOCK] Inserting {len(batch)} symbols")


# Usage in external package pyproject.toml:
# [project.entry-points."dqx.plugins"]
# db_writer = "your_package.plugins:DatabaseWriterPlugin"
```

### Monitoring Integration Plugin (examples/plugin_monitoring.py)

```python
"""Example monitoring system integration plugin for DQX."""

import logging
from typing import Any

from dqx.common import AssertionResult, SymbolInfo

logger = logging.getLogger(__name__)


class MonitoringPlugin:
    """Send metrics to monitoring system (e.g., Prometheus, DataDog)."""

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize monitoring client."""
        self.endpoint = config.get("endpoint", "http://localhost:9091")
        self.namespace = config.get("namespace", "dqx")
        self.labels = config.get("default_labels", {})

        logger.info(f"Monitoring plugin configured for: {self.endpoint}")

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """Send validation metrics to monitoring system."""
        # Calculate metrics
        metrics = self._calculate_metrics(results, suite_name)

        # Send to monitoring system
        self._push_metrics(metrics, context)

        logger.info(f"Pushed {len(metrics)} metrics to monitoring system")

    def _calculate_metrics(
        self,
        results: list[AssertionResult],
        suite_name: str
    ) -> dict[str, float]:
        """Calculate metrics from results."""
        total = len(results)
        failures = sum(1 for r in results if r.status == "FAILURE")

        # Group by severity
        severity_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
        for r in results:
            if r.status == "FAILURE":
                severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1

        return {
            f"{self.namespace}_total_assertions": total,
            f"{self.namespace}_failed_assertions": failures,
            f"{self.namespace}_success_rate": (total - failures) / total if total > 0 else 1.0,
            **{f"{self.namespace}_failures_{sev}": count for sev, count in severity_counts.items()}
        }

    def _push_metrics(
        self,
        metrics: dict[str, float],
        context: dict[str, Any]
    ) -> None:
        """Push metrics to monitoring system."""
        # Mock implementation - in reality, use prometheus_client or similar
        labels = {
            **self.labels,
            "suite": context.get("suite_name", "unknown"),
            "date": str(context.get("key", {}).get("yyyy_mm_dd", "unknown")),
        }

        for metric_name, value in metrics.items():
            logger.debug(f"[MOCK] Push metric: {metric_name}={value} labels={labels}")


# Usage in external package pyproject.toml:
# [project.entry-points."dqx.plugins"]
# monitoring = "your_package.plugins:MonitoringPlugin"
```

### Configuration Examples

```python
# In your verification suite code:
from dqx.api import VerificationSuite
from dqx.orm.repositories import MetricDB

# Configure plugins
plugin_config = {
    "email_alerts": {
        "smtp_host": "smtp.example.com",
        "from_email": "dqx-alerts@example.com",
        "to_emails": ["team@example.com"],
        "severity_levels": ["P0", "P1"],
    },
    "db_writer": {
        "database_url": "postgresql://user:pass@localhost/monitoring",
        "table_prefix": "dqx_prod_",
    },
    "monitoring": {
        "endpoint": "http://prometheus-pushgateway:9091",
        "namespace": "dqx_prod",
        "default_labels": {"env": "production"},
    }
}

# Create suite with plugin configuration
suite = VerificationSuite(
    checks=[check1, check2],
    db=MetricDB(),
    name="Production Data Quality",
    plugin_config=plugin_config
)
```

## Success Criteria

1. Plugins can be discovered and loaded from entry points
2. Plugins receive validation results after suite execution
3. Plugin failures don't break validation flow
4. Full type annotations pass mypy strict mode
5. 100% test coverage for plugin code
6. Clear documentation for plugin developers
