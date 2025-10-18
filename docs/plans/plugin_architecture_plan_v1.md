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

## Task Group 4: Documentation and Examples

### Step 4.1: Create Email Alerts Example

Create `examples/plugin_email_alerts.py`:

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

### Step 4.2: Create Database Writer Example

Create `examples/plugin_database_writer.py`:

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

### Step 4.3: Update README

Add this section to `README.md`:

```markdown
## Plugin System

DQX supports a plugin system that allows external packages to process validation results. Plugins can be used to:

- Send alerts when assertions fail
- Write results to external databases
- Integrate with monitoring systems
- Generate custom reports

### Creating a Plugin

Plugins must implement the `ResultProcessor` protocol:

```python
from typing import Any
from dqx.common import AssertionResult, SymbolInfo

class MyPlugin:
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        """Process validation results."""
        pass
```

### Registering a Plugin

External packages register plugins via `pyproject.toml`:

```toml
[project.entry-points."dqx.plugins"]
my_plugin = "mypackage.plugins:MyPlugin"
```

### Using Plugins

Configure plugins when creating a VerificationSuite:

```python
suite = VerificationSuite(
    checks=checks,
    db=db,
    name="My Suite",
    plugin_config={
        "my_plugin": {
            "setting": "value"
        }
    }
)
```

See `examples/plugin_*.py` for complete examples.
```

### Step 4.4: Add Entry Point Example to pyproject.toml

Add this commented section to `pyproject.toml`:

```toml
# Example: How external packages register DQX plugins
# [project.entry-points."dqx.plugins"]
# email_alerts = "mypackage.plugins:EmailAlertsPlugin"
# db_writer = "mypackage.plugins:DatabaseWriterPlugin"
```

### Step 4.5: Validation

```bash
# Type check examples
uv run mypy examples/plugin_email_alerts.py
uv run mypy examples/plugin_database_writer.py

# Run full test suite to ensure nothing broke
uv run pytest tests/ -v

# Run pre-commit hooks
bin/run-hooks.sh
```

### Step 4.6: Git Commit

```bash
git add examples/plugin_*.py README.md pyproject.toml
git commit -m "docs(plugins): add plugin examples and documentation

- Add email alerts plugin example
- Add database writer plugin example
- Update README with plugin development guide
- Include configuration examples"
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

## Success Criteria

After completing all task groups, you should have:

1. ✅ A working plugin system in `src/dqx/plugins.py`
2. ✅ Integration with VerificationSuite in `src/dqx/api.py`
3. ✅ Comprehensive tests with 100% coverage
4. ✅ Example plugins demonstrating usage
5. ✅ Updated documentation
6. ✅ Clean git history with atomic commits

The plugin system is now ready for external packages to create and register their own result processors!
