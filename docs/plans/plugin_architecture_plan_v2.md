# Plugin Architecture Implementation Plan v2

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
3. **Plugin Metadata**: All plugins must provide metadata through a static method
4. **Execution Context Dataclass**: Rich context object with convenience methods for metrics
5. **No Configuration**: Plugins handle their own configuration (environment variables, config files)
6. **Full Type Annotations**: Ensuring mypy compliance with `disallow_untyped_defs = true`

## Key Data Structures

### PluginMetadata
```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PluginMetadata:
    """Immutable metadata that plugins must provide."""
    name: str
    version: str
    author: str
    description: str
    capabilities: set[str] = field(default_factory=set)
```

### PluginExecutionContext
```python
@dataclass
class PluginExecutionContext:
    """Execution context passed to plugins."""
    suite_name: str  # Suite name is now part of context
    datasources: list[str]
    key: ResultKey
    timestamp: float
    duration_ms: float
    results: list[AssertionResult]
    symbols: list[SymbolInfo]

    def total_assertions(self) -> int:
        """Total number of assertions."""
        return len(self.results)

    def failed_assertions(self) -> int:
        """Number of failed assertions."""
        return sum(1 for r in self.results if r.status == "FAILURE")

    def passed_assertions(self) -> int:
        """Number of passed assertions."""
        return sum(1 for r in self.results if r.status == "OK")

    def assertion_pass_rate(self) -> float:
        """Pass rate as percentage (0-100)."""
        if not self.results:
            return 100.0
        return (self.passed_assertions() / len(self.results)) * 100

    def total_symbols(self) -> int:
        """Total number of symbols."""
        return len(self.symbols)

    def failed_symbols(self) -> int:
        """Number of symbols with failed computations."""
        return sum(1 for s in self.symbols if s.value.is_failure())

    def assertions_by_severity(self) -> dict[str, int]:
        """Count of assertions grouped by severity."""
        from collections import Counter
        return dict(Counter(r.severity for r in self.results))

    def failures_by_severity(self) -> dict[str, int]:
        """Count of failures grouped by severity."""
        from collections import Counter
        failures = [r for r in self.results if r.status == "FAILURE"]
        return dict(Counter(f.severity for f in failures))
```

### ResultProcessor Protocol
```python
class ResultProcessor(Protocol):
    """Protocol for DQX result processor plugins."""

    @staticmethod
    def metadata() -> PluginMetadata:
        """Return plugin metadata."""
        ...

    def process(self, context: PluginExecutionContext) -> None:
        """
        Process validation results after suite execution.

        Args:
            context: Execution context with results, suite name, and convenience methods
        """
        ...
```

## Implementation Guide

Each task group below follows strict TDD methodology. Write tests first, confirm they fail, then implement minimal code to make them pass. Each group ends with full validation and a clean git commit.

---

## Task Group 1: Core Data Structures (TDD)

### Step 1.1: Write Tests for Core Data Structures

Create `tests/test_plugin_dataclasses.py`:

```python
"""Tests for plugin data structures."""

import pytest
from dataclasses import FrozenInstanceError

from dqx.common import (
    AssertionResult,
    PluginExecutionContext,
    PluginMetadata,
    ResultKey,
    ResultValue,
    SymbolInfo,
)


def test_plugin_metadata_frozen() -> None:
    """Test that PluginMetadata is immutable."""
    metadata = PluginMetadata(
        name="test",
        version="1.0.0",
        author="Test Author",
        description="Test plugin",
        capabilities={"test"}
    )

    # Should not be able to modify
    with pytest.raises(FrozenInstanceError):
        metadata.name = "changed"

    with pytest.raises(FrozenInstanceError):
        metadata.capabilities = {"changed"}


def test_plugin_metadata_creation() -> None:
    """Test PluginMetadata creation with all fields."""
    metadata = PluginMetadata(
        name="my_plugin",
        version="2.1.0",
        author="John Doe",
        description="A test plugin for testing",
        capabilities={"email", "slack", "webhook"}
    )

    assert metadata.name == "my_plugin"
    assert metadata.version == "2.1.0"
    assert metadata.author == "John Doe"
    assert metadata.description == "A test plugin for testing"
    assert metadata.capabilities == {"email", "slack", "webhook"}


def test_plugin_metadata_default_capabilities() -> None:
    """Test PluginMetadata with default empty capabilities."""
    metadata = PluginMetadata(
        name="minimal",
        version="1.0.0",
        author="Author",
        description="Minimal plugin"
    )

    assert metadata.capabilities == set()


def test_plugin_execution_context_creation() -> None:
    """Test PluginExecutionContext creation."""
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check1",
            assertion="assert1",
            severity="P1",
            status="OK",
            metric="metric1",
            expression="x > 0",
            tags={}
        )
    ]

    symbols = [
        SymbolInfo(
            name="x",
            metric="count",
            dataset="ds1",
            value=ResultValue.success(10),
            yyyy_mm_dd="2024-01-01",
            suite="test",
            tags={}
        )
    ]

    context = PluginExecutionContext(
        suite_name="test",
        datasources=["ds1", "ds2"],
        key=ResultKey("2024-01-01", {"env": "prod"}),
        timestamp=1704067200.0,
        duration_ms=2500.0,
        results=results,
        symbols=symbols
    )

    assert context.suite_name == "test"
    assert context.datasources == ["ds1", "ds2"]
    assert context.key.yyyy_mm_dd == "2024-01-01"
    assert context.key.tags == {"env": "prod"}
    assert context.timestamp == 1704067200.0
    assert context.duration_ms == 2500.0
    assert len(context.results) == 1
    assert len(context.symbols) == 1


def test_context_total_assertions() -> None:
    """Test total_assertions method."""
    context = PluginExecutionContext(
        suite_name="test",
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=[
            AssertionResult(
                yyyy_mm_dd="2024-01-01",
                suite="test",
                check="check",
                assertion="assert",
                severity="P1",
                status="OK",
                metric="metric",
                expression="x > 0",
                tags={}
            ) for _ in range(5)
        ],
        symbols=[]
    )

    assert context.total_assertions() == 5


def test_context_failed_assertions() -> None:
    """Test failed_assertions method."""
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check",
            assertion=f"assert{i}",
            severity="P1",
            status="FAILURE" if i < 3 else "OK",
            metric="metric",
            expression="x > 0",
            tags={}
        ) for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=results,
        symbols=[]
    )

    assert context.failed_assertions() == 3
    assert context.passed_assertions() == 2


def test_context_assertion_pass_rate() -> None:
    """Test assertion_pass_rate calculation."""
    # 3 passed, 2 failed = 60% pass rate
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check",
            assertion=f"assert{i}",
            severity="P1",
            status="OK" if i < 3 else "FAILURE",
            metric="metric",
            expression="x > 0",
            tags={}
        ) for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=results,
        symbols=[]
    )

    assert context.assertion_pass_rate() == 60.0

    # Empty results = 100% pass rate
    empty_context = PluginExecutionContext(
        suite_name="test",
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=[],
        symbols=[]
    )

    assert empty_context.assertion_pass_rate() == 100.0


def test_context_symbol_methods() -> None:
    """Test symbol-related methods."""
    symbols = [
        SymbolInfo(
            name=f"sym{i}",
            metric="count",
            dataset="ds1",
            value=ResultValue.success(i) if i < 3 else ResultValue.failure(Exception("Error")),
            yyyy_mm_dd="2024-01-01",
            suite="test",
            tags={}
        ) for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        datasources=["ds1"],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=[],
        symbols=symbols
    )

    assert context.total_symbols() == 5
    assert context.failed_symbols() == 2


def test_context_assertions_by_severity() -> None:
    """Test assertions_by_severity grouping."""
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check",
            assertion=f"assert{i}",
            severity=f"P{i % 3}",  # P0, P1, P2, P0, P1
            status="OK",
            metric="metric",
            expression="x > 0",
            tags={}
        ) for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=results,
        symbols=[]
    )

    by_severity = context.assertions_by_severity()
    assert by_severity == {"P0": 2, "P1": 2, "P2": 1}


def test_context_failures_by_severity() -> None:
    """Test failures_by_severity grouping."""
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check",
            assertion=f"assert{i}",
            severity=f"P{i}",
            status="FAILURE" if i < 3 else "OK",  # P0, P1, P2 fail
            metric="metric",
            expression="x > 0",
            tags={}
        ) for i in range(5)
    ]

    context = PluginExecutionContext(
        suite_name="test",
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_ms=0.0,
        results=results,
        symbols=[]
    )

    failures = context.failures_by_severity()
    assert failures == {"P0": 1, "P1": 1, "P2": 1}
    # P3 and P4 passed, so not in failures
```

### Step 1.2: Run Tests to Confirm They Fail

```bash
# This should fail since we haven't implemented the dataclasses yet
uv run pytest tests/test_plugin_dataclasses.py -v

# Expected: ImportError or AttributeError for missing classes
```

### Step 1.3: Implement Core Data Structures

Now implement the minimal code to make tests pass. Add to `src/dqx/common.py`:

```python
# Add these imports at the top if not present
from collections import Counter
from dataclasses import dataclass, field

# Add after existing dataclasses
@dataclass(frozen=True)
class PluginMetadata:
    """Immutable metadata that plugins must provide."""
    name: str
    version: str
    author: str
    description: str
    capabilities: set[str] = field(default_factory=set)


@dataclass
class PluginExecutionContext:
    """Execution context passed to plugins."""
    datasources: list[str]
    key: ResultKey
    timestamp: float
    duration_seconds: float
    results: list[AssertionResult]
    symbols: list[SymbolInfo]

    def total_assertions(self) -> int:
        """Total number of assertions."""
        return len(self.results)

    def failed_assertions(self) -> int:
        """Number of failed assertions."""
        return sum(1 for r in self.results if r.status == "FAILURE")

    def passed_assertions(self) -> int:
        """Number of passed assertions."""
        return sum(1 for r in self.results if r.status == "OK")

    def assertion_pass_rate(self) -> float:
        """Pass rate as percentage (0-100)."""
        if not self.results:
            return 100.0
        return (self.passed_assertions() / len(self.results)) * 100

    def total_symbols(self) -> int:
        """Total number of symbols."""
        return len(self.symbols)

    def failed_symbols(self) -> int:
        """Number of symbols with failed computations."""
        return sum(1 for s in self.symbols if s.value.is_failure())

    def assertions_by_severity(self) -> dict[str, int]:
        """Count of assertions grouped by severity."""
        return dict(Counter(r.severity for r in self.results))

    def failures_by_severity(self) -> dict[str, int]:
        """Count of failures grouped by severity."""
        failures = [r for r in self.results if r.status == "FAILURE"]
        return dict(Counter(f.severity for f in failures))
```

### Step 1.2: Create the Plugin Module

Create `src/dqx/plugins.py` with the complete implementation:

```python
"""Plugin system for DQX result processing."""

import importlib.metadata
import logging
import time
from typing import Protocol

from dqx.common import (
    PluginExecutionContext,
    PluginMetadata,
)
from dqx.timer import TimeLimiting, TimeLimitExceededError
from rich.console import Console
from rich.table import Table
from rich import box

logger = logging.getLogger(__name__)

# Hard time limit for plugin execution
PLUGIN_TIMEOUT_SECONDS = 60


class ResultProcessor(Protocol):
    """Protocol for DQX result processor plugins."""

    @staticmethod
    def metadata() -> PluginMetadata:
        """Return plugin metadata."""
        ...

    def process(self, context: PluginExecutionContext) -> None:
        """
        Process validation results after suite execution.

        Args:
            context: Execution context with results, suite name, and convenience methods
        """
        ...


class PluginManager:
    """Manages DQX result processor plugins."""

    def __init__(self, *, _timeout_seconds: int | None = None) -> None:
        """
        Initialize the plugin manager.

        Args:
            _timeout_seconds: Internal parameter for testing. Users should not set this.
                            Defaults to PLUGIN_TIMEOUT_SECONDS.
        """
        self._plugins: dict[str, ResultProcessor] = {}
        self._timeout = _timeout_seconds or PLUGIN_TIMEOUT_SECONDS
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

                    # Get metadata
                    metadata = plugin_class.metadata()

                    # Verify it implements the protocol
                    if not isinstance(plugin, ResultProcessor):
                        logger.error(
                            f"Plugin {ep.name} does not implement ResultProcessor protocol"
                        )
                        continue

                    # Store the plugin
                    self._plugins[ep.name] = plugin

                    logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")

                except Exception as e:
                    logger.warning(f"Plugin not available: {ep.name}")

        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")

    def get_plugins(self) -> dict[str, ResultProcessor]:
        """
        Get all loaded plugins.

        Returns:
            Dictionary mapping plugin names to plugin instances
        """
        return self._plugins

    def get_metadata(self) -> dict[str, PluginMetadata]:
        """
        Get all plugin metadata by calling their static methods.

        Returns:
            Dictionary mapping plugin names to metadata
        """
        return {
            name: plugin.__class__.metadata()
            for name, plugin in self._plugins.items()
        }

    def process_all(self, context: PluginExecutionContext) -> None:
        """
        Process results through all loaded plugins with time limits.

        Args:
            context: Execution context with results, suite name, and convenience methods
        """
        if not self._plugins:
            logger.debug("No plugins loaded, skipping plugin processing")
            return

        logger.info(f"Processing results through {len(self._plugins)} plugin(s)")

        for name, plugin in self._plugins.items():
            try:
                # Hard time limit for plugin execution
                with TimeLimiting(self._timeout) as timer:
                    plugin.process(context)

                logger.info(f"Plugin {name} processed results in {timer.elapsed_ms():.2f}ms")

            except TimeLimitExceededError:
                logger.error(f"Plugin {name} exceeded {self._timeout}s time limit")
            except Exception as e:
                # Log error but don't fail the entire suite
                logger.error(f"Plugin {name} failed during processing: {e}")


class AuditPlugin:
    """
    DQX built-in audit plugin for tracking suite execution.

    This plugin provides basic auditing functionality including:
    - Execution timing
    - Result statistics with Rich table display
    - Performance metrics with colors
    """

    @staticmethod
    def metadata() -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="audit",
            version="1.0.0",
            author="DQX Team",
            description="Display execution audit report with Rich tables",
            capabilities={"console_output", "statistics"}
        )

    def __init__(self) -> None:
        """Initialize the audit plugin."""
        self.console = Console()

    def process(
        self,
        context: PluginExecutionContext,
        suite_name: str
    ) -> None:
        """
        Process and display validation results.

        Args:
            context: Execution context with results and convenience methods
            suite_name: Name of the verification suite
        """
        # Use context methods for statistics
        total = context.total_assertions()
        passed = context.passed_assertions()
        failed = context.failed_assertions()
        pass_rate = context.assertion_pass_rate()

        # Display header
        self.console.print()
        self.console.print("[bold blue]═══ DQX Audit Report ═══[/bold blue]")
        self.console.print(f"[cyan]Suite:[/cyan] {suite_name}")
        self.console.print(f"[cyan]Date:[/cyan] {context.key.yyyy_mm_dd}")
        self.console.print(f"[cyan]Duration:[/cyan] {context.duration_ms:.2f}ms")
        if context.datasources:
            self.console.print(f"[cyan]Datasets:[/cyan] {', '.join(context.datasources)}")
        self.console.print()

        # Create summary table using context methods
        summary_table = Table(title="Execution Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", justify="right")
        summary_table.add_column("Rate", justify="right")

        summary_table.add_row("Total Assertions", str(total), "")

        if total > 0:
            summary_table.add_row(
                "[green]Passed ✓[/green]",
                f"[green]{passed}[/green]",
                f"[green]{pass_rate:.1f}%[/green]"
            )
            summary_table.add_row(
                "[red]Failed ✗[/red]",
                f"[red]{failed}[/red]",
                f"[red]{100 - pass_rate:.1f}%[/red]"
            )
        else:
            summary_table.add_row("Passed ✓", "0", "0.0%")
            summary_table.add_row("Failed ✗", "0", "0.0%")

        self.console.print(summary_table)

        # Show failures by severity if any
        failures_by_sev = context.failures_by_severity()
        if failures_by_sev:
            self.console.print()
            severity_table = Table(title="Failures by Severity", box=box.ROUNDED)
            severity_table.add_column("Severity", style="bold")
            severity_table.add_column("Count", justify="right")

            # Color code by severity
            severity_colors = {
                "P0": "red bold",
                "P1": "orange1",
                "P2": "yellow",
                "P3": "blue",
                "P4": "grey50"
            }

            for severity, count in sorted(failures_by_sev.items()):
                color = severity_colors.get(severity, "white")
                severity_table.add_row(
                    f"[{color}]{severity}[/{color}]",
                    f"[{color}]{count}[/{color}]"
                )

            self.console.print(severity_table)

        # Show symbol statistics if any
        if context.symbols:
            successful_symbols = context.total_symbols() - context.failed_symbols()
            failed_symbols = context.failed_symbols()

            self.console.print()
            symbol_table = Table(title="Symbol Statistics", box=box.ROUNDED)
            symbol_table.add_column("Metric", style="cyan")
            symbol_table.add_column("Count", justify="right")

            symbol_table.add_row("Total Symbols", str(context.total_symbols()))
            symbol_table.add_row(
                "[green]Successful[/green]",
                f"[green]{successful_symbols}[/green]"
            )
            symbol_table.add_row(
                "[red]Failed[/red]",
                f"[red]{failed_symbols}[/red]"
            )

            self.console.print(symbol_table)

        self.console.print()
        self.console.print("[bold blue]══════════════════════[/bold blue]")
        self.console.print()
```

### Step 1.3: Initial Validation

```bash
# Type check both files
uv run mypy src/dqx/common.py src/dqx/plugins.py

# Lint check
uv run ruff check src/dqx/common.py src/dqx/plugins.py

# Fix any linting issues
uv run ruff check --fix src/dqx/common.py src/dqx/plugins.py
```

### Step 1.4: Final Validation and Commit

```bash
# Run all tests to ensure nothing is broken
uv run pytest tests/ -v

# Run full pre-commit checks
bin/run-hooks.sh

# If any issues are found:
# - For ruff: uv run ruff check --fix src/dqx/common.py src/dqx/plugins.py
# - For mypy: manually fix type annotations
# - For test failures: investigate and fix

# Re-run tests after any fixes
uv run pytest tests/ -v

# Once everything passes, commit:
git add src/dqx/common.py src/dqx/plugins.py
git commit -m "feat(plugins): add plugin system with metadata and context dataclass

- Add PluginMetadata and PluginExecutionContext to common.py
- Implement ResultProcessor protocol with metadata requirement
- Create PluginManager with entry point discovery and metadata collection
- Add AuditPlugin with Rich table display using context methods
- Include full type annotations for mypy compliance"
```

---

## Task Group 2: API Integration

### Step 2.1: Update Imports in api.py

Add these imports at the top of `src/dqx/api.py`:

```python
from dqx.common import PluginExecutionContext
from dqx.plugins import PluginManager
```

Also add `time` import if not already present:

```python
import time
```

### Step 2.2: Modify VerificationSuite Constructor

Update the `__init__` method to add plugin support attributes:

```python
def __init__(
    self,
    checks: Sequence[CheckProducer | DecoratedCheck],
    db: MetricDB,
    name: str,
) -> None:
    """
    Initialize the verification suite.

    Args:
        checks: Sequence of check functions to execute
        db: Database for storing and retrieving metrics
        name: Human-readable name for the suite

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
```

### Step 2.3: Add Plugin Manager Property

Add this property after the constructor:

```python
@property
def plugin_manager(self) -> PluginManager:
    """Get or create plugin manager instance (lazy loading)."""
    if self._plugin_manager is None:
        self._plugin_manager = PluginManager()
    return self._plugin_manager
```

### Step 2.4: Update run() Method to Use TimeLimiting

Update the `run()` method to add the `enable_plugins` parameter and use TimeLimiting for timing:

```python
def run(
    self,
    datasources: dict[str, SqlDataSource],
    key: ResultKey,
    *,
    enable_plugins: bool = True
) -> None:
    """
    Run the verification suite against provided data sources.

    Args:
        datasources: Dictionary mapping dataset names to SQL data sources
        key: Key for storing/retrieving results
        enable_plugins: Whether to execute plugins after evaluation (default: True)

    Raises:
        DQXError: If suite already evaluated, graph building fails, or evaluation fails
    """
    # Setup/validation (not timed)
    if self.is_evaluated:
        raise DQXError(f"Suite '{self._name}' has already been evaluated")

    # Store the key for result collection
    self._key = key

    # Build graph if needed (not timed)
    if not self._graph_built:
        self._build_graph()

    # Time only the actual evaluation
    with TimeLimiting(None) as validation_timer:
        # Run the actual evaluation
        evaluator = Evaluator(self._context.graph)
        evaluator.run(datasources)

    # Mark as evaluated (outside timer)
    self.is_evaluated = True

    # Execute plugins with timing info
    if enable_plugins:
        self._execute_plugins(datasources, key, validation_timer.tick, validation_timer.elapsed_ms())
```

### Step 2.5: Add Plugin Execution Method

Add a new private method to handle plugin execution:

```python
def _execute_plugins(self, datasources: dict[str, SqlDataSource], key: ResultKey, timestamp: float, validation_duration_ms: float) -> None:
    """Execute plugins with validation timing."""
    if not self.plugin_manager.get_plugins():
        return

    logger.info("Executing result processor plugins...")

    # Create execution context
    context = PluginExecutionContext(
        suite_name=self._name,
        datasources=list(datasources.keys()),
        key=key,
        timestamp=timestamp,  # Passed directly from timer.tick
        duration_ms=validation_duration_ms,  # Pass ms directly, no conversion
        results=self.collect_results(),
        symbols=self.collect_symbols()
    )

    # Process through plugins
    self.plugin_manager.process_all(context)
```

### Step 2.6: Call Plugin Execution in run()

In the `run()` method, after the line `self.is_evaluated = True`, add:

```python
# Execute plugins (including audit if enabled)
self._execute_plugins(datasources, key)
```

### Step 2.7: Initial Validation

```bash
# Type check
uv run mypy src/dqx/api.py

# Lint check
uv run ruff check src/dqx/api.py

# Run existing API tests to ensure backward compatibility
uv run pytest tests/test_api.py -v
```

### Step 2.8: Final Validation and Commit

```bash
# Run all tests to ensure integration works
uv run pytest tests/ -v

# Run full pre-commit checks
bin/run-hooks.sh

# Fix any issues found:
# - For ruff: uv run ruff check --fix src/dqx/api.py
# - For mypy: manually fix type annotations
# - For test failures: debug and fix

# Re-run tests after fixes
uv run pytest tests/ -v

# Once all checks pass, commit:
git add src/dqx/api.py
git commit -m "feat(api): integrate plugin execution with PluginExecutionContext

- Add plugin_config parameter to VerificationSuite
- Implement lazy loading of PluginManager
- Create PluginExecutionContext with duration tracking
- Execute plugins after suite evaluation
- Pass context object with convenience methods to plugins"
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

from dqx.common import (
    AssertionResult,
    PluginExecutionContext,
    PluginMetadata,
    ResultKey,
    ResultValue,
    SymbolInfo,
)
from dqx.plugins import PluginManager, ResultProcessor


class MockSuccessPlugin:
    """Mock plugin that tracks all calls."""

    @staticmethod
    def metadata() -> PluginMetadata:
        """Return mock metadata."""
        return PluginMetadata(
            name="mock_success",
            version="1.0.0",
            author="Test Author",
            description="Test plugin",
            capabilities={"test", "mock"}
        )

    def __init__(self) -> None:
        self.processed = False
        self.received_context: PluginExecutionContext | None = None
        self.received_suite_name = ""

    def process(self, context: PluginExecutionContext) -> None:
        """Track processing."""
        self.processed = True
        self.received_context = context
        self.received_suite_name = context.suite_name


class MockFailurePlugin:
    """Mock plugin that raises exceptions."""

    @staticmethod
    def metadata() -> PluginMetadata:
        """Return mock metadata."""
        return PluginMetadata(
            name="mock_failure",
            version="1.0.0",
            author="Test Author",
            description="Failing test plugin"
        )

    def process(
        self,
        context: PluginExecutionContext,
        suite_name: str
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
        assert isinstance(manager._metadata, dict)
        assert len(manager._plugins) == 0
        assert len(manager._metadata) == 0


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

        # Check metadata was collected
        assert len(manager._metadata) == 1
        assert "test_plugin" in manager._metadata
        metadata = manager._metadata["test_plugin"]
        assert metadata.name == "mock_success"
        assert metadata.version == "1.0.0"


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
        assert len(manager._metadata) == 0


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


def test_get_plugins_and_metadata() -> None:
    """Test getting loaded plugins and metadata."""
    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        plugins = manager.get_plugins()
        metadata = manager.get_metadata()

        assert isinstance(plugins, dict)
        assert isinstance(metadata, dict)
        assert len(plugins) == 0
        assert len(metadata) == 0


def test_process_all_no_plugins() -> None:
    """Test process_all when no plugins are loaded."""
    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()

        # Create test context
        context = PluginExecutionContext(
            datasources=["test_ds"],
            key=ResultKey("2024-01-01", {}),
            timestamp=0.0,
            duration_ms=1000.0,
            results=[],
            symbols=[]
        )

        # Should not raise any errors
        manager.process_all(context, "test_suite")


def test_process_all_with_plugin() -> None:
    """Test processing results through a plugin with context."""
    # Create mock plugin
    mock_plugin = MockSuccessPlugin()

    # Create test data
    result = AssertionResult(
        yyyy_mm_dd="2024-01-01",
        suite="test_suite",
        check="test_check",
        assertion="test_assertion",
        severity="P1",
        status="FAILURE",
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

    context = PluginExecutionContext(
        datasources=["test_dataset"],
        key=ResultKey("2024-01-01", {"env": "test"}),
            timestamp=1704067200.0,
            duration_ms=2500.0,
            results=[result],
            symbols=[symbol]
    )

    config = {"test_plugin": {"setting": "value"}}

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["test_plugin"] = mock_plugin
        manager._metadata["test_plugin"] = mock_plugin.metadata()

        manager.process_all(
            context=context,
            suite_name="test_suite"
        )

        # Verify plugin was processed
        assert mock_plugin.processed
        assert mock_plugin.received_context == context
        assert mock_plugin.received_suite_name == "test_suite"


def test_context_methods() -> None:
    """Test PluginExecutionContext convenience methods."""
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check1",
            assertion="assert1",
            severity="P0",
            status="FAILURE",
            metric="metric1",
            expression="x > 0",
            tags={}
        ),
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check2",
            assertion="assert2",
            severity="P1",
            status="FAILURE",
            metric="metric2",
            expression="y < 10",
            tags={}
        ),
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check3",
            assertion="assert3",
            severity="P2",
            status="OK",
            metric="metric3",
            expression="z == 0",
            tags={}
        ),
    ]

    symbols = [
        SymbolInfo(
            name="x",
            metric="count",
            dataset="ds1",
            value=ResultValue.success(10),
            yyyy_mm_dd="2024-01-01",
            suite="test",
            tags={}
        ),
        SymbolInfo(
            name="y",
            metric="average",
            dataset="ds2",
            value=ResultValue.failure(Exception("Error")),
            yyyy_mm_dd="2024-01-01",
            suite="test",
            tags={}
        ),
    ]

    context = PluginExecutionContext(
        datasources=["ds1", "ds2"],
        key=ResultKey("2024-01-01", {}),
            timestamp=0.0,
            duration_ms=1000.0,
            results=results,
            symbols=symbols
    )

    # Test assertion methods
    assert context.total_assertions() == 3
    assert context.failed_assertions() == 2
    assert context.passed_assertions() == 1
    assert context.assertion_pass_rate() == pytest.approx(33.33, rel=0.01)

    # Test symbol methods
    assert context.total_symbols() == 2
    assert context.failed_symbols() == 1

    # Test grouping methods
    assert context.assertions_by_severity() == {"P0": 1, "P1": 1, "P2": 1}
    assert context.failures_by_severity() == {"P0": 1, "P1": 1}


def test_context_empty_results() -> None:
    """Test context methods with empty results."""
    context = PluginExecutionContext(
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_seconds=0.0,
        results=[],
        symbols=[]
    )

    assert context.total_assertions() == 0
    assert context.failed_assertions() == 0
    assert context.passed_assertions() == 0
    assert context.assertion_pass_rate() == 100.0  # No assertions = 100% pass
    assert context.total_symbols() == 0
    assert context.failed_symbols() == 0
    assert context.assertions_by_severity() == {}
    assert context.failures_by_severity() == {}


def test_process_all_plugin_failure() -> None:
    """Test that plugin failures don't crash the suite."""
    # Create failing plugin
    mock_plugin = MockFailurePlugin()

    context = PluginExecutionContext(
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_seconds=0.0,
        results=[],
        symbols=[]
    )

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["failing_plugin"] = mock_plugin
        manager._metadata["failing_plugin"] = mock_plugin.metadata()

        # Should not raise - failures are logged
        manager.process_all(context, "test_suite")


def test_process_all_without_config() -> None:
    """Test processing without plugin configuration."""
    mock_plugin = MockSuccessPlugin()

    context = PluginExecutionContext(
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_seconds=0.0,
        results=[],
        symbols=[]
    )

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["test_plugin"] = mock_plugin
        manager._metadata["test_plugin"] = mock_plugin.metadata()

        manager.process_all(
            context=context,
            suite_name="test_suite"
        )

        assert mock_plugin.processed


def test_multiple_plugins() -> None:
    """Test processing through multiple plugins."""
    plugin1 = MockSuccessPlugin()
    plugin2 = MockSuccessPlugin()

    context = PluginExecutionContext(
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_seconds=0.0,
        results=[],
        symbols=[]
    )

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["plugin1"] = plugin1
        manager._plugins["plugin2"] = plugin2
        manager._metadata["plugin1"] = plugin1.metadata()
        manager._metadata["plugin2"] = plugin2.metadata()

        manager.process_all(context, "test_suite")

        assert plugin1.processed
        assert plugin2.processed


def test_plugin_execution_time_logging() -> None:
    """Test that plugin execution time is logged."""
    mock_plugin = MockSuccessPlugin()

    context = PluginExecutionContext(
        datasources=[],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_seconds=0.0,
        results=[],
        symbols=[]
    )

    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager()
        manager._plugins["test_plugin"] = mock_plugin

        with patch("dqx.plugins.logger") as mock_logger:
            manager.process_all(context, "test_suite")

            # Check that execution time was logged
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("processed results in" in call for call in info_calls)


def test_plugin_timeout() -> None:
    """Test that plugins are terminated after timeout."""
    import time

    class SlowPlugin:
        @staticmethod
        def metadata() -> PluginMetadata:
            return PluginMetadata(
                name="slow",
                version="1.0.0",
                author="Test",
                description="Slow plugin"
            )

        def process(self, context: PluginExecutionContext, suite_name: str) -> None:
            # Sleep for 2 seconds - will exceed 1 second timeout
            time.sleep(2)

    # Create manager with 1 second timeout for testing
    with patch("importlib.metadata.entry_points", return_value=[]):
        manager = PluginManager(_timeout_seconds=1)  # Internal parameter
        manager._plugins["slow"] = SlowPlugin()

        context = PluginExecutionContext(
            datasources=[],
            key=ResultKey("2024-01-01", {}),
            timestamp=0.0,
            duration_ms=0.0,
            results=[],
            symbols=[]
        )

        # Process should complete without raising (timeout is handled)
        with patch("dqx.plugins.logger") as mock_logger:
            manager.process_all(context, "test")

            # Check that timeout error was logged
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any("exceeded 1s time limit" in call for call in error_calls)
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

### Step 3.3: Initial Validation

```bash
# Type check
uv run mypy tests/test_plugins.py

# Run the new tests
uv run pytest tests/test_plugins.py -v

# Check coverage
uv run pytest tests/test_plugins.py -v --cov=dqx.plugins --cov-report=term-missing
```

### Step 3.4: Final Validation and Commit

```bash
# Run all tests to ensure nothing is broken
uv run pytest tests/ -v

# Run full pre-commit checks
bin/run-hooks.sh

# Fix any issues:
# - For test failures: debug and fix
# - For linting: uv run ruff check --fix tests/test_plugins.py
# - For mypy: fix type annotations

# Verify coverage is still 100%
uv run pytest tests/test_plugins.py -v --cov=dqx.plugins --cov-report=term-missing

# Once all checks pass, commit:
git add tests/test_plugins.py tests/test_api.py
git commit -m "test(plugins): add comprehensive plugin system tests

- Test plugin discovery with metadata collection
- Test PluginExecutionContext convenience methods
- Test error handling for invalid/failing plugins
- Test configuration passing and processing
- Add mock plugins with metadata support
- Ensure 100% code coverage"
```

---

## Task Group 4: Built-in Audit Plugin

### Step 4.1: Add to pyproject.toml Entry Points

Update `pyproject.toml` to register the built-in audit plugin:

```toml
[project.entry-points."dqx.plugins"]
audit = "dqx.plugins:AuditPlugin"
```

### Step 4.2: Create Tests for Audit Plugin

Create `tests/test_audit_plugin.py`:

```python
"""Tests for the built-in audit plugin."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dqx.common import (
    AssertionResult,
    PluginExecutionContext,
    ResultKey,
    ResultValue,
    SymbolInfo,
)
from dqx.plugins import AuditPlugin


def create_test_context() -> PluginExecutionContext:
    """Create test context for audit plugin testing."""
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test_suite",
            check="check1",
            assertion="assertion1",
            severity="P1",
            status="OK",
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

    return PluginExecutionContext(
        datasources=["dataset1", "dataset2"],
        key=ResultKey("2024-01-01", {"env": "test"}),
        timestamp=1704067200.0,  # 2024-01-01 00:00:00 UTC
        duration_ms=2500.0,
        results=results,
        symbols=symbols
    )


def test_audit_plugin_metadata() -> None:
    """Test audit plugin provides correct metadata."""
    metadata = AuditPlugin.metadata()
    assert metadata.name == "audit"
    assert metadata.version == "1.0.0"
    assert metadata.author == "DQX Team"
    assert "statistics" in metadata.capabilities
    assert "console_output" in metadata.capabilities


def test_audit_plugin_initialization() -> None:
    """Test audit plugin initializes correctly."""
    plugin = AuditPlugin()
    assert hasattr(plugin, "console")


def test_audit_plugin_displays_rich_tables() -> None:
    """Test audit plugin displays results using Rich tables."""
    plugin = AuditPlugin()

    context = create_test_context()

    # Mock the console to capture output
    with patch.object(plugin, "console") as mock_console:
        plugin.process(context, "Test Suite")

        # Verify console.print was called multiple times
        assert mock_console.print.called
        calls = mock_console.print.call_args_list

        # Verify header was printed
        header_found = any(
            "DQX Audit Report" in str(call) for call in calls
        )
        assert header_found

        # Verify tables were created
        table_calls = [
            call for call in calls
            if len(call[0]) > 0 and hasattr(call[0][0], "title")
        ]
        assert len(table_calls) >= 2  # Summary + severity tables


def test_audit_plugin_handles_empty_results() -> None:
    """Test audit plugin handles empty results gracefully."""
    plugin = AuditPlugin()

    context = PluginExecutionContext(
        datasources=[],
        key=ResultKey("2024-01-01", {}),
            timestamp=0.0,
            duration_ms=0.0,
            results=[],
            symbols=[]
    )

    # Mock console to verify it still prints
    with patch.object(plugin, "console") as mock_console:
        plugin.process(context, "Empty Suite")

        # Should still print header and summary
        assert mock_console.print.called


def test_audit_plugin_uses_context_methods() -> None:
    """Test audit plugin uses context convenience methods."""
    plugin = AuditPlugin()

    context = create_test_context()

    # Spy on context methods
    with patch.object(context, "total_assertions", wraps=context.total_assertions) as mock_total:
        with patch.object(context, "failed_assertions", wraps=context.failed_assertions) as mock_failed:
            with patch.object(context, "assertion_pass_rate", wraps=context.assertion_pass_rate) as mock_rate:
                with patch.object(context, "failures_by_severity", wraps=context.failures_by_severity) as mock_severity:
                    with patch.object(plugin, "console"):
                        plugin.process(context, "Test Suite")

                        # Verify context methods were called
                        mock_total.assert_called()
                        mock_failed.assert_called()
                        mock_rate.assert_called()
                        mock_severity.assert_called()


def test_audit_plugin_color_coding() -> None:
    """Test audit plugin uses appropriate colors."""
    plugin = AuditPlugin()

    context = create_test_context()

    with patch.object(plugin, "console") as mock_console:
        plugin.process(context, "Color Test")

        # Check that color tags are used
        print_calls = str(mock_console.print.call_args_list)

        # Verify color codes
        assert "[green]" in print_calls  # For passed
        assert "[red]" in print_calls    # For failed
        assert "[cyan]" in print_calls   # For headers


def test_audit_plugin_severity_colors() -> None:
    """Test audit plugin colors severities appropriately."""
    # Create results with different severities
    results = []
    for severity in ["P0", "P1", "P2", "P3", "P4"]:
        results.append(
            AssertionResult(
                yyyy_mm_dd="2024-01-01",
                suite="test",
                check="check",
                assertion=f"assertion_{severity}",
                severity=severity,
                status="FAILURE",
                metric="metric",
                expression="x > 0",
                tags={}
            )
        )

    context = PluginExecutionContext(
        datasources=[],
        key=ResultKey("2024-01-01", {}),
            timestamp=0.0,
            duration_ms=0.0,
            results=[],
            symbols=[]
    )

    plugin = AuditPlugin()

    with patch.object(plugin, "console") as mock_console:
        plugin.process(context, "Severity Test")

        # Verify severity table was created with colors
        table_calls = [
            call for call in mock_console.print.call_args_list
            if len(call[0]) > 0 and hasattr(call[0][0], "title")
        ]

        # Find the severity table
        severity_table = None
        for call in table_calls:
            if hasattr(call[0][0], "title") and "Severity" in str(call[0][0].title):
                severity_table = call[0][0]
                break

        assert severity_table is not None


def test_audit_plugin_no_failures() -> None:
    """Test audit plugin when all checks pass."""
    results = [
        AssertionResult(
            yyyy_mm_dd="2024-01-01",
            suite="test",
            check="check",
            assertion="assertion",
            severity="P1",
            status="OK",
            metric="metric",
            expression="x > 0",
            tags={}
        )
    ]

    context = PluginExecutionContext(
        datasources=["test_ds"],
        key=ResultKey("2024-01-01", {}),
        timestamp=0.0,
        duration_ms=1000.0,
        results=results,
        symbols=[]
    )

    plugin = AuditPlugin()

    with patch.object(plugin, "console") as mock_console:
        plugin.process(context, "Success Suite")

        # Should not show severity table
        table_calls = [
            call for call in mock_console.print.call_args_list
            if len(call[0]) > 0 and hasattr(call[0][0], "title")
        ]

        severity_tables = [
            call for call in table_calls
            if "Severity" in str(call[0][0].title)
        ]

        assert len(severity_tables) == 0


def test_audit_plugin_displays_duration() -> None:
    """Test audit plugin displays execution duration."""
    plugin = AuditPlugin()

    context = PluginExecutionContext(
        datasources=["ds1"],
        key=ResultKey("2024-01-01", {}),
        timestamp=1704067200.0,
        duration_ms=3141.59,
        results=[],
        symbols=[]
    )

    with patch.object(plugin, "console") as mock_console:
        plugin.process(context, "Duration Test")

        # Check duration is displayed
        print_calls = str(mock_console.print.call_args_list)
        assert "3.14" in print_calls  # Duration rounded to 2 decimal places
```

### Step 4.3: Initial Validation

```bash
# Type check
uv run mypy src/dqx/plugins.py tests/test_audit_plugin.py

# Lint check
uv run ruff check src/dqx/plugins.py tests/test_audit_plugin.py

# Run tests
uv run pytest tests/test_audit_plugin.py -v

# Check coverage
uv run pytest tests/test_audit_plugin.py -v --cov=dqx.plugins --cov-report=term-missing
```

### Step 4.4: Final Validation and Commit

```bash
# Run all tests including new audit plugin tests
uv run pytest tests/ -v

# Run full pre-commit checks
bin/run-hooks.sh

# Fix any issues:
# - For ruff: uv run ruff check --fix src/dqx/plugins.py tests/test_audit_plugin.py
# - For mypy: fix type annotations
# - For test failures: debug and fix

# Verify plugin coverage is still 100%
uv run pytest tests/test_plugins.py tests/test_audit_plugin.py -v --cov=dqx.plugins --cov-report=term-missing

# Once all checks pass, commit:
git add src/dqx/plugins.py tests/test_audit_plugin.py pyproject.toml
git commit -m "feat(plugins): register built-in audit plugin via entry point

- Add audit plugin entry point to pyproject.toml
- Update AuditPlugin to use context convenience methods
- Add comprehensive tests for audit display functionality
- Test metadata, duration display, and color coding
- Ensure 100% test coverage"
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
from dqx.common import PluginExecutionContext, ResultKey
from dqx.orm.repositories import MetricDB
from dqx.provider import DataSource

from tests.test_plugins import MockSuccessPlugin


def test_full_plugin_integration() -> None:
    """Test complete plugin integration from suite execution to plugin processing."""
    # Create checks
    @check(name="Price Check")
    def price_check(mp, ctx):
        ctx.assert_that(mp.average("price"))\
           .where(name="Average price is positive")\
           .is_positive()

    @check(name="Count Check")
    def count_check(mp, ctx):
        ctx.assert_that(mp.count("id"))\
           .where(name="Has records", severity="P0")\
           .is_gt(0)

    # Create mock plugin
    mock_plugin = MockSuccessPlugin()

    # Mock the plugin manager to use our mock plugin
    with patch("dqx.plugins.PluginManager") as MockPluginManager:
        mock_manager = MockPluginManager.return_value
        mock_manager.get_plugins.return_value = {"test": mock_plugin}
        mock_manager._metadata = {"test": mock_plugin.metadata()}

        # Capture process_all calls
        def capture_process_all(context, suite_name, config=None):
            mock_plugin.process(context, suite_name)

        mock_manager.process_all.side_effect = capture_process_all

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

        # Verify plugin received correct data
        assert mock_plugin.processed
        assert mock_plugin.received_suite_name == "Integration Test Suite"

        # Verify context structure
        context = mock_plugin.received_context
        assert isinstance(context, PluginExecutionContext)
        assert context.datasources == ["sales"]
        assert context.key == key
        assert context.duration_ms > 0

        # Verify context methods work
        assert context.total_assertions() == 2
        assert context.assertion_pass_rate() > 0


def test_enable_plugins_kill_switch() -> None:
    """Test that enable_plugins=False prevents plugin execution."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB
    from dqx.provider import DataSource

    @check(name="Test Check")
    def test_check(mp, ctx):
        ctx.assert_that(mp.count("id"))\
           .where(name="Has records")\
           .is_gt(0)

    # Create mock plugin
    mock_plugin = MockSuccessPlugin()

    with patch("dqx.plugins.PluginManager") as MockPluginManager:
        mock_manager = MockPluginManager.return_value
        mock_manager.get_plugins.return_value = {"test": mock_plugin}

        # Track if process_all is called
        process_all_called = False
        def mock_process_all(context):
            nonlocal process_all_called
            process_all_called = True
            mock_plugin.process(context)

        mock_manager.process_all.side_effect = mock_process_all

        # Create suite
        db = MetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite"
        )

        # Run suite with enable_plugins=False
        datasources = {"test": DataSource.from_records([{"id": 1}])}
        key = ResultKey(date.today(), {})
        suite.run(datasources, key, enable_plugins=False)

        # Verify plugin manager was NOT called at all
        assert not process_all_called
        assert not mock_plugin.processed


def test_enable_plugins_default_true() -> None:
    """Test that enable_plugins defaults to True."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB
    from dqx.provider import DataSource

    @check(name="Test Check")
    def test_check(mp, ctx):
        ctx.assert_that(mp.count("id"))\
           .where(name="Has records")\
           .is_gt(0)

    # Create mock plugin
    mock_plugin = MockSuccessPlugin()

    with patch("dqx.plugins.PluginManager") as MockPluginManager:
        mock_manager = MockPluginManager.return_value
        mock_manager.get_plugins.return_value = {"test": mock_plugin}

        # Track if process_all is called
        process_all_called = False
        def mock_process_all(context):
            nonlocal process_all_called
            process_all_called = True
            mock_plugin.process(context)

        mock_manager.process_all.side_effect = mock_process_all

        # Create suite
        db = MetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite"
        )

        # Run suite without specifying enable_plugins (should default to True)
        datasources = {"test": DataSource.from_records([{"id": 1}])}
        key = ResultKey(date.today(), {})
        suite.run(datasources, key)

        # Verify plugin manager WAS called (default behavior)
        assert process_all_called
        assert mock_plugin.processed
```

### Step 5.2: Initial Validation

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

### Step 5.3: Final Validation and Commit

```bash
# Run complete test suite
uv run pytest tests/ -v

# Run full pre-commit checks on all files
bin/run-hooks.sh

# Fix any issues found:
# - For ruff: uv run ruff check --fix tests/test_plugin_integration.py
# - For mypy: fix type annotations
# - For test failures: debug and fix

# Verify all tests still pass
uv run pytest tests/ -v

# Once everything passes, commit:
git add tests/test_plugin_integration.py
git commit -m "test(plugins): add full integration test with context dataclass

- Test complete flow from suite execution to plugin processing
- Verify PluginExecutionContext is passed correctly
- Test disabled plugin configuration
- Ensure context methods work in integration"
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
import os
from dqx.common import PluginExecutionContext, PluginMetadata

class MyPlugin:
    @staticmethod
    def metadata() -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            author="Your Name",
            description="My custom DQX plugin",
            capabilities={"alerts", "email"}
        )

    def __init__(self) -> None:
        """Initialize plugin - read config from environment."""
        self.smtp_host = os.getenv("SMTP_HOST", "localhost")
        self.recipients = os.getenv("ALERT_RECIPIENTS", "").split(",")

    def process(
        self,
        context: PluginExecutionContext,
        suite_name: str
    ) -> None:
        """Process validation results."""
        if context.failed_assertions() > 0:
            failures_by_sev = context.failures_by_severity()
            critical = failures_by_sev.get("P0", 0) + failures_by_sev.get("P1", 0)

            if critical > 0:
                # Send alert
                print(f"Alert: {critical} critical failures in {suite_name}")
                print(f"Pass rate: {context.assertion_pass_rate():.1f}%")
```

### 2. Register in pyproject.toml

```toml
[project.entry-points."dqx.plugins"]
my_plugin = "mypackage.plugins:MyPlugin"
```

### 3. Use in DQX

```python
from dqx.api import VerificationSuite

# Plugin will be automatically loaded via entry points
suite = VerificationSuite(
    checks=checks,
    db=db,
    name="My Suite"
)

# Set environment variables for plugin configuration
# export SMTP_HOST=mail.example.com
# export ALERT_RECIPIENTS=alerts@example.com,team@example.com
```

## Plugin Protocol

Plugins must implement two methods:

### `metadata() -> PluginMetadata` (static method)
Return plugin metadata including name, version, author, description, and capabilities.

### `process(context: PluginExecutionContext, suite_name: str) -> None`
Called after suite execution with results and context.

## PluginExecutionContext

The context object provides convenient methods for accessing results:

```python
# Assertion metrics
context.total_assertions()      # Total number of assertions
context.failed_assertions()     # Number of failed assertions
context.passed_assertions()     # Number of passed assertions
context.assertion_pass_rate()   # Pass rate as percentage (0-100)

# Symbol metrics
context.total_symbols()         # Total number of symbols
context.failed_symbols()        # Number of failed symbol computations

# Grouping methods
context.assertions_by_severity()  # {"P0": 5, "P1": 3, ...}
context.failures_by_severity()    # {"P0": 2, "P1": 1, ...}

# Raw data access
context.results                 # List[AssertionResult]
context.symbols                 # List[SymbolInfo]
context.datasources            # List[str] - dataset names
context.key                    # ResultKey with date and tags
context.timestamp              # Unix timestamp of execution
context.duration_ms            # Execution duration in milliseconds
```

## Built-in Plugins

### Audit Plugin

DQX includes a built-in audit plugin that displays colorful execution statistics:

- Automatically loaded via entry point: `audit`
- Shows execution summary with pass/fail rates
- Displays failures grouped by severity with color coding
- Shows symbol computation statistics
- Uses Rich tables for beautiful console output

## Best Practices

1. **Use metadata** - Provide clear metadata for plugin identification
2. **Handle errors gracefully** - Don't let plugin failures crash the validation suite
3. **Use context methods** - Leverage the convenience methods instead of manual calculations
4. **Log appropriately** - Use Python's logging module for debugging
5. **Keep it focused** - Each plugin should do one thing well
6. **Document configuration** - Clearly document required and optional settings
7. **Test thoroughly** - Include unit tests for your plugin logic
```

### Step 6.3: Update pyproject.toml Comments

Add plugin registration example to pyproject.toml:

```toml
# Plugin System
# DQX uses Python entry points for plugin discovery.
# Plugins must implement the ResultProcessor protocol with metadata.
#
# Example registration:
# [project.entry-points."dqx.plugins"]
# email_alerts = "mypackage.plugins:EmailAlertsPlugin"
# slack_notifier = "mypackage.plugins:SlackPlugin"
# datadog_metrics = "mypackage.plugins:DatadogPlugin"
```

### Step 6.4: Initial Validation

```bash
# Verify markdown syntax
cat docs/plugins.md

# Run full test suite
uv run pytest tests/ -v
```

### Step 6.5: Final Validation and Commit

```bash
# Run complete test suite one final time
uv run pytest tests/ -v

# Run full pre-commit checks
bin/run-hooks.sh

# Fix any issues:
# - Documentation formatting
# - Any test failures
# - Linting issues

# Final verification - all tests should pass
uv run pytest tests/ -v

# Once everything is green, commit:
git add README.md docs/plugins.md pyproject.toml
git commit -m "docs(plugins): add plugin development guide with v2 features

- Update README with brief plugin mention
- Create comprehensive plugin guide showing PluginExecutionContext usage
- Document metadata requirements and convenience methods
- Show how to disable plugins with PluginConfig protocol
- Add pyproject.toml registration examples"
```

---

## Task Group 7: Timer Coverage to 100%

### Step 7.1: Update timer.py Tests

Add these test cases to `tests/test_timer.py` to achieve 100% coverage:

```python
def test_timer_context_manager() -> None:
    """Test Timer as a context manager with Registry."""
    registry = Registry()
    timer = registry.timer("test_metric")

    with timer:
        sleep(0.05)

    assert timer.elapsed_ms() > 50
    assert registry["test_metric"] > 50


def test_timer_elapsed_before_exit() -> None:
    """Test Timer.elapsed_ms() called before timer stops."""
    registry = Registry()
    metric = Metric("test", registry)
    timer = Timer(metric)

    # Enter context but don't exit yet
    timer.__enter__()

    # Should raise RuntimeError when called before exit
    with pytest.raises(RuntimeError, match="Timer has not been stopped yet"):
        timer.elapsed_ms()

    # Clean up
    timer.__exit__(None, None, None)


def test_time_limiting_no_limit() -> None:
    """Test TimeLimiting with None time_limit (no alarm)."""
    with TimeLimiting(None) as timer:
        sleep(0.1)

    assert timer.elapsed_ms() > 100
    # Should complete without any signal handling


def test_metric_collection() -> None:
    """Test Metric.collect() and value property."""
    registry = Registry()
    metric = Metric("test_metric", registry)

    # Initially no value
    assert metric.value is None

    # Collect a value
    metric.collect(123.45)

    # Check it was stored
    assert metric.value == 123.45
    assert registry["test_metric"] == 123.45


def test_registry_timer_method() -> None:
    """Test Registry.timer() returns a Timer instance."""
    registry = Registry()

    timer = registry.timer("my_metric")

    assert isinstance(timer, Timer)
    assert timer.collector.name == "my_metric"
    assert timer.collector.collector is registry


def test_timed_decorator_with_kwargs() -> None:
    """Test @Timer.timed decorator with keyword arguments."""
    metric = Metric("with_kwargs", Registry())

    @Timer.timed(collector=metric, extra_arg="test")
    def fn_with_args(x: int, y: int = 10) -> int:
        sleep(0.05)
        return x + y

    result = fn_with_args(5, y=20)

    assert result == 25
    assert metric.value is not None
    assert metric.value > 50
```

### Step 7.2: Initial Validation

```bash
# Run timer tests with coverage
uv run pytest tests/test_timer.py -v --cov=dqx.timer --cov-report=term-missing

# Verify we now have 100% coverage
```

### Step 7.3: Final Validation and Commit

```bash
# Run all tests to ensure nothing is broken
uv run pytest tests/ -v

# Run full pre-commit checks
bin/run-hooks.sh

# Fix any issues:
# - For ruff: uv run ruff check --fix tests/test_timer.py
# - For mypy: fix type annotations
# - For test failures: debug and fix

# Verify timer.py has 100% coverage
uv run pytest tests/test_timer.py -v --cov=dqx.timer --cov-report=term-missing

# Once everything passes, commit:
git add tests/test_timer.py
git commit -m "test(timer): achieve 100% code coverage

- Test Timer context manager with Registry
- Test Timer.elapsed_ms() error case before exit
- Test TimeLimiting with None time_limit
- Test Metric collection and value property
- Test Registry.timer() method
- Test @Timer.timed decorator with kwargs"
```

---

## Success Criteria

After completing all task groups, you should have:

1. ✅ Plugin system with `PluginMetadata` and `PluginExecutionContext`
2. ✅ Integration with VerificationSuite tracking duration
3. ✅ Built-in audit plugin using context convenience methods
4. ✅ Comprehensive tests with 100% coverage
5. ✅ Documentation showing v2 features
6. ✅ Clean git history with atomic commits
7. ✅ Timer module with 100% code coverage

The v2 plugin system is now ready with:
- **PluginMetadata** for better plugin identification
- **PluginExecutionContext** dataclass with convenience methods
- Built-in audit plugin using the new context methods
- Full documentation showing how to leverage the simplified API
- No backward compatibility code or complex features
- **100% test coverage** for all modules including timer.py

## Changelog

### Changes from v1 to v2

1. **Removed all configuration support**:
   - No `initialize()` method in the ResultProcessor protocol
   - No `plugin_config` parameter in VerificationSuite
   - No `PluginConfig` protocol
   - Plugins handle their own configuration (e.g., environment variables)

2. **Simplified plugin protocol** to just 2 methods:
   - `metadata() -> PluginMetadata` - Return plugin metadata
   - `process(context: PluginExecutionContext, suite_name: str) -> None` - Process results

3. **Made PluginMetadata immutable**:
   - Added `@dataclass(frozen=True)` for immutability

4. **Improved plugin discovery logging**:
   - Changed from error to warning: `logger.warning(f"Plugin not available: {ep.name}")`

5. **Removed metadata caching**:
   - Removed `_metadata` dict from PluginManager
   - `get_metadata()` now calls `plugin.__class__.metadata()` on demand

6. **Converted to property access**:
   - Changed `_get_plugin_manager()` method to `@property plugin_manager`

7. **Followed strict TDD methodology**:
   - Each task group starts with failing tests
   - Implements minimal code to pass tests
   - Ends with clean commit-ready state

This v2 plan simplifies the plugin architecture by focusing on just two key improvements:
1. Adding `PluginMetadata` for better plugin identification
2. Converting the execution context to a proper dataclass with convenient methods

No complex features like timeouts, circuit breakers, or operational monitoring. Just clean, simple enhancements.
