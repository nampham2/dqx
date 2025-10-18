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

### Entry Point Registration

External plugins register via `pyproject.toml`:

```toml
[project.optional-dependencies]
bkng = ["bkng-dqx>=1.1"]

[project.entry-points."dqx.plugins"]
bkng_dqx = "external.bkng.dqx:Integration"
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

## Example Plugin Implementation

```python
# External package: bkng-dqx
from typing import Any
from dqx.common import AssertionResult, SymbolInfo

class BkngIntegration:
    def initialize(self, config: dict[str, Any]) -> None:
        self.email_config = config.get("email", {})
        self.severity_threshold = config.get("severity_threshold", ["P0", "P1"])

    def process(
        self,
        results: list[AssertionResult],
        symbols: list[SymbolInfo],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        failures = [
            r for r in results
            if r.status == "FAILURE" and r.severity in self.severity_threshold
        ]
        if failures:
            self._send_alert_email(failures, suite_name, context)

    def _send_alert_email(
        self,
        failures: list[AssertionResult],
        suite_name: str,
        context: dict[str, Any]
    ) -> None:
        # Email sending logic here
        pass
```

## Success Criteria

1. Plugins can be discovered and loaded from entry points
2. Plugins receive validation results after suite execution
3. Plugin failures don't break validation flow
4. Full type annotations pass mypy strict mode
5. 100% test coverage for plugin code
6. Clear documentation for plugin developers
