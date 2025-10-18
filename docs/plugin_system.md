# DQX Plugin System Documentation

The DQX Plugin System allows extending validation result processing with custom plugins. Plugins can generate reports, send notifications, store metrics, or perform any custom processing after validation completes.

## Overview

The plugin system provides:
- **Extensibility**: Add custom result processors without modifying core DQX code
- **Isolation**: Plugin failures don't affect validation execution
- **Performance**: Built-in timeouts prevent slow plugins from blocking
- **Rich Context**: Access to all validation results, symbols, and metadata

## Architecture

```
VerificationSuite
    │
    ├── Executes validations
    │
    └── PluginManager.process_all()
            │
            ├── AuditPlugin (built-in)
            ├── CustomPlugin1
            └── CustomPlugin2
```

## Creating a Plugin

Plugins must implement the `ResultProcessor` protocol:

```python
from dqx.common import PluginExecutionContext, PluginMetadata
from dqx.plugins import ResultProcessor


class MyPlugin:
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            author="Your Name",
            description="Description of what your plugin does",
            capabilities={"reporting", "notifications"},  # Optional
        )

    def process(self, context: PluginExecutionContext) -> None:
        """Process validation results."""
        # Your plugin logic here
        pass
```

## Plugin Context

The `PluginExecutionContext` provides comprehensive information:

```python
@dataclass
class PluginExecutionContext:
    suite_name: str  # Name of the verification suite
    datasources: list[str]  # Data sources used
    key: ResultKey  # Date and tags
    timestamp: float  # Unix timestamp
    duration_ms: float  # Execution time in milliseconds
    results: list[AssertionResult]  # All assertion results
    symbols: list[SymbolInfo]  # All computed symbols
```

Available convenience methods:
- `total_assertions() -> int` - Total number of assertions
- `failed_assertions() -> int` - Number of failed assertions
- `passed_assertions() -> int` - Number of passed assertions
- `assertion_pass_rate() -> float` - Percentage of passed assertions
- `total_symbols() -> int` - Total number of symbols
- `failed_symbols() -> int` - Number of failed symbols
- `assertions_by_severity() -> dict[str, int]` - Count by severity
- `failures_by_severity() -> dict[str, int]` - Failures by severity

## Built-in Plugins

### AuditPlugin

The built-in audit plugin displays a Rich-formatted summary of validation results:

```python
from dqx.api import VerificationSuite

# Create suite - plugins are managed internally
suite = VerificationSuite(checks, db, "MyDataQuality")

# Plugins are enabled by default when run() is called
suite.run(datasources, key)  # Plugins enabled

# Or explicitly disable plugins during run
suite.run(datasources, key, enable_plugins=False)
```

Output example:
```
═══ DQX Audit Report ═══
Suite: MyDataQuality
Date: 2025-10-18
Duration: 342.50ms
Datasets: products, inventory

     Execution Summary
╭──────────────────┬───────┬───────╮
│ Metric           │ Count │  Rate │
├──────────────────┼───────┼───────┤
│ Total Assertions │     4 │       │
│ Passed ✓         │     3 │ 75.0% │
│ Failed ✗         │     1 │ 25.0% │
╰──────────────────┴───────┴───────╯
```

## Example Plugins

### JSON Reporter

```python
class JSONReporterPlugin:
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="json_reporter",
            version="1.0.0",
            author="Your Team",
            description="Outputs results as JSON",
        )

    def process(self, context: PluginExecutionContext) -> None:
        import json

        report = {
            "suite": context.suite_name,
            "date": context.key.yyyy_mm_dd.isoformat(),
            "duration_ms": context.duration_ms,
            "summary": {
                "total": context.total_assertions(),
                "passed": context.passed_assertions(),
                "failed": context.failed_assertions(),
                "pass_rate": context.assertion_pass_rate(),
            },
            "failures_by_severity": context.failures_by_severity(),
        }

        with open("validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
```

### Slack Notifier

```python
class SlackNotifierPlugin:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="slack_notifier",
            version="1.0.0",
            author="DevOps Team",
            description="Sends validation alerts to Slack",
        )

    def process(self, context: PluginExecutionContext) -> None:
        if context.failed_assertions() > 0:
            import requests

            failures = context.failures_by_severity()
            message = {
                "text": f"⚠️ Data Quality Alert: {context.suite_name}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Suite:* {context.suite_name}\n"
                            f"*Pass Rate:* {context.assertion_pass_rate():.1f}%\n"
                            f"*Failures:* {failures}",
                        },
                    }
                ],
            }

            requests.post(self.webhook_url, json=message)
```

### Metrics Collector

```python
class MetricsCollectorPlugin:
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="metrics_collector",
            version="1.0.0",
            author="Analytics Team",
            description="Collects validation metrics",
        )

    def process(self, context: PluginExecutionContext) -> None:
        # Send metrics to monitoring system
        import statsd

        client = statsd.StatsClient("localhost", 8125)

        # Send gauges
        client.gauge(
            f"dqx.{context.suite_name}.assertions.total", context.total_assertions()
        )
        client.gauge(
            f"dqx.{context.suite_name}.assertions.failed", context.failed_assertions()
        )
        client.gauge(
            f"dqx.{context.suite_name}.pass_rate", context.assertion_pass_rate()
        )

        # Send timing
        client.timing(f"dqx.{context.suite_name}.duration", context.duration_ms)
```

## Plugin Registration

### Method 1: Entry Points (Recommended)

Register plugins in your package's `pyproject.toml`:

```toml
[project.entry-points."dqx.plugins"]
json_reporter = "mypackage.plugins:JSONReporterPlugin"
slack_notifier = "mypackage.plugins:SlackNotifierPlugin"
```

This approach supports automatic plugin discovery when your package is installed.

### Method 2: Manual Registration

For testing or dynamic plugin loading:

```python
from dqx.api import VerificationSuite

# Create suite
suite = VerificationSuite(checks, db, "MyData")

# Register custom plugins using fully qualified class names
suite.plugin_manager.register_plugin("mypackage.plugins.MyCustomPlugin")

# Clear default plugins if needed
suite.plugin_manager.clear_plugins()

# Register only your plugins
suite.plugin_manager.register_plugin("mypackage.plugins.JSONReporterPlugin")

# Note: For plugins with constructor arguments, you'll need to use entry points
# or create a wrapper class that handles initialization
```

## Error Handling

Plugins are executed with comprehensive error handling:

1. **Plugin Exceptions**: Caught and logged, suite continues
2. **Timeouts**: Plugins exceeding 60 seconds are terminated
3. **Invalid Plugins**: Rejected during discovery

Example error handling:

```python
class RobustPlugin:
    def process(self, context: PluginExecutionContext) -> None:
        try:
            # Risky operation
            external_api_call()
        except Exception as e:
            # Plugin should handle its own errors gracefully
            logger.error(f"Plugin error: {e}")
            # Can still do partial processing
            self.write_local_backup(context)
```

## Performance Considerations

1. **Timeouts**: Default 60-second timeout per plugin
2. **Async Operations**: Plugins run synchronously; use threads/async internally if needed
3. **Resource Usage**: Plugins should clean up resources in case of timeout

```python
class EfficientPlugin:
    def process(self, context: PluginExecutionContext) -> None:
        # Use context methods for efficiency
        if context.assertion_pass_rate() > 99.0:
            # Skip expensive processing for perfect runs
            return

        # Process only failures
        for result in context.results:
            if result.status == "FAILURE":
                self.process_failure(result)
```

## Testing Plugins

```python
import pytest
from datetime import datetime
from dqx.common import PluginExecutionContext, ResultKey, AssertionResult
from returns.result import Success


def test_my_plugin():
    # Create test context
    context = PluginExecutionContext(
        suite_name="TestSuite",
        datasources=["test_db"],
        key=ResultKey(datetime.now().date(), {}),
        timestamp=time.time(),
        duration_ms=100.0,
        results=[
            AssertionResult(
                yyyy_mm_dd=datetime.now().date(),
                suite="TestSuite",
                check="test_check",
                assertion="test_assertion",
                severity="P1",
                status="OK",
                metric=Success(1.0),
            )
        ],
        symbols=[],
    )

    # Test plugin
    plugin = MyPlugin()
    plugin.process(context)

    # Assert expected behavior
    assert os.path.exists("expected_output.json")
```

## Best Practices

1. **Be Resilient**: Handle errors gracefully
2. **Be Fast**: Complete processing quickly
3. **Be Focused**: Do one thing well
4. **Be Testable**: Write unit tests for your plugins
5. **Be Documented**: Include clear metadata and docstrings

## Future Enhancements

The plugin system is designed for extension. Potential future features:
- Async plugin execution
- Plugin dependencies
- Plugin configuration system
- Built-in plugin marketplace
- Plugin health monitoring

## See Also

- [Plugin Demo](../examples/plugin_demo.py) - Complete working example
- [API Reference](api_reference.md) - Full API documentation
- [Architecture](architecture.md) - System design details
