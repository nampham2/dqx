# Plugin Instance Registration Implementation Plan v3 - Enhanced TDD Approach

## Overview

This plan details the implementation of support for registering `PostProcessor` instances directly in the `PluginManager.register_plugin` method, while maintaining backward compatibility with the existing string-based registration.

This version incorporates all feedback from the v2 review, adding comprehensive edge case testing, better error handling, and extensive documentation.

### Current State

The `register_plugin` method currently only accepts strings representing fully qualified class names:
```python
manager.register_plugin("dqx.plugins.AuditPlugin")
```

### Desired State

Support both string-based and instance-based registration with proper type hints:
```python
# String-based (existing)
manager.register_plugin("dqx.plugins.AuditPlugin")

# Instance-based (new)
plugin = MyPlugin(config={"threshold": 0.95})
manager.register_plugin(plugin)
```

## TDD Implementation Strategy

Each task group must:
- ✅ Pass `uv run pytest tests/ -v`
- ✅ Pass `uv run mypy src/dqx/plugins.py`
- ✅ Pass `uv run ruff check`
- ✅ Pass pre-commit hooks
- ✅ Be committable with a meaningful change

## Task Groups

### Task Group 1: Test Infrastructure & Type Stubs

**Goal**: Set up test fixtures and type hints with defensive error handling

#### Step 1.1: Add test fixtures to `tests/test_plugin_manager.py`

Add these test classes at the module level (after imports):

```python
import datetime
import time
from dqx.common import PluginMetadata, PluginExecutionContext, ResultKey


class ValidInstancePlugin:
    """Valid plugin for testing instance registration."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="instance_plugin",
            version="1.0.0",
            author="Test",
            description="Test instance plugin"
        )

    def process(self, context: PluginExecutionContext) -> None:
        pass


class PluginWithConstructor:
    """Plugin that accepts constructor arguments."""

    def __init__(self, threshold: float, debug: bool = False):
        self.threshold = threshold
        self.debug = debug

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="configured_plugin",
            version="1.0.0",
            author="Test",
            description="Plugin with configuration"
        )

    def process(self, context: PluginExecutionContext) -> None:
        pass


class InvalidInstancePlugin:
    """Invalid plugin missing process method."""

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="invalid",
            version="1.0.0",
            author="Test",
            description="Invalid plugin"
        )


class StatefulPlugin:
    """Plugin that maintains state across calls."""

    def __init__(self):
        self.call_count = 0
        self.processed_suites = []

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="stateful_plugin",
            version="1.0.0",
            author="Test",
            description="Plugin with internal state"
        )

    def process(self, context: PluginExecutionContext) -> None:
        self.call_count += 1
        self.processed_suites.append(context.suite_name)
```

#### Step 1.2: Add type stubs to `src/dqx/plugins.py`

Add import at the top:
```python
from typing import overload
```

Add overload stubs and modify `register_plugin` with defensive error handling:

```python
    @overload
    def register_plugin(self, class_name: str) -> None:
        """Register a plugin by its fully qualified class name."""
        ...

    @overload
    def register_plugin(self, plugin: PostProcessor) -> None:
        """Register a plugin by passing a PostProcessor instance directly."""
        ...

    def register_plugin(self, plugin: str | PostProcessor) -> None:
        """
        Register a plugin either by class name or PostProcessor instance.

        Args:
            plugin: Either a fully qualified class name string or a PostProcessor instance

        Raises:
            ValueError: If the plugin is invalid or doesn't implement PostProcessor
            NotImplementedError: If PostProcessor instance registration is not yet supported
        """
        # Defensive type checking
        if isinstance(plugin, str):
            # Handle string case (existing implementation)
            class_name = plugin
            try:
                # ... (existing implementation remains here unchanged)
            except Exception as e:
                # ... (existing error handling)
        elif isinstance(plugin, PostProcessor):
            # PostProcessor instance (not yet implemented)
            raise NotImplementedError(
                "PostProcessor instance registration not yet supported. "
                "This feature is being implemented. Please use string-based "
                "registration for now: register_plugin('module.Class')"
            )
        else:
            # Invalid type
            raise ValueError(
                f"Invalid plugin type: {type(plugin).__name__}. "
                f"Expected either a string (module.Class format) or a "
                f"PostProcessor instance."
            )
```

#### Validation
```bash
uv run pytest tests/ -v  # All should pass
uv run mypy src/dqx/plugins.py  # Should pass
uv run ruff check  # Should pass
```

**Commit**: `test: add test fixtures and defensive type stubs for plugin instance registration`

### Task Group 2: First Working Test + Minimal Implementation

**Goal**: Implement basic instance registration with one passing test

**Note**: This task group implements minimal validation. Full validation will be added in Task Group 3.

#### Step 2.1: Write failing test in `tests/test_plugin_manager.py`

```python
def test_register_plugin_instance_stores_reference() -> None:
    """Test registering a PostProcessor instance stores the exact reference."""
    manager = PluginManager()
    plugin = ValidInstancePlugin()

    manager.register_plugin(plugin)

    assert manager.plugin_exists("instance_plugin")
    # Verify exact same instance is stored (not a copy)
    assert manager.get_plugins()["instance_plugin"] is plugin
```

Run test to see it fail:
```bash
uv run pytest tests/test_plugin_manager.py::test_register_plugin_instance_stores_reference -v
# Should fail with NotImplementedError
```

#### Step 2.2: Implement minimal code in `src/dqx/plugins.py`

Refactor the `register_plugin` method:

```python
    def register_plugin(self, plugin: str | PostProcessor) -> None:
        """
        Register a plugin either by class name or PostProcessor instance.

        Args:
            plugin: Either a fully qualified class name string or a PostProcessor instance

        Raises:
            ValueError: If the plugin is invalid or doesn't implement PostProcessor
        """
        if isinstance(plugin, str):
            self._register_from_class(plugin)
        else:
            # Note: Minimal implementation - full validation in Task Group 3
            self._register_from_instance(plugin)

    def _register_from_class(self, class_name: str) -> None:
        """Register a plugin from a class name string (existing logic)."""
        try:
            # Move ALL existing register_plugin logic here unchanged
            # Parse the class name
            parts = class_name.rsplit(".", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid class name format: {class_name}")

            module_name, cls_name = parts

            # Import the module
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                raise ValueError(f"Cannot import module {module_name}: {e}")

            # Get the class
            if not hasattr(module, cls_name):
                raise ValueError(f"Module {module_name} has no class {cls_name}")

            plugin_class = getattr(module, cls_name)

            # Instantiate the plugin
            plugin = plugin_class()

            # Use isinstance to check protocol implementation (KISS principle)
            if not isinstance(plugin, PostProcessor):
                raise ValueError(f"Plugin class {class_name} doesn't implement PostProcessor protocol")

            # Validate metadata returns correct type
            metadata = plugin.metadata()
            if not isinstance(metadata, PluginMetadata):
                raise ValueError(f"Plugin class {class_name}'s metadata() must return a PluginMetadata instance")

            plugin_name = metadata.name

            # Store the plugin
            self._plugins[plugin_name] = plugin
            logger.info(f"Registered plugin: {plugin_name} (from {class_name})")

        except Exception as e:
            # Re-raise ValueError, let other exceptions propagate
            if not isinstance(e, ValueError):
                raise ValueError(f"Failed to register plugin {class_name}: {e}")
            raise

    def _register_from_instance(self, plugin: PostProcessor) -> None:
        """Register a PostProcessor instance directly."""
        # Minimal implementation to pass test
        # Full validation will be added in Task Group 3
        metadata = plugin.metadata()
        plugin_name = metadata.name
        self._plugins[plugin_name] = plugin
        logger.info(f"Registered plugin: {plugin_name} (instance)")
```

#### Validation
```bash
uv run pytest tests/ -v  # All should pass including new test
uv run mypy src/dqx/plugins.py  # Should pass
uv run ruff check  # Should pass
```

**Commit**: `feat(plugins): implement minimal PostProcessor instance registration`

### Task Group 3: Comprehensive Validation Tests + Implementation

**Goal**: Add error handling with extensive tests

#### Step 3.1: Write validation tests in `tests/test_plugin_manager.py`

```python
def test_register_plugin_instance_rejects_non_protocol_types() -> None:
    """Test that non-PostProcessor instances are rejected with helpful error."""
    manager = PluginManager()

    # Non-PostProcessor object
    with pytest.raises(ValueError) as exc_info:
        manager.register_plugin(42)  # type: ignore

    assert "doesn't implement PostProcessor protocol" in str(exc_info.value)
    assert "Ensure your plugin has both metadata() and process() methods" in str(exc_info.value)

    # Object without proper protocol
    invalid = InvalidInstancePlugin()
    with pytest.raises(ValueError) as exc_info:
        manager.register_plugin(invalid)

    assert "doesn't implement PostProcessor protocol" in str(exc_info.value)


def test_register_plugin_instance_validates_metadata_return_type() -> None:
    """Test instance with invalid metadata return type is rejected."""

    class BadMetadataPlugin:
        @staticmethod
        def metadata() -> dict:  # Wrong return type
            return {"name": "bad"}

        def process(self, context: PluginExecutionContext) -> None:
            pass

    manager = PluginManager()
    plugin = BadMetadataPlugin()

    with pytest.raises(ValueError) as exc_info:
        manager.register_plugin(plugin)

    assert "metadata() must return a PluginMetadata instance" in str(exc_info.value)
    assert "BadMetadataPlugin" in str(exc_info.value)
```

Run tests to see them fail:
```bash
uv run pytest tests/test_plugin_manager.py::test_register_plugin_instance_rejects_non_protocol_types -v
uv run pytest tests/test_plugin_manager.py::test_register_plugin_instance_validates_metadata_return_type -v
```

#### Step 3.2: Enhance `_register_from_instance` with full validation

Update the method in `src/dqx/plugins.py`:

```python
    def _register_from_instance(self, plugin: PostProcessor) -> None:
        """
        Register a PostProcessor instance directly.

        The registered instance is stored by reference, so any state maintained
        by the plugin persists across multiple process() calls.
        """
        # Validate it's actually a PostProcessor
        if not isinstance(plugin, PostProcessor):
            raise ValueError(
                f"Plugin {type(plugin).__name__} doesn't implement PostProcessor protocol. "
                f"Ensure your plugin has both metadata() and process() methods."
            )

        # Get and validate metadata
        metadata = plugin.metadata()
        if not isinstance(metadata, PluginMetadata):
            raise ValueError(
                f"Plugin {type(plugin).__name__}'s metadata() must return a PluginMetadata instance"
            )

        # Register using name from metadata
        plugin_name = metadata.name
        self._plugins[plugin_name] = plugin
        logger.info(f"Registered plugin: {plugin_name} (instance)")
```

#### Validation
```bash
uv run pytest tests/ -v  # All should pass
uv run mypy src/dqx/plugins.py  # Should pass
uv run ruff check  # Should pass
```

**Commit**: `feat(plugins): add comprehensive validation for instance registration`

### Task Group 4: Edge Case and Integration Tests

**Goal**: Test edge cases and mixed usage scenarios

#### Step 4.1: Add edge case tests to `tests/test_plugin_manager.py`

```python
def test_register_same_plugin_via_string_then_instance() -> None:
    """Test registering same plugin via different methods (second registration wins)."""
    manager = PluginManager()
    manager.clear_plugins()

    # First, register via string
    manager.register_plugin("dqx.plugins.AuditPlugin")
    assert manager.plugin_exists("audit")

    # Create a custom audit plugin instance
    custom_audit = AuditPlugin()
    # Mark it somehow to verify it's our instance
    custom_audit._custom_marker = "test_marker"

    # Register same plugin via instance (should replace)
    manager.register_plugin(custom_audit)

    # Verify our instance is now active
    assert manager.plugin_exists("audit")
    registered = manager.get_plugins()["audit"]
    assert hasattr(registered, "_custom_marker")
    assert registered._custom_marker == "test_marker"


def test_plugin_instance_state_persistence() -> None:
    """Test that plugin instance state persists across process() calls."""
    manager = PluginManager()

    # Create a stateful plugin
    plugin = StatefulPlugin()
    manager.register_plugin(plugin)

    # Create contexts
    context1 = PluginExecutionContext(
        suite_name="suite1",
        datasources=[],
        key=ResultKey(datetime.date.today(), {}),
        timestamp=time.time(),
        duration_ms=100.0,
        results=[],
        symbols=[]
    )

    context2 = PluginExecutionContext(
        suite_name="suite2",
        datasources=[],
        key=ResultKey(datetime.date.today(), {}),
        timestamp=time.time(),
        duration_ms=200.0,
        results=[],
        symbols=[]
    )

    # Process multiple times
    manager.process_all(context1)
    manager.process_all(context2)

    # Verify state persisted
    assert plugin.call_count == 2
    assert plugin.processed_suites == ["suite1", "suite2"]

    # Verify it's the same instance
    assert manager.get_plugins()["stateful_plugin"] is plugin


def test_mixed_registration_methods() -> None:
    """Test using both string and instance registration in same manager."""
    manager = PluginManager()

    # Register via string
    manager.clear_plugins()  # Clear default plugins first
    manager.register_plugin("dqx.plugins.AuditPlugin")

    # Register via instance
    instance_plugin = ValidInstancePlugin()
    manager.register_plugin(instance_plugin)

    # Both should exist
    assert manager.plugin_exists("audit")
    assert manager.plugin_exists("instance_plugin")

    # Verify the instance plugin is the exact object we registered
    assert manager.get_plugins()["instance_plugin"] is instance_plugin

    # Process all should work
    context = PluginExecutionContext(
        suite_name="test",
        datasources=[],
        key=ResultKey(datetime.date.today(), tags={}),
        timestamp=time.time(),
        duration_ms=100.0,
        results=[],
        symbols=[]
    )

    # Should not raise
    manager.process_all(context)


def test_register_plugin_instance_with_constructor_preserves_config() -> None:
    """Test registering a plugin with constructor args preserves configuration."""
    manager = PluginManager()
    plugin = PluginWithConstructor(threshold=0.95, debug=True)

    manager.register_plugin(plugin)

    assert manager.plugin_exists("configured_plugin")
    registered = manager.get_plugins()["configured_plugin"]
    assert registered is plugin  # Same instance
    assert registered.threshold == 0.95
    assert registered.debug is True
```

#### Step 4.2: Add public API test to `tests/test_plugin_public_api.py`

```python
def test_register_plugin_overload_typing() -> None:
    """Test that overloaded signatures work correctly for type checking."""
    manager = PluginManager()

    # Clear existing plugins
    manager.clear_plugins()

    # These should be valid according to type hints
    manager.register_plugin("tests.test_plugin_public_api.ValidPlugin")

    plugin = ValidPlugin()
    manager.register_plugin(plugin)

    # Both plugins should be registered
    assert len(manager.get_plugins()) == 2

    # The instance should be the exact object we passed
    plugins = manager.get_plugins()
    # Find our instance (it has a different name than string-registered one)
    instance_plugin = next(p for p in plugins.values() if p is plugin)
    assert instance_plugin is plugin
```

#### Validation
```bash
uv run pytest tests/ -v  # All should pass
uv run mypy src/dqx/plugins.py  # Should pass
uv run ruff check  # Should pass
bin/run-hooks.sh --all  # Full validation
```

**Commit**: `test(plugins): add comprehensive edge case and integration tests`

### Task Group 5: Type Checking Test File

**Goal**: Create dedicated mypy test file

#### Step 5.1: Create `tests/test_plugin_typing.py`

```python
"""Test file specifically for mypy type checking of plugin registration."""

from dqx.plugins import PluginManager, PostProcessor, PluginMetadata
from dqx.common import PluginExecutionContext


def test_type_hints_string() -> None:
    """String-based registration should type check correctly."""
    manager = PluginManager()
    # This should type check
    manager.register_plugin("dqx.plugins.AuditPlugin")


def test_type_hints_instance() -> None:
    """Instance-based registration should type check correctly."""

    class MyPlugin:
        @staticmethod
        def metadata() -> PluginMetadata:
            return PluginMetadata(
                name="my_plugin",
                version="1.0.0",
                author="Test",
                description="Test plugin"
            )

        def process(self, context: PluginExecutionContext) -> None:
            pass

    manager = PluginManager()
    plugin = MyPlugin()
    # This should type check
    manager.register_plugin(plugin)


def test_invalid_types_should_fail() -> None:
    """Invalid types should be caught by mypy."""
    manager = PluginManager()

    # These should fail mypy type checking
    # manager.register_plugin(42)  # type: ignore
    # manager.register_plugin([])  # type: ignore
    # manager.register_plugin(None)  # type: ignore
    # manager.register_plugin({"name": "dict"})  # type: ignore

    # The type: ignore comments are necessary to prevent mypy errors
    # during testing, but demonstrate that mypy would catch these
```

#### Validation
```bash
uv run pytest tests/ -v  # All should pass
uv run mypy tests/test_plugin_typing.py  # Should pass
```

**Commit**: `test(plugins): add type checking validation tests`

### Task Group 6: Comprehensive Documentation Update

**Goal**: Update plugin documentation with lifecycle, thread safety, and examples

#### Step 6.1: Update `docs/plugin_system.md`

Add new sections after the existing plugin registration documentation:

```markdown
### Instance-Based Registration

In addition to string-based registration, you can now register plugin instances directly:

```python
from dqx.plugins import PluginManager

# Create a plugin with configuration
class ConfiguredPlugin:
    def __init__(self, threshold: float):
        self.threshold = threshold

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="configured",
            version="1.0.0",
            author="Your Name",
            description="A configurable plugin"
        )

    def process(self, context: PluginExecutionContext) -> None:
        # Use self.threshold in processing
        pass

# Register the configured instance
manager = PluginManager()
plugin = ConfiguredPlugin(threshold=0.95)
manager.register_plugin(plugin)
```

This approach is useful when:
- Your plugin needs constructor arguments
- You want to use dependency injection
- You're testing with mock plugins
- You need to share state between plugin calls

### Plugin Lifecycle and State Management

When registering plugin instances, it's important to understand the lifecycle:

#### Shared State
The registered instance is stored directly, so any state maintained by the plugin persists across multiple `process()` calls:

```python
class AccumulatingPlugin:
    def __init__(self):
        self.call_count = 0
        self.total_assertions = 0

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(name="accumulator", version="1.0.0",
                              author="Test", description="Accumulates stats")

    def process(self, context: PluginExecutionContext) -> None:
        self.call_count += 1
        self.total_assertions += context.total_assertions()
        # This state persists between calls
```

#### Thread Safety
**Important**: PluginManager is not thread-safe. If concurrent access is needed, external synchronization is required.

```python
# If you need thread safety, wrap access:
import threading

lock = threading.Lock()

def register_plugin_threadsafe(manager: PluginManager, plugin: PostProcessor) -> None:
    with lock:
        manager.register_plugin(plugin)
```

#### Resource Management
Plugins with resources (connections, file handles) should implement proper cleanup:

```python
class DatabasePlugin:
    def __init__(self, connection_string: str):
        self.connection = None
        self.connection_string = connection_string

    def __enter__(self):
        self.connection = create_connection(self.connection_string)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(name="db_plugin", version="1.0.0",
                              author="Test", description="Database plugin")

    def process(self, context: PluginExecutionContext) -> None:
        # Use self.connection
        pass

# Usage with context manager
with DatabasePlugin("postgresql://...") as plugin:
    manager.register_plugin(plugin)
    # Plugin will be cleaned up when context exits
```

### Advanced Usage Examples

#### Dependency Injection
Use instance registration for dependency injection:

```python
# External dependencies
class MetricsCollector:
    def record(self, metric: str, value: float) -> None:
        # Send to monitoring system
        pass

class ConnectionPool:
    def get_connection(self):
        # Return database connection
        pass

# Plugin using dependency injection
class MonitoredDatabasePlugin:
    def __init__(self, metrics: MetricsCollector, pool: ConnectionPool):
        self.metrics = metrics
        self.pool = pool

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="monitored_db",
            version="1.0.0",
            author="Your Team",
            description="Database plugin with monitoring"
        )

    def process(self, context: PluginExecutionContext) -> None:
        start_time = time.time()

        with self.pool.get_connection() as conn:
            # Process results
            results_count = len(context.results)
            # Store to database
            conn.execute("INSERT INTO audit_log ...", results_count)

        # Record metrics
        elapsed = time.time() - start_time
        self.metrics.record("plugin.execution_time", elapsed)
        self.metrics.record("plugin.results_processed", results_count)

# Wire up dependencies
metrics = MetricsCollector()
pool = ConnectionPool()
plugin = MonitoredDatabasePlugin(metrics, pool)

# Register the configured plugin
manager = PluginManager()
manager.register_plugin(plugin)
```

#### Testing with Mock Plugins
Instance registration makes testing easier:

```python
# Test plugin that captures calls
class TestPlugin:
    def __init__(self):
        self.process_calls = []

    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(name="test", version="1.0.0",
                              author="Test", description="Test plugin")

    def process(self, context: PluginExecutionContext) -> None:
        self.process_calls.append({
            "suite_name": context.suite_name,
            "total_assertions": context.total_assertions(),
            "failed_assertions": context.failed_assertions()
        })

# In tests
def test_plugin_execution():
    manager = PluginManager()
    test_plugin = TestPlugin()
    manager.register_plugin(test_plugin)

    # Run your verification suite
    suite = VerificationSuite("test_suite")
    # ... add checks ...
    suite.run(date(2024, 1, 1))

    # Verify plugin was called correctly
    assert len(test_plugin.process_calls) == 1
    call = test_plugin.process_calls[0]
    assert call["suite_name"] == "test_suite"
```

### Migration Guide

If you have existing string-based plugin registrations, no changes are needed. Both methods are supported:

```python
# These all work together
manager = PluginManager()

# String-based (existing code continues to work)
manager.register_plugin("mycompany.plugins.AuditPlugin")

# Instance-based (new capability)
custom_plugin = CustomPlugin(config_file="/etc/dqx/plugin.conf")
manager.register_plugin(custom_plugin)

# Both plugins will execute during process_all()
```

### Best Practices

1. **Use string-based registration when**:
   - Plugin needs no configuration
   - Plugin is stateless
   - Plugin is distributed as a package

2. **Use instance-based registration when**:
   - Plugin requires constructor arguments
   - Plugin maintains state between calls
   - Testing with mock plugins
   - Using dependency injection
   - Plugin needs external resources

3. **Resource Management**:
   - Clean up resources in plugin destructors
   - Consider using context managers
   - Document resource requirements

4. **State Management**:
   - Document if your plugin maintains state
   - Make state thread-safe if needed
   - Consider state reset methods for testing
```

#### Final Validation
```bash
uv run pytest tests/ -v
bin/run-hooks.sh --all
```

**Commit**: `docs(plugins): add comprehensive instance registration documentation`

## Success Criteria

- ✅ Each task group passes all tests independently
- ✅ No linting or type checking errors at any stage
- ✅ Backward compatibility maintained throughout
- ✅ TDD approach: tests written before implementation
- ✅ Clean, meaningful commits after each task group
- ✅ Comprehensive edge case coverage
- ✅ Defensive error handling with helpful messages
- ✅ Complete documentation including lifecycle and examples

## Implementation Notes

### Key Improvements in v3

1. **Defensive Type Checking**: Task Group 1 now properly handles all types, not just PostProcessor instances
2. **Explicit Validation Stages**: Task Group 2 clearly notes minimal validation
3. **Comprehensive Edge Cases**: Task Group 4 includes state persistence and duplicate registration tests
4. **Better Test Names**: All tests use descriptive names indicating expected behavior
5. **Helpful Error Messages**: Validation errors guide users on how to fix issues
6. **Complete Documentation**: Includes lifecycle, thread safety, dependency injection, and testing examples

### Behavior Clarifications

- **Duplicate Registration**: Last registration wins (same as string-based)
- **Instance State**: Shared reference - state persists across calls
- **Thread Safety**: Not provided - external synchronization needed
- **Resource Management**: Plugin responsible for cleanup
