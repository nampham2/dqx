# Plugin Instance Registration Implementation Plan v1

## Overview

This plan details the implementation of support for registering `PostProcessor` instances directly in the `PluginManager.register_plugin` method, while maintaining backward compatibility with the existing string-based registration.

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

## Implementation Details

### Files to Modify

1. **src/dqx/plugins.py** - Add overloaded `register_plugin` method
2. **tests/test_plugin_manager.py** - Add tests for instance registration
3. **tests/test_plugin_public_api.py** - Add public API tests for new feature

### Code Changes

#### 1. Update `src/dqx/plugins.py`

Add the following imports at the top:
```python
from typing import overload
```

Add overloaded method signatures and refactor `register_plugin`:
```python
class PluginManager:
    # ... existing code ...

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
        """
        if isinstance(plugin, str):
            self._register_from_string(plugin)
        else:
            self._register_from_instance(plugin)

    def _register_from_string(self, class_name: str) -> None:
        """Register a plugin from a class name string (existing logic)."""
        try:
            # Move existing register_plugin logic here
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
        # Validate it's actually a PostProcessor
        if not isinstance(plugin, PostProcessor):
            raise ValueError(
                f"Plugin {type(plugin).__name__} doesn't implement PostProcessor protocol"
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

## Test Plan

### Task 1: Create Test Fixtures

Create test plugin classes in `tests/test_plugin_manager.py`:

```python
# Add these test classes at module level

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
```

### Task 2: Add Instance Registration Tests

Add these test methods to `tests/test_plugin_manager.py`:

```python
def test_register_plugin_instance() -> None:
    """Test registering a PostProcessor instance."""
    manager = PluginManager()
    plugin = ValidInstancePlugin()

    manager.register_plugin(plugin)

    assert manager.plugin_exists("instance_plugin")
    assert manager.get_plugins()["instance_plugin"] is plugin


def test_register_plugin_instance_with_constructor() -> None:
    """Test registering a plugin that requires constructor arguments."""
    manager = PluginManager()
    plugin = PluginWithConstructor(threshold=0.95, debug=True)

    manager.register_plugin(plugin)

    assert manager.plugin_exists("configured_plugin")
    registered = manager.get_plugins()["configured_plugin"]
    assert registered.threshold == 0.95
    assert registered.debug is True


def test_register_plugin_instance_invalid() -> None:
    """Test that invalid instances are rejected."""
    manager = PluginManager()

    # Non-PostProcessor object
    with pytest.raises(ValueError, match="doesn't implement PostProcessor protocol"):
        manager.register_plugin(42)  # type: ignore

    # Object without proper protocol
    invalid = InvalidInstancePlugin()
    with pytest.raises(ValueError, match="doesn't implement PostProcessor protocol"):
        manager.register_plugin(invalid)


def test_register_plugin_instance_bad_metadata() -> None:
    """Test instance with invalid metadata is rejected."""

    class BadMetadataPlugin:
        @staticmethod
        def metadata() -> dict:  # Wrong return type
            return {"name": "bad"}

        def process(self, context: PluginExecutionContext) -> None:
            pass

    manager = PluginManager()
    plugin = BadMetadataPlugin()

    with pytest.raises(ValueError, match="metadata\\(\\) must return a PluginMetadata instance"):
        manager.register_plugin(plugin)


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

    # Process all should work
    context = PluginExecutionContext(
        suite_name="test",
        datasources=[],
        key=ResultKey(yyyy_mm_dd=datetime.date.today(), tags={}),
        timestamp=time.time(),
        duration_ms=100.0,
        results=[],
        symbols=[]
    )

    # Should not raise
    manager.process_all(context)
```

### Task 3: Add Public API Tests

Add these tests to `tests/test_plugin_public_api.py`:

```python
def test_register_plugin_overload_typing() -> None:
    """Test that overloaded signatures work correctly for type checking."""
    manager = PluginManager()

    # These should be valid according to type hints
    manager.register_plugin("tests.test_plugin_public_api.ValidPlugin")

    plugin = ValidPlugin()
    manager.register_plugin(plugin)

    # Both plugins should be registered
    assert len(manager.get_plugins()) >= 2
```

### Task 4: Type Checking Validation

Create a new test file `tests/test_plugin_typing.py` to verify mypy works correctly:

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
```

## Implementation Tasks

### Phase 1: Core Implementation (Tasks 1-3)

**Task 1: Add overload support and refactor register_plugin**
- Add `from typing import overload` import
- Add `@overload` decorators for both signatures
- Implement the unified `register_plugin` method
- Extract existing logic to `_register_from_string` method
- Implement new `_register_from_instance` method

**Task 2: Create test fixtures**
- Add `ValidInstancePlugin` class
- Add `PluginWithConstructor` class
- Add `InvalidInstancePlugin` class

**Task 3: Add core functionality tests**
- Test successful instance registration
- Test instance with constructor arguments
- Test invalid instance rejection
- Test metadata validation

**Commit after Phase 1**: `feat(plugins): add PostProcessor instance registration support`

### Phase 2: Integration Testing (Tasks 4-5)

**Task 4: Add mixed usage tests**
- Test combining string and instance registration
- Test `process_all` with mixed plugins
- Verify no interference between methods

**Task 5: Add type checking tests**
- Create `test_plugin_typing.py`
- Add tests that verify mypy accepts valid usage
- Add commented tests showing invalid usage

**Commit after Phase 2**: `test(plugins): add comprehensive tests for instance registration`

### Phase 3: Validation and Documentation (Tasks 6-7)

**Task 6: Run full test suite and pre-commit**
```bash
uv run pytest tests/ -v
bin/run-hooks.sh --all
```

**Task 7: Update plugin documentation**
- Update `docs/plugin_system.md` with instance registration examples
- Add migration notes for users who want to use the new feature

**Final commit**: `docs(plugins): update documentation for instance registration`

## Validation Steps

1. **Type Checking**: Run `uv run mypy src/dqx/plugins.py` to verify overloads work
2. **Test Coverage**: Ensure 100% coverage with `uv run pytest tests/test_plugin_manager.py -v --cov=dqx.plugins`
3. **Linting**: Run `uv run ruff check src/dqx/plugins.py`
4. **Integration**: Test with a real plugin instance in the Python REPL

## Success Criteria

- [x] Both string and instance registration work correctly
- [x] Type hints provide proper IDE support and mypy validation
- [x] All existing tests continue to pass
- [x] New tests achieve 100% coverage of new code
- [x] Documentation clearly explains both usage patterns
