# Plugin Instance Registration Implementation Plan v2 - TDD Approach

## Overview

This plan details the implementation of support for registering `PostProcessor` instances directly in the `PluginManager.register_plugin` method, while maintaining backward compatibility with the existing string-based registration.

This version follows strict Test-Driven Development (TDD) with independently committable task groups.

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

**Goal**: Set up test fixtures and type hints that don't break existing functionality

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
```

#### Step 1.2: Add type stubs to `src/dqx/plugins.py`

Add import at the top:
```python
from typing import overload
```

Add overload stubs and modify `register_plugin` (keep existing implementation working):

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
        """
        # For now, only handle string case to keep tests passing
        if not isinstance(plugin, str):
            raise NotImplementedError("PostProcessor instances not yet supported")

        # Keep existing implementation for strings
        class_name = plugin
        try:
            # ... (existing implementation remains here unchanged)
```

#### Validation
```bash
uv run pytest tests/ -v  # All should pass
uv run mypy src/dqx/plugins.py  # Should pass
uv run ruff check  # Should pass
```

**Commit**: `test: add test fixtures and type stubs for plugin instance registration`

### Task Group 2: First Working Test + Implementation

**Goal**: Implement basic instance registration with one passing test

#### Step 2.1: Write failing test in `tests/test_plugin_manager.py`

```python
def test_register_plugin_instance() -> None:
    """Test registering a PostProcessor instance."""
    manager = PluginManager()
    plugin = ValidInstancePlugin()

    manager.register_plugin(plugin)

    assert manager.plugin_exists("instance_plugin")
    assert manager.get_plugins()["instance_plugin"] is plugin
```

Run test to see it fail:
```bash
uv run pytest tests/test_plugin_manager.py::test_register_plugin_instance -v
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

**Commit**: `feat(plugins): implement basic PostProcessor instance registration`

### Task Group 3: Validation Tests + Implementation

**Goal**: Add error handling with tests

#### Step 3.1: Write validation tests in `tests/test_plugin_manager.py`

```python
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
```

Run tests to see them fail:
```bash
uv run pytest tests/test_plugin_manager.py::test_register_plugin_instance_invalid -v
uv run pytest tests/test_plugin_manager.py::test_register_plugin_instance_bad_metadata -v
```

#### Step 3.2: Enhance `_register_from_instance` with validation

Update the method in `src/dqx/plugins.py`:

```python
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

#### Validation
```bash
uv run pytest tests/ -v  # All should pass
uv run mypy src/dqx/plugins.py  # Should pass
uv run ruff check  # Should pass
```

**Commit**: `feat(plugins): add validation for instance registration`

### Task Group 4: Integration Tests

**Goal**: Test mixed usage scenarios

#### Step 4.1: Add integration tests to `tests/test_plugin_manager.py`

```python
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


def test_register_plugin_instance_with_constructor() -> None:
    """Test registering a plugin that requires constructor arguments."""
    manager = PluginManager()
    plugin = PluginWithConstructor(threshold=0.95, debug=True)

    manager.register_plugin(plugin)

    assert manager.plugin_exists("configured_plugin")
    registered = manager.get_plugins()["configured_plugin"]
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
```

#### Validation
```bash
uv run pytest tests/ -v  # All should pass
uv run mypy src/dqx/plugins.py  # Should pass
uv run ruff check  # Should pass
bin/run-hooks.sh --all  # Full validation
```

**Commit**: `test(plugins): add integration tests for mixed registration`

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
```

#### Validation
```bash
uv run pytest tests/ -v  # All should pass
uv run mypy tests/test_plugin_typing.py  # Should pass
```

**Commit**: `test(plugins): add type checking validation tests`

### Task Group 6: Documentation Update

**Goal**: Update plugin documentation

#### Step 6.1: Update `docs/plugin_system.md`

Add a new section after the existing plugin registration documentation:

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
```

#### Final Validation
```bash
uv run pytest tests/ -v
bin/run-hooks.sh --all
```

**Commit**: `docs(plugins): add instance registration documentation`

## Success Criteria

- ✅ Each task group passes all tests independently
- ✅ No linting or type checking errors at any stage
- ✅ Backward compatibility maintained throughout
- ✅ TDD approach: tests written before implementation
- ✅ Clean, meaningful commits after each task group
