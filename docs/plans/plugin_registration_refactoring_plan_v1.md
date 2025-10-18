# Plugin Registration Refactoring Plan

## Overview

This plan refactors the DQX plugin system to use class names for registration instead of plugin instances. This simplifies the API and creates consistency between manual registration and automatic discovery via entry points.

## Key Changes

1. **`register_plugin` method signature change**
   - From: `register_plugin(self, name: str, plugin: object)`
   - To: `register_plugin(self, class_name: str)`
   - Plugin name will be derived from the plugin's metadata

2. **Remove `get_metadata` method**
   - Only used in one demo and one test file
   - Not part of core functionality

3. **Rename `plugin_name_exists` to `plugin_exists`**
   - Better naming consistency
   - No external usage found

4. **Refactor `_discover_plugins`**
   - Use the new `register_plugin` method
   - Simplify implementation

## Implementation Phases

### Phase 1: Core Plugin Manager Refactoring

**Task 1.1: Refactor register_plugin method**
- File: `src/dqx/plugins.py`
- Update method signature to accept class_name
- Add dynamic import using `importlib`
- Extract module and class names from the full path
- Instantiate the plugin class
- Get plugin name from metadata
- Add validation for ResultProcessor protocol
- Store plugin using metadata name as key

**Task 1.2: Add validation helper**
- File: `src/dqx/plugins.py`
- Create `_validate_plugin_class` method
- Check for callable `metadata` method
- Check for callable `process` method
- Verify metadata returns PluginMetadata instance
- Provide clear error messages

**Task 1.3: Update _discover_plugins**
- File: `src/dqx/plugins.py`
- Use `str(ep.value)` to get class path
- Call new `register_plugin` with class path
- Remove duplicate instantiation code
- Maintain same error handling pattern

### Phase 2: API Cleanup

**Task 2.1: Remove get_metadata method**
- File: `src/dqx/plugins.py`
- Delete the `get_metadata` method completely
- Update docstrings if needed

**Task 2.2: Rename plugin_name_exists**
- File: `src/dqx/plugins.py`
- Rename method to `plugin_exists`
- Update method docstring

**Task 2.3: Run type checking**
- Run `uv run mypy src/dqx/plugins.py`
- Fix any type errors that arise
- Ensure all type annotations are correct

### Phase 3: Update Tests

**Task 3.1: Update test_plugin_integration.py**
- File: `tests/test_plugin_integration.py`
- Update 4 `register_plugin` calls to use class names
- Move nested test classes to module level if needed
- Format: `manager.register_plugin("tests.test_plugin_integration.ClassName")`

**Task 3.2: Update test_plugin_public_api.py**
- File: `tests/test_plugin_public_api.py`
- Update 13 `register_plugin` calls to use class names
- Remove or update `test_get_metadata` test
- Update error messages in tests to match new implementation
- Move nested test classes to module level

**Task 3.3: Update test_enable_plugins.py**
- File: `tests/test_enable_plugins.py`
- Update 3 `register_plugin` calls to use class names
- Ensure test plugin classes are importable

### Phase 4: Update Examples and Documentation

**Task 4.1: Update plugin_demo.py**
- File: `examples/plugin_demo.py`
- Remove usage of `get_metadata()`
- Replace with direct plugin access if needed
- Update any documentation in the demo

**Task 4.2: Update test_plugin_manager.py**
- File: `tests/test_plugin_manager.py`
- Remove `test_get_metadata` test method
- Update any references to removed methods

**Task 4.3: Final validation**
- Run full test suite: `uv run pytest tests/`
- Run linting: `bin/run-hooks.sh`
- Ensure 100% test coverage maintained

## Example Usage

### Before
```python
# Manual registration
plugin = CustomPlugin()
manager.register_plugin("custom", plugin)

# Discovery
for ep in entry_points:
    plugin_class = ep.load()
    plugin = plugin_class()
    self._plugins[ep.name] = plugin
```

### After
```python
# Manual registration
manager.register_plugin("mypackage.plugins.CustomPlugin")

# Discovery
for ep in entry_points:
    class_path = str(ep.value)
    self.register_plugin(class_path)
```

## Testing Strategy

1. Run tests after each phase to ensure nothing breaks
2. Use TDD approach for new validation logic
3. Ensure all existing tests pass with modifications
4. Add new tests for edge cases:
   - Invalid module path
   - Class not found
   - Class doesn't implement protocol
   - Invalid metadata

## Rollback Plan

If issues arise:
1. Git revert to previous state
2. Each phase is atomic and can be reverted independently
3. Tests ensure no regression

## Success Criteria

- All tests pass
- No type errors from mypy
- Plugin registration works via class names
- Entry point discovery uses same mechanism
- Cleaner, more consistent API
