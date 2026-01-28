# Multi-Plugin Registration Technical Specification

## Problem Statement

The current `register_plugin()` method accepts only a single plugin (either a string or PostProcessor instance). Users who want to register multiple plugins must make separate method calls:

```python
manager = PluginManager()
manager.register_plugin("com.example.Plugin1")
manager.register_plugin("com.example.Plugin2")
manager.register_plugin(my_custom_plugin)
```

This is verbose and doesn't align with common Python patterns for batch operations. A more ergonomic API would allow registering multiple plugins in a single call:

```python
manager = PluginManager()
manager.register_plugin(
    "com.example.Plugin1",
    "com.example.Plugin2",
    my_custom_plugin
)
```

**Key requirements**:
1. Support variadic arguments: `register_plugin(*plugins)`
2. Accept mixed types: strings and PostProcessor instances
3. Maintain backward compatibility: single-plugin calls must work unchanged
4. Fail-fast error handling: stop on first error, no rollback
5. Preserve type safety with `@overload` decorators

## Architecture Decisions

### Decision 1: Variadic Args with Mixed Types

**Rationale**: Maximum flexibility while maintaining type safety.

**API Design**:
```python
# All valid call patterns
manager.register_plugin(plugin1)                          # Single plugin
manager.register_plugin(plugin1, plugin2)                 # Multiple plugins
manager.register_plugin("pkg.Plugin1", instance2)         # Mixed types
manager.register_plugin(*plugin_list)                     # Unpacking
```

**Type constraints**:
- Each argument must be either `str` or `PostProcessor`
- Empty calls `register_plugin()` are **FORBIDDEN** (caught by mypy)
- Minimum 1 argument (required), maximum unlimited

**Alternatives considered**:

1. **Separate `register_plugins()` method (plural)**
   - Rejected: Two methods doing the same thing is confusing
   - Users would need to remember which method to use
   - Duplicates implementation and tests

2. **Accept list/tuple instead of variadic args**
   - Rejected: Less Pythonic, requires wrapping in list
   - Compare: `register_plugin([p1, p2])` vs `register_plugin(p1, p2)`
   - Variadic args are standard Python for batch operations

3. **Only allow homogeneous types (all strings OR all instances)**
   - Rejected: Unnecessarily restrictive
   - Common use case: mix built-in (string) + custom (instance)

### Decision 2: Fail-Fast with No Rollback

**Rationale**: Simpler implementation, consistent with DQX error handling patterns.

**Behavior**:
```python
manager.register_plugin(valid_plugin1, invalid_plugin, valid_plugin2)
# Result:
# - valid_plugin1 is registered (succeeds)
# - invalid_plugin raises ValueError (stops here)
# - valid_plugin2 is NOT processed (never reached)
```

**Why fail-fast**:
- Plugin registration errors indicate code/configuration problems
- Errors should be fixed immediately, not worked around
- Partial registration state is visible for debugging
- Consistent with existing `register_plugin()` behavior

**Why no rollback**:
- Plugin registration has no side effects beyond memory state
- Rollback adds complexity without clear benefit
- Users can call `clear_plugins()` if needed
- No transactional guarantees required

**Error message quality**:
```python
# Clear error shows which plugin failed
ValueError: Failed to register plugin 'com.invalid.Plugin': Cannot import module com.invalid
# User knows: valid_plugin1 succeeded, this one failed, others not attempted
```

**Alternatives considered**:

1. **Validate all plugins before registering any**
   - Rejected: Requires instantiating all plugins twice
   - Performance cost for large plugin lists
   - Still doesn't guarantee atomicity (metadata could change between validation and registration)

2. **Register all valid plugins, skip invalid ones with warnings**
   - Rejected: Silent failures are dangerous
   - Users might not notice missing plugins
   - Inconsistent with Python's explicit error handling philosophy

3. **Transactional with automatic rollback**
   - Rejected: Over-engineered for this use case
   - Plugin registration is not a critical transaction
   - Adds complexity for minimal benefit

### Decision 3: Type Hints with @overload (Simplified 2-Overload Pattern)

**Rationale**: Preserve type safety with minimal overloads while maintaining IDE autocomplete.

**Simplified overload strategy** (2 overloads instead of 4):
```python
from typing import overload

# Overload 1: Single plugin (any type)
@overload
def register_plugin(self, plugin: str | PostProcessor) -> None:
    """Register a single plugin."""
    ...

# Overload 2: Multiple plugins (variadic)
@overload
def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
    """Register multiple plugins."""
    ...

# Implementation (requires at least one argument)
def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
    """Register one or more plugins.

    Args:
        plugin: First plugin (class name string or PostProcessor instance)
        *plugins: Additional plugins (optional)

    Raises:
        ValueError: If any plugin is invalid or fails to load

    Examples:
        manager.register_plugin("pkg.Plugin")
        manager.register_plugin(my_plugin)
        manager.register_plugin("pkg.P1", "pkg.P2", my_plugin)
    """
    all_plugins = (plugin,) + plugins
    for p in all_plugins:
        if isinstance(p, str):
            self._register_from_class(p)
        else:
            self._register_from_instance(p)
```

**Why this works**:
- MyPy checks calls against both overloads
- First overload matches single-argument calls (both `str` and `PostProcessor`)
- Second overload matches multi-argument calls (2+)
- Implementation signature requires at least one argument (forbids empty calls)
- Simpler than 4 overloads, same type safety guarantees

**Type checking behavior**:
```python
# ✅ Valid - matches overload 1
manager.register_plugin("pkg.Plugin")

# ✅ Valid - matches overload 1
manager.register_plugin(instance)

# ✅ Valid - matches overload 2
manager.register_plugin("pkg.P1", instance, "pkg.P2")

# ❌ Invalid - mypy rejects empty call (REQUIRED BEHAVIOR)
manager.register_plugin()
# Error: All overload variants of "register_plugin" require at least one argument

# ❌ Invalid - type error
manager.register_plugin(123)  # mypy error: incompatible type

# ❌ Invalid - type error
manager.register_plugin([plugin1, plugin2])  # mypy error: expected str | PostProcessor, got list
```

**Alternatives considered**:

1. **Four overloads (empty, single str, single instance, multiple)**
   - Rejected: Over-complicated, unnecessary distinction between str and instance
   - Empty call support is questionable (why register nothing?)
   - Simplified to 2 overloads with empty calls forbidden

2. **Single overload with variadic args**
   ```python
   @overload
   def register_plugin(self, *plugins: str | PostProcessor) -> None: ...

   def register_plugin(self, *plugins: str | PostProcessor) -> None: ...
   ```
   - Rejected: Would allow empty calls `register_plugin()`
   - Weaker type safety (mypy can't enforce "at least one argument")

3. **Union type without overloads**
   ```python
   def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
   ```
   - Rejected: No overloads means worse IDE hints
   - Single vs multiple not distinguished in autocomplete

4. **Type narrowing with TypeGuard**
   - Rejected: Unnecessary complexity
   - Overloads are standard Python for this pattern

### Decision 4: Logging Strategy

**Rationale**: Maintain per-plugin logging for debugging, add summary for multi-plugin calls.

**Logging behavior**:
```python
# Single plugin (existing behavior, unchanged)
manager.register_plugin("pkg.Plugin")
# Log: "Registered plugin: my_plugin (string)"

# Multiple plugins (new behavior)
manager.register_plugin("pkg.P1", instance, "pkg.P2")
# Log: "Registered plugin: p1 (string)"
# Log: "Registered plugin: p2 (instance)"
# Log: "Registered plugin: p3 (string)"
# Log: "Successfully registered 3 plugin(s)"
```

**Why log each plugin**:
- Useful for debugging which plugins loaded
- Consistent with existing behavior
- Shows order of registration

**Why add summary**:
- Confirms all plugins loaded successfully
- Provides count for verification
- Only logged for multi-plugin calls (2+ plugins)

**Log level**: `INFO` (existing level, unchanged)

**Alternatives considered**:

1. **Only log summary for multi-plugin, skip per-plugin logs**
   - Rejected: Loses debugging information
   - Harder to diagnose which plugins loaded

2. **Log summary for all calls (including single plugin)**
   - Rejected: Adds noise for common case
   - "Successfully registered 1 plugin(s)" is redundant

## API Design

### Method Signatures

```python
class PluginManager:
    @overload
    def register_plugin(self, plugin: str | PostProcessor) -> None:
        """Register a single plugin by class name or instance.

        Args:
            plugin: Fully qualified class name (e.g., "dqx.plugins.AuditPlugin")
                    or PostProcessor instance

        Raises:
            ValueError: If plugin cannot be imported or is invalid
        """
        ...

    @overload
    def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
        """Register multiple plugins (strings and/or instances).

        Args:
            plugin: First plugin to register
            *plugins: Additional plugins to register

        Raises:
            ValueError: If any plugin is invalid (stops on first error)
        """
        ...

    def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
        """
        Register one or more plugins by class name or PostProcessor instance.

        Plugins can be mixed types (strings and instances). Registration proceeds
        sequentially and fails fast on the first error without rolling back
        previously registered plugins.

        Args:
            plugin: First plugin (required) - either:
                - A fully qualified class name string (e.g., "dqx.plugins.AuditPlugin")
                - A PostProcessor instance
            *plugins: Additional plugins (optional), same types as `plugin`

        Raises:
            ValueError: If any plugin is invalid or fails to load. Previously
                registered plugins remain registered (no rollback).

        Examples:
            >>> # Single plugin (backward compatible)
            >>> manager.register_plugin("dqx.plugins.AuditPlugin")

            >>> # Single instance
            >>> manager.register_plugin(my_plugin)

            >>> # Multiple plugins
            >>> manager.register_plugin(
            ...     "com.example.Plugin1",
            ...     "com.example.Plugin2"
            ... )

            >>> # Mixed types
            >>> custom = MyCustomPlugin()
            >>> manager.register_plugin(
            ...     "com.example.BuiltIn",
            ...     custom,
            ...     "com.example.AnotherBuiltIn"
            ... )

            >>> # Empty call - FORBIDDEN (mypy error)
            >>> manager.register_plugin()  # Error: requires at least one argument
        """
        # Implementation in next section
```

### Integration Points

**Unchanged methods**:
- `__init__()` - Still calls `register_plugin()` for built-in plugins
- `unregister_plugin(name)` - No changes needed
- `clear_plugins()` - No changes needed
- `get_plugins()` - No changes needed
- `plugin_exists(name)` - No changes needed
- `process_all(context)` - No changes needed

**Internal methods** (existing, unchanged):
- `_register_from_class(class_name: str) -> None`
- `_register_from_instance(plugin: PostProcessor) -> None`

These helpers remain as-is; `register_plugin()` delegates to them.

## Implementation Approach

### Code Changes (Pseudo-code)

```python
def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
    """Register one or more plugins."""
    # Combine first plugin with optional additional plugins
    all_plugins = (plugin,) + plugins

    # Single plugin: existing behavior (no summary log)
    if len(all_plugins) == 1:
        p = all_plugins[0]
        if isinstance(p, str):
            self._register_from_class(p)
        else:
            self._register_from_instance(p)
        return

    # Multiple plugins: new behavior
    for p in all_plugins:
        if isinstance(p, str):
            self._register_from_class(p)
        else:
            self._register_from_instance(p)

    # Log summary (only for multi-plugin)
    logger.info(f"Successfully registered {len(all_plugins)} plugin(s)")
```

### Error Handling

**Current error behavior** (preserved):
```python
# _register_from_class() errors
- ValueError: Invalid class name format
- ValueError: Cannot import module
- ValueError: Module has no class
- ValueError: Plugin doesn't implement PostProcessor protocol
- ValueError: metadata() returns wrong type

# _register_from_instance() errors
- ValueError: Doesn't implement PostProcessor protocol
- ValueError: Failed to get metadata
- ValueError: metadata() must return PluginMetadata
```

**New behavior** (for multi-plugin):
```python
manager.register_plugin(valid1, invalid, valid2)
# Execution:
# 1. valid1: registers successfully, logs "Registered plugin: valid1"
# 2. invalid: raises ValueError with descriptive message
# 3. valid2: NEVER REACHED (fail-fast)
# 4. No summary log (error interrupted)

# Result: valid1 is registered, invalid and valid2 are not
```

## Backward Compatibility

### Guaranteed Compatibility

**All existing code continues to work unchanged**:

```python
# Single string registration
manager.register_plugin("dqx.plugins.AuditPlugin")  # ✅ Works

# Single instance registration
manager.register_plugin(my_plugin)  # ✅ Works

# Called in __init__
def __init__(self):
    self.register_plugin("dqx.plugins.AuditPlugin")  # ✅ Works
```

**Type checking**:
- Existing overloads preserved
- MyPy sees same signatures for single-plugin calls
- No type errors introduced

**Behavior**:
- Same error messages
- Same logging output (per-plugin logs, no summary for single plugin)
- Same exceptions raised
- Same delegation to `_register_from_class()` and `_register_from_instance()`

### Breaking Changes

**None.** This is a pure backward-compatible extension.

## Performance Considerations

### Current Performance
- Single plugin: O(1) - Direct delegation to helper method
- N separate calls: O(N) - N method calls, N plugin registrations

### New Performance
- Single plugin: O(1) - Same as before (no overhead)
- N plugins in one call: O(N) - Loop overhead negligible

**Conclusion**: No meaningful performance impact.

## Non-Goals

**Explicitly out of scope for this feature**:

1. **Atomic transactions with rollback**
   - Not implementing: Too complex, no clear benefit
   - Users can call `clear_plugins()` if needed

2. **Validation-then-register pattern**
   - Not implementing: Requires double instantiation
   - Fail-fast is simpler and sufficient

3. **Return list of registered plugin names**
   - Not implementing: Current API returns `None`
   - Adding return value would be breaking change
   - Users can call `get_plugins()` after registration

4. **Duplicate detection across call**
   - Not implementing: Current API allows overwriting
   - Same behavior for multi-plugin (last one wins)

5. **Parallel plugin loading**
   - Not implementing: Registration is fast (no I/O)
   - Sequential is simpler and sufficient

6. **Plugin dependency resolution**
   - Not implementing: Plugins are independent
   - Registration order controlled by caller

## Testing Strategy

### Test Coverage Areas

1. **Backward compatibility**
   - Single string plugin
   - Single instance plugin
   - Mixed string and instance (separate calls)

2. **Multi-plugin scenarios**
   - Multiple strings
   - Multiple instances
   - Mixed strings and instances
   - Single plugin via variadic (regression)

3. **Error handling**
   - First plugin invalid
   - Middle plugin invalid
   - Last plugin invalid
   - All plugins invalid
   - Partial registration state after error

4. **Type checking**
   - Valid overload matches (single str, single instance, multiple mixed)
   - Invalid type rejection (int, list, etc.)
   - **Empty call rejection** - mypy must reject `register_plugin()`
   - Mixed type acceptance

5. **Logging**
   - Per-plugin logs present
   - Summary log for multi-plugin
   - No summary for single plugin

6. **Edge cases**
   - Duplicate names (overwrite)
   - Very large number of plugins (stress test)
   - Unpacking syntax

### Integration with Existing Tests

All existing tests in `tests/test_plugin_manager.py` must pass unchanged. New tests will be added in a separate test class.

## Documentation Updates

### User-Facing Docs

**`docs/plugin_system.md`** (to be updated):
- Add examples of multi-plugin registration
- Document fail-fast behavior
- Show mixed type usage

### Inline Documentation

**Docstrings**:
- Update `register_plugin()` docstring with examples
- Keep existing docstrings for overloads
- Follow Google style (AGENTS.md §docstrings)

**Type hints**:
- All overloads properly typed
- Implementation signature accepts variadic args

## Migration Guide

### For Users

**Before**:
```python
manager = PluginManager()
manager.register_plugin("pkg.Plugin1")
manager.register_plugin("pkg.Plugin2")
manager.register_plugin(custom_plugin)
```

**After** (optional, not required):
```python
manager = PluginManager()
manager.register_plugin(
    "pkg.Plugin1",
    "pkg.Plugin2",
    custom_plugin
)
```

**No action required**: Existing code continues to work.

## Future Enhancements

**Possible future additions** (not in this spec):

1. **Bulk unregister**: `unregister_plugins(*names)`
2. **Return registered names**: `register_plugin(...) -> list[str]`
3. **Validation mode**: `register_plugin(..., validate_only=True)`
4. **Transactional mode**: `register_plugin(..., atomic=True)`

These can be added later without breaking changes.

## Summary

This specification defines a backward-compatible extension to `register_plugin()` that:
- Accepts variadic arguments for batch registration (minimum 1 plugin required)
- Supports mixed types (strings and instances)
- Fails fast on errors without rollback
- Maintains type safety with 2 simplified overload decorators
- **Forbids empty calls** - caught by mypy at type-checking phase
- Preserves all existing behavior for single-plugin calls
- Adds minimal logging overhead (summary only for multi-plugin)

**Key design decisions**:
- **2 overloads** instead of 4 (simpler, cleaner)
- **Empty calls forbidden** (requires at least one argument)
- **Fail-fast error handling** (no rollback, partial registration visible)
- **Per-plugin + summary logging** (summary only for 2+ plugins)

The implementation is straightforward, testable, and aligns with Python conventions for variadic operations while enforcing semantic correctness (must register at least one plugin).
