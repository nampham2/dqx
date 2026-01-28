# Multi-Plugin Registration Context for Implementation

This document provides background context for implementing multi-plugin registration in the DQX plugin system.

## DQX Architecture Overview

### Relevant Components

#### PluginManager (`src/dqx/plugins.py`, lines 120-287)

**Purpose**: Manages registration, lifecycle, and execution of result processor plugins

**Key attributes**:
- `_plugins: dict[str, PostProcessor]` - Registry of loaded plugins (name → instance)
- `_timeout_seconds: float` - Execution timeout for plugin processing

**Key methods**:
- `__init__()` - Initializes manager, registers built-in AuditPlugin
- `register_plugin(plugin)` - Current single-plugin registration (lines 163-186)
- `_register_from_class(class_name: str)` - Helper for string-based registration (lines 199-232)
- `_register_from_instance(plugin: PostProcessor)` - Helper for instance registration (lines 233-254)
- `get_plugins()` - Returns registered plugins dictionary
- `process_all(context)` - Executes all plugins with timeout

**How multi-plugin registration relates**:
- Will extend `register_plugin()` to accept `*args`
- Will reuse `_register_from_class()` and `_register_from_instance()` helpers unchanged
- Will maintain same error handling patterns
- Will add summary logging for batch operations

#### PostProcessor Protocol (`src/dqx/plugins.py`, lines 101-118)

**Purpose**: Protocol defining plugin interface

**Required methods**:
- `metadata() -> PluginMetadata` - Static method returning plugin info
- `process(context: PluginExecutionContext) -> None` - Process validation results

**How multi-plugin registration relates**:
- Each plugin in variadic args must satisfy this protocol
- `isinstance(plugin, PostProcessor)` check works at runtime
- Type hints use `PostProcessor` in overloads

#### Plugin Registration Flow

**Current flow** (single plugin):
```
register_plugin(plugin)
    ↓
isinstance check: str or PostProcessor?
    ↓
str → _register_from_class(plugin)
    ↓
    1. Parse class name ("pkg.module.Class")
    2. Import module
    3. Get class from module
    4. Instantiate class
    5. Call _register_from_instance(instance)

PostProcessor → _register_from_instance(plugin)
    ↓
    1. Check isinstance(plugin, PostProcessor)
    2. Call plugin.metadata()
    3. Validate metadata is PluginMetadata
    4. Store in self._plugins[name] = plugin
    5. Log registration
```

**New flow** (multi-plugin):
```
register_plugin(plugin1, plugin2, plugin3)
    ↓
Loop over plugins:
    For each plugin:
        isinstance check: str or PostProcessor?
            ↓
        str → _register_from_class(plugin)
        PostProcessor → _register_from_instance(plugin)
            ↓
        [FAIL-FAST: If error, stop loop, raise ValueError]
    ↓
Log summary: "Successfully registered N plugin(s)"
```

## Code Patterns to Follow

**IMPORTANT**: All patterns reference AGENTS.md standards.

### Pattern 1: Variadic Arguments

**When to use**: Functions accepting variable number of same-typed arguments

**Example from Python stdlib**:
```python
def print(*values, sep=' ', end='\n'):
    # Loop over values
    pass
```

**Reference**: Standard Python pattern

**Apply to register_plugin()**:
```python
def register_plugin(self, *plugins: str | PostProcessor) -> None:
    """Register one or more plugins."""
    if not plugins:
        return  # Empty call is no-op

    for plugin in plugins:
        # Process each plugin
        pass
```

### Pattern 2: Overload Decorators

**When to use**: Provide type hints for multiple function signatures

**Example from DQX** (`src/dqx/plugins.py`, lines 163-172):
```python
@overload
def register_plugin(self, plugin: str) -> None:
    """Register by class name."""
    ...

@overload
def register_plugin(self, plugin: PostProcessor) -> None:
    """Register by instance."""
    ...

def register_plugin(self, plugin: str | PostProcessor) -> None:
    """Implementation."""
    pass
```

**Reference**: AGENTS.md §type-hints (strict mode)

**Apply to multi-plugin**:
```python
@overload
def register_plugin(self) -> None:
    """Empty call."""
    ...

@overload
def register_plugin(self, plugin: str) -> None:
    """Single string."""
    ...

@overload
def register_plugin(self, plugin: PostProcessor) -> None:
    """Single instance."""
    ...

@overload
def register_plugin(self, *plugins: str | PostProcessor) -> None:
    """Multiple plugins."""
    ...

def register_plugin(self, *plugins: str | PostProcessor) -> None:
    """Implementation accepts all patterns."""
    pass
```

### Pattern 3: Required First Argument Pattern

**When to use**: Variadic functions that must receive at least one argument

**Example pattern**:
```python
def process_items(self, item: Item, *items: Item) -> None:
    all_items = (item,) + items
    for i in all_items:
        # Process item
        pass
```

**Why**: Enforces semantic correctness - calling `register_plugin()` with nothing doesn't make sense

**Reference**: Python conventions - `max(arg1, *args)` requires at least one argument

**Apply to register_plugin()**:
```python
def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
    # No need for empty check - signature enforces it
    all_plugins = (plugin,) + plugins
    # Continue with registration
```

**Type safety**:
- MyPy rejects: `register_plugin()` - Error: missing required argument
- MyPy accepts: `register_plugin(p1)`, `register_plugin(p1, p2)`

### Pattern 4: Fail-Fast Error Handling

**When to use**: Batch operations where partial failure should halt processing

**Example from DQX** (error handling pattern):
```python
def validate_plugins(self, plugins: list[Plugin]) -> None:
    for plugin in plugins:
        if not self._is_valid(plugin):
            raise ValueError(f"Invalid plugin: {plugin}")
        # If error raised, loop stops, remaining plugins not processed
```

**Reference**: DQX error handling philosophy (explicit failures)

**Apply to register_plugin()**:
```python
def register_plugin(self, *plugins: str | PostProcessor) -> None:
    for plugin in plugins:
        # Any ValueError from helpers propagates immediately
        if isinstance(plugin, str):
            self._register_from_class(plugin)  # May raise ValueError
        else:
            self._register_from_instance(plugin)  # May raise ValueError
    # Loop stops on first error, no rollback
```

### Pattern 5: Conditional Logging

**When to use**: Different log messages for single vs batch operations

**Example pattern**:
```python
def process_items(self, *items: Item) -> None:
    if len(items) == 1:
        # Single item: simple log
        logger.info(f"Processed item: {items[0].name}")
    else:
        # Multiple items: individual logs + summary
        for item in items:
            logger.info(f"Processed item: {item.name}")
        logger.info(f"Processed {len(items)} items")
```

**Reference**: DQX logging patterns

**Apply to register_plugin()**:
```python
def register_plugin(self, *plugins: str | PostProcessor) -> None:
    if not plugins:
        return  # No logging for empty

    if len(plugins) == 1:
        # Single: existing behavior (no summary)
        plugin = plugins[0]
        # ... register plugin (logs per-plugin message)
        return

    # Multiple: new behavior (per-plugin + summary)
    for plugin in plugins:
        # ... register plugin (logs per-plugin message)

    logger.info(f"Successfully registered {len(plugins)} plugin(s)")
```

## Code Standards Reference

**All code must follow AGENTS.md standards**:

### Import Order (AGENTS.md §import-order)
```python
from __future__ import annotations  # Always first

import logging
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

# Existing imports unchanged
```

### Type Hints (AGENTS.md §type-hints)
- Strict mode: All parameters and return types annotated
- Use `|` for unions: `str | PostProcessor`
- Variadic: `*plugins: str | PostProcessor`
- Return type: `-> None` (register_plugin doesn't return values)

### Docstrings (AGENTS.md §docstrings)
- Google style required
- Args section: Describe `*plugins` parameter
- Raises section: Document ValueError behavior
- Examples section: Show single, multiple, mixed usage

**Example docstring**:
```python
def register_plugin(self, *plugins: str | PostProcessor) -> None:
    """
    Register one or more plugins by class name or PostProcessor instance.

    Plugins can be mixed types. Registration proceeds sequentially and fails
    fast on the first error without rolling back previously registered plugins.

    Args:
        *plugins: Zero or more plugins. Each is either a fully qualified
            class name string or a PostProcessor instance.

    Raises:
        ValueError: If any plugin is invalid or fails to load. Previously
            registered plugins remain registered (no rollback).

    Examples:
        >>> # Single plugin (backward compatible)
        >>> manager.register_plugin("dqx.plugins.AuditPlugin")

        >>> # Multiple plugins
        >>> manager.register_plugin(
        ...     "com.example.Plugin1",
        ...     "com.example.Plugin2"
        ... )

        >>> # Mixed types
        >>> custom = MyCustomPlugin()
        >>> manager.register_plugin("pkg.Plugin", custom)
    """
```

### Error Messages (AGENTS.md §error-handling)
- Clear and actionable
- Include context (which plugin failed)
- Preserve existing error messages for backward compatibility

## Testing Patterns

**Reference**: AGENTS.md §testing-patterns

### Test Organization

**Pattern**: Group related tests in classes
```python
class TestMultiPluginRegistration:
    """Tests for multi-plugin registration feature."""

    def test_register_multiple_strings(self) -> None:
        """Test registering multiple plugins by class name."""
        # Test implementation

    def test_register_mixed_types(self) -> None:
        """Test registering mixed strings and instances."""
        # Test implementation
```

### Fixtures

**Use existing fixtures** from `tests/test_plugin_manager.py`:
- `plugin_manager` - Fresh PluginManager instance
- `valid_plugin` - ValidInstancePlugin instance
- `empty_context` - PluginExecutionContext for testing

**Pattern**:
```python
def test_multi_plugin_registration(
    self, plugin_manager: PluginManager, valid_plugin: ValidInstancePlugin
) -> None:
    """Test with fixtures."""
    plugin_manager.register_plugin(
        "dqx.plugins.AuditPlugin",
        valid_plugin
    )
    assert len(plugin_manager.get_plugins()) == 2
```

### Error Testing

**Pattern**: Use `pytest.raises` with match
```python
def test_invalid_plugin_in_batch(self) -> None:
    """Test error handling for invalid plugin in batch."""
    manager = PluginManager()
    manager.clear_plugins()

    with pytest.raises(ValueError, match="Cannot import module"):
        manager.register_plugin(
            "dqx.plugins.AuditPlugin",
            "invalid.module.Plugin"
        )

    # Verify partial state: first plugin registered
    assert "audit" in manager.get_plugins()
    assert len(manager.get_plugins()) == 1
```

### Logging Testing

**Pattern**: Use monkeypatch to capture logs
```python
def test_summary_log_for_multi_plugin(
    self, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test summary logging for multi-plugin registration."""
    log_messages: list[str] = []

    def capture_log(msg: str, *args: object) -> None:
        log_messages.append(msg)

    import logging
    logger = logging.getLogger("dqx.plugins")
    monkeypatch.setattr(logger, "info", capture_log)

    manager = PluginManager()
    manager.register_plugin(
        "dqx.plugins.AuditPlugin",
        "dqx.plugins.AuditPlugin"
    )

    # Verify summary log
    assert any("Successfully registered 2 plugin(s)" in msg for msg in log_messages)
```

## Common Pitfalls

### Pitfall 1: Modifying Single-Plugin Behavior

**Problem**: Accidentally changing existing behavior while adding multi-plugin support

**Solution**:
- Preserve exact single-plugin code path
- Use `if len(plugins) == 1:` branch for backward compatibility
- Test extensively with existing tests

**Correct approach**:
```python
def register_plugin(self, *plugins: str | PostProcessor) -> None:
    if not plugins:
        return

    # Single plugin: PRESERVE existing code path exactly
    if len(plugins) == 1:
        plugin = plugins[0]
        if isinstance(plugin, str):
            self._register_from_class(plugin)
        else:
            self._register_from_instance(plugin)
        return  # No summary log for single plugin

    # Multi-plugin: NEW code path
    for plugin in plugins:
        # ...
```

### Pitfall 2: Overload Order (Simplified Pattern)

**Problem**: MyPy errors if overloads are in wrong order

**Solution**: With simplified 2-overload pattern, order from most specific to most general
1. Single plugin (any type) `(plugin: str | PostProcessor)`
2. Multiple plugins (variadic) `(plugin: str | PostProcessor, *plugins: str | PostProcessor)`

**Correct order**:
```python
@overload
def register_plugin(self, plugin: str | PostProcessor) -> None: ...

@overload
def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None: ...

def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
    # Implementation requires at least one argument
```

**Key difference from 4-overload pattern**:
- No empty call overload (empty calls are FORBIDDEN)
- Single `str | PostProcessor` instead of separate str and PostProcessor overloads
- Simpler, less prone to ordering issues

### Pitfall 3: Not Testing Partial Registration State

**Problem**: Forgetting to verify state after error in middle of batch

**Solution**: Explicitly test partial registration
```python
def test_partial_registration_on_error(self) -> None:
    """Test that successful plugins remain registered after error."""
    manager = PluginManager()
    manager.clear_plugins()

    with pytest.raises(ValueError):
        manager.register_plugin(
            "dqx.plugins.AuditPlugin",  # Will succeed
            "invalid.Plugin",            # Will fail here
            "never.reached.Plugin"       # Never processed
        )

    # Verify partial state
    plugins = manager.get_plugins()
    assert "audit" in plugins          # First succeeded
    assert len(plugins) == 1           # Only one registered
```

### Pitfall 4: Circular Imports

**Problem**: DQX has complex dependencies

**Solution**: Use `TYPE_CHECKING` for type hints
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dqx.cache import CacheStats  # Only imported for type checking
```

**Reference**: AGENTS.md §type-hints

## Related PRs and Issues

### Similar Patterns in DQX

**Variadic functions**:
- None currently (this is the first)
- Inspired by Python stdlib patterns (`print`, `max`, etc.)

**Batch operations**:
- `process_all(context)` - Processes all registered plugins
- Similar pattern: iterate, fail-fast on errors

**Protocol-based validation**:
- `isinstance(plugin, PostProcessor)` - Runtime protocol checking
- Same pattern used throughout plugin system

## Implementation Strategy

### Step-by-Step Approach

1. **Update signature**: Change to `(plugin: str | PostProcessor, *plugins: str | PostProcessor)`
2. **Add overloads**: Two simplified overload decorators for type safety
3. **Remove empty handling**: Signature forbids empty calls (no check needed)
4. **Preserve single**: Keep existing behavior for single plugin
5. **Add loop**: Iterate for multiple plugins (combine first + rest)
6. **Add logging**: Summary log for multi-plugin only (2+)
7. **Test exhaustively**: All scenarios covered + negative test for empty

### Key Decisions

**Why require first argument**:
- Semantic correctness - registering "nothing" doesn't make sense
- MyPy enforces at compile time (better than runtime check)
- Consistent with Python stdlib (`max(arg1, *args)`)

**Why 2 overloads instead of 4**:
- Simpler code, easier to maintain
- Same type safety guarantees
- Better mypy error messages

**Why reuse helpers**:
- `_register_from_class()` and `_register_from_instance()` are well-tested
- No need to duplicate logic
- Easier to maintain

**Why fail-fast**:
- Simpler than transactional approach
- Consistent with DQX philosophy
- Errors should be fixed, not worked around

**Why no return value**:
- Current API returns `None`
- Changing return type would break backward compatibility
- Users can call `get_plugins()` if needed

### Testing Strategy

**Coverage requirements** (AGENTS.md §coverage-requirements):
- Target: 100% (no exceptions)
- Cover all branches: single, multiple (2, 3, 5 plugins)
- Cover all error paths: first, middle, last
- Cover all types: string, instance, mixed
- **Negative test**: Verify mypy rejects empty calls

**Test organization**:
- New test class: `TestMultiPluginRegistration`
- Backward compatibility tests
- Error handling tests
- Type checking tests (including empty call rejection)

## Documentation

After implementation, update:

### User-Facing Documentation

**`docs/plugin_system.md`**:
- Add "Batch Plugin Registration" section
- Show multi-plugin examples
- Document fail-fast behavior
- Provide migration guidance

**Inline docstrings** (AGENTS.md §docstrings):
- Already included in implementation (Google style)
- Examples show all usage patterns
- Clear Args and Raises sections

### Example Documentation Addition

```markdown
## Batch Plugin Registration

DQX supports registering multiple plugins in a single call:

```python
from dqx.plugins import PluginManager

manager = PluginManager()

# Register multiple plugins at once
manager.register_plugin(
    "com.example.Plugin1",
    "com.example.Plugin2",
    custom_plugin_instance
)
```

**Behavior**:
- Plugins are registered sequentially
- Registration fails fast on the first error
- Previously registered plugins remain registered (no rollback)
- Mixed types (strings and instances) are supported

**Error Handling**:
```python
try:
    manager.register_plugin(
        "valid.Plugin1",      # Succeeds
        "invalid.Plugin2",    # Fails here
        "never.reached.Plugin3"  # Never processed
    )
except ValueError as e:
    # valid.Plugin1 is registered
    # invalid.Plugin2 and never.reached.Plugin3 are not
    print(f"Registration failed: {e}")
```
```

## Quality Checklist

Before implementation completion:

### Code Quality
- [ ] All code follows AGENTS.md standards
- [ ] Type hints: strict mode, all parameters annotated
- [ ] Docstrings: Google style, comprehensive examples
- [ ] Imports: correct order, no circular dependencies
- [ ] Formatting: ruff format applied

### Testing
- [ ] 100% test coverage achieved
- [ ] All existing tests pass unchanged
- [ ] New tests cover all scenarios
- [ ] Error paths tested exhaustively
- [ ] Type checking validated

### Documentation
- [ ] Inline docstrings complete
- [ ] User-facing docs updated
- [ ] Examples tested and verified
- [ ] Migration guide clear

### Quality Gates (AGENTS.md §quality-gates)
- [ ] `uv run pytest` - All tests pass
- [ ] `uv run pytest --cov=src/dqx --cov-report=term-missing` - 100% coverage
- [ ] `uv run mypy src tests` - No type errors
- [ ] `uv run ruff format` - Code formatted
- [ ] `uv run ruff check --fix` - No linting errors
- [ ] `uv run pre-commit run --all-files` - All hooks pass

## Summary

This context document provides:
- **Architecture overview**: How PluginManager works
- **Code patterns**: Required first arg, simplified overloads, fail-fast
- **Standards reference**: All patterns link to AGENTS.md
- **Testing guidance**: Fixtures, error testing, logging, negative tests
- **Common pitfalls**: What to avoid (with 2-overload pattern)
- **Implementation strategy**: Step-by-step approach

Key principles:
- Maintain backward compatibility
- Reuse existing helpers
- Fail-fast on errors
- **Forbid empty calls** (semantic correctness)
- **Simplified 2 overloads** (not 4)
- Comprehensive testing (100% coverage)
- Clear documentation

**Design evolution**:
- Initial plan: 4 overloads with empty call support
- **Final design**: 2 overloads, empty calls forbidden
- **Rationale**: Simpler, more correct semantically, better mypy errors
