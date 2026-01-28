# Multi-Plugin Registration Implementation Guide

## Overview

Extend `register_plugin()` to accept variadic arguments, allowing registration of multiple plugins in a single call. This is a backward-compatible change that maintains all existing behavior while adding ergonomic batch registration.

**Implementation strategy**: Minimal changes to existing code, maximum reuse of helper methods.

## Prerequisites

### Files to Read Before Starting

- `src/dqx/plugins.py` - Lines 120-255 (PluginManager class)
  - Current `register_plugin()` implementation (lines 163-186)
  - Helper methods `_register_from_class()` and `_register_from_instance()` (lines 199-254)
- `tests/test_plugin_manager.py` - Lines 114-220 (existing registration tests)
- `tests/test_plugin_type_checking.py` - Lines 44-60 (overload type checking)

### Related Components

**`PluginManager` class** (`src/dqx/plugins.py`):
- `__init__()` - Line 123: Calls `register_plugin()` for built-in plugins
- `register_plugin()` - Lines 163-186: Current implementation with @overload
- `_register_from_class()` - Lines 199-232: Helper for string registration
- `_register_from_instance()` - Lines 233-254: Helper for instance registration

**Test patterns**:
- Test class organization: `class TestPluginManager:`
- Fixture usage: `plugin_manager` fixture
- Error testing: `pytest.raises(ValueError, match="...")`
- Type checking: `tests/test_plugin_type_checking.py`

## Phase Breakdown

### Phase 1: Update Method Signature and Add Overloads

**Goal**: Modify `register_plugin()` signature to accept variadic args with simplified 2-overload pattern (forbid empty calls).

**Duration estimate**: 1 hour

**Files to modify**:
- `src/dqx/plugins.py` - Lines 163-186 (method signature and overloads)

**Tests to write**:
```python
class TestMultiPluginRegistration:
    def test_register_single_string_backward_compat(self): ...
    def test_register_single_instance_backward_compat(self): ...
    def test_register_two_strings(self): ...
    def test_register_two_instances(self): ...
    def test_register_mixed_string_and_instance(self): ...
    def test_register_three_mixed_types(self): ...
```

**Implementation notes**:
- **Simplify to 2 overloads** (not 4):
  - Overload 1: Single plugin (any type)
  - Overload 2: Multiple plugins (variadic)
- **Implementation requires first argument** (forbids empty calls)
- **Remove empty call handling** - not needed with new signature
- Keep single-plugin path unchanged (preserve exact behavior)
- Add loop for multi-plugin case
- Delegate to existing `_register_from_class()` and `_register_from_instance()`

**Code structure**:
```python
# src/dqx/plugins.py (lines 163-186)

@overload
def register_plugin(self, plugin: str | PostProcessor) -> None:
    """Register a single plugin."""
    ...

@overload
def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
    """Register multiple plugins."""
    ...

def register_plugin(self, plugin: str | PostProcessor, *plugins: str | PostProcessor) -> None:
    """
    Register one or more plugins by class name or PostProcessor instance.

    Plugins can be mixed types. Registration proceeds sequentially and fails
    fast on the first error without rolling back previously registered plugins.

    Args:
        plugin: First plugin (required) - class name string or PostProcessor instance
        *plugins: Additional plugins (optional), same types as plugin

    Raises:
        ValueError: If any plugin is invalid. Previously registered plugins
            remain registered (no rollback).

    Examples:
        >>> # Single plugin (backward compatible)
        >>> manager.register_plugin("dqx.plugins.AuditPlugin")

        >>> # Multiple plugins
        >>> manager.register_plugin(
        ...     "com.example.Plugin1",
        ...     "com.example.Plugin2"
        ... )

        >>> # Mixed types
        >>> manager.register_plugin("pkg.Plugin", custom_instance)
    """
    # Combine first plugin with optional additional plugins
    all_plugins = (plugin,) + plugins

    # Single plugin: preserve existing behavior exactly
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

    # Log summary for multi-plugin calls
    logger.info(f"Successfully registered {len(all_plugins)} plugin(s)")
```

**Success criteria**:
- [ ] Two @overload decorators present (single, multiple)
- [ ] Implementation requires first argument `plugin: str | PostProcessor`
- [ ] Implementation accepts optional `*plugins: str | PostProcessor`
- [ ] No empty call handling (signature forbids it)
- [ ] Single plugin uses existing code path (no behavioral changes)
- [ ] Multi-plugin loops and delegates to helpers
- [ ] Summary log only for multi-plugin calls (2+)
- [ ] All phase tests passing
- [ ] Coverage: 100% for new code
- [ ] Pre-commit hooks: passing

**Commit message**: `feat(plugins): support variadic args in register_plugin()`

---

### Phase 2: Test Backward Compatibility

**Goal**: Verify all existing single-plugin behavior remains unchanged.

**Duration estimate**: 1 hour

**Files to modify**:
- `tests/test_plugin_manager.py` - Add tests to verify backward compatibility

**Tests to write**:
```python
class TestMultiPluginBackwardCompatibility:
    def test_single_string_registration_unchanged(self): ...
    def test_single_instance_registration_unchanged(self): ...
    def test_init_still_registers_audit_plugin(self): ...
    def test_logging_format_unchanged_for_single(self): ...
    def test_error_messages_unchanged_for_single(self): ...
```

**Implementation notes**:
- Run existing test suite: All tests must pass without modification
- Add explicit tests verifying behavior hasn't changed
- Test logging output format (no summary for single plugin)
- Verify error messages match existing format
- Check that `__init__()` call works correctly

**Success criteria**:
- [ ] All existing tests pass without modification
- [ ] Single string registration behavior identical
- [ ] Single instance registration behavior identical
- [ ] Logging output unchanged for single plugin
- [ ] Error messages unchanged
- [ ] Coverage: 100%
- [ ] Pre-commit hooks: passing

**Commit message**: `test(plugins): verify backward compatibility for register_plugin()`

---

### Phase 3: Test Multi-Plugin Scenarios

**Goal**: Comprehensive testing of new multi-plugin functionality.

**Duration estimate**: 2 hours

**Files to modify**:
- `tests/test_plugin_manager.py` - Add multi-plugin test class

**Tests to write**:
```python
class TestMultiPluginRegistration:
    def test_register_multiple_strings(self): ...
    def test_register_multiple_instances(self): ...
    def test_register_mixed_strings_and_instances(self): ...
    def test_register_three_plugins(self): ...
    def test_register_five_plugins(self): ...
    def test_unpack_list_of_plugins(self): ...
    def test_unpack_tuple_of_plugins(self): ...
    def test_summary_log_for_multi_plugin(self): ...
    def test_no_summary_log_for_single_plugin(self): ...
    def test_duplicate_names_last_wins(self): ...
```

**Implementation notes**:
- Test 2, 3, 5 plugin combinations
- Test all-strings, all-instances, and mixed
- Use monkeypatch to verify logging behavior
- Verify summary log format: "Successfully registered N plugin(s)"
- Test unpacking: `manager.register_plugin(*plugin_list)`
- Verify plugins are registered in order
- Check duplicate handling (should overwrite like current behavior)

**Success criteria**:
- [ ] All multi-plugin scenarios covered
- [ ] Logging verified (per-plugin + summary)
- [ ] Order verification tests
- [ ] Unpacking syntax works
- [ ] Duplicate handling consistent with existing behavior
- [ ] Coverage: 100%
- [ ] Pre-commit hooks: passing

**Commit message**: `test(plugins): add comprehensive multi-plugin registration tests`

---

### Phase 4: Test Error Handling and Fail-Fast

**Goal**: Verify fail-fast behavior and partial registration state.

**Duration estimate**: 1.5 hours

**Files to modify**:
- `tests/test_plugin_manager.py` - Add error handling tests

**Tests to write**:
```python
class TestMultiPluginErrorHandling:
    def test_first_plugin_invalid_stops_immediately(self): ...
    def test_middle_plugin_invalid_stops_at_failure(self): ...
    def test_last_plugin_invalid_previous_registered(self): ...
    def test_partial_registration_state_after_error(self): ...
    def test_invalid_class_name_in_multi_plugin(self): ...
    def test_invalid_instance_in_multi_plugin(self): ...
    def test_import_error_in_multi_plugin(self): ...
    def test_no_summary_log_on_error(self): ...
    def test_error_message_identifies_failed_plugin(self): ...
```

**Implementation notes**:
- Test error at each position (first, middle, last)
- Verify partial state: plugins before error are registered
- Verify plugins after error are NOT registered
- Check error messages are clear and identify which plugin failed
- Verify no summary log when error occurs
- Test all error types: ImportError, ValueError, etc.

**Success criteria**:
- [ ] Fail-fast behavior verified at all positions
- [ ] Partial registration state correct after error
- [ ] Error messages clear and helpful
- [ ] No summary log on error
- [ ] All error types tested
- [ ] Coverage: 100%
- [ ] Pre-commit hooks: passing

**Commit message**: `test(plugins): verify fail-fast and error handling for multi-plugin`

---

### Phase 5: Type Checking Validation

**Goal**: Verify type hints and overloads work correctly with MyPy.

**Duration estimate**: 1 hour

**Files to modify**:
- `tests/test_plugin_type_checking.py` - Add overload validation tests

**Tests to write**:
```python
class TestMultiPluginTypeChecking:
    def test_overload_single_string(self): ...
    def test_overload_single_instance(self): ...
    def test_overload_multiple_mixed(self): ...
    def test_invalid_type_rejected(self): ...
    def test_list_argument_rejected(self): ...
    def test_mixed_types_accepted(self): ...
    def test_empty_call_rejected_by_mypy(self): ...  # NEW: Verify empty calls forbidden
```

**Implementation notes**:
- Verify MyPy accepts valid calls
- **Verify MyPy REJECTS empty calls** `register_plugin()`
- Verify MyPy rejects invalid calls (lists, integers, etc.)
- Test IDE autocomplete suggestions (manual verification)
- Check overload resolution for each pattern
- Verify type inference for return value (None)

**Success criteria**:
- [ ] MyPy passes on all test files
- [ ] Overloads match all usage patterns
- [ ] Invalid types rejected by type checker
- [ ] IDE autocomplete works correctly
- [ ] Coverage: 100%
- [ ] Pre-commit hooks: passing (mypy must pass)

**Commit message**: `test(plugins): validate type checking for multi-plugin overloads`

---

### Phase 6: Documentation and Examples

**Goal**: Update documentation with examples and usage guidance.

**Duration estimate**: 1 hour

**Files to modify**:
- `docs/plugin_system.md` - Add multi-plugin examples
- `src/dqx/plugins.py` - Docstring already updated in Phase 1

**Documentation to add**:
- Basic multi-plugin example
- Mixed type example
- Fail-fast behavior explanation
- Migration guide (optional upgrade)
- Common patterns and best practices

**Implementation notes**:
- Add section "Batch Plugin Registration" to plugin_system.md
- Include code examples with output
- Document fail-fast behavior clearly
- Show unpacking pattern
- Add troubleshooting section for errors

**Success criteria**:
- [ ] Documentation updated with examples
- [ ] Fail-fast behavior documented
- [ ] Migration guide clear
- [ ] Examples tested and verified
- [ ] Pre-commit hooks: passing

**Commit message**: `docs(plugins): add multi-plugin registration examples and guide`

---

## Phase Dependencies

**Sequential execution required**:
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
```

**Why sequential**:
- Phase 1 changes the API (all other phases depend on this)
- Phase 2 verifies no regressions (must pass before new features)
- Phases 3-4 can run in parallel (both test new functionality)
- Phase 5 validates types (depends on implementation)
- Phase 6 documents completed feature (depends on all tests passing)

**Potential parallelization**:
- Phases 3 and 4 could be done in parallel (both test new functionality)
- Phase 5 could overlap with Phase 4 (type checking independent of runtime tests)

## Rollback Strategy

### If Issues Arise

**During Phase 1-2** (before multi-plugin tests):
- Revert commit from Phase 1
- All existing tests still pass
- No user-facing changes

**During Phase 3-5** (multi-plugin testing):
- Fix issues in implementation (Phase 1)
- Re-run affected test phases
- Existing functionality unaffected

**After completion**:
- Backward compatibility guarantees no rollback needed
- Single-plugin behavior unchanged
- Multi-plugin is optional feature

### Revert Command

```bash
# If needed (unlikely)
git revert <commit-hash>
```

## Estimated Total Time

**Phase durations**:
- Phase 1: 1 hour (implementation)
- Phase 2: 1 hour (backward compatibility)
- Phase 3: 2 hours (multi-plugin tests)
- Phase 4: 1.5 hours (error handling)
- Phase 5: 1 hour (type checking)
- Phase 6: 1 hour (documentation)

**Total**: ~7.5 hours

**Buffer for issues**: +1.5 hours

**Total with buffer**: ~9 hours (approximately 1 day)

## Quality Checklist

### After Each Phase

- [ ] All new tests pass
- [ ] All existing tests still pass
- [ ] Coverage: 100% (run `uv run pytest --cov=src/dqx --cov-report=term-missing`)
- [ ] MyPy passes (run `uv run mypy src tests`)
- [ ] Ruff format applied (run `uv run ruff format`)
- [ ] Ruff linting passes (run `uv run ruff check --fix`)
- [ ] Pre-commit hooks pass (run `uv run pre-commit run --all-files`)
- [ ] Commit message follows convention (see AGENTS.md §commit-conventions)

### Before Final PR

- [ ] All 6 phases completed
- [ ] All tests pass (110+ existing + 30+ new)
- [ ] 100% coverage maintained
- [ ] Documentation updated
- [ ] Type checking passes
- [ ] Pre-commit hooks pass
- [ ] No linting errors
- [ ] Commit history clean (6 atomic commits)

## Common Issues and Solutions

### Issue: MyPy Complains About Overload Order

**Symptom**: `error: Overloaded function signature does not match implementation`

**Solution**: Ensure overloads are ordered from most specific to least specific:
1. Single plugin (any type) - Overload 1
2. Multiple plugins (variadic) - Overload 2

**Note**: With simplified 2-overload pattern, order issues are less likely.

### Issue: Tests Fail Due to Logging Changes

**Symptom**: Tests checking log output fail

**Solution**: Update tests to expect summary log only for multi-plugin calls:
- Single plugin: No summary log (existing behavior)
- Multiple plugins: Summary log present (new behavior)

### Issue: Coverage Drops Below 100%

**Symptom**: `coverage report` shows uncovered lines

**Solution**:
1. Identify uncovered lines with `--cov-report=term-missing`
2. Add tests for those specific paths
3. Consider `# pragma: no cover` for defensive code (use sparingly)

### Issue: Existing Tests Break

**Symptom**: Tests that worked before now fail

**Solution**:
- Check single-plugin code path (should be unchanged)
- Verify no accidental behavior changes
- May need to adjust test fixtures or mocks

## Testing Strategy Summary

### Test Organization

```
tests/test_plugin_manager.py
├── TestMultiPluginRegistration (Phase 3)
│   ├── Multi-plugin scenarios
│   └── Logging verification
├── TestMultiPluginBackwardCompatibility (Phase 2)
│   ├── Single plugin unchanged
│   └── Error message preservation
├── TestMultiPluginErrorHandling (Phase 4)
│   ├── Fail-fast behavior
│   └── Partial registration state
└── Existing test classes (unchanged)

tests/test_plugin_type_checking.py
└── TestMultiPluginTypeChecking (Phase 5)
    ├── Overload validation
    └── Invalid type rejection
```

### Coverage Requirements

**Target**: 100% coverage (no exceptions)

**Key areas**:
- Single plugin path (unchanged, backward compatible)
- Multi-plugin loop (new functionality)
- Error handling at each position (fail-fast)
- Type checking for both overloads
- Mypy rejection of empty calls (negative test)
- Logging branches (single vs multi-plugin)

### Test Count Estimate

**New tests**: ~28-32 tests
**Existing tests**: 110+ tests (all must still pass)
**Total test suite**: ~138-142 tests

## Final Verification

### Manual Testing Checklist

Before marking complete:

```python
# Test 1: Empty call - SHOULD ERROR (mypy rejects, runtime TypeError)
manager = PluginManager()
try:
    manager.register_plugin()  # TypeError: missing required argument 'plugin'
except TypeError:
    print("Empty call correctly rejected!")

# Test 2: Single string (existing behavior)
manager.register_plugin("dqx.plugins.AuditPlugin")

# Test 3: Single instance (existing behavior)
manager.register_plugin(ValidInstancePlugin())

# Test 4: Multiple plugins
manager.register_plugin(
    "dqx.plugins.AuditPlugin",
    ValidInstancePlugin(),
    "tests.test_plugin_manager.ValidInstancePlugin"
)

# Test 5: Error handling
try:
    manager.register_plugin(
        "valid.Plugin",
        "invalid.NonExistent",
        "never.Reached"
    )
except ValueError as e:
    print(f"Expected error: {e}")
    # Verify "valid.Plugin" is registered
    # Verify "never.Reached" is NOT registered
```

### Pre-PR Checklist

- [ ] All 6 phases completed successfully
- [ ] Manual testing completed
- [ ] Documentation reviewed and accurate
- [ ] All commits have proper messages
- [ ] No unnecessary file changes
- [ ] Ready for code review
