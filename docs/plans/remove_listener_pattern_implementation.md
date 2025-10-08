# Implementation Plan: Remove Listener Pattern and Assertion Chaining

## Executive Summary

This plan details the refactoring of DQX's assertion system to:
1. Remove the AssertListener protocol and listener pattern from AssertBuilder
2. Make AssertionNode immutable (remove all setter methods)
3. Remove assertion chaining capability (each assertion stands alone)
4. Keep the `on()` method but have it store values in AssertBuilder

## Current Architecture Problems

The current implementation has an overcomplicated design:
- **AssertionNode** has mutable setter methods that allow changing fields after creation
- **AssertBuilder** uses a listener pattern to notify nodes when configuration changes
- Assertion chaining creates multiple AssertionNode instances for a single expression
- The builder pattern is not properly implemented - it allows incomplete nodes to be created

## Proposed Solution

### Core Changes

1. **Make AssertionNode Immutable**
   - Remove `set_label()`, `set_severity()`, `set_validator()` methods
   - All fields must be provided at construction time
   - Once created, an AssertionNode cannot be modified

2. **Simplify AssertBuilder**
   - Remove the `listeners` field and `AssertListener` protocol
   - Store label/severity internally until assertion creation
   - Each assertion method creates exactly one AssertionNode

3. **Remove Assertion Chaining**
   - Methods like `is_gt()`, `is_eq()` will return `None` instead of `AssertBuilder`
   - Users must create separate assertions for each condition

## Detailed Task Breakdown

### Task 1: Write failing tests for new immutable AssertionNode behavior

**File**: `tests/test_api.py`

**What to do**:
Add tests to verify the new immutable behavior of AssertionNode and AssertBuilder.

**Example test to add**:
```python
def test_assertion_node_is_immutable():
    """AssertionNode should be immutable after creation."""
    # Test will be added in test_api.py to verify immutability through the API
    # We'll test that the builder creates complete nodes, not that setters don't exist
    pass
```

**How to test**: Run `uv run pytest tests/test_api.py::test_assertion_node_is_immutable -v`

**Expected**: Test should fail because setter methods still exist

### Task 2: Remove setter methods from AssertionNode

**File**: `src/dqx/graph/nodes.py`

**What to do**:
1. Find the `AssertionNode` class (around line 175)
2. Remove these three methods completely:
   - `set_label(self, label: str) -> None`
   - `set_severity(self, severity: SeverityLevel) -> None`
   - `set_validator(self, validator: SymbolicValidator) -> None`

**How to test**:
1. Run the test from Task 1: `uv run pytest tests/test_api.py::test_assertion_node_is_immutable -v`
2. It should now pass

**Commit**: `git add -p && git commit -m "refactor: remove setter methods from AssertionNode"`

### Task 3: Write failing tests for AssertBuilder without listeners

**File**: `tests/test_api.py`

**What to do**:
1. Write a test that verifies AssertBuilder no longer accepts listeners parameter
2. Write a test that verifies assertion methods return None (not AssertBuilder)
3. Write a test that verifies `on()` method stores values internally

**Example test**:
```python
def test_assert_builder_no_listeners():
    """AssertBuilder should not use listeners."""
    expr = sp.Symbol("x")

    # Should not accept listeners parameter
    with pytest.raises(TypeError):
        AssertBuilder(actual=expr, listeners=[], context=None)

    # Should work without listeners
    builder = AssertBuilder(actual=expr, context=None)
    assert builder is not None

def test_assertion_methods_return_none():
    """Assertion methods should not return AssertBuilder for chaining."""
    context = Context("test", InMemoryMetricDB())
    builder = context.assert_that(sp.Symbol("x"))

    # These should return None, not AssertBuilder
    result = builder.is_gt(0)
    assert result is None
```

**How to test**: Run `uv run pytest tests/test_api.py::test_assert_builder_no_listeners -v`

**Expected**: Tests should fail

### Task 4: Remove AssertListener protocol and update AssertBuilder

**File**: `src/dqx/api.py`

**What to do**:

1. Remove the `AssertListener` protocol (around line 50):
   ```python
   @runtime_checkable
   class AssertListener(Protocol):
       """Protocol for objects that listen to assertion configuration changes."""
       def set_label(self, label: str) -> None: ...
       def set_severity(self, severity: SeverityLevel) -> None: ...
       def set_validator(self, validator: SymbolicValidator) -> None: ...
   ```

2. Update `AssertBuilder.__init__` (around line 65):
   - Remove `listeners` parameter
   - Remove `self.listeners = listeners` assignment

3. Update `AssertBuilder.clone()` method:
   - Remove it entirely (no longer needed without chaining)

4. Update `AssertBuilder.on()` method (around line 95):
   - Remove the loop that notifies listeners
   - Just store the values internally

5. Update all assertion methods (`is_geq`, `is_gt`, `is_leq`, `is_lt`, `is_eq`, `is_negative`, `is_positive`):
   - Change return type from `AssertBuilder` to `None`
   - Instead of returning a new AssertBuilder, create the AssertionNode and return None

6. Update `_create_assertion_node()` method:
   - Create AssertionNode with all fields (label, severity, validator) at once
   - Don't create a new AssertBuilder for chaining

**Example for `is_gt()`:
```python
def is_gt(self, other: float, tol: float = functions.EPSILON) -> None:
    """Assert that the expression is greater than the given value."""
    validator = SymbolicValidator(f"> {other}", lambda x: functions.is_gt(x, other, tol))
    self._create_assertion_node(validator)
```

**How to test**:
1. Run `uv run mypy src/dqx/api.py` - should have no errors
2. Run `uv run pytest tests/test_api.py::test_assert_builder_no_listeners -v` - should pass

**Commit**: `git add -p && git commit -m "refactor: remove AssertListener protocol and update AssertBuilder"`

### Task 5: Update Context.assert_that() to not pass listeners

**File**: `src/dqx/api.py`

**What to do**:
1. Find `Context.assert_that()` method (around line 295)
2. Update the return statement to not pass listeners:
   ```python
   return AssertBuilder(actual=expr, context=self)
   ```

**How to test**: Run `uv run pytest tests/test_api.py -k "assert_that" -v`

**Commit**: `git add -p && git commit -m "refactor: update Context.assert_that to not use listeners"`

### Task 6: Fix all chained assertion tests

**File**: `tests/test_api.py`

**What to do**:
For each test that uses chained assertions, split them into separate assertions.

**Example transformation**:
```python
# OLD (chained):
ctx.assert_that(ratio).is_geq(0.95).is_leq(1.05)

# NEW (separate):
ctx.assert_that(ratio).is_geq(0.95)
ctx.assert_that(ratio).is_leq(1.05)
```

**Specific tests to fix**:
1. `test_check_datasets_with_multiple_datasets` - has 1 chained assertion
2. `test_advanced_symbolic_check` - has 4 chained assertions
3. `test_chained_assertions` - this entire test is about chaining, needs complete rewrite

For `test_chained_assertions`, transform it to test that separate assertions work correctly:
```python
def test_multiple_assertions_on_same_metric():
    """Test that multiple separate assertions can be made on the same metric."""
    # ... setup code ...

    # Multiple assertions on same metric (not chained)
    ctx.assert_that(metric).on(label="Greater than 40").is_gt(40)
    ctx.assert_that(metric).on(label="Less than 60").is_lt(60)

    # ... rest of test ...
```

**How to test**: Run `uv run pytest tests/test_api.py -v`

**Commit**: `git add -p && git commit -m "test: update tests to remove assertion chaining"`

### Task 7: Update README documentation

**File**: `README.md`

**What to do**:

1. Remove the section "### Assertion Chaining" (around line 185)

2. Update all examples that use chaining. Find and replace:
   ```markdown
   # OLD:
   ctx.assert_that(ratio).is_geq(0.95).is_leq(1.05)

   # NEW:
   ctx.assert_that(ratio).is_geq(0.95)
   ctx.assert_that(ratio).is_leq(1.05)
   ```

3. Update the features list - remove "fluent assertion chaining"

4. Update the Quick Start example if it uses chaining

5. Search for any other references to "chain" and update accordingly

**How to test**:
1. Visually review the README
2. Check that all code examples are valid Python

**Commit**: `git add README.md && git commit -m "docs: remove assertion chaining from documentation"`

### Task 8: Run full test suite and fix any remaining issues

**What to do**:
1. Run the full test suite: `uv run pytest -v`
2. Fix any failing tests that were missed
3. Run type checking: `uv run mypy src/`
4. Run linting: `uv run ruff check src/ tests/`
**Expected**: All tests should pass, type checking should pass

**Commit**: `git add -p && git commit -m "fix: resolve remaining test failures after removing chaining"`

### Task 9: Update the Recent Improvements section

**File**: `README.md`

**What to do**:
Add a new entry to the "Recent Improvements" section documenting this change:

```markdown
### v0.4.0 (Architecture Simplification)
- 🚨 **Breaking:** Removed assertion chaining - each assertion now stands alone
- 🚨 **Breaking:** Made AssertionNode immutable - all fields must be provided at construction
- ✅ Removed complex listener pattern in favor of simpler builder pattern
- ✅ Simplified AssertBuilder implementation
- ✅ Improved code clarity and maintainability
```

**Commit**: `git add README.md && git commit -m "docs: document v0.4.0 breaking changes"`

## Testing Strategy

For each task:
1. Write the test FIRST (TDD) - only in `tests/test_api.py`
2. Run the test to see it fail
3. Implement the minimal code to make it pass
4. Run the test again to confirm it passes
5. Refactor if needed while keeping tests green
6. Commit frequently

Note: We're only testing the new behavior through the public API in `test_api.py`, not creating separate unit tests for internal components.

## Validation Checklist

Before considering the implementation complete:

- [ ] All setter methods removed from AssertionNode
- [ ] AssertListener protocol completely removed
- [ ] AssertBuilder no longer uses listeners
- [ ] Assertion methods return None (not AssertBuilder)
- [ ] All tests pass
- [ ] Type checking passes (`uv run mypy src/`)
- [ ] Linting passes (`uv run ruff check src/ tests/`)
- [ ] README is updated with new examples
- [ ] Breaking changes are documented

## Migration Guide for Users

Users will need to update their code:

```python
# Old style (chained):
ctx.assert_that(metric).on(label="Range check").is_geq(0).is_leq(100)

# New style (separate assertions):
ctx.assert_that(metric).on(label="Minimum check").is_geq(0)
ctx.assert_that(metric).on(label="Maximum check").is_leq(100)
```

## Benefits of This Change

1. **Simpler Mental Model**: No hidden listeners or complex state management
2. **True Immutability**: AssertionNodes cannot be modified after creation
3. **Clearer Test Output**: Each assertion is independent with its own label
4. **Easier Debugging**: No need to trace through chained calls
5. **Better Error Messages**: Each assertion can have specific error context
