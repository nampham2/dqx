# Implementation Plan: Mandatory Assertion Names

## Overview

This plan implements mandatory naming for assertions in DQX by introducing a two-stage assertion building process:
1. **AssertionDraft** - Initial stage that only allows calling `where()` with a required name
2. **AssertionReady** - Final stage with all assertion methods available

This is a **breaking change** that will make assertion names required for all assertions.

## Background for Engineers

### What are Assertions in DQX?

DQX is a data quality framework where users write "checks" that contain "assertions" about their data:

```python
@check(name="Price validation")
def validate_prices(mp: MetricProvider, ctx: Context) -> None:
    # This creates an assertion that average price should be positive
    ctx.assert_that(mp.average("price")).is_positive()
```

### Current Problem

Assertions can currently be created without names, making debugging and reporting difficult:
```python
# Current (bad) - no description of what this assertion validates
ctx.assert_that(mp.average("price")).is_positive()

# Current (good) - clear description
ctx.assert_that(mp.average("price")).where(name="Average price must be positive").is_positive()
```

### Solution

Force users to always provide a name by splitting the assertion API into two stages.

### Important Notes

- **No backward compatibility** - This is a breaking change
- **No migration tools** - All existing tests will be updated manually after implementation
- The `where()` method will require a `name` parameter - empty calls are not supported

## Implementation Tasks

### Task 1: Create AssertionDraft Class

**Goal**: Create the first stage of assertion building that only exposes `where()` method.

**Files to modify**:
- `src/dqx/api.py`

**What to implement**:
```python
class AssertionDraft:
    """
    Initial assertion builder that requires a name before making assertions.

    This is the first stage of assertion building. You must call where()
    with a name before you can make any assertions.

    Example:
        draft = ctx.assert_that(mp.average("price"))
        ready = draft.where(name="Price is positive")
        ready.is_positive()
    """

    def __init__(self, actual: sp.Expr, context: Context | None = None) -> None:
        """
        Initialize assertion draft.

        Args:
            actual: The symbolic expression to evaluate
            context: The Context instance (needed to create assertion nodes)
        """
        self._actual = actual
        self._context = context

    def where(self, *, name: str, severity: SeverityLevel | None = None) -> AssertionReady:
        """
        Provide a descriptive name for this assertion.

        Args:
            name: Required description of what this assertion validates
            severity: Optional severity level (P0, P1, P2, P3)

        Returns:
            AssertionReady instance with all assertion methods available

        Raises:
            ValueError: If name is empty or too long
        """
        if not name or not name.strip():
            raise ValueError("Assertion name cannot be empty")
        if len(name) > 255:
            raise ValueError("Assertion name is too long (max 255 characters)")

        return AssertionReady(
            actual=self._actual,
            name=name.strip(),
            severity=severity,
            context=self._context
        )
```

**How to test**:
- Run: `uv run pytest tests/test_api.py::test_assertion_draft_creation -xvs`
- Test should verify AssertionDraft can be created and has only `where()` method

**Test to write** (in `tests/test_api.py`):
```python
def test_assertion_draft_creation() -> None:
    """AssertionDraft should only expose where() method."""
    expr = sp.Symbol("x")
    draft = AssertionDraft(actual=expr, context=None)

    # Should have where method
    assert hasattr(draft, "where")

    # Should NOT have assertion methods
    assert not hasattr(draft, "is_gt")
    assert not hasattr(draft, "is_eq")
    assert not hasattr(draft, "is_positive")
```

**Before starting**: Run `uv run mypy src/dqx/api.py` to ensure no existing type errors.

### Task 2: Create AssertionReady Class

**Goal**: Create the second stage that has all assertion methods.

**Files to modify**:
- `src/dqx/api.py`

**What to implement**:
```python
class AssertionReady:
    """
    Named assertion ready to perform validations.

    This assertion has been properly named and can now use any of the
    validation methods like is_gt(), is_eq(), etc.
    """

    def __init__(
        self,
        actual: sp.Expr,
        name: str,
        severity: SeverityLevel | None = None,
        context: Context | None = None
    ) -> None:
        """
        Initialize ready assertion.

        Args:
            actual: The symbolic expression to evaluate
            name: Required description of the assertion
            severity: Optional severity level
            context: The Context instance
        """
        self._actual = actual
        self._name = name
        self._severity = severity
        self._context = context

    def is_geq(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is greater than or equal to the given value."""
        validator = SymbolicValidator(f"≥ {other}", lambda x: functions.is_geq(x, other, tol))
        self._create_assertion_node(validator)

    def is_gt(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is greater than the given value."""
        validator = SymbolicValidator(f"> {other}", lambda x: functions.is_gt(x, other, tol))
        self._create_assertion_node(validator)

    def is_leq(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is less than or equal to the given value."""
        validator = SymbolicValidator(f"≤ {other}", lambda x: functions.is_leq(x, other, tol))
        self._create_assertion_node(validator)

    def is_lt(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is less than the given value."""
        validator = SymbolicValidator(f"< {other}", lambda x: functions.is_lt(x, other, tol))
        self._create_assertion_node(validator)

    def is_eq(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression equals the given value within tolerance."""
        validator = SymbolicValidator(f"= {other}", lambda x: functions.is_eq(x, other, tol))
        self._create_assertion_node(validator)

    def is_negative(self, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is negative."""
        validator = SymbolicValidator("< 0", lambda x: functions.is_negative(x, tol))
        self._create_assertion_node(validator)

    def is_positive(self, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is positive."""
        validator = SymbolicValidator("> 0", lambda x: functions.is_positive(x, tol))
        self._create_assertion_node(validator)

    def _create_assertion_node(self, validator: SymbolicValidator) -> None:
        """Create a new assertion node and attach it to the current check."""
        if self._context is None:
            return

        current = self._context.current_check
        if not current:
            raise DQXError(
                "Cannot create assertion outside of check context. "
                "Assertions must be created within a @check decorated function."
            )

        # Create assertion node with all fields
        node = self._context.create_assertion(
            actual=self._actual,
            name=self._name,  # Always has a name now!
            severity=self._severity,
            validator=validator
        )

        # Attach to the current check node
        current.add_child(node)
```

**Note**: Copy the implementation of `_create_assertion_node` from the existing `AssertBuilder` class.

**Test to write**:
```python
def test_assertion_ready_has_all_methods() -> None:
    """AssertionReady should have all assertion methods."""
    expr = sp.Symbol("x")
    ready = AssertionReady(actual=expr, name="Test assertion", context=None)

    # Should have all assertion methods
    assert hasattr(ready, "is_gt")
    assert hasattr(ready, "is_geq")
    assert hasattr(ready, "is_lt")
    assert hasattr(ready, "is_leq")
    assert hasattr(ready, "is_eq")
    assert hasattr(ready, "is_positive")
    assert hasattr(ready, "is_negative")

    # Should NOT have where method
    assert not hasattr(ready, "where")
```

### Task 3: Update Context.assert_that Method

**Goal**: Change `assert_that` to return `AssertionDraft` instead of `AssertBuilder`.

**Files to modify**:
- `src/dqx/api.py`

**What to change**:
```python
# In the Context class, replace the assert_that method:
def assert_that(self, expr: sp.Expr) -> AssertionDraft:
    """
    Create an assertion draft for the given expression.

    You must provide a name using where() before making assertions:

    Example:
        ctx.assert_that(mp.average("price"))
           .where(name="Average price is positive")
           .is_positive()

    Args:
        expr: Symbolic expression to assert on

    Returns:
        AssertionDraft that requires where() to be called
    """
    return AssertionDraft(actual=expr, context=self)
```

**Test to write**:
```python
def test_context_assert_that_returns_draft() -> None:
    """Context.assert_that should return AssertionDraft."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    draft = context.assert_that(sp.Symbol("x"))
    assert isinstance(draft, AssertionDraft)
```

### Task 4: Remove Old AssertBuilder Class

**Goal**: Delete the old `AssertBuilder` class since we're not maintaining backward compatibility.

**Files to modify**:
- `src/dqx/api.py`

**What to do**:
1. Delete the entire `AssertBuilder` class
2. Remove any imports or references to `AssertBuilder`

**How to verify**:
- Run `uv run ruff check src/dqx/api.py` to ensure no unused imports
- Run `uv run mypy src/dqx/api.py` to ensure no type errors

### Task 5: Update All Existing Tests

**Goal**: Update all tests to use the new assertion pattern.

**Files to check and update**:
- `tests/test_api.py`
- `tests/e2e/test_api_e2e.py`
- Any other test files using assertions

**Pattern to replace**:
```python
# OLD PATTERN - will no longer work
ctx.assert_that(metric).is_gt(0)

# NEW PATTERN - required
ctx.assert_that(metric).where(name="Metric is positive").is_gt(0)
```

**How to find all occurrences**:
1. Search for `assert_that` in the tests directory
2. For each occurrence, ensure it has `.where(name=...)` before any assertion method

**Example test update**:
```python
# Before
def test_something():
    ctx.assert_that(mp.num_rows()).is_gt(0)
    ctx.assert_that(mp.average("price")).is_positive()

# After
def test_something():
    ctx.assert_that(mp.num_rows()).where(name="Has data").is_gt(0)
    ctx.assert_that(mp.average("price")).where(name="Price is positive").is_positive()
```

### Task 6: Add Comprehensive Tests for New Behavior

**Goal**: Ensure the new two-stage assertion system works correctly.

**File to modify**:
- `tests/test_api.py`

**Tests to add**:

```python
def test_assertion_workflow_end_to_end() -> None:
    """Test complete assertion workflow from draft to execution."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        # Create draft
        draft = ctx.assert_that(sp.Symbol("x"))

        # Convert to ready with name
        ready = draft.where(name="X is positive")

        # Make assertion
        ready.is_positive()

        # Verify assertion was created
        assert len(ctx.current_check.children) == 1
        assert ctx.current_check.children[0].name == "X is positive"

    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.collect(context, key=key)


def test_cannot_use_assertion_methods_on_draft() -> None:
    """Assertion methods should not be available on AssertionDraft."""
    draft = AssertionDraft(sp.Symbol("x"), context=None)

    # These should raise AttributeError
    with pytest.raises(AttributeError):
        draft.is_gt(0)  # type: ignore

    with pytest.raises(AttributeError):
        draft.is_positive()  # type: ignore


def test_where_requires_name_parameter() -> None:
    """The where() method should require name parameter."""
    draft = AssertionDraft(sp.Symbol("x"), context=None)

    # Should fail without name
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'name'"):
        draft.where()  # type: ignore

    # Should work with name
    ready = draft.where(name="Valid name")
    assert isinstance(ready, AssertionReady)


def test_assertion_ready_always_has_name() -> None:
    """AssertionReady should always have a name set."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(sp.Symbol("x")).where(name="My assertion").is_positive()

        # Check that the assertion node has the name
        assertion_node = ctx.current_check.children[0]
        assert assertion_node.name == "My assertion"

    suite = VerificationSuite([test_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.collect(context, key=key)


def test_where_validates_name() -> None:
    """The where() method should validate name parameter."""
    draft = AssertionDraft(sp.Symbol("x"), context=None)

    # Empty string should fail
    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        draft.where(name="")

    # Whitespace only should fail
    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        draft.where(name="   ")

    # Too long name should fail
    with pytest.raises(ValueError, match="Assertion name is too long"):
        draft.where(name="x" * 256)

    # Valid name should work
    ready = draft.where(name="Valid assertion name")
    assert isinstance(ready, AssertionReady)

    # Name should be stripped
    ready2 = draft.where(name="  Trimmed name  ")
    assert ready2._name == "Trimmed name"


def test_thread_safety() -> None:
    """Test that assertion building is thread-safe."""
    import threading
    db = InMemoryMetricDB()
    context = Context("test", db)
    results = []

    @check(name="Thread Test")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        # Each thread creates its own assertion
        thread_id = threading.current_thread().ident
        ctx.assert_that(sp.Symbol("x")).where(
            name=f"Assertion from thread {thread_id}"
        ).is_positive()

    def run_check():
        suite = VerificationSuite([test_check], db, "test")
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.collect(context, key=key)
        results.append(len(context._graph.root.children[0].children))

    # Run in multiple threads
    threads = [threading.Thread(target=run_check) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread should have created one assertion
    assert all(count == 1 for count in results)
```

### Task 7: Update Documentation

**Goal**: Update examples and documentation to reflect the new API.

**Files to update**:
- `README.md` - Update all assertion examples
- `docs/design.md` - Add note about mandatory assertion names

**Example updates for README.md**:

Search for assertion examples and update them:
```python
# OLD
ctx.assert_that(mp.null_count("customer_id")).is_eq(0)

# NEW
ctx.assert_that(mp.null_count("customer_id")).where(name="No null customer IDs").is_eq(0)
```

### Task 8: Run Full Test Suite and Fix Any Issues

**Goal**: Ensure all tests pass with the new implementation.

**Commands to run**:
1. `uv run mypy src/` - Check for type errors
2. `uv run ruff check src/ tests/` - Check for linting issues
3. `uv run pytest -xvs` - Run all tests
4. `uv run pytest --cov=dqx` - Check test coverage

**Common issues to watch for**:
- Import errors (if AssertBuilder was imported anywhere)
- Tests that use the old assertion pattern
- Type errors in the new classes

### Task 9: Final Cleanup

**Goal**: Remove any remaining references to the old API.

**What to do**:
1. Search entire codebase for "AssertBuilder" - should find no results
2. Search for `.is_gt(`, `.is_eq(`, etc. without preceding `.where(` - update any found
3. Run `uv run ruff check --fix src/ tests/` to fix any formatting issues

## Testing Strategy

### Unit Tests (TDD Approach)

For each task, write the test FIRST, then implement the code to make it pass:

1. **Test AssertionDraft**:
   - Can be created
   - Has only `where()` method
   - `where()` returns AssertionReady

2. **Test AssertionReady**:
   - Has all assertion methods
   - Does not have `where()` method
   - Creates assertion nodes correctly

3. **Test Context Integration**:
   - `assert_that()` returns AssertionDraft
   - Full workflow works end-to-end

4. **Test Error Cases**:
   - Cannot call assertion methods on draft
   - Must provide name to `where()`
   - Proper error messages

### How to Run Tests During Development

```bash
# Run specific test
uv run pytest tests/test_api.py::test_assertion_draft_creation -xvs

# Run all api tests
uv run pytest tests/test_api.py -xvs

# Run with coverage
uv run pytest tests/test_api.py --cov=dqx.api

# Check types
uv run mypy src/dqx/api.py

# Check style
uv run ruff check src/dqx/api.py
```

## Commit Strategy

Make frequent, small commits after each task:

1. "Add AssertionDraft class with where() method"
2. "Add AssertionReady class with assertion methods"
3. "Update Context.assert_that to return AssertionDraft"
4. "Remove deprecated AssertBuilder class"
5. "Update tests to use new assertion API"
6. "Add comprehensive tests for assertion workflow"
7. "Update documentation with new assertion examples"

## Definition of Done

- [ ] All tests pass (`uv run pytest`)
- [ ] No type errors (`uv run mypy src/`)
- [ ] No linting issues (`uv run ruff check src/ tests/`)
- [ ] Coverage maintained at 100% for modified files
- [ ] All documentation examples updated
- [ ] No references to AssertBuilder remain

## Quick Reference for the Engineer

### Old API (no longer works):
```python
ctx.assert_that(metric).is_positive()
ctx.assert_that(metric).where(name="optional").is_positive()
```

### New API (required):
```python
ctx.assert_that(metric).where(name="Description required").is_positive()
```

### Key Classes:
- `AssertionDraft` - First stage, only has `where()` method
- `AssertionReady` - Second stage, has all assertion methods
- `Context.assert_that()` - Returns AssertionDraft

### Key Principle:
Every assertion MUST have a descriptive name. This helps with debugging and understanding test failures.
