# Check Decorator Mandatory Name Implementation Plan

## Overview
This plan details how to refactor the `@check` decorator to:
1. Make the `name` parameter mandatory
2. Remove support for `@check()` with zero arguments
3. Remove unsafe `_check_metadata` attribute usage for better type safety

## Background Context

### What is the @check decorator?
The `@check` decorator is used in DQX (Data Quality eXcellence) to mark functions that define data quality validation rules. These functions receive a MetricProvider and Context, and use them to create assertions about data quality.

### Current Problems
1. **Inconsistent API**: The decorator supports three forms: `@check`, `@check()`, and `@check(name="...")`. This creates confusion.
2. **Type Safety Issues**: The decorator stores metadata using `wrapped._check_metadata`, which requires `# type: ignore` comments and bypasses mypy type checking.
3. **Implicit Behavior**: Using function names as check names by default can lead to unclear check names in reports.

### Solution Approach
We'll simplify the API to require explicit names and remove the metadata storage entirely, passing information directly to the check creation function.

## Prerequisites

### Development Environment Setup
```bash
# Clone the repository if not already done
git clone git@gitlab.booking.com:npham/dqx.git
cd dqx

# Install with uv (the project's package manager)
uv sync

# Verify tests pass before starting
uv run pytest tests/test_api.py -v
uv run mypy src/dqx/api.py
uv run ruff check src/dqx/api.py
```

### Key Files to Understand
- `src/dqx/api.py` - Contains the check decorator implementation
- `tests/test_api.py` - Unit tests for the API module
- `tests/e2e/test_api_e2e.py` - End-to-end tests using the check decorator
- `README.md` - Documentation showing check decorator usage examples

## Implementation Tasks

### Task 1: Write Failing Tests for New Behavior

**Files to modify**: `tests/test_api.py`

**What to do**:
1. Create a new test file section for the mandatory name requirement
2. Write tests that verify the decorator fails without a name
3. Write tests that verify the decorator works with a name

**Code to add**:
```python
def test_check_decorator_requires_name():
    """Test that @check decorator requires name parameter."""
    db = InMemoryMetricDB()
    mp = MetricProvider(db)
    ctx = Context("Test Suite", db)

    # This should raise TypeError because name is required
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'name'"):
        @check()  # Missing required name parameter
        def my_check(mp: MetricProvider, ctx: Context) -> None:
            pass

def test_check_decorator_without_parentheses_not_allowed():
    """Test that @check without parentheses is not allowed."""
    # This test verifies compile-time behavior - the decorator
    # should not be callable without parentheses anymore
    # Note: This will be a syntax/type error after implementation
    pass

def test_check_decorator_with_name_works():
    """Test that @check with name parameter works correctly."""
    db = InMemoryMetricDB()

    @check(name="Valid Check")
    def my_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).is_gt(0)

    # Verify the check can be used in a suite
    suite = VerificationSuiteBuilder("Test", db).add_check(my_check).build()
    assert suite is not None
```

**How to test**:
```bash
# Run only the new tests (they should fail initially)
uv run pytest tests/test_api.py::test_check_decorator_requires_name -v
```

**Commit**: `test: add failing tests for mandatory check name`

### Task 2: Update Existing Tests

**Files to modify**:
- `tests/test_api.py`
- `tests/e2e/test_api_e2e.py`

**What to do**:
1. Find all uses of `@check` without parentheses
2. Find all uses of `@check()` with empty parentheses
3. Update them to use `@check(name="...")`

**Examples to fix**:
```python
# OLD - tests/test_api.py
@check
def simple_check(mp: MetricProvider, ctx: Context) -> None:
    pass

# NEW
@check(name="Simple Check")
def simple_check(mp: MetricProvider, ctx: Context) -> None:
    pass

# OLD - tests/test_api.py
@check()
def empty_paren_check(mp: MetricProvider, ctx: Context) -> None:
    pass

# NEW
@check(name="Empty Paren Check")
def empty_paren_check(mp: MetricProvider, ctx: Context) -> None:
    pass
```

**How to find all occurrences**:
```bash
# Find @check without parentheses (look for @check followed by newline)
grep -n "@check$" tests/

# Find @check() with empty parentheses
grep -n "@check()" tests/
```

**How to test**:
```bash
# These tests should still fail because implementation isn't done yet
uv run pytest tests/test_api.py -v
uv run pytest tests/e2e/test_api_e2e.py -v
```

**Commit**: `test: update existing tests for mandatory check name`

### Task 3: Remove CheckMetadata and DecoratedCheck Protocol

**Files to modify**: `src/dqx/api.py`

**What to do**:
1. Remove the `CheckMetadata` dataclass (lines ~33-39)
2. Remove the `_check_metadata` attribute from `DecoratedCheck` protocol
3. Simplify the protocol to just define the callable signature

**Code changes**:
```python
# DELETE this entire dataclass
@dataclass
class CheckMetadata:
    """Metadata stored on decorated check functions."""
    name: str  # The function name
    datasets: list[str] | None = None
    tags: list[str] = field(default_factory=list)
    display_name: str | None = None  # User-provided name

# CHANGE the DecoratedCheck protocol from:
@runtime_checkable
class DecoratedCheck(Protocol):
    """Protocol for check functions with metadata."""
    __name__: str
    _check_metadata: CheckMetadata
    def __call__(self, mp: MetricProvider, ctx: "Context") -> None: ...

# TO:
@runtime_checkable
class DecoratedCheck(Protocol):
    """Protocol for check functions."""
    __name__: str
    def __call__(self, mp: MetricProvider, ctx: "Context") -> None: ...
```

**How to test**:
```bash
# Type checking should pass
uv run mypy src/dqx/api.py
```

**Commit**: `refactor: remove CheckMetadata and simplify DecoratedCheck protocol`

### Task 4: Update Check Decorator Overloads

**Files to modify**: `src/dqx/api.py`

**What to do**:
1. Remove the overload for `@check` without parentheses
2. Remove the overload for `@check()` with optional parameters
3. Keep only the overload with required name parameter

**Code changes**:
```python
# DELETE these overloads:
@overload
def check(_check: CheckProducer) -> DecoratedCheck: ...

@overload
def check() -> Callable[[CheckProducer], DecoratedCheck]: ...

# CHANGE this overload from:
@overload
def check(
    *, name: str | None = None, tags: list[str] = [], datasets: list[str] | None = None
) -> Callable[[CheckProducer], DecoratedCheck]: ...

# TO:
@overload
def check(
    *, name: str, tags: list[str] = [], datasets: list[str] | None = None
) -> Callable[[CheckProducer], DecoratedCheck]: ...
```

**Commit**: `refactor: update check decorator overloads for mandatory name`

### Task 5: Simplify Check Decorator Implementation

**Files to modify**: `src/dqx/api.py`

**What to do**:
1. Remove the branch handling `_check` parameter (decorator without parentheses)
2. Make `name` parameter required
3. Remove all `_check_metadata` assignments
4. Simplify the decorator to only handle the parametrized form

**Code changes**:
```python
def check(
    *,
    name: str,  # Changed from str | None to str (required)
    tags: list[str] = [],
    datasets: list[str] | None = None,
) -> Callable[[CheckProducer], DecoratedCheck]:
    """
    Decorator for creating data quality check functions.

    Must be used with parentheses and a name:

    @check(name="Important Check", tags=["critical"], datasets=["ds1"])
    def my_labeled_check(mp: MetricProvider, ctx: Context) -> None:
        # check logic

    Args:
        name: Human-readable name for the check (required)
        tags: Optional tags for categorizing the check
        datasets: Optional list of datasets the check applies to

    Returns:
        Decorated check function
    """
    def decorator(fn: CheckProducer) -> DecoratedCheck:
        wrapped = functools.wraps(fn)(
            functools.partial(
                _create_check,
                _check=fn,
                name=name,  # Always use the provided name
                tags=tags,
                datasets=datasets
            )
        )
        # No metadata storage needed!
        return cast(DecoratedCheck, wrapped)

    return decorator
```

**Remove the entire first branch** that handles `if _check is not None:` case.

**How to test**:
```bash
# All tests should now pass!
uv run pytest tests/test_api.py -v
uv run pytest tests/e2e/test_api_e2e.py -v

# Type checking should pass without any ignores
uv run mypy src/dqx/api.py

# Linting should pass
uv run ruff check src/dqx/api.py
```

**Commit**: `refactor: simplify check decorator with mandatory name`

### Task 6: Update Documentation

**Files to modify**:
- `README.md`
- Any example files in `examples/`

**What to do**:
1. Update all code examples to use `@check(name="...")`
2. Remove any mentions of using `@check` without parentheses
3. Update the decorator documentation section

**Example changes in README.md**:
```python
# Find and replace patterns like:
@check
def validate_orders(mp, ctx):

# With:
@check(name="Order validation")
def validate_orders(mp, ctx):
```

**How to test**:
```bash
# Manually review that all examples are updated
grep -n "@check\s*$" README.md examples/
grep -n "@check()" README.md examples/
```

**Commit**: `docs: update examples for mandatory check name`

### Task 7: Run Full Test Suite and Quality Checks

**What to do**:
1. Run the complete test suite
2. Check code coverage
3. Run type checking on the entire codebase
4. Run linting

**Commands**:
```bash
# Full test suite
uv run pytest -v

# With coverage
uv run pytest --cov=dqx --cov-report=html

# Type checking
uv run mypy src/

# Linting and formatting
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Pre-commit hooks (if configured)
./bin/run-hooks.sh
```

**Fix any issues that arise**.

**Commit**: `chore: fix linting and type issues after refactor`

## Testing Strategy

### Unit Tests
- Verify `TypeError` is raised when name is not provided
- Verify decorator works correctly with required name
- Verify all optional parameters still work (tags, datasets)

### Integration Tests
- Verify existing e2e tests still pass with updated decorator syntax
- Verify check names appear correctly in verification results

### Manual Testing
Create a simple test script:
```python
# test_manual.py
from dqx.api import check, VerificationSuiteBuilder, Context
from dqx.provider import MetricProvider
from dqx.orm.repositories import InMemoryMetricDB

db = InMemoryMetricDB()

# This should fail at decoration time
try:
    @check()  # Missing name
    def bad_check(mp, ctx):
        pass
except TypeError as e:
    print(f"✓ Got expected error: {e}")

# This should work
@check(name="Good Check")
def good_check(mp: MetricProvider, ctx: Context):
    ctx.assert_that(mp.num_rows()).is_gt(0)

print("✓ Decorator with name works!")
```

## Rollback Plan

If issues are discovered after deployment:
1. Git revert the merge commit
2. Fix the issues in a new branch
3. Re-deploy after testing

The changes are backwards-incompatible, so coordinate with any teams using DQX.

## Success Criteria

1. ✅ All tests pass
2. ✅ No type checking errors (mypy)
3. ✅ No linting errors (ruff)
4. ✅ Documentation is updated
5. ✅ Check decorator requires explicit name
6. ✅ No `_check_metadata` attributes in code
7. ✅ Cleaner, more maintainable code

## Common Pitfalls to Avoid

1. **Don't forget to update examples** - Check README.md and any example files
2. **Don't leave old tests** - Remove or update tests for the old behavior
3. **Don't skip TDD** - Write failing tests first, then implement
4. **Don't make large commits** - Each task should be a separate commit
5. **Don't ignore type errors** - Fix them properly, don't add `# type: ignore`

## Summary

This refactor will:
- Improve API consistency by requiring explicit check names
- Enhance type safety by removing dynamic attribute assignment
- Simplify the codebase by removing unnecessary metadata storage
- Make the framework more maintainable and easier to understand

Total estimated time: 2-3 hours for an experienced developer following TDD.
