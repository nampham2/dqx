# is_between Assertion Implementation Plan v2

## Overview
Add a new assertion method `is_between` to the DQX framework that checks if a metric value falls within a specified range (inclusive on both ends).

## Background
The DQX framework currently supports individual comparison assertions (is_gt, is_lt, is_geq, is_leq, is_eq) but lacks a convenient method to assert that a value falls within a range. The `is_between` assertion will simplify range checking and improve code readability.

## Design Decisions
- **Bounds**: Inclusive on both ends (lower ≤ value ≤ upper)
- **Tolerance**: Single tolerance parameter for floating-point comparisons on both bounds
- **Validation**: Check that lower ≤ upper to catch user errors early
- **Display Format**: Use ASCII `in [lower, upper]` for better terminal compatibility
- **No backward compatibility concerns**: This is a new feature addition

## Implementation Tasks

### Phase 1: Core Function Implementation (Tasks 1-3)

#### Task 1: Implement is_between function in functions.py
**File**: `src/dqx/functions.py`

Add the following function after the existing comparison functions (around line 130, after `is_negative`):

```python
def is_between(a: float, lower: float, upper: float, tol: float = EPSILON) -> bool:
    """
    Check if a value is between two bounds (inclusive).

    Args:
        a: The value to check.
        lower: The lower bound.
        upper: The upper bound.
        tol: Tolerance for floating-point comparisons (applies to both bounds).

    Returns:
        bool: True if lower ≤ a ≤ upper (within tolerance), False otherwise.
    """
    return is_geq(a, lower, tol) and is_leq(a, upper, tol)
```

**Testing**: Verify the function works correctly by running Python interactively:
```bash
uv run python
>>> from dqx.functions import is_between, EPSILON
>>> is_between(5, 1, 10)  # Should return True
>>> is_between(0, 1, 10)  # Should return False
>>> is_between(11, 1, 10)  # Should return False
>>> is_between(1, 1, 10)  # Should return True (inclusive)
>>> is_between(10, 1, 10)  # Should return True (inclusive)
```

#### Task 2: Write unit tests for is_between function
**File**: `tests/test_functions.py`

Add comprehensive tests for the `is_between` function. Look for the test class that contains tests for other comparison functions and add:

```python
def test_is_between():
    """Test is_between function with various inputs."""
    # Basic integer tests
    assert is_between(5, 1, 10) is True
    assert is_between(0, 1, 10) is False
    assert is_between(11, 1, 10) is False

    # Boundary tests (inclusive)
    assert is_between(1, 1, 10) is True
    assert is_between(10, 1, 10) is True

    # Floating point tests
    assert is_between(5.5, 5.0, 6.0) is True
    assert is_between(4.9, 5.0, 6.0) is False

    # Tolerance tests
    epsilon = 1e-9
    assert is_between(5.0 - epsilon/2, 5.0, 10.0, tol=epsilon) is True
    assert is_between(5.0 - epsilon*2, 5.0, 10.0, tol=epsilon) is False
    assert is_between(10.0 + epsilon/2, 5.0, 10.0, tol=epsilon) is True
    assert is_between(10.0 + epsilon*2, 5.0, 10.0, tol=epsilon) is False

    # Equal bounds
    assert is_between(5, 5, 5) is True
    assert is_between(4, 5, 5) is False
    assert is_between(6, 5, 5) is False

    # Negative numbers
    assert is_between(-5, -10, -1) is True
    assert is_between(-11, -10, -1) is False
    assert is_between(0, -10, -1) is False

    # Mixed positive/negative
    assert is_between(0, -5, 5) is True
    assert is_between(-3, -5, 5) is True
    assert is_between(3, -5, 5) is True
    assert is_between(-6, -5, 5) is False
    assert is_between(6, -5, 5) is False
```

**Testing**: Run the tests to ensure they pass:
```bash
uv run pytest tests/test_functions.py::test_is_between -v
```

#### Task 3: Commit Phase 1 changes
After verifying tests pass:
```bash
git add src/dqx/functions.py tests/test_functions.py
git commit -m "feat: Add is_between function with comprehensive tests"
```

### Phase 2: API Integration (Tasks 4-6)

#### Task 4: Add is_between method to AssertionReady class
**File**: `src/dqx/api.py`

Locate the `AssertionReady` class (around line 86) and add the method after `is_eq` (around line 131):

```python
def is_between(self, lower: float, upper: float, tol: float = functions.EPSILON) -> None:
    """Assert that the expression is between two values (inclusive)."""
    if lower > upper:
        raise ValueError(
            f"Invalid range: lower bound ({lower}) must be less than or equal to upper bound ({upper})"
        )

    validator = SymbolicValidator(
        f"in [{lower}, {upper}]",
        lambda x: functions.is_between(x, lower, upper, tol)
    )
    self._create_assertion_node(validator)
```

**Testing**: Verify the method is added correctly by checking syntax:
```bash
uv run python -m py_compile src/dqx/api.py
```

#### Task 5: Write integration tests for API
**File**: `tests/test_api.py`

Based on the existing test patterns, we need to:

1. **Update the method existence test** - Find `test_assertion_ready_has_all_methods()` and add `is_between`:

```python
def test_assertion_ready_has_all_methods() -> None:
    """AssertionReady should have all assertion methods."""
    from dqx.api import AssertionReady

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
    assert hasattr(ready, "is_between")  # ADD THIS LINE

    # Should NOT have where method
    assert not hasattr(ready, "where")
```

2. **Add a workflow test** - Add this new test function after the existing workflow tests:

```python
def test_is_between_assertion_workflow() -> None:
    """Test is_between assertion in complete workflow."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Range Check")
    def range_check(mp: MetricProvider, ctx: Context) -> None:
        # Test normal range
        ctx.assert_that(sp.Symbol("x")).where(name="X is between 10 and 20").is_between(10.0, 20.0)

        # Test with same bounds
        ctx.assert_that(sp.Symbol("y")).where(name="Y equals 5").is_between(5.0, 5.0)

        # Verify assertions were created
        assert ctx.current_check is not None
        assert len(ctx.current_check.children) == 2
        assert ctx.current_check.children[0].name == "X is between 10 and 20"
        assert ctx.current_check.children[1].name == "Y equals 5"

    suite = VerificationSuite([range_check], db, "test")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.build_graph(context, key=key)


def test_is_between_invalid_bounds() -> None:
    """Test is_between with invalid bounds raises ValueError."""
    db = InMemoryMetricDB()
    context = Context("test", db)

    @check(name="Invalid Range Check")
    def invalid_check(mp: MetricProvider, ctx: Context) -> None:
        # This should raise ValueError
        with pytest.raises(ValueError, match="Invalid range: lower bound .* must be less than or equal to upper bound"):
            ctx.assert_that(sp.Symbol("x")).where(name="Invalid range").is_between(20.0, 10.0)

    # Execute the check to verify the error is raised
    invalid_check(context.provider, context)
```

**Testing**: Run the API tests:
```bash
uv run pytest tests/test_api.py::test_assertion_ready_has_all_methods -v
uv run pytest tests/test_api.py::test_is_between_assertion_workflow -v
uv run pytest tests/test_api.py::test_is_between_invalid_bounds -v
```

#### Task 6: Commit Phase 2 changes
```bash
git add src/dqx/api.py tests/test_api.py
git commit -m "feat: Add is_between assertion to API with validation"
```

### Phase 3: Final Validation (Task 7)

#### Task 7: Final validation and commit
Run all quality checks:

```bash
# Run mypy type checking
uv run mypy src/dqx/functions.py src/dqx/api.py

# Run ruff linting
uv run ruff check src/dqx/functions.py src/dqx/api.py tests/test_functions.py tests/test_api.py

# Run all tests to ensure nothing broke
uv run pytest tests/test_functions.py tests/test_api.py -v

# Run pre-commit hooks
bin/run-hooks.sh

# If all passes, we're done!
```

## Testing Strategy

### Unit Tests
- Test the `is_between` function with various numeric inputs
- Test boundary conditions (values equal to bounds)
- Test tolerance behavior for floating-point comparisons
- Test edge cases (equal bounds, negative numbers)

### Integration Tests
- Verify `is_between` method exists on AssertionReady
- Test the complete workflow from draft to assertion creation
- Test that ValueError is raised for invalid bounds
- Verify the validator function works correctly within the framework

## Common Pitfalls to Avoid

1. **Floating-point comparisons**: Always use the tolerance parameter, don't use direct equality
2. **Bound validation**: Remember to check that lower ≤ upper before creating the validator
3. **Import statements**: Don't forget to import `functions` in api.py if not already imported
4. **Test isolation**: Make sure tests don't depend on external state
5. **ASCII format**: Use `in [lower, upper]` not Unicode characters for terminal compatibility

## Verification Steps

After implementation:
1. Run `uv run pytest` to ensure all tests pass
2. Run `uv run mypy src/dqx` to check types
3. Run `uv run ruff check src/dqx tests` to check code style
4. Run `bin/run-hooks.sh` to run pre-commit hooks

## Success Criteria

- [ ] `is_between` function implemented and tested
- [ ] `is_between` API method implemented with validation
- [ ] All tests pass (unit, integration)
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Code committed to feature branch

## Changes from v1

This version incorporates the following review feedback:
1. Uses ASCII format `in [lower, upper]` instead of Unicode `∈ [lower, upper]`
2. Follows existing test patterns by updating `test_assertion_ready_has_all_methods` and adding workflow tests
3. Enhanced documentation to clarify tolerance applies to both bounds
4. Improved error message for clarity
5. Removed e2e tests and demo as per project requirements
