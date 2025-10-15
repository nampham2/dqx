# is_between Assertion Implementation Plan v1

## Overview
Add a new assertion method `is_between` to the DQX framework that checks if a metric value falls within a specified range (inclusive on both ends).

## Background
The DQX framework currently supports individual comparison assertions (is_gt, is_lt, is_geq, is_leq, is_eq) but lacks a convenient method to assert that a value falls within a range. The `is_between` assertion will simplify range checking and improve code readability.

## Design Decisions
- **Bounds**: Inclusive on both ends (lower ≤ value ≤ upper)
- **Tolerance**: Single tolerance parameter for floating-point comparisons on both bounds
- **Validation**: Check that lower ≤ upper to catch user errors early
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
        tol: Tolerance for floating-point comparisons.

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
        raise ValueError(f"Lower bound ({lower}) must be ≤ upper bound ({upper})")

    validator = SymbolicValidator(
        f"∈ [{lower}, {upper}]",
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

Find where other assertion methods are tested and add tests for `is_between`:

```python
def test_assertion_ready_is_between(mock_context):
    """Test is_between assertion method."""
    # Create an assertion ready instance
    actual = sp.Symbol("x")
    assertion = AssertionReady(actual=actual, name="Test assertion", context=mock_context)

    # Test normal case
    assertion.is_between(10.0, 20.0)

    # Verify the assertion node was created with correct validator
    mock_context.current_check.add_assertion.assert_called_once()
    call_args = mock_context.current_check.add_assertion.call_args
    assert call_args.kwargs["name"] == "Test assertion"
    assert call_args.kwargs["validator"].name == "∈ [10.0, 20.0]"

    # Test the validator function
    validator_fn = call_args.kwargs["validator"].fn
    assert validator_fn(15.0) is True
    assert validator_fn(10.0) is True
    assert validator_fn(20.0) is True
    assert validator_fn(9.9) is False
    assert validator_fn(20.1) is False

def test_assertion_ready_is_between_invalid_bounds(mock_context):
    """Test is_between with invalid bounds raises ValueError."""
    actual = sp.Symbol("x")
    assertion = AssertionReady(actual=actual, name="Test assertion", context=mock_context)

    # Test with lower > upper
    with pytest.raises(ValueError) as exc_info:
        assertion.is_between(20.0, 10.0)

    assert "Lower bound (20.0) must be ≤ upper bound (10.0)" in str(exc_info.value)
```

Also add an end-to-end test where other assertion tests are located:

```python
def test_is_between_assertion_e2e():
    """Test is_between assertion in full context."""
    @check(name="Range Check", datasets=["test_ds"])
    def range_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("value"))
           .where(name="Average is between 10 and 20")
           .is_between(10.0, 20.0)

    # Set up test data and verify assertion works
    # (Follow the pattern of other e2e tests in the file)
```

**Testing**: Run the API tests:
```bash
uv run pytest tests/test_api.py -k "is_between" -v
```

#### Task 6: Commit Phase 2 changes
```bash
git add src/dqx/api.py tests/test_api.py
git commit -m "feat: Add is_between assertion to API with validation"
```

### Phase 3: Final Testing (Task 7)

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
- Test the API method creates correct assertion nodes
- Test the ValueError is raised for invalid bounds
- Test the validator function works correctly


## Common Pitfalls to Avoid

1. **Floating-point comparisons**: Always use the tolerance parameter, don't use direct equality
2. **Bound validation**: Remember to check that lower ≤ upper before creating the validator
3. **Import statements**: Don't forget to import `functions` in api.py if not already imported
4. **Test isolation**: Make sure tests don't depend on external state

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
