# Fix Check Decorator Name Parameter Implementation Plan

## Problem Summary

The `@check` decorator in `src/dqx/api.py` is broken. Someone made the `name` parameter required, but this breaks the simple usage pattern `@check` (without parentheses). When using `@check` without arguments, the code tries to reference an undefined `name` variable, causing a `NameError`.

**Current broken behavior:**
```python
@check  # NameError: name 'name' is not defined
def my_check(mp, ctx):
    pass
```

**Expected behavior:**
```python
@check  # Should use function name 'my_check' as the name
def my_check(mp, ctx):
    pass

@check(name="My Custom Check")  # Should use "My Custom Check" as the name
def my_check(mp, ctx):
    pass
```

## Background Context

### What is DQX?

DQX (Data Quality eXcellence) is a data quality validation framework. Users write "checks" (functions that validate data quality) and the framework executes them against datasets.

### What is the @check decorator?

The `@check` decorator is used to mark functions as data quality checks. It:
1. Wraps the function to integrate it with the DQX framework
2. Stores metadata about the check (name, tags, datasets)
3. Can be used in two ways:
   - Simple form: `@check` (should use function name)
   - Parametrized form: `@check(name="My Check", tags=["critical"])`

### Key Files

- `src/dqx/api.py` - Contains the check decorator and related classes
- `tests/test_api.py` - Tests for the API module
- `tests/e2e/test_api_e2e.py` - End-to-end tests showing real usage

## Implementation Tasks

Follow these tasks in order. Use Test-Driven Development (TDD) - write the test first, see it fail, then make it pass.

### Task 1: Set Up and Understand the Current State

**Time estimate:** 15 minutes

1. **Check your environment:**
   ```bash
   # Make sure you're in the project root
   pwd  # Should show /Users/npham/git-tree/dqx

   # Check Python version
   uv run python --version  # Should be 3.11 or 3.12

   # Install dependencies if needed
   uv sync
   ```

2. **Create a new branch:**
   ```bash
   git checkout -b fix-check-decorator-name
   ```

3. **Run existing tests to see failures:**
   ```bash
   # Run API tests
   uv run pytest tests/test_api.py -v

   # You'll see errors with "# type: ignore[arg-type]" comments
   # This indicates the tests know something is wrong
   ```

4. **Commit the starting point:**
   ```bash
   git add -A
   git commit -m "chore: starting fix for check decorator name parameter"
   ```

### Task 2: Write a Failing Test

**Time estimate:** 20 minutes

1. **Create a new test file to isolate our fix:**
   ```bash
   touch tests/test_check_decorator_fix.py
   ```

2. **Write the test file:**
   ```python
   """Test the fixed check decorator behavior."""
   import datetime
   from dqx.api import check, MetricProvider, Context, VerificationSuite
   from dqx.common import ResultKey
   from dqx.orm.repositories import InMemoryMetricDB


   def test_simple_check_uses_function_name():
       """Test that @check without params uses function name."""
       # Create a simple check without parameters
       @check
       def validate_orders(mp: MetricProvider, ctx: Context) -> None:
           ctx.assert_that(mp.num_rows()).is_gt(0)

       # The metadata should use the function name
       assert hasattr(validate_orders, '_check_metadata')
       assert validate_orders._check_metadata['name'] == 'validate_orders'
       assert validate_orders._check_metadata['display_name'] is None


   def test_parametrized_check_uses_provided_name():
       """Test that @check with name parameter uses that name."""
       @check(name="Order Validation Check", tags=["critical"])
       def validate_orders(mp: MetricProvider, ctx: Context) -> None:
           ctx.assert_that(mp.num_rows()).is_gt(0)

       # The metadata should store both names
       assert validate_orders._check_metadata['name'] == 'validate_orders'  # function name
       assert validate_orders._check_metadata['display_name'] == "Order Validation Check"  # provided name
       assert validate_orders._check_metadata['tags'] == ["critical"]


   def test_simple_check_works_in_suite():
       """Test that simple @check works in a verification suite."""
       @check
       def my_simple_check(mp: MetricProvider, ctx: Context) -> None:
           ctx.assert_that(mp.num_rows()).is_gt(0)

       # Should be able to use in a suite without errors
       db = InMemoryMetricDB()
       suite = VerificationSuite([my_simple_check], db, "Test Suite")

       # Collect checks (this is where it would fail with NameError)
       context = Context("test", db)
       key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
       suite.collect(context, key)

       # Verify the check was registered correctly
       checks = list(context._graph.root.children)
       assert len(checks) == 1
       assert checks[0].name == "my_simple_check"
   ```

3. **Run the test to see it fail:**
   ```bash
   uv run pytest tests/test_check_decorator_fix.py -v

   # You should see NameError: name 'name' is not defined
   ```

4. **Commit the failing test:**
   ```bash
   git add tests/test_check_decorator_fix.py
   git commit -m "test: add failing tests for check decorator name fix"
   ```

### Task 3: Convert CheckMetadata to Dataclass

**Time estimate:** 15 minutes

1. **Open `src/dqx/api.py` and find the CheckMetadata TypedDict (around line 30):**
   ```python
   # Find this:
   class CheckMetadata(TypedDict):
       """Metadata stored on decorated check functions."""
       name: str  # The function name
       datasets: list[str] | None
       tags: list[str]
       display_name: str | None  # WAS: label: str | None
   ```

2. **Add the dataclass import at the top of the file:**
   ```python
   # Add this to the imports section (around line 3-4)
   from dataclasses import dataclass, field
   ```

3. **Replace CheckMetadata TypedDict with a dataclass:**
   ```python
   # Replace the TypedDict with:
   @dataclass
   class CheckMetadata:
       """Metadata stored on decorated check functions."""
       name: str  # The function name
       datasets: list[str] | None = None
       tags: list[str] = field(default_factory=list)
       display_name: str | None = None  # User-provided name
   ```

4. **Run tests to ensure nothing else broke:**
   ```bash
   uv run pytest tests/test_check_decorator_fix.py -v
   # Still failing, but that's expected
   ```

5. **Commit this change:**
   ```bash
   git add src/dqx/api.py
   git commit -m "refactor: convert CheckMetadata from TypedDict to dataclass"
   ```

### Task 4: Fix the Simple Check Decorator

**Time estimate:** 25 minutes

1. **Find the check decorator function (around line 410):**
   ```python
   # Look for this section:
   def check(
       _check: CheckProducer | None = None,
       *,
       name: str,
       tags: list[str] = [],
       datasets: list[str] | None = None,
   ) -> DecoratedCheck | Callable[[CheckProducer], DecoratedCheck]:
   ```

2. **Make the name parameter optional in the function signature:**
   ```python
   def check(
       _check: CheckProducer | None = None,
       *,
       name: str | None = None,  # Make this optional
       tags: list[str] = [],
       datasets: list[str] | None = None,
   ) -> DecoratedCheck | Callable[[CheckProducer], DecoratedCheck]:
   ```

3. **Fix the simple decorator case (find the `if _check is not None:` block):**
   ```python
   if _check is not None:
       # Simple @check decorator without parentheses
       wrapped = functools.wraps(_check)(
           functools.partial(
               _create_check,
               _check=_check,
               name=_check.__name__,  # Use function name
               tags=tags,
               datasets=datasets
           )
       )
       # Store metadata using dataclass
       wrapped._check_metadata = CheckMetadata(
           name=_check.__name__,
           datasets=datasets,
           tags=tags,
           display_name=None
       )
       return cast(DecoratedCheck, wrapped)
   ```

4. **Fix the parametrized decorator case:**
   ```python
   # Add validation for the parametrized form
   if name is None:
       raise TypeError("check() missing required keyword argument: 'name'")

   def decorator(fn: CheckProducer) -> DecoratedCheck:
       wrapped = functools.wraps(fn)(
           functools.partial(
               _create_check,
               _check=fn,
               name=name,  # Use provided name
               tags=tags,
               datasets=datasets
           )
       )
       # Store metadata using dataclass
       wrapped._check_metadata = CheckMetadata(
           name=fn.__name__,  # Store function name
           datasets=datasets,
           tags=tags,
           display_name=name  # Store user-provided name
       )
       return cast(DecoratedCheck, wrapped)

   return decorator
   ```

5. **Run your tests:**
   ```bash
   uv run pytest tests/test_check_decorator_fix.py -v
   # Should pass now!
   ```

6. **Run all API tests:**
   ```bash
   uv run pytest tests/test_api.py -v
   ```

7. **Commit the fix:**
   ```bash
   git add src/dqx/api.py
   git commit -m "fix: make @check decorator work without name parameter

   - Simple @check now uses function __name__ as the name
   - Parametrized @check(name=...) requires explicit name
   - Store both function name and display name in metadata"
   ```

### Task 5: Update the Overload Signatures

**Time estimate:** 10 minutes

The overload signatures tell type checkers what forms of the decorator are valid.

1. **Find the overload definitions (around line 400):**
   ```python
   @overload
   def check(_check: CheckProducer) -> DecoratedCheck: ...

   @overload
   def check(
       *, name: str, tags: list[str] = [], datasets: list[str] | None = None
   ) -> Callable[[CheckProducer], DecoratedCheck]: ...
   ```

2. **The overloads are already correct!** The first one handles `@check` without parameters, the second handles `@check(name="...")`. No changes needed here.

### Task 6: Remove type: ignore Comments from Tests

**Time estimate:** 15 minutes

1. **Open `tests/test_api.py` and find all `# type: ignore[arg-type]` comments:**
   ```bash
   grep -n "type: ignore\[arg-type\]" tests/test_api.py
   ```

2. **Remove these comments from the @check decorators:**
   ```python
   # Change this:
   @check  # type: ignore[arg-type]
   def test_check(mp: MetricProvider, ctx: Context) -> None:

   # To this:
   @check
   def test_check(mp: MetricProvider, ctx: Context) -> None:
   ```

3. **Run the tests to make sure they still work:**
   ```bash
   uv run pytest tests/test_api.py -v
   ```

4. **Check type errors:**
   ```bash
   uv run mypy src/dqx/api.py
   ```

5. **Commit the cleanup:**
   ```bash
   git add tests/test_api.py
   git commit -m "chore: remove type: ignore comments from tests"
   ```

### Task 7: Run Full Test Suite

**Time estimate:** 10 minutes

1. **Run all tests:**
   ```bash
   # Run all tests
   uv run pytest -v

   # Run with coverage
   uv run pytest --cov=dqx.api tests/test_api.py tests/test_check_decorator_fix.py
   ```

2. **Run code quality checks:**
   ```bash
   # Type checking
   uv run mypy src/dqx/api.py

   # Linting
   uv run ruff check src/dqx/api.py

   # Format check
   uv run ruff format --check src/dqx/api.py
   ```

3. **If any formatting issues, fix them:**
   ```bash
   uv run ruff format src/dqx/api.py
   uv run ruff check --fix src/dqx/api.py
   ```

4. **Final commit:**
   ```bash
   git add -A
   git commit -m "chore: fix formatting and linting issues"
   ```

### Task 8: Update Documentation

**Time estimate:** 10 minutes

1. **Check if README examples still work:**
   ```bash
   grep -A5 -B5 "@check" README.md
   ```

2. **Verify the examples show both usage patterns:**
   - Simple: `@check`
   - Parametrized: `@check(name="My Check", tags=["critical"])`

3. **If needed, update documentation to clarify both patterns are supported.**

## Testing Strategy

### Unit Tests
- Test simple decorator form (`@check`)
- Test parametrized form (`@check(name="...")`)
- Test that metadata is stored correctly
- Test that missing name in parametrized form raises TypeError

### Integration Tests
- Test that checks work in VerificationSuite
- Test that check names appear correctly in results
- Test existing e2e tests still pass

### How to Run Tests

```bash
# Run specific test file
uv run pytest tests/test_check_decorator_fix.py -v

# Run all API tests
uv run pytest tests/test_api.py -v

# Run with coverage
uv run pytest --cov=dqx.api

# Run all tests
uv run pytest

# Run e2e tests
uv run pytest tests/e2e/test_api_e2e.py -v
```

## Common Pitfalls

1. **Don't forget the dataclass import** - Without it, you'll get NameError for `@dataclass`

2. **Watch the metadata structure** - We store both `name` (function name) and `display_name` (user-provided)

3. **Don't change the overload signatures** - They're already correct

4. **Test both forms** - Make sure both `@check` and `@check(name=...)` work

5. **Check for type errors** - Run mypy to catch type issues

## Verification Steps

1. **All tests pass:**
   ```bash
   uv run pytest
   ```

2. **No type errors:**
   ```bash
   uv run mypy src/dqx/api.py
   ```

3. **Code is formatted:**
   ```bash
   uv run ruff format --check src/
   uv run ruff check src/
   ```

4. **Simple decorator works:**
   ```python
   # This should work without errors
   @check
   def my_check(mp, ctx):
       pass
   ```

5. **Parametrized decorator works:**
   ```python
   # This should also work
   @check(name="My Check", tags=["test"])
   def my_check(mp, ctx):
       pass
   ```

## Summary

This fix makes the `@check` decorator work in both forms:
- Simple form uses the function name
- Parametrized form requires an explicit name

The implementation:
1. Converts CheckMetadata to a dataclass (cleaner than TypedDict)
2. Makes name parameter optional internally
3. Uses function `__name__` for simple decorator
4. Validates name is provided for parametrized form
5. Stores both function name and display name in metadata

Total estimated time: ~2 hours with careful testing and commits.
