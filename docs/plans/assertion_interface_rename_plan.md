# DQX Assertion Interface Rename Implementation Plan

## Overview

This plan describes how to rename the assertion interface in the DQX (Data Quality eXcellence) framework to improve API consistency. The main changes are:

1. Rename `AssertBuilder.on()` method to `AssertBuilder.where()`
2. Rename all `label` parameters to `name` throughout the API
3. Consolidate CheckNode's `name` and `label` fields into a single `name` field

## Background Context

### What is DQX?
DQX is a data quality framework that allows users to write checks (validation rules) for their data. Users define checks using a decorator pattern and make assertions about metrics computed from their data.

### Current API Example
```python
# Current API (what we're changing FROM)
@check(label="Order validation", tags=["critical"])
def validate_orders(mp, ctx):
    ctx.assert_that(mp.null_count("id")).on(label="No null IDs").is_eq(0)
```

### New API Example
```python
# New API (what we're changing TO)
@check(name="Order validation", tags=["critical"])
def validate_orders(mp, ctx):
    ctx.assert_that(mp.null_count("id")).where(name="No null IDs").is_eq(0)
```

## Development Environment Setup

### Tools You'll Need
- Python 3.11 or 3.12
- `uv` (Python package manager) - The project uses this instead of pip
- VS Code or similar editor with Python support

### Initial Setup
```bash
# Clone and setup
cd /Users/npham/git-tree/dqx
uv sync  # Installs all dependencies

# Run tests to ensure everything works
uv run pytest tests/test_api.py -v  # Should pass before starting
```

### Key Commands
```bash
# Run specific tests
uv run pytest tests/test_api.py::test_name -v

# Run linting
uv run ruff check src/
uv run ruff check src/ --fix  # Auto-fix issues

# Run type checking
uv run mypy src/

# Run all quality checks (before committing)
uv run ruff check src/ tests/ && uv run mypy src/
```

## Implementation Tasks

### Task 1: Write Failing Tests for New API (TDD)

**Goal**: Create tests that use the new API (they will fail initially).

**Files to create/modify**:
- `tests/test_assertion_rename.py` (new file)

**Code to write**:
```python
# tests/test_assertion_rename.py
import pytest
import sympy as sp
from dqx.api import check, Context, VerificationSuiteBuilder
from dqx.provider import MetricProvider
from dqx.orm.repositories import InMemoryMetricDB


def test_assert_builder_where_method():
    """Test that AssertBuilder has where() method instead of on()."""
    db = InMemoryMetricDB()
    ctx = Context("test_suite", db)

    # Create a simple expression
    expr = sp.Symbol("x")

    # This should work with new API
    assert_builder = ctx.assert_that(expr)

    # Should have where method
    assert hasattr(assert_builder, "where")

    # Should NOT have on method
    assert not hasattr(assert_builder, "on")

    # Test method chaining
    result = assert_builder.where(name="Test assertion")
    assert result is assert_builder  # Should return self


def test_where_method_parameters():
    """Test that where() accepts name parameter instead of label."""
    db = InMemoryMetricDB()
    ctx = Context("test_suite", db)

    # Should accept name parameter
    ctx.assert_that(sp.Symbol("x")).where(name="Test name").is_eq(42)

    # Should NOT accept label parameter
    with pytest.raises(TypeError, match="unexpected keyword argument 'label'"):
        ctx.assert_that(sp.Symbol("x")).where(label="Test label").is_eq(42)


def test_check_decorator_with_name():
    """Test that @check decorator accepts name instead of label."""

    # Define a check with name parameter
    @check(name="My validation check", tags=["test"])
    def my_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    # Verify metadata is stored correctly
    assert hasattr(my_check, "_check_metadata")
    assert my_check._check_metadata["display_name"] == "My validation check"
    assert my_check._check_metadata["name"] == "my_check"  # Function name


def test_check_decorator_without_name():
    """Test that @check decorator uses function name when name not provided."""

    @check(tags=["test"])
    def validate_something(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).is_gt(0)

    # Should use function name
    assert my_check._check_metadata["name"] == "validate_something"
    assert my_check._check_metadata["display_name"] is None
```

**How to run**:
```bash
# This will fail initially - that's expected!
uv run pytest tests/test_assertion_rename.py -v

# Commit the failing tests
git add tests/test_assertion_rename.py
git commit -m "test: add failing tests for assertion interface rename"
```

### Task 2: Update AssertBuilder Class

**Goal**: Rename the `on()` method to `where()` and update parameter names.

**File to modify**: `src/dqx/api.py`

**Changes to make**:

1. Find the `AssertBuilder` class (around line 50-120)
2. Make these specific changes:

```python
# CHANGE 1: Update __init__ to use _name instead of _label
def __init__(self, actual: sp.Expr, context: Context | None = None) -> None:
    self._actual = actual
    self._name: str | None = None  # WAS: self._label
    self._severity: SeverityLevel | None = None
    self._validator: SymbolicValidator | None = None
    self._context = context


# CHANGE 2: Rename on() method to where()
def where(
    self, *, name: str | None = None, severity: SeverityLevel | None = None
) -> Self:
    """
    Configure the assertion with optional name and severity.

    Args:
        name: Human-readable description of the assertion  # WAS: label
        severity: Severity level for assertion failures

    Returns:
        Self for method chaining
    """
    self._name = name  # WAS: self._label = label
    self._severity = severity
    return self


# CHANGE 3: Update _create_assertion_node to use self._name
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
        name=self._name,  # WAS: label=self._label
        severity=self._severity,
        validator=validator,
    )

    # Attach to the current check node
    current.add_child(node)
```

**How to test**:
```bash
# Run just the AssertBuilder tests
uv run pytest tests/test_assertion_rename.py::test_assert_builder_where_method -v
uv run pytest tests/test_assertion_rename.py::test_where_method_parameters -v

# Some should start passing!
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "refactor: rename AssertBuilder.on() to where() and label to name"
```

### Task 3: Update Context Methods

**Goal**: Update the Context class methods to use `name` instead of `label`.

**File to modify**: `src/dqx/api.py`

**Changes to make**:

1. Find the `Context` class (around line 130-250)
2. Update the `create_assertion` method:

```python
def create_assertion(
    self,
    actual: sp.Expr,
    name: str | None = None,  # WAS: label: str | None = None
    severity: SeverityLevel | None = None,
    validator: SymbolicValidator | None = None,
) -> AssertionNode:
    """
    Factory method to create an assertion node.

    Args:
        actual: Symbolic expression to evaluate
        name: Optional human-readable description  # WAS: label
        severity: Optional severity level for failures
        validator: Optional validator function

    Returns:
        AssertionNode that can access context through its root node
    """
    return AssertionNode(
        actual=actual,
        name=name,  # WAS: label=label
        severity=severity,
        validator=validator,
    )
```

3. Update the `create_check` method to remove `label` parameter:

```python
def create_check(
    self,
    name: str,
    tags: list[str] | None = None,
    datasets: list[str] | None = None,
) -> CheckNode:
    """
    Factory method to create a check node.

    Args:
        name: Name for the check (either user-provided or function name)
        tags: Optional tags for categorizing the check
        datasets: Optional list of datasets the check applies to

    Returns:
        CheckNode that can access context through its root node
    """
    return CheckNode(
        name=name,
        tags=tags,
        datasets=datasets,
    )
```

**How to test**:
```bash
# Test that Context methods work correctly
uv run pytest tests/test_api.py -k "test_context" -v
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "refactor: update Context methods to use name instead of label"
```

### Task 4: Update Check Decorator

**Goal**: Update the @check decorator to accept `name` instead of `label`.

**File to modify**: `src/dqx/api.py`

**Changes to make**:

1. Update `CheckMetadata` TypedDict (around line 30):

```python
class CheckMetadata(TypedDict):
    """Metadata stored on decorated check functions."""

    name: str  # The function name
    datasets: list[str] | None
    tags: list[str]
    display_name: str | None  # WAS: label: str | None
```

2. Update the `check` decorator function signatures (around line 470-520):

```python
@overload
def check(_check: CheckProducer) -> DecoratedCheck: ...


@overload
def check(
    *,
    tags: list[str] = [],
    name: str | None = None,
    datasets: list[str] | None = None,  # WAS: label
) -> Callable[[CheckProducer], DecoratedCheck]: ...


def check(
    _check: CheckProducer | None = None,
    *,
    tags: list[str] = [],
    name: str | None = None,  # WAS: label: str | None = None
    datasets: list[str] | None = None,
) -> DecoratedCheck | Callable[[CheckProducer], DecoratedCheck]:
    """
    Decorator for creating data quality check functions.

    Can be used with or without parameters:

    @check
    def my_check(mp: MetricProvider, ctx: Context) -> None:
        # check logic

    @check(name="Important Check", tags=["critical"], datasets=["ds1"])  # WAS: label=
    def my_labeled_check(mp: MetricProvider, ctx: Context) -> None:
        # check logic

    Args:
        _check: The check function (when used without parentheses)
        tags: Optional tags for categorizing the check
        name: Optional human-readable name for the check  # WAS: label
        datasets: Optional list of datasets the check applies to.

    Returns:
        Decorated check function or decorator function
    """
    if _check is not None:
        # Simple @check decorator without parentheses
        wrapped = functools.wraps(_check)(
            functools.partial(
                _create_check,
                _check=_check,
                tags=tags,
                display_name=name,
                datasets=datasets,
            )
        )
        # Store metadata for validation
        wrapped._check_metadata = {  # type: ignore[attr-defined]
            "name": _check.__name__,
            "datasets": datasets,
            "tags": tags,
            "display_name": name,  # WAS: "label": label
        }
        return cast(DecoratedCheck, wrapped)

    def decorator(fn: CheckProducer) -> DecoratedCheck:
        wrapped = functools.wraps(fn)(
            functools.partial(
                _create_check,
                _check=fn,
                tags=tags,
                display_name=name,
                datasets=datasets,
            )
        )
        # Store metadata for validation
        wrapped._check_metadata = {  # type: ignore[attr-defined]
            "name": fn.__name__,
            "datasets": datasets,
            "tags": tags,
            "display_name": name,  # WAS: "label": label
        }
        return cast(DecoratedCheck, wrapped)

    return decorator
```

3. Update `_create_check` function (around line 430-460):

```python
def _create_check(
    provider: MetricProvider,
    context: Context,
    _check: CheckProducer,
    tags: list[str] = [],
    display_name: str | None = None,  # WAS: label
    datasets: list[str] | None = None,
) -> None:
    """
    Internal function to create and register a check node in the context graph.

    Args:
        provider: Metric provider for the check
        context: Execution context
        _check: Check function to execute
        tags: Optional tags for the check
        display_name: Optional human-readable name  # WAS: label
        datasets: Optional list of datasets the check applies to

    Raises:
        DQXError: If a check with the same name already exists
    """
    # Use display_name if provided, otherwise use function name
    node_name = display_name if display_name else _check.__name__

    # Use context factory method
    node = context.create_check(name=node_name, tags=tags, datasets=datasets)

    if context._graph.root.exists(node):
        raise DQXError(f"Check {node.name} already exists in the graph!")

    context._graph.root.add_child(node)

    # Call the symbolic check to collect assertions for this check node
    with context.check_context(node):
        _check(provider, context)
```

**How to test**:
```bash
# Test the decorator changes
uv run pytest tests/test_assertion_rename.py::test_check_decorator_with_name -v
uv run pytest tests/test_assertion_rename.py::test_check_decorator_without_name -v
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "refactor: update @check decorator to use name instead of label"
```

### Task 5: Update Graph Node Classes

**Goal**: Update CheckNode and AssertionNode to use `name` instead of `label`.

**File to modify**: `src/dqx/graph/nodes.py`

**Changes to make**:

1. Update `CheckNode` class (around line 140-180):

```python
class CheckNode(CompositeNode["AssertionNode"]):
    """
    Node representing a data quality check.

    CheckNode manages a collection of AssertionNode children and derives
    its state from their evaluation results.
    """

    def __init__(
        self,
        name: str,
        tags: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        """
        Initialize a check node.

        Args:
            name: Name for the check (either user-provided or function name)
            tags: Optional tags for categorizing the check
            datasets: Optional list of datasets this check applies to
        """
        super().__init__()
        self.name = name
        self.tags = tags or []
        self.datasets = datasets or []

    def node_name(self) -> str:
        """Get the display name of the node."""
        return self.name  # No more label fallback
```

2. Update `AssertionNode` class (around line 210-250):

```python
class AssertionNode(BaseNode):
    """
    Node representing an assertion to be evaluated.

    AssertionNodes are leaf nodes and cannot have children.
    """

    def __init__(
        self,
        actual: sp.Expr,
        name: str | None = None,  # WAS: label
        severity: SeverityLevel | None = None,
        validator: SymbolicValidator | None = None,
    ) -> None:
        """
        Initialize an assertion node.

        Args:
            actual: The symbolic expression to evaluate
            name: Optional human-readable description  # WAS: label
            severity: Optional severity level for failures
            validator: Optional validation function to apply
        """
        super().__init__()
        self.actual = actual
        self.name = name  # WAS: self.label = label
        self.severity = severity
        self.datasets: list[str] = []
        self.validator = validator
        self._value: Result[float, dict[SymbolicMetric | sp.Expr, str]]
```

3. Update any error messages that reference `self.label` to use `self.name`:

```python
# In AssertionNode.impute_datasets method
raise DQXError(
    f"The assertion {str(self.actual) or self.name} requires datasets {self.datasets} but got {datasets}"
)  # WAS: self.label
```

**How to test**:
```bash
# Run graph tests
uv run pytest tests/test_graph_display.py -v
uv run pytest tests/graph/test_base.py -v
```

**Commit**:
```bash
git add src/dqx/graph/nodes.py
git commit -m "refactor: update graph nodes to use name instead of label"
```

### Task 6: Update All Test Files

**Goal**: Update existing tests to use the new API.

**Files to modify**:
- `tests/test_api.py`
- `tests/e2e/test_api_e2e.py`
- `tests/test_display.py`
- `tests/test_graph_display.py`

**Changes to make**:

1. Replace all `.on(label=...)` with `.where(name=...)`
2. Replace all `@check(label=...)` with `@check(name=...)`
3. Replace all `CheckNode(..., label=...)` with `CheckNode(name=...)`
4. Replace all `AssertionNode(..., label=...)` with `AssertionNode(..., name=...)`

**Example changes**:

```python
# OLD
ctx.assert_that(metric).on(label="Greater than 40").is_gt(40)

# NEW
ctx.assert_that(metric).where(name="Greater than 40").is_gt(40)


# OLD
@check(label="Order validation", tags=["critical"])
def validate_orders(mp, ctx):
    pass


# NEW
@check(name="Order validation", tags=["critical"])
def validate_orders(mp, ctx):
    pass


# OLD
node = CheckNode("check1", label="Check One")

# NEW
node = CheckNode(name="Check One")
```

**How to find all occurrences**:
```bash
# Find all .on( occurrences
grep -r "\.on(" tests/ --include="*.py"

# Find all label= occurrences
grep -r "label=" tests/ --include="*.py"
```

**How to test**:
```bash
# Run all tests
uv run pytest tests/ -v

# If any fail, run them individually to debug
uv run pytest tests/test_api.py::specific_test_name -v
```

**Commit**:
```bash
git add tests/
git commit -m "test: update all tests to use new assertion interface"
```

### Task 7: Update Example Files

**Goal**: Update example code to use the new API.

**File to modify**: `examples/display_demo.py`

**Changes**: Same as test files - replace all occurrences of:
- `CheckNode(..., label=...)` → `CheckNode(name=...)`
- `AssertionNode(..., label=...)` → `AssertionNode(..., name=...)`

**How to test**:
```bash
# Run the example to ensure it works
uv run python examples/display_demo.py
```

**Commit**:
```bash
git add examples/
git commit -m "docs: update examples to use new assertion interface"
```

### Task 8: Update Documentation

**Goal**: Update all documentation to reflect the new API.

**Files to modify**:
- `README.md`
- `docs/design.md`
- `docs/graph_improvement_plan.md`
- `docs/plans/remove_listener_pattern_implementation.md`

**Changes**:
1. Replace all `@check(label=...)` with `@check(name=...)`
2. Replace all `.on(label=...)` with `.where(name=...)`
3. Update any API documentation sections

**Example**:
```markdown
# OLD
@check(label="Order validation", tags=["critical"])
def validate_orders(mp, ctx):
    ctx.assert_that(mp.null_count("id")).on(label="No null IDs").is_eq(0)

# NEW
@check(name="Order validation", tags=["critical"])
def validate_orders(mp, ctx):
    ctx.assert_that(mp.null_count("id")).where(name="No null IDs").is_eq(0)
```

**How to verify**:
```bash
# Search for any remaining old API usage
grep -r "\.on(" docs/ README.md --include="*.md"
grep -r "@check.*label=" docs/ README.md --include="*.md"
```

**Commit**:
```bash
git add README.md docs/
git commit -m "docs: update documentation to use new assertion interface"
```

### Task 9: Final Quality Checks

**Goal**: Ensure all changes are consistent and the codebase is clean.

**Steps**:

1. Run all tests:
```bash
uv run pytest tests/ -v
# All should pass!
```

2. Run linting:
```bash
uv run ruff check src/ tests/
uv run ruff check src/ tests/ --fix  # Fix any issues
```

3. Run type checking:
```bash
uv run mypy src/
```

4. Format code:
```bash
uv run ruff format src/ tests/
```

5. Delete the temporary test file:
```bash
rm tests/test_assertion_rename.py
git add -u
git commit -m "test: remove temporary assertion rename tests"
```

**Final commit**:
```bash
git add -A
git commit -m "chore: final cleanup for assertion interface rename"
```

## Testing the Complete Change

### Manual Testing Script

Create a temporary file to test the new API:

```python
# test_new_api.py
import pyarrow as pa
from dqx.api import check, VerificationSuiteBuilder
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import ResultKey
import datetime as dt


# Define a check using the new API
@check(name="Data validation check", tags=["test"])
def validate_data(mp, ctx):
    # Test the new where() method
    ctx.assert_that(mp.num_rows()).where(name="Has data").is_gt(0)
    ctx.assert_that(mp.null_count("value")).where(name="No nulls").is_eq(0)

    # Test without name (should still work)
    ctx.assert_that(mp.average("value")).is_gt(10)


# Create test data
data = pa.table({"id": [1, 2, 3, 4, 5], "value": [15, 20, 25, 30, 35]})

# Build and run suite
db = InMemoryMetricDB()
suite = VerificationSuiteBuilder("Test Suite", db).add_check(validate_data).build()

# Execute
ds = ArrowDataSource(data)
key = ResultKey(yyyy_mm_dd=dt.date.today())
context = suite.run({"test_data": ds}, key)

# Verify results
print("Test completed successfully!")
for assertion in context._graph.assertions():
    if assertion._value:
        print(f"✓ {assertion.name}: {assertion._value}")
```

Run this script:
```bash
uv run python test_new_api.py
# Should print success messages

# Clean up
rm test_new_api.py
```

## Common Issues and Solutions

### Issue 1: Import Errors
**Problem**: `ImportError: cannot import name 'on' from 'dqx.api'`
**Solution**: You haven't updated all imports. Search for any code that tries to import `on`.

### Issue 2: Test Failures
**Problem**: Tests fail with `AttributeError: 'AssertBuilder' object has no attribute 'on'`
**Solution**: You missed updating some test files. Search for `.on(` in the tests directory.

### Issue 3: Type Checking Errors
**Problem**: `mypy` reports type errors
**Solution**:
- Ensure all method signatures are updated consistently
- Check that Context methods match the node constructors

### Issue 4: Documentation Inconsistencies
**Problem**: Examples in docs don't match the code
**Solution**: Search documentation for both `.on(` and `label=` patterns

## Verification Checklist

Before considering the task complete, verify:

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/ tests/`
- [ ] Type checking passes: `uv run mypy src/`
- [ ] Documentation is updated: No occurrences of `.on(` or `@check(label=`
- [ ] Examples run successfully
- [ ] Manual test script works

## Migration Guide for Users

If you need to document this change for users, here's a template:

```markdown
# DQX API Update: Assertion Interface Changes

## What Changed

The assertion interface has been renamed for better consistency:
- `.on()` method is now `.where()`
- `label` parameter is now `name` throughout the API

## Migration Examples

### Before:
```python
@check(label="My Check")
def my_check(mp, ctx):
    ctx.assert_that(mp.num_rows()).on(label="Row count check").is_gt(0)
```

### After:
```python
@check(name="My Check")
def my_check(mp, ctx):
    ctx.assert_that(mp.num_rows()).where(name="Row count check").is_gt(0)
```

## Backwards Compatibility

This is a breaking change. All code using the old API must be updated.
```

## Summary

This implementation plan provides a comprehensive, step-by-step guide to rename the DQX assertion interface. The key principles followed are:

1. **TDD (Test-Driven Development)**: Start with failing tests
2. **Small, focused commits**: Each task results in a meaningful commit
3. **Incremental progress**: Changes build upon each other
4. **Comprehensive testing**: Verify each change works before moving on
5. **Clear documentation**: Update all docs and examples

The total effort should take approximately 2-4 hours for a developer familiar with Python but new to the DQX codebase.

Good luck with the implementation!
