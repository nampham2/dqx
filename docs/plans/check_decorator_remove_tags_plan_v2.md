# Implementation Plan: Remove tags Parameter from @check Decorator (v2)

## Overview

This plan details the complete removal of the unused `tags` parameter from the `@check` decorator and all related components in the DQX codebase. The `tags` parameter is currently accepted but never used for any functionality, adding unnecessary complexity to the API.

## Scope of Changes

The `tags` parameter needs to be removed from:
1. Core graph node classes (`CheckNode`, `RootNode`)
2. API layer (`@check` decorator, `_create_check` function)
3. Test files that use tags
4. Documentation and code examples

## Implementation Strategy

The implementation follows TDD principles and is organized into 5 task groups, each independently committable with passing tests and no linting issues.

---

## Task Group 1: Update Core Graph Classes

**Objective**: Remove tags parameter from CheckNode and RootNode classes

### Task 1.1: Update CheckNode class

**File**: `src/dqx/graph/nodes.py`

**Current code (lines 79-95)**:
```python
class CheckNode(CompositeNode["RootNode", "AssertionNode"]):
    """Node representing a data quality check.

    Parent type is RootNode (never None).
    Child type is AssertionNode.
    """

    def __init__(
        self,
        parent: RootNode,
        name: str,
        tags: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        """Initialize a check node.

        Args:
            parent: The RootNode parent (required)
            name: Name for the check
            tags: Optional tags for categorizing the check
            datasets: Optional list of datasets this check applies to
        """
        super().__init__(parent)
        self.name = name
        self.tags = tags or []
        self.datasets = datasets or []
```

**Change to**:
```python
class CheckNode(CompositeNode["RootNode", "AssertionNode"]):
    """Node representing a data quality check.

    Parent type is RootNode (never None).
    Child type is AssertionNode.
    """

    def __init__(
        self,
        parent: RootNode,
        name: str,
        datasets: list[str] | None = None,
    ) -> None:
        """Initialize a check node.

        Args:
            parent: The RootNode parent (required)
            name: Name for the check
            datasets: Optional list of datasets this check applies to
        """
        super().__init__(parent)
        self.name = name
        self.datasets = datasets or []
```

### Task 1.2: Update RootNode.add_check() method

**File**: `src/dqx/graph/nodes.py`

**Current code (lines 48-68)**:
```python
def add_check(
    self,
    name: str,
    tags: list[str] | None = None,
    datasets: list[str] | None = None,
) -> CheckNode:
    """Factory method to create and add a check node.

    This ensures the check has the correct parent type.

    Args:
        name: Name for the check
        tags: Optional tags for categorizing the check
        datasets: Optional list of datasets this check applies to

    Returns:
        The newly created CheckNode
    """
    check = CheckNode(parent=self, name=name, tags=tags, datasets=datasets)
    self.add_child(check)
    return check
```

**Change to**:
```python
def add_check(
    self,
    name: str,
    datasets: list[str] | None = None,
) -> CheckNode:
    """Factory method to create and add a check node.

    This ensures the check has the correct parent type.

    Args:
        name: Name for the check
        datasets: Optional list of datasets this check applies to

    Returns:
        The newly created CheckNode
    """
    check = CheckNode(parent=self, name=name, datasets=datasets)
    self.add_child(check)
    return check
```

### Task 1.3: Verification

Run the following commands to verify changes:
```bash
uv run mypy src/dqx/graph/nodes.py
uv run ruff check src/dqx/graph/nodes.py
uv run pytest tests/graph/test_typed_parents.py -v
```

**Expected**: All checks pass. Tests may fail due to tags usage - will be fixed in Task Group 3.

---

## Task Group 2: Update API Layer

**Objective**: Remove tags parameter from the API functions and decorator

### Task 2.1: Update _create_check function

**File**: `src/dqx/api.py`

**Current code (lines 686-707)**:
```python
def _create_check(
    provider: MetricProvider,
    context: Context,
    _check: CheckProducer,
    name: str,
    tags: list[str] = [],
    datasets: list[str] | None = None,
) -> None:
    """
    Internal function to create and register a check node in the context graph.

    Args:
        provider: Metric provider for the check
        context: Execution context
        _check: Check function to execute
        tags: Optional tags for the check
        display_name: Optional human-readable name
        datasets: Optional list of datasets the check applies to

    Raises:
        DQXError: If a check with the same name already exists
    """
    # Create the check node using root's factory method
    # This will automatically add it to the root and set the parent
    node = context._graph.root.add_check(name=name, tags=tags, datasets=datasets)

    # Call the symbolic check to collect assertions for this check node
    with context.check_context(node):
        _check(provider, context)
```

**Change to**:
```python
def _create_check(
    provider: MetricProvider,
    context: Context,
    _check: CheckProducer,
    name: str,
    datasets: list[str] | None = None,
) -> None:
    """
    Internal function to create and register a check node in the context graph.

    Args:
        provider: Metric provider for the check
        context: Execution context
        _check: Check function to execute
        name: Name for the check
        datasets: Optional list of datasets the check applies to

    Raises:
        DQXError: If a check with the same name already exists
    """
    # Create the check node using root's factory method
    # This will automatically add it to the root and set the parent
    node = context._graph.root.add_check(name=name, datasets=datasets)

    # Call the symbolic check to collect assertions for this check node
    with context.check_context(node):
        _check(provider, context)
```

### Task 2.2: Update check decorator

**File**: `src/dqx/api.py`

**Current code (lines 711-743)**:
```python
def check(
    *,
    name: str,
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

    Raises:
        TypeError: If called without the required 'name' parameter
    """

    def decorator(fn: CheckProducer) -> DecoratedCheck:
        wrapped = functools.wraps(fn)(
            functools.partial(_create_check, _check=fn, name=name, tags=tags, datasets=datasets)
        )
        # No metadata storage needed anymore
        return cast(DecoratedCheck, wrapped)

    return decorator
```

**Change to**:
```python
def check(
    *,
    name: str,
    datasets: list[str] | None = None,
) -> Callable[[CheckProducer], DecoratedCheck]:
    """
    Decorator for creating data quality check functions.

    Must be used with parentheses and a name:

    @check(name="Important Check", datasets=["ds1"])
    def my_labeled_check(mp: MetricProvider, ctx: Context) -> None:
        # check logic

    Args:
        name: Human-readable name for the check (required)
        datasets: Optional list of datasets the check applies to

    Returns:
        Decorated check function

    Raises:
        TypeError: If called without the required 'name' parameter
    """

    def decorator(fn: CheckProducer) -> DecoratedCheck:
        wrapped = functools.wraps(fn)(
            functools.partial(_create_check, _check=fn, name=name, datasets=datasets)
        )
        # No metadata storage needed anymore
        return cast(DecoratedCheck, wrapped)

    return decorator
```

### Task 2.3: Verification

Run the following commands:
```bash
uv run mypy src/dqx/api.py
uv run ruff check src/dqx/api.py
uv run pytest tests/test_api.py::test_parametrized_check_uses_provided_name -v
```

**Expected**: Type checking and linting pass. Some tests may fail due to tags usage.

---

## Task Group 3: Update Test Files

**Objective**: Remove all usage of tags parameter in test files

### Task 3.1: Update test_api.py

**File**: `tests/test_api.py`

**Line 121 - Remove tags from decorator**:
```python
# BEFORE:
@check(name="Order Validation Check", tags=["critical"])
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

# AFTER:
@check(name="Order Validation Check")
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)
```

### Task 3.2: Update test_api_coverage.py

**File**: `tests/test_api_coverage.py`

**Lines 164-167 - Remove tags from check decorator**:
```python
# BEFORE:
@check(name="Check 2", tags=["important"])
def check2(mp: MetricProvider, ctx: Context) -> None:
    # This will create a metric for ds2
    ctx.assert_that(mp.average("col2")).where(name="Avg col2").is_gt(10)

# AFTER:
@check(name="Check 2")
def check2(mp: MetricProvider, ctx: Context) -> None:
    # This will create a metric for ds2
    ctx.assert_that(mp.average("col2")).where(name="Avg col2").is_gt(10)
```

**Lines 185-189 - Remove tags verification**:
```python
# BEFORE:
# Verify all checks were added
checks = list(suite.graph.checks())
assert len(checks) == 3
assert {c.name for c in checks} == {"Check 1", "Check 2", "Check 3"}

# Verify tags and datasets
check2_node = next(c for c in checks if c.name == "Check 2")
assert check2_node.tags == ["important"]

check3_node = next(c for c in checks if c.name == "Check 3")
assert check3_node.datasets == ["orders"]

# AFTER:
# Verify all checks were added
checks = list(suite.graph.checks())
assert len(checks) == 3
assert {c.name for c in checks} == {"Check 1", "Check 2", "Check 3"}

# Verify datasets
check3_node = next(c for c in checks if c.name == "Check 3")
assert check3_node.datasets == ["orders"]
```

### Task 3.3: Update test_typed_parents.py

**File**: `tests/graph/test_typed_parents.py`

**Lines 52-55 - Remove tags from add_check call**:
```python
# BEFORE:
# Use factory method to create check
check = root.add_check("my_check", tags=["important"])
assert check.parent is root
assert check in root.children

# AFTER:
# Use factory method to create check
check = root.add_check("my_check")
assert check.parent is root
assert check in root.children
```

### Task 3.4: Verification

Run all affected tests:
```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_api_coverage.py -v
uv run pytest tests/graph/test_typed_parents.py -v
```

**Expected**: All tests pass.

---

## Task Group 4: Update Documentation and Examples

**Objective**: Remove tags references from documentation and code examples

### Task 4.1: Update Graph traversal documentation

**File**: `src/dqx/graph/traversal.py`

**Update docstring example (around lines 287-291)**:
```python
# BEFORE (in Graph class docstring):
>>> all_checks = list(graph.checks())
>>> for check in all_checks:
...     print(f"Check: {check.name}, Tags: {check.tags}")
>>>
>>> tag_counts = Counter(
...     tag for check in all_checks for tag in check.tags
... )

# AFTER:
>>> all_checks = list(graph.checks())
>>> for check in all_checks:
...     print(f"Check: {check.name}")
```

### Task 4.2: Verification

```bash
uv run mypy src/dqx/graph/traversal.py
uv run ruff check src/dqx/graph/traversal.py
```

**Expected**: All checks pass.

---

## Task Group 5: Final Verification and Commit

**Objective**: Ensure all changes work together and create final commit

### Task 5.1: Run full test suite

```bash
uv run pytest tests/ -v
```

**Expected**: All tests pass.

### Task 5.2: Run type checking on all modified files

```bash
uv run mypy src/dqx/graph/nodes.py src/dqx/api.py src/dqx/graph/traversal.py
```

**Expected**: No type errors.

### Task 5.3: Run linting and auto-fix

```bash
uv run ruff check --fix src/dqx/graph/nodes.py src/dqx/api.py src/dqx/graph/traversal.py
```

**Expected**: No linting errors.

### Task 5.4: Run pre-commit hooks

```bash
bin/run-hooks.sh
```

**Expected**: All hooks pass.

### Task 5.5: Run code coverage

```bash
uv run pytest tests/ -v --cov=dqx --cov-report=term-missing
```

**Expected**: Coverage remains at 100%.

### Task 5.6: Create final commit

```bash
git add -A
git commit -m "refactor: remove tags parameter from @check decorator

- Remove tags parameter from CheckNode class constructor
- Remove tags parameter from RootNode.add_check() method
- Remove tags parameter from @check decorator and _create_check function
- Update all test files to remove tags usage
- Update documentation to remove tags references

The tags parameter was unused and added unnecessary complexity to the API.
This is a breaking change for any code using @check(tags=[...])."
```

---

## Breaking Changes

This change is breaking for any code that uses the `tags` parameter in:
- `@check(name="...", tags=[...])`
- Direct instantiation of `CheckNode` with tags
- Direct calls to `RootNode.add_check()` with tags

## Migration Guide

For users upgrading:
1. Remove the `tags` parameter from all `@check` decorators
2. If tags were used for organization, consider using the check name or datasets parameter instead
3. No functional changes are needed as tags were never used in the evaluation logic

## Notes for Implementation

- Each task group should be completed and verified before moving to the next
- If any test fails during a task group, fix it before proceeding
- The changes are organized to minimize risk: core classes first, then API, then tests
- All changes should be made in a single commit after all verifications pass
