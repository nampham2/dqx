# Implementation Plan: Make Assertion Severity Mandatory with P1 Default

## Overview
We're making assertion severity mandatory in the DQX data quality framework. Currently, severity can be `None` or one of the defined levels (P0-P3). After this change:
- Severity will be **required** (cannot be `None`)
- Default severity will be `"P1"`
- All assertions must have an explicit severity level

## Background Context

### What is DQX?
DQX is a data quality framework that allows users to write assertions (tests) about their data. Each assertion can have a severity level:
- **P0**: Critical (system breaking)
- **P1**: Important (should be fixed soon)
- **P2**: Medium priority
- **P3**: Low priority
- **None**: No severity (current default)

### Current Behavior
```python
# Currently, severity is optional and defaults to None
ctx.assert_that(metric).where(name="Check something").is_gt(0)  # severity=None

# Users can explicitly set severity or None
ctx.assert_that(metric).where(name="Check something", severity="P1").is_gt(0)
ctx.assert_that(metric).where(name="Check something", severity=None).is_gt(0)  # Allowed
```

### Desired Behavior
```python
# After our change, this will create an assertion with severity="P1"
ctx.assert_that(metric).where(name="Check something").is_gt(0)  # severity="P1" (default)

# Users can override to any valid severity
ctx.assert_that(metric).where(name="Check something", severity="P0").is_gt(0)
ctx.assert_that(metric).where(name="Check something", severity="P2").is_gt(0)

# This will NO LONGER be allowed - will cause type error
# ctx.assert_that(metric).where(name="Check something", severity=None).is_gt(0)  # ERROR!
```

## Development Environment Setup

### Tools You'll Use
- **uv**: Python package manager (like pip but faster)
- **pytest**: Testing framework
- **mypy**: Type checker
- **ruff**: Linter and formatter

### Commands Reference
```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_api.py

# Run tests with coverage
uv run pytest --cov=dqx

# Type checking
uv run mypy src/

# Linting and auto-fix
uv run ruff check src/ --fix

# Code formatting
uv run ruff format src/ tests/

# Run pre-commit hooks
./bin/run-hooks.sh --fast  # Skip slow checks like mypy
./bin/run-hooks.sh --all   # Run all checks
```

## Implementation Tasks

### Task 1: Write Tests First (TDD)
**File to create/modify:** `tests/test_api.py`

**What to do:**
1. Add a new test function that verifies the default severity behavior
2. The test should check that assertions created without explicit severity get "P1"
3. Also test that explicit severity still works (including None)

**Code to add:**
```python
def test_assertion_severity_is_mandatory_with_p1_default():
    """Test that assertions require severity and default to P1."""
    # Create a mock context and provider
    db = InMemoryMetricDB()
    context = Context(suite="test_suite", db=db)

    # Create a check to hold our assertions
    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        # Create assertion without severity - should default to P1
        ctx.assert_that(sp.Symbol("x")).where(name="Default severity test").is_gt(0)

        # Create assertion with explicit P0 severity
        ctx.assert_that(sp.Symbol("y")).where(name="Explicit P0 test", severity="P0").is_gt(0)

        # Create assertion with explicit P2 severity
        ctx.assert_that(sp.Symbol("z")).where(name="Explicit P2 test", severity="P2").is_gt(0)

    # Execute the check
    test_check(context.provider, context)

    # Get the check node and its assertions
    check_node = list(context._graph.root.children)[0]
    assertions = list(check_node.children)

    # Verify severities
    assert len(assertions) == 3
    assert assertions[0].severity == "P1"  # Default
    assert assertions[1].severity == "P0"  # Explicit P0
    assert assertions[2].severity == "P2"  # Explicit P2

    # Verify that severity is never None
    for assertion in assertions:
        assert assertion.severity is not None
        assert assertion.severity in ["P0", "P1", "P2", "P3"]
```

**Run the test (it should FAIL):**
```bash
uv run pytest tests/test_api.py::test_assertion_severity_is_mandatory_with_p1_default -v
```

**Expected output:** Test should fail because the default is still None

**Commit:**
```bash
git add tests/test_api.py
git commit -m "test: add failing test for P1 default severity"
```

### Task 2: Implement the Core Changes
**Files to modify:**
1. `src/dqx/api.py` - Multiple locations
2. `src/dqx/graph/nodes.py` - AssertionNode class

**Changes in `src/dqx/api.py`:**

**Change 1 - AssertionDraft.where() method (around line 49):**
```python
# BEFORE
def where(self, *, name: str, severity: SeverityLevel | None = None) -> AssertionReady:

# AFTER
def where(self, *, name: str, severity: SeverityLevel = "P1") -> AssertionReady:
```

**Change 2 - AssertionReady.__init__() method (around line 78):**
```python
# BEFORE
def __init__(
    self, actual: sp.Expr, name: str, severity: SeverityLevel | None = None, context: Context | None = None
) -> None:

# AFTER
def __init__(
    self, actual: sp.Expr, name: str, severity: SeverityLevel = "P1", context: Context | None = None
) -> None:
```

**Change 3 - Context.create_assertion() method (around line 245):**
```python
# BEFORE
def create_assertion(
    self,
    actual: sp.Expr,
    name: str | None = None,
    severity: SeverityLevel | None = None,
    validator: SymbolicValidator | None = None,
) -> AssertionNode:

# AFTER
def create_assertion(
    self,
    actual: sp.Expr,
    name: str | None = None,
    severity: SeverityLevel = "P1",
    validator: SymbolicValidator | None = None,
) -> AssertionNode:
```

**Changes in `src/dqx/graph/nodes.py`:**

**Change 4 - AssertionNode.__init__() method:**
```python
# BEFORE
def __init__(
    self,
    actual: sp.Expr,
    name: str | None = None,
    severity: SeverityLevel | None = None,
    validator: SymbolicValidator | None = None,
) -> None:

# AFTER
def __init__(
    self,
    actual: sp.Expr,
    name: str | None = None,
    severity: SeverityLevel = "P1",
    validator: SymbolicValidator | None = None,
) -> None:
```

**Run the test again (it should PASS):**
```bash
uv run pytest tests/test_api.py::test_assertion_severity_is_mandatory_with_p1_default -v
```

**Run ALL tests to ensure nothing broke:**
```bash
uv run pytest
```

**Commit:**
```bash
git add src/dqx/api.py
git commit -m "feat: change default assertion severity to P1"
```

### Task 3: Update Documentation
**Files to modify:**
1. `src/dqx/api.py` - Update docstring
2. `README.md` - Update examples if needed

**Docstring updates in `src/dqx/api.py`:**

1. **AssertionDraft.where() docstring:**
```python
def where(self, *, name: str, severity: SeverityLevel = "P1") -> AssertionReady:
    """
    Provide a descriptive name for this assertion.

    Args:
        name: Required description of what this assertion validates
        severity: Severity level (P0, P1, P2, P3). Defaults to "P1".
                 All assertions must have a severity level.

    Returns:
        AssertionReady instance with all assertion methods available

    Raises:
        ValueError: If name is empty or too long
    """
```

2. **Context.create_assertion() docstring:**
```python
def create_assertion(
    self,
    actual: sp.Expr,
    name: str | None = None,
    severity: SeverityLevel = "P1",
    validator: SymbolicValidator | None = None,
) -> AssertionNode:
    """
    Factory method to create an assertion node.

    Args:
        actual: Symbolic expression to evaluate
        name: Optional human-readable description
        severity: Severity level for failures (P0, P1, P2, P3). Defaults to "P1".
        validator: Optional validator function

    Returns:
        AssertionNode that can access context through its root node
    """
```

**Check README.md for any examples that might need updating:**
```bash
# Search for severity mentions in README
grep -n "severity" README.md
```

If you find examples showing `severity=None` as default behavior, update them.

**Commit:**
```bash
git add src/dqx/api.py README.md
git commit -m "docs: update documentation for P1 default severity"
```

### Task 4: Run Quality Checks
**What to do:** Run all code quality tools to ensure everything is clean

```bash
# Type checking
uv run mypy src/

# Linting
uv run ruff check src/ tests/

# Auto-fix any issues
uv run ruff check src/ tests/ --fix

# Format code
uv run ruff format src/ tests/
```

**If any files were changed by the tools, commit them:**
```bash
git add -u
git commit -m "style: apply linting and formatting fixes"
```

### Task 5: Run Full Test Suite with Coverage
**What to do:** Ensure all tests pass and coverage remains high

```bash
# Run all tests with coverage
uv run pytest --cov=dqx

# Run E2E tests specifically
uv run pytest tests/e2e/ -v
```

**Expected:** All tests should pass, coverage should remain at or above previous levels

### Task 5: Fix Tests That Use severity=None
**Files to check and potentially modify:**
- `tests/test_api.py`
- `tests/test_display.py`
- `tests/test_graph_display.py`

**What to do:**
Search for any tests that explicitly use `severity=None` and update them:

```bash
# Find tests that might need updating
grep -r "severity=None" tests/
```

If you find any, either:
1. Remove the `severity=None` parameter (to use default P1)
2. Change to a valid severity like `severity="P2"`

**Example fix:**
```python
# BEFORE
assertion = AssertionNode(actual=symbol, severity=None)

# AFTER (option 1 - use default)
assertion = AssertionNode(actual=symbol)

# AFTER (option 2 - explicit severity)
assertion = AssertionNode(actual=symbol, severity="P2")
```

**Commit if any changes were made:**
```bash
git add tests/
git commit -m "test: update tests to remove severity=None usage"
```

### Task 6: Manual Verification
**What to do:** Create a simple script to manually verify the behavior

**Create `verify_severity_change.py` in the project root:**
```python
"""Manual verification script for severity default change."""
import pyarrow as pa
from dqx.api import VerificationSuiteBuilder, check
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import ResultKey
import datetime as dt

# Define a check without explicit severity
@check(name="Verification Check")
def verify_check(mp, ctx):
    ctx.assert_that(mp.num_rows()).where(name="Row count check").is_gt(0)

# Create test data
data = pa.table({"value": [1, 2, 3, 4, 5]})

# Run the check
db = InMemoryMetricDB()
suite = VerificationSuiteBuilder("Verify Suite", db).add_check(verify_check).build()

# Execute
data_source = ArrowDataSource(data)
key = ResultKey(yyyy_mm_dd=dt.date.today(), tags={})
suite.run({"test": data_source}, key)

# Print assertion details
context = suite._context
for check_node in context._graph.root.children:
    print(f"Check: {check_node.name}")
    for assertion in check_node.children:
        print(f"  - Assertion: {assertion.name}")
        print(f"    Severity: {assertion.severity}")  # Should print "P1"
```

**Run it:**
```bash
uv run python verify_severity_change.py
```

**Expected output:**
```
Check: Verification Check
  - Assertion: Row count check
    Severity: P1
```

**Clean up (don't commit this file):**
```bash
rm verify_severity_change.py
```

### Task 7: Final Commit
**What to do:** Create a summary commit if needed

```bash
# Check what's been done
git log --oneline -5

# If everything looks good, you're done!
```

## Testing Checklist
- [ ] New test for mandatory severity passes
- [ ] All existing tests still pass (after fixing severity=None usage)
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] Code coverage maintained
- [ ] E2E tests pass
- [ ] Manual verification shows P1 default and no None values

## Common Issues and Solutions

### Issue: Import errors in tests
**Solution:** Make sure you import all necessary modules:
```python
from dqx.api import Context, check
from dqx.provider import MetricProvider
from dqx.orm.repositories import InMemoryMetricDB
import sympy as sp
```

### Issue: Type checking fails
**Solution:** The severity parameter type is now just `SeverityLevel` (no None). Make sure:
1. All severity parameters are typed as `SeverityLevel` not `SeverityLevel | None`
2. Default value "P1" is a valid `SeverityLevel`:
```python
# This is already defined in src/dqx/common.py
SeverityLevel = Literal["P0", "P1", "P2", "P3"]
```

### Issue: Tests fail with "None is not a valid severity"
**Solution:** Find and update any test that passes `severity=None`:
```python
# Search for the issue
grep -r "severity=None" tests/

# Fix by removing the parameter or using a valid severity
```

### Issue: Tests fail unexpectedly
**Solution:** Debug by printing the actual values:
```python
print(f"Expected: P1, Actual: {assertion.severity}")
```

## Summary
This change makes severity mandatory for all assertions with P1 as the default. This ensures every assertion has an explicit importance level, making the data quality system more structured and preventing ambiguous "no severity" states.

**Breaking change**: Code that previously used `severity=None` will need to be updated.

Total estimated time: 45-60 minutes
Number of files changed: 3-5
Risk level: Medium (breaking API change, but straightforward to fix)
