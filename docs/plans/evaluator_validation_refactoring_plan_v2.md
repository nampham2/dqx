# Evaluator Validation Refactoring Implementation Plan v2

## Overview

### What We're Changing
Currently, our data quality framework (DQX) only checks if metrics can be computed from data. It doesn't validate if those metrics meet the assertion criteria. This refactoring adds the missing validation step.

**Example of the problem:**
```python
# Current behavior
ctx.assert_that(mp.average("price")).where(name="Price is positive").is_gt(0)
# This only checks if we can calculate average(price), not if it's actually > 0!
```

### Key Changes
1. Rename `AssertionNode._value` → `AssertionNode._metric` (clarity)
2. Add `AssertionNode._result` to store validation outcome ("OK" or "FAILURE")
3. Update evaluator to apply validator functions after computing metrics
4. Fix result collection to report actual assertion pass/fail status
5. Show complete validation expressions (e.g., "average(price) > 0")

### Breaking Changes
⚠️ **No backward compatibility** - This changes the API structure for `AssertionResult`.

## Critical Coding Guidelines

### Comment Rules
**IMPORTANT**: All code comments must describe WHAT the code does or WHY it exists, never the refactoring history.

❌ **NEVER write comments like these:**
```python
# Changed from _value to _metric for clarity
self._metric: Result[float, list[EvaluationFailure]]

# Now we also validate the assertion, not just compute the metric
match node._metric:
    case Success(value):
        # New validation step added in refactoring
        passed = node.validator.fn(value)

# Using the new field name
metric = (assertion._metric,)  # was: value
```

✅ **ALWAYS write comments like these:**
```python
# Stores the computed metric result
self._metric: Result[float, list[EvaluationFailure]]

# Apply validator to determine if assertion passes
match node._metric:
    case Success(value):
        # validator.fn returns True if assertion passes
        passed = node.validator.fn(value)

# The metric computation result
metric = (assertion._metric,)
```

Remember: Git tracks history. Comments explain the present.

## Prerequisites

### Development Environment
```bash
# Ensure you have the project set up
cd /Users/npham/git-tree/dqx

# Create a new branch for this work
git checkout -b feat/evaluator-validation-refactoring

# Run tests to ensure starting from clean state
uv run pytest tests/ -v

# Check type hints
uv run mypy src/

# Check linting
uv run ruff check src/
```

### Key Files You'll Touch
- `src/dqx/common.py` - Add type definitions
- `src/dqx/graph/nodes.py` - Update AssertionNode
- `src/dqx/evaluator.py` - Add validation logic
- `src/dqx/api.py` - Update result collection
- `examples/result_collection_demo.py` - Update demo
- Various test files (detailed below)

## Implementation Tasks

### Task 1: Add AssertionStatus Type Alias

**File:** `src/dqx/common.py`

**What to do:**
1. Add the new type alias near other type definitions (around line 70)
2. This represents the validation outcome

**Code to add:**
```python
# Add after the existing type aliases (near SeverityLevel)
AssertionStatus = Literal["OK", "FAILURE"]
```

**Test it:**
```bash
# Just check imports work
uv run python -c "from dqx.common import AssertionStatus; print(AssertionStatus)"
```

**Commit:**
```bash
git add src/dqx/common.py
git commit -m "feat: add AssertionStatus type alias for validation results"
```

### Task 2: Update AssertionNode Fields

**File:** `src/dqx/graph/nodes.py`

**What to do:**
1. Find the `AssertionNode` class (around line 127)
2. Rename `_value` to `_metric`
3. Add new `_result` field
4. Import the new type

**Changes:**
```python
# At the top, add to imports
from dqx.common import (
    AssertionStatus,
    EvaluationFailure,
    SeverityLevel,
    SymbolicValidator,
)

# In AssertionNode.__init__, change:
# OLD:
# self._value: Result[float, list[EvaluationFailure]]

# NEW:
# Stores the computed metric result
self._metric: Result[float, list[EvaluationFailure]]
# Stores whether the assertion passes validation
self._result: AssertionStatus
```

**Important:** Don't initialize these fields - they're set by the evaluator later.

**Test it:**
```bash
# This will fail some tests - that's expected!
uv run pytest tests/test_api.py::test_assertion_node_is_immutable -v
```

**Commit:**
```bash
git add src/dqx/graph/nodes.py
git commit -m "refactor: rename AssertionNode._value to _metric and add _result field"
```

### Task 3: Update Evaluator to Validate Assertions

**File:** `src/dqx/evaluator.py`

**What to do:**
1. Update the `visit` method to apply validators
2. Use pattern matching for clean code
3. Store both metric and validation results

**Changes:**
```python
# Add to imports at the top
from dqx.common import AssertionStatus, DQXError


# Replace the visit method (around line 240):
def visit(self, node: BaseNode) -> None:
    """Visit a node in the DQX graph and evaluate assertions.

    For AssertionNodes:
    1. Evaluates the metric expression
    2. Applies the validator function if metric succeeds
    3. Stores both metric result and validation status
    """
    if isinstance(node, AssertionNode):
        # Evaluate the metric
        node._metric = self.evaluate(node.actual)

        # Apply validator to determine pass/fail
        match node._metric:
            case Success(value):
                try:
                    # validator.fn returns True if assertion passes
                    passed = node.validator.fn(value)
                    node._result = "OK" if passed else "FAILURE"
                except Exception as e:
                    raise DQXError(f"Validator execution failed: {str(e)}") from e
            case Failure(_):
                # If metric computation failed, assertion fails
                node._result = "FAILURE"
```

**Test the evaluator in isolation:**
```bash
# Run evaluator tests
uv run pytest tests/test_evaluator.py -v

# These will fail - we need to update them!
```

**Commit:**
```bash
git add src/dqx/evaluator.py
git commit -m "feat: add validation step to evaluator with pattern matching"
```

### Task 4: Update AssertionResult Dataclass

**File:** `src/dqx/common.py`

**What to do:**
1. Find `AssertionResult` dataclass (around line 26)
2. Rename `value` field to `metric`
3. Change status type to use AssertionStatus
4. Update the docstring

**Changes:**
```python
# Add to imports at the top (if not already there)
from typing import Literal


@dataclass
class AssertionResult:
    """Result of a single assertion evaluation.

    This dataclass captures the complete state of an assertion after evaluation,
    including its location in the hierarchy (suite/check/assertion), the actual
    result value, and any error information if the assertion failed.

    Attributes:
        yyyy_mm_dd: Date from the ResultKey used during evaluation
        suite: Name of the verification suite
        check: Name of the parent check
        assertion: Name of the assertion (always present, names are mandatory)
        severity: Priority level (P0, P1, P2, P3)
        status: Validation result ("OK" or "FAILURE")
        metric: The metric computation result (Success with value or Failure with errors)
        expression: Full validation expression (e.g., "average(price) > 0")
        tags: Tags from the ResultKey (e.g., {"env": "prod"})
    """

    yyyy_mm_dd: datetime.date
    suite: str
    check: str
    assertion: str
    severity: SeverityLevel
    status: AssertionStatus  # Uses the type we defined in Task 1
    metric: Result[float, list[EvaluationFailure]]  # The metric computation result
    expression: str | None = None
    tags: Tags = field(default_factory=dict)
```

**Commit:**
```bash
git add src/dqx/common.py
git commit -m "refactor: update AssertionResult to use metric field and AssertionStatus type"
```

### Task 5: Update collect_results in VerificationSuite

**File:** `src/dqx/api.py`

**What to do:**
1. Find `collect_results` method (around line 510)
2. Update to use new field names
3. Use assertion validation result for status
4. Construct full validation expression

**Changes:**
```python
def collect_results(self) -> list[AssertionResult]:
    """Collect all assertion results after suite execution.

    This method traverses the evaluation graph and extracts results from
    all assertions, converting them into AssertionResult objects suitable
    for persistence or reporting. The ResultKey used during run() is
    automatically applied to all results.

    Returns:
        List of AssertionResult instances, one for each assertion in the suite.
        Results are returned in graph traversal order (breadth-first).

    Raises:
        DQXError: If called before run() has been executed successfully.

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> results = suite.collect_results()  # No key needed!
        >>> for r in results:
        ...     print(f"{r.check}/{r.assertion}: {r.status}")
        ...     if r.status == "FAILURE":
        ...         failures = r.metric.failure()
        ...         for f in failures:
        ...             print(f"  Error: {f.error_message}")
    """
    if not self.is_evaluated:
        raise DQXError(
            "Cannot collect results before suite execution. Call run() first to evaluate assertions."
        )

    if self._key is None:
        raise DQXError(
            "No ResultKey available. This should not happen after successful run()."
        )

    key = self._key
    results = []

    for assertion in self._context._graph.assertions():
        check_node = assertion.parent

        assertion_name = assertion.name
        if assertion_name is None:
            # This shouldn't happen with the new where() API
            assertion_name = f"Unnamed assertion ({str(assertion.actual)[:50]})"

        # Construct full validation expression
        expression = f"{assertion.actual} {assertion.validator.name}"

        result = AssertionResult(
            yyyy_mm_dd=key.yyyy_mm_dd,
            suite=self._name,
            check=check_node.name,
            assertion=assertion_name,
            severity=assertion.severity,
            status=assertion._result,  # Direct use of AssertionStatus
            metric=assertion._metric,
            expression=expression,
            tags=key.tags,
        )
        results.append(result)

    return results
```

**Commit:**
```bash
git add src/dqx/api.py
git commit -m "fix: update collect_results to use validation status and full expressions"
```

### Task 6: Fix Broken Tests

Now we need to update all tests that reference the old field names and status values.

**Files to update:**
1. `tests/test_api.py`
2. `tests/test_evaluator.py`
3. Any other test that uses `assertion._value` or `result.value`

**Common changes needed:**
```python
# OLD:
assertion._value
result.value
result.status == "SUCCESS"

# NEW:
assertion._metric
result.metric
result.status == "OK"
```

**Find all occurrences:**
```bash
# Find files that need updating
grep -r "_value" tests/ | grep -E "(assertion|node)\._value"
grep -r "\.value" tests/ | grep "result\.value"
grep -r "SUCCESS" tests/ | grep "status"
```

**Run tests incrementally:**
```bash
# Start with evaluator tests
uv run pytest tests/test_evaluator.py -v

# Then API tests
uv run pytest tests/test_api.py -v

# Then integration tests
uv run pytest tests/test_api_validation_integration.py -v

# Finally, all tests
uv run pytest tests/ -v
```

**Commit after each file:**
```bash
git add tests/test_evaluator.py
git commit -m "test: update evaluator tests for new field names and status values"

git add tests/test_api.py
git commit -m "test: update API tests for new field names and status values"
```

### Task 7: Write New Tests for Validation Logic

**File:** Create `tests/test_assertion_validation.py`

**What to test:**
1. Metric succeeds, validation passes → status = "OK"
2. Metric succeeds, validation fails → status = "FAILURE"
3. Metric fails → status = "FAILURE"
4. Validator throws exception → DQXError
5. Expression format is correct

**Example test structure:**
```python
import datetime
import pytest
import sympy as sp
from returns.result import Success, Failure

from dqx.api import Context, VerificationSuite, check
from dqx.common import DQXError, ResultKey, EvaluationFailure
from dqx.orm.repositories import InMemoryMetricDB
from dqx.extensions.pyarrow_ds import ArrowDataSource
import pyarrow as pa


def test_assertion_passes_when_metric_and_validation_succeed():
    """Test that assertion status is OK when both metric computation and validation pass."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp, ctx):
        # This should pass: 5.0 > 0
        ctx.assert_that(mp.average("value")).where(name="Average is positive").is_gt(0)

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create test data
    data = pa.table({"value": [5.0, 5.0, 5.0]})
    ds = ArrowDataSource(data)

    # Run suite
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run({"test": ds}, key)

    # Collect results
    results = suite.collect_results()
    assert len(results) == 1
    assert results[0].status == "OK"  # Note: "OK", not "SUCCESS"
    assert results[0].metric.is_ok()  # Metric computation succeeded
    assert results[0].metric.unwrap() == 5.0
    assert results[0].expression == "average(value) > 0"  # Full expression


def test_assertion_fails_when_validation_fails():
    """Test that assertion status is FAILURE when metric succeeds but validation fails."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp, ctx):
        # This should fail: -5.0 is not > 0
        ctx.assert_that(mp.average("value")).where(name="Average is positive").is_gt(0)

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create test data with negative values
    data = pa.table({"value": [-5.0, -5.0, -5.0]})
    ds = ArrowDataSource(data)

    # Run suite
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run({"test": ds}, key)

    # Collect results
    results = suite.collect_results()
    assert len(results) == 1
    assert results[0].status == "FAILURE"  # Validation failed
    assert results[0].metric.is_ok()  # But metric computation succeeded
    assert results[0].metric.unwrap() == -5.0
    assert results[0].expression == "average(value) > 0"


def test_assertion_fails_when_metric_fails():
    """Test that assertion status is FAILURE when metric computation fails."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp, ctx):
        # Create expression that will fail: division by zero
        zero_sum = mp.sum("zeros")  # Column of zeros
        reciprocal = sp.Integer(1) / zero_sum
        ctx.assert_that(reciprocal).where(name="Reciprocal calculation").is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create test data with zeros
    data = pa.table({"zeros": [0.0, 0.0, 0.0]})
    ds = ArrowDataSource(data)

    # Run suite
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run({"test": ds}, key)

    # Collect results
    results = suite.collect_results()
    assert len(results) == 1
    assert results[0].status == "FAILURE"  # Assertion failed
    assert results[0].metric.is_err()  # Metric computation failed (infinity)
    failures = results[0].metric.failure()
    assert "infinity" in failures[0].error_message.lower()
    # Expression still shows the full validation
    assert "1/sum(zeros) > 0" in results[0].expression


def test_validator_exception_raises_dqx_error():
    """Test that exceptions in validator functions are wrapped in DQXError."""
    # This test would need to mock a validator that throws an exception
    # Implementation depends on your testing strategy
    pass
```

**Run the new tests:**
```bash
uv run pytest tests/test_assertion_validation.py -v
```

**Commit:**
```bash
git add tests/test_assertion_validation.py
git commit -m "test: add comprehensive tests for assertion validation logic"
```

### Task 8: Update result_collection_demo.py

**File:** `examples/result_collection_demo.py`

**What to do:**
1. Update field references from `value` to `metric`
2. Update status checks from "SUCCESS" to "OK"
3. Add clear examples of validation failures vs metric failures
4. Update the output display to show both types of failures

**Key changes needed:**
```python
# In print_results_table function, change:
# OLD: if result.value.is_ok() else 'failed'
# NEW: if result.metric.is_ok() else 'failed'

# OLD: value_text = f"{result.value.unwrap():.2f}"
# NEW: value_text = f"{result.metric.unwrap():.2f}"

# OLD: failures = result.value.failure()
# NEW: failures = result.metric.failure()

# OLD: if result.status == "SUCCESS":
# NEW: if result.status == "OK":


# Add a new check that demonstrates validation failures:
@check(name="Validation Failures", datasets=["orders"])
def validation_failures(mp: MetricProvider, ctx: Context) -> None:
    """Metrics that compute successfully but fail validation."""
    # Average order is 120.5, but we assert it should be > 150
    ctx.assert_that(mp.average("amount")).where(
        name="Average order > 150", severity="P1"
    ).is_gt(150)

    # Count is 10, but we assert it should be >= 20
    ctx.assert_that(mp.num_rows()).where(
        name="At least 20 orders", severity="P2"
    ).is_geq(20)

    # Return rate is 0.0, but we assert it should be > 0.01
    return_rate = mp.sum("returns") / mp.num_rows()
    ctx.assert_that(return_rate).where(name="Return rate > 1%", severity="P1").is_gt(
        0.01
    )
```

**Update the output to distinguish failure types:**
```python
# In print_results_table, enhance the value display:
if result.status == "OK":
    value_text = f"✓ {result.metric.unwrap():.2f}"
else:
    if result.metric.is_ok():
        # Metric computed but validation failed
        metric_val = result.metric.unwrap()
        value_text = f"✗ {metric_val:.2f} (validation failed)"
    else:
        # Metric computation failed
        failures = result.metric.failure()
        if failures:
            error = failures[0].error_message
            if "infinity" in error.lower():
                value_text = "∞ (infinity error)"
            elif "nan" in error.lower():
                value_text = "NaN (not a number)"
            elif "complex" in error.lower():
                value_text = "ℂ (complex number)"
            elif "timeout" in error.lower():
                value_text = "⏱ (database timeout)"
            else:
                value_text = f"⚠ {error[:30]}..."
        else:
            value_text = "⚠ Unknown error"

# Update the summary statistics section:
print("\nSummary by Status:")
ok_count = sum(1 for r in results if r.status == "OK")
failure_count = sum(1 for r in results if r.status == "FAILURE")
print(f"✅ OK: {ok_count}")
print(f"❌ FAILURE: {failure_count}")
print(f"📊 Total: {len(results)}")

# Add expression display in the table header:
print(
    f"{'Check':<{check_width}} | {'Assertion':<{assertion_width}} | {'Sev':<5} | {'Status':<8} | {'Value/Error':<20} | Expression"
)
```

**Commit:**
```bash
git add examples/result_collection_demo.py
git commit -m "feat: update demo to showcase validation vs computation failures"
```

### Task 9: Final Verification

**Run all quality checks:**
```bash
# Type checking
uv run mypy src/

# Linting
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# All tests with coverage
uv run pytest tests/ -v --cov=dqx

# Pre-commit hooks
./bin/run-hooks.sh --all
```

**Make sure:**
- All tests pass
- No type errors
- No linting issues
- Coverage hasn't dropped

### Task 10: Update Documentation

**Files to check:**
- `README.md` - Update any examples that show result structure
- Any other docs that mention assertion results

**Look for:**
```bash
# Find documentation that might need updates
grep -r "result\.value" docs/ README.md
grep -r "assertion\._value" docs/ README.md
grep -r "SUCCESS" docs/ README.md | grep -i status
```

**Update any occurrences to use:**
- `result.metric` instead of `result.value`
- `"OK"` instead of `"SUCCESS"` for status values

**Final commit:**
```bash
git add -A
git commit -m "docs: update examples for new assertion validation behavior"
```

## Testing the Complete Change

Create a simple test script to verify everything works:

```python
# test_manual.py
import datetime
import pyarrow as pa
from dqx.api import VerificationSuiteBuilder, check
from dqx.common import ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB


@check(name="Value checks")
def check_values(mp, ctx):
    # This should pass: metric=20, validation: 20 > 0 ✓
    ctx.assert_that(mp.average("good")).where(name="Good values are positive").is_gt(0)

    # This should fail: metric=-20, validation: -20 > 0 ✗
    ctx.assert_that(mp.average("bad")).where(name="Bad values are positive").is_gt(0)

    # This should fail: metric=20, validation: 20 > 50 ✗
    ctx.assert_that(mp.average("good")).where(name="Good values exceed 50").is_gt(50)


# Create test data
data = pa.table({"good": [10.0, 20.0, 30.0], "bad": [-10.0, -20.0, -30.0]})

# Run checks
db = InMemoryMetricDB()
suite = VerificationSuiteBuilder("Test", db).add_check(check_values).build()
suite.run(
    {"data": ArrowDataSource(data)},
    ResultKey(yyyy_mm_dd=datetime.date.today(), tags={}),
)

# Check results
results = suite.collect_results()
for r in results:
    metric_str = f"{r.metric.unwrap():.1f}" if r.metric.is_ok() else "failed"
    print(f"{r.assertion}: {r.status} (metric={metric_str}) - {r.expression}")
```

Expected output:
```
Good values are positive: OK (metric=20.0) - average(good) > 0
Bad values are positive: FAILURE (metric=-20.0) - average(bad) > 0
Good values exceed 50: FAILURE (metric=20.0) - average(good) > 50
```

## Common Pitfalls

1. **Don't initialize _metric and _result in __init__** - They're set by the evaluator
2. **Remember pattern matching syntax** - Use `case Success(value):` not `case Success as value:`
3. **The validator.fn returns bool** - True means pass, False means fail
4. **DQXError on validator exceptions** - Wrap any validator errors properly
5. **No refactoring comments** - Comments describe current behavior, not history
6. **Status values are "OK"/"FAILURE"** - Not "SUCCESS"/"FAILURE"

## Rollback Plan

If something goes wrong:
```bash
# Stash your changes
git stash

# Return to main branch
git checkout main

# Delete the feature branch
git branch -D feat/evaluator-validation-refactoring
```

## Success Criteria

- [ ] All tests pass
- [ ] Type checking passes
- [ ] Assertions now validate values, not just compute them
- [ ] Result collection shows actual pass/fail status
- [ ] No performance regression
- [ ] Demo clearly shows both failure types
- [ ] Full validation expressions are displayed

## Next Steps

After this refactoring:
1. Update any dashboards that consume assertion results (note the status value change)
2. Notify teams about the breaking API change
3. Consider adding more detailed validation error messages
4. Update monitoring to distinguish metric failures from validation failures
