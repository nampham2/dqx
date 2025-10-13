# Implementation Plan: Assertion Result Collection (v2)

## Executive Summary

We need to add functionality to collect and export assertion evaluation results from the DQX verification framework after test execution. This will enable users to persist results to databases, create reports, or transform results into other formats.

This v2 plan incorporates architectural feedback and API improvements for cleaner design and better user experience.

## Background Context

### What is DQX?

DQX is a data quality framework that allows users to define checks (validation rules) on datasets. The framework:
- Uses a graph-based architecture to manage dependencies
- Evaluates assertions using symbolic expressions (SymPy)
- Returns results as `Result[float, list[EvaluationFailure]]` types (from the `returns` library)

### Key Concepts You Need to Know

1. **VerificationSuite**: The main entry point that runs data quality checks
2. **Graph**: A hierarchical structure containing:
   - **RootNode**: Top-level container
   - **CheckNode**: Groups related assertions (has a name and tags)
   - **AssertionNode**: Leaf nodes with actual validation logic
3. **Result Type**: A functional programming type that can be either:
   - `Success(value)`: Contains a float value
   - `Failure(errors)`: Contains a list of EvaluationFailure objects

### Current Workflow

```python
# Users currently do this:
suite = VerificationSuite(checks, db, "My Suite")
suite.run(datasources, key)  # Evaluates all assertions
# But have no way to extract the results programmatically
```

## Requirements

Add a method `collect_results()` to VerificationSuite that:
1. Returns a list of dataclass objects containing assertion results
2. Works only after `run()` has been called
3. Includes both successes and failures
4. Provides all context needed for reporting/persistence
5. Does NOT require the ResultKey again (stores it from run())

## Key Design Improvements in v2

1. **Single-run enforcement**: A suite can only be run once. Attempting to run again raises an error.
2. **Stored execution context**: The ResultKey is stored during run() and used automatically in collect_results().
3. **Cleaner error handling**: Uses Python's match statement for Result type handling.
4. **No exception masking**: is_evaluated is only set after successful completion (no try/finally).

## Implementation Tasks

### Task 1: Add the AssertionResult Dataclass

**File**: `src/dqx/common.py`

**What to do**:
1. Find the imports section and add: `from dataclasses import dataclass, field`
2. Add this new dataclass near the other dataclasses (around line 50-60):

```python
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
        assertion: Name or description of the assertion
        severity: Priority level (P0, P1, P2, P3)
        status: Either "SUCCESS" or "FAILURE"
        value: The actual Result object containing value or errors
        error_message: JSON-serialized error details if status is FAILURE
        expression: String representation of the symbolic expression
        tags: Tags from the ResultKey (e.g., {"env": "prod"})
    """

    yyyy_mm_dd: datetime.date
    suite: str
    check: str
    assertion: str
    severity: SeverityLevel
    status: Literal["SUCCESS", "FAILURE"]
    value: Result[float, list[EvaluationFailure]]
    error_message: str | None = None
    expression: str | None = None
    tags: Tags = field(default_factory=dict)
```

**Testing this step**:
```bash
# Run type checking to ensure imports are correct
uv run mypy src/dqx/common.py

# Create a simple test file to verify the dataclass works
# test_assertion_result.py
from dqx.common import AssertionResult, SeverityLevel
from returns.result import Success
import datetime

result = AssertionResult(
    yyyy_mm_dd=datetime.date.today(),
    suite="test",
    check="test_check",
    assertion="test_assertion",
    severity="P1",
    status="SUCCESS",
    value=Success(42.0)
)
print(result)
```

**Commit**:
```bash
git add src/dqx/common.py
git commit -m "feat: add AssertionResult dataclass for collecting evaluation results"
```

### Task 2: Update VerificationSuite with State Tracking

**File**: `src/dqx/api.py`

**What to do**:
1. Find the `VerificationSuite.__init__` method (around line 380)
2. Add these lines at the end of `__init__`:
```python
self.is_evaluated = False  # Track if assertions have been evaluated
self._key: ResultKey | None = None  # Store the key used during run()
```

3. Find the `run()` method (around line 450)
4. Add this check at the very beginning of the method:
```python
def run(
    self, datasources: dict[str, SqlDataSource], key: ResultKey, threading: bool = False
) -> None:
    """Execute the verification suite against the provided data sources."""
    # Prevent multiple runs
    if self.is_evaluated:
        raise DQXError(
            "Verification suite has already been executed. "
            "Create a new suite instance to run again."
        )

    logger.info(
        f"Running verification suite '{self._name}' with datasets: {list(datasources.keys())}"
    )

    # ... rest of existing validation ...
```

5. After the datasources validation, add:
```python
# Store the key for later use in collect_results
self._key = key
```

6. At the very end of the `run()` method (after the evaluator.bfs line), add:
```python
# Mark suite as evaluated only after successful completion
self.is_evaluated = True
```

**Why**:
- Prevents accidental re-runs which could corrupt state
- Stores execution context for later use
- Only marks as evaluated after successful completion

**Testing this step**:
Write a simple test to verify the single-run behavior:
```python
# test_single_run.py
from dqx.api import VerificationSuite, check
from dqx.orm.repositories import InMemoryMetricDB
from dqx.common import ResultKey, DQXError
import datetime
import pytest


@check(name="dummy")
def dummy_check(mp, ctx):
    ctx.assert_that(mp.num_rows()).where(name="test").is_gt(0)


db = InMemoryMetricDB()
suite = VerificationSuite([dummy_check], db, "Test Suite")

# Should be False initially
assert suite.is_evaluated == False

# First run should succeed (with proper datasource)
# key = ResultKey(...)
# suite.run({"test": datasource}, key)
# assert suite.is_evaluated == True

# Second run should raise error
# with pytest.raises(DQXError) as exc_info:
#     suite.run({"test": datasource}, key)
# assert "already been executed" in str(exc_info.value)
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "feat: add single-run enforcement and key storage to VerificationSuite"
```

### Task 3: Implement collect_results Method with Pattern Matching

**File**: `src/dqx/api.py`

**What to do**:
1. Add these imports at the top if not already present:
```python
import json
from dqx.common import AssertionResult
```

2. Add this method to the VerificationSuite class (after the `run` method):

```python
def collect_results(self) -> list[AssertionResult]:
    """
    Collect all assertion results after suite execution.

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
    """
    if not self.is_evaluated:
        raise DQXError(
            "Cannot collect results before suite execution. "
            "Call run() first to evaluate assertions."
        )

    # Use the stored key
    key = self._key
    results = []

    # Use the graph's built-in method to get all assertions
    for assertion in self._context._graph.assertions():
        # Extract parent hierarchy
        check_node = assertion.parent  # Parent is always a CheckNode

        # Use pattern matching for cleaner Result handling
        error_msg = None
        match assertion._value:
            case Success(value):
                status = "SUCCESS"
            case Failure(failures):
                status = "FAILURE"
                if failures:
                    # Serialize all failure information
                    error_data = [
                        {
                            "error_message": f.error_message,
                            "expression": f.expression,
                            "symbols": [
                                {
                                    "name": s.name,
                                    "metric": s.metric,
                                    "dataset": s.dataset,
                                    "value": str(s.value),
                                }
                                for s in f.symbols
                            ],
                        }
                        for f in failures
                    ]
                    error_msg = json.dumps(error_data)

        # Create result record
        result = AssertionResult(
            yyyy_mm_dd=key.yyyy_mm_dd,
            suite=self._name,
            check=check_node.name,
            assertion=assertion.name or f"Unnamed ({assertion.actual})",
            severity=assertion.severity,
            status=status,
            value=assertion._value,
            error_message=error_msg,
            expression=str(assertion.actual),
            tags=key.tags,
        )
        results.append(result)

    return results
```

**Note**: The pattern matching syntax requires Python 3.10+. Make sure to import Success and Failure from returns.result at the top of the file:
```python
from returns.result import Success, Failure
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "feat: implement collect_results with pattern matching and stored key"
```

### Task 4: Write Comprehensive Tests

**File**: Create `tests/test_assertion_result_collection.py`

**What to do**:
Create a new test file with comprehensive test coverage:

```python
"""Tests for assertion result collection functionality."""

import datetime
import json
import pytest
from returns.result import Success, Failure

from dqx.api import VerificationSuite, VerificationSuiteBuilder, check
from dqx.common import (
    AssertionResult,
    DQXError,
    EvaluationFailure,
    ResultKey,
    SymbolInfo,
)
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB
import pyarrow as pa


class TestAssertionResultCollection:
    """Test suite for collect_results functionality."""

    def test_collect_results_before_run_raises_error(self):
        """Should raise DQXError if collect_results called before run()."""

        @check(name="dummy check")
        def dummy_check(mp, ctx):
            ctx.assert_that(mp.num_rows()).where(name="rows > 0").is_gt(0)

        db = InMemoryMetricDB()
        suite = VerificationSuite([dummy_check], db, "Test Suite")

        # Should not be evaluated initially
        assert suite.is_evaluated is False

        # Should raise error when trying to collect results
        with pytest.raises(DQXError) as exc_info:
            suite.collect_results()

        assert "Cannot collect results before suite execution" in str(exc_info.value)

    def test_suite_cannot_run_twice(self):
        """Should raise DQXError if suite.run() called twice."""

        @check(name="dummy check")
        def dummy_check(mp, ctx):
            ctx.assert_that(mp.num_rows()).where(name="rows > 0").is_gt(0)

        db = InMemoryMetricDB()
        suite = VerificationSuite([dummy_check], db, "Test Suite")

        # Create test data
        data = pa.table({"x": [1, 2, 3]})
        datasource = ArrowDataSource(data)
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # First run should succeed
        suite.run({"data": datasource}, key)
        assert suite.is_evaluated is True

        # Second run should raise error
        with pytest.raises(DQXError) as exc_info:
            suite.run({"data": datasource}, key)

        assert "already been executed" in str(exc_info.value)

    def test_collect_results_no_key_needed(self):
        """Should not require key parameter since it's stored from run()."""

        @check(name="simple check")
        def simple_check(mp, ctx):
            ctx.assert_that(mp.num_rows()).where(name="row count").is_eq(3)

        db = InMemoryMetricDB()
        suite = VerificationSuite([simple_check], db, "Test Suite")

        # Run with specific key
        data = pa.table({"x": [1, 2, 3]})
        key = ResultKey(
            yyyy_mm_dd=datetime.date(2024, 1, 15),
            tags={"env": "test", "version": "1.0"},
        )
        suite.run({"data": ArrowDataSource(data)}, key)

        # Collect results without passing key
        results = suite.collect_results()  # No key parameter!

        # Verify key was stored and used
        assert len(results) == 1
        assert results[0].yyyy_mm_dd == datetime.date(2024, 1, 15)
        assert results[0].tags == {"env": "test", "version": "1.0"}

    def test_collect_results_after_successful_run(self):
        """Should return results for all assertions after successful run."""
        # Create test data
        data = pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "category": ["A", "B", "A", "B", "A"],
            }
        )
        datasource = ArrowDataSource(data)

        # Define checks with multiple assertions
        @check(name="data validation")
        def validate_data(mp, ctx):
            # This should pass
            ctx.assert_that(mp.num_rows()).where(
                name="Has 5 rows", severity="P0"
            ).is_eq(5)

            # This should also pass
            ctx.assert_that(mp.average("value")).where(
                name="Average value is 30", severity="P1"
            ).is_eq(30.0)

            # This should fail
            ctx.assert_that(mp.minimum("value")).where(
                name="Minimum value check", severity="P2"
            ).is_gt(
                20.0
            )  # Will fail since min is 10

        # Build and run suite
        db = InMemoryMetricDB()
        suite = (
            VerificationSuiteBuilder("Test Suite", db).add_check(validate_data).build()
        )

        key = ResultKey(
            yyyy_mm_dd=datetime.date(2024, 1, 15),
            tags={"env": "test", "version": "1.0"},
        )

        # Run the suite
        suite.run({"test_data": datasource}, key)

        # Should be evaluated after run
        assert suite.is_evaluated is True

        # Collect results (no key needed!)
        results = suite.collect_results()

        # Verify we got all 3 assertions
        assert len(results) == 3

        # Check first assertion (rows count - should pass)
        r1 = results[0]
        assert r1.yyyy_mm_dd == datetime.date(2024, 1, 15)
        assert r1.suite == "Test Suite"
        assert r1.check == "data validation"
        assert r1.assertion == "Has 5 rows"
        assert r1.severity == "P0"
        assert r1.status == "SUCCESS"
        assert r1.value.is_ok()
        assert r1.value.unwrap() == 5.0
        assert r1.error_message is None
        assert "num_rows" in r1.expression
        assert r1.tags == {"env": "test", "version": "1.0"}

        # Check second assertion (average - should pass)
        r2 = results[1]
        assert r2.assertion == "Average value is 30"
        assert r2.severity == "P1"
        assert r2.status == "SUCCESS"
        assert r2.value.unwrap() == 30.0

        # Check third assertion (minimum - should fail)
        r3 = results[2]
        assert r3.assertion == "Minimum value check"
        assert r3.severity == "P2"
        assert r3.status == "FAILURE"
        assert r3.value.is_err()
        assert r3.error_message is not None

        # Verify JSON error format
        error_data = json.loads(r3.error_message)
        assert isinstance(error_data, list)
        assert len(error_data) > 0
        assert "error_message" in error_data[0]
        assert "20" in error_data[0]["error_message"]  # Should mention expected value

    def test_json_error_serialization_format(self):
        """Should serialize error details as JSON with symbols."""
        data = pa.table({"x": [5]})
        datasource = ArrowDataSource(data)

        @check(name="json error test")
        def error_check(mp, ctx):
            # Create an assertion that will fail
            ctx.assert_that(mp.num_rows()).where(name="impossible check").is_gt(10)

        db = InMemoryMetricDB()
        suite = VerificationSuite([error_check], db, "Test Suite")

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.run({"data": datasource}, key)

        results = suite.collect_results()
        assert len(results) == 1

        result = results[0]
        assert result.status == "FAILURE"
        assert result.error_message is not None

        # Parse and validate JSON structure
        error_data = json.loads(result.error_message)
        assert isinstance(error_data, list)
        assert len(error_data) > 0

        first_error = error_data[0]
        assert "error_message" in first_error
        assert "expression" in first_error
        assert "symbols" in first_error
        assert isinstance(first_error["symbols"], list)

    def test_is_evaluated_only_set_on_success(self):
        """Should NOT set is_evaluated if run() fails."""

        @check(name="failing check")
        def failing_check(mp, ctx):
            # This will cause an error during execution
            raise RuntimeError("Simulated failure")

        db = InMemoryMetricDB()
        suite = VerificationSuite([failing_check], db, "Test Suite")

        assert suite.is_evaluated is False

        # Run should fail
        data = pa.table({"x": [1]})
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # This should raise an exception
        with pytest.raises(RuntimeError):
            suite.run({"data": ArrowDataSource(data)}, key)

        # Flag should NOT be set on failure
        assert suite.is_evaluated is False

        # collect_results should still raise error
        with pytest.raises(DQXError) as exc_info:
            suite.collect_results()
        assert "Cannot collect results" in str(exc_info.value)

    def test_collect_results_with_unnamed_assertion(self):
        """Should handle assertions without explicit names."""
        data = pa.table({"x": [1, 2, 3]})
        datasource = ArrowDataSource(data)

        @check(name="unnamed test")
        def unnamed_check(mp, ctx):
            # Create assertion without name (edge case)
            # Note: In real API this shouldn't happen, but let's test it
            assertion = ctx.create_assertion(
                mp.num_rows(), name=None, severity="P1"  # No name provided
            )

        db = InMemoryMetricDB()
        suite = VerificationSuite([unnamed_check], db, "Test Suite")

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.run({"data": datasource}, key)

        results = suite.collect_results()

        # Should create a fallback name using the expression
        assert len(results) == 1
        assert "Unnamed" in results[0].assertion
        assert "num_rows" in results[0].assertion

    def test_collect_results_preserves_result_object(self):
        """Should preserve the full Result object for advanced usage."""
        data = pa.table({"x": [1, 2, 3]})
        datasource = ArrowDataSource(data)

        @check(name="result preservation")
        def check_results(mp, ctx):
            ctx.assert_that(mp.average("x")).where(name="avg check").is_eq(2.0)

        db = InMemoryMetricDB()
        suite = VerificationSuite([check_results], db, "Suite")

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.run({"data": datasource}, key)

        results = suite.collect_results()

        # The value field should be the actual Result object
        assert len(results) == 1
        result_obj = results[0].value

        # Should be able to use Result methods
        assert result_obj.is_ok()
        assert result_obj.unwrap() == 2.0

        # Should be able to map/bind on the Result
        doubled = result_obj.map(lambda x: x * 2)
        assert doubled.unwrap() == 4.0

    def test_empty_suite_returns_empty_results(self):
        """Should return empty list if no assertions in suite."""

        # Suite with a check that has no assertions
        @check(name="empty check")
        def empty_check(mp, ctx):
            pass  # No assertions

        db = InMemoryMetricDB()
        suite = VerificationSuite([empty_check], db, "Empty Suite")
        data = pa.table({"x": [1]})

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.run({"data": ArrowDataSource(data)}, key)

        results = suite.collect_results()
        assert results == []


def test_assertion_result_dataclass():
    """Test AssertionResult dataclass behavior."""
    # Test creation with all fields
    result = AssertionResult(
        yyyy_mm_dd=datetime.date(2024, 1, 1),
        suite="Test Suite",
        check="Test Check",
        assertion="Test Assertion",
        severity="P1",
        status="SUCCESS",
        value=Success(42.0),
        error_message=None,
        expression="x > 0",
        tags={"env": "prod"},
    )

    # Should be able to access all fields
    assert result.yyyy_mm_dd == datetime.date(2024, 1, 1)
    assert result.suite == "Test Suite"
    assert result.check == "Test Check"
    assert result.assertion == "Test Assertion"
    assert result.severity == "P1"
    assert result.status == "SUCCESS"
    assert result.value.unwrap() == 42.0
    assert result.error_message is None
    assert result.expression == "x > 0"
    assert result.tags == {"env": "prod"}

    # Test with failure and JSON error
    error_json = json.dumps(
        [
            {
                "error_message": "Value 5 is not greater than 10",
                "expression": "x > 10",
                "symbols": [],
            }
        ]
    )

    failure_result = AssertionResult(
        yyyy_mm_dd=datetime.date(2024, 1, 1),
        suite="Test Suite",
        check="Test Check",
        assertion="Failed Assertion",
        severity="P0",
        status="FAILURE",
        value=Failure(
            [
                EvaluationFailure(
                    error_message="Value 5 is not greater than 10",
                    expression="x > 10",
                    symbols=[],
                )
            ]
        ),
        error_message=error_json,
        expression="x > 10",
        tags={},
    )

    assert failure_result.status == "FAILURE"
    assert failure_result.error_message == error_json
    assert failure_result.value.is_err()

    # Verify we can parse the JSON error
    parsed_error = json.loads(failure_result.error_message)
    assert parsed_error[0]["error_message"] == "Value 5 is not greater than 10"
```

**Running the tests**:
```bash
# Run just the new tests
uv run pytest tests/test_assertion_result_collection.py -v

# Run with coverage
uv run pytest tests/test_assertion_result_collection.py -v --cov=dqx.api --cov=dqx.common

# Run all tests to ensure nothing broke
uv run pytest tests/ -v
```

**Commit**:
```bash
git add tests/test_assertion_result_collection.py
git commit -m "test: add comprehensive tests for v2 features including single-run enforcement"
```

### Task 5: Integration Testing

Create a practical example showing how to use the feature:

**File**: Create `examples/result_collection_demo.py`

```python
"""Demo of collecting and using assertion results with v2 improvements."""

import datetime
import json
import pandas as pd
import pyarrow as pa
from dqx.api import VerificationSuiteBuilder, check
from dqx.common import ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB


# Create sample e-commerce data
orders_data = pa.table(
    {
        "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "customer_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "amount": [50.0, 150.0, 75.0, 200.0, 30.0, 120.0, 90.0, 180.0, 60.0, 250.0],
        "status": [
            "completed",
            "completed",
            "pending",
            "completed",
            "cancelled",
            "completed",
            "completed",
            "pending",
            "completed",
            "completed",
        ],
        "created_at": [
            "2024-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-04",
            "2024-01-04",
            "2024-01-05",
            "2024-01-05",
        ],
    }
)


# Define data quality checks
@check(name="Order Amount Validation")
def validate_order_amounts(mp, ctx):
    """Validate order amounts are within expected ranges."""
    ctx.assert_that(mp.minimum("amount")).where(
        name="No negative amounts", severity="P0"
    ).is_geq(0.0)

    ctx.assert_that(mp.average("amount")).where(
        name="Average order value check", severity="P1"
    ).is_between(50.0, 200.0)

    ctx.assert_that(mp.maximum("amount")).where(
        name="Maximum order limit", severity="P2"
    ).is_leq(1000.0)


@check(name="Order Completeness")
def validate_completeness(mp, ctx):
    """Check for data completeness."""
    ctx.assert_that(mp.null_count("order_id")).where(
        name="Order ID completeness", severity="P0"
    ).is_eq(0)

    ctx.assert_that(mp.null_count("customer_id")).where(
        name="Customer ID completeness", severity="P0"
    ).is_eq(0)


@check(name="Business Rules")
def validate_business_rules(mp, ctx):
    """Validate business logic constraints."""
    total_orders = mp.num_rows()

    # This will fail - we have 2 pending orders
    ctx.assert_that(mp.num_rows()).where(
        name="Minimum order volume", severity="P1"
    ).is_geq(
        20
    )  # Will fail - only have 10 orders


def main():
    """Run the demo."""
    # Setup
    db = InMemoryMetricDB()
    suite = (
        VerificationSuiteBuilder("E-commerce Data Quality", db)
        .add_check(validate_order_amounts)
        .add_check(validate_completeness)
        .add_check(validate_business_rules)
        .build()
    )

    # Create data source
    datasource = ArrowDataSource(orders_data)

    # Run validation
    key = ResultKey(
        yyyy_mm_dd=datetime.date.today(),
        tags={"environment": "production", "version": "1.0"},
    )

    print("Running data quality checks...")
    suite.run({"orders": datasource}, key)

    # Check evaluation status
    print(f"Suite evaluated: {suite.is_evaluated}")

    # Collect results - no key needed anymore!
    results = suite.collect_results()

    print(f"\nCollected {len(results)} assertion results")
    print("=" * 60)

    # Display results
    for result in results:
        status_icon = "✅" if result.status == "SUCCESS" else "❌"
        print(f"{status_icon} [{result.severity}] {result.check} / {result.assertion}")
        if result.status == "SUCCESS":
            print(f"   Value: {result.value.unwrap():.2f}")
        else:
            # Parse and display JSON error details
            error_data = json.loads(result.error_message)
            for error in error_data:
                print(f"   Error: {error['error_message']}")
                if error.get("symbols"):
                    print(f"   Expression: {error['expression']}")
                    for symbol in error["symbols"]:
                        print(f"     - {symbol['name']}: {symbol['value']}")
        print()

    # Convert to pandas DataFrame for analysis
    print("\nConverting to DataFrame...")
    df = pd.DataFrame(
        [
            {
                "date": r.yyyy_mm_dd,
                "suite": r.suite,
                "check": r.check,
                "assertion": r.assertion,
                "severity": r.severity,
                "status": r.status,
                "value": r.value.unwrap() if r.value.is_ok() else None,
                "error": r.error_message,
            }
            for r in results
        ]
    )

    print("\nSummary by Status:")
    print(df.groupby("status").size())

    print("\nFailed Assertions:")
    failed = df[df["status"] == "FAILURE"]
    if not failed.empty:
        print(failed[["check", "assertion", "severity"]])
    else:
        print("No failures!")

    # Demonstrate that suite cannot be run again
    print("\nAttempting to run suite again...")
    try:
        suite.run({"orders": datasource}, key)
    except Exception as e:
        print(f"Expected error: {e}")

    # Example: Create DuckDB relation
    try:
        import duckdb

        conn = duckdb.connect()
        relation = conn.from_pandas(df)
        print("\nDuckDB Relation created successfully!")
        print(
            relation.query(
                "SELECT check, COUNT(*) as assertion_count FROM df GROUP BY check"
            ).df()
        )
    except ImportError:
        print("\nSkipping DuckDB example (not installed)")


if __name__ == "__main__":
    main()
```

**Commit**:
```bash
git add examples/result_collection_demo.py
git commit -m "docs: add demo showing v2 API improvements"
```

### Task 6: Update Documentation

**File**: Update the main README.md to document the new feature

Add this section after the "Collecting Metrics Without Execution" section:

```markdown
### Collecting Assertion Results

After running a verification suite, you can collect detailed results for all assertions. The API has been designed for simplicity - no need to pass the ResultKey again:

```python
# Run the suite once
suite.run(datasources, key)

# Check if evaluation is complete
if suite.is_evaluated:
    # Collect results - no key parameter needed!
    results = suite.collect_results()

    # Process results
    for result in results:
        print(f"{result.check}/{result.assertion}: {result.status}")
        if result.status == "FAILURE":
            # Error message contains JSON with full details
            error_data = json.loads(result.error_message)
            for error in error_data:
                print(f"  Error: {error['error_message']}")

# Convert to DataFrame for analysis
import pandas as pd

df = pd.DataFrame(
    [
        {
            "date": r.yyyy_mm_dd,
            "check": r.check,
            "assertion": r.assertion,
            "status": r.status,
            "value": r.value.unwrap() if r.value.is_ok() else None,
        }
        for r in results
    ]
)

# Or create a DuckDB relation
import duckdb

conn = duckdb.connect()
relation = conn.from_pandas(df)
```

**Key Features**:
- **Single-run enforcement**: A suite can only be run once per instance
- **No redundant parameters**: ResultKey is stored during run() and reused automatically
- **Pattern matching**: Clean Result type handling using Python 3.10+ match statements
- **Rich error details**: JSON-serialized error information including symbol values

The `AssertionResult` dataclass provides:
- Full context (suite/check/assertion hierarchy)
- Success/failure status
- The actual Result object for advanced usage
- JSON-serialized error details with symbol information
- Severity levels and tags

The `is_evaluated` flag indicates whether the suite has been executed, ensuring results are available before collection.
```

**Commit**:
```bash
git add README.md
git commit -m "docs: document assertion result collection v2 features"
```

## Final Checklist

Before considering this complete:

1. **Type Checking**: Run `uv run mypy src/` and fix any type errors
2. **Linting**: Run `uv run ruff check src/` and fix any issues
3. **Tests Pass**: Run `uv run pytest tests/` - all should pass
4. **Coverage**: Run `uv run pytest tests/test_assertion_result_collection.py --cov=dqx.api --cov=dqx.common` - aim for 100%
5. **Documentation**: Ensure all docstrings are complete and accurate

## Design Principles Followed

1. **TDD**: Write tests first, then implementation
2. **DRY**: Reuse existing graph traversal methods, avoid redundant parameters
3. **YAGNI**: Only added what was requested, no extra features
4. **Single Responsibility**: Each component has one clear purpose
5. **Fail Fast**: Clear errors when preconditions not met
6. **Clean API**: No redundant parameters, intuitive usage

## Common Pitfalls to Avoid

1. Don't try to access `assertion._value` without checking `is_evaluated` first
2. Remember that `assertion.parent` is always a CheckNode, never None
3. The Result type is from the `returns` library, not standard Python
4. Pattern matching requires Python 3.10+ - ensure compatibility
5. Tags in ResultKey are a dict, not a list
6. Severity levels are strings ("P0", "P1", etc.), not integers
7. A suite can only run once - create new instances for multiple runs

## Questions You Might Have

**Q: Why can't I run a suite multiple times?**
A: This prevents state corruption and ensures results are consistent. Create a new suite instance for each run.

**Q: What if I want to filter results by status?**
A: After collecting, use list comprehension:
```python
failures = [r for r in results if r.status == "FAILURE"]
```

**Q: Can I serialize AssertionResult to JSON?**
A: Not directly because of the Result type. You'll need to extract the value first:
```python
json_safe = {
    "assertion": r.assertion,
    "value": r.value.unwrap() if r.value.is_ok() else None,
    "error": r.error_message,
}
```

**Q: What order are results returned in?**
A: Breadth-first traversal order of the graph (all checks, then all assertions within each check).

**Q: How do I handle the pattern matching if using Python < 3.10?**
A: Use traditional if/isinstance checks:
```python
if assertion._value.is_ok():
    status = "SUCCESS"
    error_msg = None
else:
    status = "FAILURE"
    failures = assertion._value.failure()
    # ... serialize errors
```

This completes the v2 implementation plan with all feedback incorporated.
