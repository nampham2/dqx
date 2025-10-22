# Implementation Plan: Cache Results and Critical Level Detection

## Overview

This plan implements two features for the VerificationSuite class:
1. **In-memory caching** for `collect_results()` and `collect_symbols()` methods
2. **Critical level detection** via a new `is_critical()` method that returns True if any P0 assertion failed

## Background

Currently, `collect_results()` and `collect_symbols()` recompute their results every time they're called. For large verification suites, this can be inefficient. Additionally, there's no easy way to determine if a suite has critical (P0) failures without manually iterating through all results.

## Design Decisions

### Caching Strategy
- **In-memory caching**: Results are cached after first computation and persist for the lifetime of the VerificationSuite instance
- **No cache clearing**: As requested, the cache doesn't need to be cleared
- **Cache invalidation**: Natural invalidation occurs when a new VerificationSuite instance is created

### Critical Level Detection
- A suite is considered "critical" if **any** P0 severity assertion has failed
- The method leverages the cached `collect_results()` for efficiency

## Implementation Plan

### Task Group 1: Test Infrastructure Setup

#### Task 1.1: Create test file for caching functionality
**File**: `tests/test_suite_caching.py`

```python
"""Tests for VerificationSuite caching functionality."""

import pytest
from datetime import date

from dqx.api import VerificationSuite, check, Context
from dqx.common import DQXError, ResultKey
from dqx.provider import MetricProvider
from tests.fixtures.datasource import create_datasource
from tests.fixtures.db import InMemoryMetricDB


class TestSuiteCaching:
    """Test suite for collect_results and collect_symbols caching."""

    def test_collect_results_returns_same_object_reference(self):
        """Multiple calls to collect_results should return the same cached object."""
        db = InMemoryMetricDB()

        @check(name="Price Check")
        def price_check(mp: MetricProvider, ctx: Context) -> None:
            price = mp.average("price")
            ctx.assert_that(price).where(name="Price is positive").is_positive()

        suite = VerificationSuite([price_check], db, "Test Suite")
        ds = create_datasource("test", {"price": [10, 20, 30]})
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # Get results twice
        results1 = suite.collect_results()
        results2 = suite.collect_results()

        # Should be the exact same object (not just equal)
        assert results1 is results2

    def test_collect_symbols_returns_same_object_reference(self):
        """Multiple calls to collect_symbols should return the same cached object."""
        db = InMemoryMetricDB()

        @check(name="Metric Check")
        def metric_check(mp: MetricProvider, ctx: Context) -> None:
            avg_price = mp.average("price")
            sum_quantity = mp.sum("quantity")
            ctx.assert_that(avg_price + sum_quantity).where(
                name="Combined metric check"
            ).is_positive()

        suite = VerificationSuite([metric_check], db, "Test Suite")
        ds = create_datasource("test", {"price": [10, 20], "quantity": [5, 15]})
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # Get symbols twice
        symbols1 = suite.collect_symbols()
        symbols2 = suite.collect_symbols()

        # Should be the exact same object
        assert symbols1 is symbols2

    def test_caching_works_after_successful_run(self):
        """Caching should work correctly after a successful suite run."""
        db = InMemoryMetricDB()

        @check(name="Simple Check")
        def simple_check(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            ctx.assert_that(total).where(name="Total is 100").is_eq(100)

        suite = VerificationSuite([simple_check], db, "Test Suite")
        ds = create_datasource("test", {"value": [25, 25, 25, 25]})
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # First calls - should compute and cache
        results1 = suite.collect_results()
        symbols1 = suite.collect_symbols()

        # Verify we got results
        assert len(results1) == 1
        assert len(symbols1) == 1

        # Second calls - should return cached
        results2 = suite.collect_results()
        symbols2 = suite.collect_symbols()

        # Verify caching
        assert results1 is results2
        assert symbols1 is symbols2

    def test_cache_before_run_raises_error(self):
        """Attempting to access cache before run() should raise DQXError."""
        db = InMemoryMetricDB()

        @check(name="Never Run")
        def never_run(mp: MetricProvider, ctx: Context) -> None:
            pass

        suite = VerificationSuite([never_run], db, "Test Suite")

        # Should raise error before run
        with pytest.raises(DQXError, match="not been executed"):
            suite.collect_results()

        with pytest.raises(DQXError, match="not been executed"):
            suite.collect_symbols()
```

**Actions**:
- Write test file with comprehensive caching tests
- Run tests to verify they fail (TDD approach)

#### Task 1.2: Create test file for is_critical functionality
**File**: `tests/test_suite_critical.py`

```python
"""Tests for VerificationSuite critical level detection."""

import pytest
from datetime import date

from dqx.api import VerificationSuite, check, Context
from dqx.common import DQXError, ResultKey
from dqx.provider import MetricProvider
from tests.fixtures.datasource import create_datasource
from tests.fixtures.db import InMemoryMetricDB


class TestSuiteCritical:
    """Test suite for is_critical functionality."""

    def test_is_critical_with_p0_failure(self):
        """Suite with P0 failures should be critical."""
        db = InMemoryMetricDB()

        @check(name="Critical Check")
        def critical_check(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            # This will fail - expecting 1000 but sum is 100
            ctx.assert_that(total).where(
                name="Critical metric", severity="P0"
            ).is_eq(1000)

        suite = VerificationSuite([critical_check], db, "Test Suite")
        ds = create_datasource("test", {"value": [25, 25, 25, 25]})
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # Should be critical due to P0 failure
        assert suite.is_critical() is True

    def test_is_critical_with_no_p0_failures(self):
        """Suite without P0 failures should not be critical."""
        db = InMemoryMetricDB()

        @check(name="Non-Critical Checks")
        def non_critical_checks(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            # P1 assertion that passes
            ctx.assert_that(total).where(
                name="P1 check", severity="P1"
            ).is_eq(100)
            # P2 assertion that fails
            ctx.assert_that(total).where(
                name="P2 check", severity="P2"
            ).is_eq(200)

        suite = VerificationSuite([non_critical_checks], db, "Test Suite")
        ds = create_datasource("test", {"value": [25, 25, 25, 25]})
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # Should not be critical (no P0 failures)
        assert suite.is_critical() is False

    def test_is_critical_with_mixed_severities(self):
        """Suite with mixed severities including P0 failure should be critical."""
        db = InMemoryMetricDB()

        @check(name="Mixed Severity Check")
        def mixed_check(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            # P0 that fails
            ctx.assert_that(total).where(
                name="Critical check", severity="P0"
            ).is_eq(1000)
            # P1 that passes
            ctx.assert_that(total).where(
                name="Important check", severity="P1"
            ).is_positive()
            # P2 that fails
            ctx.assert_that(total).where(
                name="Minor check", severity="P2"
            ).is_negative()

        suite = VerificationSuite([mixed_check], db, "Test Suite")
        ds = create_datasource("test", {"value": [25, 25, 25, 25]})
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # Should be critical due to P0 failure
        assert suite.is_critical() is True

    def test_is_critical_before_run_raises_error(self):
        """Calling is_critical before run should raise DQXError."""
        db = InMemoryMetricDB()

        @check(name="Never Run")
        def never_run(mp: MetricProvider, ctx: Context) -> None:
            pass

        suite = VerificationSuite([never_run], db, "Test Suite")

        with pytest.raises(DQXError, match="not been executed"):
            suite.is_critical()

    def test_is_critical_with_no_assertions(self):
        """Suite with no assertions should not be critical."""
        db = InMemoryMetricDB()

        @check(name="Empty Check")
        def empty_check(mp: MetricProvider, ctx: Context) -> None:
            # No assertions
            pass

        suite = VerificationSuite([empty_check], db, "Test Suite")
        ds = create_datasource("test", {"value": [1, 2, 3]})
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # Should not be critical (no assertions at all)
        assert suite.is_critical() is False

    def test_is_critical_uses_cached_results(self):
        """is_critical should use cached collect_results for efficiency."""
        db = InMemoryMetricDB()

        @check(name="P0 Failure")
        def p0_failure(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            ctx.assert_that(total).where(
                name="Will fail", severity="P0"
            ).is_eq(999)

        suite = VerificationSuite([p0_failure], db, "Test Suite")
        ds = create_datasource("test", {"value": [1, 2, 3]})
        key = ResultKey(date.today(), {"env": "test"})

        suite.run({"test": ds}, key)

        # First call to collect_results
        results1 = suite.collect_results()

        # Call is_critical
        is_critical = suite.is_critical()
        assert is_critical is True

        # Get results again - should be cached
        results2 = suite.collect_results()
        assert results1 is results2  # Same object reference
```

**Actions**:
- Write test file for is_critical functionality
- Run tests to verify they fail (TDD approach)

#### Task 1.3: Run initial tests and commit
**Commands**:
```bash
# Run the new test files to verify they fail
uv run pytest tests/test_suite_caching.py -xvs
uv run pytest tests/test_suite_critical.py -xvs

# Check with mypy
uv run mypy tests/test_suite_caching.py
uv run mypy tests/test_suite_critical.py

# Check with ruff
uv run ruff check tests/test_suite_caching.py tests/test_suite_critical.py

# Commit the test infrastructure
git add tests/test_suite_caching.py tests/test_suite_critical.py
git commit -m "test: add test infrastructure for caching and is_critical features"
```

### Task Group 2: Implement Caching Infrastructure

#### Task 2.1: Add cache attributes to VerificationSuite
**File**: `src/dqx/api.py`

In the `__init__` method of VerificationSuite (around line 320), add cache attributes:

```python
def __init__(
    self,
    checks: Sequence[CheckProducer | DecoratedCheck],
    db: MetricDB,
    name: str,
) -> None:
    """
    Initialize the verification suite.

    Args:
        checks: Sequence of check functions to execute
        db: Database for storing and retrieving metrics
        name: Human-readable name for the suite

    Raises:
        DQXError: If no checks provided or name is empty
    """
    if not checks:
        raise DQXError("At least one check must be provided")
    if not name.strip():
        raise DQXError("Suite name cannot be empty")

    self._checks: Sequence[CheckProducer | DecoratedCheck] = checks
    self._name = name.strip()

    # Create a context
    self._context = Context(suite=self._name, db=db)

    # State tracking for result collection
    self._is_evaluated = False  # Track if assertions have been evaluated
    self._key: ResultKey | None = None  # Store the key used during run()

    # Lazy-loaded plugin manager
    self._plugin_manager: PluginManager | None = None

    # Timer for analyzing phase
    self._analyze_ms = timer_registry.timer("analyzing.time_ms")

    # Cache for collect_results and collect_symbols
    self._cached_results: list[AssertionResult] | None = None
    self._cached_symbols: list[SymbolInfo] | None = None
```

**Actions**:
- Add the two cache attributes
- Run tests to see progress

#### Task 2.2: Implement caching in collect_results
**File**: `src/dqx/api.py`

Modify the `collect_results` method (around line 520):

```python
def collect_results(self) -> list[AssertionResult]:
    """
    Collect all assertion results after suite execution.

    This method traverses the evaluation graph and extracts results from
    all assertions, converting them into AssertionResult objects suitable
    for persistence or reporting. The ResultKey used during run() is
    automatically applied to all results.

    Results are cached after the first call for efficiency.

    Returns:
        List of AssertionResult instances, one for each assertion in the suite.
        Results are returned in graph traversal order (breadth-first).
        Multiple calls return the same cached list object.

    Raises:
        DQXError: If called before run() has been executed successfully.

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> results = suite.collect_results()  # No key needed!
        >>> for r in results:
        ...     print(f"{r.check}/{r.assertion}: {r.status}")
        ...     if r.status == "FAILURE":
        ...         failures = r.value.failure()
        ...         for f in failures:
        ...             print(f"  Error: {f.error_message}")
    """
    # Return cached results if available
    if self._cached_results is not None:
        return self._cached_results

    # Only collect results after evaluation
    self.assert_is_evaluated()
    key = self.key
    results = []

    # Use the graph's built-in method to get all assertions
    for assertion in self.graph.assertions():
        # Extract parent hierarchy
        check_node = assertion.parent  # Parent is always a CheckNode

        result = AssertionResult(
            yyyy_mm_dd=key.yyyy_mm_dd,
            suite=self._name,
            check=check_node.name,
            assertion=assertion.name,
            severity=assertion.severity,
            status=assertion._result,
            metric=assertion._metric,
            expression=f"{assertion.actual} {assertion.validator.name}",
            tags=key.tags,
        )
        results.append(result)

    # Cache the results before returning
    self._cached_results = results
    return self._cached_results
```

**Actions**:
- Add caching logic to collect_results
- Run caching tests to verify some pass

#### Task 2.3: Implement caching in collect_symbols
**File**: `src/dqx/api.py`

Modify the `collect_symbols` method (around line 570):

```python
def collect_symbols(self) -> list[SymbolInfo]:
    """
    Collect all symbol values after suite execution.

    This method retrieves information about all symbols (metrics) that were
    registered during suite setup, evaluates them, and returns their values
    along with metadata. Symbols are sorted by name for consistent ordering.

    Results are cached after the first call for efficiency.

    Returns:
        List of SymbolInfo instances, sorted by symbol name in natural numeric
        order (x_1, x_2, ..., x_10, x_11, etc. rather than lexicographic).
        Each contains the symbol name, metric description, dataset,
        computed value, and context information (date, suite, tags).
        Multiple calls return the same cached list object.

    Raises:
        DQXError: If called before run() has been executed successfully.

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> symbols = suite.collect_symbols()
        >>> for s in symbols:
        ...     if s.value.is_success():
        ...         print(f"{s.metric}: {s.value.unwrap()}")
    """
    # Return cached symbols if available
    if self._cached_symbols is not None:
        return self._cached_symbols

    # Only collect symbols after evaluation
    self.assert_is_evaluated()
    symbols = []

    # Iterate through all registered symbols
    for symbolic_metric in self._context.provider.symbolic_metrics:
        # Calculate the effective key for this symbol
        effective_key = symbolic_metric.key_provider.create(self.key)

        # Try to evaluate the symbol to get its value
        try:
            value = symbolic_metric.fn(effective_key)
        except Exception:
            # In tests, the symbol might not be evaluable
            from returns.result import Failure

            value = Failure("Not evaluated")

        # Create SymbolInfo with all fields
        symbol_info = SymbolInfo(
            name=str(symbolic_metric.symbol),
            metric=str(symbolic_metric.metric_spec),
            dataset=symbolic_metric.dataset,
            value=value,
            yyyy_mm_dd=effective_key.yyyy_mm_dd,  # Use effective date!
            suite=self._name,
            tags=effective_key.tags,
        )
        symbols.append(symbol_info)

    # Sort by symbol numeric suffix for natural ordering (x_1, x_2, ..., x_10)
    # instead of lexicographic ordering (x_1, x_10, x_2)
    sorted_symbols = sorted(symbols, key=lambda s: int(s.name.split("_")[1]))

    # Cache the sorted symbols before returning
    self._cached_symbols = sorted_symbols
    return self._cached_symbols
```

**Actions**:
- Add caching logic to collect_symbols
- Run all caching tests - they should now pass

#### Task 2.4: Run tests and commit caching implementation
**Commands**:
```bash
# Run caching tests
uv run pytest tests/test_suite_caching.py -xvs

# Run type checking
uv run mypy src/dqx/api.py

# Run linting
uv run ruff check src/dqx/api.py

# If all pass, commit
git add src/dqx/api.py
git commit -m "feat: implement in-memory caching for collect_results and collect_symbols"
```

### Task Group 3: Implement is_critical Method

#### Task 3.1: Add is_critical method to VerificationSuite
**File**: `src/dqx/api.py`

Add the new method after `collect_symbols` (around line 620):

```python
def is_critical(self) -> bool:
    """
    Determine if the verification suite is critical.

    A suite is considered critical if any P0 severity assertion has failed.
    This method uses the cached results from collect_results() for efficiency.

    Returns:
        True if any P0 assertion failed, False otherwise

    Raises:
        DQXError: If called before run() has been executed successfully

    Example:
        >>> suite = VerificationSuite(checks, db, "My Suite")
        >>> suite.run(datasources, key)
        >>> if suite.is_critical():
        ...     print("CRITICAL: P0 failures detected!")
        ...     send_alert()
    """
    # Leverage cached collect_results
    results = self.collect_results()

    # Check if any P0 assertion failed
    for result in results:
        if result.severity == "P0" and result.status == "FAILURE":
            return True

    return False
```

**Actions**:
- Add the is_critical method
- Run critical tests to verify they pass

#### Task 3.2: Run all tests and commit
**Commands**:
```bash
# Run critical level tests
uv run pytest tests/test_suite_critical.py -xvs

# Run all API tests to ensure nothing broke
uv run pytest tests/test_api.py -xvs

# Run type checking
uv run mypy src/dqx/api.py

# Run linting
uv run ruff check src/dqx/api.py

# If all pass, commit
git add src/dqx/api.py
git commit -m "feat: add is_critical method for P0 failure detection"
```

### Task Group 4: Integration Testing and Examples

#### Task 4.1: Create integration test for both features
**File**: `tests/test_suite_cache_critical_integration.py`

```python
"""Integration tests for caching and critical level detection."""

import pytest
from datetime import date

from dqx.api import VerificationSuite, check, Context
from dqx.common import ResultKey
from dqx.provider import MetricProvider
from tests.fixtures.datasource import create_datasource
from tests.fixtures.db import InMemoryMetricDB


def test_cache_and_critical_integration():
    """Test that caching and critical detection work together correctly."""
    db = InMemoryMetricDB()

    @check(name="Multi-Severity Check")
    def multi_check(mp: MetricProvider, ctx: Context) -> None:
        total = mp.sum("amount")
        avg = mp.average("amount")

        # P0 that will fail (total is 150, not 1000)
        ctx.assert_that(total).where(
            name="Critical total check", severity="P0"
        ).is_eq(1000)

        # P1 that will pass
        ctx.assert_that(avg).where(
            name="Average check", severity="P1"
        ).is_positive()

        # P2 that will fail
        ctx.assert_that(total).where(
            name="Minor check", severity="P2"
        ).is_lt(100)

    suite = VerificationSuite([multi_check], db, "Integration Test")
    ds = create_datasource("test", {"amount": [50, 50, 50]})
    key = ResultKey(date.today(), {"env": "test"})

    # Run the suite
    suite.run({"test": ds}, key)

    # First access - computes and caches
    results1 = suite.collect_results()
    symbols1 = suite.collect_symbols()
    is_critical1 = suite.is_critical()

    # Verify results
    assert len(results1) == 3  # Three assertions
    assert len(symbols1) == 2  # Two metrics (sum, average)
    assert is_critical1 is True  # Has P0 failure

    # Second access - uses cache
    results2 = suite.collect_results()
    symbols2 = suite.collect_symbols()
    is_critical2 = suite.is_critical()

    # Verify same cached objects
    assert results1 is results2
    assert symbols1 is symbols2
    assert is_critical1 == is_critical2

    # Verify critical detection is working
    p0_failures = [r for r in results1 if r.severity == "P0" and r.status == "FAILURE"]
    assert len(p0_failures) == 1
    assert p0_failures[0].assertion == "Critical total check"
```

**Actions**:
- Write integration test
- Run test to verify it passes

#### Task 4.2: Update example to demonstrate new features
**File**: `examples/cache_and_critical_demo.py`

```python
"""Demonstration of caching and critical level detection features."""

import logging
from datetime import date

from dqx.api import VerificationSuite, check, Context
from dqx.common import ResultKey
from dqx.provider import MetricProvider
from tests.fixtures.datasource import create_datasource
from tests.fixtures.db import InMemoryMetricDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@check(name="Data Quality Checks")
def data_quality_checks(mp: MetricProvider, ctx: Context) -> None:
    """Example check with multiple severity levels."""
    # Critical check - data freshness
    row_count = mp.count()
    ctx.assert_that(row_count).where(
        name="Data freshness - must have recent data",
        severity="P0"
    ).is_gt(1000)  # Will fail if less than 1000 rows

    # Important check - completeness
    null_count = mp.count_null("email")
    ctx.assert_that(null_count).where(
        name="Email completeness",
        severity="P1"
    ).is_eq(0)

    # Regular check - data range
    avg_age = mp.average("age")
    ctx.assert_that(avg_age).where(
        name="Average age in expected range",
        severity="P2"
    ).is_between(18, 65)


def main():
    """Demonstrate caching and critical level detection."""
    # Setup
    db = InMemoryMetricDB()
    suite = VerificationSuite([data_quality_checks], db, "Production Data Quality")

    # Create test data (only 500 rows - will trigger P0 failure)
    data = {
        "email": ["user@example.com"] * 450 + [None] * 50,  # Some nulls
        "age": list(range(20, 70)) * 10,  # Ages 20-69 repeated
    }
    ds = create_datasource("users", data)
    key = ResultKey(date.today(), {"env": "prod", "region": "us-east-1"})

    # Run the suite
    logger.info("Running verification suite...")
    suite.run({"users": ds}, key)

    # Check if critical
    if suite.is_critical():
        logger.critical("🚨 CRITICAL FAILURES DETECTED! P0 assertions failed.")
    else:
        logger.info("✅ No critical failures.")

    # Collect results (first call - computes and caches)
    logger.info("\nCollecting results (first call - will compute)...")
    results = suite.collect_results()

    # Show summary
    failures = [r for r in results if r.status == "FAILURE"]
    logger.info(f"Total assertions: {len(results)}")
    logger.info(f"Failed assertions: {len(failures)}")

    # Show failures by severity
    for severity in ["P0", "P1", "P2", "P3"]:
        severity_failures = [f for f in failures if f.severity == severity]
        if severity_failures:
            logger.warning(f"{severity} failures: {len(severity_failures)}")
            for failure in severity_failures:
                logger.warning(f"  - {failure.assertion}")

    # Collect results again (uses cache)
    logger.info("\nCollecting results (second call - will use cache)...")
    results_cached = suite.collect_results()
    logger.info(f"Results are same object: {results is results_cached}")

    # Collect symbols (demonstrates symbol caching)
    logger.info("\nCollecting symbols...")
    symbols = suite.collect_symbols()
    logger.info(f"Total symbols computed: {len(symbols)}")

    # Show symbol values
    for symbol in symbols:
        if symbol.value.is_success():
            logger.info(f"  {symbol.metric}: {symbol.value.unwrap()}")
        else:
            logger.error(f"  {symbol.metric}: FAILED")

    # Collect symbols again (uses cache)
    symbols_cached = suite.collect_symbols()
    logger.info(f"Symbols are same object: {symbols is symbols_cached}")


if __name__ == "__main__":
    main()
```

**Actions**:
- Write example demonstrating both features
- Test the example runs correctly

#### Task 4.3: Run all tests and commit
**Commands**:
```bash
# Run integration test
uv run pytest tests/test_suite_cache_critical_integration.py -xvs

# Run the example
uv run python examples/cache_and_critical_demo.py

# Run all related tests
uv run pytest tests/test_suite_caching.py tests/test_suite_critical.py tests/test_suite_cache_critical_integration.py -xvs

# If all pass, commit
git add tests/test_suite_cache_critical_integration.py examples/cache_and_critical_demo.py
git commit -m "test: add integration tests and example for cache and critical features"
```

### Task Group 5: Documentation and Final Verification

#### Task 5.1: Update API documentation
**File**: `src/dqx/api.py`

Ensure all docstrings are complete and accurate for:
- `__init__` method (mentioning cache attributes)
- `collect_results` method (mentioning caching behavior)
- `collect_symbols` method (mentioning caching behavior)
- `is_critical` method (new method)

#### Task 5.2: Run comprehensive test suite
**Commands**:
```bash
# Run all tests to ensure nothing is broken
uv run pytest tests/ -xvs

# Run mypy on entire codebase
uv run mypy src/

# Run ruff on entire codebase
uv run ruff check src/

# Run pre-commit hooks
uv run pre-commit run --all-files
```

#### Task 5.3: Update changelog and commit
**File**: `CHANGELOG.md`

Add entry for the new features under the appropriate version section.

**Commands**:
```bash
# If all tests pass, commit documentation updates
git add -A
git commit -m "docs: update documentation for cache and critical level features"
```

## Summary

This plan implements:

1. **In-memory caching** for `collect_results()` and `collect_symbols()`:
   - Cache persists for the lifetime of the VerificationSuite instance
   - No cache clearing mechanism (as requested)
   - Significant performance improvement for repeated calls

2. **Critical level detection** via `is_critical()`:
   - Returns True if any P0 assertion failed
   - Leverages cached results for efficiency
   - Simple boolean interface for alert systems

The implementation follows TDD principles with comprehensive test coverage and maintains backward compatibility. The caching is transparent to users and the is_critical method provides a clean API for detecting critical failures.
