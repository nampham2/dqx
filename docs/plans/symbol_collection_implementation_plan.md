# Symbol Collection Implementation Plan

## Overview
This plan outlines the implementation of a new feature to collect all symbols (metrics) after a VerificationSuite has been executed. The feature allows users to retrieve a comprehensive list of all computed metrics with their values, metadata, and evaluation context.

## Background

### What is DQX?
DQX is a data quality framework that:
- Validates data using symbolic expressions (e.g., `average(price) > 0`)
- Uses symbols like `x_1`, `x_2` to represent metrics internally
- Evaluates these symbols against actual data to produce validation results

### Current State
- `VerificationSuite` can run data quality checks and collect assertion results
- `SymbolInfo` dataclass exists but lacks context information (date, suite name, tags)
- `MetricProvider` tracks all symbols but there's no way to collect them after evaluation

### Goal
Add a `collect_symbols()` method to `VerificationSuite` that returns a list of `SymbolInfo` objects containing:
- Symbol name (e.g., "x_1")
- Human-readable metric name
- Dataset name
- Computed value
- Evaluation date
- Suite name
- Tags

### Important Notes
- **No backward compatibility is required** - This is a breaking change and existing code using SymbolInfo will need to be updated
- Per project policy, explicit permission from Nam is required before implementing ANY backward compatibility
- All comments in code should describe what the code does, not reference the refactoring

## Implementation Tasks

### Task 1: Extend SymbolInfo Dataclass

**File to modify**: `src/dqx/common.py`

**Current code** (around line 25):
```python
@dataclass
class SymbolInfo:
    """Information about a symbol in an expression"""
    name: str  # Symbol name (e.g., "x_1")
    metric: str  # Human-readable metric name
    dataset: str  # Dataset name
    value: Result[float, str]  # Success(10.5) or Failure("error")
```

**Change to**:
```python
@dataclass
class SymbolInfo:
    """Information about a symbol in an expression"""
    name: str  # Symbol name (e.g., "x_1")
    metric: str  # Human-readable metric name
    dataset: str  # Dataset name
    value: Result[float, str]  # Success(10.5) or Failure("error")
    yyyy_mm_dd: datetime.date  # Date when the metric was evaluated
    suite: str  # Name of the verification suite
    tags: Tags  # Additional metadata tags
```

**Testing approach**:
1. Create a test that instantiates SymbolInfo with all fields
2. Verify all fields are accessible
3. Ensure existing tests are updated to provide all required fields

**Test file**: Create `test_symbol_info_extension.py` in project root:
```python
import datetime
from returns.result import Success
from dqx.common import SymbolInfo

def test_symbol_info_with_new_fields():
    info = SymbolInfo(
        name="x_1",
        metric="average(price)",
        dataset="orders",
        value=Success(100.5),
        yyyy_mm_dd=datetime.date(2025, 1, 13),
        suite="My Suite",
        tags={"env": "prod"}
    )
    assert info.yyyy_mm_dd == datetime.date(2025, 1, 13)
    assert info.suite == "My Suite"
    assert info.tags == {"env": "prod"}
```

**Run test**: `uv run pytest test_symbol_info_extension.py -v`

**Commit**: `git commit -m "feat: extend SymbolInfo with context fields"`

---

### Task 2: Update Evaluator to Populate New Fields

**Files to modify**:
- `src/dqx/evaluator.py`
- `src/dqx/api.py`

**Step 2.1: Update Evaluator constructor**

In `src/dqx/evaluator.py`, find the `__init__` method (around line 30):

**Current**:
```python
def __init__(self, provider: MetricProvider, key: ResultKey):
    self.provider = provider
    self.key = key
    self._metrics: dict[sp.Basic, Result[float, str]] | None = None
```

**Change to**:
```python
def __init__(self, provider: MetricProvider, key: ResultKey, suite_name: str):
    self.provider = provider
    self.key = key
    self.suite_name = suite_name
    self._metrics: dict[sp.Basic, Result[float, str]] | None = None
```

**Step 2.2: Update _gather method**

In `src/dqx/evaluator.py`, find the `_gather` method (around line 95):

**Current code creates SymbolInfo like**:
```python
symbol_info = SymbolInfo(
    name=str(sym),
    metric=str(sm.metric_spec),
    dataset=sm.dataset or "",
    value=metric_result
)
```

**Change to**:
```python
symbol_info = SymbolInfo(
    name=str(sym),
    metric=str(sm.metric_spec),
    dataset=sm.dataset,
    value=metric_result,
    yyyy_mm_dd=self.key.yyyy_mm_dd,
    suite=self.suite_name,
    tags=self.key.tags
)
```

**Step 2.3: Update VerificationSuite.run()**

In `src/dqx/api.py`, find where Evaluator is created (around line 435):

**Current**:
```python
evaluator = Evaluator(self.provider, key)
```

**Change to**:
```python
evaluator = Evaluator(self.provider, key, self._name)
```

**Testing approach**:
1. Run existing tests to ensure no regression
2. Create a test that verifies SymbolInfo objects have new fields populated during evaluation

**Test**: Add to existing test files or create new test
```python
def test_evaluator_populates_symbol_info_fields():
    # Setup suite and run it
    # Then check that any EvaluationFailure has SymbolInfo with all fields
    pass  # Implementation depends on existing test structure
```

**Run tests**: `uv run pytest tests/test_evaluator.py -v`

**Commit**: `git commit -m "feat: update Evaluator to populate SymbolInfo context fields"`

---

### Task 3: Enhance get_symbol to Accept Strings

**File to modify**: `src/dqx/provider.py`

**Find the get_symbol method** (around line 47):

**Current**:
```python
def get_symbol(self, symbol: sp.Symbol) -> SymbolicMetric:
    """Find the first symbol data that matches the given symbol."""
    first_or_none = next(filter(lambda s: s.symbol == symbol, self._metrics), None)
    if not first_or_none:
        raise DQXError(f"Symbol {symbol} not found.")
    return first_or_none
```

**Change to**:
```python
def get_symbol(self, symbol: sp.Symbol | str) -> SymbolicMetric:
    """Find the first symbol data that matches the given symbol.

    Args:
        symbol: Either a sympy Symbol or string representation (e.g., "x_1")

    Returns:
        The SymbolicMetric containing metadata for the symbol

    Raises:
        DQXError: If the symbol is not found
    """
    # Convert string to Symbol if needed
    if isinstance(symbol, str):
        symbol = sp.Symbol(symbol)

    first_or_none = next(filter(lambda s: s.symbol == symbol, self._metrics), None)
    if not first_or_none:
        raise DQXError(f"Symbol {symbol} not found.")

    return first_or_none
```

**Testing**:
```python
def test_get_symbol_accepts_string():
    provider = MetricProvider(db=Mock())
    # Register a metric
    sym = provider.average("price", dataset="orders")

    # Should work with Symbol
    result1 = provider.get_symbol(sym)

    # Should also work with string
    result2 = provider.get_symbol(str(sym))  # e.g., "x_1"

    assert result1 == result2
```

**Commit**: `git commit -m "feat: enhance get_symbol to accept string symbols"`

---

### Task 4: Add collect_symbols Method

**File to modify**: `src/dqx/api.py`

**Add new method to VerificationSuite class** (after collect_results, around line 475):

```python
def collect_symbols(self) -> list[SymbolInfo]:
    """
    Collect all symbol values after suite execution.

    This method retrieves information about all symbols (metrics) that were
    registered during suite setup, evaluates them, and returns their values
    along with metadata.

    Returns:
        List of SymbolInfo instances, one for each registered symbol.
        Each contains the symbol name, metric description, dataset,
        computed value, and context information (date, suite, tags).

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
    if not self.is_evaluated:
        raise DQXError(
            "Cannot collect symbols before suite execution. "
            "Call run() first to evaluate assertions."
        )

    if self._key is None:
        raise DQXError("No ResultKey available. This should not happen after successful run().")

    symbols = []

    # Iterate through all registered symbols
    for symbolic_metric in self._context.provider.symbolic_metrics:
        # Evaluate the symbol to get its value
        value = symbolic_metric.fn(self._key)

        # Create SymbolInfo with all fields
        symbol_info = SymbolInfo(
            name=str(symbolic_metric.symbol),
            metric=str(symbolic_metric.metric_spec),
            dataset=symbolic_metric.dataset,
            value=value,
            yyyy_mm_dd=self._key.yyyy_mm_dd,
            suite=self._name,
            tags=self._key.tags
        )
        symbols.append(symbol_info)

    return symbols
```

**Testing approach**:
1. Create a simple suite with known metrics
2. Run the suite
3. Call collect_symbols()
4. Verify all symbols are returned with correct values

**Test example**:
```python
def test_collect_symbols_after_run():
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("price"))
           .where(name="Price check")
           .is_positive()

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Should fail before run
    with pytest.raises(DQXError, match="Cannot collect symbols"):
        suite.collect_symbols()

    # Run suite
    datasource = create_test_datasource()  # Helper to create test data
    key = ResultKey(yyyy_mm_dd=date.today(), tags={"test": "true"})
    suite.run({"test": datasource}, key)

    # Now collect symbols
    symbols = suite.collect_symbols()

    assert len(symbols) == 1
    assert symbols[0].name.startswith("x_")
    assert symbols[0].metric == "average(price)"
    assert symbols[0].suite == "Test Suite"
    assert symbols[0].yyyy_mm_dd == date.today()
    assert symbols[0].tags == {"test": "true"}
```

**Commit**: `git commit -m "feat: add collect_symbols method to VerificationSuite"`

---

### Task 5: Update Examples

**Task 5.1: Fix evaluation_failure_demo.py**

**File**: `examples/evaluation_failure_demo.py`

The `print_failure_details` function needs to handle SymbolInfo objects that may or may not have the new fields (for backward compatibility with mocked data).

**Find the function** (around line 20):

Since this is a demo/example file that might use mocked data, we need to handle both old and new SymbolInfo formats. However, this is an exception - the actual library code will require all fields.

**Update to**:
```python
def print_failure_details(failures: list[EvaluationFailure]) -> None:
    """Pretty print evaluation failure details."""
    for i, failure in enumerate(failures, 1):
        print(f"\n🚨 Failure {i}:")
        print(f"   Error: {failure.error_message}")
        print(f"   Expression: {failure.expression}")

        if failure.symbols:
            print(f"   Symbols involved ({len(failure.symbols)}):")
            for symbol in failure.symbols:
                status = "✅ Success" if isinstance(symbol.value, Success) else "❌ Failed"
                print(f"      - {symbol.name}: {symbol.metric} from '{symbol.dataset}' [{status}]")

                # Display additional context if present
                if hasattr(symbol, 'yyyy_mm_dd'):
                    print(f"        Date: {symbol.yyyy_mm_dd}")
                if hasattr(symbol, 'suite'):
                    print(f"        Suite: {symbol.suite}")
                if hasattr(symbol, 'tags') and symbol.tags:
                    print(f"        Tags: {symbol.tags}")

                if isinstance(symbol.value, Failure):
                    print(f"        Error: {symbol.value.failure()}")
                else:
                    print(f"        Value: {symbol.value.unwrap()}")
```

**Commit**: `git commit -m "fix: update evaluation_failure_demo to handle new SymbolInfo fields"`

**Task 5.2: Add symbol collection demo to result_collection_demo.py**

**File**: `examples/result_collection_demo.py`

**Add at the end of main() function** (before the final `if __name__ == "__main__"`):

```python
    # Demonstrate symbol collection
    print("\n" + "=" * 60)
    print("SYMBOL COLLECTION DEMO")
    print("=" * 60)

    # Collect all symbols
    symbols = suite.collect_symbols()
    print(f"\nCollected {len(symbols)} symbols")

    # Display symbols in a table
    print("\nAll Computed Metrics:")
    print("-" * 100)
    print(f"{'Symbol':<10} | {'Metric':<30} | {'Dataset':<15} | {'Value':<20} | {'Status'}")
    print("-" * 100)

    for sym in symbols:
        if isinstance(sym.value, Success):
            value_str = f"{sym.value.unwrap():.2f}"
            status = "✅ Success"
        else:
            value_str = "N/A"
            status = "❌ Failed"

        metric_str = sym.metric[:30]
        if len(sym.metric) > 30:
            metric_str = metric_str[:27] + "..."

        print(f"{sym.name:<10} | {metric_str:<30} | {sym.dataset:<15} | {value_str:<20} | {status}")

    # Group by dataset
    print("\n\nMetrics by Dataset:")
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for sym in symbols:
        by_dataset[sym.dataset].append(sym)

    for dataset, dataset_symbols in sorted(by_dataset.items()):
        print(f"\n  {dataset}: {len(dataset_symbols)} metrics")
        for sym in dataset_symbols[:3]:  # Show first 3
            print(f"    - {sym.metric}")
        if len(dataset_symbols) > 3:
            print(f"    ... and {len(dataset_symbols) - 3} more")
```

**Commit**: `git commit -m "feat: add symbol collection demo to result_collection_demo.py"`

---

### Task 6: Write Comprehensive Tests

**Create new test file**: `tests/test_symbol_collection.py`

```python
"""Test symbol collection functionality."""

import datetime
from unittest.mock import Mock

import pytest
from returns.result import Failure, Success

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import DQXError, ResultKey, SymbolInfo
from dqx.extensions.duck_ds import DuckDbDataSource
from dqx.orm.repositories import InMemoryMetricDB


class TestSymbolCollection:
    """Test suite for symbol collection feature."""

    def test_symbol_info_has_all_fields(self):
        """Test that SymbolInfo can be created with all fields."""
        info = SymbolInfo(
            name="x_1",
            metric="average(price)",
            dataset="orders",
            value=Success(100.5),
            yyyy_mm_dd=datetime.date(2025, 1, 13),
            suite="Test Suite",
            tags={"env": "test"}
        )

        assert info.name == "x_1"
        assert info.metric == "average(price)"
        assert info.dataset == "orders"
        assert info.value == Success(100.5)
        assert info.yyyy_mm_dd == datetime.date(2025, 1, 13)
        assert info.suite == "Test Suite"
        assert info.tags == {"env": "test"}

    def test_collect_symbols_before_run_fails(self):
        """Test that collect_symbols fails if called before run()."""
        db = InMemoryMetricDB()

        @check(name="Test")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="Has rows").is_positive()

        suite = VerificationSuite([test_check], db, "Test Suite")

        with pytest.raises(DQXError, match="Cannot collect symbols before suite execution"):
            suite.collect_symbols()

    def test_collect_symbols_returns_all_symbols(self):
        """Test that collect_symbols returns all registered symbols."""
        db = InMemoryMetricDB()

        @check(name="Multiple Metrics", datasets=["test"])
        def multi_check(mp: MetricProvider, ctx: Context) -> None:
            # Register multiple metrics
            avg = mp.average("value", dataset="test")
            count = mp.num_rows(dataset="test")
            min_val = mp.minimum("value", dataset="test")

            ctx.assert_that(avg).where(name="Avg positive").is_positive()
            ctx.assert_that(count).where(name="Has data").is_gt(0)
            ctx.assert_that(min_val).where(name="Min check").is_geq(0)

        suite = VerificationSuite([multi_check], db, "Multi Metric Suite")

        # Create test data
        datasource = DuckDbDataSource.from_dict({
            "value": [10.0, 20.0, 30.0]
        })

        key = ResultKey(
            yyyy_mm_dd=datetime.date(2025, 1, 13),
            tags={"env": "test", "version": "1.0"}
        )

        suite.run({"test": datasource}, key)

        # Collect symbols
        symbols = suite.collect_symbols()

        # Should have 3 symbols
        assert len(symbols) == 3

        # Check all symbols have proper fields
        for symbol in symbols:
            assert symbol.name.startswith("x_")
            assert symbol.dataset == "test"
            assert symbol.yyyy_mm_dd == datetime.date(2025, 1, 13)
            assert symbol.suite == "Multi Metric Suite"
            assert symbol.tags == {"env": "test", "version": "1.0"}
            assert isinstance(symbol.value, Success)

        # Check specific metrics
        metrics = [s.metric for s in symbols]
        assert "average(value)" in metrics
        assert "num_rows()" in metrics
        assert "minimum(value)" in metrics

    def test_collect_symbols_with_failures(self):
        """Test that collect_symbols includes failed metrics."""
        db = InMemoryMetricDB()

        @check(name="Test")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            mp.sum("amount", dataset="test")
            mp.average("price", dataset="test")

        suite = VerificationSuite([test_check], db, "Test Suite")

        # Mock some metrics to fail
        provider = suite.provider
        for metric in provider.symbolic_metrics:
            if "sum" in str(metric.metric_spec):
                provider._symbol_index[metric.symbol].fn = lambda k: Failure("Database error")

        # Run with empty datasource (other metrics will fail naturally)
        datasource = DuckDbDataSource.from_dict({})
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Run will complete even with failures
        suite.run({"test": datasource}, key)

        symbols = suite.collect_symbols()

        # Should still return all symbols
        assert len(symbols) == 2

        # Check we have both success and failure
        failures = [s for s in symbols if isinstance(s.value, Failure)]
        assert len(failures) > 0

    def test_get_symbol_with_string(self):
        """Test that get_symbol accepts string symbol names."""
        provider = MetricProvider(db=Mock())

        # Register a metric
        sym = provider.average("price", dataset="orders")
        sym_name = str(sym)  # e.g., "x_1"

        # Should work with Symbol object
        result1 = provider.get_symbol(sym)

        # Should also work with string
        result2 = provider.get_symbol(sym_name)

        assert result1 == result2
        assert result1.symbol == sym
        assert result1.metric_spec.name == "average(price)"

    def test_get_symbol_string_not_found(self):
        """Test that get_symbol raises error for unknown string."""
        provider = MetricProvider(db=Mock())

        with pytest.raises(DQXError, match="Symbol x_999 not found"):
            provider.get_symbol("x_999")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Run tests**:
```bash
uv run pytest tests/test_symbol_collection.py -v
```

**Commit**: `git commit -m "test: add comprehensive tests for symbol collection"`

---

### Task 7: Update Documentation

**Create/Update**: `docs/symbol_collection_guide.md`

```markdown
# Symbol Collection Guide

## Overview
The symbol collection feature allows you to retrieve all computed metrics (symbols) after running a VerificationSuite. This is useful for:
- Debugging metric calculations
- Exporting metric values for reporting
- Understanding what metrics were computed during validation

## Usage

```python
from dqx.api import VerificationSuite, check
from dqx.common import ResultKey

# Define your checks
@check(name="Revenue Checks")
def revenue_checks(mp, ctx):
    ctx.assert_that(mp.sum("amount"))
       .where(name="Total revenue positive")
       .is_positive()

# Create and run suite
suite = VerificationSuite([revenue_checks], db, "Daily Revenue")
suite.run(datasources, key)

# Collect all symbols
symbols = suite.collect_symbols()

# Process results
for symbol in symbols:
    if symbol.value.is_success():
        print(f"{symbol.metric}: {symbol.value.unwrap()}")
    else:
        print(f"{symbol.metric}: FAILED - {symbol.value.failure()}")
```

##
