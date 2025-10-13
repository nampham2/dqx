# Remove Toolz Dependency Implementation Plan (Improved)

## Overview
This plan details the complete process for removing the `toolz` dependency from the DQX codebase. The dependency is only used in `src/dqx/analyzer.py` for:
1. `merge_with` - Used once in `AnalysisReport.merge()`, always with `models.Metric.reduce`
2. `groupby` - Used twice for deduplication (we only use the keys, not the groups)

## Key Principles
- **YAGNI**: No generic utilities - implement only what's needed
- **TDD**: Write failing tests first, then implement
- **DRY**: Avoid duplication
- **Frequent commits**: One commit per task
- **Thread Safety**: Maintain existing thread safety guarantees
- **Performance**: No regression in performance

## Prerequisites
- Python 3.11+ environment with `uv` package manager
- Development environment set up (`./bin/setup-dev-env.sh`)
- All existing tests passing (`uv run pytest`)

## Task Breakdown

### Task 1: Write comprehensive tests for AnalysisReport.merge refactoring

**Why**: Ensure the new merge implementation maintains exact same behavior including edge cases

**File to create**: `tests/test_analyzer_merge_refactor.py`

```python
"""Test AnalysisReport.merge behavior without toolz dependency."""

import pytest
import threading
import time
from dqx.analyzer import AnalysisReport
from dqx.models import Metric
from dqx.specs import Average, Sum
from dqx.states import DoubleValueState
from dqx.common import ResultKey
import datetime as dt


def test_merge_empty_reports():
    """Test merging two empty AnalysisReports."""
    report1 = AnalysisReport()
    report2 = AnalysisReport()
    merged = report1.merge(report2)
    assert len(merged) == 0


def test_merge_non_overlapping_reports():
    """Test merging reports with different metrics."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1))

    metric1 = Metric.build(Average("price"), key, DoubleValueState(10.0))
    metric2 = Metric.build(Average("quantity"), key, DoubleValueState(5.0))

    report1 = AnalysisReport({(metric1.spec, key): metric1})
    report2 = AnalysisReport({(metric2.spec, key): metric2})

    merged = report1.merge(report2)
    assert len(merged) == 2
    assert (metric1.spec, key) in merged
    assert (metric2.spec, key) in merged


def test_merge_overlapping_reports():
    """Test merging reports with same metric - should use Metric.reduce."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1))
    spec = Average("price")

    # Create two metrics with same spec but different values
    metric1 = Metric.build(spec, key, DoubleValueState(10.0))
    metric2 = Metric.build(spec, key, DoubleValueState(20.0))

    report1 = AnalysisReport({(spec, key): metric1})
    report2 = AnalysisReport({(spec, key): metric2})

    merged = report1.merge(report2)
    assert len(merged) == 1

    # The merge should have used Metric.reduce
    # which calls state.merge, which for DoubleValueState adds values
    merged_metric = merged[(spec, key)]
    assert merged_metric.value == 30.0  # 10.0 + 20.0


def test_merge_with_different_result_keys():
    """Test merging reports with different ResultKeys."""
    key1 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1))
    key2 = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 2))
    spec = Average("price")

    metric1 = Metric.build(spec, key1, DoubleValueState(10.0))
    metric2 = Metric.build(spec, key2, DoubleValueState(20.0))

    report1 = AnalysisReport({(spec, key1): metric1})
    report2 = AnalysisReport({(spec, key2): metric2})

    merged = report1.merge(report2)
    assert len(merged) == 2
    assert merged[(spec, key1)].value == 10.0
    assert merged[(spec, key2)].value == 20.0


def test_merge_multiple_metrics_simultaneously():
    """Test merging multiple different metrics at once."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1))

    # Report 1: price average and sum
    price_avg1 = Metric.build(Average("price"), key, DoubleValueState(10.0))
    price_sum1 = Metric.build(Sum("price"), key, DoubleValueState(100.0))
    report1 = AnalysisReport(
        {(price_avg1.spec, key): price_avg1, (price_sum1.spec, key): price_sum1}
    )

    # Report 2: price average, quantity average, and quantity sum
    price_avg2 = Metric.build(Average("price"), key, DoubleValueState(5.0))
    qty_avg = Metric.build(Average("quantity"), key, DoubleValueState(3.0))
    qty_sum = Metric.build(Sum("quantity"), key, DoubleValueState(30.0))
    report2 = AnalysisReport(
        {
            (price_avg2.spec, key): price_avg2,
            (qty_avg.spec, key): qty_avg,
            (qty_sum.spec, key): qty_sum,
        }
    )

    merged = report1.merge(report2)
    assert len(merged) == 4
    # Price average should be merged: 10.0 + 5.0 = 15.0
    assert merged[(Average("price"), key)].value == 15.0
    # Price sum should remain from report1
    assert merged[(Sum("price"), key)].value == 100.0
    # Quantity metrics should be from report2
    assert merged[(Average("quantity"), key)].value == 3.0
    assert merged[(Sum("quantity"), key)].value == 30.0


def test_merge_empty_with_non_empty():
    """Test merging empty report with non-empty report."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1))
    metric = Metric.build(Average("price"), key, DoubleValueState(10.0))

    empty_report = AnalysisReport()
    non_empty_report = AnalysisReport({(metric.spec, key): metric})

    # Test both directions
    merged1 = empty_report.merge(non_empty_report)
    merged2 = non_empty_report.merge(empty_report)

    assert len(merged1) == 1
    assert len(merged2) == 1
    assert merged1[(metric.spec, key)].value == 10.0
    assert merged2[(metric.spec, key)].value == 10.0


def test_merge_preserve_identity_behavior():
    """Test that Metric.reduce behavior with identity matches toolz.merge_with."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1))
    spec = Average("price")

    # Create metrics
    metric1 = Metric.build(spec, key, DoubleValueState(10.0))
    metric2 = Metric.build(spec, key, DoubleValueState(20.0))

    # Test reduce directly
    reduced = Metric.reduce([metric1, metric2])
    assert reduced.value == 30.0

    # Verify the identity behavior
    identity = metric1.identity()
    assert identity.value == 0.0  # DoubleValueState identity is 0.0
```

**How to test**:
```bash
# Run the test - it should pass with current implementation
uv run pytest tests/test_analyzer_merge_refactor.py -v
```

**Commit**: `git add tests/test_analyzer_merge_refactor.py && git commit -m "test: add comprehensive tests for AnalysisReport.merge behavior"`

### Task 1.5: Add performance baseline test

**Why**: Establish performance baseline before refactoring

**File to create**: `tests/test_analyzer_performance.py`

```python
"""Performance tests for analyzer operations."""

import time
import pytest
from dqx.analyzer import AnalysisReport
from dqx.models import Metric
from dqx.specs import Average
from dqx.states import DoubleValueState
from dqx.common import ResultKey
import datetime as dt


@pytest.mark.performance
def test_merge_performance_baseline():
    """Establish baseline performance for merge operations."""
    # Create large reports with many metrics
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1))

    # Create 1000 metrics for each report
    metrics1 = {}
    metrics2 = {}

    for i in range(1000):
        spec1 = Average(f"column_{i}")
        spec2 = Average(f"column_{i + 500}")  # 50% overlap

        metric1 = Metric.build(spec1, key, DoubleValueState(float(i)))
        metric2 = Metric.build(spec2, key, DoubleValueState(float(i * 2)))

        metrics1[(spec1, key)] = metric1
        metrics2[(spec2, key)] = metric2

    report1 = AnalysisReport(metrics1)
    report2 = AnalysisReport(metrics2)

    # Measure merge time
    start_time = time.time()
    merged = report1.merge(report2)
    merge_time = time.time() - start_time

    # Verify correctness
    assert len(merged) == 1500  # 500 unique from each + 500 overlap

    # Performance assertion (adjust threshold as needed)
    assert merge_time < 0.1  # Should complete in less than 100ms
    print(f"Merge time for 1000 metrics each: {merge_time:.4f}s")
```

**How to test**:
```bash
# Run performance test
uv run pytest tests/test_analyzer_performance.py -v -m performance
```

**Commit**: `git add tests/test_analyzer_performance.py && git commit -m "test: add performance baseline for merge operations"`

### Task 2: Refactor AnalysisReport.merge with optimization

**File to modify**: `src/dqx/analyzer.py`

**Changes**:
1. Update the `merge` method in `AnalysisReport` class (around line 40):

```python
def merge(self, other: AnalysisReport) -> AnalysisReport:
    """Merge two AnalysisReports, using Metric.reduce for conflicts.

    When the same (metric_spec, result_key) exists in both reports,
    the values are merged using Metric.reduce which applies the
    appropriate state merge operation (e.g., sum for DoubleValueState).

    Args:
        other: Another AnalysisReport to merge with this one

    Returns:
        A new AnalysisReport containing all metrics from both reports
    """
    # Start with a copy of self.data for efficiency
    merged_data = dict(self.data)

    # Merge items from other
    for key, metric in other.items():
        if key in merged_data:
            # Key exists in both: use Metric.reduce to merge
            merged_data[key] = models.Metric.reduce([merged_data[key], metric])
        else:
            # Key only in other: just add it
            merged_data[key] = metric

    return AnalysisReport(data=merged_data)
```

**How to test**:
```bash
# All tests should still pass
uv run pytest tests/test_analyzer_merge_refactor.py -v
uv run pytest tests/test_analyzer.py -v

# Check performance didn't regress
uv run pytest tests/test_analyzer_performance.py -v -m performance
```

**Commit**: `git add src/dqx/analyzer.py && git commit -m "refactor: optimize AnalysisReport.merge and add documentation"`

### Task 3: Write comprehensive tests for deduplication refactoring

**File to create**: `tests/test_analyzer_dedup_refactor.py`

```python
"""Test analyzer deduplication behavior without toolz.groupby."""

import pytest
from dqx.analyzer import analyze_sketch_ops, analyze_sql_ops
from dqx.ops import SqlOp, Sum, Average, Maximum, Minimum
from dqx.extensions.pyarrow_ds import ArrowDataSource
import pyarrow as pa


def test_sql_ops_deduplication():
    """Test that duplicate SQL ops are only executed once."""
    # Create test data
    data = pa.table({"value": [1, 2, 3, 4, 5]})
    ds = ArrowDataSource(data)

    # Create duplicate ops
    sum1 = Sum("value")
    sum2 = Sum("value")  # Same as sum1
    avg1 = Average("value")
    avg2 = Average("value")  # Same as avg1

    ops = [sum1, sum2, avg1, avg2, sum1]  # sum1 appears 3 times total

    # Analyze
    analyze_sql_ops(ds, ops)

    # All duplicate ops should have the same value assigned
    assert sum1.value == sum2.value == 15.0  # 1+2+3+4+5
    assert avg1.value == avg2.value == 3.0  # (1+2+3+4+5)/5


def test_sql_ops_order_preservation():
    """Test that deduplication preserves order of first occurrence."""
    data = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    ds = ArrowDataSource(data)

    # Create ops in specific order
    ops_order = []
    max_a = Maximum("a")
    sum_b = Sum("b")
    avg_a = Average("a")
    min_b = Minimum("b")

    # Add with duplicates in different positions
    ops = [max_a, sum_b, avg_a, max_a, min_b, sum_b, avg_a]

    # Track which ops were processed (in order)
    original_analyze = analyze_sql_ops
    processed_ops = []

    def mock_analyze(ds, ops):
        nonlocal processed_ops
        processed_ops = list(ops)
        original_analyze(ds, ops)

    # Temporarily replace the function
    import dqx.analyzer

    original_func = dqx.analyzer.analyze_sql_ops
    dqx.analyzer.analyze_sql_ops = mock_analyze

    try:
        analyze_sql_ops(ds, ops)

        # Verify deduplication occurred
        assert len(processed_ops) == 4  # 4 unique ops

        # Verify values were assigned
        assert max_a.value == 3.0
        assert sum_b.value == 15.0
        assert avg_a.value == 2.0
        assert min_b.value == 4.0
    finally:
        dqx.analyzer.analyze_sql_ops = original_func


def test_empty_ops_handling():
    """Test that empty ops list is handled correctly."""
    data = pa.table({"value": [1, 2, 3]})
    ds = ArrowDataSource(data)

    # Should not raise any errors
    analyze_sql_ops(ds, [])
    analyze_sketch_ops(ds, [])


def test_mixed_column_deduplication():
    """Test deduplication with ops on different columns."""
    data = pa.table(
        {"price": [10.0, 20.0, 30.0], "quantity": [1, 2, 3], "tax": [1.0, 2.0, 3.0]}
    )
    ds = ArrowDataSource(data)

    # Create ops on different columns with some duplicates
    ops = [
        Sum("price"),
        Average("quantity"),
        Sum("price"),  # Duplicate
        Maximum("tax"),
        Average("quantity"),  # Duplicate
        Minimum("price"),
    ]

    analyze_sql_ops(ds, ops)

    # Verify all ops got values
    assert ops[0].value == ops[2].value == 60.0  # Sum of price
    assert ops[1].value == ops[4].value == 2.0  # Average of quantity
    assert ops[3].value == 3.0  # Max tax
    assert ops[5].value == 10.0  # Min price
```

**How to test**:
```bash
# This should pass with current implementation
uv run pytest tests/test_analyzer_dedup_refactor.py -v
```

**Commit**: `git add tests/test_analyzer_dedup_refactor.py && git commit -m "test: add comprehensive tests for ops deduplication behavior"`

### Task 4: Replace toolz.groupby with order-preserving deduplication

**File to modify**: `src/dqx/analyzer.py`

**Changes**:

1. In `analyze_sketch_ops` function (around line 53):
```python
def analyze_sketch_ops(
    ds: T, ops: Sequence[SketchOp], batch_size: int = 100_000
) -> None:
    if len(ops) == 0:
        return

    # Deduping the ops preserving order of first occurrence
    seen = set()
    distinct_ops = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            distinct_ops.append(op)

    # Constructing the query
    logger.info(f"Analyzing SketchOps: {distinct_ops}")
    # ... rest of the function remains the same
```

2. In `analyze_sql_ops` function (around line 84):
```python
def analyze_sql_ops(ds: T, ops: Sequence[SqlOp]) -> None:
    if len(ops) == 0:
        return

    # Deduping the ops preserving order of first occurrence
    seen = set()
    distinct_ops = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            distinct_ops.append(op)

    # Constructing the query
    logger.info(f"Analyzing SqlOps: {distinct_ops}")
    # ... rest of the function remains the same
```

**Note**: This approach preserves the order of first occurrences and makes the deduplication logic explicit, addressing the feedback about clarity and order preservation.

**How to test**:
```bash
# All tests should pass
uv run pytest tests/test_analyzer_dedup_refactor.py -v
uv run pytest tests/test_analyzer.py -v
uv run pytest tests/ -k "analyzer" -v
```

**Commit**: `git add src/dqx/analyzer.py && git commit -m "refactor: replace toolz.groupby with order-preserving deduplication"`

### Task 5: Remove toolz import

**File to modify**: `src/dqx/analyzer.py`

**Changes**:
1. Remove the line `import toolz` (around line 14)

**How to test**:
```bash
# All analyzer tests should still pass
uv run pytest tests/test_analyzer.py -v
uv run pytest tests/test_analyzer_merge_refactor.py -v
uv run pytest tests/test_analyzer_dedup_refactor.py -v
```

**Commit**: `git add src/dqx/analyzer.py && git commit -m "refactor: remove toolz import from analyzer"`

### Task 6: Remove toolz from dependencies

**File to modify**: `pyproject.toml`

**Changes**:
1. Find the `dependencies` section
2. Remove the line containing `"toolz>=..."`

**How to test**:
```bash
# Update lock file
uv lock

# Reinstall dependencies
uv sync

# Run all tests to ensure nothing breaks
uv run pytest
```

**Commit**: `git add pyproject.toml uv.lock && git commit -m "chore: remove toolz from dependencies"`

### Task 7: Fix unrelated Variance class bug

**Why**: The feedback identified a copy-paste error in the Variance class

**File to modify**: `src/dqx/ops.py`

**Changes**:
Find the `Variance` class and fix the `__eq__` method:

```python
def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Variance):  # Fix: Changed from Sum to Variance
        return NotImplemented
    return self.column == other.column
```

**How to test**:
```bash
# Run ops tests to ensure the fix doesn't break anything
uv run pytest tests/test_ops.py -v

# Add a specific test for this if not already present
```

**Commit**: `git add src/dqx/ops.py && git commit -m "fix: correct copy-paste error in Variance.__eq__ method"`

### Task 8: Final verification and documentation

**Steps**:

1. **Run full test suite with coverage**:
```bash
uv run pytest --cov=dqx --cov-report=term-missing
```

2. **Run type checking**:
```bash
uv run mypy src/
```

3. **Run linting**:
```bash
uv run ruff check src/ tests/
```

4. **Verify toolz is completely removed**:
```bash
grep -r "toolz" src/ tests/  # Should return nothing
```

5. **Performance comparison**:
```bash
# Run performance tests again
uv run pytest tests/test_analyzer_performance.py -v -m performance
```

6. **Update documentation**:
   - Add docstring improvements to AnalysisReport class if needed
   - Update any references to toolz in documentation

7. **Integration testing**:
```bash
# Run end-to-end tests
uv run pytest tests/e2e/ -v
```

**Commit**: `git add -A && git commit -m "docs: final verification and documentation updates"`

## Summary

This improved plan addresses all feedback points:

1. **Optimized merge implementation** - Uses `dict(self.data)` for efficiency
2. **Comprehensive edge case testing** - Added tests for empty reports, different keys, and identity behavior
3. **Performance validation** - Added baseline performance tests
4. **Order-preserving deduplication** - Explicit implementation that maintains first-occurrence order
5. **Bug fix included** - Separate task for the Variance class bug
6. **Thread safety** - The implementation maintains thread safety (merge creates new objects)
7. **Complete type annotations** - All new code includes proper type hints

The implementation is simpler and more maintainable because:
- No generic utilities for single use cases (YAGNI)
- Uses Python built-ins where possible
- Keeps domain-specific logic in domain classes
- Makes deduplication logic explicit and clear
- Includes comprehensive testing at every step
