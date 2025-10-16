# Symbol Collection Ordering Fix Implementation Plan v1

## Overview
This plan addresses the issue with symbol ordering in the `collect_symbols` method of `VerificationSuite`. Currently, symbols are sorted lexicographically (x_1, x_10, x_2), but we need natural numeric ordering (x_1, x_2, x_10).

## Background

### Current Issue
- The `collect_symbols` method uses simple string sorting: `sorted(symbols, key=lambda s: s.name)`
- This produces lexicographic ordering: x_1, x_10, x_11, ..., x_19, x_2, x_20, ..., x_9
- Users expect natural numeric ordering: x_1, x_2, x_3, ..., x_9, x_10, x_11, ..., x_20

### Solution
- Extract the numeric part after "x_" and sort by integer value
- All symbols follow the pattern "x_N" where N is a positive integer
- No backward compatibility is required (confirmed by Nam)

### Technical Context
- **Project**: DQX (Data Quality eXcellence) - A high-performance data quality framework
- **File**: `src/dqx/api.py` - Contains the `VerificationSuite` class
- **Method**: `collect_symbols()` - Returns a list of `SymbolInfo` objects
- **Testing**: Project uses pytest with high coverage standards (98%+)
- **Tools**: uv package manager, mypy for type checking, ruff for linting

## Git Strategy
- Create new branch: `feature/natural-symbol-sorting`
- Make focused commits for each task group
- Only commit when all tests pass

## Implementation Tasks

### Task Group 1: Core Implementation (Tasks 1-3)

#### Task 1: Create the test file first (TDD approach)
**File to create**: `tests/test_symbol_ordering.py`

```python
"""Test natural ordering of symbols in collect_symbols method."""

from datetime import date
from returns.result import Success

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.extensions.duck_ds import DuckDbDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_collect_symbols_natural_ordering():
    """Test that symbols are sorted in natural numeric order."""
    db = InMemoryMetricDB()

    @check(name="Many Metrics", datasets=["test"])
    def many_metrics(mp: MetricProvider, ctx: Context) -> None:
        # Create metrics that will generate x_1 through x_15
        # Intentionally create them out of order to test sorting
        mp.sum("col_5", dataset="test")      # x_1
        mp.average("col_2", dataset="test")   # x_2
        mp.minimum("col_10", dataset="test")  # x_3
        mp.maximum("col_1", dataset="test")   # x_4
        mp.variance("col_8", dataset="test")  # x_5
        mp.sum("col_12", dataset="test")     # x_6
        mp.average("col_3", dataset="test")   # x_7
        mp.minimum("col_11", dataset="test")  # x_8
        mp.maximum("col_4", dataset="test")   # x_9
        mp.variance("col_7", dataset="test")  # x_10
        mp.sum("col_9", dataset="test")      # x_11
        mp.average("col_6", dataset="test")   # x_12
        mp.minimum("col_13", dataset="test")  # x_13
        mp.maximum("col_14", dataset="test")  # x_14
        mp.variance("col_15", dataset="test") # x_15

    suite = VerificationSuite([many_metrics], db, "Test Suite")

    # Create test data with all columns
    data = {f"col_{i}": [float(i)] for i in range(1, 16)}
    datasource = DuckDbDataSource.from_dict(data)

    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    suite.run({"test": datasource}, key)

    symbols = suite.collect_symbols()

    # Extract symbol names
    names = [s.name for s in symbols]

    # Should have all 15 symbols
    assert len(names) == 15

    # Verify natural ordering (x_1, x_2, ..., x_10, x_11, ..., x_15)
    expected = [f"x_{i}" for i in range(1, 16)]
    assert names == expected, f"Expected {expected}, but got {names}"

    # Specifically check the problematic transition from single to double digits
    x_9_index = names.index("x_9")
    x_10_index = names.index("x_10")
    assert x_10_index == x_9_index + 1, "x_10 should come immediately after x_9"


def test_collect_symbols_large_numbers():
    """Test natural ordering with larger numbers (x_99, x_100, x_101)."""
    db = InMemoryMetricDB()

    @check(name="Large Numbers", datasets=["test"])
    def large_numbers(mp: MetricProvider, ctx: Context) -> None:
        # Create 105 metrics to test x_1 through x_105
        for i in range(105):
            mp.sum(f"col_{i}", dataset="test")

    suite = VerificationSuite([large_numbers], db, "Test Suite")

    # Create test data
    data = {f"col_{i}": [float(i)] for i in range(105)}
    datasource = DuckDbDataSource.from_dict(data)

    key = ResultKey(yyyy_mm_dd=date.today(), tags={})
    suite.run({"test": datasource}, key)

    symbols = suite.collect_symbols()
    names = [s.name for s in symbols]

    # Check specific transitions
    assert names.index("x_99") < names.index("x_100")
    assert names.index("x_100") < names.index("x_101")
    assert names.index("x_9") < names.index("x_10")
    assert names.index("x_10") < names.index("x_100")

    # Verify complete ordering
    expected = [f"x_{i}" for i in range(1, 106)]
    assert names == expected
```

**How to run**:
```bash
uv run pytest tests/test_symbol_ordering.py -v
```

This test will fail initially (expected behavior in TDD).

#### Task 2: Fix the sorting in collect_symbols
**File to modify**: `src/dqx/api.py`

Find the `collect_symbols` method (around line 515) and locate this line:
```python
# Sort by symbol name before returning
return sorted(symbols, key=lambda s: s.name)
```

Change it to:
```python
# Sort by symbol numeric suffix for natural ordering
return sorted(symbols, key=lambda s: int(s.name.split("_")[1]))
```

**What this does**:
- `s.name.split("_")` splits "x_10" into ["x", "10"]
- `[1]` gets the second element: "10"
- `int()` converts "10" to integer 10
- Sort uses integer comparison: 2 < 10 (not "10" < "2")

#### Task 3: Verify the fix works
Run the test again:
```bash
uv run pytest tests/test_symbol_ordering.py -v
```

The test should now pass.

**Commit after tests pass**:
```bash
git add tests/test_symbol_ordering.py src/dqx/api.py
git commit -m "fix: implement natural numeric ordering for collect_symbols"
```

---

### Task Group 2: Integration Testing (Tasks 4-5)

#### Task 4: Run existing symbol collection tests
Ensure our change doesn't break existing functionality:

```bash
# Run specific test file
uv run pytest tests/test_symbol_collection.py -v

# Run API tests that might use collect_symbols
uv run pytest tests/test_api.py::TestVerificationSuite -v

# Check the e2e test
uv run pytest tests/e2e/test_api_e2e.py::TestCollectSymbols -v
```

All tests should pass. If any fail, investigate and fix.

#### Task 5: Run full test suite
```bash
# Run all tests with coverage
uv run pytest --cov=dqx --cov-report=term-missing

# Ensure coverage hasn't dropped below 98%
```

**Only proceed if all tests pass!**

---

### Task Group 3: Code Quality & Documentation (Tasks 6-8)

#### Task 6: Run code quality checks
```bash
# Type checking
uv run mypy src/dqx/api.py

# Linting
uv run ruff check src/dqx/api.py tests/test_symbol_ordering.py

# If ruff finds issues, fix them:
uv run ruff check --fix src/dqx/api.py tests/test_symbol_ordering.py
```

#### Task 7: Update documentation (if needed)
Check if any documentation mentions the ordering:
```bash
grep -r "collect_symbols" docs/ examples/
```

If found, update to mention natural ordering. Otherwise, no action needed.

#### Task 8: Run pre-commit hooks and final checks
```bash
# Run the pre-commit script
bin/run-hooks.sh

# This runs mypy, ruff, and other checks
```

Fix any issues that arise.

**Final commit**:
```bash
git add -u  # Add any files modified by linting/formatting
git commit -m "style: apply linting and formatting fixes"
```

---

## Summary

### What We're Changing
1. The `collect_symbols` method in `VerificationSuite` to use natural numeric sorting
2. Adding comprehensive tests to verify the ordering

### Key Files
- **Modify**: `src/dqx/api.py` - Change one line in `collect_symbols` method
- **Create**: `tests/test_symbol_ordering.py` - New test file

### Testing Strategy
1. TDD: Write failing test first
2. Fix the implementation
3. Run integration tests
4. Run full test suite
5. Code quality checks

### Expected Outcome
- Symbols will be ordered naturally: x_1, x_2, ..., x_9, x_10, x_11, ..., x_100
- No breaking changes to existing functionality
- All tests pass with maintained coverage

### Time Estimate
- Task Group 1: 15 minutes
- Task Group 2: 10 minutes
- Task Group 3: 10 minutes
- Total: ~35 minutes

### Troubleshooting

**If tests fail after the change**:
1. Check that all symbol names follow the "x_N" pattern
2. Verify the split operation works correctly
3. Look for any symbols with non-standard naming

**If mypy complains**:
- The change should not introduce type issues, but if it does, ensure proper type annotations

**If coverage drops**:
- The new test file should increase coverage, not decrease it
- If it drops, ensure the new test is being discovered by pytest
