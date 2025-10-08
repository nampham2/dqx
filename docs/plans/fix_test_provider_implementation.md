# Implementation Plan: Fix test_provider.py Tests

## Executive Summary

The tests in `tests/test_provider.py` are failing because they were written for an older version of the `SymbolicMetric` dataclass and `_register` method. The tests need to be updated to match the current implementation without modifying the source code.

**Total Estimated Time**: 2-3 hours  
**Risk Level**: Low (only test modifications)  
**Dependencies**: None

## Background Context

### What is Provider?

The `provider.py` module is part of the DQX (Data Quality eXcellence) framework. It provides:
- **MetricProvider**: Creates symbolic representations of metrics (averages, sums, counts, etc.)
- **SymbolicMetric**: A dataclass that holds metric metadata
- **ExtendedMetricProvider**: Provides advanced metrics like day-over-day comparisons

### Current Problem

10 out of 33 tests are failing due to mismatches between test expectations and actual implementation:

1. Tests expect `SymbolicMetric` to have `dependencies` and `datasets` (plural) attributes
2. Actual implementation has `metric_spec` and `dataset` (singular) attributes
3. The `_register` method has different parameter order in tests vs implementation

## Pre-Implementation Checklist

- [ ] Ensure you're in the project root: `/Users/npham/git-tree/dqx`
- [ ] Verify git status is clean: `git status`
- [ ] Create a new branch: `git checkout -b fix-test-provider`
- [ ] Run initial test to see failures: `uv run pytest tests/test_provider.py -v`

## Implementation Tasks

### Task 1: Fix TestSymbolicMetric Class Tests

**Files to modify**: `tests/test_provider.py`

**What's wrong**:
- Tests are trying to create `SymbolicMetric` with `dependencies` and `datasets` parameters
- Actual class expects `metric_spec` and `dataset` parameters

**Step-by-step fix**:

1. **Fix test_symbolic_metric_creation** (around line 20):
   ```python
   # OLD CODE (WRONG):
   metric = SymbolicMetric(
       name="test_metric",
       symbol=symbol,
       fn=fn,
       key_provider=key_provider,
       dependencies=dependencies,
       datasets=["dataset1", "dataset2"],
   )
   
   # NEW CODE (CORRECT):
   metric = SymbolicMetric(
       name="test_metric",
       symbol=symbol,
       fn=fn,
       key_provider=key_provider,
       metric_spec=Mock(spec=specs.MetricSpec),
       dataset="dataset1",  # Note: singular, and only one dataset
   )
   ```

2. **Fix test_symbolic_metric_default_datasets** (around line 45):
   ```python
   # OLD CODE (WRONG):
   metric = SymbolicMetric(
       name="test_metric", symbol=symbol, fn=fn, key_provider=key_provider, dependencies=dependencies
   )
   assert metric.datasets == []
   
   # NEW CODE (CORRECT):
   metric = SymbolicMetric(
       name="test_metric", 
       symbol=symbol, 
       fn=fn, 
       key_provider=key_provider, 
       metric_spec=Mock(spec=specs.MetricSpec)
   )
   assert metric.dataset is None  # Default is None, not []
   ```

**How to test this task**:
```bash
uv run pytest tests/test_provider.py::TestSymbolicMetric -v
```

**Expected outcome**: 2 tests should pass

**Commit after success**:
```bash
git add tests/test_provider.py
git commit -m "fix: update TestSymbolicMetric tests to match current implementation"
```

### Task 2: Fix _register Method Calls

**What's wrong**:
- Tests call `_register` with 6 parameters in wrong order
- Actual method expects different parameters

**Current test signature**: `_register(symbol, name, fn, key_provider, dependencies, datasets)`  
**Actual signature**: `_register(symbol, name, fn, key, metric_spec, dataset)`

**Step-by-step fix**:

1. **Fix test_register in TestSymbolicMetricBase** (around line 150):
   ```python
   # OLD CODE (WRONG):
   base._register(symbol, "test_metric", fn, key_provider, dependencies, datasets)
   
   # NEW CODE (CORRECT):
   base._register(
       symbol=symbol,
       name="test_metric",
       fn=fn,
       key=key_provider,  # Note: parameter name is 'key' not 'key_provider'
       metric_spec=Mock(spec=specs.MetricSpec),
       dataset="dataset1"  # singular, not plural
   )
   ```

2. **Update assertions in test_register**:
   ```python
   # Remove these lines:
   assert registered_metric.dependencies == dependencies
   assert registered_metric.datasets == datasets
   
   # Add these lines:
   assert registered_metric.metric_spec is not None
   assert registered_metric.dataset == "dataset1"
   ```

3. **Fix all other _register calls** (search for `base._register` and `provider._register`):
   - test_symbols_with_registered_symbols (line ~85)
   - test_get_symbol_success (line ~98)
   - test_evaluate_success (line ~175)
   - test_evaluate_failure (line ~185)

**How to test this task**:
```bash
uv run pytest tests/test_provider.py::TestSymbolicMetricBase -v
```

**Expected outcome**: All TestSymbolicMetricBase tests should pass

**Commit after success**:
```bash
git add tests/test_provider.py
git commit -m "fix: update _register method calls to match implementation"
```

### Task 3: Fix MetricProvider Tests

**What's wrong**:
- Tests expect `datasets` (plural) but implementation uses `dataset` (singular)
- Tests check for `dependencies` attribute that doesn't exist

**Step-by-step fix**:

1. **Fix test_metric** (around line 235):
   ```python
   # OLD CODE (WRONG):
   datasets = ["dataset1"]
   symbol = provider.metric(mock_metric_spec, mock_key_provider, datasets)
   
   # NEW CODE (CORRECT):
   dataset = "dataset1"  # singular
   symbol = provider.metric(mock_metric_spec, mock_key_provider, dataset)
   ```

2. **Update assertions**:
   ```python
   # OLD CODE (WRONG):
   assert registered_metric.datasets == datasets
   assert registered_metric.dependencies[0] == (mock_metric_spec, mock_key_provider)
   
   # NEW CODE (CORRECT):
   assert registered_metric.dataset == dataset
   assert registered_metric.metric_spec == mock_metric_spec
   ```

3. **Fix test_metric_with_defaults** (around line 255):
   ```python
   # OLD CODE (WRONG):
   assert registered_metric.datasets == []
   
   # NEW CODE (CORRECT):
   assert registered_metric.dataset is None
   ```

**How to test this task**:
```bash
uv run pytest tests/test_provider.py::TestMetricProvider -v
```

**Expected outcome**: All TestMetricProvider tests should pass

**Commit after success**:
```bash
git add tests/test_provider.py
git commit -m "fix: update MetricProvider tests for dataset/datasets mismatch"
```

### Task 4: Fix ExtendedMetricProvider Tests

**What's wrong**: Same issues - `datasets` vs `dataset` and missing `dependencies`

**Files to modify**: `tests/test_provider.py`

**Step-by-step fix**:

1. **Fix test_day_over_day** (around line 440):
   ```python
   # OLD CODE (WRONG):
   datasets = ["dataset1"]
   symbol = ext_provider.day_over_day(mock_metric, mock_key_provider, datasets)
   assert registered_metric.datasets == datasets
   assert registered_metric.dependencies[0] == (mock_metric, mock_key_provider)
   
   # NEW CODE (CORRECT):
   dataset = "dataset1"
   symbol = ext_provider.day_over_day(mock_metric, mock_key_provider, dataset)
   assert registered_metric.dataset == dataset
   assert registered_metric.metric_spec == mock_metric
   ```

2. **Fix test_day_over_day_defaults** (around line 455):
   ```python
   # OLD CODE (WRONG):
   assert registered_metric.datasets == []
   
   # NEW CODE (CORRECT):
   assert registered_metric.dataset is None
   ```

3. **Apply same fixes to test_stddev and test_stddev_defaults**

**How to test this task**:
```bash
uv run pytest tests/test_provider.py::TestExtendedMetricProvider -v
```

**Expected outcome**: All ExtendedMetricProvider tests should pass

**Commit after success**:
```bash
git add tests/test_provider.py
git commit -m "fix: update ExtendedMetricProvider tests"
```

### Task 5: Remove Invalid Type Alias Test

**What's wrong**: The test expects a `Dependency` type alias that doesn't exist

**Step-by-step fix**:

1. **Remove test_dependency_type** method entirely from TestTypeAliases class
   - This test references a `Dependency` type that doesn't exist in the implementation

**How to test this task**:
```bash
uv run pytest tests/test_provider.py::TestTypeAliases -v
```

**Expected outcome**: Remaining TestTypeAliases test should pass

**Commit after success**:
```bash
git add tests/test_provider.py
git commit -m "fix: remove test for non-existent Dependency type alias"
```

### Task 6: Final Verification

1. **Run all tests**:
   ```bash
   uv run pytest tests/test_provider.py -v
   ```
   
   **Expected**: All 33 tests should pass

2. **Check code quality**:
   ```bash
   uv run mypy src/dqx/provider.py
   uv run ruff check tests/test_provider.py
   ```

3. **Run coverage to ensure tests are comprehensive**:
   ```bash
   uv run pytest tests/test_provider.py -v --cov=dqx.provider
   ```

4. **Final commit**:
   ```bash
   git add -A
   git commit -m "fix: all test_provider tests now passing"
   ```

## Common Pitfalls to Avoid

1. **Don't change the source code** - Only modify tests
2. **Watch for plural vs singular** - `dataset` not `datasets`
3. **Parameter names matter** - `key` not `key_provider` in _register
4. **Don't assume empty list defaults** - `dataset` defaults to `None` not `[]`
5. **Remove references to non-existent attributes** - No `dependencies` attribute exists

## Testing Philosophy (TDD)

Even though we're fixing existing tests, follow TDD principles:
1. Run the test first to see it fail
2. Make the minimal change to make it pass
3. Refactor if needed (but keep changes minimal)
4. Commit frequently

## Quick Reference Commands

```bash
# Run all provider tests
uv run pytest tests/test_provider.py -v

# Run specific test class
uv run pytest tests/test_provider.py::TestSymbolicMetric -v

# Run with coverage
uv run pytest tests/test_provider.py -v --cov=dqx.provider

# Check code quality
uv run ruff check tests/test_provider.py
uv run mypy src/dqx/provider.py

# Git commands
git status
git diff
git add tests/test_provider.py
git commit -m "fix: [description]"
```

## Success Criteria

- [ ] All 33 tests in test_provider.py pass
- [ ] No changes made to src/dqx/provider.py
- [ ] Code passes ruff and mypy checks
- [ ] Each fix is committed separately
- [ ] Final test run shows 100% pass rate

## Time Estimates

- Task 1: 20 minutes
- Task 2: 30 minutes
- Task 3: 25 minutes
- Task 4: 25 minutes
- Task 5: 10 minutes
- Task 6: 10 minutes
- **Total**: ~2 hours

## Questions You Might Have

**Q: Why are we not changing the source code?**  
A: The source code is the source of truth. The tests were written for an older version and need to be updated.

**Q: What if I find a bug in the source code?**  
A: Note it down, but don't fix it in this task. Create a separate issue for source code bugs.

**Q: Can I refactor the tests to be cleaner?**  
A: Keep changes minimal. The goal is to make tests pass, not to refactor. You can propose refactoring in a separate task.
