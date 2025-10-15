# Remove Batch Support Implementation Plan v2

## Overview
This plan outlines the complete removal of batch processing support from DQX, focusing on simplifying the analyzer to support only single-pass SQL data sources. This includes removing `BatchSqlDataSource` protocol, `ArrowBatchDataSource` implementation, and all threading/batch processing infrastructure.

## Background
- **Goal**: Simplify the analyzer for the first release by removing batch processing complexity
- **Rationale**: Focus on single-pass analyzers to improve performance and reduce complexity
- **Branch**: `feature/remove-batch-support`

## Impact Analysis

### Functionality Changes
- **Lost Capability**: Processing large datasets in chunks (e.g., multiple Parquet files that exceed memory)
- **Performance Impact**: All data must now fit in memory during analysis
- **Use Case Impact**: Users processing TB-scale datasets will need to pre-filter or sample their data

### Migration Path for Current Users
- For large datasets: Pre-aggregate data using external tools (e.g., DuckDB, Spark)
- For Parquet files: Use DuckDB's native Parquet reading with filters/projections
- For memory constraints: Sample data before analysis

## Pre-requisites
- Ensure you're on the `feature/remove-batch-support` branch
- Run `uv run pytest` to verify all tests pass before starting
- Understand the current architecture:
  - `BatchSqlDataSource` is a Protocol in `src/dqx/common.py`
  - `ArrowBatchDataSource` is an implementation in `src/dqx/extensions/pyarrow_ds.py`
  - Analyzer has batch processing methods in `src/dqx/analyzer.py`

## Implementation Phases

### Phase 1: Core Protocol Changes

#### Task 1: Remove BatchSqlDataSource Protocol from common.py

**File**: `src/dqx/common.py`

**Changes**:
1. Remove the entire `BatchSqlDataSource` Protocol class (lines ~195-225)
2. Remove `BatchSqlDataSource` from imports in other files that use it

**What to remove**:
```python
@runtime_checkable
class BatchSqlDataSource(Protocol):
    """
    Protocol for batch SQL data sources that provide data in multiple batches.
    ...
    """
    def batches(self) -> Iterable[SqlDataSource]:
        ...
```

**Testing**: Run `uv run mypy src/dqx/common.py` to ensure no type errors

#### Task 2: Update Analyzer Protocol in common.py

**File**: `src/dqx/common.py`

**Changes**:
1. Find the `Analyzer` Protocol (around line ~260)
2. Update the `analyze()` method signature:
   - Change `ds: SqlDataSource | BatchSqlDataSource` to `ds: SqlDataSource`
   - Remove the `threading: bool = False` parameter completely
3. Update the docstring to remove references to batch processing and threading

**What to change**:
```python
# Before:
def analyze(
    self,
    ds: SqlDataSource | BatchSqlDataSource,
    metrics: Sequence[MetricSpec],
    key: ResultKey,
    threading: bool = False,
) -> AnalysisReport:

# After:
def analyze(
    self,
    ds: SqlDataSource,
    metrics: Sequence[MetricSpec],
    key: ResultKey,
) -> AnalysisReport:
```

**Testing**: Run `uv run mypy src/dqx/common.py` to ensure no type errors

**Phase 1 Commit**: `git commit -m "refactor: remove batch processing protocols"`

### Phase 2: Analyzer Implementation

#### Task 3: Remove batch processing imports from analyzer.py

**File**: `src/dqx/analyzer.py`

**Changes**:
Remove these imports completely:
```python
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
```

Also remove `BatchSqlDataSource` from the imports from `dqx.common`

**Note**: Keep the `Lock` import if `_mutex` is retained (see Task 4)

**Testing**: Run `uv run ruff check src/dqx/analyzer.py` to check for unused imports

#### Task 4: Remove batch processing methods from analyzer.py

**File**: `src/dqx/analyzer.py`

**Changes**:
1. Remove the `_analyze_batches()` method (entire method)
2. Remove the `_analyze_batches_threaded()` method (entire method)
3. **KEEP the `_mutex` attribute** in `__init__` - it's still needed for thread-safe merge in `analyze_single()`
4. Keep the `from threading import Lock` import

**Important**: The mutex is retained because `analyze_single()` could still be called from multiple threads and needs thread-safe report merging.

**Testing**: Run `uv run mypy src/dqx/analyzer.py` to ensure no type errors

#### Task 4.5: Verify AnalysisReport.merge() Usage

**File**: `src/dqx/analyzer.py`

**Analysis**:
1. Check the `analyze_single()` method's use of `merge()`
2. Confirm it's still needed for combining metric results
3. Document why merge is retained despite batch removal

**Expected Finding**: The merge is still needed because:
- Multiple metrics can produce results that need to be combined
- The analyzer accumulates results across multiple analyze_single calls
- Thread safety is still required for concurrent usage

**No code changes needed** - just verify the understanding

#### Task 5: Simplify analyze() method in analyzer.py

**File**: `src/dqx/analyzer.py`

**Changes**:
1. Update the `analyze()` method signature to remove `threading` parameter completely
2. Remove all the conditional logic for batch processing
3. Make it simply call `analyze_single()` directly

**What to change**:
```python
def analyze(
    self,
    ds: SqlDataSource,
    metrics: Sequence[MetricSpec],
    key: ResultKey,
) -> AnalysisReport:
    return self.analyze_single(ds, metrics, key)
```

**Testing**:
- Run `uv run mypy src/dqx/analyzer.py`
- Run `uv run pytest tests/test_analyzer.py::TestAnalyzer::test_analyzer_single_analysis -v`

**Phase 2 Commit**: `git commit -m "refactor: simplify analyzer to single-source only"`

### Phase 3: Extension & Test Cleanup

#### Task 6: Remove ArrowBatchDataSource from pyarrow_ds.py

**File**: `src/dqx/extensions/pyarrow_ds.py`

**Changes**:
1. Remove the entire `ArrowBatchDataSource` class
2. Remove `ArrowBatchDataSource` from module exports (`__all__` if present)
3. Update the module docstring to remove references to batch processing
4. Remove these imports that are only used by `ArrowBatchDataSource`:
   - `from typing import Iterable`
   - `from pyarrow.dataset import dataset`
5. Remove the `MAX_ARROW_BATCH_SIZE` constant

**What to remove**:
- The entire class from `class ArrowBatchDataSource:` to the end of its methods
- Update docstring to remove mentions of `ArrowBatchDataSource`

**Testing**: Run `uv run mypy src/dqx/extensions/pyarrow_ds.py`

#### Task 7: Remove batch-related tests from test_analyzer.py

**File**: `tests/test_analyzer.py`

**Changes**:
1. Remove the `FakeBatchSqlDataSource` class
2. Remove the entire `TestBatchAnalysis` class
3. Remove the `TestThreadingDetails` class
4. In `test_analyze_unsupported_data_source`, update it to test a different unsupported type (e.g., a plain dict)

**Testing**: Run `uv run pytest tests/test_analyzer.py -v` to ensure remaining tests pass

#### Task 7.5: Add Batch Rejection Test

**File**: `tests/test_analyzer.py`

**Add this test**:
```python
def test_analyze_rejects_batch_datasource():
    """Test that analyzer properly rejects batch data sources."""
    analyzer = Analyzer()

    # Create a mock that would have implemented BatchSqlDataSource
    class MockBatchDS:
        name = "mock_batch"
        def batches(self):
            return []

    with pytest.raises(DQXError, match="Unsupported data source"):
        analyzer.analyze(MockBatchDS(), [], ResultKey())
```

**Testing**: Run `uv run pytest tests/test_analyzer.py::test_analyze_rejects_batch_datasource -v`

#### Task 8: Remove ArrowBatchDataSource tests

**File**: `tests/extensions/test_pyarrow_ds.py`

**Changes**:
1. Remove the import of `ArrowBatchDataSource`
2. Remove these test functions:
   - `test_arrow_batch_datasource_init`
   - `test_arrow_batch_datasource_arrow_ds`
   - `test_arrow_batch_datasource_from_parquets`
   - `test_arrow_batch_datasource_with_record_batches`

**Testing**: Run `uv run pytest tests/extensions/test_pyarrow_ds.py -v`

**Phase 3 Commit**: `git commit -m "refactor: remove batch implementations and update tests"`

### Phase 4: Documentation & Examples

#### Task 9: Update design.md documentation

**File**: `docs/design.md`

**Specific changes**:
1. **Scalability section**: Change "TB-scale data through batch processing" to "Efficient single-pass analysis with DuckDB's query engine"
2. **Remove entire "Parallel Execution" section** if it exists
3. **Update code examples**: Remove any that show `threading=True` parameter
4. **Remove example**: The one showing `ArrowBatchDataSource.from_parquets`

**Key sections to update**:
- Find all instances of "batch", "threading", "parallel", "TB-scale"
- Update to emphasize single-pass efficiency and memory considerations

#### Task 10: Update other documentation files

**Search for files**: Use `grep -r "batch\|BatchSqlDataSource\|threading" docs/` to find other references

**Files to check and update**:
- README.md (if it mentions batch processing)
- Any guide documents that reference threading or batch capabilities
- Remove references to `ArrowBatchDataSource` in all docs

#### Task 11.5: Update Examples

**Directory**: `examples/`

**Check for**:
- Any use of `ArrowBatchDataSource`
- Any use of `threading=True` parameter
- Any references to batch processing

**Note**: Initial search found no examples using batch processing, but verify manually

#### Task 11: Final code cleanup

1. Search for any remaining references:
   ```bash
   grep -r "BatchSqlDataSource" src/
   grep -r "ArrowBatchDataSource" src/
   grep -r "threading" src/
   grep -r "_analyze_batches" src/
   ```

2. Remove any found references
3. Check for any orphaned imports

**Phase 4 Commit**: `git commit -m "docs: remove batch processing references"`

### Phase 5: Verification & Validation

#### Task 12: Run pre-commit hooks

**Commands**:
```bash
cd /Users/npham/git-tree/dqx
bin/run-hooks.sh
```

**Expected issues**:
- Ruff may find unused imports
- Mypy may find type errors
- Code formatting issues

**Fix any issues found and amend the previous commit**

#### Task 13: Run full test suite

**Commands**:
```bash
cd /Users/npham/git-tree/dqx
uv run pytest -v
```

**What to verify**:
- All tests pass
- Coverage remains at 100%
- No deprecation warnings about removed features

**If tests fail**:
1. Fix the failing tests
2. Run pre-commit hooks again
3. Amend the commit

#### Task 14: Final verification

1. **Verify no batch references remain**:
   ```bash
   grep -r "BatchSqlDataSource\|ArrowBatchDataSource\|_analyze_batches" src/ tests/ docs/
   ```

2. **Check the simplified API**:
   - Analyzer only accepts SqlDataSource
   - No threading parameter in method signature
   - No batch processing methods

4. **Run final pre-commit and tests**:
   ```bash
   bin/run-hooks.sh
   uv run pytest
   ```

**Phase 5 Commit**: `git commit -m "style: fix linting and finalize batch removal"`

## Testing the Changes

After implementation, test the simplified analyzer:

```python
# This should work
from dqx.analyzer import Analyzer
from dqx.extensions.pyarrow_ds import ArrowDataSource
import pyarrow as pa

data = pa.table({"col": [1, 2, 3]})
ds = ArrowDataSource(data)
analyzer = Analyzer()
result = analyzer.analyze(ds, metrics, key)  # No threading parameter

# This should NOT work (and shouldn't exist)
# from dqx.extensions.pyarrow_ds import ArrowBatchDataSource  # Should fail
```

## Rollback Plan

If issues arise at any phase:
1. `git log --oneline` to see the commits
2. `git reset --hard <commit-before-phase>` to rollback to a specific phase
3. Re-evaluate the approach for that phase
4. Continue with a modified approach

## Success Criteria

- [ ] All tests pass with 100% coverage
- [ ] No references to batch processing in code
- [ ] No references to threading in analyzer (except mutex for thread safety)
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] API is simplified and cleaner
- [ ] Threading parameter is completely removed from API

## Notes for the Developer

- **TDD Approach**: Run tests after each task to catch issues early
- **Commit Frequently**: Each phase has its own commit for easy rollback
- **Check Types**: Run mypy after each change to catch type errors
- **Be Thorough**: Use grep to find all references before declaring a phase complete
- **Thread Safety**: Remember that `_mutex` stays for thread-safe merge operations

Remember: The goal is to simplify the codebase while maintaining all existing single-source functionality. The threading parameter is completely removed, and thread safety is maintained for the merge operation.
