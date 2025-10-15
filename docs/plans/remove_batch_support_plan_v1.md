# Remove Batch Support Implementation Plan v1

## Overview
This plan outlines the complete removal of batch processing support from DQX, focusing on simplifying the analyzer to support only single-pass SQL data sources. This includes removing `BatchSqlDataSource` protocol, `ArrowBatchDataSource` implementation, and all threading/batch processing infrastructure.

## Background
- **Goal**: Simplify the analyzer for the first release by removing batch processing complexity
- **Rationale**: Focus on single-pass analyzers to improve performance and reduce complexity
- **Branch**: `feature/remove-batch-support`

## Pre-requisites
- Ensure you're on the `feature/remove-batch-support` branch
- Run `uv run pytest` to verify all tests pass before starting
- Understand the current architecture:
  - `BatchSqlDataSource` is a Protocol in `src/dqx/common.py`
  - `ArrowBatchDataSource` is an implementation in `src/dqx/extensions/pyarrow_ds.py`
  - Analyzer has batch processing methods in `src/dqx/analyzer.py`

## Implementation Tasks

### Task 1: Remove BatchSqlDataSource Protocol from common.py

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

**Commit**: `git commit -m "refactor: remove BatchSqlDataSource protocol from common.py"`

### Task 2: Update Analyzer Protocol in common.py

**File**: `src/dqx/common.py`

**Changes**:
1. Find the `Analyzer` Protocol (around line ~260)
2. Update the `analyze()` method signature:
   - Change `ds: SqlDataSource | BatchSqlDataSource` to `ds: SqlDataSource`
   - Remove the `threading: bool = False` parameter
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

**Commit**: `git commit -m "refactor: simplify Analyzer protocol to single-source only"`

### Task 3: Remove batch processing imports from analyzer.py

**File**: `src/dqx/analyzer.py`

**Changes**:
1. Remove these imports:
   - `import multiprocessing`
   - `from concurrent.futures import ThreadPoolExecutor`
   - `from threading import Lock`
2. Remove `BatchSqlDataSource` from the imports from `dqx.common`

**Testing**: Run `uv run ruff check src/dqx/analyzer.py` to check for unused imports

**Commit**: `git commit -m "refactor: remove batch processing imports from analyzer"`

### Task 4: Remove batch processing methods from analyzer.py

**File**: `src/dqx/analyzer.py`

**Changes**:
1. Remove the `_analyze_batches()` method (entire method)
2. Remove the `_analyze_batches_threaded()` method (entire method)
3. Remove the `_mutex` attribute from `__init__` if it's only used for threading
4. Check if `_mutex` is still needed for the merge operation in `analyze_single`

**Testing**: Run `uv run mypy src/dqx/analyzer.py` to ensure no type errors

**Commit**: `git commit -m "refactor: remove batch processing methods from analyzer"`

### Task 5: Simplify analyze() method in analyzer.py

**File**: `src/dqx/analyzer.py`

**Changes**:
1. Update the `analyze()` method signature to remove `threading` parameter
2. Remove all the conditional logic for batch processing
3. Make it simply call `analyze_single()` directly

**What to change**:
```python
# The new analyze method should look like:
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

**Commit**: `git commit -m "refactor: simplify analyze() to single-source only"`

### Task 6: Remove ArrowBatchDataSource from pyarrow_ds.py

**File**: `src/dqx/extensions/pyarrow_ds.py`

**Changes**:
1. Remove the entire `ArrowBatchDataSource` class
2. Remove `ArrowBatchDataSource` from module exports
3. Update the module docstring to remove references to batch processing
4. Remove the `MAX_ARROW_BATCH_SIZE` constant if it's only used by `ArrowBatchDataSource`

**What to remove**:
- The entire class from `class ArrowBatchDataSource:` to the end of the file
- Any imports that were only used by this class (like `dataset` from pyarrow if not used elsewhere)

**Testing**: Run `uv run mypy src/dqx/extensions/pyarrow_ds.py`

**Commit**: `git commit -m "refactor: remove ArrowBatchDataSource from extensions"`

### Task 7: Remove batch-related tests from test_analyzer.py

**File**: `tests/test_analyzer.py`

**Changes**:
1. Remove the `FakeBatchSqlDataSource` class
2. Remove the entire `TestBatchAnalysis` class
3. Remove the `TestThreadingDetails` class
4. In `test_analyze_unsupported_data_source`, update it to test a different unsupported type
5. Update any remaining references to threading or batch processing

**Testing**: Run `uv run pytest tests/test_analyzer.py -v` to ensure remaining tests pass

**Commit**: `git commit -m "test: remove batch-related tests from analyzer"`

### Task 8: Remove ArrowBatchDataSource tests

**File**: `tests/extensions/test_pyarrow_ds.py`

**Changes**:
1. Remove the import of `ArrowBatchDataSource`
2. Remove these test functions:
   - `test_arrow_batch_datasource_init`
   - `test_arrow_batch_datasource_arrow_ds`
   - `test_arrow_batch_datasource_from_parquets`
   - `test_arrow_batch_datasource_with_record_batches`

**Testing**: Run `uv run pytest tests/extensions/test_pyarrow_ds.py -v`

**Commit**: `git commit -m "test: remove ArrowBatchDataSource tests"`

### Task 9: Update design.md documentation

**File**: `docs/design.md`

**Changes**:
1. Remove references to:
   - Batch processing
   - TB-scale data handling
   - Multi-threading
   - `ArrowBatchDataSource`
   - `threading=True` parameter
2. Update the scalability section to focus on single-pass efficiency
3. Remove the example showing `ArrowBatchDataSource.from_parquets`

**Key sections to update**:
- Find and update the "Scalability" bullet point
- Remove the "Parallel Execution" section
- Update any code examples that use `threading=True`

**Commit**: `git commit -m "docs: remove batch processing references from design.md"`

### Task 10: Update other documentation files

**Search for files**: Use `grep -r "batch\|BatchSqlDataSource\|threading" docs/` to find other references

**Files to check and update**:
- Any plan documents that mention batch processing capabilities
- Any examples showing `threading=True`
- References to `ArrowBatchDataSource`

**Commit**: `git commit -m "docs: remove remaining batch processing references"`

### Task 11: Final code cleanup

1. Search for any remaining references:
   ```bash
   grep -r "BatchSqlDataSource" src/
   grep -r "ArrowBatchDataSource" src/
   grep -r "threading" src/
   grep -r "_analyze_batches" src/
   ```

2. Remove any found references

**Commit**: `git commit -m "refactor: final cleanup of batch processing references"`

### Task 12: Run pre-commit hooks

**Commands**:
```bash
cd /Users/npham/git-tree/dqx
bin/run-hooks.sh
```

**Expected issues**:
- Ruff may find unused imports
- Mypy may find type errors
- Code formatting issues

**Fix any issues found and commit**:
```bash
git add -u
git commit -m "style: fix linting and type checking issues"
```

### Task 13: Run full test suite

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
3. Commit the fixes

### Task 14: Final verification

1. **Verify no batch references remain**:
   ```bash
   grep -r "BatchSqlDataSource\|ArrowBatchDataSource\|_analyze_batches\|threading" src/ tests/ docs/
   ```

2. **Check the simplified API**:
   - Analyzer only accepts SqlDataSource
   - No threading parameter anywhere
   - No batch processing methods

3. **Update README.md** if it mentions batch processing

4. **Run final pre-commit and tests**:
   ```bash
   bin/run-hooks.sh
   uv run pytest
   ```

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
# analyzer.analyze(ds, metrics, key, threading=True)  # Should fail
```

## Rollback Plan

If issues arise:
1. `git checkout main`
2. `git branch -D feature/remove-batch-support`
3. Start over with a more gradual approach

## Success Criteria

- [ ] All tests pass with 100% coverage
- [ ] No references to batch processing in code
- [ ] No references to threading in analyzer
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] API is simplified and cleaner

## Notes for the Developer

- **TDD Approach**: Run tests after each task to catch issues early
- **Commit Frequently**: Each task should have its own commit
- **Check Types**: Run mypy after each change to catch type errors
- **Be Thorough**: Use grep to find all references before declaring a task complete
- **Ask Questions**: If unsure about a change, investigate the impact first

Remember: The goal is to simplify the codebase while maintaining all existing single-source functionality. Be careful not to break anything that depends on the single-source path.
