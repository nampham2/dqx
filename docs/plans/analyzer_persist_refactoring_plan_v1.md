# Analyzer Persist Refactoring Plan v1

## Overview
This plan outlines the steps to refactor the persist functionality from the `Analyzer` class to the `AnalysisReport` class. This refactoring improves separation of concerns by making AnalysisReport responsible for its own persistence.

## Background
- **Current State**: The `Analyzer` class has `persist()` and `_merge_persist()` methods that save AnalysisReport data to MetricDB
- **Desired State**: Move these methods to `AnalysisReport` where the data naturally belongs
- **No Backward Compatibility**: We can remove methods from Analyzer entirely
- **Thread Safety**: Not a concern - callers will handle thread safety if needed

## Prerequisites
- Working DQX development environment with `uv` installed
- All tests passing before starting (`uv run pytest`)
- Clean git working directory on branch `refactor/move-persist-to-report`

## Implementation Plan

### Phase 1: Move Methods to AnalysisReport (Tasks 1-3)

#### Task 1: Copy persist methods to AnalysisReport
**File**: `src/dqx/analyzer.py`

1. Open `src/dqx/analyzer.py`
2. Locate the `AnalysisReport` class (around line 33)
3. Add the following methods to `AnalysisReport` class:

```python
def persist(self, db: MetricDB, overwrite: bool = True) -> None:
    """Persist the analysis report to the metric database.

    Args:
        db: MetricDB instance for persistence
        overwrite: If True, overwrite existing metrics. If False, merge with existing.
    """
    if len(self) == 0:
        logger.warning("Try to save an EMPTY analysis report!")
        return

    if overwrite:
        logger.info("Overwriting analysis report ...")
        db.persist(self.values())
    else:
        logger.info("Merging analysis report ...")
        self._merge_persist(db)

def _merge_persist(self, db: MetricDB) -> None:
    """Merge with existing metrics in the database before persisting.

    Args:
        db: MetricDB instance for persistence
    """
    db_report = AnalysisReport()

    for key, metric in self.items():
        # Find the metric in DB
        db_metric = db.get(metric.key, metric.spec)
        if db_metric is not None:
            db_report[key] = db_metric.unwrap()

    # Merge and persist
    merged_report = self.merge(db_report)
    db.persist(merged_report.values())
```

4. Run tests to ensure methods work in new location:
   ```bash
   uv run pytest tests/test_analyzer.py -xvs
   ```

#### Task 2: Update imports in AnalysisReport if needed
**File**: `src/dqx/analyzer.py`

1. Ensure `logger` is available at module level (it should already be imported)
2. Ensure `MetricDB` type is imported (check imports at top of file)
3. No additional imports should be needed as they're already present

#### Task 3: Test the new methods work correctly
1. Create a simple test script to verify the methods work:
   ```bash
   uv run python -c "from dqx.analyzer import AnalysisReport; from dqx.orm.repositories import InMemoryMetricDB; print('Import successful - methods are accessible')"
   ```

2. If successful, commit this phase:
   ```bash
   git add src/dqx/analyzer.py
   git commit -m "feat: Add persist methods to AnalysisReport"
   ```

### Phase 2: Update API Usage (Tasks 4-5)

#### Task 4: Update VerificationSuite to use new location
**File**: `src/dqx/api.py`

1. Open `src/dqx/api.py`
2. Find the line in `VerificationSuite.run()` method (around line 410):
   ```python
   analyzer.persist(self.provider._db)
   ```
3. Change it to:
   ```python
   analyzer.report.persist(self.provider._db)
   ```

4. Test the change:
   ```bash
   uv run pytest tests/test_api.py -xvs -k "test_verification"
   ```

#### Task 5: Run integration tests
1. Run broader test suite to ensure nothing breaks:
   ```bash
   uv run pytest tests/test_api.py tests/test_analyzer.py -xvs
   ```

2. If all tests pass, commit:
   ```bash
   git add src/dqx/api.py
   git commit -m "feat: Update API to use AnalysisReport.persist()"
   ```

### Phase 3: Clean Up Analyzer (Tasks 6-8)

#### Task 6: Remove persist methods from Analyzer
**File**: `src/dqx/analyzer.py`

1. Open `src/dqx/analyzer.py`
2. Delete the `persist()` method (around line 217)
3. Delete the `_merge_persist()` method (around line 203)
4. Delete the TODO comment: `# TODO(npham): Move persist to the analysis report`

#### Task 7: Remove mutex lock from Analyzer
**File**: `src/dqx/analyzer.py`

1. Remove the mutex import:
   ```python
   from threading import Lock
   ```
2. Remove mutex initialization in `__init__`:
   ```python
   self._mutex = Lock()
   ```
3. Remove all uses of `self._mutex` (in the `analyze` method around line 194-195)

4. Test that Analyzer still works:
   ```bash
   uv run pytest tests/test_analyzer.py -xvs -k "test_analyze"
   ```

#### Task 8: Update analyzer tests
**File**: `tests/test_analyzer.py`

1. Open `tests/test_analyzer.py`
2. Find all occurrences of `analyzer.persist(` and replace with `analyzer.report.persist(`
3. Look for test methods that might be testing the old persist behavior:
   - `test_persist_empty_report`
   - `test_persist_with_merge`
   - Any mocks of `analyzer.persist`

4. Update each test to use the new API:
   ```python
   # Old
   analyzer.persist(db)

   # New
   analyzer.report.persist(db)
   ```

5. Run all analyzer tests:
   ```bash
   uv run pytest tests/test_analyzer.py -xvs
   ```

### Phase 4: Final Validation (Tasks 9-10)

#### Task 9: Run full test suite
1. Run pre-commit hooks:
   ```bash
   bin/run-hooks.sh
   ```

2. Run full test suite:
   ```bash
   uv run pytest -xvs
   ```

3. Check test coverage hasn't decreased:
   ```bash
   uv run pytest --cov=dqx.analyzer --cov-report=term-missing
   ```

#### Task 10: Final commit and cleanup
1. If all tests pass and pre-commit is clean:
   ```bash
   git add -A
   git commit -m "refactor: Complete move of persist methods to AnalysisReport"
   ```

2. Update the memory bank with this change:
   - Add to `memory-bank/activeContext.md` under "Recently Completed Work"
   - Note that persist functionality moved from Analyzer to AnalysisReport

## Testing Strategy

### Unit Tests to Update
- `tests/test_analyzer.py`: Update all calls to `analyzer.persist()`
- Look for any mocked persist methods that need updating

### Integration Tests to Verify
- `tests/test_api.py`: Ensure VerificationSuite still works
- `tests/e2e/test_api_e2e.py`: DO NOT MODIFY (critical ground truth tests)

### Manual Testing Commands
```bash
# Quick smoke test
uv run python examples/display_demo.py

# Run specific test
uv run pytest tests/test_analyzer.py::test_persist_empty_report -xvs
```

## Rollback Plan
If issues arise:
```bash
git checkout main
git branch -D refactor/move-persist-to-report
```

## Success Criteria
- [ ] All tests pass
- [ ] No linting errors
- [ ] Coverage maintained at 98%+
- [ ] AnalysisReport can persist itself
- [ ] Analyzer no longer has persist methods
- [ ] API updated to use new location
