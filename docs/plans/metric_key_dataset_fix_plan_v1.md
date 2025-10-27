# MetricKey Dataset Fix Implementation Plan

## Problem Statement

The current `MetricKey` type alias is defined as `tuple[MetricSpec, ResultKey]`, which causes collisions when the same metric is used across multiple datasets on the same date. This leads to incorrect symbol assignments in cross-dataset verification suites, where symbols from one dataset incorrectly override values from another dataset.

## Root Cause

1. **MetricKey Collision**: When analyzing multiple datasets with the same metrics on the same date, the `(MetricSpec, ResultKey)` tuple is identical, causing the second dataset's values to overwrite the first in the AnalysisReport dictionary.

2. **Database Query Issue**: The MetricDB interface methods (`get_metric_value`, `get_metric_window`, etc.) don't include dataset in their WHERE clauses, potentially returning arbitrary results when multiple datasets have the same metric.

## Solution Overview

1. Add a `DatasetName` type alias for semantic clarity
2. Update `MetricKey` to include dataset: `tuple[MetricSpec, ResultKey, DatasetName]`
3. Update all MetricKey creation and usage sites
4. Fix MetricDB interface to include dataset parameter
5. Add comprehensive tests to prevent regression

## Implementation Tasks

### Task Group 1: Type Definitions and Common Updates

**Task 1.1: Add DatasetName type alias**
- File: `src/dqx/common.py`
- Add type alias: `DatasetName = str`
- Import this in files that need it

**Task 1.2: Update MetricKey type definition**
- File: `src/dqx/analyzer.py`
- Change from: `MetricKey = tuple[MetricSpec, ResultKey]`
- To: `MetricKey = tuple[MetricSpec, ResultKey, DatasetName]`
- Update import to include DatasetName

**Task 1.3: Write tests for new type definitions**
- File: `tests/test_analyzer.py`
- Add test case `test_report_with_dataset_in_key` in `TestAnalysisReport`
- Verify that MetricKey with dataset prevents collisions

### Task Group 2: Analyzer Updates

**Task 2.1: Update metric_key creation in analyzer**
- File: `src/dqx/analyzer.py` (~line 418)
- Change: `metric_key = (metric, key)`
- To: `metric_key = (metric, key, ds.name)`

**Task 2.2: Update Analyzer constructor**
- File: `src/dqx/analyzer.py` (~line 266)
- Update symbol_lookup parameter type:
  ```python
  symbol_lookup: dict[tuple[MetricSpec, ResultKey, DatasetName], str] | None = None
  ```

**Task 2.3: Add cross-dataset analyzer test**
- File: `tests/test_analyzer.py`
- Add test `test_analyzer_multiple_datasets_same_metrics` in `TestAnalyzer`
- Verify no collision when same metrics used across datasets

### Task Group 3: API and Symbol Lookup Updates

**Task 3.1: Update symbol lookup in api.py**
- File: `src/dqx/api.py` (~line 656)
- Update type annotation: `symbol_lookup: dict[tuple[MetricSpec, ResultKey, DatasetName], str] = {}`
- Update assignment: `symbol_lookup[(sym_metric.metric_spec, effective_key, ds.name)] = str(sym_metric.symbol)`

**Task 3.2: Add symbol mapping test**
- File: `tests/test_analyzer.py`
- Add test `test_analyzer_symbol_mapping_with_datasets`
- Verify symbol mapping works with dataset in MetricKey

**Task 3.3: Run tests and fix any issues**
- Run: `uv run pytest tests/test_analyzer.py -v`
- Fix any failing tests

### Task Group 4: Data Display Updates

**Task 4.1: Update MetricKey unpacking in data.py**
- File: `src/dqx/data.py`
- Find all occurrences of `for (metric_spec, result_key), metric in`
- Update to: `for (metric_spec, result_key, dataset), metric in`
- Handle the unpacked dataset variable appropriately

**Task 4.2: Write display tests**
- File: `tests/test_data.py` (or create if doesn't exist)
- Add test for display functions handling 3-tuple MetricKey
- Verify display works correctly with new key format

**Task 4.3: Run tests and verify display**
- Run: `uv run pytest tests/test_data*.py -v`
- Fix any display-related issues

### Task Group 5: Database Interface Updates

**Task 5.1: Update MetricDB method signatures**
- File: `src/dqx/orm/repositories.py`
- Update methods to include dataset parameter:
  ```python
  def get(self, key: ResultKey, spec: MetricSpec, dataset: DatasetName) -> Maybe[models.Metric]:
  def get_metric_value(self, metric: MetricSpec, key: ResultKey, dataset: DatasetName) -> Maybe[float]:
  def get_metric_window(self, metric: MetricSpec, key: ResultKey, dataset: DatasetName, lag: int, window: int) -> Maybe[TimeSeries]:
  ```

**Task 5.2: Update database queries**
- File: `src/dqx/orm/repositories.py`
- Add `Metric.dataset == dataset` to WHERE clauses in:
  - `_get_by_key` method
  - `get_metric_value` method
  - `get_metric_window` method

**Task 5.3: Update compute.py to pass dataset**
- File: `src/dqx/compute.py`
- Update all MetricDB method calls to include dataset parameter
- This will require determining where to get the dataset from

### Task Group 6: Integration Testing and Verification

**Task 6.1: Add integration test for cross-dataset verification**
- File: `tests/test_api_e2e.py` or similar
- Create a test similar to the failing e2e test
- Verify the fix resolves the symbol collision issue

**Task 6.2: Update existing tests**
- Search for tests that create MetricKey tuples directly
- Update them to use 3-tuple format
- Ensure backward compatibility where needed

**Task 6.3: Run full test suite**
- Run: `uv run pytest tests/ -v`
- Run: `bin/run-hooks.sh`
- Ensure all tests pass and no linting issues

### Task Group 7: Final Verification

**Task 7.1: Run the original failing test**
- Run: `uv run pytest tests/e2e/test_api_e2e.py::test_cross_dataset_verification_suite -xvs`
- Verify it now passes

**Task 7.2: Clean up debug files**
- Remove any temporary debug files created during investigation
- Clean up any debug print statements

**Task 7.3: Final checks**
- Run: `uv run mypy src/`
- Run: `uv run ruff check --fix`
- Run: `uv run pytest tests/ -v --cov=dqx`
- Verify coverage hasn't decreased

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
   - MetricKey creation with dataset
   - Database queries with dataset filter
   - Symbol lookup with new key format

2. **Integration Tests**: Test component interactions
   - Cross-dataset analysis without collisions
   - Symbol assignment across datasets
   - Database persistence and retrieval

3. **Regression Tests**: Ensure existing functionality works
   - Single dataset analysis
   - Backward compatibility where applicable

## Rollback Plan

If issues arise:
1. Revert MetricKey type definition
2. Remove dataset from tuple creation sites
3. Revert database interface changes

## Success Criteria

1. The failing e2e test passes
2. No existing tests are broken
3. Code coverage remains at or above current level
4. All linting and type checking passes
5. Cross-dataset metrics have unique symbols and correct values
