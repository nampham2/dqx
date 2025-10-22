# DataSource Refactoring Implementation Plan v1

## Overview

This plan details the migration of `DuckRelationDataSource` from `dqx.extensions.duckds` to `dqx.datasource` and the complete removal of the extensions module. This is a breaking change with no backward compatibility requirements.

## Motivation

1. **Simplify Structure**: The extensions module contains only one class, making it an unnecessary abstraction layer
2. **Better Organization**: Data sources are core components and should be at the top level of the module hierarchy
3. **KISS Principle**: Removing the extensions module reduces complexity without losing functionality

## Approach

Following Test-Driven Development (TDD) principles with committable task groups:
1. Each task group completes a full Red-Green-Refactor cycle
2. No failing tests are committed - each group ends with all tests passing
3. Each commit represents a working state of the codebase
4. Systematic migration minimizes risk

## Implementation Plan

### Task Group 1: Core Migration with TDD

**Objective**: Migrate DuckRelationDataSource to new location following TDD within a single committable group

#### Task 1.1: Update primary test file import (Red Phase)
**File**: `tests/extensions/test_duck_ds.py`

Change:
```python
from dqx.extensions.duckds import DuckRelationDataSource
```

To:
```python
from dqx.datasource import DuckRelationDataSource
```

**Verification**: Run `uv run pytest tests/extensions/test_duck_ds.py -v`
- Expected: ImportError - cannot import name 'DuckRelationDataSource' from 'dqx.datasource'
- This confirms our test is properly checking the import (Red phase)

#### Task 1.2: Create datasource.py module (Green Phase)
**File**: `src/dqx/datasource.py`

Create new file with complete content from `src/dqx/extensions/duckds.py`, updating only the module docstring and import references in docstrings:


```python
"""Data source implementations for DQX framework.

This module provides data source adapters that implement the SqlDataSource protocol,
enabling various data formats to be analyzed within the DQX data quality framework.

The primary implementation is DuckRelationDataSource, which wraps DuckDB relations
and provides the necessary interface for the DQX analyzer to execute SQL queries.
"""

from __future__ import annotations

import datetime
from typing import Self

import duckdb
import pyarrow as pa

from dqx.utils import random_prefix


class DuckRelationDataSource:
    """Adapter for DuckDB relations to work as DQX data sources.

    This class wraps a DuckDB relation and implements the SqlDataSource protocol,
    enabling it to be used with the DQX analyzer. It provides a temporary table
    name for SQL queries and methods to execute queries against the wrapped relation.

    The adapter is particularly useful when you have complex DuckDB queries or
    transformations that you want to analyze for data quality metrics without
    materializing the results to disk.

    Attributes:
        name: Identifier for this data source type, always "duckdb"
        dialect: SQL dialect used for query generation, always "duckdb"

    Example:
        >>> # From an existing DuckDB query
        >>> relation = duckdb.sql("SELECT * FROM sales WHERE year = 2023")
        >>> ds = DuckRelationDataSource(relation)
        >>>
        >>> # Use with analyzer
        >>> report = analyzer.analyze_single(ds, metrics, key)
    """

    name: str = "duckdb"
    dialect: str = "duckdb"

    def __init__(self, relation: duckdb.DuckDBPyRelation) -> None:
        """Initialize the DuckDB relation data source.

        Creates a wrapper around a DuckDB relation with a randomly generated
        internal table name for use in SQL queries. The table name is prefixed
        with an underscore and followed by 6 random characters to avoid collisions.

        Args:
            relation: A DuckDB relation object to wrap. This can be the result
                     of any DuckDB query or transformation.

        Example:
            >>> conn = duckdb.connect()
            >>> rel = conn.sql("SELECT * FROM 'data.csv'")
            >>> ds = DuckRelationDataSource(rel)
        """
        self._relation = relation
        self._table_name = random_prefix(k=6)

    def cte(self, nominal_date: datetime.date) -> str:
        """Get the CTE for this data source.

        Args:
            nominal_date: The date for filtering (currently ignored)

        Returns:
            The CTE SQL string
        """
        return f"SELECT * FROM {self._table_name}"

    def query(self, query: str, nominal_date: datetime.date) -> duckdb.DuckDBPyRelation:
        """Execute a query against the DuckDB relation.

        Args:
            query: The SQL query to execute
            nominal_date: The date for filtering (currently ignored)

        Returns:
            Query results as a DuckDB relation
        """
        return self._relation.query(self._table_name, query)

    @classmethod
    def from_arrow(cls, table: pa.RecordBatch | pa.Table) -> Self:
        """Create a DuckRelationDataSource from PyArrow data structures.

        This factory method provides a convenient way to create a DuckDB data source
        directly from PyArrow Tables or RecordBatches. It leverages DuckDB's native
        Arrow integration to create a relation from the Arrow data, then wraps it
        in a DuckRelationDataSource for use with the DQX analyzer.

        This method is particularly useful when you have data in Arrow format
        (e.g., from Parquet files, Arrow IPC, or computational results) and want
        to analyze it using DQX without intermediate conversions.

        Args:
            table: A PyArrow Table or RecordBatch containing the data to analyze.
                   Both types are supported and will be converted to a DuckDB
                   relation automatically.

        Returns:
            A new DuckRelationDataSource instance wrapping the Arrow data.

        Example:
            >>> import pyarrow as pa
            >>> from dqx.datasource import DuckRelationDataSource
            >>>
            >>> # From a PyArrow Table
            >>> arrow_table = pa.table({
            ...     'id': [1, 2, 3, 4],
            ...     'value': [10.5, 20.3, 30.1, 40.7]
            ... })
            >>> ds = DuckRelationDataSource.from_arrow(arrow_table)
            >>>
            >>> # From a RecordBatch
            >>> batch = pa.record_batch([
            ...     pa.array([1, 2, 3]),
            ...     pa.array(['a', 'b', 'c'])
            ... ], names=['id', 'category'])
            >>> ds = DuckRelationDataSource.from_arrow(batch)
            >>>
            >>> # Use with analyzer
            >>> analyzer = Analyzer()
            >>> metrics = [MetricSpec.num_rows(), MetricSpec.cardinality('category')]
            >>> report = analyzer.analyze_single(ds, metrics, key)
        """
        relation: duckdb.DuckDBPyRelation = duckdb.arrow(table)
        return cls(relation)
```

#### Task 1.3: Verify test now passes
```bash
uv run pytest tests/extensions/test_duck_ds.py -v
```
- Expected: All tests should pass (Green phase achieved)

#### Task 1.4: Run all quality checks
```bash
# Type checking
uv run mypy src/dqx/datasource.py tests/extensions/test_duck_ds.py

# Linting
uv run ruff check src/dqx/datasource.py tests/extensions/test_duck_ds.py

# Ensure no regression in other tests that still use old import
uv run pytest tests/test_analyzer.py -v
```
- Expected: mypy and ruff pass, analyzer tests still pass with old import

#### Task 1.5: Commit the working migration
```bash
git add src/dqx/datasource.py tests/extensions/test_duck_ds.py
git commit -m "feat: migrate DuckRelationDataSource to datasource module with TDD"
```

**Important**: At this point we have:
- A working test that imports from the new location
- The new module with DuckRelationDataSource
- All tests passing (both updated and non-updated)
- Clean linting and type checking

### Task Group 2: Update Analyzer and Related Tests

**Objective**: Update analyzer and related test imports as a cohesive group

#### Task 2.1: Update analyzer and coverage tests
#### Task 2.1: Update analyzer and coverage tests
**Files**:
- `tests/test_analyzer.py`
- `tests/test_analyzer_coverage.py`

Change all imports from:
```python
from dqx.extensions.duckds import DuckRelationDataSource
```

To:
```python
from dqx.datasource import DuckRelationDataSource
```

#### Task 2.2: Verify tests pass
```bash
uv run pytest tests/test_analyzer.py tests/test_analyzer_coverage.py -v
```

#### Task 2.3: Run quality checks
```bash
uv run mypy tests/test_analyzer.py tests/test_analyzer_coverage.py
uv run ruff check tests/test_analyzer.py tests/test_analyzer_coverage.py
```

#### Task 2.4: Commit analyzer test updates
```bash
git add tests/test_analyzer.py tests/test_analyzer_coverage.py
git commit -m "test: update analyzer test imports to use datasource module"
```

### Task Group 3: Update Validation and Integration Tests

**Objective**: Update validation-related test imports

#### Task 3.1: Update evaluator and lag tests
**Files**:
- `tests/test_evaluator_validation.py`
- `tests/test_lag_date_handling.py`
- `tests/test_duplicate_count_integration.py`

Same import change as previous groups.

#### Task 3.2: Verify tests pass
```bash
uv run pytest tests/test_evaluator_validation.py tests/test_lag_date_handling.py tests/test_duplicate_count_integration.py -v
```

#### Task 3.3: Run quality checks and commit
```bash
uv run mypy tests/test_evaluator_validation.py tests/test_lag_date_handling.py tests/test_duplicate_count_integration.py
uv run ruff check tests/test_evaluator_validation.py tests/test_lag_date_handling.py tests/test_duplicate_count_integration.py

git add tests/test_evaluator_validation.py tests/test_lag_date_handling.py tests/test_duplicate_count_integration.py
git commit -m "test: update validation and integration test imports"
```

### Task Group 4: Update API Tests

**Objective**: Update all API-related test imports

#### Task 4.1: Update API tests
#### Task 4.1: Update API tests
**Files**:
- `tests/test_api.py`
- `tests/test_api_coverage.py`
- `tests/e2e/test_api_e2e.py`
- `tests/test_assertion_result_collection.py`

Same import change as above.

#### Task 4.2: Verify and commit
```bash
uv run pytest tests/test_api.py tests/test_api_coverage.py tests/e2e/test_api_e2e.py tests/test_assertion_result_collection.py -v
uv run mypy tests/test_api*.py tests/e2e/test_api_e2e.py tests/test_assertion_result_collection.py
uv run ruff check tests/test_api*.py tests/e2e/test_api_e2e.py tests/test_assertion_result_collection.py

git add tests/test_api.py tests/test_api_coverage.py tests/e2e/test_api_e2e.py tests/test_assertion_result_collection.py
git commit -m "test: update API test imports to use datasource module"
```

### Task Group 4.5: Comprehensive Reference Search

**Objective**: Ensure no imports were missed before moving test files

#### Task 4.5.1: Search for all remaining references
```bash
# Comprehensive search for any remaining references
grep -r "extensions\.duckds" src/ tests/ examples/ docs/ || echo "No references found"
grep -r "from dqx\.extensions" src/ tests/ examples/ docs/ || echo "No references found"

# Also check for string literals that might reference the module
grep -r "\"dqx\.extensions" src/ tests/ examples/ docs/ || echo "No string references found"
grep -r "'dqx\.extensions" src/ tests/ examples/ docs/ || echo "No string references found"
```

#### Task 4.5.2: Check __init__.py files
```bash
# Check if extensions __init__.py exists and what it contains
cat src/dqx/extensions/__init__.py 2>/dev/null || echo "No __init__.py in extensions"

# Check if main __init__.py imports from extensions
grep -n "extensions" src/dqx/__init__.py || echo "No extensions imports in main __init__.py"
```

### Task Group 5: Update Suite and Remaining Tests

**Objective**: Update remaining test imports and move test file

#### Task 5.1: Update suite tests
**Files**:
- `tests/test_suite_critical.py`
- `tests/test_suite_caching.py`
- `tests/test_symbol_ordering.py`
- `tests/test_extended_metric_symbol_info.py`
- `tests/test_dialect.py`

Same import change as above.

#### Task 5.2: Verify tests before moving
```bash
# Ensure all tests pass before moving files
uv run pytest tests/test_suite_critical.py tests/test_suite_caching.py tests/test_symbol_ordering.py tests/test_extended_metric_symbol_info.py tests/test_dialect.py -v
```

#### Task 5.3: Move test file
```bash
mv tests/extensions/test_duck_ds.py tests/test_datasource.py
```

Update the docstring in `tests/test_datasource.py`:
```python
"""Tests for data source implementations."""
```

#### Task 5.4: Check and remove extensions test directory
```bash
# Check if extensions test directory is empty
ls -la tests/extensions/

# If empty, remove it
rmdir tests/extensions/
```

#### Task 5.5: Verify and commit
```bash
uv run pytest tests/test_suite_critical.py tests/test_suite_caching.py tests/test_symbol_ordering.py tests/test_extended_metric_symbol_info.py tests/test_dialect.py tests/test_datasource.py -v

uv run mypy tests/test_suite_*.py tests/test_symbol_ordering.py tests/test_extended_metric_symbol_info.py tests/test_dialect.py tests/test_datasource.py
uv run ruff check tests/test_suite_*.py tests/test_symbol_ordering.py tests/test_extended_metric_symbol_info.py tests/test_dialect.py tests/test_datasource.py

git add tests/test_suite_critical.py tests/test_suite_caching.py tests/test_symbol_ordering.py tests/test_extended_metric_symbol_info.py tests/test_dialect.py tests/test_datasource.py
git rm -r tests/extensions/
git commit -m "test: complete test migration and relocate datasource tests"
```

### Task Group 6: Update Example Files

**Objective**: Update all example files to use the new import

#### Task 6.1: Update example imports
**Files**:
- `examples/suite_symbol_collection_demo.py`
- `examples/result_collection_demo.py`
- `examples/suite_cache_and_critical_demo.py`
- `examples/sql_formatting_demo.py`

Change all imports from:
```python
from dqx.extensions.duckds import DuckRelationDataSource
```

To:
```python
from dqx.datasource import DuckRelationDataSource
```

**Note**: Also check for any docstring references to the extensions module (the actual module name is `duckds`, not `duck_ds`).

#### Task 6.2: Verify examples still work
```bash
# Run a couple of examples to verify
uv run python examples/sql_formatting_demo.py
uv run python examples/result_collection_demo.py
```

#### Task 6.3: Commit example updates
```bash
git add examples/
git commit -m "docs: update example imports for datasource module"
```

### Task Group 7: Clean Up Old Structure and Check Documentation

**Objective**: Remove the now-empty extensions module and check for documentation updates

#### Task 7.1: Check extensions module before removal
```bash
# Check what's in the extensions directory
ls -la src/dqx/extensions/

# Check extensions __init__.py content
cat src/dqx/extensions/__init__.py 2>/dev/null || echo "No __init__.py"

# Verify it's safe to remove
find src/dqx/extensions/ -type f -name "*.py" | grep -v __pycache__
```

#### Task 7.2: Check documentation and configuration files
```bash
# Check README for extensions references
grep -n "extensions" README.md || echo "No extensions in README"

# Check documentation files
grep -r "extensions\.duckds" docs/ || echo "No extensions.duckds in docs"
grep -r "dqx\.extensions" docs/ || echo "No dqx.extensions in docs"

# Check pyproject.toml
grep -n "extensions" pyproject.toml || echo "No extensions in pyproject.toml"

# Check if datasource module needs to be added anywhere
grep -n "packages" pyproject.toml || echo "No packages config found"
```

#### Task 7.3: Remove extensions source directory
```bash
rm -rf src/dqx/extensions/
```

#### Task 7.4: Search for any remaining references
```bash
# Ensure no references remain
grep -r "extensions\.duckds" src/ tests/ examples/ || echo "No references found"
grep -r "from dqx\.extensions" src/ tests/ examples/ || echo "No references found"

# Check for dynamic imports or string references
grep -r "\"extensions\.duckds\"" src/ tests/ examples/ || echo "No string references"
grep -r "'extensions\.duckds'" src/ tests/ examples/ || echo "No string references"
grep -r "importlib.*extensions" src/ tests/ || echo "No dynamic imports found"
```

#### Task 7.5: Update documentation if needed
If any documentation references were found in Task 7.2, update them here.

#### Task 7.6: Final verification before commit
```bash
# Run all tests
uv run pytest tests/ -v

# Check coverage
uv run pytest tests/ -v --cov=dqx --cov-report=term-missing

# Run type checking
uv run mypy src/dqx tests/

# Run linting
uv run ruff check src/dqx tests/
```

#### Task 7.7: Commit cleanup
```bash
git add -A
git commit -m "refactor: remove extensions module after completing datasource migration"
```

### Task Group 8: Final Verification

**Objective**: Ensure everything works correctly and no references were missed

#### Task 8.1: Final comprehensive search
```bash
# One final search for any missed references
echo "=== Searching for 'extensions' string in all files ==="
grep -r "extensions" src/ tests/ examples/ docs/ --exclude-dir=__pycache__ | grep -v "# Task" | grep -v "git commit" || echo "No references found"

echo "=== Searching for string literals with extensions ==="
grep -r "\".*extensions.*\"" src/ tests/ examples/ --exclude-dir=__pycache__ || echo "No string literals found"
grep -r "'.*extensions.*'" src/ tests/ examples/ --exclude-dir=__pycache__ || echo "No string literals found"
```

#### Task 8.2: Run pre-commit hooks
```bash
bin/run-hooks.sh
```

#### Task 8.3: Run full test suite with coverage
```bash
uv run pytest tests/ -v --cov=dqx --cov-report=term-missing
```
- Expected: 100% coverage

#### Task 8.4: Verify examples run correctly
```bash
# Run all examples to ensure they work
for example in examples/*.py; do
    echo "Running $example..."
    uv run python "$example" || echo "Failed: $example"
done
```

#### Task 8.5: Final commit (if any fixes were needed)
If any issues were found and fixed during verification, commit them:
```bash
git add -A
git commit -m "fix: final adjustments after datasource migration"
```

## Success Criteria

1. All tests pass with 100% coverage
2. No type errors from mypy
3. No linting errors from ruff
4. All examples run successfully
5. No remaining references to the extensions module (including in documentation)
6. No dynamic imports or string references to extensions.duckds
7. Clean git history with meaningful commits
8. Documentation and configuration files updated if needed

## Risk Assessment

**Low Risk** due to:
- TDD approach ensuring tests catch issues early
- Committable task groups preventing broken commits
- Comprehensive verification steps including:
  - Searching for string literals and dynamic imports
  - Checking __init__.py files
  - Verifying documentation and configuration
  - Running all examples
- Simple rollback strategy

## Rollback Plan

If issues arise, the changes can be reverted by:
```bash
git checkout main
git branch -D refactor/move-datasource-remove-extensions
```

This will restore the original structure with the extensions module intact.
