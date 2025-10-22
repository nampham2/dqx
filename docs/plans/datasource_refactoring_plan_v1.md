# DataSource Refactoring Implementation Plan v1

## Overview

This plan details the migration of `DuckRelationDataSource` from `dqx.extensions.duckds` to `dqx.datasource` and the complete removal of the extensions module. This is a breaking change with no backward compatibility requirements.

## Motivation

1. **Simplify Structure**: The extensions module contains only one class, making it an unnecessary abstraction layer
2. **Better Organization**: Data sources are core components and should be at the top level of the module hierarchy
3. **KISS Principle**: Removing the extensions module reduces complexity without losing functionality

## Approach

Following Test-Driven Development (TDD) principles:
1. Update test imports first (Red phase)
2. Create new module with minimal changes (Green phase)
3. Systematically update all imports (Refactor phase)
4. Verify and clean up

## Implementation Plan

### Task Group 1: TDD Setup - Update Test Imports (Red Phase)

**Objective**: Update test imports to expect the new location, ensuring they fail first

#### Task 1.1: Update primary test file import
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

#### Task 1.2: Update one integration test
**File**: `tests/test_analyzer.py`

Change all occurrences of:
```python
from dqx.extensions.duckds import DuckRelationDataSource
```

To:
```python
from dqx.datasource import DuckRelationDataSource
```

**Verification**: Run `uv run pytest tests/test_analyzer.py -v`
- Expected: ImportError

#### Task 1.3: Commit the failing tests
```bash
git add tests/extensions/test_duck_ds.py tests/test_analyzer.py
git commit -m "test: update imports to expect DuckRelationDataSource in datasource module"
```

### Task Group 2: Create New Module (Green Phase)

**Objective**: Create the new datasource.py module to make tests pass

#### Task 2.1: Create datasource.py with DuckRelationDataSource
**File**: `src/dqx/datasource.py`

Create new file with complete content from `src/dqx/extensions/duckds.py`, updating only the module docstring:

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

#### Task 2.2: Verify tests now pass
```bash
uv run pytest tests/extensions/test_duck_ds.py tests/test_analyzer.py -v
```
- Expected: All tests should pass

#### Task 2.3: Run type checking
```bash
uv run mypy src/dqx/datasource.py
```
- Expected: No type errors

#### Task 2.4: Commit the new module
```bash
git add src/dqx/datasource.py
git commit -m "feat: create datasource module with DuckRelationDataSource"
```

### Task Group 3: Update Test Imports

**Objective**: Systematically update all test file imports

#### Task 3.1: Update remaining analyzer-related tests
**Files**:
- `tests/test_analyzer_coverage.py`
- `tests/test_evaluator_validation.py`
- `tests/test_lag_date_handling.py`

Change all imports from:
```python
from dqx.extensions.duckds import DuckRelationDataSource
```

To:
```python
from dqx.datasource import DuckRelationDataSource
```

**Verification**: Run each test file individually to ensure they pass

#### Task 3.2: Update API and integration tests
**Files**:
- `tests/test_api.py`
- `tests/test_api_coverage.py`
- `tests/e2e/test_api_e2e.py`
- `tests/test_duplicate_count_integration.py`
- `tests/test_assertion_result_collection.py`

Same import change as above.

**Verification**: `uv run pytest tests/test_api*.py tests/e2e/test_api_e2e.py -v`

#### Task 3.3: Update remaining tests
**Files**:
- `tests/test_suite_critical.py`
- `tests/test_suite_caching.py`
- `tests/test_symbol_ordering.py`
- `tests/test_extended_metric_symbol_info.py`
- `tests/test_dialect.py`

Same import change as above.

#### Task 3.4: Move and update test file
Move `tests/extensions/test_duck_ds.py` to `tests/test_datasource.py` and update its imports.

```bash
mv tests/extensions/test_duck_ds.py tests/test_datasource.py
```

Update import in the file and also update the docstring:
```python
"""Tests for data source implementations."""
```

#### Task 3.5: Commit test updates
```bash
git add tests/
git commit -m "test: update all test imports for datasource module"
```

### Task Group 4: Update Example Files

**Objective**: Update all example files to use the new import

#### Task 4.1: Update example imports
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

Also update any docstring references from `dqx.extensions.duck_ds` to `dqx.datasource`.

#### Task 4.2: Verify examples still work
Run one example to verify:
```bash
uv run python examples/sql_formatting_demo.py
```

#### Task 4.3: Commit example updates
```bash
git add examples/
git commit -m "docs: update example imports for datasource module"
```

### Task Group 5: Clean Up Old Structure

**Objective**: Remove the now-empty extensions module

#### Task 5.1: Remove extensions directory
```bash
rm -rf src/dqx/extensions/
rm -rf tests/extensions/
```

#### Task 5.2: Update source file docstrings
In `src/dqx/datasource.py`, update any remaining references in docstrings from:
- `from dqx.extensions.duck_ds import DuckRelationDataSource`

To:
- `from dqx.datasource import DuckRelationDataSource`

#### Task 5.3: Final verification
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

#### Task 5.4: Commit cleanup
```bash
git add -A
git commit -m "refactor: remove extensions module after datasource migration"
```

### Task Group 6: Final Verification

**Objective**: Ensure everything works correctly

#### Task 6.1: Run pre-commit hooks
```bash
bin/run-hooks.sh
```

#### Task 6.2: Run full test suite with coverage
```bash
uv run pytest tests/ -v --cov=dqx --cov-report=term-missing
```
- Expected: 100% coverage

#### Task 6.3: Search for any remaining references
```bash
grep -r "extensions.duckds" src/ tests/ examples/ docs/ || echo "No references found"
grep -r "extensions.duck_ds" src/ tests/ examples/ docs/ || echo "No references found"
```

## Success Criteria

1. All tests pass with 100% coverage
2. No type errors from mypy
3. No linting errors from ruff
4. All examples run successfully
5. No remaining references to the extensions module
6. Clean git history with meaningful commits

## Rollback Plan

If issues arise, the changes can be reverted by:
```bash
git checkout main
git branch -D refactor/move-datasource-remove-extensions
```

This will restore the original structure with the extensions module intact.
