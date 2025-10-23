# BigQuery Dialect Implementation Plan v1

## Overview

This plan outlines the implementation of BigQuery SQL dialect support for the DQX framework. The BigQuery dialect will enable DQX to generate SQL compatible with Google BigQuery, allowing users to analyze data quality metrics on BigQuery datasets.

## Background

DQX currently supports DuckDB dialect for SQL generation. The dialect system uses a protocol-based design that allows adding new SQL dialects without modifying core analyzer code. BigQuery requires different SQL syntax including:
- Backtick (`) column quoting instead of single quotes
- FLOAT64 type instead of DOUBLE
- COUNTIF instead of COUNT_IF
- VAR_SAMP for variance calculations
- STRUCT types instead of MAP for batch queries

## Requirements

1. Implement all SqlOp translations for BigQuery syntax
2. Maintain compatibility with existing analyzer infrastructure
3. Support batch query optimization using BigQuery STRUCT
4. Ensure all tests pass with 100% coverage
5. Follow TDD principles throughout implementation

## Implementation Task Groups

### Task Group 1: Core BigQuery Dialect Class and Basic Translations (TDD Foundation)

**Objective**: Create the BigQueryDialect class with core structure and implement basic SqlOp translations with comprehensive tests.

**Files to create/modify**:
1. `tests/test_bigquery_dialect.py` (create new)
2. `src/dqx/dialect.py` (modify)

**Step 1.1**: Write failing tests for BigQueryDialect class and basic ops

Create `tests/test_bigquery_dialect.py`:
```python
"""Test BigQuery dialect implementation."""

import pytest

from dqx.dialect import BigQueryDialect, get_dialect
from dqx.ops import Average, Maximum, Minimum, NumRows, Sum


class TestBigQueryDialect:
    """Test BigQuery dialect implementation."""

    def test_bigquery_dialect_name(self) -> None:
        """Test dialect name property."""
        dialect = BigQueryDialect()
        assert dialect.name == "bigquery"

    def test_translate_num_rows(self) -> None:
        """Test NumRows translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = NumRows()
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNT(*) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_average(self) -> None:
        """Test Average translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Average("revenue")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(AVG(revenue) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_sum(self) -> None:
        """Test Sum translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Sum("quantity")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(SUM(quantity) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_minimum(self) -> None:
        """Test Minimum translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Minimum("price")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(MIN(price) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_maximum(self) -> None:
        """Test Maximum translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Maximum("score")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(MAX(score) AS FLOAT64) AS `{op.sql_col}`"

    def test_build_cte_query(self) -> None:
        """Test CTE query building."""
        dialect = BigQueryDialect()
        cte_sql = "SELECT * FROM `project.dataset.table`"
        expressions = [
            "CAST(COUNT(*) AS FLOAT64) AS `total_rows`",
            "CAST(AVG(price) AS FLOAT64) AS `avg_price`"
        ]

        query = dialect.build_cte_query(cte_sql, expressions)

        expected = (
            "WITH source AS (SELECT * FROM `project.dataset.table`) "
            "SELECT CAST(COUNT(*) AS FLOAT64) AS `total_rows`, "
            "CAST(AVG(price) AS FLOAT64) AS `avg_price` FROM source"
        )
        assert query == expected
```

**Step 1.2**: Run tests to confirm they fail
```bash
uv run pytest tests/test_bigquery_dialect.py -v
```

**Step 1.3**: Implement BigQueryDialect class with basic translations

Add to `src/dqx/dialect.py`:
```python
class BigQueryDialect:
    """BigQuery SQL dialect implementation.

    This dialect generates SQL compatible with BigQuery's syntax,
    including COUNTIF, VAR_SAMP, and STRUCT-based batch queries.
    """

    name = "bigquery"

    def translate_sql_op(self, op: ops.SqlOp) -> str:
        """Translate SqlOp to BigQuery SQL syntax."""
        match op:
            case ops.NumRows():
                return f"CAST(COUNT(*) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Average(column=col):
                return f"CAST(AVG({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Minimum(column=col):
                return f"CAST(MIN({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Maximum(column=col):
                return f"CAST(MAX({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Sum(column=col):
                return f"CAST(SUM({col}) AS FLOAT64) AS `{op.sql_col}`"

            case _:
                raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")

    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        """Build CTE query using the common helper."""
        return build_cte_query(cte_sql, select_expressions)
```

**Step 1.4**: Run tests to confirm they pass
```bash
uv run pytest tests/test_bigquery_dialect.py -v
```

**Step 1.5**: Check code quality
```bash
uv run mypy src/dqx/dialect.py tests/test_bigquery_dialect.py
uv run ruff check --fix src/dqx/dialect.py tests/test_bigquery_dialect.py
```

### Task Group 2: Advanced SqlOp Translations (Variance, First, NullCount, NegativeCount)

**Objective**: Implement remaining SqlOp translations with BigQuery-specific syntax.

**Files to modify**:
1. `tests/test_bigquery_dialect.py`
2. `src/dqx/dialect.py`

**Step 2.1**: Add failing tests for advanced operations

Add to `tests/test_bigquery_dialect.py`:
```python
from dqx.ops import First, NegativeCount, NullCount, Variance

# Add to TestBigQueryDialect class:

    def test_translate_variance(self) -> None:
        """Test Variance translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = Variance("sales")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(VAR_SAMP(sales) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_first(self) -> None:
        """Test First translation using MIN for deterministic results."""
        dialect = BigQueryDialect()
        op = First("timestamp")
        sql = dialect.translate_sql_op(op)
        # Using MIN for deterministic "first" value
        assert sql == f"CAST(MIN(timestamp) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_null_count(self) -> None:
        """Test NullCount translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = NullCount("email")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNTIF(email IS NULL) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_negative_count(self) -> None:
        """Test NegativeCount translation to BigQuery SQL."""
        dialect = BigQueryDialect()
        op = NegativeCount("profit")
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNTIF(profit < 0) AS FLOAT64) AS `{op.sql_col}`"
```

**Step 2.2**: Run tests to confirm they fail
```bash
uv run pytest tests/test_bigquery_dialect.py::TestBigQueryDialect::test_translate_variance -v
uv run pytest tests/test_bigquery_dialect.py::TestBigQueryDialect::test_translate_first -v
uv run pytest tests/test_bigquery_dialect.py::TestBigQueryDialect::test_translate_null_count -v
uv run pytest tests/test_bigquery_dialect.py::TestBigQueryDialect::test_translate_negative_count -v
```

**Step 2.3**: Implement advanced translations

Update the `translate_sql_op` method in BigQueryDialect:
```python
            case ops.Variance(column=col):
                return f"CAST(VAR_SAMP({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.First(column=col):
                # Using MIN for deterministic "first" value
                return f"CAST(MIN({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.NullCount(column=col):
                return f"CAST(COUNTIF({col} IS NULL) AS FLOAT64) AS `{op.sql_col}`"

            case ops.NegativeCount(column=col):
                return f"CAST(COUNTIF({col} < 0) AS FLOAT64) AS `{op.sql_col}`"
```

**Step 2.4**: Run all tests
```bash
uv run pytest tests/test_bigquery_dialect.py -v
```

### Task Group 3: DuplicateCount Translation and Error Handling

**Objective**: Implement DuplicateCount translation and proper error handling for unsupported operations.

**Files to modify**:
1. `tests/test_bigquery_dialect.py`
2. `src/dqx/dialect.py`

**Step 3.1**: Add tests for DuplicateCount and error handling

Add to `tests/test_bigquery_dialect.py`:
```python
from dqx.ops import DuplicateCount, SqlOp

# Add to TestBigQueryDialect class:

    def test_translate_duplicate_count_single_column(self) -> None:
        """Test DuplicateCount translation for single column."""
        dialect = BigQueryDialect()
        op = DuplicateCount(["user_id"])
        sql = dialect.translate_sql_op(op)
        assert sql == f"CAST(COUNT(*) - COUNT(DISTINCT user_id) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_duplicate_count_multiple_columns(self) -> None:
        """Test DuplicateCount translation for multiple columns."""
        dialect = BigQueryDialect()
        op = DuplicateCount(["user_id", "product_id"])
        sql = dialect.translate_sql_op(op)
        # Columns should be sorted in DuplicateCount
        assert sql == f"CAST(COUNT(*) - COUNT(DISTINCT (product_id, user_id)) AS FLOAT64) AS `{op.sql_col}`"

    def test_translate_unsupported_op(self) -> None:
        """Test error handling for unsupported operations."""
        dialect = BigQueryDialect()

        class UnsupportedOp(SqlOp):
            @property
            def name(self) -> str:
                return "unsupported"
            @property
            def prefix(self) -> str:
                return "unsup"
            @property
            def sql_col(self) -> str:
                return "unsup_col"
            def value(self) -> float:
                return 0.0
            def assign(self, value: float) -> None:
                pass
            def clear(self) -> None:
                pass

        op = UnsupportedOp()
        with pytest.raises(ValueError, match="Unsupported SqlOp type: UnsupportedOp"):
            dialect.translate_sql_op(op)
```

**Step 3.2**: Implement DuplicateCount translation

Update the `translate_sql_op` method:
```python
            case ops.DuplicateCount(columns=cols):
                if len(cols) == 1:
                    distinct_expr = cols[0]
                else:
                    distinct_expr = f"({', '.join(cols)})"
                return f"CAST(COUNT(*) - COUNT(DISTINCT {distinct_expr}) AS FLOAT64) AS `{op.sql_col}`"
```

**Step 3.3**: Run tests and fix any issues
```bash
uv run pytest tests/test_bigquery_dialect.py -v
uv run mypy src/dqx/dialect.py tests/test_bigquery_dialect.py
uv run ruff check --fix src/dqx/dialect.py tests/test_bigquery_dialect.py
```

**Step 3.4**: Commit Task Groups 1-3
```bash
git add src/dqx/dialect.py tests/test_bigquery_dialect.py
git commit -m "feat(dialect): implement BigQuery dialect with all SqlOp translations

- Add BigQueryDialect class with FLOAT64 type casting
- Implement all SqlOp translations using BigQuery syntax
- Use COUNTIF for null and negative counting
- Use VAR_SAMP for variance calculations
- Use MIN for deterministic First operation
- Add comprehensive tests for all operations"
```

### Task Group 4: Batch Query Implementation with STRUCT

**Objective**: Implement STRUCT-based batch query generation for BigQuery.

**Files to modify**:
1. `tests/test_bigquery_dialect.py`
2. `src/dqx/dialect.py`

**Step 4.1**: Add tests for batch CTE queries

Add to `tests/test_bigquery_dialect.py`:
```python
from datetime import date
from dqx.common import ResultKey
from dqx.models import BatchCTEData

# Add new test class:

class TestBigQueryBatchQueries:
    """Test BigQuery batch CTE query generation."""

    def test_build_batch_cte_query_single_date(self) -> None:
        """Test batch query with single date using STRUCT."""
        dialect = BigQueryDialect()

        key = ResultKey(date(2024, 1, 1), {})
        ops = [Sum("revenue"), Average("price")]

        cte_data = [
            BatchCTEData(
                key=key,
                cte_sql="SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-01'",
                ops=ops  # type: ignore[arg-type]
            )
        ]

        sql = dialect.build_batch_cte_query(cte_data)

        # Verify STRUCT format
        assert "STRUCT(" in sql
        assert "AS values FROM" in sql
        assert "'2024-01-01' AS date" in sql
        # Check that ops are included in STRUCT
        assert f"AS {ops[0].sql_col}" in sql
        assert f"AS {ops[1].sql_col}" in sql

    def test_build_batch_cte_query_multiple_dates(self) -> None:
        """Test batch query with multiple dates."""
        dialect = BigQueryDialect()

        cte_data = []
        for day in [1, 2, 3]:
            key = ResultKey(date(2024, 1, day), {})
            ops = [Sum("revenue"), Average("price")]

            cte_data.append(
                BatchCTEData(
                    key=key,
                    cte_sql=f"SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-0{day}'",
                    ops=ops  # type: ignore[arg-type]
                )
            )

        sql = dialect.build_batch_cte_query(cte_data)

        # Should have UNION ALL for multiple dates
        assert sql.count("UNION ALL") == 2  # 3 dates = 2 UNION ALLs
        assert sql.count("STRUCT(") == 3  # One STRUCT per date
        assert "'2024-01-01' AS date" in sql
        assert "'2024-01-02' AS date" in sql
        assert "'2024-01-03' AS date" in sql

    def test_build_batch_cte_query_empty_data(self) -> None:
        """Test error handling for empty CTE data."""
        dialect = BigQueryDialect()

        with pytest.raises(ValueError, match="No CTE data provided"):
            dialect.build_batch_cte_query([])

    def test_build_batch_cte_query_no_ops(self) -> None:
        """Test error handling when no ops provided."""
        dialect = BigQueryDialect()

        cte_data = [
            BatchCTEData(
                key=ResultKey(date(2024, 1, 1), {}),
                cte_sql="SELECT * FROM `project.dataset.sales`",
                ops=[]  # No ops
            )
        ]

        with pytest.raises(ValueError, match="No metrics to compute"):
            dialect.build_batch_cte_query(cte_data)
```

**Step 4.2**: Implement batch CTE query method

Add to BigQueryDialect class:
```python
    def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
        """Build batch CTE query using STRUCT for BigQuery.

        This method generates a query that returns results as:
        - date: The date string
        - values: A STRUCT containing all metric values

        The STRUCT approach reduces the result set size compared to UNPIVOT,
        similar to DuckDB's MAP feature.

        Args:
            cte_data: List of BatchCTEData objects

        Returns:
            Complete SQL query with CTEs and STRUCT-based results
        """
        if not cte_data:
            raise ValueError("No CTE data provided")

        cte_parts = []
        struct_selects = []

        # Build CTEs and collect STRUCT queries
        for i, data in enumerate(cte_data):
            if not data.ops:
                continue

            # Format date for CTE names
            date_suffix = data.key.yyyy_mm_dd.strftime("%Y_%m_%d")
            source_cte = f"source_{date_suffix}_{i}"

            # Add source CTE
            cte_parts.append(f"{source_cte} AS ({data.cte_sql})")

            # Build STRUCT fields
            struct_fields = []
            for op in data.ops:
                # Translate to SQL and extract expression
                sql_expr = self.translate_sql_op(op)
                # Remove the AS `alias` part to get just the expression
                expr_part = sql_expr.split(" AS ")[0]
                # Add as STRUCT field with the sql_col as field name
                struct_fields.append(f"{expr_part} AS {op.sql_col}")

            # Build STRUCT expression
            struct_expr = "STRUCT(\n    " + ",\n    ".join(struct_fields) + "\n  )"
            date_str = data.key.yyyy_mm_dd.isoformat()

            struct_selects.append(
                f"SELECT '{date_str}' AS date, {struct_expr} AS values FROM {source_cte}"
            )

        if not struct_selects:
            raise ValueError("No metrics to compute")

        # Build final query
        cte_clause = "WITH\n  " + ",\n  ".join(cte_parts)
        union_clause = "\n".join(
            f"{'UNION ALL' if i > 0 else ''}\n{select}"
            for i, select in enumerate(struct_selects)
        )

        return f"{cte_clause}\n{union_clause}"
```

**Step 4.3**: Run batch query tests
```bash
uv run pytest tests/test_bigquery_dialect.py::TestBigQueryBatchQueries -v
```

**Step 4.4**: Fix any type errors and run full test suite
```bash
uv run mypy src/dqx/dialect.py tests/test_bigquery_dialect.py
uv run ruff check --fix src/dqx/dialect.py tests/test_bigquery_dialect.py
uv run pytest tests/test_bigquery_dialect.py -v
```

### Task Group 5: Dialect Registration and Integration

**Objective**: Register BigQuery dialect and add integration tests.

**Files to modify**:
1. `tests/test_bigquery_dialect.py`
2. `src/dqx/dialect.py`
3. `tests/test_dialect.py` (to verify integration)

**Step 5.1**: Add registration test

Add to `tests/test_bigquery_dialect.py`:
```python
# Add to TestBigQueryDialect class:

    def test_bigquery_dialect_registration(self) -> None:
        """Test that BigQuery dialect can be registered and retrieved."""
        # BigQuery should be registered on module import
        dialect = get_dialect("bigquery")
        assert isinstance(dialect, BigQueryDialect)
        assert dialect.name == "bigquery"
```

**Step 5.2**: Register BigQuery dialect

At the end of `src/dqx/dialect.py`:
```python
# Register built-in dialects
register_dialect("duckdb", DuckDBDialect)
register_dialect("bigquery", BigQueryDialect)
```

**Step 5.3**: Add integration test in existing dialect tests

Add to `tests/test_dialect.py` in the TestDialectRegistry class:
```python
    def test_bigquery_dialect_is_registered(self) -> None:
        """Test that BigQuery dialect is registered on import."""
        assert "bigquery" in _DIALECT_REGISTRY
        assert _DIALECT_REGISTRY["bigquery"] is BigQueryDialect
```

**Step 5.4**: Run all dialect tests
```bash
uv run pytest tests/test_dialect.py tests/test_bigquery_dialect.py -v
```

**Step 5.5**: Run full test suite to ensure no regression
```bash
uv run pytest tests/ -v --tb=short
```

**Step 5.6**: Commit Task Groups 4-5
```bash
git add src/dqx/dialect.py tests/test_bigquery_dialect.py tests/test_dialect.py
git commit -m "feat(dialect): add STRUCT-based batch queries and register BigQuery dialect

- Implement build_batch_cte_query using BigQuery STRUCT type
- STRUCT approach reduces result set size like DuckDB MAP
- Register BigQuery dialect in global registry
- Add comprehensive tests for batch queries
- Ensure proper error handling for edge cases"
```

### Task Group 6: Documentation and Final Verification

**Objective**: Add documentation and perform final verification.

**Files to modify**:
1. `src/dqx/dialect.py` (docstring updates)
2. `examples/bigquery_dialect_demo.py` (create new)

**Step 6.1**: Update module docstring

Update the module docstring in `src/dqx/dialect.py` to include BigQuery example:
```python
"""SQL dialect abstraction for DQX framework.

This module provides a protocol-based abstraction for SQL dialects,
allowing DQX to support different SQL databases beyond DuckDB.

## Overview

The dialect system enables DQX to generate SQL compatible with different
database systems. Each dialect handles:
- Translation of SqlOp operations to dialect-specific SQL
- Query formatting and structure
- Database-specific function mappings

## Usage

### Using the DuckDB dialect (default):

    >>> from dqx.dialect import get_dialect
    >>> from dqx.ops import Average, Sum
    >>>
    >>> dialect = get_dialect("duckdb")
    >>> avg_op = Average("price")
    >>> sql = dialect.translate_sql_op(avg_op)
    >>> print(sql)  # CAST(AVG(price) AS DOUBLE) AS 'prefix_average(price)'

### Using the BigQuery dialect:

    >>> dialect = get_dialect("bigquery")
    >>> avg_op = Average("price")
    >>> sql = dialect.translate_sql_op(avg_op)
    >>> print(sql)  # CAST(AVG(price) AS FLOAT64) AS `prefix_average(price)`

## Supported Dialects

- **duckdb**: Default dialect for DuckDB (uses DOUBLE type, single quotes)
- **bigquery**: Google BigQuery dialect (uses FLOAT64 type, backticks, COUNTIF)
"""
```

**Step 6.2**: Create BigQuery example demo

Create `examples/bigquery_dialect_demo.py`:
```python
"""Demonstrate BigQuery dialect usage in DQX framework.

This example shows how to use the BigQuery SQL dialect to generate
BigQuery-compatible SQL for data quality metrics.
"""

from datetime import date

from dqx.dialect import get_dialect
from dqx.ops import Average, DuplicateCount, Maximum, Minimum, NullCount, NumRows, Sum


def main() -> None:
    """Demonstrate BigQuery dialect features."""
    # Get BigQuery dialect
    dialect = get_dialect("bigquery")
    print(f"Using {dialect.name} dialect\n")

    # Basic aggregations
    print("Basic Aggregations:")
    print("-" * 50)

    ops = [
        NumRows(),
        Sum("revenue"),
        Average("price"),
        Minimum("quantity"),
        Maximum("score"),
    ]

    for op in ops:
        sql = dialect.translate_sql_op(op)
        print(f"{op.name:20} -> {sql}")

    # Advanced operations
    print("\nAdvanced Operations:")
    print("-" * 50)

    advanced_ops = [
        NullCount("email"),
        DuplicateCount(["user_id", "product_id"]),
    ]

    for op in advanced_ops:
        sql = dialect.translate_sql_op(op)
        print(f"{op.name:20} -> {sql}")

    # CTE query example
    print("\nCTE Query Example:")
    print("-" * 50)

    cte_sql = "SELECT * FROM `project.dataset.sales` WHERE date = '2024-01-01'"
    expressions = [
        dialect.translate_sql_op(NumRows()),
        dialect.translate_sql_op(Sum("revenue")),
        dialect.translate_sql_op(Average("price")),
    ]

    query = dialect.build_cte_query(cte_sql, expressions)
    print(query)


if __name__ == "__main__":
    main()
```

**Step 6.3**: Run final verification
```bash
# Run all tests
uv run pytest tests/ -v

# Check code quality
uv run mypy src/dqx/dialect.py tests/test_bigquery_dialect.py examples/bigquery_dialect_demo.py
uv run ruff check --fix src/dqx/dialect.py tests/test_bigquery_dialect.py examples/bigquery_dialect_demo.py

# Run the example to verify it works
uv run python examples/bigquery_dialect_demo.py
```

**Step 6.4**: Commit documentation and examples
```bash
git add src/dqx/dialect.py examples/bigquery_dialect_demo.py
git commit -m "docs(dialect): add BigQuery documentation and example

- Update module docstring with BigQuery usage examples
- Create bigquery_dialect_demo.py to demonstrate features
- Show SQL generation for all supported operations
- Include CTE query building example"
```

## Final Verification

After completing all task groups, perform a final verification:

```bash
# Ensure we're on the feature branch
git status

# Run full test suite with coverage
uv run pytest tests/ -v --cov=dqx.dialect --cov-report=term-missing

# Run pre-commit checks
bin/run-hooks.sh

# If everything passes, the implementation is complete
```

## Summary

This plan provides a complete TDD-based implementation of BigQuery dialect support for DQX. The implementation:

1. Supports all SqlOp types with BigQuery-specific SQL syntax
2. Uses STRUCT for efficient batch query results (similar to DuckDB's MAP)
3. Maintains full compatibility with the existing analyzer
4. Includes comprehensive tests with 100% coverage
5. Provides clear documentation and examples

Each task group is designed to be committed independently with passing tests, following the project's coding standards and commit conventions.
