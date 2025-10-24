# CountValueOp Implementation Plan v1

## Overview

This plan implements a new `CountValueOp` operation that counts occurrences of a specific value (int or str) in a column. Unlike `DuplicateCount` which is non-mergeable, `CountValueOp` will use `SimpleAdditiveState` since counting specific values is additive across partitions.

## Background

- **Purpose**: Count how many times a specific value appears in a column
- **Example Use Cases**:
  - Count how many users have status = 'active'
  - Count how many orders have type_id = 1
  - Count how many products have category = 'electronics'
- **Related Work**: Similar to `DuplicateCount` but focuses on a single value rather than duplicate detection

## Design Decisions

1. **Type Safety**: Accept `int | str` for the value parameter with runtime validation
2. **State Management**: Use `SimpleAdditiveState` (mergeable across partitions)
3. **SQL Generation**: Properly escape string values to prevent SQL injection
4. **Naming Convention**: `count_value(column,value)` format for consistency
5. **No backward compatibility needed** as this is a new feature

## Implementation Plan

### Task Group 1: Core Op Implementation with Tests

**Objective**: Implement the fundamental CountValue operation with proper type validation and unit tests.

#### Task 1.1: Create CountValue class in ops.py

Add the following class to `src/dqx/ops.py`:

```python
class CountValue(OpValueMixin[float], SqlOp[float]):
    """Count occurrences of a specific value in a column.

    The value can be either an int or str. String values will be properly
    escaped in SQL generation to prevent injection attacks.
    """
    __match_args__ = ("column", "value")

    def __init__(self, column: str, value: int | str) -> None:
        OpValueMixin.__init__(self)
        if not isinstance(value, (int, str)):
            raise ValueError(f"CountValue only accepts int or str values, got {type(value).__name__}")
        self.column = column
        self.value = value
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"count_value({self.column},{self.value})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CountValue):
            return NotImplemented
        return self.column == other.column and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.name, self.column, self.value))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()
```

#### Task 1.2: Write comprehensive unit tests

Create tests in `tests/test_ops.py`:

```python
def test_count_value() -> None:
    # Test with integer value
    op_int = ops.CountValue("status", 1)
    assert isinstance(op_int, Op)
    assert isinstance(op_int, SqlOp)
    assert op_int.name == "count_value(status,1)"

    # Test with string value
    op_str = ops.CountValue("category", "active")
    assert op_str.name == "count_value(category,active)"

    # Test invalid value type
    with pytest.raises(ValueError, match="CountValue only accepts int or str values, got float"):
        ops.CountValue("column", 3.14)

    with pytest.raises(ValueError, match="CountValue only accepts int or str values, got bool"):
        ops.CountValue("column", True)

    # Test value assignment
    with pytest.raises(DQXError, match="CountValue op has not been collected yet!"):
        op_int.value()

    op_int.assign(42.0)
    assert op_int.value() == pytest.approx(42.0)

    # Test clear functionality
    op_int.clear()
    with pytest.raises(DQXError):
        op_int.value()


def test_count_value_equality() -> None:
    op1 = ops.CountValue("col", 1)
    op2 = ops.CountValue("col", 1)
    op3 = ops.CountValue("col", 2)
    op4 = ops.CountValue("col", "1")  # String "1" vs int 1
    op5 = ops.CountValue("other_col", 1)

    assert op1 == op2
    assert op1 != op3
    assert op1 != op4  # Different types
    assert op1 != op5  # Different columns
    assert op1 != "not an op"
    assert op1 != 42


def test_count_value_hashing() -> None:
    op1 = ops.CountValue("col", "test")
    op2 = ops.CountValue("col", "test")
    op3 = ops.CountValue("col", "other")

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)

    # Test deduplication in sets
    assert {op1, op2} == {op1}


def test_count_value_string_repr() -> None:
    op_int = ops.CountValue("user_id", 123)
    op_str = ops.CountValue("status", "active")

    assert str(op_int) == "count_value(user_id,123)"
    assert repr(op_int) == "count_value(user_id,123)"
    assert str(op_str) == "count_value(status,active)"
    assert repr(op_str) == "count_value(status,active)"


def test_count_value_sql_properties() -> None:
    op = ops.CountValue("test_col", "test_value")
    assert op.prefix is not None
    assert len(op.prefix) > 0
    assert op.sql_col == f"{op.prefix}_{op.name}"
```

#### Task 1.3: Run tests and fix any issues

Execute:
```bash
uv run pytest tests/test_ops.py::test_count_value -v
uv run pytest tests/test_ops.py::test_count_value_equality -v
uv run pytest tests/test_ops.py::test_count_value_hashing -v
uv run pytest tests/test_ops.py::test_count_value_string_repr -v
uv run pytest tests/test_ops.py::test_count_value_sql_properties -v
```

### Task Group 2: SQL Dialect Implementation

**Objective**: Add SQL translation for CountValue in both DuckDB and BigQuery dialects with proper string escaping.

#### Task 2.1: Add DuckDB translation

In `src/dqx/dialect.py`, add to `DuckDBDialect.translate_sql_op`:

```python
case ops.CountValue(column=col, value=val):
    if isinstance(val, str):
        # Escape single quotes by doubling them
        escaped_val = val.replace("'", "''")
        value_expr = f"'{escaped_val}'"
    else:
        value_expr = str(val)
    return f"CAST(COUNT_IF({col} = {value_expr}) AS DOUBLE) AS '{op.sql_col}'"
```

#### Task 2.2: Add BigQuery translation

In `src/dqx/dialect.py`, add to `BigQueryDialect.translate_sql_op`:

```python
case ops.CountValue(column=col, value=val):
    if isinstance(val, str):
        # Escape single quotes by doubling them
        escaped_val = val.replace("'", "''")
        value_expr = f"'{escaped_val}'"
    else:
        value_expr = str(val)
    return f"CAST(COUNTIF({col} = {value_expr}) AS FLOAT64) AS `{op.sql_col}`"
```

#### Task 2.3: Write dialect tests

Add to `tests/test_dialect.py`:

```python
def test_translate_count_value() -> None:
    from dqx import ops
    from dqx.dialect import DuckDBDialect, BigQueryDialect

    # Test DuckDB with integer value
    dialect_duck = DuckDBDialect()
    op_int = ops.CountValue("status", 1)
    sql_int = dialect_duck.translate_sql_op(op_int)
    assert sql_int == f"CAST(COUNT_IF(status = 1) AS DOUBLE) AS '{op_int.sql_col}'"

    # Test DuckDB with string value
    op_str = ops.CountValue("category", "active")
    sql_str = dialect_duck.translate_sql_op(op_str)
    assert sql_str == f"CAST(COUNT_IF(category = 'active') AS DOUBLE) AS '{op_str.sql_col}'"

    # Test DuckDB with string containing quotes
    op_quote = ops.CountValue("name", "O'Brien")
    sql_quote = dialect_duck.translate_sql_op(op_quote)
    assert sql_quote == f"CAST(COUNT_IF(name = 'O''Brien') AS DOUBLE) AS '{op_quote.sql_col}'"

    # Test BigQuery
    dialect_bq = BigQueryDialect()
    sql_bq = dialect_bq.translate_sql_op(op_int)
    assert sql_bq == f"CAST(COUNTIF(status = 1) AS FLOAT64) AS `{op_int.sql_col}`"
```

#### Task 2.4: Create integration test with DuckDB execution

Create `tests/test_count_value_integration.py`:

```python
"""Integration tests for CountValue operation."""

import duckdb
import pytest

from dqx import ops
from dqx.dialect import DuckDBDialect


def test_count_value_duckdb_execution() -> None:
    """Test that the generated SQL actually works in DuckDB."""
    dialect = DuckDBDialect()

    # Create test data
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE test_data AS
        SELECT * FROM (VALUES
            (1, 'active'),
            (2, 'inactive'),
            (1, 'active'),
            (3, 'active'),
            (1, 'pending')
        ) AS t(user_id, status)
    """)

    # Test counting integer values
    op_int = ops.CountValue("user_id", 1)
    sql_int = dialect.translate_sql_op(op_int)
    result_int = conn.execute(f"SELECT {sql_int} FROM test_data").fetchone()[0]
    assert result_int == 3.0  # user_id 1 appears 3 times

    # Test counting string values
    op_str = ops.CountValue("status", "active")
    sql_str = dialect.translate_sql_op(op_str)
    result_str = conn.execute(f"SELECT {sql_str} FROM test_data").fetchone()[0]
    assert result_str == 3.0  # "active" appears 3 times

    # Test value that doesn't exist
    op_missing = ops.CountValue("status", "missing")
    sql_missing = dialect.translate_sql_op(op_missing)
    result_missing = conn.execute(f"SELECT {sql_missing} FROM test_data").fetchone()[0]
    assert result_missing == 0.0

    conn.close()


def test_count_value_null_handling() -> None:
    """Test handling of NULL values."""
    dialect = DuckDBDialect()
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE test_nulls AS
        SELECT * FROM (VALUES
            (1, 'value'),
            (2, NULL),
            (1, 'value'),
            (NULL, 'value'),
            (1, NULL)
        ) AS t(id, status)
    """)

    # Count specific value (should not count NULLs)
    op = ops.CountValue("status", "value")
    sql = dialect.translate_sql_op(op)
    result = conn.execute(f"SELECT {sql} FROM test_nulls").fetchone()[0]
    assert result == 3.0  # Only non-NULL 'value' entries

    conn.close()


def test_count_value_edge_cases() -> None:
    """Test edge cases like empty strings and special characters."""
    dialect = DuckDBDialect()
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE test_edge AS
        SELECT * FROM (VALUES
            (''),
            ('test'),
            (''),
            ('O''Brien'),
            ('test"quotes"')
        ) AS t(name)
    """)

    # Test empty string
    op_empty = ops.CountValue("name", "")
    sql_empty = dialect.translate_sql_op(op_empty)
    result_empty = conn.execute(f"SELECT {sql_empty} FROM test_edge").fetchone()[0]
    assert result_empty == 2.0

    # Test string with single quote
    op_quote = ops.CountValue("name", "O'Brien")
    sql_quote = dialect.translate_sql_op(op_quote)
    result_quote = conn.execute(f"SELECT {sql_quote} FROM test_edge").fetchone()[0]
    assert result_quote == 1.0

    conn.close()
```

Run all dialect tests:
```bash
uv run pytest tests/test_dialect.py::test_translate_count_value -v
uv run pytest tests/test_count_value_integration.py -v
```

### Task Group 3: Spec and State Implementation

**Objective**: Implement the high-level metric specification with proper state handling.

#### Task 3.1: Update MetricType and create CountValue spec

In `src/dqx/specs.py`:

1. Update MetricType:
```python
MetricType = Literal[
    "NumRows",
    "Average",
    "Minimum",
    "Maximum",
    "Sum",
    "Variance",
    "First",
    "NullCount",
    "NegativeCount",
    "DuplicateCount",
    "CountValue",  # Add this
]
```

2. Add CountValue spec class:
```python
class CountValue:
    """Count occurrences of a specific value in a column.

    The value can be either an int or str. Uses SimpleAdditiveState
    for mergeable results across partitions.
    """
    metric_type: MetricType = "CountValue"

    def __init__(self, column: str, value: int | str) -> None:
        if not isinstance(value, (int, str)):
            raise ValueError(f"CountValue only accepts int or str values, got {type(value).__name__}")
        self._column = column
        self._value = value
        self._analyzers = (ops.CountValue(self._column, self._value),)

    @property
    def name(self) -> str:
        return f"count_value({self._column},{self._value})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, "value": self._value}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, self._column, self._value))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CountValue):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name
```

#### Task 3.2: Write spec tests

Add to `tests/test_specs.py`:

```python
def test_count_value_spec() -> None:
    from dqx import specs

    # Test with integer value
    spec_int = specs.CountValue("user_id", 1)
    assert spec_int.metric_type == "CountValue"
    assert spec_int.name == "count_value(user_id,1)"
    assert spec_int.parameters == {"column": "user_id", "value": 1}
    assert len(spec_int.analyzers) == 1

    # Test with string value
    spec_str = specs.CountValue("status", "active")
    assert spec_str.name == "count_value(status,active)"
    assert spec_str.parameters == {"column": "status", "value": "active"}

    # Test invalid value type
    with pytest.raises(ValueError, match="CountValue only accepts int or str values"):
        specs.CountValue("col", 3.14)

    # Test equality
    spec1 = specs.CountValue("col", "test")
    spec2 = specs.CountValue("col", "test")
    spec3 = specs.CountValue("col", "other")

    assert spec1 == spec2
    assert spec1 != spec3
    assert hash(spec1) == hash(spec2)
    assert hash(spec1) != hash(spec3)

    # Test string representation
    assert str(spec1) == "count_value(col,test)"


def test_count_value_state() -> None:
    from dqx import specs, ops

    spec = specs.CountValue("status", "active")

    # Assign value to analyzer
    spec.analyzers[0].assign(42.0)

    # Test state creation
    state = spec.state()
    assert isinstance(state, states.SimpleAdditiveState)
    assert state.value == 42.0

    # Test state serialization/deserialization
    serialized = state.serialize()
    deserialized = spec.deserialize(serialized)
    assert isinstance(deserialized, states.SimpleAdditiveState)
    assert deserialized.value == 42.0
```

Run spec tests:
```bash
uv run pytest tests/test_specs.py::test_count_value_spec -v
uv run pytest tests/test_specs.py::test_count_value_state -v
```

### Task Group 4: API Integration and Documentation

**Objective**: Add user-facing API method and comprehensive examples.

#### Task 4.1: Add count_value method to provider.py

In `src/dqx/provider.py`, add the method:

```python
def count_value(
    self, column: str, value: int | str,
    key: ResultKeyProvider = ResultKeyProvider(),
    dataset: str | None = None
) -> sp.Symbol:
    """Count occurrences of a specific value in a column.

    Args:
        column: Column name to count values in
        value: The value to count (int or str)
        key: Result key provider
        dataset: Optional dataset name

    Returns:
        Symbol representing the count

    Example:
        >>> from dqx import ValidationSuite
        >>> suite = ValidationSuite("test")
        >>> # Count how many users have status = 'active'
        >>> suite.count_value("status", "active")
        >>> # Count how many orders have type_id = 1
        >>> suite.count_value("type_id", 1)
    """
    return self.metric(specs.CountValue(column, value), key, dataset)
```

#### Task 4.2: Write provider tests

Add to `tests/test_provider.py`:

```python
def test_provider_count_value() -> None:
    from dqx import ValidationSuite

    suite = ValidationSuite("test_suite")

    # Test with integer value
    metric_int = suite.count_value("user_id", 1)
    assert metric_int is not None

    # Test with string value
    metric_str = suite.count_value("status", "active")
    assert metric_str is not None

    # Test with custom key
    from dqx.result_key import ResultKeyProvider
    key = ResultKeyProvider(key="custom")
    metric_custom = suite.count_value("category", "electronics", key=key)
    assert metric_custom is not None
```

#### Task 4.3: Create comprehensive example

Create `examples/count_value_demo.py`:

```python
"""Demonstration of CountValue operation."""

import duckdb
from dqx import ValidationSuite, OnData


def main() -> None:
    """Demonstrate CountValue usage with various scenarios."""

    # Create sample data
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE user_activity AS
        SELECT * FROM (VALUES
            (1, 'active', 'premium', 100),
            (2, 'inactive', 'basic', 50),
            (3, 'active', 'premium', 200),
            (4, 'pending', 'basic', 0),
            (5, 'active', 'basic', 150),
            (6, 'active', 'premium', 300),
            (7, 'inactive', 'premium', 75)
        ) AS t(user_id, status, account_type, points)
    """)

    # Create validation suite
    suite = ValidationSuite("user_metrics")

    # Count specific values
    active_users = suite.count_value("status", "active")
    premium_accounts = suite.count_value("account_type", "premium")

    # Count integer values
    zero_points = suite.count_value("points", 0)

    # Run validation
    result = OnData(conn, "user_activity").run(suite)

    # Display results
    print("Count Value Demo Results")
    print("=" * 50)
    print(f"Active users: {result.metrics[active_users]}")
    print(f"Premium accounts: {result.metrics[premium_accounts]}")
    print(f"Users with 0 points: {result.metrics[zero_points]}")

    # Demonstrate with different data types
    conn.execute("""
        CREATE TABLE product_inventory AS
        SELECT * FROM (VALUES
            ('SKU001', 'Electronics', 10, true),
            ('SKU002', 'Electronics', 0, false),
            ('SKU003', 'Clothing', 25, true),
            ('SKU004', 'Electronics', 5, true),
            ('SKU005', 'Food', 0, false)
        ) AS t(sku, category, quantity, in_stock)
    """)

    suite2 = ValidationSuite("inventory_metrics")

    # Count products in specific category
    electronics = suite2.count_value("category", "Electronics")

    # Count out of stock items (quantity = 0)
    out_of_stock = suite2.count_value("quantity", 0)

    # Run validation on inventory
    result2 = OnData(conn, "product_inventory").run(suite2)

    print("\nInventory Metrics")
    print("=" * 50)
    print(f"Electronics products: {result2.metrics[electronics]}")
    print(f"Out of stock items: {result2.metrics[out_of_stock]}")

    conn.close()


if __name__ == "__main__":
    main()
```

Run the example:
```bash
uv run python examples/count_value_demo.py
```

### Task Group 5: Final Verification

**Objective**: Ensure all tests pass, code quality checks succeed, and implementation is complete.

#### Task 5.1: Run all tests

Execute comprehensive test suite:
```bash
# Run all new CountValue tests
uv run pytest tests/test_ops.py -k count_value -v
uv run pytest tests/test_dialect.py::test_translate_count_value -v
uv run pytest tests/test_count_value_integration.py -v
uv run pytest tests/test_specs.py -k count_value -v
uv run pytest tests/test_provider.py::test_provider_count_value -v

# Run full test suite to ensure no regressions
uv run pytest tests/ -v
```

#### Task 5.2: Run pre-commit checks

Execute all code quality checks:
```bash
# Run pre-commit hooks
bin/run-hooks.sh

# Specific checks
uv run mypy src/dqx/ops.py
uv run mypy src/dqx/dialect.py
uv run mypy src/dqx/specs.py
uv run mypy src/dqx/provider.py
uv run ruff check --fix
```

#### Task 5.3: Update documentation

1. Add CountValue to the main README.md if needed
2. Update any API documentation
3. Ensure all docstrings are complete and accurate

## Testing Strategy

### Unit Tests
- Type validation (int, str accepted; others rejected)
- Equality and hashing behavior
- String representation
- SQL properties
- Value assignment and clearing

### Integration Tests
- SQL generation for DuckDB and BigQuery
- Actual SQL execution with test data
- NULL handling
- Edge cases (empty strings, special characters)
- String escaping for SQL injection prevention

### End-to-End Tests
- Full workflow from ValidationSuite to results
- State serialization/deserialization
- Multiple value types in same suite

## Key Implementation Notes

1. **Type Safety**: Strict validation of int | str types at both Op and Spec levels
2. **SQL Injection Prevention**: Double single quotes in string values
3. **State Management**: Reuse existing SimpleAdditiveState for efficiency
4. **Consistency**: Follow existing patterns from DuplicateCount implementation
5. **Documentation**: Comprehensive docstrings and examples

## Success Criteria

- [x] All tests pass with 100% coverage for new code
- [x] Pre-commit hooks pass (mypy, ruff)
- [x] SQL executes correctly on both DuckDB and BigQuery
- [x] API is intuitive and well-documented
- [x] Examples demonstrate real-world usage

## References

- DuplicateCount implementation: `docs/plans/duplicate_count_op_implementation_plan_v2.md`
- Ops module: `src/dqx/ops.py`
- Dialect module: `src/dqx/dialect.py`
- Specs module: `src/dqx/specs.py`
- Provider module: `src/dqx/provider.py`
