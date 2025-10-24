# CountValues Op Implementation Plan v2

## Overview

This plan implements a new `CountValues` operation that counts occurrences of one or more specific values in a column. Unlike v1's single-value `CountValue`, this supports both single values and lists for efficiency and flexibility. It uses `SimpleAdditiveState` since counting specific values is additive across partitions.

## Background

- **Purpose**: Count how many times specific value(s) appear in a column
- **Example Use Cases**:
  - Single value: `count_values("status", "active")`
  - Multiple values: `count_values("status", ["active", "pending"])`
  - Count multiple product categories in one query
  - Check for presence of specific user IDs
- **Key Improvement**: Single query for multiple values instead of multiple separate queries

## Design Decisions

1. **Type Flexibility**: Accept `int`, `str`, `list[int]`, or `list[str]` with runtime validation
2. **No Mixed Types**: Lists must be homogeneous (all int or all str)
3. **Empty List Handling**: Raise error on empty lists
4. **State Management**: Use `SimpleAdditiveState` (mergeable across partitions)
5. **SQL Generation**: Use `=` for single values, `IN` for multiple values
6. **String Escaping**: Properly escape backslashes and quotes to prevent SQL injection
7. **Naming Convention**:
   - Single: `count_values(column,value)`
   - Multiple: `count_values(column,[val1,val2,...])`

## Implementation Plan

### Task Group 1: Core Op Implementation with Tests

**Objective**: Implement the fundamental CountValues operation with proper type validation and unit tests.

#### Task 1.1: Create CountValues class in ops.py

Add the following class to `src/dqx/ops.py`:

```python
class CountValues(OpValueMixin[float], SqlOp[float]):
    """Count occurrences of specific value(s) in a column.

    Accepts single values (int or str) or lists of values (list[int] or list[str]).
    Lists must be homogeneous - all integers or all strings, not mixed.
    String values will be properly escaped in SQL generation to prevent injection.
    """
    __match_args__ = ("column", "values")

    def __init__(self, column: str, values: int | str | list[int] | list[str]) -> None:
        OpValueMixin.__init__(self)

        # Normalize to list for internal consistency
        if isinstance(values, (int, str)):
            self._values = [values]
            self._is_single = True
        elif isinstance(values, list):
            if not values:
                raise ValueError("CountValues requires at least one value")

            # Check homogeneous types
            if not (all(isinstance(v, int) for v in values) or
                    all(isinstance(v, str) for v in values)):
                raise ValueError("CountValues list must contain all integers or all strings, not mixed")

            self._values = values
            self._is_single = False
        else:
            raise ValueError(
                f"CountValues accepts int, str, list[int], or list[str], "
                f"got {type(values).__name__}"
            )

        self.column = column
        self.values = values  # Store original format for equality checks
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        if self._is_single:
            return f"count_values({self.column},{self._values[0]})"
        else:
            # Format as [val1,val2,...] without quotes
            values_str = "[" + ",".join(str(v) for v in self._values) + "]"
            return f"count_values({self.column},{values_str})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CountValues):
            return NotImplemented
        return self.column == other.column and self.values == other.values

    def __hash__(self) -> int:
        # Convert lists to tuples for hashing
        hashable_values = self.values if not isinstance(self.values, list) else tuple(self.values)
        return hash((self.name, self.column, hashable_values))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()
```

#### Task 1.2: Write comprehensive unit tests

Create tests in `tests/test_ops.py`:

```python
def test_count_values_single() -> None:
    # Test with single integer value
    op_int = ops.CountValues("status", 1)
    assert isinstance(op_int, Op)
    assert isinstance(op_int, SqlOp)
    assert op_int.name == "count_values(status,1)"

    # Test with single string value
    op_str = ops.CountValues("category", "active")
    assert op_str.name == "count_values(category,active)"

    # Test value assignment
    with pytest.raises(DQXError, match="CountValues op has not been collected yet!"):
        op_int.value()

    op_int.assign(42.0)
    assert op_int.value() == pytest.approx(42.0)

    # Test clear functionality
    op_int.clear()
    with pytest.raises(DQXError):
        op_int.value()


def test_count_values_multiple() -> None:
    # Test with multiple integer values
    op_ints = ops.CountValues("type_id", [1, 2, 3])
    assert op_ints.name == "count_values(type_id,[1,2,3])"

    # Test with multiple string values
    op_strs = ops.CountValues("status", ["active", "pending"])
    assert op_strs.name == "count_values(status,[active,pending])"

    # Test single-item list
    op_single_list = ops.CountValues("category", ["electronics"])
    assert op_single_list.name == "count_values(category,[electronics])"


def test_count_values_invalid_types() -> None:
    # Test invalid single value type
    with pytest.raises(ValueError, match="CountValues accepts int, str, list\\[int\\], or list\\[str\\]"):
        ops.CountValues("column", 3.14)

    with pytest.raises(ValueError, match="CountValues accepts int, str, list\\[int\\], or list\\[str\\]"):
        ops.CountValues("column", True)

    # Test empty list
    with pytest.raises(ValueError, match="CountValues requires at least one value"):
        ops.CountValues("column", [])

    # Test mixed type list
    with pytest.raises(ValueError, match="CountValues list must contain all integers or all strings"):
        ops.CountValues("column", [1, "two", 3])


def test_count_values_equality() -> None:
    # Single values
    op1 = ops.CountValues("col", 1)
    op2 = ops.CountValues("col", 1)
    op3 = ops.CountValues("col", 2)
    op4 = ops.CountValues("col", "1")  # String "1" vs int 1

    assert op1 == op2
    assert op1 != op3
    assert op1 != op4  # Different types

    # List values
    op5 = ops.CountValues("col", [1, 2])
    op6 = ops.CountValues("col", [1, 2])
    op7 = ops.CountValues("col", [2, 1])  # Different order

    assert op5 == op6
    assert op5 != op7  # Order matters

    # Single vs list
    op8 = ops.CountValues("col", 1)
    op9 = ops.CountValues("col", [1])
    assert op8 != op9  # Different formats


def test_count_values_hashing() -> None:
    op1 = ops.CountValues("col", ["test", "values"])
    op2 = ops.CountValues("col", ["test", "values"])
    op3 = ops.CountValues("col", ["different", "values"])

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)

    # Test deduplication in sets
    assert {op1, op2} == {op1}


def test_count_values_string_repr() -> None:
    op_single = ops.CountValues("user_id", 123)
    op_list = ops.CountValues("status", ["active", "pending", "completed"])

    assert str(op_single) == "count_values(user_id,123)"
    assert repr(op_single) == "count_values(user_id,123)"
    assert str(op_list) == "count_values(status,[active,pending,completed])"
    assert repr(op_list) == "count_values(status,[active,pending,completed])"


def test_count_values_special_characters() -> None:
    # Test backslashes (Windows paths)
    op_path = ops.CountValues("path", "C:\\Users\\test")
    assert op_path.name == "count_values(path,C:\\Users\\test)"

    # Test quotes in strings
    op_quote = ops.CountValues("name", "O'Brien")
    assert op_quote.name == "count_values(name,O'Brien)"

    # Test Unicode
    op_unicode = ops.CountValues("name", "José")
    assert op_unicode.name == "count_values(name,José)"

    # Test mixed quotes
    op_mixed = ops.CountValues("text", 'He said "Hello"')
    assert op_mixed.name == 'count_values(text,He said "Hello")'
```

#### Task 1.3: Run tests and fix any issues

Execute:
```bash
uv run pytest tests/test_ops.py::test_count_values_single -v
uv run pytest tests/test_ops.py::test_count_values_multiple -v
uv run pytest tests/test_ops.py::test_count_values_invalid_types -v
uv run pytest tests/test_ops.py::test_count_values_equality -v
uv run pytest tests/test_ops.py::test_count_values_hashing -v
uv run pytest tests/test_ops.py::test_count_values_string_repr -v
uv run pytest tests/test_ops.py::test_count_values_special_characters -v
```

### Task Group 2: SQL Dialect Implementation

**Objective**: Add SQL translation for CountValues in both DuckDB and BigQuery dialects with proper string escaping.

#### Task 2.1: Add DuckDB translation

In `src/dqx/dialect.py`, add to `DuckDBDialect.translate_sql_op`:

```python
case ops.CountValues(column=col, values=vals):
    # Internal _values is always a list
    if isinstance(vals, list):
        value_list = vals
    else:
        value_list = [vals]

    # Escape string values
    escaped_values = []
    for val in value_list:
        if isinstance(val, str):
            # Escape backslashes first, then quotes
            escaped = val.replace("\\", "\\\\").replace("'", "''")
            escaped_values.append(f"'{escaped}'")
        else:
            escaped_values.append(str(val))

    # Generate SQL based on single vs multiple values
    if len(escaped_values) == 1:
        condition = f"{col} = {escaped_values[0]}"
    else:
        values_clause = ", ".join(escaped_values)
        condition = f"{col} IN ({values_clause})"

    return f"CAST(COUNT_IF({condition}) AS DOUBLE) AS '{op.sql_col}'"
```

#### Task 2.2: Add BigQuery translation

In `src/dqx/dialect.py`, add to `BigQueryDialect.translate_sql_op`:

```python
case ops.CountValues(column=col, values=vals):
    # Internal _values is always a list
    if isinstance(vals, list):
        value_list = vals
    else:
        value_list = [vals]

    # Escape string values
    escaped_values = []
    for val in value_list:
        if isinstance(val, str):
            # Escape backslashes first, then quotes
            escaped = val.replace("\\", "\\\\").replace("'", "''")
            escaped_values.append(f"'{escaped}'")
        else:
            escaped_values.append(str(val))

    # Generate SQL based on single vs multiple values
    if len(escaped_values) == 1:
        condition = f"{col} = {escaped_values[0]}"
    else:
        values_clause = ", ".join(escaped_values)
        condition = f"{col} IN ({values_clause})"

    return f"CAST(COUNTIF({condition}) AS FLOAT64) AS `{op.sql_col}`"
```

#### Task 2.3: Write dialect tests

Add to `tests/test_dialect.py`:

```python
def test_translate_count_values() -> None:
    from dqx import ops
    from dqx.dialect import DuckDBDialect, BigQueryDialect

    # Test DuckDB with single integer value
    dialect_duck = DuckDBDialect()
    op_int = ops.CountValues("status", 1)
    sql_int = dialect_duck.translate_sql_op(op_int)
    assert sql_int == f"CAST(COUNT_IF(status = 1) AS DOUBLE) AS '{op_int.sql_col}'"

    # Test DuckDB with single string value
    op_str = ops.CountValues("category", "active")
    sql_str = dialect_duck.translate_sql_op(op_str)
    assert sql_str == f"CAST(COUNT_IF(category = 'active') AS DOUBLE) AS '{op_str.sql_col}'"

    # Test DuckDB with multiple values
    op_multi = ops.CountValues("status", ["active", "pending"])
    sql_multi = dialect_duck.translate_sql_op(op_multi)
    assert sql_multi == f"CAST(COUNT_IF(status IN ('active', 'pending')) AS DOUBLE) AS '{op_multi.sql_col}'"

    # Test DuckDB with special characters
    op_backslash = ops.CountValues("path", "C:\\Users\\test")
    sql_backslash = dialect_duck.translate_sql_op(op_backslash)
    assert sql_backslash == f"CAST(COUNT_IF(path = 'C:\\\\Users\\\\test') AS DOUBLE) AS '{op_backslash.sql_col}'"

    op_quote = ops.CountValues("name", "O'Brien")
    sql_quote = dialect_duck.translate_sql_op(op_quote)
    assert sql_quote == f"CAST(COUNT_IF(name = 'O''Brien') AS DOUBLE) AS '{op_quote.sql_col}'"

    # Test BigQuery
    dialect_bq = BigQueryDialect()
    sql_bq_single = dialect_bq.translate_sql_op(op_int)
    assert sql_bq_single == f"CAST(COUNTIF(status = 1) AS FLOAT64) AS `{op_int.sql_col}`"

    sql_bq_multi = dialect_bq.translate_sql_op(op_multi)
    assert sql_bq_multi == f"CAST(COUNTIF(status IN ('active', 'pending')) AS FLOAT64) AS `{op_multi.sql_col}`"
```

#### Task 2.4: Create integration test with DuckDB execution

Create `tests/test_count_values_integration.py`:

```python
"""Integration tests for CountValues operation."""

import duckdb
import pytest

from dqx import ops
from dqx.dialect import DuckDBDialect


def test_count_values_single_duckdb() -> None:
    """Test single value counting in DuckDB."""
    dialect = DuckDBDialect()
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

    # Test single integer value
    op_int = ops.CountValues("user_id", 1)
    sql_int = dialect.translate_sql_op(op_int)
    result_int = conn.execute(f"SELECT {sql_int} FROM test_data").fetchone()[0]
    assert result_int == 3.0  # user_id 1 appears 3 times

    # Test single string value
    op_str = ops.CountValues("status", "active")
    sql_str = dialect.translate_sql_op(op_str)
    result_str = conn.execute(f"SELECT {sql_str} FROM test_data").fetchone()[0]
    assert result_str == 3.0  # "active" appears 3 times

    conn.close()


def test_count_values_multiple_duckdb() -> None:
    """Test multiple value counting in DuckDB."""
    dialect = DuckDBDialect()
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE test_data AS
        SELECT * FROM (VALUES
            (1, 'active'),
            (2, 'pending'),
            (3, 'active'),
            (4, 'inactive'),
            (5, 'pending')
        ) AS t(id, status)
    """)

    # Test multiple string values
    op_multi = ops.CountValues("status", ["active", "pending"])
    sql_multi = dialect.translate_sql_op(op_multi)
    result_multi = conn.execute(f"SELECT {sql_multi} FROM test_data").fetchone()[0]
    assert result_multi == 4.0  # "active" + "pending" = 2 + 2 = 4

    # Test multiple integer values
    conn.execute("""
        CREATE TABLE test_numbers AS
        SELECT * FROM (VALUES (1), (2), (1), (3), (2), (4)) AS t(num)
    """)

    op_ints = ops.CountValues("num", [1, 2])
    sql_ints = dialect.translate_sql_op(op_ints)
    result_ints = conn.execute(f"SELECT {sql_ints} FROM test_numbers").fetchone()[0]
    assert result_ints == 4.0  # 1 appears 2 times, 2 appears 2 times

    conn.close()


def test_count_values_null_handling() -> None:
    """Test handling of NULL values - should not count them."""
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
    op = ops.CountValues("status", "value")
    sql = dialect.translate_sql_op(op)
    result = conn.execute(f"SELECT {sql} FROM test_nulls").fetchone()[0]
    assert result == 3.0  # Only non-NULL 'value' entries

    # Multiple values also shouldn't match NULL
    op_multi = ops.CountValues("status", ["value", "other"])
    sql_multi = dialect.translate_sql_op(op_multi)
    result_multi = conn.execute(f"SELECT {sql_multi} FROM test_nulls").fetchone()[0]
    assert result_multi == 3.0  # Still only counts 'value'

    conn.close()


def test_count_values_empty_string() -> None:
    """Test that empty strings are counted, not NULLs."""
    dialect = DuckDBDialect()
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE test_empty AS
        SELECT * FROM (VALUES
            (''),
            ('test'),
            (''),
            (NULL),
            ('')
        ) AS t(name)
    """)

    # Count empty strings
    op_empty = ops.CountValues("name", "")
    sql_empty = dialect.translate_sql_op(op_empty)
    result_empty = conn.execute(f"SELECT {sql_empty} FROM test_empty").fetchone()[0]
    assert result_empty == 3.0  # Three empty strings, not NULL

    # Count empty string in a list
    op_list = ops.CountValues("name", ["", "test"])
    sql_list = dialect.translate_sql_op(op_list)
    result_list = conn.execute(f"SELECT {sql_list} FROM test_empty").fetchone()[0]
    assert result_list == 4.0  # 3 empty + 1 "test"

    conn.close()


def test_count_values_special_characters() -> None:
    """Test edge cases with special characters."""
    dialect = DuckDBDialect()
    conn = duckdb.connect(":memory:")

    conn.execute(r"""
        CREATE TABLE test_special AS
        SELECT * FROM (VALUES
            ('C:\Users\test'),
            ('normal'),
            ('C:\Users\test'),
            ('O''Brien'),
            ('He said "Hello"'),
            ('José'),
            ('C:\Users\test')
        ) AS t(text)
    """)

    # Test backslashes
    op_backslash = ops.CountValues("text", r"C:\Users\test")
    sql_backslash = dialect.translate_sql_op(op_backslash)
    result_backslash = conn.execute(f"SELECT {sql_backslash} FROM test_special").fetchone()[0]
    assert result_backslash == 3.0

    # Test single quotes
    op_quote = ops.CountValues("text", "O'Brien")
    sql_quote = dialect.translate_sql_op(op_quote)
    result_quote = conn.execute(f"SELECT {sql_quote} FROM test_special").fetchone()[0]
    assert result_quote == 1.0

    # Test multiple special values
    op_multi = ops.CountValues("text", [r"C:\Users\test", "O'Brien"])
    sql_multi = dialect.translate_sql_op(op_multi)
    result_multi = conn.execute(f"SELECT {sql_multi} FROM test_special").fetchone()[0]
    assert result_multi == 4.0  # 3 + 1

    conn.close()
```

Run all dialect tests:
```bash
uv run pytest tests/test_dialect.py::test_translate_count_values -v
uv run pytest tests/test_count_values_integration.py -v
```

### Task Group 3: Spec and State Implementation

**Objective**: Implement the high-level metric specification with proper state handling.

#### Task 3.1: Update MetricType and create CountValues spec

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
    "CountValues",  # Add this
]
```

2. Add CountValues spec class:
```python
class CountValues:
    """Count occurrences of specific value(s) in a column.

    Accepts single values (int or str) or lists of values (list[int] or list[str]).
    Uses SimpleAdditiveState for mergeable results across partitions.

    Note: This operation counts only the specified values, never NULLs.
    Empty strings are counted as values, not as NULLs.
    """
    metric_type: MetricType = "CountValues"

    def __init__(self, column: str, values: int | str | list[int] | list[str]) -> None:
        # Validation is done in the Op class
        self._column = column
        self._values = values
        self._analyzers = (ops.CountValues(self._column, self._values),)

    @property
    def name(self) -> str:
        # Delegate to Op for consistent naming
        return self._analyzers[0].name

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, "values": self._values}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def __hash__(self) -> int:
        # Convert lists to tuples for hashing
        hashable_values = self._values if not isinstance(self._values, list) else tuple(self._values)
        return hash((self.name, self._column, hashable_values))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CountValues):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name
```

#### Task 3.2: Write spec tests

Add to `tests/test_specs.py`:

```python
def test_count_values_spec_single() -> None:
    from dqx import specs

    # Test with single integer value
    spec_int = specs.CountValues("user_id", 1)
    assert spec_int.metric_type == "CountValues"
    assert spec_int.name == "count_values(user_id,1)"
    assert spec_int.parameters == {"column": "user_id", "values": 1}
    assert len(spec_int.analyzers) == 1

    # Test with single string value
    spec_str = specs.CountValues("status", "active")
    assert spec_str.name == "count_values(status,active)"
    assert spec_str.parameters == {"column": "status", "values": "active"}


def test_count_values_spec_multiple() -> None:
    from dqx import specs

    # Test with multiple integer values
    spec_ints = specs.CountValues("type_id", [1, 2, 3])
    assert spec_ints.name == "count_values(type_id,[1,2,3])"
    assert spec_ints.parameters == {"column": "type_id", "values": [1, 2, 3]}

    # Test with multiple string values
    spec_strs = specs.CountValues("status", ["active", "pending"])
    assert spec_strs.name == "count_values(status,[active,pending])"
    assert spec_strs.parameters == {"column": "status", "values": ["active", "pending"]}


def test_count_values_spec_invalid() -> None:
    from dqx import specs

    # Invalid types are caught by the Op class
    with pytest.raises(ValueError, match="CountValues accepts"):
        specs.CountValues("col", 3.14)

    with pytest.raises(ValueError, match="CountValues requires at least one value"):
        specs.CountValues("col", [])

    with pytest.raises(ValueError, match="CountValues list must contain all integers or all strings"):
        specs.CountValues("col", [1, "two"])


def test_count_values_spec_equality() -> None:
    from dqx import specs

    # Single values
    spec1 = specs.CountValues("col", "test")
    spec2 = specs.CountValues("col", "test")
    spec3 = specs.CountValues("col", "other")

    assert spec1 == spec2
    assert spec1 != spec3
    assert hash(spec1) == hash(spec2)
    assert hash(spec1) != hash(spec3)

    # List values
    spec4 = specs.CountValues("col", [1, 2])
    spec5 = specs.CountValues("col", [1, 2])
    spec6 = specs.CountValues("col", [2, 1])

    assert spec4 == spec5
    assert spec4 != spec6  # Order matters
    assert hash(spec4) == hash(spec5)

    # Test string representation
    assert str(spec1) == "count_values(col,test)"
    assert str(spec4) == "count_values(col,[1,2])"


def test_count_values_state() -> None:
    from dqx import specs, ops, states

    spec = specs.CountValues("status", ["active", "pending"])

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
uv run pytest tests/test_specs.py::test_count_values_spec_single -v
uv run pytest tests/test_specs.py::test_count_values_spec_multiple -v
uv run pytest tests/test_specs.py::test_count_values_spec_invalid -v
uv run pytest tests/test_specs.py::test_count_values_spec_equality -v
uv run pytest tests/test_specs.py::test_count_values_state -v
```

### Task Group 4: API Integration and Documentation

**Objective**: Add user-facing API method with type overloads and comprehensive examples.

#### Task 4.1: Add count_values method to provider.py with type overloads

In `src/dqx/provider.py`, first add the import and then the overloaded method:

```python
from typing import overload

class MetricProvider(SymbolicMetricBase):
    # ... existing code ...

    @overload
    def count_values(
        self, column: str, value: int,
        key: ResultKeyProvider = ResultKeyProvider(),
        dataset: str | None = None
    ) -> sp.Symbol:
        """Count occurrences of a specific integer value in a column.

        Args:
            column: Column name to count values in
            value: The integer value to count
            key: Result key provider
            dataset: Optional dataset name

        Returns:
            Symbol representing the count

        Example:
            >>> suite.count_values("user_id", 123)
            >>> suite.count_values("type_id", 1)
        """
        ...

    @overload
    def count_values(
        self, column: str, value: str,
        key: ResultKeyProvider = ResultKeyProvider(),
        dataset: str | None = None
    ) -> sp.Symbol:
        """Count occurrences of a specific string value in a column.

        Args:
            column: Column name to count values in
            value: The string value to count
            key: Result key provider
            dataset: Optional dataset name

        Returns:
            Symbol representing the count

        Example:
            >>> suite.count_values("status", "active")
            >>> suite.count_values("category", "electronics")
        """
        ...

    @overload
    def count_values(
        self, column: str, values: list[int],
        key: ResultKeyProvider = ResultKeyProvider(),
        dataset: str | None = None
    ) -> sp.Symbol:
        """Count occurrences of multiple integer values in a column.

        Args:
            column: Column name to count values in
            values: List of integer values to count
            key: Result key provider
            dataset: Optional dataset name

        Returns:
            Symbol representing the count

        Example:
            >>> suite.count_values("user_id", [123, 456, 789])
            >>> suite.count_values("type_id", [1, 2, 3])
        """
        ...

    @overload
    def count_values(
        self, column: str, values: list[str],
        key: ResultKeyProvider = ResultKeyProvider(),
        dataset: str | None = None
    ) -> sp.Symbol:
        """Count occurrences of multiple string values in a column.

        Args:
            column: Column name to count values in
            values: List of string values to count
            key: Result key provider
            dataset: Optional dataset name

        Returns:
            Symbol representing the count

        Example:
            >>> suite.count_values("status", ["active", "pending"])
            >>> suite.count_values("category", ["electronics", "books", "clothing"])
        """
        ...

    def count_values(
        self, column: str, values: int | str | list[int] | list[str],
        key: ResultKeyProvider = ResultKeyProvider(),
        dataset: str | None = None
    ) -> sp.Symbol:
        """Count occurrences of specific value(s) in a column.

        This operation counts only the specified values, never NULLs.
        Empty strings are counted as values, not as NULLs.

        Args:
            column: Column name to count values in
            values: Value(s) to count - single int/str or list of int/str
            key: Result key provider
            dataset: Optional dataset name

        Returns:
            Symbol representing the count

        Examples:
            >>> from dqx import ValidationSuite
            >>> suite = ValidationSuite("test")

            >>> # Count single value
            >>> suite.count_values("status", "active")

            >>> # Count multiple values efficiently in one query
            >>> suite.count_values("status", ["active", "pending"])

            >>> # Count integer values
            >>> suite.count_values("type_id", [1, 2, 3])

        Performance Note:
            Counting multiple values with a list is more efficient than
            making multiple separate count_values calls.
        """
        return self.metric(specs.CountValues(column, values), key, dataset)
```

#### Task 4.2: Write provider tests

Add to `tests/test_provider.py`:

```python
def test_provider_count_values_single() -> None:
    from dqx import ValidationSuite

    suite = ValidationSuite("test_suite")

    # Test with single integer value
    metric_int = suite.count_values("user_id", 1)
    assert metric_int is not None

    # Test with single string value
    metric_str = suite.count_values("status", "active")
    assert metric_str is not None


def test_provider_count_values_multiple() -> None:
    from dqx import ValidationSuite

    suite = ValidationSuite("test_suite")

    # Test with multiple integer values
    metric_ints = suite.count_values("user_id", [1, 2, 3])
    assert metric_ints is not None

    # Test with multiple string values
    metric_strs = suite.count_values("status", ["active", "pending"])
    assert metric_strs is not None

    # Test with custom key
    from dqx.result_key import ResultKeyProvider
    key = ResultKeyProvider(key="custom")
    metric_custom = suite.count_values("category", ["electronics", "books"], key=key)
    assert metric_custom is not None
```

#### Task 4.3: Create comprehensive example

Create `examples/count_values_demo.py`:

```python
"""Demonstration of CountValues operation for single and multiple values."""

import duckdb
from dqx import ValidationSuite, OnData


def main() -> None:
    """Demonstrate CountValues usage with various scenarios."""

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

    # Count single values
    active_users = suite.count_values("status", "active")
    premium_accounts = suite.count_values("account_type", "premium")

    # Count multiple values efficiently
    active_or_pending = suite.count_values("status", ["active", "pending"])
    all_account_types = suite.count_values("account_type", ["premium", "basic"])

    # Count integer values
    zero_points = suite.count_values("points", 0)
    low_points = suite.count_values("points", [0, 50, 75])

    # Run validation
    result = OnData(conn, "user_activity").run(suite)

    # Display results
    print("CountValues Demo Results")
    print("=" * 50)
    print("\nSingle Value Counts:")
    print(f"  Active users: {result.metrics[active_users]}")
    print(f"  Premium accounts: {result.metrics[premium_accounts]}")
    print(f"  Users with 0 points: {result.metrics[zero_points]}")

    print("\nMultiple Value Counts (more efficient):")
    print(f"  Active or pending users: {result.metrics[active_or_pending]}")
    print(f"  All account types: {result.metrics[all_account_types]}")
    print(f"  Users with low points (0, 50, or 75): {result.metrics[low_points]}")

    # Demonstrate with different data types including special characters
    conn.execute(r"""
        CREATE TABLE file_system AS
        SELECT * FROM (VALUES
            ('C:\Users\admin\documents', 'folder', 'admin'),
            ('C:\Users\john\downloads', 'folder', 'john'),
            ('report.pdf', 'file', 'admin'),
            ('C:\Users\admin\pictures', 'folder', 'admin'),
            ("O'Brien's report.docx", 'file', 'O''Brien'),
            ('backup.zip', 'file', 'admin')
        ) AS t(path, type, owner)
    """)

    suite2 = ValidationSuite("file_metrics")

    # Count paths with backslashes
    admin_folders = suite2.count_values("path", [
        r"C:\Users\admin\documents",
        r"C:\Users\admin\pictures"
    ])

    # Count by type
    folders = suite2.count_values("type", "folder")

    # Count special owner names
    special_owners = suite2.count_values("owner", ["admin", "O'Brien"])

    # Run validation on file system data
    result2 = OnData(conn, "file_system").run(suite2)

    print("\n\nFile System Metrics (Special Characters)")
    print("=" * 50)
    print(f"Admin folders: {result2.metrics[admin_folders]}")
    print(f"Total folders: {result2.metrics[folders]}")
    print(f"Files owned by admin or O'Brien: {result2.metrics[special_owners]}")

    # Demonstrate NULL and empty string handling
    conn.execute("""
        CREATE TABLE test_nulls AS
        SELECT * FROM (VALUES
            ('value'),
            (''),
            (NULL),
            (''),
            ('value'),
            ('')
        ) AS t(data)
    """)

    suite3 = ValidationSuite("null_metrics")

    # Count empty strings (not NULLs)
    empty_strings = suite3.count_values("data", "")
    values = suite3.count_values("data", "value")
    empty_or_value = suite3.count_values("data", ["", "value"])

    result3 = OnData(conn, "test_nulls").run(suite3)

    print("\n\nNULL vs Empty String Handling")
    print("=" * 50)
    print(f"Empty strings (not NULLs): {result3.metrics[empty_strings]}")
    print(f"'value' occurrences: {result3.metrics[values]}")
    print(f"Empty strings OR 'value': {result3.metrics[empty_or_value]}")
    print("Note: NULLs are never counted")

    conn.close()


if __name__ == "__main__":
    main()
```

Run the example:
```bash
uv run python examples/count_values_demo.py
```

### Task Group 5: Final Verification

**Objective**: Ensure all tests pass, code quality checks succeed, and implementation is complete.

#### Task 5.1: Run all tests

Execute comprehensive test suite:
```bash
# Run all new CountValues tests
uv run pytest tests/test_ops.py -k count_values -v
uv run pytest tests/test_dialect.py::test_translate_count_values -v
uv run pytest tests/test_count_values_integration.py -v
uv run pytest tests/test_specs.py -k count_values -v
uv run pytest tests/test_provider.py -k count_values -v

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

1. Add CountValues to the main README.md if needed
2. Update any API documentation
3. Ensure all docstrings are complete and accurate

## Testing Strategy

### Unit Tests
- Type validation (int, str, list[int], list[str] accepted; others rejected)
- Empty list rejection
- Mixed type list rejection
- Single vs multiple value handling
- Equality and hashing behavior
- String representation
- SQL properties
- Value assignment and clearing

### Integration Tests
- SQL generation for DuckDB and BigQuery
- Single value with `=` operator
- Multiple values with `IN` operator
- Actual SQL execution with test data
- NULL handling (never counted)
- Empty string handling (counted as value)
- Edge cases with special characters:
  - Backslashes (Windows paths)
  - Single quotes
  - Unicode characters
- String escaping for SQL injection prevention

### End-to-End Tests
- Full workflow from ValidationSuite to results
- State serialization/deserialization
- Multiple value types in same suite
- Performance comparison single vs batch

## Key Implementation Notes

1. **Type Safety**: Strict validation at Op level, supports int, str, list[int], list[str]
2. **SQL Injection Prevention**: Double escaping for backslashes and quotes
3. **Performance**: Single query for multiple values using SQL IN operator
4. **State Management**: Reuse existing SimpleAdditiveState for efficiency
5. **NULL Behavior**: Explicitly documented that NULLs are never counted
6. **Empty Strings**: Counted as values, not NULLs
7. **API Flexibility**: Four overloads for optimal type hints and IDE support

## Success Criteria

- [ ] All tests pass with 100% coverage for new code
- [ ] Pre-commit hooks pass (mypy, ruff)
- [ ] SQL executes correctly on both DuckDB and BigQuery
- [ ] Type hints work correctly in IDEs
- [ ] Examples demonstrate real-world usage
- [ ] Performance benefit of batch counting documented

## Key Differences from v1

1. **Multiple Values**: Support for counting multiple values in one operation
2. **Enhanced Type System**: Four overloads instead of two for better type safety
3. **Improved SQL**: Uses IN operator for multiple values
4. **Better Escaping**: Handles backslashes in addition to quotes
5. **Clearer Documentation**: Explicit about NULL and empty string behavior

## References

- CountValue v1 plan: `docs/plans/count_value_op_implementation_plan_v1.md`
- DuplicateCount implementation: `docs/plans/duplicate_count_op_implementation_plan_v2.md`
- Ops module: `src/dqx/ops.py`
- Dialect module: `src/dqx/dialect.py`
- Specs module: `src/dqx/specs.py`
- Provider module: `src/dqx/provider.py`
