# UniqueCount Implementation Plan v1

## Overview

This plan describes the implementation of a new `UniqueCount` operation for DQX that counts the number of distinct/unique values in a column (cardinality). This operation is equivalent to SQL's `COUNT(DISTINCT column)`.

## Background

- **Purpose**: Count unique/distinct values in a categorical column
- **SQL Equivalent**: `COUNT(DISTINCT column)`
- **Example Use Case**: Counting unique categories, unique user IDs, unique products, etc.
- **Null Handling**: NULLs are automatically excluded (standard SQL behavior)
- **Mergeability**: UniqueCount is NOT mergeable across partitions (similar to DuplicateCount)

## Implementation Details

### Task Group 1: Core Operation Implementation (TDD)

#### Task 1.1: Write Tests for UniqueCount Operation
Create comprehensive tests for the new `UniqueCount` operation class.

**File**: `tests/test_ops.py`

Add the following test:

```python
def test_unique_count() -> None:
    """Test UniqueCount operation basic functionality."""
    op = ops.UniqueCount("product_id")
    assert op.name == "unique_count(product_id)"
    assert op.column == "product_id"
    assert op.prefix is not None
    assert op.sql_col == f"{op.prefix}_unique_count(product_id)"

    # Test value assignment
    with pytest.raises(DQXError, match="UniqueCount op has not been collected yet!"):
        op.value()

    op.assign(42.0)
    assert op.value() == pytest.approx(42.0)

    # Test clear
    op.clear()
    with pytest.raises(DQXError):
        op.value()

    # Test equality
    op2 = ops.UniqueCount("product_id")
    op3 = ops.UniqueCount("user_id")
    assert op == op2
    assert op != op3
    assert op != ops.Average("product_id")

    # Test hash
    assert hash(op) == hash(op2)
    assert hash(op) != hash(op3)

    # Test string representation
    assert str(op) == "unique_count(product_id)"
    assert repr(op) == "unique_count(product_id)"
```

Also update existing protocol tests:
- In `test_op_protocol()`: Add `assert isinstance(ops.UniqueCount("col"), ops.Op)`
- In `test_sql_op_protocol()`: Add `assert isinstance(ops.UniqueCount("col"), ops.SqlOp)`
- In `test_sql_op_properties()`: Add `ops.UniqueCount("col")` to the list
- In `test_op_value_assignment_and_clearing()`: Add `ops.UniqueCount("col")` to the list

#### Task 1.2: Implement UniqueCount Operation Class
Create the `UniqueCount` operation following the established pattern.

**File**: `src/dqx/ops.py`

Add the following class (place it after `NegativeCount` class for logical grouping):

```python
class UniqueCount(OpValueMixin[float], SqlOp[float]):
    """Count distinct/unique values in a column.

    This operation counts the number of distinct non-null values
    in the specified column, equivalent to SQL's COUNT(DISTINCT column).

    Args:
        column: The column name to count distinct values in

    Example:
        >>> op = UniqueCount("category")
        >>> # After collection, op.value() returns the count of unique categories
    """

    __match_args__ = ("column",)

    def __init__(self, column: str) -> None:
        OpValueMixin.__init__(self)
        self.column = column
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"unique_count({self.column})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        """SQL column alias for query results."""
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UniqueCount) and self.column == other.column

    def __hash__(self) -> int:
        return hash((type(self), self.column))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
```

#### Task 1.3: Run Tests and Fix Issues
```bash
uv run pytest tests/test_ops.py::test_unique_count -v
uv run mypy src/dqx/ops.py
uv run ruff check --fix src/dqx/ops.py
```

### Task Group 2: Generic NonMergeable State Implementation

#### Task 2.1: Write Tests for NonMergeable State
Create tests for the generic NonMergeable state.

**File**: `tests/test_states.py`

Add the following tests:

```python
def test_nonmergeable_state() -> None:
    """Test generic NonMergeable state behavior."""
    from dqx import states
    from dqx.common import DQXError

    # Test with UniqueCount metric type
    state1 = states.NonMergeable(value=5.0, metric_type="UniqueCount")
    state2 = states.NonMergeable(value=3.0, metric_type="UniqueCount")

    # Test that merge raises error
    with pytest.raises(DQXError, match="UniqueCount state cannot be merged"):
        state1.merge(state2)

    # Test that identity raises error
    with pytest.raises(DQXError, match="NonMergeable state does not support identity"):
        states.NonMergeable.identity()

    # Test value property
    assert state1.value == 5.0
    assert state2.value == 3.0

    # Test serialization
    serialized = state1.serialize()
    deserialized = states.NonMergeable.deserialize(serialized)
    assert deserialized.value == state1.value
    assert deserialized == state1

    # Test with DuplicateCount metric type
    state3 = states.NonMergeable(value=10.0, metric_type="DuplicateCount")
    with pytest.raises(DQXError, match="DuplicateCount state cannot be merged"):
        state3.merge(state3)
```

#### Task 2.2: Implement Generic NonMergeable State Class
Create the generic non-mergeable state class.

**File**: `src/dqx/states.py`

Add the following class (place it after `Maximum` class, before `DuplicateCount`):

```python
class NonMergeable(State):
    """Generic non-mergeable state for metrics that cannot be merged across partitions.

    This state is used for metrics where merging results from different partitions
    would produce incorrect results (e.g., duplicate counts, unique counts).

    Args:
        value: The metric value
        metric_type: The type of metric (e.g., "DuplicateCount", "UniqueCount")
                    used to customize error messages
    """

    def __init__(self, value: float, metric_type: str) -> None:
        self._value = float(value)
        self._metric_type = metric_type

    @classmethod
    def identity(cls) -> NonMergeable:
        raise DQXError(
            "NonMergeable state does not support identity. "
            "Metrics using this state must be computed on the entire dataset in a single pass "
            "because results from different partitions cannot be accurately merged."
        )

    @property
    def value(self) -> float:
        return self._value

    def serialize(self) -> bytes:
        return msgpack.packb((self._value, self._metric_type))

    @classmethod
    def deserialize(cls, data: bytes) -> NonMergeable:
        value, metric_type = msgpack.unpackb(data)
        return cls(value=value, metric_type=metric_type)

    def merge(self, other: NonMergeable) -> NonMergeable:
        raise DQXError(
            f"{self._metric_type} state cannot be merged across partitions. "
            f"The {self._metric_type} metric must be computed on the entire dataset in a single pass "
            "to ensure accurate results."
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NonMergeable):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> NonMergeable:
        return NonMergeable(value=self._value, metric_type=self._metric_type)
```

#### Task 2.3: Migrate DuplicateCount to Use NonMergeable
Replace the existing `DuplicateCount` state class with a deprecation alias.

**File**: `src/dqx/states.py`

Replace the existing `DuplicateCount` class with:

```python
# For backward compatibility - will be removed in a future version
DuplicateCount = NonMergeable
```

Update the `DuplicateCount` spec in `src/dqx/specs.py`:

```python
# In the DuplicateCount spec class, update these methods:
def state(self) -> states.NonMergeable:
    return states.NonMergeable(value=self._analyzers[0].value(), metric_type="DuplicateCount")

@classmethod
def deserialize(cls, state: bytes) -> states.State:
    return states.NonMergeable.deserialize(state)
```

#### Task 2.4: Run State Tests
```bash
uv run pytest tests/test_states.py::test_nonmergeable_state -v
uv run pytest tests/test_states.py -v  # Ensure existing tests still pass
uv run mypy src/dqx/states.py
uv run ruff check --fix src/dqx/states.py
```

### Task Group 3: Spec and Provider Implementation

#### Task 3.1: Write Tests for UniqueCount Spec
Update spec tests to include UniqueCount.

**File**: `tests/test_specs.py`

Update the existing test to verify `"UniqueCount"` is in MetricType and registry:
- Add `"UniqueCount"` to the expected list in the relevant test

#### Task 3.2: Implement UniqueCount Spec
Add the spec implementation.

**File**: `src/dqx/specs.py`

1. Add `"UniqueCount"` to the `MetricType` literal (in alphabetical order)

2. Add the following class (place it after `CountValues` class):

```python
class UniqueCount:
    """Specification for counting distinct values in a column.

    Counts the number of unique non-null values in the specified column.
    Uses the non-mergeable UniqueCount state because unique counts cannot
    be merged across partitions (same values may appear in multiple partitions).

    Args:
        column: The column to count distinct values in

    Example:
        >>> spec = UniqueCount("user_id")
        >>> # Use with MetricProvider to count unique users
    """

    metric_type: MetricType = "UniqueCount"
    is_extended: bool = False

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.UniqueCount(self._column),)

    @property
    def name(self) -> str:
        return f"unique_count({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.NonMergeable:
        return states.NonMergeable(value=self._analyzers[0].value(), metric_type="UniqueCount")

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.NonMergeable.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UniqueCount):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name
```

#### Task 3.3: Add unique_count Method to Provider
Implement the provider method.

**File**: `src/dqx/provider.py`

Add the following method (place it after `count_values` method):

```python
def unique_count(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    """Count distinct/unique values in a column.

    This counts the number of distinct non-null values in the specified column,
    equivalent to SQL's COUNT(DISTINCT column).

    Args:
        column: Column name to count distinct values
        lag: Lag offset in days (default: 0)
        dataset: Optional dataset name

    Returns:
        Symbol representing the distinct count

    Examples:
        >>> # Count unique categories
        >>> ctx.assert_that(mp.unique_count("category")).where(
        ...     name="Should have 5 unique categories"
        ... ).is_eq(5)

        >>> # Count unique users
        >>> ctx.assert_that(mp.unique_count("user_id")).where(
        ...     name="Should have at least 100 unique users"
        ... ).is_gte(100)
    """
    return self.metric(specs.UniqueCount(column), lag, dataset)
```

#### Task 3.4: Run Tests
```bash
uv run pytest tests/test_specs.py -v
uv run mypy src/dqx/specs.py src/dqx/provider.py
uv run ruff check --fix src/dqx/
```

### Task Group 4: SQL Dialect Implementation

#### Task 4.1: Write Tests for SQL Translation
Add dialect translation tests.

**File**: `tests/test_dialect.py`

Add the following test:

```python
def test_translate_unique_count() -> None:
    """Test UniqueCount translation for both dialects."""
    from dqx import ops

    dialect_duck = DuckDBDialect()
    dialect_bq = BigQueryDialect()

    op = ops.UniqueCount("user_id")

    # Test DuckDB translation
    sql_duck = dialect_duck.translate_sql_op(op)
    assert sql_duck == f"CAST(COUNT(DISTINCT user_id) AS DOUBLE) AS '{op.sql_col}'"

    # Test BigQuery translation
    sql_bq = dialect_bq.translate_sql_op(op)
    assert sql_bq == f"CAST(COUNT(DISTINCT user_id) AS FLOAT64) AS `{op.sql_col}`"
```

Also add `ops.UniqueCount("col")` to the batch optimization test lists.

#### Task 4.2: Implement SQL Translation
Add SQL translation for both dialects.

**File**: `src/dqx/dialect.py`

1. In `DuckDBDialect.translate_sql_op` method, add the following case (after `NegativeCount`):

```python
case ops.UniqueCount(column=col):
    return f"CAST(COUNT(DISTINCT {col}) AS DOUBLE) AS '{op.sql_col}'"
```

2. In `BigQueryDialect.translate_sql_op` method, add the following case (after `NegativeCount`):

```python
case ops.UniqueCount(column=col):
    return f"CAST(COUNT(DISTINCT {col}) AS FLOAT64) AS `{op.sql_col}`"
```

#### Task 4.3: Run Dialect Tests
```bash
uv run pytest tests/test_dialect.py::test_translate_unique_count -v
uv run pytest tests/test_dialect_batch_optimization.py -v
uv run mypy src/dqx/dialect.py
```

### Task Group 5: API Integration Tests

#### Task 5.1: Create Comprehensive API Tests
Create API-level integration tests.

**File**: `tests/test_api_unique_count.py`

```python
"""API-level tests for UniqueCount functionality.

This module tests the UniqueCount feature through the high-level API,
ensuring it correctly counts distinct values in columns.
"""

import datetime

import pyarrow as pa

from dqx import specs
from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_unique_count_basic() -> None:
    """Test UniqueCount through the check decorator API."""
    data = pa.table(
        {
            "category": ["A", "B", "A", "C", "B", "A", "D"],
            "user_id": [1, 2, 1, 3, 2, 4, 5],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Unique Count Check")
    def unique_count_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("category")).where(
            name="Number of unique categories should be 4"
        ).is_eq(4)

        ctx.assert_that(mp.unique_count("user_id")).where(
            name="Number of unique users should be 5"
        ).is_eq(5)

    db = InMemoryMetricDB()
    suite = VerificationSuite([unique_count_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([ds], key)


def test_unique_count_with_nulls() -> None:
    """Test UniqueCount behavior with null values."""
    data = pa.table(
        {
            "product": ["A", None, "B", "A", None, "C", "B"],
            "score": [10, 20, None, 10, 30, None, 20],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Null Handling Check")
    def null_handling_check(mp: MetricProvider, ctx: Context) -> None:
        # COUNT(DISTINCT) excludes nulls
        ctx.assert_that(mp.unique_count("product")).where(
            name="Unique products should be 3 (nulls excluded)"
        ).is_eq(3)

        ctx.assert_that(mp.unique_count("score")).where(
            name="Unique scores should be 3 (nulls excluded)"
        ).is_eq(3)

    db = InMemoryMetricDB()
    suite = VerificationSuite([null_handling_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([ds], key)


def test_unique_count_edge_cases() -> None:
    """Test UniqueCount with edge cases."""
    # Empty column
    empty_data = pa.table({"col": pa.array([], type=pa.string())})

    # All nulls
    null_data = pa.table({"col": [None, None, None]})

    # All same value
    same_data = pa.table({"col": ["X", "X", "X", "X"]})

    @check(name="Empty Data Check")
    def empty_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("col")).where(
            name="Empty column should have 0 unique values"
        ).is_eq(0)

    @check(name="All Nulls Check")
    def null_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("col")).where(
            name="All nulls should have 0 unique values"
        ).is_eq(0)

    @check(name="Same Value Check")
    def same_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("col")).where(
            name="All same value should have 1 unique value"
        ).is_eq(1)

    db = InMemoryMetricDB()
    suite = VerificationSuite([empty_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([DuckRelationDataSource.from_arrow(empty_data, "empty")], key)

    suite = VerificationSuite([null_check], db, "Test Suite")
    suite.run([DuckRelationDataSource.from_arrow(null_data, "nulls")], key)

    suite = VerificationSuite([same_check], db, "Test Suite")
    suite.run([DuckRelationDataSource.from_arrow(same_data, "same")], key)


def test_unique_count_with_spec_directly() -> None:
    """Test using UniqueCount spec directly."""
    data = pa.table({"region": ["US", "EU", "US", "APAC", "EU", "US"]})
    ds = DuckRelationDataSource.from_arrow(data, "data")

    unique_count_spec = specs.UniqueCount("region")

    @check(name="Spec Direct Check")
    def spec_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.metric(unique_count_spec)).where(
            name="Unique regions should be 3"
        ).is_eq(3)

    db = InMemoryMetricDB()
    suite = VerificationSuite([spec_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([ds], key)


def test_unique_count_various_types() -> None:
    """Test UniqueCount with different data types."""
    data = pa.table(
        {
            "strings": ["apple", "banana", "apple", "cherry", "banana"],
            "integers": [100, 200, 100, 300, 200],
            "floats": [1.5, 2.5, 1.5, 3.5, 2.5],
            "booleans": [True, False, True, False, True],
        }
    )

    ds = DuckRelationDataSource.from_arrow(data, "data")

    @check(name="Type Variety Check")
    def type_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.unique_count("strings")).where(
            name="Unique strings"
        ).is_eq(3)

        ctx.assert_that(mp.unique_count("integers")).where(
            name="Unique integers"
        ).is_eq(3)

        ctx.assert_that(mp.unique_count("floats")).where(
            name="Unique floats"
        ).is_eq(3)

        ctx.assert_that(mp.unique_count("booleans")).where(
            name="Unique booleans"
        ).is_eq(2)

    db = InMemoryMetricDB()
    suite = VerificationSuite([type_check], db, "Test Suite")
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
    suite.run([ds], key)


def test_unique_count_symbol_info() -> None:
    """Test that UniqueCount symbols have correct metadata."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    symbol = provider.unique_count("customer_id")
    symbol_info = provider.get_symbol(symbol)

    assert symbol_info.name == "unique_count(customer_id)"
    assert isinstance(symbol_info.metric_spec, specs.UniqueCount)
    assert symbol_info.metric_spec.parameters["column"] == "customer_id"
```

#### Task 5.2: Run API Tests
```bash
uv run pytest tests/test_api_unique_count.py -v
```

### Task Group 6: Final Verification

#### Task 6.1: Run All Related Tests
```bash
# Run all ops tests
uv run pytest tests/test_ops.py -v

# Run spec tests
uv run pytest tests/test_specs.py -v

# Run dialect tests
uv run pytest tests/test_dialect.py -v

# Run API tests
uv run pytest tests/test_api_unique_count.py -v

# Run batch optimization tests
uv run pytest tests/test_dialect_batch_optimization.py -v
```

#### Task 6.2: Run Type Checking and Linting
```bash
# Type checking
uv run mypy src/dqx/

# Linting and formatting
uv run ruff check --fix src/
uv run ruff check --fix tests/

# Run pre-commit hooks
uv run hooks
```

#### Task 6.3: Run Full Test Suite
```bash
# Run all tests to ensure nothing is broken
uv run pytest tests/ -v

# Check code coverage
uv run coverage tests/
```

## Summary

This plan implements the `UniqueCount` operation following TDD principles:

1. **Operation**: Counts distinct values using `COUNT(DISTINCT column)`
2. **State**: Uses generic `NonMergeable` state (shared with `DuplicateCount`)
   - Introduces a generic NonMergeable state class for all non-mergeable metrics
   - Migrates existing DuplicateCount to use the generic state
   - Raises errors on `merge()` and `identity()` with metric-specific messages
   - Ensures accurate counting by requiring single-pass computation
3. **SQL Support**: Works with both DuckDB and BigQuery dialects
4. **Testing**: Comprehensive unit and integration tests, including state merge error handling
5. **API**: Simple method `mp.unique_count("column")` for easy use

The implementation follows all existing patterns in the codebase and introduces a reusable NonMergeable state pattern for future non-mergeable metrics.
