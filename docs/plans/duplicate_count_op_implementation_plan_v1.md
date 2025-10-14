# DuplicateCountOp Implementation Plan v1

## Overview

This document provides a comprehensive, step-by-step plan to implement a new operation called `DuplicateCountOp` in the DQX data quality framework. This operation counts duplicate rows in a dataset based on a specified list of columns.

### What is DuplicateCountOp?

DuplicateCountOp is a SQL-based operation that:
- Takes a list of column names as input
- Counts how many duplicate rows exist based on those columns
- Returns the count as a float value

Example: If you have a table with columns `[user_id, product_id, date]` and you want to know how many duplicate `(user_id, product_id)` combinations exist, DuplicateCountOp will calculate: `COUNT(*) - COUNT(DISTINCT (user_id, product_id))`.

### Architecture Context

DQX uses a layered architecture:
1. **Ops** (`ops.py`) - Define what metrics to collect
2. **Specs** (`specs.py`) - High-level metric specifications that create Ops
3. **States** (`states.py`) - Store and merge metric results
4. **Dialect** (`dialect.py`) - Translate Ops to SQL
5. **Provider** (`provider.py`) - User-facing API to create metrics

## Prerequisites

Before starting:
1. Ensure you have the development environment set up:
   ```bash
   ./bin/setup-dev-env.sh
   ```
2. Familiarize yourself with Test-Driven Development (TDD)
3. Run existing tests to ensure everything works:
   ```bash
   uv run pytest -v
   ```

## Task Groups

The implementation is organized into 4 sequential task groups that can be implemented and committed independently:

### Task Group 1: Core Op Implementation (Tasks 1-4)
**Focus**: Implement the fundamental DuplicateCount operation and SQL translation

- Task 1: Write tests for DuplicateCount Op
- Task 2: Implement DuplicateCount Op
- Task 3: Write tests for Dialect Translation (including DuckDB execution)
- Task 4: Implement Dialect Translation

**Why this grouping works**:
- Creates the core operation that everything else depends on
- Can be fully tested with unit tests
- Includes the SQL generation which can be verified independently
- After this group, you have a working Op that generates valid SQL

**Verification**: Run `uv run pytest tests/test_ops.py tests/test_dialect.py -v`

**Suggested commit**: `feat: Add DuplicateCount op with SQL dialect translation`

---

### Task Group 2: State and Spec Implementation (Tasks 5-8)
**Focus**: Implement state management and high-level metric specification

- Task 5: Write tests for DuplicateCount State
- Task 6: Implement DuplicateCount State (with identity error)
- Task 7: Write tests for DuplicateCount Spec
- Task 8: Implement DuplicateCount Spec (including MetricType update)

**Why this grouping works**:
- State and Spec are closely related (Spec creates States)
- Can be tested independently of the API layer
- After this group, you have the complete metric specification system

**Verification**: Run `uv run pytest tests/test_states.py tests/test_specs.py -v`

**Suggested commit**: `feat: Add DuplicateCount state and spec with registry support`

---

### Task Group 3: API and Integration (Tasks 9-11)
**Focus**: User-facing API and end-to-end testing

- Task 9: Write tests for Provider Method
- Task 10: Implement Provider Method
- Task 11: Integration Tests

**Why this grouping works**:
- Completes the user-facing API
- Integration tests verify the entire stack works together
- Natural point to ensure everything integrates properly

**Verification**: Run `uv run pytest tests/test_provider.py tests/test_duplicate_count_integration.py -v`

**Suggested commit**: `feat: Add provider API and integration tests for DuplicateCount`

---

### Task Group 4: Documentation and Final Checks (Tasks 12-13)
**Focus**: Documentation and final quality assurance

- Task 12: Documentation
- Task 13: Final Verification (all tests, linting, manual testing)

**Why this grouping works**:
- Documentation can reference the complete implementation
- Final verification ensures quality before declaring done
- Natural completion point

**Verification**: Run full test suite and pre-commit hooks

**Suggested commit**: `docs: Add documentation and final verification for DuplicateCount`

---

### Implementation Tips for Each Group:

1. **After completing each group**:
   - Run the group-specific tests
   - Run `uv run mypy` on the modified files
   - Run `uv run ruff check --fix` on the modified files
   - Make a single commit with all changes from that group
   - Take a break before starting the next group

2. **Quality checks between groups**:
   - Ensure all tests pass before moving to the next group
   - Check that the code follows the project's style guidelines
   - Verify that type hints are properly added

3. **If issues arise**:
   - Each group is designed to be independent
**What to do**:
1. Add the new `DuplicateCount` class at the end of the file, before the final comments
2. Follow the pattern of existing ops like `NullCount` or `Average`

**Code to add**:
```python
class DuplicateCount(OpValueMixin[float], SqlOp[float]):
    __match_args__ = ("columns",)

    def __init__(self, columns: list[str]) -> None:
        OpValueMixin.__init__(self)
        if not columns:
            raise ValueError("DuplicateCount requires at least one column")
        self.columns = columns
        self._prefix = random_prefix()

    @property
    def name(self) -> str:
        return f"duplicate_count({','.join(self.columns)})"

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def sql_col(self) -> str:
        return f"{self.prefix}_{self.name}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DuplicateCount):
            return NotImplemented
        return self.columns == other.columns

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.columns)))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()
```

**How to test**:
```bash
# Run the tests from Task 1 - they should now pass
uv run pytest tests/test_ops.py -k duplicate_count -v
```

**Commit**:
```bash
git add src/dqx/ops.py
git commit -m "feat: Implement DuplicateCount op"
```

### Task 3: Write Tests for Dialect Translation

**File**: `tests/test_dialect.py`

**What to do**:
1. Add tests for DuplicateCount SQL translation, including a test that executes the SQL in DuckDB

**Code to add**:
```python
def test_translate_duplicate_count() -> None:
    from dqx import ops
    from dqx.dialect import DuckDBDialect

    dialect = DuckDBDialect()

    # Test single column
    op1 = ops.DuplicateCount(["user_id"])
    sql1 = dialect.translate_sql_op(op1)
    assert sql1 == f"CAST(COUNT(*) - COUNT(DISTINCT (user_id)) AS DOUBLE) AS '{op1.sql_col}'"

    # Test multiple columns
    op2 = ops.DuplicateCount(["user_id", "product_id", "date"])
    sql2 = dialect.translate_sql_op(op2)
    assert sql2 == f"CAST(COUNT(*) - COUNT(DISTINCT (user_id, product_id, date)) AS DOUBLE) AS '{op2.sql_col}'"


def test_translate_duplicate_count_with_duckdb_execution() -> None:
    """Test that the generated SQL actually works in DuckDB."""
    from dqx import ops
    from dqx.dialect import DuckDBDialect
    import duckdb

    dialect = DuckDBDialect()

    # Create test data
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE test_data AS
        SELECT * FROM (VALUES
            (1, 'A', 100),
            (1, 'A', 100),  -- Duplicate
            (2, 'B', 200),
            (2, 'B', 300),  -- Same user_id and name, different amount
            (3, 'C', 400)
        ) AS t(user_id, name, amount)
    """)

    # Test single column
    op1 = ops.DuplicateCount(["user_id"])
    sql1 = dialect.translate_sql_op(op1)

    # Execute the generated SQL
    result1 = conn.execute(f"SELECT {sql1} FROM test_data").fetchone()[0]
    assert result1 == 2.0  # 5 rows - 3 unique user_ids = 2 duplicates

    # Test multiple columns
    op2 = ops.DuplicateCount(["user_id", "name"])
    sql2 = dialect.translate_sql_op(op2)

    result2 = conn.execute(f"SELECT {sql2} FROM test_data").fetchone()[0]
    assert result2 == 1.0  # 5 rows - 4 unique (user_id, name) pairs = 1 duplicate

    # Test all columns
    op3 = ops.DuplicateCount(["user_id", "name", "amount"])
    sql3 = dialect.translate_sql_op(op3)

    result3 = conn.execute(f"SELECT {sql3} FROM test_data").fetchone()[0]
    assert result3 == 1.0  # 5 rows - 4 unique combinations = 1 duplicate

    conn.close()
```

**How to test**:
```bash
uv run pytest tests/test_dialect.py::test_translate_duplicate_count -v
uv run pytest tests/test_dialect.py::test_translate_duplicate_count_with_duckdb_execution -v
```

**Commit**:
```bash
git add tests/test_dialect.py
git commit -m "test: Add dialect translation tests for DuplicateCount with DuckDB execution"
```

### Task 4: Implement Dialect Translation

**File**: `src/dqx/dialect.py`

**What to do**:
1. Find the `translate_sql_op` method in the `DuckDBDialect` class
2. Add a new case for `DuplicateCount` before the default case

**Code to add** (inside the match statement):
```python
            case ops.DuplicateCount(columns=cols):
                column_list = ", ".join(cols)
                return f"CAST(COUNT(*) - COUNT(DISTINCT ({column_list})) AS DOUBLE) AS '{op.sql_col}'"
```

**Important**: You'll also need to import the ops module at the top if not already imported:
```python
from dqx import ops
```

**How to test**:
```bash
# Run the dialect test - should now pass
uv run pytest tests/test_dialect.py::test_translate_duplicate_count -v
```

**Commit**:
```bash
git add src/dqx/dialect.py
git commit -m "feat: Add DuplicateCount translation to DuckDB dialect"
```

### Task 5: Write Tests for DuplicateCount State

**File**: `tests/test_states.py`

**What to do**:
1. Add tests for the new non-mergeable DuplicateCount state

**Code to add**:
```python
def test_duplicate_count_state() -> None:
    from dqx import states
    from dqx.common import DQXError
    from copy import copy

    # Test basic functionality
    state = states.DuplicateCount(value=42.0)
    assert state.value == 42.0

    # Test identity raises error
    with pytest.raises(DQXError, match="DuplicateCount state does not support identity"):
        states.DuplicateCount.identity()

    # Test serialization
    serialized = state.serialize()
    deserialized = states.DuplicateCount.deserialize(serialized)
    assert deserialized.value == 42.0
    assert state == deserialized

    # Test copy
    copied = copy(state)
    assert copied.value == 42.0
    assert copied == state

    # Test merge raises error
    state1 = states.DuplicateCount(value=10.0)
    state2 = states.DuplicateCount(value=20.0)

    with pytest.raises(DQXError, match="DuplicateCount state cannot be merged"):
        state1.merge(state2)


def test_duplicate_count_state_equality() -> None:
    from dqx import states

    state1 = states.DuplicateCount(value=42.0)
    state2 = states.DuplicateCount(value=42.0)
    state3 = states.DuplicateCount(value=43.0)

    assert state1 == state2
    assert state1 != state3
    assert state1 != "not a state"
    assert state1 != 42
```

**How to test**:
```bash
uv run pytest tests/test_states.py::test_duplicate_count_state -v
uv run pytest tests/test_states.py::test_duplicate_count_state_equality -v
```

**Commit**:
```bash
git add tests/test_states.py
git commit -m "test: Add tests for DuplicateCount state with identity error"
```

### Task 6: Implement DuplicateCount State

**File**: `src/dqx/states.py`

**What to do**:
1. Add the new `DuplicateCount` state class at the end of the file

**Code to add**:
```python
class DuplicateCount(State):
    """Non-mergeable state for duplicate count metrics.

    Duplicate counts cannot be merged across partitions because the same
    value might appear in multiple partitions, leading to incorrect counts.

    This state does not support identity or merge operations.
    """

    def __init__(self, value: float) -> None:
        self._value = float(value)

    @classmethod
    def identity(cls) -> DuplicateCount:
        raise DQXError(
            "DuplicateCount state does not support identity. "
            "Duplicate counts must be computed on the entire dataset in a single pass."
        )

    @property
    def value(self) -> float:
        return self._value

    def serialize(self) -> bytes:
        return msgpack.packb(self._value)

    @classmethod
    def deserialize(cls, data: bytes) -> DuplicateCount:
        return cls(value=msgpack.unpackb(data))

    def merge(self, other: DuplicateCount) -> DuplicateCount:
        raise DQXError(
            "DuplicateCount state cannot be merged across partitions. "
            "Duplicate counts must be computed on the entire dataset."
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DuplicateCount):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> DuplicateCount:
        return DuplicateCount(value=self._value)
```

**How to test**:
```bash
# Run the state tests - should now pass
uv run pytest tests/test_states.py -k duplicate_count -v
```

**Commit**:
```bash
git add src/dqx/states.py
git commit -m "feat: Implement non-mergeable DuplicateCount state with identity error"
```

### Task 7: Update MetricType and Write Tests for DuplicateCount Spec

**File**: `tests/test_specs.py`

**What to do**:
1. Add tests for the DuplicateCount metric spec

**Code to add**:
```python
def test_duplicate_count_spec() -> None:
    from dqx import specs, ops, states

    # Test basic creation
    spec = specs.DuplicateCount(["col1", "col2"])
    assert spec.name == "duplicate_count(col1,col2)"
    assert spec.parameters == {"columns": ["col1", "col2"]}
    assert spec.metric_type == "DuplicateCount"

    # Test empty columns raises error
    with pytest.raises(ValueError, match="DuplicateCount requires at least one column"):
        specs.DuplicateCount([])

    # Test analyzers
    analyzers = spec.analyzers
    assert len(analyzers) == 1
    assert isinstance(analyzers[0], ops.DuplicateCount)
    assert analyzers[0].columns == ["col1", "col2"]

    # Test state creation
    analyzers[0].assign(42.0)
    state = spec.state()
    assert isinstance(state, states.DuplicateCount)
    assert state.value == 42.0

    # Test deserialization
    serialized = state.serialize()
    deserialized = spec.deserialize(serialized)
    assert isinstance(deserialized, states.DuplicateCount)
    assert deserialized.value == 42.0


def test_duplicate_count_spec_equality() -> None:
    from dqx import specs

    spec1 = specs.DuplicateCount(["col1", "col2"])
    spec2 = specs.DuplicateCount(["col1", "col2"])
    spec3 = specs.DuplicateCount(["col1"])
    spec4 = specs.DuplicateCount(["col2", "col1"])  # Different order

    assert spec1 == spec2
    assert spec1 != spec3
    assert spec1 != spec4
    assert spec1 != "not a spec"

    # Test hash
    assert hash(spec1) == hash(spec2)
    assert hash(spec1) != hash(spec3)


def test_duplicate_count_spec_str() -> None:
    from dqx import specs

    spec = specs.DuplicateCount(["user_id", "product_id", "date"])
    assert str(spec) == "duplicate_count(user_id,product_id,date)"
```

**How to test**:
```bash
uv run pytest tests/test_specs.py::test_duplicate_count_spec -v
uv run pytest tests/test_specs.py::test_duplicate_count_spec_equality -v
uv run pytest tests/test_specs.py::test_duplicate_count_spec_str -v
```

**Commit**:
```bash
git add tests/test_specs.py
git commit -m "test: Add tests for DuplicateCount spec"
```

### Task 8: Implement DuplicateCount Spec

**File**: `src/dqx/specs.py`

**What to do**:
1. First, update the `MetricType` literal at the top of the file to include "DuplicateCount"
2. Add the `DuplicateCount` class implementation

**Step 1 - Update MetricType**:
```python
MetricType = Literal[
    "NumRows",
    "First",
    "Average",
    "Variance",
    "Minimum",
    "Maximum",
    "Sum",
    "NullCount",
    "NegativeCount",
    "ApproxCardinality",
    "DuplicateCount",  # Add this line
]
```

**Step 2 - Add the class** (at the end before the registry code):
```python
class DuplicateCount:
    metric_type: MetricType = "DuplicateCount"

    def __init__(self, columns: list[str]) -> None:
        if not columns:
            raise ValueError("DuplicateCount requires at least one column")
        self._columns = columns
        self._analyzers = (ops.DuplicateCount(self._columns),)

    @property
    def name(self) -> str:
        return f"duplicate_count({','.join(self._columns)})"

    @property
    def parameters(self) -> Parameters:
        return {"columns": self._columns}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.DuplicateCount:
        return states.DuplicateCount(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.DuplicateCount.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DuplicateCount):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name
```

**How to test**:
```bash
# Run the spec tests - should now pass
uv run pytest tests/test_specs.py -k duplicate_count -v

# Also verify the registry picks it up
uv run python -c "from dqx.specs import registry; print('DuplicateCount' in registry)"
```

**Commit**:
```bash
git add src/dqx/specs.py
git commit -m "feat: Implement DuplicateCount spec with registry support"
```

### Task 9: Write Tests for Provider Method

**File**: `tests/test_provider.py`

**What to do**:
1. Add a test for the duplicate_count provider method

**Code to add**:
```python
def test_duplicate_count_provider() -> None:
    from dqx.provider import MetricProvider
    from dqx.orm.repositories import MetricDB
    from dqx.common import ResultKeyProvider
    import sympy as sp

    db = MetricDB()
    provider = MetricProvider(db)

    # Test single column
    symbol1 = provider.duplicate_count(["user_id"])
    assert isinstance(symbol1, sp.Symbol)

    # Test multiple columns
    symbol2 = provider.duplicate_count(["user_id", "product_id", "date"])
    assert isinstance(symbol2, sp.Symbol)

    # Test with custom key and dataset
    key = ResultKeyProvider()
    symbol3 = provider.duplicate_count(["col1", "col2"], key=key, dataset="test_dataset")
    assert isinstance(symbol3, sp.Symbol)

    # Verify the symbolic metric was registered correctly
    sm = provider.get_symbol(symbol2)
    assert sm.name == "duplicate_count(user_id,product_id,date)"
    assert sm.metric_spec.name == "duplicate_count(user_id,product_id,date)"
    assert sm.dataset is None

    sm3 = provider.get_symbol(symbol3)
    assert sm3.dataset == "test_dataset"
```

**How to test**:
```bash
uv run pytest tests/test_provider.py::test_duplicate_count_provider -v
```

**Commit**:
```bash
git add tests/test_provider.py
git commit -m "test: Add tests for duplicate_count provider method"
```

### Task 10: Implement Provider Method

**File**: `src/dqx/provider.py`

**What to do**:
1. Add the `duplicate_count` method to the `MetricProvider` class
2. Place it after the other metric methods (like `approx_cardinality`)

**Code to add**:
```python
    def duplicate_count(
        self, columns: list[str], key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        return self.metric(specs.DuplicateCount(columns), key, dataset)
```

**How to test**:
```bash
# Run the provider test - should now pass
uv run pytest tests/test_provider.py::test_duplicate_count_provider -v
```

**Commit**:
```bash
git add src/dqx/provider.py
git commit -m "feat: Add duplicate_count method to MetricProvider"
```

### Task 11: Integration Tests

**File**: Create `tests/test_duplicate_count_integration.py`

**What to do**:
1. Create a new test file for end-to-end integration testing

**Code to add**:
```python
"""Integration tests for DuplicateCount functionality."""

import tempfile
from pathlib import Path

import duckdb
import pytest

from dqx.analyzer import Analyzer
from dqx.common import ResultKey, ResultKeyProvider, DuckDataSource
from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider
from dqx.specs import DuplicateCount


def test_duplicate_count_end_to_end() -> None:
    """Test DuplicateCount from data source to final result."""
    # Create test data with duplicates
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = duckdb.connect(str(db_path))

        # Create test table with known duplicates
        conn.execute("""
            CREATE TABLE test_data AS
            SELECT * FROM (VALUES
                ('user1', 'prod1', '2024-01-01'),
                ('user1', 'prod1', '2024-01-01'),  -- Duplicate
                ('user1', 'prod2', '2024-01-01'),
                ('user2', 'prod1', '2024-01-01'),
                ('user2', 'prod1', '2024-01-02'),
                ('user2', 'prod1', '2024-01-02'),  -- Duplicate
                ('user3', 'prod3', '2024-01-03')
            ) AS t(user_id, product_id, order_date)
        """)

        # Create data source
        ds = DuckDataSource(
            name="test_data",
            cte="SELECT * FROM test_data",
            conn=conn,
            dialect="duckdb"
        )

        # Test 1: Count duplicates on (user_id, product_id)
        spec1 = DuplicateCount(["user_id", "product_id"])
        analyzer = Analyzer()
        key = ResultKey(yyyy_mm_dd=None, datasource="test")

        report = analyzer.analyze(ds, [spec1], key)
        metric = report[(spec1, key)]

        # We have 7 total rows, 5 unique (user_id, product_id) combinations
        # So duplicates = 7 - 5 = 2
        assert metric.value == 2.0

        # Test 2: Count duplicates on all three columns
        spec2 = DuplicateCount(["user_id", "product_id", "order_date"])
        report2 = analyzer.analyze(ds, [spec2], key)
        metric2 = report2[(spec2, key)]

        # We have 7 total rows, 6 unique combinations
        # So duplicates = 7 - 6 = 1
        assert metric2.value == 1.0

        # Test 3: Count duplicates on single column
        spec3 = DuplicateCount(["user_id"])
        report3 = analyzer.analyze(ds, [spec3], key)
        metric3 = report3[(spec3, key)]

        # We have 7 rows, 3 unique user_ids
        # So duplicates = 7 - 3 = 4
        assert metric3.value == 4.0

        conn.close()


def test_duplicate_count_with_provider() -> None:
    """Test DuplicateCount using MetricProvider API."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up database and provider
        db_path = Path(tmpdir) / "test.db"
        conn = duckdb.connect(str(db_path))

        # Create test data
        conn.execute("""
            CREATE TABLE products AS
            SELECT * FROM (VALUES
                (1, 'Widget', 'Electronics'),
                (2, 'Gadget', 'Electronics'),
                (2, 'Gadget', 'Electronics'),  -- Exact duplicate
                (3, 'Tool', 'Hardware'),
                (3, 'Tool', 'Software')        -- Same id and name, different category
            ) AS t(id, name, category)
        """)

        ds = DuckDataSource(
            name="products",
            cte="SELECT * FROM products",
            conn=conn,
            dialect="duckdb"
        )

        # Use provider API
        metric_db = MetricDB()
        provider = MetricProvider(metric_db)

        # Create symbolic metric
        dup_symbol = provider.duplicate_count(["id", "name"])

        # Get the metric spec
        symbolic_metric = provider.get_symbol(dup_symbol)
        spec = symbolic_metric.metric_spec

        # Analyze
        analyzer = Analyzer()
        key = ResultKey(yyyy_mm_dd=None, datasource="products")
        report = analyzer.analyze(ds, [spec], key)

        # We have 5 rows, 4 unique (id, name) combinations
        # Duplicates = 5 - 4 = 1
        metric = report[(spec, key)]
        assert metric.value == 1.0

        conn.close()


def test_duplicate_count_no_duplicates() -> None:
    """Test DuplicateCount when there are no duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = duckdb.connect(str(db_path))

        # Create test table with no duplicates
        conn.execute("""
            CREATE TABLE unique_data AS
            SELECT * FROM (VALUES
                (1, 'A'),
                (2, 'B'),
                (3, 'C'),
                (4, 'D')
            ) AS t(id, value)
        """)

        ds = DuckDataSource(
            name="unique_data",
            cte="SELECT * FROM unique_data",
            conn=conn,
            dialect="duckdb"
        )

        spec = DuplicateCount(["id", "value"])
        analyzer = Analyzer()
        key = ResultKey(yyyy_mm_dd=None, datasource="unique")

        report = analyzer.analyze(ds, [spec], key)
        metric = report[(spec, key)]

        # No duplicates: 4 rows - 4 unique = 0
        assert metric.value == 0.0

        conn.close()


def test_duplicate_count_batch_processing_error() -> None:
    """Test that DuplicateCount raises error when trying to merge across batches."""
    from dqx.common import DQXError
    from dqx import states

    # Create two states from different batches
    state1 = states.DuplicateCount(value=10.0)
    state2 = states.DuplicateCount(value=5.0)

    # Attempting to merge should raise an error
    with pytest.raises(DQXError, match="DuplicateCount state cannot be merged"):
        state1.merge(state2)
```

**How to test**:
```bash
uv run pytest tests/test_duplicate_count_integration.py -v
```

**Commit**:
```bash
git add tests/test_duplicate_count_integration.py
git commit -m "test: Add integration tests for DuplicateCount"
```

### Task 12: Documentation

**File**: Create `docs/duplicate_count_usage.md`

**What to do**:
1. Create user documentation for the new feature

**Code to add**:
```markdown
# DuplicateCount Metric

The `DuplicateCount` metric counts the number of duplicate rows in a dataset based on specified columns.

## Overview

DuplicateCount calculates: `COUNT(*) - COUNT(DISTINCT (column1, column2, ...))`

This gives you the number of rows that are duplicates based on the specified column combination.

## Usage

### Basic Usage

```python
from dqx.provider import MetricProvider
from dqx.orm.repositories import MetricDB

# Create provider
db = MetricDB()
provider = MetricProvider(db)

# Count duplicates based on a single column
user_duplicates = provider.duplicate_count(["user_id"])

# Count duplicates based on multiple columns
order_duplicates = provider.duplicate_count(["user_id", "product_id", "order_date"])
```

### With VerificationSuite

```python
from dqx.api import DQCheck, VerificationSuite
from dqx.common import DuckDataSource

# Create verification suite
suite = VerificationSuite("duplicate_check_suite")

# Add duplicate count check
suite.add_check(
    DQCheck("no_duplicate_orders")
    .on(DuckDataSource("orders", "SELECT * FROM orders"))
    .with_metric(provider.duplicate_count(["user_id", "product_id", "order_date"]))
    .with_assertion(lambda x: x == 0, name="No duplicates allowed")
)

# Run the suite
results = suite.run()
```

### Examples

#### Example 1: Find duplicate customer records
```python
# Count how many duplicate customer records exist
customer_dups = provider.duplicate_count(["email"])
```

#### Example 2: Detect duplicate transactions
```python
# Find duplicate transactions (same user, amount, timestamp)
transaction_dups = provider.duplicate_count(["user_id", "amount", "timestamp"])
```

#### Example 3: Identify duplicate product SKUs
```python
# Check for duplicate product identifiers
sku_dups = provider.duplicate_count(["sku", "vendor_id"])
```

## Important Notes

1. **Non-Mergeable State**: DuplicateCount cannot be merged across data partitions. If you're using batch processing, ensure all data is processed in a single batch.

2. **Column Order Matters**: `duplicate_count(["a", "b"])` is different from `duplicate_count(["b", "a"])`.

3. **Performance**: The operation uses `COUNT(DISTINCT)` which can be expensive on large datasets with many unique values.

4. **Null Handling**: NULL values are considered equal for duplicate detection (standard SQL behavior).
```

**Commit**:
```bash
git add docs/duplicate_count_usage.md
git commit -m "docs: Add user documentation for DuplicateCount metric"
```

### Task 13: Final Verification

**What to do**:
1. Run all tests to ensure everything works
2. Check code formatting and linting
3. Verify the feature works end-to-end

**Steps**:
```bash
# 1. Run all tests
uv run pytest -v

# 2. Run mypy type checking
uv run mypy src/dqx/ops.py src/dqx/states.py src/dqx/specs.py src/dqx/dialect.py src/dqx/provider.py

# 3. Run ruff linting
uv run ruff check --fix src/dqx/ops.py src/dqx/states.py src/dqx/specs.py src/dqx/dialect.py src/dqx/provider.py

# 4. Run the pre-commit hooks
./bin/run-hooks.sh

# 5. Create a simple test script to verify end-to-end
cat > test_duplicate_count_manual.py << 'EOF'
"""Manual test to verify DuplicateCount works end-to-end."""

import duckdb
from dqx.analyzer import Analyzer
from dqx.common import DuckDataSource, ResultKey
from dqx.specs import DuplicateCount

# Create test data
conn = duckdb.connect(":memory:")
conn.execute("""
    CREATE TABLE test AS
    SELECT * FROM (VALUES
        (1, 'A'),
        (1, 'A'),  -- Duplicate
        (2, 'B'),
        (3, 'C')
    ) AS t(id, value)
""")

# Create data source and analyze
ds = DuckDataSource("test", "SELECT * FROM test", conn, dialect="duckdb")
spec = DuplicateCount(["id", "value"])
analyzer = Analyzer()
key = ResultKey(yyyy_mm_dd=None, datasource="test")

report = analyzer.analyze(ds, [spec], key)
metric = report[(spec, key)]

print(f"Duplicate count: {metric.value}")
print(f"Expected: 1.0")
assert metric.value == 1.0
print("✅ Test passed!")
EOF

uv run python test_duplicate_count_manual.py

# Clean up
rm test_duplicate_count_manual.py
```

## Summary

This implementation plan provides a complete, step-by-step guide to implementing DuplicateCountOp in the DQX framework. The implementation follows these principles:

1. **Test-Driven Development**: Write tests first, then implement
2. **Incremental Changes**: Small, focused commits
3. **YAGNI**: Only implement what's needed
4. **DRY**: Reuse existing patterns and infrastructure

### Key Design Decisions

1. **Multi-column support**: Unlike other ops that take a single column, DuplicateCount accepts a list of columns
2. **Non-mergeable state**: Duplicate counts cannot be accurately merged across partitions
3. **SQL implementation**: Uses `COUNT(*) - COUNT(DISTINCT (...))` for efficiency
4. **Consistent API**: Follows the same pattern as other metrics in the provider

### Testing Strategy

1. **Unit tests**: Test each component in isolation
2. **Integration tests**: Test the full flow from data source to result
3. **Edge cases**: Test empty data, no duplicates, and error conditions

### Next Steps

After implementing this feature:
1. Monitor for any issues in production
2. Consider adding optimizations for large datasets
3. Add support for other SQL dialects as needed
