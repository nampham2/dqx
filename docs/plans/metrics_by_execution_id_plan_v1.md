# Metrics by Execution ID Implementation Plan v1

## Overview

This feature adds the ability to retrieve all metrics persisted during a specific VerificationSuite execution, including metrics from extended operations (e.g., day_over_day) that create lagged metrics. This enables comprehensive metric collection for analysis and debugging.

### Key Goals
1. Add unique execution ID to each VerificationSuite instance
2. Tag all persisted metrics with the suite's execution ID
3. Provide a utility function to retrieve all metrics by execution ID
4. Ensure lagged metrics from extended operations are included

## Technical Design

### How SymbolInfo is Created

`SymbolInfo` is a dataclass in `dqx.common` that represents metadata about a computed metric symbol:

```python
@dataclass
class SymbolInfo:
    name: str                          # Symbol identifier (e.g., "x_1", "x_2")
    metric: str                        # Human-readable description (e.g., "average(price)")
    dataset: str | None                # Dataset name
    value: Result[float, str]          # Success(float) or Failure(error)
    yyyy_mm_dd: datetime.date          # Evaluation date
    tags: Tags = field(default_factory=dict)  # Additional metadata
```

### Execution ID Flow

1. **Generation**: Each VerificationSuite instance generates a unique UUID in `__init__`
2. **Propagation**: The execution ID is injected into ResultKey tags before metric computation
3. **Persistence**: All metrics are persisted with the `__execution_id` tag
4. **Retrieval**: Query MetricDB using the execution ID to retrieve all related metrics

### Key Implementation Details

- **Tag Injection Point**: In `VerificationSuite.run()`, before calling `_run()`
- **Tag Key**: Use `__execution_id` (double underscore prefix for internal tags)
- **Metric Naming**: Since database doesn't store symbol names, generate placeholder names (e.g., "m_1", "m_2")
- **Lagged Metrics**: Automatically included as they share the same execution ID

## Implementation Tasks

### Task Group 1: Core Infrastructure (TDD)

#### Task 1.1: Create test for execution ID generation
**File**: `tests/test_api.py`

```python
def test_verification_suite_execution_id():
    """Test that VerificationSuite generates unique execution ID."""
    from dqx import VerificationSuite
    from dqx.orm.repositories import InMemoryMetricDB
    import uuid

    db = InMemoryMetricDB()
    suite1 = VerificationSuite([], db, "Test Suite 1")
    suite2 = VerificationSuite([], db, "Test Suite 2")

    # Should have execution_id property
    assert hasattr(suite1, 'execution_id')
    assert hasattr(suite2, 'execution_id')

    # Should be valid UUIDs
    uuid.UUID(suite1.execution_id)  # Raises if invalid
    uuid.UUID(suite2.execution_id)  # Raises if invalid

    # Should be unique
    assert suite1.execution_id != suite2.execution_id
```

#### Task 1.2: Implement execution ID generation
**File**: `src/dqx/api.py`

In `VerificationSuite.__init__`, add:
```python
import uuid

class VerificationSuite:
    def __init__(self, ...):
        # ... existing code ...
        self._execution_id = str(uuid.uuid4())

    @property
    def execution_id(self) -> str:
        """Unique identifier for this suite execution."""
        return self._execution_id
```

#### Task 1.3: Create test for tag injection
**File**: `tests/test_api.py`

```python
def test_execution_id_tag_injection(mocker):
    """Test that execution_id is injected into ResultKey tags."""
    from dqx import VerificationSuite, ResultKey
    from dqx.orm.repositories import InMemoryMetricDB
    from datetime import date

    db = InMemoryMetricDB()
    suite = VerificationSuite([], db, "Test Suite")

    # Mock _run to capture the enriched key
    mock_run = mocker.patch.object(suite, '_run')

    # Run with original key
    original_key = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={"env": "test"})
    suite.run([], original_key)

    # Verify _run was called with enriched key
    mock_run.assert_called_once()
    enriched_key = mock_run.call_args[0][1]  # Second positional arg

    assert enriched_key.yyyy_mm_dd == original_key.yyyy_mm_dd
    assert enriched_key.tags["env"] == "test"
    assert enriched_key.tags["__execution_id"] == suite.execution_id
```

### Task Group 2: Tag Injection Implementation

#### Task 2.1: Implement tag injection in run method
**File**: `src/dqx/api.py`

In `VerificationSuite.run()`:
```python
def run(self, datasources: list[SqlDataSource], key: ResultKey, *, enable_plugins: bool = True) -> None:
    # ... existing validation ...

    # Enrich the key with execution_id
    enriched_tags = {**key.tags, "__execution_id": self.execution_id}
    enriched_key = ResultKey(yyyy_mm_dd=key.yyyy_mm_dd, tags=enriched_tags)

    # Use enriched_key instead of original key
    self._run(datasources, enriched_key, enable_plugins)
```

#### Task 2.2: Run tests and verify
```bash
uv run pytest tests/test_api.py::test_verification_suite_execution_id -v
uv run pytest tests/test_api.py::test_execution_id_tag_injection -v
```

### Task Group 3: Data Retrieval Module (TDD)

#### Task 3.1: Create test for metrics_by_execution_id function
**File**: `tests/test_data.py`

```python
"""Tests for data retrieval utilities."""
import uuid
from datetime import date
from unittest.mock import Mock

import pytest
from returns.result import Success

from dqx.common import SymbolInfo
from dqx.orm.repositories import Metric, MetricDB


class TestMetricsByExecutionId:
    """Test the metrics_by_execution_id function."""

    def test_retrieves_metrics_by_execution_id(self):
        """Test successful retrieval and conversion of metrics."""
        from dqx.data import metrics_by_execution_id

        # Setup
        exec_id = str(uuid.uuid4())
        mock_db = Mock(spec=MetricDB)

        # Create mock Metric objects
        mock_metric1 = Mock(spec=Metric)
        mock_metric1.dataset = "sales"
        mock_metric1.value = 42.5
        mock_metric1.yyyy_mm_dd = date(2024, 1, 1)
        mock_metric1.tags = {"env": "prod", "__execution_id": exec_id}

        mock_spec1 = Mock()
        mock_spec1.name = "average(price)"
        mock_metric1.to_spec.return_value = mock_spec1

        mock_metric2 = Mock(spec=Metric)
        mock_metric2.dataset = "sales"
        mock_metric2.value = 40.0
        mock_metric2.yyyy_mm_dd = date(2023, 12, 31)  # Lagged metric
        mock_metric2.tags = {"env": "prod", "__execution_id": exec_id}

        mock_spec2 = Mock()
        mock_spec2.name = "average(price)"
        mock_metric2.to_spec.return_value = mock_spec2

        mock_db.search.return_value = [mock_metric1, mock_metric2]

        # Execute
        results = metrics_by_execution_id(mock_db, exec_id)

        # Verify
        assert len(results) == 2

        # Check first metric
        assert results[0].name == "m_1"
        assert results[0].metric == "average(price)"
        assert results[0].dataset == "sales"
        assert results[0].value == Success(40.0)  # Older date comes first
        assert results[0].yyyy_mm_dd == date(2023, 12, 31)
        assert results[0].tags["__execution_id"] == exec_id

        # Check second metric
        assert results[1].name == "m_2"
        assert results[1].metric == "average(price)"
        assert results[1].value == Success(42.5)
        assert results[1].yyyy_mm_dd == date(2024, 1, 1)

        # Verify search was called correctly
        mock_db.search.assert_called_once()
        # The exact call depends on SQLAlchemy expression, just verify it was called

    def test_empty_results(self):
        """Test handling of no metrics found."""
        from dqx.data import metrics_by_execution_id

        exec_id = str(uuid.uuid4())
        mock_db = Mock(spec=MetricDB)
        mock_db.search.return_value = []

        results = metrics_by_execution_id(mock_db, exec_id)

        assert results == []
        assert isinstance(results, list)
```

#### Task 3.2: Create dqx.data module
**File**: `src/dqx/data.py`

```python
"""Data retrieval utilities for DQX."""

from collections.abc import Sequence

from returns.result import Success

from dqx.common import SymbolInfo
from dqx.orm.repositories import Metric, MetricDB

__all__ = ["metrics_by_execution_id"]


def metrics_by_execution_id(db: MetricDB, execution_id: str) -> Sequence[SymbolInfo]:
    """Retrieve all metrics persisted with a specific execution ID.

    This function queries the MetricDB for all metrics tagged with the given
    execution ID and converts them to SymbolInfo objects. This includes metrics
    from extended operations (e.g., day_over_day) that create lagged metrics.

    Args:
        db: MetricDB instance to query
        execution_id: The execution ID to search for (UUID string)

    Returns:
        List of SymbolInfo objects sorted by date and metric name.
        Symbol names are generated as "m_1", "m_2", etc. since the
        database doesn't store the original symbol names.

    Example:
        >>> from dqx.orm.repositories import InMemoryMetricDB
        >>> db = InMemoryMetricDB()
        >>> suite = VerificationSuite([my_check], db, "My Suite")
        >>> suite.run([datasource], key)
        >>>
        >>> # Retrieve all metrics from this execution
        >>> metrics = metrics_by_execution_id(db, suite.execution_id)
        >>> for m in metrics:
        ...     print(f"{m.yyyy_mm_dd}: {m.metric} = {m.value}")
    """
    # Query metrics with the execution_id in tags
    # Using dictionary containment check for JSON field
    metrics = db.search(Metric.tags['"__execution_id"'].astext == execution_id)

    # Convert to SymbolInfo objects
    symbols = []
    for i, metric in enumerate(sorted(metrics, key=lambda m: (m.yyyy_mm_dd, m.to_spec().name))):
        # Create SymbolInfo from Metric
        symbol_info = SymbolInfo(
            name=f"m_{i+1}",  # Generate placeholder names
            metric=metric.to_spec().name,  # e.g., "average(price)"
            dataset=metric.dataset,
            value=Success(metric.value),  # Wrap in Success
            yyyy_mm_dd=metric.yyyy_mm_dd,
            tags=metric.tags
        )
        symbols.append(symbol_info)

    return symbols
```

#### Task 3.3: Update __init__.py to export the function
**File**: `src/dqx/__init__.py`

Add to the imports and __all__:
```python
from dqx.data import metrics_by_execution_id

__all__ = [
    # ... existing exports ...
    "metrics_by_execution_id",
]
```

### Task Group 4: Integration Testing

#### Task 4.1: Create integration test with InMemoryDB
**File**: `tests/test_data.py`

```python
def test_metrics_by_execution_id_integration():
    """Integration test: write metrics to DB and retrieve by execution_id."""
    import pyarrow as pa
    from dqx import check, MetricProvider, Context, VerificationSuite, ResultKey
    from dqx.datasource import DuckRelationDataSource
    from dqx.orm.repositories import InMemoryMetricDB
    from dqx.data import metrics_by_execution_id
    from datetime import date

    # Setup
    db = InMemoryMetricDB()

    # Create a check that uses extended metrics (to test lagged metrics)
    @check(name="Sales Check", severity="P1")
    def sales_check(mp: MetricProvider, ctx: Context):
        # Regular metric
        avg_price = mp.average("price", dataset="sales")
        ctx.assert_that(avg_price).where(name="Average price > 0").is_gt(0)

        # Extended metric that creates lagged metrics
        dow_avg = mp.ext.day_over_day(avg_price, dataset="sales")
        ctx.assert_that(dow_avg).where(name="DoD growth < 50%").is_lt(0.5)

    # Create suite and run
    suite = VerificationSuite([sales_check], db, "Sales Suite")
    exec_id = suite.execution_id  # Capture for later

    # Create test data for today
    data_today = pa.table({
        "price": [10.0, 15.0, 20.0, 25.0],
        "yyyy_mm_dd": [date(2024, 1, 2)] * 4
    })

    # Create test data for yesterday (for day_over_day)
    data_yesterday = pa.table({
        "price": [8.0, 12.0, 16.0, 20.0],
        "yyyy_mm_dd": [date(2024, 1, 1)] * 4
    })

    # Combine data
    data = pa.concat_tables([data_yesterday, data_today])
    datasource = DuckRelationDataSource.from_arrow(data, "sales")

    # Run the suite
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 2), tags={"env": "test"})
    suite.run([datasource], key)

    # Retrieve metrics using execution_id
    persisted = metrics_by_execution_id(db, exec_id)

    # Assertions
    assert len(persisted) >= 2  # At least avg_price for 2 days

    # Check we got metrics for different dates (lagged metrics)
    dates = {s.yyyy_mm_dd for s in persisted}
    assert date(2024, 1, 2) in dates  # Nominal date
    assert date(2024, 1, 1) in dates  # Lagged date

    # Check all have the execution_id
    for symbol in persisted:
        assert symbol.tags.get("__execution_id") == exec_id
        assert symbol.tags.get("env") == "test"  # Original tags preserved

    # Check metric names
    metric_names = {s.metric for s in persisted}
    assert "average(price)" in metric_names

    # Check values are Success
    for symbol in persisted:
        assert symbol.value.is_success()
        assert isinstance(symbol.value.unwrap(), float)


def test_multiple_suite_executions_isolated():
    """Test that different suite executions don't interfere."""
    import pyarrow as pa
    from dqx import check, MetricProvider, Context, VerificationSuite, ResultKey
    from dqx.datasource import DuckRelationDataSource
    from dqx.orm.repositories import InMemoryMetricDB
    from dqx.data import metrics_by_execution_id
    from datetime import date

    db = InMemoryMetricDB()

    @check(name="Simple Check", severity="P1")
    def simple_check(mp: MetricProvider, ctx: Context):
        rows = mp.num_rows(dataset="test")
        ctx.assert_that(rows).where(name="Has rows").is_gt(0)

    # Create and run first suite
    suite1 = VerificationSuite([simple_check], db, "Suite 1")
    exec_id1 = suite1.execution_id

    data = pa.table({"x": [1, 2, 3]})
    datasource = DuckRelationDataSource.from_arrow(data, "test")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})

    suite1.run([datasource], key)

    # Create and run second suite
    suite2 = VerificationSuite([simple_check], db, "Suite 2")
    exec_id2 = suite2.execution_id

    suite2.run([datasource], key)

    # Verify different execution IDs
    assert exec_id1 != exec_id2

    # Retrieve metrics for each execution
    metrics1 = metrics_by_execution_id(db, exec_id1)
    metrics2 = metrics_by_execution_id(db, exec_id2)

    # Both should have metrics
    assert len(metrics1) > 0
    assert len(metrics2) > 0

    # Metrics should be isolated
    for m in metrics1:
        assert m.tags["__execution_id"] == exec_id1

    for m in metrics2:
        assert m.tags["__execution_id"] == exec_id2
```

#### Task 4.2: Run all tests
```bash
uv run pytest tests/test_data.py -v
```

### Task Group 5: Final Verification

#### Task 5.1: Run all tests
```bash
# Run mypy type checking
uv run mypy src/dqx/data.py

# Run ruff linting
uv run ruff check src/dqx/data.py

# Run all new tests
uv run pytest tests/test_api.py::test_verification_suite_execution_id -v
uv run pytest tests/test_api.py::test_execution_id_tag_injection -v
uv run pytest tests/test_data.py -v

# Run pre-commit checks
bin/run-hooks.sh
```

#### Task 5.2: Verify complete test coverage
```bash
uv run pytest tests/ -v --cov=dqx --cov-report=term-missing
```

## Notes for Implementation

1. **Backward Compatibility**: This feature adds new functionality without breaking existing APIs
2. **Performance**: The execution_id tag adds minimal overhead (one UUID per suite)
3. **Security**: The `__execution_id` tag uses double underscore to indicate internal use
4. **Testing**: Full TDD approach with unit and integration tests
5. **Documentation**: Comprehensive docstrings for all new functions

## Expected Outcomes

After implementation:
- Each VerificationSuite has a unique `execution_id` property
- All metrics persisted during suite execution are tagged with this ID
- The `metrics_by_execution_id` function retrieves all metrics from a specific execution
- Lagged metrics from extended operations are automatically included
- Full test coverage with unit and integration tests
