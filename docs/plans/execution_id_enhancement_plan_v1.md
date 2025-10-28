# Execution ID Enhancement Implementation Plan v1

## Overview

This plan details the implementation of mandatory execution_id parameter throughout the metric retrieval pipeline in DQX. The goal is to ensure that all metric retrievals are properly scoped to a specific execution, preventing data mixing between different verification suite runs.

## Background

Currently, execution_id is stored in metric metadata but not consistently used during retrieval. This can lead to incorrect metric values when multiple executions exist for the same date/dataset combination.

## Design Summary

1. **Add ExecutionId type alias** for better type safety
2. **Add execution_id property** to MetricProvider
3. **Pass execution_id through the entire retrieval chain**:
   - VerificationSuite → MetricProvider → Lazy functions → Compute functions → Database methods
4. **Filter all database queries** by execution_id

## Implementation Tasks

### Task Group 1: Foundation (Type Alias & Database Layer)

#### Task 1.1: Add ExecutionId Type Alias
**File**: `src/dqx/common.py`
```python
# Add after existing type definitions
ExecutionId = str
```

#### Task 1.2: Update Database Methods - get_metric_value
**File**: `src/dqx/orm/repositories.py`
```python
def get_metric_value(
    self,
    metric: MetricSpec,
    key: ResultKey,
    dataset: str,
    execution_id: ExecutionId  # ADD THIS
) -> Maybe[float]:
    """Get a single metric value for a specific dataset and execution.

    Args:
        metric: The metric specification.
        key: The result key containing date and tags.
        dataset: The dataset name.
        execution_id: The execution ID to filter by.

    Returns:
        Maybe containing the metric value if found, Nothing otherwise.
    """
    query = (
        select(Metric.value)
        .where(
            Metric.metric_type == metric.metric_type,
            Metric.parameters == metric.parameters,
            Metric.yyyy_mm_dd == key.yyyy_mm_dd,
            Metric.tags == key.tags,
            Metric.dataset == dataset,
            func.json_extract(Metric.meta, "$.execution_id") == execution_id,  # ADD THIS
        )
        .order_by(Metric.created.desc())
        .limit(1)
    )
    return Maybe.from_optional(self.new_session().scalar(query))
```

#### Task 1.3: Update Database Methods - get_metric_window
**File**: `src/dqx/orm/repositories.py`
```python
def get_metric_window(
    self,
    metric: MetricSpec,
    key: ResultKey,
    lag: int,
    window: int,
    dataset: str,
    execution_id: ExecutionId  # ADD THIS
) -> Maybe[TimeSeries]:
    """Get metric values over a time window for a specific dataset and execution.

    Args:
        metric: The metric specification.
        key: The result key for the base date.
        lag: Number of days to lag from the base date.
        window: Number of days to include in the window.
        dataset: The dataset name.
        execution_id: The execution ID to filter by.

    Returns:
        Maybe containing the TimeSeries if found, Nothing otherwise.
    """
    from_date, until_date = key.range(lag, window)

    # Create CTE for finding latest metrics per day within execution
    latest_metrics_cte = (
        select(
            Metric.yyyy_mm_dd,
            Metric.value,
            func.row_number().over(
                partition_by=Metric.yyyy_mm_dd, order_by=Metric.created.desc()
            ).label("rn"),
        ).where(
            Metric.metric_type == metric.metric_type,
            Metric.parameters == metric.parameters,
            Metric.yyyy_mm_dd >= from_date,
            Metric.yyyy_mm_dd <= until_date,
            Metric.tags == key.tags,
            Metric.dataset == dataset,
            func.json_extract(Metric.meta, "$.execution_id") == execution_id,  # ADD THIS
        )
    ).cte("latest_metrics")

    # Rest of the method remains the same...
```

#### Task 1.4: Update imports in repositories.py
**File**: `src/dqx/orm/repositories.py`
```python
# Update the import line
from dqx.common import DQXError, ExecutionId, Metadata, ResultKey, Tags, TimeSeries
```

#### Task 1.5: Write Tests for Database Methods
**File**: `tests/test_repositories.py`
Create tests to verify execution_id filtering works correctly:
```python
def test_get_metric_value_with_execution_id(metric_db: MetricDB) -> None:
    """Test get_metric_value filters by execution_id."""
    # Create test data
    spec = specs.NumRows()
    key = ResultKey(yyyy_mm_dd=date(2023, 1, 1), tags={})
    dataset = "test_dataset"

    # Create metrics with different execution_ids
    metric1 = create_test_metric(
        metric_spec=spec,
        key=key,
        dataset=dataset,
        value=10.0,
        execution_id="exec-1"
    )
    metric2 = create_test_metric(
        metric_spec=spec,
        key=key,
        dataset=dataset,
        value=20.0,
        execution_id="exec-2"
    )

    metric_db.persist([metric1, metric2])

    # Test retrieving by execution_id
    result1 = metric_db.get_metric_value(spec, key, dataset, "exec-1")
    assert result1 == Some(10.0)

    result2 = metric_db.get_metric_value(spec, key, dataset, "exec-2")
    assert result2 == Some(20.0)

    # Test non-existent execution_id
    result3 = metric_db.get_metric_value(spec, key, dataset, "exec-999")
    assert result3 == Nothing
```

**Verification**: Run `uv run pytest tests/test_repositories.py -v` and ensure all tests pass.

### Task Group 2: Compute Functions Update

#### Task 2.1: Update compute.py imports
**File**: `src/dqx/compute.py`
```python
# Update imports
from dqx.common import ExecutionId, ResultKey, TimeSeries
```

#### Task 2.2: Update simple_metric function
**File**: `src/dqx/compute.py`
```python
def simple_metric(
    db: MetricDB,
    metric: MetricSpec,
    dataset: str,
    nominal_key: ResultKey,
    execution_id: ExecutionId  # ADD THIS
) -> Result[float, str]:
    """Retrieve a simple metric value from the database for a specific dataset and execution.

    Fetches the value of a metric for the given specification, date, dataset, and execution.

    Args:
        db: The metric database instance.
        metric: The metric specification to retrieve.
        dataset: The dataset name where the metric was computed.
        nominal_key: The result key containing date and tags.
        execution_id: The execution ID to filter by.

    Returns:
        Success with the metric value if found, Failure with error message otherwise.
    """
    value = db.get_metric_value(metric, nominal_key, dataset, execution_id)  # ADD execution_id
    error_msg = f"Metric {metric.name} for {nominal_key.yyyy_mm_dd.isoformat()} on dataset '{dataset}' in execution {execution_id} not found!"  # UPDATE MESSAGE
    return convert_maybe_to_result(value, error_msg)
```

#### Task 2.3: Update day_over_day, week_over_week, and stddev functions
**File**: `src/dqx/compute.py`

For each function, add execution_id parameter and pass it to database calls:

```python
def day_over_day(
    db: MetricDB,
    metric: MetricSpec,
    dataset: str,
    nominal_key: ResultKey,
    execution_id: ExecutionId  # ADD THIS
) -> Result[float, str]:
    """Calculate day-over-day ratio for a metric in a specific execution.

    ... (update docstring) ...
    execution_id: The execution ID to filter by.
    """
    base_key = nominal_key.lag(0)
    maybe_ts = db.get_metric_window(
        metric, base_key, lag=0, window=2, dataset=dataset, execution_id=execution_id  # ADD
    )
    # Rest remains the same...

# Similar changes for week_over_week and stddev
```

#### Task 2.4: Update compute function tests
**File**: `tests/test_compute.py`

Update all test functions to include execution_id parameter:
```python
def test_simple_metric_success(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_result_key: ResultKey
) -> None:
    """Test simple_metric with successful retrieval."""
    execution_id = "test-exec-123"
    mock_db.get_metric_value.return_value = Some(42.0)

    result = compute.simple_metric(
        mock_db, mock_metric, "test_dataset", mock_result_key, execution_id
    )

    assert isinstance(result, Success)
    assert result.unwrap() == 42.0
    mock_db.get_metric_value.assert_called_once_with(
        mock_metric, mock_result_key, "test_dataset", execution_id
    )
```

**Verification**: Run `uv run pytest tests/test_compute.py -v` and ensure all tests pass.

### Task Group 3: Provider Updates

#### Task 3.1: Add execution_id property to MetricProvider
**File**: `src/dqx/provider.py`
```python
class MetricProvider(SymbolicMetricBase):
    def __init__(self, db: MetricDB, execution_id: ExecutionId) -> None:
        super().__init__()
        self._db = db
        self._execution_id = execution_id

    @property
    def execution_id(self) -> ExecutionId:
        """The execution ID for this provider instance."""
        return self._execution_id
```

#### Task 3.2: Update lazy retrieval functions
**File**: `src/dqx/provider.py`
```python
def _create_lazy_retrieval_fn(
    provider: "MetricProvider",
    metric_spec: MetricSpec,
    symbol: sp.Symbol
) -> RetrievalFn:
    def lazy_retrieval_fn(key: ResultKey) -> Result[float, str]:
        try:
            symbolic_metric = provider.get_symbol(symbol)
        except DQXError as e:
            return Failure(f"Failed to resolve symbol {symbol}: {str(e)}")

        if symbolic_metric.dataset is None:
            return Failure(f"Dataset not imputed for metric {symbolic_metric.name}")

        return compute.simple_metric(
            provider._db,
            metric_spec,
            symbolic_metric.dataset,
            key,
            provider.execution_id  # ADD THIS
        )

    return lazy_retrieval_fn
```

#### Task 3.3: Update _create_lazy_extended_fn
**File**: `src/dqx/provider.py`
```python
def _create_lazy_extended_fn(
    provider: "MetricProvider",
    compute_fn: Callable[[MetricDB, MetricSpec, str, ResultKey, ExecutionId], Result[float, str]],  # UPDATE TYPE
    metric_spec: MetricSpec,
    symbol: sp.Symbol,
) -> RetrievalFn:
    def lazy_extended_fn(key: ResultKey) -> Result[float, str]:
        try:
            symbolic_metric = provider.get_symbol(symbol)
        except DQXError as e:
            return Failure(f"Failed to resolve symbol {symbol}: {str(e)}")

        if symbolic_metric.dataset is None:
            return Failure(f"Dataset not imputed for metric {symbolic_metric.name}")

        return compute_fn(
            provider._db,
            metric_spec,
            symbolic_metric.dataset,
            key,
            provider.execution_id  # ADD THIS
        )

    return lazy_extended_fn
```

#### Task 3.4: Update stddev lambda in ExtendedMetricProvider
**File**: `src/dqx/provider.py`
```python
# In ExtendedMetricProvider.stddev() method, update the lambda:
fn = _create_lazy_extended_fn(
    self._provider,
    lambda db, metric, dataset, key, execution_id: compute.stddev(
        db, metric, n, dataset, key, execution_id
    ),  # ADD execution_id parameter
    spec,
    sym
)
```

#### Task 3.5: Update ExtendedMetricProvider to expose execution_id
**File**: `src/dqx/provider.py`
```python
class ExtendedMetricProvider(RegistryMixin):
    # ... existing code ...

    @property
    def execution_id(self) -> ExecutionId:
        """The execution ID from the parent provider."""
        return self._provider.execution_id
```

#### Task 3.6: Update provider tests
**File**: `tests/test_provider.py`
```python
@pytest.fixture
def provider(mock_db: Mock) -> MetricProvider:
    """Create a MetricProvider instance with execution_id."""
    return MetricProvider(mock_db, execution_id="test-exec-123")

def test_execution_id_property(provider: MetricProvider) -> None:
    """Test execution_id property."""
    assert provider.execution_id == "test-exec-123"
```

**Verification**: Run `uv run pytest tests/test_provider.py -v` and ensure all tests pass.

### Task Group 4: VerificationSuite Integration

#### Task 4.1: Update VerificationSuite.run to pass execution_id
**File**: `src/dqx/api.py`

Verify that VerificationSuite creates MetricProvider with execution_id:
```python
def run(self, datasources: list[DataSource], key: ResultKey) -> None:
    # ... existing code ...

    # Create provider with execution_id
    mp = MetricProvider(db=self._db, execution_id=self._execution_id)

    # Rest remains the same...
```

#### Task 4.2: Create integration test
**File**: `tests/test_execution_id_integration.py`
```python
"""Integration tests for execution_id functionality."""
import datetime as dt
from dqx import check, VerificationSuite
from dqx.data import InMemoryDataSource
from dqx.orm.in_memory import InMemoryMetricDB
from dqx.common import ResultKey


def test_execution_isolation() -> None:
    """Test that metrics from different executions are properly isolated."""
    # Create test data
    data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    datasource = InMemoryDataSource(data)
    db = InMemoryMetricDB()

    # Create two suites with different execution_ids
    @check("Test Check")
    def test_check(mp, key):
        rows = mp.num_rows()
        return rows > 0

    suite1 = VerificationSuite([test_check], db, execution_id="exec-1")
    suite2 = VerificationSuite([test_check], db, execution_id="exec-2")

    # Run both on the same date
    key = ResultKey(yyyy_mm_dd=dt.date(2023, 1, 1), tags={})

    suite1.run([datasource], key)
    suite2.run([datasource], key)

    # Verify metrics are isolated
    # This would require adding a method to retrieve metrics by execution_id
    # For now, we just verify both suites ran successfully
    assert len(db._metrics) >= 2  # At least one metric per execution
```

**Verification**: Run `uv run pytest tests/test_execution_id_integration.py -v`

### Task Group 5: Final Verification

#### Task 5.1: Run all tests
```bash
uv run pytest tests/ -v
```

#### Task 5.2: Run linting and type checking
```bash
uv run mypy src/dqx
uv run ruff check --fix src/dqx tests/
```

#### Task 5.3: Run pre-commit hooks
```bash
uv run hooks
```

#### Task 5.4: Check test coverage
```bash
uv run pytest --cov=dqx tests/
```

## Success Criteria

1. All tests pass with execution_id parameter
2. No linting or type checking errors
3. Test coverage maintained or improved
4. Metrics are properly isolated by execution_id
5. No breaking changes to existing functionality

## Notes for Implementer

- Follow TDD: Write tests first, then implementation
- Use type annotations for all new parameters
- Update docstrings to mention execution_id filtering
- Commit after each task group completion
- If you encounter any issues, check existing patterns in the codebase

## Dependencies

No external dependencies. This change builds on existing DQX infrastructure.
