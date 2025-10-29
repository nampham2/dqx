# Extended Metrics Fix Implementation Plan v1

## Problem Summary

When using extended metrics that depend on other extended metrics (e.g., `stddev(day_over_day(average(tax)))`), the system fails to create all necessary dependency metrics. Specifically, when `stddev` creates its required metrics for the time window, it uses `provider.metric()` which only works for simple metrics and doesn't properly handle extended metrics like DayOverDay.

## Root Cause Analysis

The issue occurs in the extended metric methods (`day_over_day`, `week_over_week`, `stddev`) when they create their required dependencies:

```python
# In stddev method:
required = [self.provider.metric(spec, lag=i, dataset=symbolic_metric.dataset)
            for i in range(lag, lag + n)]
```

The problem is that `provider.metric()` is designed for simple metrics only. It:
1. Creates a symbol for the metric
2. Uses `compute.simple_metric()` for retrieval
3. Doesn't understand how to create extended metrics with their dependencies

When `spec` is a DayOverDay spec, this fails because:
- It doesn't call the appropriate `day_over_day()` method
- It doesn't create the necessary base metrics (e.g., `average(tax)` at lag and lag+1)
- The resulting symbol can't be computed

## Solution Overview

Add a unified `create_metric()` method to MetricProvider that:
1. Checks if the metric spec is simple or extended
2. Routes to the appropriate creation method
3. Ensures all transitive dependencies are created

This allows extended metric methods to create dependencies correctly, regardless of whether those dependencies are simple or extended metrics.

## Implementation Tasks

### Task Group 1: Write Tests for the Bug (TDD)

**Task 1.1: Create test for complex nested metrics**
```python
# Create tests/test_extended_metrics_integration.py
"""Integration tests for complex extended metric dependencies."""
import datetime as dt
from datetime import date

import pyarrow as pa
import pytest

from dqx import data, specs
from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_stddev_of_dod_creates_dependencies() -> None:
    """Test that stddev(dod(average(tax))) creates all necessary dependencies."""
    # Setup
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec-123")

    # Create the complex metric
    avg_tax = provider.average("tax")
    dod_avg_tax = provider.ext.day_over_day(avg_tax)
    stddev_dod_avg_tax = provider.ext.stddev(dod_avg_tax, lag=0, n=7)

    # Verify the stddev metric has correct dependencies
    stddev_metric = provider.get_symbol(stddev_dod_avg_tax)
    assert len(stddev_metric.required_metrics) == 7  # 7 days of DoD values

    # Create test data for 10 days
    tables = []
    for day in range(1, 11):
        table = pa.table({
            "tax": [100.0 + day, 200.0 + day, 300.0 + day],
            "date": [f"2024-01-{day:02d}"] * 3
        })
        tables.append((date(2024, 1, day), table))

    # Create check
    @check(name="Tax DoD Stddev Check")
    def tax_dod_stddev_check(mp: MetricProvider, ctx: Context) -> None:
        avg = mp.average("tax")
        dod = mp.ext.day_over_day(avg)
        std = mp.ext.stddev(dod, lag=0, n=7)
        ctx.assert_that(std).where(name="Collect stddev").noop()

    # Run suite for each day
    for key_date, table_data in tables:
        suite = VerificationSuite([tax_dod_stddev_check], db, "Test Suite")
        datasource = DuckRelationDataSource.from_arrow(table_data, "tax_data")
        key = ResultKey(yyyy_mm_dd=key_date, tags={"test": "stddev_dod"})
        suite.run([datasource], key)

    # Verify all required metrics exist in DB
    # Should have average(tax) for days 1-9 (to compute DoD for days 2-9)
    for day in range(1, 10):
        key_day = ResultKey(yyyy_mm_dd=date(2024, 1, day), tags={"test": "stddev_dod"})
        avg_metric = db.get(key_day, specs.Average("tax"))
        assert avg_metric.is_ok(), f"Missing average(tax) for day {day}"

    # Verify DoD metrics exist for days 2-9
    for day in range(2, 10):
        key_day = ResultKey(yyyy_mm_dd=date(2024, 1, day), tags={"test": "stddev_dod"})
        dod_spec = specs.DayOverDay.from_base_spec(specs.Average("tax"))
        dod_metric = db.get(key_day, dod_spec)
        assert dod_metric.is_ok(), f"Missing dod(average(tax)) for day {day}"

    # Verify the final stddev metric exists
    final_key = ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={"test": "stddev_dod"})
    stddev_spec = specs.Stddev.from_base_spec(
        specs.DayOverDay.from_base_spec(specs.Average("tax")),
        lag=0, n=7
    )
    final_metric = db.get(final_key, stddev_spec)
    assert final_metric.is_ok()


def test_nested_extended_metrics_combinations() -> None:
    """Test various combinations of nested extended metrics."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, "test-exec-124")

    # Test 1: WoW of DoD
    avg = provider.average("revenue")
    dod = provider.ext.day_over_day(avg)
    wow_dod = provider.ext.week_over_week(dod)

    wow_metric = provider.get_symbol(wow_dod)
    assert len(wow_metric.required_metrics) == 2  # DoD at lag 0 and lag 7

    # Test 2: DoD of WoW
    sum_metric = provider.sum("cost")
    wow = provider.ext.week_over_week(sum_metric)
    dod_wow = provider.ext.day_over_day(wow)

    dod_metric = provider.get_symbol(dod_wow)
    assert len(dod_metric.required_metrics) == 2  # WoW at lag 0 and lag 1

    # Test 3: Stddev of WoW
    min_metric = provider.minimum("price")
    wow_min = provider.ext.week_over_week(min_metric)
    stddev_wow = provider.ext.stddev(wow_min, lag=0, n=5)

    stddev_metric = provider.get_symbol(stddev_wow)
    assert len(stddev_metric.required_metrics) == 5  # 5 days of WoW values
```

**Task 1.2: Run tests to confirm they fail**
```bash
uv run pytest tests/test_extended_metrics_integration.py -v
# Should fail with missing dependencies
```

### Task Group 2: Implement create_metric Method

**Task 2.1: Add create_metric to MetricProvider**
```python
# In src/dqx/provider.py, add to MetricProvider class:

def create_metric(self, spec: MetricSpec, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
    """Unified metric creation that handles both simple and extended metrics.

    This method examines the metric spec and routes to the appropriate
    creation method, ensuring all dependencies are properly created.

    Args:
        spec: The metric specification
        lag: The lag to apply (in days)
        dataset: Optional dataset name

    Returns:
        Symbol representing the metric
    """
    if not spec.is_extended:
        # Simple metrics use the existing metric() method
        return self.metric(spec, lag=lag, dataset=dataset)

    # Handle extended metrics based on their type
    if spec.metric_type == "DayOverDay":
        dod_spec = typing.cast(specs.DayOverDay, spec)
        # Create base metric first (recursively handles nested extended metrics)
        base_metric = self.create_metric(dod_spec.base_spec, lag=lag, dataset=dataset)
        return self.ext.day_over_day(base_metric, dataset=dataset)

    elif spec.metric_type == "WeekOverWeek":
        wow_spec = typing.cast(specs.WeekOverWeek, spec)
        base_metric = self.create_metric(wow_spec.base_spec, lag=lag, dataset=dataset)
        return self.ext.week_over_week(base_metric, dataset=dataset)

    elif spec.metric_type == "Stddev":
        stddev_spec = typing.cast(specs.Stddev, spec)
        # For stddev, create the base metric without additional lag
        # as stddev manages its own lag internally
        base_metric = self.create_metric(stddev_spec.base_spec, lag=0, dataset=dataset)
        return self.ext.stddev(base_metric, lag=stddev_spec._lag + lag, n=stddev_spec._n, dataset=dataset)

    else:
        raise ValueError(f"Unknown extended metric type: {spec.metric_type}")
```

**Task 2.2: Add required import**
```python
# At the top of src/dqx/provider.py, ensure typing is imported:
import typing
```

**Task 2.3: Run type checking and fix any issues**
```bash
uv run mypy src/dqx/provider.py
uv run ruff check --fix src/dqx/provider.py
```

### Task Group 3: Update Extended Metric Methods

**Task 3.1: Update day_over_day method**
```python
# In src/dqx/provider.py, in ExtendedMetricProvider class, update day_over_day:

def day_over_day(self, metric: sp.Symbol, dataset: str | None = None) -> sp.Symbol:
    """Create day-over-day metric."""
    symbolic_metric = self._provider.get_symbol(metric)
    spec = symbolic_metric.metric_spec

    # Use create_metric instead of metric for proper handling
    lag_0 = self.provider.create_metric(spec, lag=0, dataset=symbolic_metric.dataset)
    lag_1 = self.provider.create_metric(spec, lag=1, dataset=symbolic_metric.dataset)

    # Rest of the method remains unchanged...
    # (Keep the existing symbol generation and registration logic)
```

**Task 3.2: Update week_over_week method**
```python
# In src/dqx/provider.py, in ExtendedMetricProvider class, update week_over_week:

def week_over_week(self, metric: sp.Symbol, dataset: str | None = None) -> sp.Symbol:
    """Create week-over-week metric."""
    symbolic_metric = self._provider.get_symbol(metric)
    spec = symbolic_metric.metric_spec

    # Use create_metric instead of metric for proper handling
    lag_0 = self.provider.create_metric(spec, lag=0, dataset=symbolic_metric.dataset)
    lag_7 = self.provider.create_metric(spec, lag=7, dataset=symbolic_metric.dataset)

    # Rest of the method remains unchanged...
```

**Task 3.3: Update stddev method**
```python
# In src/dqx/provider.py, in ExtendedMetricProvider class, update stddev:

def stddev(self, metric: sp.Symbol, lag: int, n: int, dataset: str | None = None) -> sp.Symbol:
    """Create standard deviation metric."""
    symbolic_metric = self._provider.get_symbol(metric)
    spec = symbolic_metric.metric_spec

    # Use create_metric for proper handling of extended metrics
    required = []
    for i in range(lag, lag + n):
        required_metric = self.provider.create_metric(spec, lag=i, dataset=symbolic_metric.dataset)
        required.append(required_metric)

    # Rest of the method remains unchanged...
```

### Task Group 4: Unit Tests for create_metric

**Task 4.1: Add unit tests to test_provider.py**
```python
# Add to tests/test_provider.py in TestMetricProvider class:

def test_create_metric_simple(self, provider: MetricProvider) -> None:
    """Test create_metric with simple metric specs."""
    # Test with Average spec
    avg_spec = specs.Average("revenue")
    symbol = provider.create_metric(avg_spec, lag=3, dataset="test_ds")

    assert isinstance(symbol, sp.Symbol)
    metric = provider.get_symbol(symbol)
    assert metric.lag == 3
    assert metric.dataset == "test_ds"
    assert metric.metric_spec == avg_spec

def test_create_metric_day_over_day(self, provider: MetricProvider) -> None:
    """Test create_metric with DayOverDay spec."""
    base_spec = specs.Average("revenue")
    dod_spec = specs.DayOverDay.from_base_spec(base_spec)

    symbol = provider.create_metric(dod_spec, lag=2, dataset="test_ds")

    # Verify DoD metric was created
    metric = provider.get_symbol(symbol)
    assert isinstance(metric.metric_spec, specs.DayOverDay)

    # Verify dependencies were created (base at lag 2 and lag 3)
    assert len(metric.required_metrics) == 2

def test_create_metric_nested_extended(self, provider: MetricProvider) -> None:
    """Test create_metric with nested extended metrics (stddev of DoD)."""
    base_spec = specs.Average("revenue")
    dod_spec = specs.DayOverDay.from_base_spec(base_spec)
    stddev_spec = specs.Stddev.from_base_spec(dod_spec, lag=0, n=7)

    symbol = provider.create_metric(stddev_spec, lag=0)

    # Verify the full dependency chain was created
    metric = provider.get_symbol(symbol)
    assert isinstance(metric.metric_spec, specs.Stddev)
    assert len(metric.required_metrics) == 7  # 7 days of DoD values

def test_create_metric_unknown_type(self, provider: MetricProvider) -> None:
    """Test create_metric with unknown extended metric type."""
    # Create a mock spec with unknown type
    mock_spec = Mock(spec=specs.MetricSpec)
    mock_spec.is_extended = True
    mock_spec.metric_type = "UnknownType"

    with pytest.raises(ValueError, match="Unknown extended metric type: UnknownType"):
        provider.create_metric(mock_spec)
```

**Task 4.2: Run all unit tests**
```bash
uv run pytest tests/test_provider.py -v
```

### Task Group 5: Integration Testing and Verification

**Task 5.1: Run the integration tests**
```bash
uv run pytest tests/test_extended_metrics_integration.py -v
```

**Task 5.2: Run the original failing test**
```bash
uv run pytest tests/e2e/test_api_e2e.py::test_e2e_suite -k "stddev" -v
```

**Task 5.3: Run existing extended metric tests**
```bash
uv run pytest tests/test_dod_integration.py -v
uv run pytest tests/ -k "extended or dod or wow or stddev" -v
```

### Task Group 6: Final Verification

**Task 6.1: Check test coverage**
```bash
uv run pytest tests/ --cov=src/dqx/provider --cov-report=term-missing
```

**Task 6.2: Run all linting and type checks**
```bash
uv run mypy src/dqx/provider.py
uv run ruff check src/dqx/provider.py
```

**Task 6.3: Run pre-commit hooks**
```bash
uv run hooks
```

**Task 6.4: Run full test suite**
```bash
uv run pytest tests/ -v
```

## Success Criteria

1. ✅ Complex nested metrics like `stddev(dod(average(tax)))` work correctly
2. ✅ All required dependency metrics are created automatically
3. ✅ All existing tests continue to pass
4. ✅ No regression in functionality
5. ✅ Type checking and linting pass
6. ✅ Test coverage maintained or improved
7. ✅ The solution is extensible for future metric types

## Notes

- The solution maintains backward compatibility by not changing existing method signatures
- The `create_metric` method provides a single entry point for metric creation
- The recursive nature of `create_metric` handles arbitrary nesting levels
- The type-based dispatch is contained in one method for maintainability
- Future extended metric types just need to be added to `create_metric`

## Future Improvements

While the current solution uses type-based dispatch, future improvements could include:
- Adding a `create_lagged` method to MetricSpec protocol
- Using a registry pattern for extended metric creation
- Refactoring to use a visitor pattern for metric creation

These improvements can be considered once the immediate issue is resolved and the pattern is proven to work well.
