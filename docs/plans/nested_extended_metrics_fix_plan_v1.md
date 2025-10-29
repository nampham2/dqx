# Nested Extended Metrics Fix Implementation Plan v1

## Problem Summary

When using nested extended metrics (e.g., `stddev(day_over_day(average(tax)))`), the system raises a `TypeError: unhashable type: 'dict'`. This occurs because the `__hash__` method in extended metric classes attempts to hash the `parameters` property, which contains nested dictionaries for nested metrics.

## Root Cause Analysis

The issue stems from how extended metrics are implemented:

1. Extended metrics (DayOverDay, WeekOverWeek, Stddev) store their base metric as string type and parameters dict
2. The `__hash__` method tries to hash a tuple containing these parameters
3. For nested extended metrics, the parameters contain dictionaries, which are unhashable in Python
4. Example problematic structure:
   ```python
   # For stddev(day_over_day(average(tax))):
   parameters = {
       "base_metric_type": "DayOverDay",
       "base_parameters": {  # This nested dict causes the hash error
           "base_metric_type": "Average",
           "base_parameters": {"column": "tax"}
       },
       "lag": 1,
       "n": 7
   }
   ```

## Solution Overview

Refactor extended metric classes to store and use the actual base spec object internally while maintaining the current parameters format for database compatibility:

1. Keep the current constructor signature for database compatibility
2. Reconstruct and store the base spec object during initialization
3. Use the base spec object for hashing and equality operations
4. Maintain the `parameters` property for database serialization

## Implementation Tasks

### Task Group 1: Write Tests for Nested Extended Metrics (TDD)

**Task 1.1: Create test file for nested metrics**
```python
# Create tests/test_specs_nested.py
"""Tests for nested extended metric specifications."""
import pytest
from dqx import specs


class TestNestedExtendedMetrics:
    """Test extended metrics with nested base metrics."""

    def test_nested_dayoverday_hash(self) -> None:
        """Test that DayOverDay of DayOverDay can be hashed."""
        # Create base metric
        avg = specs.Average("price")

        # Create first level DoD
        dod1 = specs.DayOverDay.from_base_spec(avg)

        # Create nested DoD - this should not raise TypeError
        dod2 = specs.DayOverDay.from_base_spec(dod1)

        # Should be able to hash without error
        hash_value = hash(dod2)
        assert isinstance(hash_value, int)

    def test_stddev_of_dayoverday_hash(self) -> None:
        """Test that Stddev of DayOverDay can be hashed."""
        avg = specs.Average("tax")
        dod = specs.DayOverDay.from_base_spec(avg)
        stddev = specs.Stddev.from_base_spec(dod, lag=1, n=7)

        # This is the exact case from the failing test
        hash_value = hash(stddev)
        assert isinstance(hash_value, int)

    def test_deeply_nested_metrics(self) -> None:
        """Test 3+ levels of nesting."""
        # stddev(dod(wow(sum(revenue))))
        sum_spec = specs.Sum("revenue")
        wow = specs.WeekOverWeek.from_base_spec(sum_spec)
        dod = specs.DayOverDay.from_base_spec(wow)
        stddev = specs.Stddev.from_base_spec(dod, lag=1, n=14)

        # All operations should work
        assert hash(stddev) is not None
        assert stddev.name == "stddev(dod(wow(sum(revenue))), lag=1, n=14)"
```

**Task 1.2: Run tests to confirm they fail**
```bash
uv run pytest tests/test_specs_nested.py -v
# Should see TypeError: unhashable type: 'dict'
```

### Task Group 2: Refactor DayOverDay Class

**Task 2.1: Update DayOverDay to store base spec internally**
```python
# In src/dqx/specs.py, update DayOverDay class:

class DayOverDay:
    """Day-over-day change percentage metric."""

    metric_type: MetricType = "DayOverDay"
    is_extended: bool = True

    def __init__(self, base_metric_type: str, base_parameters: dict[str, Any]) -> None:
        """Initialize with backward-compatible parameters."""
        self._base_metric_type = base_metric_type
        self._base_parameters = base_parameters
        self._analyzers = ()

        # NEW: Reconstruct and store the base spec for internal operations
        metric_type_cast = typing.cast(MetricType, self._base_metric_type)
        self._base_spec = registry[metric_type_cast](**self._base_parameters)

    @classmethod
    def from_base_spec(cls, base_spec: MetricSpec) -> Self:
        """Create from a base metric spec."""
        return cls(
            base_metric_type=base_spec.metric_type,
            base_parameters=base_spec.parameters
        )

    @property
    def base_spec(self) -> MetricSpec:
        """Get the base metric specification."""
        return self._base_spec

    @property
    def name(self) -> str:
        """Get display name."""
        return f"dod({self._base_spec.name})"

    @property
    def parameters(self) -> Parameters:
        """Get parameters for database storage."""
        return {
            "base_metric_type": self._base_metric_type,
            "base_parameters": self._base_parameters,
        }

    @property
    def analyzers(self) -> Analyzers:
        """Extended metrics have no analyzers."""
        return self._analyzers

    def state(self) -> State:
        """Get initial state."""
        return states.SimpleAdditiveState(0.0)

    @staticmethod
    def deserialize(data: bytes) -> State:
        """Deserialize state from bytes."""
        return states.SimpleAdditiveState.deserialize(data)

    def __hash__(self) -> int:
        """Hash using the base spec object."""
        # NEW: Use base_spec which handles nested hashing properly
        return hash(("DayOverDay", self._base_spec))

    def __eq__(self, other: Any) -> bool:
        """Check equality using base spec."""
        if not isinstance(other, DayOverDay):
            return False
        # NEW: Compare base specs directly
        return self._base_spec == other._base_spec

    def __str__(self) -> str:
        """String representation."""
        return self.name
```

**Task 2.2: Run tests for DayOverDay**
```bash
uv run pytest tests/test_specs_nested.py::TestNestedExtendedMetrics::test_nested_dayoverday_hash -v
uv run pytest tests/test_specs.py::TestDayOverDay -v
```

**Task 2.3: Fix any linting issues**
```bash
uv run mypy src/dqx/specs.py
uv run ruff check --fix src/dqx/specs.py
```

### Task Group 3: Refactor WeekOverWeek Class

**Task 3.1: Update WeekOverWeek using same pattern**
```python
# In src/dqx/specs.py, update WeekOverWeek class:

class WeekOverWeek:
    """Week-over-week change percentage metric."""

    metric_type: MetricType = "WeekOverWeek"
    is_extended: bool = True

    def __init__(self, base_metric_type: str, base_parameters: dict[str, Any]) -> None:
        """Initialize with backward-compatible parameters."""
        self._base_metric_type = base_metric_type
        self._base_parameters = base_parameters
        self._analyzers = ()

        # Reconstruct and store the base spec
        metric_type_cast = typing.cast(MetricType, self._base_metric_type)
        self._base_spec = registry[metric_type_cast](**self._base_parameters)

    @classmethod
    def from_base_spec(cls, base_spec: MetricSpec) -> Self:
        """Create from a base metric spec."""
        return cls(
            base_metric_type=base_spec.metric_type,
            base_parameters=base_spec.parameters
        )

    @property
    def base_spec(self) -> MetricSpec:
        """Get the base metric specification."""
        return self._base_spec

    @property
    def name(self) -> str:
        """Get display name."""
        return f"wow({self._base_spec.name})"

    @property
    def parameters(self) -> Parameters:
        """Get parameters for database storage."""
        return {
            "base_metric_type": self._base_metric_type,
            "base_parameters": self._base_parameters,
        }

    @property
    def analyzers(self) -> Analyzers:
        """Extended metrics have no analyzers."""
        return self._analyzers

    def state(self) -> State:
        """Get initial state."""
        return states.SimpleAdditiveState(0.0)

    @staticmethod
    def deserialize(data: bytes) -> State:
        """Deserialize state from bytes."""
        return states.SimpleAdditiveState.deserialize(data)

    def __hash__(self) -> int:
        """Hash using the base spec object."""
        return hash(("WeekOverWeek", self._base_spec))

    def __eq__(self, other: Any) -> bool:
        """Check equality using base spec."""
        if not isinstance(other, WeekOverWeek):
            return False
        return self._base_spec == other._base_spec

    def __str__(self) -> str:
        """String representation."""
        return self.name
```

**Task 3.2: Add test for WeekOverWeek nesting**
```python
# Add to tests/test_specs_nested.py:
def test_nested_weekoverweek_hash(self) -> None:
    """Test that WeekOverWeek of extended metrics can be hashed."""
    min_spec = specs.Minimum("cost")
    dod = specs.DayOverDay.from_base_spec(min_spec)
    wow = specs.WeekOverWeek.from_base_spec(dod)

    hash_value = hash(wow)
    assert isinstance(hash_value, int)
    assert wow.name == "wow(dod(minimum(cost)))"
```

**Task 3.3: Run tests and fix linting**
```bash
uv run pytest tests/test_specs_nested.py -k "weekoverweek" -v
uv run pytest tests/test_specs.py::TestWeekOverWeek -v
uv run mypy src/dqx/specs.py
uv run ruff check --fix src/dqx/specs.py
```

### Task Group 4: Refactor Stddev Class

**Task 4.1: Update Stddev with additional parameters**
```python
# In src/dqx/specs.py, update Stddev class:

class Stddev:
    """Standard deviation metric over a time window."""

    metric_type: MetricType = "Stddev"
    is_extended: bool = True

    def __init__(
        self,
        base_metric_type: str,
        base_parameters: dict[str, Any],
        lag: int,
        n: int
    ) -> None:
        """Initialize with backward-compatible parameters."""
        self._base_metric_type = base_metric_type
        self._base_parameters = base_parameters
        self._lag = lag
        self._n = n
        self._analyzers = ()

        # Reconstruct and store the base spec
        metric_type_cast = typing.cast(MetricType, self._base_metric_type)
        self._base_spec = registry[metric_type_cast](**self._base_parameters)

    @classmethod
    def from_base_spec(cls, base_spec: MetricSpec, lag: int, n: int) -> Self:
        """Create from a base metric spec."""
        return cls(
            base_metric_type=base_spec.metric_type,
            base_parameters=base_spec.parameters,
            lag=lag,
            n=n
        )

    @property
    def base_spec(self) -> MetricSpec:
        """Get the base metric specification."""
        return self._base_spec

    @property
    def name(self) -> str:
        """Get display name."""
        return f"stddev({self._base_spec.name}, lag={self._lag}, n={self._n})"

    @property
    def parameters(self) -> Parameters:
        """Get parameters for database storage."""
        return {
            "base_metric_type": self._base_metric_type,
            "base_parameters": self._base_parameters,
            "lag": self._lag,
            "n": self._n,
        }

    @property
    def analyzers(self) -> Analyzers:
        """Extended metrics have no analyzers."""
        return self._analyzers

    def state(self) -> State:
        """Get initial state."""
        return states.SimpleAdditiveState(0.0)

    @staticmethod
    def deserialize(data: bytes) -> State:
        """Deserialize state from bytes."""
        return states.SimpleAdditiveState.deserialize(data)

    def __hash__(self) -> int:
        """Hash using the base spec object and window parameters."""
        return hash((
            "Stddev",
            self._base_spec,
            self._lag,
            self._n
        ))

    def __eq__(self, other: Any) -> bool:
        """Check equality using base spec and parameters."""
        if not isinstance(other, Stddev):
            return False
        return (
            self._base_spec == other._base_spec and
            self._lag == other._lag and
            self._n == other._n
        )

    def __str__(self) -> str:
        """String representation."""
        return self.name
```

**Task 4.2: Run all nested metric tests**
```bash
uv run pytest tests/test_specs_nested.py -v
uv run pytest tests/test_specs.py -k "Stddev or DayOverDay or WeekOverWeek" -v
```

### Task Group 5: Integration Testing

**Task 5.1: Test database round-trip**
```python
# Add to tests/test_specs_nested.py:
def test_nested_metric_database_roundtrip(self) -> None:
    """Test that nested metrics can be stored and retrieved."""
    from dqx.orm.repositories import InMemoryMetricDB
    from dqx.common import ResultKey
    from dqx import models
    import datetime as dt

    # Create nested metric
    avg = specs.Average("tax")
    dod = specs.DayOverDay.from_base_spec(avg)
    stddev = specs.Stddev.from_base_spec(dod, lag=1, n=7)

    # Store in database
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 10, 24), tags={})
    metric = models.Metric.build(
        spec=stddev,
        key=key,
        dataset="test_dataset",
        state=stddev.state()
    )

    persisted = list(db.persist([metric]))[0]

    # Retrieve and verify
    retrieved = db.get(persisted.metric_id).unwrap()
    assert retrieved.spec == stddev
    assert retrieved.spec.name == "stddev(dod(average(tax)), lag=1, n=7)"
```

**Task 5.2: Run the original failing e2e test**
```bash
uv run pytest tests/e2e/test_api_e2e.py::test_e2e_suite -v
```

**Task 5.3: Run full test suite**
```bash
uv run pytest tests/ -v
```

### Task Group 6: Final Verification

**Task 6.1: Check test coverage**
```bash
uv run pytest tests/ --cov=src/dqx/specs --cov-report=term-missing
```

**Task 6.2: Run all linting and type checks**
```bash
uv run mypy src/dqx/specs.py
uv run ruff check src/dqx/specs.py
```

**Task 6.3: Run pre-commit hooks**
```bash
uv run hooks
```

## Success Criteria

1. ✅ Nested extended metrics can be hashed without TypeError
2. ✅ All existing tests continue to pass
3. ✅ Database serialization/deserialization works correctly
4. ✅ The failing e2e test now passes
5. ✅ No regression in functionality
6. ✅ Type checking and linting pass
7. ✅ Test coverage maintained or improved

## Notes

- The solution maintains backward compatibility by keeping constructor signatures unchanged
- The `_base_spec` is reconstructed from stored parameters to ensure consistency
- The registry lookup happens during initialization, so all metric types must be registered
- This approach fixes the immediate issue while setting up for potential future improvements
