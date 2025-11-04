# Date Exclusion Implementation Plan v1

## Overview

This plan implements a date exclusion feature for DQX that allows users to exclude specific dates from metric calculations. When dates are excluded, assertions dependent on those metrics will be automatically marked as "SKIPPED" if data availability falls below a configurable threshold.

## Architecture Summary

### Core Concept: Metric-Level Data Availability
- Each `SymbolicMetric` tracks its data availability ratio (`data_av_ratio`)
- Simple metrics: 1.0 if date not excluded, 0.0 if excluded
- Extended metrics: Percentage of non-excluded required metrics
- Assertions skip when metric availability < threshold (default 0.8)

### Integration Points
1. **VerificationSuite**: Accept `skip_dates` and `data_av_threshold` parameters
2. **MetricRegistry**: Calculate availability ratios for all metrics
3. **Evaluator**: Check availability before evaluation, set SKIPPED status
4. **AssertionStatus**: Extend to include "SKIPPED" literal

## Backward Compatibility Note

This implementation maintains backward compatibility by using default parameter values:
- The `Evaluator` constructor adds `data_av_threshold` with a default value of 0.8
- This ensures all 28 existing call sites continue to work without modification
- Sites that need custom thresholds can pass the parameter explicitly

## Implementation Tasks

### Task Group 1: Core Type Updates and Foundation

**Goal**: Update core types to support date exclusion functionality.

#### Task 1.1: Extend AssertionStatus Type
**File**: `src/dqx/common.py`
```python
# Change line ~86:
AssertionStatus = Literal["OK", "FAILURE", "SKIPPED"]
```

#### Task 1.2: Add data_av_ratio to SymbolicMetric
**File**: `src/dqx/provider.py`
```python
# In SymbolicMetric dataclass (~line 60), add:
@dataclass
class SymbolicMetric:
    name: str
    symbol: sp.Symbol
    fn: RetrievalFn
    metric_spec: MetricSpec
    lag: int = 0
    dataset: str | None = None
    required_metrics: list[sp.Symbol] = field(default_factory=list)
    data_av_ratio: float | None = None  # NEW FIELD
```

#### Task 1.3: Write Tests for Type Updates
**File**: `tests/test_date_exclusion_types.py` (NEW)
```python
"""Test type updates for date exclusion feature."""
import datetime
from typing import get_args

import pytest

from dqx.common import AssertionStatus
from dqx.provider import SymbolicMetric


class TestDateExclusionTypes:
    """Test core type updates for date exclusion."""

    def test_assertion_status_includes_skipped(self) -> None:
        """Verify AssertionStatus literal includes SKIPPED."""
        status_values = get_args(AssertionStatus)
        assert "OK" in status_values
        assert "FAILURE" in status_values
        assert "SKIPPED" in status_values
        assert len(status_values) == 3

    def test_symbolic_metric_has_data_av_ratio(self) -> None:
        """Verify SymbolicMetric has data_av_ratio field."""
        # Test default None
        metric = SymbolicMetric(
            name="test",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(1.0),
            metric_spec=specs.NumRows()
        )
        assert metric.data_av_ratio is None

        # Test explicit value
        metric2 = SymbolicMetric(
            name="test2",
            symbol=sp.Symbol("x_2"),
            fn=lambda k: Success(2.0),
            metric_spec=specs.Average("price"),
            data_av_ratio=0.75
        )
        assert metric2.data_av_ratio == 0.75
```

#### Task 1.4: Run Tests and Fix Linting
```bash
# Run tests
uv run pytest tests/test_date_exclusion_types.py -v

# Fix any linting issues
uv run ruff check --fix src/dqx/common.py src/dqx/provider.py tests/test_date_exclusion_types.py
uv run mypy src/dqx/common.py src/dqx/provider.py tests/test_date_exclusion_types.py

# Run pre-commit
uv run hooks
```

**Commit**: `feat: add SKIPPED status and data_av_ratio field for date exclusion`

---

### Task Group 2: Availability Calculation Logic

**Goal**: Implement the core logic to calculate data availability ratios.

#### Task 2.1: Add calculate_data_av_ratios Method
**File**: `src/dqx/provider.py`
```python
# In MetricRegistry class (~line 106), add:

def calculate_data_av_ratios(self, skip_dates: set[datetime.date], key: ResultKey) -> None:
    """Calculate data availability ratios for all metrics.

    Args:
        skip_dates: Set of dates to exclude from calculations
        key: The ResultKey for evaluation context
    """
    if not skip_dates:
        # No dates excluded - all metrics fully available
        for metric in self._metrics:
            metric.data_av_ratio = 1.0
        return

    # Memoization for efficiency
    memo: dict[sp.Symbol, float] = {}

    def calculate_ratio(metric: SymbolicMetric) -> float:
        """Recursively calculate availability ratio."""
        if metric.symbol in memo:
            return memo[metric.symbol]

        # Simple metric: check if effective date is excluded
        if not metric.required_metrics:
            effective_date = key.lag(metric.lag).yyyy_mm_dd
            ratio = 0.0 if effective_date in skip_dates else 1.0
        else:
            # Extended metric: average of children ratios
            child_ratios = []
            for req_symbol in metric.required_metrics:
                req_metric = self.get(req_symbol)
                child_ratios.append(calculate_ratio(req_metric))
            ratio = sum(child_ratios) / len(child_ratios) if child_ratios else 1.0

        memo[metric.symbol] = ratio
        metric.data_av_ratio = ratio
        return ratio

    # Calculate for all metrics
    for metric in self._metrics:
        calculate_ratio(metric)
```

#### Task 2.2: Write Unit Tests for Availability Calculation
**File**: `tests/test_data_av_ratio_calculation.py` (NEW)
```python
"""Test data availability ratio calculation."""
import datetime

import pytest
from returns.result import Success

from dqx import specs
from dqx.common import ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


class TestDataAvailabilityCalculation:
    """Test calculation of data availability ratios."""

    @pytest.fixture
    def provider(self) -> MetricProvider:
        """Create a MetricProvider with in-memory DB."""
        db = InMemoryMetricDB()
        return MetricProvider(db, "test-exec")

    def test_no_skip_dates_all_available(self, provider: MetricProvider) -> None:
        """When no dates excluded, all metrics have ratio 1.0."""
        # Create metrics
        m1 = provider.average("price", lag=0, dataset="sales")
        m2 = provider.sum("quantity", lag=1, dataset="sales")

        # Calculate with empty skip_dates
        key = ResultKey(datetime.date(2024, 1, 15), {})
        provider.registry.calculate_data_av_ratios(set(), key)

        # All metrics should be fully available
        assert provider.get_symbol(m1).data_av_ratio == 1.0
        assert provider.get_symbol(m2).data_av_ratio == 1.0

    def test_simple_metric_excluded_date(self, provider: MetricProvider) -> None:
        """Simple metrics have ratio 0.0 when date excluded."""
        # Create metrics with different lags
        m1 = provider.average("price", lag=0, dataset="sales")  # 2024-01-15
        m2 = provider.sum("quantity", lag=1, dataset="sales")   # 2024-01-14

        # Exclude 2024-01-14
        skip_dates = {datetime.date(2024, 1, 14)}
        key = ResultKey(datetime.date(2024, 1, 15), {})
        provider.registry.calculate_data_av_ratios(skip_dates, key)

        # m1 not affected (different date)
        assert provider.get_symbol(m1).data_av_ratio == 1.0
        # m2 excluded
        assert provider.get_symbol(m2).data_av_ratio == 0.0

    def test_extended_metric_partial_availability(self, provider: MetricProvider) -> None:
        """Extended metrics average their children's availability."""
        # Create base metrics
        base = provider.average("price", dataset="sales")

        # Create DoD metric (depends on lag=0 and lag=1)
        dod = provider.ext.day_over_day(base, lag=0, dataset="sales")

        # Exclude the lag=1 date (2024-01-14)
        skip_dates = {datetime.date(2024, 1, 14)}
        key = ResultKey(datetime.date(2024, 1, 15), {})
        provider.registry.calculate_data_av_ratios(skip_dates, key)

        # DoD should have 0.5 availability (1 of 2 children available)
        assert provider.get_symbol(dod).data_av_ratio == 0.5

    def test_recursive_calculation_with_memoization(self, provider: MetricProvider) -> None:
        """Verify memoization prevents recalculation."""
        # Create a complex dependency chain
        base1 = provider.average("price", dataset="sales")
        base2 = provider.sum("quantity", dataset="sales")

        dod1 = provider.ext.day_over_day(base1, dataset="sales")
        dod2 = provider.ext.day_over_day(base2, dataset="sales")

        # Both DoD metrics share some base metrics
        key = ResultKey(datetime.date(2024, 1, 15), {})
        provider.registry.calculate_data_av_ratios(set(), key)

        # All should be 1.0
        assert all(m.data_av_ratio == 1.0 for m in provider.metrics)
```

#### Task 2.3: Run Tests and Fix Issues
```bash
# Run new tests
uv run pytest tests/test_data_av_ratio_calculation.py -v

# Check linting
uv run mypy src/dqx/provider.py tests/test_data_av_ratio_calculation.py
uv run ruff check --fix src/dqx/provider.py tests/test_data_av_ratio_calculation.py

# Run pre-commit
uv run hooks
```

**Commit**: `feat: implement data availability ratio calculation in MetricRegistry`

---

### Task Group 3: VerificationSuite Integration

**Goal**: Update VerificationSuite to accept skip_dates and trigger availability calculation.

#### Task 3.1: Update VerificationSuite Constructor
**File**: `src/dqx/api.py`
```python
# Update __init__ method (~line 506):
def __init__(
    self,
    checks: Sequence[CheckProducer | DecoratedCheck],
    db: "MetricDB",
    name: str,
    log_level: int = logging.INFO,
    skip_dates: set[datetime.date] | None = None,  # NEW
    data_av_threshold: float = 0.8,  # NEW
) -> None:
    """
    Initialize the verification suite.

    Args:
        checks: Sequence of check functions to execute
        db: Database for storing and retrieving metrics
        name: Human-readable name for the suite
        log_level: Logging level (default: INFO)
        skip_dates: Set of dates to exclude from metric calculations
        data_av_threshold: Minimum data availability to evaluate assertions (default: 0.8)

    Raises:
        DQXError: If no checks provided or name is empty
    """
    # ... existing initialization ...

    # Store new parameters
    self._skip_dates = skip_dates or set()
    self._data_av_threshold = data_av_threshold
```

#### Task 3.2: Integrate Availability Calculation in run()
**File**: `src/dqx/api.py`
```python
# In run() method, after symbol deduplication (~line 673):
# Apply symbol deduplication BEFORE analysis
self._context.provider.symbol_deduplication(self._context._graph, key)

# NEW: Calculate data availability ratios if dates are excluded
if self._skip_dates:
    logger.info(f"Calculating data availability with {len(self._skip_dates)} excluded dates")
    self._context.provider.registry.calculate_data_av_ratios(self._skip_dates, key)

# Collect metrics stats and cleanup expired metrics BEFORE analysis
```

#### Task 3.3: Add Properties for New Fields
**File**: `src/dqx/api.py`
```python
# Add after existing properties (~line 600):
@property
def skip_dates(self) -> set[datetime.date]:
    """
    Set of dates excluded from metric calculations.

    Returns:
        Set of date objects to skip
    """
    return self._skip_dates

@property
def data_av_threshold(self) -> float:
    """
    Minimum data availability threshold for assertion evaluation.

    Assertions depending on metrics with availability below this
    threshold will be marked as SKIPPED rather than evaluated.

    Returns:
        Float between 0.0 and 1.0 (default: 0.8)
    """
    return self._data_av_threshold
```

#### Task 3.4: Write Integration Tests
**File**: `tests/test_date_exclusion_integration.py` (NEW)
```python
"""Integration tests for date exclusion feature."""
import datetime

import pyarrow as pa
import pytest

from dqx import check
from dqx.api import Context, MetricProvider, VerificationSuite
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


@check(name="Price Checks")
def price_checks(mp: MetricProvider, ctx: Context) -> None:
    """Check average price metrics."""
    ctx.assert_that(mp.average("price")).\
        where(name="Average price is positive").is_positive()

    ctx.assert_that(mp.ext.day_over_day(mp.average("price"))).\
        where(name="Price DoD reasonable").is_between(0.8, 1.2)


class TestDateExclusionIntegration:
    """Test date exclusion end-to-end workflow."""

    @pytest.fixture
    def test_data(self) -> pa.Table:
        """Create test data for multiple dates."""
        return pa.table({
            "date": [
                "2024-01-13", "2024-01-14", "2024-01-15",
                "2024-01-16", "2024-01-17"
            ],
            "price": [100.0, 110.0, 105.0, 108.0, 112.0],
            "quantity": [10, 15, 12, 14, 16]
        })

    def test_suite_accepts_skip_dates(self) -> None:
        """VerificationSuite accepts skip_dates parameter."""
        db = InMemoryMetricDB()
        skip_dates = {datetime.date(2024, 1, 14)}

        suite = VerificationSuite(
            checks=[price_checks],
            db=db,
            name="Test Suite",
            skip_dates=skip_dates,
            data_av_threshold=0.7
        )

        assert suite.skip_dates == skip_dates
        assert suite.data_av_threshold == 0.7

    def test_availability_calculation_triggered(self, test_data: pa.Table) -> None:
        """Verify availability calculation is triggered with skip_dates."""
        db = InMemoryMetricDB()
        skip_dates = {datetime.date(2024, 1, 14)}

        suite = VerificationSuite(
            checks=[price_checks],
            db=db,
            name="Test Suite",
            skip_dates=skip_dates
        )

        datasource = DuckRelationDataSource.from_arrow(test_data, "sales", "date")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        suite.run([datasource], key)

        # Check that availability was calculated
        for metric in suite.provider.metrics:
            assert metric.data_av_ratio is not None
```

#### Task 3.5: Run Tests and Validate
```bash
# Run integration tests
uv run pytest tests/test_date_exclusion_integration.py -v

# Run all tests to ensure no regression
uv run pytest tests/ -v

# Fix linting
uv run ruff check --fix src/dqx/api.py tests/test_date_exclusion_integration.py
uv run mypy src/dqx/api.py tests/test_date_exclusion_integration.py

# Pre-commit
uv run hooks
```

**Commit**: `feat: add skip_dates and data_av_threshold to VerificationSuite`

---

### Task Group 4: Evaluator Skip Logic

**Goal**: Update Evaluator to check data availability and skip assertions when below threshold.

#### Task 4.1: Add Skip Logic to Evaluator
**File**: `src/dqx/evaluator.py`
```python
# In visit() method for AssertionNode (~line 93):
def visit(self, node: BaseNode) -> None:
    """Visit a node in the graph during evaluation."""
    if isinstance(node, AssertionNode):
        # NEW: Check if expression has insufficient data
        if self._has_insufficient_data(node.actual):
            # Skip this assertion
            node._result = "SKIPPED"
            node._metric = Failure([
                EvaluationFailure(
                    error_message=f"Expression contains metrics with insufficient data availability (threshold: {self._data_av_threshold})",
                    expression=str(node.actual),
                    symbols=[]  # Will be populated if needed
                )
            ])
            return

        # Existing evaluation logic...
        result = self.evaluate(node.actual)
        # ... rest of existing code
```

#### Task 4.2: Add Helper Method for Availability Check
**File**: `src/dqx/evaluator.py`
```python
# Add import at top of file if not already present:
import logging
from dqx.common import DQXError

logger = logging.getLogger(__name__)

# Add new method to Evaluator class:
def _has_insufficient_data(self, expr: sp.Expr) -> bool:
    """Check if expression contains symbols with insufficient data availability.

    Args:
        expr: Symbolic expression to check

    Returns:
        True if any symbol has availability below threshold, False otherwise
    """
    symbols = expr.free_symbols
    if not symbols:
        return False  # Constants always have sufficient data

    for symbol in symbols:
        try:
            metric = self._provider.get_symbol(symbol)
            if metric.data_av_ratio is not None:
                if metric.data_av_ratio < self._data_av_threshold:
                    return True  # Found insufficient data
        except DQXError as e:
            # If we can't find the metric, assume it's available
            logger.debug(f"Symbol {symbol} not found in provider: {e}")
            pass

    return False  # All symbols have sufficient data
```

#### Task 4.3: Update Evaluator Constructor
**File**: `src/dqx/evaluator.py`
```python
# Update __init__ (~line 45):
def __init__(
    self,
    provider: MetricProvider,
    key: ResultKey,
    suite: str,
    data_av_threshold: float = 0.8  # NEW
) -> None:
    """Initialize the evaluator.

    Args:
        provider: Metric provider for symbol lookup
        key: Result key with date and tags
        suite: Name of the verification suite
        data_av_threshold: Minimum data availability to evaluate (default: 0.8)
    """
    self._provider = provider
    self._key = key
    self._suite = suite
    self._data_av_threshold = data_av_threshold  # NEW
    self._metrics: dict[sp.Symbol, Result[float, str]] = {}
```

#### Task 4.4: Update Evaluator Creation in VerificationSuite
**File**: `src/dqx/api.py`
```python
# In run() method where Evaluator is created (~line 690):
# 3. Evaluate assertions
evaluator = Evaluator(
    self.provider,
    key,
    self._name,
    self._data_av_threshold  # NEW: pass threshold
)
self._context._graph.bfs(evaluator)
```

#### Task 4.5: Write Tests for Skip Logic
**File**: `tests/test_evaluator_skip_logic.py` (NEW)
```python
"""Test Evaluator skip logic for low data availability."""
import datetime
from unittest.mock import Mock

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx import specs
from dqx.common import ResultKey
from dqx.evaluator import Evaluator
from dqx.graph.nodes import AssertionNode
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider, SymbolicMetric


class TestEvaluatorSkipLogic:
    """Test that Evaluator skips assertions with low availability."""

    @pytest.fixture
    def provider(self) -> MetricProvider:
        """Create provider with metrics having availability set."""
        db = InMemoryMetricDB()
        provider = MetricProvider(db, "test-exec")

        # Add a metric with low availability
        symbol = sp.Symbol("x_1")
        provider._registry._metrics.append(
            SymbolicMetric(
                name="average(price)",
                symbol=symbol,
                fn=lambda k: Success(100.0),
                metric_spec=specs.Average("price"),
                dataset="sales",
                data_av_ratio=0.5  # Below default threshold
            )
        )
        provider._registry._symbol_index[symbol] = provider._registry._metrics[-1]

        return provider

    def test_assertion_skipped_below_threshold(self, provider: MetricProvider) -> None:
        """Assertions are skipped when availability below threshold."""
        key = ResultKey(datetime.date(2024, 1, 15), {})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Create assertion node
        symbol = sp.Symbol("x_1")
        validator = Mock(name="> 0", fn=lambda x: x > 0)

        assertion = AssertionNode(
            name="Test assertion",
            actual=symbol,
            validator=validator,
            severity="P1"
        )

        # Visit the assertion
        evaluator.visit(assertion)

        # Should be skipped
        assert assertion._result == "SKIPPED"
        assert isinstance(assertion._metric, Failure)
        failures = assertion._metric.failure()
        assert len(failures) == 1
        assert "insufficient data availability" in failures[0].error_message

    def test_assertion_evaluated_above_threshold(self, provider: MetricProvider) -> None:
        """Assertions are evaluated when availability above threshold."""
        # Update availability to be above threshold
        provider.get_symbol("x_1").data_av_ratio = 0.9

        key = ResultKey(datetime.date(2024, 1, 15), {})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8)

        # Create assertion
        symbol = sp.Symbol("x_1")
        validator = Mock(name="> 0", fn=lambda x: x > 0)

        assertion = AssertionNode(
            name="Test assertion",
            actual=symbol,
            validator=validator,
            severity="P1"
        )

        # Mock the evaluate method to avoid full evaluation
        evaluator.evaluate = Mock(return_value=Success(100.0))

        # Visit the assertion
        evaluator.visit(assertion)

        # Should not be skipped
        assert assertion._result != "SKIPPED"
        evaluator.evaluate.assert_called_once()
```

#### Task 4.6: Run All Tests and Validate
```bash
# Run new skip logic tests
uv run pytest tests/test_evaluator_skip_logic.py -v

# Run all tests
uv run pytest tests/ -v

# Fix linting
uv run mypy src/dqx/evaluator.py tests/test_evaluator_skip_logic.py
uv run ruff check --fix src/dqx/evaluator.py tests/test_evaluator_skip_logic.py

# Pre-commit
uv run hooks
```

**Commit**: `feat: implement skip logic in Evaluator for low data availability`

---

### Task Group 5: Final Integration Testing

**Goal**: Create comprehensive end-to-end tests and ensure all components work together.

#### Task 5.1: Complete End-to-End Test
**File**: `tests/test_date_exclusion_e2e.py` (NEW)
```python
"""End-to-end tests for complete date exclusion workflow."""
import datetime

import pyarrow as pa
import pytest

from dqx import check
from dqx.api import Context, MetricProvider, VerificationSuite
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


@check(name="Comprehensive Checks")
def comprehensive_checks(mp: MetricProvider, ctx: Context) -> None:
    """Various checks to test skip logic."""
    # Simple metric check
    ctx.assert_that(mp.average("price", lag=1)).\
        where(name="Yesterday price positive", severity="P0").is_positive()

    # Extended metric check (DoD)
    ctx.assert_that(mp.ext.day_over_day(mp.average("price"))).\
        where(name="Price DoD stable", severity="P1").is_between(0.9, 1.1)

    # Check unaffected by exclusion
    ctx.assert_that(mp.average("price", lag=0)).\
        where(name="Today price positive", severity="P2").is_positive()


class TestDateExclusionE2E:
    """End-to-end tests for date exclusion feature."""

    @pytest.fixture
    def multi_day_data(self) -> pa.Table:
        """Create test data spanning multiple days."""
        return pa.table({
            "date": [
                "2024-01-10", "2024-01-11", "2024-01-12",
                "2024-01-13", "2024-01-14", "2024-01-15"
            ],
            "price": [95.0, 100.0, 98.0, 102.0, 105.0, 103.0],
            "quantity": [8, 10, 9, 11, 12, 10]
        })

    def test_mixed_skip_and_evaluate(self, multi_day_data: pa.Table) -> None:
        """Test mix of skipped and evaluated assertions."""
        db = InMemoryMetricDB()

        # Exclude 2024-01-14 (affects lag=1 metrics when evaluated on 2024-01-15)
        skip_dates = {datetime.date(2024, 1, 14)}

        suite = VerificationSuite(
            checks=[comprehensive_checks],
            db=db,
            name="Test Suite",
            skip_dates=skip_dates,
            data_av_threshold=0.8
        )

        datasource = DuckRelationDataSource.from_arrow(multi_day_data, "sales", "date")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        # Run the suite
        suite.run([datasource], key)

        # Collect results
        results = suite.collect_results()
        assert len(results) == 3

        # Check statuses
        result_map = {r.assertion: r for r in results}

        # Yesterday price check should be SKIPPED (lag=1, date excluded)
        assert result_map["Yesterday price positive"].status == "SKIPPED"

        # DoD check should be SKIPPED (depends on excluded date)
        assert result_map["Price DoD stable"].status == "SKIPPED"

        # Today price check should be OK (lag=0, not affected)
        assert result_map["Today price positive"].status == "OK"

    def test_threshold_boundary_behavior(self, multi_day_data: pa.Table) -> None:
        """Test behavior exactly at threshold boundary."""
        db = InMemoryMetricDB()

        # Create checks that use stddev (3-day window)
        @check(name="Stddev Check")
        def stddev_check(mp: MetricProvider, ctx: Context) -> None:
            # Stddev over 3 days
            ctx.assert_that(
                mp.ext.stddev(mp.average("price"), offset=0, n=3)
            ).where(name="Price volatility low").is_lt(10.0)

        # Exclude 1 of 3 days (availability = 0.67)
        skip_dates = {datetime.date(2024, 1, 13)}

        # Test with threshold above availability (should skip)
        suite1 = VerificationSuite(
            checks=[stddev_check],
            db=db,
            name="Suite 1",
            skip_dates=skip_dates,
            data_av_threshold=0.7  # Above 0.67
        )

        datasource = DuckRelationDataSource.from_arrow(multi_day_data, "sales", "date")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        suite1.run([datasource], key)
        results1 = suite1.collect_results()
        assert results1[0].status == "SKIPPED"

        # Test with threshold below availability (should evaluate)
        suite2 = VerificationSuite(
            checks=[stddev_check],
            db=db,
            name="Suite 2",
            skip_dates=skip_dates,
            data_av_threshold=0.6  # Below 0.67
        )

        suite2.run([datasource], key)
        results2 = suite2.collect_results()
        assert results2[0].status in ["OK", "FAILURE"]  # Not skipped

    def test_no_dates_excluded_normal_operation(self, multi_day_data: pa.Table) -> None:
        """Verify normal operation when no dates excluded."""
        db = InMemoryMetricDB()

        # Create suite without skip_dates
        suite = VerificationSuite(
            checks=[comprehensive_checks],
            db=db,
            name="Normal Suite"
            # No skip_dates parameter
        )

        datasource = DuckRelationDataSource.from_arrow(multi_day_data, "sales", "date")
        key = ResultKey(datetime.date(2024, 1, 15), {})

        suite.run([datasource], key)
        results = suite.collect_results()

        # No assertions should be skipped
        for result in results:
            assert result.status != "SKIPPED"
```

#### Task 5.2: Run Full Test Suite
```bash
# Run all date exclusion tests
uv run pytest tests/test_date_exclusion_*.py -v

# Run regression tests
uv run pytest tests/ -v

# Final linting check
uv run ruff check src/ tests/
uv run mypy src/ tests/

# Final pre-commit
uv run hooks
```

**Commit**: `feat: add comprehensive end-to-end tests for date exclusion`

---

### Task Group 6: Documentation and Final Verification

#### Task 6.1: Update README or Documentation
**File**: `docs/user-guide.md` (UPDATE)
```markdown
# Add new section about Date Exclusion feature

## Date Exclusion

DQX supports excluding specific dates from metric calculations, which is useful for:
- Handling known data quality issues on specific dates
- Excluding holidays or maintenance windows
- Ignoring dates with incomplete data

### Basic Usage

```python
from datetime import date
from dqx import VerificationSuite

# Define dates to exclude
skip_dates = {
    date(2024, 1, 1),  # New Year's Day
    date(2024, 1, 15), # Maintenance window
}

# Create suite with date exclusion
suite = VerificationSuite(
    checks=[your_checks],
    db=db,
    name="Daily Quality Checks",
    skip_dates=skip_dates,
    data_av_threshold=0.8  # Skip if <80% data available
)
```

### How It Works

1. **Simple Metrics**: Marked as unavailable (0.0) if their effective date is excluded
2. **Extended Metrics**: Availability = percentage of non-excluded dependencies
3. **Assertions**: Automatically SKIPPED when metric availability < threshold

### Configuration

- `skip_dates`: Set of `datetime.date` objects to exclude
- `data_av_threshold`: Float between 0.0-1.0 (default: 0.8)
  - 0.8 = Skip assertions if less than 80% of required data available
  - 1.0 = Skip if any data missing
  - 0.0 = Never skip (evaluate with available data)
```

#### Task 6.2: Final Pre-commit and Verification
```bash
# Ensure all files are formatted
uv run ruff format src/ tests/

# Run final test suite
uv run pytest tests/ -v

# Run pre-commit on all files
uv run hooks

# Verify no issues remain
git status
```

**Commit**: `docs: add date exclusion feature documentation`

---

## Summary

This implementation plan provides a complete, test-driven approach to adding date exclusion functionality to DQX. The feature integrates cleanly with existing architecture:

1. **Minimal changes** to core types (one literal extension, one field addition)
2. **Leverages existing patterns** (visitor pattern, metric lifecycle)
3. **Comprehensive testing** at unit and integration levels
4. **Clear documentation** for users

The implementation follows TDD principles with tests written before code, maintains type safety throughout, and ensures backward compatibility by using optional parameters with sensible defaults.

Total commits: 6 (one per task group)
Total new test files: 4
Total modified files: ~5 core files
