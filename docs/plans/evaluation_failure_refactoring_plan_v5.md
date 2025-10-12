# EvaluationFailure Refactoring Implementation Plan (v5)

## Overview

This plan transforms the failure output of `Evaluator.evaluate()` from `dict[SymbolicMetric | sp.Expr, str]` to structured dataclasses that are human-readable and suitable for database persistence.

## Background

Currently, when DQX evaluates data quality assertions, failures are returned as a dictionary where:
- Keys are either `SymbolicMetric` objects or `sp.Expr` objects
- Values are error message strings

This is problematic because:
1. The keys are Python objects, not human-readable strings
2. The structure doesn't capture important context like dataset names
3. It's difficult to persist to a database

## Critical Pre-refactoring Fix

There is an existing type mismatch bug in `evaluator.py` that must be fixed before proceeding with the refactoring:

```python
# Current buggy code in _gather method:
failures: dict[SymbolicMetric, str] = {}
# ...
failures[sym] = err  # BUG: sym is sp.Symbol, not SymbolicMetric!
```

## Solution Design (v5)

### New Data Structures

```python
from dataclasses import dataclass
from returns.result import Result

@dataclass
class SymbolInfo:
    """Information about a symbol in an expression"""
    name: str                    # Symbol name (e.g., "x_1")
    metric: str                  # Human-readable metric name
    dataset: str                 # Dataset name
    value: Result[float, str]    # Success(10.5) or Failure("error")

@dataclass
class EvaluationFailure:
    """Failure information for evaluation errors"""
    error_message: str           # Overall error message
    expression: str              # The symbolic expression
    symbols: list[SymbolInfo]    # List of symbol information
```

### Key Design Decisions

1. **No failure_type field** - The structure itself indicates whether it's a metric failure (symbol has Failure value) or expression failure (all symbols have Success values but expression failed)
2. **Strongly typed with dataclasses** - Better type safety and self-documenting code
3. **Single EvaluationFailure from _gather** - One failure object containing all symbol information
4. **Result types for values** - Natural representation using the returns library
5. **Unified NaN/infinity handling** - Single code path with different error messages
6. **Efficient data flow** - Avoid rebuilding symbol information by passing it through the evaluation pipeline
7. **No serialization yet** - Following YAGNI principle, serialization will be added when needed

## Implementation Tasks

### Task 0: Fix existing type mismatch bug

**Branch**: `fix/evaluator-type-mismatch`

**Files to modify**:
- `src/dqx/evaluator.py` - Fix the type annotation in `_gather`

**Implementation**:
```python
def _gather(self, expr: sp.Expr) -> Result[dict[sp.Symbol, float], dict[sp.Symbol, str]]:
    """Gather metric values for all symbols in an expression."""
    successes: dict[sp.Symbol, float] = {}
    failures: dict[sp.Symbol, str] = {}  # Fixed: Use sp.Symbol as key type

    for sym in expr.free_symbols:
        if sym not in self.metrics:
            sm = self.metric_for_symbol(sym)
            raise DQXError(f"Symbol {sm.name} not found in collected metrics.")

        match self.metrics[sym]:
            case Failure(err):
                failures[sym] = err  # Now correctly uses sp.Symbol
            case Success(v):
                successes[sym] = v

    if failures:
        return Failure(failures)
    return Success(successes)
```

**Testing**:
```bash
# Run existing tests to ensure fix doesn't break anything
uv run pytest tests/ -v -k "evaluator"
```

**Commit**: `fix: correct type annotation for failures dict in Evaluator._gather`

### Task 1: Create the dataclasses

**Branch**: `feat/evaluation-failure-v5` (create from fix/evaluator-type-mismatch)

**Files to modify**:
- `src/dqx/common.py` - Add the new dataclasses

**Implementation**:
```python
from dataclasses import dataclass
from returns.result import Result

@dataclass
class SymbolInfo:
    """Information about a symbol in an expression"""
    name: str                    # Symbol name (e.g., "x_1")
    metric: str                  # Human-readable metric name
    dataset: str                 # Dataset name
    value: Result[float, str]    # Success(10.5) or Failure("error")

@dataclass
class EvaluationFailure:
    """Failure information for evaluation errors"""
    error_message: str           # Overall error message
    expression: str              # The symbolic expression
    symbols: list[SymbolInfo]    # List of symbol information
```

**Commit**: `feat: add EvaluationFailure and SymbolInfo dataclasses`

### Task 2: Write comprehensive failing tests

**Branch**: Continue on `feat/evaluation-failure-v5`

**Files to create/modify**:
- `tests/test_evaluator.py` - Create this new test file

**Implementation**:
```python
import math
import sympy as sp
import pytest
from datetime import date
from dqx.common import EvaluationFailure, SymbolInfo, ResultKey, DQXError
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.specs import Average
from returns.result import Success, Failure
from unittest.mock import Mock, patch


class TestEvaluatorFailureHandling:
    """Test that Evaluator returns EvaluationFailure objects."""

    def test_metric_failure_returns_evaluation_failure(self):
        """Test that metric failures are converted to EvaluationFailure."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today())
        evaluator = Evaluator(provider, key)

        # Create a symbol that will fail
        symbol = sp.Symbol("x_1")
        symbolic_metric = SymbolicMetric(
            name="x_1",
            symbol=symbol,
            fn=lambda k: Failure("Database error"),
            key_provider=Mock(),
            metric_spec=Average("price"),
            dataset="orders"
        )

        # Mock the provider methods
        provider.symbolic_metrics = [symbolic_metric]
        provider.get_symbol.return_value = symbolic_metric

        # Mock the metrics collection to return a failure
        evaluator._metrics = {symbol: Failure("Database error")}

        # Evaluate an expression with the failing symbol
        result = evaluator.evaluate(symbol)

        # Assert we get EvaluationFailure objects
        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, EvaluationFailure)
        assert failure.error_message == "One or more metrics failed to evaluate"
        assert failure.expression == str(symbol)
        assert len(failure.symbols) == 1

        symbol_info = failure.symbols[0]
        assert symbol_info.name == "x_1"
        assert symbol_info.metric == "average(price)"
        assert symbol_info.dataset == "orders"
        assert isinstance(symbol_info.value, Failure)
        assert symbol_info.value.failure() == "Database error"

    def test_expression_nan_returns_evaluation_failure(self):
        """Test that NaN results are converted to EvaluationFailure."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today())
        evaluator = Evaluator(provider, key)

        # Create symbols that will divide to NaN
        x1 = sp.Symbol("x_1")
        x2 = sp.Symbol("x_2")

        # Create symbolic metrics
        sm1 = SymbolicMetric(
            name="x_1", symbol=x1,
            fn=lambda k: Success(0.0),
            key_provider=Mock(),
            metric_spec=Mock(name="average(price)"),
            dataset="orders"
        )
        sm2 = SymbolicMetric(
            name="x_2", symbol=x2,
            fn=lambda k: Success(0.0),
            key_provider=Mock(),
            metric_spec=Mock(name="sum(quantity)"),
            dataset="inventory"
        )

        provider.symbolic_metrics = [sm1, sm2]
        provider.get_symbol.side_effect = lambda s: sm1 if s == x1 else sm2

        # Mock successful metric collection
        evaluator._metrics = {x1: Success(0.0), x2: Success(0.0)}

        # Evaluate 0/0 which gives NaN
        result = evaluator.evaluate(x1 / x2)

        # Assert we get EvaluationFailure
        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        failure = failures[0]
        assert failure.error_message == "Validating value is NaN"
        assert failure.expression == "x_1/x_2"
        assert len(failure.symbols) == 2

        # Check symbols
        assert failure.symbols[0].name == "x_1"
        assert failure.symbols[0].metric == "average(price)"
        assert failure.symbols[0].value == Success(0.0)
        assert failure.symbols[1].name == "x_2"
        assert failure.symbols[1].metric == "sum(quantity)"
        assert failure.symbols[1].value == Success(0.0)

    def test_symbol_not_in_provider(self):
        """Test error when symbol is not found in provider."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today())
        evaluator = Evaluator(provider, key)

        # Create a symbol that doesn't exist in provider
        unknown_symbol = sp.Symbol("x_unknown")

        # Mock provider to raise error
        provider.get_symbol.side_effect = DQXError("Symbol not found")

        # Mock metrics without the symbol
        evaluator._metrics = {}

        # Attempt to evaluate should handle the error gracefully
        with pytest.raises(DQXError, match="Symbol"):
            evaluator.evaluate(unknown_symbol)

    def test_complex_expression_with_multiple_operators(self):
        """Test complex expression like (a + b) / (c - d)."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today())
        evaluator = Evaluator(provider, key)

        # Create symbols
        a, b, c, d = sp.symbols('a b c d')

        # Create symbolic metrics
        metrics = {
            a: ("metric_a", "dataset1", 10.0),
            b: ("metric_b", "dataset1", 20.0),
            c: ("metric_c", "dataset2", 15.0),
            d: ("metric_d", "dataset2", 15.0),  # Will make denominator 0
        }

        for sym, (name, dataset, value) in metrics.items():
            sm = SymbolicMetric(
                name=str(sym),
                symbol=sym,
                fn=lambda k, v=value: Success(v),
                key_provider=Mock(),
                metric_spec=Mock(name=name),
                dataset=dataset
            )
            provider.get_symbol.side_effect = lambda s, m=metrics: Mock(
                name=m[s][0], dataset=m[s][1], metric_spec=Mock(name=m[s][0])
            )

        # Mock metrics
        evaluator._metrics = {sym: Success(val[2]) for sym, val in metrics.items()}

        # Evaluate (a + b) / (c - d) = 30 / 0 = infinity
        result = evaluator.evaluate((a + b) / (c - d))

        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert failures[0].error_message == "Validating value is infinity"

    def test_empty_expression(self):
        """Test expression with no free symbols."""
        provider = Mock(spec=MetricProvider)
        key = ResultKey(yyyy_mm_dd=date.today())
        evaluator = Evaluator(provider, key)

        # Evaluate a constant expression
        result = evaluator.evaluate(sp.sympify(42))

        assert isinstance(result, Success)
        assert result.unwrap() == 42.0
```

**How to run tests**:
```bash
# Run just the new tests
uv run pytest tests/test_evaluator.py -v

# Run with coverage
uv run pytest tests/test_evaluator.py -v --cov=dqx.evaluator
```

**Commit**: `test: add comprehensive tests for EvaluationFailure in Evaluator`

### Task 2.5: Identify affected existing tests

**Branch**: Continue on same branch

**Steps**:
1. Run the full test suite to identify which tests will fail with the new types:
```bash
uv run pytest tests/ -v > test_baseline.txt
```

2. Document affected tests in a temporary file:
```bash
echo "# Tests that will need updates for EvaluationFailure refactoring" > affected_tests.md
echo "Run this after implementing the changes to see which tests fail" >> affected_tests.md
```

**Commit**: `docs: document test baseline before refactoring`

### Task 3: Update Evaluator._gather() method with improved error handling

**Branch**: Continue on same branch

**Files to modify**:
- `src/dqx/evaluator.py` - Update the `_gather` method

**Implementation steps**:
1. Import dataclasses: `from dqx.common import EvaluationFailure, SymbolInfo`
2. Change return type and implementation:

```python
def _gather(self, expr: sp.Expr) -> Result[tuple[dict[sp.Symbol, float], list[SymbolInfo]], EvaluationFailure]:
    """Gather metric values for all symbols in an expression.

    Returns both the successes dict and symbol info list to avoid rebuilding later.
    """
    successes: dict[sp.Symbol, float] = {}
    symbol_infos: list[SymbolInfo] = []
    has_failures = False

    for sym in expr.free_symbols:
        if sym not in self.metrics:
            # Handle case where metric_for_symbol might also fail
            try:
                sm = self.metric_for_symbol(sym)
                raise DQXError(f"Symbol {sm.name} not found in collected metrics.")
            except DQXError as e:
                # If metric_for_symbol fails, provide a more generic error
                if "not found in provider" in str(e):
                    raise DQXError(f"Symbol {sym} not found in provider.")
                raise

        sm = self.metric_for_symbol(sym)
        metric_result = self.metrics[sym]

        symbol_info = SymbolInfo(
            name=str(sym),
            metric=sm.metric_spec.name,
            dataset=sm.dataset,
            value=metric_result
        )
        symbol_infos.append(symbol_info)

        match metric_result:
            case Failure(_):
                has_failures = True
            case Success(v):
                successes[sym] = v

    if has_failures:
        failure = EvaluationFailure(
            error_message="One or more metrics failed to evaluate",
            expression=str(expr),
            symbols=symbol_infos
        )
        return Failure(failure)

    return Success((successes, symbol_infos))
```

**How to verify**:
```bash
# Check types with mypy
uv run mypy src/dqx/evaluator.py

# Run the specific test for metric failures
uv run pytest tests/test_evaluator.py::TestEvaluatorFailureHandling::test_metric_failure_returns_evaluation_failure -v
```

**Commit**: `feat: update Evaluator._gather with improved error handling and symbol info`

### Task 4: Update Evaluator.evaluate() method with efficient data flow

**Branch**: Continue on same branch

**Files to modify**:
- `src/dqx/evaluator.py` - Update the `evaluate` method

**Implementation steps**:
1. Change return type from `Result[float, dict[SymbolicMetric | sp.Expr, str]]` to `Result[float, list[EvaluationFailure]]`
2. Update to use symbol info from _gather:

```python
def evaluate(self, expr: sp.Expr) -> Result[float, list[EvaluationFailure]]:
    """Evaluate a symbolic expression by substituting collected metric values."""
    sv = self._gather(expr)
    match sv:
        case Success((symbol_values, symbol_infos)):
            expr_val = float(sp.N(expr.subs(symbol_values), 6))

            # Unified handling for NaN and infinity
            if math.isnan(expr_val) or math.isinf(expr_val):
                # Reuse symbol_infos from _gather instead of rebuilding
                error_msg = "Validating value is NaN" if math.isnan(expr_val) else "Validating value is infinity"

                failure = EvaluationFailure(
                    error_message=error_msg,
                    expression=str(expr),
                    symbols=symbol_infos
                )
                return Failure([failure])

            return Success(expr_val)

        case Failure(error):
            # _gather returns single EvaluationFailure, wrap in list
            return Failure([error])

    # Unreachable state
    raise RuntimeError("Unreachable state in evaluation.")
```

**Update docstrings**:
```python
def _gather(self, expr: sp.Expr) -> Result[tuple[dict[sp.Symbol, float], list[SymbolInfo]], EvaluationFailure]:
    """Gather metric values for all symbols in an expression.

    Args:
        expr: Symbolic expression containing symbols to gather values for

    Returns:
        Success containing a tuple of (symbol_values dict, symbol_infos list) if
        all symbols evaluated successfully. Failure containing an EvaluationFailure
        if any symbols failed.

    Raises:
        DQXError: If a symbol in the expression is not found in metrics or provider
    """

def evaluate(self, expr: sp.Expr) -> Result[float, list[EvaluationFailure]]:
    """Evaluate a symbolic expression by substituting collected metric values.

    Args:
        expr: Symbolic expression to evaluate

    Returns:
        Success containing the evaluated float value if evaluation succeeds.
        Failure containing a list of EvaluationFailure objects if any symbols
        fail to evaluate or if the result is NaN/infinity.
    """
```

**How to verify**:
```bash
# Run all evaluator tests
uv run pytest tests/test_evaluator.py -v

# Check linting
uv run ruff check src/dqx/evaluator.py
```

**Commit**: `feat: update Evaluator.evaluate with efficient data flow`

### Task 5: Update AssertionNode type annotation

**Branch**: Continue on same branch

**Files to modify**:
- `src/dqx/graph/nodes.py` - Update AssertionNode class

**Implementation steps**:
1. Add import: `from dqx.common import EvaluationFailure`
2. Update the `_value` type annotation and docstring:

```python
class AssertionNode(BaseNode["CheckNode"]):
    """Node representing an assertion to be evaluated.

    Attributes:
        _value: The evaluation result, either Success[float] or
                Failure[list[EvaluationFailure]]
    """

    def __init__(self, ...):
        # ... existing code ...
        self._value: Result[float, list[EvaluationFailure]]  # Updated type
```

**How to verify**:
```bash
# Type check
uv run mypy src/dqx/graph/nodes.py

# Run any existing tests that use AssertionNode
uv run pytest tests/graph/ -v
```

**Commit**: `feat: update AssertionNode to use EvaluationFailure type`

### Task 6: Integration testing

**Branch**: Continue on same branch

**Files to create/modify**:
- `tests/test_evaluator_integration.py` - Create integration tests

**Implementation**:
```python
"""Integration tests for EvaluationFailure with real data flow."""
import sympy as sp
from datetime import date
from dqx.common import EvaluationFailure, SymbolInfo, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider
from dqx.specs import Average, Sum
from dqx.orm.repositories import InMemoryMetricDB
from returns.result import Success, Failure


def test_end_to_end_metric_failure():
    """Test complete flow from provider to assertion with metric failure."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create metrics
    avg_price = provider.average("price", dataset="orders")
    sum_qty = provider.sum("quantity", dataset="inventory")

    key = ResultKey(yyyy_mm_dd=date.today())
    evaluator = Evaluator(provider, key)

    # Evaluate expression
    result = evaluator.evaluate(avg_price + sum_qty)

    # Verify failure structure
    assert isinstance(result, Failure)
    failures = result.failure()
    assert len(failures) == 1
    assert isinstance(failures[0], EvaluationFailure)

def test_end_to_end_expression_failure():
    """Test complete flow for expression-level failures."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create metrics that will result in NaN
    metric1 = provider.average("col1", dataset="data1")
    metric2 = provider.average("col2", dataset="data2")

    key = ResultKey(yyyy_mm_dd=date.today())
    evaluator = Evaluator(provider, key)

    # Evaluate 0/0
    result = evaluator.evaluate(metric1 / metric2)

    # Verify expression failure
    assert isinstance(result, Failure)
    failures = result.failure()
    assert failures[0].error_message == "Validating value is NaN"
    assert len(failures[0].symbols) == 2
```

**Commit**: `test: add integration tests for EvaluationFailure flow`

### Task 7: Update existing tests

**Branch**: Continue on same branch

**Files to check and potentially update**:
- Based on Task 2.5 results, update affected tests

**Steps**:
1. Run all tests to find failures:
```bash
uv run pytest tests/ -v
```

2. Update test assertions to handle new types:
```python
# Old assertion style
assert isinstance(result.failure(), dict)
assert result.failure()[some_key] == "error message"

# New assertion style
assert isinstance(result.failure(), list)
assert len(result.failure()) == 1
assert isinstance(result.failure()[0], EvaluationFailure)
assert result.failure()[0].error_message == "error message"
```

3. Fix each failing test file

**Commit**: `test: update existing tests for EvaluationFailure changes`

### Task 8: Final verification and cleanup

**Branch**: Continue on same branch

**Steps**:
1. Run full test suite with coverage:
   ```bash
   uv run pytest tests/ -v --cov=dqx
   ```

2. Run type checking:
   ```bash
   uv run mypy src/
   ```

3. Run linting and formatting:
   ```bash
   uv run ruff check src/ tests/ --fix
   uv run ruff format src/ tests/
   ```

4. Run pre-commit hooks:
   ```bash
   ./bin/run-hooks.sh --all
   ```

**Commit**: `chore: cleanup and format code`

### Task 9: Documentation and Demo

**Branch**: Continue on same branch

**Files to update**:
- `README.md` - Update any examples that show error handling
- `src/dqx/evaluator.py` - Ensure all docstrings are updated
- `src/dqx/graph/nodes.py` - Update AssertionNode docstring
- Create `docs/evaluation_failure_guide.md` if needed
- Create `examples/evaluation_failure_demo.py` - Demo script

**Demo Script Implementation** (`examples/evaluation_failure_demo.py`):
```python
"""Demo script showing EvaluationFailure error reporting in various scenarios."""
import sympy as sp
from datetime import date
from dqx.common import EvaluationFailure, SymbolInfo, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.specs import Average, Sum, Count, Min, Max
from returns.result import Success, Failure
from unittest.mock import Mock
from typing import List


def print_evaluation_failure(failures: List[EvaluationFailure]) -> None:
    """Pretty print evaluation failures."""
    for i, failure in enumerate(failures, 1):
        print(f"\n{'='*80}")
        print(f"FAILURE #{i}")
        print(f"{'='*80}")
        print(f"Error: {failure.error_message}")
        print(f"Expression: {failure.expression}")
        print(f"\nSymbol Details:")
        print(f"{'Symbol':<10} {'Metric':<25} {'Dataset':<15} {'Value':<15} {'Status'}")
        print(f"{'-'*10} {'-'*25} {'-'*15} {'-'*15} {'-'*6}")

        for symbol in failure.symbols:
            status = "✓" if symbol.value.is_success() else "✗"
            if symbol.value.is_success():
                value = f"{symbol.value.unwrap():.2f}"
                error = ""
            else:
                value = "N/A"
                error = f" ({symbol.value.failure()})"

            print(f"{symbol.name:<10} {symbol.metric:<25} {symbol.dataset:<15} {value:<15} {status}{error}")


def scenario_1_multiple_metric_failures():
    """Scenario 1: Multiple metric failures with 5+ symbols."""
    print("\n" + "="*80)
    print("SCENARIO 1: Multiple Metric Failures")
    print("="*80)
    print("Expression: (revenue - costs) / (users * conversion_rate * avg_order_value)")

    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today())
    evaluator = Evaluator(provider, key)

    # Create symbols
    revenue = sp.Symbol("revenue")
    costs = sp.Symbol("costs")
    users = sp.Symbol("users")
    conversion_rate = sp.Symbol("conv_rate")
    avg_order_value = sp.Symbol("avg_order")

    # Create symbolic metrics with various failures
    metrics = {
        revenue: SymbolicMetric(
            name="revenue", symbol=revenue,
            fn=lambda k: Failure("Database connection timeout"),
            key_provider=Mock(),
            metric_spec=Sum("revenue"),
            dataset="transactions"
        ),
        costs: SymbolicMetric(
            name="costs", symbol=costs,
            fn=lambda k: Success(50000.0),
            key_provider=Mock(),
            metric_spec=Sum("operating_costs"),
            dataset="finance"
        ),
        users: SymbolicMetric(
            name="users", symbol=users,
            fn=lambda k: Failure("Permission denied: insufficient privileges"),
            key_provider=Mock(),
            metric_spec=Count("user_id"),
            dataset="users"
        ),
        conversion_rate: SymbolicMetric(
            name="conv_rate", symbol=conversion_rate,
            fn=lambda k: Success(0.05),
            key_provider=Mock(),
            metric_spec=Average("conversion_rate"),
            dataset="analytics"
        ),
        avg_order_value: SymbolicMetric(
            name="avg_order", symbol=avg_order_value,
            fn=lambda k: Failure("Column 'order_value' not found"),
            key_provider=Mock(),
            metric_spec=Average("order_value"),
            dataset="orders"
        ),
    }

    # Mock provider methods
    provider.symbolic_metrics = list(metrics.values())
    provider.get_symbol.side_effect = lambda s: metrics[s]

    # Mock metrics collection
    evaluator._metrics = {sym: metric.fn(key) for sym, metric in metrics.items()}

    # Evaluate expression
    expr = (revenue - costs) / (users * conversion_rate * avg_order_value)
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print_evaluation_failure(result.failure())


def scenario_2_expression_nan():
    """Scenario 2: Expression results in NaN with 5+ symbols."""
    print("\n" + "="*80)
    print("SCENARIO 2: Expression Results in NaN")
    print("="*80)
    print("Expression: (sales_na - sales_eu) / (sales_apac + sales_latam - sales_total)")

    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today())
    evaluator = Evaluator(provider, key)

    # Create symbols
    sales_na = sp.Symbol("sales_na")
    sales_eu = sp.Symbol("sales_eu")
    sales_apac = sp.Symbol("sales_apac")
    sales_latam = sp.Symbol("sales_latam")
    sales_total = sp.Symbol("sales_total")

    # Create metrics that will cause division by zero
    metrics = {
        sales_na: SymbolicMetric(
            name="sales_na", symbol=sales_na,
            fn=lambda k: Success(100000.0),
            key_provider=Mock(),
            metric_spec=Sum("sales_amount"),
            dataset="sales_north_america"
        ),
        sales_eu: SymbolicMetric(
            name="sales_eu", symbol=sales_eu,
            fn=lambda k: Success(80000.0),
            key_provider=Mock(),
            metric_spec=Sum("sales_amount"),
            dataset="sales_europe"
        ),
        sales_apac: SymbolicMetric(
            name="sales_apac", symbol=sales_apac,
            fn=lambda k: Success(60000.0),
            key_provider=Mock(),
            metric_spec=Sum("sales_amount"),
            dataset="sales_asia_pacific"
        ),
        sales_latam: SymbolicMetric(
            name="sales_latam", symbol=sales_latam,
            fn=lambda k: Success(40000.0),
            key_provider=Mock(),
            metric_spec=Sum("sales_amount"),
            dataset="sales_latin_america"
        ),
        sales_total: SymbolicMetric(
            name="sales_total", symbol=sales_total,
            fn=lambda k: Success(100000.0),  # Equals sum of apac + latam
            key_provider=Mock(),
            metric_spec=Sum("sales_amount"),
            dataset="sales_global"
        ),
    }

    # Mock provider methods
    provider.symbolic_metrics = list(metrics.values())
    provider.get_symbol.side_effect = lambda s: metrics[s]

    # Mock metrics collection
    evaluator._metrics = {sym: metric.fn(key) for sym, metric in metrics.items()}

    # Evaluate expression (denominator will be 0)
    expr = (sales_na - sales_eu) / (sales_apac + sales_latam - sales_total)
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print_evaluation_failure(result.failure())


def scenario_3_expression_infinity():
    """Scenario 3: Expression results in infinity with 6 symbols."""
    print("\n" + "="*80)
    print("SCENARIO 3: Expression Results in Infinity")
    print("="*80)
    print("Expression: (views * clicks * conversions) / (bounces - sessions + errors)")

    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today())
    evaluator = Evaluator(provider, key)

    # Create symbols
    views = sp.Symbol("views")
    clicks = sp.Symbol("clicks")
    conversions = sp.Symbol("conversions")
    bounces = sp.Symbol("bounces")
    sessions = sp.Symbol("sessions")
    errors = sp.Symbol("errors")

    # Create metrics that will cause division by very small number
    metrics = {
        views: SymbolicMetric(
            name="views", symbol=views,
            fn=lambda k: Success(1000000.0),
            key_provider=Mock(),
            metric_spec=Count("page_view"),
            dataset="web_analytics"
        ),
        clicks: SymbolicMetric(
            name="clicks", symbol=clicks,
            fn=lambda k: Success(50000.0),
            key_provider=Mock(),
            metric_spec=Count("click_event"),
            dataset="web_analytics"
        ),
        conversions: SymbolicMetric(
            name="conversions", symbol=conversions,
            fn=lambda k: Success(1000.0),
            key_provider=Mock(),
            metric_spec=Count("conversion"),
            dataset="web_analytics"
        ),
        bounces: SymbolicMetric(
            name="bounces", symbol=bounces,
            fn=lambda k: Success(10000.0),
            key_provider=Mock(),
            metric_spec=Count("bounce"),
            dataset="web_analytics"
        ),
        sessions: SymbolicMetric(
            name="sessions", symbol=sessions,
            fn=lambda k: Success(10000.0),
            key_provider=Mock(),
            metric_spec=Count("session"),
            dataset="web_analytics"
        ),
        errors = SymbolicMetric(
            name="errors", symbol=errors,
            fn=lambda k: Success(0.000001),  # Very small value
            key_provider=Mock(),
            metric_spec=Count("error"),
            dataset="web_analytics"
        ),
    }

    # Mock provider methods
    provider.symbolic_metrics = list(metrics.values())
    provider.get_symbol.side_effect = lambda s: metrics[s]

    # Mock metrics collection
    evaluator._metrics = {sym: metric.fn(key) for sym, metric in metrics.items()}

    # Evaluate expression (very large numerator / very small denominator)
    expr = (views * clicks * conversions) / (bounces - sessions + errors)
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print_evaluation_failure(result.failure())


def scenario_4_complex_mixed():
    """Scenario 4: Complex expression with mixed failures and 8 symbols."""
    print("\n" + "="*80)
    print("SCENARIO 4: Complex Mixed Expression")
    print("="*80)
    print("Expression: ((revenue * margin - fixed_costs) / active_users) + ")
    print("            (ad_spend / (impressions * ctr * conversion))")

    provider = Mock(spec=MetricProvider)
    key = ResultKey(yyyy_mm_dd=date.today())
    evaluator = Evaluator(provider, key)

    # Create symbols
    revenue = sp.Symbol("revenue")
    margin = sp.Symbol("margin")
    fixed_costs = sp.Symbol("fixed_costs")
    active_users = sp.Symbol("active_users")
    ad_spend = sp.Symbol("ad_spend")
    impressions = sp.Symbol("impressions")
    ctr = sp.Symbol("ctr")
    conversion = sp.Symbol("conversion")

    # Mix of successful and failed metrics
    metrics = {
        revenue: SymbolicMetric(
            name="revenue", symbol=revenue,
            fn=lambda k: Success(1000000.0),
            key_provider=Mock(),
            metric_spec=Sum("revenue"),
            dataset="financial"
        ),
        margin: SymbolicMetric(
            name="margin", symbol=margin,
            fn=lambda k: Failure("ETL pipeline failed: data quality check failed"),
            key_provider=Mock(),
            metric_spec=Average("profit_margin"),
            dataset="financial"
        ),
        fixed_costs: SymbolicMetric(
            name="fixed_costs", symbol=fixed_costs,
            fn=lambda k: Success(200000.0),
            key_provider=Mock(),
            metric_spec=Sum("fixed_cost"),
            dataset="financial"
        ),
        active_users: SymbolicMetric(
            name="active_users", symbol=active_users,
            fn=lambda k: Failure("Redis cache unavailable"),
            key_provider=Mock(),
            metric_spec=Count("distinct user_id"),
            dataset="user_activity"
        ),
        ad_spend: SymbolicMetric(
            name="ad_spend", symbol=ad_spend,
            fn=lambda k: Success(50000.0),
            key_provider=Mock(),
            metric_spec=Sum("spend"),
            dataset="marketing"
        ),
        impressions: SymbolicMetric(
            name="impressions", symbol=impressions,
            fn=lambda k: Success(10000000.0),
            key_provider=Mock(),
            metric_spec=Count("impression"),
            dataset="marketing"
        ),
        ctr: SymbolicMetric(
            name="ctr", symbol=ctr,
            fn=lambda k: Failure("Division by zero in CTR calculation"),
            key_provider=Mock(),
            metric_spec=Average("click_through_rate"),
            dataset="marketing"
        ),
        conversion: SymbolicMetric(
            name="conversion", symbol=conversion,
            fn=lambda k: Success(0.02),
            key_provider=Mock(),
            metric_spec=Average("conversion_rate"),
            dataset="marketing"
        ),
    }

    # Mock provider methods
    provider.symbolic_metrics = list(metrics.values())
    provider.get_symbol.side_effect = lambda s: metrics[s]

    # Mock metrics collection
    evaluator._metrics = {sym: metric.fn(key) for sym, metric in metrics.items()}

    # Evaluate complex expression
    expr = ((revenue * margin - fixed_costs) / active_users) + \
           (ad_spend / (impressions * ctr * conversion))
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print_evaluation_failure(result.failure())


if __name__ == "__main__":
    print("DQX EvaluationFailure Demo")
    print("=" * 80)
    print("This demo shows how DQX reports errors in various failure scenarios.")
    print("Each scenario uses expressions with at least 5 symbols.")

    scenario_1_multiple_metric_failures()
    scenario_2_expression_nan()
    scenario_3_expression_infinity()
    scenario_4_complex_mixed()

    print("\n" + "="*80)
    print("Demo completed!")
```

**Example documentation**:
```markdown
## Error Handling

When assertions fail, DQX provides structured error information:

```python
# If evaluation fails, you get EvaluationFailure objects
if isinstance(assertion._value, Failure):
    for failure in assertion._value.failure():
        print(f"Error: {failure.error_message}")
        print(f"Expression: {failure.expression}")
        for symbol in failure.symbols:
            status = "✓" if symbol.value.is_success() else "✗"
            value = symbol.value.value_or("N/A")
            error = symbol.value.swap().value_or("")
            print(f"  {status} {symbol.name} ({symbol.metric}): {value} {error}")
```

For a comprehensive demonstration of error reporting, see `examples/evaluation_failure_demo.py`.
```

**Commit**: `docs: update documentation and add comprehensive error demo`

## Development Workflow

1. **Fix the bug first**:
   ```bash
   git checkout -b fix/evaluator-type-mismatch
   # Implement Task 0
   git push origin fix/evaluator-type-mismatch
   ```

2. **Create feature branch from the fix**:
   ```bash
   git checkout -b feat/evaluation-failure-v5
   ```

3. **Follow TDD cycle**:
   - Write failing test
   - Implement minimum code to pass
   - Refactor if needed
   - Commit frequently

4. **Testing commands**:
   ```bash
   # Run specific test file
   uv run pytest tests/test_evaluator.py -v

   # Run with coverage
   uv run pytest tests/test_evaluator.py -v --cov=dqx.evaluator

   # Run all tests
   uv run pytest tests/ -v
   ```

5. **Code quality checks**:
   ```bash
   # Type checking
   uv run mypy src/dqx/evaluator.py

   # Linting
   uv run ruff check src/dqx/evaluator.py

   # Formatting
   uv run ruff format src/dqx/evaluator.py
   ```

## Key Principles to Follow

1. **Fix bugs before refactoring** - Ensure a clean baseline
2. **YAGNI (You Aren't Gonna Need It)** - Only implement what's needed now (no serialization)
3. **DRY (Don't Repeat Yourself)** - Reuse data structures to avoid redundant work
4. **TDD (Test Driven Development)** - Write tests first, including edge cases
5. **Frequent Commits** - Keep commits atomic and focused
6. **Proper error handling** - Handle edge cases gracefully

## Success Criteria

- [ ] Type mismatch bug is fixed
- [ ] All tests pass
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] Code coverage maintained or improved
- [ ] EvaluationFailure objects contain all required information
- [ ] SymbolInfo uses Result types naturally
- [ ] NaN and infinity handling is unified
- [ ] _gather returns single EvaluationFailure with symbol info
- [ ] No redundant symbol info rebuilding
- [ ] All docstrings are updated

## Notes for Implementation

- The `SymbolicMetric` class is in `src/dqx/provider.py`
- The `sp.Expr` type comes from sympy
- The `Result` type comes from the `returns` library
- Use `match` statements (Python 3.10+) for pattern matching
- Remember that AssertionNode already contains the expression in its `actual` field
- Always handle the case where `metric_for_symbol()` might fail
- No serialization utilities needed at this time (YAGNI)
