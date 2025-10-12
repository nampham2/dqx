# EvaluationFailure Refactoring Implementation Plan (v2)

## Overview

This plan transforms the failure output of `Evaluator.evaluate()` from `dict[SymbolicMetric | sp.Expr, str]` to a structured `EvaluationFailure` dataclass that is human-readable and suitable for database persistence.

## Background

Currently, when DQX evaluates data quality assertions, failures are returned as a dictionary where:
- Keys are either `SymbolicMetric` objects or `sp.Expr` objects
- Values are error message strings

This is problematic because:
1. The keys are Python objects, not human-readable strings
2. The structure doesn't capture important context like dataset names
3. It's difficult to persist to a database

## Solution Design (v2)

### New Data Structure

```python
from typing import Any

@dataclass
class EvaluationFailure:
    """Flattened failure information for evaluation errors"""
    error_message: str         # The actual error
    expression: str            # The symbolic expression (as string)
    symbols: list[dict[str, Any]]  # List of symbol information
```

### Symbol Information Structure

Each symbol in the `symbols` list contains:
```python
{
    "name": "x_1",              # Symbol name (e.g., x_1, x_2)
    "metric": "average(price)", # Human-readable metric name
    "dataset": "orders",        # Dataset name
    "value": 10.5,              # Actual value (None if failed)
    "error": None               # Error message (None if successful)
}
```

### Key Design Decisions

1. **No failure_type field** - The structure itself indicates whether it's a metric failure (symbol has error) or expression failure (all symbols succeeded but expression failed)
2. **Flattened structure** - Direct access to all fields without nesting
3. **Unified NaN/infinity handling** - Single code path with different error messages
4. **JSON-serializable** - Can be stored directly in database

## Implementation Tasks

### Task 1: Create the EvaluationFailure dataclass

**Branch**: `feat/evaluation-failure-v2`

**Files to modify**:
- `src/dqx/common.py` - Add the new dataclass

**Implementation**:
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class EvaluationFailure:
    """Flattened failure information for evaluation errors"""
    error_message: str         # The actual error
    expression: str            # The symbolic expression (as string)
    symbols: list[dict[str, Any]]  # List of symbol information
```

**Testing**:
- No direct tests needed for a simple dataclass
- Will be tested through usage in subsequent tasks

**Commit**: `feat: add EvaluationFailure dataclass for structured error reporting`

### Task 2: Write failing tests for Evaluator changes

**Branch**: Continue on `feat/evaluation-failure-v2`

**Files to create/modify**:
- `tests/test_evaluator.py` - Create this new test file

**Implementation**:
```python
import math
import sympy as sp
import pytest
from datetime import date
from dqx.common import EvaluationFailure, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.specs import Average
from returns.result import Success, Failure
from unittest.mock import Mock, patch


class TestEvaluatorFailureHandling:
    """Test that Evaluator returns EvaluationFailure objects."""

    def test_metric_failure_returns_evaluation_failure(self):
        """Test that metric failures are converted to EvaluationFailure."""
        # This test will fail initially - that's expected!
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
        assert failure.error_message == "Database error"
        assert failure.expression == str(symbol)
        assert len(failure.symbols) == 1
        assert failure.symbols[0]["name"] == "x_1"
        assert failure.symbols[0]["metric"] == "average(price)"
        assert failure.symbols[0]["dataset"] == "orders"
        assert failure.symbols[0]["value"] is None
        assert failure.symbols[0]["error"] == "Database error"

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
        # Check first symbol
        assert failure.symbols[0]["name"] == "x_1"
        assert failure.symbols[0]["metric"] == "average(price)"
        assert failure.symbols[0]["value"] == 0.0
        assert failure.symbols[0]["error"] is None
        # Check second symbol
        assert failure.symbols[1]["name"] == "x_2"
        assert failure.symbols[1]["metric"] == "sum(quantity)"
        assert failure.symbols[1]["value"] == 0.0
        assert failure.symbols[1]["error"] is None

    def test_expression_infinity_returns_evaluation_failure(self):
        """Test that infinity results are converted to EvaluationFailure."""
        # Similar to NaN test but with division by zero
        # resulting in infinity

    def test_multiple_metric_failures(self):
        """Test handling of multiple metric failures in one expression."""
        # Test expression with multiple failing symbols
```

**Key testing principles**:
1. Write tests BEFORE implementation (TDD)
2. Each test should test ONE specific behavior
3. Tests should be independent and isolated
4. Use descriptive test names that explain what is being tested
5. Follow Arrange-Act-Assert pattern

**How to run tests**:
```bash
# Run just the new tests
uv run pytest tests/test_evaluator.py -v

# Run with coverage
uv run pytest tests/test_evaluator.py -v --cov=dqx.evaluator
```

**Commit**: `test: add failing tests for EvaluationFailure in Evaluator`

### Task 3: Update Evaluator._gather() method

**Branch**: Continue on same branch

**Files to modify**:
- `src/dqx/evaluator.py` - Update the `_gather` method

**Implementation steps**:
1. Import EvaluationFailure: `from dqx.common import EvaluationFailure`
2. Change return type of `_gather` from `Result[dict[sp.Symbol, float], dict[SymbolicMetric, str]]` to `Result[dict[sp.Symbol, float], list[EvaluationFailure]]`
3. Update the failure handling logic:

```python
def _gather(self, expr: sp.Expr) -> Result[dict[sp.Symbol, float], list[EvaluationFailure]]:
    """Gather metric values for all symbols in an expression."""
    successes: dict[sp.Symbol, float] = {}
    symbol_infos: list[dict[str, Any]] = []
    has_failures = False

    for sym in expr.free_symbols:
        if sym not in self.metrics:
            sm = self.metric_for_symbol(sym)
            raise DQXError(f"Symbol {sm.name} not found in collected metrics.")

        sm = self.metric_for_symbol(sym)
        symbol_info = {
            "name": str(sym),
            "metric": sm.metric_spec.name,
            "dataset": sm.dataset,
            "value": None,
            "error": None
        }

        match self.metrics[sym]:
            case Failure(err):
                symbol_info["error"] = err
                has_failures = True
            case Success(v):
                symbol_info["value"] = v
                successes[sym] = v

        symbol_infos.append(symbol_info)

    if has_failures:
        # Create a single EvaluationFailure with all symbol info
        failure = EvaluationFailure(
            error_message="One or more metrics failed to evaluate",
            expression=str(expr),
            symbols=symbol_infos
        )
        return Failure([failure])

    return Success(successes)
```

**How to verify**:
```bash
# Check types with mypy
uv run mypy src/dqx/evaluator.py

# Run the specific test for metric failures
uv run pytest tests/test_evaluator.py::TestEvaluatorFailureHandling::test_metric_failure_returns_evaluation_failure -v
```

**Commit**: `feat: update Evaluator._gather to return EvaluationFailure objects`

### Task 4: Update Evaluator.evaluate() method

**Branch**: Continue on same branch

**Files to modify**:
- `src/dqx/evaluator.py` - Update the `evaluate` method

**Implementation steps**:
1. Change return type from `Result[float, dict[SymbolicMetric | sp.Expr, str]]` to `Result[float, list[EvaluationFailure]]`
2. Update the NaN/infinity handling with unified approach:

```python
def evaluate(self, expr: sp.Expr) -> Result[float, list[EvaluationFailure]]:
    """Evaluate a symbolic expression by substituting collected metric values."""
    sv = self._gather(expr)
    match sv:
        case Success(symbol_values):
            expr_val = float(sp.N(expr.subs(symbol_values), 6))

            # Unified handling for NaN and infinity
            if math.isnan(expr_val) or math.isinf(expr_val):
                # Build symbol info for all symbols in expression
                symbol_infos = []
                for sym in expr.free_symbols:
                    sm = self.metric_for_symbol(sym)
                    symbol_infos.append({
                        "name": str(sym),
                        "metric": sm.metric_spec.name,
                        "dataset": sm.dataset,
                        "value": symbol_values.get(sym),
                        "error": None
                    })

                # Choose appropriate error message
                if math.isnan(expr_val):
                    error_msg = "Validating value is NaN"
                else:
                    error_msg = "Validating value is infinity"

                failure = EvaluationFailure(
                    error_message=error_msg,
                    expression=str(expr),
                    symbols=symbol_infos
                )
                return Failure([failure])

            return Success(expr_val)

        case Failure(errors):
            return Failure(errors)

    # Unreachable state
    raise RuntimeError("Unreachable state in evaluation.")
```

**How to verify**:
```bash
# Run all evaluator tests
uv run pytest tests/test_evaluator.py -v

# Check linting
uv run ruff check src/dqx/evaluator.py
```

**Commit**: `feat: update Evaluator.evaluate to handle expression failures`

### Task 5: Update AssertionNode type annotation

**Branch**: Continue on same branch

**Files to modify**:
- `src/dqx/graph/nodes.py` - Update AssertionNode class

**Implementation steps**:
1. Add import: `from dqx.common import EvaluationFailure`
2. Update the `_value` type annotation:

```python
class AssertionNode(BaseNode["CheckNode"]):
    """Node representing an assertion to be evaluated."""

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
from dqx.common import EvaluationFailure, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider
from dqx.specs import Average, Sum
from dqx.orm.repositories import InMemoryMetricDB
from returns.result import Success, Failure


def test_end_to_end_metric_failure():
    """Test complete flow from provider to assertion with metric failure."""
    # Create a real MetricProvider with in-memory DB
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create metrics
    avg_price = provider.average("price", dataset="orders")
    sum_qty = provider.sum("quantity", dataset="inventory")

    # Mock one metric to fail
    # This would require setting up the DB to return failure
    # or mocking at the compute level

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

    # Set up metrics to return 0.0
    # Mock or set up DB appropriately

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
- Any test files that use `Evaluator.evaluate()` or check `AssertionNode._value`
- Search for affected tests:

```bash
# Find files that might need updates
grep -r "evaluate.*Failure" tests/
grep -r "assertion\._value" tests/
grep -r "SymbolicMetric.*str" tests/
```

**How to find and fix**:
1. Run all tests: `uv run pytest tests/ -v`
2. Fix any failures related to the type changes
3. Update assertions to check for `EvaluationFailure` objects

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

### Task 9: Documentation

**Branch**: Continue on same branch

**Files to update**:
- `README.md` - Update any examples that show error handling
- Create `docs/evaluation_failure_guide.md` if needed

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
            print(f"  - {symbol['name']} ({symbol['metric']}): "
                  f"value={symbol['value']}, error={symbol['error']}")
```
```

**Commit**: `docs: update documentation for EvaluationFailure`

## Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feat/evaluation-failure-v2
   ```

2. **Follow TDD cycle**:
   - Write failing test
   - Implement minimum code to pass
   - Refactor if needed
   - Commit frequently

3. **Commit guidelines**:
   - Use conventional commits (feat:, test:, docs:, chore:)
   - Keep commits small and focused
   - Write clear commit messages

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

1. **YAGNI (You Aren't Gonna Need It)**:
   - Only implement what's specified
   - Don't add extra features or generalizations

2. **DRY (Don't Repeat Yourself)**:
   - Reuse the symbol info building logic
   - Extract common patterns into helper functions if needed

3. **TDD (Test Driven Development)**:
   - Write tests first
   - Only write enough code to make tests pass
   - Refactor while keeping tests green

4. **Frequent Commits**:
   - Commit after each test passes
   - Commit after each refactoring
   - Keep commits atomic and focused

## Success Criteria

- [ ] All tests pass
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] Code coverage maintained or improved
- [ ] EvaluationFailure objects contain all required information
- [ ] Flattened structure is JSON-serializable for DB persistence
- [ ] NaN and infinity handling is unified
- [ ] No failure_type field needed

## Troubleshooting

1. **Import errors**: Make sure to import from the correct modules
2. **Type errors**: Use `uv run mypy` to catch type issues early
3. **Test failures**: Read error messages carefully, they usually point to the issue
4. **Mocking issues**: Use `unittest.mock` or `pytest-mock` for mocking the MetricProvider

## Notes for Implementation

- The `SymbolicMetric` class is in `src/dqx/provider.py`
- The `sp.Expr` type comes from sympy
- The `Result` type comes from the `returns` library
- Use `match` statements (Python 3.10+) for pattern matching
- The symbols list must be JSON-serializable (use basic Python types)
- Remember that AssertionNode already contains the expression in its `actual` field
