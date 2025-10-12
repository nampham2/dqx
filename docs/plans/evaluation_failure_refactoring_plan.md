# EvaluationFailure Refactoring Implementation Plan

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

## Solution Design

### New Data Structure

```python
@dataclass
class EvaluationFailure:
    """Unified failure information for evaluation errors"""
    name: str                  # Metric name OR expression string
    error_message: str         # The actual error
    dataset: str | None        # Single dataset, "multiple", or None
    context: dict[str, str]    # String-only context for DB persistence
```

### Context Field Specification

For metric failures:
```python
context = {
    "failure_type": "metric",
    "metric_spec": "Average('price')",     # String representation of the spec
    "symbol": "x_1",                       # The symbolic variable name
}
```

For expression failures (NaN/infinity):
```python
context = {
    "failure_type": "expression",
    "symbol_count": "2",                   # Number of symbols in expression
    "symbol_0_name": "average(price)",     # Name of first symbol
    "symbol_0_dataset": "orders",          # Dataset of first symbol
    "symbol_1_name": "sum(quantity)",      # Name of second symbol
    "symbol_1_dataset": "inventory",       # Dataset of second symbol
    # ... pattern continues for more symbols
}
```

## Implementation Tasks

### Task 1: Create the EvaluationFailure dataclass

**Branch**: `feat/evaluation-failure-dataclass`

**Files to modify**:
- `src/dqx/common.py` - Add the new dataclass

**Implementation**:
1. Add import: `from dataclasses import dataclass`
2. Add the EvaluationFailure dataclass definition (see design above)

**Testing**:
- No direct tests needed for a simple dataclass
- Will be tested through usage in subsequent tasks

**Commit**: `feat: add EvaluationFailure dataclass for structured error reporting`

### Task 2: Write failing tests for Evaluator changes

**Branch**: Continue on `feat/evaluation-failure-dataclass`

**Files to create/modify**:
- `tests/test_evaluator.py` - Create this new test file

**Implementation**:
```python
import math
import sympy as sp
import pytest
from dqx.common import EvaluationFailure, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.specs import Average
from returns.result import Success, Failure


class TestEvaluatorFailureHandling:
    """Test that Evaluator returns EvaluationFailure objects."""

    def test_metric_failure_returns_evaluation_failure(self):
        """Test that metric failures are converted to EvaluationFailure."""
        # This test will fail initially - that's expected!
        # Setup a mock provider that returns a failure
        provider = MetricProvider(db=None)  # We'll mock this
        key = ResultKey(yyyy_mm_dd=date.today())
        evaluator = Evaluator(provider, key)

        # Create a symbol that will fail
        symbol = sp.Symbol("x_1")
        # Mock the metric to return a failure
        # ... test implementation

        # Evaluate an expression with the failing symbol
        result = evaluator.evaluate(symbol)

        # Assert we get EvaluationFailure objects
        assert isinstance(result, Failure)
        failures = result.failure()
        assert len(failures) == 1
        assert isinstance(failures[0], EvaluationFailure)
        assert failures[0].name == "average(price)"
        assert failures[0].error_message == "Database error"
        assert failures[0].dataset == "orders"
        assert failures[0].context["failure_type"] == "metric"

    def test_expression_nan_returns_evaluation_failure(self):
        """Test that NaN results are converted to EvaluationFailure."""
        # Test for divide by zero resulting in NaN
        # ... test implementation

    def test_expression_infinity_returns_evaluation_failure(self):
        """Test that infinity results are converted to EvaluationFailure."""
        # Test for expressions that evaluate to infinity
        # ... test implementation

    def test_multiple_metric_failures(self):
        """Test handling of multiple metric failures in one expression."""
        # Test expression with multiple failing symbols
        # ... test implementation
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
    failures: list[EvaluationFailure] = []

    for sym in expr.free_symbols:
        if sym not in self.metrics:
            sm = self.metric_for_symbol(sym)
            raise DQXError(f"Symbol {sm.name} not found in collected metrics.")

        match self.metrics[sym]:
            case Failure(err):
                # Convert to EvaluationFailure
                sm = self.metric_for_symbol(sym)
                failure = EvaluationFailure(
                    name=sm.name,
                    error_message=err,
                    dataset=sm.dataset,
                    context={
                        "failure_type": "metric",
                        "metric_spec": str(sm.metric_spec),
                        "symbol": str(sym),
                    }
                )
                failures.append(failure)
            case Success(v):
                successes[sym] = v

    if failures:
        return Failure(failures)
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
2. Update the NaN/infinity handling:

```python
def evaluate(self, expr: sp.Expr) -> Result[float, list[EvaluationFailure]]:
    """Evaluate a symbolic expression by substituting collected metric values."""
    sv = self._gather(expr)
    match sv:
        case Success(symbol_values):
            expr_val = float(sp.N(expr.subs(symbol_values), 6))

            # Handling nan and inf values
            if math.isnan(expr_val):
                # Build context with all symbols involved
                involved_symbols = []
                for i, sym in enumerate(expr.free_symbols):
                    sm = self.metric_for_symbol(sym)
                    involved_symbols.append({
                        f"symbol_{i}_name": sm.name,
                        f"symbol_{i}_dataset": sm.dataset or "",
                    })

                context = {
                    "failure_type": "expression",
                    "symbol_count": str(len(expr.free_symbols)),
                }
                context.update({k: v for d in involved_symbols for k, v in d.items()})

                failure = EvaluationFailure(
                    name=str(expr),
                    error_message="Validating value is NaN",
                    dataset="multiple" if len(expr.free_symbols) > 1 else None,
                    context=context
                )
                return Failure([failure])

            elif math.isinf(expr_val):
                # Similar handling for infinity
                # ... (same pattern as NaN)

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

def test_end_to_end_metric_failure():
    """Test complete flow from provider to assertion with metric failure."""
    # Create a real MetricProvider with mocked DB
    # Create symbols that will fail
    # Build expressions
    # Evaluate and verify EvaluationFailure structure

def test_end_to_end_expression_failure():
    """Test complete flow for expression-level failures."""
    # Test divide by zero, NaN, infinity cases
    # Verify context includes all symbol information
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

**Commit**: `docs: update documentation for EvaluationFailure`

## Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feat/evaluation-failure-dataclass
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
   - Reuse the context building logic
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
- [ ] No backward compatibility issues (not required for this change)
- [ ] EvaluationFailure objects contain all required information
- [ ] Context field uses only string values for DB persistence

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
- The context dictionary must only contain string keys and values
