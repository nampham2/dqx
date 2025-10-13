# DatasetValidator Implementation Plan

## Overview

This document provides a step-by-step implementation plan for adding a new validator called `DatasetValidator` to the DQX data quality framework. The validator will catch dataset mismatches between CheckNodes and their AssertionNodes' symbols.

### What is DQX?
DQX is a data quality framework that validates data using a graph-based architecture. Key concepts:
- **CheckNode**: Groups related data quality assertions (e.g., "validate prices")
- **AssertionNode**: Individual validation rules (e.g., "price > 0")
- **Symbol**: Represents a metric to compute (e.g., average(price))
- **Dataset**: The data source a metric operates on (e.g., "production", "staging")

### The Problem
Currently, when a CheckNode specifies datasets (e.g., ["production", "staging"]) and an AssertionNode uses a symbol with a different dataset (e.g., "testing"), this mismatch isn't caught until runtime. We want to catch this during validation phase.

### Solution Approach
Add a new validator that runs during the validation phase to detect these mismatches early.

## Prerequisites

### Development Environment Setup
```bash
# 1. Clone the repository (if not already done)
git clone git@gitlab.com:booking-com/personal/nam.pham/dqx.git
cd dqx

# 2. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Set up development environment
./bin/setup-dev-env.sh

# 4. Verify setup
uv run pytest tests/test_validator.py -v  # Should pass
```

### Key Commands You'll Use
```bash
# Run tests
uv run pytest tests/test_validator.py -v

# Run specific test
uv run pytest tests/test_validator.py::test_dataset_validator -v

# Check code quality
uv run mypy src/dqx/validator.py
uv run ruff check src/dqx/validator.py

# Auto-fix linting issues
uv run ruff check --fix src/dqx/validator.py

# Run all quality checks
./bin/run-hooks.sh --all
```

## Implementation Tasks

### Task 1: Write Failing Test for DatasetValidator

**Goal**: Follow TDD - write the test first, watch it fail.

**File to create**: `tests/test_dataset_validator.py`

```python
"""Tests for DatasetValidator."""

import pytest
from dqx.graph.nodes import RootNode
from dqx.provider import MetricProvider
from dqx.validator import DatasetValidator, ValidationIssue


def test_dataset_validator_detects_mismatch():
    """Test that DatasetValidator catches dataset mismatches."""
    # Arrange: Create a graph with dataset mismatch
    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production", "staging"])

    # Create a mock provider with a symbol that has "testing" dataset
    provider = MockProvider()
    symbol = provider.average("price", dataset="testing")

    # Add assertion using the symbol
    check.add_assertion(symbol, name="avg price > 0")

    # Act: Run the validator
    validator = DatasetValidator(provider)
    validator.process_node(check)
    validator.finalize()

    # Assert: Should have one error
    issues = validator.get_issues()
    assert len(issues) == 1
    assert issues[0].rule == "dataset_mismatch"
    assert "testing" in issues[0].message
    assert "production" in issues[0].message


class MockProvider:
    """Minimal mock provider for testing."""

    def __init__(self):
        self._metrics = []
        self._symbol_index = {}

    def average(self, column, dataset=None):
        # Implementation will be refined later
        pass
```

**Commit after writing test**:
```bash
git add tests/test_dataset_validator.py
git commit -m "test: add failing test for DatasetValidator"
```

### Task 2: Create Minimal DatasetValidator Class

**Goal**: Make the test compile (but still fail).

**File to modify**: `src/dqx/validator.py`

Add this class after the existing validators:

```python
class DatasetValidator(BaseValidator):
    """Detects dataset mismatches between CheckNodes and their AssertionNodes' symbols."""

    name = "dataset_mismatch"
    is_error = True

    def __init__(self, provider: MetricProvider | None = None) -> None:
        """Initialize validator with optional provider."""
        super().__init__()
        self._provider = provider

    def process_node(self, node: BaseNode) -> None:
        """Process a node to check for dataset mismatches."""
        # TODO: Implement in next task
        pass
```

**Also add import at top of file**:
```python
from dqx.provider import MetricProvider
```

**Run test to confirm it fails**:
```bash
uv run pytest tests/test_dataset_validator.py -v
# Should fail because process_node is not implemented
```

**Commit**:
```bash
git add src/dqx/validator.py
git commit -m "feat: add minimal DatasetValidator class structure"
```

### Task 3: Implement Basic Dataset Validation Logic

**Goal**: Make the test pass with minimal implementation.

**File to modify**: `src/dqx/validator.py`

Update the `DatasetValidator.process_node` method:

```python
def process_node(self, node: BaseNode) -> None:
    """Process a node to check for dataset mismatches."""
    if not self._provider:
        return

    if isinstance(node, CheckNode) and node.datasets:
        # Store check datasets for later validation
        self._check_datasets[node] = node.datasets

    elif isinstance(node, AssertionNode):
        # Check if parent has datasets
        parent_check = node.parent
        if parent_check not in self._check_datasets:
            return

        parent_datasets = self._check_datasets[parent_check]

        # Extract symbols from assertion expression
        symbols = node.actual.free_symbols

        for symbol in symbols:
            try:
                metric = self._provider.get_symbol(symbol)
                if metric.dataset and metric.dataset not in parent_datasets:
                    self._issues.append(
                        ValidationIssue(
                            rule=self.name,
                            message=(
                                f"Symbol '{metric.name}' in assertion '{node.name}' "
                                f"has dataset '{metric.dataset}' which is not in "
                                f"parent check '{parent_check.name}' datasets: {parent_datasets}"
                            ),
                            node_path=[
                                "root",
                                f"check:{parent_check.name}",
                                f"assertion:{node.name}",
                            ],
                        )
                    )
            except Exception:
                # Symbol not found in provider, skip
                pass
```

**Also add to `__init__`**:
```python
def __init__(self, provider: MetricProvider | None = None) -> None:
    """Initialize validator with optional provider."""
    super().__init__()
    self._provider = provider
    self._check_datasets: dict[CheckNode, list[str]] = {}
```

**And update `reset` method**:
```python
def reset(self) -> None:
    """Reset validator state."""
    super().reset()
    self._check_datasets.clear()
```

**Run test**:
```bash
uv run pytest tests/test_dataset_validator.py -v
# Still might fail - need to refine the test mock
```

**Commit**:
```bash
git add src/dqx/validator.py
git commit -m "feat: implement basic dataset validation logic"
```

### Task 4: Fix Test with Proper Mocking

**Goal**: Create a working mock provider to make test pass.

**File to modify**: `tests/test_dataset_validator.py`

Replace the mock with a complete implementation:

```python
import sympy as sp
from dataclasses import dataclass
from dqx.specs import Average


@dataclass
class MockSymbolicMetric:
    """Mock SymbolicMetric for testing."""

    name: str
    symbol: sp.Symbol
    dataset: str | None = None
    metric_spec: Any = None


class MockProvider:
    """Mock provider that tracks symbols and their datasets."""

    def __init__(self):
        self._symbol_counter = 0
        self._symbol_index = {}

    def average(self, column: str, dataset: str | None = None) -> sp.Symbol:
        """Create a mock average metric."""
        self._symbol_counter += 1
        symbol = sp.Symbol(f"x_{self._symbol_counter}")

        metric = MockSymbolicMetric(
            name=f"average({column})",
            symbol=symbol,
            dataset=dataset,
            metric_spec=Average(column),
        )

        self._symbol_index[symbol] = metric
        return symbol

    def get_symbol(self, symbol: sp.Symbol) -> MockSymbolicMetric:
        """Get the metric for a symbol."""
        if symbol not in self._symbol_index:
            raise KeyError(f"Symbol {symbol} not found")
        return self._symbol_index[symbol]
```

**Run test again**:
```bash
uv run pytest tests/test_dataset_validator.py -v
# Should pass now!
```

**Commit**:
```bash
git add tests/test_dataset_validator.py
git commit -m "test: fix DatasetValidator test with proper mocking"
```

### Task 5: Add Test for Valid Configuration

**Goal**: Ensure validator doesn't flag valid configurations.

**File to modify**: `tests/test_dataset_validator.py`

Add this test:

```python
def test_dataset_validator_allows_valid_configuration():
    """Test that DatasetValidator allows matching datasets."""
    # Arrange: Create a graph with matching datasets
    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production", "staging"])

    provider = MockProvider()
    # Symbol has dataset that IS in check's datasets
    symbol = provider.average("price", dataset="production")

    check.add_assertion(symbol, name="avg price > 0")

    # Act: Run the validator
    validator = DatasetValidator(provider)
    validator.process_node(check)
    # Process assertion manually since we're not using full traversal
    for child in check.children:
        validator.process_node(child)
    validator.finalize()

    # Assert: Should have no errors
    issues = validator.get_issues()
    assert len(issues) == 0
```

**Run test**:
```bash
uv run pytest tests/test_dataset_validator.py::test_dataset_validator_allows_valid_configuration -v
```

**Commit**:
```bash
git add tests/test_dataset_validator.py
git commit -m "test: add test for valid dataset configuration"
```

### Task 6: Add Test for Edge Cases

**Goal**: Handle edge cases properly.

**File to modify**: `tests/test_dataset_validator.py`

Add these tests:

```python
def test_dataset_validator_skips_when_no_datasets_specified():
    """Test that validator skips validation when check has no datasets."""
    root = RootNode("test_suite")
    # Check has no datasets specified
    check = root.add_check("price_check")

    provider = MockProvider()
    symbol = provider.average("price", dataset="testing")
    check.add_assertion(symbol, name="avg price > 0")

    validator = DatasetValidator(provider)
    validator.process_node(check)
    for child in check.children:
        validator.process_node(child)
    validator.finalize()

    # Should have no errors since check doesn't specify datasets
    assert len(validator.get_issues()) == 0


def test_dataset_validator_skips_when_symbol_has_no_dataset():
    """Test that validator skips when symbol has no dataset."""
    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production"])

    provider = MockProvider()
    # Symbol has no dataset specified
    symbol = provider.average("price", dataset=None)
    check.add_assertion(symbol, name="avg price > 0")

    validator = DatasetValidator(provider)
    validator.process_node(check)
    for child in check.children:
        validator.process_node(child)
    validator.finalize()

    # Should have no errors since symbol doesn't specify dataset
    assert len(validator.get_issues()) == 0


def test_dataset_validator_works_without_provider():
    """Test that validator handles missing provider gracefully."""
    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production"])

    # No provider passed
    validator = DatasetValidator(provider=None)
    validator.process_node(check)

    # Should not crash, should have no issues
    assert len(validator.get_issues()) == 0
```

**Run all tests**:
```bash
uv run pytest tests/test_dataset_validator.py -v
```

**Commit**:
```bash
git add tests/test_dataset_validator.py
git commit -m "test: add edge case tests for DatasetValidator"
```

### Task 7: Integrate DatasetValidator into SuiteValidator

**Goal**: Make DatasetValidator run as part of suite validation.

**File to modify**: `src/dqx/validator.py`

1. Update `SuiteValidator.__init__` to create DatasetValidator:

```python
def __init__(self) -> None:
    """Initialize with built-in validators."""
    self._validators = [
        DuplicateCheckNameValidator(),
        EmptyCheckValidator(),
        DuplicateAssertionNameValidator(),
        # Note: DatasetValidator requires provider, will be set in validate()
    ]
    self._dataset_validator = DatasetValidator()  # Keep separate reference
```

2. Update `validate` method signature and implementation:

```python
def validate(
    self, graph: Graph, provider: MetricProvider | None = None
) -> ValidationReport:
    """Run validation on a graph.

    Args:
        graph: The graph to validate
        provider: Optional provider for dataset validation

    Returns:
        ValidationReport with all issues found
    """
    # Build validator list, including DatasetValidator if provider available
    validators = self._validators.copy()
    if provider:
        self._dataset_validator._provider = provider
        validators.append(self._dataset_validator)

    # Create composite with current validators
    self._composite = CompositeValidationVisitor(validators)

    # Single-pass traversal
    graph.bfs(self._composite)

    # Get all issues
    issues = self._composite.get_all_issues()

    # Build report
    report = ValidationReport()
    for error in issues["errors"]:
        report.add_error(error)
    for warning in issues["warnings"]:
        report.add_warning(warning)

    return report
```

**Run validator tests to ensure nothing broke**:
```bash
uv run pytest tests/test_validator.py -v
```

**Commit**:
```bash
git add src/dqx/validator.py
git commit -m "feat: integrate DatasetValidator into SuiteValidator"
```

### Task 8: Update API to Pass Provider

**Goal**: Update API calls to pass provider when available.

**Files to check**:
- `src/dqx/api.py` - Look for calls to `validate()`

**Search for validate calls**:
```bash
grep -n "validate(" src/dqx/api.py
```

You'll find two places to update:

1. In the `plan` method:
```python
# Change from:
return self._validator.validate(temp_context._graph)
# To:
return self._validator.validate(temp_context._graph, temp_context.provider)
```

2. In the `run` method:
```python
# Change from:
report = self._validator.validate(context._graph)
# To:
report = self._validator.validate(context._graph, context.provider)
```

**Run API tests to ensure nothing broke**:
```bash
uv run pytest tests/test_api.py -v -k validate
```

**Commit**:
```bash
git add src/dqx/api.py
git commit -m "feat: update API to pass provider to validator"
```

### Task 9: Add Integration Test

**Goal**: Test the full flow from API to validation.

**File to create**: `tests/test_dataset_validator_integration.py`

```python
"""Integration tests for DatasetValidator with full API flow."""

import pytest
from dqx.api import VerificationSuiteBuilder, check
from dqx.common import DQXError
from dqx.orm.repositories import InMemoryMetricDB


def test_dataset_validator_prevents_suite_execution():
    """Test that dataset mismatch prevents suite execution."""
    db = InMemoryMetricDB()

    @check(name="price_check", datasets=["production", "staging"])
    def validate_prices(mp, ctx):
        # This creates a symbol with "testing" dataset - mismatch!
        avg_price = mp.average("price", dataset="testing")
        ctx.assert_that(avg_price).where(name="avg price > 0").is_gt(0)

    suite = (
        VerificationSuiteBuilder("test_suite", db).add_check(validate_prices).build()
    )

    # Act & Assert: Should raise validation error
    with pytest.raises(DQXError, match="dataset_mismatch"):
        report = suite.plan()
        # If we got here, validation didn't catch the error
        if report.has_errors():
            raise DQXError(str(report))


def test_dataset_validator_allows_valid_flow():
    """Test that valid dataset configuration works."""
    db = InMemoryMetricDB()

    @check(name="price_check", datasets=["production", "staging"])
    def validate_prices(mp, ctx):
        # Symbol uses "production" which IS in check's datasets
        avg_price = mp.average("price", dataset="production")
        ctx.assert_that(avg_price).where(name="avg price > 0").is_gt(0)

    suite = (
        VerificationSuiteBuilder("test_suite", db).add_check(validate_prices).build()
    )

    # Should not raise
    report = suite.plan()
    assert not report.has_errors()
```

**Run integration test**:
```bash
uv run pytest tests/test_dataset_validator_integration.py -v
```

**Commit**:
```bash
git add tests/test_dataset_validator_integration.py
git commit -m "test: add integration tests for DatasetValidator"
```

### Task 10: Run Full Test Suite

**Goal**: Ensure all tests still pass.

```bash
# Run all tests
uv run pytest -v

# Run code quality checks
uv run mypy src/dqx/validator.py
uv run ruff check src/dqx/validator.py

# Run all pre-commit hooks
./bin/run-hooks.sh --all
```

Fix any issues that arise.

**Final commit**:
```bash
git add -A
git commit -m "feat: complete DatasetValidator implementation"
```

## Testing Your Implementation

### Manual Testing

Create a test file `test_manual.py`:

```python
from dqx.api import VerificationSuiteBuilder, check
from dqx.orm.repositories import InMemoryMetricDB

db = InMemoryMetricDB()


@check(name="test_check", datasets=["prod", "staging"])
def my_check(mp, ctx):
    # This should trigger an error - "testing" not in ["prod", "staging"]
    metric = mp.average("column", dataset="testing")
    ctx.assert_that(metric).where(name="test").is_gt(0)


suite = VerificationSuiteBuilder("test", db).add_check(my_check).build()

try:
    report = suite.plan()
    print("Validation report:", report)
except Exception as e:
    print("Caught expected error:", e)
```

Run it:
```bash
uv run python test_manual.py
```

### Verify Error Messages

The error message should be clear and helpful:
```
Symbol 'average(column)' in assertion 'test' has dataset 'testing' which is not in parent check 'test_check' datasets: ['prod', 'staging']
```

## Common Issues and Solutions

### Issue: Import Errors
**Solution**: Make sure to add necessary imports at the top of files:
```python
from dqx.provider import MetricProvider  # In validator.py
import sympy as sp  # In tests
```

### Issue: Tests Failing Due to Missing Methods
**Solution**: Check that you've implemented all required methods:
- `__init__`
- `process_node`
- `reset`
- `finalize` (if needed)

### Issue: MyPy Type Errors
**Solution**: Add type annotations:
```python
from typing import Optional

provider: Optional[MetricProvider] = None
```

### Issue: Circular Import
**Solution**: Use string literals for type hints:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dqx.provider import MetricProvider

# Then use quotes: provider: "MetricProvider | None"
```

## Summary

By following this plan, you will have:
1. Created a new DatasetValidator that catches dataset mismatches
2. Followed TDD by writing tests first
3. Made small, focused commits
4. Integrated the validator into the existing validation system
5. Updated the API to support the new validator
6. Added comprehensive tests including edge cases

The implementation follows YAGNI (only what's needed), DRY (reuses existing base classes), and maintains backward compatibility (provider parameter is optional).
