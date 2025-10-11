# DatasetValidator Implementation Plan (v1)

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
Add a new validator that runs during the validation phase to detect these mismatches early. The validator will:
- Only process AssertionNodes (no need to track CheckNodes separately)
- Access parent datasets via `node.parent.datasets`
- Require a MetricProvider to look up symbol datasets

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

**Goal**: Follow TDD - write the test first using real MetricProvider.

**File to create**: `tests/test_dataset_validator.py`

```python
"""Tests for DatasetValidator."""

import pytest
from dqx.graph.nodes import RootNode
from dqx.provider import MetricProvider
from dqx.orm.repositories import InMemoryMetricDB
from dqx.validator import DatasetValidator, ValidationIssue


def test_dataset_validator_detects_mismatch():
    """Test that DatasetValidator catches dataset mismatches."""
    # Arrange: Create a real provider
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create a graph with dataset mismatch
    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production", "staging"])

    # Create a symbol with "testing" dataset - mismatch!
    symbol = provider.average("price", dataset="testing")

    # Add assertion using the symbol
    check.add_assertion(symbol, name="avg price > 0")

    # Act: Run the validator
    validator = DatasetValidator(provider)
    # Process only the assertion node
    for child in check.children:
        validator.process_node(child)

    # Assert: Should have one error
    issues = validator.get_issues()
    assert len(issues) == 1
    assert issues[0].rule == "dataset_mismatch"
    assert "testing" in issues[0].message
    assert "production" in issues[0].message
    assert "staging" in issues[0].message
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

    def __init__(self, provider: MetricProvider) -> None:
        """Initialize validator with provider."""
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

### Task 3: Implement Dataset Validation Logic

**Goal**: Make the test pass with clean implementation.

**File to modify**: `src/dqx/validator.py`

Update the `DatasetValidator.process_node` method:

```python
def process_node(self, node: BaseNode) -> None:
    """Process a node to check for dataset mismatches."""
    if not isinstance(node, AssertionNode):
        return

    parent_check = node.parent

    # Only validate if parent check has datasets specified
    if not parent_check.datasets:
        return

    parent_datasets = parent_check.datasets

    # Extract symbols from assertion expression
    symbols = node.actual.free_symbols

    for symbol in symbols:
        try:
            metric = self._provider.get_symbol(symbol)
            # Only validate if symbol has a dataset specified
            if metric.dataset and metric.dataset not in parent_datasets:
                self._issues.append(
                    ValidationIssue(
                        rule=self.name,
                        message=(
                            f"Symbol '{metric.name}' in assertion '{node.name}' "
                            f"has dataset '{metric.dataset}' which is not in "
                            f"parent check '{parent_check.name}' datasets: {parent_datasets}"
                        ),
                        node_path=["root", f"check:{parent_check.name}", f"assertion:{node.name}"]
                    )
                )
        except Exception:
            # Symbol not found in provider, skip silently
            # This can happen during early validation before all symbols are registered
            pass
```

**Run test**:
```bash
uv run pytest tests/test_dataset_validator.py -v
# Should pass now!
```

**Commit**:
```bash
git add src/dqx/validator.py
git commit -m "feat: implement dataset validation logic"
```

### Task 4: Add Test for Valid Configuration

**Goal**: Ensure validator doesn't flag valid configurations.

**File to modify**: `tests/test_dataset_validator.py`

Add this test:

```python
def test_dataset_validator_allows_valid_configuration():
    """Test that DatasetValidator allows matching datasets."""
    # Arrange: Create a real provider
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create a graph with matching datasets
    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production", "staging"])

    # Symbol has dataset that IS in check's datasets
    symbol = provider.average("price", dataset="production")

    check.add_assertion(symbol, name="avg price > 0")

    # Act: Run the validator
    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

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

### Task 5: Add Test for Edge Cases

**Goal**: Handle edge cases properly.

**File to modify**: `tests/test_dataset_validator.py`

Add these tests:

```python
def test_dataset_validator_skips_when_no_datasets_specified():
    """Test that validator skips validation when check has no datasets."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    # Check has no datasets specified
    check = root.add_check("price_check")

    symbol = provider.average("price", dataset="testing")
    check.add_assertion(symbol, name="avg price > 0")

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have no errors since check doesn't specify datasets
    assert len(validator.get_issues()) == 0


def test_dataset_validator_skips_when_symbol_has_no_dataset():
    """Test that validator skips when symbol has no dataset."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production"])

    # Symbol has no dataset specified
    symbol = provider.average("price", dataset=None)
    check.add_assertion(symbol, name="avg price > 0")

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have no errors since symbol doesn't specify dataset
    assert len(validator.get_issues()) == 0


def test_dataset_validator_handles_multiple_symbols():
    """Test validator with multiple symbols in one assertion."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production"])

    # Create multiple symbols - one invalid, one valid
    valid_symbol = provider.average("price", dataset="production")
    invalid_symbol = provider.average("cost", dataset="testing")

    # Assertion with expression using both symbols
    check.add_assertion(
        valid_symbol + invalid_symbol,
        name="combined metric"
    )

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have one error for the invalid symbol
    issues = validator.get_issues()
    assert len(issues) == 1
    assert "average(cost)" in issues[0].message
    assert "testing" in issues[0].message
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

### Task 6: Update SuiteValidator to Include DatasetValidator

**Goal**: Integrate DatasetValidator into the validation system.

**File to modify**: `src/dqx/validator.py`

1. Update `SuiteValidator.__init__` to store a reference to DatasetValidator:

```python
def __init__(self) -> None:
    """Initialize with built-in validators."""
    self._validators = [
        DuplicateCheckNameValidator(),
        EmptyCheckValidator(),
        DuplicateAssertionNameValidator(),
    ]
    # DatasetValidator needs special handling due to provider requirement
    self._dataset_validator: DatasetValidator | None = None
```

2. Update `validate` method signature and implementation:

```python
def validate(self, graph: Graph, provider: MetricProvider | None = None) -> ValidationReport:
    """Run validation on a graph.

    Args:
        graph: The graph to validate
        provider: Optional provider for dataset validation

    Returns:
        ValidationReport with all issues found
    """
    # Build validator list
    validators = self._validators.copy()

    # Add DatasetValidator if provider is available
    if provider:
        dataset_validator = DatasetValidator(provider)
        validators.append(dataset_validator)

    # Create composite with current validators
    composite = CompositeValidationVisitor(validators)

    # Single-pass traversal
    graph.bfs(composite)

    # Get all issues
    issues = composite.get_all_issues()

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

### Task 7: Update API to Pass Provider

**Goal**: Update API calls to pass provider when available.

**File to modify**: `src/dqx/api.py`

Find and update the two validation calls:

1. In the `plan` method (around line 250-260):
```python
# Change from:
return self._validator.validate(temp_context._graph)
# To:
return self._validator.validate(temp_context._graph, temp_context.provider)
```

2. In the `run` method (around line 280-290):
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

### Task 8: Add Integration Test

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
        VerificationSuiteBuilder("test_suite", db)
        .add_check(validate_prices)
        .build()
    )

    # Act & Assert: plan() should raise validation error
    report = suite.plan()
    assert report.has_errors()
    assert "dataset_mismatch" in str(report)
    assert "testing" in str(report)


def test_dataset_validator_allows_valid_flow():
    """Test that valid dataset configuration works."""
    db = InMemoryMetricDB()

    @check(name="price_check", datasets=["production", "staging"])
    def validate_prices(mp, ctx):
        # Symbol uses "production" which IS in check's datasets
        avg_price = mp.average("price", dataset="production")
        ctx.assert_that(avg_price).where(name="avg price > 0").is_gt(0)

    suite = (
        VerificationSuiteBuilder("test_suite", db)
        .add_check(validate_prices)
        .build()
    )

    # Should not have validation errors
    report = suite.plan()
    assert not report.has_errors()


def test_dataset_validator_with_multiple_checks():
    """Test validator across multiple checks with different datasets."""
    db = InMemoryMetricDB()

    @check(name="prod_check", datasets=["production"])
    def check_production(mp, ctx):
        # Valid: uses production dataset
        metric = mp.average("revenue", dataset="production")
        ctx.assert_that(metric).where(name="prod revenue > 0").is_gt(0)

    @check(name="staging_check", datasets=["staging", "testing"])
    def check_staging(mp, ctx):
        # Invalid: uses production dataset in staging check
        metric = mp.average("users", dataset="production")
        ctx.assert_that(metric).where(name="staging users > 0").is_gt(0)

    suite = (
        VerificationSuiteBuilder("multi_check_suite", db)
        .add_check(check_production)
        .add_check(check_staging)
        .build()
    )

    report = suite.plan()
    assert report.has_errors()
    errors = [issue.message for issue in report.errors]

    # Should only have error for staging check
    assert len(errors) == 1
    assert "staging_check" in errors[0]
    assert "production" in errors[0]
    assert "staging" in errors[0]
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

### Task 9: Run Full Test Suite and Fix Issues

**Goal**: Ensure all tests pass and code quality is good.

```bash
# Run all tests
uv run pytest -v

# Run code quality checks
uv run mypy src/dqx/validator.py
uv run ruff check src/dqx/validator.py

# Fix any linting issues
uv run ruff check --fix src/dqx/validator.py

# Run all pre-commit hooks
./bin/run-hooks.sh --all
```

Fix any issues that arise.

**Commit any fixes**:
```bash
git add -A
git commit -m "fix: address linting and type checking issues"
```

### Task 10: Add Documentation

**Goal**: Document the new validator for other developers.

**File to modify**: `src/dqx/validator.py`

Add comprehensive docstring to DatasetValidator:

```python
class DatasetValidator(BaseValidator):
    """Detects dataset mismatches between CheckNodes and their AssertionNodes' symbols.

    This validator ensures that when a CheckNode specifies allowed datasets,
    all symbols used in its assertions must either:
    - Have no dataset specified (will be imputed later), or
    - Have a dataset that is in the check's allowed datasets

    Example of invalid configuration that this catches:
        @check(name="prod_check", datasets=["production"])
        def my_check(mp, ctx):
            # ERROR: Symbol uses "staging" but check only allows "production"
            metric = mp.average("revenue", dataset="staging")
            ctx.assert_that(metric).where(name="revenue > 0").is_gt(0)

    The validator requires a MetricProvider to look up symbol datasets,
    so it only runs when a provider is available (during plan() and run()).

    Attributes:
        _provider: MetricProvider used to look up symbol information
    """
```

**Final commit**:
```bash
git add src/dqx/validator.py
git commit -m "docs: add comprehensive documentation for DatasetValidator"
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

report = suite.plan()
print("Validation report:")
print(report)
print(f"\nHas errors: {report.has_errors()}")
```

Run it:
```bash
uv run python test_manual.py
```

Expected output should show:
```
1 ERROR(S):
  [dataset_mismatch] Symbol 'average(column)' in assertion 'test' has dataset 'testing' which is not in parent check 'test_check' datasets: ['prod', 'staging']
    Path: root > check:test_check > assertion:test
```

## Key Design Improvements in v1

1. **Mandatory Provider**: The provider parameter is now required, ensuring the validator always has the necessary context to perform validation.

2. **Simplified Implementation**: No need to track `_check_datasets` - we access parent datasets directly via `node.parent.datasets`.

3. **Real Classes in Tests**: Using actual `MetricProvider` and `InMemoryMetricDB` instead of mocks makes tests more realistic and easier to understand.

4. **Focused Processing**: Only processes AssertionNodes, making the logic clearer and more efficient.

## Common Issues and Solutions

### Issue: Import Errors
**Solution**: Make sure to add necessary imports:
```python
from dqx.provider import MetricProvider  # In validator.py
from dqx.orm.repositories import InMemoryMetricDB  # In tests
```

### Issue: Type Checking Errors
**Solution**: The provider is now mandatory, so type hints are simpler:
```python
def __init__(self, provider: MetricProvider) -> None:
```

### Issue: Tests Failing Due to Symbol Not Found
**Solution**: The implementation handles this gracefully with try/except, allowing early validation before all symbols are registered.

## Summary

This implementation:
- Creates a focused DatasetValidator that catches dataset mismatches early
- Uses clean, simple code that leverages existing graph structure
- Follows TDD with comprehensive test coverage
- Integrates seamlessly with the existing validation system
- Provides clear, actionable error messages

The validator will help catch configuration errors during the planning phase, preventing runtime failures and improving the developer experience.
