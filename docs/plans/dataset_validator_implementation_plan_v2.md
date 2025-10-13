# DatasetValidator Implementation Plan (v2)

## Overview

This document provides a step-by-step implementation plan for adding a new validator called `DatasetValidator` to the DQX data quality framework. The validator will catch dataset mismatches and ambiguities between CheckNodes and their AssertionNodes' symbols.

### What is DQX?
DQX is a data quality framework that validates data using a graph-based architecture. Key concepts:
- **CheckNode**: Groups related data quality assertions (e.g., "validate prices")
- **AssertionNode**: Individual validation rules (e.g., "price > 0")
- **Symbol**: Represents a metric to compute (e.g., average(price))
- **Dataset**: The data source a metric operates on (e.g., "production", "staging")

### The Problem
Two critical issues need to be caught during validation:
1. **Dataset Mismatch**: When a CheckNode specifies datasets (e.g., ["production", "staging"]) and an AssertionNode uses a symbol with a different dataset (e.g., "testing")
2. **Dataset Ambiguity**: When a symbol has no dataset specified (`dataset=None`) but the parent check has multiple datasets, making it impossible to determine which dataset to use during imputation

### Solution Approach
Add a new validator that runs during the validation phase to detect these issues early. The validator will:
- Only process AssertionNodes (no need to track CheckNodes separately)
- Access parent datasets via `node.parent.datasets`
- Require a MetricProvider to look up symbol datasets
- Detect both mismatches and ambiguous configurations

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

### Task 3: Implement Dataset Validation Logic (Updated)

**Goal**: Make the test pass with comprehensive validation logic that handles ambiguity.

**File to modify**: `src/dqx/validator.py`

Update the `DatasetValidator.process_node` method with the enhanced logic:

```python
def process_node(self, node: BaseNode) -> None:
    """Process a node to check for dataset mismatches and ambiguities."""
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

            if metric.dataset is None:
                # If check has multiple datasets, this is ambiguous
                if len(parent_datasets) > 1:
                    self._issues.append(
                        ValidationIssue(
                            rule=self.name,
                            message=(
                                f"Symbol '{metric.name}' in assertion '{node.name}' "
                                f"has no dataset specified, but parent check '{parent_check.name}' "
                                f"has multiple datasets: {parent_datasets}. Unable to determine which dataset to use."
                            ),
                            node_path=[
                                "root",
                                f"check:{parent_check.name}",
                                f"assertion:{node.name}",
                            ],
                        )
                    )
                # If check has exactly one dataset, imputation will handle it
                continue

            # Validate symbol's dataset is in parent's datasets
            if metric.dataset not in parent_datasets:
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
git commit -m "feat: implement dataset validation with ambiguity detection"
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

### Task 5: Add Comprehensive Edge Case Tests (Updated)

**Goal**: Handle all edge cases including the ambiguity scenario.

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


def test_dataset_validator_errors_on_ambiguous_none_dataset():
    """Test that validator errors when symbol has no dataset but check has multiple."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    # Check has multiple datasets
    check = root.add_check("price_check", datasets=["production", "staging"])

    # Symbol has no dataset - ambiguous!
    symbol = provider.average("price", dataset=None)
    check.add_assertion(symbol, name="avg price > 0")

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have error about ambiguity
    issues = validator.get_issues()
    assert len(issues) == 1
    assert "no dataset specified" in issues[0].message
    assert "multiple datasets" in issues[0].message
    assert "Unable to determine" in issues[0].message


def test_dataset_validator_allows_none_dataset_with_single_check_dataset():
    """Test that validator allows None dataset when check has single dataset."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    # Check has single dataset
    check = root.add_check("price_check", datasets=["production"])

    # Symbol has no dataset - OK, will be imputed
    symbol = provider.average("price", dataset=None)
    check.add_assertion(symbol, name="avg price > 0")

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have no errors
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
    check.add_assertion(valid_symbol + invalid_symbol, name="combined metric")

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
git commit -m "test: add comprehensive edge case tests including ambiguity"
```

### Task 6: Update SuiteValidator with Mandatory Provider (Updated)

**Goal**: Integrate DatasetValidator into the validation system with mandatory provider.

**File to modify**: `src/dqx/validator.py`

1. Update `SuiteValidator.__init__` (no changes needed):

```python
def __init__(self) -> None:
    """Initialize with built-in validators."""
    self._validators = [
        DuplicateCheckNameValidator(),
        EmptyCheckValidator(),
        DuplicateAssertionNameValidator(),
    ]
```

2. Update `validate` method signature to make provider mandatory:

```python
def validate(self, graph: Graph, provider: MetricProvider) -> ValidationReport:
    """Run validation on a graph.

    Args:
        graph: The graph to validate
        provider: MetricProvider for dataset validation (required)

    Returns:
        ValidationReport with all issues found
    """
    # Build validator list including DatasetValidator
    validators = self._validators.copy()
    dataset_validator = DatasetValidator(provider)
    validators.append(dataset_validator)

    # Create composite with all validators
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
git commit -m "feat: integrate DatasetValidator with mandatory provider"
```

### Task 7: Update API to Pass Provider

**Goal**: Update API calls to pass provider (now mandatory).

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
git commit -m "feat: update API to pass mandatory provider to validator"
```

### Task 8: Add Integration Tests (Updated)

**Goal**: Test the full flow including ambiguity detection.

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

    # Act & Assert: plan() should raise validation error
    report = suite.plan()
    assert report.has_errors()
    assert "dataset_mismatch" in str(report)
    assert "testing" in str(report)


def test_dataset_validator_catches_ambiguity():
    """Test that ambiguous dataset configuration is caught."""
    db = InMemoryMetricDB()

    @check(name="multi_dataset_check", datasets=["production", "staging"])
    def validate_ambiguous(mp, ctx):
        # Symbol has no dataset - ambiguous with multiple parent datasets!
        metric = mp.average("revenue")  # dataset=None by default
        ctx.assert_that(metric).where(name="revenue > 0").is_gt(0)

    suite = (
        VerificationSuiteBuilder("ambiguous_suite", db)
        .add_check(validate_ambiguous)
        .build()
    )

    # Should catch the ambiguity
    report = suite.plan()
    assert report.has_errors()
    assert "no dataset specified" in str(report)
    assert "multiple datasets" in str(report)


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

    # Should not have validation errors
    report = suite.plan()
    assert not report.has_errors()


def test_dataset_validator_with_single_dataset_imputation():
    """Test that None dataset works with single parent dataset."""
    db = InMemoryMetricDB()

    @check(name="single_dataset_check", datasets=["production"])
    def validate_single(mp, ctx):
        # Symbol has no dataset but check has single dataset - OK
        metric = mp.average("users")  # dataset=None
        ctx.assert_that(metric).where(name="users > 0").is_gt(0)

    suite = (
        VerificationSuiteBuilder("single_dataset_suite", db)
        .add_check(validate_single)
        .build()
    )

    # Should not have errors - imputation will handle it
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
git commit -m "test: add integration tests including ambiguity detection"
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

### Task 10: Add Comprehensive Documentation

**Goal**: Document the new validator including ambiguity detection.

**File to modify**: `src/dqx/validator.py`

Add comprehensive docstring to DatasetValidator:

```python
class DatasetValidator(BaseValidator):
    """Detects dataset mismatches and ambiguities between CheckNodes and their AssertionNodes' symbols.

    This validator ensures dataset consistency by checking two conditions:

    1. **Dataset Mismatch**: When a symbol has a specific dataset, it must be
       in the parent check's allowed datasets.

    2. **Dataset Ambiguity**: When a symbol has no dataset (dataset=None) and
       the parent check has multiple datasets, it's impossible to determine
       which dataset to use during imputation. This is flagged as an error.

    Valid configurations:
    - Symbol has dataset that matches one of check's datasets ✓
    - Symbol has no dataset and check has single dataset (will be imputed) ✓
    - Check has no datasets specified (no validation performed) ✓

    Invalid configurations:
    - Symbol has dataset not in check's datasets ✗
    - Symbol has no dataset and check has multiple datasets (ambiguous) ✗

    Examples:
        # Invalid - dataset mismatch
        @check(name="prod_check", datasets=["production"])
        def my_check(mp, ctx):
            metric = mp.average("revenue", dataset="staging")  # ERROR

        # Invalid - ambiguous
        @check(name="multi_check", datasets=["prod", "staging"])
        def my_check(mp, ctx):
            metric = mp.average("revenue")  # ERROR: which dataset?

        # Valid - will be imputed
        @check(name="single_check", datasets=["production"])
        def my_check(mp, ctx):
            metric = mp.average("revenue")  # OK: will use production

    The validator requires a MetricProvider to look up symbol datasets,
    and runs during plan() and run() phases.

    Attributes:
        _provider: MetricProvider used to look up symbol information
    """
```

**Final commit**:
```bash
git add src/dqx/validator.py
git commit -m "docs: add comprehensive documentation for DatasetValidator v2"
```

## Testing Your Implementation

### Manual Testing

Create a test file `test_manual.py`:

```python
from dqx.api import VerificationSuiteBuilder, check
from dqx.orm.repositories import InMemoryMetricDB

db = InMemoryMetricDB()


# Test 1: Dataset mismatch
@check(name="mismatch_check", datasets=["prod", "staging"])
def mismatch_check(mp, ctx):
    # This should trigger an error - "testing" not in ["prod", "staging"]
    metric = mp.average("column", dataset="testing")
    ctx.assert_that(metric).where(name="test").is_gt(0)


# Test 2: Ambiguous dataset
@check(name="ambiguous_check", datasets=["prod", "staging"])
def ambiguous_check(mp, ctx):
    # This should trigger an error - ambiguous which dataset to use
    metric = mp.average("column")  # dataset=None
    ctx.assert_that(metric).where(name="test").is_gt(0)


# Test 3: Valid single dataset imputation
@check(name="valid_check", datasets=["prod"])
def valid_check(mp, ctx):
    # This is OK - will be imputed to "prod"
    metric = mp.average("column")  # dataset=None
    ctx.assert_that(metric).where(name="test").is_gt(0)


suite = (
    VerificationSuiteBuilder("test", db)
    .add_check(mismatch_check)
    .add_check(ambiguous_check)
    .add_check(valid_check)
    .build()
)

report = suite.plan()
print("Validation report:")
print(report)
print(f"\nHas errors: {report.has_errors()}")
print(f"Error count: {len(report.errors)}")
```

Run it:
```bash
uv run python test_manual.py
```

Expected output should show 2 errors:
1. Dataset mismatch for "testing"
2. Ambiguous dataset for the None case with multiple parent datasets

## Key Design Improvements in v2

1. **Ambiguity Detection**: Now catches the case where `dataset=None` with multiple parent datasets, preventing confusing runtime errors.

2. **Mandatory Provider**: The provider is now mandatory in `SuiteValidator.validate()`, making the API clearer and more consistent.

3. **Comprehensive Validation**: Handles all combinations of dataset specifications:
   - Explicit dataset → validate against parent
   - No dataset + single parent dataset → allow (will be imputed)
   - No dataset + multiple parent datasets → error (ambiguous)

4. **Clear Error Messages**: Different messages for mismatches vs ambiguity, helping developers understand exactly what's wrong.

## Common Issues and Solutions

### Issue: Import Errors
**Solution**: Make sure to add necessary imports:
```python
from dqx.provider import MetricProvider  # In validator.py
from dqx.orm.repositories import InMemoryMetricDB  # In tests
```

### Issue: Understanding Ambiguity Errors
**Solution**: The error message clearly states the problem:
- "has no dataset specified" - the symbol doesn't specify a dataset
- "has multiple datasets" - the parent check allows multiple datasets
- "Unable to determine which dataset to use" - explains why it's an error

### Issue: Tests Failing Due to Symbol Not Found
**Solution**: The implementation handles this gracefully with try/except, allowing early validation before all symbols are registered.

## Summary

This v2 implementation:
- Detects both dataset mismatches and ambiguous configurations
- Provides clear, actionable error messages for each scenario
- Uses mandatory provider parameter for consistency
- Includes comprehensive test coverage for all edge cases
- Prevents confusing runtime errors during dataset imputation

The validator will significantly improve the developer experience by catching configuration errors early and providing clear guidance on how to fix them.
