# Dataset Imputation Visitor Implementation Plan

## Overview

### Current Problem
The dataset imputation logic is scattered across multiple node classes (RootNode, CheckNode, AssertionNode), each implementing their own `impute_datasets()` method. This violates the Single Responsibility Principle and makes the code harder to maintain and test.

### Proposed Solution
Replace the scattered `impute_datasets()` methods with a centralized `DatasetImputationVisitor` that uses the Visitor pattern to traverse the graph and handle all dataset propagation logic in one place.

### Key Concepts for Context
- **Dataset**: A named data source (e.g., "production", "staging") that metrics are computed against
- **Graph Structure**: RootNode → CheckNode → AssertionNode → SymbolicMetric (via symbols in expressions)
- **Visitor Pattern**: A design pattern that separates algorithms from the object structure they operate on
- **SymPy**: Python library for symbolic mathematics; assertions use SymPy expressions containing symbols

### Dataset Flow
```
Available Datasets (from user)
    ↓
RootNode (validates non-empty)
    ↓
CheckNode (inherits or validates own datasets)
    ↓
SymbolicMetric (accessed via symbols in AssertionNode expressions)
```

## Implementation Tasks

### Task 1: Write Tests for DatasetImputationVisitor (TDD First!)

**Files to create/modify:**
- `tests/graph/test_visitor.py` (create new file or add to existing)

**What to do:**
1. Create/update test file with basic imports
2. Write test cases BEFORE implementing the visitor:
   - Test successful propagation: root → check → symbolic metric
   - Test inconsistency error: symbolic metric requires dataset not in check
   - Test confusion error: symbolic metric has no dataset but check has multiple
   - Test preservation of existing datasets
   - Test idempotency (running twice produces same result)
   - Test handling of missing symbols gracefully

**Example test structure:**
```python
import pytest
from dqx.graph.nodes import RootNode, CheckNode, AssertionNode
from dqx.graph.visitors import DatasetImputationVisitor  # Will implement later
from dqx.provider import MetricProvider
from dqx.common import DQXError

# If file already exists, add to it. Otherwise create with:
"""Tests for visitor classes in the graph module."""

class TestDatasetImputationVisitor:
    def test_propagates_datasets_from_root_to_check(self):
        """When CheckNode has no datasets, it inherits from available datasets."""
        # Arrange
        root = RootNode("test_suite")
        check = CheckNode("test_check")  # No datasets specified
        root.add_child(check)

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=None)

        # Act
        visitor.visit(check)

        # Assert
        assert check.datasets == ["prod", "staging"]

    def test_preserves_existing_check_datasets(self):
        """When CheckNode has datasets, they are preserved if valid."""
        # Similar structure...

    def test_error_on_invalid_check_dataset(self):
        """When CheckNode specifies dataset not in available, raise error."""
        # Test should expect DQXError

    def test_imputation_is_idempotent(self):
        """Running imputation twice produces the same result."""
        # Set up graph
        # Run imputation once
        # Capture state
        # Run imputation again
        # Assert state unchanged

    def test_preserves_explicitly_set_datasets(self):
        """Datasets explicitly set on SymbolicMetric are preserved."""
        # Create metric with dataset="prod"
        # Run imputation with ["prod", "staging"]
        # Assert metric still has dataset="prod"

    def test_handles_missing_symbols_gracefully(self):
        """Visitor handles symbols without SymbolicMetrics."""
        # Create assertion with symbol not in provider
        # Run imputation
        # Should not fail, just skip the symbol

    # More tests for SymbolicMetric validation...
```

**How to run tests:**
```bash
uv run pytest tests/graph/test_visitor.py::TestDatasetImputationVisitor -v
```

**Commit after:** Writing all test cases (they will fail - that's OK!)
```bash
git add tests/graph/test_visitor.py
git commit -m "test: add test cases for DatasetImputationVisitor in test_visitor.py"
```

---

### Task 2: Implement Basic DatasetImputationVisitor Structure

**Files to modify:**
- `src/dqx/graph/visitors.py` (add new class)

**What to do:**
1. Add the basic visitor class structure (no inheritance - NodeVisitor is a Protocol)
2. Implement visit and visit_async methods to satisfy the Protocol
3. Add helper methods for each node type

**Code to add:**
```python
class DatasetImputationVisitor:  # No inheritance needed - NodeVisitor is a Protocol
    """Visitor that propagates and validates dataset information through the graph.

    This visitor handles all dataset-related logic:
    - Propagates available datasets from root to leaf nodes
    - Validates dataset consistency between parent and child nodes
    - Reports clear errors for inconsistencies
    - Ensures idempotent behavior (running multiple times produces same result)
    """

    def __init__(self, available_datasets: list[str], provider: MetricProvider):
        """Initialize the visitor with available datasets.

        Args:
            available_datasets: List of dataset names available for computation
            provider: MetricProvider instance to access SymbolicMetrics
        """
        if not available_datasets:
            raise DQXError("At least one dataset must be provided!")

        self.available_datasets = available_datasets
        self.provider = provider
        self.errors: list[str] = []

    def visit(self, node: BaseNode) -> None:
        """Visit a node and handle dataset logic based on node type.

        Implementation of NodeVisitor protocol.
        """
        if isinstance(node, RootNode):
            self._visit_root(node)
        elif isinstance(node, CheckNode):
            self._visit_check(node)
        elif isinstance(node, AssertionNode):
            self._visit_assertion(node)

    async def visit_async(self, node: BaseNode) -> None:
        """Async visit implementation.

        Since dataset imputation doesn't require async operations,
        this simply delegates to the synchronous visit method.
        """
        self.visit(node)

    def _visit_root(self, node: RootNode) -> None:
        """Process RootNode - just validate available datasets."""
        # Nothing to do - available_datasets already validated in __init__
        pass

    def _visit_check(self, node: CheckNode) -> None:
        """Process CheckNode - validate or inherit datasets."""
        # TODO: Implement in next task
        pass

    def _visit_assertion(self, node: AssertionNode) -> None:
        """Process AssertionNode - validate symbols' datasets."""
        # TODO: Implement in next task
        pass
```

**How to test:**
```bash
# Run your tests - some should start passing
uv run pytest tests/graph/test_visitor.py::TestDatasetImputationVisitor::test_propagates_datasets_from_root_to_check -v
```

**Commit after:** Basic structure is working
```bash
git add src/dqx/graph/visitors.py
git commit -m "feat: add basic DatasetImputationVisitor structure"
```

---

### Task 3: Implement CheckNode Dataset Logic

**Files to modify:**
- `src/dqx/graph/visitors.py` (update `_visit_check` method)

**What to do:**
1. Implement validation for existing datasets
2. Implement inheritance when no datasets specified
3. Add error collection

**Code to implement:**
```python
def _visit_check(self, node: CheckNode) -> None:
    """Process CheckNode - validate or inherit datasets."""
    if node.datasets:
        # Validate existing datasets
        invalid_datasets = [ds for ds in node.datasets if ds not in self.available_datasets]
        if invalid_datasets:
            error_msg = (
                f"CheckNode '{node.name}' requires datasets {invalid_datasets} "
                f"but only {self.available_datasets} are available"
            )
            self.errors.append(error_msg)
            # Don't propagate to children on error
            return
    else:
        # No datasets specified - inherit all available
        node.datasets = self.available_datasets.copy()
```

**How to test:**
```bash
# These tests should now pass
uv run pytest tests/graph/test_visitor.py::TestDatasetImputationVisitor -k "check" -v
```

**Commit after:** CheckNode logic is working
```bash
git add src/dqx/graph/visitors.py
git commit -m "feat: implement CheckNode dataset validation and inheritance"
```

---

### Task 4: Implement AssertionNode/SymbolicMetric Logic

**Files to modify:**
- `src/dqx/graph/visitors.py` (update `_visit_assertion` method)

**What to do:**
1. Extract symbols from assertion expression
2. Look up SymbolicMetrics for each symbol
3. Validate/impute datasets for each SymbolicMetric

**Code to implement:**
```python
def _visit_assertion(self, node: AssertionNode) -> None:
    """Process AssertionNode - validate symbols' datasets."""
    # Get parent CheckNode
    parent_check = node.parent
    if not isinstance(parent_check, CheckNode):
        self.errors.append(f"AssertionNode '{node.name}' has invalid parent")
        return

    # Extract symbols from the assertion's expression
    symbols = node.actual.free_symbols

    for symbol in symbols:
        try:
            symbolic_metric = self.provider.get_symbol(symbol)
        except DQXError:
            # Symbol not found - this is a different kind of error
            continue

        if symbolic_metric.dataset:
            # Validate specified dataset
            if symbolic_metric.dataset not in parent_check.datasets:
                error_msg = (
                    f"SymbolicMetric '{symbolic_metric.name}' requires dataset "
                    f"'{symbolic_metric.dataset}' but CheckNode '{parent_check.name}' "
                    f"only provides {parent_check.datasets}"
                )
                self.errors.append(error_msg)
        else:
            # No dataset specified - try to impute
            if len(parent_check.datasets) == 1:
                # Single dataset - use it
                symbolic_metric.dataset = parent_check.datasets[0]
            else:
                # Multiple datasets - confusion!
                error_msg = (
                    f"SymbolicMetric '{symbolic_metric.name}' has no dataset specified "
                    f"but CheckNode '{parent_check.name}' provides multiple datasets "
                    f"{parent_check.datasets}. Cannot determine which to use."
                )
                self.errors.append(error_msg)
```

**How to test:**
```bash
# All tests should pass now
uv run pytest tests/graph/test_visitor.py::TestDatasetImputationVisitor -v
```

**Commit after:** Full visitor implementation is working
```bash
git add src/dqx/graph/visitors.py
git commit -m "feat: implement AssertionNode symbol dataset validation"
```

---

### Task 5: Add Error Reporting Methods

**Files to modify:**
- `src/dqx/graph/visitors.py` (add methods)

**What to do:**
Add methods for better error access and reporting:

```python
def has_errors(self) -> bool:
    """Check if any validation errors occurred."""
    return len(self.errors) > 0

def get_errors(self) -> list[str]:
    """Get list of individual error messages."""
    return self.errors.copy()

def get_error_summary(self) -> str:
    """Get summary of all errors."""
    if not self.errors:
        return ""
    error_count = len(self.errors)
    return f"Dataset validation failed with {error_count} error(s):\n" + \
           "\n".join(f"  - {err}" for err in self.errors)
```

**Commit after:** Adding error methods
```bash
git add src/dqx/graph/visitors.py
git commit -m "feat: add error reporting methods to DatasetImputationVisitor"
```

---

### Task 6: Remove impute_datasets from Node Classes

**Files to modify:**
- `src/dqx/graph/nodes.py` (remove methods only)
- `src/dqx/graph/traversal.py` (update Graph class)

**What to do:**

1. **In `nodes.py`**, remove these methods:
   - `RootNode.impute_datasets()` method
   - `CheckNode.impute_datasets()` method
   - `AssertionNode.impute_datasets()` method
   - Remove `self.datasets: list[str] = []` line from AssertionNode.__init__
   - (Note: AssertionNode.__init__ doesn't take datasets parameter, just initializes empty list)

2. **In `traversal.py`**, replace Graph.impute_datasets():
```python
def impute_datasets(self, datasets: list[str], provider: MetricProvider) -> None:
    """Propagate dataset information through the graph using visitor pattern.

    Args:
        datasets: List of available dataset names
        provider: MetricProvider to access SymbolicMetrics

    Raises:
        DQXError: If validation fails
    """
    from dqx.graph.visitors import DatasetImputationVisitor

    visitor = DatasetImputationVisitor(datasets, provider)
    self.dfs(visitor)  # Use DFS to ensure parents are processed before children

    if visitor.has_errors():
        raise DQXError(visitor.get_error_summary())
```

**How to test:**
```bash
# Run existing tests to ensure nothing breaks
uv run pytest tests/graph/ -v
uv run pytest tests/test_api.py -v
```

**Commit after:** Removing old implementation
```bash
git add src/dqx/graph/nodes.py src/dqx/graph/traversal.py
git commit -m "refactor: remove impute_datasets methods from nodes"
```

---

### Task 7: Update API Layer

**Files to modify:**
- `src/dqx/api.py` (update VerificationSuite.run)

**What to do:**
Find where `impute_datasets` is called and update to pass the provider:

```python
# In VerificationSuite.run() method, change:
self._context._graph.impute_datasets(list(datasources.keys()))

# To:
self._context._graph.impute_datasets(list(datasources.keys()), self.provider)
```

**How to test:**
```bash
# Run end-to-end tests
uv run pytest tests/e2e/test_api_e2e.py -v
```

**Commit after:** API layer is updated
```bash
git add src/dqx/api.py
git commit -m "fix: update API to pass provider to impute_datasets"
```

---

### Task 8: Add Integration Tests

**Files to modify:**
- `tests/graph/test_visitor.py` (add integration test section)

**What to do:**
Add integration tests to the same visitor test file:

```python
# Add to tests/graph/test_visitor.py

class TestDatasetImputationIntegration:
    """Integration tests for dataset imputation across the full graph."""

    def test_full_dataset_propagation_flow(self):
        """Test dataset propagation through entire graph."""
        # Create a full graph structure
        # Run visitor with multiple datasets
        # Verify datasets are properly propagated

    def test_dataset_validation_errors_surface_correctly(self):
        """Test that validation errors are properly reported."""
        # Create a graph with invalid dataset configuration
        # Verify DQXError is raised with clear message
```

**Commit after:** Integration tests pass
```bash
git add tests/graph/test_visitor.py
git commit -m "test: add integration tests for dataset imputation"
```

---

### Task 9: Update Documentation

**Files to modify:**
- `README.md` (if it mentions dataset behavior)
- `src/dqx/graph/visitors.py` (ensure docstrings are complete)

**What to do:**
1. Update any documentation that references the old impute_datasets behavior
2. Add comprehensive docstrings to DatasetImputationVisitor
3. Document the dataset validation rules clearly

**Commit after:** Documentation is updated
```bash
git add README.md src/dqx/graph/visitors.py
git commit -m "docs: update documentation for new dataset imputation"
```

---

### Task 10: Final Cleanup and Validation

**What to do:**
1. Run full test suite: `uv run pytest -v`
2. Run type checking: `uv run mypy src/`
3. Run linting: `uv run ruff check src/ --fix`
4. Check test coverage: `uv run pytest --cov=dqx.graph`

**Final commit:**
```bash
git add -u
git commit -m "chore: final cleanup and formatting"
```

## Testing Strategy

### Unit Tests
- Test each visitor method in isolation
- Mock the MetricProvider when needed
- Test error cases thoroughly

### Integration Tests
- Test full graph traversal
- Test with real MetricProvider
- Verify end-to-end behavior

### Test Data Setup
```python
# Helper to create test graphs
def create_test_graph():
    root = RootNode("test")
    check = CheckNode("check1", datasets=["prod"])
    assertion = AssertionNode(
        actual=sp.Symbol("x_1"),  # References a metric
        name="test assertion"
    )
    root.add_child(check)
    check.add_child(assertion)
    return root
```

## Common Pitfalls to Avoid

1. **Don't forget the provider parameter** - The visitor needs MetricProvider to look up symbols
2. **Handle missing symbols gracefully** - Not all symbols may have corresponding SymbolicMetrics
3. **Test error paths** - Ensure validation errors are clear and actionable
4. **Preserve immutability where possible** - Only modify datasets when imputing, not when validating

## Success Criteria

1. All existing tests still pass
2. New visitor tests have >95% coverage
3. Dataset validation errors are clear and actionable
4. No performance regression (visitor should be as fast as old implementation)
5. Code is cleaner and more maintainable

## How to Verify Everything Works

```bash
# 1. Run all tests
uv run pytest -v

# 2. Check coverage
uv run pytest --cov=dqx.graph --cov-report=html

# 3. Run type checking
uv run mypy src/

# 4. Run linting
uv run ruff check src/

# 5. Try the example from README
python examples/quickstart.py  # Should work without errors
