# Hierarchical Dataset Imputation Implementation Plan

## Overview

This document provides a comprehensive, step-by-step plan to improve the dataset imputation logic in DQX (Data Quality eXcellence framework). The improvement introduces a hierarchical dataset propagation model where datasets flow from parent to child nodes in the graph structure.

## Background for Engineers

### What is DQX?

DQX is a data quality framework that allows users to define quality checks on datasets. The framework uses a graph-based architecture where:

- **RootNode**: The top-level container for all checks
- **CheckNode**: Represents a data quality check (e.g., "validate that prices are positive")
- **AssertionNode**: Individual assertions within a check (e.g., "average(price) > 0")
- **Dataset**: A data source (like a database table or file) that checks run against

### What is Dataset Imputation?

When users define checks, they may not always specify which dataset the check should run on. Dataset imputation is the process of automatically determining which datasets a check should use based on:
- What datasets are available
- What the parent node specifies
- Validation rules to ensure consistency

### Current Problem

Currently, CheckNodes directly look at the list of available datasets when imputing. This breaks the hierarchical structure of the graph and makes the logic harder to follow.

### Proposed Solution

Make dataset imputation truly hierarchical:
1. RootNode gets a `datasets` field that's populated with available datasets
2. CheckNodes inherit/validate against their parent's datasets (not the global available list)

## Development Environment Setup

### Prerequisites

```bash
# This project uses Python 3.11 or 3.12
python --version  # Should show 3.11.x or 3.12.x

# The project uses uv for dependency management
# If you don't have uv, install it:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project Setup

```bash
# Clone the repository if you haven't already
cd /Users/npham/git-tree/dqx

# Install dependencies
uv sync

# Run tests to ensure everything works
uv run pytest tests/graph/test_visitor.py -v

# Run type checking
uv run mypy src/dqx/graph/

# Run linting
uv run ruff check src/dqx/graph/
```

## Implementation Tasks

### Task 1: Add datasets field to RootNode

**File to modify**: `src/dqx/graph/nodes.py`

**What to do**:
1. Find the `RootNode` class (around line 10-50)
2. Add a `datasets` attribute initialized as an empty list
3. Follow the existing pattern for other attributes

**Code to add**:
```python
class RootNode(CompositeNode[None, "CheckNode"]):
    """Root node of the verification graph hierarchy.

    ... existing docstring ...

    Attributes:
        name: Human-readable name identifying this verification suite
        datasets: List of dataset names available in this suite (populated during imputation)
        _context: Reference to the Context instance that owns the symbol table
    """

    def __init__(self, name: str) -> None:
        """Initialize a root node.

        Args:
            name: Human-readable name for the verification suite
        """
        super().__init__(parent=None)  # Root always has None parent
        self.name = name
        self.datasets: list[str] = []  # ADD THIS LINE
```

**How to test manually**:
```python
# In Python REPL
from dqx.graph.nodes import RootNode
root = RootNode("test")
print(root.datasets)  # Should print: []
```

**Commit this change**:
```bash
git add src/dqx/graph/nodes.py
git commit -m "feat: add datasets field to RootNode for hierarchical imputation"
```

### Task 2: Implement RootNode visitor method

**File to modify**: `src/dqx/graph/visitors.py`

**What to do**:
1. Find the `DatasetImputationVisitor` class
2. Add a `_visit_root_node` method that sets the root's datasets
3. Update the main `visit` method to handle RootNode

**Code to add**:

First, add the import at the top of the file:
```python
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode  # Add RootNode
```

Then add the visitor method:
```python
def visit(self, node: BaseNode) -> None:
    """Visit a node and perform dataset validation/imputation.

    Args:
        node: The node to visit
    """
    if isinstance(node, RootNode):  # ADD THIS
        self._visit_root_node(node)  # ADD THIS
    elif isinstance(node, CheckNode):
        self._visit_check_node(node)
    elif isinstance(node, AssertionNode):
        self._visit_assertion_node(node)

def _visit_root_node(self, node: RootNode) -> None:  # ADD THIS METHOD
    """Set available datasets on the RootNode.

    This establishes the top-level datasets that will flow down
    through the hierarchy.

    Args:
        node: The RootNode to process
    """
    node.datasets = self.available_datasets.copy()
```

**How to test manually**:
```python
# In Python REPL
from dqx.graph.nodes import RootNode
from dqx.graph.visitors import DatasetImputationVisitor

root = RootNode("test")
visitor = DatasetImputationVisitor(["prod", "staging"], provider=None)
visitor.visit(root)
print(root.datasets)  # Should print: ['prod', 'staging']
```

**Commit this change**:
```bash
git add src/dqx/graph/visitors.py
git commit -m "feat: add _visit_root_node to set datasets on RootNode"
```

### Task 3: Update CheckNode validation to use parent datasets

**File to modify**: `src/dqx/graph/visitors.py`

**What to do**:
1. Find the `_visit_check_node` method
2. Change it to use `parent.datasets` instead of `self.available_datasets`
3. Update error messages accordingly

**Code changes**:
```python
def _visit_check_node(self, node: CheckNode) -> None:
    """Validate and impute datasets for a CheckNode.

    If the CheckNode has no datasets, impute from parent's datasets.
    If it has datasets, validate they are all in parent's datasets.

    Args:
        node: The CheckNode to process
    """
    # Get parent's datasets (RootNode should have them by now)
    parent_datasets = node.parent.datasets  # CHANGE THIS LINE

    if not node.datasets:
        # Impute from parent datasets
        node.datasets = parent_datasets.copy()  # CHANGE THIS LINE
    else:
        # Validate existing datasets against parent
        for dataset in node.datasets:
            if dataset not in parent_datasets:  # CHANGE THIS LINE
                self._errors.append(
                    f"Check '{node.name}' specifies dataset '{dataset}' "
                    f"which is not in parent datasets: {parent_datasets}"  # CHANGE THIS LINE
                )
```

**Testing consideration**: This change will break existing tests! We'll fix them in Task 4.

**Commit this change**:
```bash
git add src/dqx/graph/visitors.py
git commit -m "feat: update CheckNode validation to use parent datasets"
```

### Task 4: Fix existing tests

**File to modify**: `tests/graph/test_visitor.py`

**What to do**:
Many tests will fail because they don't set datasets on the RootNode. We need to update them.

**Pattern to follow** - For each failing test:
1. After creating the visitor, visit the root node first
2. Then visit the check node

**Example fix**:
```python
def test_propagates_datasets_from_root_to_check(self) -> None:
    """When CheckNode has no datasets, it inherits from parent's datasets."""
    # Arrange
    root = RootNode("test_suite")
    check = root.add_check("test_check")  # No datasets specified

    visitor = DatasetImputationVisitor(["prod", "staging"], provider=None)

    # Act
    visitor.visit(root)  # ADD THIS LINE - Visit root first
    visitor.visit(check)

    # Assert
    assert check.datasets == ["prod", "staging"]
```

**Run tests frequently**:
```bash
# Run after each test fix to see progress
uv run pytest tests/graph/test_visitor.py::TestDatasetImputationVisitor::test_propagates_datasets_from_root_to_check -v
```

**Commit after fixing all tests**:
```bash
git add tests/graph/test_visitor.py
git commit -m "test: update existing tests for hierarchical dataset imputation"
```

### Task 5: Add new tests for RootNode dataset handling

**File to modify**: `tests/graph/test_visitor.py`

**What to do**:
Add comprehensive tests for the new RootNode behavior.

**Tests to add** (add these to the `TestDatasetImputationVisitor` class):

```python
def test_root_node_receives_available_datasets(self) -> None:
    """RootNode should be populated with available datasets when visited."""
    # Arrange
    root = RootNode("test_suite")
    visitor = DatasetImputationVisitor(["prod", "staging", "dev"], provider=None)

    # Act
    visitor.visit(root)

    # Assert
    assert root.datasets == ["prod", "staging", "dev"]

def test_root_node_datasets_are_copied_not_referenced(self) -> None:
    """RootNode should get a copy of datasets, not a reference."""
    # Arrange
    available = ["prod", "staging"]
    root = RootNode("test_suite")
    visitor = DatasetImputationVisitor(available, provider=None)

    # Act
    visitor.visit(root)
    available.append("dev")  # Modify original list

    # Assert
    assert root.datasets == ["prod", "staging"]  # Should not include "dev"

def test_check_validates_against_parent_not_available(self) -> None:
    """CheckNode should validate against parent datasets, not available."""
    # Arrange
    root = RootNode("test_suite")
    check = root.add_check("test_check", datasets=["dev"])

    # Manually set root datasets to simulate a filtered scenario
    root.datasets = ["prod", "staging"]  # "dev" is not included

    visitor = DatasetImputationVisitor(["prod", "staging", "dev"], provider=None)

    # Act - Don't visit root (it already has datasets set)
    visitor.visit(check)

    # Assert
    assert visitor.has_errors()
    errors = visitor.get_errors()
    assert any("parent datasets" in err for err in errors)
    assert any("dev" in err for err in errors)

def test_hierarchical_flow_root_to_check_to_assertion(self) -> None:
    """Test complete hierarchical flow from root to assertion."""
    # Arrange
    root = RootNode("test_suite")
    check = root.add_check("test_check")  # No datasets
    assertion = check.add_assertion(actual=sp.Symbol("x_1"), name="test")

    # Mock provider
    provider = Mock(spec=MetricProvider)
    metric = Mock(spec=SymbolicMetric)
    metric.name = "x_1"
    metric.dataset = None
    provider.get_symbol.return_value = metric

    visitor = DatasetImputationVisitor(["prod"], provider=provider)

    # Act - Visit in hierarchical order
    visitor.visit(root)      # Sets root.datasets = ["prod"]
    visitor.visit(check)     # Sets check.datasets = ["prod"] from parent
    visitor.visit(assertion) # Imputes metric.dataset = "prod"

    # Assert
    assert root.datasets == ["prod"]
    assert check.datasets == ["prod"]
    assert metric.dataset == "prod"
    assert not visitor.has_errors()
```

**Run the new tests**:
```bash
uv run pytest tests/graph/test_visitor.py -k "test_root_node" -v
uv run pytest tests/graph/test_visitor.py -k "test_hierarchical_flow" -v
```

**Commit the new tests**:
```bash
git add tests/graph/test_visitor.py
git commit -m "test: add comprehensive tests for RootNode dataset handling"
```

### Task 6: Run full test suite and fix any issues

**What to do**:
1. Run all visitor tests
2. Run type checking
3. Run linting
4. Fix any issues that arise

**Commands**:
```bash
# Run all visitor tests
uv run pytest tests/graph/test_visitor.py -v

# Run type checking
uv run mypy src/dqx/graph/

# Run linting and auto-fix
uv run ruff check src/dqx/graph/ --fix

# Run the pre-commit hooks
./bin/run-hooks.sh src/dqx/graph/ tests/graph/test_visitor.py
```

**Common issues you might encounter**:

1. **Type errors**: The parent might not have a datasets attribute
   - Solution: Add type guards or assertions

2. **Test failures**: Some integration tests might fail
   - Solution: Update them to visit RootNode first

3. **Linting issues**: Code style problems
   - Solution: Use `ruff check --fix` to auto-fix

**Final commit**:
```bash
git add -u
git commit -m "chore: fix type checking and linting issues"
```

## Testing Your Implementation

### Manual Testing Script

Create a file `test_imputation.py`:

```python
from dqx.graph.nodes import RootNode
from dqx.graph.visitors import DatasetImputationVisitor
from dqx.graph.traversal import GraphTraversal

# Create graph structure
root = RootNode("test_suite")
check1 = root.add_check("check_without_datasets")
check2 = root.add_check("check_with_datasets", datasets=["prod"])

# Create visitor
visitor = DatasetImputationVisitor(["prod", "staging"], provider=None)

# Use graph traversal to visit all nodes
traversal = GraphTraversal()
traversal.traverse(root, visitor)

# Verify results
print(f"Root datasets: {root.datasets}")  # Should be ["prod", "staging"]
print(f"Check1 datasets: {check1.datasets}")  # Should be ["prod", "staging"]
print(f"Check2 datasets: {check2.datasets}")  # Should be ["prod"]

# Check for errors
if visitor.has_errors():
    print("Errors:", visitor.get_error_summary())
```

Run it:
```bash
uv run python test_imputation.py
```

### Integration Testing

Check if the changes work with the full system:

```bash
# Run end-to-end tests
uv run pytest tests/e2e/test_api_e2e.py -v

# If any fail, they might need updates for the new behavior
```

## Troubleshooting Guide

### Problem: "AttributeError: 'RootNode' object has no attribute 'datasets'"
**Solution**: Make sure you added the `self.datasets = []` line in RootNode.__init__

### Problem: "Check specifies dataset which is not in parent datasets"
**Solution**: This is the new validation working! The check is specifying a dataset that the parent doesn't have.

### Problem: Tests pass individually but fail when run together
**Solution**: Tests might be sharing state. Make sure each test creates fresh nodes.

### Problem: Type checker complains about parent.datasets
**Solution**: Add a type assertion or check:
```python
if hasattr(node.parent, 'datasets'):
    parent_datasets = node.parent.datasets
else:
    # Handle error case
```

## Design Principles Applied

### DRY (Don't Repeat Yourself)
- We reuse the existing visitor pattern instead of creating new traversal logic
- Dataset validation logic is centralized in the visitor

### YAGNI (You Aren't Gonna Need It)
- We only add the datasets field to RootNode (not all nodes)
- We don't add complex dataset transformation logic

### TDD (Test Driven Development)
- Write tests for each behavior before implementing
- Use tests to drive the design
- Each commit should have passing tests

## Verification Checklist

Before considering the implementation complete:

- [ ] All existing tests pass
- [ ] New tests for RootNode dataset handling pass
- [ ] Type checking passes (`uv run mypy src/dqx/graph/`)
- [ ] Linting passes (`uv run ruff check src/dqx/graph/`)
- [ ] Manual testing script works correctly
- [ ] No hardcoded values or debug prints left in code
- [ ] All commits have descriptive messages
- [ ] Code follows existing patterns in the codebase

## Summary

This implementation improves the dataset imputation logic by making it truly hierarchical. The key insight is that child nodes should only know about their parent's datasets, not the global list of available datasets. This creates a cleaner, more maintainable architecture that follows the natural graph hierarchy.

The implementation is minimal (only ~20 lines of production code changes) but requires careful attention to test updates to maintain the test suite.
