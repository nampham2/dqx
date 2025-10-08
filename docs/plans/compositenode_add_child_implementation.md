# CompositeNode add_child Method Implementation Plan

## Overview
This document provides a step-by-step implementation plan for adding the `add_child` method to the `CompositeNode` class in the DQX project. This plan is written for an engineer with Python experience but no knowledge of the DQX codebase.

## Background

### What is DQX?
DQX (Data Quality eXcellence) is a data quality framework that validates data using a graph-based architecture. Think of it like unit tests, but for data instead of code.

### What are Graph Nodes?
The project uses a tree structure to represent data quality checks:
- **BaseNode**: The base class for all nodes (like a tree node)
- **CompositeNode**: A node that can have children (like a branch)
- **Various concrete nodes**: CheckNode, AssertionNode, etc. (specific types of validations)

### The Problem
Currently, to add a child to a CompositeNode, you must:
```python
parent.children.append(child)  # Add to list
child.parent = parent          # Manually set parent - easy to forget!
```

We want a single method that does both operations.

## Project Setup

### Tools You'll Need
- Python 3.11 or 3.12
- The `uv` tool (Python package manager used in this project)

### Project Structure
```
dqx/
├── src/
│   └── dqx/
│       ├── common.py         # Contains DQXError exception
│       └── graph/
│           ├── base.py       # Contains CompositeNode class
│           └── nodes.py      # Uses CompositeNode
├── tests/
│   └── graph/
│       └── test_base.py      # Tests for base.py
└── docs/
    └── plans/                # This document
```

## Implementation Tasks

### Task 1: Write the First Failing Test (TDD Step 1)

**Files to edit**: `tests/graph/test_base.py`

**What to do**: Add a new test class at the end of the file for testing `add_child`.

**Code to add**:
```python
class TestCompositeNodeAddChild:
    """Test suite for CompositeNode.add_child method."""

    def test_add_child_adds_to_children_list(self) -> None:
        """Test that add_child adds the child to children list."""
        parent = MockCompositeNode()
        child = MockNode()

        result = parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] is child
        assert result is parent  # Method should return self
```

**How to run the test**:
```bash
# From project root directory
uv run pytest tests/graph/test_base.py::TestCompositeNodeAddChild::test_add_child_adds_to_children_list -v
```

**Expected result**: The test should FAIL with `AttributeError: 'MockCompositeNode' object has no attribute 'add_child'`

**Commit**:
```bash
git add tests/graph/test_base.py
git commit -m "test: add failing test for CompositeNode.add_child method"
```

### Task 2: Implement Minimal Code to Pass (TDD Step 2)

**Files to edit**: `src/dqx/graph/base.py`

**Step 1**: Add import at the top of the file (after other imports):
```python
from dqx.common import DQXError
```

**Step 2**: Add the method to the `CompositeNode` class (add this after the `is_leaf` method):
```python
    def add_child(self, child: TChild) -> CompositeNode[TChild]:
        """Add a child node and set its parent reference.

        Args:
            child: The child node to add

        Returns:
            Self for method chaining

        Raises:
            DQXError: If the child is already in the children list
        """
        if child in self.children:
            raise DQXError("Child node is already in the children list")

        self.children.append(child)
        child.parent = self
        return self
```

**How to test**:
```bash
# Run the specific test
uv run pytest tests/graph/test_base.py::TestCompositeNodeAddChild::test_add_child_adds_to_children_list -v
```

**Expected result**: Test should now PASS

**Commit**:
```bash
git add src/dqx/graph/base.py
git commit -m "feat: implement add_child method in CompositeNode"
```

### Task 3: Add Test for Parent Reference (TDD Cycle 2)

**Files to edit**: `tests/graph/test_base.py`

**What to do**: Add another test method to `TestCompositeNodeAddChild` class:

```python
    def test_add_child_sets_parent_reference(self) -> None:
        """Test that add_child sets the child's parent reference."""
        parent = MockCompositeNode()
        child = MockNode()

        parent.add_child(child)

        assert child.parent is parent
        assert child.is_root is False
```

**How to test**:
```bash
uv run pytest tests/graph/test_base.py::TestCompositeNodeAddChild -v
```

**Expected result**: Both tests should PASS (the implementation already handles this)

**Commit**:
```bash
git add tests/graph/test_base.py
git commit -m "test: add test for parent reference in add_child"
```

### Task 4: Add Test for Duplicate Prevention

**Files to edit**: `tests/graph/test_base.py`

**What to do**: Add test for the error case:

```python
    def test_add_child_raises_error_on_duplicate(self) -> None:
        """Test that add_child raises DQXError when adding duplicate child."""
        from dqx.common import DQXError

        parent = MockCompositeNode()
        child = MockNode()

        # Add child first time - should work
        parent.add_child(child)

        # Try to add same child again - should raise error
        with pytest.raises(DQXError, match="Child node is already in the children list"):
            parent.add_child(child)
```

**How to test**:
```bash
uv run pytest tests/graph/test_base.py::TestCompositeNodeAddChild::test_add_child_raises_error_on_duplicate -v
```

**Commit**:
```bash
git add tests/graph/test_base.py
git commit -m "test: add test for duplicate child prevention"
```

### Task 5: Add Test for Re-parenting

**Files to edit**: `tests/graph/test_base.py`

**What to do**: Test that a child can be moved between parents:

```python
    def test_add_child_allows_reparenting(self) -> None:
        """Test that a child can be moved from one parent to another."""
        parent1 = MockCompositeNode()
        parent2 = MockCompositeNode()
        child = MockNode()

        # Add to first parent
        parent1.add_child(child)
        assert child.parent is parent1
        assert child in parent1.children

        # Move to second parent
        parent2.add_child(child)
        assert child.parent is parent2
        assert child in parent2.children
        assert child in parent1.children  # Still in first parent's list!
```

**How to test**:
```bash
uv run pytest tests/graph/test_base.py::TestCompositeNodeAddChild -v
```

**Commit**:
```bash
git add tests/graph/test_base.py
git commit -m "test: add test for child re-parenting behavior"
```

### Task 6: Add Test for Method Chaining

**Files to edit**: `tests/graph/test_base.py`

**What to do**: Verify the fluent interface pattern:

```python
    def test_add_child_returns_self_for_chaining(self) -> None:
        """Test that add_child returns self to enable method chaining."""
        parent = MockCompositeNode()
        child1 = MockNode()
        child2 = MockNode()

        # Chain multiple add_child calls
        result = parent.add_child(child1).add_child(child2)

        assert result is parent
        assert len(parent.children) == 2
        assert child1.parent is parent
        assert child2.parent is parent
```

**How to test**: Run all tests for this class:
```bash
uv run pytest tests/graph/test_base.py::TestCompositeNodeAddChild -v
```

**Commit**:
```bash
git add tests/graph/test_base.py
git commit -m "test: add test for method chaining in add_child"
```

### Task 7: Update Existing Tests

**Files to edit**: `tests/graph/test_base.py`

**What to do**: Find the test `test_parent_child_relationship_with_composite` and update it:

**Original code**:
```python
parent.children.append(child1)
parent.children.append(child2)
# Manually set parent (in real code, add_child method would do this)
child1.parent = parent
child2.parent = parent
```

**Replace with**:
```python
parent.add_child(child1)
parent.add_child(child2)
```

**How to test**: Run all tests to ensure nothing breaks:
```bash
uv run pytest tests/graph/test_base.py -v
```

**Commit**:
```bash
git add tests/graph/test_base.py
git commit -m "refactor: update existing tests to use add_child method"
```

### Task 8: Check Code Coverage

**What to do**: Ensure we maintain 100% coverage for the graph module.

**How to check coverage**:
```bash
# Run tests with coverage for the specific module
uv run pytest tests/graph/test_base.py -v --cov=dqx.graph.base --cov-report=term-missing
```

**Expected output**: Should show 100% coverage. If not, add tests for any missed lines.

### Task 9: Run Linting and Type Checking

**What to do**: Ensure code quality standards are met.

**Commands**:
```bash
# Type checking
uv run mypy src/dqx/graph/base.py

# Linting
uv run ruff check src/dqx/graph/base.py tests/graph/test_base.py

# Auto-fix any linting issues
uv run ruff check --fix src/dqx/graph/base.py tests/graph/test_base.py
```

**Fix any issues** that are reported.

**Commit** if you made changes:
```bash
git add -u
git commit -m "style: fix linting issues"
```

### Task 10: Update Usage in nodes.py

**Files to check**: `src/dqx/graph/nodes.py`

**What to do**: Search for places where children are manually added and parent is set. Look for patterns like:
- `self.children.append(`
- `.parent =`

The `RootNode` class already has an `exists()` method that checks if a child is in the children list, which will work correctly with our implementation.

**Note**: Based on the current code, it seems the actual usage of adding children happens elsewhere (possibly in `api.py` or other modules). No changes needed in `nodes.py` at this time.

### Task 11: Final Test Run

**What to do**: Run the full test suite to ensure nothing is broken.

```bash
# Run all tests
uv run pytest tests/ -v

# Or if that's too many, at least run graph tests
uv run pytest tests/graph/ -v
```

**Expected**: All tests should pass.

## Summary

You've successfully implemented the `add_child` method following TDD principles:

1. ✅ Wrote failing tests first
2. ✅ Implemented minimal code to pass
3. ✅ Added comprehensive test coverage
4. ✅ Updated existing code to use the new method
5. ✅ Maintained code quality standards

## Common Issues and Solutions

### Import Errors
If you get `ImportError: cannot import name 'DQXError'`:
- Make sure you added the import at the top of `base.py`
- The import should be: `from dqx.common import DQXError`

### Type Errors
If mypy complains about types:
- The method signature uses generics: `TChild` is defined at the module level
- Make sure return type is `CompositeNode[TChild]` not just `CompositeNode`

### Test Failures
If tests fail unexpectedly:
- Run a single test with `-v` flag to see detailed output
- Check that you're using the mock classes defined in the test file
- Ensure you're running tests from the project root directory

## Next Steps

After completing this implementation:
1. The `add_child` method can be used throughout the codebase
2. Future code reviews should enforce using `add_child` instead of manual parent setting
3. Consider adding a `remove_child` method if needed in the future
