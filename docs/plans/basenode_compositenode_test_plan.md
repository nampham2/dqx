# Unit Test Implementation Plan for BaseNode and CompositeNode

## Table of Contents
1. [Overview](#overview)
2. [Background & Context](#background--context)
3. [Test Design Principles](#test-design-principles)
4. [Development Environment Setup](#development-environment-setup)
5. [Implementation Tasks](#implementation-tasks)
6. [Code Examples](#code-examples)
7. [Testing Checklist](#testing-checklist)
8. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
9. [Commit Strategy](#commit-strategy)

## Overview

This document provides a step-by-step guide for implementing comprehensive unit tests for the `BaseNode` and `CompositeNode` classes in the DQX data quality framework. These classes form the foundation of DQX's graph-based architecture for managing data quality checks.

**Time Estimate**: 4-6 hours for complete implementation
**Required Skills**: Python, pytest, basic understanding of design patterns (Composite, Visitor)

## Background & Context

### What are BaseNode and CompositeNode?

**BaseNode** (`src/dqx/graph/base.py`):
- Abstract base class for all nodes in the DQX verification graph
- Implements basic node functionality: parent tracking, visitor pattern support
- Cannot have children (leaf nodes inherit from this)

**CompositeNode** (`src/dqx/graph/base.py`):
- Extends BaseNode to support child nodes (Composite pattern)
- Generic class that maintains a typed list of children
- Used by nodes that need to contain other nodes (RootNode, CheckNode)

### Why Test These Classes?

These classes are fundamental building blocks. Any bugs here will cascade throughout the entire system. We need to ensure:
- Parent-child relationships work correctly
- The visitor pattern is properly implemented
- Type safety is maintained for generic children
- Edge cases are handled gracefully

### Project Structure
```
dqx/
├── src/
│   └── dqx/
│       └── graph/
│           ├── base.py          # Classes we're testing
│           └── nodes.py         # Concrete implementations
└── tests/
    └── graph/                   # Where our tests will go
        └── test_base.py         # Test file we'll create
```

## Test Design Principles

### 1. Test Isolation
Each test should be independent and not rely on other tests. Use fresh instances for each test.

### 2. Arrange-Act-Assert (AAA)
Structure each test with:
- **Arrange**: Set up test data and conditions
- **Act**: Execute the code being tested
- **Assert**: Verify the expected outcome

### 3. Test One Thing
Each test method should verify a single behavior or edge case.

### 4. Clear Test Names
Use descriptive names that explain what is being tested:
- ✅ `test_is_root_returns_true_when_parent_is_none`
- ❌ `test_root`

### 5. Use Type Hints
Always include type hints in test code for clarity and IDE support.

### 6. No Mocking When Possible
Prefer using real objects over mocks. Only mock external dependencies.

## Development Environment Setup

### Prerequisites Check
```bash
# Verify Python version (should be 3.11 or 3.12)
python --version

# Verify uv is installed
uv --version

# Verify you're in the project directory
pwd  # Should show: /Users/npham/git-tree/dqx
```

### Install Dependencies
```bash
# Sync project dependencies
uv sync

# Verify pytest is available
uv run pytest --version
```

### Verify Existing Tests Run
```bash
# Run existing tests to ensure environment is working
uv run pytest tests/ -v
```

## Implementation Tasks

### Task 1: Create Test File Structure
**File**: `tests/graph/test_base.py`
**Time**: 15 minutes

1. Create the test directory if it doesn't exist:
   ```bash
   mkdir -p tests/graph
   ```

2. Create the test file with proper imports:
   ```python
   """Unit tests for BaseNode and CompositeNode classes."""
   from __future__ import annotations

   import pytest
   from typing import Any

   from dqx.graph.base import BaseNode, CompositeNode, NodeVisitor
   ```

3. Commit:
   ```bash
   git add tests/graph/test_base.py
   git commit -m "test: add test file structure for BaseNode and CompositeNode"
   ```

### Task 2: Create Test Implementations for BaseNode
**Time**: 30 minutes

Since BaseNode has an abstract method `is_leaf()`, we need a concrete implementation for testing:

1. Create a test implementation class:
   ```python
   class TestNode(BaseNode):
       """Concrete implementation of BaseNode for testing."""

       def is_leaf(self) -> bool:
           """Test implementation always returns True."""
           return True
   ```

2. Create another implementation that returns False:
   ```python
   class TestNonLeafNode(BaseNode):
       """Concrete implementation that claims not to be a leaf."""

       def is_leaf(self) -> bool:
           return False
   ```

3. Commit:
   ```bash
   git add -A
   git commit -m "test: add concrete BaseNode implementations for testing"
   ```

### Task 3: Test BaseNode Initialization
**Time**: 20 minutes

Write tests for BaseNode initialization:

```python
class TestBaseNode:
    """Test suite for BaseNode functionality."""

    def test_init_sets_parent_to_none(self) -> None:
        """Test that BaseNode initializes with parent set to None."""
        # Arrange & Act
        node = TestNode()

        # Assert
        assert node.parent is None

    def test_is_root_returns_true_when_parent_is_none(self) -> None:
        """Test that is_root returns True when node has no parent."""
        # Arrange
        node = TestNode()

        # Act
        result = node.is_root

        # Assert
        assert result is True

    def test_is_root_returns_false_when_parent_exists(self) -> None:
        """Test that is_root returns False when node has a parent."""
        # Arrange
        child = TestNode()
        parent = TestNode()
        child.parent = parent

        # Act
        result = child.is_root

        # Assert
        assert result is False
```

Run tests: `uv run pytest tests/graph/test_base.py::TestBaseNode -v`

Commit: `git commit -am "test: add BaseNode initialization and is_root tests"`

### Task 4: Test Visitor Pattern Implementation
**Time**: 30 minutes

1. Create a test visitor:
   ```python
   class TestVisitor:
       """Test visitor that records visited nodes."""

       def __init__(self) -> None:
           self.visited_nodes: list[BaseNode] = []

       def visit(self, node: BaseNode) -> None:
           """Record the visited node."""
           self.visited_nodes.append(node)

       async def visit_async(self, node: BaseNode) -> None:
           """Record the visited node asynchronously."""
           self.visited_nodes.append(node)
   ```

2. Test synchronous visitor:
   ```python
   def test_accept_calls_visitor_visit_method(self) -> None:
       """Test that accept() correctly calls visitor's visit method."""
       # Arrange
       node = TestNode()
       visitor = TestVisitor()

       # Act
       node.accept(visitor)

       # Assert
       assert len(visitor.visited_nodes) == 1
       assert visitor.visited_nodes[0] is node
   ```

3. Test asynchronous visitor:
   ```python
   @pytest.mark.asyncio
   async def test_accept_async_calls_visitor_visit_async_method(self) -> None:
       """Test that accept_async() correctly calls visitor's async method."""
       # Arrange
       node = TestNode()
       visitor = TestVisitor()

       # Act
       await node.accept_async(visitor)

       # Assert
       assert len(visitor.visited_nodes) == 1
       assert visitor.visited_nodes[0] is node
   ```

Note: You'll need to install pytest-asyncio: `uv add --dev pytest-asyncio`

Commit: `git commit -am "test: add visitor pattern tests for BaseNode"`

### Task 5: Test is_leaf Abstract Method
**Time**: 15 minutes

```python
def test_is_leaf_must_be_implemented_by_subclasses(self) -> None:
    """Test that is_leaf raises NotImplementedError in base class."""
    # We can't test this directly on BaseNode since it's abstract,
    # but we can verify our test implementations work correctly

    # Arrange
    leaf_node = TestNode()
    non_leaf_node = TestNonLeafNode()

    # Act & Assert
    assert leaf_node.is_leaf() is True
    assert non_leaf_node.is_leaf() is False
```

Commit: `git commit -am "test: add is_leaf implementation tests"`

### Task 6: Create Test Implementation for CompositeNode
**Time**: 20 minutes

1. Create a typed composite node for testing:
   ```python
   class TestCompositeNode(CompositeNode[TestNode]):
       """Concrete implementation of CompositeNode for testing."""
       pass
   ```

2. Create a test child node type:
   ```python
   class TestChildNode(BaseNode):
       """A specific child node type for testing type constraints."""

       def __init__(self, name: str) -> None:
           super().__init__()
           self.name = name

       def is_leaf(self) -> bool:
           return True
   ```

Commit: `git commit -am "test: add CompositeNode test implementations"`

### Task 7: Test CompositeNode Initialization
**Time**: 20 minutes

```python
class TestCompositeNode:
    """Test suite for CompositeNode functionality."""

    def test_init_creates_empty_children_list(self) -> None:
        """Test that CompositeNode initializes with empty children list."""
        # Arrange & Act
        node = TestCompositeNode()

        # Assert
        assert isinstance(node.children, list)
        assert len(node.children) == 0

    def test_init_calls_parent_init(self) -> None:
        """Test that CompositeNode properly initializes BaseNode attributes."""
        # Arrange & Act
        node = TestCompositeNode()

        # Assert
        assert node.parent is None
        assert node.is_root is True

    def test_children_list_is_instance_attribute(self) -> None:
        """Test that each instance has its own children list."""
        # Arrange
        node1 = TestCompositeNode()
        node2 = TestCompositeNode()
        child = TestNode()

        # Act
        node1.children.append(child)

        # Assert
        assert len(node1.children) == 1
        assert len(node2.children) == 0  # Should not be shared
```

Run tests: `uv run pytest tests/graph/test_base.py::TestCompositeNode -v`

Commit: `git commit -am "test: add CompositeNode initialization tests"`

### Task 8: Test CompositeNode Type Safety
**Time**: 25 minutes

```python
def test_children_type_safety(self) -> None:
    """Test that children list maintains type safety."""
    # This is more of a documentation test since Python doesn't
    # enforce generics at runtime, but it's good to show intent

    # Arrange
    node = TestCompositeNode()
    child = TestNode()

    # Act
    node.children.append(child)

    # Assert
    assert all(isinstance(child, TestNode) for child in node.children)

def test_is_leaf_returns_false(self) -> None:
    """Test that CompositeNode.is_leaf always returns False."""
    # Arrange
    node = TestCompositeNode()

    # Act
    result = node.is_leaf()

    # Assert
    assert result is False

def test_is_leaf_returns_false_even_with_no_children(self) -> None:
    """Test that is_leaf returns False regardless of children."""
    # Arrange
    node = TestCompositeNode()
    assert len(node.children) == 0  # Verify no children

    # Act
    result = node.is_leaf()

    # Assert
    assert result is False  # Still not a leaf conceptually
```

Commit: `git commit -am "test: add CompositeNode type safety and is_leaf tests"`

### Task 9: Integration Tests
**Time**: 30 minutes

Test how BaseNode and CompositeNode work together:

```python
class TestBaseNodeCompositeNodeIntegration:
    """Test integration between BaseNode and CompositeNode."""

    def test_composite_node_with_visitor_pattern(self) -> None:
        """Test visitor pattern works with composite nodes."""
        # Arrange
        composite = TestCompositeNode()
        visitor = TestVisitor()

        # Act
        composite.accept(visitor)

        # Assert
        assert len(visitor.visited_nodes) == 1
        assert visitor.visited_nodes[0] is composite

    def test_parent_child_relationship_with_composite(self) -> None:
        """Test setting parent on nodes in composite structure."""
        # Arrange
        parent = TestCompositeNode()
        child1 = TestNode()
        child2 = TestNode()

        # Act
        parent.children.append(child1)
        parent.children.append(child2)
        # Manually set parent (in real code, add_child method would do this)
        child1.parent = parent
        child2.parent = parent

        # Assert
        assert child1.parent is parent
        assert child2.parent is parent
        assert child1.is_root is False
        assert child2.is_root is False
        assert parent.is_root is True
```

Commit: `git commit -am "test: add BaseNode and CompositeNode integration tests"`

### Task 10: Edge Cases and Error Conditions
**Time**: 25 minutes

```python
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_node_can_change_parent(self) -> None:
        """Test that a node's parent can be changed."""
        # Arrange
        node = TestNode()
        parent1 = TestNode()
        parent2 = TestNode()

        # Act & Assert
        node.parent = parent1
        assert node.parent is parent1
        assert node.is_root is False

        node.parent = parent2
        assert node.parent is parent2

        node.parent = None
        assert node.parent is None
        assert node.is_root is True

    def test_visitor_with_none_handling(self) -> None:
        """Test visitor behavior with edge cases."""
        # Arrange
        node = TestNode()

        class ErrorVisitor:
            def visit(self, node: BaseNode) -> None:
                raise ValueError("Test error")

            async def visit_async(self, node: BaseNode) -> None:
                raise ValueError("Test async error")

        visitor = ErrorVisitor()

        # Act & Assert
        with pytest.raises(ValueError, match="Test error"):
            node.accept(visitor)

        with pytest.raises(ValueError, match="Test async error"):
            import asyncio
            asyncio.run(node.accept_async(visitor))
```

Commit: `git commit -am "test: add edge case tests for BaseNode and CompositeNode"`

### Task 11: Run Full Test Suite and Check Coverage
**Time**: 15 minutes

1. Run all tests with coverage:
   ```bash
   uv run pytest tests/graph/test_base.py -v --cov=dqx.graph.base --cov-report=term-missing
   ```

2. Ensure 100% coverage. If not, add missing tests.

3. Run with mypy to check types:
   ```bash
   uv run mypy tests/graph/test_base.py
   ```

4. Run with ruff:
   ```bash
   uv run ruff check tests/graph/test_base.py
   ```

5. Final commit:
   ```bash
   git add -A
   git commit -m "test: complete test suite for BaseNode and CompositeNode with 100% coverage"
   ```

## Code Examples

### Complete Test Structure
Here's what your final test file structure should look like:

```python
"""Unit tests for BaseNode and CompositeNode classes."""
from __future__ import annotations

import asyncio
import pytest
from typing import Any

from dqx.graph.base import BaseNode, CompositeNode, NodeVisitor


# Test implementations
class TestNode(BaseNode):
    """Concrete implementation of BaseNode for testing."""

    def is_leaf(self) -> bool:
        """Test implementation always returns True."""
        return True


class TestNonLeafNode(BaseNode):
    """Concrete implementation that claims not to be a leaf."""

    def is_leaf(self) -> bool:
        return False


class TestChildNode(BaseNode):
    """A specific child node type for testing type constraints."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def is_leaf(self) -> bool:
        return True


class TestCompositeNode(CompositeNode[TestNode]):
    """Concrete implementation of CompositeNode for testing."""
    pass


class TestVisitor:
    """Test visitor that records visited nodes."""

    def __init__(self) -> None:
        self.visited_nodes: list[BaseNode] = []

    def visit(self, node: BaseNode) -> None:
        """Record the visited node."""
        self.visited_nodes.append(node)

    async def visit_async(self, node: BaseNode) -> None:
        """Record the visited node asynchronously."""
        self.visited_nodes.append(node)


# Test classes
class TestBaseNode:
    """Test suite for BaseNode functionality."""
    # ... test methods ...


class TestCompositeNode:
    """Test suite for CompositeNode functionality."""
    # ... test methods ...


class TestBaseNodeCompositeNodeIntegration:
    """Test integration between BaseNode and CompositeNode."""
    # ... test methods ...


class TestEdgeCases:
    """Test edge cases and error conditions."""
    # ... test methods ...
```

## Testing Checklist

Before considering the tests complete, verify:

- [ ] All test files follow the naming convention `test_*.py`
- [ ] All test classes start with `Test`
- [ ] All test methods start with `test_`
- [ ] Each test has a clear docstring explaining what it tests
- [ ] Tests follow the Arrange-Act-Assert pattern
- [ ] No test depends on another test
- [ ] All tests pass when run individually and together
- [ ] Coverage is at 100% for both BaseNode and CompositeNode
- [ ] Type hints are used throughout
- [ ] No unnecessary mocking is used
- [ ] Edge cases are covered
- [ ] Tests run quickly (< 1 second total)
- [ ] Code passes mypy type checking
- [ ] Code passes ruff linting

## Common Pitfalls to Avoid

### 1. Testing Implementation Details
❌ **Don't**: Test private methods or internal state
✅ **Do**: Test public interface and observable behavior

### 2. Overly Complex Tests
❌ **Don't**: Write tests with complex setup and multiple assertions
✅ **Do**: Keep tests simple and focused on one behavior

### 3. Misleading Test Names
❌ **Don't**: `test_node_1`, `test_composite_works`
✅ **Do**: `test_is_root_returns_true_when_parent_is_none`

### 4. Shared State Between Tests
❌ **Don't**: Use class-level attributes that tests modify
✅ **Do**: Create fresh instances in each test

### 5. Testing Python Language Features
❌ **Don't**: Test that `isinstance()` works or that lists are lists
✅ **Do**: Test your specific logic and edge cases

### 6. Forgetting Async Tests
❌ **Don't**: Skip testing async methods because they're "the same"
✅ **Do**: Test both sync and async paths explicitly

### 7. Ignoring Type Safety
❌ **Don't**: Use `Any` types or ignore mypy errors
✅ **Do**: Use proper type hints to catch issues early

## Commit Strategy

Follow these commit guidelines:

1. **Prefix commits** with category:
   - `test:` for adding tests
   - `fix:` if you find and fix bugs while testing
   - `docs:` for documentation updates

2. **Commit frequently**:
   - After each task completion
   - When all tests for a class pass
   - Before refactoring

3. **Write clear messages**:
   ```bash
   # Good
   git commit -m "test: add visitor pattern tests for BaseNode"

   # Bad
   git commit -m "added tests"
   ```

4. **Keep commits atomic**:
   - One logical change per commit
   - Don't mix test additions with code fixes

## Final Steps

1. **Run final quality checks**:
   ```bash
   # Run all tests
   uv run pytest tests/graph/test_base.py -v

   # Check coverage
   uv run pytest tests/graph/test_base.py --cov=dqx.graph.base --cov-report=term-missing

   # Type checking
   uv run mypy tests/graph/test_base.py

   # Linting
   uv run ruff check tests/graph/test_base.py
   ```

2. **Update documentation** if needed:
   - Add notes about any discovered edge cases
   - Document any decisions made during testing

3. **Create a summary commit**:
   ```bash
   git commit -m "test: complete BaseNode and CompositeNode test suite

   - Added comprehensive tests for BaseNode parent/child relationships
   - Added tests for visitor pattern (sync and async)
   - Added tests for CompositeNode children management
   - Added integration and edge case tests
   - Achieved 100% code coverage
   - All tests pass mypy and ruff checks"
   ```

## Questions or Issues?

If you encounter any issues:

1. Check existing tests in the codebase for patterns
2. Refer to pytest documentation
3. Ensure you're using the virtual environment: `source .venv/bin/activate`
4. Run a single test for debugging: `uv run pytest tests/graph/test_base.py::TestClass::test_method -v`

Remember: Good tests are an investment in code quality. Take your time and do it right!
