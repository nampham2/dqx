# Implementation Plan: Strongly Typed Parent Hierarchy

## Overview

This plan guides you through adding type-safe parent properties to DQX's graph node hierarchy. Currently, all nodes have `parent: BaseNode | None`, but we want each node type to know exactly what type its parent is.

## Background Context

DQX is a data quality framework that uses a graph structure to represent validation checks:
- **RootNode**: Top of the hierarchy, has no parent (parent is always None)
- **CheckNode**: Represents a quality check, parent is always a RootNode
- **AssertionNode**: Represents a specific validation rule, parent is always a CheckNode

Current problem: The parent property is weakly typed, allowing incorrect parent-child relationships at compile time.

## Goal

Transform the parent property from:
```python
class BaseNode:
    parent: BaseNode | None  # Too generic!
```

To:
```python
class RootNode:
    parent: None  # Always None, it's the root!

class CheckNode:
    parent: RootNode  # Always has a RootNode parent, never None!

class AssertionNode:
    parent: CheckNode  # Always has a CheckNode parent, never None!
```

## Development Environment Setup

Before starting:
```bash
# Ensure you're in the project root
cd /Users/npham/git-tree/dqx

# Create a new branch
git checkout -b feature/strongly-typed-parents

# Verify tests pass before starting
uv run pytest tests/graph/ -v
uv run mypy src/dqx/graph/
```

## Implementation Tasks

### Task 1: Write Failing Tests for New Parent Types

**Why**: Test Driven Development - write tests first to define the behavior we want.

**Files to create/modify**:
- `tests/graph/test_typed_parents.py` (new file)

**What to do**:
1. Create a new test file specifically for testing the typed parent behavior
2. Write tests that will fail with current implementation but pass with our new design

**Code to write**:
```python
# tests/graph/test_typed_parents.py
"""Tests for strongly typed parent hierarchy."""
import pytest
from dqx.graph.nodes import RootNode, CheckNode, AssertionNode
import sympy as sp


def test_root_node_has_none_parent():
    """Root nodes should always have None as parent."""
    root = RootNode("test_suite")
    assert root.parent is None
    assert root.is_root is True


def test_check_node_requires_root_parent():
    """CheckNode should require a RootNode parent at construction."""
    root = RootNode("test_suite")

    # This should work
    check = CheckNode(parent=root, name="my_check")
    assert check.parent is root
    assert isinstance(check.parent, RootNode)
    assert check.is_root is False

    # This should not work (type checker should complain)
    # check = CheckNode(parent=None, name="orphan")  # Should fail type checking


def test_assertion_node_requires_check_parent():
    """AssertionNode should require a CheckNode parent at construction."""
    root = RootNode("test_suite")
    check = CheckNode(parent=root, name="my_check")

    # This should work
    assertion = AssertionNode(
        parent=check,
        actual=sp.Symbol("x"),
        name="test_assertion"
    )
    assert assertion.parent is check
    assert isinstance(assertion.parent, CheckNode)

    # This should not work (type checker should complain)
    # assertion = AssertionNode(parent=None, actual=sp.Symbol("x"))  # Should fail


def test_factory_methods_create_proper_hierarchy():
    """Test that factory methods maintain proper parent-child relationships."""
    root = RootNode("test_suite")

    # Use factory method to create check
    check = root.add_check("my_check", tags=["important"])
    assert check.parent is root
    assert check in root.children

    # Use factory method to create assertion
    assertion = check.add_assertion(
        actual=sp.Symbol("x"),
        name="x > 0"
    )
    assert assertion.parent is check
    assert assertion in check.children


def test_type_safety_at_runtime():
    """Verify runtime type safety for parent assignments."""
    root = RootNode("test_suite")
    check = CheckNode(parent=root, name="check")

    # These should fail at runtime if someone bypasses type checking
    with pytest.raises(TypeError):
        # Trying to set wrong parent type
        another_check = CheckNode(parent=root, name="another")
        assertion = AssertionNode(
            parent=another_check,  # This is correct
            actual=sp.Symbol("x")
        )
        # This should fail - can't set assertion's parent to root
        assertion.parent = root  # type: ignore
```

**How to test**:
```bash
# Run the new tests (they should fail)
uv run pytest tests/graph/test_typed_parents.py -v

# Verify existing tests still pass
uv run pytest tests/graph/test_base.py -v
```

**Commit**:
```bash
git add tests/graph/test_typed_parents.py
git commit -m "test: add failing tests for strongly typed parent hierarchy"
```

### Task 2: Update Base Classes with Generic Parent Type

**Why**: Make BaseNode generic over its parent type to enable type-safe parent properties.

**Files to modify**:
- `src/dqx/graph/base.py`

**What to do**:
1. Add a `TParent` type variable
2. Make `BaseNode` generic over `TParent`
3. Update `CompositeNode` to also be generic over parent type
4. Update constructors to require parent parameter

**Code changes**:
```python
# src/dqx/graph/base.py
from __future__ import annotations
from typing import Generic, Protocol, TypeVar, Optional, Union

from dqx.common import DQXError

# Type variables
TParent = TypeVar("TParent", bound=Union["BaseNode", None])
TChild = TypeVar("TChild", bound="BaseNode")
TNode = TypeVar("TNode", bound="BaseNode")


class NodeVisitor(Protocol):
    # ... existing code remains unchanged ...


class BaseNode(Generic[TParent]):
    """Base class for all nodes in the graph.

    Now generic over TParent to enable strongly typed parent relationships.
    Each node type can specify exactly what type its parent should be.
    """

    def __init__(self, parent: TParent) -> None:
        """Initialize a base node with its parent.

        Args:
            parent: The parent node. Type depends on the specific node class.
                   RootNode has None parent, all others have specific parent types.
        """
        self.parent: TParent = parent

    @property
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.parent is None

    def accept(self, visitor: NodeVisitor) -> None:
        """Accept a visitor for traversal."""
        return visitor.visit(self)

    async def accept_async(self, visitor: NodeVisitor) -> None:
        """Accept an asynchronous visitor for traversal."""
        return await visitor.visit_async(self)

    def is_leaf(self) -> bool:
        """Check if this node has children."""
        raise NotImplementedError("Subclasses must implement is_leaf method.")


class CompositeNode(BaseNode[TParent], Generic[TParent, TChild]):
    """Base class for nodes that can have children.

    Now generic over both TParent (parent type) and TChild (children type).
    """

    def __init__(self, parent: TParent) -> None:
        """Initialize a composite node with its parent."""
        super().__init__(parent)
        self.children: list[TChild] = []

    def is_leaf(self) -> bool:
        """Composite nodes are not leaves (they can have children)."""
        return False

    def add_child(self, child: TChild) -> CompositeNode[TParent, TChild]:
        """Add a child node.

        Note: We no longer set child.parent here because the child
        already has its parent set in its constructor.

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
        # Don't set child.parent - it's already set in constructor!
        return self
```

**How to test**:
```python
# Quick verification in Python REPL
uv run python
>>> from dqx.graph.base import BaseNode, CompositeNode
>>> # Should work without errors
```

**Run type checking**:
```bash
uv run mypy src/dqx/graph/base.py
```

**Commit**:
```bash
git add src/dqx/graph/base.py
git commit -m "refactor: make BaseNode generic over parent type"
```

### Task 3: Update Node Implementations

**Why**: Apply the generic parent types to actual node classes.

**Files to modify**:
- `src/dqx/graph/nodes.py`

**What to do**:
1. Update RootNode to have None parent
2. Update CheckNode to require RootNode parent
3. Update AssertionNode to require CheckNode parent
4. Add factory methods for safe child creation

**Code changes**:
```python
# src/dqx/graph/nodes.py
from __future__ import annotations
from typing import TYPE_CHECKING

import sympy as sp
from returns.result import Result

from dqx.common import SeverityLevel, SymbolicValidator
from dqx.graph.base import BaseNode, CompositeNode
from dqx.provider import SymbolicMetric

if TYPE_CHECKING:
    # Avoid circular imports
    pass


class RootNode(CompositeNode[None, "CheckNode"]):
    """Root node of the verification graph hierarchy.

    Parent type is None (roots have no parent).
    Child type is CheckNode.
    """

    def __init__(self, name: str) -> None:
        """Initialize a root node.

        Args:
            name: Human-readable name for the verification suite
        """
        super().__init__(parent=None)  # Root always has None parent
        self.name = name

    def add_check(
        self,
        name: str,
        tags: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> CheckNode:
        """Factory method to create and add a check node.

        This ensures the check has the correct parent type.

        Args:
            name: Name for the check
            tags: Optional tags for categorizing the check
            datasets: Optional list of datasets this check applies to

        Returns:
            The newly created CheckNode
        """
        check = CheckNode(parent=self, name=name, tags=tags, datasets=datasets)
        self.add_child(check)
        return check

    def exists(self, child: CheckNode) -> bool:
        """Check if a specific CheckNode exists as a direct child."""
        return child in self.children


class CheckNode(CompositeNode["RootNode", "AssertionNode"]):
    """Node representing a data quality check.

    Parent type is RootNode (never None).
    Child type is AssertionNode.
    """

    def __init__(
        self,
        parent: RootNode,  # Required, not optional!
        name: str,
        tags: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        """Initialize a check node.

        Args:
            parent: The RootNode parent (required)
            name: Name for the check
            tags: Optional tags for categorizing the check
            datasets: Optional list of datasets this check applies to
        """
        super().__init__(parent)
        self.name = name
        self.tags = tags or []
        self.datasets = datasets or []

    def add_assertion(
        self,
        actual: sp.Expr,
        name: str | None = None,
        severity: SeverityLevel = "P1",
        validator: SymbolicValidator | None = None,
    ) -> AssertionNode:
        """Factory method to create and add an assertion node.

        This ensures the assertion has the correct parent type.

        Args:
            actual: The symbolic expression to evaluate
            name: Optional human-readable description
            severity: Severity level for failures
            validator: Optional validation function

        Returns:
            The newly created AssertionNode
        """
        assertion = AssertionNode(
            parent=self,
            actual=actual,
            name=name,
            severity=severity,
            validator=validator
        )
        self.add_child(assertion)
        return assertion


class AssertionNode(BaseNode["CheckNode"]):
    """Node representing an assertion to be evaluated.

    Parent type is CheckNode (never None).
    AssertionNodes are leaf nodes and cannot have children.
    """

    def __init__(
        self,
        parent: CheckNode,  # Required, not optional!
        actual: sp.Expr,
        name: str | None = None,
        severity: SeverityLevel = "P1",
        validator: SymbolicValidator | None = None,
    ) -> None:
        """Initialize an assertion node.

        Args:
            parent: The CheckNode parent (required)
            actual: The symbolic expression to evaluate
            name: Optional human-readable description
            severity: Severity level for failures
            validator: Optional validation function
        """
        super().__init__(parent)
        self.actual = actual
        self.name = name
        self.severity = severity
        self.validator = validator
        self._value: Result[float, dict[SymbolicMetric | sp.Expr, str]]

    def is_leaf(self) -> bool:
        """Assertion nodes are always leaves."""
        return True
```

**How to test**:
```bash
# Run the typed parent tests (some should pass now)
uv run pytest tests/graph/test_typed_parents.py -v

# Check types
uv run mypy src/dqx/graph/
```

**Commit**:
```bash
git add src/dqx/graph/nodes.py
git commit -m "feat: implement strongly typed parent hierarchy in nodes"
```

### Task 4: Fix Existing Code That Creates Nodes

**Why**: The new design requires parents at construction time, breaking existing code.

**Files to check and potentially modify**:
- Search for node construction patterns
- Update tests
- Update any code that builds graphs

**How to find affected code**:
```bash
# Find CheckNode constructions
uv run rg "CheckNode\(" --type py

# Find AssertionNode constructions
uv run rg "AssertionNode\(" --type py

# Find places where parent is set after construction
uv run rg "\.parent\s*=" --type py
```

**Common patterns to fix**:

Old pattern:
```python
check = CheckNode(name="test")
root.add_child(check)  # Parent set implicitly
```

New pattern:
```python
check = root.add_check(name="test")  # Parent set explicitly via factory
# OR
check = CheckNode(parent=root, name="test")
root.add_child(check)
```

**Run tests to find breakages**:
```bash
# This will show you what needs fixing
uv run pytest tests/ -v
```

### Task 5: Update Existing Tests

**Why**: Existing tests create nodes without parents, which is no longer allowed.

**Files to modify**:
- `tests/graph/test_base.py`
- Any other test files that create nodes

**Example fixes**:
```python
# Old test pattern
def test_something():
    check = CheckNode(name="test")  # No parent!

# New test pattern
def test_something():
    root = RootNode("test_suite")
    check = CheckNode(parent=root, name="test")  # Parent required!
    # OR use factory
    check = root.add_check(name="test")
```

**How to test**:
```bash
# Fix tests incrementally
uv run pytest tests/graph/test_base.py -v
# Fix failures one by one
```

**Commit after each file**:
```bash
git add tests/graph/test_base.py
git commit -m "test: update graph tests for typed parent hierarchy"
```

### Task 6: Update Code That Uses Parent Property

**Why**: Code might assume parent can be None for non-root nodes.

**Files to check**:
- `src/dqx/graph/visitors.py`
- `src/dqx/display.py`

**Look for patterns like**:
```python
# Old assumption
if node.parent is not None:
    # do something

# New reality - only root has None parent
if not node.is_root:
    # do something - parent is guaranteed to exist
```

### Task 7: Add Runtime Type Validation (Optional Safety)

**Why**: Python's type hints aren't enforced at runtime. Add validation for extra safety.

**What to add** (in base.py):
```python
class BaseNode(Generic[TParent]):
    def __init__(self, parent: TParent) -> None:
        # Optional: Add runtime validation
        if self.__class__.__name__ == "RootNode" and parent is not None:
            raise TypeError("RootNode must have None parent")
        elif self.__class__.__name__ == "CheckNode" and not isinstance(parent, RootNode):
            raise TypeError("CheckNode must have RootNode parent")
        elif self.__class__.__name__ == "AssertionNode" and not isinstance(parent, CheckNode):
            raise TypeError("AssertionNode must have CheckNode parent")

        self.parent: TParent = parent
```

### Task 8: Update Documentation

**Why**: The API has changed, documentation needs to reflect this.

**Files to update**:
- `README.md` - Update examples showing node creation
- Docstrings in modified files

**Example README update**:
```markdown
# Old example
check = CheckNode(name="validation")

# New example
root = RootNode("suite")
check = root.add_check(name="validation")
# OR
check = CheckNode(parent=root, name="validation")
```

### Task 9: Final Verification

**Run all tests**:
```bash
# All tests should pass
uv run pytest tests/ -v

# Type checking should pass
uv run mypy src/dqx/

# Linting
uv run ruff check src/ tests/
```

**Run example code** to ensure it works:
```python
# Verify in REPL
uv run python
>>> from dqx.graph.nodes import RootNode
>>> root = RootNode("test")
>>> check = root.add_check("my_check")
>>> assertion = check.add_assertion(sp.Symbol("x"), name="x > 0")
>>>
>>> # Verify types
>>> assert isinstance(check.parent, RootNode)
>>> assert isinstance(assertion.parent, CheckNode)
>>> print("Success!")
```

## Common Pitfalls to Avoid

1. **Don't forget to update child.parent assignment**: Since parent is now set in constructor, don't set it again in `add_child()`

2. **Don't create orphaned nodes**: Every non-root node needs a parent at construction

3. **Watch for circular imports**: Use `TYPE_CHECKING` guards if needed

4. **Test incrementally**: Fix and commit one file at a time

5. **Don't skip type checking**: Run mypy frequently to catch issues

## Testing Strategy

1. **Unit tests first**: Get `test_typed_parents.py` passing
2. **Fix existing tests**: Update them to use new construction pattern
3. **Integration tests**: Ensure the full graph still works
4. **Type checking**: Mypy should have no errors
5. **Runtime testing**: Actually build and traverse a graph

## Commit Strategy

Make small, focused commits:
- One commit per test file fixed
- One commit per source file updated
- Separate commits for refactoring vs new features
- Use conventional commit messages (feat:, fix:, test:, docs:)

## Final Checklist

- [ ] All tests pass
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] Documentation updated
- [ ] Examples in README work
- [ ] No uses of `# type: ignore` added
- [ ] Parent is required at construction for all non-root nodes
- [ ] Factory methods available for convenient child creation

## Questions You Might Have

**Q: Why not just use property overrides?**
A: Generic types give us compile-time safety. Property overrides would only give runtime checks.

**Q: Why require parent at construction?**
A: It prevents orphaned nodes and makes the relationship explicit and immutable.

**Q: What about backward compatibility?**
A: This is a breaking change. Existing code needs updates, but the new API is safer.

**Q: Why factory methods?**
A: They ensure correct parent-child relationships and make the API more convenient.

Remember: If you get stuck, look at the test file first - it shows exactly what behavior we want!
