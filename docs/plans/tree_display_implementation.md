# Tree Display Implementation Plan

## Overview

This plan guides you through adding a tree visualization feature to the DQX data quality framework. The feature will allow users to print the graph structure to the console using the Rich library.

## Background for the Engineer

### What is DQX?
- A data quality framework that uses a graph structure to represent checks and assertions
- The graph has three levels: RootNode → CheckNode → AssertionNode  
- Each node has a `parent` attribute (except the root)
- The `is_leaf()` method returns `True` for AssertionNode and `False` for CompositeNode subclasses

### Key Files You'll Need to Understand:
- `src/dqx/graph/base.py` - Defines BaseNode and CompositeNode classes
- `src/dqx/graph/nodes.py` - Defines RootNode, CheckNode, AssertionNode
- `src/dqx/graph/traversal.py` - Contains Graph class with DFS traversal
- `src/dqx/graph/visitors.py` - Example of visitor pattern (NodeCollector)

### Important Node Details:
- **CheckNode** has a `node_name()` method that returns `self.label or self.name`
- **AssertionNode** doesn't have a name attribute - only `label` and `actual` (the expression)
- **RootNode** has a `name` attribute

### Project Setup:
- Uses `uv` for dependency management
- Run commands with `uv run <command>`
- Tests use pytest
- Type checking with mypy
- Linting with ruff

## Implementation Tasks

### Task 1: Set Up the Display Module

**Goal:** Create the foundation for the display feature

**Steps:**
1. Create new file: `src/dqx/display.py`
2. Create test file: `tests/test_display.py`
3. Add module docstring explaining the purpose

**What to implement:**
- A protocol called `NodeFormatter` with one method: `format_node(node: BaseNode) -> str`
- A class `SimpleNodeFormatter` that implements this protocol
  - Should check for `node_name()` method first (for CheckNode compatibility)
  - Then check for `name` attribute if it exists and is not empty
  - Otherwise return the class name

**Implementation details:**
```python
def format_node(self, node: BaseNode) -> str:
    # Try node_name() method first (for CheckNode)
    if hasattr(node, 'node_name') and callable(node.node_name):
        return node.node_name()
    # Then try name attribute
    elif hasattr(node, 'name') and node.name:
        return node.name
    # Finally, use class name
    else:
        return node.__class__.__name__
```

**How to test:**
- Create mock nodes with and without name attributes
- Test CheckNode with both name and label
- Test that formatter returns expected strings
- Run: `uv run pytest tests/test_display.py -v`

**Commit:** "feat: add display module with NodeFormatter protocol"

### Task 2: Create the Tree Builder Visitor

**Goal:** Implement a visitor that builds a Rich Tree during graph traversal

**Background:**
- The visitor pattern separates algorithms from the objects they operate on
- Look at `NodeCollector` in `src/dqx/graph/visitors.py` for an example
- The Graph.dfs() method will call your visitor's `visit()` method for each node
- DFS visits nodes in parent-first order, so parent is always visited before children

**Steps:**
1. In `src/dqx/display.py`, create class `TreeBuilderVisitor`
2. Add constructor that accepts a `NodeFormatter` parameter
3. Add instance variables:
   - `tree` (Optional[Tree]) - The Rich tree being built
   - `node_to_tree_map` (Dict[BaseNode, Tree]) - Maps graph nodes to tree nodes

**What to implement:**
- `visit(node: BaseNode) -> None` method that:
  - Formats the node using the formatter
  - If it's the first node (tree is None), creates the root Tree
  - Otherwise, finds the parent's tree node and adds this node as a child
  - Stores the mapping for future children
  - Raises ValueError with specific error message if parent wasn't visited first (defensive programming - this shouldn't happen with DFS but we check anyway)
- `visit_async(node: BaseNode) -> None` - Just delegate to `visit()`

**Error handling:**
```python
if node.parent not in self.node_to_tree_map:
    raise ValueError(
        f"Parent of node '{formatter.format_node(node)}' was not visited before the child. "
        f"This indicates an issue with the traversal order."
    )
```

**How to test:**
- Create mock nodes with parent relationships
- Call visit() in the correct order and verify tree structure
- Test error case: visiting child before parent
- Run: `uv run pytest tests/test_display.py::test_tree_builder -v`

**Things to check:**
- Import Rich's Tree class: `from rich.tree import Tree`
- Review how parent relationships work in `src/dqx/graph/base.py`

**Commit:** "feat: add TreeBuilderVisitor for building Rich trees"

### Task 3: Add Public API Function

**Goal:** Create a simple function users can call to print graphs

**Steps:**
1. In `src/dqx/display.py`, add function `print_graph(graph: Graph, formatter: Optional[NodeFormatter] = None)`
2. Import necessary types from `dqx.graph.traversal`

**What to implement:**
- If formatter is None, create a SimpleNodeFormatter
- Create a TreeBuilderVisitor with the formatter
- Call graph.dfs(visitor) to traverse and build the tree
- If visitor.tree exists, create a Rich Console and print the tree

**How to test:**
- Mock the Console class to verify print is called
- Test with default formatter
- Test with custom formatter
- Verify the tree passed to console.print has expected structure
- Run: `uv run pytest tests/test_display.py::test_print_graph -v`

**Dependencies to import:**
- `from rich.console import Console`
- `from dqx.graph.traversal import Graph`

**Commit:** "feat: add print_graph public API function"

### Task 4: Add Convenience Method to Graph Class

**Goal:** Allow users to call `graph.print_tree()` directly

**Steps:**
1. Open `src/dqx/graph/traversal.py`
2. Add method to the Graph class
3. Handle circular import using TYPE_CHECKING

**What to implement:**
- Add imports at the top:
  ```python
  from typing import TYPE_CHECKING, Optional
  if TYPE_CHECKING:
      from dqx.display import NodeFormatter
  ```
- Add method `print_tree(self, formatter: Optional["NodeFormatter"] = None)`
- Method should import print_graph and call it with self and formatter

**How to test:**
- Create new test file: `tests/test_graph_display.py`
- Mock the print_graph function
- Verify graph.print_tree() calls print_graph correctly
- Run: `uv run pytest tests/test_graph_display.py -v`

**Why TYPE_CHECKING?**
- Prevents circular imports at runtime
- Still provides type hints for development

**Commit:** "feat: add print_tree method to Graph class"

### Task 5: Integration Testing

**Goal:** Verify everything works with real graph nodes

**Steps:**
1. Add integration tests to `tests/test_display.py`
2. Use real node classes from `dqx.graph.nodes`

**What to test:**
- Build a real graph with RootNode, CheckNode, AssertionNode
- Call graph.print_tree()
- Mock Console to capture output
- Verify tree structure matches graph structure
- Test that nodes use their name attribute when available
- Test that CheckNode uses node_name() method (label or name)
- Test that AssertionNode shows class name (no name attribute)

**Important notes:**
- AssertionNode doesn't have a name attribute - should show class name
- CheckNode has both name and label - we use node_name() which returns label if available
- Import sympy for creating assertion expressions: `import sympy as sp`

**Manual testing:**
1. Create a script `examples/display_demo.py` (keep for future reference)
2. Build a sample graph
3. Call graph.print_tree()
4. Run: `uv run python examples/display_demo.py`
5. Verify output looks correct in terminal

**Commit:** "test: add integration tests for tree display"

## Testing Guidelines

### Running tests efficiently:
- Test only new code: `uv run pytest tests/test_display.py tests/test_graph_display.py -v`
- Run specific test: `uv run pytest tests/test_display.py::test_name -v`
- Check coverage: `uv run pytest tests/test_display.py --cov=dqx.display`

### Before each commit:
1. Run mypy: `uv run mypy src/dqx/display.py`
2. Run ruff: `uv run ruff check src/dqx/display.py`
3. Fix any issues before committing

## Troubleshooting

### Common issues:

1. **Import errors**
   - Make sure you're using `uv run` for all commands
   - Check that Rich is in pyproject.toml dependencies

2. **Type errors with Optional or Dict**
   - Import from typing: `from typing import Optional, Dict`
   - Or use `Union[Tree, None]` for older Python

3. **Parent not found error**
   - This means DFS order is wrong or node.parent is not set
   - Check parent relationships in your test setup
   - The error message will specify which node caused the issue

4. **Rich not displaying**
   - Terminal might not support Unicode/colors
   - Try setting: `TERM=xterm-256color`

5. **Circular import issues**
   - Use TYPE_CHECKING pattern shown in Task 4
   - Import inside functions if necessary

6. **Node formatting issues**
   - Remember CheckNode has node_name() method
   - AssertionNode has no name, only label
   - Use hasattr() to check for methods/attributes

## Key Principles to Follow

1. **TDD (Test-Driven Development)**
   - Write tests before implementation
   - Run tests to see them fail
   - Implement just enough to make tests pass
   - Refactor if needed

2. **YAGNI (You Aren't Gonna Need It)**
   - Start with just displaying node names
   - Don't add features not in the requirements
   - Keep it simple

3. **DRY (Don't Repeat Yourself)**
   - Reuse existing DFS traversal
   - Use the visitor pattern to separate concerns
   - Don't duplicate logic

4. **Frequent Commits**
   - Commit after each task
   - Use descriptive commit messages
   - Keep commits focused and small

## Expected Final Structure

When complete, you should have:
- `src/dqx/display.py` - Contains NodeFormatter, SimpleNodeFormatter, TreeBuilderVisitor, print_graph
- `tests/test_display.py` - Tests for all display components
- `tests/test_graph_display.py` - Tests for Graph.print_tree method
- `examples/display_demo.py` - Manual testing script (kept for reference)
- Modified `src/dqx/graph/traversal.py` - Added print_tree method

The feature allows users to:
```python
# Simple usage
graph.print_tree()

# With custom formatter
graph.print_tree(formatter=MyCustomFormatter())
```

## Summary

This implementation adds tree visualization to DQX by:
1. Creating a flexible formatter protocol that handles different node types
2. Building a visitor that constructs Rich Trees with proper error handling
3. Providing a clean public API
4. Integrating with the existing Graph class
5. Following best practices (TDD, YAGNI, DRY)

The design is minimal but extensible for future enhancements, with special consideration for the different node types and their naming conventions.
