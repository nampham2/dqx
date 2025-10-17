"""Additional tests to improve coverage for graph/traversal.py."""

import asyncio

import pytest

from dqx.common import DQXError
from dqx.graph.base import BaseNode, NodeVisitor
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph


class AsyncTestVisitor(NodeVisitor):
    """Test visitor that records visited nodes asynchronously."""

    def __init__(self) -> None:
        self.visited_nodes: list[BaseNode] = []
        self.visit_count = 0

    def visit(self, node: BaseNode) -> None:
        """Synchronous visit method."""
        self.visited_nodes.append(node)
        self.visit_count += 1

    async def visit_async(self, node: BaseNode) -> None:
        """Asynchronous visit method with simulated async work."""
        # Simulate some async work
        await asyncio.sleep(0.001)
        self.visited_nodes.append(node)
        self.visit_count += 1


@pytest.mark.asyncio
async def test_async_bfs_traversal() -> None:
    """Test async_bfs method traverses graph correctly."""
    # Create a simple graph structure
    root = RootNode("test_suite")
    check1 = root.add_check("check1")
    check2 = root.add_check("check2")

    # Add some assertions to checks
    assertion1 = check1.add_assertion(
        actual=None,  # type: ignore
        name="assertion1",
        validator=None,  # type: ignore
    )
    assertion2 = check1.add_assertion(
        actual=None,  # type: ignore
        name="assertion2",
        validator=None,  # type: ignore
    )
    assertion3 = check2.add_assertion(
        actual=None,  # type: ignore
        name="assertion3",
        validator=None,  # type: ignore
    )

    # Create graph and visitor
    graph = Graph(root)
    visitor = AsyncTestVisitor()

    # Perform async BFS traversal
    await graph.async_bfs(visitor)

    # Verify all nodes were visited
    assert visitor.visit_count == 6  # root + 2 checks + 3 assertions
    assert len(visitor.visited_nodes) == 6

    # Verify BFS order (level by level)
    # Level 0: Root
    assert visitor.visited_nodes[0] == root
    # Level 1: Checks
    assert visitor.visited_nodes[1] == check1
    assert visitor.visited_nodes[2] == check2
    # Level 2: Assertions
    assert visitor.visited_nodes[3] == assertion1
    assert visitor.visited_nodes[4] == assertion2
    assert visitor.visited_nodes[5] == assertion3


@pytest.mark.asyncio
async def test_async_bfs_with_composite_nodes() -> None:
    """Test async_bfs correctly handles CompositeNode type checking."""
    # Create a graph with composite nodes
    root = RootNode("test_suite")
    check = root.add_check("test_check")

    # Add nested structure
    for i in range(3):
        check.add_assertion(
            actual=None,  # type: ignore
            name=f"assertion_{i}",
            validator=None,  # type: ignore
        )

    graph = Graph(root)
    visitor = AsyncTestVisitor()

    # Perform async BFS
    await graph.async_bfs(visitor)

    # Verify traversal
    assert visitor.visit_count == 5  # root + check + 3 assertions

    # Verify that CompositeNode branches were followed
    visited_types = [type(node).__name__ for node in visitor.visited_nodes]
    assert visited_types == ["RootNode", "CheckNode", "AssertionNode", "AssertionNode", "AssertionNode"]


@pytest.mark.asyncio
async def test_async_bfs_empty_children() -> None:
    """Test async_bfs with nodes that have no children."""
    # Create a minimal graph
    root = RootNode("empty_suite")
    # Root has no checks

    graph = Graph(root)
    visitor = AsyncTestVisitor()

    await graph.async_bfs(visitor)

    # Only root should be visited
    assert visitor.visit_count == 1
    assert visitor.visited_nodes[0] == root


def test_impute_datasets_with_errors() -> None:
    """Test impute_datasets raises DQXError when visitor has errors."""
    from unittest.mock import Mock, patch

    # Create a simple graph
    root = RootNode("test_suite")
    root.add_check("test_check")
    graph = Graph(root)

    # Mock the visitor to simulate errors
    with patch("dqx.graph.visitors.DatasetImputationVisitor") as MockVisitor:
        # Create mock visitor instance
        mock_visitor = Mock()
        mock_visitor.has_errors.return_value = True
        mock_visitor.get_error_summary.return_value = "Test error summary"

        # Configure the mock class to return our mock instance
        MockVisitor.return_value = mock_visitor

        # Mock provider
        mock_provider = Mock()

        # Call impute_datasets - should raise error
        with pytest.raises(DQXError, match="Test error summary"):
            graph.impute_datasets(["dataset1", "dataset2"], mock_provider)

        # Verify visitor was created with correct arguments
        MockVisitor.assert_called_once_with(["dataset1", "dataset2"], mock_provider)

        # Verify DFS was called on the visitor
        mock_visitor.has_errors.assert_called_once()
        mock_visitor.get_error_summary.assert_called_once()


def test_impute_datasets_without_errors() -> None:
    """Test impute_datasets completes successfully when no errors."""
    from unittest.mock import Mock, patch

    # Create a simple graph
    root = RootNode("test_suite")
    root.add_check("test_check")
    graph = Graph(root)

    # Mock the visitor to simulate no errors
    with patch("dqx.graph.visitors.DatasetImputationVisitor") as MockVisitor:
        # Create mock visitor instance
        mock_visitor = Mock()
        mock_visitor.has_errors.return_value = False

        # Configure the mock class to return our mock instance
        MockVisitor.return_value = mock_visitor

        # Mock provider
        mock_provider = Mock()

        # Call impute_datasets - should not raise error
        graph.impute_datasets(["dataset1"], mock_provider)

        # Verify visitor was created and used correctly
        MockVisitor.assert_called_once_with(["dataset1"], mock_provider)
        mock_visitor.has_errors.assert_called_once()
        # get_error_summary should NOT be called when there are no errors
        mock_visitor.get_error_summary.assert_not_called()


@pytest.mark.asyncio
async def test_async_bfs_preserves_order() -> None:
    """Test that async_bfs maintains breadth-first order."""
    # Create a more complex graph
    root = RootNode("complex_suite")

    # Add multiple levels
    checks = []
    for i in range(3):
        check = root.add_check(f"check_{i}")
        checks.append(check)

        # Add assertions to each check
        for j in range(2):
            check.add_assertion(
                actual=None,  # type: ignore
                name=f"assertion_{i}_{j}",
                validator=None,  # type: ignore
            )

    graph = Graph(root)
    visitor = AsyncTestVisitor()

    await graph.async_bfs(visitor)

    # Verify correct number of visits
    assert visitor.visit_count == 10  # 1 root + 3 checks + 6 assertions

    # Verify breadth-first order
    node_names = []
    for node in visitor.visited_nodes:
        if hasattr(node, "name"):
            node_names.append(node.name)

    expected_order = [
        "complex_suite",  # Level 0
        "check_0",
        "check_1",
        "check_2",  # Level 1
        "assertion_0_0",
        "assertion_0_1",
        "assertion_1_0",
        "assertion_1_1",
        "assertion_2_0",
        "assertion_2_1",  # Level 2
    ]

    assert node_names == expected_order
